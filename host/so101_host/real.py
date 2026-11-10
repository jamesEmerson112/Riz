"""The real SO-101 behind the same interface as SimArm, so the host, the watchdog and the Hermes plugin
run unchanged on hardware. It talks to the servos through LeRobot (`SO101Follower`) and adds what the
plain follower lacks: an arrival check, the joints' load, the servos' Status register and an overload
timer.

    arm = RealArm(port="/dev/ttyACM0", robot_id="my_so101").connect()
    arm = RealArm(robot=MockRobot())          # tests without hardware (mockrobot.py)

What differs from the sim, on purpose:
  - Moves go between poses taught on this arm (`python -m so101_host teach <pose>`), in joint space:
    the sim's zero isn't this arm's calibrated zero.
  - There is no wrist distance sensor, so the watchdog's distance rules stay quiet; joint stalls,
    overload and servo alarm bits still park the arm.
  - Servo registers are probed at connect: one that LeRobot's table doesn't name is skipped, not fatal.

Status: written for LeRobot 0.7's `lerobot.robots.so_follower` API and tested against the mock robot.
"""
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from .arm import ARM_JOINTS, JOINTS, OVERLOAD, OVERLOAD_AFTER_S

STALL_NM = 2.94          # STS3215 at 12 V, about 30 kg*cm: load fractions are reported in N*m against it
LOAD_ALARM = 0.95        # held at this fraction of full load for OVERLOAD_AFTER_S, report OVERLOAD


def data_dir():
    """Where taught poses and motions live: $SO101_HOME, else ~/.so101-arm."""
    return Path(os.environ.get("SO101_HOME") or Path.home() / ".so101-arm")


def taught_library(home=None):
    """(poses, motions) taught on this arm. poses.json is {name: [6 values]}; motions/<name>.json are
    {name, fps, joints, frames}. A skill pose not taught yet is None, so goto() refuses it."""
    home = Path(home or data_dir())
    poses_file = home / "poses.json"
    poses = json.loads(poses_file.read_text()) if poses_file.exists() else {}
    for name in ("rest", "home", "above_block", "pick"):
        poses.setdefault(name, None)
    motions = {}
    for f in sorted((home / "motions").glob("*.json")) if (home / "motions").is_dir() else []:
        motions[f.stem] = json.loads(f.read_text())
    return poses, motions


def lerobot_follower(port, robot_id, *, calibration_dir=None, max_relative_target=None):
    """LeRobot's SO-101 follower. Imported here so the sim never needs LeRobot installed."""
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise SystemExit("the real arm needs LeRobot: pip install -e 'host[real]' "
                         f"(LeRobot 0.7 with the feetech extra). Import failed: {exc}")
    kw = {"port": port, "id": robot_id}
    if calibration_dir:
        kw["calibration_dir"] = Path(calibration_dir)
    if max_relative_target:
        kw["max_relative_target"] = float(max_relative_target)
    return SO101Follower(SO101FollowerConfig(**kw))


class RealArm:
    kind = "real"
    has_sensor = False       # no wrist distance sensor on the real arm (yet)
    use_ik = False           # joint-space moves between taught poses
    kinematics_ok = False
    taught_poses = None

    def __init__(self, *, robot=None, recorder=None, fps=50, port=None, robot_id=None, calibration_dir=None,
                 max_relative_target=None):
        if robot is None:
            port = port or os.environ.get("SO101_PORT")
            if not port or not Path(port).exists():
                raise SystemExit(f"no arm at SO101_PORT={port!r}. Plug in the arm (USB and 12 V power), find its "
                                 "port (`lerobot-find-port`, or /dev/serial/by-id/), and set SO101_PORT.")
            robot = lerobot_follower(port, robot_id or os.environ.get("SO101_ID") or "so101_follower",
                                     calibration_dir=calibration_dir or os.environ.get("SO101_CALIBRATION_DIR"),
                                     max_relative_target=max_relative_target or os.environ.get("SO101_MAX_STEP_DEG"))
        self.robot = robot
        self.fps, self.recorder = fps, recorder
        self.hooks, self.watchers, self.faults = [], [], {}
        self.d_rows = deque(maxlen=200)       # stays empty: no distance sensor
        self.halt_reason, self._parking = None, False
        self.t0 = time.monotonic()
        self._goal = None
        self._torque_on = False
        self._at_limit = np.zeros(6)
        self._last_tick = None
        self._last_d = None
        self.registers = {}

    # ---- lifecycle ---------------------------------------------------------------------------
    def connect(self):
        if hasattr(self.robot, "calibration") and not self.robot.calibration:     # LeRobot found no file
            fpath = getattr(self.robot, "calibration_fpath", "the LeRobot calibration folder")
            raise SystemExit(f"no calibration for this arm at {fpath}. Calibrate it once with LeRobot "
                             "(`lerobot-calibrate --robot.type=so101_follower --robot.port=... --robot.id=...`), "
                             "using the same id as SO101_ID.")
        try:
            self.robot.connect(calibrate=False)    # never LeRobot's interactive calibration from the host
        except TypeError:
            self.robot.connect()                   # the mock robot
        self._torque_on = True
        self.registers = {name: self._probe(name) for name in ("Present_Load", "Status", "Present_Velocity")}
        self._goal = self._read_vec()
        return self

    def disconnect(self):
        if getattr(self.robot, "is_connected", True):
            self.robot.disconnect()

    def _probe(self, name):
        try:
            self.robot.bus.sync_read(name, normalize=False)
            return True
        except Exception:                   # LeRobot raises KeyError (or similar) for an unknown name
            return False

    # ---- the SimArm interface ----------------------------------------------------------------
    @property
    def t(self):
        return time.monotonic() - self.t0

    def _read_vec(self):
        obs = self.robot.get_observation()
        return np.array([float(obs[f"{j}.pos"]) for j in JOINTS])

    def read(self):
        return {f"{j}.pos": float(v) for j, v in zip(JOINTS, self._read_vec())}

    def _vec(self, pose):
        if isinstance(pose, dict):
            cur = self._goal if self._goal is not None else self._read_vec()
            return np.array([float(pose.get(f"{j}.pos", pose.get(j, c))) for j, c in zip(JOINTS, cur)])
        return np.array(pose, dtype=float)

    def send(self, action):
        # taught_poses is set by the host; {} with nothing taught means no pose on this arm is safe to use.
        if self.taught_poses is not None and not any(v is not None for v in self.taught_poses.values()):
            raise RuntimeError("no taught poses on this arm: teach them first, with torque off: "
                               "python -m so101_host teach rest home above_block pick")
        if not self._torque_on:
            self.robot.bus.enable_torque()
            self._torque_on = True
        self._goal = self._vec(action)
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, self._goal)})

    def _loads(self):
        if not self.registers.get("Present_Load"):
            return np.zeros(6)
        raw = self.robot.bus.sync_read("Present_Load", normalize=False)
        mags = []
        for j in JOINTS:
            v = int(raw.get(j, 0))
            # LeRobot 0.7 already returns a signed value, so -78 means 7.8% the other way; masking it
            # with 0x3FF would read 94.6% and raise false overloads. A raw sign-magnitude value
            # (bit 10 = direction, as the mock bus sends) still works.
            mags.append((-v if v < 0 else v & 0x3FF) / 1000.0)
        return np.array(mags) * STALL_NM

    def tick(self):
        """One control period: read, update the load timers, record, pace to the control rate."""
        q = self._read_vec()
        loads = self._loads()
        now = time.monotonic()
        dt = 0.0 if self._last_tick is None else min(0.2, now - self._last_tick)
        self._last_tick = now
        self._at_limit = np.where(loads >= LOAD_ALARM * STALL_NM, self._at_limit + dt, 0.0)
        self._q, self._load = q, loads
        if self.recorder:
            self.recorder.joints(self.t, self._goal if self._goal is not None else q, q, loads, self.status())
        for hook in self.hooks:
            hook(self)
        nxt = now + 1.0 / self.fps
        while time.monotonic() < nxt:
            time.sleep(max(0.0, nxt - time.monotonic()))

    def halted(self):
        return self.halt_reason is not None and not self._parking

    def hold(self, seconds):
        for _ in range(max(1, round(seconds * self.fps))):
            self.tick()

    def goto(self, pose, duration=1.5, *, tol_deg=2.0, settle_timeout=2.0):
        if pose is None:
            raise RuntimeError("that pose isn't taught on this arm: teach it with torque off: "
                               "python -m so101_host teach <name>")
        start = self._read_vec() if self._goal is None else self._goal.copy()
        goal = self._vec(pose)
        n = max(1, round(duration * self.fps))
        for i in range(1, n + 1):
            if self.halted():
                return self._stop()
            a = i / n
            self.send(start + (goal - start) * (a * a * (3 - 2 * a)))
            self.tick()
        return self.wait_arrived(goal, tol_deg=tol_deg, timeout=settle_timeout)

    def _stop(self):
        self.send(self._read_vec())                   # hold where it is
        return {"arrived": False, "halted": self.halt_reason}

    def wait_arrived(self, goal=None, *, tol_deg=2.0, timeout=2.0):
        goal = self._goal if goal is None else np.asarray(goal, dtype=float)
        t0, prev = self.t, None
        while True:
            if self.halted():
                return self._stop()
            self.tick()
            err = np.abs(goal - self._q)[:5]
            moving = prev is not None and np.abs(self._q - prev)[:5].max() > 0.05
            prev = self._q.copy()
            if err.max() <= tol_deg and not moving:
                return {"arrived": True, "max_err_deg": round(float(err.max()), 2), "seconds": round(self.t - t0, 2)}
            if self.t - t0 >= timeout:
                return {"arrived": False, "max_err_deg": round(float(err.max()), 2), "seconds": round(self.t - t0, 2),
                        "worst_joint": ARM_JOINTS[int(err.argmax())]}

    def play(self, prim, speed=1.0):
        names = prim.get("joints") or list(JOINTS)
        cur = self._goal if self._goal is not None else self._read_vec()
        order = [names.index(j) if j in names else None for j in JOINTS]
        frames = [[f[i] if i is not None else c for i, c in zip(order, cur)] for f in prim["frames"]]
        r = self.goto(frames[0], duration=1.0)
        if r.get("halted"):
            return r
        src_fps = prim.get("fps", 50) * speed
        n = max(1, round(len(frames) / src_fps * self.fps))
        for i in range(n):
            if self.halted():
                return self._stop()
            self.send(frames[min(len(frames) - 1, int(i * src_fps / self.fps))])
            self.tick()
        return self.wait_arrived(timeout=1.0)

    def park(self, rest):
        self._parking = True
        try:
            r = self.goto(rest)
        finally:
            self._parking = False
        self.relax()
        return {**r, "torque": "off"}

    def relax(self):
        self.robot.bus.disable_torque()
        self._torque_on = False

    def status(self):
        """Servo alarm bits: the Status register where LeRobot names it, OR the overload timer."""
        bits = [OVERLOAD if s >= OVERLOAD_AFTER_S else 0 for s in self._at_limit]
        if self.registers.get("Status"):
            raw = self.robot.bus.sync_read("Status", normalize=False)
            bits = [b | int(raw.get(j, 0)) for b, j in zip(bits, JOINTS)]
        return bits

    def commanded(self):
        """The pose the servos are being told to hold (LeRobot units, JOINTS order)."""
        return (self._goal if self._goal is not None else self._read_vec()).tolist()

    def load_nm(self, joint):
        loads = getattr(self, "_load", None)
        return float(loads[JOINTS.index(joint)]) if loads is not None else 0.0

    def grip_limit_nm(self):
        return STALL_NM

    def grasp_check(self, *, min_open=4.0, min_load=0.25):
        q = getattr(self, "_q", None)
        if q is None:
            self.tick()
            q = self._q
        load = self.load_nm("gripper")
        return {"holding": float(q[5]) > min_open and load > min_load, "gripper": round(float(q[5]), 1),
                "load_nm": round(load, 3)}

    # ---- sim-only features, refused politely --------------------------------------------------
    def inject(self, fault, **params):
        raise ValueError("faults can't be planted on the real arm from software; plant them by hand "
                         "(hold a joint, block the arm) or use the sim")

    def clear(self, fault=None):
        return None

    def distance(self):
        return {"mm": None, "us": None, "st": "no_sensor"}

    def predicted_mm(self, *_, **__):
        return None
