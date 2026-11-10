"""A simulated SO-101 (MuJoCo, the Menagerie `robotstudio_so101` model) with the same methods as the
real arm (`real.py`): `read`, `send`, `goto`, `play`, `relax`, plus an arrival check, a grasp load
check and servo status bits.

Units follow LeRobot: arm joints in degrees, gripper 0-100 (0 = closed). The sim's zero pose isn't a
real arm's calibrated zero, so the real arm uses poses taught on it (`python -m so101_host teach`).
"""
from collections import deque

import mujoco
import numpy as np

from .model import BLOCK_HALF, BLOCK_POS, BLOCK_TOP, build

# What forward kinematics can see in the scene: the block's top face (x, y, top_z, half-size).
SCENE_OBJECTS = [(BLOCK_POS[0], BLOCK_POS[1], BLOCK_TOP, BLOCK_HALF[0])]

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
JOINTS = ARM_JOINTS + (GRIPPER,)

# STS3215 Status register bits: 1 voltage, 2 angle sensor, 4 overheat,
# 8 over-current, 32 overload. Servo alarms never raise; they have to be polled.
OVERLOAD = 32
OVERLOAD_AFTER_S = 0.5     # force at the limit this long sets the overload bit

SENSOR_RATE_HZ = 10        # the wrist ultrasonic reports at 10 Hz
US_JITTER = 0.5            # µs of echo jitter, so a healthy trace is never perfectly flat
MIN_MM = 20                # echoes under ~116 µs (about 20 mm) read as out_of_range


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class SimArm:
    def __init__(self, model=None, *, fps=50, recorder=None, seed=0, start=None):
        self.m = model if model is not None else build()
        self.d = mujoco.MjData(self.m)
        self.fps = fps
        self.substeps = max(1, round(1.0 / fps / self.m.opt.timestep))
        self.recorder = recorder
        self.hooks = []                      # called once per control tick (GIF, viewer)
        self.watchers = []                   # called with each new distance row (the watchdog)
        self.faults = {}                     # active planted faults, see inject()
        self.d_rows = deque(maxlen=200)      # recent D rows with ts and pred_mm (20 s at 10 Hz)
        self.halt_reason = None              # set by the watchdog: motion stops at the next tick
        self._parking = False
        self.rng = np.random.default_rng(seed)
        self._qadr = np.array([self.m.joint(j).qposadr[0] for j in JOINTS])
        self._dadr = np.array([self.m.joint(j).dofadr[0] for j in JOINTS])
        self._act = np.array([self.m.actuator(j).id for j in JOINTS])
        self._lo, self._hi = self.m.actuator_ctrlrange[self._act].T.copy()
        self._flimit = np.abs(self.m.actuator_forcerange[self._act]).max(axis=1)
        self._gain = self.m.actuator_gainprm[self._act].copy()
        self._bias = self.m.actuator_biasprm[self._act].copy()
        self._at_limit = np.zeros(len(JOINTS))
        self._sensor = self.m.sensor("ultrasonic").adr[0]
        self._tip = self.m.site("gripperframe").id
        self._next_d = 0.0
        self._last_d = None
        if start is not None:
            self.d.qpos[self._qadr] = self.to_ctrl(start)
        mujoco.mj_forward(self.m, self.d)
        self.d.ctrl[self._act] = self.d.qpos[self._qadr]

    # ---- units ---------------------------------------------------------------------------
    def to_ctrl(self, pose):
        """LeRobot units (dict of `<joint>.pos`, or a list in JOINTS order) -> servo radians."""
        q = self._raw(pose)
        # Planted joint faults (the fault factory): the software believes one angle, the servo
        # physically goes to another.
        if "joint_offset" in self.faults:          # bad calibration: a constant offset on one joint
            f = self.faults["joint_offset"]
            q[f["joint"]] += np.deg2rad(f["deg"])
        if "joint_reversed" in self.faults:        # wrong direction: the joint mirrors about its zero
            q[self.faults["joint_reversed"]["joint"]] *= -1.0
        return np.clip(q, self._lo, self._hi)

    def _raw(self, pose):
        """LeRobot units -> radians, with no faults applied (the software's own view)."""
        if isinstance(pose, dict):
            cur = self._units(self._believed(self.d.ctrl[self._act]))
            pose = [pose.get(f"{j}.pos", pose.get(j, c)) for j, c in zip(JOINTS, cur)]
        vals = np.array(pose, dtype=float)
        g = self._lo[5] + (self._hi[5] - self._lo[5]) * np.clip(vals[5], 0, 100) / 100.0
        return np.append(np.deg2rad(vals[:5]), g)

    def _belief_data(self):
        """MjData posed where the software believes the arm is (the physical pose if no joint fault)."""
        if "joint_offset" not in self.faults and "joint_reversed" not in self.faults:
            return self.d
        if not hasattr(self, "_dbelief"):
            self._dbelief = mujoco.MjData(self.m)
        d = self._dbelief
        d.qpos[:] = self.d.qpos
        d.qpos[self._qadr] = self._believed(self.d.qpos[self._qadr])
        mujoco.mj_kinematics(self.m, d)
        return d

    def _units(self, q):
        q = np.asarray(q, dtype=float)
        g = (q[5] - self._lo[5]) / (self._hi[5] - self._lo[5]) * 100.0
        return np.append(np.rad2deg(q[:5]), g)

    def _believed(self, q):
        """What the encoders report. With a planted calibration fault the software believes a
        different angle than the physical one; the residual check exists to catch exactly that."""
        q = np.array(q, dtype=float)
        if "joint_offset" in self.faults:
            f = self.faults["joint_offset"]
            q[f["joint"]] -= np.deg2rad(f["deg"])
        if "joint_reversed" in self.faults:
            q[self.faults["joint_reversed"]["joint"]] *= -1.0
        return q

    # ---- the arm interface (shared with real.py) ----------------------------------------------
    @property
    def t(self):
        return float(self.d.time)

    def read(self):
        believed = self._believed(self.d.qpos[self._qadr])
        return {f"{j}.pos": float(v) for j, v in zip(JOINTS, self._units(believed))}

    def send(self, action):
        self._torque(True)
        self.d.ctrl[self._act] = self.to_ctrl(action)

    def tick(self):
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
        dt = self.substeps * self.m.opt.timestep
        force = np.abs(self.d.actuator_force[self._act])
        self._at_limit = np.where(force >= 0.95 * self._flimit, self._at_limit + dt, 0.0)
        if self.recorder:
            self.recorder.joints(self.t, self._units(self._believed(self.d.ctrl[self._act])),
                                 self._units(self._believed(self.d.qpos[self._qadr])),
                                 self.d.actuator_force[self._act], self.status())
        if self.t + 1e-9 >= self._next_d:
            self._next_d += 1.0 / SENSOR_RATE_HZ
            row = self.distance()
            pred = self.predicted_mm(objects=SCENE_OBJECTS)
            row["pred_mm"] = None if pred is None else round(pred, 1)
            if self.recorder:
                self.recorder.distance(self.t, row)
            self.d_rows.append({"ts": self.t, **row})
            for watch in self.watchers:
                watch(self)
        for hook in self.hooks:
            hook(self)

    def halted(self):
        return self.halt_reason is not None and not self._parking

    def hold(self, seconds):
        for _ in range(max(1, round(seconds * self.fps))):
            self.tick()

    def goto(self, pose, duration=1.5, *, tol_deg=2.0, settle_timeout=2.0):
        """Ease to `pose`, then confirm arrival."""
        start, goal = self.d.ctrl[self._act].copy(), self.to_ctrl(pose)
        self._torque(True)
        n = max(1, round(duration * self.fps))
        for i in range(1, n + 1):
            if self.halted():
                return self._stop()
            self.d.ctrl[self._act] = start + (goal - start) * smooth(i / n)
            self.tick()
        return self.wait_arrived(goal, tol_deg=tol_deg, timeout=settle_timeout)

    def _stop(self):
        """Watchdog halt: hold where the arm is and say why."""
        self.d.ctrl[self._act] = self.d.qpos[self._qadr]
        return {"arrived": False, "halted": self.halt_reason}

    def wait_arrived(self, goal=None, *, tol_deg=2.0, timeout=2.0):
        goal = self.d.ctrl[self._act] if goal is None else goal
        t0 = self.t
        while True:
            if self.halted():
                return self._stop()
            err = np.rad2deg(np.abs(goal - self.d.qpos[self._qadr]))[:5]
            speed = np.abs(self.d.qvel[self._dadr])[:5]
            if err.max() <= tol_deg and speed.max() < 0.05:
                return {"arrived": True, "max_err_deg": round(float(err.max()), 2), "seconds": round(self.t - t0, 2)}
            if self.t - t0 >= timeout:
                return {"arrived": False, "max_err_deg": round(float(err.max()), 2), "seconds": round(self.t - t0, 2),
                        "worst_joint": ARM_JOINTS[int(err.argmax())]}
            self.tick()

    def play(self, prim, speed=1.0):
        """Play a recorded motion ({fps, joints, frames}); ease to its first frame first."""
        names = prim.get("joints") or list(JOINTS)
        order = [names.index(j) if j in names else None for j in JOINTS]
        frames = [[f[i] if i is not None else c for i, c in zip(order, self._units(self.d.ctrl[self._act]))]
                  for f in prim["frames"]]
        self.goto(frames[0], duration=1.0)
        src_fps = prim.get("fps", 50) * speed
        n = max(1, round(len(frames) / src_fps * self.fps))
        for i in range(n):
            if self.halted():
                return self._stop()
            k = min(len(frames) - 1, int(i * src_fps / self.fps))
            self.d.ctrl[self._act] = self.to_ctrl(frames[k])
            self.tick()
        return self.wait_arrived(timeout=1.0)

    def park(self, rest):
        """Go to the rest pose even while halted, then torque off (the watchdog's move)."""
        self._parking = True
        try:
            r = self.goto(rest)
        finally:
            self._parking = False
        self.relax()
        return {**r, "torque": "off"}

    def relax(self):
        """Torque off. On the real arm this drops it, so park at a rest pose first."""
        self._torque(False)

    def _torque(self, on):
        if on:
            self.m.actuator_gainprm[self._act] = self._gain
            self.m.actuator_biasprm[self._act] = self._bias
            if "joint_stuck" in self.faults:     # a stuck servo stays stuck when torque comes back
                a = self._act[self.faults["joint_stuck"]["joint"]]
                self.m.actuator_gainprm[a] = 0.0
                self.m.actuator_biasprm[a] = 0.0
        else:
            self.m.actuator_gainprm[self._act] = 0.0
            self.m.actuator_biasprm[self._act] = 0.0

    # ---- checks -------------------------------------------------------------------------------
    def status(self):
        return [OVERLOAD if s >= OVERLOAD_AFTER_S else 0 for s in self._at_limit]

    kind = "sim"
    use_ik = True             # the sim's kinematics are exact; the real arm's need bench calibration
    kinematics_ok = True

    def commanded(self):
        """The pose the servos are being told to hold (LeRobot units, JOINTS order)."""
        return self._units(self._believed(self.d.ctrl[self._act])).tolist()

    def load_nm(self, joint):
        """Torque the joint's servo is applying now (N*m)."""
        return float(abs(self.d.actuator_force[self._act][JOINTS.index(joint)]))

    def grip_limit_nm(self):
        """The gripper servo's torque limit (N*m); the overload bit trips when held near it."""
        return float(self._flimit[5])

    def grasp_check(self, *, min_open=4.0, min_load=0.25):
        """Holding = the gripper stopped short of fully closed and its servo is pushing."""
        g = float(self._units(self.d.qpos[self._qadr])[5])
        load = float(abs(self.d.actuator_force[self._act][5]))
        return {"holding": g > min_open and load > min_load, "gripper": round(g, 1), "load_nm": round(load, 3)}

    # ---- the wrist ultrasonic (an HC-SR04-style rangefinder) -----------------------------------
    def distance(self):
        """One distance row: {mm, us, st}; trust mm only when st == "ok".

        Like an HC-SR04 driver: the echo time `us`, then mm = (us*343+1000)/2000 in integer math.
        """
        if "us_frozen" in self.faults and self._last_d is not None:
            return dict(self._last_d)            # the same reading, forever: a frozen sensor
        dist_m = float(self.d.sensordata[self._sensor])
        if dist_m < 0 or dist_m >= self.m.sensor_cutoff[self.m.sensor("ultrasonic").id] - 1e-6:
            row = {"mm": None, "us": None, "st": "no_echo"}    # nothing within 4 m: no echo comes back
        else:
            true_mm = dist_m * 1000.0
            if "us_offset" in self.faults:      # calibration: reads N mm long or short everywhere
                true_mm += self.faults["us_offset"]["mm"]
            if "us_scale" in self.faults:       # a clock/units bug, e.g. a half-speed timer
                true_mm *= self.faults["us_scale"]["factor"]
            jitter = self.faults.get("us_noisy", {}).get("us", US_JITTER)
            us = int(round(max(true_mm, 0.0) * 2000.0 / 343.0 + self.rng.normal(0.0, jitter)))
            mm = (us * 343 + 1000) // 2000
            row = {"mm": mm, "us": us, "st": "ok" if mm >= MIN_MM else "out_of_range"}
        self._last_d = row
        return dict(row)

    def sensor_m(self):
        """The rangefinder's true distance in metres, or None when nothing is in range (no echo)."""
        dist_m = float(self.d.sensordata[self._sensor])
        if dist_m < 0 or dist_m >= self.m.sensor_cutoff[self.m.sensor("ultrasonic").id] - 1e-6:
            return None
        return dist_m

    def inject(self, fault, **params):
        """Plant a fault (sim only). Sensor: `us_frozen`, `us_offset` (mm), `us_scale` (factor),
        `us_noisy` (us). Joints: `joint_stuck`, `joint_offset` (deg), `joint_reversed` (joint index).
        Scene: `block_shift` (dx_mm, dy_mm): the block is somewhere else than where the arm was taught."""
        if fault == "block_shift":
            self.shift_block(params.get("dx_mm", 30.0), params.get("dy_mm", 0.0))
        self.faults[fault] = {"at": round(self.t, 3), **params}
        if fault == "joint_stuck":               # the servo stops following: no torque on that joint
            a = self._act[params["joint"]]
            self.m.actuator_gainprm[a] = 0.0
            self.m.actuator_biasprm[a] = 0.0
        return self.faults[fault]

    def clear(self, fault=None):
        if fault in (None, "block_shift") and "block_shift" in self.faults:
            self.shift_block(0.0, 0.0)
        if fault is None:
            self.faults.clear()
        else:
            self.faults.pop(fault, None)

    def shift_block(self, dx_mm=0.0, dy_mm=0.0):
        """Put the block dx_mm forward and dy_mm left of its taught spot, upright and still."""
        j = self.m.body("box").jntadr[0]
        q, v = self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]
        self.d.qpos[q:q + 7] = [BLOCK_POS[0] + dx_mm / 1000.0, BLOCK_POS[1] + dy_mm / 1000.0, BLOCK_POS[2], 1, 0, 0, 0]
        self.d.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        return self.block_mm()

    def block_mm(self):
        """Where the block is now, in mm from its taught spot (dx forward, dy left): ground truth for tests."""
        p = self.d.xpos[self.m.body("box").id]
        return [round(float(p[0] - BLOCK_POS[0]) * 1000.0, 1), round(float(p[1] - BLOCK_POS[1]) * 1000.0, 1)]

    # ---- kinematics (MuJoCo's own; replaces Placo for the residual check) --------------------
    def tip(self):
        """Fingertip position (m) and approach axis (unit vector), as the software believes them."""
        d = self._belief_data()
        return d.site_xpos[self._tip].copy(), d.site_xmat[self._tip].reshape(3, 3)[:, 0].copy()

    def predicted_mm(self, table_z=0.0, *, objects=()):
        """Distance the sensor *should* read, from what the encoders believe: the same ray the
        rangefinder casts, against the whole model posed at the believed angles. That includes the
        arm's own links and the moving jaw (closed, it sits in the beam at about 7 mm), the block
        and the table. None when nothing is in range. (`table_z` and `objects` are kept for older
        callers; the model already holds the table and the block.)"""
        sid = self.m.site("ultrasonic").id
        d = self._belief_data()
        p = d.site_xpos[sid].copy()
        a = d.site_xmat[sid].reshape(3, 3)[:, 2].copy()
        geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.m, d, p, a, None, 1, int(self.m.site_bodyid[sid]), geomid)
        cutoff = float(self.m.sensor_cutoff[self.m.sensor("ultrasonic").id])
        if dist < 0 or dist >= cutoff - 1e-6:
            return None
        return float(dist) * 1000.0

    def solve(self, xyz, *, seed=None, iters=200):
        """Joint pose (LeRobot units) that puts the fingertips at `xyz` pointing straight down."""
        m, d = self.m, mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        # Solve in the software's own (believed) joint space; the servos apply any planted fault.
        d.qpos[self._qadr] = self._raw(seed) if seed is not None else self._believed(self.d.qpos[self._qadr])
        cols = self._dadr[:4]                         # pan, lift, elbow, wrist flex
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        down = np.array([0.0, 0.0, -1.0])
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            pos = d.site_xpos[self._tip]
            axis = d.site_xmat[self._tip].reshape(3, 3)[:, 0]
            err = np.concatenate([np.asarray(xyz) - pos, 0.5 * np.cross(axis, down)])
            if np.linalg.norm(err[:3]) < 5e-4 and np.linalg.norm(err[3:]) < 5e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self._tip)
            J = np.vstack([jacp[:, cols], 0.5 * jacr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = d.qpos[self._qadr[:4]] + dq
            d.qpos[self._qadr[:4]] = np.clip(q, self._lo[:4], self._hi[:4])
        mujoco.mj_forward(m, d)
        miss = float(np.linalg.norm(np.asarray(xyz) - d.site_xpos[self._tip]))
        tilt = float(np.degrees(np.arccos(np.clip(-d.site_xmat[self._tip].reshape(3, 3)[2, 0], -1, 1))))
        if miss > 0.003 or tilt > 3.0:
            raise ValueError(f"can't reach {np.round(xyz, 3).tolist()} pointing down "
                             f"(off by {miss * 1000:.0f} mm, tilted {tilt:.0f} deg)")
        out = self._units(d.qpos[self._qadr])
        out[5] = self._units(self._believed(self.d.ctrl[self._act]))[5]
        return out.tolist()
