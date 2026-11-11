"""The arm host: one long-lived process that owns the arm and runs skills on request.

Newline-delimited JSON over a Unix socket, one request per line, one reply per line. LeRobot's
follower takes seconds to start, so the real arm also wants exactly one long-lived owner of the bus.

    {"op": "skill", "name": "approach"}       -> {"ok": true, "skill": "approach", "result": {...}}
    {"op": "play", "motion": "wave"}           play a recorded motion (sim: the built-in wave)
    {"op": "status"} / {"op": "read"} / {"op": "distance"} / {"op": "incidents"} / {"op": "motions"}
    {"op": "repair", "action": "restart_sensor" | "verify"}   the healer's tools: fix, then re-check
    {"op": "handoff", "id": "inc-1", "reason": ...}          a person takes an open incident
    {"op": "freeze", "by": ...} / {"op": "unfreeze", "by": ...}   the operator's kill switch
    {"op": "inject", "fault": "us_frozen"} / {"op": "clear"}  plant or clear a fault (sim only, demos)

The watchdog is plain code (`level1.py`), never a model: on each distance row and after each move it
checks the recent data. The first critical finding opens an incident: motion halts, the arm parks, and
motion stays blocked until a repair is verified by a move home and fresh readings. A joint that still
won't move after a checked retry is handed to a person; only an explicit `repair verify` (after the
person's fix) closes it. The model chooses skills; it never sends joint angles.
"""
import argparse
import base64
import json
import math
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import level1, prims
from .arm import JOINTS, SimArm
from .client import socket_path
from .record import RunRecorder
from .real import data_dir

SOCK = Path(socket_path())


class Skills:
    """The skill menu. Each skill moves the arm and reports how it went. The sim uses its own solved
    poses; the real arm uses poses taught on it (the sim's joint zero isn't a real arm's zero)."""

    NEEDED = ("rest", "home", "above_block", "pick")
    MENU = ("go_home", "reach", "approach", "grip", "lift", "release", "park")

    def __init__(self, arm, *, poses=None, motions=None):
        self.arm = arm
        self.p = poses if poses is not None else prims.poses()
        self.motions = motions if motions is not None else {"wave": prims.load("wave")}
        if getattr(arm, "kind", "sim") == "sim":
            missing = [n for n in self.NEEDED if n not in self.p]
            if missing:
                raise SystemExit(f"sim poses missing: {', '.join(missing)} (python -m so101_host poses)")

    def _keep(self, **joints):
        cur = self.arm.read()
        cur.update({f"{k}.pos": v for k, v in joints.items()})
        return cur

    def go_home(self):
        return self.arm.goto(self.p["home"])

    def reach(self):
        return self.arm.goto(self.p["above_block"])

    def grip(self):
        """Close; if the jaws stall on something, back off to just inside the stall so the servo isn't
        held at its torque limit (an STS3215 held there sets its overload bit and the guard blocks)."""
        r = self.arm.goto(self._keep(gripper=prims.CLOSED), duration=0.8, settle_timeout=0.4)
        g = self.arm.grasp_check()
        if g["holding"]:
            limit = self.arm.grip_limit_nm()
            lo, hi = float(prims.CLOSED), g["gripper"] + 1.0
            for _ in range(8):                       # a squeeze between 40% and 90% of the limit
                if 0.4 * limit <= g["load_nm"] < 0.9 * limit:
                    break
                mid = (lo + hi) / 2
                self.arm.send(self._keep(gripper=mid))
                self.arm.hold(0.15)
                g = self.arm.grasp_check(min_load=0.2)
                if g["load_nm"] >= 0.9 * limit:
                    lo = mid                         # still at the limit: open a little more
                else:
                    hi = mid                         # too loose (or let go): close a little
            self.arm.hold(0.6)                       # long enough for a stale overload count to clear
            g = self.arm.grasp_check(min_load=0.2)
        return {**{k: v for k, v in r.items() if k != "arrived"}, "grasp": g, "status": self.arm.status()}

    def release(self):
        return self.arm.goto(self._keep(gripper=prims.OPEN), duration=0.6)

    def park(self):
        return self.arm.park(self.p["rest"])

    def approach(self, offset_mm=(0.0, 0.0)):
        if not getattr(self.arm, "has_sensor", True):
            if any(offset_mm):
                return {"arrived": False, "aborted": "offsets need the sim's kinematics; the real arm moves "
                                                     "between taught poses only"}
            return approach_joint(self.arm, self.p)
        return approach(self.arm, self.p, offset_mm=offset_mm)

    def lift(self):
        return lift(self.arm, self.p)

    def play(self, name, speed=1.0):
        motion = self.motions.get(name)
        if motion is None:
            return {"arrived": False, "aborted": f"no motion named {name!r}; known: {sorted(self.motions)}"}
        return self.arm.play(motion, speed=speed)


MAX_OFFSET_MM = 40.0                         # the agent's aim correction, per axis: enforced here, in code
SETBACK_MM = 50                              # the sensor sits 50 mm behind the fingertips
SWITCH_MM = max(60, SETBACK_MM + 25)         # stop stepping here, then a fixed final move


def _median_reading(arm, n=3, timeout=1.0):
    """Wait for n fresh distance rows ("step and settle"), then the median of the valid ones."""
    start = len(arm.d_rows) and arm.d_rows[-1]["ts"]
    t0 = arm.t
    while arm.t - t0 < timeout:
        arm.tick()
        fresh = [r for r in arm.d_rows if r["ts"] > start]
        if len(fresh) >= n:
            ok = sorted(r["mm"] for r in fresh[-n:] if r["st"] == "ok")
            return (ok[len(ok) // 2] if ok else None), fresh[-n:]
    return None, []


def check_offset(offset_mm):
    """The agent's aim correction: two numbers, each within MAX_OFFSET_MM. Returns (offset, error)."""
    try:
        dx, dy = (float(v) for v in (offset_mm or (0.0, 0.0)))
    except (TypeError, ValueError):
        return None, "offset_mm must be two numbers: [dx_mm forward, dy_mm left]"
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None, "offset_mm must be finite"
    if abs(dx) > MAX_OFFSET_MM or abs(dy) > MAX_OFFSET_MM:
        return None, f"offset ({dx:.0f}, {dy:.0f}) mm is beyond the {MAX_OFFSET_MM:.0f} mm limit per axis"
    return (dx, dy), None


def approach(arm, poses, *, step_m=0.005, max_steps=20, offset_mm=(0.0, 0.0)):
    """Step down 5 mm at a time over the block, settling on 3 readings each step, until the median
    distance is inside the switch distance; then a fixed final move to the grasp pose. A reading far
    below what the pose predicts aborts with a finding (the watchdog opens the incident).
    offset_mm moves the target from the taught block spot (the agent's correction, already bounded)."""
    from .model import BLOCK_POS, BLOCK_TOP
    x, y = BLOCK_POS[0] + offset_mm[0] / 1000.0, BLOCK_POS[1] + offset_mm[1] / 1000.0
    above, pick = poses["above_block"], poses["pick"]
    if any(offset_mm):
        seed, above, why = list(poses["above_block"]), None, ""
        for h in (0.06, 0.05, 0.04):              # far out, the wrist can't point down from as high
            try:
                above = arm.solve((x, y, BLOCK_TOP + h), seed=seed, iters=400)
                break
            except ValueError as exc:
                why = str(exc)
        try:
            if above is None:
                raise ValueError(why)
            pick = arm.solve((x, y, BLOCK_TOP - 0.015), seed=seed, iters=400)
        except ValueError as exc:
            return {"arrived": False, "aborted": f"can't reach that offset: {exc}"}
        above[5] = pick[5] = prims.OPEN
    start = arm.goto(above)                   # always start over the target with the jaws open
    if not start.get("arrived"):
        return {**start, "aborted": "couldn't get over the block to start the approach"}
    tip, _ = arm.tip()
    readings, invalid = [], 0
    for step in range(max_steps + 1):
        d, rows = _median_reading(arm)
        if arm.halted():
            return arm._stop()
        if d is None:
            invalid += len(rows) or 3
            if invalid >= 6:
                return {"arrived": False, "aborted": "6 invalid readings in a row", "readings": readings}
        else:
            invalid = 0
            readings.append(d)
            pred = rows[-1].get("pred_mm")
            if pred is not None and pred - d > 20:
                return {"arrived": False, "aborted": f"reading {d} mm is far below the expected {pred:.0f} mm",
                        "readings": readings, "finding": level1.residual_finding(d, pred)}
            if d <= SWITCH_MM:
                r = arm.goto(pick, duration=0.8)                   # the fixed final move
                return {**r, "steps": step, "readings": readings, "switched_at_mm": d,
                        "offset_mm": list(offset_mm)}
        tip[2] -= step_m
        try:
            pose = arm.solve((x, y, tip[2]), seed=list(arm.read().values()))
        except ValueError as exc:
            return {"arrived": False, "aborted": str(exc), "readings": readings}
        r = arm.goto(pose, duration=0.3)
        if r.get("halted"):
            return {**r, "readings": readings}
    return {"arrived": False, "aborted": f"no switch distance after {max_steps} steps", "readings": readings}


def _lerp_pose(a, b, f):
    return [x + (y - x) * f for x, y in zip(a, b)]


def _as_list(pose):
    return [pose[j] for j in JOINTS] if isinstance(pose, dict) else list(pose)


def approach_joint(arm, poses):
    """No distance sensor (the real arm): over the block, then the taught pick pose, in joint space."""
    start = arm.goto(poses["above_block"])
    if not start.get("arrived"):
        return {**start, "aborted": "couldn't get over the block to start the approach", "mode": "joint"}
    return {**arm.goto(poses["pick"], duration=1.0), "mode": "joint"}


def lift(arm, poses, *, up_m=0.03):
    """Lift 30 mm straight up, then check the grasp still holds. Without IK (the real arm), go
    halfway back along the taught pick -> above_block line instead, keeping the grip's squeeze."""
    if not getattr(arm, "use_ik", True):
        if poses.get("pick") is None or poses.get("above_block") is None:
            return {"arrived": False, "aborted": "teach pick and above_block first", "grasp": arm.grasp_check()}
        target = _lerp_pose(_as_list(poses["pick"]), _as_list(poses["above_block"]), 0.5)
        target[5] = arm.commanded()[5]
        r = arm.goto(target, duration=0.8)
        arm.hold(0.3)
        return {**r, "grasp": arm.grasp_check(), "mode": "joint"}
    tip, _ = arm.tip()
    try:
        pose = arm.solve((tip[0], tip[1], tip[2] + up_m), seed=list(arm.read().values()))  # keeps the squeeze
    except ValueError as exc:
        return {"arrived": False, "aborted": str(exc), "grasp": arm.grasp_check()}
    r = arm.goto(pose, duration=0.8)
    arm.hold(0.3)
    return {**r, "grasp": arm.grasp_check()}


SENSOR_KINDS = ("us_frozen", "us_stuck", "us_scale", "us_offset", "us_noisy", "us_missing_echo", "us_residual")


class Watchdog:
    """Plain code, no model: level-1 findings over the recent rows open incidents. A fix is only
    trusted after `verify()`: a move home and fresh readings that pass the checks."""

    REPAIRS = ("restart_sensor", "verify")

    def __init__(self, arm, rec):
        self.arm, self.rec, self.incidents = arm, rec, []
        self.clock = lambda: arm.t
        self.lock = threading.RLock()
        self.home = None                      # the pose verify() checks from (set by serve)
        arm.watchers.append(self)

    def open(self):
        with self.lock:
            return [i for i in self.incidents if i["state"] == "open"]

    def __call__(self, arm):
        if self.open():
            return
        snap = {"now": self.clock(), "D": list(arm.d_rows), "holding": arm.grasp_check()["holding"]}
        found = [f for f in level1.run(snap) if f["severity"] in ("critical", "warn")]
        if found:
            self._open(found[0])

    def arrival(self, result):
        """After a skill: a finding it returned, or a move that didn't arrive (joint_stall)."""
        if self.open():
            return None
        if result.get("finding"):
            return self._open(result["finding"])
        load = self.arm.load_nm(result["worst_joint"]) if result.get("worst_joint") else None
        found = level1.check_arrival(result, load_nm=load)
        return self._open(found[0]) if found else None

    def _open(self, f):
        with self.lock:
            inc = {"id": f"inc-{len(self.incidents) + 1}", "state": "open", "opened_at": round(self.clock(), 2),
                   "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **f}
            self.incidents.append(inc)
            self.arm.halt_reason = f["kind"]
        print(f"WATCHDOG {inc['id']} {f['kind']}: {f['summary']}", flush=True)
        return inc

    def repair(self, action):
        """The healer's tools. restart_sensor: restart the wrist sensor (in sim, clears a planted freeze).
        verify: no fix, just re-check. Either way, verify() decides what is fixed."""
        if action not in self.REPAIRS:
            return {"ok": False, "error": f"unknown repair {action!r}; known: {list(self.REPAIRS)}"}
        if action == "restart_sensor":
            if not getattr(self.arm, "has_sensor", True):
                return {"ok": False, "error": "this arm has no wrist distance sensor to restart"}
            self.arm.clear("us_frozen")
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for inc in self.open():
            inc.setdefault("actions", []).append({"tool": f"arm_{action}", "at": at})
        return {"ok": True, "action": action, **self.verify(f"repair: {action}")}

    def verify(self, why):
        """Go home with the jaws open even though motion is halted, hold still on fresh readings,
        then close what the checks say is fixed. Needs-hands faults are handed to a person."""
        if not self.open():
            self.arm.hold(0.6)                    # no incident: refresh the servo status bits
            return {"verified": True, "open": [], "status": self.arm.status()}
        arm = self.arm
        arm._parking = True                       # the watchdog's own move is allowed while halted
        try:
            moved = arm.goto(self.home) if self.home is not None else {"arrived": True}
        finally:
            arm._parking = False
        arm.d_rows.clear()
        arm.hold(3.5)
        snap = {"now": self.clock(), "D": list(arm.d_rows)}
        still = [f for f in level1.run(snap) if f["severity"] in ("critical", "warn")]
        rows_ok = sum(1 for r in arm.d_rows if r.get("st") == "ok")
        need_rows = 10 if getattr(arm, "has_sensor", True) else 0
        resolved, handoffs = [], []
        with self.lock:
            for inc in self.open():
                if inc["kind"] == "joint_stall":
                    fixed = bool(moved.get("arrived"))
                elif inc["kind"] in SENSOR_KINDS:
                    fixed = not still and rows_ok >= need_rows
                else:
                    fixed = not still
                if fixed:
                    inc.update(state="resolved", resolved_at=round(self.clock(), 2), fix=why)
                    resolved.append(inc)
                else:
                    # Needs hands: a joint that still won't move after a checked retry, or a sensor
                    # that fails two checks. The arm stays parked until a person's check passes.
                    inc["verify_failures"] = inc.get("verify_failures", 0) + 1
                    if not inc.get("handed_off") and (inc["kind"] == "joint_stall" or inc["verify_failures"] >= 2):
                        inc.update(handed_off=round(self.clock(), 2), handoff_reason=(
                            "still failing after a checked retry: " + ("; ".join(f["summary"] for f in still)
                                                                     or f"home move arrived={moved.get('arrived')}")))
                        handoffs.append(inc)
            if not self.open():
                arm.halt_reason = None
        for inc in handoffs:
            print(f"WATCHDOG {inc['id']} handed to a person: {inc['handoff_reason']}", flush=True)
        for inc in resolved:
            print(f"WATCHDOG {inc['id']} resolved ({why})", flush=True)
        return {"verified": not self.open(), "home_arrived": moved.get("arrived"), "ok_rows": rows_ok,
                "still": [f["summary"] for f in still], "resolved": [i["id"] for i in resolved],
                "handed_off": [i["id"] for i in handoffs], "open": [i["id"] for i in self.open()]}

    def handoff(self, iid, reason):
        """A person takes an open incident: the arm stays parked until a person's `repair verify`."""
        with self.lock:
            inc = next((i for i in self.open() if i["id"] == iid), None)
            if inc is None:
                return {"ok": False, "error": f"no open incident {iid!r}"}
            if not inc.get("handed_off"):
                inc.update(handed_off=round(self.clock(), 2), handoff_reason=reason)
                print(f"WATCHDOG {iid} handed to a person: {reason}", flush=True)
        return {"ok": True, "id": iid, "handed_off": inc["handed_off"], "handoff_reason": inc["handoff_reason"]}


def _brief(inc):
    keys = ("id", "kind", "severity", "summary", "at", "handed_off", "handoff_reason", "next_steps")
    return {k: inc[k] for k in keys if inc.get(k) is not None}


def serve(*, arm_kind="sim", gif=False, view=False, realtime=False, sock=None):
    """arm_kind="sim": the MuJoCo SO-101 with a simulated wrist ultrasonic and plantable faults.
    arm_kind="real": a real SO-101 through LeRobot ($SO101_PORT, $SO101_ID), with taught poses."""
    sock = Path(sock or SOCK)
    runs = data_dir() / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    rec = RunRecorder(runs, task="arm host session", source=arm_kind, arm=f"{arm_kind}-so101", joints=JOINTS)
    if arm_kind == "real":
        from .real import RealArm, taught_library
        poses, motions = taught_library()
        arm = RealArm(recorder=rec).connect()
        arm.taught_poses = poses
        skills = Skills(arm, poses=poses, motions=motions)
        taught = sorted(n for n, v in poses.items() if v is not None)
        print(f"real arm connected; taught poses: {taught or 'none'}; motions: {sorted(motions) or 'none'}", flush=True)
    else:
        arm = SimArm(recorder=rec, start=prims.REST)
        skills = Skills(arm)
    dog = Watchdog(arm, rec)
    dog.home = skills.p.get("home")
    if gif and arm_kind == "sim":
        from .render import Gif
        gif = Gif(arm, runs / f"{rec.run_id}.gif", every=10,
                  caption=lambda a: ("PARKED: " + dog.open()[0]["summary"]) if dog.open() else "")
    viewer = None
    if view and arm_kind == "sim":
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(arm.m, arm.d)
        arm.hooks.append(lambda a: viewer.sync())
    if (view or realtime) and arm_kind == "sim":
        clock = {"wall": time.monotonic(), "sim": arm.t}

        def pace(a):
            ahead = (a.t - clock["sim"]) - (time.monotonic() - clock["wall"])
            if ahead < -0.25:                         # the host sat idle: re-anchor, don't sprint
                clock.update(wall=time.monotonic(), sim=a.t)
            elif ahead > 0:
                time.sleep(ahead)
        arm.hooks.append(pace)

    sock.parent.mkdir(parents=True, exist_ok=True)
    if sock.exists():
        sock.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock))
    os.chmod(sock, 0o600)                             # only this user can command the arm
    srv.listen(4)
    srv.settimeout(0.5)
    state = {"parked": True, "frozen": None, "eye": None}
    print(f"so101 arm host ({arm_kind}) listening on {sock}; run record {rec.path}", flush=True)

    def park():
        state["parked"] = True
        return skills.park()

    def after_move(name, result):
        reply = {"ok": True, "skill": name, "result": result, "t": round(arm.t, 2)}
        if not result.get("halted") and name != "park":
            dog.arrival(result)
        if result.get("halted") or dog.open():
            reply.update(parked=park(), incident=_brief(dog.open()[0]) if dog.open() else None)
        elif name == "park":
            state["parked"] = True
        return reply

    def refuse_motion(name):
        if state["frozen"] and name != "park":
            return {"ok": False, "error": f"the arm is frozen by the operator ({state['frozen']}); "
                                          "only a person can unfreeze it"}
        if dog.open() and name != "park":
            return {"ok": False, "error": "incident open; the arm stays parked until a repair is verified",
                    "incidents": [_brief(i) for i in dog.open()]}
        return None

    def handle(req):
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "arm": arm_kind, "t": arm.t}
        if op == "skill":
            name = req.get("name")
            if name not in Skills.MENU:
                return {"ok": False, "error": f"unknown skill {name!r}; the menu is {list(Skills.MENU)}"}
            offset = None
            if req.get("offset_mm") is not None:
                if name != "approach":
                    return {"ok": False, "error": "offset_mm applies to approach only"}
                offset, why = check_offset(req["offset_mm"])
                if why:
                    return {"ok": False, "error": why}
            no = refuse_motion(name)
            if no:
                return no
            state["parked"] = name == "park"
            result = skills.approach(offset) if offset is not None else getattr(skills, name)()
            return after_move(name, result)
        if op == "play":
            no = refuse_motion("play")
            if no:
                return no
            speed = min(max(float(req.get("speed") or 1.0), 0.25), 1.0)     # never faster than recorded
            state["parked"] = False
            return after_move(f"play:{req.get('motion')}", skills.play(req.get("motion"), speed=speed))
        if op == "motions":
            return {"ok": True, "motions": sorted(skills.motions)}
        if op == "look":
            if arm_kind != "sim":
                return {"ok": False, "error": "no camera on the real arm in this version"}
            aim, why = check_offset(req.get("aim_mm") or (0.0, 0.0))
            if why:
                return {"ok": False, "error": why}
            try:
                if state.get("eye") is None:
                    from .camera import Eye
                    state["eye"] = Eye(arm, data_dir() / "frames")
                seen = state["eye"].look(aim)
                far = seen.get("block_mm") and max(abs(v) for v in seen["block_mm"]) > MAX_OFFSET_MM
                if far:                               # the camera sees it, but no correction may reach it
                    seen["to_pick_it"] = ("none: the block is out of reach. Don't pick; tell the user to move it "
                                          "back near the marked spot (0, 0 on the map).")
                    seen["reachable"] = False
                    seen["note"] = (f"The block is more than {MAX_OFFSET_MM:g} mm from the taught spot, beyond what "
                                    "the arm may correct (a limit in code). Don't try to pick it: tell the user "
                                    "it needs to go back near the marked spot (0, 0 on the map).")
                elif seen.get("block_mm"):
                    seen["reachable"] = True
                if req.get("inline"):                 # the top view's bytes, for a tool result the model sees
                    seen["top_png_b64"] = base64.b64encode(Path(seen["top"]).read_bytes()).decode()
                return {"ok": True, **seen}
            except Exception as exc:                  # no OpenGL here: the arm still works, only the camera doesn't
                return {"ok": False, "error": f"the sim camera can't render here ({type(exc).__name__}: {exc}). "
                                              "On headless Linux set MUJOCO_GL=egl (or osmesa) before starting "
                                              "the host; on macOS no setting is needed."}
        if op == "read":
            return {"ok": True, "pos": arm.read()}
        if op == "distance":
            return {"ok": True, "row": arm.distance()}
        if op == "status":
            return {"ok": True, "arm": arm_kind, "status": arm.status(), "grasp": arm.grasp_check(),
                    "parked": state["parked"], "frozen": state["frozen"], "has_sensor": getattr(arm, "has_sensor", True),
                    "incidents": [_brief(i) for i in dog.open()],
                    "handed_off": [_brief(i) for i in dog.open() if i.get("handed_off")]}
        if op == "incidents":
            return {"ok": True, "incidents": dog.incidents[-50:]}
        if op == "repair":
            if state["frozen"]:
                return {"ok": False, "error": f"the arm is frozen by the operator ({state['frozen']})"}
            return dog.repair(req.get("action"))
        if op == "handoff":
            return dog.handoff(req.get("id"), req.get("reason") or "handed to a person by the agent")
        if op == "freeze":
            state["frozen"] = f"{req.get('by') or 'operator'} at {datetime.now().strftime('%H:%M:%S')}"
            reply = {"ok": True, "frozen": state["frozen"]}
            if not state["parked"]:
                reply["parked"] = park()
            return reply
        if op == "unfreeze":
            state["frozen"] = None
            return {"ok": True, "frozen": None}
        if op == "inject":
            if arm_kind != "sim":
                return {"ok": False, "error": "faults are planted in the sim only"}
            fault = req.get("fault")
            known = ("us_frozen", "us_offset", "us_scale", "us_noisy", "joint_stuck", "joint_offset",
                     "joint_reversed", "block_shift")
            if fault not in known:
                return {"ok": False, "error": f"unknown fault {fault!r}; known: {list(known)}"}
            return {"ok": True, "fault": arm.inject(fault, **req.get("params", {}))}
        if op == "clear":
            arm.clear(req.get("fault"))
            return {"ok": True}
        if op == "shutdown":
            return {"ok": True, "bye": True}
        return {"ok": False, "error": f"unknown op {op!r}"}

    def chores():
        """Idle work on the main thread: an incident while unparked parks the arm."""
        try:
            if dog.open() and not state["parked"]:
                park()
        except Exception as exc:                      # a servo-bus read failing must not stop the host
            print(f"CHORES failed ({type(exc).__name__}: {exc}); still serving", flush=True)

    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:                    # idle
                chores()
                continue
            conn.settimeout(30)                       # a silent client can't hold the host
            try:
                with conn, conn.makefile("rw", encoding="utf-8") as f:
                    for line in f:
                        try:
                            reply = handle(json.loads(line))
                        except Exception as exc:      # report, keep serving
                            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        print(json.dumps({"req": line.strip()[:120], "reply": reply}, default=str)[:400], flush=True)
                        f.write(json.dumps(reply, default=str) + "\n")
                        f.flush()
                        if reply.get("bye"):
                            raise SystemExit
            except (BrokenPipeError, ConnectionResetError, TimeoutError) as exc:
                print(f"client gone before the reply ({type(exc).__name__}); still serving", flush=True)
            chores()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        srv.close()
        sock.unlink(missing_ok=True)
        rec.close("host stopped", findings=[{k: i[k] for k in ("kind", "severity", "summary")} for i in dog.incidents])
        if arm_kind == "real":
            try:
                arm.disconnect()
            except Exception:
                pass
        if gif:
            print("gif:", gif.save(), flush=True)
        if viewer:
            viewer.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the SO-101 arm host (sim or real).")
    ap.add_argument("--real", action="store_true", help="drive a real SO-101 through LeRobot ($SO101_PORT, $SO101_ID)")
    ap.add_argument("--gif", action="store_true", help="sim: save a GIF of the session next to the run record")
    ap.add_argument("--view", action="store_true", help="sim: open MuJoCo's viewer (on macOS, run with mjpython)")
    ap.add_argument("--realtime", action="store_true", help="sim: pace to wall-clock time")
    ap.add_argument("--sock", default=None, help="socket path (default $SO101_SOCK or <temp>/so101-arm.sock)")
    a = ap.parse_args(argv)
    serve(arm_kind="real" if a.real else "sim", gif=a.gif, view=a.view, realtime=a.realtime, sock=a.sock)


if __name__ == "__main__":
    main()
