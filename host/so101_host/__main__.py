"""python -m so101_host <command>

  serve [--real] [--gif] [--view] [--realtime]   run the arm host (default: the MuJoCo sim)
  status                                          ask the running host how the arm is
  fault <name> [key=value ...]                    sim: plant a fault for a demo (us_frozen, us_scale
                                                  factor=0.5, us_offset mm=40, us_noisy us=30,
                                                  joint_stuck joint=2, joint_offset joint=1 deg=15)
  clear [<name>]                                  sim: clear planted faults
  freeze [--by NAME] / unfreeze [--by NAME]       the operator's kill switch: freeze parks the arm and
                                                  refuses every move until a person unfreezes it
  check                                           a person's check at home after a fix (repair verify)
  teach <pose> [<pose> ...]                       real arm: torque off, move it by hand, Enter to save
                                                  (rest, home, above_block, pick)
  record <motion> [--seconds 10]                  real arm: torque off, record a motion by hand at 50 fps
  poses                                           sim: re-solve the sim's poses and its wave
  fetch-model                                     download MuJoCo Menagerie's SO-101 model (git, Apache-2.0)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

USAGE = __doc__

# MuJoCo renders offscreen through EGL on a headless Linux box (the camera tool); a Mac needs nothing.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def _call(op, **fields):
    from .client import call, socket_path
    try:
        return call(op, **fields)
    except OSError as exc:
        raise SystemExit(f"no arm host at {socket_path()} ({exc}); start it: python -m so101_host serve")


def _kv(args):
    out = {}
    for a in args:
        k, _, v = a.partition("=")
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = float(v)
    return out


def _flag(args, name, default=None):
    if name in args:
        i = args.index(name)
        return args[i + 1]
    return default


def cmd_poses():
    from . import prims
    from .arm import JOINTS, SimArm
    arm = SimArm()
    p = prims.make_poses(arm)
    prims.save({"units": "degrees; gripper 0-100 (0 = closed)", "joints": list(JOINTS), "poses": p}, "poses")
    prims.save(prims.make_wave(p["present"]), "wave")
    for name, pose in p.items():
        print(f"{name:12s} {pose}")


MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_SHA = "98a70c569c4c1b6da467a20012f766ee1f8dc521"     # the commit this sim was built and tested on


def cmd_fetch_model():
    from .model import MENAGERIE
    dest = MENAGERIE.parent
    if (MENAGERIE / "scene_box.xml").exists():
        print(f"{MENAGERIE} is already there")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"fetching robotstudio_so101 from {MENAGERIE_URL} at {MENAGERIE_SHA[:12]} into {dest}")
    git = ["git", "-C", str(dest)]
    for step in (["init", "-q"], ["remote", "add", "origin", MENAGERIE_URL],
                 ["sparse-checkout", "set", "robotstudio_so101"],
                 ["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_SHA],
                 ["checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(git + step, check=True)
    print("done")


def _real_arm():
    from .real import RealArm
    return RealArm().connect()


def cmd_teach(names):
    from .arm import JOINTS
    from .real import data_dir
    if not names:
        raise SystemExit("name the poses to teach: rest home above_block pick")
    home = data_dir()
    home.mkdir(parents=True, exist_ok=True)
    poses_file = home / "poses.json"
    poses = json.loads(poses_file.read_text()) if poses_file.exists() else {}
    arm = _real_arm()
    try:
        arm.relax()
        print("Torque is off. Support the arm: it will drop if you let go.")
        for name in names:
            input(f"Move the arm to '{name}' by hand, then press Enter... ")
            q = arm._read_vec()
            poses[name] = [round(float(v), 2) for v in q]
            print(f"  {name}: {dict(zip(JOINTS, poses[name]))}")
        poses_file.write_text(json.dumps(poses, indent=1) + "\n")
        print(f"saved {poses_file}")
    finally:
        arm.disconnect()


def cmd_record(name, seconds):
    from .arm import JOINTS
    from .real import data_dir
    out = data_dir() / "motions"
    out.mkdir(parents=True, exist_ok=True)
    arm = _real_arm()
    try:
        arm.relax()
        input(f"Torque is off. Press Enter, then move the arm through '{name}' for {seconds:g} s... ")
        frames, fps = [], 50
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            frames.append([round(float(v), 2) for v in arm._read_vec()])
            time.sleep(max(0.0, t0 + len(frames) / fps - time.monotonic()))
        motion = {"name": name, "fps": fps, "joints": list(JOINTS), "duration_s": round(len(frames) / fps, 2),
                  "meta": {"source": "recorded by hand on this arm"}, "frames": frames}
        path = out / f"{name}.json"
        path.write_text(json.dumps(motion) + "\n")
        print(f"saved {len(frames)} frames to {path}")
    finally:
        arm.disconnect()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args.pop(0) if args else ""
    if cmd == "serve":
        from .host import main as serve_main
        return serve_main(args)
    if cmd == "status":
        return print(json.dumps(_call("status"), indent=1))
    if cmd == "fault":
        if not args:
            raise SystemExit("name a fault, e.g. us_frozen or joint_stuck joint=2")
        return print(json.dumps(_call("inject", fault=args[0], params=_kv(args[1:]))))
    if cmd == "clear":
        return print(json.dumps(_call("clear", fault=args[0] if args else None)))
    if cmd in ("freeze", "unfreeze"):
        return print(json.dumps(_call(cmd, by=_flag(args, "--by", "operator"))))
    if cmd == "check":
        return print(json.dumps(_call("repair", action="verify"), indent=1))
    if cmd == "teach":
        return cmd_teach(args)
    if cmd == "record":
        if not args:
            raise SystemExit("name the motion: python -m so101_host record pack --seconds 12")
        return cmd_record(args[0], float(_flag(args, "--seconds", 10)))
    if cmd == "poses":
        return cmd_poses()
    if cmd == "fetch-model":
        return cmd_fetch_model()
    print(USAGE)
    return None


if __name__ == "__main__":
    main()
