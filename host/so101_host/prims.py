"""The sim's poses and motions, in one JSON format: {name, fps, joints, duration_s, meta,
frames: [[5 arm joints in degrees, gripper 0-100]]}. The same format holds motions taught on a real arm.
"""
import json
import math
from pathlib import Path

from .arm import JOINTS
from .model import BLOCK_POS, BLOCK_TOP

LIB = Path(__file__).resolve().parent / "library"

# Folded and low: park here before torque off. Wrist roll stays at the working 90.8 deg; a rest
# pose with roll 0 put the jaws on the floor, and rolling out of it stalled the wrist.
REST = [0.0, -95.0, 90.0, 55.0, 90.8, 0.0]
PRESENT = [0.0, -45.0, 30.0, -20.0, 90.0, 60.0]  # raised, facing out: waves start here
OPEN, CLOSED = 60.0, 0.0


def load(name):
    return json.loads((LIB / f"{name}.json").read_text(encoding="utf-8"))


def poses():
    return load("poses")["poses"]


def save(obj, name):
    LIB.mkdir(parents=True, exist_ok=True)
    (LIB / f"{name}.json").write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def make_poses(arm):
    """Named poses, solved with MuJoCo's own kinematics for the block in the scene.

    Reach limit: pointing straight down, the fingertips can't rise above ~9 cm at 0.17-0.25 m
    out, because wrist flex tops out near 95 deg. So the approach starts 6 cm over a 3 cm block,
    not a higher hover.
    """
    x, y, _ = BLOCK_POS
    seed = [0.0, 0.0, 27.1, 67.4, 90.8, OPEN]    # Menagerie's own pickup keyframe
    def at(px, z, grip):
        pose = arm.solve((px, y, z), seed=seed, iters=400)
        pose[5] = grip
        return [round(v, 2) for v in pose]
    return {
        "rest": REST,
        "present": PRESENT,
        "home": at(x - 0.04, 0.08, OPEN),               # ready: highest straight-down pose, 4 cm back
        "above_block": at(x, BLOCK_TOP + 0.06, OPEN),   # where the approach starts
        "pick": at(x, BLOCK_TOP - 0.015, OPEN),         # jaws around the top 1.5 cm of the block
    }


def make_wave(start, *, cycles=3, hz=1.25, amp_deg=25.0, fps=50):
    n = round(cycles / hz * fps)
    frames = []
    for i in range(n):
        s = math.sin(2.0 * math.pi * hz * i / fps)
        f = list(start)
        f[3] = start[3] + amp_deg * s     # wrist flex
        f[0] = start[0] + 8.0 * s         # a little pan sway
        frames.append([round(v, 2) for v in f])
    return {"name": "wave", "fps": fps, "joints": list(JOINTS), "duration_s": round(n / fps, 2),
            "meta": {"source": "so101_host, synthetic", "start_pose": "present"},
            "frames": frames}
