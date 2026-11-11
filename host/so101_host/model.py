"""The simulated bench: MuJoCo Menagerie's SO-101 (`robotstudio_so101`), a block in reach,
and a wrist ultrasonic modelled as a rangefinder.

An HC-SR04 on the wrist bracket points along the gripper's approach axis, mounted 50 mm or more
behind the fingertips. The rangefinder site below copies that placement.

The model files come from MuJoCo Menagerie (Apache-2.0). `python -m so101_host fetch-model` puts them in
host/vendor/ at a pinned commit; $SO101_MENAGERIE points at another robotstudio_so101 folder.
"""
import os
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
MENAGERIE = Path(os.environ.get("SO101_MENAGERIE") or ROOT / "vendor" / "menagerie" / "robotstudio_so101")

# Where Menagerie's own "pickup" keyframe grasps the box: about 0.21 m in front of the base.
# A low, hard, flat-topped block: a 3 cm cube, about the smallest the sensor can see.
# Pointing straight down, the fingertips can't rise above ~9 cm here (wrist flex tops out near
# 95 deg), so a tall block leaves no room for a step-and-settle approach; 3 cm leaves about 6 cm.
# A 4 cm block didn't fit between the jaws: they landed on it and pushed it away.
BLOCK_HALF = (0.015, 0.015, 0.015)
BLOCK_POS = (0.21, 0.0, BLOCK_HALF[2])
BLOCK_TOP = BLOCK_POS[2] + BLOCK_HALF[2]

# On the gripper body the jaws point along -Z, and `gripperframe` sits at the fingertips
# (z = -0.098). The sensor goes on the jaw centreline, 50 mm behind them;
# quat (0,1,0,0) turns the site's +Z (the rangefinder's ray) to point along the approach axis.
# A mount 26 mm to the side never saw a 4 cm block. With the jaws closed, the moving jaw sits in
# the beam (reads ~7 mm), which is the real "jaw in the beam" failure, so approach with jaws open.
SENSOR_POS = (0.012, -0.000218, -0.048)
SENSOR_QUAT = (0.0, 1.0, 0.0, 0.0)
SENSOR_CUTOFF_M = 4.0  # HC-SR04 range is 2-400 cm


def _body(spec, name):
    finder = getattr(spec, "body", None) or getattr(spec, "find_body")
    body = finder(name)
    if body is None:
        raise KeyError(f"no body {name!r} in {MENAGERIE / 'scene_box.xml'}")
    return body


INK, PAPER = 0.08, 0.96        # the bench's look: black and white


def _black_and_white(spec):
    """A light, black-and-white bench: a white arm with black servos, a black block, a pale checker table
    and a white sky. Looks only; the camera's map repaints the block for its own measurement."""
    for mat in spec.materials:
        r, g, b, a = mat.rgba
        shade = PAPER if 0.3 * r + 0.59 * g + 0.11 * b > 0.4 else INK
        mat.rgba = [shade, shade, shade, a]
        if mat.name == "groundplane":
            mat.reflectance = 0.0
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb1, tex.rgb2 = [1.0, 1.0, 1.0], [0.85, 0.85, 0.85]
        elif tex.name == "groundplane":
            tex.rgb1, tex.rgb2, tex.markrgb = [0.95, 0.95, 0.95], [0.72, 0.72, 0.72], [0.55, 0.55, 0.55]
    spec.visual.rgba.haze = [1.0, 1.0, 1.0, 1.0]
    spec.visual.rgba.rangefinder = [INK, INK, INK, 1.0]
    finder = getattr(spec, "geom", None) or getattr(spec, "find_geom")
    finder("box").rgba = [INK, INK, INK, 1.0]


def build(block_pos=BLOCK_POS) -> mujoco.MjModel:
    if not (MENAGERIE / "scene_box.xml").exists():
        raise FileNotFoundError(f"{MENAGERIE} is missing; run: python -m so101_host fetch-model")
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "scene_box.xml"))
    _body(spec, "box").pos = list(block_pos)
    finder = getattr(spec, "geom", None) or getattr(spec, "find_geom")
    finder("box").size = list(BLOCK_HALF)
    for key in list(spec.keys):          # Menagerie's "pickup" keyframe expects the old box
        spec.delete(key)
    _black_and_white(spec)
    _body(spec, "gripper").add_site(
        name="ultrasonic", pos=list(SENSOR_POS), quat=list(SENSOR_QUAT),
        size=[0.006, 0.006, 0.006], rgba=[INK, INK, INK, 1.0], group=1,
    )
    spec.add_sensor(
        name="ultrasonic", type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ultrasonic", cutoff=SENSOR_CUTOFF_M,
        intprm=[1, 0, 0],  # MuJoCo 4.0: output the distance only (the XML parser's default)
    )
    return spec.compile()
