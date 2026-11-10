"""The sim's eyes for the agent: a top-down view of the work area with a 10 mm grid drawn on the table,
plus a side view. The agent reads the block's position off the grid and corrects its aim with a bounded
offset (arm_approach dx_mm/dy_mm), the way a person would with a ruler.

The grid and the aim mark are drawn as geometry in the 3D scene, so they line up with the block exactly;
only the text labels are placed by projection.
"""
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .model import BLOCK_POS, BLOCK_TOP

SIZE = 480
TOP = {"distance": 0.30, "azimuth": 0.0, "elevation": -90.0}        # forward (+x) is up in the image
SIDE = {"distance": 0.58, "azimuth": 235.0, "elevation": -24.0}


def _font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _line(scene, a, b, rgba, width, glow=0.0):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.array(a, float), np.array(b, float))
    g.emission = glow                                       # marks keep their colour under the scene's lights
    scene.ngeom += 1


class Eye:
    def __init__(self, arm, out_dir):
        self.arm, self.out = arm, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.r = mujoco.Renderer(arm.m, SIZE, SIZE)
        self.n = 0
        self.font, self.small = _font(15), _font(12)

    def _camera(self, spec, lookat):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = spec["distance"], spec["azimuth"], spec["elevation"]
        return cam

    def _px(self, x, y, z=0.0):
        """World point -> top-view pixel (straight-down camera at TOP['distance'] over the block)."""
        fovy = math.radians(self.arm.m.vis.global_.fovy)
        s = (SIZE / 2) / ((TOP["distance"] - z) * math.tan(fovy / 2))
        return SIZE / 2 - (y - BLOCK_POS[1]) * s, SIZE / 2 - (x - BLOCK_POS[0]) * s

    def look(self, aim_mm=(0.0, 0.0)):
        """Save a top view (with the grid and the aim) and a side view. Returns their paths."""
        self.n += 1
        x0, y0 = BLOCK_POS[0], BLOCK_POS[1]
        ax, ay = x0 + aim_mm[0] / 1000.0, y0 + aim_mm[1] / 1000.0
        opt = mujoco.MjvOption()
        top_opt = mujoco.MjvOption()
        top_opt.geomgroup[1:5] = 0                             # the table map: arm hidden, so nothing covers the grid
        top_opt.sitegroup[:] = 0
        top_opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.r.update_scene(self.arm.d, self._camera(TOP, (x0, y0, 0.0)), top_opt)
        floor = [g for g in range(self.arm.m.ngeom) if self.arm.m.geom_bodyid[g] == 0]
        for i in range(self.r.scene.ngeom):                    # a plain table: no checker lines across the grid
            g = self.r.scene.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in floor:
                g.matid = -1
                g.reflectance = 0.0                            # no mirror image of the grid on the floor
                g.rgba[:] = (0.16, 0.20, 0.28, 1.0)
        z = BLOCK_TOP + 0.0008      # the grid lies in the plane of the block's top face: no parallax when reading it
        for k in range(-6, 7):                                  # a 10 mm grid, +-60 mm around the taught spot
            strong = k % 5 == 0
            rgba, width = ((0.95, 0.95, 1.0, 0.95), 2.0) if strong else ((0.75, 0.8, 0.9, 0.6), 1.0)
            off = k / 100.0
            _line(self.r.scene, (x0 - 0.06, y0 + off, z), (x0 + 0.06, y0 + off, z), rgba, width)
            _line(self.r.scene, (x0 + off, y0 - 0.06, z), (x0 + off, y0 + 0.06, z), rgba, width)
        tip, _ = self.arm.tip()                                 # where the jaws are now (blue)
        blue = (0.35, 0.65, 1.0, 1.0)
        for s in (1, -1):
            _line(self.r.scene, (tip[0] - 0.006, tip[1] - s * 0.006, z + 0.0025),
                  (tip[0] + 0.006, tip[1] + s * 0.006, z + 0.0025), blue, 4.0, glow=1.0)
        red = (1.0, 0.15, 0.15, 1.0)                            # the aim: where arm_approach will go
        _line(self.r.scene, (ax - 0.011, ay, z + 0.0015), (ax + 0.011, ay, z + 0.0015), red, 5.0, glow=1.0)
        _line(self.r.scene, (ax, ay - 0.011, z + 0.0015), (ax, ay + 0.011, z + 0.0015), red, 5.0, glow=1.0)
        top = Image.fromarray(self.r.render()).convert("RGB")
        d = ImageDraw.Draw(top, "RGBA")
        for k in (-50, 0, 50):
            px, py = self._px(x0 + k / 1000.0, y0 - 0.062, z)
            d.text((px + 4, py - 8), f"x{k:+d}" if k else "x 0", font=self.small, fill=(255, 255, 255, 230))
            px, py = self._px(x0 - 0.062, y0 + k / 1000.0, z)
            d.text((px - 14, py + 2), f"y{k:+d}" if k else "y 0", font=self.small, fill=(255, 255, 255, 230))
        d.rectangle([0, 0, SIZE, 24], fill=(10, 14, 22, 210))
        d.text((8, 4), f"table map (arm hidden) · grid 10 mm · red + aim ({aim_mm[0]:+.0f}, {aim_mm[1]:+.0f}) mm · blue x jaws",
               font=self.small, fill=(240, 244, 252))
        d.rectangle([0, SIZE - 22, SIZE, SIZE], fill=(10, 14, 22, 210))
        d.text((8, SIZE - 19), "up = forward (+x, away from the base) · left = +y", font=self.small,
               fill=(220, 226, 236))
        top_path = self.out / f"look-{self.n:03d}-top.png"
        top.save(top_path)

        self.r.update_scene(self.arm.d, self._camera(SIDE, (0.14, 0.0, 0.08)), opt)
        side = Image.fromarray(self.r.render()).convert("RGB")
        side_path = self.out / f"look-{self.n:03d}-side.png"
        side.save(side_path)
        return {"top": str(top_path), "side": str(side_path), "grid_mm": 10, "aim_mm": list(aim_mm),
                "how_to_read": "In the top view, forward (+x) is up and +y is left. Find the block's centre, "
                               "count grid squares (10 mm each) from the red aim mark, and pass that offset as "
                               "dx_mm (forward) and dy_mm (left) to arm_approach or arm_pick_up."}
