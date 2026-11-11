"""Rehearsal: try candidate aims on simulated copies of the arm before the main arm tries again.

Each copy is a fresh sim (its own model, so no fault planted on the main arm carries over) with the block
placed where the main arm's camera measured it, never where the main sim really has it: the copies know
only what the arm saw. Each copy runs the same pick the agent would (home, approach at the aim, grip,
lift) and reports whether it held. A copy's pick takes about 0.15 s of CPU (the physics runs ~70x faster
than real time), so 16 candidates take a few seconds on a laptop; MuJoCo Warp or MJX on a GPU is the way
to thousands.
"""
import base64
import io

import mujoco
from PIL import Image, ImageDraw

from . import prims
from .arm import SimArm
from .model import BLOCK_POS, build

MAX_COPIES = 16
TILE_W, TILE_H = 240, 180


def _camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [BLOCK_POS[0], BLOCK_POS[1], 0.03]
    cam.distance, cam.azimuth, cam.elevation = 0.34, 215.0, -28.0
    return cam


def _pick(model, block_mm, aim_mm, path=None):
    """One copy: the block where it was seen, then the agent's pick at this aim. Returns (result, copy).
    With `path` (a list), the copy's pose at every control tick is appended to it, for a replay."""
    from .host import Skills                                  # host imports this module: import late
    copy = SimArm(model, start=prims.REST)
    copy.shift_block(*block_mm)
    if path is not None:
        copy.hooks.append(lambda a: path.append(a.d.qpos.copy()))
    skills = Skills(copy)
    skills.go_home()
    steps = {"approach": skills.approach(tuple(aim_mm))}
    if steps["approach"].get("arrived"):
        steps["grip"] = skills.grip()
        if (steps["grip"].get("grasp") or {}).get("holding"):
            steps["lift"] = skills.lift()
    held = bool((steps.get("lift") or {}).get("grasp", {}).get("holding"))
    return {"dx_mm": aim_mm[0], "dy_mm": aim_mm[1], "held": held,
            "failed_at": None if held else list(steps)[-1]}, copy


def _best(results, block_mm):
    """The aim to send to the main arm: of the copies that held, the one nearest where the block was seen.
    None if none held. (The middle of the held aims would lean toward whatever edge went untried.)"""
    held = [r for r in results if r["held"]]
    if not held:
        return None
    return min(held, key=lambda r: (r["dx_mm"] - block_mm[0]) ** 2 + (r["dy_mm"] - block_mm[1]) ** 2)


def around(block_mm, step=8.0, limit=40.0):
    """Nine aims: where the block was seen and a step to each side, each kept within the arm's limit."""
    clip = lambda v: max(-limit, min(limit, round(v, 1)))
    return [(clip(block_mm[0] + i * step), clip(block_mm[1] + j * step)) for i in (-1, 0, 1) for j in (-1, 0, 1)]


def rehearse(block_mm, candidates, *, picture=True, show=None):
    """Run each candidate aim on its own copy. Returns the ranked results, the best aim, and a contact
    sheet of the copies' last frames (PNG, base64) when `picture`. With `show`, every copy's motion is
    handed over afterwards as show(model, paths, held), for the viewer to replay them side by side."""
    model = build()
    renderer = mujoco.Renderer(model, TILE_H, TILE_W) if picture else None
    results, tiles, paths = [], [], []
    for aim in candidates[:MAX_COPIES]:
        path = [] if show else None
        result, copy = _pick(model, block_mm, aim, path)
        paths.append(path)
        results.append(result)
        if renderer:
            renderer.update_scene(copy.d, _camera())
            tile = Image.fromarray(renderer.render()).convert("RGB")
            d = ImageDraw.Draw(tile)
            d.rectangle([0, 0, TILE_W, 20], fill=(255, 255, 255) if result["held"] else (0, 0, 0))
            d.text((6, 4), f"#{len(results)}  dx {aim[0]:+g}  dy {aim[1]:+g}  " + ("HELD" if result["held"] else
                    f"missed ({result['failed_at']})"), fill=(0, 0, 0) if result["held"] else (255, 255, 255))
            tiles.append(tile)
    if show:
        show(model, paths, [r["held"] for r in results])
    best = _best(results, block_mm)
    out = {"copies": len(results), "held": sum(r["held"] for r in results), "block_mm_seen": list(block_mm),
           "best": best and {"dx_mm": best["dx_mm"], "dy_mm": best["dy_mm"]}, "results": results}
    if tiles:
        cols = 4 if len(tiles) > 4 else 2 if len(tiles) > 1 else 1
        rows = -(-len(tiles) // cols)
        sheet = Image.new("RGB", (cols * TILE_W + (cols - 1) * 4, rows * TILE_H + (rows - 1) * 4), (0, 0, 0))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * (TILE_W + 4), (i // cols) * (TILE_H + 4)))
        buf = io.BytesIO()
        sheet.save(buf, "PNG")
        out["sheet_png_b64"] = base64.b64encode(buf.getvalue()).decode()
    if renderer:
        renderer.close()
    return out
