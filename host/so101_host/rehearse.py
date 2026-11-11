"""Rehearsal: try candidate aims on simulated copies of the arm before the main arm tries again.

Each copy is a fresh sim (its own data, so no fault planted on the main arm carries over) with the block
placed where the main arm's camera measured it, never where the main sim really has it: the copies know
only what the arm saw. Each copy runs the same pick the agent would (home, approach at the aim, grip,
lift) and reports whether it held.

When the arm is noisy (`joint_noise`, `camera_noise`), the copies get the same noise levels (never the
main arm's actual draws) and each aim is tried several times, so the result is a hit rate per aim and the
best aim is the most reliable one. The copies run together, one control tick at a time, so a viewer can
show them live (stage.py). A copy's pick takes about 0.15 s of CPU (the physics runs ~70x faster than real
time); MuJoCo Warp or MJX on a GPU is the way to thousands.
"""
import base64
import io
import threading
import time

import mujoco
from PIL import Image, ImageDraw

from . import prims
from .arm import SimArm
from .model import BLOCK_POS, build
from .stage import layout

MAX_AIMS, MAX_TRIALS, MAX_COPIES = 16, 4, 24
TILE_W, TILE_H = 240, 180
NOISE = ("joint_noise", "camera_noise")


def noise_levels(faults):
    """The arm's noise, as levels the copies can use: the planted faults' settings, never their draws."""
    return {k: {p: v for p, v in faults[k].items() if p != "at"} for k in NOISE if k in faults}


def _camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [BLOCK_POS[0], BLOCK_POS[1], 0.03]
    cam.distance, cam.azimuth, cam.elevation = 0.34, 215.0, -28.0
    return cam


class Gate:
    """Copies step together: each copy's tick waits here until the driver (the caller's thread) has drawn
    every copy and lets them all take the next step. A copy that ends leaves the gate."""

    def __init__(self, n):
        self.cv = threading.Condition()
        self.active, self.waiting, self.gen = n, 0, 0

    def tick(self, arm=None):                     # a copy's hook, in the copy's own thread
        with self.cv:
            gen = self.gen
            self.waiting += 1
            self.cv.notify_all()
            self.cv.wait_for(lambda: self.gen != gen)

    def leave(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def step(self, draw):
        """Wait for every running copy, draw them (they're all paused), let them go. False once none is left."""
        with self.cv:
            self.cv.wait_for(lambda: self.waiting >= self.active)
            if self.active == 0:
                return False
        draw()
        with self.cv:
            self.waiting, self.gen = 0, self.gen + 1
            self.cv.notify_all()
        return True


def _copy(model, block_mm, noise, seed):
    """A fresh copy: the block where it was seen (and, with a noisy camera, where it might really be)."""
    copy = SimArm(model, start=prims.REST, seed=seed)
    block = list(block_mm)
    if "camera_noise" in noise:
        sd = noise["camera_noise"].get("mm", 4.0)
        block = [v + copy.rng.normal(0.0, sd) for v in block]
    copy.shift_block(*block)
    if "joint_noise" in noise:
        copy.faults["joint_noise"] = dict(noise["joint_noise"])
    return copy


def _pick(copy, aim_mm):
    """The agent's pick on one copy: home, approach at the aim, grip, lift. Which step it stopped at."""
    from .host import Skills                                  # host imports this module: import late
    skills = Skills(copy)
    skills.go_home()
    steps = {"approach": skills.approach(tuple(aim_mm))}
    if steps["approach"].get("arrived"):
        steps["grip"] = skills.grip()
        if (steps["grip"].get("grasp") or {}).get("holding"):
            steps["lift"] = skills.lift()
    held = bool((steps.get("lift") or {}).get("grasp", {}).get("holding"))
    return {"held": held, "failed_at": None if held else list(steps)[-1]}


def _best(per_aim, block_mm):
    """The aim to send to the main arm: the highest hit rate, then the nearest to where the block was seen.
    None if no copy held. (The middle of the held aims would lean toward whatever edge went untried.)"""
    held = [r for r in per_aim if r["held"]]
    if not held:
        return None
    return max(held, key=lambda r: (r["held"] / r["trials"],
                                    -((r["dx_mm"] - block_mm[0]) ** 2 + (r["dy_mm"] - block_mm[1]) ** 2)))


def around(block_mm, step=8.0, limit=40.0):
    """Nine aims: where the block was seen and a step to each side, each kept within the arm's limit."""
    clip = lambda v: max(-limit, min(limit, round(v, 1)))
    return [(clip(block_mm[0] + i * step), clip(block_mm[1] + j * step)) for i in (-1, 0, 1) for j in (-1, 0, 1)]


def cross(block_mm, step=6.0, limit=40.0):
    """Five aims: where the block was seen, and a step forward, back, left and right of it."""
    clip = lambda v: max(-limit, min(limit, round(v, 1)))
    return [(clip(block_mm[0] + i * step), clip(block_mm[1] + j * step))
            for i, j in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]


def rehearse(block_mm, aims, *, trials=1, noise=None, picture=True, stage=None, video=None, fps=50):
    """Run every aim on `trials` copies each, all together. Returns the hit rate per aim, the best aim, and a
    contact sheet of the copies' last frames (PNG, base64) when `picture`. With a `stage` (the viewer) the
    copies are drawn live at real-time pace and stay on it afterwards; with a `video`, they're filmed too."""
    noise = noise or {}
    model = build()
    runs = [(aim, t) for aim in aims for t in range(trials)]
    copies = [_copy(model, block_mm, noise, seed=1000 + i) for i in range(len(runs))]
    offsets = layout(len(aims), trials)
    results = [None] * len(runs)
    gate = Gate(len(runs))

    def run(i):
        try:
            copies[i].hooks.append(gate.tick)
            results[i] = _pick(copies[i], runs[i][0])
        except Exception as exc:                  # a copy that breaks counts as a miss, and says why
            results[i] = {"held": False, "failed_at": f"error: {type(exc).__name__}"}
        finally:
            gate.leave()

    live = [(c.d, o, None) for c, o in zip(copies, offsets)]
    if stage:
        stage.frame(offsets)
        stage.show(model, live)
    clock = {"t0": time.monotonic(), "n": 0}

    def draw():
        clock["n"] += 1
        if stage:
            stage.draw()
        if video and clock["n"] % 4 == 0:           # the copies play at about twice their speed in the video
            video.frame()
        if stage and stage.viewer:                # real time, so a person can watch them
            ahead = clock["t0"] + clock["n"] / fps - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(runs))]
    for th in threads:
        th.start()
    while gate.step(draw):
        pass
    for th in threads:
        th.join()

    if stage:                                     # they stay up, red where they missed, until the next job
        stage.show(model, [(c.d, o, r["held"]) for c, o, r in zip(copies, offsets, results)])
    if video:
        video.frame(hold=12)
    per_aim = []
    for a, aim in enumerate(aims):
        rs = results[a * trials:(a + 1) * trials]
        per_aim.append({"dx_mm": aim[0], "dy_mm": aim[1], "held": sum(r["held"] for r in rs), "trials": trials,
                        "failed_at": [r["failed_at"] for r in rs if not r["held"]]})
    best = _best(per_aim, block_mm)
    out = {"copies": len(runs), "trials": trials, "held": sum(r["held"] for r in results),
           "block_mm_seen": list(block_mm), "noise": noise,
           "best": best and {k: best[k] for k in ("dx_mm", "dy_mm", "held", "trials")}, "results": per_aim}
    if picture:
        out["sheet_png_b64"] = _sheet(model, copies, runs, results, trials)
    return out


def _sheet(model, copies, runs, results, trials):
    """One tile per copy, an aim to a row when each aim had several tries."""
    renderer = mujoco.Renderer(model, TILE_H, TILE_W)
    tiles = []
    for i, (copy, (aim, t), r) in enumerate(zip(copies, runs, results)):
        renderer.update_scene(copy.d, _camera())
        tile = Image.fromarray(renderer.render()).convert("RGB")
        d = ImageDraw.Draw(tile)
        d.rectangle([0, 0, TILE_W, 20], fill=(255, 255, 255) if r["held"] else (0, 0, 0))
        label = f"#{i // trials + 1}" + (f".{t + 1}" if trials > 1 else "") + f"  dx {aim[0]:+g}  dy {aim[1]:+g}  "
        d.text((6, 4), label + ("HELD" if r["held"] else f"missed ({r['failed_at']})"),
               fill=(0, 0, 0) if r["held"] else (255, 255, 255))
        tiles.append(tile)
    renderer.close()
    cols = trials if trials > 1 else 4 if len(tiles) > 4 else 2 if len(tiles) > 1 else 1
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * TILE_W + (cols - 1) * 4, rows * TILE_H + (rows - 1) * 4), (0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * (TILE_W + 4), (i // cols) * (TILE_H + 4)))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()
