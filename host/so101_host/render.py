"""Offscreen frames of a run, saved as a GIF, so a run can be checked without the viewer window."""
import mujoco
from PIL import Image, ImageDraw


class Gif:
    def __init__(self, arm, path, *, every=5, width=480, height=360, caption=None):
        self.arm, self.path, self.every = arm, path, every
        self.caption = caption or (lambda a: "")
        self.r = mujoco.Renderer(arm.m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.15, 0.0, 0.12]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.8, 135.0, -20.0
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True   # draw the ultrasonic ray
        self.frames, self.n = [], 0
        arm.hooks.append(self)

    def __call__(self, arm):
        self.n += 1
        if self.n % self.every:
            return
        self.r.update_scene(arm.d, self.cam, self.opt)
        img = Image.fromarray(self.r.render())
        d = arm._last_d or {}
        text = f"t={arm.t:5.2f}s  d={d.get('mm')} mm  us={d.get('us')}  {d.get('st', '')}  {self.caption(arm)}"
        ImageDraw.Draw(img).text((8, 8), text, fill=(255, 255, 0))
        self.frames.append(img)

    def save(self):
        if self.frames:
            ms = int(1000 * self.every / self.arm.fps)
            self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:], duration=ms, loop=0)
        return self.path
