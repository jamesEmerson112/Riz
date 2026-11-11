"""Offscreen frames of a run, so a run can be checked or shown without the viewer window: a GIF of the arm
(`serve --gif`), or an MP4 with the Hermes log beside it (`serve --video`)."""
import textwrap

import mujoco
from PIL import Image, ImageDraw, ImageFont


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


class Video:
    """An MP4 of the session: the arm (and a rehearsal's copies) through the viewer's camera, with the Hermes
    log beside it in black and white. Frames come from motion (every `every` control ticks) and from new log
    lines (held about half a second), so the seconds spent waiting on the model aren't filmed.

    Frames are piped to ffmpeg as a fragmented MP4, which still plays if the host is killed mid-write."""

    W, H, LOG_W = 1280, 720, 470

    def __init__(self, arm, path, *, board, stage, fps=25, every=4):     # every 4th tick: ~2x speed, and
                                                                        # the arm keeps close to real time
        import shutil
        import subprocess
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SystemExit("--video needs ffmpeg on the PATH (macOS: brew install ffmpeg; Debian: apt install ffmpeg)")
        self.arm, self.path, self.board, self.stage, self.every, self.n = arm, path, board, stage, every, 0
        view_w = self.W - self.LOG_W
        g = arm.m.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, view_w), max(g.offheight, self.H)
        self.r = mujoco.Renderer(arm.m, self.H, view_w)
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
        self.font, self.bold = _mono(17), _mono(19)
        self.proc = subprocess.Popen(
            [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(path)],
            stdin=subprocess.PIPE)

    def __call__(self, arm):                      # an arm hook: one frame every `every` control ticks
        self.n += 1
        if self.n % self.every == 0:
            self.frame()

    def frame(self, hold=1):
        self.r.update_scene(self.arm.d, self.stage.camera(), self.opt)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0        # a shadow pass over a stage of copies
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0    # costs ~0.4 s a frame; the viewer keeps its own
        self.stage.add_to(self.r.scene)
        img = Image.new("RGB", (self.W, self.H), (255, 255, 255))
        img.paste(Image.fromarray(self.r.render()), (0, 0))
        self._log(ImageDraw.Draw(img), self.W - self.LOG_W)
        try:
            raw = img.tobytes()
            for _ in range(hold):
                self.proc.stdin.write(raw)
        except (BrokenPipeError, ValueError):     # ffmpeg is gone: keep the arm running, stop filming
            pass

    def _log(self, d, x0):
        """The Board, brutalist: black on white, monospace, hard rules, an inverted title bar."""
        W, H, pad = self.W, self.H, 14
        d.rectangle([x0, 0, W, H], fill=(255, 255, 255))
        d.rectangle([x0, 0, x0 + 3, H], fill=(0, 0, 0))
        d.rectangle([x0, 0, W, 40], fill=(0, 0, 0))
        d.text((x0 + pad, 9), "HERMES + ARM", font=self.bold, fill=(255, 255, 255))
        y, chars = 52, (self.LOG_W - 2 * pad) // 10
        rows, room = [], H - 245 - y              # the newest lines that fit above the context block
        for t, text in reversed(self.board.agent):
            lines = textwrap.wrap(text, chars - 9)[:2] or [""]
            room -= 21 * len(lines) + 5
            if room < 0:
                break
            rows.insert(0, (t, lines))
        for t, lines in rows:
            d.text((x0 + pad, y), t, font=self.font, fill=(110, 110, 110))
            for k, line in enumerate(lines):
                d.text((x0 + pad + 92, y + 21 * k), line, font=self.font, fill=(0, 0, 0))
            y += 21 * len(lines) + 5
        top = H - 230
        d.rectangle([x0, top, W, top + 3], fill=(0, 0, 0))
        d.text((x0 + pad, top + 12), "CONTEXT", font=self.bold, fill=(0, 0, 0))
        y = top + 40
        for line in textwrap.wrap(self.board.context, chars)[:2]:
            d.text((x0 + pad, y), line, font=self.font, fill=(0, 0, 0))
            y += 22
        for t, text in list(self.board.memory)[-6:]:
            d.text((x0 + pad, y + 6), f"{t}  {text}"[:chars], font=self.font, fill=(90, 90, 90))
            y += 22

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            pass
        self.r.close()
        return self.path


def _mono(size):
    for name in ("Menlo.ttc", "/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)
