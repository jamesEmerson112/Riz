"""Record the README's GIFs from the sim (MuJoCo offscreen; on Linux set MUJOCO_GL=egl).

    python tools/make_media.py          # writes media/correct.gif, pick.gif, heal.gif, stall.gif and card.png
    python tools/make_media.py heal     # just one

Each GIF drives the same Skills and Watchdog the arm host uses, step by step, and captions every frame
with the tool call the agent would make and what the watchdog said. The GIFs play at 45 ms a frame.

Captions use DejaVu Sans when DejaVuSans.ttf and DejaVuSans-Bold.ttf are on the font path (the current
folder counts) and fall back to Arial, so run it where DejaVu is found to match the committed media.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))
from so101_host import prims  # noqa: E402
from so101_host.arm import SimArm  # noqa: E402
from so101_host.host import Skills, Watchdog  # noqa: E402
from so101_host.record import RunRecorder  # noqa: E402

W, H = 560, 350
OUT = ROOT / "media"
PIXEL = 5                   # the films' look: 5 px blocks in four greys


def pixelate(img):
    """Shrink the render PIXEL times, round it to four greys, and scale it back up with hard edges."""
    small = img.convert("L").resize((img.width // PIXEL, img.height // PIXEL), Image.BOX)
    small = small.point(lambda v: min(255, round(v / 85) * 85))
    return small.resize(img.size, Image.NEAREST).convert("RGB")


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_SMALL = font(15)


def font_named(names, size):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return font(size)


def fit(text, size, room=W - 24):
    """The largest font, from size down, that fits the text in the bar: a caption never runs off the frame."""
    for s in range(size, 10, -1):
        f = font(s)
        if f.getlength(text) <= room:
            return f
    raise ValueError(f"caption too long for a {W}px frame: {text!r}")


class Film:
    """Frames of the sim with a caption bar: the tool call on top, the watchdog's words below."""

    def __init__(self, arm, *, every=8):
        self.arm, self.every, self.n = arm, every, 0
        self.r = mujoco.Renderer(arm.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.14, 0.0, 0.08]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.58, 235.0, -24.0
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True     # draw the ultrasonic ray
        self.frames, self.call, self.note, self.tone = [], "", "", "ok"
        arm.hooks.append(self)

    def __call__(self, arm):
        self.n += 1
        if self.n % self.every == 0:
            self.snap()

    def snap(self, repeat=1):
        self.r.update_scene(self.arm.d, self.cam, self.opt)
        img = pixelate(Image.fromarray(self.r.render()))
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, W, 34], fill=(12, 16, 24, 215))
        d.text((12, 6), self.call, font=fit(self.call, 20), fill=(235, 240, 250))
        if self.note:
            colour = {"bad": (255, 120, 110), "warn": (255, 196, 90), "ok": (130, 230, 160)}[self.tone]
            d.rectangle([0, H - 30, W, H], fill=(12, 16, 24, 215))
            d.text((12, H - 25), self.note, font=fit(self.note, 15), fill=colour)
        row = self.arm._last_d or {}
        d.text((W - 150, 42), f"wrist: {row.get('mm', '-')} mm", font=F_SMALL, fill=(255, 255, 255))
        for _ in range(repeat):
            self.frames.append(img)

    def show(self, path, call, note="", tone="ok", hold=30):
        """A still (the agent's camera image) full-frame, with the same caption bars."""
        img = Image.open(path).convert("RGB")
        img.thumbnail((W, H - 64))
        frame = Image.new("RGB", (W, H), (12, 16, 24))
        frame.paste(img, ((W - img.width) // 2, 34 + (H - 64 - img.height) // 2))
        d = ImageDraw.Draw(frame, "RGBA")
        d.text((12, 6), call, font=fit(call, 20), fill=(235, 240, 250))
        if note:
            colour = {"bad": (255, 120, 110), "warn": (255, 196, 90), "ok": (130, 230, 160)}[tone]
            d.text((12, H - 25), note, font=fit(note, 15), fill=colour)
        self.frames.extend([frame] * hold)

    def say(self, call, note="", tone="ok", hold=8):
        self.call, self.note, self.tone = call, note, tone
        self.snap(repeat=hold)

    def save(self, name, ms=45):
        # 45 ms a frame plays the films at twice their recorded pace; a held caption is repeated frames, so it
        # speeds up with them. GIF stores delays in 10 ms steps and Pillow rounds down, so a lone frame plays
        # at 40 ms (browsers clamp anything under 20 ms, so don't go lower).
        path = OUT / name
        frames = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in self.frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
        self.r.close()
        print(f"{path}  {len(frames)} frames  {path.stat().st_size // 1024} KB")
        return path


def bench():
    arm = SimArm(recorder=RunRecorder("/tmp", task="media"), start=prims.REST)
    skills = Skills(arm)
    dog = Watchdog(arm, arm.recorder)
    dog.home = skills.p["home"]
    return arm, skills, dog


def run(film, skills, dog, name, call):
    film.say(call, hold=3)
    r = getattr(skills, name)()
    if not r.get("halted"):
        dog.arrival(r)
    if dog.open():
        inc = dog.open()[0]
        film.say(call, f"watchdog: {inc['summary']}", "bad", hold=22)
        film.say("arm parked (torque off)", f"watchdog: {inc['summary']}", "bad", hold=4)
        skills.park()
        film.say("arm parked: motion blocked", "guard: fix the cause first (arm_restart_sensor)", "warn", hold=18)
    return r


def pick():
    arm, skills, dog = bench()
    film = Film(arm)
    for name, call in (("go_home", "arm_go_home"), ("approach", "arm_approach"), ("grip", "arm_grip"),
                       ("lift", "arm_lift")):
        r = run(film, skills, dog, name, call)
    held = (r.get("grasp") or {}).get("holding")
    film.say("arm_lift", f"holding: {held} (gripper stopped short of closed, under load)", "ok", hold=20)
    run(film, skills, dog, "release", "arm_release")
    run(film, skills, dog, "go_home", "arm_go_home")
    film.say("done", "picked up and put back, every step checked", "ok", hold=14)
    return film.save("pick.gif")


def heal():
    arm, skills, dog = bench()
    film = Film(arm)
    run(film, skills, dog, "go_home", "arm_go_home")
    arm.inject("us_frozen")
    film.say("(a fault is planted: the wrist sensor freezes)", "the sensor keeps sending its last reading", "warn", hold=16)
    run(film, skills, dog, "approach", "arm_approach")
    film.say("arm_restart_sensor", "the agent fixes the cause; the watchdog re-checks at home", "warn", hold=10)
    v = dog.repair("restart_sensor")
    film.say("arm_restart_sensor", f"verified: {v['verified']}, incident closed", "ok", hold=18)
    for name, call in (("approach", "arm_approach"), ("grip", "arm_grip"), ("lift", "arm_lift")):
        r = run(film, skills, dog, name, call)
    film.say("arm_pick_up: resumed", f"holding: {(r.get('grasp') or {}).get('holding')} after the heal", "ok", hold=20)
    return film.save("heal.gif")


def correct():
    """The hero: the block was bumped, the pick misses, the agent looks at the grid and corrects its aim."""
    from so101_host.camera import Eye
    arm, skills, dog = bench()
    film = Film(arm)
    run(film, skills, dog, "go_home", "arm_go_home")
    arm.shift_block(30, -10)
    film.say("(someone bumps the block)", "it moved 30 mm forward and 10 mm right; the arm still aims at the old spot",
             "warn", hold=16)
    run(film, skills, dog, "approach", "arm_pick_up")
    g = run(film, skills, dog, "grip", "arm_pick_up")
    film.say("arm_pick_up", f"failed at grip: the jaws closed on nothing (holding: {g['grasp']['holding']})", "bad",
             hold=20)
    run(film, skills, dog, "release", "arm_release")
    run(film, skills, dog, "go_home", "arm_go_home")
    eye = Eye(arm, tempfile.mkdtemp(prefix="so101-media-"))
    seen = eye.look((0, 0))
    film.show(seen["top"], "arm_look", f"block_mm {seen['block_mm']}: measured on the map; the image confirms it",
              "warn", hold=40)
    preview = eye.look((30, -10))
    eye.r.close()
    film.show(preview["top"], "arm_look dx_mm=30 dy_mm=-10", "preview: the aim now sits on the block", "ok", hold=28)
    film.say("arm_pick_up dx_mm=30 dy_mm=-10", "corrections are limited to 40 mm per axis, in code", "ok", hold=8)
    skills.approach((30.0, -10.0))
    g = skills.grip()
    l = skills.lift()
    film.say("arm_pick_up dx_mm=30 dy_mm=-10", f"picked up: holding {l['grasp']['holding']} after the correction",
             "ok", hold=30)
    return film.save("correct.gif")


def stall():
    arm, skills, dog = bench()
    film = Film(arm)
    run(film, skills, dog, "go_home", "arm_go_home")
    arm.inject("joint_stuck", joint=2)
    film.say("(a fault is planted: the elbow servo dies)", "", "warn", hold=12)
    run(film, skills, dog, "reach", "arm_reach")
    film.say("arm_check", "the watchdog retries the move home", "warn", hold=8)
    v = dog.verify("repair: verify")
    who = "handed to a person" if v.get("handed_off") else "fixed"
    film.say("arm_check", f"still stuck: {who}; the arm stays parked", "bad", hold=26)
    return film.save("stall.gif")


def card():
    """The catalog card (2:1, 1200x600): the arm holding the block, and what the plugin is."""
    arm, skills, dog = bench()
    for name in ("go_home", "approach", "grip", "lift"):
        getattr(skills, name)()
    arm.m.vis.global_.offwidth, arm.m.vis.global_.offheight = 1200, 600
    r = mujoco.Renderer(arm.m, 600, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.0, 0.10]
    cam.distance, cam.azimuth, cam.elevation = 0.50, 215.0, -18.0
    r.update_scene(arm.d, cam)
    shot = Image.fromarray(r.render()).convert("RGB")
    r.close()
    img = Image.new("RGB", (1200, 600), (12, 16, 24))
    img.paste(shot, (0, 0))
    d = ImageDraw.Draw(img)
    bold = font_named(("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                       "Arial Bold.ttf"), 60)
    d.text((680, 70), "so101-arm", font=bold, fill=(245, 248, 255))
    d.text((682, 150), "Give Hermes a robot arm, safely", font=font(30), fill=(150, 200, 255))
    lines = ("Skills, never joint angles", "A grid camera to correct its aim", "A watchdog that parks it on faults",
             "Heals, verifies, or calls a person", "A kill switch only a person lifts")
    for i, line in enumerate(lines):
        y = 232 + i * 52
        d.ellipse([684, y + 9, 696, y + 21], fill=(130, 230, 160))
        d.text((712, y), line, font=font(26), fill=(225, 230, 240))
    d.text((682, 520), "MuJoCo sim on a laptop · real SO-101 via LeRobot", font=fit("MuJoCo sim on a laptop · real SO-101 via LeRobot", 19, 500), fill=(140, 150, 170))
    img.save(OUT / "card.png")
    print(OUT / "card.png")


MEDIA = {"pick": pick, "heal": heal, "stall": stall, "correct": correct, "card": card}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    names = sys.argv[1:]
    if names:
        for name in names:
            MEDIA[name]()
    else:
        # One process per file: an OpenGL context that one film's renderer leaves behind can be torn
        # down in the middle of the next film's frames, which turns them to noise.
        for name in MEDIA:
            subprocess.run([sys.executable, __file__, name], check=True)
