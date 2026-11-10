"""Record the README's GIFs from the sim (MuJoCo offscreen; on Linux set MUJOCO_GL=egl).

    python tools/make_media.py            # writes media/pick.gif, media/heal.gif, media/stall.gif, media/still.png

Each GIF drives the same Skills and Watchdog the arm host uses, step by step, and captions every frame
with the tool call the agent would make and what the watchdog said.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))
from so101_host import prims  # noqa: E402
from so101_host.arm import SimArm  # noqa: E402
from so101_host.host import Skills, Watchdog  # noqa: E402
from so101_host.record import RunRecorder  # noqa: E402

W, H = 640, 400
OUT = ROOT / "media"


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_BIG, F_SMALL = font(20), font(15)


class Film:
    """Frames of the sim with a caption bar: the tool call on top, the watchdog's words below."""

    def __init__(self, arm, *, every=4):
        self.arm, self.every, self.n = arm, every, 0
        self.r = mujoco.Renderer(arm.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.16, 0.0, 0.08]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.62, 140.0, -22.0
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
        img = Image.fromarray(self.r.render()).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, W, 34], fill=(12, 16, 24, 215))
        d.text((12, 6), self.call, font=F_BIG, fill=(235, 240, 250))
        if self.note:
            colour = {"bad": (255, 120, 110), "warn": (255, 196, 90), "ok": (130, 230, 160)}[self.tone]
            d.rectangle([0, H - 30, W, H], fill=(12, 16, 24, 215))
            d.text((12, H - 25), self.note, font=F_SMALL, fill=colour)
        row = self.arm._last_d or {}
        d.text((W - 150, 42), f"wrist: {row.get('mm', '-')} mm", font=F_SMALL, fill=(255, 255, 255))
        for _ in range(repeat):
            self.frames.append(img)

    def say(self, call, note="", tone="ok", hold=8):
        self.call, self.note, self.tone = call, note, tone
        self.snap(repeat=hold)

    def save(self, name, ms=60):
        path = OUT / name
        frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in self.frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
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


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pick()
    heal()
    stall()
    arm, skills, _ = bench()
    skills.go_home()
    Film(arm).snap()
    f = Film(arm)
    f.say("so101-arm: the MuJoCo SO-101, ready", "", hold=1)
    f.frames[-1].save(OUT / "still.png")
    print(OUT / "still.png")
    _ = np  # numpy is imported by the sim; kept for clarity when editing
