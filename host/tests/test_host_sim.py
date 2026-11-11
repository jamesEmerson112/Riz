"""The arm host on the MuJoCo sim, through its socket, as the plugin uses it: python -m pytest host/tests

Needs the sim extra (`pip install -e 'host[sim]'`) and the model (`python -m so101_host fetch-model`).
"""
import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
from so101_host.client import call  # noqa: E402
from so101_host.model import MENAGERIE  # noqa: E402

if not (MENAGERIE / "so101.xml").exists():
    pytest.skip(f"no SO-101 model at {MENAGERIE} (python -m so101_host fetch-model)", allow_module_level=True)


@pytest.fixture(scope="module")
def host():
    tmp = Path(tempfile.mkdtemp(prefix="so101-"))
    sock = tmp / "arm.sock"
    env = dict(os.environ, SO101_SOCK=str(sock), SO101_HOME=str(tmp / "home"))
    proc = subprocess.Popen([sys.executable, "-m", "so101_host", "serve"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(200):
        if sock.exists():
            break
        time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("the host didn't start: " + (proc.stdout.read() if proc.stdout else ""))

    def ask(op, **fields):
        return call(op, sock=sock, timeout=180, **fields)
    yield ask
    try:
        ask("shutdown")
    except OSError:
        pass
    proc.wait(10)


def test_the_pick_sequence_holds_the_block(host):
    host("clear")
    assert host("skill", name="go_home")["result"]["arrived"]
    assert host("skill", name="approach")["result"]["arrived"]
    assert host("skill", name="grip")["result"]["grasp"]["holding"]
    lifted = host("skill", name="lift")["result"]
    assert lifted["arrived"] and lifted["grasp"]["holding"]
    assert host("skill", name="release")["result"]["arrived"]
    assert host("play", motion="wave")["result"]["arrived"]
    assert host("motions")["motions"] == ["wave"]


def test_a_frozen_sensor_parks_the_arm_until_the_restart_is_verified(host):
    host("clear")
    host("skill", name="go_home")
    host("inject", fault="us_frozen")
    r = host("skill", name="approach")
    assert r["incident"]["kind"].startswith("us_") and r["parked"]["torque"] == "off", r
    refused = host("skill", name="reach")
    assert not refused["ok"] and "incident open" in refused["error"]
    fixed = host("repair", action="restart_sensor")
    assert fixed["verified"] and fixed["resolved"], fixed
    assert host("skill", name="go_home")["result"]["arrived"]


def test_a_dead_joint_goes_to_a_person_and_only_their_check_closes_it(host):
    host("clear")
    host("skill", name="go_home")
    host("inject", fault="joint_stuck", params={"joint": 2})
    r = host("skill", name="reach")
    assert r["incident"]["kind"] == "joint_stall", r
    first = host("repair", action="verify")
    assert not first["verified"] and first["handed_off"], first
    assert host("status")["handed_off"]
    host("clear")                                       # the person fixes the servo
    second = host("repair", action="verify")
    assert second["verified"], second


def test_freeze_refuses_motion_until_a_person_unfreezes(host):
    host("clear")
    froze = host("freeze", by="test")
    assert froze["frozen"]
    no = host("skill", name="go_home")
    assert not no["ok"] and "frozen" in no["error"]
    assert not host("repair", action="verify")["ok"]
    assert host("skill", name="park")["ok"]             # parking is always allowed
    host("unfreeze", by="test")
    assert host("skill", name="go_home")["result"]["arrived"]


def test_the_agent_corrects_its_aim_from_the_grid_when_the_block_moved(host):
    host("clear")
    host("skill", name="go_home")
    host("inject", fault="block_shift", params={"dx_mm": 30, "dy_mm": -10})
    host("skill", name="approach")
    missed = host("skill", name="grip")["result"]["grasp"]
    assert not missed["holding"], missed                 # aimed at the taught spot: closes on nothing
    host("skill", name="release")
    host("skill", name="go_home")
    seen = host("look")
    assert Path(seen["top"]).exists() and Path(seen["side"]).exists() and seen["grid_mm"] == 10
    assert "top_png_b64" not in seen
    assert seen["block_seen"] and seen["block_mm"] == [30, -10]         # measured from the image, plain code
    assert seen["to_pick_it"] == {"tool": "arm_pick_up", "dx_mm": 30, "dy_mm": -10}
    inline = host("look", aim_mm=[30, -10], inline=True)              # what the plugin hands a vision model
    assert base64.b64decode(inline["top_png_b64"]).startswith(b"\x89PNG") and inline["aim_mm"] == [30, -10]
    assert host("skill", name="approach", offset_mm=[30, -10])["result"]["arrived"]
    assert host("skill", name="grip")["result"]["grasp"]["holding"]
    assert host("skill", name="lift")["result"]["grasp"]["holding"]
    host("skill", name="release")
    host("clear")


def test_an_open_gripper_is_not_holding_right_after_a_release(host):
    host("clear")
    host("skill", name="go_home")
    host("skill", name="approach")
    host("skill", name="grip")
    assert host("skill", name="lift")["result"]["grasp"]["holding"]
    host("skill", name="release")
    assert not host("status")["grasp"]["holding"]                      # it settles with load, but it's open
    host("clear")


def test_every_sensor_incident_says_restart_first():
    from so101_host.host import _brief
    inc = {"id": "inc-1", "kind": "us_missing_echo", "summary": "no echo", "next_steps": ["a person checks the wires"]}
    steps = _brief(inc)["next_steps"]
    assert "arm_restart_sensor" in steps[0] and steps[1] == "if it comes back: a person checks the wires"
    assert _brief({**inc, "kind": "joint_stall"})["next_steps"] == ["a person checks the wires"]


def test_a_block_beyond_the_limit_is_seen_but_not_offered(host):
    host("clear")
    host("skill", name="go_home")
    host("inject", fault="block_shift", params={"dx_mm": 55, "dy_mm": 0})
    seen = host("look")
    assert seen["block_mm"] == [55, 0] and seen["reachable"] is False and seen["to_pick_it"].startswith("none")
    assert "tell the user" in seen["note"]
    host("clear")


def test_offsets_are_bounded_in_code(host):
    host("clear")
    for bad in ([41, 0], [0, -60], ["x", 1], [float("inf"), 0]):
        r = host("skill", name="approach", offset_mm=bad)
        assert not r["ok"], (bad, r)
    assert not host("skill", name="grip", offset_mm=[5, 5])["ok"]     # only the approach takes an aim
    assert not host("inject", fault="meteor")["ok"]


def test_unknown_skills_and_motions_are_refused(host):
    assert not host("skill", name="dance")["ok"]
    r = host("play", motion="dance")
    assert "no motion named" in r["result"]["aborted"]


def test_the_viewer_log_keeps_the_agents_lines_and_the_context_apart(host):
    from so101_host.host import Board, _board_line
    shown = []
    board = Board(show=shown.append, keep=3)
    for i in range(5):
        board.add("agent", f"step {i}")
    board.add("context", "12.1k tokens = 4% of 272.0k")
    board.add("memory", "call 7: 11.2k in, 97% cached, 27 out → ascii only")
    (a1, a2), (m1, m2) = shown[-1]
    assert a2.splitlines()[1:] == ["step 2", "step 3", "step 4"]          # the newest lines, oldest first
    assert "12.1k tokens = 4% of 272.0k" in m2 and m2.isascii()
    assert _board_line({"op": "freeze", "by": "/arm"}, {"ok": True}).startswith("FROZEN by /arm")
    assert _board_line({"op": "status"}, {"ok": True}) is None
    assert host("note", lane="agent", text="arm_go_home -> ok")["ok"]      # the plugin's line, over the socket


def test_rehearsal_tries_aims_on_copies_built_from_what_the_camera_saw(host):
    host("clear")
    host("repair", action="verify")
    host("inject", fault="block_shift", params={"dx_mm": 30, "dy_mm": -10})
    before = host("status")
    out = host("rehearse", candidates=[[0, 0], [30, -10], [30, 10]])
    assert out["ok"] and out["copies"] == 3 and out["block_mm_seen"] == [30, -10]
    held = {(r["dx_mm"], r["dy_mm"]): r["held"] for r in out["results"]}
    assert held == {(0.0, 0.0): 0, (30.0, -10.0): 1, (30.0, 10.0): 0} and out["trials"] == 1
    assert out["best"] == {"dx_mm": 30.0, "dy_mm": -10.0, "held": 1, "trials": 1} and "sheet_png_b64" not in out
    after = host("status")                                    # the main arm didn't move or pick anything up
    assert after["parked"] == before["parked"] and not after["grasp"]["holding"] and not after["incidents"]
    nine = host("rehearse", inline=True)                      # no aims given: nine around the seen block
    assert nine["copies"] == 9 and nine["held"] >= 1 and base64.b64decode(nine["sheet_png_b64"])[:4] == b"\x89PNG"
    assert "beyond the 40 mm" in host("rehearse", candidates=[[45, 0]])["error"]
    host("clear")


def test_joint_noise_lands_each_move_a_little_off_and_repeats_with_the_seed():
    from so101_host import prims
    from so101_host.arm import SimArm
    from so101_host.host import Skills

    def landed(seed, noise=True):
        arm = SimArm(start=prims.REST, seed=seed)
        if noise:
            arm.inject("joint_noise", deg=1.5)
        home = Skills(arm).p["home"]
        assert arm.goto(home)["arrived"]                     # it arrives where it was (noisily) sent
        return [abs(v - h) for v, h in zip(list(arm.read().values())[:5], home[:5])], arm

    off, arm = landed(7)
    assert 0.05 < max(off) < 6.0                             # a little off, joint by joint
    assert landed(7)[0] == off and landed(8)[0] != off      # the same seed lands the same way
    assert max(landed(7, noise=False)[0]) < 0.1
    before = list(arm.read().values())[:5]                   # a gripper-only move leaves the arm joints be:
    Skills(arm).release()                                    # noise jolting them at the bottom of a pick
    after = list(arm.read().values())[:5]                    # overloaded the shoulder against the block
    assert max(abs(a - b) for a, b in zip(after, before)) < 0.2
    arm.clear()
    assert "joint_noise" not in arm.faults


def test_a_noisy_camera_measures_the_block_a_little_off_each_look(host):
    host("clear")
    host("inject", fault="block_shift", params={"dx_mm": 20, "dy_mm": 10})
    clean = host("look")["block_mm"]
    host("inject", fault="camera_noise", params={"mm": 4})
    looks = [host("look")["block_mm"] for _ in range(6)]
    assert len({tuple(b) for b in looks}) > 1               # each look is off differently
    for axis in (0, 1):
        assert abs(sum(b[axis] for b in looks) / len(looks) - clean[axis]) < 6
    host("clear", fault="camera_noise")
    assert host("look")["block_mm"] == clean
    host("clear")


def test_a_noisy_rehearsal_tries_each_aim_several_times_and_picks_the_most_reliable(host):
    host("clear")
    host("repair", action="verify")
    host("inject", fault="block_shift", params={"dx_mm": 30, "dy_mm": -10})
    host("inject", fault="joint_noise", params={"deg": 1.0})
    host("inject", fault="camera_noise", params={"mm": 2})
    out = host("rehearse", candidates=[[30, -10], [0, 0]], trials=3)
    assert out["ok"] and out["copies"] == 6 and out["trials"] == 3
    assert out["noise"] == {"joint_noise": {"deg": 1.0}, "camera_noise": {"mm": 2}}
    by_aim = {(r["dx_mm"], r["dy_mm"]): r for r in out["results"]}
    assert by_aim[(0.0, 0.0)]["held"] == 0 and len(by_aim[(0.0, 0.0)]["failed_at"]) == 3
    assert by_aim[(30.0, -10.0)]["held"] >= 1 and out["best"]["dx_mm"] == 30.0 and out["best"]["trials"] == 3
    chosen = host("rehearse")                                # noisy and no aims: five around the seen block, 3 each
    assert chosen["copies"] == 15 and len(chosen["results"]) == 5
    assert "trials must be 1 to 4" in host("rehearse", trials=5)["error"]
    assert "at most 24 copies" in host("rehearse", candidates=[[30, -10]] * 9, trials=3)["error"]
    host("clear")


def test_the_copies_step_together_and_leave_no_threads_behind():
    import threading
    from so101_host.rehearse import rehearse
    before = threading.active_count()
    out = rehearse([30, -10], [(30.0, -10.0), (0.0, 0.0)], trials=2, noise={"joint_noise": {"deg": 1.0}},
                   picture=False)
    assert out["copies"] == 4 and [r["trials"] for r in out["results"]] == [2, 2]
    assert out["results"][1]["held"] == 0 and threading.active_count() == before


def test_the_video_is_an_mp4_of_the_arm_with_the_log_beside_it(tmp_path):
    import json as _json
    import shutil
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("no ffmpeg")
    from so101_host import prims
    from so101_host.arm import SimArm
    from so101_host.host import Board
    from so101_host.render import Video
    from so101_host.stage import Stage
    arm, board = SimArm(start=prims.REST), Board()
    board.add("agent", "arm_go_home -> ok")
    board.add("context", "12.1k tokens = 4% of 272.0k")
    video = Video(arm, tmp_path / "run.mp4", board=board, stage=Stage())
    arm.hooks.append(video)
    arm.hold(0.4)                                            # 20 ticks: 10 frames
    video.frame(hold=5)
    path = video.close()
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,nb_read_frames", "-of", "json", str(path)],
                           capture_output=True, text=True, check=True)
    stream = _json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (1280, 720) and int(stream["nb_read_frames"]) == 15
