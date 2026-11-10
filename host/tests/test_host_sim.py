"""The arm host on the MuJoCo sim, through its socket, as the plugin uses it: python -m pytest host/tests

Needs the sim extra (`pip install -e 'host[sim]'`) and the model (`python -m so101_host fetch-model`).
"""
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
    assert host("skill", name="approach", offset_mm=[30, -10])["result"]["arrived"]
    assert host("skill", name="grip")["result"]["grasp"]["holding"]
    assert host("skill", name="lift")["result"]["grasp"]["holding"]
    host("skill", name="release")
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
