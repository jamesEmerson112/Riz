"""The real plugin against the real arm host on the sim: the calls an agent makes, in the order Hermes makes
them (the pre_tool_call guard first, then the tool), with nothing faked.

    python -m pytest host/tests/test_plugin_on_sim.py
"""
import base64
import importlib.util
import json
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

ROOT = Path(__file__).resolve().parents[2]


class Ctx:
    def __init__(self, sock):
        self.tools, self.hooks, self.commands, self.skills, self.sock = {}, {}, {}, {}, sock

    def get_config(self, key, default=None):
        return str(self.sock) if key == "socket" else default

    def register_tool(self, *, name, toolset, schema, handler, **kw):
        self.tools[name] = handler

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_command(self, name, handler, **kw):
        self.commands[name] = handler

    def register_skill(self, name, path, **kw):
        self.skills[name] = path


@pytest.fixture(scope="module")
def agent():
    tmp = Path(tempfile.mkdtemp(prefix="so101a-"))
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
    spec = importlib.util.spec_from_file_location("so101_arm_plugin_live", ROOT / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    ctx = Ctx(sock)
    plugin.register(ctx)

    def use(tool, args=None):
        """What Hermes does: ask the guard, then run the tool (or hand the agent the guard's message)."""
        block = ctx.hooks["pre_tool_call"](tool_name=tool, args=args or {}, task_id="t")
        if block:
            return {"blocked": block["message"]}
        out = ctx.tools[tool](args or {}, task_id="t")
        return out if isinstance(out, dict) else json.loads(out)

    def host(op, **fields):
        return call(op, sock=sock, timeout=180, **fields)

    yield use, host, ctx
    try:
        host("shutdown")
    except OSError:
        pass
    proc.wait(10)


def test_look_and_correct_through_the_plugin(agent):
    use, host, _ = agent
    host("clear")
    assert use("arm_go_home")["ok"]
    host("inject", fault="block_shift", params={"dx_mm": 30, "dy_mm": -10})
    missed = use("arm_pick_up")
    assert not missed["picked_up"] and missed["failed_at"] in ("grip", "lift"), missed
    use("arm_release")
    use("arm_go_home")
    seen = use("arm_look")
    assert seen["_multimodal"] is True                                  # the image goes to the model itself
    text, image = seen["content"]
    assert base64.b64decode(image["image_url"]["url"].split(",", 1)[1]).startswith(b"\x89PNG")
    assert json.loads(text["text"].split("\n\n")[0])["aim_mm"] == [0.0, 0.0]
    assert use("arm_pick_up", {"dx_mm": 30, "dy_mm": -10})["picked_up"]
    use("arm_release")
    host("clear")


def test_a_frozen_sensor_is_blocked_then_healed_through_the_plugin(agent):
    use, host, _ = agent
    host("clear")
    use("arm_go_home")
    host("inject", fault="us_frozen")
    first = use("arm_pick_up")
    assert not first["picked_up"]                                         # the watchdog parked it mid-approach
    blocked = use("arm_reach")
    assert "arm_restart_sensor" in blocked["blocked"]                     # the guard names the way out
    fixed = use("arm_restart_sensor")
    assert fixed["ok"] and fixed["verified"], fixed
    assert use("arm_go_home")["ok"] and use("arm_pick_up")["picked_up"]
    use("arm_release")


def test_only_the_slash_command_unfreezes(agent):
    use, host, ctx = agent
    host("clear")
    ctx.commands["arm"]("freeze")
    assert "/arm unfreeze" in use("arm_play", {"motion": "wave"})["blocked"]
    assert not any(name for name in ctx.tools if "freeze" in name)        # no tool for it
    assert not host("skill", name="go_home")["ok"]                         # the host refuses too, guard or not
    ctx.commands["arm"]("unfreeze")
    assert use("arm_play", {"motion": "wave"})["ok"]
