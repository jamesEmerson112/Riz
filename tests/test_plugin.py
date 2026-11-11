"""The plugin itself: registration, every tool, the pre_tool_call guard and /arm, against a fake arm host.
No Hermes, no MuJoCo, no hardware: python -m pytest tests"""
import importlib.util
import json
import os
import socket
import tempfile
import threading
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_plugin():
    spec = importlib.util.spec_from_file_location("so101_arm_plugin", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCtx:
    """Records what register() asks for, like Hermes's PluginContext."""

    def __init__(self, config=None):
        self.tools, self.hooks, self.commands, self.skills = {}, {}, {}, {}
        self.config = config or {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, *, name, toolset, schema, handler, **kw):
        assert schema["name"] == name and schema["description"]
        self.tools[name] = (toolset, schema, handler)

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_command(self, name, handler, description="", args_hint="", **kw):
        self.commands[name] = handler
        self.command_hints = getattr(self, "command_hints", {})
        self.command_hints[name] = args_hint

    def register_skill(self, name, path, description="", **kw):
        self.skills[name] = Path(path)
        self.skill_descriptions = getattr(self, "skill_descriptions", {})
        self.skill_descriptions[name] = description


class FakeHost:
    """A stand-in arm host on a Unix socket: canned replies, and a log of every request."""

    def __init__(self, path):
        self.path, self.log = str(path), []
        self.state = {"incidents": [], "status": [0] * 6, "frozen": None, "grasp_holding": True}
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(8)
        threading.Thread(target=self._serve, daemon=True).start()

    def reply(self, req):
        op, st = req.get("op"), self.state
        if op == "status":
            return {"ok": True, "arm": "sim", "status": st["status"], "frozen": st["frozen"], "parked": True,
                    "incidents": st["incidents"], "handed_off": [i for i in st["incidents"] if i.get("handed_off")],
                    "grasp": {"holding": st["grasp_holding"]}}
        if op == "skill":
            res = {"arrived": True}
            if req["name"] in ("grip", "lift"):
                res["grasp"] = {"holding": st["grasp_holding"]}
            return {"ok": True, "skill": req["name"], "result": res}
        if op == "play":
            return {"ok": True, "skill": "play:" + req["motion"], "result": {"arrived": True}}
        if op == "read":
            return {"ok": True, "pos": {"gripper.pos": 60.0}}
        if op == "distance":
            return {"ok": True, "row": {"mm": 120, "us": 700, "st": "ok"}}
        if op == "motions":
            return {"ok": True, "motions": ["wave"]}
        if op == "look":
            reply = {"ok": True, "top": "/tmp/look-001-top.png", "side": "/tmp/look-001-side.png",
                     "aim_mm": req["aim_mm"], "grid_mm": 10}
            if req.get("inline") and st.get("inline_png", True):
                reply["top_png_b64"] = "iVBORw0KGgo="
            return reply
        if op == "repair":
            return {"ok": True, "action": req["action"], "verified": True}
        if op == "handoff":
            return {"ok": True, "id": req["id"], "handed_off": 1.0, "handoff_reason": req["reason"]}
        if op in ("freeze", "unfreeze"):
            st["frozen"] = req.get("by") if op == "freeze" else None
            return {"ok": True, "frozen": st["frozen"]}
        return {"ok": False, "error": "unknown op"}

    def _serve(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            with conn, conn.makefile("rw", encoding="utf-8") as f:
                for line in f:
                    req = json.loads(line)
                    self.log.append(req)
                    f.write(json.dumps(self.reply(req)) + "\n")
                    f.flush()

    def close(self):
        self.srv.close()


@pytest.fixture
def plugin(tmp_path):
    sock = Path(tempfile.mkdtemp(prefix="so101p-")) / "arm.sock"     # short path: macOS caps socket paths
    host = FakeHost(sock)
    mod = load_plugin()
    ctx = FakeCtx(config={"socket": str(sock)})
    mod.register(ctx)
    yield mod, ctx, host
    host.close()


def run(ctx, tool, args=None):
    return json.loads(ctx.tools[tool][2](args or {}, task_id="t1"))


def test_registration_matches_the_manifest(plugin):
    mod, ctx, _ = plugin
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    assert sorted(ctx.tools) == sorted(manifest["provides_tools"])
    assert sorted(ctx.hooks) == sorted(manifest["provides_hooks"]) == ["post_api_request", "post_compaction",
                                                                       "post_tool_call", "pre_tool_call"]
    assert all(toolset == "so101_arm" for toolset, _, _ in ctx.tools.values())
    assert "arm" in ctx.commands and "operate" in ctx.skills and ctx.skills["operate"].name == "SKILL.md"
    assert ctx.skills["operate"].exists() and ctx.skill_descriptions["operate"].startswith("How to run the SO-101")
    assert ctx.command_hints["arm"] == "status|park|check|freeze|unfreeze"
    assert manifest["requires_hermes"].startswith(">=") and isinstance(manifest["version"], str)


def test_every_tool_returns_json_and_reaches_the_host(plugin):
    mod, ctx, host = plugin
    for tool in ("arm_go_home", "arm_reach", "arm_approach", "arm_grip", "arm_lift", "arm_release", "arm_park"):
        out = run(ctx, tool)
        assert out["ok"] and out["skill"] == mod.SKILLS[tool][0]
    assert run(ctx, "arm_play", {"motion": "wave", "speed": 3})["ok"]
    assert host.log[-1] == {"op": "play", "motion": "wave", "speed": 1.0}     # clamped: never faster
    assert not run(ctx, "arm_play", {})["ok"]
    st = run(ctx, "arm_status")
    assert st["motions"] == ["wave"] and st["distance"]["mm"] == 120
    assert run(ctx, "arm_restart_sensor")["verified"] and run(ctx, "arm_check")["verified"]
    assert run(ctx, "arm_check")["next"].startswith("Fixed")             # what to do next, in words


def test_offsets_reach_the_host_and_bad_ones_are_refused(plugin):
    mod, ctx, host = plugin
    assert run(ctx, "arm_approach", {"dx_mm": 30, "dy_mm": -10})["ok"]
    assert host.log[-1] == {"op": "skill", "name": "approach", "offset_mm": [30.0, -10.0]}
    assert run(ctx, "arm_approach", {})["ok"] and "offset_mm" not in host.log[-1]
    assert not run(ctx, "arm_approach", {"dx_mm": "north"})["ok"]
    out = run(ctx, "arm_pick_up", {"dx_mm": 20})
    assert out["picked_up"] and host.log[-3]["offset_mm"] == [20.0, 0.0]
    assert ctx.hooks["pre_tool_call"]("arm_look", {}, task_id="t1") is None    # looking never moves the arm


def test_look_hands_the_model_the_top_view(plugin):
    """Hermes's multimodal envelope: the image for a vision model, a text summary for any other."""
    mod, ctx, host = plugin
    out = ctx.tools["arm_look"][2]({"dx_mm": 5}, task_id="t1")
    assert host.log[-1] == {"op": "look", "aim_mm": [5.0, 0.0], "inline": True}
    assert out["_multimodal"] is True and isinstance(out["content"], list)
    text, image = out["content"]
    assert text["type"] == "text" and json.loads(text["text"].split("\n\n")[0])["top"].endswith("top.png")
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert "top_png_b64" not in text["text"] and "to_pick_it" in out["text_summary"]
    host.state["inline_png"] = False                       # an older host: the paths, as plain JSON
    plain = json.loads(ctx.tools["arm_look"][2]({}, task_id="t1"))
    assert plain["ok"] and plain["side"].endswith("side.png")


def test_three_missed_picks_in_a_row_say_stop_and_ask(plugin):
    mod, ctx, host = plugin
    host.state["grasp_holding"] = False
    hints = [json.loads(ctx.tools["arm_pick_up"][2]({}, task_id="t9"))["hint"] for _ in range(3)]
    assert "stop retrying" not in hints[1] and "3 missed picks in a row: stop retrying" in hints[2]
    assert "stop retrying" not in json.loads(ctx.tools["arm_pick_up"][2]({}, task_id="other"))["hint"]
    host.state["grasp_holding"] = True
    assert run(ctx, "arm_pick_up")["picked_up"]                        # a success resets the count


def test_six_missed_picks_stop_picking_until_a_person_checks(plugin):
    mod, ctx, host = plugin
    guard = ctx.hooks["pre_tool_call"]
    host.state["grasp_holding"] = False
    for _ in range(6):
        assert guard("arm_pick_up", {}, task_id="t7") is None
        ctx.tools["arm_pick_up"][2]({}, task_id="t7")
    block = guard("arm_pick_up", {}, task_id="t7")
    assert block["action"] == "block" and "/arm check" in block["message"]
    assert guard("arm_go_home", {}, task_id="t7") is None             # other moves (and other tasks) are fine
    assert guard("arm_pick_up", {}, task_id="t8") is None
    ctx.commands["arm"]("check")                                     # a person checked: picks allowed again
    assert guard("arm_pick_up", {}, task_id="t7") is None


def test_grips_by_hand_count_toward_the_stop_too(plugin):
    mod, ctx, host = plugin
    guard = ctx.hooks["pre_tool_call"]
    host.state["grasp_holding"] = False
    for _ in range(6):
        ctx.tools["arm_grip"][2]({}, task_id="t5")
    assert "stop retrying" in json.loads(ctx.tools["arm_grip"][2]({}, task_id="t5"))["next"]
    assert guard("arm_approach", {}, task_id="t5")["action"] == "block"
    host.state["grasp_holding"] = True
    ctx.commands["arm"]("check")
    out = json.loads(ctx.tools["arm_grip"][2]({}, task_id="t5"))
    assert "next" not in out and mod._recent_misses("t5") == []          # a hold resets the count


def test_pick_up_stops_at_the_first_failed_step(plugin):
    mod, ctx, host = plugin
    assert run(ctx, "arm_pick_up")["picked_up"]
    host.state["grasp_holding"] = False
    out = run(ctx, "arm_pick_up")
    assert not out["picked_up"] and out["failed_at"] == "grip" and len(out["steps"]) == 2


def test_the_guard_blocks_motion_while_an_incident_is_open(plugin):
    mod, ctx, host = plugin
    guard = ctx.hooks["pre_tool_call"]
    assert guard("arm_reach", {}, task_id="t1") is None
    host.state["incidents"] = [{"id": "inc-1", "kind": "us_frozen", "summary": "Ultrasonic output is frozen"}]
    block = guard("arm_reach", {}, task_id="t1")
    assert block["action"] == "block" and "arm_restart_sensor" in block["message"]
    assert guard("arm_status", {}, task_id="t1") is None            # reading and repairing stay allowed
    assert guard("arm_restart_sensor", {}, task_id="t1") is None
    assert guard("arm_park", {}, task_id="t1") is None
    assert guard("terminal", {}, task_id="t1") is None               # other plugins' tools: not ours to judge
    host.state["incidents"] = [{"id": "inc-2", "kind": "joint_stall", "summary": "Elbow flex stopped",
                                "handed_off": 3.0}]
    assert "person" in guard("arm_pick_up", {}, task_id="t1")["message"]


def test_the_guard_blocks_on_servo_alarms_and_a_freeze(plugin):
    mod, ctx, host = plugin
    guard = ctx.hooks["pre_tool_call"]
    host.state["status"] = [0, 0, 0, 0, 0, 32]
    assert "alarm" in guard("arm_grip", {}, task_id="t1")["message"]
    host.state["status"] = [0] * 6
    ctx.commands["arm"]("freeze")
    assert "/arm unfreeze" in guard("arm_go_home", {}, task_id="t1")["message"]
    ctx.commands["arm"]("unfreeze")
    assert guard("arm_go_home", {}, task_id="t1") is None


def test_the_viewer_log_gets_tool_steps_guard_blocks_and_context_numbers(plugin):
    mod, ctx, host = plugin
    notes = lambda: [(r["lane"], r["text"]) for r in host.log if r["op"] == "note"]
    ctx.hooks["post_tool_call"](tool_name="arm_pick_up", args={"dx_mm": 30, "dy_mm": -10},
                                result=json.dumps({"ok": False, "picked_up": False, "failed_at": "grip"}))
    ctx.hooks["post_tool_call"](tool_name="terminal", args={"command": "secret stuff"}, result="whatever")
    ctx.commands["arm"]("freeze")
    ctx.hooks["pre_tool_call"]("arm_play", {"motion": "wave"}, task_id="t1")
    ctx.hooks["post_api_request"](usage={"prompt_tokens": 31200, "cache_read_tokens": 28000, "output_tokens": 120},
                                  context_length=400000, message_count=14, api_call_count=3,
                                  assistant_message="never sent", response={"x": "never sent"})
    ctx.hooks["post_compaction"](kind="auto", reason="threshold", before_tokens=12500, after_tokens=3100,
                                 messages_removed=9)
    said = notes()
    assert ("agent", "arm_pick_up(dx=30, dy=-10) -> FAILED at grip") in said
    assert ("agent", "terminal") in said                          # other tools: the name only, never the args
    assert any(lane == "agent" and text.startswith("GUARD blocked arm_play: The arm is frozen") for lane, text in said)
    assert ("context", "31.2k tokens = 8% of 400.0k, 90% cached, 14 messages") in said
    assert ("memory", "COMPACTED (auto): 12.5k -> 3.1k tokens, 9 messages folded") in said
    assert "never sent" not in json.dumps(said) and "secret" not in json.dumps(said)


def test_handoff_needs_an_open_incident(plugin):
    mod, ctx, host = plugin
    assert not run(ctx, "arm_handoff", {"reason": "x"})["ok"]
    host.state["incidents"] = [{"id": "inc-3", "kind": "us_noisy", "summary": "jitters"}]
    out = run(ctx, "arm_handoff", {"reason": "sensor keeps jittering after two restarts"})
    assert out["ok"] and out["id"] == "inc-3" and "/arm check" in out["next"]


def test_slash_command_status_reads_like_a_sentence(plugin):
    mod, ctx, host = plugin
    host.state["incidents"] = [{"id": "inc-1", "kind": "us_frozen", "summary": "frozen at 130 mm"}]
    text = ctx.commands["arm"]("")
    assert "inc-1 us_frozen: frozen at 130 mm" in text
    assert ctx.commands["arm"]("dance").startswith("usage")


def test_no_host_is_an_error_reply_not_an_exception(tmp_path):
    mod = load_plugin()
    ctx = FakeCtx(config={"socket": str(tmp_path / "nobody-home.sock")})
    mod.register(ctx)
    out = run(ctx, "arm_go_home")
    assert not out["ok"] and "python -m so101_host serve" in out["error"]
    assert ctx.hooks["pre_tool_call"]("arm_go_home", {}, task_id="t1") is None   # the tool reports it
    ctx.hooks["post_tool_call"](tool_name="arm_go_home", args={}, result=json.dumps(out))   # the log stays quiet
    ctx.hooks["post_api_request"](usage={"prompt_tokens": 10}, context_length=None, message_count=2, api_call_count=1)


def test_the_plugin_needs_only_the_standard_library():
    src = (ROOT / "__init__.py").read_text()
    imports = {line.split()[1].split(".")[0] for line in src.splitlines() if line.startswith(("import ", "from "))}
    assert imports <= {"json", "os", "socket", "tempfile", "time", "pathlib"}, imports
    assert not any(p.suffix in (".pyc", ".pyo") for p in ROOT.rglob("*") if ".git" not in p.parts
                   and "vendor" not in p.parts and "__pycache__" not in p.parts)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
