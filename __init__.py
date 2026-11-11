"""so101-arm: give Hermes a robot arm, safely.

The plugin turns an SO-101 arm (the MuJoCo sim, or a real arm through LeRobot) into Hermes tools. Each
tool asks the local arm host (host/, a separate process the user starts) to run one named skill: go
home, reach, approach, grip, lift, release, park, play a taught motion. The model picks skills; it never
sends joint angles, and no model sits in the real-time loop.

Safety lives outside the model, in two places:
  - the arm host's watchdog (plain code) parks the arm on a fault (a frozen or wrong distance reading, a
    joint that stalls, a servo alarm), and the host refuses motion until a repair is verified, or while
    a person has frozen the arm;
  - this plugin's pre_tool_call hook blocks every motion tool in those same states, and on a servo
    alarm, before the call reaches the host, and tells the agent the way out.
No tool in this plugin can unfreeze the arm: a person does that with `/arm unfreeze`. (Other tools, a
terminal for one, can still reach the host's socket like any program you run; see the README.)

What it touches: one Unix socket on this machine ($SO101_SOCK or the plugin's `socket` setting, else
<temp dir>/so101-arm.sock). No network, no shell commands, no files outside its own folder (it reads its
SKILL.md), no background processes, no credentials.
"""
import json
import os
import socket
import tempfile
import time
from pathlib import Path

TOOLSET = "so101_arm"
_STATE = {"socket": None, "misses": {}}
MISS_LIMIT = 3          # failed picks in a row (per task) before the reply says to stop and ask the user
MISS_BLOCK = 6          # ...and before the guard refuses more picks until a person's /arm check
MISS_FORGET_S = 600     # a miss older than this no longer counts
PICKING = {"arm_pick_up", "arm_approach", "arm_grip"}

SKILLS = {
    "arm_go_home": ("go_home", "Move the arm to its ready pose over the work area. Use it first, and to reset "
                               "before retrying a task."),
    "arm_reach": ("reach", "Move the open gripper over the block, ready to approach it."),
    "arm_approach": ("approach", "Move over the block with the jaws open, then step down using the wrist distance "
                                 "sensor (5 mm at a time, sim) or the taught pick pose (real arm), so the jaws "
                                 "straddle the block. dx_mm/dy_mm (each within 40 mm) aim off the taught spot "
                                 "when arm_look shows the block has moved. To pick something up: arm_approach, "
                                 "arm_grip, arm_lift (or arm_pick_up, which does all three with a check after each)."),
    "arm_grip": ("grip", "Close the gripper. The result's grasp.holding says whether it holds something (the "
                         "jaws stopped short of closed, with load)."),
    "arm_lift": ("lift", "Lift the gripper about 30 mm and check it is still holding."),
    "arm_release": ("release", "Open the gripper. Right after arm_lift, this sets the block back down where it was "
                               "picked up (the lift is only about 30 mm), which is how to put it back."),
    "arm_park": ("park", "Fold the arm to its rest pose and switch torque off. Always allowed, even during a fault."),
}
MOTION = {"arm_go_home", "arm_reach", "arm_approach", "arm_grip", "arm_lift", "arm_release", "arm_play",
          "arm_pick_up"}
TOOLS = [*SKILLS, "arm_play", "arm_pick_up", "arm_look", "arm_status", "arm_restart_sensor", "arm_check",
         "arm_handoff"]
OFFSET = {"dx_mm": {"type": "number", "description": "aim this many mm forward (+) or back (-) of the taught "
                                                     "block spot, at most 40"},
          "dy_mm": {"type": "number", "description": "aim this many mm left (+) or right (-), at most 40"}}


def _socket_path():
    return (_STATE["socket"] or os.environ.get("SO101_SOCK")
            or os.path.join(tempfile.gettempdir(), "so101-arm.sock"))


def _call(op, timeout=120.0, **fields):
    """One request to the arm host. Never raises: an unreachable host is an error reply."""
    path = _socket_path()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(path)
            s.sendall((json.dumps({"op": op, **fields}) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        return json.loads(buf or b'{"ok": false, "error": "no reply from the arm host"}')
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"arm host not reachable at {path}: {exc}. Start it in the plugin's "
                                      "repo: `python -m so101_host serve` (sim) or `serve --real`."}


def _schema(name, description, properties=None, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}


def _json(obj):
    return json.dumps(obj, default=str)


def _offset(args):
    """dx_mm/dy_mm from the tool call, or None. The host enforces the 40 mm limit; this only parses."""
    args = args or {}
    if args.get("dx_mm") in (None, "") and args.get("dy_mm") in (None, ""):
        return None
    try:
        return [float(args.get("dx_mm") or 0.0), float(args.get("dy_mm") or 0.0)]
    except (TypeError, ValueError):
        return "bad"


def _incident_next(inc):
    """The one repair to try when the watchdog parked the arm mid-move."""
    kind = str((inc or {}).get("kind", ""))
    if (inc or {}).get("handed_off"):
        return "A person has this incident: tell the user what failed and wait for their /arm check."
    if kind.startswith("us_"):
        return "The watchdog parked the arm on a wrist-sensor fault. Run arm_restart_sensor, then retry the task."
    return ("The watchdog parked the arm: a joint did not reach its target. Call arm_check now, before you answer "
            "the user: it re-tests at home and, if the joint still fails, hands it to a person. Then tell the user "
            "what it found.")


def _with_incident_next(reply):
    if reply.get("incident"):
        return {"next": _incident_next(reply["incident"]), **reply}
    return reply


def _skill_handler(skill):
    def handler(args=None, **kwargs):
        if skill == "approach":
            off = _offset(args)
            if off == "bad":
                return _json({"ok": False, "error": "dx_mm and dy_mm must be numbers"})
            if off is not None:
                return _json(_with_incident_next(_call("skill", name=skill, offset_mm=off)))
        return _json(_with_incident_next(_call("skill", name=skill)))
    return handler


def play(args=None, **kwargs):
    args = args or {}
    motion = str(args.get("motion") or "").strip()
    if not motion:
        return _json({"ok": False, "error": "say which motion: arm_status lists them"})
    speed = args.get("speed", 1.0)
    try:
        speed = min(max(float(speed), 0.25), 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    return _json(_with_incident_next(_call("play", motion=motion, speed=speed)))


def _recent_misses(task_id):
    now = time.monotonic()
    return [t for t in _STATE["misses"].get(task_id, []) if now - t < MISS_FORGET_S]


def _count_miss(task_id, missed):
    """Consecutive failed picks for this task: a physical retry shouldn't loop forever."""
    now = time.monotonic()
    misses = [t for t in _STATE["misses"].get(task_id, []) if now - t < MISS_FORGET_S] if missed else []
    if missed:
        misses.append(now)
    _STATE["misses"][task_id] = misses
    return len(misses)


def pick_up(args=None, **kwargs):
    """approach -> grip -> lift, stopping at the first step that fails, so the agent sees exactly where
    a pick went wrong (and the watchdog has already parked the arm if it was a fault)."""
    off = _offset(args)
    if off == "bad":
        return _json({"ok": False, "error": "dx_mm and dy_mm must be numbers"})
    steps = []
    for name, ok in (("approach", lambda r: r.get("arrived")),
                     ("grip", lambda r: (r.get("grasp") or {}).get("holding")),
                     ("lift", lambda r: r.get("arrived") and (r.get("grasp") or {}).get("holding"))):
        reply = _call("skill", name=name, **({"offset_mm": off} if name == "approach" and off else {}))
        result = reply.get("result") or {}
        steps.append({"step": name, "ok": bool(reply.get("ok") and ok(result)), "reply": reply})
        if not steps[-1]["ok"]:
            measured = None
            if reply.get("error"):                      # the host refused the call itself
                hint = f"The arm host refused it: {reply['error']}. Don't retry the same call."
            elif reply.get("incident"):
                hint = _incident_next(reply["incident"])
            elif name == "approach":
                hint = "The arm is parked by a fault: check arm_status and repair first."
            else:
                hint, measured = _missed_block_hint()
            # the verdict and what to do first, the details after: small models read from the top
            if name != "approach" or reply.get("error"):
                n = _count_miss(kwargs.get("task_id") or "", True)
                if n >= MISS_LIMIT:
                    hint = (f"That is {n} missed picks in a row: stop retrying. Tell the user what happened and what "
                            "arm_look shows, and let them check the block. " + hint)
            out = {"ok": False, "picked_up": False, "failed_at": name, "hint": hint}
            if measured:
                out["measured"] = measured
            return _json({**out, "steps": steps})
    _count_miss(kwargs.get("task_id") or "", False)
    return _json({"ok": True, "picked_up": True,
                  "next": "Holding the block about 30 mm up. To put it back, call arm_release: it sets the block "
                          "down where it was picked up.", "steps": steps})


def _missed_block_hint():
    """After a grip that closed on nothing: measure where the block really is (the sim's camera, plain code)
    and say exactly what to do next. Without a camera (the real arm), say how to find out."""
    seen = _call("look", aim_mm=[0.0, 0.0], timeout=30.0)
    if "no camera" in str(seen.get("error", "")):     # the real arm: nothing to look with
        return ("The jaws closed on nothing, so the block isn't on the taught spot. Do arm_release and tell the user "
                "to put the block back on its marked spot; this arm has no camera to find it."), None
    if not seen.get("ok") or not seen.get("block_seen"):
        return ("The jaws closed on nothing, so the block isn't where the arm aimed. Do arm_release, then arm_look "
                "to see where it is."), None
    bx, by = seen["block_mm"]
    measured = {"block_mm": seen["block_mm"], "reachable": seen.get("reachable", True)}
    if seen.get("reachable") is False:
        return (f"The jaws closed on nothing: the block was moved to dx_mm={bx}, dy_mm={by}, past the 40 mm the arm "
                "may correct. Do arm_release, then tell the user to move it back near the marked spot."), measured
    return (f"The jaws closed on nothing: the block was moved. Plain code measured it at dx_mm={bx}, dy_mm={by}. "
            f"Do arm_release, then arm_pick_up with dx_mm={bx} and dy_mm={by}."), measured


def look(args=None, **kwargs):
    """Render the sim's cameras. The top view comes back inside the tool result (Hermes's multimodal
    envelope), so a vision model reads the grid itself; Hermes sends the text summary instead to a model
    or provider that can't take images in tool results, and that summary points at vision_analyze."""
    off = _offset(args)
    if off == "bad":
        return _json({"ok": False, "error": "dx_mm and dy_mm must be numbers"})
    reply = _call("look", aim_mm=off or [0.0, 0.0], inline=True)
    png = reply.pop("top_png_b64", None)
    text = _json(reply)
    if not (reply.get("ok") and png):
        return text
    return {"_multimodal": True,
            "content": [{"type": "text", "text": text + "\n\nThe top view is attached: read the grid with your "
                                                         "own vision."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}],
            "text_summary": text + "\n\nThe numbers above were measured for you: use to_pick_it. (The map "
                                   "image is at the 'top' path, for vision_analyze if you want to look.)",
            "meta": {"image_path": reply.get("top"), "native_vision": True}}


def status(args=None, **kwargs):
    st = _call("status")
    if not st.get("ok"):
        return _json(st)
    return _json({"status": st, "joints": _call("read").get("pos"), "distance": _call("distance").get("row"),
                  "motions": _call("motions").get("motions")})


def _with_next(reply):
    """Say what comes next after a repair or a hand-off, in words the agent can pass on to the user. It goes
    first in the reply: small models read from the top."""
    if not reply.get("ok"):
        return reply
    if reply.get("handed_off"):
        nxt = ("This needs a person. Stop moving the arm, and tell the user what failed and that a person must fix "
               "it and then type /arm check. The arm stays parked until then.")
    elif reply.get("verified"):
        nxt = "Fixed and checked at home. Carry on with the task."
    elif reply.get("verified") is False:
        nxt = ("Not fixed. Read arm_status; if it still fails, hand it to a person with arm_handoff and tell the "
               "user what failed.")
    else:
        return reply
    return {"next": nxt, **reply}


def restart_sensor(args=None, **kwargs):
    return _json(_with_next(_call("repair", action="restart_sensor")))


def check(args=None, **kwargs):
    return _json(_with_next(_call("repair", action="verify")))


def handoff(args=None, **kwargs):
    args = args or {}
    st = _call("status")
    if not st.get("ok"):
        return _json(st)
    open_ = st.get("incidents") or []
    if not open_:
        return _json({"ok": False, "error": "no open incident to hand off"})
    reason = str(args.get("reason") or "the agent asked for a person").strip()[:300]
    return _json(_with_next(_call("handoff", id=open_[0]["id"], reason=reason)))


def guard(tool_name, args=None, task_id="", **kwargs):
    """Block motion while the arm isn't safe to move. Returns None (allow) or a block directive."""
    if tool_name not in MOTION:
        return None
    if tool_name in PICKING and len(_recent_misses(task_id or "")) >= MISS_BLOCK:
        return {"action": "block", "message": f"{MISS_BLOCK} picks in a row missed the block, so no more tries in "
                                              "this task. Tell the user what happened and what arm_look shows; a "
                                              "person can check the block and type /arm check to allow picks again."}
    st = _call("status", timeout=10.0)
    if not st.get("ok"):
        return None                                   # the tool itself reports the host problem
    if st.get("frozen"):
        return {"action": "block", "message": f"The arm is frozen by the operator ({st['frozen']}). No tool can "
                                              "unfreeze it: a person does, with /arm unfreeze. Tell the user and "
                                              "wait."}
    incidents = st.get("incidents") or []
    if incidents:
        said = "; ".join(f"{i['id']} {i['kind']}: {i['summary']}" for i in incidents)
        if any(i.get("handed_off") for i in incidents):
            way = ("It is handed to a person: tell the user what failed and wait for their fix and their "
                   "/arm check.")
        elif any(i["kind"].startswith("us_") for i in incidents):
            way = "Fix the cause first: arm_restart_sensor (a stuck or wrong distance), then retry."
        else:
            way = "Fix the cause first: arm_check re-tests at home; if it still fails, it goes to a person."
        return {"action": "block", "message": f"The arm is parked: {said}. {way}"}
    if any(st.get("status") or []):
        return {"action": "block", "message": f"A servo reports an alarm (status bits {st['status']}). "
                                              "Park the arm (arm_park) and report it instead of moving."}
    return None


def slash_arm(raw_args="", **kwargs):
    """/arm [status|park|check|freeze|unfreeze]: the person's controls. The kill switch lives here only."""
    words = (raw_args or "").split()
    cmd = words[0] if words else "status"
    if cmd == "status":
        st = _call("status")
        if not st.get("ok"):
            return st.get("error", "arm host not reachable")
        lines = [f"arm: {st.get('arm')} · parked: {st.get('parked')} · frozen: {st.get('frozen') or 'no'}"]
        for inc in st.get("incidents") or []:
            who = " (with a person)" if inc.get("handed_off") else ""
            lines.append(f"  {inc['id']} {inc['kind']}{who}: {inc['summary']}")
        if not st.get("incidents"):
            lines.append("  no open incidents")
        return "\n".join(lines)
    if cmd == "park":
        return _json(_call("skill", name="park"))
    if cmd == "check":
        _STATE["misses"].clear()                      # a person looked: picks are allowed again
        return _json(_call("repair", action="verify"))
    if cmd in ("freeze", "unfreeze"):
        return _json(_call(cmd, by="/arm (a person in the chat)"))
    return "usage: /arm [status|park|check|freeze|unfreeze]"


def register(ctx):
    try:
        configured = ctx.get_config("socket", default="")
    except Exception:                                 # older Hermes without plugin settings
        configured = ""
    _STATE["socket"] = configured or None

    for tool, (skill, description) in SKILLS.items():
        props = OFFSET if tool == "arm_approach" else None
        ctx.register_tool(name=tool, toolset=TOOLSET, schema=_schema(tool, description, props),
                          handler=_skill_handler(skill))
    ctx.register_tool(
        name="arm_play", toolset=TOOLSET, handler=play,
        schema=_schema("arm_play", "Play a motion taught on this arm (arm_status lists them; the sim has 'wave'). "
                                   "Speed 0.25-1.0 of the recorded speed, never faster.",
                       {"motion": {"type": "string", "description": "the motion's name"},
                        "speed": {"type": "number", "description": "0.25 to 1.0 (default 1.0)"}}, ["motion"]))
    ctx.register_tool(
        name="arm_pick_up", toolset=TOOLSET, handler=pick_up,
        schema=_schema("arm_pick_up", "Pick up the block: approach, grip, then lift, checking after each step "
                                      "and stopping at the first that fails. Returns every step's result. "
                                      "dx_mm/dy_mm correct the aim (read them off arm_look's grid).", OFFSET))
    ctx.register_tool(
        name="arm_look", toolset=TOOLSET, handler=look,
        schema=_schema("arm_look", "Look at the work area (sim): a top-down map of the table with a 10 mm grid "
                                   "numbered in dx_mm/dy_mm, the block, a red + where the arm will aim and a blue "
                                   "x where the jaws are. The result has the map as an image, and block_mm: where "
                                   "plain code measured the block's centre on that grid. To pick the block up, "
                                   "call arm_pick_up with the to_pick_it numbers (dx_mm, dy_mm); check them "
                                   "against the image first. If it says reachable: false, the block is past the "
                                   "40 mm the arm may correct: don't try, tell the user to move it back near "
                                   "the marked spot. If you can't see the image, open the 'top' path "
                                   "with vision_analyze. Pass dx_mm/dy_mm here to preview an aim without "
                                   "moving.", OFFSET))
    ctx.register_tool(
        name="arm_status", toolset=TOOLSET, handler=status,
        schema=_schema("arm_status", "Read the arm: open incidents (and whether a person has one), servo alarm "
                                     "bits, grasp, joint positions, the wrist distance sensor, frozen or not, and "
                                     "the taught motions. Call it before moving and after anything fails."))
    ctx.register_tool(
        name="arm_restart_sensor", toolset=TOOLSET, handler=restart_sensor,
        schema=_schema("arm_restart_sensor", "Repair: restart the wrist distance sensor, then the watchdog moves "
                                             "the arm home and checks fresh readings. Fixes a frozen, stuck or "
                                             "wrong distance. The result says verified true or false."))
    ctx.register_tool(
        name="arm_check", toolset=TOOLSET, handler=check,
        schema=_schema("arm_check", "Re-check without a fix: the watchdog moves the arm home and re-tests, closing "
                                    "what passes. Use after a joint stall. A joint that still won't move is "
                                    "handed to a person automatically."))
    ctx.register_tool(
        name="arm_handoff", toolset=TOOLSET, handler=handoff,
        schema=_schema("arm_handoff", "Hand the open incident to a person when the tools can't fix it. The arm "
                                      "stays parked until the person fixes it and runs /arm check.",
                       {"reason": {"type": "string", "description": "what failed and what you tried"}}))
    ctx.register_hook("pre_tool_call", guard)
    ctx.register_command("arm", slash_arm, description="SO-101 arm: status, park, check, freeze, unfreeze",
                         args_hint="status|park|check|freeze|unfreeze")

    skills_dir = Path(__file__).parent / "skills"
    if skills_dir.is_dir() and hasattr(ctx, "register_skill"):
        for child in sorted(skills_dir.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                ctx.register_skill(child.name, skill_md, description=_skill_description(skill_md))


def _skill_description(skill_md):
    """The `description:` line of a SKILL.md's frontmatter, for the skill's row in <available_skills>."""
    lines = skill_md.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line.startswith("description:"):
                return line.partition(":")[2].strip().strip('"')
    return ""
