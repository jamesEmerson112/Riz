"""Matched planner comparison: Astra vs Sol on ONE frozen snapshot (same prompt, images, history, tools, effort).
Only the model differs. Never moves the arm. Run from prototypes/arm-sim."""
import json, shutil, sys, time
from pathlib import Path
import armsim.gentest as g

SNAP = Path(sys.argv[1]); SNAP.mkdir(parents=True, exist_ok=True)
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODELS = {"astra": "gpt-6-astra", "sol": "gpt-6-sol"}

# freeze the history exactly as the live loop builds it
history = []
for old in (g.OUT / "trials.jsonl", g.PLAN_LOG):
    for line in old.read_text().splitlines():
        try:
            h = json.loads(line); history.append({**h, "pan_offset": h.get("pan_offset"), "speed": h.get("speed", 1.0)})
        except json.JSONDecodeError:
            pass
for k, h in enumerate(history):
    if str(h.get("try", "")).startswith("g") and k + 1 < len(history) and history[k + 1].get("hermes_debug_before"):
        history[k]["debug"] = history[k + 1]["hermes_debug_before"]
        history[k]["bag_estimate"] = history[k + 1].get("hermes_bag_before")
last = history[-1]
before, after = SNAP / "before.jpg", SNAP / "after.jpg"
shutil.copy(last["before"], before); shutil.copy(last["after"], after)
g.OUT = SNAP                                  # context.json goes here, not into the live loop's folder
print(f"snapshot: {len(history)} tries, last {last['try']} {last['recording']} pan {last['pan_offset']} | images {last['after']}", flush=True)

rows = []
for i in range(RUNS):
    for name, model in MODELS.items():         # alternate, so neither model always goes first
        g.HERMES_MODEL = ["-m", model, "--provider", "openai-codex"]
        t0 = time.time()
        p = g.plan(history, before, after)
        wall = round(time.time() - t0, 1)
        n = p["next"] or {}
        row = {"model": name, "run": i + 1, "wall_s": wall, "parsed": bool(p["next"]), "verdict": p["verdict"],
               "next": f"{n.get('recording')} pan {n.get('pan')} speed {n.get('speed')}" if n else None,
               "bag": p.get("bag"), "debug": p["debug"][:160], "why": n.get("why", "")[:160]}
        rows.append(row)
        print(json.dumps(row), flush=True)
(SNAP / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
