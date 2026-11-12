"""Matched reasoning-effort comparison on Astra: one frozen snapshot, same prompt/images/model; only --reasoning differs.
Times both calls the loop makes (plan, probe go/no-go). Never moves the arm. Run from prototypes/arm-sim."""
import json, shutil, sys, time
from pathlib import Path
import armsim.gentest as g

SNAP = Path(sys.argv[1]); SNAP.mkdir(parents=True, exist_ok=True)
MODES, RUNS = ["full", "lean"], 3
history = []
for old in (g.OUT / "trials.jsonl", g.PLAN_LOG):
    for line in old.read_text().splitlines():
        try:
            h = json.loads(line); history.append({**h, "pan_offset": h.get("pan_offset"), "speed": h.get("speed", 1.0)})
        except json.JSONDecodeError:
            pass
for k, h in enumerate(history):
    if k + 1 < len(history) and history[k + 1].get("hermes_debug_before"):
        history[k]["debug"] = history[k + 1]["hermes_debug_before"]
        history[k]["bag_estimate"] = history[k + 1].get("hermes_bag_before")
last = history[-1]
before, after = SNAP / "before.jpg", SNAP / "after.jpg"
shutil.copy(last["before"], before); shutil.copy(last["after"], after)
bag = last.get("bag_estimate") or {"near_edge_m": [0.135, -0.082]}
g.OUT = SNAP
print(f"snapshot: {len(history)} tries, last {last['try']} {last['recording']} pan {last['pan_offset']}", flush=True)
rows = []
for i in range(RUNS):
    for lvl in MODES:
        g.LEAN = (lvl == "lean")
        t0 = time.time(); p = g.plan(history, before, after); plan_s = round(time.time() - t0, 1)
        go, why, check_s, cells = g.probe_check(after, bag)
        n = p["next"] or {}
        row = {"mode": lvl, "run": i + 1, "plan_s": plan_s, "check_s": check_s, "plan_parsed": bool(p["next"]),
               "next": f"{n.get('recording')} pan {n.get('pan')}" if n else None,
               "plan_cells": [ (p.get('bag') or {}).get('jaws_cell'), (p.get('bag') or {}).get('edge_cell')],
               "check": {"go": go, **cells}, "why": why[:120]}
        rows.append(row); print(json.dumps(row), flush=True)
(SNAP / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
