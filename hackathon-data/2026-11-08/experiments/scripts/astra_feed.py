"""Forward each Hermes-guided trial (runs/gentest/hermes-trials.jsonl on the Mac) to Mission Control's
commander feed on the Pi (its runs/events.jsonl), as Astra's diagnosis, the test it designed, and the result."""
import json, subprocess, sys, time
from pathlib import Path
log = Path(sys.argv[1])
CMD = ("cd ~/sage_hardware_healer/Naive-Nous-YC-Hackathon/prototypes/arm-sim && PYTHONPATH=. .venv/bin/python -c "
       "\"import json,sys; from armsim import events; [events.emit('astra', 'commander', t, echo=False) "
       "for t in json.load(sys.stdin)]\"")
def post(texts):
    subprocess.run(["ssh", "-o", "BatchMode=yes", "pi@192.168.50.2", CMD], input=json.dumps(texts), text=True,
                   capture_output=True, timeout=30)
seen = 0
post(["Astra (Hermes) is debugging the chips pick: one test per trial until the bag is in the box"])
while True:
    lines = log.read_text().splitlines() if log.exists() else []
    for line in lines[seen:]:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = r.get("plan") or {}
        post([f"Astra debug: {r.get('hermes_debug_before', '')[:420]}",
              f"Astra test {r.get('try')}: {p.get('recording')} pan {p.get('pan'):+} speed {p.get('speed')} - {p.get('why', '')[:200]}",
              f"Result {r.get('try')}: {'GRASPED' if r.get('grasped') else 'no grasp'} (held {r.get('holding_s')} s, widest "
              f"{r.get('widest_hold')}), grasp point {((r.get('close_at') or {}).get('tip_m'))} m"])
    seen = len(lines)
    time.sleep(2)
