"""Watch gentest's try log; on each new success (a success row, or an INSIDE verdict), build the success GIF(s)."""
import json, subprocess, sys, time
from pathlib import Path
LOG = Path("runs/gentest/hermes-trials.jsonl")
seen = LOG.stat().st_size if LOG.exists() else 0
print(f"watching {LOG} from byte {seen}", flush=True)
while True:
    time.sleep(3)
    size = LOG.stat().st_size if LOG.exists() else 0
    if size <= seen:
        continue
    with open(LOG) as f:
        f.seek(seen); new = f.read()
    seen = size
    for line in new.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("success") or r.get("hermes_verdict_before") == "INSIDE":
            out = subprocess.run([sys.executable, "-m", "armsim.gifs"], capture_output=True, text=True,
                                 env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"})
            print(f"SUCCESS seen at {time.strftime('%H:%M:%S')} (try {r.get('of') or r.get('try')}):", out.stdout.strip().replace("\n", " | "), out.stderr[-200:], flush=True)
