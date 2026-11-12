"""Wait for the running trial's result line, then SIGINT the loop (it is planning, the arm is at rest) and restart it."""
import os, signal, subprocess, sys, time
LOG = "runs/gentest/live-1211.log"
def pid():
    r = subprocess.run(["pgrep", "-f", r"^\.venv/bin/python -u -m armsim.gentest run"], capture_output=True, text=True)
    return int(r.stdout.split()[0]) if r.stdout.strip() else None
start = open(LOG).read(); n = start.count("  -> ")
print(f"waiting for trial result #{n + 1}", flush=True)
while True:
    t = open(LOG).read()
    new = t[len(start):]
    if pid() is None or "SUCCESS" in new or "stopping:" in new:
        print("loop ended by itself; not restarting"); sys.exit(0)
    if t.count("  -> ") > n:
        break
    time.sleep(0.5)
p = pid(); os.kill(p, signal.SIGTERM)   # started in the background, the loop ignores SIGINT
while pid() == p:
    time.sleep(0.5)
print("old loop stopped:", open(LOG).read().strip().splitlines()[-1][:160], flush=True)
with open(LOG, "a") as f:
    f.write(f"---- restart: no re-grab (it dropped thin edge grips), full motion straight through {time.strftime('%H:%M:%S')} ----\n")
new = subprocess.Popen([".venv/bin/python", "-u", "-m", "armsim.gentest", "run", "--guide", "hermes", "--only", "recording_2", "--probe-once", "--reasoning", "none,low,medium", "--max-trials", "300"],
                       stdout=open(LOG, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                       env={**os.environ, "PYTHONPATH": "."}, start_new_session=True,
                       preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
print("restarted pid", new.pid, flush=True)
