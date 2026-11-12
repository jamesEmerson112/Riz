# Hackathon data, Sun Nov 8 2026 (local only)

Everything the Naïve Nous YC hackathon produced on James's Mac, gathered from the temporary Claude worktrees and scratch folders on Nov 9. Published here on Nov 11, when the hackathon repo's local copy was removed. Before upload it was scanned for API keys, tokens and private keys, and none were found. The `.env` files and local notes were left out. Paths below such as `prototypes/arm-sim` and `docker compose` refer to the hackathon repo, not this one.

| Folder | What it is | Size |
|---|---|---|
| `arm-runs/gentest/` | The chips generate-and-test runs on the real SO-101 (red bag into the box), 11:54–14:38 | 58 MB |
| `arm-runs/runs-other/` | The rest of the arm-sim `runs/`: earlier run records (Nov 7–8), dispatcher, escalation and event logs | 81 MB |
| `arm-runs/README.txt` | File-by-file guide to `arm-runs/`, written when it was first backed up (17:23) | |
| `romulus/db/romulus-2026-11-08.sql.gz` | `pg_dump` of the Romulus database: devices, incidents, work orders, events, metric samples, recordings, labels | 124 KB |
| `romulus/db/row-counts.txt` | Rows per table at dump time | |
| `romulus/data/` | Romulus's `./data` bind mount: blobs, raw telemetry datasets, the arm twin's records, level-2 results, sim faults | 89 MB |
| `experiments/` | Matched Astra tests: `sol-vs-astra/`, `reasoning-effort/`, `full-vs-lean/`, `lean-12-tries/` (each has its frozen snapshot, script and `results.jsonl`) | |
| `experiments/scripts/` | Session helpers: the Astra feed forwarder, the restart-between-moves switcher, the success-GIF watcher, `fold_home.py` (fold the arm home in stages after a power cut), `try_gallery.py` (per-try GIF gallery), and an unfinished `--task`/`drop_pan` patch | |
| `agent-worktrees/` | Test leftovers from two subagent worktrees (Mission Control split view, feed hover) | 164 KB |
| `pi-runs/` | Empty: the Pi's own logs (see below) | |

## The gentest log in short
- `arm-runs/gentest/hermes-trials.jsonl`: one row per try (96).
  - **What was run:** recording, pan, speed, effort.
  - **What happened:** grip time and load, the jaw tip and joints at close and open.
  - **What Astra said:** its diagnosis, bag estimate, grid cells and verdict.
  - **Extras:** probe results, re-grab attempts, frame sequences.
- `trials.jsonl` is the first fixed run (6 tries). `live-1211.log` is the loop's full console log.
- `astra/` holds the exact prompts sent to Astra and its raw replies, from 13:55 on.
- `*.jpg` are the camera frames: before, jaws close, probe, after, and samples every 4 s. `*-grid*` files are the gridded copies Astra saw, and `*-sheet.jpg` files are timed frame sheets of whole tries. `gallery/` has one GIF per try.
- **Best try:** 13:50:27, `recording_2`, pan −4, speed 0.5, carried three bags for 20.7 s. Its jaws never opened, so it carried them home.

## Charts
`python -m armsim.report` (in `prototypes/arm-sim`) reads this folder and rewrites the README charts in `prototypes/arm-sim/media/gentest/`. Only the charts and a few downscaled frames are committed; the raw data stays here.

## Bringing Romulus back
The live database is the Docker volume `scheduled-md-sync-task-49ac70_pgdata`. It was kept when the stack was stopped (`docker compose down` without `-v`).
- **Same volume:** from the repo root, run `COMPOSE_PROJECT_NAME=scheduled-md-sync-task-49ac70 docker compose up -d`. Copy `romulus/data/` to `./data` first, since the API bind-mounts `./data`.
- **From the dump instead:** start a fresh stack, then `gzip -dc romulus/db/romulus-2026-11-08.sql.gz | docker compose exec -T db psql -U <user> -d <db>`.

## Not here
- **The Pi's own logs** were never copied, and the Pi was powered down before anyone could: the arm host's session records (servo ticks), the Pi-side event log behind Mission Control's commander feed, and the podd and edge-bridge logs. arm-1's status reports and incidents did reach Romulus, so they're in the dump. If the Pi comes back, copy them into `pi-runs/`:
  `rsync -a pi@192.168.50.2:~/sage_hardware_healer/Naive-Nous-YC-Hackathon/prototypes/arm-sim/runs/ pi-runs/`
- **Python environments and the MuJoCo model** (`.venv`, `vendor/`) aren't kept; `setup.sh` rebuilds them.
