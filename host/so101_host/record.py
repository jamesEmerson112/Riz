"""Run records: one JSON Lines file per run, the same shape in sim and on the real arm.

Line 1 is a header (`kind: "run"`). Then two row kinds:
- `J` at the control rate: commanded and actual joints, actuator load, status bits;
- `D` at the ultrasonic's rate: the distance fields `mm`, `us`, `st`.
The last line (`kind: "end"`) holds the outcome and any findings, so a sim run and a real-arm run
can be labeled and compared the same way.
"""
import json
import time
import uuid
from pathlib import Path

SCHEMA = "arm-run/v1"


class RunRecorder:
    def __init__(self, out_dir, *, task, source="sim", arm="sim-so101", joints=(), label=None):
        self.run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = Path(out_dir) / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._write({
            "kind": "run", "schema": SCHEMA, "run_id": self.run_id, "source": source, "arm": arm,
            "task": task, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "joints": list(joints),
            "units": {"pos": "deg (arm joints), 0-100 (gripper, 0 = closed)", "load": "N*m",
                      "dist": "mm, us as an HC-SR04 driver reports them"},
            "label": label or {"fault": None},
        })
        self.rows = {"J": 0, "D": 0}

    def _write(self, obj):
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def joints(self, t, cmd, pos, load, status):
        self.rows["J"] += 1
        self._write({"kind": "J", "t": round(t, 4), "cmd": [round(x, 3) for x in cmd],
                     "pos": [round(x, 3) for x in pos], "load": [round(x, 4) for x in load],
                     "status": list(status)})

    def distance(self, t, row):
        self.rows["D"] += 1
        self._write({"kind": "D", "t": round(t, 4), **row})

    def close(self, outcome, findings=(), label=None):
        end = {"kind": "end", "outcome": outcome, "findings": list(findings), "rows": self.rows}
        if label is not None:
            end["label"] = label            # the final label, e.g. with the fault's onset time
        self._write(end)
        self._f.close()
        return self.path
