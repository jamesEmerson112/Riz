---
license: other
license_name: per-subset
license_link: https://huggingface.co/datasets/Forgis/FactoryNet#licensing
pretty_name: FactoryNet
size_categories:
  - 100M<n<1B
task_categories:
  - time-series-forecasting
  - tabular-classification
tags:
  - industrial
  - robotics
  - anomaly-detection
  - time-series
  - sim-to-real
  - manufacturing
  - predictive-maintenance
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/*.parquet
---

# FactoryNet

A multi-embodiment industrial time-series corpus: **56,591 end-to-end task executions**
(14,847 real, 41,744 simulated), **113M logged timesteps**, **7 embodiments**, **4 tasks**,
**27 annotated anomaly types** with healthy baselines and counterfactual pairs.

Every signal from every source is mapped into one control-theoretic schema —
**Setpoint, Effort, Feedback, Context (S-E-F-C)** — so a single dataloader works across a
6-DOF arm and a 4-axis CNC gantry, and *commanded versus realized* dynamics are readable
as explicit prediction residuals rather than opaque reconstruction scores.


## Quick start

```python
from datasets import load_dataset

ds = load_dataset("Forgis/FactoryNet", split="train", streaming=True)
print(next(iter(ds)))
```

Or read the Parquet directly, which is usually what you want for time series.
`cnc_000.parquet` is 1.1 MB, so this is a fast first look:

```python
import pandas as pd
from huggingface_hub import hf_hub_download

p = hf_hub_download("Forgis/FactoryNet", "data/cnc_000.parquet",
                    repo_type="dataset")
df = pd.read_parquet(p)

# the same four prefixes on every machine, without vendor-specific names
S = [c for c in df.columns if c.startswith("setpoint_")]
E = [c for c in df.columns if c.startswith("effort_")]
F = [c for c in df.columns if c.startswith("feedback_")]
C = [c for c in df.columns if c.startswith("ctx_")]
```

The same five lines work unchanged on a 6-DOF arm. The consolidated real UR
recording, `data/factorywave_ur_consolidated.parquet`, is 992 MB with 7,129,261
rows and 125 columns (44 Setpoint, 19 Effort, 25 Feedback, 30 Context), so pass
`columns=[...]` to `read_parquet` rather than loading it whole.

### What is where

| Prefix in `data/` | Source | Files |
|---|---|---|
| `factorywave_*` | our laboratory UR3 and KUKA KR10 recordings, incl. screwdriver telemetry and episode metadata | 6 |
| `voraus_*` | voraus-AD | 60 |
| `aursad_*` | AURSAD | 32 |
| `cnc_*` | UMich CNC milling | 1 |
| `simulations_baseline_*` | simulated nominal episodes | 11 |
| `simulations_counterfactual_*` | matched simulated counterfactuals | 10 |

`data/` is 13.7 GB of Parquet. The repository also carries the paired sim-to-real
gap artifacts used for the validation campaign: `real_csv/` and `sim_csv/` (episodes
paired by filename), `pick_configs/` (per-episode simulation parameters), and
`gap_reports/` + `summary/` (per-episode and aggregated gap analysis).

## The S-E-F-C schema

One wide table per source: one row per control tick, joined to an episode table by
`episode_id`. The column prefix carries the role; the suffix `_i` is the positional joint
index 0..5 (base → wrist), so the same column means the same slot on every arm. A column a
source does not expose is written as null rather than dropped, so all sources share one
schema.

| Role | Prefix | What it is | Examples |
|---|---|---|---|
| **Setpoint** | `setpoint_*` | commanded intent | target joint position, velocity, acceleration; target TCP pose; `gripper_command` |
| **Effort** | `effort_*` | actuation energy expended | motor current, joint torque, commanded torque, estimated contact force |
| **Feedback** | `feedback_*` | measured physical outcome | encoder position and velocity, TCP pose, tool accelerometer |
| **Context** | `ctx_*` | environment and static state | payload mass, task phase, fault labels, safety/robot mode, speed scaling, I/O bits |

This separation is the point of the corpus. Existing industrial datasets log sensor
*outcomes* without distinguishing what the controller asked for from what the machine did,
which makes cross-machine dynamics learning hard to set up at all.

## Composition (the corpus as reported in the paper)

| Source | Machine | Tasks | Faults | Episodes | Timesteps |
|---|---|---|---|---|---|
| Lab (real) | UR3 | P&P, Screw, Peg | yes | 8,863 | 11M |
| Lab (real) | KUKA KR10 | P&P | yes | 1,799 | 4M |
| Open (real) | voraus-AD (Yu-Cobot) | P&P | yes | 2,122 | 16M |
| Open (real) | AURSAD (UR3e) | Screw | yes | 2,045 | 3M |
| Open (real) | UMich CNC | Machining | yes | 18 | 18K |
| Synthetic | Isaac Sim — UR3, UR5, UR10, UR30, KR10 | P&P | yes | 41,744 | 79M |
| **Total** | | | | **56,591** | **113M** |

1,553 real counterfactual episodes accompany the faulty runs. Of the 10,662 lab episodes,
~28% are healthy and ~72% contain an injected fault.

### Synthetic track — read this before comparing against the paper

**This snapshot carries the earlier Isaac Sim 4.5.0 synthetic campaign**
(`simulations_baseline_*` and `simulations_counterfactual_*`), not the five-arm
41,744-episode campaign the paper reports. That campaign is Isaac Sim 5.1 / Isaac Lab
2.3 across UR3, UR5, UR10, UR30 and KR10, and is being uploaded during the review
period. Every real subset here is final and matches the paper.

The 5.1 campaign runs GPU-batched PhysX at 500 Hz, with control and logging at 125 Hz
for the UR arms and 83.3 Hz for the KR10 (the KRC interpolation cycle), no resampling.
Six fault classes: payload addition, motor miscommutation, gripper activation failure,
gripper release, collision, path obstacle.

Every episode passes a two-stage validity gate — structural (plausible length, complete
phases, part placed, injected event present, non-zero contact force) and trace (every
fault must leave a measurable physical effect). **41,744 of 50,000 generated episodes
pass.** This guarantees that a fault label corresponds to a real physical effect, but it
also biases the synthetic faults toward more detectable instances, and acceptance rates
differ by fault class. UR3 and KR10 motion profiles are fitted to real recordings; UR5,
UR10 and UR30 have no real counterpart and are scaled from the UR3 fit.

## Known limitations

Stated here because they determine what the corpus can be used for:

- **UR5, UR10 and UR30 exist only in simulation.** Any multi-arm result on them is sim-to-sim.
- **The validity gate favours detectable faults**, so synthetic fault difficulty is not representative.
- **Recording context partly predicts fault labels** on both real robots (program, day, speed override), so real fault evaluation should stay within a single recording session.
- **Simulated motor current is not calibrated** to the real robots out of the box. Task and cycle structure transfer from simulation; fault signatures largely do not.
- The KUKA KR10 (KSS 8.3) does not expose joint velocities, commanded TCP pose, or TCP force/torque over RSI; those channels are null.

## Licensing

Licensing is per subset:

| Subset | License |
|---|---|
| Laboratory (`factorywave_*`) and synthetic (`simulations_*`) | MIT |
| `voraus_*` (voraus-AD) | CC BY-NC-SA 4.0: **NonCommercial and ShareAlike**; works derived from it must keep this license |
| `aursad_*` (AURSAD) | CC BY 4.0 |
| `cnc_*` (UMich CNC) | follows its source terms |

Cite the original sources when using the adapted subsets.

## Code

Adapters, simulation configs and the transfer study: https://github.com/Forgis-Labs/FactoryNet

## Citation

```bibtex
@article{petersen2026factorynet,
  title   = {FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models},
  author  = {Petersen, Jonas and Jawahar, Pratik and Maggioni, Riccardo and Huou, Alexandre and Othman, Karim and Mazzoleni, Camilla and Cozzi, Matteo and Gajewski, Marek and Paloschi, Davide and Lombardi, Gian-Alessandro and Ignuta-Ciuncanu, Matei C. and Porath, Lior and Martelli, Federico and Petersen, Philipp Christian},
  journal = {arXiv preprint arXiv:2605.09081},
  year    = {2026}
}
```
