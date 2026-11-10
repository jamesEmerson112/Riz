"""Level 1: plain code turns number streams into one-sentence findings (no LLM).

Every finding has one shape: {kind, severity, summary, evidence, hypotheses, next_steps}. Rows carry
a timestamp `ts`; distance rows are {ts, mm, us, st} (plus `pred_mm`, what the arm's pose predicts).

- `check_ultrasonic_frozen` needs no arm: a real sensor always has a noise floor (an HC-SR04's echo
  time jitters by about 1 us), so zero variation over a few seconds means stale data.
- `check_distance_residual` uses the arm: it compares what the sensor reads with what forward
  kinematics says it should read (rows with a `pred_mm` field), and names the pattern.
"""
import statistics


def _f(kind, severity, summary, evidence=None, hypotheses=None, next_steps=None):
    return {"kind": kind, "severity": severity, "summary": summary, "evidence": evidence or {},
            "hypotheses": hypotheses or [], "next_steps": next_steps or []}


def _std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _window(recs, now, seconds):
    return [r for r in recs if r["ts"] >= now - seconds]


def check_ultrasonic_frozen(s, *, seconds=4.0, min_rows=30):
    """Sensor-only rule. `s` is a snapshot: {"now": ts, "D": [rows]}."""
    ok = [d for d in _window(s["D"], s["now"], seconds) if d.get("st") == "ok"]
    if len(ok) < min_rows:
        return []
    us = [d["us"] for d in ok]
    if _std(us) == 0:
        return [_f("us_frozen", "critical",
                   "Ultrasonic output is frozen: %d readings, echo time identical at %d us (zero noise)" % (len(ok), us[0]),
                   {"n": len(ok), "us": us[0], "mm": ok[0]["mm"], "std_us": 0.0},
                   ["the sensor driver is re-sending a cached distance (its read loop stalled)",
                    "a jaw or held object sits in the beam at a fixed distance"],
                   ["restart the sensor (arm_restart_sensor)", "re-read at another gripper position"])]
    return []


def check_distance_residual(s, *, seconds=2.0, min_rows=12, moved_mm=10.0):
    """Arm-aware rule. D rows here carry `pred_mm`: the distance the arm's pose predicts.
    Suspended while the gripper holds something (`s["holding"]`): a held block sits in the beam."""
    if s.get("holding"):
        return []
    rows = [d for d in _window(s["D"], s["now"], seconds) if d.get("st") == "ok" and d.get("pred_mm") is not None]
    if len(rows) < min_rows:
        return []
    # Stuck: the newest readings all have the same echo time while the arm's pose says the
    # distance changed. Look at the trailing run only, so rows from before a freeze don't hide it.
    tail = [rows[-1]]
    for d in reversed(rows[:-1]):
        if d["us"] != rows[-1]["us"]:
            break
        tail.append(d)
    tail_moved = max(d["pred_mm"] for d in tail) - min(d["pred_mm"] for d in tail)
    if len(tail) >= min_rows and tail_moved >= moved_mm:
        return [_f("us_stuck", "critical",
                   "Distance stuck at %d mm while the arm moved %d mm" % (tail[0]["mm"], round(tail_moved)),
                   {"n": len(tail), "stuck_mm": tail[0]["mm"], "stuck_us": tail[0]["us"], "arm_moved_mm": round(tail_moved),
                    "pred_mm": [round(tail[-1]["pred_mm"]), round(tail[0]["pred_mm"])]},
                   ["the sensor's reading froze (stale data)", "a jaw or held object is in the beam"],
                   ["restart the sensor (arm_restart_sensor)", "re-read at another gripper position"])]
    if len(tail) >= 5:
        return []      # the reading has stopped changing: let the stuck test above decide
    meas = [float(d["mm"]) for d in rows]
    pred = [float(d["pred_mm"]) for d in rows]
    moved = max(pred) - min(pred)
    if moved < moved_mm:
        # Holding still: a steady gap to the kinematics still shows a sensor that reads wrong.
        # At one height an offset and a scale error look the same, so name the gap and its ratio.
        res = [m - p for m, p in zip(meas, pred)]
        gap, spread = statistics.median(res), _std(res)
        ratio = statistics.median(m / p for m, p in zip(meas, pred) if p > 0)
        if spread < 3 and abs(gap) > 8:
            return [_f("us_offset", "warn",
                       "Reads %d mm %s against the arm's kinematics (%.2fx)" % (abs(round(gap)), "long" if gap > 0 else "short", ratio),
                       {"n": len(rows), "gap_mm": round(gap, 1), "ratio": round(ratio, 3), "spread_mm": round(spread, 2)},
                       ["a calibration offset or a units/clock bug in the sensor firmware", "the sensor mount moved"],
                       ["move to two more heights: a constant gap means offset, a constant ratio means scale",
                        "check the sensor's firmware version and clock"])]
        return []
    # Fit measured = a * predicted + b, but only over enough travel to tell a scale from an offset
    # (over ~15 mm a slope of 0.92 with an 8 mm intercept fits the same points as slope 1 and no
    # gap), and only when the reading actually follows the pose (correlation >= 0.95): a flat or
    # erratic reading is not a calibration or scale error.
    if moved < 2 * moved_mm:
        return []
    mp, mm_ = statistics.fmean(pred), statistics.fmean(meas)
    sp, sm = _std(pred), _std(meas)
    if sm < 1.0 or sp == 0:
        return []
    r = sum((p - mp) * (m - mm_) for p, m in zip(pred, meas)) / (len(rows) * sp * sm)
    if r < 0.95:
        return []
    a = r * sm / sp
    gap = statistics.median(m - p for m, p in zip(meas, pred))     # the offset, measured directly
    ratio = statistics.median(m / p for m, p in zip(meas, pred) if p > 0)
    ev = {"n": len(rows), "slope": round(a, 3), "ratio": round(ratio, 3), "gap_mm": round(gap, 1), "r": round(r, 3),
          "arm_moved_mm": round(moved)}
    # A scale error changes every reading by the same factor, so the readings' own ratio to the
    # prediction must be off too: a steep fitted slope alone comes from a fast move.
    if abs(a - 1) > 0.2 and abs(ratio - 1) > 0.1:
        return [_f("us_scale", "critical", "Reads %.2fx the kinematics distance" % a, ev,
                   ["a clock or units bug in the sensor firmware (for example a half-speed clock halves distances)"],
                   ["check the sensor's firmware version and clock", "roll the sensor firmware back"])]
    if abs(gap) > 8 and abs(a - 1) <= 0.1:
        return [_f("us_offset", "warn", "Reads %d mm %s at every height" % (abs(round(gap)), "long" if gap > 0 else "short"), ev,
                   ["the distance calibration offset is wrong", "the sensor mount moved"],
                   ["correct the offset, then re-sweep 5 heights"])]
    return []


def check_ultrasonic_noise(s, *, seconds=2.0, min_rows=12, max_std_us=4.0):
    """While the arm holds still, a healthy HC-SR04 jitters about 1 us. Much more is abnormal.
    Plain code still filters it; this names it."""
    if s.get("holding"):
        return []
    rows = [d for d in _window(s["D"], s["now"], seconds) if d.get("st") == "ok" and d.get("pred_mm") is not None]
    if len(rows) < min_rows or max(d["pred_mm"] for d in rows) - min(d["pred_mm"] for d in rows) > 2.0:
        return []
    # Robust spread (1.4826 x the median absolute deviation, which equals the std for Gaussian
    # jitter): one reading taken as the beam crosses an edge must not count as jitter.
    us = [d["us"] for d in rows]
    med = statistics.median(us)
    sd = 1.4826 * statistics.median(abs(u - med) for u in us)
    if sd <= max_std_us:
        return []
    return [_f("us_noisy", "warn", "Distance jitters +-%.1f mm at rest (a healthy sensor jitters under 0.2 mm)" % (sd * 343 / 2000),
               {"n": len(rows), "std_us": round(sd, 1), "std_mm": round(sd * 343 / 2000, 2)},
               ["a loose ECHO/TRIG wire or marginal connection", "electrical noise from the servo power", "a soft or angled target"],
               ["take more readings and use the median", "a person reseats the sensor's wires"])]


def check_missing_echo(s, *, seconds=2.0, min_rows=12, frac=0.7):
    """No echo while the arm's pose says a surface is within range: the sensor points somewhere
    the software doesn't think it does (a reversed joint, a bent mount) or it lost its echo."""
    if s.get("holding"):
        return []
    rows = [d for d in _window(s["D"], s["now"], seconds) if d.get("pred_mm") is not None and 30 <= d["pred_mm"] <= 400]
    if len(rows) < min_rows:
        return []
    bad = [d for d in rows if d.get("st") != "ok"]
    if len(bad) / len(rows) < frac:
        return []
    near = statistics.median(d["pred_mm"] for d in rows)
    return [_f("us_missing_echo", "warn",
               "No echo for %d of %d readings where the arm's pose says a surface is %d mm away" % (len(bad), len(rows), round(near)),
               {"n": len(rows), "no_echo": len(bad), "pred_mm": round(near)},
               ["the sensor points somewhere the software doesn't expect (a reversed joint, a bent mount)",
                "the ECHO or TRIG wire is off"],
               ["compare the joint readings with a camera frame", "a person checks the sensor's wires"])]


def check_arrival(result, *, load_nm=None):
    """A move that didn't arrive becomes a sentence, e.g. "Elbow flex stopped 31 deg short of its target"."""
    if result.get("arrived", True) or result.get("halted") or not result.get("worst_joint"):
        return []
    joint, err = result["worst_joint"], float(result.get("max_err_deg", 0))
    load = f" (load {load_nm:.1f} N*m)" if load_nm is not None else ""
    return [_f("joint_stall", "critical",
               "%s stopped %d deg short of its target%s" % (joint.replace("_", " ").capitalize(), round(err), load),
               {"joint": joint, "short_by_deg": round(err, 1), "load_nm": None if load_nm is None else round(load_nm, 2)},
               ["the servo is stuck, unpowered or overloaded", "something blocks the arm", "the joint's calibration is off"],
               ["read the servo Status register (overload, voltage, overheat bits)",
                "compare calibration with the last good file", "park and call a person if it persists"])]


def residual_finding(mm, pred_mm):
    """One reading far from what the pose predicts, seen while stepping down on the target
    (the approach checks this every step), e.g. a bad sensor firmware that halves every distance."""
    ratio = mm / pred_mm if pred_mm else 0.0
    return _f("us_residual", "critical",
              "Distance reads %d mm where the arm's pose says %d mm (%.2fx)" % (mm, round(pred_mm), ratio),
              {"mm": mm, "pred_mm": round(pred_mm, 1), "ratio": round(ratio, 3)},
              ["the sensor's firmware or clock changed (a bad image halves time and distance)",
               "something sits in the beam", "the arm's calibration is off"],
              ["restart the sensor", "re-check at home"])



def run(s):
    findings = (check_ultrasonic_frozen(s) + check_distance_residual(s) + check_ultrasonic_noise(s)
                + check_missing_echo(s))
    order = {"critical": 0, "warn": 1, "info": 2}
    return sorted(findings, key=lambda f: order.get(f["severity"], 3))
