"""
Local backtest for src/lib/strategy.py (not part of the app).

Replays past races lap by lap from FastF1 timing data and checks whether
the projected finishing order beats the current running order as a
predictor of the result, how well lap times are predicted, and how stable
the regret bands are.

    python scripts/backtest_strategy.py 2024:1 2024:2 ...
"""
import os
import sys

import fastf1
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.strategy import (  # noqa: E402
    COMPOUNDS, analyse_race, estimate_condition, split_stints, visible_laps,
)
from src.lib.tyres import get_tyre_compound_int  # noqa: E402

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
WET_RACES = ("2024:9", "2024:12")
DEFAULT_RACES = ["2024:1", "2024:2", "2024:4", "2024:5", "2024:6", "2024:10",
                 "2024:11", "2024:13", "2024:15", "2024:16", "2024:9", "2024:12"]


def load_race(year, rnd):
    session = fastf1.get_session(year, rnd, "R")
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps
    lap_times = {}
    for _, row in laps.iterrows():
        if pd.isna(row["LapNumber"]) or pd.isna(row["Time"]):
            continue
        compound = row["Compound"]
        lap_times.setdefault(row["Driver"], []).append({
            "lap": int(row["LapNumber"]),
            "time_s": row["LapTime"].total_seconds() if pd.notna(row["LapTime"]) else -1.0,
            "end_time_s": row["Time"].total_seconds(),
            "tyre": get_tyre_compound_int(compound) if pd.notna(compound) else -1,
            "tyre_life": int(row["TyreLife"]) if pd.notna(row["TyreLife"]) else 0,
            "is_pit_entry": pd.notna(row["PitInTime"]),
            "is_pit_affected": pd.notna(row["PitInTime"]),
            "is_out_lap": pd.notna(row["PitOutTime"]),
            "is_accurate": bool(row["IsAccurate"]),
        })
    # Interval to the car that crossed the line just before, on the same lap
    by_lap = {}
    for code, entries in lap_times.items():
        for entry in entries:
            by_lap.setdefault(entry["lap"], []).append(entry)
    for entries in by_lap.values():
        entries.sort(key=lambda e: e["end_time_s"])
        for ahead, entry in zip(entries, entries[1:]):
            entry["official_interval_to_ahead_s"] = entry["end_time_s"] - ahead["end_time_s"]

    total_laps = int(laps["LapNumber"].max())
    leader_end = {}
    for lap, entries in by_lap.items():
        leader_end[lap] = entries[0]["end_time_s"]

    def lap_at(t):
        return 1 + sum(1 for end in leader_end.values() if end <= t)

    status_laps = []
    status = session.track_status
    times = [t.total_seconds() for t in status["Time"]]
    for i, code in enumerate(status["Status"].astype(str)):
        if code not in ("4", "5", "6", "7"):
            continue
        end = times[i + 1] if i + 1 < len(times) else times[i]
        status_laps.append({"status": code, "start_lap": min(lap_at(times[i]), total_laps),
                            "end_lap": min(lap_at(end), total_laps)})

    results = session.results
    final = {row["Abbreviation"]: float(row["Position"]) for _, row in results.iterrows()
             if pd.notna(row["Position"])}
    return session.event["EventName"], lap_times, total_laps, status_laps, leader_end, final


def rank_error(order, final_rank):
    return float(np.mean([abs(i + 1 - final_rank[c]) for i, c in enumerate(order)]))


def backtest(year, rnd):
    name, lap_times, total_laps, status_laps, leader_end, final = load_race(year, rnd)
    finishers = [c for c, entries in lap_times.items()
                 if max(e["lap"] for e in entries) >= total_laps - 3 and c in final]
    final_order = sorted(finishers, key=lambda c: final[c])
    final_rank = {c: i + 1 for i, c in enumerate(final_order)}

    conditions = {}
    rows, pace_errors = [], []
    weight_rows = {}
    last_result = None
    for lap in range(1, total_laps - 2):
        cutoff = leader_end.get(lap)
        if cutoff is None:
            continue
        completed = {}
        for code, entries in lap_times.items():
            done = sum(1 for e in entries if e["end_time_s"] <= cutoff + 1e-6)
            retired = max(e["lap"] for e in entries) < total_laps - 3 and \
                done >= max(e["lap"] for e in entries)
            if done > 0 and not retired:
                completed[code] = done
        seen = visible_laps(lap_times, completed)
        conditions[lap] = estimate_condition(seen, lap)["condition"]
        if lap < 5 or lap % 5:
            continue

        current = sorted(
            completed,
            key=lambda c: (-completed[c],
                           next(e["end_time_s"] for e in lap_times[c] if e["lap"] == completed[c])),
        )
        positions = {c: i + 1 for i, c in enumerate(current)}
        status_now = next((p["status"] for p in status_laps
                           if p["start_lap"] <= lap + 1 <= p["end_lap"]), None)
        result = analyse_race(lap_times, completed, total_laps, status_laps,
                              dict(conditions), positions=positions, current_status=status_now)
        last_result = (lap, completed, positions, dict(conditions), result)

        projected = [r["code"] for r in result["standings"] if r["code"] in final_rank]
        naive = [c for c in current if c in final_rank and c in projected]
        projected = [c for c in projected if c in naive]
        if len(naive) < 5:
            continue
        sub_rank = {c: i + 1 for i, c in enumerate(sorted(naive, key=lambda c: final_rank[c]))}
        rows.append((lap, rank_error(naive, sub_rank), rank_error(projected, sub_rank)))
        for w in WEIGHTS:
            d = result["drivers"]
            alt = sorted(naive, key=lambda c: d[c]["elapsed_s"] + d[c]["plan"].remaining_s
                         - (1 - w) * d[c]["pace_vs_field_s"] * (total_laps - d[c]["completed"]))
            weight_rows.setdefault(w, []).append(rank_error(alt, sub_rank))

        # Predict the next five laps of the current stint
        model = result["model"]
        for code, info in result["drivers"].items():
            for ahead in range(1, 6):
                target = next((e for e in lap_times[code] if e["lap"] == info["completed"] + ahead), None)
                if target is None or target["time_s"] <= 0 or target["is_pit_entry"] \
                        or target["is_out_lap"] or not target["is_accurate"] \
                        or target["tyre"] != info["compound"] or info["clean_laps"] < 5:
                    break
                if any(p["start_lap"] <= target["lap"] <= p["end_lap"] + 1 for p in status_laps):
                    break
                predicted = model.lap_time(code, info["compound"], info["tyre_age"] + ahead,
                                           target["lap"], conditions.get(lap, "DRY"))
                pace_errors.append(target["time_s"] - predicted)

    # Band stability when degradation is 30% lower or higher
    stable = None
    if last_result:
        lap, completed, positions, conds, result = last_result
        bands = {c: d["band"] for c, d in result["drivers"].items()}
        agree, count = 0, 0
        for factor in (0.7, 1.3):
            other = analyse_race(lap_times, completed, total_laps, status_laps, conds,
                                 positions=positions, deg_factor=factor)
            for code, band in bands.items():
                if code in other["drivers"]:
                    count += 1
                    agree += other["drivers"][code]["band"] == band
        stable = agree / count if count else None

    rows = np.array(rows)
    errors = np.array(pace_errors)
    summary = {
        "race": f"{year} {name}", "laps": total_laps,
        "naive": rows[:, 1].mean(), "projected": rows[:, 2].mean(),
        "better": int((rows[:, 2] < rows[:, 1] - 1e-9).sum()),
        "worse": int((rows[:, 2] > rows[:, 1] + 1e-9).sum()),
        "checkpoints": len(rows),
        "pace_mae": float(np.abs(errors).mean()) if len(errors) else float("nan"),
        "pace_bias": float(errors.mean()) if len(errors) else float("nan"),
        "stable": stable,
        "weights": {w: float(np.mean(v)) for w, v in weight_rows.items()},
        "rows": rows,
        "last": last_result,
    }
    return summary


def main():
    fastf1.Cache.enable_cache(os.environ.get("FASTF1_CACHE", ".fastf1-cache"))
    fastf1.set_log_level("ERROR")
    races = sys.argv[1:] or DEFAULT_RACES
    summaries = []
    for spec in races:
        year, rnd = (int(x) for x in spec.split(":"))
        try:
            s = backtest(year, rnd)
        except Exception as exc:  # keep going; one bad race should not end the run
            print(f"{spec}: failed ({type(exc).__name__}: {exc})", flush=True)
            continue
        s["wet"] = spec in WET_RACES
        summaries.append(s)
        print(f"{s['race']:<32} naive {s['naive']:.2f}  projected {s['projected']:.2f}  "
              f"better/worse {s['better']}/{s['worse']} of {s['checkpoints']}  "
              f"pace MAE {s['pace_mae']:.2f} bias {s['pace_bias']:+.2f}  "
              f"bands stable {s['stable']:.0%}", flush=True)
        lap, _, _, _, result = s["last"]
        model = result["model"]
        print(f"   lap {lap}: pit loss {result['pit_loss_s']:.1f}s ({result['pit_stops_seen']} stops)  "
              f"deg S/M/H {model.deg[0]:.3f}/{model.deg[1]:.3f}/{model.deg[2]:.3f}  "
              f"offset M/H {model.offset[1]:+.2f}/{model.offset[2]:+.2f}  evo {model.evolution:+.3f}  sigma {model.sigma:.2f}",
              flush=True)
        by_phase = {}
        for lap_no, naive, projected in s["rows"]:
            phase = min(3, int(4 * lap_no / s["laps"]))
            by_phase.setdefault(phase, []).append((naive, projected))
        print("   by quarter: " + "  ".join(
            f"Q{p + 1} {np.mean([n for n, _ in v]):.2f}->{np.mean([q for _, q in v]):.2f}"
            for p, v in sorted(by_phase.items())), flush=True)
    for label, group in (("DRY", [s for s in summaries if not s["wet"]]),
                         ("WET", [s for s in summaries if s["wet"]])):
        if not group:
            continue
        print(f"\n{label} ({len(group)} races): naive {np.mean([s['naive'] for s in group]):.3f}  "
              f"pace MAE {np.nanmean([s['pace_mae'] for s in group]):.2f}  projected by pace weight: "
              + "  ".join(f"{w}: {np.mean([s['weights'][w] for s in group]):.3f}" for w in WEIGHTS))
        print("   per race at each weight: " + " | ".join(
            s["race"][5:9] + " " + "/".join(f"{s['weights'][w]:.2f}" for w in WEIGHTS) for s in group))
    if summaries:
        print(f"\nMEAN position error: naive {np.mean([s['naive'] for s in summaries]):.3f}  "
              f"projected {np.mean([s['projected'] for s in summaries]):.3f}  "
              f"pace MAE {np.nanmean([s['pace_mae'] for s in summaries]):.2f}")


if __name__ == "__main__":
    main()
