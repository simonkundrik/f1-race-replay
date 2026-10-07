"""
Local study: tyre degradation per driver and team across one season.

For every completed race, each dry stint's lap times are corrected for fuel
and track evolution and a slope against tyre age is fitted. Slopes are then
compared with the field's median for the same race and compound, so that
tracks and compound allocations cancel out, and averaged over the season.

    python scripts/season_deg.py 2026
"""
import csv
import datetime
import os
import sys

import fastf1
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.lib.strategy as st  # noqa: E402
import backtest_strategy as bt  # noqa: E402

COMPOUND_NAMES = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
MIN_LAPS = 6


def race_stints(year, rnd):
    name, lap_times, total_laps, status_laps, leader_end, final = bt.load_race(year, rnd)
    session = fastf1.get_session(year, rnd, "R")
    session.load(telemetry=False, weather=False, messages=False)
    teams = {r["Abbreviation"]: r["TeamName"] for _, r in session.results.iterrows()}

    conditions = {}
    for lap in range(1, total_laps + 1):
        seen = {c: [e for e in entries if e["lap"] <= lap] for c, entries in lap_times.items()}
        conditions[lap] = st.estimate_condition(seen, lap)["condition"]
    model = st.fit_pace_model(lap_times, total_laps, status_laps, conditions)
    neutral = st.neutralised_laps(status_laps)

    rows = []
    for code, entries in lap_times.items():
        for stint in st.split_stints(entries):
            if stint["tyre"] not in st.DRY_COMPOUNDS:
                continue
            laps = []
            for entry in stint["laps"]:
                lap = int(entry["lap"])
                if not st._is_clean(entry, neutral) or conditions.get(lap) != st.CONDITION_DRY:
                    continue
                age = stint["first_age"] + (lap - stint["start_lap"])
                corrected = (entry["time_s"] - model.fuel_per_lap * (total_laps - lap)
                             - model.evolution * lap)
                interval = entry.get("official_interval_to_ahead_s")
                laps.append((age, corrected, interval is not None and interval < st.TRAFFIC_INTERVAL_S))
            clear = [r for r in laps if not r[2]]
            if len(clear) >= MIN_LAPS:
                laps = clear
            if len(laps) < MIN_LAPS:
                continue
            arr = np.array([(a, t) for a, t, _ in laps])
            arr = arr[arr[:, 1] <= np.median(arr[:, 1]) + 2.0]
            if len(arr) < MIN_LAPS or np.ptp(arr[:, 0]) < 4:
                continue
            slope = float(stats.theilslopes(arr[:, 1], arr[:, 0])[0])
            rows.append({
                "round": rnd, "race": name, "driver": code, "team": teams.get(code, "?"),
                "compound": COMPOUND_NAMES[stint["tyre"]], "laps": len(arr),
                "first_age": int(arr[:, 0].min()), "last_age": int(arr[:, 0].max()),
                "deg_s_per_lap": slope,
            })
    by_compound = {}
    for row in rows:
        by_compound.setdefault(row["compound"], []).append(row["deg_s_per_lap"])
    for row in rows:
        field = by_compound[row["compound"]]
        row["field_deg"] = float(np.median(field)) if len(field) >= 4 else None
        row["delta"] = row["deg_s_per_lap"] - row["field_deg"] if row["field_deg"] is not None else None
    return name, rows, model


def summarise(rows, key):
    groups = {}
    for row in rows:
        if row["delta"] is not None:
            groups.setdefault(row[key], []).append(row)
    out = []
    for name, items in groups.items():
        deltas = np.array([r["delta"] for r in items])
        weights = np.array([r["laps"] for r in items], dtype=float)
        mean = float(np.average(deltas, weights=weights))
        se = float(np.std(deltas) / np.sqrt(len(deltas))) if len(deltas) > 1 else float("nan")
        out.append((name, mean, se, len(items), len({r["round"] for r in items})))
    return sorted(out, key=lambda r: r[1])


def split_half(rows, key):
    """Correlation between the averages from odd and even rounds."""
    halves = ({}, {})
    for row in rows:
        if row["delta"] is not None:
            halves[row["round"] % 2].setdefault(row[key], []).append(row["delta"])
    common = [k for k in halves[0] if k in halves[1] and len(halves[0][k]) >= 3 and len(halves[1][k]) >= 3]
    if len(common) < 5:
        return None, len(common)
    a = [np.mean(halves[0][k]) for k in common]
    b = [np.mean(halves[1][k]) for k in common]
    return float(np.corrcoef(a, b)[0, 1]), len(common)


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.date.today().year
    fastf1.Cache.enable_cache(".fastf1-cache")
    fastf1.set_log_level("ERROR")
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    today = pd.Timestamp(datetime.date.today())
    rounds = [int(r["RoundNumber"]) for _, r in schedule.iterrows() if r["EventDate"] < today]
    print(f"{year}: {len(rounds)} completed rounds", flush=True)

    all_rows = []
    for rnd in rounds:
        try:
            name, rows, model = race_stints(year, rnd)
        except Exception as exc:
            print(f"round {rnd}: failed ({type(exc).__name__}: {exc})", flush=True)
            continue
        all_rows.extend(rows)
        meds = {c: np.median([r["deg_s_per_lap"] for r in rows if r["compound"] == c])
                for c in COMPOUND_NAMES.values() if any(r["compound"] == c for r in rows)}
        print(f"R{rnd:<2} {name:<28} stints {len(rows):<3} evo {model.evolution:+.3f}  field deg "
              + "  ".join(f"{c[0]} {v:.3f}" for c, v in meds.items()), flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"deg_stints_{year}.csv")
    if all_rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\n{len(all_rows)} stints written to {path}")

    for key in ("team", "driver"):
        print(f"\nDegradation vs field by {key} (s/lap, negative = kinder on tyres):")
        for name, mean, se, n, races in summarise(all_rows, key):
            print(f"  {name:<22} {mean:+.4f}  +/-{se:.4f}  stints {n:<3} races {races}")
        r, n = split_half(all_rows, key)
        print(f"  odd/even round consistency: r = {r if r is None else round(r, 2)} over {n} {key}s")


if __name__ == "__main__":
    main()
