"""Local: lap-by-lap Strategy Predictor output for one race, with what really happened."""
import json
import os
import sys

import fastf1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.lib.strategy as st  # noqa: E402
import backtest_strategy as bt  # noqa: E402

year, rnd = int(sys.argv[1]), int(sys.argv[2])
fastf1.Cache.enable_cache(".fastf1-cache")
fastf1.set_log_level("ERROR")
name, lap_times, total_laps, status_laps, leader_end, final = bt.load_race(year, rnd)
stints = {c: st.split_stints(e) for c, e in lap_times.items()}

conditions, laps_out = {}, []
for lap in range(3, total_laps):
    cutoff = leader_end.get(lap)
    if cutoff is None:
        continue
    done = {c: sum(1 for e in es if e["end_time_s"] <= cutoff + 1e-6) for c, es in lap_times.items()}
    done = {c: n for c, n in done.items() if n > 0}
    order = sorted(done, key=lambda c: (-done[c], next(e["end_time_s"] for e in lap_times[c] if e["lap"] == done[c])))
    position = {c: i + 1 for i, c in enumerate(order)}
    status = st.status_at_lap(status_laps, lap + 1)
    result = st.predict_next_moves(lap_times, done, total_laps, status_laps,
                                   {k: v for k, v in conditions.items() if k < lap}, status)
    conditions[lap] = result["condition"]["condition"]
    cars = []
    for code, car in result["drivers"].items():
        nxt = next((s for s in stints[code] if s["start_lap"] > car["completed"]), None)
        cars.append({
            "c": code, "p": car["position"], "t": car["compound"], "a": car["tyre_age"],
            "call": car["call"], "n": car["next_tyres"], "w": car["window"],
            "pl": car["places_lost"], "free": car["free_stop"], "why": car["reason"],
            "uc": [car["car_behind"], round(car["gap_behind_s"], 1)] if car["undercut_threat"] else None,
            "actual": [nxt["start_lap"] - 1, nxt["tyre"]] if nxt else None,
        })
    cars.sort(key=lambda r: r["p"])
    laps_out.append({
        "lap": lap, "status": status, "loss": round(result["pit_loss_s"], 1),
        "lossNow": round(result["pit_loss_now_s"], 1), "cond": conditions[lap],
        "life": {str(k): round(v) for k, v in result["tyre_life"].items()}, "cars": cars,
    })

# Score this race: one sample per car per lap
calls = [(r["call"], r["actual"] is not None) for lap in laps_out for r in lap["cars"]]
stop_ok = sum((call != "likely_stay_out") == stopped for call, stopped in calls) / len(calls)
tyre = [(r["n"][0] == r["actual"][1]) for lap in laps_out for r in lap["cars"]
        if r["actual"] and r["n"] and r["actual"][0] - lap["lap"] == 3]
print(json.dumps({
    "race": f"{year} {name} (Madrid)", "total": total_laps, "laps": laps_out,
    "score": {"stop": round(stop_ok, 3), "samples": len(calls),
              "tyre": round(sum(tyre) / len(tyre), 3) if tyre else None, "tyreStops": len(tyre)},
    "stints": {c: [[s["tyre"], s["start_lap"], s["end_lap"]] for s in ss] for c, ss in stints.items()},
}, separators=(",", ":")))
