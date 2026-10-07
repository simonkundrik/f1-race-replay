"""
Local test: can the model call a car's next tyre and pit lap a few laps ahead?

For every real pit stop, the model is run on what was known LOOKAHEAD laps
before it and its recommended next compound and stop window are compared
with what the team did, alongside naive guesses.

    python scripts/next_tyre_test.py 2024 2026
"""
import collections
import datetime
import os
import sys

import fastf1
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.lib.strategy as st  # noqa: E402
import backtest_strategy as bt  # noqa: E402

LOOKAHEAD = 3
NAMES = {0: "S", 1: "M", 2: "H", 3: "I", 4: "W"}


LIFE_PRIOR = {0: 0.30, 1: 0.45, 2: 0.65}  # share of race distance a set is assumed to last


def tyre_life(seen, total_laps):
    """Laps each dry compound can do: longest stint seen so far, or a default."""
    life = {c: LIFE_PRIOR[c] * total_laps for c in st.DRY_COMPOUNDS}
    for entries in seen.values():
        for stint in st.split_stints(entries):
            if stint["tyre"] in life:
                age = stint["first_age"] + stint["end_lap"] - stint["start_lap"]
                life[stint["tyre"]] = max(life[stint["tyre"]], age)
    return life


def allowed(compound, current, used):
    """A final stint must leave the car having used two dry compounds."""
    dry_used = {c for c in used if c in st.DRY_COMPOUNDS}
    wet_used = any(c not in st.DRY_COMPOUNDS for c in used)
    return wet_used or len(dry_used) >= 2 or compound != current


def pick_softest_that_lasts(need, current, used, life):
    for compound in st.DRY_COMPOUNDS:  # soft first
        if life[compound] >= need and allowed(compound, current, used):
            return compound
    # Nothing reaches the end: another stop will follow, so cover half
    for compound in st.DRY_COMPOUNDS:
        if life[compound] >= need / 2:
            return compound
    return st.HARD


def pick_popular_that_lasts(need, current, used, life, counts):
    ranked = [c for c, _ in counts.most_common() if c in st.DRY_COMPOUNDS]
    ranked += [c for c in (st.HARD, st.MEDIUM, st.SOFT) if c not in ranked]
    for compound in ranked:
        if life[compound] >= need and allowed(compound, current, used):
            return compound
    for compound in ranked:
        if allowed(compound, current, used):
            return compound
    return ranked[0]


def stop_worth_it(model, current, age, remaining, new, pit_loss):
    """Seconds gained over the rest of the race by fitting ``new`` now, minus the pit loss."""
    laps = np.arange(1, remaining + 1)
    stay = model.offset[current] + model.deg[current] * (age + laps)
    fresh = model.offset[new] + model.deg[new] * laps
    return float((stay - fresh).sum() - pit_loss)


def test_race(year, rnd):
    name, lap_times, total_laps, status_laps, leader_end, final = bt.load_race(year, rnd)
    stints = {code: st.split_stints(entries) for code, entries in lap_times.items()}
    wet_race = any(s["tyre"] not in st.DRY_COMPOUNDS for ss in stints.values() for s in ss)
    all_stops = sorted(
        (ss[i]["end_lap"], code, i, ss[i + 1]["tyre"])
        for code, ss in stints.items() for i in range(len(ss) - 1)
    )

    def completed_at(cutoff):
        return {code: n for code, n in (
            (code, sum(1 for e in entries if e["end_time_s"] <= cutoff + 1e-6))
            for code, entries in lap_times.items()) if n > 0}

    conditions, fits = {}, {}
    for lap in sorted(leader_end):
        seen = st.visible_laps(lap_times, completed_at(leader_end[lap]))
        conditions[lap] = st.estimate_condition(seen, lap)["condition"]

    def fit_at(lap):
        if lap not in fits:
            completed = completed_at(leader_end[lap])
            seen = st.visible_laps(lap_times, completed)
            past = [p for p in status_laps if p["start_lap"] <= lap + 1]
            conds = {n: c for n, c in conditions.items() if n <= lap}
            for future in range(lap + 1, total_laps + 1):
                conds[future] = conditions[lap]
            fits[lap] = (completed, seen, past, conds,
                         st.fit_pace_model(seen, total_laps, past, conds),
                         st.estimate_pit_loss(seen, past)[0])
        return fits[lap]

    rows = []
    for pit_lap, code, index, actual in all_stops:
        if pit_lap <= 3 or pit_lap >= total_laps - 1 or actual not in st.COMPOUNDS:
            continue
        entry = next((e for e in lap_times[code] if e["lap"] == pit_lap - LOOKAHEAD), None)
        if entry is None:
            continue
        leader_lap = max((lap for lap, end in leader_end.items() if end <= entry["end_time_s"] + 1e-6),
                         default=None)
        if leader_lap is None or leader_lap < 3:
            continue
        completed, seen, past, conds, model, pit_loss = fit_at(leader_lap)
        mine = st.split_stints(seen.get(code, []))
        if not mine or code not in model.base or len(mine) != index + 1:
            continue  # only score the stop that is actually next
        current = mine[-1]
        done = current["end_lap"]
        age = current["first_age"] + (done - current["start_lap"])
        used = tuple(s["tyre"] for s in mine)
        status_now = st.status_at_lap(status_laps, leader_lap + 1)
        plan = st.optimise_remaining(model, code, done, current["tyre"], age, used, pit_loss,
                                     pit_loss * st.pit_loss_factor(status_now), conds)

        earlier = [(lap, new, i) for lap, c, i, new in all_stops
                   if lap <= leader_lap and c != code and leader_end.get(lap, 1e18) <= entry["end_time_s"]]
        counts = collections.Counter(new for _, new, _ in earlier)
        common = counts.most_common(1)[0][0] if counts else st.HARD
        different = [c for c, _ in counts.most_common() if c != current["tyre"]]
        rule_guess = different[0] if different else (st.HARD if current["tyre"] != st.HARD else st.MEDIUM)
        same_number = [lap for lap, _, i in earlier if i == index]

        life = tyre_life(seen, total_laps)
        need = total_laps - done
        dry_now = conds[leader_lap] == st.CONDITION_DRY
        soft_rule = pick_softest_that_lasts(need, current["tyre"], used, life) if dry_now else st.INTERMEDIATE
        popular_rule = pick_popular_that_lasts(need, current["tyre"], used, life, counts) if dry_now else st.INTERMEDIATE
        rows.append({
            "softest_lasts": soft_rule, "popular_lasts": popular_rule,
            "race": name, "wet": wet_race, "code": code, "pit_lap": pit_lap, "actual": actual,
            "neutral": st.status_at_lap(status_laps, pit_lap) is not None,
            "model": plan.stops[0][1] if plan.stops else None,
            "model_lap": plan.stops[0][0] if plan.stops else None,
            "window": plan.window,
            "common": common, "rule": rule_guess, "hard": st.HARD,
            "naive_lap": float(np.median(same_number)) if len(same_number) >= 3 else None,
            "from": current["tyre"],
        })
    # Is another stop worth it? Sampled every 5 laps for every running car.
    worth = []
    for leader_lap in range(5, total_laps - 4, 5):
        if leader_lap not in leader_end:
            continue
        completed, seen, past, conds, model, pit_loss = fit_at(leader_lap)
        if conds[leader_lap] != st.CONDITION_DRY:
            continue
        life = tyre_life(seen, total_laps)
        status_now = st.status_at_lap(status_laps, leader_lap + 1)
        loss_now = pit_loss * st.pit_loss_factor(status_now)
        for code, entries in seen.items():
            mine = st.split_stints(entries)
            if not mine or code not in model.base or mine[-1]["tyre"] not in st.DRY_COMPOUNDS:
                continue
            done = mine[-1]["end_lap"]
            if max(e["lap"] for e in lap_times[code]) < total_laps - 3 or done < leader_lap - 2:
                continue  # retired later, or already out
            current = mine[-1]
            age = current["first_age"] + (done - current["start_lap"])
            used = tuple(s["tyre"] for s in mine)
            remaining = total_laps - done
            rule_met = allowed(current["tyre"], None, used) and len({c for c in used}) >= 2
            reaches_end = life[current["tyre"]] >= age + remaining
            new = pick_softest_that_lasts(remaining, current["tyre"], used, life)
            margin = stop_worth_it(model, current["tyre"], age, remaining, new, loss_now)
            worth.append({
                "race": name, "code": code, "lap": leader_lap, "rule_met": rule_met,
                "reaches_end": reaches_end, "margin_s": margin,
                "life": life[current["tyre"]], "age": age, "remaining": remaining, "neutral": status_now is not None,
                "actual": any(s["end_lap"] >= done for s in stints[code][:-1]),
                "pred": (not rule_met) or (not reaches_end) or margin > 0,
                "pred_no_time": (not rule_met) or (not reaches_end),
            })
    return name, rows, worth


def report_worth(label, worth):
    if not worth:
        return
    n = len(worth)
    print(f"\n{label}: {n} car/lap samples, {np.mean([w['actual'] for w in worth]):.0%} went on to stop again")
    for key, text in (("pred", "rules + tyre life + time gained vs pit loss"),
                      ("pred_no_time", "rules + tyre life only")):
        hit = np.mean([w[key] == w["actual"] for w in worth])
        yes = [w for w in worth if w[key]]
        no = [w for w in worth if not w[key]]
        print(f"  {text:<46} correct {hit:.0%}   when it says stop: right {np.mean([w['actual'] for w in yes]) if yes else float('nan'):.0%} "
              f"({len(yes)})   when it says stay out: right {np.mean([not w['actual'] for w in no]) if no else float('nan'):.0%} ({len(no)})")
    base = max(np.mean([w['actual'] for w in worth]), 1 - np.mean([w['actual'] for w in worth]))
    print(f"  always guessing the more common outcome        correct {base:.0%}")


def report(label, rows):
    if not rows:
        return
    n = len(rows)

    def acc(key):
        return sum(r[key] == r["actual"] for r in rows) / n

    print(f"\n{label}: rule picks  softest that lasts {acc('softest_lasts'):.0%}   "
          f"most popular that lasts {acc('popular_lasts'):.0%}", end="")

    with_plan = [r for r in rows if r["model_lap"] is not None]
    in_window = [r for r in with_plan if r["window"][0] <= r["pit_lap"] <= r["window"][1]]
    widths = [r["window"][1] - r["window"][0] + 1 for r in with_plan]
    both = [r for r in with_plan if r["naive_lap"] is not None]
    print(f"\n{label}: {n} stops")
    print(f"  next tyre correct:  model {acc('model'):.0%}   most common so far {acc('common'):.0%}   "
          f"most common other than current {acc('rule'):.0%}   always hard {acc('hard'):.0%}")
    print(f"  model saw no further stop: {1 - len(with_plan) / n:.0%}")
    if with_plan:
        print(f"  stop inside predicted window: {len(in_window) / len(with_plan):.0%} "
              f"(median window {np.median(widths):.0f} laps)   "
              f"within 3 laps of best lap: {np.mean([abs(r['model_lap'] - r['pit_lap']) <= 3 for r in with_plan]):.0%}   "
              f"within 5: {np.mean([abs(r['model_lap'] - r['pit_lap']) <= 5 for r in with_plan]):.0%}")
    if both:
        print(f"  median lap error ({len(both)} stops with a naive guess): model "
              f"{np.median([abs(r['model_lap'] - r['pit_lap']) for r in both]):.1f}   "
              f"median lap of earlier stops {np.median([abs(r['naive_lap'] - r['pit_lap']) for r in both]):.1f}")
    confusion = collections.Counter((NAMES[r["actual"]], NAMES.get(r["model"], "-")) for r in rows)
    print("  actual->model: " + "  ".join(f"{a}->{m} {c}" for (a, m), c in sorted(confusion.items())))


def main():
    fastf1.Cache.enable_cache(".fastf1-cache")
    fastf1.set_log_level("ERROR")
    today = pd.Timestamp(datetime.date.today())
    rows = []
    all_worth = []
    for year in (int(y) for y in (sys.argv[1:] or ["2024", "2026"])):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for rnd in [int(r["RoundNumber"]) for _, r in schedule.iterrows() if r["EventDate"] < today]:
            try:
                name, found, worth = test_race(year, rnd)
                for w in worth:
                    w["year"] = year
                all_worth.extend(worth)
            except Exception as exc:
                print(f"{year} R{rnd}: failed ({type(exc).__name__}: {exc})", flush=True)
                continue
            for r in found:
                r["year"] = year
            rows.extend(found)
            ok = sum(r["model"] == r["actual"] for r in found)
            print(f"{year} R{rnd:<2} {name:<28} stops {len(found):<3} model tyre correct {ok}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "output", "next_tyre_test.csv"), index=False)
    dry = [r for r in rows if not r["wet"]]
    report("ALL RACES", rows)
    report("DRY RACES", dry)
    report("DRY, green-flag stops", [r for r in dry if not r["neutral"]])
    report("DRY, stops under safety car / VSC", [r for r in dry if r["neutral"]])
    report("WET RACES", [r for r in rows if r["wet"]])
    for year in sorted({r["year"] for r in rows}):
        report(f"DRY {year}", [r for r in dry if r["year"] == year])
    print("\n==== IS ANOTHER STOP WORTH IT (dry running only) ====")
    report_worth("ALL", all_worth)
    report_worth("Rule already met (stop is optional)", [w for w in all_worth if w["rule_met"]])
    report_worth("Rule met and tyre should reach the end", [w for w in all_worth if w["rule_met"] and w["reaches_end"]])
    report_worth("Under safety car / VSC", [w for w in all_worth if w["neutral"]])
    for year in sorted({w["year"] for w in all_worth}):
        report_worth(f"{year}", [w for w in all_worth if w["year"] == year])
    pd.DataFrame(all_worth).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "output", "stop_worth_test.csv"), index=False)


if __name__ == "__main__":
    main()
