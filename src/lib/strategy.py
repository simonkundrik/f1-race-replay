"""
Tyre strategy estimation from the replay's lap entries.

These functions operate on the lap entry dictionaries broadcast by the
replay server (see ``F1RaceReplayWindow._compute_lap_times``) and only
depend on numpy, so they can run in any insight window or in a test.

The model is a pace-only estimate: every car is treated on its own, so
traffic, track position and tyre set availability are not considered.
Callers must pass only laps that have already been completed (see
``visible_laps``), because the replay broadcasts the whole race up front.
"""

from dataclasses import dataclass, field

import numpy as np

SOFT, MEDIUM, HARD, INTERMEDIATE, WET = 0, 1, 2, 3, 4
COMPOUNDS = (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
DRY_COMPOUNDS = (SOFT, MEDIUM, HARD)

CATEGORY_SLICK = "SLICK"
CATEGORY_INTER = "INTER"
CATEGORY_WET = "WET"

CONDITION_DRY = "DRY"
CONDITION_DAMP = "DAMP"
CONDITION_WET = "WET"

BAND_OPTIMAL = "on_optimum"
BAND_MARGINAL = "marginal"
BAND_COSTLY = "costly"

# Priors mirror src/bayesian_tyre_model.py so both models start from the
# same assumptions: degradation in s/lap, pace offset relative to the soft
# and the fuel effect (0.032 s/kg at 1.6 kg/lap).
DEFAULT_DEG = {SOFT: 0.05, MEDIUM: 0.03, HARD: 0.01, INTERMEDIATE: 0.04, WET: 0.02}
DEFAULT_OFFSET = {SOFT: 0.0, MEDIUM: 0.5, HARD: 1.0, INTERMEDIATE: 0.0, WET: 0.0}
DEFAULT_FUEL_PER_LAP = 0.032 * 1.6

# Seconds lost per lap by running a tyre category in a condition it does not
# suit, relative to the right tyre (StateSpaceConfig.mismatch_penalties).
DEFAULT_CONDITION_DELTA = {
    (CATEGORY_SLICK, CONDITION_DRY): 0.0,
    (CATEGORY_INTER, CONDITION_DRY): 1.5,
    (CATEGORY_WET, CONDITION_DRY): 4.0,
    (CATEGORY_SLICK, CONDITION_DAMP): 2.0,
    (CATEGORY_INTER, CONDITION_DAMP): 0.0,
    (CATEGORY_WET, CONDITION_DAMP): 1.0,
    (CATEGORY_SLICK, CONDITION_WET): 8.0,
    (CATEGORY_INTER, CONDITION_WET): 0.5,
    (CATEGORY_WET, CONDITION_WET): 0.0,
}
# How much slower the whole field laps on a damp / wet track. This shifts
# every car equally, so it never changes a recommendation.
DEFAULT_CONDITION_LEVEL = {CONDITION_DRY: 0.0, CONDITION_DAMP: 8.0, CONDITION_WET: 14.0}

DEFAULT_PIT_LOSS_S = 22.0
SAFETY_CAR_PIT_FACTOR = 0.55
VSC_PIT_FACTOR = 0.7
STATUS_SAFETY_CAR = "4"
STATUS_RED_FLAG = "5"
STATUS_VSC = ("6", "7")

# Laps closer than this to the car ahead are treated as traffic and count
# for less when fitting degradation.
TRAFFIC_INTERVAL_S = 1.0
TRAFFIC_WEIGHT = 0.3

# Assumed change in degradation per degree of track temperature. A single
# race cannot separate this from fuel burn and track evolution, so it is an
# assumption for what-if scenarios rather than a fitted value.
DEFAULT_TEMP_SENSITIVITY = 0.02

# Share of a car's pace advantage over the field that is carried into the
# projected order. Below 1 because a quicker car still has to get past the
# cars ahead, which a pace-only model would hand it for free.
DEFAULT_PACE_WEIGHT = 0.5
LEVEL_TREND_DECAY = 0.7

MIN_STINT_LAPS = 5
MAX_DEG = 0.4
PIT_WINDOW_TOLERANCE_S = 1.0
_BIG = 1e9


def tyre_category(compound):
    if compound == INTERMEDIATE:
        return CATEGORY_INTER
    if compound == WET:
        return CATEGORY_WET
    return CATEGORY_SLICK


def visible_laps(lap_times, completed_laps):
    """
    Drop laps that have not happened yet at the current replay position.

    ``completed_laps`` maps driver code to the number of laps completed.
    Drivers missing from it are dropped entirely.
    """
    result = {}
    for code, entries in (lap_times or {}).items():
        done = completed_laps.get(code)
        if done is None:
            continue
        result[code] = [e for e in entries if 0 < int(e.get("lap", 0)) <= done]
    return result


def neutralised_laps(status_laps):
    """Lap numbers run under safety car, VSC or red flag."""
    laps = set()
    for period in status_laps or []:
        # The lap after a period restarts on cold tyres and bunched up
        laps.update(range(int(period["start_lap"]), int(period["end_lap"]) + 2))
    return laps


def status_at_lap(status_laps, lap):
    """Track status code in force on a lap, or None when green."""
    for period in status_laps or []:
        if int(period["start_lap"]) <= lap <= int(period["end_lap"]):
            return str(period["status"])
    return None


def pit_loss_factor(status):
    """Share of the normal pit loss paid when stopping under this status."""
    if status == STATUS_SAFETY_CAR:
        return SAFETY_CAR_PIT_FACTOR
    if status in STATUS_VSC:
        return VSC_PIT_FACTOR
    return 1.0


def temp_factor(track_temp, reference_temp, offset=0.0, sensitivity=DEFAULT_TEMP_SENSITIVITY):
    """Multiplier on degradation for a track temperature against a reference."""
    if track_temp is None or reference_temp is None:
        delta = offset
    else:
        delta = (track_temp - reference_temp) + offset
    return float(np.clip(1.0 + sensitivity * delta, 0.5, 2.0))


def split_stints(entries):
    """
    Group a driver's lap entries into stints.

    Returns a list of {"tyre", "start_lap", "end_lap", "first_age", "laps"}
    where ``first_age`` is the tyre age on the first lap of the stint, so a
    set that was already used in qualifying starts above 1.
    """
    stints = []
    previous = None
    for entry in sorted(entries, key=lambda e: e.get("lap", 0)):
        tyre = int(entry.get("tyre", -1))
        if tyre not in COMPOUNDS:
            continue
        new_stint = (
            previous is None
            or tyre != stints[-1]["tyre"]
            or previous.get("is_pit_entry")
            or entry.get("is_out_lap")
        )
        if new_stint:
            life = entry.get("tyre_life") or 0
            stints.append({
                "tyre": tyre,
                "start_lap": int(entry["lap"]),
                "end_lap": int(entry["lap"]),
                "first_age": int(life) if life and life > 0 else 1,
                "laps": [],
            })
        stints[-1]["end_lap"] = int(entry["lap"])
        stints[-1]["laps"].append(entry)
        previous = entry
    return stints


def _is_clean(entry, neutral):
    time_s = entry.get("time_s", -1.0)
    return (
        time_s is not None
        and time_s > 0
        and entry.get("lap", 0) > 1
        and entry.get("lap") not in neutral
        and not entry.get("is_pit_entry")
        and not entry.get("is_pit_affected")
        and not entry.get("is_out_lap")
        and entry.get("is_accurate", True)
    )


def _fit_rows(laps_by_driver, total_laps, neutral, condition_by_lap, condition_delta, fuel_per_lap):
    """
    One row per usable lap: (code, compound, lap, age, adjusted time, weight).

    The time has the fuel load and any tyre and condition mismatch removed.
    Laps in traffic or on the wrong tyre for the conditions count for less.
    """
    rows = []
    stint_counts = {c: 0 for c in COMPOUNDS}
    for code, entries in laps_by_driver.items():
        for stint in split_stints(entries):
            usable = 0
            for entry in stint["laps"]:
                if not _is_clean(entry, neutral):
                    continue
                lap = int(entry["lap"])
                condition = condition_by_lap.get(lap, CONDITION_DRY)
                mismatch = condition_delta[(tyre_category(stint["tyre"]), condition)]
                interval = entry.get("official_interval_to_ahead_s")
                weight = 1.0
                if interval is not None and 0 <= interval < TRAFFIC_INTERVAL_S:
                    weight = TRAFFIC_WEIGHT
                if mismatch > 0:
                    weight *= 0.5
                rows.append((
                    code, stint["tyre"], lap,
                    stint["first_age"] + (lap - stint["start_lap"]),
                    entry["time_s"] - fuel_per_lap * (total_laps - lap) - mismatch,
                    weight,
                ))
                usable += 1
            if usable >= MIN_STINT_LAPS:
                stint_counts[stint["tyre"]] += 1
    return rows, stint_counts


@dataclass
class PaceModel:
    """Fitted lap time model; see ``fit_pace_model``."""
    total_laps: int
    base: dict = field(default_factory=dict)
    offset: dict = field(default_factory=lambda: dict(DEFAULT_OFFSET))
    deg: dict = field(default_factory=lambda: dict(DEFAULT_DEG))
    stint_counts: dict = field(default_factory=lambda: {c: 0 for c in COMPOUNDS})
    evolution: float = 0.0
    lap_level: dict = field(default_factory=dict)
    level_trend: tuple = None  # (last wet lap, its level, change per lap)
    deg_se: float = 0.02
    sigma: float = 0.5
    fuel_per_lap: float = DEFAULT_FUEL_PER_LAP
    condition_delta: dict = field(default_factory=lambda: dict(DEFAULT_CONDITION_DELTA))
    condition_level: dict = field(default_factory=lambda: dict(DEFAULT_CONDITION_LEVEL))

    def track_level(self, lap, condition):
        """Field-wide pace shift on a lap: track evolution when dry, wetness otherwise."""
        if condition == CONDITION_DRY:
            return self.evolution * lap
        if lap in self.lap_level:
            return self.lap_level[lap]
        level = self.condition_level[condition]
        if self.level_trend and lap > self.level_trend[0]:
            # Carry a drying or worsening track forward, fading out quickly
            last_lap, last_level, slope = self.level_trend
            steps = lap - last_lap
            carried = slope * LEVEL_TREND_DECAY * (1 - LEVEL_TREND_DECAY ** steps) / (1 - LEVEL_TREND_DECAY)
            level = max(0.0, last_level + carried)
        return level

    def lap_time(self, code, compound, age, lap, condition=CONDITION_DRY, deg_factor=1.0):
        """Predicted lap time in seconds; None for a driver with no pace estimate."""
        base = self.base.get(code)
        if base is None:
            return None
        return (
            base
            + self.offset[compound]
            + self.deg[compound] * deg_factor * age
            + self.fuel_per_lap * (self.total_laps - lap)
            + self.condition_delta[(tyre_category(compound), condition)]
            + self.track_level(lap, condition)
        )

    def lap_costs(self, code, lap, max_age, condition=CONDITION_DRY, deg_factor=1.0):
        """Array [compound, age] of predicted times for one lap."""
        ages = np.arange(max_age, dtype=float)
        costs = np.empty((len(COMPOUNDS), max_age))
        shared = (
            self.base[code]
            + self.fuel_per_lap * (self.total_laps - lap)
            + self.track_level(lap, condition)
        )
        for compound in COMPOUNDS:
            costs[compound] = (
                shared
                + self.offset[compound]
                + self.deg[compound] * deg_factor * ages
                + self.condition_delta[(tyre_category(compound), condition)]
            )
        return costs


def fit_pace_model(laps_by_driver, total_laps, status_laps=None, condition_by_lap=None,
                   condition_delta=None, fuel_per_lap=DEFAULT_FUEL_PER_LAP):
    """
    Fit base pace per driver, compound offsets, degradation and track level.

    One regression over every usable lap of the field:

        time = base[driver] + offset[compound] + deg[compound] * tyre age
               + fuel + track level[lap]

    where the track level is a linear trend (track evolution) on dry laps
    and a free value per lap on damp or wet laps. Offsets and degradation
    are pulled towards the priors, which also settles what the data cannot
    separate (a whole field running the same compounds in the same order).
    Laps under a neutralised track and in or out laps are left out, laps in
    traffic are down-weighted. ``condition_by_lap`` maps lap number to
    DRY / DAMP / WET; missing laps are treated as dry.
    """
    model = PaceModel(total_laps=int(total_laps), fuel_per_lap=fuel_per_lap)
    if condition_delta:
        model.condition_delta.update(condition_delta)
    condition_by_lap = condition_by_lap or {}
    rows, model.stint_counts = _fit_rows(
        laps_by_driver, total_laps, neutralised_laps(status_laps),
        condition_by_lap, model.condition_delta, fuel_per_lap,
    )
    if not rows:
        return model

    codes = sorted({r[0] for r in rows})
    wet_laps = sorted({r[2] for r in rows if condition_by_lap.get(r[2], CONDITION_DRY) != CONDITION_DRY})
    n_codes, n_dry, n_all = len(codes), len(DRY_COMPOUNDS), len(COMPOUNDS)
    col_offset = n_codes
    col_deg = col_offset + n_dry
    col_evolution = col_deg + n_all
    col_level = col_evolution + 1
    n_cols = col_level + len(wet_laps)
    code_index = {code: i for i, code in enumerate(codes)}
    level_index = {lap: col_level + i for i, lap in enumerate(wet_laps)}

    design = np.zeros((len(rows), n_cols))
    target = np.empty(len(rows))
    weights = np.empty(len(rows))
    for i, (code, compound, lap, age, adjusted, weight) in enumerate(rows):
        design[i, code_index[code]] = 1.0
        if compound in DRY_COMPOUNDS:
            design[i, col_offset + compound] = 1.0
        design[i, col_deg + compound] = age
        if lap in level_index:
            design[i, level_index[lap]] = 1.0
        else:
            design[i, col_evolution] = lap
        target[i] = adjusted
        weights[i] = weight

    # Priors as extra rows: (column, value, strength in lap equivalents)
    priors = [(col_offset + c, DEFAULT_OFFSET[c], 20.0) for c in DRY_COMPOUNDS]
    priors += [(col_deg + c, DEFAULT_DEG[c], 1300.0) for c in COMPOUNDS]
    priors.append((col_evolution, 0.0, 2000.0))
    for lap, col in level_index.items():
        priors.append((col, DEFAULT_CONDITION_LEVEL[condition_by_lap[lap]], 0.5))
    prior_design = np.zeros((len(priors), n_cols))
    prior_target = np.empty(len(priors))
    for i, (col, value, strength) in enumerate(priors):
        prior_design[i, col] = np.sqrt(strength)
        prior_target[i] = value * np.sqrt(strength)

    def solve(keep):
        root = np.sqrt(weights[keep])[:, None]
        a = np.vstack([design[keep] * root, prior_design])
        b = np.concatenate([target[keep] * root[:, 0], prior_target])
        return np.linalg.lstsq(a, b, rcond=None)[0]

    keep = np.ones(len(rows), dtype=bool)
    solution = solve(keep)
    residual = target - design @ solution
    mad = np.median(np.abs(residual - np.median(residual)))
    # Refit without laps that are far off: mistakes, yellow flags, damage
    keep = np.abs(residual) <= max(1.0, 3.0 * 1.4826 * mad)
    if MIN_STINT_LAPS <= keep.sum() < len(rows):
        solution = solve(keep)
        residual = target - design @ solution

    for compound in COMPOUNDS:
        model.deg[compound] = float(np.clip(solution[col_deg + compound], 0.0, MAX_DEG))
    for compound in DRY_COMPOUNDS:
        model.offset[compound] = float(solution[col_offset + compound])
    model.evolution = float(solution[col_evolution])
    model.base = {code: float(solution[code_index[code]]) for code in codes}
    model.lap_level = {lap: float(solution[col]) for lap, col in level_index.items()}
    for condition in (CONDITION_DAMP, CONDITION_WET):
        recent = [model.lap_level[lap] for lap in wet_laps if condition_by_lap[lap] == condition][-3:]
        if recent:
            model.condition_level[condition] = float(np.median(recent))
    if len(wet_laps) >= 3 and wet_laps[-1] - wet_laps[-3] == 2:
        levels = [model.lap_level[lap] for lap in wet_laps[-4:]]
        slope = float(np.clip(np.polyfit(range(len(levels)), levels, 1)[0], -1.5, 1.5))
        model.level_trend = (wet_laps[-1], float(np.mean(levels[-2:])), slope)

    # Degradation is deliberately one value per compound for the whole field.
    # A slope per driver fits past laps better but is too noisy to carry over
    # the rest of a race, and made the projected order worse in backtests.
    # The spread between drivers is kept as the uncertainty of the estimate.
    slopes = {}
    for i, row in enumerate(rows):
        if keep[i]:
            slopes.setdefault((row[0], row[1]), []).append((row[3], residual[i]))
    deviations = []
    for laps in slopes.values():
        own = np.array(laps)
        if len(own) >= 8 and np.ptp(own[:, 0]) >= 5:
            deviations.append(float(np.polyfit(own[:, 0], own[:, 1], 1)[0]))
    if len(deviations) >= 3:
        mad = np.median(np.abs(np.array(deviations) - np.median(deviations)))
        model.deg_se = float(np.clip(1.4826 * mad / np.sqrt(len(deviations)), 0.005, 0.05))

    clean = keep & (weights >= 1.0)
    if clean.sum() >= 10:
        model.sigma = float(np.std(residual[clean]))
    return model


def estimate_pit_loss(laps_by_driver, status_laps=None, default=DEFAULT_PIT_LOSS_S):
    """
    Time lost by a green flag pit stop, from the stops made so far.

    Each stop is the in lap plus the out lap minus two normal laps for that
    driver. Returns (seconds, number of stops used); the default is blended
    in until several stops have been seen.
    """
    neutral = neutralised_laps(status_laps)
    losses = []
    for entries in laps_by_driver.values():
        by_lap = {int(e["lap"]): e for e in entries}
        for lap, entry in by_lap.items():
            out = by_lap.get(lap + 1)
            if not entry.get("is_pit_entry") or out is None or lap <= 1:
                continue
            if lap in neutral or lap + 1 in neutral:
                continue
            if entry.get("time_s", -1) <= 0 or out.get("time_s", -1) <= 0:
                continue
            nearby = [
                by_lap[n]["time_s"]
                for n in (lap - 3, lap - 2, lap - 1, lap + 2, lap + 3, lap + 4)
                if n in by_lap and _is_clean(by_lap[n], neutral)
            ]
            if len(nearby) < 2:
                continue
            loss = entry["time_s"] + out["time_s"] - 2 * float(np.median(nearby))
            if 10.0 <= loss <= 45.0:
                losses.append(loss)
    if not losses:
        return default, 0
    observed = float(np.median(losses))
    trust = len(losses) / (len(losses) + 2.0)
    return trust * observed + (1 - trust) * default, len(losses)


def estimate_condition(laps_by_driver, lap, raining=False, window=2):
    """
    Judge the track condition from the tyres the field is on and how the
    slick runners compare with those on intermediates or wets.

    Returns {"condition", "margin_s", "slick", "treaded"} where ``margin_s``
    is how much faster the treaded tyres are (negative once slicks are the
    quicker tyre, i.e. the crossover has passed) or None when the field is
    all on one type.
    """
    times = {}  # code -> list of (lap, time, compound)
    for code, entries in laps_by_driver.items():
        for entry in entries:
            if entry.get("time_s", -1) > 0 and int(entry.get("tyre", -1)) in COMPOUNDS \
                    and not entry.get("is_pit_entry") and not entry.get("is_out_lap"):
                times.setdefault(code, []).append(
                    (int(entry["lap"]), float(entry["time_s"]), int(entry["tyre"]))
                )
    recent = {
        code: [row for row in rows if lap - window < row[0] <= lap]
        for code, rows in times.items()
    }
    recent = {code: rows for code, rows in recent.items() if rows}
    slick = [c for c, rows in recent.items() if rows[-1][2] in DRY_COMPOUNDS]
    treaded = [c for c, rows in recent.items() if rows[-1][2] not in DRY_COMPOUNDS]
    wets = [c for c in treaded if recent[c][-1][2] == WET]
    result = {"condition": CONDITION_DRY, "margin_s": None,
              "slick": len(slick), "treaded": len(treaded)}

    if not treaded:
        result["condition"] = CONDITION_DAMP if raining and not slick else CONDITION_DRY
        return result
    if not slick:
        result["condition"] = CONDITION_WET if len(wets) * 2 > len(treaded) else CONDITION_DAMP
        return result

    # Mixed field: remove each driver's usual gap to the field before comparing
    lap_median = {}
    for rows in times.values():
        for row_lap, time_s, _ in rows:
            lap_median.setdefault(row_lap, []).append(time_s)
    lap_median = {n: float(np.median(v)) for n, v in lap_median.items()}

    def adjusted(code):
        history = [t - lap_median[n] for n, t, _ in times[code] if n <= lap - window]
        usual = float(np.median(history)) if len(history) >= 3 else 0.0
        return float(np.median([t for _, t, _ in recent[code]])) - usual

    margin = float(np.median([adjusted(c) for c in slick])
                   - np.median([adjusted(c) for c in treaded]))
    result["margin_s"] = margin
    if margin <= 0:
        result["condition"] = CONDITION_DRY
    else:
        result["condition"] = CONDITION_WET if len(wets) * 2 > len(treaded) else CONDITION_DAMP
    return result


def condition_delta_from_margin(condition, margin_s):
    """Override the slick against intermediate penalty with a measured margin."""
    if margin_s is None:
        return {}
    if condition == CONDITION_DRY:
        return {(CATEGORY_INTER, CONDITION_DRY): max(0.3, -margin_s)}
    if condition == CONDITION_DAMP:
        return {(CATEGORY_SLICK, CONDITION_DAMP): max(0.3, margin_s)}
    return {(CATEGORY_SLICK, CONDITION_WET): max(0.3, margin_s)}


@dataclass
class StrategyPlan:
    """Result of ``optimise_remaining``."""
    remaining_s: float
    stops: list            # [(lap the car pits at the end of, new compound)]
    window: tuple = None   # (first, last) lap of the next stop within tolerance
    rule_met: bool = True


def _used_mask(compounds):
    mask = 0
    for compound in compounds:
        if compound in DRY_COMPOUNDS:
            mask |= 1 << compound
    return mask


def optimise_remaining(model, code, completed, compound, age, used_compounds=(),
                       pit_loss=DEFAULT_PIT_LOSS_S, pit_loss_now=None,
                       condition_by_lap=None, deg_factor=1.0, allow_first_lap_stop=True):
    """
    Fastest way to run the laps after ``completed`` for one car.

    A dynamic programme over (lap, compound, tyre age, dry compounds used)
    where the car either stays out or fits a new set at the end of each lap.
    The two dry compound rule is enforced unless an intermediate or wet tyre
    has been or will be used. ``pit_loss_now`` applies to a stop at the end
    of the current lap only, for a cheap stop under a safety car.
    """
    total = model.total_laps
    condition_by_lap = condition_by_lap or {}
    n_masks = 1 << len(DRY_COMPOUNDS)
    max_age = total + int(age) + 2
    masks = np.arange(n_masks)

    def solve(enforce_rule):
        value = np.zeros((len(COMPOUNDS), max_age, n_masks, 2))
        if enforce_rule:
            for mask in masks:
                if bin(mask).count("1") < 2:
                    value[:, :, mask, 0] = _BIG
        stay_at, pit_to, pit_cost_at, lap_costs = {}, {}, {}, {}
        for lap in range(total, completed, -1):
            costs = model.lap_costs(
                code, lap, max_age, condition_by_lap.get(lap, CONDITION_DRY), deg_factor
            )
            run = np.full_like(value, _BIG)
            run[:, :-1] = costs[:, 1:, None, None] + value[:, 1:]
            loss = pit_loss
            if lap == completed + 1:
                if not allow_first_lap_stop:
                    loss = _BIG
                elif pit_loss_now is not None:
                    loss = pit_loss_now
            candidates = np.empty((len(COMPOUNDS), n_masks, 2))
            for new in COMPOUNDS:
                if new in DRY_COMPOUNDS:
                    candidates[new] = run[new, 0][masks | (1 << new), :]
                else:
                    candidates[new] = run[new, 0][:, 1][:, None]
            candidates += loss
            pit_to[lap] = candidates.argmin(axis=0)
            pit_cost_at[lap] = candidates.min(axis=0)
            stay_at[lap] = run <= pit_cost_at[lap][None, None]
            lap_costs[lap] = costs
            value = np.where(stay_at[lap], run, pit_cost_at[lap][None, None])
        return value, stay_at, pit_to, pit_cost_at, lap_costs

    mask0 = _used_mask(tuple(used_compounds) + (compound,))
    wet0 = int(any(c not in DRY_COMPOUNDS for c in tuple(used_compounds) + (compound,)))
    age0 = int(min(max(age, 0), max_age - 2))
    if completed >= total:
        return StrategyPlan(0.0, [], None, wet0 == 1 or bin(mask0).count("1") >= 2)

    value, stay_at, pit_to, pit_cost_at, lap_costs = solve(True)
    rule_met = value[compound, age0, mask0, wet0] < _BIG / 2
    if not rule_met:
        value, stay_at, pit_to, pit_cost_at, lap_costs = solve(False)

    stops = []
    cur, cur_age, mask, wet = compound, age0, mask0, wet0
    for lap in range(completed + 1, total + 1):
        if stay_at[lap][cur, cur_age, mask, wet]:
            cur_age += 1
            continue
        new = int(pit_to[lap][mask, wet])
        stops.append((lap - 1, new))
        if new in DRY_COMPOUNDS:
            mask |= 1 << new
        else:
            wet = 1
        cur, cur_age = new, 1

    # Cost of making the next stop at the end of each lap, to show how wide
    # the window around the best lap is
    window = None
    if stops:
        forced = {}
        running = 0.0
        for pit_lap in range(completed, total):
            if pit_lap > completed:
                running += lap_costs[pit_lap][compound, age0 + (pit_lap - completed)]
            forced[pit_lap] = running + pit_cost_at[pit_lap + 1][mask0, wet0]
        best_lap = stops[0][0]
        limit = forced[best_lap] + PIT_WINDOW_TOLERANCE_S
        first = last = best_lap
        while first - 1 in forced and forced[first - 1] <= limit:
            first -= 1
        while last + 1 in forced and forced[last + 1] <= limit:
            last += 1
        window = (first, last)

    return StrategyPlan(float(value[compound, age0, mask0, wet0]), stops, window, bool(rule_met))


def history_time(model, code, stints, completed, pit_loss, condition_by_lap=None, deg_factor=1.0):
    """Model time for the laps a car has already run, including its stops."""
    condition_by_lap = condition_by_lap or {}
    total = 0.0
    for stint in stints:
        for lap in range(stint["start_lap"], min(stint["end_lap"], completed) + 1):
            age = stint["first_age"] + (lap - stint["start_lap"])
            total += model.lap_time(
                code, stint["tyre"], age, lap,
                condition_by_lap.get(lap, CONDITION_DRY), deg_factor,
            )
    return total + pit_loss * max(0, len(stints) - 1)


def regret_band(regret_s, model, total_laps):
    """
    Classify time lost against the optimum, allowing for model error.

    A degradation estimate that is slightly out moves the answer by several
    seconds over a race, so small differences are reported as on optimum.
    """
    noise = max(3.0, model.deg_se * total_laps ** 2 / 8.0)
    if regret_s <= noise:
        return BAND_OPTIMAL
    if regret_s <= 3.0 * noise:
        return BAND_MARGINAL
    return BAND_COSTLY


def analyse_race(lap_times, completed_laps, total_laps, status_laps=None,
                 condition_by_lap=None, forecast=None, positions=None,
                 current_status=None, deg_factor=1.0, raining=False,
                 pace_weight=DEFAULT_PACE_WEIGHT):
    """
    Run the whole pipeline for the current replay position.

    ``lap_times`` may be the full race as broadcast; it is cut down to
    ``completed_laps`` first. ``condition_by_lap`` holds the conditions seen
    so far and ``forecast`` any what-if conditions for laps still to come
    (the current condition is assumed to continue otherwise). ``pace_weight``
    is the share of each car's pace advantage carried into the projection.

    Returns {"model", "pit_loss_s", "pit_stops_seen", "condition", "drivers",
    "standings"}. ``drivers`` maps code to the recommended plan, the regret
    against the car's own optimum and its band. ``standings`` is the
    projected finishing order, fastest first.
    """
    laps = visible_laps(lap_times, completed_laps)
    past_status = [p for p in (status_laps or [])
                   if int(p["start_lap"]) <= max(completed_laps.values(), default=0) + 1]
    leader_lap = max(completed_laps.values(), default=0)

    condition = estimate_condition(laps, leader_lap, raining)
    conditions = dict(condition_by_lap or {})
    conditions.setdefault(leader_lap, condition["condition"])
    for lap in range(leader_lap + 1, int(total_laps) + 1):
        conditions[lap] = (forecast or {}).get(lap, condition["condition"])

    model = fit_pace_model(
        laps, total_laps, past_status, conditions,
        condition_delta_from_margin(condition["condition"], condition["margin_s"]),
    )
    pit_loss, stops_seen = estimate_pit_loss(laps, past_status)
    pit_loss_now = pit_loss * pit_loss_factor(current_status)
    neutral = neutralised_laps(past_status)

    paces = [model.base[c] for c in laps if c in model.base]
    field_pace = float(np.median(paces)) if paces else 0.0

    drivers = {}
    for code, entries in laps.items():
        stints = split_stints(entries)
        if not stints or code not in model.base:
            continue
        completed = stints[-1]["end_lap"]
        current = stints[-1]
        age = current["first_age"] + (completed - current["start_lap"])
        used = tuple(s["tyre"] for s in stints)
        plan = optimise_remaining(
            model, code, completed, current["tyre"], age, used,
            pit_loss, pit_loss_now, conditions, deg_factor,
        )

        # Best race this car could have run from the start, with hindsight
        start_options = [c for c in COMPOUNDS if tyre_category(c) == tyre_category(stints[0]["tyre"])]
        ideal = min(
            optimise_remaining(model, code, 0, c, 0, (), pit_loss, None, conditions,
                               deg_factor, allow_first_lap_stop=False).remaining_s
            for c in start_options
        )
        actual = history_time(model, code, stints, completed, pit_loss, conditions, deg_factor)
        regret = actual + plan.remaining_s - ideal

        # Stops made under a safety car were cheaper than the model charges
        luck = 0.0
        for stint in stints[1:]:
            pit_lap = stint["start_lap"] - 1
            luck += pit_loss * (1.0 - pit_loss_factor(status_at_lap(past_status, pit_lap)))

        pace_vs_field = model.base[code] - field_pace
        elapsed = next(
            (e.get("end_time_s") for e in reversed(entries)
             if e.get("lap") == completed and e.get("end_time_s") is not None),
            None,
        )
        drivers[code] = {
            "completed": completed,
            "compound": current["tyre"],
            "tyre_age": age,
            "stints": [{"tyre": s["tyre"], "start_lap": s["start_lap"], "end_lap": s["end_lap"]}
                       for s in stints],
            "plan": plan,
            "regret_s": float(regret),
            "safety_car_gain_s": float(luck),
            "band": regret_band(regret, model, int(total_laps)),
            "elapsed_s": elapsed,
            "pace_vs_field_s": pace_vs_field,
            "projected_s": (
                elapsed + plan.remaining_s
                - (1.0 - pace_weight) * pace_vs_field * (int(total_laps) - completed)
                if elapsed is not None else None
            ),
            "clean_laps": sum(1 for e in entries if _is_clean(e, neutral)),
        }

    return {
        "model": model,
        "pit_loss_s": pit_loss,
        "pit_stops_seen": stops_seen,
        "condition": condition,
        "drivers": drivers,
        "standings": projected_standings(drivers, positions),
    }


def projected_standings(drivers, positions=None):
    """
    Projected finishing order: time so far plus the best remaining strategy.

    Returns a list of {"code", "projected_position", "position", "change",
    "gap_s"}, fastest first. ``change`` is places gained against the current
    position (positive is a gain) and is None without a current position.
    """
    positions = positions or {}
    ranked = sorted(
        ((code, d["projected_s"]) for code, d in drivers.items() if d["projected_s"] is not None),
        key=lambda item: item[1],
    )
    if not ranked:
        return []
    # Rank current positions among the cars being projected so that a
    # retirement elsewhere does not show up as a place gained
    current_order = sorted((positions[c], c) for c, _ in ranked if positions.get(c) is not None)
    current_rank = {code: i + 1 for i, (_, code) in enumerate(current_order)}
    leader_time = ranked[0][1]
    return [
        {
            "code": code,
            "projected_position": i + 1,
            "position": positions.get(code),
            "change": current_rank[code] - (i + 1) if code in current_rank else None,
            "gap_s": float(projected - leader_time),
        }
        for i, (code, projected) in enumerate(ranked)
    ]


STOP_MUST = "must_stop"
STOP_LIKELY = "likely_stop"
STOP_UNLIKELY = "likely_stay_out"

# Share of the race distance a set is assumed to last until the race itself
# has shown a longer stint on that compound.
DEFAULT_TYRE_LIFE_SHARE = {SOFT: 0.30, MEDIUM: 0.45, HARD: 0.65}
UNDERCUT_RANGE_S = 2.0
# Sets usually go a little further than the longest stint seen so far, and
# teams very rarely make an optional stop close to the end.
TYRE_LIFE_SLACK = 1.1
LATE_RACE_LAPS = 12
LATE_RACE_SLACK = 1.4


def tyre_life(laps_by_driver, total_laps):
    """Laps each dry compound can do: the longest stint seen so far, or a default."""
    life = {c: DEFAULT_TYRE_LIFE_SHARE[c] * total_laps for c in DRY_COMPOUNDS}
    for entries in laps_by_driver.values():
        for stint in split_stints(entries):
            if stint["tyre"] in life:
                age = stint["first_age"] + stint["end_lap"] - stint["start_lap"]
                life[stint["tyre"]] = max(life[stint["tyre"]], age)
    return life


def fitted_counts(laps_by_driver):
    """How many times each compound has been fitted at a pit stop so far."""
    counts = {}
    for entries in laps_by_driver.values():
        for stint in split_stints(entries)[1:]:
            counts[stint["tyre"]] = counts.get(stint["tyre"], 0) + 1
    return counts


def two_compound_rule_met(used):
    """True once a car has used two dry compounds, or any intermediate or wet."""
    return any(c not in DRY_COMPOUNDS for c in used) or len(set(used)) >= 2


def likely_next_tyres(need, current, used, life, counts):
    """
    The dry compounds a car is most likely to fit next, best guess first.

    Follows what the field has been fitting, skipping compounds that would
    not last ``need`` laps or would leave the two compound rule unmet.
    """
    ranked = sorted((c for c in counts if c in DRY_COMPOUNDS), key=lambda c: -counts[c])
    ranked += [c for c in (HARD, MEDIUM, SOFT) if c not in ranked]
    legal = [c for c in ranked if two_compound_rule_met(tuple(used)) or c != current]
    lasting = [c for c in legal if life[c] >= need]
    return lasting + [c for c in legal if c not in lasting]


def gaps_behind(laps_by_driver):
    """
    For each car, the cars behind it on track as [(gap in seconds, code)],
    closest first, measured at the last lap both have completed.
    """
    line_times = {
        code: {int(e["lap"]): e["end_time_s"] for e in entries if e.get("end_time_s") is not None}
        for code, entries in laps_by_driver.items()
    }
    result = {}
    for code, times in line_times.items():
        if not times:
            continue
        behind = []
        for other, other_times in line_times.items():
            if other == code or not other_times:
                continue
            lap = min(max(times), max(other_times))
            if lap in times and lap in other_times and other_times[lap] > times[lap]:
                behind.append((other_times[lap] - times[lap], other))
        result[code] = sorted(behind)
    return result


def predict_next_moves(lap_times, completed_laps, total_laps, status_laps=None,
                       condition_by_lap=None, current_status=None, raining=False):
    """
    What each car is most likely to do next: stop again or not, on which
    tyre, roughly when, and where it would rejoin.

    The stop and tyre calls come from the rules and from what the field has
    done so far, which predicted real stops better than lap time modelling
    in backtests. Lap times are only used for the timing of the stop.

    Returns {"pit_loss_s", "pit_loss_now_s", "pit_stops_seen", "condition",
    "tyre_life", "drivers"} where ``drivers`` maps code to {"completed",
    "position", "compound", "tyre_age", "call", "reason", "next_tyres", "window",
    "gap_behind_s", "car_behind", "places_lost", "free_stop",
    "undercut_threat"}.
    """
    total_laps = int(total_laps)
    laps = visible_laps(lap_times, completed_laps)
    leader_lap = max(completed_laps.values(), default=0)
    past_status = [p for p in (status_laps or []) if int(p["start_lap"]) <= leader_lap + 1]

    condition = estimate_condition(laps, leader_lap, raining)
    conditions = dict(condition_by_lap or {})
    conditions.setdefault(leader_lap, condition["condition"])
    for lap in range(leader_lap + 1, total_laps + 1):
        conditions[lap] = condition["condition"]
    dry = condition["condition"] == CONDITION_DRY

    model = fit_pace_model(
        laps, total_laps, past_status, conditions,
        condition_delta_from_margin(condition["condition"], condition["margin_s"]),
    )
    pit_loss, stops_seen = estimate_pit_loss(laps, past_status)
    pit_loss_now = pit_loss * pit_loss_factor(current_status)
    life = tyre_life(laps, total_laps)
    counts = fitted_counts(laps)
    behind = gaps_behind(laps)

    drivers = {}
    for code, entries in laps.items():
        stints = split_stints(entries)
        if not stints:
            continue
        current = stints[-1]
        done = current["end_lap"]
        if done < leader_lap - 4 or done >= total_laps:
            continue  # retired, or nothing left to decide
        compound = current["tyre"]
        age = current["first_age"] + (done - current["start_lap"])
        used = tuple(s["tyre"] for s in stints)
        remaining = total_laps - done
        on_slicks = compound in DRY_COMPOUNDS

        slack = LATE_RACE_SLACK if remaining <= LATE_RACE_LAPS else TYRE_LIFE_SLACK
        next_tyres = likely_next_tyres(remaining, compound, used, life, counts)
        if dry and not on_slicks:
            call, reason = STOP_LIKELY, "Slicks are now the quicker tyre"
        elif not dry and on_slicks:
            call, reason = STOP_LIKELY, "Track is too wet for slicks"
            next_tyres = [WET if condition["condition"] == CONDITION_WET else INTERMEDIATE]
        elif not dry:
            call, reason = STOP_UNLIKELY, "On the right tyre for the conditions"
            next_tyres = []
        elif not two_compound_rule_met(used):
            call, reason = STOP_MUST, "Still has to run a second compound"
        elif life[compound] * slack < age + remaining:
            call, reason = STOP_LIKELY, "This set is unlikely to reach the end"
        else:
            call, reason = STOP_UNLIKELY, "Can reach the end on this set"

        window = None
        if call != STOP_UNLIKELY:
            if dry and on_slicks and code in model.base:
                window = optimise_remaining(model, code, done, compound, age, used,
                                            pit_loss, pit_loss_now, conditions).window
            if window is None:
                last = done + 2
                if dry and on_slicks:
                    last = done + max(1, int(life[compound] - age))
                window = (done, max(done, min(total_laps - 1, last)))

        cars_behind = behind.get(code, [])
        gap, car_behind = cars_behind[0] if cars_behind else (None, None)
        places_lost = sum(1 for g, _ in cars_behind if g < pit_loss_now)
        drivers[code] = {
            "completed": done,
            "compound": compound,
            "tyre_age": age,
            "call": call,
            "reason": reason,
            "next_tyres": next_tyres[:2],
            "window": window,
            "gap_behind_s": gap,
            "car_behind": car_behind,
            "places_lost": places_lost,
            "free_stop": places_lost == 0,
            "undercut_threat": call != STOP_UNLIKELY and gap is not None and gap < UNDERCUT_RANGE_S,
        }

    # Running order from the timing itself, so that the rejoin position is
    # counted on the same basis as the gaps
    def line_time(code):
        done = drivers[code]["completed"]
        return next((e["end_time_s"] for e in laps[code]
                     if e["lap"] == done and e.get("end_time_s") is not None), float("inf"))

    order = sorted(drivers, key=lambda c: (-drivers[c]["completed"], line_time(c)))
    for position, code in enumerate(order, start=1):
        drivers[code]["position"] = position
        drivers[code]["places_lost"] = min(drivers[code]["places_lost"], len(order) - position)
        drivers[code]["free_stop"] = drivers[code]["places_lost"] == 0

    return {
        "pit_loss_s": pit_loss,
        "pit_loss_now_s": pit_loss_now,
        "pit_stops_seen": stops_seen,
        "condition": condition,
        "tyre_life": life,
        "drivers": drivers,
    }
