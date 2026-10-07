# Strategy Predictor

## Overview

The Strategy Predictor is an insight window that shows, for every car still running, what it is most likely to do next: whether another pit stop is coming, which tyre it will probably fit, roughly when, and where it would rejoin if it stopped now.

Open it from the Insights Menu under **Race Analysis**, or run it on its own while a replay is streaming:

```bash
python -m src.insights.strategy_predictor_window
```

It only uses laps that have already happened at the current replay position, so scrubbing backwards gives the prediction that could have been made at that point.

## Columns

| Column | Meaning |
|--------|---------|
| Tyre | Current compound and its age in laps |
| Stop again? | `MUST STOP`, `LIKELY` or `STAYS OUT` (see below) |
| Likely tyre | Most likely next compound, with the second choice in brackets |
| Window | Laps at the end of which the stop is expected |
| If they stop now | Position the car would rejoin in, or "Free stop" when no car is within the pit loss behind |
| Situation | Why the call was made, plus an undercut warning when a car is within 2 seconds behind |

The summary line shows the estimated time lost by a pit stop (lower under a safety car or VSC) and how many laps each compound is expected to last.

## How the calls are made

All logic lives in [`src/lib/strategy.py`](../src/lib/strategy.py) (`predict_next_moves`) and has no GUI dependencies.

**Stop again?**
- `MUST STOP`: the car has not yet used two dry compounds.
- `LIKELY`: the current set is not expected to reach the end. A set's life is the longest stint seen on that compound so far in the race (or a default share of race distance early on), with a 10% allowance, rising to 40% in the last 12 laps.
- `STAYS OUT`: the rule is met and the set should reach the end.

**Likely tyre:** the compound the field has fitted most often so far, skipping any that would not last the remaining distance or would leave the two compound rule unmet.

**Window:** from a lap time model fitted to the race so far (base pace per driver, degradation per compound, track evolution), which finds the cheapest lap to stop. When the model sees no stop, the window runs to the lap the set is expected to wear out.

**If they stop now:** the pit loss is measured from the stops already made in the race. Cars within that gap behind are the places lost.

**Conditions:** the track is judged damp or wet from the tyres the field is on and from how slick runners compare with those on intermediates.

## How well it works

Checked against every pit stop in the 2024 season and the first 16 races of 2026 (40 races), using only what was known at the time:

| Prediction | Result | Naive guess |
|------------|--------|-------------|
| Will the car stop again? (dry running) | 82% correct | 60% (always the more common outcome) |
| Next tyre (dry races, 3 laps ahead) | 69% correct | 63% (always the hard) |
| Stop lap | median 4 laps out | 8 laps (median lap of earlier stops) |

How often each call was followed by another stop: `MUST STOP` 99.6%, `LIKELY` 65%, `STAYS OUT` 21%.

The tyre life allowances were chosen on the same races they are scored on, so expect slightly lower figures on new races. Before the allowances the stop call was 80% correct.

## Limitations

- Each car is judged on its own. Traffic, track position and covering a rival are not modelled beyond the rejoin position and the undercut warning.
- Which tyre sets a car has left is not known.
- Wet and changing conditions are much less reliable: the next tyre was right 43% of the time in wet races.
- The opening laps have little to go on; nothing is shown before lap 4.
- Safety cars cannot be foreseen. Once one is out, the cheaper stop is shown.
- A lap time model alone was tried for the tyre choice and for whether an optional stop pays for itself. It was right 22% of the time on the tyre and did not improve the stop call, because drivers manage their pace and the measured wear rates understate what a fresh set gains. That is why those two calls use rules instead.
