# Handoff: Strategy Predictor for f1-race-replay

Working notes for continuing this on another machine. Not part of the upstream pull request.

## What this is

A contribution to [IAmTomShaw/f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) from the fork `simonkundrik/f1-race-replay`, branches `feat/optimal-strategy-insight` (the clean feature, for the pull request) and `work/strategy-research` (the feature plus these scripts and a local workaround; use this one to keep working). It adds a **Strategy Predictor** insight window: per car, whether another pit stop is coming, the likely next tyre, the stop window, and where the car would rejoin.

## Setup on a new machine

```bash
git clone https://github.com/simonkundrik/f1-race-replay.git
cd f1-race-replay
git checkout work/strategy-research
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt scipy
mkdir .fastf1-cache
.venv\Scripts\python -m pytest
```

Python 3.13 was used. `scipy` is needed by the app but missing from `requirements.txt`.

Run the replay (Windows; `PYTHONUTF8=1` avoids a console encoding crash in upstream code when output is redirected):

```bash
set PYTHONUTF8=1 && .venv\Scripts\python main.py --viewer --year 2026 --round 14 --playback-speed 4
```

Then open **Strategy Predictor** under Race Analysis in the Insights Menu, or run `python -m src.insights.strategy_predictor_window` while a replay is streaming. Round 14 of 2026 is the Madrid race ("Spanish Grand Prix"); round 7 is Barcelona.

## Files

Feature (intended for the pull request):
- `src/lib/strategy.py`: all logic, numpy only. Entry point `predict_next_moves`.
- `src/insights/strategy_predictor_window.py`: the window (`PitWallWindow` subclass).
- `src/gui/insights_menu.py`: button and launcher added.
- `tests/lib/test_strategy.py`: unit tests.
- `docs/StrategyPredictor.md`: user-facing description, accuracy and limits.

Local only (keep out of the pull request):
- `src/f1_data.py`: `get_circuit_rotation` guard so Madrid 2026 launches. Upstream PRs 338 and 340 address the same bug.
- `scripts/`: backtests and studies (see below).

## Scripts

All run from the repo root with the venv Python. FastF1 data downloads into `.fastf1-cache` on first use.

| Script | Purpose |
|---|---|
| `backtest_strategy.py` | Projected finishing order vs running order, pace error |
| `ablate.py` | Which model terms help the projection |
| `season_deg.py` | Per-stint degradation for a season, driver and team trends |
| `deg_tables.py` | f1pace-style lap-weighted tables from the stint CSV |
| `next_tyre_test.py` | Next tyre, stop lap and stop-or-stay accuracy at every real stop |
| `life_margin.py` | Tyre life allowance tuning from `stop_worth_test.csv` |
| `window_smoke.py` | Feed the window a real race through the stream, save screenshots |
| `live_check.py` | Attach the window to a running replay and save a screenshot |
| `madrid_data.py`, `dash_data.py` | JSON for the dashboard |
| `output/` | CSV results, screenshots, dashboard HTML and its template |

Dashboard build: `dash_data.py > output/dash.json`, `madrid_data.py 2026 14 > output/madrid.json`, then replace `__DATA__` and `__MADRID__` in `output/dashboard_template.html`. Published at https://claude.ai/artifact/47DFeh2HSfABJukU8GgZva.

## What was found (2024 season + 2026 rounds 1 to 16)

- **Per-driver or per-team tyre wear is not a usable signal.** Odd/even round consistency was 0.10 for drivers and -0.01 for teams. Lap time slope reflects how a stint was driven, not tyre capability.
- **A lap time model is poor at choosing the next tyre** (22% vs 63% for "always hard") and does not improve the stop-or-stay call. It almost always picks the soft because there is no tyre life limit and soft wear is understated.
- **Rules work.** Stop-or-stay call 82% correct (60% by guessing); next tyre 69% in dry races (63% naive); stop lap median error 4 laps (8 naive). Calls were followed by a stop: must stop 99.6%, likely 65%, stays out 21%.
- **Projected finishing order** from the pace model beats the running order modestly in dry races (12% on tuned races, 4% held out), is worse in the wet and at Monaco. Not used by the window.
- The tyre life allowances (10%, 40% in the last 12 laps) and the 0.5 pace weight were tuned on the same races they were scored on.

## Open items

- Decide whether to delete the unused projection and regret code in `strategy.py` (`analyse_race`, `projected_standings`, `regret_band`, `history_time`) before the pull request.
- Upstream asks for a feature issue first and screenshots in the PR. Neither has been drafted or posted. Related upstream: issue 151, PR 283.
- The window has been verified on the real stream up to lap 12 of Madrid 2026; a full race run-through has not been watched.
- The window's position column comes from timing data and can differ from the replay leaderboard, which has known upstream ordering bugs.
- Not built: rejoin-traffic penalty in the model, dirty-air wear factor, pre-race priors from practice.
- Wet races and the first few laps remain unreliable.
