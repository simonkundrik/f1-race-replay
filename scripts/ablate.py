"""Local ablation: which per-driver terms help the projected order."""
import os
import sys

import fastf1
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.lib.strategy as strategy  # noqa: E402
import backtest_strategy as bt  # noqa: E402

original = strategy.fit_pace_model
RACES = [r for r in bt.DEFAULT_RACES if r not in bt.WET_RACES]


def variant(no_form, no_driver_deg):
    def fit(*args, **kwargs):
        model = original(*args, **kwargs)
        if no_form:
            model.form = {}
        if no_driver_deg:
            model.driver_deg = {}
        return model
    return fit


fastf1.Cache.enable_cache(".fastf1-cache")
fastf1.set_log_level("ERROR")
for name, flags in (("no form", (True, False)), ("no driver deg", (False, True)),
                    ("neither", (True, True))):
    strategy.fit_pace_model = variant(*flags)
    out = []
    for spec in RACES:
        year, rnd = (int(x) for x in spec.split(":"))
        out.append(bt.backtest(year, rnd))
    print(f"{name:<14} naive {np.mean([s['naive'] for s in out]):.3f}  pace MAE "
          f"{np.mean([s['pace_mae'] for s in out]):.2f}  by weight: "
          + "  ".join(f"{w}: {np.mean([s['weights'][w] for s in out]):.3f}" for w in bt.WEIGHTS),
          flush=True)
