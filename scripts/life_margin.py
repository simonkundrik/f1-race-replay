"""Local: how much slack on tyre life gives the best stop-again call."""
import os

import pandas as pd

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "stop_worth_test.csv"))
late = df[df["remaining"] <= 12]
for margin in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
    def acc(d):
        pred = (~d["rule_met"]) | (d["life"] * margin < d["age"] + d["remaining"])
        return (pred == d["actual"]).mean()
    print(f"margin {margin:.1f}: all {acc(df):.3f}  2024 {acc(df[df.year == 2024]):.3f}  "
          f"2026 {acc(df[df.year == 2026]):.3f}  last 12 laps {acc(late):.3f}  "
          f"rule met {acc(df[df.rule_met]):.3f}")
print("last 12 laps: share that stopped again", round(late["actual"].mean(), 3), "n", len(late))
