"""Local: lap-weighted absolute degradation tables from deg_stints_<year>.csv."""
import os
import sys

import pandas as pd

year = sys.argv[1] if len(sys.argv) > 1 else "2026"
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", f"deg_stints_{year}.csv"))


def wavg(g):
    return (g["deg_s_per_lap"] * g["laps"]).sum() / g["laps"].sum()


print("Per tyre:", {c: round(wavg(g), 3) for c, g in df.groupby("compound")})
print("Share of laps per tyre:", (df.groupby("compound")["laps"].sum() / df["laps"].sum()).round(2).to_dict())
for key in ("team", "driver"):
    table = df.groupby(key).apply(
        lambda g: pd.Series({"all": wavg(g), "laps": g["laps"].sum(), **{
            c[0]: wavg(g[g["compound"] == c]) if (g["compound"] == c).any() else float("nan")
            for c in ("SOFT", "MEDIUM", "HARD")}}), include_groups=False).sort_values("all")
    print(f"\nBy {key} (s/lap, lap-weighted):")
    print(table.round(3).to_string())
print("\nStint slope spread (s/lap): std", round(df["deg_s_per_lap"].std(), 3),
      " median laps per stint", df["laps"].median())
