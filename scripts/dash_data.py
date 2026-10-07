"""Local: JSON for the dashboard from deg_stints_2026.csv."""
import json
import os

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "output", "deg_stints_2026.csv"))
rel = df.dropna(subset=["delta"])


def group(key):
    out = []
    for name, g in rel.groupby(key):
        halves = [g[g["round"] % 2 == h]["delta"] for h in (1, 0)]
        a = df[df[key] == name]
        row = {
            "name": name,
            "delta": round(float(np.average(g["delta"], weights=g["laps"])), 4),
            "se": round(float(g["delta"].std(ddof=0) / np.sqrt(len(g))), 4),
            "stints": int(len(g)), "races": int(g["round"].nunique()),
            "odd": round(float(halves[0].mean()), 4) if len(halves[0]) >= 3 else None,
            "even": round(float(halves[1].mean()), 4) if len(halves[1]) >= 3 else None,
            "abs": round(float((a["deg_s_per_lap"] * a["laps"]).sum() / a["laps"].sum()), 3),
        }
        for c in ("SOFT", "MEDIUM", "HARD"):
            s = a[a["compound"] == c]
            row[c[0]] = round(float((s["deg_s_per_lap"] * s["laps"]).sum() / s["laps"].sum()), 3) if len(s) else None
        if key == "driver":
            row["team"] = g["team"].mode()[0]
        out.append(row)
    return sorted(out, key=lambda r: r["delta"])


races = []
for (rnd, race), g in df.groupby(["round", "race"]):
    row = {"round": int(rnd), "race": race.replace(" Grand Prix", ""), "stints": int(len(g))}
    for c in ("SOFT", "MEDIUM", "HARD"):
        s = g[g["compound"] == c]
        row[c[0]] = round(float(s["deg_s_per_lap"].median()), 3) if len(s) >= 4 else None
    races.append(row)

print(json.dumps({"teams": group("team"), "drivers": group("driver"), "races": races},
                 separators=(",", ":")))
