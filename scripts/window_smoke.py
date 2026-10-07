"""
Local smoke test: feed the Strategy Predictor window a real race through the
telemetry stream and save screenshots, without starting the replay.

    python scripts/window_smoke.py 2026 16 20 40
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import fastf1  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_strategy as bt  # noqa: E402
from src.services.stream import TelemetryStreamServer  # noqa: E402
from src.insights.strategy_predictor_window import StrategyPredictorWindow  # noqa: E402

year, rnd = int(sys.argv[1]), int(sys.argv[2])
laps = [int(x) for x in sys.argv[3:]] or [20]
fastf1.Cache.enable_cache(".fastf1-cache")
fastf1.set_log_level("ERROR")
name, lap_times, total_laps, status_laps, leader_end, final = bt.load_race(year, rnd)

server = TelemetryStreamServer()
server.start()
app = QApplication(sys.argv)
window = StrategyPredictorWindow()
window.show()


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


pump(1.5)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
for lap in laps:
    cutoff = leader_end[lap]
    done = {c: sum(1 for e in es if e["end_time_s"] <= cutoff) for c, es in lap_times.items()}
    order = sorted((c for c in done if done[c]), key=lambda c: (
        -done[c], next(e["end_time_s"] for e in lap_times[c] if e["lap"] == done[c])))
    alive = [c for c in order if done[c] >= lap - 3]
    status = next((p["status"] for p in status_laps if p["start_lap"] <= lap + 1 <= p["end_lap"]), "1")
    payload = {
        "frame": {
            "drivers": {c: {"lap": done[c] + 1, "position": i + 1, "tyre": 0.0} for i, c in enumerate(alive)},
            "lap": lap + 1, "weather": {"rain_state": "DRY"},
        },
        "track_status": status,
        "session_data": {"lap": lap + 1, "total_laps": total_laps},
        "lap_times": lap_times, "status_laps": status_laps,
    }
    server.broadcast(payload)
    pump(4.0)
    path = os.path.join(out, f"predictor_{year}_{rnd}_lap{lap}.png")
    window.grab().save(path)
    print(f"{name} lap {lap}: rows {window.table.rowCount()}  status {status}  -> {path}")
    print("   " + window.summary_label.text())
window.close()
server.stop()
