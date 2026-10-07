"""Local: attach the Strategy Predictor to the running replay's stream and save a screenshot."""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.insights.strategy_predictor_window import StrategyPredictorWindow  # noqa: E402

app = QApplication(sys.argv)
window = StrategyPredictorWindow()
window.show()
end = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 20
while time.time() < end:
    app.processEvents()
    time.sleep(0.02)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "predictor_live.png")
window.grab().save(path)
print("messages", window.message_count, "rows", window.table.rowCount(), window.lap_label.text())
print(window.summary_label.text())
window.close()
