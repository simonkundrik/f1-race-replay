import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtGui import QFont, QColor, QBrush
from PySide6.QtCore import Qt, QTimer
from src.gui.pit_wall_window import PitWallWindow
from src.lib.strategy import (
    CONDITION_DRY, STOP_LIKELY, STOP_MUST, STOP_UNLIKELY,
    STATUS_RED_FLAG, STATUS_SAFETY_CAR, STATUS_VSC,
    predict_next_moves,
)


# Keyed by the compound ints used in the telemetry stream (src/lib/tyres.py)
TYRE_COLOURS = {
    0: ("#E8002D", "S", "Soft"),
    1: ("#FFF200", "M", "Medium"),
    2: ("#CACACA", "H", "Hard"),
    3: ("#39B54A", "I", "Inter"),
    4: ("#0067FF", "W", "Wet"),
}

CALL_STYLE = {
    STOP_MUST:     ("MUST STOP",  "#e10600"),
    STOP_LIKELY:   ("LIKELY",     "#ff9a1f"),
    STOP_UNLIKELY: ("STAYS OUT",  "#888888"),
}

BG          = "#0f0f0f"
ROW_BG      = "#1a1a1a"
ROW_ALT     = "#141414"
HEADER_BG   = "#111111"
ACCENT      = "#e10600"
TEXT_WHITE  = "#ffffff"
TEXT_DIM    = "#888888"
BORDER      = "#2a2a2a"
GOOD        = "#39B54A"

COLUMNS = ["P", "DRIVER", "TYRE", "STOP AGAIN?", "LIKELY TYRE", "WINDOW", "IF THEY STOP NOW", "SITUATION"]

# The opening laps have too little to go on
MIN_LAPS = 3


class StrategyPredictorWindow(PitWallWindow):
    """
    Per driver: whether another stop is coming, the likely tyre, roughly
    when, and where the car would rejoin. See docs/StrategyPredictor.md.
    """

    def __init__(self):
        # attrs FIRST — super().__init__() calls setup_ui() immediately
        self.lap_times     = {}
        self.status_laps   = []
        self.completed     = {}
        self.conditions    = {}
        self.total_laps    = 0
        self.track_status  = None
        self.raining       = False
        self._last_key     = None
        self._dirty        = False

        super().__init__()
        self.setWindowTitle("F1 Strategy Predictor")
        self.statusBar().hide()

        # Recalculating on every frame would be wasted work; once a lap is plenty
        self._timer = QTimer()
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ------------------------------------------------------------------ UI --

    def setup_ui(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {BG}; }}
            QWidget {{ background: {BG}; color: {TEXT_WHITE}; }}
            QTableWidget {{ background: {ROW_BG}; alternate-background-color: {ROW_ALT};
                            gridline-color: {BORDER}; border: none; }}
            QHeaderView::section {{ background: {HEADER_BG}; color: {TEXT_DIM}; border: none;
                                    border-bottom: 1px solid {BORDER}; padding: 6px; }}
        """)
        self.resize(1040, 720)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(52)
        header.setStyleSheet(f"background:{HEADER_BG}; border-bottom:1px solid {BORDER};")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(14, 0, 14, 0)

        title = QLabel("STRATEGY PREDICTOR")
        title.setFont(QFont("Arial", 15, QFont.Bold))
        title.setStyleSheet(f"color:{TEXT_WHITE}; letter-spacing:3px; border:none;")
        h_layout.addWidget(title)
        h_layout.addStretch()

        self.status_pill = QLabel("")
        self.status_pill.setFont(QFont("Arial", 9, QFont.Bold))
        h_layout.addWidget(self.status_pill)

        self.lap_label = QLabel("LAP — / —")
        self.lap_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.lap_label.setStyleSheet(f"color:{TEXT_WHITE}; border:none;")
        h_layout.addWidget(self.lap_label)
        outer.addWidget(header)

        # ── Race summary ──────────────────────────────────────────────────
        summary = QFrame()
        summary.setFixedHeight(30)
        summary.setStyleSheet(f"background:{HEADER_BG}; border-bottom:1px solid {BORDER};")
        s_layout = QHBoxLayout(summary)
        s_layout.setContentsMargins(14, 0, 14, 0)
        self.summary_label = QLabel("Waiting for telemetry…")
        self.summary_label.setFont(QFont("Arial", 9))
        self.summary_label.setStyleSheet(f"color:{TEXT_DIM}; border:none;")
        s_layout.addWidget(self.summary_label)
        s_layout.addStretch()
        outer.addWidget(summary)

        # ── Driver table ──────────────────────────────────────────────────
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        head = self.table.horizontalHeader()
        head.setFont(QFont("Arial", 8, QFont.Bold))
        head.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for col in range(len(COLUMNS) - 1):
            head.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)
        outer.addWidget(self.table, 1)

        # ── Footer ────────────────────────────────────────────────────────
        footer = QFrame()
        footer.setFixedHeight(40)
        footer.setStyleSheet(f"background:#0a0a0a; border-top:1px solid {BORDER};")
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(12, 0, 12, 0)
        note = QLabel(
            "Estimates from the tyre rules, the longest stints seen so far and the gaps on track. "
            "Checked against 40 past races: stop call right about 80% of the time, tyre about 69%, "
            "stop lap within about 4 laps. Less reliable in the wet and in the opening laps."
        )
        note.setWordWrap(True)
        note.setFont(QFont("Arial", 8))
        note.setStyleSheet(f"color:{TEXT_DIM}; border:none;")
        f_layout.addWidget(note)
        outer.addWidget(footer)

    # --------------------------------------------------------- Telemetry ---

    def on_telemetry_data(self, data):
        if "frame" not in data:
            return

        frame   = data["frame"] or {}
        drivers = frame.get("drivers", {})

        if data.get("lap_times"):
            self.lap_times = data["lap_times"]
        if "status_laps" in data:
            self.status_laps = data["status_laps"] or []

        sess = data.get("session_data", {})
        if sess.get("total_laps"):
            self.total_laps = int(sess["total_laps"])

        status = str(data.get("track_status", ""))
        self.track_status = status if status in (STATUS_SAFETY_CAR, STATUS_RED_FLAG) + STATUS_VSC else None
        self.raining = (frame.get("weather") or {}).get("rain_state") == "RAINING"

        completed = {}
        for code, driver in drivers.items():
            lap = driver.get("lap")
            if lap is None:
                continue
            completed[code] = max(0, int(lap) - 1)
        self.completed = completed

        # Only the laps completed and the track status change the answer
        key = (max(completed.values(), default=0), sum(completed.values()), self.track_status)
        if key != self._last_key:
            self._last_key = key
            self._dirty = True

    # ----------------------------------------------------------- Render ----

    def _refresh(self):
        if not self._dirty:
            return
        self._dirty = False

        leader_done = max(self.completed.values(), default=0)
        self.lap_label.setText(f"LAP {leader_done + 1} / {self.total_laps or '—'}")
        self._set_status_pill()
        if not self.total_laps or not self.lap_times or leader_done < MIN_LAPS:
            self.summary_label.setText("Predictions start once a few laps have been completed.")
            self.table.setRowCount(0)
            return

        # After a rewind, forget conditions recorded for laps that are now in the future
        self.conditions = {lap: c for lap, c in self.conditions.items() if lap < leader_done}
        try:
            result = predict_next_moves(
                self.lap_times, self.completed, self.total_laps, self.status_laps,
                self.conditions, self.track_status, self.raining,
            )
        except Exception as e:
            self.summary_label.setText(f"Could not calculate predictions: {e}")
            return
        self.conditions[leader_done] = result["condition"]["condition"]

        self._set_summary(result)
        self._fill_table(result)

    def _set_status_pill(self):
        if self.track_status == STATUS_SAFETY_CAR:
            text, colour = "SAFETY CAR", "#ff9a1f"
        elif self.track_status in STATUS_VSC:
            text, colour = "VIRTUAL SAFETY CAR", "#ff9a1f"
        elif self.track_status == STATUS_RED_FLAG:
            text, colour = "RED FLAG", ACCENT
        else:
            self.status_pill.setText("")
            self.status_pill.setStyleSheet("border:none;")
            return
        self.status_pill.setText(f"  {text}  ")
        self.status_pill.setStyleSheet(
            f"color:#000000; background:{colour}; border:none; border-radius:3px; padding:2px;"
        )

    def _set_summary(self, result):
        loss = f"Pit stop costs about {result['pit_loss_s']:.0f}s"
        if result["pit_loss_now_s"] < result["pit_loss_s"] - 0.5:
            loss += f" ({result['pit_loss_now_s']:.0f}s right now)"
        elif not result["pit_stops_seen"]:
            loss += " (typical value, no stops yet)"
        life = result["tyre_life"]
        lasts = "  ".join(f"{TYRE_COLOURS[c][1]} {life[c]:.0f}" for c in sorted(life))
        condition = result["condition"]["condition"]
        track = "Dry track" if condition == CONDITION_DRY else f"{condition.title()} track (low confidence)"
        self.summary_label.setText(f"{loss}   │   Longest a set should last, in laps:  {lasts}   │   {track}")

    def _fill_table(self, result):
        drivers = result["drivers"]
        codes = sorted(drivers, key=lambda c: drivers[c]["position"])
        self.table.setRowCount(len(codes))

        for row, code in enumerate(codes):
            car = drivers[code]
            # Order from the timing at the last completed lap, which is what
            # the gaps and the rejoin position are measured against
            position = car["position"]
            colour, abbr, _ = TYRE_COLOURS.get(car["compound"], ("#888888", "?", "Unknown"))
            call_text, call_colour = CALL_STYLE[car["call"]]

            if car["next_tyres"] and car["call"] != STOP_UNLIKELY:
                first = TYRE_COLOURS[car["next_tyres"][0]]
                next_text, next_colour = first[2], first[0]
                if len(car["next_tyres"]) > 1:
                    next_text += f"  (or {TYRE_COLOURS[car['next_tyres'][1]][2]})"
            else:
                next_text, next_colour = "—", TEXT_DIM

            if car["window"] and car["call"] != STOP_UNLIKELY:
                first_lap, last_lap = car["window"]
                # The window is the lap the car pits at the end of
                window_text = f"Lap {first_lap}" if first_lap == last_lap else f"Laps {first_lap}–{last_lap}"
            else:
                window_text = "—"

            if car["free_stop"]:
                rejoin_text, rejoin_colour = "Free stop, keeps position", GOOD
            else:
                rejoin_text = f"Rejoins P{position + car['places_lost']}  (−{car['places_lost']})"
                rejoin_colour = TEXT_WHITE

            situation = car["reason"]
            if car["call"] != STOP_UNLIKELY and self.track_status in (STATUS_SAFETY_CAR,) + STATUS_VSC:
                situation = "Cheap stop available now  ·  " + situation
            if car["undercut_threat"]:
                situation += f"  ·  {car['car_behind']} {car['gap_behind_s']:.1f}s behind, undercut threat"

            cells = [
                (str(position), TEXT_DIM, False),
                (code, TEXT_WHITE, True),
                (f"{abbr}  {car['tyre_age']} laps", colour, True),
                (call_text, call_colour, True),
                (next_text, next_colour, False),
                (window_text, TEXT_WHITE, False),
                (rejoin_text, rejoin_colour, False),
                (situation, TEXT_DIM, False),
            ]
            for col, (text, text_colour, bold) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(QBrush(QColor(text_colour)))
                item.setFont(QFont("Arial", 10, QFont.Bold if bold else QFont.Normal))
                self.table.setItem(row, col, item)


def main():
    app = QApplication(sys.argv)
    window = StrategyPredictorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
