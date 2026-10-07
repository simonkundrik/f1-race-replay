import pytest

from src.lib.strategy import (
    BAND_COSTLY,
    BAND_OPTIMAL,
    CONDITION_DAMP,
    CONDITION_DRY,
    DRY_COMPOUNDS,
    HARD,
    INTERMEDIATE,
    MEDIUM,
    SOFT,
    STOP_LIKELY,
    STOP_MUST,
    STOP_UNLIKELY,
    PaceModel,
    analyse_race,
    estimate_condition,
    estimate_pit_loss,
    fit_pace_model,
    gaps_behind,
    likely_next_tyres,
    predict_next_moves,
    tyre_life,
    optimise_remaining,
    projected_standings,
    regret_band,
    split_stints,
    temp_factor,
    visible_laps,
)

TOTAL_LAPS = 50
FUEL = 0.05


def _flat_model(deg, total_laps=TOTAL_LAPS):
    """Model where every dry compound has the same pace and degradation."""
    model = PaceModel(total_laps=total_laps, fuel_per_lap=0.0)
    model.base = {"AAA": 90.0}
    for compound in DRY_COMPOUNDS:
        model.offset[compound] = 0.0
        model.deg[compound] = deg
    return model


def _stint(start_lap, end_lap, tyre, base=90.0, deg=0.1, total_laps=TOTAL_LAPS,
           first_age=1, start_time=0.0, interval=None):
    """Lap entries for one stint following the model exactly."""
    entries = []
    clock = start_time
    for lap in range(start_lap, end_lap + 1):
        age = first_age + (lap - start_lap)
        time_s = base + deg * age + FUEL * (total_laps - lap)
        clock += time_s
        entry = {
            "lap": lap, "time_s": time_s, "end_time_s": clock, "tyre": tyre,
            "tyre_life": age, "is_pit_entry": False, "is_pit_affected": False,
            "is_out_lap": False, "is_accurate": True,
        }
        if interval is not None:
            entry["official_interval_to_ahead_s"] = interval
        entries.append(entry)
    return entries


def _race(stints, pit_loss=20.0, **kwargs):
    """Join stints with an in lap and out lap that share the pit loss."""
    entries = []
    clock = 0.0
    for start_lap, end_lap, tyre in stints:
        laps = _stint(start_lap, end_lap, tyre, start_time=clock, **kwargs)
        if entries:
            entries[-1]["is_pit_entry"] = True
            entries[-1]["is_pit_affected"] = True
            entries[-1]["time_s"] += pit_loss / 2
            laps[0]["is_out_lap"] = True
            laps[0]["time_s"] += pit_loss / 2
        entries.extend(laps)
        clock = entries[-1]["end_time_s"]
    return entries


class TestOptimiseRemaining:
    def test_one_stop_splits_the_race_evenly(self):
        # 0.05 s/lap: one stop costs 52.5 s of wear and pit loss, two cost 62.1 s
        plan = optimise_remaining(_flat_model(0.05), "AAA", 0, SOFT, 0, pit_loss=20.0,
                                  allow_first_lap_stop=False)
        assert [lap for lap, _ in plan.stops] == [25]
        assert plan.stops[0][1] != SOFT
        assert plan.rule_met

    def test_higher_degradation_adds_a_stop(self):
        # 0.15 s/lap: one stop 117.5 s, two stops 106.3 s, three stops 110.7 s
        plan = optimise_remaining(_flat_model(0.15), "AAA", 0, SOFT, 0, pit_loss=20.0,
                                  allow_first_lap_stop=False)
        assert len(plan.stops) == 2

    def test_hotter_track_brings_the_stop_forward(self):
        model = _flat_model(0.05)
        model.deg[SOFT] = 0.08
        normal = optimise_remaining(model, "AAA", 0, SOFT, 0, pit_loss=20.0,
                                    allow_first_lap_stop=False)
        hot = optimise_remaining(model, "AAA", 0, SOFT, 0, pit_loss=20.0,
                                 deg_factor=temp_factor(45.0, 30.0),
                                 allow_first_lap_stop=False)
        assert hot.stops[0][0] <= normal.stops[0][0]
        assert hot.remaining_s > normal.remaining_s

    def test_two_compound_rule_forces_a_stop(self):
        # No wear at all, so stopping is only ever for the rule
        plan = optimise_remaining(_flat_model(0.0), "AAA", 0, MEDIUM, 0, pit_loss=20.0,
                                  allow_first_lap_stop=False)
        assert len(plan.stops) == 1
        assert plan.stops[0][1] in (SOFT, HARD)

    def test_rule_waived_after_intermediates(self):
        plan = optimise_remaining(_flat_model(0.0), "AAA", 10, MEDIUM, 5,
                                  used_compounds=(INTERMEDIATE,), pit_loss=20.0)
        assert plan.stops == []
        assert plan.rule_met

    def test_rain_forecast_switches_to_intermediates(self):
        forecast = {lap: CONDITION_DAMP for lap in range(20, TOTAL_LAPS + 1)}
        plan = optimise_remaining(_flat_model(0.02), "AAA", 10, MEDIUM, 10,
                                  pit_loss=20.0, condition_by_lap=forecast)
        assert plan.stops[0][1] == INTERMEDIATE
        assert 18 <= plan.stops[0][0] <= 20

    def test_cheap_stop_under_safety_car_is_taken_now(self):
        model = _flat_model(0.05)
        green = optimise_remaining(model, "AAA", 18, SOFT, 18, pit_loss=20.0)
        safety_car = optimise_remaining(model, "AAA", 18, SOFT, 18, pit_loss=20.0,
                                        pit_loss_now=11.0)
        assert green.stops[0][0] > 18
        assert safety_car.stops[0][0] == 18

    def test_window_contains_the_best_lap(self):
        plan = optimise_remaining(_flat_model(0.05), "AAA", 5, SOFT, 5, pit_loss=20.0)
        first, last = plan.window
        assert first <= plan.stops[0][0] <= last
        assert last > first

    def test_finished_race_has_nothing_left(self):
        plan = optimise_remaining(_flat_model(0.05), "AAA", TOTAL_LAPS, HARD, 20,
                                  used_compounds=(SOFT,))
        assert plan.remaining_s == 0.0
        assert plan.stops == []


class TestFitPaceModel:
    def test_recovers_degradation_and_base_pace(self):
        # Cars stop on different laps and in a different compound order, which
        # is what lets tyre wear be told apart from track evolution
        bases = {"AAA": 90.0, "BBB": 90.4, "CCC": 91.0}
        bases.update({f"D{i:02d}": 90.0 + 0.1 * i for i in range(9)})
        laps = {}
        for i, (code, base) in enumerate(bases.items()):
            stop = 14 + 2 * i
            order = (MEDIUM, HARD) if i % 2 else (HARD, MEDIUM)
            laps[code] = _race([(1, stop, order[0]), (stop + 1, 45, order[1])], base=base, deg=0.1)
        model = fit_pace_model(laps, TOTAL_LAPS, fuel_per_lap=FUEL)
        assert model.deg[MEDIUM] == pytest.approx(0.1, abs=0.02)
        assert model.deg[HARD] == pytest.approx(0.1, abs=0.02)
        assert model.evolution == pytest.approx(0.0, abs=0.01)
        assert model.base["BBB"] - model.base["AAA"] == pytest.approx(0.4, abs=0.05)
        assert model.base["CCC"] - model.base["AAA"] == pytest.approx(1.0, abs=0.05)

    def test_recovers_track_evolution(self):
        laps = {}
        for i in range(12):
            stop = 14 + 2 * i
            order = (MEDIUM, HARD) if i % 2 else (HARD, MEDIUM)
            entries = _race([(1, stop, order[0]), (stop + 1, 45, order[1])], deg=0.1)
            for entry in entries:
                entry["time_s"] -= 0.03 * entry["lap"]
            laps[f"D{i:02d}"] = entries
        model = fit_pace_model(laps, TOTAL_LAPS, fuel_per_lap=FUEL)
        assert model.evolution == pytest.approx(-0.03, abs=0.01)
        assert model.deg[MEDIUM] == pytest.approx(0.1, abs=0.02)

    def test_single_stint_trend_is_still_predicted(self):
        # With one stint each, wear and evolution cannot be separated, but
        # the next lap of the stint must still be predicted correctly
        laps = {f"D{i:02d}": _stint(1, 25, MEDIUM, deg=0.1) for i in range(12)}
        model = fit_pace_model(laps, TOTAL_LAPS, fuel_per_lap=FUEL)
        expected = 90.0 + 0.1 * 26 + FUEL * (TOTAL_LAPS - 26)
        assert model.lap_time("D00", MEDIUM, 26, 26) == pytest.approx(expected, abs=0.1)

    def test_traffic_laps_do_not_inflate_degradation(self):
        clean = {c: _stint(1, 25, MEDIUM, deg=0.1) for c in ("AAA", "BBB", "CCC")}
        held_up = {c: _stint(1, 25, MEDIUM, deg=0.1) for c in ("AAA", "BBB", "CCC")}
        for entries in held_up.values():
            for entry in entries:
                if entry["lap"] >= 18:
                    entry["time_s"] += 0.8
                    entry["official_interval_to_ahead_s"] = 0.5
        reference = fit_pace_model(clean, TOTAL_LAPS, fuel_per_lap=FUEL).deg[MEDIUM]
        unflagged = {
            c: [{k: v for k, v in e.items() if k != "official_interval_to_ahead_s"} for e in es]
            for c, es in held_up.items()
        }
        with_filter = fit_pace_model(held_up, TOTAL_LAPS, fuel_per_lap=FUEL).deg[MEDIUM]
        without_filter = fit_pace_model(unflagged, TOTAL_LAPS, fuel_per_lap=FUEL).deg[MEDIUM]
        assert abs(with_filter - reference) < abs(without_filter - reference)

    def test_safety_car_laps_are_ignored(self):
        laps = {f"D{i:02d}": _stint(1, 25, MEDIUM, deg=0.1) for i in range(12)}
        for entries in laps.values():
            for entry in entries:
                if 10 <= entry["lap"] <= 13:
                    entry["time_s"] += 30.0
        status = [{"status": "4", "start_lap": 10, "end_lap": 13}]
        model = fit_pace_model(laps, TOTAL_LAPS, status, fuel_per_lap=FUEL)
        assert model.deg[MEDIUM] + model.evolution == pytest.approx(0.1, abs=0.02)
        assert model.sigma < 0.1


class TestHelpers:
    def test_visible_laps_hides_the_future(self):
        laps = {"AAA": _stint(1, 30, SOFT), "BBB": _stint(1, 30, SOFT)}
        seen = visible_laps(laps, {"AAA": 12})
        assert list(seen) == ["AAA"]
        assert max(e["lap"] for e in seen["AAA"]) == 12

    def test_split_stints_keeps_used_tyre_age(self):
        entries = _race([(1, 10, SOFT), (11, 30, HARD)], first_age=3)
        stints = split_stints(entries)
        assert [(s["tyre"], s["start_lap"], s["end_lap"]) for s in stints] == [
            (SOFT, 1, 10), (HARD, 11, 30),
        ]
        assert stints[0]["first_age"] == 3

    def test_pit_loss_measured_from_stops(self):
        laps = {
            code: _race([(1, 20, MEDIUM), (21, 40, HARD)], pit_loss=24.0, deg=0.0)
            for code in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
        }
        loss, stops = estimate_pit_loss(laps)
        assert stops == 6
        assert 22.0 < loss < 24.5

    def test_pit_loss_defaults_before_any_stop(self):
        assert estimate_pit_loss({"AAA": _stint(1, 10, SOFT)}) == (22.0, 0)

    def test_regret_band_allows_for_model_noise(self):
        model = _flat_model(0.05)
        assert regret_band(1.0, model, TOTAL_LAPS) == BAND_OPTIMAL
        assert regret_band(60.0, model, TOTAL_LAPS) == BAND_COSTLY


class TestEstimateCondition:
    def _field(self, slick_time, inter_time):
        laps = {}
        for i in range(6):
            tyre, pace = (MEDIUM, slick_time) if i < 3 else (INTERMEDIATE, inter_time)
            laps[f"D{i}"] = (
                _stint(1, 8, INTERMEDIATE, base=100.0, deg=0.0)
                + _stint(9, 10, tyre, base=pace, deg=0.0)
            )
        return laps

    def test_all_slicks_is_dry(self):
        laps = {"AAA": _stint(1, 10, SOFT), "BBB": _stint(1, 10, MEDIUM)}
        assert estimate_condition(laps, 10)["condition"] == CONDITION_DRY

    def test_all_intermediates_is_damp(self):
        laps = {"AAA": _stint(1, 10, INTERMEDIATE), "BBB": _stint(1, 10, INTERMEDIATE)}
        assert estimate_condition(laps, 10)["condition"] == CONDITION_DAMP

    def test_slicks_quicker_means_crossover_has_passed(self):
        result = estimate_condition(self._field(slick_time=95.0, inter_time=99.0), 10)
        assert result["condition"] == CONDITION_DRY
        assert result["margin_s"] < 0

    def test_intermediates_quicker_means_still_damp(self):
        result = estimate_condition(self._field(slick_time=104.0, inter_time=99.0), 10)
        assert result["condition"] == CONDITION_DAMP
        assert result["margin_s"] > 0


class TestAnalyseRace:
    def _laps(self):
        return {
            "AAA": _race([(1, 25, MEDIUM), (26, 50, HARD)], base=90.0, deg=0.05),
            "BBB": _race([(1, 25, MEDIUM), (26, 50, HARD)], base=90.3, deg=0.05),
            "CCC": _race([(1, 8, SOFT), (9, 50, HARD)], base=90.1, deg=0.05),
        }

    def test_future_laps_do_not_change_the_answer(self):
        full = self._laps()
        completed = {code: 20 for code in full}
        cut = {code: [e for e in entries if e["lap"] <= 20] for code, entries in full.items()}
        from_full = analyse_race(full, completed, TOTAL_LAPS)
        from_cut = analyse_race(cut, completed, TOTAL_LAPS)
        for code in full:
            assert from_full["drivers"][code]["plan"].stops == from_cut["drivers"][code]["plan"].stops
            assert from_full["drivers"][code]["projected_s"] == pytest.approx(
                from_cut["drivers"][code]["projected_s"]
            )

    def test_projection_accounts_for_a_stop_still_owed(self):
        # CCC has stopped and runs just behind AAA and BBB, who have not
        laps = self._laps()
        result = analyse_race(laps, {code: 20 for code in laps}, TOTAL_LAPS,
                              positions={"AAA": 1, "BBB": 2, "CCC": 3})
        order = [row["code"] for row in result["standings"]]
        assert order.index("CCC") < order.index("BBB")
        assert next(r for r in result["standings"] if r["code"] == "CCC")["change"] > 0

    def test_very_early_stop_is_worse_than_an_even_split(self):
        laps = self._laps()
        result = analyse_race(laps, {code: 40 for code in laps}, TOTAL_LAPS)
        assert result["drivers"]["CCC"]["regret_s"] > result["drivers"]["AAA"]["regret_s"]


def test_projected_standings_ranks_by_time():
    drivers = {"AAA": {"projected_s": 5010.0}, "BBB": {"projected_s": 5000.0},
               "CCC": {"projected_s": None}}
    standings = projected_standings(drivers, {"AAA": 1, "BBB": 2})
    assert [row["code"] for row in standings] == ["BBB", "AAA"]
    assert standings[0]["change"] == 1
    assert standings[1]["gap_s"] == pytest.approx(10.0)


class TestNextMoves:
    def _laps(self):
        return {
            "AAA": _race([(1, 30, MEDIUM), (31, 50, HARD)], base=90.0, deg=0.05),
            "BBB": _race([(1, 12, SOFT), (13, 50, HARD)], base=90.3, deg=0.05),
            "CCC": _race([(1, 14, SOFT), (15, 50, HARD)], base=90.6, deg=0.05),
        }

    def test_tyre_life_grows_with_the_longest_stint_seen(self):
        life = tyre_life({"AAA": _stint(1, 40, HARD)}, TOTAL_LAPS)
        assert life[HARD] == 40
        assert life[SOFT] == pytest.approx(0.30 * TOTAL_LAPS)

    def test_next_tyre_follows_the_field(self):
        life = {SOFT: 15, MEDIUM: 25, HARD: 40}
        assert likely_next_tyres(30, MEDIUM, (MEDIUM,), life, {HARD: 5, MEDIUM: 2})[0] == HARD

    def test_next_tyre_must_satisfy_the_two_compound_rule(self):
        life = {SOFT: 15, MEDIUM: 25, HARD: 40}
        picks = likely_next_tyres(20, HARD, (HARD,), life, {HARD: 6, MEDIUM: 1})
        assert picks[0] == MEDIUM
        assert HARD not in picks

    def test_next_tyre_must_last_the_distance(self):
        life = {SOFT: 15, MEDIUM: 25, HARD: 40}
        assert likely_next_tyres(30, SOFT, (SOFT,), life, {MEDIUM: 5, HARD: 1})[0] == HARD

    def test_gaps_behind_are_measured_on_a_shared_lap(self):
        laps = {"AAA": _stint(1, 10, SOFT, base=90.0), "BBB": _stint(1, 10, SOFT, base=91.0)}
        gaps = gaps_behind(laps)
        assert gaps["BBB"] == []
        assert gaps["AAA"][0][1] == "BBB"
        assert gaps["AAA"][0][0] == pytest.approx(10.0, abs=0.01)

    def test_car_on_one_compound_must_stop(self):
        laps = self._laps()
        result = predict_next_moves(laps, {code: 20 for code in laps}, TOTAL_LAPS)
        car = result["drivers"]["AAA"]
        assert car["call"] == STOP_MUST
        assert car["next_tyres"][0] == HARD
        assert car["window"][0] <= car["window"][1]

    def test_car_that_has_stopped_and_can_finish_stays_out(self):
        laps = self._laps()
        result = predict_next_moves(laps, {code: 45 for code in laps}, TOTAL_LAPS)
        assert result["drivers"]["BBB"]["call"] == STOP_UNLIKELY
        assert result["drivers"]["BBB"]["window"] is None

    def test_stop_costs_the_places_within_the_pit_loss(self):
        laps = self._laps()
        result = predict_next_moves(laps, {code: 10 for code in laps}, TOTAL_LAPS)
        leader = result["drivers"]["AAA"]
        assert leader["places_lost"] == 2
        assert not leader["free_stop"]
        assert result["drivers"]["CCC"]["free_stop"]

    def test_future_laps_do_not_change_the_calls(self):
        full = self._laps()
        completed = {code: 20 for code in full}
        cut = {code: [e for e in entries if e["lap"] <= 20] for code, entries in full.items()}
        assert predict_next_moves(full, completed, TOTAL_LAPS)["drivers"] == \
            predict_next_moves(cut, completed, TOTAL_LAPS)["drivers"]

    def test_wet_track_sends_slick_runners_to_intermediates(self):
        # Rain arrives on lap 8: four cars change to intermediates, one stays out
        laps = {
            f"D{i}": _stint(1, 7, MEDIUM) + _stint(8, 10, INTERMEDIATE, base=100.0)
            for i in range(4)
        }
        laps["AAA"] = _stint(1, 7, MEDIUM) + _stint(8, 10, MEDIUM, base=110.0, first_age=8)
        result = predict_next_moves(laps, {code: 10 for code in laps}, TOTAL_LAPS, raining=True)
        assert result["drivers"]["AAA"]["call"] == STOP_LIKELY
        assert result["drivers"]["AAA"]["next_tyres"] == [INTERMEDIATE]
        assert result["drivers"]["D0"]["call"] == STOP_UNLIKELY
