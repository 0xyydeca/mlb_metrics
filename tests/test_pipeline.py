import datetime
import os

import pandas as pd
import pytest

from mlb_metrics import pipeline


def test_build_all_pitch_events_keeps_every_row_no_pre_filtering():
    # No pre-filter on pitch_type/zone here - a row with a null pitch_type
    # but a perfectly real description/zone (e.g. a real ball/strike call)
    # must still come through, since hitters.compute_plate_discipline
    # needs it even though pitchers.compute_pitch_arsenal (which filters
    # on pitch_type itself) would drop it downstream.
    df = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-15"), "batter": 10, "pitcher": 1, "pitch_type": "FF", "description": "ball", "zone": 5},
        {"game_date": pd.Timestamp("2026-06-15"), "batter": 10, "pitcher": 1, "pitch_type": None, "description": "called_strike", "zone": None},
        {"game_date": pd.Timestamp("2026-06-16"), "batter": 20, "pitcher": 2, "pitch_type": "SL", "description": "swinging_strike", "zone": 12},
    ])

    result = pipeline.build_all_pitch_events(df)

    assert len(result) == 3
    assert list(result.columns) == ["game_date", "batter", "pitcher", "pitch_type", "description", "zone"]
    assert set(result["pitcher"]) == {1, 2}


def test_build_all_pitch_events_degrades_gracefully_when_columns_missing():
    # An older/narrower synthetic df predating pitch_type/description/zone
    # - degrades every missing column to null rather than a KeyError, same
    # "missing optional input becomes an honest null" precedent
    # build_pitch_events already establishes for its own quality columns.
    df = pd.DataFrame([{"game_date": pd.Timestamp("2026-06-15"), "batter": 10, "pitcher": 1}])

    result = pipeline.build_all_pitch_events(df)

    assert len(result) == 1
    assert list(result.columns) == ["game_date", "batter", "pitcher", "pitch_type", "description", "zone"]


def test_run_excludes_games_on_or_after_as_of_date(monkeypatch, tmp_path):
    """The pipeline must only ever compute metrics from games strictly
    before --as-of-date, regardless of what the fetch/persist layer hands
    back - this is the fix for the original script's implicit same-day
    leakage risk (see mlb_metrics.pipeline module docstring)."""
    raw = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-06-18", "2026-06-19", "2026-06-20"]),
        "value": [1, 2, 3],
    })

    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: raw)
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)

    captured = {}

    def fake_compute_outputs(df):
        captured["dates"] = sorted(df["game_date"].dt.strftime("%Y-%m-%d").tolist())
        return {"wave": pd.DataFrame(), "pave": pd.DataFrame(), "confidence": pd.DataFrame()}

    monkeypatch.setattr(pipeline, "compute_outputs", fake_compute_outputs)

    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        log_predictions=False,
    )

    assert captured["dates"] == ["2026-06-18", "2026-06-19"]


def test_run_rejects_as_of_date_before_season_start(tmp_path):
    try:
        pipeline.run(
            datetime.date(2026, 1, 1),
            raw_dir=str(tmp_path / "raw"),
            output_dir=str(tmp_path / "out"),
            log_predictions=False,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def _no_schedule(date):
    raise RuntimeError("no network in tests")


def test_run_logs_and_resolves_predictions(monkeypatch, tmp_path):
    """End-to-end check that run() both logs today's picks and resolves any
    pick from a previous run whose target date has now happened, using
    exactly the data fetched for this run (no extra data source)."""
    raw = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-06-18", "2026-06-18", "2026-06-19", "2026-06-19"]),
        "batter": [1, 2, 1, 2],
        # Both of 06-19's picks get a hit -> the Beat the Streak day should succeed.
        "events": ["field_out", "field_out", "single", "double"],
    })
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: raw)
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    # This test only cares about Game_Hit_Probability picks/streak wiring,
    # not matchup blending or game picks - and must not depend on live
    # network access (statsapi is reachable in CI but not in every sandbox,
    # which would make this test's outcome nondeterministic depending on
    # where it runs).
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", _no_schedule)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", _no_schedule)

    wave = pd.DataFrame([
        {
            "key_mlbam": 1, "name_first": "Test", "name_last": "PlayerOne", "team": "NYY",
            "PA_L": 0, "PA_R": 40, "probability_L": 0, "probability_R": 0.9, "probability": 0.9,
            "Game_Hit_Probability": 0.8, "Consistency": -0.1, "Approach": 0.72, "Expected_Bases": 1.5,
        },
        {
            "key_mlbam": 2, "name_first": "Test", "name_last": "PlayerTwo", "team": "BOS",
            "PA_L": 0, "PA_R": 35, "probability_L": 0, "probability_R": 0.8, "probability": 0.8,
            # Above DAILY_PICK_MIN_PROBABILITY (0.80) too, so both picks
            # qualify as "recommended" and the streak wiring below covers a
            # genuine 2-hit day, not just a 1-pick day.
            "Game_Hit_Probability": 0.85, "Consistency": -0.1, "Approach": 0.56, "Expected_Bases": 1.2,
        },
    ])
    monkeypatch.setattr(
        pipeline, "compute_outputs",
        lambda df: {"wave": wave, "pave": pd.DataFrame(), "confidence": pd.DataFrame()},
    )

    predictions_dir = str(tmp_path / "predictions")
    log_path = f"{predictions_dir}/predictions.csv"

    # First run: as_of_date=2026-06-19 logs a pending pick for that date.
    pipeline.run(
        datetime.date(2026, 6, 19),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
        log_predictions=True,
    )
    logged = pd.read_csv(log_path, parse_dates=["date"])
    assert len(logged) == 2  # both ranked picks for 06-19
    assert logged["actual_hit"].isna().all()

    # Second run: as_of_date=2026-06-20 fetches through 06-19, which resolves
    # both pending 06-19 picks (both batters got a hit that day) and logs new
    # pending picks for 06-20.
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
        log_predictions=True,
    )
    logged = pd.read_csv(log_path, parse_dates=["date"])
    assert len(logged) == 4
    resolved_rows = logged[logged["date"] == "2026-06-19"]
    assert (resolved_rows["actual_hit"] == 1.0).all()
    pending_rows = logged[logged["date"] == "2026-06-20"]
    assert pending_rows["actual_hit"].isna().all()

    # write_beat_the_streak_export() should have run after each pass and
    # reflect that 06-19's day (both picks hit) added 2 to the streak.
    picks_export = pd.read_csv(f"{tmp_path}/out/beat_the_streak_picks.csv", parse_dates=["date"])
    assert set(picks_export["date"].dt.strftime("%Y-%m-%d")) == {"2026-06-19", "2026-06-20"}
    resolved_export_rows = picks_export[picks_export["date"] == "2026-06-19"]
    assert len(resolved_export_rows) == 2  # both picks cleared the recommendation threshold
    assert (resolved_export_rows["status"] == "hit").all()

    summary_export = pd.read_csv(f"{tmp_path}/out/beat_the_streak_summary.csv")
    assert summary_export.loc[0, "n_days_resolved"] == 1
    assert summary_export.loc[0, "current_streak"] == 2
    assert summary_export.loc[0, "longest_streak"] == 2

    # Every pick here was logged by this run's own select_picks() call (no
    # legacy rows mixed in), so the current-model-version row matches the
    # all_time one exactly.
    by_version = pd.read_csv(f"{tmp_path}/out/beat_the_streak_summary_by_version.csv")
    assert set(by_version["model_version"]) == {"all_time", pipeline.config.HITTER_MODEL_VERSION}
    current_row = by_version[by_version["model_version"] == pipeline.config.HITTER_MODEL_VERSION].iloc[0]
    assert current_row["n_days_resolved"] == 1
    assert current_row["current_streak"] == 2


def _minimal_outputs():
    wave = pd.DataFrame([
        {
            "key_mlbam": 1, "name_first": "Test", "name_last": "PlayerOne", "team": "NYY",
            "PA_L": 0, "PA_R": 40, "WAVE": 0.33, "probability_L": 0, "probability_R": 0.9, "probability": 0.9,
            "Game_Hit_Probability": 0.85, "Consistency": -0.1, "Approach": 0.72, "Expected_Bases": 1.5,
            # dfs_ml.build_hitter_features (pipeline.run's Model_Hit_Probability
            # tier) requires every dfs_ml.HITTER_FEATURE_COLUMNS column to
            # exist on wave - a real outputs["wave"] always carries these
            # (hitters.compute_extended_dk_rates/compute_quality_of_contact),
            # so this fixture must too.
            "Expected_BB": 0.3, "Expected_HBP": 0.05, "Expected_RBI": 0.5,
            "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
        },
    ])
    # PAVE/PAVE_PLUS chosen so this fixture's Matchup_Hit_Probability clears
    # HITTER_MIN_PROBABILITY (see test_run_logs_matchup_probability_when_schedule_fetch_succeeds).
    pave = pd.DataFrame([{
        "key_mlbam": 999, "name_first": "Probable", "name_last": "Starter",
        "PAVE": 0.25, "PAVE_PLUS": 0.9, "Power_A_PLUS": 0.9,
    }])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.28, "Bullpen_PAVE_PLUS": 1.0}])
    return {"wave": wave, "pave": pave, "confidence": confidence}


def test_run_logs_matchup_probability_when_schedule_fetch_succeeds(monkeypatch, tmp_path):
    """Matchup_Hit_Probability is no longer a separate parallel metric row -
    it's merged into the same Game_Hit_Probability-tagged pick pool and
    factored into both the joint qualifier gate and the ranking (see
    pipeline.run). This batter must still qualify: probability=0.9,
    Game_Hit_Probability=0.85, and (per _minimal_outputs' comment)
    Matchup_Hit_Probability all clear HITTER_MIN_PROBABILITY."""
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs())

    schedule_df = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"), "team": "NYY", "opponent": "BOS",
        "probable_pitcher_key_mlbam": 999, "is_home": True, "game_pk": 824651,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", lambda date: schedule_df)
    monkeypatch.setattr(pipeline.schedule, "fetch_hitter_schedule", lambda date: schedule_df)
    # Scoped to hitter-pick metrics only - game picks get their own tests below.
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: pd.DataFrame())
    # This test is scoped to the Matchup_Approach tier specifically (its
    # own docstring/name), not the Model_Hit_Probability tier - force the
    # fallback so it stays deterministic rather than depending on the real
    # committed model artifact's arbitrary output on this tiny fixture.
    monkeypatch.setattr(pipeline.dfs_ml, "predict_hitter_hit_probability", lambda features: pd.DataFrame(columns=["key_mlbam", "game_pk", "Model_Hit_Probability"]))

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert set(logged["metric"]) == {"Game_Hit_Probability"}
    assert (logged["key_mlbam"] == 1).all()
    assert logged.loc[0, "predicted_probability"] == pytest.approx(0.85)  # still Game_Hit_Probability, not the matchup blend

    probable_pitchers = pd.read_csv(f"{tmp_path}/out/probable_pitchers.csv")
    assert len(probable_pitchers) == 1
    assert probable_pitchers.loc[0, "team"] == "NYY"
    assert probable_pitchers.loc[0, "opponent"] == "BOS"
    assert probable_pitchers.loc[0, "pitcher_name"] == "Probable Starter"
    assert probable_pitchers.loc[0, "PAVE_PLUS"] == pytest.approx(0.9)


def test_run_excludes_pick_with_a_bad_matchup_even_with_strong_probability_and_ghp(monkeypatch, tmp_path):
    """A batter with strong probability/Game_Hit_Probability but a tough
    matchup (a dominant opposing starter+bullpen) must be excluded once
    schedule/matchup data is available - a good matchup is now required just
    as much as the other two signals (see predictions.select_picks)."""
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)

    def _outputs_with_tough_opponent(df):
        outputs = _minimal_outputs()
        # A dominant probable starter + bullpen (both well below league
        # PAVE) should drag this batter's Matchup_Hit_Probability under
        # HITTER_MIN_PROBABILITY despite their strong standalone numbers.
        outputs["pave"] = pd.DataFrame([{
            "key_mlbam": 999, "name_first": "Tough", "name_last": "Opponent",
            "PAVE": 0.08, "PAVE_PLUS": 0.08 / 0.25,
        }])
        outputs["confidence"] = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.08, "Bullpen_PAVE_PLUS": 1.0}])
        return outputs

    monkeypatch.setattr(pipeline, "compute_outputs", _outputs_with_tough_opponent)

    schedule_df = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"), "team": "NYY", "opponent": "BOS",
        "probable_pitcher_key_mlbam": 999, "is_home": True, "game_pk": 824651,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", lambda date: schedule_df)
    monkeypatch.setattr(pipeline.schedule, "fetch_hitter_schedule", lambda date: schedule_df)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: pd.DataFrame())
    # Scoped to the Matchup_Approach tier specifically (its own docstring/
    # name) - force the fallback so it stays deterministic rather than
    # depending on the real committed model artifact's arbitrary output on
    # this tiny fixture.
    monkeypatch.setattr(pipeline.dfs_ml, "predict_hitter_hit_probability", lambda features: pd.DataFrame(columns=["key_mlbam", "game_pk", "Model_Hit_Probability"]))

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert logged.empty


def test_run_logs_model_hit_probability_when_model_predicts(monkeypatch, tmp_path):
    """When dfs_ml.predict_hitter_hit_probability returns a real (non-empty)
    prediction, Model_Hit_Probability gets merged onto the pick pool and
    logged on the resulting pick, with the current (bumped) model_version -
    predicted_probability/metric still report Game_Hit_Probability
    regardless. This fixture has only 1 hitter, so it can't exercise the
    model-shortlist truncation itself (see
    test_run_model_shortlist_excludes_heuristic_favorite_outside_model_top_n
    below for that) - it's just a smoke test that the column reaches the
    log at all."""
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs())
    monkeypatch.setattr(
        pipeline.dfs_ml, "predict_hitter_hit_probability",
        lambda features: pd.DataFrame({
            "key_mlbam": features["key_mlbam"].values,
            "game_pk": features["game_pk"].values if "game_pk" in features.columns else [pd.NA] * len(features),
            "Model_Hit_Probability": [0.9] * len(features),
        }),
    )

    schedule_df = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"), "team": "NYY", "opponent": "BOS",
        "probable_pitcher_key_mlbam": 999, "is_home": True, "game_pk": 824651,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", lambda date: schedule_df)
    monkeypatch.setattr(pipeline.schedule, "fetch_hitter_schedule", lambda date: schedule_df)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: pd.DataFrame())

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert len(logged) == 1
    assert logged.loc[0, "Model_Hit_Probability"] == pytest.approx(0.9)
    assert logged.loc[0, "predicted_probability"] == pytest.approx(0.85)  # still Game_Hit_Probability
    assert logged.loc[0, "metric"] == "Game_Hit_Probability"
    assert logged.loc[0, "model_version"] == pipeline.config.HITTER_MODEL_VERSION


def test_run_model_shortlist_excludes_heuristic_favorite_outside_model_top_n(monkeypatch, tmp_path):
    """Integration-level proof of the model-shortlist-then-heuristic design
    through pipeline.run() itself, not just predictions.select_picks in
    isolation: real feedback after v3 shipped live (2026-08-05, "bts is
    just picking Freddie... get the top ten from the model and then use
    our method... to narrow that down to two"). 11 hitters; one
    (key_mlbam=99) has by far the best Approach/Matchup_Approach but the
    model rates it worst of all 11 - it must be excluded from the logged
    pick even though it would win a pure-heuristic ranking."""
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)

    def _outputs_with_many_hitters(df):
        rows = [
            {
                "key_mlbam": key, "name_first": f"F{key}", "name_last": f"L{key}", "team": "NYY",
                "PA_L": 0, "PA_R": 40, "WAVE": 0.30, "probability_L": 0, "probability_R": 0.60, "probability": 0.60,
                "Game_Hit_Probability": 0.60, "Consistency": 0.0, "Approach": 0.36, "Expected_Bases": 1.0,
                "Expected_BB": 0.3, "Expected_HBP": 0.05, "Expected_RBI": 0.5,
                "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
            }
            for key in range(1, 11)
        ]
        rows.append({
            "key_mlbam": 99, "name_first": "Heuristic", "name_last": "Favorite", "team": "NYY",
            "PA_L": 0, "PA_R": 40, "WAVE": 0.40, "probability_L": 0, "probability_R": 0.95, "probability": 0.95,
            "Game_Hit_Probability": 0.95, "Consistency": 0.0, "Approach": 0.9025, "Expected_Bases": 1.5,
            "Expected_BB": 0.3, "Expected_HBP": 0.05, "Expected_RBI": 0.5,
            "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
        })
        wave = pd.DataFrame(rows)
        pave = pd.DataFrame([{
            "key_mlbam": 999, "name_first": "Probable", "name_last": "Starter",
            "PAVE": 0.25, "PAVE_PLUS": 0.9, "Power_A_PLUS": 0.9,
        }])
        confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.28, "Bullpen_PAVE_PLUS": 1.0}])
        return {"wave": wave, "pave": pave, "confidence": confidence}

    monkeypatch.setattr(pipeline, "compute_outputs", _outputs_with_many_hitters)

    schedule_df = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"), "team": "NYY", "opponent": "BOS",
        "probable_pitcher_key_mlbam": 999, "is_home": True, "game_pk": 824651,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", lambda date: schedule_df)
    monkeypatch.setattr(pipeline.schedule, "fetch_hitter_schedule", lambda date: schedule_df)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: pd.DataFrame())

    # Same Matchup_Hit_Probability for everyone (clears HITTER_MIN_PROBABILITY,
    # not the point of this test) - controlled directly so this test doesn't
    # depend on tuning realistic PAVE/confidence inputs for 11 hitters.
    monkeypatch.setattr(
        pipeline.matchup, "compute_matchup_hit_probability",
        lambda wave, pave, confidence, schedule_df: pd.DataFrame({
            "key_mlbam": wave["key_mlbam"].values,
            "game_pk": schedule_df["game_pk"].iloc[0],
            "Matchup_Hit_Probability": 0.75,
        }),
    )
    # The model rates the heuristic favorite (99) WORST of all 11.
    model_scores = {key: 0.60 + key * 0.01 for key in range(1, 11)}
    model_scores[99] = 0.1
    monkeypatch.setattr(
        pipeline.dfs_ml, "predict_hitter_hit_probability",
        lambda features: pd.DataFrame({
            "key_mlbam": features["key_mlbam"].values,
            "game_pk": features["game_pk"].values,
            "Model_Hit_Probability": features["key_mlbam"].map(model_scores).values,
        }),
    )

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert not logged.empty
    assert 99 not in set(logged["key_mlbam"])


def test_run_continues_without_matchup_when_schedule_fetch_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs())

    def raise_error(date):
        raise RuntimeError("statsapi is down")

    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", raise_error)
    monkeypatch.setattr(pipeline.schedule, "fetch_hitter_schedule", raise_error)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", raise_error)

    predictions_dir = str(tmp_path / "predictions")
    # Must not raise - a new external dependency can't be allowed to break
    # the whole daily update.
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert set(logged["metric"]) == {"Game_Hit_Probability"}  # no Matchup_Hit_Probability logged
    assert not os.path.exists(f"{predictions_dir}/game_predictions.csv")  # no game picks logged either
    assert not os.path.exists(f"{tmp_path}/out/probable_pitchers.csv")  # no probable-pitchers list either


def _minimal_outputs_with_confidence():
    """Like _minimal_outputs, but with the confidence.csv columns
    game_picks.compute_game_win_probabilities needs for two teams."""
    outputs = _minimal_outputs()
    # Bullpen_Power_A_PLUS/Power_A_PLUS mirror the PAVE_PLUS-side values (a
    # reasonable synthetic stand-in) so GAME_PICK_SUSCEPTIBILITY_WEIGHT's
    # blend doesn't dilute this fixture's intended separation toward neutral.
    outputs["confidence"] = pd.DataFrame([
        {
            "team": "NYY", "pyth_Strength": 1.1, "pyth_Confidence": 1.05,
            "suppression_resistance": 1.0, "true_power": 1.0,
            "Bullpen_PAVE_PLUS": 0.9, "Bullpen_Power_A_PLUS": 0.9,
            # win_rate_CI_Low/High (teams.compute_team_win_rate_ci's real
            # output columns, merged into a real assemble_team_metrics
            # frame) - required by game_picks.apply_kelly_uncertainty, which
            # select_game_picks now calls whenever `confidence` is given.
            "win_rate_CI_Low": 0.40, "win_rate_CI_High": 0.60,
        },
        {
            "team": "BOS", "pyth_Strength": 0.9, "pyth_Confidence": 0.95,
            "suppression_resistance": 1.0, "true_power": 1.0,
            "Bullpen_PAVE_PLUS": 1.1, "Bullpen_Power_A_PLUS": 1.1,
            "win_rate_CI_Low": 0.40, "win_rate_CI_High": 0.60,
        },
    ])
    outputs["pave"] = pd.DataFrame([
        {"key_mlbam": 501, "PAVE_PLUS": 0.8, "Power_A_PLUS": 0.8},  # NYY probable starter (tough)
        {"key_mlbam": 502, "PAVE_PLUS": 1.2, "Power_A_PLUS": 1.2},  # BOS probable starter (easy)
    ])
    return outputs


def test_run_logs_game_picks_when_schedule_fetch_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs_with_confidence())
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", _no_schedule)

    schedule_games_df = pd.DataFrame([{
        "game_pk": 100, "date": pd.Timestamp("2026-06-20"), "home_team": "NYY", "away_team": "BOS",
        "home_probable_pitcher_key_mlbam": 501, "away_probable_pitcher_key_mlbam": 502,
        "status": "Scheduled", "home_score": None, "away_score": None,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: schedule_games_df)

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/game_predictions.csv", parse_dates=["date"])
    assert len(logged) == 1
    assert logged.loc[0, "game_pk"] == 100
    assert logged.loc[0, "predicted_winner"] == "NYY"  # NYY has the better composite + faces the weaker pitching
    assert logged.loc[0, "metric"] == "GamePick_Win_Probability"
    assert pd.isna(logged.loc[0, "game_played"])

    summary = pd.read_csv(f"{tmp_path}/out/game_picks_summary.csv")
    # No real market data mocked here (fetch_market_home_win_probabilities
    # isn't patched, so pipeline.run()'s own try/except catches the real
    # network failure and logs with market_probabilities=None) - no bets
    # can be advised without a real market price to compare against.
    assert summary.loc[0, "n_bets_advised"] == 0


def test_run_continues_without_game_picks_when_schedule_fetch_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs_with_confidence())
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", _no_schedule)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", _no_schedule)

    predictions_dir = str(tmp_path / "predictions")
    # Must not raise - a new external dependency can't be allowed to break
    # the whole daily update.
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    assert not os.path.exists(f"{predictions_dir}/game_predictions.csv")


def test_run_resolves_game_picks_across_two_runs(monkeypatch, tmp_path):
    """A game pick logged on day N as pending gets resolved on day N+1 once
    schedule.fetch_game_results reports it Final, and the exported
    game_picks_picks.csv/game_picks_summary.csv reflect the win."""
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs_with_confidence())
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", _no_schedule)

    schedule_games_day1 = pd.DataFrame([{
        "game_pk": 100, "date": pd.Timestamp("2026-06-19"), "home_team": "NYY", "away_team": "BOS",
        "home_probable_pitcher_key_mlbam": 501, "away_probable_pitcher_key_mlbam": 502,
        "status": "Scheduled", "home_score": None, "away_score": None,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: schedule_games_day1)

    # A real, generously-priced market underdog line on NYY (home) - the
    # fixture's own model favors NYY substantially (better composite,
    # weaker opposing pitcher), so this real vigged price (implied
    # probabilities sum to 1.0348, a real ~3.5% vig) guarantees a real,
    # unambiguous positive edge on the home side regardless of the exact
    # model probability this fixture happens to produce.
    market_day1 = pd.DataFrame([{
        "home_team": "NYY", "away_team": "BOS", "market_home_win_probability": 0.4203,
        "market_provider": "DraftKings", "home_moneyline": 130, "away_moneyline": -150,
    }])
    monkeypatch.setattr(pipeline.market_odds, "fetch_market_home_win_probabilities", lambda date: market_day1)

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 19),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    # Day 2: no new games today, but day 1's pick resolves - NYY (predicted
    # winner) actually won 5-3.
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: pd.DataFrame())

    def fake_fetch_results(date):
        assert date == datetime.date(2026, 6, 19)
        return pd.DataFrame([{"game_pk": 100, "status": "Final", "home_score": 5, "away_score": 3}])

    monkeypatch.setattr(pipeline.schedule, "fetch_game_results", fake_fetch_results)

    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/game_predictions.csv")
    assert logged.loc[0, "actual_winner"] == "NYY"
    assert logged.loc[0, "game_played"] == 1
    # A real bet was advised on NYY (home) - real, positive Kelly units -
    # and NYY winning for real resolves it into a real positive profit,
    # via game_predictions.resolve_game_predictions' own extended logic.
    assert logged.loc[0, "bet_units"] > 0
    assert logged.loc[0, "bet_team"] == "NYY"
    assert logged.loc[0, "bet_profit_units"] > 0

    picks_export = pd.read_csv(f"{tmp_path}/out/game_picks_picks.csv")
    assert picks_export.loc[0, "status"] == "win"

    summary_export = pd.read_csv(f"{tmp_path}/out/game_picks_summary.csv")
    assert summary_export.loc[0, "n_bets_advised"] == 1
    assert summary_export.loc[0, "bets_won"] == 1
    assert summary_export.loc[0, "win_rate_on_advised_bets"] == 1.0
    assert summary_export.loc[0, "total_profit_units"] > 0

    by_version = pd.read_csv(f"{tmp_path}/out/game_picks_summary_by_version.csv")
    assert set(by_version["model_version"]) == {"all_time", pipeline.config.GAME_PICK_MODEL_VERSION}
    current_row = by_version[by_version["model_version"] == pipeline.config.GAME_PICK_MODEL_VERSION].iloc[0]
    assert current_row["n_bets_advised"] == 1
    assert current_row["bets_won"] == 1
