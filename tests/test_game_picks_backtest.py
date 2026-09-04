import subprocess

import pandas as pd
import pytest

from mlb_metrics import config, game_picks_backtest, game_predictions


def _statcast_game(game_pk, date, home_team, away_team, home_pitcher, away_pitcher, home_score, away_score):
    """A minimal 4-row single-game Statcast fixture: Top(home pitching)/
    Bot(away pitching) alternating, score climbing to the given final."""
    return [
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": home_pitcher, "inning_topbot": "Top", "at_bat_number": 1,
         "post_home_score": 0, "post_away_score": 0},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": away_pitcher, "inning_topbot": "Bot", "at_bat_number": 2,
         "post_home_score": min(1, home_score), "post_away_score": 0},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": home_pitcher, "inning_topbot": "Top", "at_bat_number": 3,
         "post_home_score": min(1, home_score), "post_away_score": min(1, away_score)},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": away_pitcher, "inning_topbot": "Bot", "at_bat_number": 4,
         "post_home_score": home_score, "post_away_score": away_score},
    ]


def test_derive_historical_schedule_games_uses_real_game_pk_not_reconstructed_id():
    # Two real doubleheader games (same two teams, same date, different
    # game_pk and different final scores) - a synthetic per-run game_id
    # reconstruction could conflate or fragment these; using Statcast's own
    # game_pk directly must not.
    rows = (
        _statcast_game(100, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
        + _statcast_game(101, pd.Timestamp("2026-06-01"), "NYY", "BOS", 102, 202, 1, 3)
    )
    games = game_picks_backtest.derive_historical_schedule_games(pd.DataFrame(rows)).set_index("game_pk")

    assert len(games) == 2
    assert games.loc[100, "home_score"] == 5 and games.loc[100, "away_score"] == 2
    assert games.loc[100, "home_probable_pitcher_key_mlbam"] == 101
    assert games.loc[100, "away_probable_pitcher_key_mlbam"] == 201
    assert games.loc[101, "home_score"] == 1 and games.loc[101, "away_score"] == 3
    assert games.loc[101, "status"] == "Final"
    assert (games["schedule_backtest_mode"] == "actual_starter").all()
    assert (games["schedule_source"] == "actual_starter_diagnostic").all()


def test_derive_historical_schedule_games_excludes_zero_zero_artifacts():
    # MLB games can't end 0-0 - any reconstructed 0-0 "final" is a data
    # artifact (see module docstring) and must be dropped, not treated as a
    # real result to backtest against.
    rows = [
        {"game_pk": 999, "game_date": pd.Timestamp("2026-06-01"), "home_team": "NYY", "away_team": "BOS",
         "pitcher": 101, "inning_topbot": "Top", "at_bat_number": 1, "post_home_score": 0, "post_away_score": 0},
    ]
    games = game_picks_backtest.derive_historical_schedule_games(pd.DataFrame(rows))

    assert games.empty


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True, text=True)


def _confidence_csv(rows):
    # Bullpen_Power_A_PLUS mirrors Bullpen_PAVE_PLUS (a reasonable synthetic
    # stand-in) so GAME_PICK_SUSCEPTIBILITY_WEIGHT's blend doesn't dilute
    # these fixtures' intended pitching-quality separation toward neutral.
    return pd.DataFrame(
        [
            {
                "team": team, "pyth_Strength": s, "pyth_Confidence": c,
                "suppression_resistance": sr, "true_power": tp,
                "Bullpen_PAVE_PLUS": bp, "Bullpen_Power_A_PLUS": bp,
            }
            for team, s, c, sr, tp, bp in rows
        ]
    )


def _init_repo_with_confidence_history(tmp_path):
    repo = tmp_path / "repo"
    data_dir = repo / "docs" / "data"
    data_dir.mkdir(parents=True)
    confidence_path = data_dir / "confidence.csv"
    pave_path = data_dir / "pave.csv"

    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    # Day 1: NYY (home) favored over BOS - faces BOS's weaker starter+bullpen.
    _confidence_csv(
        [("NYY", 1.1, 1.05, 1.0, 1.0, 0.9), ("BOS", 0.9, 0.95, 1.0, 1.0, 1.1)]
    ).to_csv(confidence_path, index=False)
    pd.DataFrame([
        {"key_mlbam": 101, "PAVE_PLUS": 0.8, "Power_A_PLUS": 0.8},
        {"key_mlbam": 201, "PAVE_PLUS": 1.2, "Power_A_PLUS": 1.2},
    ]).to_csv(pave_path, index=False)
    _git(["add", "docs/data/confidence.csv", "docs/data/pave.csv"], repo)
    _git(["commit", "-q", "-m", "day1", "--date=2026-06-01T12:00:00"], repo)

    return repo, confidence_path, pave_path


def test_reconstruct_historical_game_picks_replays_and_resolves_immediately(tmp_path):
    repo, _, _ = _init_repo_with_confidence_history(tmp_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks(
        repo_dir=str(repo), raw_dir=str(raw_dir), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert len(picks) == 1
    row = picks.iloc[0]
    assert row["game_pk"] == 555
    assert row["predicted_winner"] == "NYY"  # better composite + faces BOS's weaker pitching
    # Already resolved - a backtested pick's outcome is known at reconstruction time.
    assert row["game_played"] == 1
    assert row["actual_winner"] == "NYY"
    # Reconstructed rows are tagged "legacy", not the current live model
    # version - they replay old-era confidence.csv/pave.csv, not current logic.
    assert row["model_version"] == game_predictions.LEGACY_MODEL_VERSION


def test_reconstruct_historical_game_picks_skips_commits_missing_required_columns(tmp_path):
    repo = tmp_path / "repo"
    data_dir = repo / "docs" / "data"
    data_dir.mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    # A commit predating pyth_Strength/true_power etc. - old schema, must be skipped.
    pd.DataFrame([{"team": "NYY", "current": 1.0}]).to_csv(data_dir / "confidence.csv", index=False)
    pd.DataFrame([{"key_mlbam": 101, "PAVE_PLUS": 0.8}]).to_csv(data_dir / "pave.csv", index=False)
    _git(["add", "docs/data/confidence.csv", "docs/data/pave.csv"], repo)
    _git(["commit", "-q", "-m", "old-schema", "--date=2026-06-01T12:00:00"], repo)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks(
        repo_dir=str(repo), raw_dir=str(raw_dir), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert picks.empty


def test_reconstruct_historical_game_picks_days_limits_the_replay_window(tmp_path):
    repo = tmp_path / "repo"
    data_dir = repo / "docs" / "data"
    data_dir.mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    pave = pd.DataFrame([
        {"key_mlbam": 101, "PAVE_PLUS": 0.8, "Power_A_PLUS": 0.8},
        {"key_mlbam": 201, "PAVE_PLUS": 1.2, "Power_A_PLUS": 1.2},
    ])

    # Slightly different content each day - identical content would leave
    # nothing to commit (git commit fails with no changes staged).
    for day, date, nyy_strength in [(1, "2026-06-01", 1.1), (2, "2026-06-02", 1.2)]:
        _confidence_csv(
            [("NYY", nyy_strength, 1.05, 1.0, 1.0, 0.9), ("BOS", 0.9, 0.95, 1.0, 1.0, 1.1)]
        ).to_csv(data_dir / "confidence.csv", index=False)
        pave.to_csv(data_dir / "pave.csv", index=False)
        _git(["add", "docs/data/confidence.csv", "docs/data/pave.csv"], repo)
        _git(["commit", "-q", "-m", f"day{day}", f"--date={date}T12:00:00"], repo)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2) + _statcast_game(
        556, pd.Timestamp("2026-06-02"), "NYY", "BOS", 101, 201, 3, 1
    )
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks(
        repo_dir=str(repo), raw_dir=str(raw_dir), season=2026, days=1,
        schedule_backtest_mode="actual_starter",
    )

    assert len(picks) == 1
    assert picks.iloc[0]["game_pk"] == 556  # only the most recent day's commit replayed


def _fake_current_outputs():
    return {
        "confidence": _confidence_csv(
            [("NYY", 1.1, 1.05, 1.0, 1.0, 0.9), ("BOS", 0.9, 0.95, 1.0, 1.0, 1.1)]
        ),
        "pave": pd.DataFrame([
            {"key_mlbam": 101, "PAVE_PLUS": 0.8, "Power_A_PLUS": 0.8},
            {"key_mlbam": 201, "PAVE_PLUS": 1.2, "Power_A_PLUS": 1.2},
        ]),
    }


def test_reconstruct_historical_game_picks_from_persisted_tags_current_model_version(tmp_path, monkeypatch):
    # compute_outputs itself needs a live get_name_register() network call and
    # full hitters/pitchers/teams assembly - stubbed out here the same way
    # test_pipeline.py stubs it, so this test exercises only the per-date
    # replay/resolve wiring, not the real metric computation.
    monkeypatch.setattr(game_picks_backtest.pipeline, "compute_outputs", lambda df: _fake_current_outputs())

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Two days: the first only supplies "prior history" for the second (the
    # first date itself has nothing before it to build a pick from, and is
    # asserted against separately below).
    rows = _statcast_game(554, pd.Timestamp("2026-05-31"), "NYY", "BOS", 101, 201, 2, 1) + _statcast_game(
        555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2
    )
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks_from_persisted(
        raw_dir=str(raw_dir), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert len(picks) == 1
    row = picks.iloc[0]
    assert row["game_pk"] == 555
    assert row["predicted_winner"] == "NYY"  # better composite + faces BOS's weaker pitching
    assert row["game_played"] == 1
    assert row["actual_winner"] == "NYY"
    # Every replayed date uses today's live logic end to end - tagged with
    # the current model version, not "legacy" (contrast with the
    # git-history version's test above).
    assert row["model_version"] == config.GAME_PICK_MODEL_VERSION


def test_reconstruct_historical_game_picks_from_persisted_skips_a_date_with_no_prior_history(tmp_path, monkeypatch):
    # The very first calendar date with any persisted data has nothing
    # strictly before it to build a pick from (same as pipeline.run()'s
    # as_of_date cutoff) - it must be skipped, not crash or use an empty
    # history as if it were real.
    monkeypatch.setattr(game_picks_backtest.pipeline, "compute_outputs", lambda df: _fake_current_outputs())

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2) + _statcast_game(
        556, pd.Timestamp("2026-06-02"), "NYY", "BOS", 101, 201, 3, 1
    )
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks_from_persisted(
        raw_dir=str(raw_dir), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert len(picks) == 1
    assert picks.iloc[0]["game_pk"] == 556


def test_reconstruct_historical_game_picks_from_persisted_days_limits_the_replay_window(tmp_path, monkeypatch):
    monkeypatch.setattr(game_picks_backtest.pipeline, "compute_outputs", lambda df: _fake_current_outputs())

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = (
        _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
        + _statcast_game(556, pd.Timestamp("2026-06-02"), "NYY", "BOS", 101, 201, 3, 1)
        + _statcast_game(557, pd.Timestamp("2026-06-03"), "NYY", "BOS", 101, 201, 4, 0)
    )
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks_from_persisted(
        raw_dir=str(raw_dir), season=2026, days=1,
        schedule_backtest_mode="actual_starter",
    )

    assert len(picks) == 1
    assert picks.iloc[0]["game_pk"] == 557  # only the most recent replayable day


def test_reconstruct_historical_game_picks_from_persisted_no_persisted_data_returns_empty(tmp_path):
    picks = game_picks_backtest.reconstruct_historical_game_picks_from_persisted(
        raw_dir=str(tmp_path / "nonexistent"), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert picks.empty
    assert list(picks.columns) == game_predictions.GAME_PREDICTION_COLUMNS


def test_assemble_game_pick_log_has_expected_schema_and_no_lookahead(tmp_path, monkeypatch):
    # Same stubbing approach as the reconstruct_..._from_persisted tests
    # above - compute_outputs needs live network access, irrelevant to what
    # this test exercises (the per-date replay/merge wiring).
    monkeypatch.setattr(game_picks_backtest.pipeline, "compute_outputs", lambda df: _fake_current_outputs())

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Two days: the first only supplies "prior history" for the second (the
    # first date has nothing strictly before it, so it must be skipped -
    # same no-lookahead guarantee as reconstruct_..._from_persisted).
    rows = _statcast_game(554, pd.Timestamp("2026-05-31"), "NYY", "BOS", 101, 201, 2, 1) + _statcast_game(
        555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2
    )
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    log = game_picks_backtest.assemble_game_pick_log(
        raw_dir=str(raw_dir), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert list(log.columns) == game_picks_backtest.GAME_PICK_LOG_COLUMNS
    assert len(log) == 1  # 05-31 skipped (no prior history), only 06-01 logged
    row = log.iloc[0]
    assert row["game_pk"] == 555
    assert row["home_team"] == "NYY"
    assert row["away_team"] == "BOS"
    # NYY (home) really won 5-2 - the real outcome, not derivable from any
    # of GAME_PICK_FEATURE_COLUMNS (those come from confidence.csv/pave.csv
    # snapshots computed off history strictly before this date, never the
    # game's own result).
    assert row["Home_Won"] == 1
    # home_win_probability must match what compute_game_win_probabilities
    # itself would produce off the exact same (confidence, pave, schedule)
    # inputs - carried through as a tracked comparison column, not
    # recomputed some other way.
    outputs = _fake_current_outputs()
    todays_games = game_picks_backtest.derive_historical_schedule_games(pd.DataFrame(rows))
    todays_games = todays_games[todays_games["date"] == pd.Timestamp("2026-06-01")]
    expected = game_picks_backtest.game_picks.compute_game_win_probabilities(
        outputs["confidence"], outputs["pave"], todays_games
    )
    assert row["home_win_probability"] == pytest.approx(expected.iloc[0]["home_win_probability"])
    for col in game_picks_backtest.game_picks.GAME_PICK_FEATURE_COLUMNS:
        assert col in log.columns


def test_assemble_game_pick_log_populates_bullpen_recent_workload(tmp_path, monkeypatch):
    # Real columns present (unlike _statcast_game's minimal fixture) ->
    # home_bullpen_recent_outs should populate with a real value derived
    # from NYY's own relief appearance the day before.
    monkeypatch.setattr(game_picks_backtest.pipeline, "compute_outputs", lambda df: _fake_current_outputs())

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    day1 = [
        # NYY (home) starter - earliest at_bat_number for NYY in this game.
        {"game_pk": 554, "game_date": pd.Timestamp("2026-05-31"), "home_team": "NYY", "away_team": "BOS",
         "pitcher": 101, "inning_topbot": "Top", "at_bat_number": 1, "events": "field_out",
         "post_home_score": 0, "post_away_score": 0},
        # NYY (home) reliever - later at_bat_number, real strikeout (1 out).
        {"game_pk": 554, "game_date": pd.Timestamp("2026-05-31"), "home_team": "NYY", "away_team": "BOS",
         "pitcher": 102, "inning_topbot": "Top", "at_bat_number": 2, "events": "strikeout",
         "post_home_score": 0, "post_away_score": 0},
        {"game_pk": 554, "game_date": pd.Timestamp("2026-05-31"), "home_team": "NYY", "away_team": "BOS",
         "pitcher": 201, "inning_topbot": "Bot", "at_bat_number": 3, "events": "single",
         "post_home_score": 0, "post_away_score": 0},
        {"game_pk": 554, "game_date": pd.Timestamp("2026-05-31"), "home_team": "NYY", "away_team": "BOS",
         "pitcher": 201, "inning_topbot": "Bot", "at_bat_number": 4, "events": "field_out",
         "post_home_score": 0, "post_away_score": 1},
    ]
    day2 = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
    for row in day2:
        row["events"] = None
    pd.DataFrame(day1 + day2).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    log = game_picks_backtest.assemble_game_pick_log(
        raw_dir=str(raw_dir), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert len(log) == 1
    row = log.iloc[0]
    assert row["home_bullpen_recent_outs"] == 1  # NYY's reliever's real strikeout the day before
    assert pd.isna(row["away_bullpen_recent_outs"])  # BOS had no relief appearance (one pitcher, is_starter)

    # Wider windows (1d/3d/5d) still see the same single day of history -
    # same real value, not zeroed out just because the window changed.
    for suffix in ("1d", "3d", "5d"):
        assert row[f"home_bullpen_recent_outs_{suffix}"] == 1

    # Exactly one distinct reliever used (pitcher 102).
    assert row["home_bullpen_distinct_relievers"] == 1

    # No back-to-back reliever: only ONE prior date (05-31) exists in
    # history at all, so there's no real "day before that" to compare
    # against - a genuine cold-start NaN, not a fabricated 0.
    assert pd.isna(row["home_bullpen_back_to_back_relievers"])


def test_assemble_game_pick_log_populates_back_to_back_relievers_when_real(tmp_path, monkeypatch):
    # NYY's reliever (102) pitches on BOTH 05-30 and 05-31 - the two
    # consecutive dates strictly before the 06-01 target date - a real
    # back-to-back appearance that must populate as a real 1, not NaN.
    monkeypatch.setattr(game_picks_backtest.pipeline, "compute_outputs", lambda df: _fake_current_outputs())

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    def _relief_game(game_pk, date):
        return [
            {"game_pk": game_pk, "game_date": date, "home_team": "NYY", "away_team": "BOS",
             "pitcher": 101, "inning_topbot": "Top", "at_bat_number": 1, "events": "field_out",
             "post_home_score": 0, "post_away_score": 0},
            {"game_pk": game_pk, "game_date": date, "home_team": "NYY", "away_team": "BOS",
             "pitcher": 102, "inning_topbot": "Top", "at_bat_number": 2, "events": "strikeout",
             "post_home_score": 0, "post_away_score": 0},
            {"game_pk": game_pk, "game_date": date, "home_team": "NYY", "away_team": "BOS",
             "pitcher": 201, "inning_topbot": "Bot", "at_bat_number": 3, "events": "field_out",
             "post_home_score": 0, "post_away_score": 0},
        ]

    day_minus_2 = _relief_game(553, pd.Timestamp("2026-05-30"))
    day_minus_1 = _relief_game(554, pd.Timestamp("2026-05-31"))
    day0 = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
    for row in day0:
        row["events"] = None
    pd.DataFrame(day_minus_2 + day_minus_1 + day0).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    log = game_picks_backtest.assemble_game_pick_log(
        raw_dir=str(raw_dir), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert len(log) == 1
    assert log.iloc[0]["home_bullpen_back_to_back_relievers"] == 1


def test_assemble_game_pick_log_home_loss_recorded_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(game_picks_backtest.pipeline, "compute_outputs", lambda df: _fake_current_outputs())

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_game(554, pd.Timestamp("2026-05-31"), "NYY", "BOS", 101, 201, 2, 1) + _statcast_game(
        555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 1, 6
    )
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    log = game_picks_backtest.assemble_game_pick_log(
        raw_dir=str(raw_dir), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert log.iloc[0]["Home_Won"] == 0


def test_assemble_game_pick_log_days_limits_the_replay_window(tmp_path, monkeypatch):
    monkeypatch.setattr(game_picks_backtest.pipeline, "compute_outputs", lambda df: _fake_current_outputs())

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = (
        _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
        + _statcast_game(556, pd.Timestamp("2026-06-02"), "NYY", "BOS", 101, 201, 3, 1)
        + _statcast_game(557, pd.Timestamp("2026-06-03"), "NYY", "BOS", 101, 201, 4, 0)
    )
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    log = game_picks_backtest.assemble_game_pick_log(
        raw_dir=str(raw_dir), season=2026, days=1,
        schedule_backtest_mode="actual_starter",
    )

    assert len(log) == 1
    assert log.iloc[0]["game_pk"] == 557


def test_assemble_game_pick_log_no_persisted_data_returns_empty(tmp_path):
    log = game_picks_backtest.assemble_game_pick_log(
        raw_dir=str(tmp_path / "nonexistent"), season=2026, days=40,
        schedule_backtest_mode="actual_starter",
    )

    assert log.empty
    assert list(log.columns) == game_picks_backtest.GAME_PICK_LOG_COLUMNS
