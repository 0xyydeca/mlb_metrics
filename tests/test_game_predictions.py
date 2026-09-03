import pandas as pd
import pytest

from mlb_metrics import config, game_predictions, kelly


def _win_probabilities(rows):
    """rows: list of dicts with game_pk, date, home_team, away_team, home_win_probability."""
    return pd.DataFrame(rows)


def test_select_game_picks_logs_every_game_and_flags_above_threshold():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},  # clears threshold, home favored
        {"game_pk": 2, "date": pd.Timestamp("2026-07-22"), "home_team": "LAD", "away_team": "SF",
         "home_win_probability": 0.55},  # below threshold - still logged, just flagged False
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"))

    # Every scheduled game is logged now, not just the ones clearing the
    # threshold - the dashboard publishes the complete slate and highlights
    # the flagged ones instead of hiding the rest.
    assert len(picks) == 2
    assert set(picks["game_pk"]) == {1, 2}

    game1 = picks[picks["game_pk"] == 1].iloc[0]
    assert game1["predicted_winner"] == "NYY"
    assert game1["predicted_probability"] == 0.65
    assert game1["above_threshold"] == True  # noqa: E712
    assert game1["metric"] == "GamePick_Win_Probability"
    assert pd.isna(game1["actual_winner"])
    assert pd.isna(game1["game_played"])
    assert game1["model_version"] == config.GAME_PICK_MODEL_VERSION

    game2 = picks[picks["game_pk"] == 2].iloc[0]
    assert game2["above_threshold"] == False  # noqa: E712


def test_select_game_picks_stamps_provenance_fields():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65, "home_probable_pitcher_key_mlbam": 99,
         "away_probable_pitcher_key_mlbam": 88},
    ])
    status = {
        "loaded": True, "fallback_used": False, "fallback_reason": None,
        "artifact_id": "cal123", "training_data_cutoff": "2026-07-01",
        "feature_schema_hash": "hash1",
    }

    picks = game_predictions.select_game_picks(
        win_probs, pd.Timestamp("2026-07-22"),
        model_status=status, probability_source="calibrated_home_win_probability",
    )

    row = picks.iloc[0]
    assert list(picks.columns) == game_predictions.GAME_PREDICTION_COLUMNS
    assert row["selection_metric"] == "predicted_probability"
    assert row["selection_score"] == pytest.approx(0.65)
    assert row["probability_source"] == "calibrated_home_win_probability"
    assert row["model_artifact_id"] == "cal123"
    assert row["prediction_snapshot_type"] == "morning"
    assert row["lineup_status"] == "unconfirmed"
    assert row["starter_status"] == "probable"
    assert row["fallback_used"] == False  # noqa: E712


def test_select_game_picks_records_calibration_fallback():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},
    ])

    picks = game_predictions.select_game_picks(
        win_probs, pd.Timestamp("2026-07-22"),
        fallback_used=True, fallback_reason="missing_artifact",
        probability_source="home_win_probability",
    )

    assert picks.iloc[0]["fallback_used"] == True  # noqa: E712
    assert picks.iloc[0]["fallback_reason"] == "missing_artifact"
    assert picks.iloc[0]["probability_source"] == "home_win_probability"


def test_append_game_predictions_migrates_provenance_on_legacy_csv(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy = pd.DataFrame([{
        "date": "2026-07-19", "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.6, "above_threshold": True,
        "metric": "GamePick_Win_Probability", "actual_winner": "NYY", "game_played": 1,
        "model_version": "legacy",
    }])
    legacy.to_csv(log_path, index=False)

    fresh = game_predictions.select_game_picks(
        _win_probabilities([{
            "game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD", "away_team": "SF",
            "home_win_probability": 0.7,
        }]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(fresh, log_path)

    assert set(game_predictions.PROVENANCE_COLUMNS).issubset(combined.columns)
    legacy_row = combined[combined["game_pk"] == 1].iloc[0]
    assert legacy_row["prediction_code_sha"] == "legacy"
    assert legacy_row["actual_winner"] == "NYY"  # no data loss


def test_append_game_predictions_migrates_a_log_written_before_model_version_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1,
    }])
    legacy_log.to_csv(log_path, index=False)

    new_pick = game_predictions.select_game_picks(
        _win_probabilities([{"game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD",
                              "away_team": "SF", "home_win_probability": 0.65}]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(new_pick, log_path)

    row1 = combined[combined["game_pk"] == 1].iloc[0]
    assert row1["model_version"] == game_predictions.LEGACY_MODEL_VERSION
    row2 = combined[combined["game_pk"] == 2].iloc[0]
    assert row2["model_version"] == config.GAME_PICK_MODEL_VERSION


def test_append_game_predictions_migrates_a_log_written_before_above_threshold_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1, "model_version": "v1",
    }])
    legacy_log.to_csv(log_path, index=False)
    assert "above_threshold" not in legacy_log.columns

    new_pick = game_predictions.select_game_picks(
        _win_probabilities([{"game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD",
                              "away_team": "SF", "home_win_probability": 0.52}]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(new_pick, log_path)

    # The legacy row already cleared the old hard filter by definition
    # (it's a real logged row from before above_threshold existed) - True
    # is the factually correct backfill, not an arbitrary default.
    row1 = combined[combined["game_pk"] == 1].iloc[0]
    assert row1["above_threshold"] == True  # noqa: E712
    row2 = combined[combined["game_pk"] == 2].iloc[0]
    assert row2["above_threshold"] == False  # noqa: E712


def test_select_game_picks_away_favored():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.3},  # away (BOS) favored at .7
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"))

    assert picks.iloc[0]["predicted_winner"] == "BOS"
    assert picks.iloc[0]["predicted_probability"] == 0.7


def test_select_game_picks_still_logs_a_game_that_clears_no_threshold():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.52},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"))

    assert len(picks) == 1
    assert picks.iloc[0]["above_threshold"] == False  # noqa: E712


def test_append_game_predictions_dedupes_keeping_existing_resolved_row(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")

    first = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1,
    }])
    game_predictions.append_game_predictions(first, log_path)

    # Re-logging the same (date, game_pk, metric) with a fresh/unresolved
    # row must not clobber the already-resolved outcome.
    relogged = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    result = game_predictions.append_game_predictions(relogged, log_path)

    assert len(result) == 1
    assert result.iloc[0]["actual_winner"] == "NYY"
    assert result.iloc[0]["game_played"] == 1


def test_resolve_game_predictions_final_game_sets_winner(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 5, "away_score": 3}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert resolved.iloc[0]["game_played"] == 1
    assert resolved.iloc[0]["actual_winner"] == "NYY"


def test_resolve_game_predictions_away_team_wins(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 2, "away_score": 6}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert resolved.iloc[0]["actual_winner"] == "BOS"


def test_resolve_game_predictions_leaves_non_final_games_pending(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Postponed", "home_score": None, "away_score": None}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert pd.isna(resolved.iloc[0]["game_played"])
    assert pd.isna(resolved.iloc[0]["actual_winner"])


def test_resolve_game_predictions_one_bad_date_does_not_block_others(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
            "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
            "actual_winner": pd.NA, "game_played": pd.NA,
        },
        {
            "date": pd.Timestamp("2026-07-20"), "game_pk": 2, "home_team": "LAD", "away_team": "SF",
            "predicted_winner": "LAD", "predicted_probability": 0.6, "metric": "GamePick_Win_Probability",
            "actual_winner": pd.NA, "game_played": pd.NA,
        },
    ])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        if date == pd.Timestamp("2026-07-19").date():
            raise RuntimeError("statsapi is down for this date")
        return pd.DataFrame([{"game_pk": 2, "status": "Final", "home_score": 4, "away_score": 1}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    row1 = resolved[resolved["game_pk"] == 1].iloc[0]
    row2 = resolved[resolved["game_pk"] == 2].iloc[0]
    assert pd.isna(row1["game_played"])  # left pending, no crash
    assert row2["game_played"] == 1
    assert row2["actual_winner"] == "LAD"


def test_resolve_game_predictions_migrates_a_log_written_before_model_version_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    legacy_log.to_csv(log_path, index=False)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 5, "away_score": 3}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert resolved.iloc[0]["model_version"] == game_predictions.LEGACY_MODEL_VERSION


def _market_probabilities(rows):
    """rows: list of dicts with home_team, away_team, market_home_win_probability."""
    return pd.DataFrame(rows)


def test_select_game_picks_merges_a_matching_market_probability():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},
    ])
    market = _market_probabilities([
        {"home_team": "NYY", "away_team": "BOS", "market_home_win_probability": 0.62, "market_provider": "DraftKings"},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"), market_probabilities=market)

    assert picks.iloc[0]["market_home_win_probability"] == pytest.approx(0.62)


def test_select_game_picks_market_probability_is_nan_when_no_match():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},
    ])
    # Market data exists for a different game entirely - a real "ESPN
    # doesn't have this matchup" case, not a crash.
    market = _market_probabilities([
        {"home_team": "LAD", "away_team": "SF", "market_home_win_probability": 0.55, "market_provider": "DraftKings"},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"), market_probabilities=market)

    assert pd.isna(picks.iloc[0]["market_home_win_probability"])


def test_select_game_picks_market_probability_is_nan_when_none_given():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"))

    assert "market_home_win_probability" in picks.columns
    assert pd.isna(picks.iloc[0]["market_home_win_probability"])


def test_append_game_predictions_migrates_a_log_written_before_market_column_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1, "model_version": "v1", "above_threshold": True,
    }])
    legacy_log.to_csv(log_path, index=False)
    assert "market_home_win_probability" not in legacy_log.columns

    new_pick = game_predictions.select_game_picks(
        _win_probabilities([{"game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD",
                              "away_team": "SF", "home_win_probability": 0.65}]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(new_pick, log_path)

    row1 = combined[combined["game_pk"] == 1].iloc[0]
    assert pd.isna(row1["market_home_win_probability"])


def test_resolve_game_predictions_migrates_a_log_written_before_market_column_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    legacy_log.to_csv(log_path, index=False)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 5, "away_score": 3}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert pd.isna(resolved.iloc[0]["market_home_win_probability"])


# ---------------------------------------------------------------------
# advise_bets - moved here (unchanged logic) from scripts/recommend_bets.py's
# former build_bet_recommendations, since pipeline.run() now shares this
# same real decision via select_game_picks below, not just the standalone
# report script.
# ---------------------------------------------------------------------

def _bet_pick_row(game_pk, home_team, away_team, predicted_winner, predicted_probability):
    return {
        "date": pd.Timestamp("2026-08-24"), "game_pk": game_pk, "home_team": home_team,
        "away_team": away_team, "predicted_winner": predicted_winner,
        "predicted_probability": predicted_probability,
    }


def _bet_market_row(home_team, away_team, home_moneyline, away_moneyline):
    return {
        "home_team": home_team, "away_team": away_team,
        "market_home_win_probability": None, "market_provider": "DraftKings",
        "home_moneyline": home_moneyline, "away_moneyline": away_moneyline,
    }


def test_advise_bets_finds_a_real_positive_home_edge():
    picks = pd.DataFrame([_bet_pick_row(1, "NYY", "TOR", "NYY", 0.70)])
    # home implied = 150/250 = 0.6 -> edge = 0.70 - 0.6 = 0.10
    # away model prob = 0.30, away implied = 100/230 = 0.4348 -> edge < 0
    market = pd.DataFrame([_bet_market_row("NYY", "TOR", -150, 130)])

    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert len(recs) == 1
    row = recs.iloc[0]
    assert row["side"] == "home"
    assert row["team"] == "NYY"
    assert row["edge"] == pytest.approx(0.10, abs=1e-6)
    assert row["kelly_stake_fraction"] > 0


def test_advise_bets_below_min_edge_recommends_nothing():
    picks = pd.DataFrame([_bet_pick_row(1, "NYY", "TOR", "NYY", 0.52)])
    market = pd.DataFrame([_bet_market_row("NYY", "TOR", -115, 105)])

    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert recs.empty


def test_advise_bets_skips_a_game_missing_from_market():
    picks = pd.DataFrame([
        _bet_pick_row(1, "NYY", "TOR", "NYY", 0.70),
        _bet_pick_row(2, "LAD", "SF", "LAD", 0.65),
    ])
    # Only game_pk=1's matchup has real market data.
    market = pd.DataFrame([_bet_market_row("NYY", "TOR", -150, 130)])

    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert list(recs["game_pk"]) == [1]


def test_advise_bets_drops_both_sides_on_a_data_anomaly(capsys):
    # An unrealistic (negative-vig) market row - both implied probabilities
    # sum to < 1, so both sides can clear min_edge at once. Real data never
    # does this (see market_odds.devig's own test asserting real vig > 0)
    # - this exercises the defensive guard, not a real scenario.
    picks = pd.DataFrame([_bet_pick_row(1, "NYY", "TOR", "NYY", 0.50)])
    market = pd.DataFrame([_bet_market_row("NYY", "TOR", 120, 120)])

    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert recs.empty
    assert "data-quality anomaly" in capsys.readouterr().out


def test_advise_bets_sizes_off_pessimistic_probability_when_given():
    # Real follow-up (2026-08-25 - "we need the units risked to not be
    # arbitrary"): when home/away_win_probability_pessimistic columns are
    # present, the stake is sized off THAT real, per-game conservative
    # probability - not the raw model probability - but STILL scaled by
    # kelly_fraction_multiplier (2026-08-26 - "the short-priced-favorite
    # blowup risk": the CI-based pessimism and the flat multiplier are two
    # independent, stacked layers of conservatism, not a swap of one for
    # the other).
    picks = pd.DataFrame([{
        **_bet_pick_row(1, "NYY", "TOR", "NYY", 0.70),
        "home_win_probability_pessimistic": 0.61,
        "away_win_probability_pessimistic": 0.05,
    }])
    market = pd.DataFrame([_bet_market_row("NYY", "TOR", -150, 130)])

    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=0.5, min_edge=0.02)

    assert len(recs) == 1
    row = recs.iloc[0]
    assert row["side"] == "home"
    # Eligibility (min_edge) still uses the RAW model probability (0.70) -
    # a separate question from how much to actually risk.
    assert row["edge"] == pytest.approx(0.70 - (150 / 250), abs=1e-6)
    expected_stake = kelly.kelly_fraction(0.61, -150, 0.5)
    assert expected_stake > 0  # sanity - this test must exercise a real positive stake
    assert row["kelly_stake_fraction"] == pytest.approx(expected_stake, abs=1e-9)
    # Genuinely smaller than sizing at a full 1.0 multiplier off the same
    # pessimistic probability - the multiplier still does real work even
    # once the CI-based pessimism has already been applied.
    full_kelly_off_pessimistic = kelly.kelly_fraction(0.61, -150, 1.0)
    assert row["kelly_stake_fraction"] < full_kelly_off_pessimistic
    # And genuinely smaller than the old flat-half-Kelly approach on the
    # same raw 0.70 probability too - the CI-based pessimism still matters
    # on top of the multiplier, not replaced by it.
    old_flat_stake = kelly.kelly_fraction(0.70, -150, 0.5)
    assert row["kelly_stake_fraction"] < old_flat_stake


def test_advise_bets_clips_a_single_bet_to_the_max_single_bet_cap():
    # Real follow-up (2026-08-26 - "maybe we scaled it down too much...
    # this is now taking out games where the line is genuinely appealing"):
    # config.KELLY_MAX_SINGLE_BET_UNIT_CAP bounds any ONE game's stake
    # directly, regardless of how large Kelly's own number is - a huge raw
    # edge here (full Kelly would be 50% of bankroll) still clips to the
    # cap, well under config.KELLY_DAILY_UNIT_CAP so this isolates the
    # per-bet cap from the separate daily cap.
    picks = pd.DataFrame([_bet_pick_row(1, "NYY", "TOR", "NYY", 0.80)])
    market = pd.DataFrame([_bet_market_row("NYY", "TOR", -150, 130)])

    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert len(recs) == 1
    uncapped = kelly.kelly_fraction(0.80, -150, 1.0)
    single_bet_cap = config.KELLY_MAX_SINGLE_BET_UNIT_CAP * config.UNIT_SIZE_FRACTION
    assert uncapped > single_bet_cap  # sanity - the cap is genuinely binding here
    assert single_bet_cap < config.KELLY_DAILY_UNIT_CAP * config.UNIT_SIZE_FRACTION
    assert recs.iloc[0]["kelly_stake_fraction"] == pytest.approx(single_bet_cap, abs=1e-9)


def test_advise_bets_daily_cap_layers_on_top_of_the_single_bet_cap():
    # Three real edges on the SAME date: two huge ones (full Kelly would be
    # 25%+ of bankroll each) that each clip to config.KELLY_MAX_SINGLE_BET_UNIT_CAP
    # first, plus one modest one that stays under that per-bet cap on its
    # own. Their SUM after the per-bet cap still exceeds
    # config.KELLY_DAILY_UNIT_CAP, so the daily cap then scales ALL THREE
    # down further, proportionally, on top of the per-bet cap - both real
    # risk limits apply together, in the documented order, and the two
    # already-capped bets aren't exempted from the daily cap just because
    # they were already reduced once.
    picks = pd.DataFrame([
        _bet_pick_row(1, "NYY", "TOR", "NYY", 0.80),
        _bet_pick_row(2, "LAD", "SF", "LAD", 0.80),
        _bet_pick_row(3, "HOU", "SEA", "HOU", 0.346),
    ])
    market = pd.DataFrame([
        _bet_market_row("NYY", "TOR", -150, 130),
        _bet_market_row("LAD", "SF", -140, 120),
        # HOU a live underdog at +200 - a real, modestly-appealing edge
        # (~1.3 points above the 33.3% breakeven) that stays under the
        # per-bet cap on its own, unlike NYY/LAD's huge favorite edges.
        _bet_market_row("HOU", "SEA", 200, -260),
    ])

    # min_edge=0.01 (below the module-default 0.05) so HOU's modest,
    # deliberately-small edge still clears eligibility - this test is
    # about cap layering, not min_edge itself.
    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.01)

    assert len(recs) == 3
    single_bet_cap = config.KELLY_MAX_SINGLE_BET_UNIT_CAP * config.UNIT_SIZE_FRACTION
    daily_cap = config.KELLY_DAILY_UNIT_CAP * config.UNIT_SIZE_FRACTION
    hou_pre_daily_cap = kelly.kelly_fraction(0.346, 200, 1.0)
    assert hou_pre_daily_cap < single_bet_cap  # sanity - HOU isn't hitting the per-bet cap on its own
    pre_daily_total = 2 * single_bet_cap + hou_pre_daily_cap
    assert pre_daily_total > daily_cap  # sanity - the daily cap is genuinely binding on top of the per-bet cap

    # The day's total lands EXACTLY at the daily cap.
    assert recs["kelly_stake_fraction"].sum() == pytest.approx(daily_cap, abs=1e-9)
    # NYY/LAD were identically capped pre-daily-cap (both hit the same
    # per-bet cap), so they land identically post-daily-cap too.
    by_team = recs.set_index("team")
    assert by_team.loc["NYY", "kelly_stake_fraction"] == pytest.approx(
        by_team.loc["LAD", "kelly_stake_fraction"], abs=1e-9
    )
    # HOU's smaller pre-daily-cap stake is scaled down by the same real
    # ratio as NYY/LAD - proportional, not an equal three-way split.
    scale = daily_cap / pre_daily_total
    assert by_team.loc["NYY", "kelly_stake_fraction"] == pytest.approx(single_bet_cap * scale, abs=1e-9)
    assert by_team.loc["HOU", "kelly_stake_fraction"] == pytest.approx(hou_pre_daily_cap * scale, abs=1e-9)


def test_advise_bets_daily_cap_leaves_a_different_dates_stakes_untouched():
    # Three dates, each individually under BOTH caps on its own, but
    # summed together they WOULD exceed config.KELLY_DAILY_UNIT_CAP if the
    # cap were (wrongly) applied across the whole DataFrame instead of per
    # real calendar date.
    # +200 underdog picks (same modest, real edge used in the layering
    # test above) - a real edge that stays under both caps individually.
    picks = pd.DataFrame([
        {**_bet_pick_row(1, "NYY", "TOR", "NYY", 0.346), "date": pd.Timestamp("2026-08-24")},
        {**_bet_pick_row(2, "LAD", "SF", "LAD", 0.346), "date": pd.Timestamp("2026-08-25")},
        {**_bet_pick_row(3, "HOU", "SEA", "HOU", 0.346), "date": pd.Timestamp("2026-08-26")},
    ])
    market = pd.DataFrame([
        _bet_market_row("NYY", "TOR", 200, -260),
        _bet_market_row("LAD", "SF", 200, -260),
        _bet_market_row("HOU", "SEA", 200, -260),
    ])

    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.01)

    expected_each = kelly.kelly_fraction(0.346, 200, 1.0)
    single_bet_cap = config.KELLY_MAX_SINGLE_BET_UNIT_CAP * config.UNIT_SIZE_FRACTION
    daily_cap = config.KELLY_DAILY_UNIT_CAP * config.UNIT_SIZE_FRACTION
    assert expected_each < single_bet_cap
    assert expected_each < daily_cap
    assert expected_each * 3 > daily_cap  # sanity - would exceed the cap if wrongly summed across dates
    assert (recs["kelly_stake_fraction"] - expected_each).abs().max() < 1e-9


# ---------------------------------------------------------------------
# select_game_picks - bet_* columns
# ---------------------------------------------------------------------

def _full_market(rows):
    """rows: list of dicts with home_team, away_team, home_moneyline, away_moneyline."""
    return pd.DataFrame([
        {"market_home_win_probability": None, "market_provider": "DraftKings", **r} for r in rows
    ])


def test_select_game_picks_logs_a_real_advised_bet():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-08-24"), "home_team": "NYY", "away_team": "TOR",
         "home_win_probability": 0.70},
    ])
    market = _full_market([{"home_team": "NYY", "away_team": "TOR", "home_moneyline": -150, "away_moneyline": 130}])

    picks = game_predictions.select_game_picks(
        win_probs, pd.Timestamp("2026-08-24"), market_probabilities=market, kelly_fraction_multiplier=1.0,
    )

    row = picks.iloc[0]
    assert row["bet_units"] > 0  # bet_units IS the "was a bet advised" signal - 0 means no bet
    assert row["bet_side"] == "home"
    assert row["bet_team"] == "NYY"
    assert row["bet_moneyline"] == -150
    assert row["bet_stake_fraction"] > 0
    assert row["bet_units"] == pytest.approx(row["bet_stake_fraction"] / config.UNIT_SIZE_FRACTION)
    assert pd.isna(row["bet_profit_units"])  # not resolved yet


def test_select_game_picks_uses_confidence_for_a_smaller_pessimistic_stake():
    # Real follow-up (2026-08-25 - "we need the units risked to not be
    # arbitrary"): passing `confidence` (teams.assemble_team_metrics' real
    # output, carrying each team's real win_rate_CI_Low/CI_High) makes
    # select_game_picks size the stake off a real per-game conservative
    # probability, on top of the same kelly_fraction_multiplier (a small
    # 0.05 in both calls here so neither stake hits
    # config.KELLY_MAX_SINGLE_BET_UNIT_CAP - this isolates the CI-based
    # pessimism's own effect from that separate cap) - a smaller, more
    # honest stake on the exact same real edge.
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-08-24"), "home_team": "NYY", "away_team": "TOR",
         "home_win_probability": 0.70},
    ])
    market = _full_market([{"home_team": "NYY", "away_team": "TOR", "home_moneyline": -150, "away_moneyline": 130}])
    confidence = pd.DataFrame([
        {"team": "NYY", "win_rate_CI_Low": 0.62, "win_rate_CI_High": 0.78},
        {"team": "TOR", "win_rate_CI_Low": 0.46, "win_rate_CI_High": 0.54},
    ])

    with_confidence = game_predictions.select_game_picks(
        win_probs, pd.Timestamp("2026-08-24"), market_probabilities=market,
        kelly_fraction_multiplier=0.05, confidence=confidence,
    )
    without_confidence = game_predictions.select_game_picks(
        win_probs, pd.Timestamp("2026-08-24"), market_probabilities=market, kelly_fraction_multiplier=0.05,
    )

    # Same real edge clears bet eligibility either way (min_edge uses the
    # raw model probability, unaffected by `confidence`)...
    assert with_confidence.iloc[0]["bet_units"] > 0
    assert without_confidence.iloc[0]["bet_units"] > 0
    # ...but the real per-game CI-derived stake is genuinely smaller than
    # the flat full-Kelly stake on the same raw probability/edge.
    assert with_confidence.iloc[0]["bet_stake_fraction"] < without_confidence.iloc[0]["bet_stake_fraction"]

    # Persisted onto the logged row (not just used transiently) - so
    # scripts/recommend_bets.py's later re-derivation of advise_bets from
    # THIS log reuses the exact same real sizing.
    assert with_confidence.iloc[0]["home_win_probability_pessimistic"] == pytest.approx(0.70 - 0.089442, abs=1e-4)
    assert pd.notna(with_confidence.iloc[0]["away_win_probability_pessimistic"])
    # Without `confidence`, the columns still EXIST (a stable schema for the
    # log/report), just null for that game - not simply absent.
    assert "home_win_probability_pessimistic" in without_confidence.columns
    assert pd.isna(without_confidence.iloc[0]["home_win_probability_pessimistic"])
    assert pd.isna(without_confidence.iloc[0]["away_win_probability_pessimistic"])


def test_advise_bets_falls_back_to_flat_multiplier_when_pessimistic_columns_are_present_but_null():
    # Regression guard: select_game_picks always persists
    # home/away_win_probability_pessimistic (so scripts/recommend_bets.py's
    # later re-derivation from the log can reuse them), but leaves them null
    # for a game `confidence` didn't cover. advise_bets must fall back to
    # the flat kelly_fraction_multiplier for THAT row, not silently compute
    # a NaN stake just because the columns happen to exist.
    picks = pd.DataFrame([{
        **_bet_pick_row(1, "NYY", "TOR", "NYY", 0.70),
        "home_win_probability_pessimistic": pd.NA,
        "away_win_probability_pessimistic": pd.NA,
    }])
    market = pd.DataFrame([_bet_market_row("NYY", "TOR", -150, 130)])

    # A small multiplier so the resulting stake stays comfortably under
    # BOTH config.KELLY_MAX_SINGLE_BET_UNIT_CAP and config.KELLY_DAILY_UNIT_CAP
    # - this test is about the pessimistic/flat fallback choice, not either
    # cap (covered elsewhere).
    recs = game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=0.05, min_edge=0.02)

    assert len(recs) == 1
    expected_stake = kelly.kelly_fraction(0.70, -150, 0.05)
    single_bet_cap = config.KELLY_MAX_SINGLE_BET_UNIT_CAP * config.UNIT_SIZE_FRACTION
    daily_cap = config.KELLY_DAILY_UNIT_CAP * config.UNIT_SIZE_FRACTION
    assert expected_stake < single_bet_cap  # sanity - this test isn't exercising either cap
    assert expected_stake < daily_cap
    assert recs.iloc[0]["kelly_stake_fraction"] == pytest.approx(expected_stake, abs=1e-9)


def test_select_game_picks_no_bet_advised_when_no_real_edge():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-08-24"), "home_team": "NYY", "away_team": "TOR",
         "home_win_probability": 0.52},
    ])
    market = _full_market([{"home_team": "NYY", "away_team": "TOR", "home_moneyline": -115, "away_moneyline": 105}])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-08-24"), market_probabilities=market)

    row = picks.iloc[0]
    assert row["bet_units"] == 0.0
    assert pd.isna(row["bet_side"])


def test_select_game_picks_no_bet_columns_without_moneylines_backward_compat():
    # game_picks_backtest.py and older tests pass either no market_probabilities
    # at all, or the original 3-column de-vigged-only frame - bet_units must
    # stay 0.0, not KeyError.
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-08-24"), "home_team": "NYY", "away_team": "TOR",
         "home_win_probability": 0.70},
    ])
    market = pd.DataFrame([
        {"home_team": "NYY", "away_team": "TOR", "market_home_win_probability": 0.6},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-08-24"), market_probabilities=market)

    row = picks.iloc[0]
    assert row["bet_units"] == 0.0

    picks_no_market = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-08-24"))
    assert picks_no_market.iloc[0]["bet_units"] == 0.0


def test_append_game_predictions_migrates_a_log_written_before_bet_columns_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1, "model_version": "v1", "above_threshold": True,
        "market_home_win_probability": pd.NA,
    }])
    legacy_log.to_csv(log_path, index=False)
    assert "bet_units" not in legacy_log.columns

    new_pick = game_predictions.select_game_picks(
        _win_probabilities([{"game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD",
                              "away_team": "SF", "home_win_probability": 0.65}]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(new_pick, log_path)

    row1 = combined[combined["game_pk"] == 1].iloc[0]
    assert row1["bet_units"] == 0.0
    assert pd.isna(row1["bet_profit_units"])


def test_append_game_predictions_migrates_a_log_written_before_pessimistic_columns_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1, "model_version": "v1", "above_threshold": True,
        "market_home_win_probability": pd.NA, "bet_units": 0.0, "bet_side": pd.NA, "bet_team": pd.NA,
        "bet_moneyline": pd.NA, "bet_stake_fraction": pd.NA, "bet_profit_units": pd.NA,
    }])
    legacy_log.to_csv(log_path, index=False)
    assert "home_win_probability_pessimistic" not in legacy_log.columns

    new_pick = game_predictions.select_game_picks(
        _win_probabilities([{"game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD",
                              "away_team": "SF", "home_win_probability": 0.65}]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(new_pick, log_path)

    row1 = combined[combined["game_pk"] == 1].iloc[0]
    assert pd.isna(row1["home_win_probability_pessimistic"])
    assert pd.isna(row1["away_win_probability_pessimistic"])


# ---------------------------------------------------------------------
# resolve_game_predictions - real units won/lost
# ---------------------------------------------------------------------

def _advised_pick_row(game_pk, home_team, away_team, bet_side, bet_team, bet_moneyline, bet_units):
    return {
        "date": pd.Timestamp("2026-07-20"), "game_pk": game_pk, "home_team": home_team, "away_team": away_team,
        "predicted_winner": home_team, "predicted_probability": 0.6, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
        "bet_units": bet_units, "bet_side": bet_side, "bet_team": bet_team, "bet_moneyline": bet_moneyline,
        "bet_stake_fraction": bet_units * config.UNIT_SIZE_FRACTION, "bet_profit_units": pd.NA,
    }


def test_resolve_game_predictions_computes_real_profit_on_a_winning_advised_bet(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([_advised_pick_row(1, "NYY", "BOS", "home", "NYY", -150, 3.0)])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 5, "away_score": 3}])  # NYY (home) won

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    # net odds b = 100/150 -> profit = 3 * (100/150) = 2.0
    assert resolved.iloc[0]["bet_profit_units"] == pytest.approx(3 * (100 / 150))


def test_resolve_game_predictions_computes_real_loss_on_a_losing_advised_bet(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([_advised_pick_row(1, "NYY", "BOS", "home", "NYY", -150, 3.0)])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 2, "away_score": 6}])  # BOS (away) won

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert resolved.iloc[0]["bet_profit_units"] == pytest.approx(-3.0)


def test_resolve_game_predictions_leaves_profit_null_for_a_non_advised_resolved_game(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    # A real select_game_picks row always has all 6 bet_* columns together
    # (bet_units=0.0, rest NaN when not advised) - matches that real shape,
    # not a hand-trimmed one.
    picks = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA, "bet_units": 0.0,
        "bet_side": pd.NA, "bet_team": pd.NA, "bet_moneyline": pd.NA,
        "bet_stake_fraction": pd.NA, "bet_profit_units": pd.NA,
    }])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 5, "away_score": 3}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert pd.isna(resolved.iloc[0]["bet_profit_units"])
