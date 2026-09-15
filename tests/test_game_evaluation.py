import math

import pandas as pd
import pytest

from mlb_metrics import evaluation, game_evaluation


def _pick(date, game_pk, home_team, away_team, predicted_winner, predicted_probability, actual_winner, game_played,
          metric="GamePick_Win_Probability", market_home_win_probability=None,
          bet_units=0.0, bet_side=None, bet_team=None, bet_moneyline=None, bet_profit_units=None):
    return {
        "date": date, "game_pk": game_pk, "home_team": home_team, "away_team": away_team,
        "predicted_winner": predicted_winner, "predicted_probability": predicted_probability,
        "metric": metric, "actual_winner": actual_winner, "game_played": game_played,
        "market_home_win_probability": market_home_win_probability,
        "bet_units": bet_units, "bet_side": bet_side, "bet_team": bet_team, "bet_moneyline": bet_moneyline,
        "bet_profit_units": bet_profit_units,
    }


def _predictions():
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.65, "NYY", 1),  # win
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.6, "SF", 1),  # loss
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.7, "HOU", 1),  # win
        _pick("2026-07-21", 4, "ATL", "PHI", "ATL", 0.62, None, None),  # pending
        _pick("2026-07-17", 5, "TB", "TOR", "TB", 0.6, None, 0),  # not_played (postponed)
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_classify_outcome():
    df = _predictions()
    outcome = game_evaluation._classify_outcome(df)
    assert list(outcome) == ["win", "loss", "win", "pending", "not_played"]


def test_build_game_picks_export_picks_table_status_and_order():
    picks, summary = game_evaluation.build_game_picks_export(_predictions())

    # Most recent date first.
    assert picks["date"].iloc[0] == picks["date"].max()

    by_pk = picks.set_index("game_pk")
    assert by_pk.loc[1, "status"] == "win"
    assert by_pk.loc[2, "status"] == "loss"
    assert by_pk.loc[4, "status"] == "pending"
    assert by_pk.loc[5, "status"] == "not_played"


def test_build_game_picks_export_predicted_loser_and_market_predicted_winner_probability():
    # Real follow-up (2026-08-28 - the History table was "dumping almost
    # everything... jumbled," trimmed to a curated column set including
    # two new DERIVED ones): predicted_loser is whichever of home/away
    # ISN'T predicted_winner, and market_predicted_winner_probability is
    # the market's real probability for the SAME side the model favored -
    # de-vigged market_home_win_probability is always the home team's own
    # probability, so it needs flipping (1 - p) whenever the model favors
    # the AWAY team, to stay a real apples-to-apples "model vs. market"
    # comparison instead of silently comparing the wrong side.
    rows = [
        # Home team (NYY) favored - no flip needed, market_predicted_winner_probability
        # should equal the raw market_home_win_probability directly.
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.65, "NYY", 1, market_home_win_probability=0.60),
        # Away team (SF) favored - market_predicted_winner_probability
        # must be the FLIPPED value (1 - 0.40 = 0.60), not the raw
        # home-team number (0.40), which would be the WRONG side.
        _pick("2026-07-19", 2, "LAD", "SF", "SF", 0.55, "SF", 1, market_home_win_probability=0.40),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df)
    by_pk = picks.set_index("game_pk")

    assert by_pk.loc[1, "predicted_loser"] == "BOS"
    assert by_pk.loc[1, "market_predicted_winner_probability"] == pytest.approx(0.60)

    assert by_pk.loc[2, "predicted_loser"] == "LAD"
    assert by_pk.loc[2, "market_predicted_winner_probability"] == pytest.approx(0.60)


def test_build_game_picks_export_min_probability_flags_but_does_not_drop():
    preds = _predictions()
    picks, summary = game_evaluation.build_game_picks_export(preds, min_probability=0.63)

    # Every game is still published, not just the ones clearing .63 - only
    # 0.65 (pk 1) and 0.7 (pk 3) are flagged above_threshold. above_threshold
    # is still published for the dashboard, it just no longer drives the
    # headline scoring below (bet_units does - see the bet-P&L tests).
    assert set(picks["game_pk"]) == {1, 2, 3, 4, 5}
    by_pk = picks.set_index("game_pk")
    assert by_pk.loc[1, "above_threshold"] == True  # noqa: E712
    assert by_pk.loc[3, "above_threshold"] == True  # noqa: E712
    assert by_pk.loc[2, "above_threshold"] == False  # noqa: E712
    assert by_pk.loc[4, "above_threshold"] == False  # noqa: E712
    assert by_pk.loc[5, "above_threshold"] == False  # noqa: E712


def test_build_game_picks_export_only_bet_advised_games_affect_pnl_scoring():
    # A real win on a game where NO bet was advised must not move the P&L
    # numbers at all - above_threshold is irrelevant here too; only
    # bet_units > 0 (and a real resolved bet_profit_units) counts.
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.90, "NYY", 1,
              bet_units=3.0, bet_side="home", bet_team="NYY", bet_moneyline=-150,
              bet_profit_units=3 * (100 / 150)),  # advised, real win
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.50, "SF", 1),  # NOT advised, real loss - must not count
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.85, "HOU", 1),  # NOT advised, real win - must not count
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert set(picks["game_pk"]) == {1, 2, 3}  # published regardless
    assert summary.loc[0, "n_bets_advised"] == 1
    assert summary.loc[0, "bets_won"] == 1
    assert summary.loc[0, "win_rate_on_advised_bets"] == pytest.approx(1.0)
    assert summary.loc[0, "total_profit_units"] == pytest.approx(3 * (100 / 150))


def test_build_game_picks_export_ignores_other_metrics():
    preds = _predictions()
    other = preds.copy()
    other["metric"] = "SomeOtherFormula"
    other["game_pk"] = other["game_pk"] + 100
    combined = pd.concat([preds, other], ignore_index=True)

    picks, summary = game_evaluation.build_game_picks_export(combined, metric="GamePick_Win_Probability")

    assert set(picks["game_pk"]) == {1, 2, 3, 4, 5}


def test_build_game_picks_export_model_version_filters_and_labels_summary():
    preds = _predictions()
    preds["model_version"] = "v1"
    v2_row = pd.DataFrame([_pick("2026-07-22", 6, "SD", "COL", "SD", 0.66, "SD", 1)])
    v2_row["date"] = pd.to_datetime(v2_row["date"])
    v2_row["model_version"] = "v2"
    combined = pd.concat([preds, v2_row], ignore_index=True)

    all_time_picks, all_time_summary = game_evaluation.build_game_picks_export(combined)
    assert all_time_summary.loc[0, "model_version"] == "all_time"
    assert set(all_time_picks["game_pk"]) == {1, 2, 3, 4, 5, 6}

    v1_picks, v1_summary = game_evaluation.build_game_picks_export(combined, model_version="v1")
    assert v1_summary.loc[0, "model_version"] == "v1"
    assert 6 not in set(v1_picks["game_pk"])


def test_build_game_picks_export_no_resolved_games_yet():
    preds = pd.DataFrame([_pick("2026-07-21", 1, "ATL", "PHI", "ATL", 0.62, None, None)])
    preds["date"] = pd.to_datetime(preds["date"])

    picks, summary = game_evaluation.build_game_picks_export(preds)

    assert summary.loc[0, "n_bets_advised"] == 0
    assert pd.isna(summary.loc[0, "win_rate_on_advised_bets"])
    assert summary.loc[0, "current_bet_streak"] == 0
    assert summary.loc[0, "best_bet_streak"] == 0


def test_build_game_picks_export_bet_pnl_metrics_real_win_and_loss():
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.65, "NYY", 1,
              bet_units=3.0, bet_side="home", bet_team="NYY", bet_moneyline=-150,
              bet_profit_units=3 * (100 / 150)),  # win
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.6, "SF", 1,
              bet_units=1.5, bet_side="home", bet_team="LAD", bet_moneyline=-120,
              bet_profit_units=-1.5),  # loss
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.7, "HOU", 1),  # never advised - excluded entirely
        _pick("2026-07-21", 4, "ATL", "PHI", "ATL", 0.62, None, None,
              bet_units=2.25, bet_side="home", bet_team="ATL", bet_moneyline=-130,
              bet_profit_units=None),  # advised but still pending - excluded
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_bets_advised"] == 2
    assert summary.loc[0, "bets_won"] == 1
    assert summary.loc[0, "bets_lost"] == 1
    assert summary.loc[0, "win_rate_on_advised_bets"] == pytest.approx(0.5)
    assert summary.loc[0, "total_staked_units"] == pytest.approx(4.5)
    expected_profit = 3 * (100 / 150) - 1.5
    assert summary.loc[0, "total_profit_units"] == pytest.approx(expected_profit)
    assert summary.loc[0, "roi"] == pytest.approx(expected_profit / 4.5)

    # Quant-analytics item #5: win_rate's real Wilson CI (1 win of 2
    # advised bets) - informational only, not a significance test (see
    # evaluation.mean_significance's own docstring for why win-rate alone
    # isn't the right null-hypothesis test for a bets-of-varying-price
    # strategy). roi_p_value IS the real, correctly-posed test: a
    # one-sample t-test on the two real bet_profit_units values
    # (3*(100/150), -1.5) against a null of 0 (breaking even).
    expected_wr_low, expected_wr_high = evaluation.wilson_confidence_interval(1, 2)
    assert summary.loc[0, "win_rate_on_advised_bets_ci_low"] == pytest.approx(expected_wr_low)
    assert summary.loc[0, "win_rate_on_advised_bets_ci_high"] == pytest.approx(expected_wr_high)
    expected_roi_p = evaluation.mean_significance(pd.Series([3 * (100 / 150), -1.5]), null_value=0.0)
    assert summary.loc[0, "roi_p_value"] == pytest.approx(expected_roi_p)


def test_build_game_picks_export_bet_streak_is_a_day_streak_not_a_bet_streak():
    # Chronological order by DAY: profitable day, losing day, profitable
    # day -> current streak resets on the losing day then rebuilds to 1 on
    # the next profitable day - streak is scored per calendar day on that
    # day's TOTAL real profit, not per individual bet (2026-08-25 - "the
    # streak should be days... if the cumulative bets made money that day").
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.65, "NYY", 1,
              bet_units=3.0, bet_team="NYY", bet_moneyline=-150, bet_profit_units=2.0),
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.6, "SF", 1,
              bet_units=1.5, bet_team="LAD", bet_moneyline=-120, bet_profit_units=-1.5),
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.7, "HOU", 1,
              bet_units=2.25, bet_team="HOU", bet_moneyline=-130, bet_profit_units=1.73),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "current_bet_streak"] == 1
    assert summary.loc[0, "best_bet_streak"] == 1


def test_build_game_picks_export_bet_streak_scores_a_day_by_its_total_profit():
    # Two bets on the SAME day: a big winner and a smaller loser that
    # nets the day positive overall - counts as exactly ONE streak step,
    # not two, and is scored on the day's combined profit (net positive),
    # not on either individual bet.
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.65, "NYY", 1,
              bet_units=3.0, bet_team="NYY", bet_moneyline=-150, bet_profit_units=2.0),
        _pick("2026-07-18", 2, "LAD", "SF", "LAD", 0.6, "SF", 1,
              bet_units=1.0, bet_team="LAD", bet_moneyline=-120, bet_profit_units=-0.5),
        # A second day, net negative overall despite one winning bet on it
        # - resets the streak, not just leaves it unchanged.
        _pick("2026-07-19", 3, "HOU", "SEA", "HOU", 0.7, "HOU", 1,
              bet_units=1.0, bet_team="HOU", bet_moneyline=-130, bet_profit_units=0.5),
        _pick("2026-07-19", 4, "ATL", "PHI", "ATL", 0.66, None, 1,
              bet_units=2.0, bet_team="ATL", bet_moneyline=-140, bet_profit_units=-1.5),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df)

    # Day 1 (net +1.5) extends the streak to 1; day 2 (net -1.0) resets it.
    assert summary.loc[0, "current_bet_streak"] == 0
    assert summary.loc[0, "best_bet_streak"] == 1


def test_build_game_picks_export_bet_columns_missing_migration():
    # A log written before bet advice existed at all (not just null values)
    # must not crash - same migration convention as market_home_win_probability's
    # own missing-column case.
    rows = [
        {"date": "2026-07-18", "game_pk": 1, "home_team": "NYY", "away_team": "BOS", "predicted_winner": "NYY",
         "predicted_probability": 0.65, "metric": "GamePick_Win_Probability", "actual_winner": "NYY", "game_played": 1},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    assert "bet_units" not in df.columns

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_bets_advised"] == 0
    assert "bet_units" in picks.columns


def test_build_game_picks_export_market_metrics_are_nan_with_no_real_market_data():
    # _predictions() logs no market_home_win_probability values at all -
    # every game predates quant-analytics item #6 slice 2's real wiring.
    # market_accuracy/etc must be honestly NaN, never a fabricated number.
    picks, summary = game_evaluation.build_game_picks_export(_predictions())

    assert summary.loc[0, "n_market_resolved"] == 0
    assert pd.isna(summary.loc[0, "market_accuracy"])
    assert pd.isna(summary.loc[0, "market_brier_score"])
    assert pd.isna(summary.loc[0, "market_log_loss"])
    assert summary.loc[0, "n_beat_closing_line_compared"] == 0
    assert pd.isna(summary.loc[0, "beat_closing_line_rate"])
    assert "market_home_win_probability" in picks.columns

    # Quant-analytics item #5: honestly NaN, not a fabricated CI/p-value,
    # when there's no real market comparison data at all.
    assert pd.isna(summary.loc[0, "market_accuracy_ci_low"])
    assert pd.isna(summary.loc[0, "beat_closing_line_rate_ci_low"])
    assert pd.isna(summary.loc[0, "beat_closing_line_rate_p_value"])
    # No bets advised either - win_rate_on_advised_bets is honestly NaN,
    # but its CI still reports the real "no information" Wilson bound
    # (0.0, 1.0), same n=0 contract as evaluation.wilson_confidence_interval
    # itself; roi_p_value is honestly NaN (no profit samples to test).
    assert summary.loc[0, "win_rate_on_advised_bets_ci_low"] == 0.0
    assert summary.loc[0, "win_rate_on_advised_bets_ci_high"] == 1.0
    assert pd.isna(summary.loc[0, "roi_p_value"])


def test_build_game_picks_export_market_metrics_missing_column_migration():
    # A log written before slice 2's column existed at all (not just null
    # values) must not crash - same migration convention as
    # above_threshold's own missing-column case.
    rows = [
        {"date": "2026-07-18", "game_pk": 1, "home_team": "NYY", "away_team": "BOS", "predicted_winner": "NYY",
         "predicted_probability": 0.65, "metric": "GamePick_Win_Probability", "actual_winner": "NYY", "game_played": 1},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    assert "market_home_win_probability" not in df.columns

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_market_resolved"] == 0
    assert pd.isna(summary.loc[0, "market_accuracy"])


def _market_comparison_rows():
    # Three above-threshold, resolved games with real market data:
    #  - pk 1: model home prob 0.90, market 0.60 - model's squared error
    #    (0.01) beats the market's (0.16) -> a real model win.
    #  - pk 2: model favors home at 0.60 (home prob 0.60), market favors
    #    home much less (0.30); the away team (SF) actually wins - model's
    #    squared error (0.36) is worse than the market's (0.09) -> a real
    #    market win.
    #  - pk 3: model home prob 0.75, market ALSO exactly 0.75 - a real
    #    exact tie, excluded from both sides of the rate.
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.90, "NYY", 1, market_home_win_probability=0.60),
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.60, "SF", 1, market_home_win_probability=0.30),
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.75, "HOU", 1, market_home_win_probability=0.75),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_build_game_picks_export_market_accuracy_brier_log_loss():
    picks, summary = game_evaluation.build_game_picks_export(_market_comparison_rows())

    # All three market picks (NYY, SF, HOU) match the real actual winners.
    assert summary.loc[0, "n_market_resolved"] == 3
    assert summary.loc[0, "market_accuracy"] == pytest.approx(1.0)

    expected_market_brier = ((0.60 - 1) ** 2 + (0.70 - 1) ** 2 + (0.75 - 1) ** 2) / 3
    assert summary.loc[0, "market_brier_score"] == pytest.approx(expected_market_brier)

    expected_market_log_loss = -(math.log(0.60) + math.log(0.70) + math.log(0.75)) / 3
    assert summary.loc[0, "market_log_loss"] == pytest.approx(expected_market_log_loss)

    # Quant-analytics item #5: market_accuracy's real Wilson CI (3 of 3) -
    # deliberately no p-value here (0.5 isn't a genuine "no skill" null for
    # an unconditional accuracy rate, see _market_comparison_metrics'
    # own docstring).
    expected_low, expected_high = evaluation.wilson_confidence_interval(3, 3)
    assert summary.loc[0, "market_accuracy_ci_low"] == pytest.approx(expected_low)
    assert summary.loc[0, "market_accuracy_ci_high"] == pytest.approx(expected_high)


def test_build_game_picks_export_beat_closing_line_rate_requires_closing_snapshots():
    picks, summary = game_evaluation.build_game_picks_export(_market_comparison_rows())

    # Logged prediction-time prices must not count as closing comparisons.
    assert summary.loc[0, "n_beat_closing_line_compared"] == 0
    assert pd.isna(summary.loc[0, "beat_closing_line_rate"])


def test_build_game_picks_export_beat_closing_line_rate_excludes_ties():
    from mlb_metrics import market_odds

    rows = _market_comparison_rows().copy()
    snaps = []
    for _, row in rows.iterrows():
        day = pd.Timestamp(row["date"]).date()
        snaps.append({
            "provider_event_id": f"e{row['game_pk']}",
            "sportsbook": "DraftKings",
            "captured_at_utc": f"{day}T22:00:00Z",
            "game_datetime": f"{day}T23:10:00Z",
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_moneyline": -150,
            "away_moneyline": 130,
            "market_home_win_probability": float(row["market_home_win_probability"]),
            "source_status": market_odds.SOURCE_OK,
            "game_pk": int(row["game_pk"]),
            "match_method": "unique_matchup",
            "snapshot_id": f"close_{row['game_pk']}",
            "snapshot_role": "intraday",
        })
    rows["game_datetime"] = [
        f"{pd.Timestamp(d).date()}T23:10:00Z" for d in rows["date"]
    ]
    picks, summary = game_evaluation.build_game_picks_export(
        rows, odds_snapshots=pd.DataFrame(snaps),
    )

    # pk 3's exact tie is excluded from the comparison base entirely -
    # only pk 1 (model win) and pk 2 (market win) count.
    assert summary.loc[0, "n_beat_closing_line_compared"] == 2
    assert summary.loc[0, "beat_closing_line_rate"] == pytest.approx(0.5)

    # Quant-analytics item #5: this IS a well-posed 0.5 null ("whose
    # squared error is lower on this game" is a genuine coin flip under
    # "no real skill difference") - 1 of 2 is exactly the null, so the
    # real two-sided exact binomial p-value must be 1.0.
    expected_low, expected_high = evaluation.wilson_confidence_interval(1, 2)
    assert summary.loc[0, "beat_closing_line_rate_ci_low"] == pytest.approx(expected_low)
    assert summary.loc[0, "beat_closing_line_rate_ci_high"] == pytest.approx(expected_high)
    assert summary.loc[0, "beat_closing_line_rate_p_value"] == pytest.approx(1.0)


def test_build_game_picks_export_picks_out_includes_market_column():
    picks, summary = game_evaluation.build_game_picks_export(_market_comparison_rows())

    assert "market_home_win_probability" in picks.columns
    by_pk = picks.set_index("game_pk")
    assert by_pk.loc[1, "market_home_win_probability"] == pytest.approx(0.60)
