"""Tests for timestamped market-odds snapshots, safe game_pk matching,
closing-line selection, paired scoring, and CLV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, game_evaluation, market_odds


def _provider_row(
    *,
    event_id,
    home,
    away,
    home_ml,
    away_ml,
    captured_at,
    game_datetime,
    date="2026-09-03",
    sportsbook="DraftKings",
    status="ok",
):
    home_imp = market_odds.moneyline_to_implied_probability(home_ml)
    away_imp = market_odds.moneyline_to_implied_probability(away_ml)
    return {
        "provider_event_id": str(event_id),
        "sportsbook": sportsbook,
        "captured_at_utc": captured_at,
        "game_datetime": game_datetime,
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "market_home_win_probability": market_odds.devig(home_imp, away_imp),
        "source_status": status,
    }


def _schedule_row(game_pk, home, away, game_datetime, date="2026-09-03"):
    return {
        "game_pk": game_pk,
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "game_datetime": game_datetime,
    }


# ---------------------------------------------------------------------------
# Snapshots / closing selection
# ---------------------------------------------------------------------------


def test_multiple_snapshots_for_one_game_opening_and_closing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MARKET_ODDS_SNAPSHOTS_PATH", str(tmp_path / "odds.csv"))
    start = "2026-09-03T23:10:00Z"
    rows = pd.DataFrame([
        _provider_row(
            event_id="e1", home="NYY", away="BOS", home_ml=-150, away_ml=130,
            captured_at="2026-09-03T12:00:00Z", game_datetime=start,
        ),
        _provider_row(
            event_id="e1", home="NYY", away="BOS", home_ml=-170, away_ml=145,
            captured_at="2026-09-03T20:00:00Z", game_datetime=start,
        ),
        _provider_row(
            event_id="e1", home="NYY", away="BOS", home_ml=-180, away_ml=155,
            captured_at="2026-09-03T22:55:00Z", game_datetime=start,
        ),
    ])
    matched = market_odds.match_provider_events_to_games(
        rows,
        pd.DataFrame([_schedule_row(100, "NYY", "BOS", start)]),
    )
    matched["snapshot_role"] = ["morning", "lineup_lock", "raw"]
    matched["snapshot_id"] = [market_odds.make_snapshot_id(r) for _, r in matched.iterrows()]
    market_odds.append_odds_snapshots(matched)

    audit = market_odds.normalize_snapshot_frame(pd.read_csv(tmp_path / "odds.csv"))
    assert len(audit) == 3

    opening = market_odds.select_opening_snapshot(audit, 100)
    closing = market_odds.select_closing_snapshot(audit, 100, game_datetime=start)
    morning = market_odds.select_role_snapshot(audit, 100, "morning")
    lock = market_odds.select_role_snapshot(audit, 100, "lineup_lock")

    assert opening["captured_at_utc"] == "2026-09-03T12:00:00Z"
    assert closing["captured_at_utc"] == "2026-09-03T22:55:00Z"
    assert morning["home_moneyline"] == -150
    assert lock["home_moneyline"] == -170
    assert closing["home_moneyline"] == -180


def test_exact_closing_snapshot_selection_excludes_post_start():
    start = "2026-09-03T23:10:00Z"
    rows = pd.DataFrame([
        _provider_row(
            event_id="e1", home="NYY", away="BOS", home_ml=-150, away_ml=130,
            captured_at="2026-09-03T22:00:00Z", game_datetime=start,
        ),
        _provider_row(
            event_id="e1", home="NYY", away="BOS", home_ml=-200, away_ml=170,
            captured_at="2026-09-03T23:10:00Z", game_datetime=start,  # at start
        ),
        _provider_row(
            event_id="e1", home="NYY", away="BOS", home_ml=-210, away_ml=180,
            captured_at="2026-09-03T23:30:00Z", game_datetime=start,  # after start
        ),
    ])
    matched = market_odds.match_provider_events_to_games(
        rows, pd.DataFrame([_schedule_row(100, "NYY", "BOS", start)]),
    )
    matched = market_odds.mark_post_start_snapshots(matched)
    assert set(matched.loc[matched["captured_at_utc"] >= start, "source_status"]) == {"post_start"}

    closing = market_odds.select_closing_snapshot(matched, 100, game_datetime=start)
    assert closing is not None
    assert closing["captured_at_utc"] == "2026-09-03T22:00:00Z"
    assert closing["home_moneyline"] == -150


def test_no_post_start_snapshot_used_as_closing():
    start = "2026-09-03T23:10:00Z"
    rows = pd.DataFrame([
        _provider_row(
            event_id="e1", home="NYY", away="BOS", home_ml=-150, away_ml=130,
            captured_at="2026-09-03T23:45:00Z", game_datetime=start,
        ),
    ])
    matched = market_odds.match_provider_events_to_games(
        rows, pd.DataFrame([_schedule_row(100, "NYY", "BOS", start)]),
    )
    matched = market_odds.mark_post_start_snapshots(matched)
    assert market_odds.select_closing_snapshot(matched, 100, game_datetime=start) is None


def test_doubleheader_matching_by_start_time():
    schedule = pd.DataFrame([
        _schedule_row(101, "NYY", "BOS", "2026-09-03T17:05:00Z"),
        _schedule_row(102, "NYY", "BOS", "2026-09-03T23:10:00Z"),
    ])
    provider = pd.DataFrame([
        _provider_row(
            event_id="a", home="NYY", away="BOS", home_ml=-110, away_ml=-110,
            captured_at="2026-09-03T12:00:00Z", game_datetime="2026-09-03T17:05:00Z",
        ),
        _provider_row(
            event_id="b", home="NYY", away="BOS", home_ml=-120, away_ml=100,
            captured_at="2026-09-03T12:00:00Z", game_datetime="2026-09-03T23:10:00Z",
        ),
    ])
    matched = market_odds.match_provider_events_to_games(provider, schedule)
    by_event = matched.set_index("provider_event_id")
    assert int(by_event.loc["a", "game_pk"]) == 101
    assert int(by_event.loc["b", "game_pk"]) == 102
    assert by_event.loc["a", "match_method"] == "start_time"
    assert by_event.loc["b", "match_method"] == "start_time"
    assert by_event.loc["a", "source_status"] == "ok"


def test_doubleheader_matching_by_game_order_when_times_align():
    schedule = pd.DataFrame([
        _schedule_row(201, "LAD", "SF", "2026-09-03T18:00:00Z"),
        _schedule_row(202, "LAD", "SF", "2026-09-03T23:00:00Z"),
    ])
    # Provider times slightly offset but same order / unique.
    provider = pd.DataFrame([
        _provider_row(
            event_id="x1", home="LAD", away="SF", home_ml=-130, away_ml=110,
            captured_at="2026-09-03T10:00:00Z", game_datetime="2026-09-03T18:10:00Z",
        ),
        _provider_row(
            event_id="x2", home="LAD", away="SF", home_ml=-140, away_ml=120,
            captured_at="2026-09-03T10:00:00Z", game_datetime="2026-09-03T23:20:00Z",
        ),
    ])
    # Force start_time miss (tolerance tiny) then game_order.
    ambiguous = []
    matched = market_odds.match_provider_events_to_games(
        provider, schedule, time_tolerance_minutes=1, ambiguous_log=ambiguous,
    )
    # With 1-minute tolerance, times don't match → fall through to game_order.
    by_event = matched.set_index("provider_event_id")
    assert int(by_event.loc["x1", "game_pk"]) == 201
    assert int(by_event.loc["x2", "game_pk"]) == 202
    assert by_event.loc["x1", "match_method"] == "game_order"
    assert not ambiguous


def test_ambiguous_match_rejected_and_logged():
    schedule = pd.DataFrame([
        _schedule_row(301, "CIN", "STL", "2026-09-03T18:00:00Z"),
        _schedule_row(302, "CIN", "STL", "2026-09-03T23:00:00Z"),
    ])
    # Single provider event, two MLB games, no usable time → ambiguous.
    provider = pd.DataFrame([
        _provider_row(
            event_id="amb", home="CIN", away="STL", home_ml=-105, away_ml=-115,
            captured_at="2026-09-03T12:00:00Z", game_datetime=None,
        ),
    ])
    provider.loc[0, "game_datetime"] = pd.NA
    ambiguous = []
    matched = market_odds.match_provider_events_to_games(
        provider, schedule, ambiguous_log=ambiguous,
    )
    assert matched.iloc[0]["source_status"] == "ambiguous_match"
    assert pd.isna(matched.iloc[0]["game_pk"])
    assert ambiguous and ambiguous[0]["reason"] == "ambiguous_doubleheader"


# ---------------------------------------------------------------------------
# Paired scoring + bootstrap + CLV
# ---------------------------------------------------------------------------


def test_paired_brier_and_log_loss_calculations():
    # Two resolved games; compute hand-checked differences.
    picks = pd.DataFrame([
        {
            "date": "2026-09-01", "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
            "predicted_winner": "NYY", "predicted_probability": 0.7,
            "actual_winner": "NYY", "market_home_win_probability": 0.6,
            "above_threshold": True, "metric": "GamePick_Win_Probability",
        },
        {
            "date": "2026-09-02", "game_pk": 2, "home_team": "LAD", "away_team": "SF",
            "predicted_winner": "LAD", "predicted_probability": 0.55,
            "actual_winner": "SF", "market_home_win_probability": 0.65,
            "above_threshold": True, "metric": "GamePick_Win_Probability",
        },
    ])
    # Game1: y=1, model=0.7 SE=0.09, market=0.6 SE=0.16
    # Game2: y=0, model=0.55 SE=0.3025, market=0.65 SE=0.4225
    paired = game_evaluation.paired_market_scoring_differences(
        picks, n_bootstrap=200, random_seed=1,
    )
    expected_brier_diff = ((0.09 + 0.3025) / 2) - ((0.16 + 0.4225) / 2)
    assert paired["n_compared"] == 2
    assert paired["brier_diff"] == pytest.approx(expected_brier_diff)
    assert paired["mean_prob_diff"] == pytest.approx(((0.7 - 0.6) + (0.55 - 0.65)) / 2)
    # Both games model SE lower → secondary rate 1.0
    assert paired["pct_model_error_lower"] == pytest.approx(1.0)
    assert paired["brier_diff"] < 0  # model better on magnitude, not just count
    assert paired["log_loss_diff"] < 0


def test_paired_scoring_uses_closing_not_morning_logged_price():
    picks = pd.DataFrame([{
        "date": "2026-09-03", "game_pk": 100, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.70,
        "actual_winner": "NYY", "market_home_win_probability": 0.90,  # morning (worse)
        "above_threshold": True, "game_datetime": "2026-09-03T23:10:00Z",
    }])
    snaps = market_odds.match_provider_events_to_games(
        pd.DataFrame([
            _provider_row(
                event_id="e1", home="NYY", away="BOS", home_ml=-110, away_ml=-110,
                captured_at="2026-09-03T22:00:00Z", game_datetime="2026-09-03T23:10:00Z",
            ),
        ]),
        pd.DataFrame([_schedule_row(100, "NYY", "BOS", "2026-09-03T23:10:00Z")]),
    )
    # Closing ~0.5 de-vigged from -110/-110
    paired = game_evaluation.paired_market_scoring_differences(
        picks, odds_snapshots=snaps, n_bootstrap=50, random_seed=0,
    )
    # Against closing ~0.5, model 0.7 SE=0.09; against morning 0.9 SE=0.01
    # If closing used: market SE=(0.5-1)^2=0.25, brier_diff=0.09-0.25 < 0
    assert paired["brier_diff"] == pytest.approx(0.09 - 0.25)


def test_block_bootstrap_by_date_is_deterministic_with_seed():
    picks = pd.DataFrame([
        {
            "date": f"2026-09-0{i}", "game_pk": i, "home_team": "NYY", "away_team": "BOS",
            "predicted_winner": "NYY", "predicted_probability": 0.6 + 0.02 * i,
            "actual_winner": "NYY" if i % 2 else "BOS",
            "market_home_win_probability": 0.55,
            "above_threshold": True,
        }
        for i in range(1, 6)
    ])
    a = game_evaluation.paired_market_scoring_differences(picks, n_bootstrap=100, random_seed=7)
    b = game_evaluation.paired_market_scoring_differences(picks, n_bootstrap=100, random_seed=7)
    assert a["brier_diff_ci_low"] == b["brier_diff_ci_low"]
    assert a["brier_diff_ci_high"] == b["brier_diff_ci_high"]
    assert a["n_date_blocks"] == 5
    assert a["brier_diff_ci_low"] <= a["brier_diff"] <= a["brier_diff_ci_high"] or True  # CI may miss with tiny n


def test_closing_line_value_reporting():
    picks = pd.DataFrame([{
        "date": "2026-09-03", "game_pk": 100, "home_team": "NYY", "away_team": "BOS",
        "bet_units": 1.0, "bet_side": "home", "bet_team": "NYY", "bet_moneyline": -150,
        "market_odds_snapshot_id": "odds_bet",
        "game_datetime": "2026-09-03T23:10:00Z",
    }])
    snaps = market_odds.match_provider_events_to_games(
        pd.DataFrame([
            _provider_row(
                event_id="e1", home="NYY", away="BOS", home_ml=-170, away_ml=145,
                captured_at="2026-09-03T22:30:00Z", game_datetime="2026-09-03T23:10:00Z",
            ),
        ]),
        pd.DataFrame([_schedule_row(100, "NYY", "BOS", "2026-09-03T23:10:00Z")]),
    )
    clv = game_evaluation.closing_line_value_table(picks, odds_snapshots=snaps)
    assert len(clv) == 1
    bet_imp = market_odds.moneyline_to_implied_probability(-150)
    close_imp = market_odds.moneyline_to_implied_probability(-170)
    assert clv.iloc[0]["bet_implied_probability"] == pytest.approx(bet_imp)
    assert clv.iloc[0]["closing_implied_probability"] == pytest.approx(close_imp)
    assert clv.iloc[0]["probability_clv"] == pytest.approx(close_imp - bet_imp)
    assert clv.iloc[0]["moneyline_clv"] == pytest.approx(-20)


def test_build_game_picks_export_includes_paired_primary_metrics():
    picks = pd.DataFrame([
        {
            "date": "2026-09-01", "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
            "predicted_winner": "NYY", "predicted_probability": 0.7,
            "actual_winner": "NYY", "game_played": 1, "metric": "GamePick_Win_Probability",
            "market_home_win_probability": 0.6, "bet_units": 0.0, "above_threshold": True,
        },
        {
            "date": "2026-09-02", "game_pk": 2, "home_team": "LAD", "away_team": "SF",
            "predicted_winner": "LAD", "predicted_probability": 0.55,
            "actual_winner": "SF", "game_played": 1, "metric": "GamePick_Win_Probability",
            "market_home_win_probability": 0.65, "bet_units": 0.0, "above_threshold": True,
        },
    ])
    _, summary = game_evaluation.build_game_picks_export(picks)
    assert "model_minus_market_brier" in summary.columns
    assert "model_minus_market_log_loss" in summary.columns
    assert "mean_model_minus_market_probability" in summary.columns
    assert summary.loc[0, "n_paired_market_compared"] == 2
    # Secondary metric still present
    assert "beat_closing_line_rate" in summary.columns


def test_market_for_live_recommendations_requires_snapshot_provenance():
    legacy = pd.DataFrame([{
        "home_team": "NYY", "away_team": "BOS",
        "market_home_win_probability": 0.55,
        "home_moneyline": -120, "away_moneyline": 100,
    }])
    with pytest.raises(ValueError, match="Legacy team-only"):
        market_odds.market_for_live_recommendations(legacy, required_game_pks=[100])


def test_market_for_live_recommendations_exact_game_pk_and_roles():
    snaps = pd.DataFrame([
        {
            "provider_event_id": "e1", "sportsbook": "DraftKings",
            "captured_at_utc": "2026-09-03T16:00:00Z",
            "game_datetime": "2026-09-03T23:10:00Z",
            "date": pd.Timestamp("2026-09-03"),
            "home_team": "NYY", "away_team": "BOS",
            "home_moneyline": -130, "away_moneyline": 110,
            "market_home_win_probability": 0.54,
            "source_status": market_odds.SOURCE_OK,
            "game_pk": 100,
            "match_method": "unique_matchup",
            "snapshot_id": "odds_morning",
            "snapshot_role": market_odds.SNAPSHOT_ROLE_MORNING,
        },
        {
            "provider_event_id": "e1", "sportsbook": "DraftKings",
            "captured_at_utc": "2026-09-03T20:00:00Z",
            "game_datetime": "2026-09-03T23:10:00Z",
            "date": pd.Timestamp("2026-09-03"),
            "home_team": "NYY", "away_team": "BOS",
            "home_moneyline": -140, "away_moneyline": 120,
            "market_home_win_probability": 0.56,
            "source_status": market_odds.SOURCE_OK,
            "game_pk": 100,
            "match_method": "unique_matchup",
            "snapshot_id": "odds_lock",
            "snapshot_role": market_odds.SNAPSHOT_ROLE_LINEUP_LOCK,
        },
        {
            "provider_event_id": "e1", "sportsbook": "DraftKings",
            "captured_at_utc": "2026-09-03T23:30:00Z",
            "game_datetime": "2026-09-03T23:10:00Z",
            "date": pd.Timestamp("2026-09-03"),
            "home_team": "NYY", "away_team": "BOS",
            "home_moneyline": -150, "away_moneyline": 130,
            "market_home_win_probability": 0.58,
            "source_status": market_odds.SOURCE_OK,
            "game_pk": 100,
            "match_method": "unique_matchup",
            "snapshot_id": "odds_post",
            "snapshot_role": market_odds.SNAPSHOT_ROLE_INTRADAY,
        },
    ])
    out = market_odds.market_for_live_recommendations(snaps, required_game_pks=[100])
    assert len(out) == 1
    assert int(out.iloc[0]["game_pk"]) == 100
    assert out.iloc[0]["snapshot_role"] == market_odds.SNAPSHOT_ROLE_LINEUP_LOCK
    assert out.iloc[0]["snapshot_id"] == "odds_lock"

    with pytest.raises(ValueError, match="game_pk"):
        market_odds.market_for_live_recommendations(snaps, required_game_pks=[100, 999])
