"""Decision board: max price, states, gates, exposure, no auto-orders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, decision_board, paper_ledger


def test_modes_remain_fail_closed():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"


def test_max_acceptable_price_accounts_for_fees_and_edge():
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    p_model = 0.62
    min_edge = 0.02
    max_px = decision_board.max_acceptable_price(p_model, fee, min_edge=min_edge, qty=1.0)
    assert max_px is not None
    assert max_px < p_model - min_edge  # fee pushes price below raw edge budget
    # At max price, EV after fees should be ~min_edge
    env = decision_board.expected_net_value(p_model, max_px, fee, 1.0)
    assert env == pytest.approx(min_edge, abs=1e-3)
    # Above max must fail the price gate
    assert decision_board.acquisition_cost_per_contract(max_px + 0.01, fee) > (
        p_model - min_edge
    )


def test_high_probability_alone_never_paper_candidate():
    state, reason, actionable = decision_board.classify_decision_state(
        evidence_pass=False,
        game_started=False,
        quote_stale=False,
        missing_inputs=False,
        executable_buy=0.40,
        max_price=0.55,
        model_probability=0.90,
    )
    assert state == config.POLYMARKET_DECISION_STATE_PASS_INSUFFICIENT
    assert actionable is False
    assert reason == "evidence_gate_failed_or_unavailable"


def test_classify_all_explicit_states():
    cases = [
        dict(
            kwargs=dict(
                evidence_pass=True,
                game_started=True,
                quote_stale=False,
                missing_inputs=False,
                executable_buy=0.4,
                max_price=0.5,
                model_probability=0.6,
            ),
            state=config.POLYMARKET_DECISION_STATE_GAME_STARTED,
        ),
        dict(
            kwargs=dict(
                evidence_pass=True,
                game_started=False,
                quote_stale=False,
                missing_inputs=True,
                executable_buy=None,
                max_price=0.5,
                model_probability=0.6,
            ),
            state=config.POLYMARKET_DECISION_STATE_PASS_MISSING,
        ),
        dict(
            kwargs=dict(
                evidence_pass=True,
                game_started=False,
                quote_stale=True,
                missing_inputs=False,
                executable_buy=0.4,
                max_price=0.5,
                model_probability=0.6,
            ),
            state=config.POLYMARKET_DECISION_STATE_PASS_STALE,
        ),
        dict(
            kwargs=dict(
                evidence_pass=True,
                game_started=False,
                quote_stale=False,
                missing_inputs=False,
                executable_buy=0.55,
                max_price=0.50,
                model_probability=0.6,
            ),
            state=config.POLYMARKET_DECISION_STATE_PASS_PRICE,
        ),
        dict(
            kwargs=dict(
                evidence_pass=True,
                game_started=False,
                quote_stale=False,
                missing_inputs=False,
                executable_buy=0.40,
                max_price=0.50,
                model_probability=0.6,
            ),
            state=config.POLYMARKET_DECISION_STATE_PAPER_CANDIDATE,
        ),
    ]
    for case in cases:
        state, _reason, actionable = decision_board.classify_decision_state(**case["kwargs"])
        assert state == case["state"]
        if case["state"] == config.POLYMARKET_DECISION_STATE_PAPER_CANDIDATE:
            assert actionable is True
        else:
            assert actionable is False


def test_board_suppresses_actionable_while_betting_disabled(tmp_path: Path):
    now = "2026-09-15T12:00:00Z"
    registry = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "game_pk": 111,
                "home_team": "NYY",
                "away_team": "BOS",
                "long_team": "NYY",
                "event_slug": "mlb-bos-nyy-demo",
                "market_slug": "aec-demo",
                "scheduled_start_utc": "2026-09-15T23:00:00Z",
                "mapping_status": "mapped",
                "venue_id": "polymarket_us",
                "rules_hash": "demo",
            }
        ]
    )
    quotes = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "receive_time_utc": "2026-09-15T11:59:50Z",
                "best_ask": 0.45,
                "best_bid": 0.44,
                "best_ask_size": 10.0,
                "best_bid_size": 8.0,
                "eligible": True,
                "raw_response_hash": "h1",
            }
        ]
    )
    baseball = pd.DataFrame(
        [
            {
                "game_pk": 111,
                "home_starter_status": "probable",
                "away_starter_status": "probable",
                "home_lineup_status": "confirmed",
                "away_lineup_status": "confirmed",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 111,
                "home_team": "NYY",
                "away_team": "BOS",
                "predicted_winner": "NYY",
                "predicted_probability": 0.70,
                "model_version": "demo_fixture_v0",
            }
        ]
    )
    # Force evidence pass in gate dict — board must still not set actionable
    # while BETTING_MODE=disabled (stake guidance stays off).
    gate = {
        "evidence_pass": True,
        "verdict": "edge_supported",
        "validation_status": "validated_passed",
        "fail_reasons": [],
        "policy_version": "demo_policy",
        "policy_hash": "abc",
        "min_edge": 0.01,
        "size_contracts": 1.0,
        "evaluation": {"data_coverage": {"n_polymarket_labeled_eligible_dates": 200}},
    }
    board, meta = decision_board.build_decision_board(
        registry=registry,
        quotes=quotes,
        baseball=baseball,
        predictions=predictions,
        now_utc=now,
        gate=gate,
        demo_fixture=True,
    )
    assert len(board) == 2
    long_side = board[board["side_team"] == "NYY"].iloc[0]
    short_side = board[board["side_team"] == "BOS"].iloc[0]
    assert long_side["decision_state"] == config.POLYMARKET_DECISION_STATE_PAPER_CANDIDATE
    assert short_side["decision_state"] in (
        config.POLYMARKET_DECISION_STATE_PASS_PRICE,
        config.POLYMARKET_DECISION_STATE_PAPER_CANDIDATE,
        config.POLYMARKET_DECISION_STATE_PASS_MISSING,
    )
    assert not board["actionable"].astype(bool).any()
    assert not board["stake_guidance_enabled"].astype(bool).any()
    assert board["demo_fixture"].astype(bool).all()
    assert meta["orders_automated"] is False
    assert meta["n_actionable"] == 0
    assert float(long_side["executable_qty"]) == pytest.approx(10.0)
    assert float(long_side["max_acceptable_price"]) > 0

    paths = decision_board.write_decision_board(
        board,
        meta,
        board_path=str(tmp_path / "board.csv"),
        meta_path=str(tmp_path / "meta.json"),
    )
    reloaded = pd.read_csv(paths["board_path"])
    with open(paths["meta_path"], encoding="utf-8") as f:
        meta2 = json.load(f)
    assert len(reloaded) == 2
    assert meta2["evidence_verdict"] == "edge_supported"


def test_board_pass_insufficient_with_live_gate():
    gate = decision_board.load_evidence_gate()
    assert gate["evidence_pass"] is False
    assert gate["gates_ok_for_actionable"] is False
    registry = pd.DataFrame(
        [
            {
                "market_id": "m2",
                "game_pk": 222,
                "home_team": "LAD",
                "away_team": "SF",
                "long_team": "LAD",
                "event_slug": "mlb-sf-lad-demo",
                "market_slug": "aec-demo2",
                "scheduled_start_utc": "2099-01-01T00:00:00Z",
                "mapping_status": "mapped",
                "venue_id": "polymarket_us",
                "rules_hash": "demo",
            }
        ]
    )
    quotes = pd.DataFrame(
        [
            {
                "market_id": "m2",
                "receive_time_utc": "2099-01-01T00:00:00Z",
                "best_ask": 0.40,
                "best_bid": 0.39,
                "eligible": True,
            }
        ]
    )
    board, meta = decision_board.build_decision_board(
        registry=registry,
        quotes=quotes,
        baseball=pd.DataFrame(
            [
                {
                    "game_pk": 222,
                    "home_starter_status": "probable",
                    "away_starter_status": "probable",
                    "home_lineup_status": "confirmed",
                    "away_lineup_status": "confirmed",
                }
            ]
        ),
        predictions=pd.DataFrame(
            [
                {
                    "game_pk": 222,
                    "home_team": "LAD",
                    "away_team": "SF",
                    "predicted_winner": "LAD",
                    "predicted_probability": 0.80,
                }
            ]
        ),
        now_utc="2026-09-15T12:00:00Z",
        gate=gate,
        demo_fixture=True,
    )
    assert (board["decision_state"] == config.POLYMARKET_DECISION_STATE_PASS_INSUFFICIENT).all()
    assert meta["evidence_verdict"] in ("insufficient_evidence", "edge_not_supported", gate["verdict"])


def test_exposure_caps_block_correlated_and_daily():
    positions = [
        {"units": 1.0, "game_pk": 1, "side_team": "NYY", "decision_day": "2026-09-15", "status": "open"},
        {"units": 4.0, "game_pk": 2, "side_team": "BOS", "decision_day": "2026-09-15", "status": "open"},
    ]
    same_game = decision_board.exposure_check(
        proposed_units=1.0,
        existing_positions=positions,
        game_pk=1,
        team="NYY",
        per_bet_cap=1.0,
        daily_cap=10.0,
        same_game_cap=1.0,
        same_team_cap=2.0,
        decision_day="2026-09-15",
    )
    assert same_game["allowed"] is False
    assert "exceeds_same_game_cap" in same_game["reasons"]

    daily = decision_board.exposure_check(
        proposed_units=2.0,
        existing_positions=positions,
        game_pk=9,
        team="TB",
        per_bet_cap=5.0,
        daily_cap=5.0,
        same_game_cap=5.0,
        same_team_cap=5.0,
        decision_day="2026-09-15",
    )
    assert daily["allowed"] is False
    assert "exceeds_daily_cap" in daily["reasons"]

    ok = decision_board.exposure_check(
        proposed_units=1.0,
        existing_positions=positions,
        game_pk=9,
        team="TB",
        per_bet_cap=1.0,
        daily_cap=10.0,
        same_game_cap=1.0,
        same_team_cap=2.0,
        decision_day="2026-09-15",
    )
    assert ok["allowed"] is True


def test_uncertainty_note_has_no_invented_confidence():
    note = decision_board.uncertainty_note(
        model_probability=0.51,
        probability_source="market_only_fallback",
        evidence_pass=False,
        n_eligible_dates=0,
    )
    assert "No invented confidence score" in note
    assert "confidence=" not in note.lower() or "no invented" in note.lower()
    assert "market-only" in note.lower() or "market_only" in note.lower()
