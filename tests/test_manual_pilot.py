"""Manual pilot: readiness dual conclusions, pauses, fills, guidance rejects."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, decision_board, manual_pilot


def test_modes_fail_closed():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"


def test_software_vs_evidence_conclusions_separate(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        config,
        "POLYMARKET_PILOT_PAUSE_STATE_PATH",
        str(tmp_path / "pause.json"),
    )
    monkeypatch.setattr(
        config,
        "POLYMARKET_PILOT_READINESS_REPORT_PATH",
        str(tmp_path / "ready.json"),
    )
    monkeypatch.setattr(
        config,
        "POLYMARKET_PILOT_RECONCILE_REPORT_PATH",
        str(tmp_path / "recon.json"),
    )
    monkeypatch.setattr(
        config,
        "POLYMARKET_PILOT_RISK_LIMITS_PATH",
        str(tmp_path / "risk.json"),
    )
    report = manual_pilot.build_pilot_readiness_report(personal_limits=None)
    assert report["dual_conclusions"]["software_operates_correctly"] in (True, False)
    assert report["dual_conclusions"]["evidence_supports_limited_real_money_pilot"] is False
    assert report["pilot_authorized"] is False
    assert report["edge_claimed"] is False
    reasons = report["evidence_for_limited_real_money_pilot"]["no_bet_reasons"]
    assert any("insufficient" in r or "evidence_verdict" in r or "validation_status" in r for r in reasons)
    assert any("paper_units_not_usd" in r or "BETTING_MODE" in r for r in reasons)
    assert report["risk_limits"]["currency"] == "normalized_paper_unit"
    assert report["stake_guidance"]["mode"] == "normalized_paper_units"


def test_agent_paper_risk_limits_are_conservative_and_not_usd():
    limits = manual_pilot.agent_paper_risk_limits()
    assert limits["currency"] == "normalized_paper_unit"
    assert limits["real_money_usd_stakes_enabled"] is False
    assert limits["income_target_not_used"] is True
    assert float(limits["max_affordable_loss"]) <= float(limits["bankroll"])
    assert float(limits["per_bet_exposure_limit"]) <= float(limits["daily_exposure_limit"])
    ok, missing = manual_pilot.personal_limits_complete(limits)
    assert ok is True
    assert missing == []
    evidence = manual_pilot.assess_evidence_for_pilot(personal_limits=limits)
    assert evidence["supported"] is False
    assert "risk_limits_are_paper_units_not_usd_bankroll" in evidence["no_bet_reasons"]


def test_personal_limits_required_fields():
    ok, missing = manual_pilot.personal_limits_complete(None)
    assert ok is False
    assert "bankroll" in missing
    ok2, missing2 = manual_pilot.personal_limits_complete(
        {
            "bankroll": 1000,
            "max_affordable_loss": 100,
            "per_bet_exposure_limit": 10,
            "daily_exposure_limit": 50,
            "same_game_exposure_limit": 10,
            "same_team_exposure_limit": 20,
        }
    )
    assert ok2 is True
    assert missing2 == []
    ok3, missing3 = manual_pilot.personal_limits_complete(
        {"bankroll": 100, "max_affordable_loss": 200, "per_bet": 1, "daily": 1, "same_game": 1, "same_team": 1}
    )
    assert ok3 is False
    assert "max_affordable_loss_exceeds_bankroll" in missing3


def test_market_only_fallback_rejected_as_independent():
    identity = manual_pilot.verify_candidate_and_market(
        evaluation={"protocol_id": config.POLYMARKET_PILOT_REQUIRED_PROTOCOL_ID, "policy_hash": "x"},
        protocol={
            "protocol_id": config.POLYMARKET_PILOT_REQUIRED_PROTOCOL_ID,
            "market_id": config.POLYMARKET_PILOT_MARKET_ID,
            "candidates": ["market_only_fallback"],
        },
        policy={"policy_hash": "x"},
    )
    assert identity["ok"] is False
    assert any("banned_independent_candidate" in r for r in identity["reasons"])


def test_guidance_rejects_each_condition():
    evidence_ok = {"supported": True, "no_bet_reasons": []}
    evidence_bad = {"supported": False, "no_bet_reasons": ["evidence_verdict:insufficient_evidence"]}
    pause_clear = {"paused": False, "reasons": []}
    pause_on = {"paused": True, "reasons": ["operational_failure"]}
    exposure_ok = {"allowed": True, "reasons": []}
    exposure_bad = {"allowed": False, "reasons": ["exceeds_daily_cap"]}

    cases = [
        dict(
            kwargs=dict(
                evidence=evidence_bad,
                pause=pause_clear,
                quote_stale=False,
                executable_buy=0.4,
                max_acceptable_price=0.5,
                missing_inputs=False,
                exposure=exposure_ok,
                probability_source="residual",
            ),
            must="missing_or_failed_evidence",
        ),
        dict(
            kwargs=dict(
                evidence=evidence_ok,
                pause=pause_on,
                quote_stale=False,
                executable_buy=0.4,
                max_acceptable_price=0.5,
                missing_inputs=False,
                exposure=exposure_ok,
                probability_source="residual",
            ),
            must="pilot_paused",
        ),
        dict(
            kwargs=dict(
                evidence=evidence_ok,
                pause=pause_clear,
                quote_stale=True,
                executable_buy=0.4,
                max_acceptable_price=0.5,
                missing_inputs=False,
                exposure=exposure_ok,
                probability_source="residual",
            ),
            must="stale_quote",
        ),
        dict(
            kwargs=dict(
                evidence=evidence_ok,
                pause=pause_clear,
                quote_stale=False,
                executable_buy=0.55,
                max_acceptable_price=0.50,
                missing_inputs=False,
                exposure=exposure_ok,
                probability_source="residual",
            ),
            must="price_above_max_acceptable",
        ),
        dict(
            kwargs=dict(
                evidence=evidence_ok,
                pause=pause_clear,
                quote_stale=False,
                executable_buy=0.4,
                max_acceptable_price=0.5,
                missing_inputs=True,
                exposure=exposure_ok,
                probability_source="residual",
            ),
            must="missing_inputs",
        ),
        dict(
            kwargs=dict(
                evidence=evidence_ok,
                pause=pause_clear,
                quote_stale=False,
                executable_buy=0.4,
                max_acceptable_price=0.5,
                missing_inputs=False,
                exposure=exposure_bad,
                probability_source="residual",
            ),
            must="excess_exposure",
        ),
        dict(
            kwargs=dict(
                evidence=evidence_ok,
                pause=pause_clear,
                quote_stale=False,
                executable_buy=0.4,
                max_acceptable_price=0.5,
                missing_inputs=False,
                exposure=exposure_ok,
                probability_source="market_only_fallback",
            ),
            must="market_only_not_independent_model",
        ),
        dict(
            kwargs=dict(
                evidence=evidence_ok,
                pause=pause_clear,
                quote_stale=False,
                executable_buy=0.4,
                max_acceptable_price=0.5,
                missing_inputs=False,
                exposure=exposure_ok,
                probability_source="residual",
                board_age_seconds=9999,
            ),
            must="stale_board_export",
        ),
    ]
    for case in cases:
        out = manual_pilot.guidance_allowed(**case["kwargs"])
        assert out["allowed"] is False
        assert out["orders_automated"] is False
        assert case["must"] in out["reasons"]


def test_pause_conditions_preserve_exposure(tmp_path: Path):
    pause = manual_pilot.evaluate_pause_conditions(
        operational_failure=True,
        settled_real_net=-50,
        max_affordable_loss=100,
        evidence_supported=True,
        owner_pause=False,
        previous={"paused": False},
    )
    assert pause["paused"] is True
    assert "operational_failure" in pause["reasons"]
    assert pause["existing_exposure_preserved"] is True

    loss = manual_pilot.evaluate_pause_conditions(
        operational_failure=False,
        settled_real_net=-100,
        max_affordable_loss=100,
        evidence_supported=True,
        owner_pause=False,
    )
    assert "loss_limit_exceeded" in loss["reasons"]

    ev = manual_pilot.evaluate_pause_conditions(
        operational_failure=False,
        settled_real_net=0,
        max_affordable_loss=100,
        evidence_supported=False,
        owner_pause=False,
    )
    assert "evidence_deterioration" in ev["reasons"]

    path = manual_pilot.write_pause_state(pause, path=str(tmp_path / "pause.json"))
    reloaded = manual_pilot.load_pause_state(path)
    assert reloaded["paused"] is True
    assert reloaded["existing_exposure_preserved"] is True


def test_real_fill_ledger_and_reconcile_separate(tmp_path: Path):
    path = str(tmp_path / "real_fills.csv")
    pause_clear = {"paused": False, "reasons": []}
    limits = {
        "per_bet": 2.0,
        "daily": 5.0,
        "same_game": 2.0,
        "same_team": 3.0,
    }
    ok = manual_pilot.record_real_fill(
        {
            "market_id": "m1",
            "game_pk": 1,
            "side_team": "NYY",
            "purchase_price": 0.45,
            "qty": 1.0,
            "fees_paid": 0.01,
            "status": "open",
            "decision_day": "2026-09-16",
        },
        path=path,
        pause_state=pause_clear,
        limits=limits,
    )
    assert ok["accepted"] is True

    dup = manual_pilot.record_real_fill(
        {
            "market_id": "m1",
            "game_pk": 1,
            "side_team": "NYY",
            "purchase_price": 0.46,
            "qty": 1.0,
            "fees_paid": 0.01,
            "status": "open",
            "decision_day": "2026-09-16",
        },
        path=path,
        pause_state=pause_clear,
        limits=limits,
    )
    assert dup["accepted"] is False
    assert "duplicate_open_market_side" in dup["reasons"]

    paused_block = manual_pilot.record_real_fill(
        {
            "market_id": "m2",
            "game_pk": 2,
            "side_team": "BOS",
            "purchase_price": 0.40,
            "qty": 1.0,
            "fees_paid": 0.01,
            "status": "open",
            "decision_day": "2026-09-16",
        },
        path=path,
        pause_state={"paused": True, "reasons": ["owner_pause"]},
        limits=limits,
    )
    assert paused_block["accepted"] is False
    assert "pilot_paused_new_open_fills_blocked" in paused_block["reasons"]

    # Settle and reconcile
    fills = manual_pilot.load_real_fills(path)
    fills.loc[0, "status"] = "settled"
    fills.loc[0, "settlement_px"] = 1.0
    fills.loc[0, "settled_net"] = 1.0 - 0.45 - 0.01
    fills.to_csv(path, index=False)
    recon = manual_pilot.reconcile_real_vs_paper(
        real_fills=manual_pilot.load_real_fills(path),
        paper_summary={"settled_net": 9.99, "open_exposure": 0.0, "roi": 1.0, "n_settled": 1},
    )
    assert recon["streams_separate"] is True
    assert recon["real_manual"]["settled_net"] == pytest.approx(0.54)
    assert recon["paper_ledger"]["settled_net"] == 9.99
    assert recon["edge_claimed"] is False


def test_exposure_increase_protocol_forbids_recovery_boosting(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        config,
        "POLYMARKET_PILOT_EXPOSURE_REVIEW_PROTOCOL_PATH",
        str(tmp_path / "exp_review.json"),
    )
    protocol = manual_pilot.register_exposure_review_protocol()
    assert protocol["rules"]["never_increase_to_recover_losses"] is True
    assert protocol["rules"]["never_increase_to_meet_income_deadline"] is True
    assert Path(config.POLYMARKET_PILOT_EXPOSURE_REVIEW_PROTOCOL_PATH).exists()


def test_classify_pause_and_exposure_states():
    state, reason, actionable = decision_board.classify_decision_state(
        evidence_pass=True,
        game_started=False,
        quote_stale=False,
        missing_inputs=False,
        executable_buy=0.4,
        max_price=0.5,
        model_probability=0.6,
        pilot_paused=True,
    )
    assert state == config.POLYMARKET_DECISION_STATE_PASS_PAUSED
    assert actionable is False

    state2, reason2, actionable2 = decision_board.classify_decision_state(
        evidence_pass=True,
        game_started=False,
        quote_stale=False,
        missing_inputs=False,
        executable_buy=0.4,
        max_price=0.5,
        model_probability=0.6,
        excess_exposure=True,
    )
    assert state2 == config.POLYMARKET_DECISION_STATE_PASS_EXPOSURE
    assert actionable2 is False


def test_restart_safe_pause_and_fills(tmp_path: Path):
    pause_path = str(tmp_path / "pause.json")
    fills_path = str(tmp_path / "fills.csv")
    manual_pilot.write_pause_state(
        {"paused": True, "reasons": ["evidence_deterioration"], "paused_at_utc": "2026-09-16T12:00:00Z"},
        path=pause_path,
    )
    assert manual_pilot.load_pause_state(pause_path)["paused"] is True
    manual_pilot.record_real_fill(
        {
            "market_id": "m9",
            "side_team": "LAD",
            "game_pk": 9,
            "purchase_price": 0.5,
            "qty": 1,
            "fees_paid": 0,
            "status": "open",
            "decision_day": "2026-09-16",
            "force_record_during_pause": True,
        },
        path=fills_path,
        pause_state={"paused": True, "reasons": ["evidence_deterioration"]},
        limits={"per_bet": 5, "daily": 5, "same_game": 5, "same_team": 5},
    )
    assert len(manual_pilot.load_real_fills(fills_path)) == 1
