"""Prospective protocol registration and remaining-season capacity tests."""

from __future__ import annotations

import json
from pathlib import Path

from mlb_metrics import config, polymarket_research


def test_remaining_season_cannot_meet_structural_floor():
    cap = polymarket_research.remaining_season_capacity(as_of_local="2026-09-16")
    assert cap["max_remaining_regular_season_calendar_dates"] == 12  # Sep 16–27
    assert cap["remaining_regular_season_meets_structural_floor"] is False
    assert cap["remaining_regular_season_meets_power_plan"] is False
    assert cap["conclusion"] == "insufficient_for_structural_floor_this_regular_season"


def test_protocol_v2_registers_required_fields(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    policy_path = tmp_path / "policy.json"
    monkeypatch.setattr(config, "POLYMARKET_PAPER_PROTOCOL_PATH", str(protocol_path))
    monkeypatch.setattr(config, "POLYMARKET_FROZEN_POLICY_PATH", str(policy_path))
    protocol = polymarket_research.register_protocol(path=str(protocol_path))
    assert protocol["protocol_version"] == "2"
    assert protocol["protocol_id"] == "polymarket_us_game_winner_v2"
    assert "market_mid_baseline" in protocol["candidates"]
    assert protocol["decision_times"]["entry_minutes_before_scheduled_start"] == 30
    assert protocol["cohorts"]["postseason"]["do_not_transfer_regular_season_results"] is True
    assert "2026-09-14" in protocol["exploratory_dates"]
    assert protocol["chronological_periods"]["prospective_collection_start_local"] == "2026-09-16"
    assert protocol["evaluation_checkpoints"]["game_days"] == [7, 14, 28, 70]
    assert protocol["uncertainty_methods"]["no_invented_confidence_score"] is True
    assert protocol["remaining_season_capacity"]["remaining_regular_season_meets_structural_floor"] is False
    frozen = polymarket_research.write_frozen_policy(path=str(policy_path), protocol=protocol)
    frozen2 = polymarket_research.write_frozen_policy(path=str(policy_path), protocol=protocol)
    assert frozen["policy_hash"] == frozen2["policy_hash"]
    assert frozen["betting_mode"] == "disabled"


def test_protocol_registration_is_idempotent(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    monkeypatch.setattr(config, "POLYMARKET_PAPER_PROTOCOL_PATH", str(protocol_path))
    a = polymarket_research.register_protocol(path=str(protocol_path))
    b = polymarket_research.register_protocol(path=str(protocol_path))
    assert a["protocol_hash"] == b["protocol_hash"]
    loaded = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    assert loaded["protocol_hash"] == a["protocol_hash"]


def test_modes_remain_disabled_for_prospective_collection():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
