"""Hit-prop frozen paper ops: exclusions, restart, reconcile, checkpoints."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mlb_metrics import config, hit_prop_ops as ops


def _contract(**overrides):
    base = {
        "capture_id": "c1",
        "game_type": "R",
        "game_type_source": "mlb_statsapi_game_pk",
        "market_id": "m1",
        "market_slug": "slug-1",
        "game_pk": 100,
        "key_mlbam": 200,
        "player_name": "Ops Player",
        "game_mapping_status": "mapped",
        "player_mapping_status": "mapped",
        "rules_hash": "rh",
        "quarantined": False,
        "scheduled_start_utc": "2099-06-01T00:00:00Z",
        "book_status": "captured",
        "book_receive_time_utc": "2099-05-31T23:00:00Z",
        "yes_buy_price": 0.55,
        "yes_buy_size": 8.0,
        "no_buy_price": 0.48,
        "requested_local_date": "2099-05-31",
    }
    base.update(overrides)
    return base


def test_load_frozen_policy_from_disk():
    policy = ops.load_frozen_policy()
    assert policy["policy_version"] == config.HIT_PROP_PAPER_POLICY_VERSION
    assert policy["action"] == "paper_only"
    assert policy["betting_mode"] == "disabled"


def test_explicit_exclusions_not_silent_drops():
    policy = ops.load_frozen_policy()
    bad = ops.evaluate_contract_before_outcome(
        _contract(quarantined=True, game_mapping_status="ambiguous", yes_buy_price=None),
        policy=policy,
        decision_time_utc="2099-05-31T22:00:00Z",
        lineup_started=None,
    )
    assert bad["decision"]["action"] == "pass"
    assert bad["exclusion"] is not None
    assert bad["exclusion"]["dropped_as_unfavorable_outcome"] is False
    assert "quarantined" in bad["pass_reasons"]
    assert "lineup_availability_unknown" in bad["pass_reasons"]


def test_postseason_separated():
    policy = ops.load_frozen_policy()
    row = ops.evaluate_contract_before_outcome(
        _contract(requested_local_date="2026-09-29", game_type="F"),
        policy=policy,
        decision_time_utc="2026-10-05T12:00:00Z",
        lineup_started=True,
    )
    assert row["decision"]["cohort"].startswith("postseason")
    assert "postseason_cohort_separate" in row["pass_reasons"]


def test_outage_restart_drill(tmp_path):
    result = ops.outage_restart_drill(str(tmp_path))
    assert result["status"] == "passed"
    assert result["n_unique_decision_ids"] == 2
    assert result["exclusions_exist"] is True


def test_reconcile_and_history_integrity(tmp_path, monkeypatch):
    root = tmp_path / "prospective"
    monkeypatch.setattr(config, "HIT_PROP_PROSPECTIVE_DIR", str(root))
    policy = ops.load_frozen_policy()
    # Force a buy path with confirmed lineup and decision at receive time.
    contracts = [
        _contract(
            book_receive_time_utc="2099-05-31T22:00:00Z",
            scheduled_start_utc="2099-06-01T03:00:00Z",
        )
    ]
    cycle = ops.run_ops_cycle(
        contracts,
        policy=policy,
        decision_time_utc="2099-05-31T22:00:00Z",
        lineup_by_key={(100, 200): True},
        persist=True,
        store_kind="prospective",
        paths=ops.prospective_paths(str(root)),
        universe={"n_events": 1, "n_contracts_in_fetched_events": 1},
    )
    assert cycle["policy_version"] == policy["policy_version"]
    assert cycle["decisions"][0]["policy_hash"] == policy["policy_hash"]
    # With fresh quote + lineup, may buy; either way decision is persisted.
    assert (root / "decisions.csv").exists()

    rec = ops.reconcile_prospective(ops.prospective_paths(str(root)))
    assert rec["n_duplicate_decision_ids"] == 0
    assert rec["sim_separated"] is True

    integrity = ops.retrieve_history_integrity(ops.prospective_paths(str(root)))
    assert integrity["retrievable"] is True
    assert integrity["files"]["decisions"]["exists"] is True


def test_checkpoint_7_14_and_no_premature_eval():
    cp0 = ops.checkpoint_status(2)
    assert cp0["operational_checkpoint_7"] is False
    assert cp0["nested_performance_evaluation_allowed"] is False
    cp7 = ops.checkpoint_status(7)
    assert cp7["operational_checkpoint_7"] is True
    assert cp7["operational_checkpoint_14"] is False
    assert cp7["nested_performance_evaluation_allowed"] is False
    cp14 = ops.checkpoint_status(14)
    assert cp14["operational_checkpoint_14"] is True
    assert cp14["nested_performance_evaluation_allowed"] is False
    cp70 = ops.checkpoint_status(70)
    assert cp70["nested_performance_evaluation_allowed"] is True


def test_modes_fail_closed():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"


@pytest.mark.parametrize("game_type,expected", [("R", "regular_season"), ("F", "postseason"), ("D", "postseason"), ("L", "postseason"), ("W", "postseason"), (None, "unknown_season"), ("S", "unknown_season"), ("2026-10-01", "unknown_season")])
def test_cohort_uses_verified_game_type(game_type, expected):
    assert ops.classify_cohort(game_type) == expected


@pytest.mark.parametrize("receipt", [None, "bad", "2099-05-31T23:00:01Z"])
def test_missing_or_future_quote_cannot_buy(receipt):
    result = ops.evaluate_contract_before_outcome(
        _contract(book_receive_time_utc=receipt), policy=ops.load_frozen_policy(),
        decision_time_utc="2099-05-31T23:00:00Z", lineup_started=True)
    assert result["decision"]["action"] == "pass"
    assert "missing_or_future_quote_timestamp" in result["pass_reasons"]


def test_legacy_or_ineligible_dates_do_not_count(tmp_path):
    paths = ops.prospective_paths(str(tmp_path))
    pd.DataFrame([{"requested_local_date": "2026-09-29", "cohort": "regular_season"}]).to_csv(paths["decisions"], index=False)
    assert ops.eligible_decision_dates(paths) == []
    row = ops.evaluate_contract_before_outcome(_contract(), policy=ops.load_frozen_policy(), decision_time_utc="2099-05-31T23:00:00Z", lineup_started=True)["decision"]
    pd.DataFrame([row, {**row, "requested_local_date": "2099-06-01", "game_type": "F"}, {**row, "requested_local_date": "2099-06-02", "lineup_availability": "unknown"}]).to_csv(paths["decisions"], index=False)
    assert ops.eligible_decision_dates(paths) == []  # market-only baseline is not independent evidence


def test_lineup_exact_identity_timestamp_and_unknowns():
    contracts = [_contract(), _contract(game_pk=101), _contract(key_mlbam=201)]
    row = {"game_pk": 100, "key_mlbam": 200, "fetched_at_utc": "2099-05-31T22:59:00Z", "game_datetime": "2099-06-01T00:00:00Z", "source": "statsapi_schedule_lineups", "lineup_status": "confirmed", "is_confirmed_starter": True, "snapshot_id": "snapshot-test"}
    enriched, mapping = ops.attach_verified_lineups(contracts, pd.DataFrame([row]), decision_time_utc="2099-05-31T23:00:00Z")
    assert mapping == {(100, 200): True}
    assert enriched[0]["lineup_snapshot_id"] == "snapshot-test"
    assert "lineup_snapshot_id" not in contracts[0]
    for override in [{"fetched_at_utc": "2099-05-31T23:00:01Z"}, {"fetched_at_utc": "2099-05-31T22:00:00Z"}, {"game_datetime": "2099-05-31T23:00:00Z"}, {"is_confirmed_starter": pd.NA}, {"lineup_status": "unconfirmed"}]:
        _, mapping = ops.attach_verified_lineups(contracts, pd.DataFrame([{**row, **override}]), decision_time_utc="2099-05-31T23:00:00Z")
        assert mapping == {}
    _, mapping = ops.attach_verified_lineups(contracts, pd.DataFrame([row, row]), decision_time_utc="2099-05-31T23:00:00Z")
    assert mapping == {}


@pytest.mark.parametrize("overrides", [{"scheduled_start_utc": None}, {"scheduled_start_utc": "NaT"}, {"game_pk": None}, {"key_mlbam": 0}, {"game_pk": 1.5}, {"game_type_source": None}])
def test_incomplete_identity_or_schedule_cannot_buy(overrides):
    row = ops.evaluate_contract_before_outcome(_contract(**overrides), policy=ops.load_frozen_policy(), decision_time_utc="2099-05-31T23:00:00Z", lineup_started=True)
    assert row["decision"]["action"] == "pass"
