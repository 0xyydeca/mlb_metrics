"""Coverage provenance for the paper-ops command, using isolated stores."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "paper_ops_command", Path(__file__).parents[1] / "scripts/run_hit_prop_paper_ops.py"
)
command = importlib.util.module_from_spec(spec)
spec.loader.exec_module(command)


def setup_store(tmp_path, monkeypatch):
    monkeypatch.setattr(command.hit_prop_research, "research_store_dir", lambda: str(tmp_path))
    pd.DataFrame([
        {"capture_id": "older", "requested_local_date": "2026-09-18", "market_id": "a"},
        {"capture_id": "newer", "requested_local_date": "2026-09-19", "market_id": "b"},
    ]).to_csv(tmp_path / "contracts.csv", index=False)
    (tmp_path / "captures").mkdir()
    (tmp_path / "captures" / "zz-unrelated.json").write_text(json.dumps({
        "capture_id": "unrelated", "requested_local_date": "2026-09-20",
        "universe": {"n_events": 999},
    }))


def test_requested_slice_uses_its_own_capture(tmp_path, monkeypatch):
    setup_store(tmp_path, monkeypatch)
    (tmp_path / "captures" / "older.json").write_text(json.dumps({
        "capture_id": "older", "requested_local_date": "2026-09-18",
        "universe": {"n_events": 3, "n_rows_loaded": 999, "complete_denominator_claimed": True},
    }))
    rows, coverage = command._load_latest_contracts("2026-09-18")
    assert [r["market_id"] for r in rows] == ["a"]
    assert coverage["n_events"] == 3
    assert coverage["capture_id"] == "older"
    assert coverage["n_rows_loaded"] == 1
    assert coverage["complete_denominator_claimed"] is False
    assert coverage["capture_metadata_status"] == "matched"


@pytest.mark.parametrize("payload", [None, "broken JSON", [],
    {"capture_id": "wrong", "requested_local_date": "2026-09-18", "universe": {}},
    {"capture_id": "older", "requested_local_date": "2026-09-19", "universe": {}},
    {"capture_id": "older", "requested_local_date": "2026-09-18", "universe": []},
])
def test_missing_or_invalid_capture_never_borrows_coverage(tmp_path, monkeypatch, payload):
    setup_store(tmp_path, monkeypatch)
    if payload is not None:
        (tmp_path / "captures" / "older.json").write_text(
            payload if isinstance(payload, str) else json.dumps(payload)
        )
    rows, coverage = command._load_latest_contracts("2026-09-18")
    assert len(rows) == 1
    assert "n_events" not in coverage
    assert coverage["capture_metadata_status"] != "matched"
    assert coverage["capture_id"] == "older"


def test_no_contracts_for_date_does_not_attach_capture(tmp_path, monkeypatch):
    setup_store(tmp_path, monkeypatch)
    rows, coverage = command._load_latest_contracts("2026-09-30")
    assert rows == []
    assert "n_events" not in coverage
    assert coverage["n_rows_loaded"] == 0


def test_fresh_runner_uses_durable_summary(tmp_path, monkeypatch):
    setup_store(tmp_path, monkeypatch)
    (tmp_path / "capture_summaries").mkdir()
    (tmp_path / "capture_summaries" / "older.json").write_text(json.dumps({
        "capture_id": "older", "requested_local_date": "2026-09-18",
        "universe": {"n_events": 3},
    }))
    _, coverage = command._load_latest_contracts("2026-09-18")
    assert coverage["capture_metadata_status"] == "matched"
    assert coverage["n_events"] == 3
