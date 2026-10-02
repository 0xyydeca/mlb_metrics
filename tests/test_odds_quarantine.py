"""Unresolved sportsbook identities remain auditable and never enter serving."""
import pandas as pd
import pytest
from mlb_metrics import market_odds as odds


def row(sid, game_pk, status="ok"):
    return {"snapshot_id": sid, "game_pk": game_pk, "source_status": status,
            "provider_event_id": "fixture", "home_team": "NYY", "away_team": "BAL",
            "home_moneyline": -110, "away_moneyline": 100}


def test_quarantine_is_durable_idempotent_and_retains_original_fields(tmp_path):
    path = str(tmp_path / "odds.csv")
    frame = pd.DataFrame([row("good", 123), row("ambiguous", None, "ambiguous_match")])
    out = odds.append_odds_snapshots(frame, path)
    assert list(out.snapshot_id) == ["good"]
    quarantine = pd.read_csv(tmp_path / "odds_quarantine.csv")
    assert list(quarantine.snapshot_id) == ["ambiguous"]
    assert quarantine.loc[0, "source_status"] == "ambiguous_match"
    assert pd.isna(quarantine.loc[0, "game_pk"])
    assert quarantine.loc[0, "home_moneyline"] == -110
    odds.append_odds_snapshots(frame, path)
    assert len(pd.read_csv(path)) == 1
    assert len(pd.read_csv(tmp_path / "odds_quarantine.csv")) == 1


@pytest.mark.parametrize("game_pk,status", [(None,"ok"),(0,"ok"),(-1,"ok"),(1.5,"ok"),(123,"unmatched"),(123,"ambiguous_match")])
def test_invalid_identity_never_enters_primary(tmp_path, game_pk, status):
    out = odds.append_odds_snapshots(pd.DataFrame([row("bad",game_pk,status)]), str(tmp_path / "odds.csv"))
    assert out.empty
    assert len(pd.read_csv(tmp_path / "odds_quarantine.csv")) == 1


def test_migration_preserves_rejected_rows_before_primary_write(tmp_path, monkeypatch):
    path = str(tmp_path / "odds.csv")
    original = odds.normalize_snapshot_frame(pd.DataFrame([row("good",123),row("bad",None)]))
    original.to_csv(path,index=False)
    before = (tmp_path / "odds.csv").read_bytes()
    real_write = odds._write_snapshot_file
    def fail_primary(frame, destination):
        if destination == path:
            raise OSError("fixture primary write failure")
        real_write(frame,destination)
    monkeypatch.setattr(odds,"_write_snapshot_file",fail_primary)
    with pytest.raises(OSError):
        odds.append_odds_snapshots(pd.DataFrame(),path)
    assert (tmp_path / "odds.csv").read_bytes() == before
    assert list(pd.read_csv(tmp_path / "odds_quarantine.csv").snapshot_id) == ["bad"]


def test_existing_invalid_rows_migrate_without_new_capture(tmp_path):
    path = str(tmp_path / "odds.csv")
    odds.normalize_snapshot_frame(pd.DataFrame([row("good",123),row("bad",None)])).to_csv(path,index=False)
    out=odds.append_odds_snapshots(pd.DataFrame(),path)
    assert list(out.snapshot_id)==["good"]
    assert list(pd.read_csv(tmp_path / "odds_quarantine.csv").snapshot_id)==["bad"]
