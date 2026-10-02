"""Hitter export: exact game identity, dated outputs, and explicit unavailable states."""

import json
import datetime
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from mlb_metrics import ml_models

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_hitter_hit_predictions.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_hitter_hit_predictions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ConstantProbaModel:
    def __init__(self, positive_proba):
        self.positive_proba = positive_proba

    def predict_proba(self, X):
        import numpy as np
        return np.column_stack([1 - np.full(len(X), self.positive_proba), np.full(len(X), self.positive_proba)])


def _write_daily_csvs(data_dir, last_game_date="2026-06-20"):
    wave = pd.DataFrame([{
        "key_mlbam": 1, "name_first": "Test", "name_last": "Hitter", "team": "BOS",
        "PA_L": 20, "PA_R": 20, "WAVE": 0.28, "WAVE_L": 0.28, "WAVE_R": 0.28,
        "probability_L": 0.6, "probability_R": 0.6, "probability": 0.6,
        "Game_Hit_Probability": 0.70, "Consistency": 0.1, "Approach": 0.4, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
        "Last_Game_Date": last_game_date,
    }])
    pave = pd.DataFrame([{
        "key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "team": "NYY",
        "at_bats": 100, "Throws": "R", "PAVE": 0.24, "PAVE_PLUS": 1.0, "Power_A_PLUS": 1.0,
        "Expected_Hits": 1.0, "Expected_Bases": 1.5, "Expected_HRs": 0.1,
    }])
    confidence = pd.DataFrame([
        {"team": "BOS", "Bullpen_PAVE": 0.25, "Park_Factor": 1.0},
        {"team": "NYY", "Bullpen_PAVE": 0.25, "Park_Factor": 1.0},
    ])
    wave.to_csv(data_dir / "wave.csv", index=False)
    pave.to_csv(data_dir / "pave.csv", index=False)
    confidence.to_csv(data_dir / "confidence.csv", index=False)


def _schedule_df():
    return pd.DataFrame([
        {"date": "2026-06-20", "team": "BOS", "opponent": "NYY", "probable_pitcher_key_mlbam": 99, "game_pk": 1, "is_home": False},
        {"date": "2026-06-20", "team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": pd.NA, "game_pk": 1, "is_home": True},
    ])


def test_build_hitter_hit_predictions_writes_csv(tmp_path, monkeypatch):
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_csvs(data_dir)

    model_path = str(tmp_path / "model.joblib")
    ml_models.save_model(_ConstantProbaModel(0.61), model_path)
    monkeypatch.setattr(module.config, "HITTER_HIT_PROBABILITY_MODEL_PATH", model_path)
    monkeypatch.setattr(module.schedule, "fetch_hitter_schedule", lambda date: _schedule_df())

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    result = pd.read_csv(data_dir / "hitter_hit_predictions.csv")
    assert len(result) == 1
    assert result.iloc[0]["key_mlbam"] == 1
    assert result.iloc[0]["Model_Hit_Probability"] == 0.61
    assert result.iloc[0]["opponent"] == "NYY"


def test_build_hitter_hit_predictions_excludes_hitter_who_has_not_played_recently(tmp_path, monkeypatch):
    # A season-long injured-list stay (e.g. Shea Langeliers, reported
    # 2026-08-04) leaves season-to-date PA/rates looking qualified, but the
    # hitter hasn't actually played in days/weeks - must be excluded from
    # hitter_hit_predictions.csv, mirroring test_build_dfs_rankings_script.py's
    # equivalent DFS Hitters test.
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # 11 days before the 2026-06-20 as-of-date used below - well past
    # HITTER_MAX_DAYS_SINCE_LAST_GAME (5).
    _write_daily_csvs(data_dir, last_game_date="2026-06-09")

    model_path = str(tmp_path / "model.joblib")
    ml_models.save_model(_ConstantProbaModel(0.61), model_path)
    monkeypatch.setattr(module.config, "HITTER_HIT_PROBABILITY_MODEL_PATH", model_path)
    monkeypatch.setattr(module.schedule, "fetch_hitter_schedule", lambda date: _schedule_df())

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    # No qualified hitters must publish an empty current export.
    assert pd.read_csv(data_dir / "hitter_hit_predictions.csv").empty


def test_build_hitter_hit_predictions_keeps_hitter_within_recency_window(tmp_path, monkeypatch):
    # Boundary check: exactly at the 5-day cutoff must still be included
    # (predictions.select_picks's own gate uses the same <= semantics).
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_csvs(data_dir, last_game_date="2026-06-15")  # exactly 5 days before as-of-date

    model_path = str(tmp_path / "model.joblib")
    ml_models.save_model(_ConstantProbaModel(0.61), model_path)
    monkeypatch.setattr(module.config, "HITTER_HIT_PROBABILITY_MODEL_PATH", model_path)
    monkeypatch.setattr(module.schedule, "fetch_hitter_schedule", lambda date: _schedule_df())

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    result = pd.read_csv(data_dir / "hitter_hit_predictions.csv")
    assert len(result) == 1
    assert result.iloc[0]["key_mlbam"] == 1


def test_build_hitter_hit_predictions_missing_daily_csvs_clears_export(tmp_path, monkeypatch):
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    assert pd.read_csv(data_dir / "hitter_hit_predictions.csv").empty


def test_build_hitter_hit_predictions_failed_schedule_clears_stale_file(tmp_path, monkeypatch):
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    (data_dir / "hitter_hit_predictions.csv").write_text("stale,data\n1,2\n")

    def _boom(date):
        raise RuntimeError("statsapi is down")

    monkeypatch.setattr(module.schedule, "fetch_hitter_schedule", _boom)

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    assert pd.read_csv(data_dir / "hitter_hit_predictions.csv").empty


def test_build_hitter_hit_predictions_missing_model_clears_stale_file(tmp_path, monkeypatch):
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    (data_dir / "hitter_hit_predictions.csv").write_text("stale,data\n1,2\n")

    monkeypatch.setattr(module.config, "HITTER_HIT_PROBABILITY_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(module.schedule, "fetch_hitter_schedule", lambda date: _schedule_df())

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    assert pd.read_csv(data_dir / "hitter_hit_predictions.csv").empty


def test_doubleheader_predictions_keep_game_identity(tmp_path, monkeypatch):
    module = _load_module()
    _write_daily_csvs(tmp_path)
    model_path = str(tmp_path / "model.joblib")
    ml_models.save_model(_ConstantProbaModel(0.61), model_path)
    monkeypatch.setattr(module.config, "HITTER_HIT_PROBABILITY_MODEL_PATH", model_path)
    games = pd.concat([_schedule_df(), _schedule_df().assign(game_pk=2)], ignore_index=True)
    monkeypatch.setattr(module.schedule, "fetch_hitter_schedule", lambda date: games)
    monkeypatch.setattr(sys, "argv", ["script", "--data-dir", str(tmp_path), "--as-of-date", "2026-06-20"])
    module.main()
    result = pd.read_csv(tmp_path / "hitter_hit_predictions.csv")
    assert len(result) == 2
    assert set(result.game_pk) == {1, 2}
    assert result.forecast_date.eq("2026-06-20").all()
    assert not result.duplicated(["game_pk", "key_mlbam"]).any()


def test_empty_schedule_clears_stale_export_with_dated_status(tmp_path, monkeypatch):
    module = _load_module()
    _write_daily_csvs(tmp_path)
    (tmp_path / "hitter_hit_predictions.csv").write_text("stale,data\n1,2\n")
    monkeypatch.setattr(module.schedule, "fetch_hitter_schedule", lambda date: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["script", "--data-dir", str(tmp_path), "--as-of-date", "2026-06-20"])
    module.main()
    assert pd.read_csv(tmp_path / "hitter_hit_predictions.csv").empty
    status = json.loads((tmp_path / "hitter_hit_predictions_status.json").read_text())
    assert status["forecast_date"] == "2026-06-20"
    assert status["status"] == "no_scheduled_games"
    assert status["n_rows"] == 0
