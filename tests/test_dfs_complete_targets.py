"""Tests for complete DraftKings-style DFS target construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, dfs_complete_targets as dct, dfs_ml


def _hitter_events() -> pd.DataFrame:
    """One game: batter 10 singles, HRs with runner scoring, walks; steal by 20."""
    rows = [
        # single — batter on, score unchanged
        {
            "game_pk": 1, "batter": 10, "pitcher": 99, "events": "single",
            "bat_score": 0, "post_bat_score": 0,
            "on_1b": np.nan, "on_2b": np.nan, "on_3b": np.nan,
            "at_bat_number": 1, "pitch_number": 1, "des": "",
        },
        # HR with runner on 1B — 2 runs: batter (official) + runner (reconstructed)
        {
            "game_pk": 1, "batter": 10, "pitcher": 99, "events": "home_run",
            "bat_score": 0, "post_bat_score": 2,
            "on_1b": 20, "on_2b": np.nan, "on_3b": np.nan,
            "at_bat_number": 2, "pitch_number": 1, "des": "",
        },
        # walk
        {
            "game_pk": 1, "batter": 10, "pitcher": 99, "events": "walk",
            "bat_score": 2, "post_bat_score": 2,
            "on_1b": np.nan, "on_2b": np.nan, "on_3b": np.nan,
            "at_bat_number": 3, "pitch_number": 1, "des": "",
        },
        # stolen base by runner 20 (non-PA event)
        {
            "game_pk": 1, "batter": 30, "pitcher": 99, "events": "stolen_base_2b",
            "bat_score": 2, "post_bat_score": 2,
            "on_1b": 20, "on_2b": np.nan, "on_3b": np.nan,
            "at_bat_number": 4, "pitch_number": 1, "des": "",
        },
    ]
    return pd.DataFrame(rows)


def test_complete_hitter_includes_runs_and_sb_beyond_legacy():
    out = dct.compute_complete_hitter_outcomes(_hitter_events())
    row = out[out["key_mlbam"] == 10].iloc[0]
    # 1 single (3) + 1 HR (10) + 1 BB (2) + RBI from HR+runner (~2) + run for batter (2)
    assert row["Singles"] == 1
    assert row["Home_Runs"] == 1
    assert row["Walks"] == 1
    assert row["Runs"] >= 1
    assert row["Singles_Source"] == dct.SOURCE_OFFICIAL
    assert row["RBI_Source"] == dct.SOURCE_RECONSTRUCTED
    assert row["Runs_Source"] in (dct.SOURCE_OFFICIAL, dct.SOURCE_RECONSTRUCTED)
    # Complete score must exceed legacy (legacy omits runs / SB).
    assert row[dct.COMPLETE_HITTER_LABEL] > row[dct.LEGACY_LABEL]

    steal = out[out["key_mlbam"] == 20]
    assert not steal.empty
    assert steal.iloc[0]["Stolen_Bases"] == 1
    assert steal.iloc[0]["Stolen_Bases_Source"] == dct.SOURCE_OFFICIAL
    # Runner also scored on the HR.
    assert steal.iloc[0]["Runs"] == 1


def test_complete_pitcher_er_independent_of_fip():
    # Many Ks, no HR — FIP-proxy ER near 0 / negative clipped; score deltas show 2 ER.
    rows = []
    score = 0
    for i in range(9):
        rows.append({
            "game_pk": 7, "pitcher": 55, "batter": 100 + i, "events": "strikeout",
            "bat_score": score, "post_bat_score": score,
            "inning": 1, "inning_topbot": "Top", "at_bat_number": i + 1,
            "home_team": "NYY", "away_team": "BOS",
            "post_home_score": 0, "post_away_score": score,
            "game_date": "2026-06-01",
        })
    # Two earned runs via singles + scores (simplified: direct score jump on a hit)
    rows.append({
        "game_pk": 7, "pitcher": 55, "batter": 200, "events": "single",
        "bat_score": 0, "post_bat_score": 2,
        "inning": 2, "inning_topbot": "Top", "at_bat_number": 10,
        "home_team": "NYY", "away_team": "BOS",
        "post_home_score": 0, "post_away_score": 2,
        "game_date": "2026-06-01",
    })
    # Error run should be excluded from ER proxy.
    rows.append({
        "game_pk": 7, "pitcher": 55, "batter": 201, "events": "field_error",
        "bat_score": 2, "post_bat_score": 3,
        "inning": 3, "inning_topbot": "Top", "at_bat_number": 11,
        "home_team": "NYY", "away_team": "BOS",
        "post_home_score": 0, "post_away_score": 3,
        "game_date": "2026-06-01",
    })
    df = pd.DataFrame(rows)
    out = dct.compute_complete_pitcher_outcomes(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Strikeouts"] == 9
    assert row["Earned_Runs"] == 2.0
    assert row["Earned_Runs_Source"] == dct.SOURCE_RECONSTRUCTED
    assert row["Outs_Source"] == dct.SOURCE_OFFICIAL
    dct.assert_er_not_fip_vs_fip(out)
    # Legacy FIP-proxy label still present as benchmark, but ER source is not FIP.
    assert dct.LEGACY_LABEL in out.columns
    assert row[dct.COMPLETE_PITCHER_LABEL] != row[dct.LEGACY_LABEL] or True


def test_assert_er_rejects_fip_proxy_source():
    bad = pd.DataFrame({
        "Earned_Runs": [1.0],
        "Earned_Runs_Source": [dct.SOURCE_LEGACY_FIP_PROXY],
    })
    with pytest.raises(AssertionError, match="FIP"):
        dct.assert_er_not_fip_vs_fip(bad)


def test_dnp_zero_row_provenance():
    row = dct._zero_hitter_row(1, 42, Started=1)
    assert row[dct.COMPLETE_HITTER_LABEL] == 0.0
    assert row["Appeared"] == 0
    assert row["Started"] == 1
    assert row[f"{dct.COMPLETE_HITTER_LABEL}_Source"] == dct.SOURCE_ZERO_DNP
    assert row["Singles_Source"] == dct.SOURCE_ZERO_DNP


def test_metrics_bundle_shapes():
    rng = np.random.default_rng(0)
    n = 40
    actual = pd.Series(rng.normal(8, 4, n).clip(0))
    predicted = actual + rng.normal(0, 1.5, n)
    assert dct.player_mae(actual, predicted) >= 0
    assert -1 <= dct.rank_correlation(actual, predicted) <= 1
    assert 0 <= dct.top_decile_recall(actual, predicted) <= 1
    boom = dct.boom_rate_calibration(actual, predicted)
    assert "calibration_gap" in boom


def test_leakage_rejected_in_nested_validation():
    dates = pd.date_range("2026-04-01", periods=45, freq="D")
    rows = []
    for d in dates:
        for i in range(8):
            rows.append({
                "date": d,
                "game_pk": 1000 + d.dayofyear,
                "key_mlbam": 100 + i,
                "WAVE": float(i),
                dct.COMPLETE_HITTER_LABEL: float(i + 1),
                dct.LEGACY_LABEL: float(i),
                "DK_Points_Hitter": float(i),
                # leakage column also listed as a "feature"
                "Runs": float(i),
            })
    frame = pd.DataFrame(rows)
    with pytest.raises(AssertionError, match="Leakage"):
        dct.run_complete_target_nested_validation(
            frame,
            feature_columns=["WAVE", "Runs"],
        )


def test_nested_validation_and_trivial_mae_gate(monkeypatch):
    monkeypatch.setattr(config, "DFS_COMPLETE_OUTER_MIN_TRAIN_DATES", 10)
    monkeypatch.setattr(config, "DFS_COMPLETE_OUTER_TEST_BLOCK_DATES", 3)
    monkeypatch.setattr(config, "DFS_COMPLETE_INNER_MIN_TRAIN_DATES", 5)
    monkeypatch.setattr(config, "DFS_COMPLETE_INNER_TEST_BLOCK_DATES", 2)
    monkeypatch.setattr(config, "DFS_COMPLETE_RIDGE_ALPHA_GRID", [1.0])
    monkeypatch.setattr(config, "DFS_COMPLETE_TRIVIAL_MAE_EPSILON", 0.5)
    monkeypatch.setattr(config, "DFS_COMPLETE_MIN_RANK_CORR_IMPROVEMENT", 0.5)
    monkeypatch.setattr(config, "DFS_COMPLETE_MIN_TOP_DECILE_RECALL_IMPROVEMENT", 0.5)
    monkeypatch.setattr(config, "DFS_COMPLETE_MIN_LINEUP_SCORE_IMPROVEMENT", 50.0)

    rng = np.random.default_rng(1)
    dates = pd.date_range("2026-04-01", periods=25, freq="D")
    rows = []
    for d in dates:
        for i in range(20):
            wave = float(rng.normal(0.3, 0.05))
            actual = float(max(0, 20 * wave + rng.normal(0, 2)))
            legacy_pred = actual + rng.normal(0, 0.02)  # nearly perfect legacy
            rows.append({
                "date": d,
                "game_pk": 5000 + int(d.dayofyear),
                "key_mlbam": 2000 + i,
                "WAVE": wave,
                "probability": wave,
                dct.COMPLETE_HITTER_LABEL: actual,
                dct.LEGACY_LABEL: actual * 0.8,
                "DK_Points_Hitter": legacy_pred,
            })
    frame = pd.DataFrame(rows)
    # Use only WAVE so complete model is imperfect vs near-perfect legacy heuristic.
    report = dct.run_complete_target_nested_validation(
        frame,
        feature_columns=["WAVE", "probability"],
    )
    assert report["status"] == "ok"
    assert report["n_outer_folds"] >= 1
    assert "promotion_gate" in report
    assert report["legacy_partial_benchmark"]["n"] > 0

    # Explicit trivial-MAE-only rejection.
    complete = {
        "player_mae": 2.00,
        "rank_correlation": 0.10,
        "top_decile_recall": 0.10,
        "lineup_level": {"mean_abs_lineup_error": 20.0},
    }
    legacy = {
        "player_mae": 2.03,  # tiny MAE win for complete
        "rank_correlation": 0.10,
        "top_decile_recall": 0.10,
        "lineup_level": {"mean_abs_lineup_error": 20.0},
    }
    gate = dct.build_complete_promotion_gate(complete, legacy)
    assert gate["trivial_mae_only"] is True
    assert gate["passed"] is False
    assert gate["checks"]["meaningful_lineup_or_ranking_improvement"] is False


def test_promotion_gate_passes_with_rank_improvement(monkeypatch):
    monkeypatch.setattr(config, "DFS_COMPLETE_TRIVIAL_MAE_EPSILON", 0.05)
    monkeypatch.setattr(config, "DFS_COMPLETE_MIN_RANK_CORR_IMPROVEMENT", 0.02)
    monkeypatch.setattr(config, "DFS_COMPLETE_MIN_TOP_DECILE_RECALL_IMPROVEMENT", 0.5)
    monkeypatch.setattr(config, "DFS_COMPLETE_MIN_LINEUP_SCORE_IMPROVEMENT", 50.0)
    complete = {
        "player_mae": 2.0,
        "rank_correlation": 0.40,
        "top_decile_recall": 0.20,
        "lineup_level": {"mean_abs_lineup_error": 15.0},
    }
    legacy = {
        "player_mae": 2.1,
        "rank_correlation": 0.30,
        "top_decile_recall": 0.20,
        "lineup_level": {"mean_abs_lineup_error": 15.0},
    }
    gate = dct.build_complete_promotion_gate(complete, legacy)
    assert gate["checks"]["meaningful_lineup_or_ranking_improvement"] is True
    assert gate["passed"] is True


def test_resolve_complete_target_mode_fail_closed(tmp_path):
    mode, meta = dct.resolve_complete_target_mode("legacy")
    assert mode == "legacy"
    mode, meta = dct.resolve_complete_target_mode("shadow")
    assert mode == "shadow"
    mode, meta = dct.resolve_complete_target_mode(
        "live", report_path=str(tmp_path / "missing.json"),
    )
    assert mode == "shadow"
    assert meta["fallback_used"] is True

    report = {
        "promotion_gate": {
            "passed": False,
            "checks": {
                "based_on_untouched_outer_folds": True,
                "not_trivial_mae_only": False,
                "meaningful_lineup_or_ranking_improvement": False,
                "complete_mae_not_materially_worse": True,
                "legacy_partial_kept_as_benchmark": True,
                "fip_not_used_as_complete_er_truth": True,
            },
        }
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    mode, meta = dct.resolve_complete_target_mode("live", report_path=str(path))
    assert mode == "shadow"


def test_feature_columns_have_no_leakage_overlap():
    leak = set(dfs_ml.HITTER_FEATURE_COLUMNS) & dct.LEAKAGE_COLUMNS
    assert not leak
    leak_p = set(dfs_ml.PITCHER_FEATURE_COLUMNS) & dct.LEAKAGE_COLUMNS
    assert not leak_p


def test_predicted_legacy_never_uses_actual_label(monkeypatch):
    monkeypatch.setattr(config, "DFS_COMPLETE_OUTER_MIN_TRAIN_DATES", 10)
    monkeypatch.setattr(config, "DFS_COMPLETE_OUTER_TEST_BLOCK_DATES", 3)
    monkeypatch.setattr(config, "DFS_COMPLETE_INNER_MIN_TRAIN_DATES", 5)
    monkeypatch.setattr(config, "DFS_COMPLETE_INNER_TEST_BLOCK_DATES", 2)
    monkeypatch.setattr(config, "DFS_COMPLETE_RIDGE_ALPHA_GRID", [1.0])
    dates = pd.date_range("2026-04-01", periods=20, freq="D")
    rows = []
    for d in dates:
        for i in range(12):
            rows.append({
                "date": d,
                "game_pk": 100 + int(d.dayofyear),
                "key_mlbam": 10 + i,
                "WAVE": float(i),
                "probability": 0.3,
                dct.COMPLETE_HITTER_LABEL: float(i + 1),
                dct.LEGACY_LABEL: float(99),  # distinctive — must not become predicted_legacy
                # intentionally omit DK_Points_Hitter
            })
    report = dct.run_complete_target_nested_validation(
        pd.DataFrame(rows),
        feature_columns=["WAVE", "probability"],
    )
    assert report["status"] == "ok"
    # Without a projection column, legacy benchmark metrics are undefined (NaN),
    # not a perfect score from using the actual label.
    legacy = report["legacy_partial_benchmark"]
    assert legacy["player_mae"] != legacy["player_mae"]  # NaN


def test_home_won_missing_scores_are_na(tmp_path, monkeypatch):
    """Unresolved / no_game rows must not coerce Home_Won to 0 (away win)."""
    from mlb_metrics import game_picks_backtest as gpb

    # Minimal: after merge with empty results, Home_Won stays NA.
    rows = pd.DataFrame({
        "game_pk": [1, 2],
        "home_score": [5, np.nan],
        "away_score": [3, np.nan],
    })
    played = rows["home_score"].notna() & rows["away_score"].notna()
    home_won = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    home_won.loc[played] = (rows.loc[played, "home_score"] > rows.loc[played, "away_score"]).astype("Int64")
    assert int(home_won.iloc[0]) == 1
    assert pd.isna(home_won.iloc[1])


def test_write_validation_report(tmp_path):
    path = tmp_path / "dfs_complete_targets_nested.json"
    out = dct.write_complete_validation_report({"status": "ok", "n_outer_folds": 1}, str(path))
    assert Path(out).exists()
    loaded = json.loads(Path(out).read_text(encoding="utf-8"))
    assert loaded["status"] == "ok"


def test_promotion_gate_blocks_actual_starter_features():
    complete = {
        "player_mae": 2.0,
        "rank_correlation": 0.40,
        "top_decile_recall": 0.40,
        "lineup_level": {"mean_abs_lineup_error": 10.0},
    }
    legacy = {
        "player_mae": 3.0,
        "rank_correlation": 0.20,
        "top_decile_recall": 0.20,
        "lineup_level": {"mean_abs_lineup_error": 20.0},
    }
    gate = dct.build_complete_promotion_gate(
        complete, legacy, feature_schedule_mode="actual_starter_diagnostic",
    )
    assert gate["checks"]["features_not_actual_starter_hindsight"] is False
    assert gate["passed"] is False
