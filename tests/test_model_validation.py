"""Nested rolling-origin validation: fold boundaries, no overlap,
preprocessing discipline, determinism, insufficient history, and paired
bootstrap alignment. Does not change live prediction behavior.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mlb_metrics import model_validation as mv


def _synthetic_classifier_frame(n_dates=40, rows_per_date=12, seed=0):
    rng = np.random.RandomState(seed)
    dates = [pd.Timestamp("2026-04-01") + pd.Timedelta(days=i) for i in range(n_dates)]
    rows = []
    for i, date in enumerate(dates):
        for j in range(rows_per_date):
            x1 = rng.normal()
            x2 = rng.normal()
            p = 1 / (1 + np.exp(-(0.8 * x1 - 0.3 * x2)))
            rows.append({
                "date": date,
                "game_pk": 1000 + i,
                "key_mlbam": 10_000 + j,
                "x1": x1,
                "x2": x2,
                "Got_Hit": int(rng.rand() < p),
                "Total_PA": 4,
                "Game_Hit_Probability": float(np.clip(p + rng.normal(scale=0.05), 0.01, 0.99)),
            })
    return pd.DataFrame(rows)


def test_rolling_origin_fold_boundaries_exact():
    dates = [f"2026-01-{d:02d}" for d in range(1, 21)]
    folds = mv.build_rolling_origin_folds(dates, min_train_dates=10, test_block_dates=5)
    assert len(folds) == 2
    assert folds[0].train_dates == tuple(dates[:10])
    assert folds[0].test_dates == tuple(dates[10:15])
    assert folds[1].train_dates == tuple(dates[:15])
    assert folds[1].test_dates == tuple(dates[15:20])
    assert folds[0].train_start == dates[0]
    assert folds[0].train_end == dates[9]
    assert folds[0].test_start == dates[10]
    assert folds[0].test_end == dates[14]


def test_rolling_origin_folds_never_overlap_and_are_chronological():
    dates = [f"2026-02-{d:02d}" for d in range(1, 31)]
    folds = mv.build_rolling_origin_folds(dates, min_train_dates=12, test_block_dates=4)
    assert folds
    for fold in folds:
        fold.assert_no_overlap()
        assert max(fold.train_dates) < min(fold.test_dates)


def test_minimum_training_dates_enforced():
    dates = [f"2026-03-{d:02d}" for d in range(1, 11)]
    assert mv.build_rolling_origin_folds(dates, min_train_dates=10, test_block_dates=5) == []
    # Exactly min_train + one test block worth of dates → one fold.
    dates20 = [f"2026-03-{d:02d}" for d in range(1, 16)]
    folds = mv.build_rolling_origin_folds(dates20, min_train_dates=10, test_block_dates=5)
    assert len(folds) == 1
    assert len(folds[0].train_dates) == 10


def test_nested_folds_inner_confined_to_outer_train_and_freeze_excluded():
    dates = [pd.Timestamp("2026-05-01") + pd.Timedelta(days=i) for i in range(50)]
    nested, freeze = mv.build_nested_folds(
        dates,
        outer_min_train_dates=20,
        outer_test_block_dates=5,
        inner_min_train_dates=10,
        inner_test_block_dates=5,
        freeze_dates=5,
    )
    assert freeze == dates[-5:]
    assert nested
    freeze_set = set(freeze)
    for item in nested:
        item.outer.assert_no_overlap()
        assert set(item.outer.test_dates).isdisjoint(freeze_set)
        assert set(item.outer.train_dates).isdisjoint(freeze_set)
        assert item.inner_folds
        for inner in item.inner_folds:
            inner.assert_no_overlap()
            assert set(inner.train_dates).issubset(item.outer.train_dates)
            assert set(inner.test_dates).issubset(item.outer.train_dates)
            assert set(inner.test_dates).isdisjoint(item.outer.test_dates)


def test_insufficient_history_returns_empty_status():
    df = _synthetic_classifier_frame(n_dates=8, rows_per_date=5)
    candidates = [
        mv.ModelCandidate(
            name="logit",
            estimator=LogisticRegression(max_iter=500),
            feature_columns=["x1", "x2"],
        )
    ]
    report = mv.run_nested_classifier_validation(
        df, candidates,
        mv.NestedValidationConfig(
            outer_min_train_dates=30, outer_test_block_dates=5,
            inner_min_train_dates=15, inner_test_block_dates=5,
            freeze_dates=0, bootstrap_samples=10, random_seed=0,
        ),
    )
    assert report["status"] == "insufficient_history"
    assert report["n_outer_folds"] == 0


def test_preprocessor_fit_only_on_training_data():
    train = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [1.0, 1.0, 1.0]})
    test = pd.DataFrame({"a": [100.0, 200.0], "b": [1.0, 1.0]})
    pre = mv.StandardizePreprocessor().fit(train)
    # Zero-variance b dropped from training fit.
    assert pre.columns_ == ["a"]
    transformed_test = pre.transform(test)
    # Uses training mean=2, std≈2 → (100-2)/std.
    assert list(transformed_test.columns) == ["a"]
    expected = (100.0 - 2.0) / train["a"].std()
    assert transformed_test.iloc[0, 0] == pytest.approx(expected)
    # Fitting on test would produce a different mean; prove we didn't.
    sneak = mv.StandardizePreprocessor().fit(test)
    assert sneak.mean_["a"] != pre.mean_["a"]
    mv.assert_preprocessor_fit_only_on_train(pre, train)


def test_imputer_fit_only_on_training_data_test_cannot_affect_medians():
    train = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    test = pd.DataFrame({
        "a": [1000.0, np.nan],
        "b": [np.nan, 999.0],
    })
    pre = mv.ImputeStandardizePreprocessor().fit(train)
    assert pre.median_["a"] == pytest.approx(2.0)
    assert "a" in pre.missing_indicator_cols_
    assert "b" not in pre.missing_indicator_cols_

    # Transforming test must not mutate fitted medians.
    out = pre.transform(test)
    assert pre.median_["a"] == pytest.approx(2.0)
    assert "a__missing" in out.columns
    assert list(out["a__missing"]) == [0.0, 1.0]

    # Fitting on train+test would pull the median of a toward 1000; prove we didn't.
    contaminated = pd.concat([train, test], ignore_index=True)
    sneak = mv.ImputeStandardizePreprocessor().fit(contaminated)
    assert sneak.median_["a"] != pre.median_["a"]
    mv.assert_imputer_fit_only_on_train(pre, train)


def test_nested_validation_deterministic_with_fixed_seed(tmp_path):
    df = _synthetic_classifier_frame(n_dates=36, rows_per_date=10, seed=1)
    candidates = mv.expand_candidate_grid(
        families=[{
            "name": "LogisticRegression",
            "estimator": LogisticRegression(max_iter=500),
            "param_grid": {"C": [0.5, 2.0]},
        }],
        feature_subsets=[["x1", "x2"], ["x1"]],
        calibration_methods=(None,),
        thresholds=(0.5,),
        shortlist_sizes=(None,),
        n_picks_options=(1,),
    )
    cfg = mv.NestedValidationConfig(
        outer_min_train_dates=12,
        outer_test_block_dates=4,
        inner_min_train_dates=8,
        inner_test_block_dates=4,
        freeze_dates=2,
        random_seed=7,
        bootstrap_samples=50,
        report_dir=str(tmp_path / "reports"),
    )
    first = mv.run_nested_classifier_validation(
        df, candidates, cfg, champion_probability_col="Game_Hit_Probability",
    )
    second = mv.run_nested_classifier_validation(
        df, candidates, cfg, champion_probability_col="Game_Hit_Probability",
    )
    assert first["status"] == "ok"
    assert first["aggregate"]["log_loss"] == second["aggregate"]["log_loss"]
    assert first["outer_folds"][0]["selected_candidate"] == second["outer_folds"][0]["selected_candidate"]
    if first["bootstrap_comparisons"]:
        assert first["bootstrap_comparisons"][0]["point_difference"] == second["bootstrap_comparisons"][0]["point_difference"]
        assert first["bootstrap_comparisons"][0]["ci_low"] == second["bootstrap_comparisons"][0]["ci_low"]


def test_paired_bootstrap_aligns_by_date_game_player():
    rng = np.random.RandomState(0)
    keys = []
    for d in range(5):
        for g in (1, 2):
            for p in (10, 20):
                keys.append({"date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=d), "game_pk": g, "key_mlbam": p})
    base = pd.DataFrame(keys)
    base["Got_Hit"] = rng.binomial(1, 0.4, size=len(base))
    a = base.copy()
    a["predicted_probability"] = np.clip(0.4 + rng.normal(scale=0.05, size=len(a)), 0.01, 0.99)
    b = base.copy()
    b["predicted_probability"] = np.clip(0.4 + rng.normal(scale=0.08, size=len(b)), 0.01, 0.99)
    # Drop one row from b → alignment must be inner join.
    b = b.iloc[1:].copy()
    result = mv.paired_block_bootstrap_difference(
        a, b,
        keys=["date", "game_pk", "key_mlbam"],
        block_key="date",
        label_col="Got_Hit",
        metric="log_loss",
        n_bootstrap=100,
        random_seed=0,
    )
    assert result["n_aligned_rows"] == len(base) - 1
    assert result["n_blocks"] == 5
    assert result["resampled_unit"] == "date"
    assert result["aligned_keys"] == ["date", "game_pk", "key_mlbam"]
    assert result["point_difference"] == result["point_difference"]  # not NaN


def test_write_validation_report_is_machine_readable(tmp_path):
    df = _synthetic_classifier_frame(n_dates=30, rows_per_date=8, seed=2)
    candidates = [
        mv.ModelCandidate(
            name="logit",
            estimator=LogisticRegression(max_iter=400),
            feature_columns=["x1", "x2"],
            n_picks=1,
        )
    ]
    cfg = mv.NestedValidationConfig(
        outer_min_train_dates=10,
        outer_test_block_dates=5,
        inner_min_train_dates=6,
        inner_test_block_dates=3,
        freeze_dates=0,
        bootstrap_samples=20,
        random_seed=1,
        report_dir=str(tmp_path / "reports"),
    )
    report = mv.run_nested_classifier_validation(df, candidates, cfg)
    path = mv.write_validation_report(report, cfg.report_dir)
    loaded = json.loads(Path(path).read_text())
    assert "outer_folds" in loaded
    assert "aggregate" in loaded
    assert "freeze_warning" in loaded
    for fold in loaded["outer_folds"]:
        assert "train_start" in fold and "test_end" in fold
        assert "log_loss" in fold
        assert "brier_score" in fold
        assert "roc_auc" in fold
        assert "calibration_intercept" in fold
        assert "coverage" in fold


def test_expand_candidate_grid_includes_policy_and_feature_knobs():
    candidates = mv.expand_candidate_grid(
        families=[{
            "name": "LogisticRegression",
            "estimator": LogisticRegression(max_iter=200),
            "param_grid": {"C": [1.0]},
        }],
        feature_subsets=[["x1"], ["x1", "x2"]],
        calibration_methods=(None, "sigmoid"),
        thresholds=(0.4, 0.5),
        shortlist_sizes=(None, 10),
        n_picks_options=(1, 2),
    )
    # 1 param × 2 feature × 2 calib × 2 thr × 2 shortlist × 2 picks
    assert len(candidates) == 32
    assert any(c.shortlist_size == 10 and c.n_picks == 2 for c in candidates)


def test_freeze_warning_present_when_freeze_reserved():
    df = _synthetic_classifier_frame(n_dates=35, rows_per_date=6, seed=3)
    candidates = [
        mv.ModelCandidate(
            name="logit",
            estimator=LogisticRegression(max_iter=300),
            feature_columns=["x1", "x2"],
        )
    ]
    report = mv.run_nested_classifier_validation(
        df, candidates,
        mv.NestedValidationConfig(
            outer_min_train_dates=12, outer_test_block_dates=4,
            inner_min_train_dates=8, inner_test_block_dates=4,
            freeze_dates=3, bootstrap_samples=10, random_seed=0,
        ),
    )
    assert len(report["freeze_dates"]) == 3
    assert "must not" in report["freeze_warning"].lower() or "Do not" in report["freeze_warning"]
