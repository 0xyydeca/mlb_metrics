"""Champion/challenger Final_Hit_Probability selection-mode integration."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, evaluation, hitter_probability_model as hpm, predictions


def _pool(rows):
    """rows: (key, ghp, approach, final_hit, model_hit=None)."""
    out = []
    for item in rows:
        key, ghp, approach, final_hit = item[:4]
        model_hit = item[4] if len(item) > 4 else None
        out.append({
            "key_mlbam": key,
            "name_first": f"F{key}",
            "name_last": f"L{key}",
            "team": "NYY",
            "game_pk": 1000 + key,
            "PA_L": 20,
            "PA_R": 20,
            "probability": ghp,
            "Game_Hit_Probability": ghp,
            "Approach": approach,
            "Matchup_Approach": approach,
            "Matchup_Hit_Probability": ghp,
            "Final_Hit_Probability": final_hit,
            "P_Appear": 0.9,
            "Expected_PA_hat": 4.0,
            "P_Hit_Given_Appearance": final_hit / 0.9 if final_hit else 0.0,
            "Model_Hit_Probability": model_hit if model_hit is not None else ghp,
            "avg_batting_order": 2.0,
            "start_rate": 0.9,
        })
    return pd.DataFrame(out)


def test_config_default_is_shadow_without_promotion_gate():
    assert config.HITTER_SELECTION_MODE == "shadow"
    ok, details = hpm.promotion_gate_satisfied()
    assert ok is False
    assert details["passed"] is False


def test_live_configured_without_gate_falls_back_to_shadow():
    mode, meta = hpm.resolve_hitter_selection_mode("live", force_live=False)
    assert mode == "shadow"
    assert meta["fallback_used"] is True
    assert "promotion_gate" in meta["fallback_reason"] or "gate" in meta["fallback_reason"]


def test_promotion_gate_requires_all_checks():
    ok, details = hpm.evaluate_promotion_gate({"promotion_gate": {
        "lower_log_loss_than_legacy": True,
        "lower_brier_than_legacy": True,
        "non_inferior_coverage": True,
        "no_worse_no_game_rate": True,
        "improved_or_non_inferior_top_one_advance": True,
        "improved_expected_streak_utility": False,
        "no_material_high_tail_calibration_degradation": True,
        "based_on_untouched_outer_folds": True,
    }})
    assert ok is False
    assert details["checks"]["improved_expected_streak_utility"] is False

    ok2, _ = hpm.evaluate_promotion_gate({"promotion_gate": {
        "lower_log_loss_than_legacy": True,
        "lower_brier_than_legacy": True,
        "non_inferior_coverage": True,
        "no_worse_no_game_rate": True,
        "improved_or_non_inferior_top_one_advance": True,
        "improved_expected_streak_utility": True,
        "no_material_high_tail_calibration_degradation": True,
        "based_on_untouched_outer_folds": True,
    }})
    assert ok2 is True


def test_legacy_mode_preserves_shortlist_and_split_rank_log():
    # Model shortlist (size 2) keeps keys 1 and 2; Matchup_Approach then
    # prefers key 2. Logged probability stays Game_Hit_Probability.
    hitters = _pool([
        (1, 0.85, 0.50, 0.40, 0.99),
        (2, 0.80, 0.95, 0.99, 0.90),
        (3, 0.75, 0.99, 0.70, 0.10),  # highest Approach but excluded by shortlist
    ])
    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30,
        rank_metric="Matchup_Approach",
        selection_mode="legacy",
        model_shortlist_size=2,
    )
    assert list(picks["key_mlbam"]) == [2]
    assert picks.iloc[0]["selection_metric"] == "Matchup_Approach"
    assert picks.iloc[0]["probability_source"] == "Game_Hit_Probability"
    assert picks.iloc[0]["predicted_probability"] == 0.80
    assert picks.iloc[0]["selection_mode"] == "legacy"


def test_shadow_mode_preserves_official_picks_and_logs_challenger():
    hitters = _pool([
        (1, 0.90, 0.95, 0.40),
        (2, 0.70, 0.50, 0.95),
    ])
    hitters = hpm.attach_shadow_ranks(hitters)
    legacy = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30,
        rank_metric="Matchup_Approach", selection_mode="legacy",
    )
    shadow = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30,
        rank_metric="Matchup_Approach", selection_mode="shadow",
        shadow_model_status={"artifact_id": "shadow-art-1"},
    )
    assert list(shadow["key_mlbam"]) == list(legacy["key_mlbam"])
    assert shadow.iloc[0]["predicted_probability"] == legacy.iloc[0]["predicted_probability"]
    assert shadow.iloc[0]["selection_metric"] == "Matchup_Approach"
    assert shadow.iloc[0]["Final_Hit_Probability"] == 0.40
    assert shadow.iloc[0]["shadow_rank"] == 2  # key 2 has higher Final_Hit
    assert shadow.iloc[0]["shadow_model_artifact_id"] == "shadow-art-1"
    assert shadow.iloc[0]["P_Appear"] == 0.9


def test_live_mode_single_authoritative_column():
    """Rank, threshold, log, score, and display all use Final_Hit_Probability."""
    hitters = _pool([
        # High GHP / Approach / Model — but low Final_Hit → must NOT win live
        (1, 0.95, 0.99, 0.55, 0.99),
        # Lower legacy signals — high Final_Hit → MUST win live
        (2, 0.70, 0.40, 0.92, 0.10),
        (3, 0.60, 0.30, 0.50, 0.20),
    ])
    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=2, min_plate_appearances=30,
        selection_mode="live", force_live=True,
        min_probability=0.5,
        rank_metric="Matchup_Approach",  # must be ignored
    )
    assert list(picks["key_mlbam"]) == [2, 1]
    row = picks.iloc[0]
    assert row["metric"] == "Final_Hit_Probability"
    assert row["selection_metric"] == "Final_Hit_Probability"
    assert row["probability_source"] == "Final_Hit_Probability"
    assert row["selection_score"] == row["Final_Hit_Probability"]
    assert row["predicted_probability"] == row["Final_Hit_Probability"]
    assert row["selection_score"] == row["predicted_probability"]
    assert row["model_version"] == config.HITTER_MODEL_VERSION_LIVE
    assert row["selection_logic_version"] == config.HITTER_MODEL_VERSION_LIVE
    assert row["selection_mode"] == "live"
    # Diagnostics preserved
    assert row["Game_Hit_Probability"] == 0.70
    assert row["Model_Hit_Probability"] == 0.10
    assert row["Matchup_Hit_Probability"] == 0.70


def test_live_mode_no_model_shortlist():
    hitters = _pool([
        (1, 0.80, 0.50, 0.91, 0.11),  # low model, high final
        (2, 0.80, 0.50, 0.85, 0.99),
        (3, 0.80, 0.50, 0.80, 0.98),
        (4, 0.80, 0.50, 0.75, 0.97),
    ])
    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30,
        selection_mode="live", force_live=True,
        model_shortlist_size=1,  # would pick key=2 if shortlist engaged
        min_probability=0.5,
    )
    assert list(picks["key_mlbam"]) == [1]


def test_live_missing_final_hit_records_fallback():
    hitters = _pool([(1, 0.90, 0.90, 0.90)])
    hitters = hitters.drop(columns=["Final_Hit_Probability"])
    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30,
        selection_mode="live", force_live=True,
    )
    assert picks.iloc[0]["fallback_used"] in (True, 1)
    assert "missing_final_hit_probability" in str(picks.iloc[0]["fallback_reason"])
    # Official path fell back — still produced a pick via legacy/shadow
    assert picks.iloc[0]["predicted_probability"] == 0.90


def test_evaluation_live_uses_predicted_not_blend():
    preds = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"),
        "rank": 1,
        "name": "A",
        "metric": "Final_Hit_Probability",
        "predicted_probability": 0.90,
        "probability": 0.50,
        "Matchup_Hit_Probability": 0.50,
        "probability_source": "Final_Hit_Probability",
        "selection_metric": "Final_Hit_Probability",
        "selection_mode": "live",
        "model_version": config.HITTER_MODEL_VERSION_LIVE,
        "actual_hit": pd.NA,
        "at_bats": pd.NA,
    }])
    graded = evaluation.graded_daily_picks(preds, metric="Game_Hit_Probability", max_picks=2, min_probability=0.77)
    assert graded.iloc[0]["combined_probability"] == 0.90
    assert graded.iloc[0]["grade"] == "recommended"


def test_evaluation_legacy_still_blends():
    preds = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"),
        "rank": 1,
        "name": "A",
        "metric": "Game_Hit_Probability",
        "predicted_probability": 0.90,
        "probability": 0.60,
        "Matchup_Hit_Probability": 0.60,
        "probability_source": "Game_Hit_Probability",
        "selection_metric": "Matchup_Approach",
        "selection_mode": "legacy",
        "model_version": config.HITTER_MODEL_VERSION,
        "actual_hit": pd.NA,
        "at_bats": pd.NA,
    }])
    graded = evaluation.graded_daily_picks(preds, metric="Game_Hit_Probability", max_picks=2, min_probability=0.70)
    expected = (0.90 + 0.60 + 0.60) / 3
    assert abs(graded.iloc[0]["combined_probability"] - expected) < 1e-9


def test_rank_and_log_must_share_column_in_live_mode():
    """Regression: fail if live ranks by one column while logging another."""
    hitters = _pool([
        (1, 0.99, 0.99, 0.40),
        (2, 0.50, 0.10, 0.95),
    ])
    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30,
        selection_mode="live", force_live=True, min_probability=0.3,
        rank_metric="Matchup_Approach",
        metric="Game_Hit_Probability",
    )
    row = picks.iloc[0]
    # Winner is Final_Hit leader (2), not Matchup_Approach / GHP leader (1)
    assert row["key_mlbam"] == 2
    assert row["selection_metric"] == row["probability_source"] == row["metric"]
    assert row["predicted_probability"] == row["selection_score"] == row["Final_Hit_Probability"]


def test_regression_live_select_picks_source_forbids_split_authority():
    """Source-level guard: live path must not reintroduce rank≠log split."""
    src = inspect.getsource(predictions._select_picks_live)
    assert "authoritative" in src
    assert "FINAL_HIT_PROBABILITY" in src or "Final_Hit_Probability" in src
    assert "sort_values(authoritative" in src.replace(" ", "") or "sort_values(authoritative," in src
    assert 'picks["predicted_probability"] = picks[authoritative]' in src or \
        "picks['predicted_probability'] = picks[authoritative]" in src
    assert 'picks["selection_score"] = picks[authoritative]' in src or \
        "picks['selection_score'] = picks[authoritative]" in src
    # Must not reintroduce the ML shortlist as the primary live mechanism.
    assert ".head(model_shortlist_size)" not in src
    assert "JOINT_PROBABILITY_GATE_COLUMNS" not in src


def test_brier_uses_logged_predicted_probability_for_live_rows():
    preds = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"),
        "predicted_probability": 0.8,
        "actual_hit": 1.0,
        "at_bats": 3,
        "metric": "Final_Hit_Probability",
        "probability_source": "Final_Hit_Probability",
    }])
    score = evaluation.brier_score(preds)
    assert abs(score - (0.8 - 1.0) ** 2) < 1e-12
