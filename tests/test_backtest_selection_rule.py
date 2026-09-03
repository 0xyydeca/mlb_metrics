"""Unit tests for scripts/backtest_selection_rule.py's own logic
(select_and_resolve, report) against a tiny hand-built `pools` list -
bypassing build_date_pools' real Statcast/model-artifact dependency, per
this project's "the backtest tool itself is under test, not just a
one-off unchecked script" discipline."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backtest_selection_rule.py"


def _load_backtest_selection_rule_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("backtest_selection_rule", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick_pool_row(key, name, ghp, probability, approach, matchup_hp, matchup_approach, model_hp):
    return {
        "key_mlbam": key, "name_first": name, "name_last": "L", "team": "NYY",
        "PA_L": 0, "PA_R": 40, "Game_Hit_Probability": ghp, "probability": probability,
        "Approach": approach, "Matchup_Hit_Probability": matchup_hp, "Matchup_Approach": matchup_approach,
        "Model_Hit_Probability": model_hp,
    }


def test_select_and_resolve_matchup_approach_ranks_by_approach_product():
    module = _load_backtest_selection_rule_module()
    pools = [
        {
            "date": "2026-06-20",
            "pick_pool": pd.DataFrame([
                # Player A: lower Model_Hit_Probability but higher Matchup_Approach.
                _pick_pool_row(1, "A", 0.85, 0.9, 0.765, 0.6, 0.459, 0.55),
                # Player B: higher Model_Hit_Probability but lower Matchup_Approach.
                _pick_pool_row(2, "B", 0.70, 0.8, 0.560, 0.9, 0.504, 0.92),
            ]),
            "got_hit": pd.DataFrame([{"key_mlbam": 1, "Got_Hit": 1}, {"key_mlbam": 2, "Got_Hit": 0}]),
            "has_model": True,
        }
    ]

    heuristic_picks = module.select_and_resolve(
        pools, "Matchup_Approach", variant="heuristic_only", top_n=1, min_plate_appearances=30
    )
    assert list(heuristic_picks["key_mlbam"]) == [2]  # higher Matchup_Approach (0.504 > 0.459)
    assert heuristic_picks.iloc[0]["actual_hit"] == 0
    assert heuristic_picks.iloc[0]["at_bats"] == 1

    # With only 2 rows (well under HITTER_MODEL_SHORTLIST_SIZE, 10), the
    # model-shortlist step is a genuine no-op - both variants must pick
    # identically, ranked by Matchup_Approach either way.
    model_shortlist_picks = module.select_and_resolve(
        pools, "Matchup_Approach", variant="model_shortlist", top_n=1, min_plate_appearances=30
    )
    assert list(model_shortlist_picks["key_mlbam"]) == [2]


def test_select_and_resolve_model_shortlist_can_pick_differently_from_heuristic_only():
    module = _load_backtest_selection_rule_module()
    # 11 rows so the model shortlist (top HITTER_MODEL_SHORTLIST_SIZE=10) is
    # genuinely engaged. Player 1 has the single best Matchup_Approach but
    # the WORST Model_Hit_Probability - the model shortlist excludes it,
    # leaving player 2 (best Matchup_Approach among the 10 survivors) as the
    # model-shortlist pick, while the heuristic-only variant (no shortlist)
    # still picks player 1 directly.
    rows = [
        # Matchup_Hit_Probability=0.75 clears HITTER_MIN_PROBABILITY's 0.7
        # gate for the (unused, since Model_Hit_Probability is present)
        # fallback path.
        _pick_pool_row(1, "A", 0.85, 0.95, 0.8075, 0.75, 0.605625, 0.10),  # worst model score
        _pick_pool_row(2, "B", 0.72, 0.75, 0.5400, 0.75, 0.500000, 0.60),  # best Matchup_Approach among survivors
    ] + [
        _pick_pool_row(key, f"F{key}", 0.65, 0.70, 0.455, 0.75, 0.405000, 0.61 + key * 0.01)
        for key in range(3, 12)
    ]
    pools = [
        {
            "date": "2026-06-20",
            "pick_pool": pd.DataFrame(rows),
            "got_hit": pd.DataFrame([{"key_mlbam": 1, "Got_Hit": 0}, {"key_mlbam": 2, "Got_Hit": 1}]),
            "has_model": True,
        }
    ]

    heuristic_picks = module.select_and_resolve(
        pools, "Matchup_Approach", variant="heuristic_only", top_n=1, min_plate_appearances=30
    )
    assert list(heuristic_picks["key_mlbam"]) == [1]

    model_shortlist_picks = module.select_and_resolve(
        pools, "Matchup_Approach", variant="model_shortlist", top_n=1, min_plate_appearances=30
    )
    assert list(model_shortlist_picks["key_mlbam"]) == [2]  # proves the two variants genuinely diverge


def test_select_and_resolve_skips_dates_with_no_usable_model_prediction():
    module = _load_backtest_selection_rule_module()
    pools = [
        {
            "date": "2026-06-20",
            "pick_pool": pd.DataFrame([_pick_pool_row(1, "A", 0.9, 0.9, 0.81, 0.8, 0.648, 0.9)]),
            "got_hit": pd.DataFrame([{"key_mlbam": 1, "Got_Hit": 1}]),
            "has_model": False,  # e.g. the model artifact wasn't loadable that date
        }
    ]

    model_shortlist_picks = module.select_and_resolve(
        pools, "Matchup_Approach", variant="model_shortlist", top_n=1, min_plate_appearances=30
    )
    assert model_shortlist_picks.empty

    heuristic_picks = module.select_and_resolve(
        pools, "Matchup_Approach", variant="heuristic_only", top_n=1, min_plate_appearances=30
    )
    assert list(heuristic_picks["key_mlbam"]) == [1]  # heuristic_only is unaffected by has_model


def test_select_and_resolve_missing_from_got_hit_table_means_no_game():
    # A picked player absent from that date's real Got_Hit table (zero
    # at-bats that day) must resolve to at_bats=0/actual_hit=NaN, not be
    # silently dropped or treated as a miss.
    module = _load_backtest_selection_rule_module()
    pools = [
        {
            "date": "2026-06-20",
            "pick_pool": pd.DataFrame([_pick_pool_row(1, "A", 0.9, 0.9, 0.81, 0.8, 0.648, 0.9)]),
            "got_hit": pd.DataFrame(columns=["key_mlbam", "Got_Hit"]),
            "has_model": True,
        }
    ]

    picks = module.select_and_resolve(pools, "Matchup_Approach", top_n=1, min_plate_appearances=30)
    assert len(picks) == 1
    assert picks.iloc[0]["at_bats"] == 0
    assert pd.isna(picks.iloc[0]["actual_hit"])


def test_report_runs_without_error_on_empty_and_nonempty_input(capsys):
    module = _load_backtest_selection_rule_module()
    empty = pd.DataFrame(
        columns=["date", "key_mlbam", "rank", "predicted_probability", "actual_hit", "at_bats", "metric"]
    )
    module.report("empty case", empty, n_candidate_dates=0)

    nonempty = pd.DataFrame([
        {
            "date": "2026-06-20", "key_mlbam": 1, "rank": 1, "predicted_probability": 0.8,
            "actual_hit": 1, "at_bats": 3, "metric": "Game_Hit_Probability",
        },
        {
            "date": "2026-06-20", "key_mlbam": 2, "rank": 2, "predicted_probability": 0.75,
            "actual_hit": 1, "at_bats": 2, "metric": "Game_Hit_Probability",
        },
    ])
    metrics = module.report("nonempty case", nonempty, n_candidate_dates=1)

    out = capsys.readouterr().out
    assert "empty case" in out
    assert "nonempty case" in out
    assert "HEADLINE all_of_top_2_hit_rate" in out
    assert "top_2_reset_rate" in out
    assert "secondary any_of_top_2_hit_rate" in out
    assert metrics["all_of_top_2_hit_rate"] == pytest.approx(1.0)
    assert metrics["top_2_reset_rate"] == pytest.approx(0.0)


def test_report_headline_prefers_all_of_top_2_over_any_of_top_2(capsys):
    # hit/miss day: any_of=1, all_of=0, reset=1 - headline must surface all_of/reset.
    module = _load_backtest_selection_rule_module()
    picks = pd.DataFrame([
        {"date": "2026-06-20", "key_mlbam": 1, "rank": 1, "predicted_probability": 0.8,
         "actual_hit": 1, "at_bats": 3, "metric": "Game_Hit_Probability"},
        {"date": "2026-06-20", "key_mlbam": 2, "rank": 2, "predicted_probability": 0.7,
         "actual_hit": 0, "at_bats": 2, "metric": "Game_Hit_Probability"},
    ])
    metrics = module.report("mixed day", picks, n_candidate_dates=1)
    out = capsys.readouterr().out
    assert "HEADLINE all_of_top_2_hit_rate=0.0000" in out
    assert "top_2_reset_rate=1.0000" in out
    assert "secondary any_of_top_2_hit_rate=1.0000" in out
    assert metrics["any_of_top_2_hit_rate"] == pytest.approx(1.0)
    assert metrics["all_of_top_2_hit_rate"] == pytest.approx(0.0)


def test_report_bootstrap_comparison_uses_date_unit(capsys):
    module = _load_backtest_selection_rule_module()
    picks_a = pd.DataFrame([
        {"date": pd.Timestamp("2026-06-18"), "rank": 1, "predicted_probability": 0.9, "actual_hit": 1, "at_bats": 3},
        {"date": pd.Timestamp("2026-06-18"), "rank": 2, "predicted_probability": 0.8, "actual_hit": 1, "at_bats": 2},
        {"date": pd.Timestamp("2026-06-19"), "rank": 1, "predicted_probability": 0.9, "actual_hit": 1, "at_bats": 3},
        {"date": pd.Timestamp("2026-06-19"), "rank": 2, "predicted_probability": 0.8, "actual_hit": 1, "at_bats": 2},
    ])
    picks_b = pd.DataFrame([
        {"date": pd.Timestamp("2026-06-18"), "rank": 1, "predicted_probability": 0.9, "actual_hit": 0, "at_bats": 3},
        {"date": pd.Timestamp("2026-06-18"), "rank": 2, "predicted_probability": 0.8, "actual_hit": 0, "at_bats": 2},
        {"date": pd.Timestamp("2026-06-19"), "rank": 1, "predicted_probability": 0.9, "actual_hit": 0, "at_bats": 3},
        {"date": pd.Timestamp("2026-06-19"), "rank": 2, "predicted_probability": 0.8, "actual_hit": 0, "at_bats": 2},
    ])
    module.report_bootstrap_comparison(
        picks_a, picks_b, "A", "B",
        candidate_dates=[pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-19")],
        n_bootstrap=20, random_state=0,
    )
    out = capsys.readouterr().out
    assert "unit=date" in out
    assert "all_of_top_2_hit_rate" in out
    assert "top_2_reset_rate" in out
