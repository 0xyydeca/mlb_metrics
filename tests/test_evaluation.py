import math

import pandas as pd
import pytest

from mlb_metrics import evaluation


def _predictions():
    return pd.DataFrame(
        [
            {"date": "2026-06-18", "rank": 1, "predicted_probability": 0.9, "metric": "m", "actual_hit": 1},
            {"date": "2026-06-18", "rank": 2, "predicted_probability": 0.6, "metric": "m", "actual_hit": 0},
            {"date": "2026-06-19", "rank": 1, "predicted_probability": 0.8, "metric": "m", "actual_hit": 1},
            {"date": "2026-06-19", "rank": 2, "predicted_probability": 0.5, "metric": "m", "actual_hit": 1},
            {"date": "2026-06-20", "rank": 1, "predicted_probability": 0.7, "metric": "m", "actual_hit": 0},
            {"date": "2026-06-20", "rank": 2, "predicted_probability": 0.4, "metric": "m", "actual_hit": 0},
        ]
    )


def test_pick_accuracy_by_rank():
    table = evaluation.pick_accuracy_by_rank(_predictions()).set_index("rank")
    assert table.loc[1, "hit_rate"] == pytest.approx(2 / 3)
    assert table.loc[1, "n"] == 3
    assert table.loc[2, "hit_rate"] == pytest.approx(1 / 3)


def test_top_k_hit_rate_any_vs_all():
    preds = _predictions()
    assert evaluation.top_k_hit_rate(preds, 1) == pytest.approx(2 / 3)
    assert evaluation.top_k_hit_rate(preds, 2, require_all=False) == pytest.approx(2 / 3)
    assert evaluation.top_k_hit_rate(preds, 2, require_all=True) == pytest.approx(1 / 3)


def test_brier_score():
    preds = _predictions()
    expected = sum(
        (p - y) ** 2 for p, y in zip(preds["predicted_probability"], preds["actual_hit"])
    ) / len(preds)
    assert evaluation.brier_score(preds) == pytest.approx(expected)


def test_log_loss():
    preds = _predictions()
    expected = -sum(
        y * math.log(p) + (1 - y) * math.log(1 - p)
        for p, y in zip(preds["predicted_probability"], preds["actual_hit"])
    ) / len(preds)
    assert evaluation.log_loss(preds) == pytest.approx(expected)


def test_unresolved_rows_are_excluded_from_every_metric():
    preds = _predictions()
    pending = pd.DataFrame(
        [{"date": "2026-06-21", "rank": 1, "predicted_probability": 0.99, "metric": "m", "actual_hit": None}]
    )
    with_pending = pd.concat([preds, pending], ignore_index=True)

    assert evaluation.brier_score(with_pending) == pytest.approx(evaluation.brier_score(preds))
    assert evaluation.log_loss(with_pending) == pytest.approx(evaluation.log_loss(preds))
    assert evaluation.top_k_hit_rate(with_pending, 1) == pytest.approx(evaluation.top_k_hit_rate(preds, 1))


def test_calibration_table_covers_every_resolved_row():
    table = evaluation.calibration_table(_predictions(), n_bins=2)
    assert table["n"].sum() == 6


def test_wilson_confidence_interval_matches_helpers_wilson_ci_hand_derived_case():
    # Same count=50, n=100 case tests/test_helpers.py hand-derives the
    # Wilson formula for - this is a scalar wrapper around
    # helpers.wilson_ci, so it must produce the exact same numbers.
    ci_low, ci_high = evaluation.wilson_confidence_interval(50, 100)
    assert ci_low == pytest.approx(0.4038315303659956)
    assert ci_high == pytest.approx(0.5961684696340044)


def test_wilson_confidence_interval_n_zero_is_the_widest_honest_bound():
    ci_low, ci_high = evaluation.wilson_confidence_interval(0, 0)
    assert ci_low == 0.0
    assert ci_high == 1.0


def test_binomial_significance_exact_null_rate_is_maximally_insignificant():
    # 6 of 12 successes IS the null (0.5) exactly - a two-sided exact
    # binomial test must return p=1.0, zero evidence against the null.
    assert evaluation.binomial_significance(6, 12, null_probability=0.5) == pytest.approx(1.0)


def test_binomial_significance_extreme_rate_is_significant():
    # 0 of 12 successes against a null of 0.5 - vanishingly unlikely
    # under the null (two-sided exact binomial: 2 * 0.5^12).
    p = evaluation.binomial_significance(0, 12, null_probability=0.5)
    assert p == pytest.approx(2 * (0.5**12))


def test_binomial_significance_n_zero_is_nan_not_a_fabricated_pvalue():
    assert math.isnan(evaluation.binomial_significance(0, 0))


def test_mean_significance_matches_scipy_ttest_1samp_directly():
    # Reuse-not-reimplement: verify the wiring (right values passed to
    # scipy in the right shape) rather than re-deriving the t-distribution
    # by hand, same spirit as wilson_ci's own tests trusting statsmodels'
    # formula and only checking a hand-derived case once elsewhere.
    from scipy.stats import ttest_1samp as _ttest_1samp

    values = pd.Series([1.0, 2.0, 3.0, -1.5])
    expected = float(_ttest_1samp(values, 0.0).pvalue)
    assert evaluation.mean_significance(values, null_value=0.0) == pytest.approx(expected)


def test_mean_significance_fewer_than_two_real_values_is_nan():
    assert math.isnan(evaluation.mean_significance(pd.Series([5.0]), null_value=0.0))
    assert math.isnan(evaluation.mean_significance(pd.Series([], dtype=float), null_value=0.0))
    # A NaN mixed in with only one real value still leaves just 1 real
    # sample after dropna - not enough to estimate a variance.
    assert math.isnan(evaluation.mean_significance(pd.Series([5.0, None]), null_value=0.0))


def test_outcome_col_parameter_works_with_a_different_column_name():
    # Same fixture, but the outcome lives in "actual_correct" instead of
    # "actual_hit" - the shape game_evaluation.py's game-pick scoring needs
    # (see build_game_picks_export). Proves the outcome_col plumbing added
    # to resolved_only/brier_score/log_loss/etc. actually works, not just
    # that the default ("actual_hit") still does.
    preds = _predictions().rename(columns={"actual_hit": "actual_correct"})
    assert evaluation.brier_score(preds, outcome_col="actual_correct") == pytest.approx(
        evaluation.brier_score(_predictions())
    )
    assert evaluation.log_loss(preds, outcome_col="actual_correct") == pytest.approx(
        evaluation.log_loss(_predictions())
    )
    assert len(evaluation.resolved_only(preds, outcome_col="actual_correct")) == 6


def test_summarize_splits_by_metric():
    preds = _predictions()
    other_metric = preds.copy()
    other_metric["metric"] = "other"

    summary = evaluation.summarize(pd.concat([preds, other_metric], ignore_index=True)).set_index("metric")

    assert set(summary.index) == {"m", "other"}
    assert summary.loc["m", "n_resolved"] == 6
    assert summary.loc["m", "any_of_top_1_hit_rate"] == pytest.approx(2 / 3)


def test_summarize_model_version_filters_to_only_that_version():
    preds = _predictions()
    preds["model_version"] = "v1"
    new_version_row = pd.DataFrame(
        [{"date": "2026-06-21", "rank": 1, "predicted_probability": 0.9, "metric": "m",
          "actual_hit": 1, "model_version": "v2"}]
    )
    combined = pd.concat([preds, new_version_row], ignore_index=True)

    all_time = evaluation.summarize(combined).set_index("metric")
    assert all_time.loc["m", "n_resolved"] == 7  # every version blended, unchanged default behavior

    v2_only = evaluation.summarize(combined, model_version="v2").set_index("metric")
    assert v2_only.loc["m", "n_resolved"] == 1


def _pick(date, rank, predicted_probability, at_bats, actual_hit, metric="Game_Hit_Probability", name="Player"):
    return {
        "date": date, "rank": rank, "name": name, "predicted_probability": predicted_probability,
        "metric": metric, "actual_hit": actual_hit, "at_bats": at_bats,
    }


def _streak_predictions():
    """Exercises every real Beat the Streak rule from the miss/hit/no_game/
    pending model, with min_probability disabled (0.0) so every logged row
    counts - this fixture is about the cumulative-count/reset logic itself,
    not the recommendation-threshold gating (see the dedicated test below)."""
    rows = [
        _pick("2026-06-18", 1, 0.9, 4, 1),   # hit
        _pick("2026-06-18", 2, 0.85, 3, 1),  # hit -> day adds 2 -> streak=2
        _pick("2026-06-19", 1, 0.88, 3, 1),  # hit
        _pick("2026-06-19", 2, 0.82, 0, None),  # no_game (0 at-bats) -> neutral -> day adds 1 -> streak=3
        _pick("2026-06-20", 1, 0.87, 3, 1),  # hit
        _pick("2026-06-20", 2, 0.81, 2, 0),  # miss -> resets the whole day -> streak=0
        _pick("2026-06-21", 1, 0.90, 3, 1),  # single pick, hit -> streak=1
        _pick("2026-06-22", 1, 0.90, None, None),  # pending
        _pick("2026-06-22", 2, 0.85, None, None),  # pending -> whole day skipped, not a break
        _pick("2026-06-23", 1, 0.90, 1, 1),  # single pick, hit -> continues from 06-21 (06-22 skipped) -> streak=2
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_streak_progression_follows_beat_the_streak_rules():
    progression = evaluation.streak_progression(_streak_predictions(), min_probability=0.0)

    assert list(progression["date"].dt.strftime("%Y-%m-%d")) == [
        "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21", "2026-06-23",
    ]  # 06-22 (all pending) never appears
    assert list(progression["streak"]) == [2, 3, 0, 1, 2]
    assert list(progression["reset"]) == [False, False, True, False, False]


def test_longest_and_current_streak():
    preds = _streak_predictions()
    assert evaluation.longest_streak(preds, min_probability=0.0) == 3
    assert evaluation.current_streak(preds, min_probability=0.0) == 2


def test_a_miss_resets_the_streak_regardless_of_the_other_pick():
    preds = pd.DataFrame(
        [
            _pick("2026-06-18", 1, 0.9, 3, 1),
            _pick("2026-06-18", 2, 0.85, 3, 1),
            _pick("2026-06-19", 1, 0.9, 3, 1),   # hit
            _pick("2026-06-19", 2, 0.85, 2, 0),  # miss -> resets despite the hit
        ]
    )
    assert evaluation.current_streak(preds, min_probability=0.0) == 0
    assert evaluation.longest_streak(preds, min_probability=0.0) == 2


def test_zero_at_bats_pick_is_neutral_not_a_break():
    preds = pd.DataFrame(
        [
            _pick("2026-06-18", 1, 0.9, 3, 1),
            _pick("2026-06-18", 2, 0.85, 0, None),  # no_game
        ]
    )
    # Should behave exactly like a single-pick day that hit: +1, no reset.
    assert evaluation.current_streak(preds, min_probability=0.0) == 1
    assert evaluation.longest_streak(preds, min_probability=0.0) == 1


def test_graded_picks_can_be_all_recommended_mixed_or_all_speculative():
    preds = pd.DataFrame(
        [
            # Day A: both clear the bar -> both graded recommended.
            _pick("2026-06-18", 1, 0.90, 3, 1),
            _pick("2026-06-18", 2, 0.85, 3, 1),
            # Day B: only rank 1 clears it -> mixed grades.
            _pick("2026-06-19", 1, 0.90, 3, 1),
            _pick("2026-06-19", 2, 0.60, 3, 0),
            # Day C: neither clears it -> both graded speculative, but
            # STILL shown (not collapsed to a single no_pick placeholder -
            # real candidates were logged, they're just below the bar).
            _pick("2026-06-20", 1, 0.70, 3, 0),
            _pick("2026-06-20", 2, 0.65, 3, 1),
        ]
    )

    picks, summary = evaluation.build_beat_the_streak_export(preds, max_picks=2, min_probability=0.80)

    assert set(picks[picks["date"] == "2026-06-18"]["rank"]) == {1, 2}
    assert set(picks[picks["date"] == "2026-06-18"]["grade"]) == {"recommended"}

    day_b = picks[picks["date"] == "2026-06-19"].set_index("rank")
    assert day_b.loc[1, "grade"] == "recommended"
    assert day_b.loc[2, "grade"] == "speculative"

    day_c = picks[picks["date"] == "2026-06-20"]
    assert len(day_c) == 2  # both real candidates shown, no no_pick placeholder
    assert set(day_c["grade"]) == {"speculative"}
    assert set(day_c["status"]) == {"miss", "hit"}  # real outcomes still tracked
    assert day_c["combined_probability"].notna().all()

    # Speculative-only day C never enters the streak at all (not even as a
    # no-op skip in the progression table), and day A+B's recommended picks
    # both hit -> 2 + 1 = 3. Day B's speculative rank-2 miss doesn't count.
    assert summary.loc[0, "current_streak"] == 3
    assert summary.loc[0, "longest_streak"] == 3
    assert summary.loc[0, "n_days_resolved"] == 2


def test_a_date_with_no_rank_within_max_picks_still_gets_a_no_pick_row():
    # A defensive edge case, not something the live pipeline produces in
    # practice (select_picks always starts ranking at 1, so any date with
    # ANY logged candidate has a rank<=max_picks row - a real off day, e.g.
    # the All-Star break, is simply absent from the log entirely and was
    # never something this placeholder could detect either, before or
    # after this change): a date whose only logged row has rank > max_picks
    # still needs its own explicit no_pick row rather than silently
    # vanishing from picks_table.
    preds = pd.DataFrame(
        [
            _pick("2026-06-18", 1, 0.90, 3, 1),
            _pick("2026-06-19", 3, 0.90, 3, 1),  # rank 3 > max_picks (2)
        ]
    )
    picks, summary = evaluation.build_beat_the_streak_export(preds, max_picks=2, min_probability=0.80)

    no_pick_row = picks[picks["date"] == "2026-06-19"]
    assert len(no_pick_row) == 1
    assert no_pick_row.iloc[0]["status"] == "no_pick"
    assert pd.isna(no_pick_row.iloc[0]["grade"])


def test_build_beat_the_streak_export_picks_table_status_and_summary():
    preds = _streak_predictions()
    picks, summary = evaluation.build_beat_the_streak_export(preds, min_probability=0.0)

    # Most recent day first, includes pending rows.
    assert picks["date"].iloc[0] == picks["date"].max()
    pending_rows = picks[picks["date"] == "2026-06-22"]
    assert (pending_rows["status"] == "pending").all()
    no_game_row = picks[(picks["date"] == "2026-06-19") & (picks["rank"] == 2)].iloc[0]
    assert no_game_row["status"] == "no_game"
    miss_row = picks[(picks["date"] == "2026-06-20") & (picks["rank"] == 2)].iloc[0]
    assert miss_row["status"] == "miss"
    hit_row = picks[(picks["date"] == "2026-06-18") & (picks["rank"] == 1)].iloc[0]
    assert hit_row["status"] == "hit"

    assert summary.loc[0, "n_days_resolved"] == 5
    assert summary.loc[0, "longest_streak"] == 3
    assert summary.loc[0, "current_streak"] == 2
    assert summary.loc[0, "model_version"] == "all_time"  # unfiltered (model_version=None) default label

    # Quant-analytics item #5: day_survival_rate's real CI, over the
    # 4-of-5 non-reset days streak_progression established above (reset
    # == [False, False, True, False, False]).
    expected_ci_low, expected_ci_high = evaluation.wilson_confidence_interval(4, 5)
    assert summary.loc[0, "day_survival_rate_ci_low"] == pytest.approx(expected_ci_low)
    assert summary.loc[0, "day_survival_rate_ci_high"] == pytest.approx(expected_ci_high)


def test_recommended_picks_blends_probability_and_matchup_not_just_ghp():
    # predicted_probability (GHP) alone is just under the bar, but
    # `probability` and `Matchup_Hit_Probability` are both strong - the
    # blended mean clears 0.80 even though GHP alone doesn't.
    preds = pd.DataFrame([
        _pick("2026-06-18", 1, 0.79, 3, 1) | {"probability": 0.85, "Matchup_Hit_Probability": 0.83},
    ])

    picks, summary = evaluation.build_beat_the_streak_export(preds, max_picks=2, min_probability=0.80)

    assert set(picks["status"]) == {"hit"}
    assert set(picks["grade"]) == {"recommended"}  # blend = (0.79+0.85+0.83)/3 = 0.823 >= 0.80


def test_recommended_picks_blend_can_also_exclude_a_pick_ghp_alone_would_have_passed():
    # GHP alone clears 0.80, but probability/Matchup_Hit_Probability are
    # both weak - the blended mean drops below the bar, so the pick is
    # still SHOWN (real, logged candidate) but graded speculative rather
    # than counted as recommended.
    preds = pd.DataFrame([
        _pick("2026-06-18", 1, 0.90, 3, 1) | {"probability": 0.55, "Matchup_Hit_Probability": 0.50},
    ])

    picks, summary = evaluation.build_beat_the_streak_export(preds, max_picks=2, min_probability=0.80)

    assert set(picks["status"]) == {"hit"}  # real outcome still tracked, not hidden behind no_pick
    assert set(picks["grade"]) == {"speculative"}  # blend = (0.90+0.55+0.50)/3 = 0.65 < 0.80
    assert picks.iloc[0]["combined_probability"] == pytest.approx(0.65)


def test_recommended_picks_blend_ignores_missing_matchup_hit_probability_per_row():
    # A day with no schedule/matchup data that day (Matchup_Hit_Probability
    # is NaN, not 0) should blend just the two signals it has, not be
    # dragged down by treating the missing value as a zero.
    preds = pd.DataFrame([
        _pick("2026-06-18", 1, 0.85, 3, 1) | {"probability": 0.81, "Matchup_Hit_Probability": None},
    ])

    picks, summary = evaluation.build_beat_the_streak_export(preds, max_picks=2, min_probability=0.80)

    assert set(picks["status"]) == {"hit"}  # blend = (0.85+0.81)/2 = 0.83 >= 0.80, NaN excluded not zeroed


def test_build_beat_the_streak_export_speculative_only_day_does_not_affect_streak_or_other_days():
    preds = pd.DataFrame(
        [
            _pick("2026-06-18", 1, 0.90, 3, 1),  # hit -> streak=1
            _pick("2026-06-19", 1, 0.70, 3, 0),  # below bar -> speculative, shown, miss
            _pick("2026-06-19", 2, 0.65, 3, 1),  # below bar -> speculative, shown, hit
            _pick("2026-06-20", 1, 0.90, 3, 1),  # hit -> streak continues to 2
        ]
    )

    picks, summary = evaluation.build_beat_the_streak_export(preds, max_picks=2, min_probability=0.80)

    speculative_rows = picks[picks["date"] == "2026-06-19"]
    assert len(speculative_rows) == 2  # both real candidates shown, not collapsed to one no_pick row
    assert set(speculative_rows["grade"]) == {"speculative"}
    assert set(speculative_rows["status"]) == {"miss", "hit"}  # real outcomes, not hidden
    assert speculative_rows["rank"].notna().all()

    # The speculative-only day is still a true no-op for the streak - not a
    # break, not skipped-as-pending, just absent from the progression
    # entirely, exactly like a real no_pick day was before this change.
    assert summary.loc[0, "n_days_resolved"] == 2
    assert summary.loc[0, "current_streak"] == 2
    assert summary.loc[0, "longest_streak"] == 2


def test_build_beat_the_streak_export_model_version_filters_and_labels_summary():
    preds = _streak_predictions()
    preds["model_version"] = "v1"
    # A lone v2 pick that would otherwise change the v1-only picture if not filtered out.
    preds = pd.concat([preds, pd.DataFrame([
        _pick("2026-06-23", 1, 0.85, 4, 1) | {"model_version": "v2"}
    ])], ignore_index=True)

    picks, summary = evaluation.build_beat_the_streak_export(preds, min_probability=0.0, model_version="v1")

    assert "2026-06-23" not in set(picks["date"])
    assert summary.loc[0, "model_version"] == "v1"
    assert summary.loc[0, "n_days_resolved"] == 5  # unchanged from the all-v1 fixture above
    assert summary.loc[0, "day_survival_rate"] == pytest.approx(4 / 5)  # 4 of 5 resolved days didn't reset


def _bts_day(date, outcomes, predicted_probability=0.8):
    """Build one fully-resolved top-2 day from a list of (at_bats, actual_hit)."""
    rows = []
    for rank, (at_bats, actual_hit) in enumerate(outcomes, start=1):
        rows.append(_pick(date, rank, predicted_probability, at_bats, actual_hit))
    return rows


def test_classify_pick_states_hit_miss_no_game_pending():
    preds = pd.DataFrame(
        [
            _pick("2026-06-18", 1, 0.9, 3, 1),
            _pick("2026-06-18", 2, 0.8, 2, 0),
            _pick("2026-06-18", 3, 0.7, 0, None),
            _pick("2026-06-18", 4, 0.6, None, None),
        ]
    )
    assert list(evaluation.classify_pick_states(preds)) == [
        evaluation.PICK_STATE_HIT,
        evaluation.PICK_STATE_MISS,
        evaluation.PICK_STATE_NO_GAME,
        evaluation.PICK_STATE_PENDING,
    ]


def test_top_k_day_outcomes_hand_calculated_two_pick_cases():
    # Five mutually exclusive two-pick day shapes from the BTS rules.
    preds = pd.DataFrame(
        _bts_day("2026-06-18", [(3, 1), (2, 1)])  # hit/hit
        + _bts_day("2026-06-19", [(3, 1), (0, None)])  # hit/void
        + _bts_day("2026-06-20", [(0, None), (0, None)])  # void/void
        + _bts_day("2026-06-21", [(3, 1), (2, 0)])  # hit/miss
        + _bts_day("2026-06-22", [(0, None), (2, 0)])  # void/miss
    )
    preds["date"] = pd.to_datetime(preds["date"])
    days = evaluation.top_k_day_outcomes(preds, k=2)
    by_date = days.set_index("date")

    hit_hit = by_date.loc[pd.Timestamp("2026-06-18")]
    assert bool(hit_hit["survived"]) and bool(hit_hit["all_hit"]) and hit_hit["hits_added"] == 2
    assert bool(hit_hit["advanced"]) and not bool(hit_hit["reset"])

    hit_void = by_date.loc[pd.Timestamp("2026-06-19")]
    assert bool(hit_void["survived"]) and not bool(hit_void["all_hit"]) and hit_void["hits_added"] == 1
    assert bool(hit_void["advanced"]) and not bool(hit_void["reset"])

    void_void = by_date.loc[pd.Timestamp("2026-06-20")]
    assert bool(void_void["survived"]) and not bool(void_void["all_hit"]) and void_void["hits_added"] == 0
    assert not bool(void_void["advanced"]) and not bool(void_void["reset"])

    hit_miss = by_date.loc[pd.Timestamp("2026-06-21")]
    assert bool(hit_miss["reset"]) and not bool(hit_miss["survived"]) and hit_miss["hits_added"] == 0
    assert not bool(hit_miss["all_hit"]) and bool(hit_miss["any_hit"])

    void_miss = by_date.loc[pd.Timestamp("2026-06-22")]
    assert bool(void_miss["reset"]) and not bool(void_miss["survived"]) and void_miss["hits_added"] == 0
    assert not bool(void_miss["any_hit"])


def test_selection_strategy_metrics_hand_calculated_over_bts_day_shapes():
    preds = pd.DataFrame(
        _bts_day("2026-06-18", [(3, 1), (2, 1)])
        + _bts_day("2026-06-19", [(3, 1), (0, None)])
        + _bts_day("2026-06-20", [(0, None), (0, None)])
        + _bts_day("2026-06-21", [(3, 1), (2, 0)])
        + _bts_day("2026-06-22", [(0, None), (2, 0)])
    )
    # One extra candidate date with no picks -> coverage 5/6.
    metrics = evaluation.selection_strategy_metrics(preds, k=2, n_candidate_dates=6)

    assert metrics["n_candidate_dates"] == 6
    assert metrics["n_dates_with_picks"] == 5
    assert metrics["coverage_rate"] == pytest.approx(5 / 6)
    assert metrics["n_resolved_pick_rows"] == 10  # includes no_game rows
    assert metrics["no_game_rate"] == pytest.approx(4 / 10)  # four voids across the five days
    # At-bats taken: hit/hit (2), hit/void (1), hit/miss (2), void/miss (1) = 6 rows; hits = 1+1+1+1 = 4? 
    # hit/hit: 2 hits, hit/void: 1, hit/miss: 1 hit + 1 miss, void/miss: 1 miss → hits=4, taken=6
    assert metrics["conditional_hit_rate_given_at_bat"] == pytest.approx(4 / 6)
    # all_hit only on 06-18 → 1/5
    assert metrics["all_of_top_2_hit_rate"] == pytest.approx(1 / 5)
    # any_hit on 06-18, 06-19, 06-21 → 3/5
    assert metrics["any_of_top_2_hit_rate"] == pytest.approx(3 / 5)
    # survived: hit/hit, hit/void, void/void → 3/5; reset: hit/miss, void/miss → 2/5
    assert metrics["top_2_survival_rate"] == pytest.approx(3 / 5)
    assert metrics["top_2_reset_rate"] == pytest.approx(2 / 5)
    # hits_added: 2, 1, 0, 0, 0 → mean 0.6
    assert metrics["mean_hits_added_per_played_day"] == pytest.approx(0.6)
    # top-1: advance on 18,19,21 (hit); reset on none of first picks... 
    # rank1 outcomes: hit, hit, void, hit, void → advance 3/5, reset 0/5
    assert metrics["top_1_advance_rate"] == pytest.approx(3 / 5)
    assert metrics["top_1_reset_rate"] == pytest.approx(0 / 5)
    assert metrics["brier_score"] == metrics["brier_score"]  # finite
    assert metrics["log_loss"] == metrics["log_loss"]


def test_no_game_row_is_not_silently_removed_from_advancement_and_coverage():
    # A lone void day must still count as a played/survived day with
    # coverage, and must NOT inflate all_of_top_k via resolved_only dropping.
    preds = pd.DataFrame(
        [
            _pick("2026-06-18", 1, 0.9, 0, None),
            _pick("2026-06-18", 2, 0.8, 0, None),
        ]
    )
    metrics = evaluation.selection_strategy_metrics(preds, k=2, n_candidate_dates=2)

    assert metrics["n_dates_with_picks"] == 1
    assert metrics["coverage_rate"] == pytest.approx(0.5)
    assert metrics["n_resolved_pick_rows"] == 2
    assert metrics["no_game_rate"] == pytest.approx(1.0)
    assert metrics["top_2_survival_rate"] == pytest.approx(1.0)
    assert metrics["top_2_reset_rate"] == pytest.approx(0.0)
    assert metrics["all_of_top_2_hit_rate"] == pytest.approx(0.0)  # voids are not hits
    assert metrics["any_of_top_2_hit_rate"] == pytest.approx(0.0)
    assert metrics["mean_hits_added_per_played_day"] == pytest.approx(0.0)
    # Legacy top_k_hit_rate(require_all=True) would be NaN (no labeled rows)
    # or overstate if mixed - prove the new metric keeps the void day.
    days = evaluation.top_k_day_outcomes(preds, k=2)
    assert len(days) == 1
    assert bool(days.iloc[0]["survived"])
    assert days.iloc[0]["n_no_game"] == 2


def test_legacy_top_k_all_hit_overstates_hit_plus_void_day():
    # Documents why selection_strategy_metrics exists: resolved_only drops
    # the void, so legacy require_all=True treats hit+void as "all hit".
    preds = pd.DataFrame(_bts_day("2026-06-19", [(3, 1), (0, None)]))
    assert evaluation.top_k_hit_rate(preds, 2, require_all=True) == pytest.approx(1.0)
    metrics = evaluation.selection_strategy_metrics(preds, k=2, n_candidate_dates=1)
    assert metrics["all_of_top_2_hit_rate"] == pytest.approx(0.0)
    assert metrics["top_2_survival_rate"] == pytest.approx(1.0)


def test_bootstrap_strategy_metric_difference_resamples_dates_not_rows(monkeypatch):
    # Strategy A: two dates, both all-hit. Strategy B: same dates, both reset.
    picks_a = pd.DataFrame(
        _bts_day("2026-06-18", [(3, 1), (2, 1)])
        + _bts_day("2026-06-19", [(3, 1), (2, 1)])
    )
    picks_b = pd.DataFrame(
        _bts_day("2026-06-18", [(3, 0), (2, 0)])
        + _bts_day("2026-06-19", [(3, 0), (2, 0)])
    )
    candidate_dates = [pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-19")]
    picks_a["date"] = pd.to_datetime(picks_a["date"])
    picks_b["date"] = pd.to_datetime(picks_b["date"])

    drawn_high_values = []

    class _FakeRng:
        def integers(self, low, high, size=None):
            # Date-block bootstrap must draw from n_dates (=2), never from
            # n_pick_rows (=4). Recording `high` proves which population size
            # was used.
            drawn_high_values.append(high)
            import numpy as _np
            return _np.zeros(size, dtype=int)

    monkeypatch.setattr(evaluation.np.random, "default_rng", lambda seed=None: _FakeRng())

    result = evaluation.bootstrap_strategy_metric_difference(
        picks_a, picks_b, metric="all_of_top_2_hit_rate", k=2,
        n_bootstrap=5, candidate_dates=candidate_dates, random_state=0,
    )
    assert result["resampled_unit"] == "date"
    assert result["n_dates"] == 2
    assert result["point_difference"] == pytest.approx(1.0)
    assert drawn_high_values == [2, 2, 2, 2, 2]
    assert all(diff == pytest.approx(1.0) for diff in result["differences"])
    days_a = evaluation.top_k_day_outcomes(picks_a, 2)
    assert set(days_a["n_picks"]) == {2}
