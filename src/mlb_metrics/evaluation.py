"""Backtest scoring: turns a resolved predictions log (from predictions.py)
into the numbers that actually answer "does this beat a coin flip, let
alone Beat the Streak" - hit rate by pick rank, calibration, Brier score,
and log loss.

Beat the Streak day-level semantics (hit / miss / no_game) live in
`classify_pick_states`, `top_k_day_outcomes`, and `selection_strategy_metrics`.
Those keep survival, advancement, all-picks-hit, and conditional-at-bat hit
rate as DISTINCT quantities - they must not share one ambiguous metric.
`top_k_hit_rate` / `summarize` remain available for legacy callers but are
NOT the Beat the Streak headline definitions (they historically dropped
`at_bats==0` rows via `resolved_only`, which can overstate "all hit" days).
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

from mlb_metrics import helpers

# Resolved pick states for Beat the Streak / selection-rule scoring.
# Distinct from dashboard status strings that also include "pending"/"no_pick".
PICK_STATE_HIT = "hit"
PICK_STATE_MISS = "miss"
PICK_STATE_NO_GAME = "no_game"
PICK_STATE_PENDING = "pending"


def resolved_only(predictions: pd.DataFrame, outcome_col: str = "actual_hit") -> pd.DataFrame:
    """Rows with a known outcome (0/1) in `outcome_col`, i.e. the game has
    been played. `outcome_col` defaults to "actual_hit" (hitter picks); pass
    "actual_correct" for game picks (see game_evaluation.py).

    Note: a Beat the Streak `no_game` row (`at_bats==0`, `actual_hit` null)
    is intentionally NOT included here - there is no 0/1 hit outcome to
    score. Day-level BTS metrics must go through `top_k_day_outcomes` /
    `selection_strategy_metrics` instead of this filter, so voids are not
    silently erased from survival/coverage denominators."""
    resolved = predictions[predictions[outcome_col].notna()].copy()
    resolved[outcome_col] = resolved[outcome_col].astype(float)
    return resolved


def classify_pick_states(df: pd.DataFrame) -> pd.Series:
    """Per-pick state from `at_bats` / `actual_hit`:

    - hit: at_bats > 0 and actual_hit == 1
    - miss: at_bats > 0 and actual_hit == 0
    - no_game (void): at_bats == 0
    - pending: at_bats unknown (null)

    These four states are the only inputs to date-level Beat the Streak
    survival / advancement / all-hit logic. A no_game row is a first-class
    resolved state - never drop it before aggregating days."""
    at_bats = pd.to_numeric(df["at_bats"], errors="coerce")
    actual_hit = pd.to_numeric(df["actual_hit"], errors="coerce")

    state = pd.Series(PICK_STATE_PENDING, index=df.index, dtype=object)
    state[at_bats == 0] = PICK_STATE_NO_GAME
    state[(at_bats > 0) & (actual_hit == 1)] = PICK_STATE_HIT
    state[(at_bats > 0) & (actual_hit == 0)] = PICK_STATE_MISS
    return state


def pick_accuracy_by_rank(predictions: pd.DataFrame, outcome_col: str = "actual_hit") -> pd.DataFrame:
    """Hit rate for each individual pick rank (1st-ranked pick, 2nd-ranked,
    ...), independent of the others. If the model has any skill, this should
    decrease as rank increases; if it's flat, the ranking isn't doing anything.

    Scoped to rows with a 0/1 `outcome_col` (see `resolved_only`) - no_game
    voids have no hit/miss label and are excluded from this per-player rate
    by design. Use `selection_strategy_metrics` for day-level BTS rates."""
    resolved = resolved_only(predictions, outcome_col)
    if resolved.empty:
        return pd.DataFrame(columns=["rank", "hit_rate", "n"])
    grouped = resolved.groupby("rank")[outcome_col].agg(hit_rate="mean", n="count").reset_index()
    return grouped.sort_values("rank").reset_index(drop=True)


def top_k_hit_rate(predictions: pd.DataFrame, k: int, require_all: bool = False, outcome_col: str = "actual_hit") -> float:
    """Legacy per-day rate over rows with a 0/1 `outcome_col` only
    (`resolved_only`). require_all=False: any labeled hit that day;
    require_all=True: every labeled row that day is a hit.

    WARNING: because `no_game` rows have null `actual_hit`, they are dropped
    before aggregation - a hit+void day looks like "all hit" here. That is
    NOT Beat the Streak's all-picks-hit definition. Prefer
    `selection_strategy_metrics` / `top_k_day_outcomes` for BTS scoring
    (`all_of_top_k_hit_rate` requires every selected pick to be a hit;
    `top_k_survival_rate` allows voids)."""
    resolved = resolved_only(predictions, outcome_col)
    picks = resolved[resolved["rank"] <= k]
    if picks.empty:
        return float("nan")
    per_day = picks.groupby("date")[outcome_col]
    outcome = per_day.min() if require_all else per_day.max()
    return float(outcome.mean())


def brier_score(predictions: pd.DataFrame, outcome_col: str = "actual_hit") -> float:
    """Mean squared error between predicted probability and actual (0/1)
    outcome - lower is better, 0 is perfect, 0.25 is what an uninformative
    always-predict-0.5 model scores. Only rows with a real 0/1 outcome
    (at-bat taken) enter; no_game voids correctly contribute nothing."""
    resolved = resolved_only(predictions, outcome_col)
    if resolved.empty:
        return float("nan")
    return float(np.mean((resolved["predicted_probability"] - resolved[outcome_col]) ** 2))


def log_loss(predictions: pd.DataFrame, eps: float = 1e-6, outcome_col: str = "actual_hit") -> float:
    resolved = resolved_only(predictions, outcome_col)
    if resolved.empty:
        return float("nan")
    p = resolved["predicted_probability"].clip(eps, 1 - eps)
    y = resolved[outcome_col]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def top_k_day_outcomes(predictions: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per date that has at least one rank<=k pick and is FULLY
    resolved (no pending pick states). Encodes Beat the Streak's date-level
    rules without collapsing distinct quantities:

    For a two-pick day:
      - two hits -> survived, all_hit, hits_added=2, advanced
      - one hit + one no_game -> survived, not all_hit, hits_added=1, advanced
      - two no_game -> survived, not all_hit, hits_added=0, not advanced
      - any miss -> reset, hits_added=0, not survived

    Columns: date, n_picks, n_hit, n_miss, n_no_game, reset, survived,
    all_hit, any_hit, hits_added, advanced.
    """
    if predictions.empty or "at_bats" not in predictions.columns:
        return pd.DataFrame(
            columns=[
                "date", "n_picks", "n_hit", "n_miss", "n_no_game",
                "reset", "survived", "all_hit", "any_hit", "hits_added", "advanced",
            ]
        )

    picks = predictions[predictions["rank"] <= k].copy()
    if picks.empty:
        return pd.DataFrame(
            columns=[
                "date", "n_picks", "n_hit", "n_miss", "n_no_game",
                "reset", "survived", "all_hit", "any_hit", "hits_added", "advanced",
            ]
        )

    picks["state"] = classify_pick_states(picks)
    rows = []
    for date, day in picks.groupby("date", sort=True):
        if (day["state"] == PICK_STATE_PENDING).any():
            continue
        n_hit = int((day["state"] == PICK_STATE_HIT).sum())
        n_miss = int((day["state"] == PICK_STATE_MISS).sum())
        n_no_game = int((day["state"] == PICK_STATE_NO_GAME).sum())
        n_picks = len(day)
        reset = n_miss > 0
        survived = not reset
        all_hit = (n_hit == n_picks) and n_picks > 0
        any_hit = n_hit > 0
        hits_added = n_hit if survived else 0
        advanced = hits_added > 0
        rows.append(
            {
                "date": date,
                "n_picks": n_picks,
                "n_hit": n_hit,
                "n_miss": n_miss,
                "n_no_game": n_no_game,
                "reset": reset,
                "survived": survived,
                "all_hit": all_hit,
                "any_hit": any_hit,
                "hits_added": hits_added,
                "advanced": advanced,
            }
        )
    return pd.DataFrame(rows)


def _rate_or_nan(successes: int, n: int) -> float:
    return float(successes / n) if n else float("nan")


def selection_strategy_metrics(
    predictions: pd.DataFrame,
    k: int = 2,
    n_candidate_dates: int | None = None,
    probability_col: str = "predicted_probability",
) -> dict:
    """Full Beat the Streak / selection-rule metric bundle for top-`k` picks.

    Survival, advancement, all-picks-hit, and conditional-at-bat hit rate are
    reported as separate fields - they are not interchangeable:

      - top_k_survival_rate: no selected pick missed (voids allowed)
      - top_k_reset_rate: any selected pick missed
      - all_of_top_k_hit_rate: every selected pick was a hit (voids fail this)
      - any_of_top_k_hit_rate: at least one selected pick was a hit (secondary)
      - top_1_advance_rate / top_1_reset_rate: when k>=1, computed on rank-1 days
      - conditional_hit_rate_given_at_bat: among pick rows with at_bats>0
      - mean_hits_added_per_played_day: mean hits_added over fully-resolved days

    `n_candidate_dates` is the evaluation window size (holdout dates). When
    omitted, defaults to the number of distinct dates present in
    `predictions` (coverage_rate then reflects only dates that produced
    rows, not a true slate-coverage rate).
    """
    preds = predictions.copy()
    if n_candidate_dates is None:
        n_candidate_dates = int(preds["date"].nunique()) if not preds.empty and "date" in preds.columns else 0

    top = preds[preds["rank"] <= k] if not preds.empty and "rank" in preds.columns else preds.iloc[0:0]
    n_dates_with_picks = int(top["date"].nunique()) if not top.empty else 0
    coverage_rate = _rate_or_nan(n_dates_with_picks, n_candidate_dates)

    if top.empty or "at_bats" not in top.columns:
        at_bat_known = top.iloc[0:0]
        at_bat_taken = top.iloc[0:0]
        n_resolved_pick_rows = 0
        no_game_rate = float("nan")
        conditional_hit_rate = float("nan")
        brier = float("nan")
        ll = float("nan")
    else:
        at_bats = pd.to_numeric(top["at_bats"], errors="coerce")
        at_bat_known = top[at_bats.notna()]
        n_resolved_pick_rows = len(at_bat_known)
        no_game_rate = _rate_or_nan(int((pd.to_numeric(at_bat_known["at_bats"], errors="coerce") == 0).sum()), n_resolved_pick_rows)
        at_bat_taken = top[at_bats > 0].copy()
        if at_bat_taken.empty:
            conditional_hit_rate = float("nan")
            brier = float("nan")
            ll = float("nan")
        else:
            y = pd.to_numeric(at_bat_taken["actual_hit"], errors="coerce").astype(float)
            conditional_hit_rate = float(y.mean())
            # Score calibration only on rows with a real 0/1 label.
            scored = at_bat_taken.copy()
            scored["actual_hit"] = y
            scored["predicted_probability"] = pd.to_numeric(at_bat_taken[probability_col], errors="coerce")
            brier = brier_score(scored)
            ll = log_loss(scored)

    days = top_k_day_outcomes(preds, k)
    n_played_days = len(days)
    all_of_rate = _rate_or_nan(int(days["all_hit"].sum()), n_played_days) if n_played_days else float("nan")
    any_of_rate = _rate_or_nan(int(days["any_hit"].sum()), n_played_days) if n_played_days else float("nan")
    survival_rate = _rate_or_nan(int(days["survived"].sum()), n_played_days) if n_played_days else float("nan")
    reset_rate = _rate_or_nan(int(days["reset"].sum()), n_played_days) if n_played_days else float("nan")
    mean_hits_added = float(days["hits_added"].mean()) if n_played_days else float("nan")

    # Top-1 advance/reset always come from the rank-1 day slice (even when
    # reporting a k=2 bundle), so the two-pick headline metrics stay
    # distinguishable from single-pick advancement.
    days_1 = top_k_day_outcomes(preds, 1)
    n_played_1 = len(days_1)
    top_1_advance_rate = _rate_or_nan(int(days_1["advanced"].sum()), n_played_1) if n_played_1 else float("nan")
    top_1_reset_rate = _rate_or_nan(int(days_1["reset"].sum()), n_played_1) if n_played_1 else float("nan")

    return {
        "k": k,
        "n_candidate_dates": int(n_candidate_dates),
        "n_dates_with_picks": n_dates_with_picks,
        "coverage_rate": coverage_rate,
        "n_resolved_pick_rows": n_resolved_pick_rows,
        "no_game_rate": no_game_rate,
        "conditional_hit_rate_given_at_bat": conditional_hit_rate,
        "top_1_advance_rate": top_1_advance_rate,
        "top_1_reset_rate": top_1_reset_rate,
        f"all_of_top_{k}_hit_rate": all_of_rate,
        f"any_of_top_{k}_hit_rate": any_of_rate,
        f"top_{k}_survival_rate": survival_rate,
        f"top_{k}_reset_rate": reset_rate,
        "mean_hits_added_per_played_day": mean_hits_added,
        "brier_score": brier,
        "log_loss": ll,
        "n_played_days": n_played_days,
    }


def _metric_from_day_outcomes_and_rows(
    days: pd.DataFrame,
    pick_rows: pd.DataFrame,
    metric: str,
    k: int,
    n_candidate_dates: int,
) -> float:
    """Compute one named metric from an already-sliced day/row frame.
    Used by the date-block bootstrap so each resample aggregates days, not
    independent pick rows."""
    n_played = len(days)
    if metric == "coverage_rate":
        n_with_picks = int(pick_rows["date"].nunique()) if not pick_rows.empty else 0
        return _rate_or_nan(n_with_picks, n_candidate_dates)
    if metric in (f"all_of_top_{k}_hit_rate", "all_of_top_k_hit_rate"):
        return _rate_or_nan(int(days["all_hit"].sum()), n_played) if n_played else float("nan")
    if metric in (f"any_of_top_{k}_hit_rate", "any_of_top_k_hit_rate"):
        return _rate_or_nan(int(days["any_hit"].sum()), n_played) if n_played else float("nan")
    if metric in (f"top_{k}_survival_rate", "top_k_survival_rate"):
        return _rate_or_nan(int(days["survived"].sum()), n_played) if n_played else float("nan")
    if metric in (f"top_{k}_reset_rate", "top_k_reset_rate"):
        return _rate_or_nan(int(days["reset"].sum()), n_played) if n_played else float("nan")
    if metric == "mean_hits_added_per_played_day":
        return float(days["hits_added"].mean()) if n_played else float("nan")
    if metric == "top_1_advance_rate":
        days_1 = days if k == 1 else top_k_day_outcomes(pick_rows, 1)
        return _rate_or_nan(int(days_1["advanced"].sum()), len(days_1)) if len(days_1) else float("nan")
    if metric == "top_1_reset_rate":
        days_1 = days if k == 1 else top_k_day_outcomes(pick_rows, 1)
        return _rate_or_nan(int(days_1["reset"].sum()), len(days_1)) if len(days_1) else float("nan")
    if metric == "conditional_hit_rate_given_at_bat":
        if pick_rows.empty or "at_bats" not in pick_rows.columns:
            return float("nan")
        taken = pick_rows[pd.to_numeric(pick_rows["at_bats"], errors="coerce") > 0]
        if taken.empty:
            return float("nan")
        return float(pd.to_numeric(taken["actual_hit"], errors="coerce").astype(float).mean())
    raise ValueError(f"Unknown bootstrap metric {metric!r}")


def bootstrap_strategy_metric_difference(
    picks_a: pd.DataFrame,
    picks_b: pd.DataFrame,
    metric: str,
    k: int = 2,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    candidate_dates=None,
    random_state: int | None = None,
) -> dict:
    """Paired date-block bootstrap CI for metric(A) - metric(B).

    Resamples WHOLE dates with replacement from `candidate_dates` (or the
    union of dates present in either pick frame). Pick rows are never
    resampled independently - every pick from a drawn date is kept
    together - because same-day picks are dependent under Beat the Streak.

    Returns point_difference, ci_low, ci_high, n_bootstrap, n_dates,
    metric_a, metric_b, and the raw bootstrap differences series.
    """
    rng = np.random.default_rng(random_state)
    if candidate_dates is None:
        dates_a = set(picks_a["date"]) if not picks_a.empty else set()
        dates_b = set(picks_b["date"]) if not picks_b.empty else set()
        candidate_dates = sorted(dates_a | dates_b)
    else:
        candidate_dates = list(candidate_dates)

    n_dates = len(candidate_dates)
    if n_dates == 0:
        return {
            "metric": metric,
            "metric_a": float("nan"),
            "metric_b": float("nan"),
            "point_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap": n_bootstrap,
            "n_dates": 0,
            "differences": np.array([], dtype=float),
            "resampled_unit": "date",
        }

    days_a = top_k_day_outcomes(picks_a, k).set_index("date") if not picks_a.empty else pd.DataFrame()
    days_b = top_k_day_outcomes(picks_b, k).set_index("date") if not picks_b.empty else pd.DataFrame()
    rows_a = picks_a[picks_a["rank"] <= k].copy() if not picks_a.empty else picks_a
    rows_b = picks_b[picks_b["rank"] <= k].copy() if not picks_b.empty else picks_b

    metric_a = _metric_from_day_outcomes_and_rows(
        days_a.reset_index() if not days_a.empty else days_a,
        rows_a, metric, k, n_dates,
    )
    metric_b = _metric_from_day_outcomes_and_rows(
        days_b.reset_index() if not days_b.empty else days_b,
        rows_b, metric, k, n_dates,
    )
    point = metric_a - metric_b if (metric_a == metric_a and metric_b == metric_b) else float("nan")

    differences = np.empty(n_bootstrap, dtype=float)
    date_array = np.asarray(candidate_dates, dtype=object)
    for i in range(n_bootstrap):
        # Sample DATE indices, not pick-row indices - the whole point of a
        # date-block bootstrap (dependence within a day).
        sampled = date_array[rng.integers(0, n_dates, size=n_dates)]
        # Preserve multiplicity: a date drawn twice contributes its day
        # outcome twice to the aggregate (standard bootstrap with replacement).
        day_a_parts = []
        day_b_parts = []
        row_a_parts = []
        row_b_parts = []
        for date in sampled:
            if not days_a.empty and date in days_a.index:
                day_a_parts.append(days_a.loc[[date]].reset_index())
            if not days_b.empty and date in days_b.index:
                day_b_parts.append(days_b.loc[[date]].reset_index())
            if not rows_a.empty:
                row_a_parts.append(rows_a[rows_a["date"] == date])
            if not rows_b.empty:
                row_b_parts.append(rows_b[rows_b["date"] == date])
        day_a_boot = pd.concat(day_a_parts, ignore_index=True) if day_a_parts else pd.DataFrame(columns=list(days_a.reset_index().columns) if not days_a.empty else ["date"])
        day_b_boot = pd.concat(day_b_parts, ignore_index=True) if day_b_parts else pd.DataFrame(columns=list(days_b.reset_index().columns) if not days_b.empty else ["date"])
        row_a_boot = pd.concat(row_a_parts, ignore_index=True) if row_a_parts else rows_a.iloc[0:0]
        row_b_boot = pd.concat(row_b_parts, ignore_index=True) if row_b_parts else rows_b.iloc[0:0]
        a = _metric_from_day_outcomes_and_rows(day_a_boot, row_a_boot, metric, k, n_dates)
        b = _metric_from_day_outcomes_and_rows(day_b_boot, row_b_boot, metric, k, n_dates)
        differences[i] = a - b if (a == a and b == b) else np.nan

    finite = differences[np.isfinite(differences)]
    if len(finite) == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.quantile(finite, alpha / 2))
        ci_high = float(np.quantile(finite, 1 - alpha / 2))

    return {
        "metric": metric,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "point_difference": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap": n_bootstrap,
        "n_dates": n_dates,
        "differences": differences,
        "resampled_unit": "date",
    }


def wilson_confidence_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Scalar Wilson score confidence interval for one aggregate backtest
    rate (accuracy, beat_closing_line_rate, win_rate_on_advised_bets,
    day_survival_rate, ...) - quant-analytics item #5 ("backtest scope
    and statistical significance"): the honest answer to "is n big
    enough to trust this rate," not just reporting the rate itself. A
    12-game 33% "beat the market" rate and a 1200-game 33% rate are NOT
    the same amount of evidence, and a bare percentage on a dashboard
    can't tell them apart.

    Reuses helpers.wilson_ci (quant-analytics item #3's per-row
    vectorized version, already proven against
    statsmodels.stats.proportion.proportion_confint) by wrapping the
    scalar successes/n in a length-1 Series and unwrapping the result -
    not a second implementation of the same formula. n=0 returns
    (0.0, 1.0), the same "no information, could be anywhere" contract
    helpers.wilson_ci already establishes."""
    low, high = helpers.wilson_ci(pd.Series([successes]), pd.Series([n]), alpha=alpha)
    return float(low.iloc[0]), float(high.iloc[0])


def binomial_significance(successes: int, n: int, null_probability: float = 0.5) -> float:
    """Two-sided exact binomial test p-value: given only `n` real
    trials, is the observed rate distinguishable from `null_probability`
    (default 0.5 - a coin flip, i.e. "no real skill difference")? A
    small p-value means the observed rate would be unlikely if the null
    were actually true. Uses scipy.stats.binomtest - the exact test, not
    a normal approximation (unreliable at the small n this project's
    real backtests actually have, e.g. n=12) - not hand-derived.

    0.5 is only a well-posed null for a genuinely symmetric comparison
    (e.g. beat_closing_line_rate's "whose squared error was lower on
    this game," where under "no skill difference" either side is
    equally likely to win) - NOT for an unconditional accuracy rate
    (home teams win somewhat more than half of real MLB games, so 0.5
    isn't actually "no skill" there). Callers are responsible for only
    using this where the null is real, not just convenient - see
    game_evaluation.py's own callers for which metrics get a p-value at
    all versus a confidence interval only. n=0 returns NaN, not a
    fabricated 1.0 (no data is no evidence either way, not "certainly
    the null")."""
    if n == 0:
        return float("nan")
    return float(binomtest(successes, n, null_probability).pvalue)


def mean_significance(values: pd.Series, null_value: float = 0.0) -> float:
    """One-sample two-sided t-test p-value: is the real mean of `values`
    (e.g. each advised bet's real bet_profit_units) distinguishable from
    `null_value` (default 0.0 - "breaking even")? This is the honest
    test for "did the advised bets actually make money, or is this
    within noise" - deliberately NOT binomial_significance on
    win_rate_on_advised_bets, which would silently throw away each bet's
    real price (a -150 favorite winning 55% of the time and a +150
    underdog winning 55% of the time are very different real outcomes
    that a win-rate-only test can't tell apart; the real profit each bet
    produced already prices that in). Needs at least 2 real resolved
    values to estimate a variance; returns NaN otherwise, not a
    fabricated p-value off a single data point."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    return float(ttest_1samp(clean, null_value).pvalue)


def calibration_table(predictions: pd.DataFrame, n_bins: int = 10, outcome_col: str = "actual_hit") -> pd.DataFrame:
    """Bins picks by predicted probability and compares each bin's mean
    predicted probability to its actual hit rate - a well-calibrated metric
    should have predicted_mean ~= actual_rate in every bin."""
    resolved = resolved_only(predictions, outcome_col)
    if resolved.empty:
        return pd.DataFrame(columns=["bin", "predicted_mean", "actual_rate", "n"])

    bins = pd.cut(resolved["predicted_probability"], bins=n_bins, include_lowest=True)
    grouped = (
        resolved.groupby(bins, observed=True)
        .agg(predicted_mean=("predicted_probability", "mean"), actual_rate=(outcome_col, "mean"), n=(outcome_col, "size"))
        .reset_index(names="bin")
    )
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def _filter_metric(predictions: pd.DataFrame, metric: str | None) -> pd.DataFrame:
    if metric is None:
        return predictions
    # Game_Hit_Probability's live successor is Final_Hit_Probability - include
    # both when the caller asks for the historical default metric name so
    # dashboard exports spanning the cutover stay complete.
    if metric == "Game_Hit_Probability":
        return predictions[
            predictions["metric"].isin(["Game_Hit_Probability", "Final_Hit_Probability"])
        ]
    return predictions[predictions["metric"] == metric]


def _filter_model_version(predictions: pd.DataFrame, model_version: str | None) -> pd.DataFrame:
    """Restricts to rows tagged with `model_version` (see
    predictions.select_picks/config.HITTER_MODEL_VERSION) - None (the
    default everywhere below) means "all versions, unfiltered", preserving
    every existing caller's behavior byte-for-byte. Passing a specific
    version is what lets a real recalibration's effect actually show up in
    these stats instead of being diluted by pre-change history forever."""
    if model_version is None:
        return predictions
    if "model_version" not in predictions.columns:
        return predictions.iloc[0:0]
    return predictions[predictions["model_version"] == model_version]


def _legacy_combined_probability(df: pd.DataFrame) -> pd.Series:
    """Legacy blended recommendation score (pre-Final_Hit_Probability)."""
    columns = [c for c in ("predicted_probability", "probability", "Matchup_Hit_Probability") if c in df.columns]
    return df[columns].astype(float).mean(axis=1, skipna=True)


def _uses_final_hit_probability(df: pd.DataFrame) -> pd.Series:
    """Per-row: True when Final_Hit_Probability is the authoritative score."""
    source = df["probability_source"] if "probability_source" in df.columns else pd.Series(pd.NA, index=df.index)
    sel = df["selection_metric"] if "selection_metric" in df.columns else pd.Series(pd.NA, index=df.index)
    metric = df["metric"] if "metric" in df.columns else pd.Series(pd.NA, index=df.index)
    mode = df["selection_mode"] if "selection_mode" in df.columns else pd.Series(pd.NA, index=df.index)
    return (
        (source == "Final_Hit_Probability")
        | (sel == "Final_Hit_Probability")
        | (metric == "Final_Hit_Probability")
        | (mode == "live")
    )


def _combined_probability(df: pd.DataFrame) -> pd.Series:
    """Recommendation / grading score for dashboard + streak logic.

    Live / Final_Hit_Probability rows use ``predicted_probability`` alone
    (which equals Final_Hit_Probability). Older rows without that authority
    keep the legacy three-signal mean. Never silently mixes the two on a
    live row.
    """
    if df.empty:
        return pd.Series(dtype=float)
    legacy = _legacy_combined_probability(df)
    if not _uses_final_hit_probability(df).any():
        return legacy
    authoritative = pd.to_numeric(df["predicted_probability"], errors="coerce")
    use_final = _uses_final_hit_probability(df)
    return authoritative.where(use_final, legacy)


def _recommended_picks(
    predictions: pd.DataFrame,
    metric: str | None,
    max_picks: int,
    min_probability: float,
    model_version: str | None = None,
) -> pd.DataFrame:
    """The subset of logged picks that actually count toward the tracked
    streak/day_survival_rate for a given day: top-ranked, capped at
    `max_picks`, and only those whose _combined_probability clears
    `min_probability` ("a good matchup"). A day can have 0, 1, or
    `max_picks` rows here depending on how many clear the bar - it's never
    padded out to a fixed count. This is the STREAK-COUNTING definition of
    "recommended" only - see graded_daily_picks for what the dashboard
    actually displays, which is a superset of this (every logged
    candidate, not just the ones that clear the bar)."""
    df = _filter_metric(predictions, metric)
    df = _filter_model_version(df, model_version)
    df = df[(df["rank"] <= max_picks) & (_combined_probability(df) >= min_probability)]
    return df


def graded_daily_picks(
    predictions: pd.DataFrame,
    metric: str | None,
    max_picks: int,
    min_probability: float,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Every day's top `max_picks` candidates by rank, ALWAYS returned -
    unlike _recommended_picks, a day is never empty just because nobody
    cleared `min_probability`. Each row gets its own real `combined_probability`
    and a `grade` ("recommended" if that clears `min_probability` - the
    exact same bar/columns _recommended_picks gates on, so a "recommended"
    grade here is precisely what counts toward the tracked streak/
    day_survival_rate, see streak_progression - else "speculative"). A
    "speculative" pick is still a real, already-qualified candidate
    (predictions.select_picks already gated it on HITTER_MIN_PROBABILITY/
    the model shortlist before it was ever logged) - just below the
    backtested confidence bar, shown for visibility rather than hidden.

    Added because DAILY_PICK_MIN_PROBABILITY (0.77) was validated on a
    42-day historical replay where Matchup_Hit_Probability was always NaN
    (never persisted to git history at the time - see that constant's own
    docstring) - once live runs started actually carrying real
    Matchup_Hit_Probability values most days, the blended mean runs lower
    on an ordinary day than that replay ever exercised, and real live data
    hit a 5-day-straight stretch (2026-08-11 through 2026-08-15) where the
    top-ranked candidate's real combined probability landed at 0.71-0.77 -
    just under the bar every single day, producing a blank dashboard
    despite real, qualified candidates existing every one of those days.
    Rather than re-chase a moving threshold, the dashboard now always shows
    its real top candidates, graded honestly, instead of going blank."""
    df = _filter_metric(predictions, metric)
    df = _filter_model_version(df, model_version)
    df = df[df["rank"] <= max_picks].copy()
    df["combined_probability"] = _combined_probability(df)
    df["grade"] = np.where(df["combined_probability"] >= min_probability, "recommended", "speculative")
    return df


def _classify_outcome(df: pd.DataFrame) -> pd.Series:
    """Per-pick outcome for dashboard/export status strings. Same states as
    `classify_pick_states` (pending / no_game / hit / miss) - kept as a
    thin alias so streak_progression / graded exports share one definition
    with selection_strategy_metrics."""
    return classify_pick_states(df)


def streak_progression(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Day-by-day Beat the Streak simulation using the real game's actual
    rules, not a simplified win/loss-per-day model:

    - A pick with >=1 at-bat and no hit ("miss") resets the streak to 0,
      no matter what the other pick (if any) did that day.
    - Otherwise the streak increases by however many picks got a hit that
      day (0, 1, or up to max_picks) - a pick with 0 at-bats ("no_game")
      contributes nothing, positive or negative.
    - A day isn't processed at all until every pick logged for it is
      resolved (no "pending" outcomes) - it's simply skipped, not counted
      as a break, until the data catches up.
    - A day with zero recommended picks (no good matchup - see
      _recommended_picks) never appears in the log in the first place, so
      it's implicitly skipped too, which is exactly the desired no-op.

    Returns one row per resolved day, oldest first: date, the running
    streak value after that day, and whether that day reset it.
    """
    df = _recommended_picks(predictions, metric, max_picks, min_probability, model_version).copy()
    df["outcome"] = _classify_outcome(df)

    rows = []
    streak = 0
    for date, day in df.sort_values("date").groupby("date", sort=True):
        if (day["outcome"] == "pending").any():
            continue
        reset = bool((day["outcome"] == "miss").any())
        if reset:
            streak = 0
        else:
            streak += int((day["outcome"] == "hit").sum())
        rows.append({"date": date, "streak": streak, "reset": reset})

    return pd.DataFrame(rows, columns=["date", "streak", "reset"])


def longest_streak(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
    model_version: str | None = None,
) -> int:
    progression = streak_progression(predictions, metric, max_picks, min_probability, model_version)
    return int(progression["streak"].max()) if len(progression) else 0


def current_streak(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
    model_version: str | None = None,
) -> int:
    """Streak value as of the most recently *resolved* day (a trailing
    run of still-pending or no-pick days doesn't change this)."""
    progression = streak_progression(predictions, metric, max_picks, min_probability, model_version)
    return int(progression["streak"].iloc[-1]) if len(progression) else 0


def build_beat_the_streak_export(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
    model_version: str | None = None,
):
    """Build the two tables the dashboard's Beat the Streak section reads:
    (picks_table, summary_row). picks_table is every day's top `max_picks`
    candidates by rank (see graded_daily_picks) with a hit/miss/no_game/
    pending status AND a "recommended"/"speculative" grade, most recent day
    first - unlike before, a day with real logged candidates is NEVER blank
    just because none cleared `min_probability`; it always shows its real
    best options, honestly graded. summary_row has longest_streak/
    current_streak plus a day_survival_rate (fraction of resolved days that
    didn't reset the streak - a looser sanity metric than the streak count
    itself, since a single miss zeroes a long streak) - these are still
    computed from ONLY "recommended"-grade picks (see streak_progression/
    _recommended_picks, unchanged), so a "speculative" day is still a
    real no-op for the tracked streak, exactly like a no_pick day was
    before this function's picks_table stopped going blank on those days.

    A date is only absent from picks_table's real rows (and gets the
    explicit "no_pick" placeholder row instead) when NOTHING was logged for
    it at all - a genuine off day (All-Star break), a rainout across the
    whole slate, or a pipeline gap - not a weak-slate day, which now gets a
    real "speculative" row instead of vanishing.

    `model_version` (default None, i.e. every version blended together -
    unchanged behavior) restricts to picks tagged with a specific
    predictions.select_picks model_version (see config.HITTER_MODEL_VERSION)
    - the summary row's own "model_version" column is set to whatever was
    passed (or "all_time" when None), so pipeline.py can build one small
    CSV covering both views without ambiguity about which row is which."""
    picks = graded_daily_picks(predictions, metric, max_picks, min_probability, model_version).copy()
    picks["status"] = _classify_outcome(picks)
    # game_pk is optional for legacy CSV rows (null) and required on new
    # live rows - surface it when present so the export stays aligned with
    # predictions.PREDICTION_COLUMNS without breaking older logs.
    if "game_pk" not in picks.columns:
        picks["game_pk"] = pd.NA
    picks = picks[
        ["date", "game_pk", "rank", "name", "predicted_probability", "combined_probability", "actual_hit", "status", "grade"]
    ]

    # graded_daily_picks already includes every rank<=max_picks candidate
    # regardless of grade, so a date can only be missing from `picks` here
    # when NOTHING was logged for it at all (see docstring above) - surface
    # that explicitly as its own row rather than leaving the date silently
    # absent, which a reader (or the dashboard) would otherwise misread as
    # "the most recent day was some earlier date."
    filtered = _filter_model_version(_filter_metric(predictions, metric), model_version)
    no_pick_dates = sorted(set(filtered["date"]) - set(picks["date"]))
    if no_pick_dates:
        no_pick_rows = pd.DataFrame(
            {
                "date": no_pick_dates,
                "game_pk": pd.NA,
                "rank": pd.NA,
                "name": pd.NA,
                "predicted_probability": pd.NA,
                "combined_probability": pd.NA,
                "actual_hit": pd.NA,
                "status": "no_pick",
                "grade": pd.NA,
            }
        )
        picks = pd.concat([picks, no_pick_rows], ignore_index=True)

    picks = picks.sort_values(["date", "rank"], ascending=[False, True]).reset_index(drop=True)

    progression = streak_progression(predictions, metric, max_picks, min_probability, model_version)
    n_days = len(progression)
    survival_successes = int((~progression["reset"]).sum()) if n_days else 0
    survival_rate = float(survival_successes / n_days) if n_days else float("nan")
    # Quant-analytics item #5 ("backtest scope and statistical
    # significance"): CI only here, deliberately no binomial_significance
    # p-value - "recommended" picks are already gated on a high
    # min_probability bar, so there's no real symmetric "no skill" null
    # to test survival against the way beat_closing_line_rate has one
    # (see game_evaluation.py's own choice of which metrics get a
    # p-value). The CI still answers the real question a small n_days
    # raises: how much could this rate move with more data.
    survival_ci_low, survival_ci_high = wilson_confidence_interval(survival_successes, n_days)

    summary = pd.DataFrame(
        [
            {
                "model_version": model_version if model_version is not None else "all_time",
                "metric": metric,
                "max_picks": max_picks,
                "min_probability": min_probability,
                "n_days_resolved": n_days,
                "day_survival_rate": survival_rate,
                "day_survival_rate_ci_low": survival_ci_low,
                "day_survival_rate_ci_high": survival_ci_high,
                "longest_streak": int(progression["streak"].max()) if n_days else 0,
                "current_streak": int(progression["streak"].iloc[-1]) if n_days else 0,
            }
        ]
    )
    return picks, summary


def summarize(predictions: pd.DataFrame, top_k_values=(1, 2, 5), model_version: str | None = None) -> pd.DataFrame:
    """One-row-per-metric summary table, split by the `metric` column so
    multiple candidate metrics (e.g. "probability" vs "Game_Hit_Probability")
    logged into the same predictions file can be compared directly.

    `model_version` (default None, i.e. every version blended together -
    unchanged behavior) restricts to picks tagged with a specific
    predictions.select_picks model_version (see config.HITTER_MODEL_VERSION) -
    this is what lets a real recalibration's effect on accuracy/Brier/etc.
    actually show up, instead of being diluted by pre-change history."""
    predictions = _filter_model_version(predictions, model_version)
    rows = []
    for metric_name, group in predictions.groupby("metric"):
        resolved = resolved_only(group)
        row = {
            "metric": metric_name,
            "n_resolved": len(resolved),
            "brier_score": brier_score(group),
            "log_loss": log_loss(group),
        }
        for k in top_k_values:
            row[f"any_of_top_{k}_hit_rate"] = top_k_hit_rate(group, k, require_all=False)
            row[f"all_of_top_{k}_hit_rate"] = top_k_hit_rate(group, k, require_all=True)
        rows.append(row)
    return pd.DataFrame(rows)
