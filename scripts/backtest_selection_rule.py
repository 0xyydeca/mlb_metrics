"""Real backtest of predictions.select_picks' SELECTION RULE - not just
the hit-probability model's own already-known calibration accuracy
(scripts/train_hitter_hit_model.py's holdout log_loss/Brier/ROC AUC
answers that question). This answers a different, real one: does using
Model_Hit_Probability as a broad top-N shortlist gate (config.
HITTER_MODEL_SHORTLIST_SIZE, then Matchup_Approach ranks the survivors -
see predictions.select_picks/config.HITTER_MODEL_VERSION's v4 paragraph)
produce a better realized selection than the plain Matchup_Approach
heuristic alone - a question about discriminative power at the extreme
top of the distribution, not average calibration.

Headline two-pick metrics follow Beat the Streak's real rules via
evaluation.selection_strategy_metrics:
  - all_of_top_2_hit_rate (every selected pick hit - voids fail this)
  - top_2_reset_rate (any miss resets the day)
  - top_2_survival_rate (no miss; voids allowed)
any_of_top_2_hit_rate is reported only as a secondary descriptive metric.
Strategy differences use a paired DATE-block bootstrap
(evaluation.bootstrap_strategy_metric_difference) - whole dates, never
independent pick rows.

No-lookahead, reusing dfs_backtest._compute_date_outputs's per-date
recompute (the same technique every other backtest in this project uses)
and the REAL predictions.select_picks/evaluation.py functions directly -
not a reimplementation of either.

Restricted to the SAME final holdout date block
scripts/train_hitter_hit_model.py validated the saved model artifact on
(config.ML_FINAL_HOLDOUT_DATES) - evaluating on training dates would be
leakage. Uses the LAST ML_FINAL_HOLDOUT_DATES dates of whatever's
currently persisted, so any real Statcast history accumulated since the
model's original training run naturally widens the window.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py) and the
saved model artifact (config.HITTER_HIT_PROBABILITY_MODEL_PATH,
scripts/train_hitter_hit_model.py).

Usage:
    python scripts/backtest_selection_rule.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, dfs_backtest, dfs_ml, evaluation, predictions

DEFAULT_BOOTSTRAP_SAMPLES = 1000


def build_date_pools(dates, persisted: pd.DataFrame, team_schedule: pd.DataFrame) -> list[dict]:
    """One pass over `dates`: for each real historical date, builds the
    Matchup_Approach pick_pool (today's live heuristic tier) merged with
    Model_Hit_Probability (when the model predicts something for that
    date), plus that date's real outcomes. Reused by both variants in
    select_and_resolve below, so the expensive per-date pipeline recompute
    + model inference happens exactly once per date regardless of how many
    variants get evaluated afterward."""
    pools = []
    for date in dates:
        day = dfs_backtest._compute_date_outputs(persisted, team_schedule, date)
        if day is None:
            continue

        pick_pool = day["outputs"]["wave"].merge(day["matchup_probability"], on="key_mlbam", how="inner")
        pick_pool["Matchup_Approach"] = pick_pool["Approach"] * pick_pool["Matchup_Hit_Probability"]

        hitter_features = dfs_ml.build_hitter_features(
            day["outputs"]["wave"], day["outputs"]["pave"], day["outputs"]["confidence"],
            day["todays_schedule"], day["matchup_probability"],
        )
        model_predictions = dfs_ml.predict_hitter_hit_probability(hitter_features)
        has_model = not model_predictions.empty
        if has_model:
            pick_pool = pick_pool.merge(model_predictions, on=["key_mlbam", "game_pk"], how="left")

        day_events = persisted[persisted["game_date"] == date]
        got_hit = dfs_backtest.compute_actual_hitter_got_hit(
            data.completed_events(day_events, ["game_date", "game_pk", "batter", "events"])
        )
        pools.append({"date": date, "pick_pool": pick_pool, "got_hit": got_hit, "has_model": has_model})
    return pools


def select_and_resolve(
    pools: list[dict], rank_metric: str, variant: str = "heuristic_only", **select_kwargs
) -> pd.DataFrame:
    """Runs predictions.select_picks on each cached date pool (skipping a
    date the model has no usable prediction for, when
    `variant="model_shortlist"`), then resolves each returned pick against
    that date's REAL Got_Hit outcome. Returns a
    predictions.PREDICTION_COLUMNS-shaped frame with actual_hit/at_bats
    already filled in, directly consumable by evaluation.py's scoring
    functions.

    select_picks' shortlist step activates off Model_Hit_Probability's
    mere presence as a column, regardless of rank_metric - so when
    simulating `variant="heuristic_only"`, the column is dropped first.
    Otherwise a pool built for the model-shortlist simulation (which also
    carries Model_Hit_Probability for the OTHER simulation on the same
    date) would wrongly apply the shortlist step to the heuristic-only
    simulation too. Both variants always rank by the same `rank_metric`
    (Matchup_Approach) - the only difference is whether the model
    shortlist narrows the pool first."""
    rows = []
    for entry in pools:
        if variant == "model_shortlist" and not entry["has_model"]:
            continue
        pick_pool = entry["pick_pool"]
        if variant == "heuristic_only":
            pick_pool = pick_pool.drop(columns="Model_Hit_Probability", errors="ignore")
        picks = predictions.select_picks(pick_pool, entry["date"], rank_metric=rank_metric, **select_kwargs)
        if picks.empty:
            continue

        resolve_keys = ["key_mlbam"]
        if "game_pk" in picks.columns and "game_pk" in entry["got_hit"].columns:
            resolve_keys = ["game_pk", "key_mlbam"]
        picks = picks.merge(
            entry["got_hit"].rename(columns={"Got_Hit": "resolved_hit"}), on=resolve_keys, how="left"
        )
        # Present in that contest's real completed-events table => had a
        # real at-bat (at_bats=1, don't need the exact count here);
        # absent => a picked player with zero at-bats that game (no_game).
        picks["at_bats"] = picks["resolved_hit"].notna().astype(int)
        picks["actual_hit"] = picks["resolved_hit"]
        picks = picks.drop(columns="resolved_hit")
        rows.append(picks)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=predictions.PREDICTION_COLUMNS)


def _fmt_rate(value) -> str:
    return f"{value:.4f}" if value == value else "n/a"


def report(label: str, picks_df: pd.DataFrame, n_candidate_dates: int) -> dict:
    """Print Beat the Streak selection-rule metrics for one strategy.

    Headline two-pick lines are all_of_top_2_hit_rate and top_2_reset_rate
    (plus survival). any_of_top_2 is secondary only - Beat the Streak
    requires every selected pick to avoid a miss, not "at least one hit"."""
    metrics = evaluation.selection_strategy_metrics(picks_df, k=2, n_candidate_dates=n_candidate_dates)
    print(f"\n{label}")
    print(
        f"  n_candidate_dates={metrics['n_candidate_dates']}, "
        f"n_dates_with_picks={metrics['n_dates_with_picks']}, "
        f"coverage_rate={_fmt_rate(metrics['coverage_rate'])}"
    )
    print(
        f"  n_resolved_pick_rows={metrics['n_resolved_pick_rows']}, "
        f"no_game_rate={_fmt_rate(metrics['no_game_rate'])}, "
        f"conditional_hit_rate_given_at_bat={_fmt_rate(metrics['conditional_hit_rate_given_at_bat'])}"
    )
    print(
        f"  top_1_advance_rate={_fmt_rate(metrics['top_1_advance_rate'])}, "
        f"top_1_reset_rate={_fmt_rate(metrics['top_1_reset_rate'])}"
    )
    print(
        f"  HEADLINE all_of_top_2_hit_rate={_fmt_rate(metrics['all_of_top_2_hit_rate'])}, "
        f"top_2_reset_rate={_fmt_rate(metrics['top_2_reset_rate'])}, "
        f"top_2_survival_rate={_fmt_rate(metrics['top_2_survival_rate'])}"
    )
    print(
        f"  secondary any_of_top_2_hit_rate={_fmt_rate(metrics['any_of_top_2_hit_rate'])}, "
        f"mean_hits_added_per_played_day={_fmt_rate(metrics['mean_hits_added_per_played_day'])}"
    )
    print(
        f"  brier_score={_fmt_rate(metrics['brier_score'])}, "
        f"log_loss={_fmt_rate(metrics['log_loss'])}"
    )
    return metrics


def report_bootstrap_comparison(
    picks_a: pd.DataFrame,
    picks_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    candidate_dates,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_state: int = 0,
) -> None:
    """Paired date-block bootstrap for the headline two-pick metrics."""
    print(f"\nDate-block bootstrap ({n_bootstrap} resamples of whole dates): {label_a} − {label_b}")
    for metric in ("all_of_top_2_hit_rate", "top_2_reset_rate", "top_2_survival_rate"):
        result = evaluation.bootstrap_strategy_metric_difference(
            picks_a, picks_b, metric=metric, k=2,
            n_bootstrap=n_bootstrap, candidate_dates=candidate_dates, random_state=random_state,
        )
        print(
            f"  {metric}: point={_fmt_rate(result['point_difference'])} "
            f"CI=[{_fmt_rate(result['ci_low'])}, {_fmt_rate(result['ci_high'])}] "
            f"(unit={result['resampled_unit']}, n_dates={result['n_dates']})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--holdout-dates", type=int, default=config.ML_FINAL_HOLDOUT_DATES)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(args.raw_dir, season)
    if persisted is None:
        print(f"No persisted Statcast in {args.raw_dir} for season {season} - nothing to backtest.")
        return

    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    all_dates = sorted(team_schedule["date"].unique())
    dates = all_dates[-args.holdout_dates:] if len(all_dates) > args.holdout_dates else all_dates
    print(f"Backtesting the selection rule over the final {len(dates)} real dates (of {len(all_dates)} total) - "
          f"the same holdout block scripts/train_hitter_hit_model.py validates the saved model artifact on.")

    pools = build_date_pools(dates, persisted, team_schedule)
    total_dates = len(pools)
    dates_with_model = sum(1 for p in pools if p["has_model"])
    print(f"{total_dates} dates have usable prior history; {dates_with_model} of those have a usable Model_Hit_Probability prediction.")

    heuristic_picks = select_and_resolve(pools, "Matchup_Approach", variant="heuristic_only")
    report("Matchup_Approach only (no model shortlist)", heuristic_picks, total_dates)

    model_shortlist_picks = select_and_resolve(pools, "Matchup_Approach", variant="model_shortlist")
    report(
        f"Model shortlist (top {config.HITTER_MODEL_SHORTLIST_SIZE} by Model_Hit_Probability, then Matchup_Approach)",
        model_shortlist_picks, dates_with_model,
    )

    candidate_dates = [p["date"] for p in pools if p["has_model"]]
    if candidate_dates and not heuristic_picks.empty and not model_shortlist_picks.empty:
        # Restrict heuristic picks to the same model-available dates so the
        # paired bootstrap compares strategies on a shared information set.
        heuristic_on_model_dates = heuristic_picks[heuristic_picks["date"].isin(candidate_dates)]
        report_bootstrap_comparison(
            model_shortlist_picks, heuristic_on_model_dates,
            label_a="model_shortlist", label_b="heuristic_only",
            candidate_dates=candidate_dates,
            n_bootstrap=args.bootstrap_samples,
            random_state=args.bootstrap_seed,
        )


if __name__ == "__main__":
    main()
