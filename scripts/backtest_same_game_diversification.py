"""Real go/no-go backtest for quant-analytics item #4, slice 2's same-game
diversification tie-break (predictions.select_picks's
same_game_diversification_margin - see that function's own docstring and
predictions._diversify_second_pick for the real rule this validates).

No-lookahead, reusing dfs_backtest._compute_date_outputs's per-date
recompute - the same technique scripts/backtest_selection_rule.py already
established for validating a predictions.select_picks change, and this
script deliberately mirrors that one's build_date_pools/select_and_resolve/
report shape rather than reinventing it. The one real addition: a
(date, batter) -> game_pk lookup, built once from the already-loaded raw
Statcast DataFrame (no new fetch - the same source
scripts/backtest_selection_rule.py already loads), merged onto each
date's pick_pool so predictions.select_picks can actually see which
candidates share a real game.

Restricted to the SAME final holdout date block
scripts/train_hitter_hit_model.py/scripts/backtest_selection_rule.py
already validate against (config.ML_FINAL_HOLDOUT_DATES).

Sweeps same_game_diversification_margin over a small real grid. Primary
go/no-go metric: evaluation.top_k_hit_rate(picks, k=2, require_all=True) -
the REAL "both picks must hit" Beat the Streak rule (deliberately not
just require_all=False, which backtest_selection_rule.py already reports
but which doesn't actually test what diversification is meant to
improve). Reported honestly - the real same-game pair count in this
holdout window is thin (see README's "Same-game diversification"
section for the real numbers), so an inconclusive result here is a real,
expected possible outcome, not a failure of the method.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py).

Usage:
    python scripts/backtest_same_game_diversification.py
    python scripts/backtest_same_game_diversification.py --margin-grid 0,0.02,0.05,0.1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, dfs_backtest, dfs_ml, evaluation, predictions

DEFAULT_MARGIN_GRID = [0.0, 0.02, 0.05, 0.1]


def build_date_pools(dates, persisted: pd.DataFrame, team_schedule: pd.DataFrame) -> list[dict]:
    """Same shape as scripts/backtest_selection_rule.py's own
    build_date_pools (Matchup_Approach pick_pool per real historical
    date, expensive per-date recompute done once regardless of how many
    margins get evaluated afterward). Matchup already carries a real
    game_pk per batter-game, so same-game diversification sees both DH
    halves without a separate (date, batter) lookup that would collapse
    them."""
    pools = []
    for date in dates:
        day = dfs_backtest._compute_date_outputs(persisted, team_schedule, date)
        if day is None:
            continue

        pick_pool = day["outputs"]["wave"].merge(day["matchup_probability"], on="key_mlbam", how="inner")
        pick_pool["Matchup_Approach"] = pick_pool["Approach"] * pick_pool["Matchup_Hit_Probability"]

        day_events = persisted[persisted["game_date"] == date]
        got_hit = dfs_backtest.compute_actual_hitter_got_hit(
            data.completed_events(day_events, ["game_date", "game_pk", "batter", "events"])
        )
        pools.append({"date": date, "pick_pool": pick_pool, "got_hit": got_hit})
    return pools


def select_and_resolve(pools: list[dict], rank_metric: str, margin: float) -> pd.DataFrame:
    """Runs predictions.select_picks on each cached date pool at the given
    same_game_diversification_margin, then resolves each returned pick
    against that contest's REAL Got_Hit outcome - same resolve logic as
    scripts/backtest_selection_rule.py's own select_and_resolve."""
    rows = []
    for entry in pools:
        picks = predictions.select_picks(
            entry["pick_pool"], entry["date"], rank_metric=rank_metric, same_game_diversification_margin=margin
        )
        if picks.empty:
            continue

        resolve_keys = ["key_mlbam"]
        if "game_pk" in picks.columns and "game_pk" in entry["got_hit"].columns:
            resolve_keys = ["game_pk", "key_mlbam"]
        picks = picks.merge(
            entry["got_hit"].rename(columns={"Got_Hit": "resolved_hit"}), on=resolve_keys, how="left"
        )
        picks["at_bats"] = picks["resolved_hit"].notna().astype(int)
        picks["actual_hit"] = picks["resolved_hit"]
        picks = picks.drop(columns="resolved_hit")
        rows.append(picks)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=predictions.PREDICTION_COLUMNS)


def _count_same_game_pairs(pools: list[dict], margin: float, rank_metric: str) -> int:
    """How many real dates in this holdout actually had a #1/#2 sharing a
    game_pk under the GIVEN margin's resulting picks - the honest
    denominator for how much this margin could possibly have changed.
    game_pk is now part of PREDICTION_COLUMNS, so it is read off the picks
    themselves."""
    count = 0
    for entry in pools:
        picks = predictions.select_picks(
            entry["pick_pool"], entry["date"], rank_metric=rank_metric, same_game_diversification_margin=0.0
        )
        if len(picks) < 2:
            continue
        top_game = picks.iloc[0]["game_pk"]
        second_game = picks.iloc[1]["game_pk"]
        if pd.notna(top_game) and top_game == second_game:
            count += 1
    return count


def report(margin: float, picks_df: pd.DataFrame, total_dates: int) -> None:
    resolved = evaluation.resolved_only(picks_df)
    dates_with_a_pick = picks_df["date"].nunique() if not picks_df.empty else 0
    print(f"\nmargin={margin}")
    print(f"  n_scored={len(resolved)}, dates_with_a_pick={dates_with_a_pick}/{total_dates}")
    both_rate = evaluation.top_k_hit_rate(picks_df, 2, require_all=True)
    any_rate = evaluation.top_k_hit_rate(picks_df, 2, require_all=False)
    print(f"  top_2_BOTH_hit_rate (require_all=True, the real BTS rule) = {both_rate:.4f}" if both_rate == both_rate else "  top_2_BOTH_hit_rate=n/a")
    print(f"  top_2_ANY_hit_rate (require_all=False, context only)      = {any_rate:.4f}" if any_rate == any_rate else "  top_2_ANY_hit_rate=n/a")
    brier = evaluation.brier_score(picks_df)
    ll = evaluation.log_loss(picks_df)
    print(f"  brier_score={brier:.4f}" if brier == brier else "  brier_score=n/a")
    print(f"  log_loss={ll:.4f}" if ll == ll else "  log_loss=n/a")
    return both_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--holdout-dates", type=int, default=config.ML_FINAL_HOLDOUT_DATES)
    parser.add_argument("--margin-grid", default=",".join(str(x) for x in DEFAULT_MARGIN_GRID))
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(args.raw_dir, season)
    if persisted is None:
        print(f"No persisted Statcast in {args.raw_dir} for season {season} - nothing to backtest.")
        return

    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    all_dates = sorted(team_schedule["date"].unique())
    dates = all_dates[-args.holdout_dates:] if len(all_dates) > args.holdout_dates else all_dates
    print(f"Backtesting same-game diversification over the final {len(dates)} real dates (of {len(all_dates)} total).")

    pools = build_date_pools(dates, persisted, team_schedule)
    total_dates = len(pools)
    same_game_days = _count_same_game_pairs(pools, 0.0, "Matchup_Approach")
    same_game_share = f"{same_game_days / total_dates:.1%}" if total_dates else "n/a"
    print(
        f"{total_dates} dates have usable prior history; {same_game_days} of those ({same_game_share}) "
        f"have a real #1/#2 same-game pair under today's unmodified ranking (margin=0.0) - "
        f"the honest ceiling on how much this margin could change."
    )
    if same_game_days == 0:
        print("No same-game pairs at all in this holdout - nothing for a nonzero margin to possibly improve. Reporting margin=0.0 only.")

    margin_grid = [float(x) for x in args.margin_grid.split(",")]
    results = {}
    for margin in margin_grid:
        picks = select_and_resolve(pools, "Matchup_Approach", margin)
        results[margin] = report(margin, picks, total_dates)

    baseline = results.get(0.0)
    if baseline is None or baseline != baseline:
        print("\n-> strength=0 baseline unscored - cannot judge go/no-go")
        return
    candidates = {m: r for m, r in results.items() if m != 0.0 and r == r}
    if not candidates:
        print("\n-> no nonzero margin produced scored rows")
        return
    best_margin = max(candidates, key=lambda m: candidates[m])
    print(
        f"\nSmall-sample caveat: only {same_game_days} of {total_dates} real holdout dates had a same-game "
        f"#1/#2 pair to begin with - treat any margin comparison here as suggestive, not conclusive."
    )
    if candidates[best_margin] > baseline:
        print(
            f"-> GO: margin={best_margin} beats margin=0.0 (unmodified) on the real top_2_BOTH_hit_rate "
            f"({candidates[best_margin]:.4f} vs. {baseline:.4f}) - candidate for a nonzero live default."
        )
    else:
        print(
            f"-> NO-GO: margin=0.0 (unmodified) remains best on the real top_2_BOTH_hit_rate "
            f"({baseline:.4f}) - reported honestly, no live default earned on this holdout."
        )


if __name__ == "__main__":
    main()
