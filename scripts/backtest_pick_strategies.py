"""Deep, no-lookahead, full-season backtest comparing candidate Beat the
Streak selection strategies - the direct answer to "does the current,
much more elaborate automated system actually beat my original manual
heuristic (WAVE-table odds vs. probable pitcher's PAVE), and by how much,
measured in a way that resists confirmation bias?"

## Methodology (pre-registered before any candidate was scored)

Three candidates, fixed BEFORE this script was ever run against real
results - not searched over after seeing which one "wins":

  c0_legacy_approach   - pipeline.run's own real no-schedule fallback:
                          rank by Approach (Game_Hit_Probability *
                          probability) alone, no PAVE/matchup signal at
                          all. A baseline, not the main comparison.
  c1_wave_vs_pave       - the user's own original manual method: PA
                          qualifier only (no lineup/recency qualifiers,
                          no probability floor), ranked by Matchup_Approach
                          (Approach * Matchup_Hit_Probability - the WAVE
                          side blended with the probable starter's PAVE).
  c2_current_heuristic  - today's real live selection logic
                          (predictions.select_picks's full qualifier
                          chain: PA, lineup top-half/start-rate, recency,
                          teams-playing-today), ranked by Matchup_Approach.
                          The trained ML shortlist (Model_Hit_Probability)
                          is deliberately EXCLUDED here even on dates after
                          it went live: the committed model artifact was
                          fit on data through ~2026-08-25, so scoring it
                          against early-season dates would be real
                          lookahead/leakage (it already saw those
                          outcomes as training data). The narrower "does
                          the ML shortlist help" question is answered
                          separately by scripts/backtest_selection_rule.py,
                          which is already restricted to the model's own
                          genuine holdout window.

All three candidates are SELECTED with min_probability=0.0 (the
HITTER_MIN_PROBABILITY=0.7 selection-time floor is deliberately not
applied to any candidate here) - confirmed against real data that almost
no hitter clears 0.7 on probability/Game_Hit_Probability/
Matchup_Hit_Probability jointly, and in real live operation that gate is
only ever cleared in practice because Model_Hit_Probability REPLACES it
with a top-10 shortlist (excluded here for leak-safety). The real,
currently-live "recommended" bar downstream of selection is
DAILY_PICK_MIN_PROBABILITY (0.77) - applied uniformly to every
candidate's top DAILY_PICK_MAX picks at evaluation time instead (see
scripts/analyze_pick_strategy_backtest.py), so all three are graded by
the same real bar the dashboard actually uses today, not a stale
selection-time floor that predates the model shortlist that now does its
job.

Every candidate is recomputed fresh, per real historical date, from
ONLY Statcast strictly before that date (dfs_backtest._compute_date_outputs
- the same no-lookahead technique this project's other backtests already
use) - not from the live predictions.csv log, whose selection logic
actually changed over the season. This is what makes the comparison
apples-to-apples: each candidate's FIXED, CURRENT-shape logic is applied
uniformly across the whole season, not the shifting mix of logic that was
actually live day to day.

## Avoiding confirmation bias

- The 3 candidates above are fixed before running; no post-hoc "try a 4th
  variant that scores better" search.
- PRIMARY metric is per-pick hit rate with a real Wilson confidence
  interval (evaluation.wilson_confidence_interval) among each candidate's
  top DAILY_PICK_MAX picks/day - not "longest streak", which is a noisy
  extreme-value statistic (a single lucky/unlucky day swings it a lot)
  unsuitable as a primary optimization target. streak_progression's real
  longest/current streak is still reported, but as a secondary/
  illustrative stat only.
- The full window is split into first-half/second-half so an apparent
  winner can be checked for holding up in both halves independently,
  not just riding one hot stretch of the season.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, dfs_backtest, helpers, matchup, predictions

LINEUP_QUALIFIER_COLUMNS = ["avg_batting_order", "start_rate", "Last_Game_Date"]


def _resolve_picks(picks: pd.DataFrame, persisted: pd.DataFrame) -> pd.DataFrame:
    """Same at_bats/actual_hit resolution logic as
    predictions.resolve_predictions, applied in-memory against the full
    persisted history instead of round-tripping through a CSV - every
    date in this backtest is already fully in the past, so nothing here
    is ever "still pending"."""
    events = data.completed_events(persisted, ["game_date", "batter", "events"]).copy()
    events["had_hit"] = helpers.is_hit(events["events"])
    per_batter_day = (
        events.groupby(["game_date", "batter"])
        .agg(resolved_at_bats=("events", "size"), resolved_hit=("had_hit", "max"))
        .reset_index()
        .rename(columns={"game_date": "date", "batter": "key_mlbam"})
    )
    picks = picks.merge(per_batter_day, on=["date", "key_mlbam"], how="left")
    picks["at_bats"] = picks["resolved_at_bats"].fillna(0)
    got_hit = (picks["at_bats"] > 0) & (picks["resolved_hit"] == 1)
    got_out = (picks["at_bats"] > 0) & (picks["resolved_hit"] != 1)
    picks["actual_hit"] = pd.NA
    picks.loc[got_hit, "actual_hit"] = 1
    picks.loc[got_out, "actual_hit"] = 0
    return picks.drop(columns=["resolved_at_bats", "resolved_hit"])


def _select(pool: pd.DataFrame, date, *, rank_metric: str, min_probability: float,
            teams_playing_today: set[str] | None, model_version: str) -> pd.DataFrame:
    picks = predictions.select_picks(
        pool, date,
        top_n=config.BACKTEST_TOP_N,
        min_plate_appearances=config.BACKTEST_MIN_PLATE_APPEARANCES,
        metric="Game_Hit_Probability",
        rank_metric=rank_metric,
        min_probability=min_probability,
        teams_playing_today=teams_playing_today,
        model_version=model_version,
    )
    return picks


def run_backtest(raw_dir: str = "data/raw", season: int | None = None) -> pd.DataFrame:
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        raise SystemExit("No persisted Statcast history found.")

    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    dates = sorted(team_schedule["date"].unique())
    print(f"Backtesting {len(dates)} real historical dates: {dates[0]} to {dates[-1]}", flush=True)

    all_picks = []
    start = time.time()
    for i, current_date in enumerate(dates):
        day = dfs_backtest._compute_date_outputs(persisted, team_schedule, current_date)
        if day is None:
            continue

        wave = day["outputs"]["wave"]
        todays_schedule = day["todays_schedule"]
        matchup_probability = day["matchup_probability"]
        teams_playing_today = set(todays_schedule["team"])

        # --- Candidate 0: legacy Approach-only, no matchup signal at all
        # (pipeline.run's own real no-schedule fallback path). Selected
        # with min_probability=0.0 - see the note on candidate 2 below for
        # why the selection-time floor is deferred to evaluation time for
        # every candidate uniformly. ---
        c0_picks = _select(
            wave, current_date, rank_metric="Approach", min_probability=0.0,
            teams_playing_today=teams_playing_today, model_version="backtest-c0-legacy-approach",
        )
        c0_picks["strategy"] = "c0_legacy_approach"

        # --- Shared matchup pool for candidates 1 and 2 ---
        matched_pool = wave.merge(matchup_probability, on="key_mlbam", how="inner")
        matched_pool["Matchup_Approach"] = matched_pool["Approach"] * matched_pool["Matchup_Hit_Probability"]

        # --- Candidate 1: "WAVE vs PAVE" - best-faith reconstruction of
        # the user's own original manual method. PA qualifier only: no
        # lineup/recency qualifiers (dropped - column-gated no-ops, same
        # technique git_backtest.py relies on for pre-lineup-feature
        # wave.csv snapshots), no probability floor
        # (min_probability=0.0 - the gate check is `>= min_probability`,
        # so 0.0 always passes). ---
        c1_pool = matched_pool.drop(columns=[c for c in LINEUP_QUALIFIER_COLUMNS if c in matched_pool.columns])
        c1_picks = _select(
            c1_pool, current_date, rank_metric="Matchup_Approach", min_probability=0.0,
            teams_playing_today=teams_playing_today, model_version="backtest-c1-wave-vs-pave",
        )
        c1_picks["strategy"] = "c1_wave_vs_pave"

        # --- Candidate 2: today's real live selection logic (full
        # qualifier chain), minus the ML shortlist (leakage risk against
        # early-season dates - see module docstring).
        #
        # Selected with min_probability=0.0, NOT config.HITTER_MIN_PROBABILITY
        # (0.7) - confirmed against real data (both this recompute and the
        # live docs/data/wave.csv) that essentially no hitter clears 0.7 on
        # probability AND Game_Hit_Probability AND Matchup_Hit_Probability
        # jointly; in real live operation this gate is only ever cleared in
        # practice because Model_Hit_Probability REPLACES it with a top-10
        # shortlist (see select_picks's own docstring) - excluded here for
        # leak-safety, so re-applying the selection-time 0.7 floor without
        # the model that was validated alongside it would make this
        # candidate select ~0 picks on ~every date, not a fair stand-in for
        # "the current heuristic". The real, currently-live "recommended"
        # bar downstream of selection is DAILY_PICK_MIN_PROBABILITY (0.77,
        # see evaluation.graded_daily_picks) - applied uniformly to EVERY
        # candidate's top DAILY_PICK_MAX picks at evaluation time instead,
        # so all three are graded by the same real bar the dashboard
        # actually uses today. ---
        c2_picks = _select(
            matched_pool, current_date, rank_metric="Matchup_Approach", min_probability=0.0,
            teams_playing_today=teams_playing_today, model_version="backtest-c2-current-heuristic",
        )
        c2_picks["strategy"] = "c2_current_heuristic"

        all_picks.append(pd.concat([c0_picks, c1_picks, c2_picks], ignore_index=True))

        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(dates)}] {current_date} done ({elapsed:.0f}s elapsed)", flush=True)

    result = pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()
    if result.empty:
        return result

    result = _resolve_picks(result, persisted)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--output", default="data/backtest/pick_strategy_backtest.csv")
    args = parser.parse_args()

    result = run_backtest(args.raw_dir, args.season)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(result)} rows).")


if __name__ == "__main__":
    main()
