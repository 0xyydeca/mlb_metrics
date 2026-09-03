"""Follow-up to backtest_pick_strategies.py: two rounds of EXPLORATORY,
hypothesis-driven candidates, run after seeing that round 1's own results
showed (a) c1 (WAVE vs PAVE, no lineup/recency qualifiers) beat c0/c2 on
every cut, and (b) a direct swap analysis (players c1 picked that c2's
qualifiers excluded entirely hit 73.0%, n=115, vs. 66.2%, n=157, for the
ones that survived into c2's pool) - real evidence the qualifiers are
cutting good picks, not bad ones.

Unlike round 1, this is NOT a pre-registered blind comparison - it is
targeted diagnosis (which qualifier is responsible) plus a small,
declared sweep of alternative knobs. Read the results with that in mind:
more candidates tested against the same 144-date window raises real
multiple-comparison risk, so nothing here should be treated as
conclusive without the same first/second-half consistency check round 1
used, applied here too.

## Qualifier ablation (isolate WHICH qualifier drives the round-1 gap)

  c2a_no_lineup   - c2's full pool minus the lineup qualifiers
                    (avg_batting_order/start_rate) - keeps PA + recency +
                    teams-playing-today.
  c2b_no_recency  - c2's full pool minus the recency qualifier
                    (Last_Game_Date) - keeps PA + lineup + teams-playing.

If c2a's hit rate recovers to roughly c1's while c2b stays near c2's,
the lineup qualifier (not recency) is the one doing the damage - matches
the Javier Sanoja finding (excluded on avg_batting_order=7.25 despite a
100% start rate) generalizing across the whole season, not just one
player.

## Alternative ranking metrics (does Matchup_Approach's multiplicative
blend actually rank better than the alternatives?), all using c1's own
qualifier profile (PA-only) so this isolates the RANKING choice alone:

  c1_rank_matchup_only   - rank by Matchup_Hit_Probability alone (drop
                            the batter's own season-long Approach term
                            entirely - pure "today's matchup only").
  c1_rank_combined_mean  - rank by the row-wise mean of
                            Game_Hit_Probability/probability/
                            Matchup_Hit_Probability (the same
                            "combined_probability" evaluation.py already
                            uses for streak grading - an additive blend
                            instead of Matchup_Approach's multiplicative
                            one).

## PA-threshold sweep (is BACKTEST_MIN_PLATE_APPEARANCES=30 well-tuned?),
same PA-only qualifier profile and Matchup_Approach ranking as c1:

  c1_pa15  - min_plate_appearances=15 (probes whether a lower volume bar
             would have caught more real, currently-missed hot streaks -
             directly related to the real Javier Sanoja case, even though
             his actual blocker was the lineup qualifier, not PA).
  c1_pa50  - min_plate_appearances=50 (tests whether a STRICTER volume
             bar improves reliability instead).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, dfs_backtest, helpers, matchup, predictions

LINEUP_COLUMNS = ["avg_batting_order", "start_rate"]
RECENCY_COLUMNS = ["Last_Game_Date"]


def _resolve_picks(picks: pd.DataFrame, persisted: pd.DataFrame) -> pd.DataFrame:
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


def _select(pool: pd.DataFrame, date, *, rank_metric: str, min_plate_appearances: int,
            teams_playing_today: set[str] | None, model_version: str) -> pd.DataFrame:
    return predictions.select_picks(
        pool, date,
        top_n=config.BACKTEST_TOP_N,
        min_plate_appearances=min_plate_appearances,
        metric="Game_Hit_Probability",
        rank_metric=rank_metric,
        min_probability=0.0,
        teams_playing_today=teams_playing_today,
        model_version=model_version,
    )


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

        matched_pool = wave.merge(matchup_probability, on="key_mlbam", how="inner")
        matched_pool["Matchup_Approach"] = matched_pool["Approach"] * matched_pool["Matchup_Hit_Probability"]
        matched_pool["combined_probability"] = matched_pool[
            ["Game_Hit_Probability", "probability", "Matchup_Hit_Probability"]
        ].astype(float).mean(axis=1, skipna=True)

        pa_only_pool = matched_pool.drop(columns=[c for c in LINEUP_COLUMNS + RECENCY_COLUMNS if c in matched_pool.columns])

        day_picks = []

        # --- Qualifier ablation ---
        no_lineup_pool = matched_pool.drop(columns=[c for c in LINEUP_COLUMNS if c in matched_pool.columns])
        c2a = _select(no_lineup_pool, current_date, rank_metric="Matchup_Approach",
                      min_plate_appearances=config.BACKTEST_MIN_PLATE_APPEARANCES,
                      teams_playing_today=teams_playing_today, model_version="backtest-c2a-no-lineup")
        c2a["strategy"] = "c2a_no_lineup"
        day_picks.append(c2a)

        no_recency_pool = matched_pool.drop(columns=[c for c in RECENCY_COLUMNS if c in matched_pool.columns])
        c2b = _select(no_recency_pool, current_date, rank_metric="Matchup_Approach",
                      min_plate_appearances=config.BACKTEST_MIN_PLATE_APPEARANCES,
                      teams_playing_today=teams_playing_today, model_version="backtest-c2b-no-recency")
        c2b["strategy"] = "c2b_no_recency"
        day_picks.append(c2b)

        # --- Ranking metric alternatives (PA-only qualifier profile) ---
        c1_matchup_only = _select(pa_only_pool, current_date, rank_metric="Matchup_Hit_Probability",
                                   min_plate_appearances=config.BACKTEST_MIN_PLATE_APPEARANCES,
                                   teams_playing_today=teams_playing_today, model_version="backtest-c1-rank-matchup-only")
        c1_matchup_only["strategy"] = "c1_rank_matchup_only"
        day_picks.append(c1_matchup_only)

        c1_combined_mean = _select(pa_only_pool, current_date, rank_metric="combined_probability",
                                    min_plate_appearances=config.BACKTEST_MIN_PLATE_APPEARANCES,
                                    teams_playing_today=teams_playing_today, model_version="backtest-c1-rank-combined-mean")
        c1_combined_mean["strategy"] = "c1_rank_combined_mean"
        day_picks.append(c1_combined_mean)

        # --- PA threshold sweep (PA-only qualifier profile, Matchup_Approach ranking) ---
        c1_pa15 = _select(pa_only_pool, current_date, rank_metric="Matchup_Approach",
                           min_plate_appearances=15,
                           teams_playing_today=teams_playing_today, model_version="backtest-c1-pa15")
        c1_pa15["strategy"] = "c1_pa15"
        day_picks.append(c1_pa15)

        c1_pa50 = _select(pa_only_pool, current_date, rank_metric="Matchup_Approach",
                           min_plate_appearances=50,
                           teams_playing_today=teams_playing_today, model_version="backtest-c1-pa50")
        c1_pa50["strategy"] = "c1_pa50"
        day_picks.append(c1_pa50)

        all_picks.append(pd.concat(day_picks, ignore_index=True))

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
    parser.add_argument("--output", default="data/backtest/pick_strategy_backtest_v2.csv")
    args = parser.parse_args()

    result = run_backtest(args.raw_dir, args.season)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(result)} rows).")


if __name__ == "__main__":
    main()
