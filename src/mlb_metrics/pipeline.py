"""End-to-end pipeline: fetch Statcast -> persist raw -> compute hitter,
pitcher, and team metrics -> write docs/data/*.csv.

This replaces the original monolithic scripts/wave.py; that file is now a
thin entrypoint that calls `main()` here.

`run()` takes an explicit `as_of_date` and only ever uses games strictly
before it, rather than implicitly relying on the pipeline being run early
enough in the day that Statcast doesn't have today's games yet. That makes
the same-day leakage cutoff explicit and testable, and - together with raw
data persistence - lets this function be re-run against any past date for
backtesting (Phase B).
"""

import argparse
import datetime
import os

import pandas as pd

from mlb_metrics import (
    config, data, dfs_ml, evaluation, game_evaluation, game_picks, game_predictions,
    hitters, lineup, market_odds, matchup, ml_models, pitchers, predictions, schedule, teams,
)


def build_pitch_events(df: pd.DataFrame) -> pd.DataFrame:
    """Completed at-bat events with batter/p_throws, used by WAVE/WHOPS/WTB.
    Carries bat_score/post_bat_score too (helpers.estimate_rbi, via
    hitters.compute_extended_dk_rates), and the real batted-ball-quality
    columns (type, launch_speed, estimated_ba_using_speedangle,
    estimated_woba_using_speedangle, launch_speed_angle) used by
    hitters.compute_quality_of_contact - all of these already exist on
    every raw Statcast row (data.completed_events's own output is already
    one row per completed PA, and for a ball-in-play PA that row IS the
    real batted-ball-quality row), so no extra fetch/join is needed to add
    any of them here. The quality-of-contact columns are simply null on a
    non-batted-ball PA (a strikeout/walk/HBP never had a batted ball) -
    helpers.is_batted_ball is what filters those out downstream.

    Also carries `pitch_type` (the PA-ending pitch's own Statcast code),
    used by hitters.compute_pitch_family_rates - same "already on every
    real row, no extra fetch" reasoning as the quality-of-contact columns.

    Also carries `inning_topbot` ("Top"/"Bot"), used by
    hitters.compute_home_road_split to tell whether the batter's team was
    home ("Bot" - bottom of the inning is the home team's ups) or away
    ("Top") for that specific PA - real wave-logic follow-up (2026-08-24):
    "for each of our features, they should be taken with wave logic
    (someone... might randomly struggle versus lefties or at home)" - the
    same recency-windowed-blend treatment WAVE_L/WAVE_R already gives the
    platoon split, generalized to home/road via _blend_windows' existing
    `column` parameter (already proven for compute_pitch_family_rates).

    A `df` missing one or more of these optional columns entirely (never
    happens with a real pybaseball.statcast() pull - see
    data.fetch_statcast_range's docstring - but does happen with an older/
    narrower synthetic fixture, e.g. this project's own pre-existing test
    fixtures built before this feature existed) degrades those columns to
    null rather than raising - the same "missing optional input becomes an
    honest null, not a crash" precedent Bullpen_PAVE/Park_Factor/etc.
    already establish elsewhere in this pipeline."""
    quality_columns = [
        "type", "launch_speed", "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle", "launch_speed_angle", "pitch_type",
        "inning_topbot",
    ]
    missing = [c for c in quality_columns if c not in df.columns]
    if missing:
        df = df.assign(**{c: pd.NA for c in missing})
    return data.completed_events(
        df, ["game_date", "batter", "events", "p_throws", "bat_score", "post_bat_score"] + quality_columns
    )


def build_all_pitch_events(df: pd.DataFrame) -> pd.DataFrame:
    """Every real pitch thrown (not filtered down to one row per completed
    PA) - needed for pitchers.compute_pitch_arsenal's per-pitch usage-mix
    windowing and hitters.compute_plate_discipline's per-pitch swing/whiff/
    chase windowing. Every other pitcher/hitter metric in this pipeline
    only needs the PA-ENDING pitch (data.completed_events); both of these
    signals are fundamentally about every pitch thrown/seen, not just the
    ones that happened to end a plate appearance - a pitcher's real
    fastball/breaking/offspeed usage share (or a batter's real swing/whiff/
    chase rate) would be badly distorted by only counting PA-ending
    pitches (a putaway pitch is disproportionately a breaking/offspeed
    pitch and disproportionately a swing, not representative of the full
    mix either consumer needs).

    No null-filtering here - `batter`/`description` are never null on a
    real row (confirmed against the actual persisted data), and
    `pitch_type`/`zone` (the two columns that DO have real, ~0.4% nulls -
    mostly pitchouts/Statcast's own "couldn't classify" rows) are each
    filtered by their own consumer (helpers.pitch_type_family/
    is_out_of_zone), not pre-filtered here - dropping on `pitch_type`
    unconditionally would wrongly exclude real plate-discipline-relevant
    pitches that have no pitch_type but a perfectly real description/zone."""
    columns = ["game_date", "batter", "pitcher", "pitch_type", "description", "zone"]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        df = df.assign(**{c: pd.NA for c in missing})
    return df[columns]


def build_pitcher_events(df: pd.DataFrame) -> pd.DataFrame:
    """Completed at-bat events keyed by pitcher, used by PAVE. Carries
    p_throws (constant per pitcher) so pitchers.compute_pitcher_throws can
    expose each pitcher's own throwing hand for matchup.py's platoon logic."""
    return data.completed_events(df, ["game_date", "pitcher", "events", "p_throws"])


def build_pitcher_events_with_role(data_with_game_id: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Completed at-bat events keyed by pitcher, with the pitching `team` and
    `is_starter` for that appearance attached, used by compute_bullpen_pave."""
    completed = data.completed_events(
        data_with_game_id, ["game_date", "pitcher", "events", "game_id"]
    )
    return completed.merge(
        roles[["game_id", "pitcher", "team", "is_starter"]], on=["game_id", "pitcher"], how="left"
    )


def compute_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all three metric families against an already as-of-date-filtered
    Statcast dataframe. Returns {"wave": ..., "pave": ..., "confidence": ...}."""
    names = data.get_name_register()[["key_mlbam", "name_first", "name_last"]]
    latest_batter_team = data.latest_team_for_batters(df)
    latest_pitcher_team = data.latest_team_for_pitchers(df)
    data_with_game_id = data.assign_game_ids(df)
    roles = data.label_pitcher_roles(data_with_game_id)

    dt = build_pitch_events(df)
    pdf = build_pitcher_events(df)
    pdf_with_role = build_pitcher_events_with_role(data_with_game_id, roles)
    bullpen_pave = pitchers.compute_bullpen_pave(pdf_with_role)

    all_pitches = build_all_pitch_events(df)
    pitch_arsenal = pitchers.compute_pitch_arsenal(all_pitches)

    batting_order = data.assign_batting_order(data_with_game_id)
    lineup_consistency = lineup.compute_lineup_consistency(batting_order, latest_batter_team)

    return {
        "wave": hitters.assemble_hitters(
            dt, data_with_game_id, names, latest_batter_team, lineup_consistency, all_pitches
        ),
        "pave": pitchers.assemble_pitchers(pdf, names, latest_pitcher_team, pitch_arsenal),
        "confidence": teams.assemble_team_metrics(data_with_game_id, bullpen_pave),
    }


def write_beat_the_streak_export(predictions_log_path: str, output_dir: str) -> None:
    """Read the full predictions log and (re)write the CSVs the dashboard's
    Beat the Streak section reads: each day's top DAILY_PICK_MAX candidates,
    ALWAYS shown (see evaluation.graded_daily_picks) with hit/miss/no_game/
    pending status and a "recommended"/"speculative" grade (whether that
    candidate's own real combined probability clears
    DAILY_PICK_MIN_PROBABILITY) - a weak-slate day shows its real best
    options graded "speculative" rather than going blank; an all-time
    longest_streak/current_streak summary following Beat the Streak's actual
    rules, counting only "recommended"-grade picks (see
    evaluation.streak_progression), and a small by-version summary
    (all_time plus config.HITTER_MODEL_VERSION) so a selection-logic
    change's real effect on accuracy is visible without waiting for
    pre-change history to stop dominating the all-time numbers. No-op if
    nothing's logged yet."""
    if not os.path.exists(predictions_log_path):
        return
    log = pd.read_csv(predictions_log_path, parse_dates=["date"])
    picks, summary = evaluation.build_beat_the_streak_export(
        log, max_picks=config.DAILY_PICK_MAX, min_probability=config.DAILY_PICK_MIN_PROBABILITY
    )
    _, current_version_summary = evaluation.build_beat_the_streak_export(
        log, max_picks=config.DAILY_PICK_MAX, min_probability=config.DAILY_PICK_MIN_PROBABILITY,
        model_version=config.HITTER_MODEL_VERSION,
    )
    by_version_summary = pd.concat([summary, current_version_summary], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    picks.to_csv(os.path.join(output_dir, "beat_the_streak_picks.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "beat_the_streak_summary.csv"), index=False)
    by_version_summary.to_csv(os.path.join(output_dir, "beat_the_streak_summary_by_version.csv"), index=False)


def write_probable_pitchers_export(schedule_df: pd.DataFrame, pave: pd.DataFrame, output_dir: str) -> None:
    """(Re)write the small CSV the dashboard's Probable Pitchers list reads:
    one row per team playing today with their probable starter's PAVE/
    PAVE_PLUS/Power_A_PLUS (see schedule.build_probable_pitchers_table).
    Callers should only invoke this when `schedule_df` is non-empty - see
    run()'s resilience handling for a failed/empty schedule fetch."""
    table = schedule.build_probable_pitchers_table(schedule_df, pave)
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(os.path.join(output_dir, "probable_pitchers.csv"), index=False)


def write_game_picks_export(game_predictions_log_path: str, output_dir: str) -> None:
    """Read the full game-predictions log and (re)write the CSVs the
    dashboard's Automated Game Picks section reads: each picked game with a
    win/loss/not_played/pending status, an all-time accuracy/streak summary,
    and a small by-version summary (all_time plus config.GAME_PICK_MODEL_VERSION -
    see game_evaluation.build_game_picks_export), same reasoning as
    write_beat_the_streak_export's own by-version split. No-op if nothing's
    logged yet."""
    if not os.path.exists(game_predictions_log_path):
        return
    log = pd.read_csv(game_predictions_log_path, parse_dates=["date"])
    picks, summary = game_evaluation.build_game_picks_export(log)
    _, current_version_summary = game_evaluation.build_game_picks_export(
        log, model_version=config.GAME_PICK_MODEL_VERSION
    )
    by_version_summary = pd.concat([summary, current_version_summary], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    picks.to_csv(os.path.join(output_dir, "game_picks_picks.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "game_picks_summary.csv"), index=False)
    by_version_summary.to_csv(os.path.join(output_dir, "game_picks_summary_by_version.csv"), index=False)


def run(
    as_of_date: datetime.date,
    raw_dir: str = "data/raw",
    output_dir: str = "docs/data",
    predictions_dir: str = "data/predictions",
    persist_raw: bool = True,
    log_predictions: bool = True,
) -> dict[str, pd.DataFrame]:
    fetch_start = config.SEASON_START
    fetch_end = min(config.SEASON_END, as_of_date - datetime.timedelta(days=1))
    if fetch_end < fetch_start:
        raise ValueError(f"as_of_date {as_of_date} is before the season start {fetch_start}")

    fresh = data.fetch_statcast_range(fetch_start, fetch_end)
    df = data.persist_raw_statcast(fresh, raw_dir, season=fetch_start.year) if persist_raw else fresh

    # Belt-and-suspenders cutoff: even if persisted raw data (or a future
    # fetch_end miscalculation) contains rows on/after as_of_date, never let
    # them reach the metrics.
    df = df[df["game_date"] < pd.Timestamp(as_of_date)].copy()

    outputs = compute_outputs(df)

    os.makedirs(output_dir, exist_ok=True)
    outputs["wave"].to_csv(os.path.join(output_dir, "wave.csv"), index=False)
    outputs["pave"].to_csv(os.path.join(output_dir, "pave.csv"), index=False)
    outputs["confidence"].to_csv(os.path.join(output_dir, "confidence.csv"), index=False)

    if log_predictions:
        predictions_log_path = os.path.join(predictions_dir, "predictions.csv")
        game_predictions_log_path = os.path.join(predictions_dir, "game_predictions.csv")

        # `df` already covers every completed game strictly before as_of_date,
        # i.e. exactly what's needed to resolve any pick logged on an earlier
        # run whose target date has since happened.
        resolve_columns = ["game_date", "batter", "events"]
        if "game_pk" in df.columns:
            resolve_columns = ["game_date", "game_pk", "batter", "events"]
        completed = data.completed_events(df, resolve_columns)
        predictions.resolve_predictions(predictions_log_path, completed)
        # Resolving game picks needs final scores, not Statcast (see
        # schedule.fetch_game_results) - this call is internally resilient
        # per-date (see game_predictions.resolve_game_predictions), so it
        # doesn't need its own try/except here.
        game_predictions.resolve_game_predictions(game_predictions_log_path, schedule.fetch_game_results, as_of_date)

        # A new external dependency (statsapi) must not be able to break the
        # whole daily update - on failure, fall back to no schedule
        # awareness at all for this run rather than skipping Game_Hit_Probability too.
        schedule_df = None
        try:
            schedule_df = schedule.fetch_probable_pitchers(as_of_date)
        except Exception as exc:
            print(f"WARNING: failed to fetch today's schedule/probable pitchers ({exc}); "
                  f"skipping Matchup_Hit_Probability and the teams-playing-today qualifier for this run.")

        # Same resilience for the game-per-row shape Automated Game Picks
        # needs (see schedule.normalize_schedule_games) - a separate call
        # since it deliberately doesn't dedupe doubleheaders the way
        # schedule_df above does, so it can't be derived from schedule_df.
        schedule_games_df = None
        try:
            schedule_games_df = schedule.fetch_todays_games(as_of_date)
        except Exception as exc:
            print(f"WARNING: failed to fetch today's game schedule ({exc}); "
                  f"skipping Automated Game Picks for this run.")

        # None (fetch failed) means "unknown, don't filter"; an empty set
        # (fetch succeeded, zero games today) correctly excludes every pick.
        # Hitter matchup / features / picks use the all-games schedule
        # (both DH halves); probable-pitchers export keeps the first-game-
        # only shape for the dashboard list.
        hitter_schedule_df = schedule_df
        if schedule_df is not None:
            try:
                hitter_schedule_df = schedule.fetch_hitter_schedule(as_of_date)
            except Exception as exc:
                print(
                    f"WARNING: failed to fetch all-games hitter schedule ({exc}); "
                    f"falling back to first-game-only probable-pitchers schedule for matchup."
                )
                hitter_schedule_df = schedule_df

        teams_playing_today = set(hitter_schedule_df["team"]) if hitter_schedule_df is not None else None

        # Two-tier rank_metric, best available first: Matchup_Approach
        # (Approach * Matchup_Hit_Probability, today's schedule/matchup
        # available) > Approach (own-form-only, no schedule at all) - falls
        # back on a failed schedule fetch, same resilience pattern as
        # schedule_df itself. Separately, independent of rank_metric: when
        # the validated logistic regression (config.HITTER_HIT_PROBABILITY_MODEL_PATH,
        # treats matchup ingredients as independent learned features instead
        # of one hand-picked multiplier, see dfs_ml.py's module docstring)
        # loads and predicts for today's schedule, its Model_Hit_Probability
        # column gets merged onto pick_pool - select_picks then uses that
        # column's mere PRESENCE (not rank_metric) to narrow the qualified
        # pool to a broad model-approved shortlist BEFORE rank_metric picks
        # the final order among survivors (see select_picks's own docstring
        # for why the model shortlists rather than ranks directly - a
        # deliberate reversal of an earlier design, HITTER_MODEL_VERSION's
        # v3, that let it rank the whole pool). A model artifact that hasn't
        # been trained/fails to load (dfs_ml.predict_hitter_hit_probability's
        # own "return empty, never raise" contract) just means the shortlist
        # step never engages that day - rank_metric alone decides, same as
        # before this model existed at all. predicted_probability/metric
        # logged still reflect Game_Hit_Probability regardless of any of
        # this, so DAILY_PICK_MIN_PROBABILITY's calibration is unaffected.
        pick_pool = outputs["wave"]
        rank_metric = "Approach"
        hitter_model_status = None
        hitter_fallback_used = False
        hitter_fallback_reason = None
        if hitter_schedule_df is not None and not hitter_schedule_df.empty:
            matchup_probability = matchup.compute_matchup_hit_probability(
                outputs["wave"], outputs["pave"], outputs["confidence"], hitter_schedule_df
            )
            # Expand wave (one row per batter) by matchup's per-game rows
            # (key_mlbam + game_pk). Matchup already carries one row per
            # contest, so this is intentional duplication across a DH, not
            # a team-only cartesian fan-out against the schedule.
            pick_pool = outputs["wave"].merge(matchup_probability, on="key_mlbam", how="inner")
            pick_pool["Matchup_Approach"] = pick_pool["Approach"] * pick_pool["Matchup_Hit_Probability"]
            rank_metric = "Matchup_Approach"

            hitter_features = dfs_ml.build_hitter_features(
                outputs["wave"], outputs["pave"], outputs["confidence"], hitter_schedule_df, matchup_probability
            )
            hitter_model_status = ml_models.inspect_model_path(config.HITTER_HIT_PROBABILITY_MODEL_PATH)
            model_predictions = dfs_ml.predict_hitter_hit_probability(hitter_features)
            if not model_predictions.empty:
                # Guard on non-empty BEFORE merging: select_picks' shortlist
                # step is column-gated, so merging in an all-NaN
                # Model_Hit_Probability column on a day the model fails to
                # load would make every row sort last instead of correctly
                # skipping the shortlist step entirely.
                pick_pool = pick_pool.merge(
                    model_predictions, on=["key_mlbam", "game_pk"], how="left"
                )
                hitter_fallback_used = False
                hitter_fallback_reason = None
            else:
                # Model shortlist did not engage - record why so a load
                # failure is not indistinguishable from a normal heuristic day.
                hitter_fallback_used = True
                hitter_fallback_reason = (
                    hitter_model_status.get("fallback_reason")
                    if not hitter_model_status.get("loaded")
                    else "empty_predictions"
                )

            if schedule_df is not None and not schedule_df.empty:
                write_probable_pitchers_export(schedule_df, outputs["pave"], output_dir)

        game_hit_picks = predictions.select_picks(
            pick_pool,
            as_of_date,
            rank_metric=rank_metric,
            teams_playing_today=teams_playing_today,
            model_status=hitter_model_status,
            fallback_used=hitter_fallback_used,
            fallback_reason=hitter_fallback_reason,
        )
        predictions.append_predictions(game_hit_picks, predictions_log_path)

        write_beat_the_streak_export(predictions_log_path, output_dir)

        if schedule_games_df is not None and not schedule_games_df.empty:
            win_probabilities = game_picks.compute_game_win_probabilities(
                outputs["confidence"], outputs["pave"], schedule_games_df
            )
            # Quant-analytics follow-up "dig into calibration": rescales
            # the raw heuristic ratio through the saved recalibration, if
            # one has been trained and cleared its own real-holdout bar
            # (see game_picks.apply_calibration's own docstring) - a no-op
            # returning win_probabilities completely unchanged otherwise.
            calibration_status = ml_models.inspect_model_path(config.GAME_PICK_CALIBRATION_MODEL_PATH)
            win_probabilities = game_picks.apply_calibration(win_probabilities)
            if calibration_status.get("loaded"):
                game_fallback_used = False
                game_fallback_reason = None
                game_probability_source = "calibrated_home_win_probability"
            else:
                game_fallback_used = True
                game_fallback_reason = calibration_status.get("fallback_reason") or "missing_artifact"
                game_probability_source = "home_win_probability"
            # A real market-odds fetch failure must never suppress real
            # game-pick logging - deliberately a separate try/except from
            # schedule_games_df's own above, not shared with it. Quant-
            # analytics item #6, slice 2 (market_odds.py).
            try:
                market_probabilities = market_odds.fetch_market_home_win_probabilities(as_of_date)
            except Exception as exc:
                print(
                    f"WARNING: failed to fetch real ESPN market odds for {as_of_date} ({exc}); "
                    f"logging today's game picks without a market comparison."
                )
                market_probabilities = None
            todays_game_picks = game_predictions.select_game_picks(
                win_probabilities,
                as_of_date,
                market_probabilities=market_probabilities,
                confidence=outputs["confidence"],
                model_status=calibration_status,
                fallback_used=game_fallback_used,
                fallback_reason=game_fallback_reason,
                probability_source=game_probability_source,
            )
            game_predictions.append_game_predictions(todays_game_picks, game_predictions_log_path)

        write_game_picks_export(game_predictions_log_path, output_dir)

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Run the daily hitter/pitcher/team metrics pipeline."
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="YYYY-MM-DD; defaults to today. Only games strictly before this date are used.",
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="docs/data")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions")
    parser.add_argument(
        "--no-persist-raw",
        action="store_true",
        help="Skip saving the Statcast pull to --raw-dir (useful for local/backtest runs).",
    )
    parser.add_argument(
        "--no-log-predictions",
        action="store_true",
        help="Skip logging today's picks / resolving past ones (useful for local/backtest runs).",
    )
    args = parser.parse_args()

    as_of_date = (
        datetime.date.fromisoformat(args.as_of_date) if args.as_of_date else schedule.today_local()
    )
    run(
        as_of_date,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        predictions_dir=args.predictions_dir,
        persist_raw=not args.no_persist_raw,
        log_predictions=not args.no_log_predictions,
    )


if __name__ == "__main__":
    main()
