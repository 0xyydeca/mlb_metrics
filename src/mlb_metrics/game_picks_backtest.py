"""Reconstruct historical Automated Game Picks (see game_picks.py).

Two schedule information regimes exist and must not be conflated:

1. **as_of_snapshot** (production-equivalent, primary) - probable starters
   from append-only ``schedule_snapshots`` captured at or before the
   prediction timestamp (see ``mlb_metrics.schedule_snapshots``).
2. **actual_starter** (retrospective upper-bound diagnostic) - whoever
   actually started, reconstructed from persisted Statcast. Useful as a
   ceiling / sensitivity check, but **not** what the live pipeline saw.

A third mode, **missing_snapshot**, marks games with no as-of capture as
explicitly unavailable rather than silently filling actual starters.

Until enough forward schedule snapshots accumulate, report the
production-equivalent sample size honestly and never merge actual-starter
rows into an as-of evaluation set.

Game identity is Statcast's own ``game_pk`` (including doubleheaders).
Git-history replay of confidence.csv/pave.csv is unchanged in spirit.
"""

import subprocess

import pandas as pd

from mlb_metrics import (
    config, data, game_picks, game_predictions, git_backtest, pipeline, pitchers,
    schedule_snapshots,
)

CONFIDENCE_CSV_PATH = "docs/data/confidence.csv"
PAVE_CSV_PATH = "docs/data/pave.csv"

REQUIRED_CONFIDENCE_COLUMNS = {"team", "pyth_Strength", "pyth_Confidence", "suppression_resistance", "true_power"}
REQUIRED_PAVE_COLUMNS = {"key_mlbam", "PAVE_PLUS"}


def derive_historical_schedule_games(persisted_statcast: pd.DataFrame) -> pd.DataFrame:
    """Retrospective **actual-starter** schedule (diagnostic, not as-of).

    One row per historical game in the same column shape as
    ``schedule.normalize_schedule_games``, but ``*_probable_pitcher_*``
    columns hold whoever **actually started** (from Statcast pitcher
    roles), not the pregame announcement. Prefer
    ``schedule_snapshots.resolve_schedule_for_backtest(mode="as_of_snapshot")``
    for production-equivalent backtests.
    """
    data_with_game_id = persisted_statcast.rename(columns={"game_pk": "game_id"})
    roles = data.label_pitcher_roles(data_with_game_id)
    starters = roles[roles["is_starter"]][["game_id", "team", "pitcher"]]

    results = data.extract_game_results(data_with_game_id)
    # MLB games cannot end 0-0 (no ties, extra innings continue until
    # someone scores) - a reconstructed "final" score of 0-0 would only
    # ever indicate a data artifact, never a real result. Cheap, strictly
    # correct insurance regardless of cause.
    results = results[(results["home_score"] > 0) | (results["away_score"] > 0)]

    home_starters = starters.rename(columns={"team": "home_team", "pitcher": "home_probable_pitcher_key_mlbam"})
    away_starters = starters.rename(columns={"team": "away_team", "pitcher": "away_probable_pitcher_key_mlbam"})

    games = results.merge(home_starters, on=["game_id", "home_team"], how="left")
    games = games.merge(away_starters, on=["game_id", "away_team"], how="left")
    games["status"] = "Final"
    games = games.rename(columns={"game_id": "game_pk", "game_date": "date"})
    if "game_datetime" not in games.columns:
        games["game_datetime"] = pd.NA
    games["schedule_backtest_mode"] = schedule_snapshots.BACKTEST_MODE_ACTUAL
    games["schedule_source"] = schedule_snapshots.SOURCE_ACTUAL_STARTER_DIAGNOSTIC

    return games[
        [
            "game_pk", "date", "home_team", "away_team",
            "home_probable_pitcher_key_mlbam", "away_probable_pitcher_key_mlbam",
            "status", "home_score", "away_score", "game_datetime",
            "schedule_backtest_mode", "schedule_source",
        ]
    ]


def resolve_backtest_schedule_for_date(
    date,
    *,
    mode: str | None = None,
    actual_starter_games: pd.DataFrame | None = None,
    schedule_snapshots_df: pd.DataFrame | None = None,
    prediction_timestamp=None,
) -> tuple[pd.DataFrame, dict]:
    """Resolve today's games for a backtest date under the requested mode."""
    mode = mode or config.SCHEDULE_BACKTEST_MODE_DEFAULT
    pred_ts = prediction_timestamp
    if pred_ts is None and mode == schedule_snapshots.BACKTEST_MODE_AS_OF:
        pred_ts = schedule_snapshots.default_morning_prediction_timestamp(date)
    return schedule_snapshots.resolve_schedule_for_backtest(
        mode,
        prediction_timestamp=pred_ts,
        prediction_target_date=date,
        schedule_snapshots=schedule_snapshots_df,
        actual_starter_games=actual_starter_games,
    )


def reconstruct_historical_game_picks(
    repo_dir: str = ".",
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 40,
    model_version: str = game_predictions.LEGACY_MODEL_VERSION,
    schedule_backtest_mode: str | None = None,
    schedule_snapshots_path: str | None = None,
) -> pd.DataFrame:
    """Replay the last `days` daily commits of confidence.csv/pave.csv
    through game_picks.compute_game_win_probabilities +
    game_predictions.select_game_picks.

    ``schedule_backtest_mode`` defaults to production-equivalent
    ``as_of_snapshot``. Pass ``actual_starter`` explicitly for the
    retrospective Statcast-starter diagnostic. Modes are never merged.

    `model_version` defaults to game_predictions.LEGACY_MODEL_VERSION, not
    config.GAME_PICK_MODEL_VERSION - same reasoning as
    git_backtest.reconstruct_historical_picks: this reconstructs what old
    logic *would have* picked using old-era confidence.csv/pave.csv
    snapshots, so tagging it as the current live version would misrepresent
    it once both kinds of rows coexist in the same game_predictions.csv."""
    season = season or config.SEASON_START.year
    mode = schedule_backtest_mode or config.SCHEDULE_BACKTEST_MODE_DEFAULT

    commits = git_backtest.list_wave_csv_commits(repo_dir, path=CONFIDENCE_CSV_PATH)
    if days:
        commits = commits.sort_values("date").tail(days)

    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return pd.DataFrame(columns=game_predictions.GAME_PREDICTION_COLUMNS)
    actual_games = derive_historical_schedule_games(persisted)
    snaps = schedule_snapshots.load_schedule_snapshots(schedule_snapshots_path)

    all_picks = []
    for _, row in commits.iterrows():
        date = row["date"]
        try:
            confidence = git_backtest.read_csv_at_commit(row["commit"], CONFIDENCE_CSV_PATH, repo_dir)
            pave = git_backtest.read_csv_at_commit(row["commit"], PAVE_CSV_PATH, repo_dir)
        except subprocess.CalledProcessError:
            continue
        if confidence.empty or not REQUIRED_CONFIDENCE_COLUMNS.issubset(confidence.columns):
            continue
        if not REQUIRED_PAVE_COLUMNS.issubset(pave.columns):
            continue

        todays_games, sched_meta = resolve_backtest_schedule_for_date(
            date,
            mode=mode,
            actual_starter_games=actual_games,
            schedule_snapshots_df=snaps,
        )
        schedule_snapshots.assert_no_actual_starter_substitution(sched_meta)
        if todays_games.empty:
            continue

        win_probabilities = game_picks.compute_game_win_probabilities(confidence, pave, todays_games)
        picks = game_predictions.select_game_picks(win_probabilities, date, model_version=model_version)
        if picks.empty:
            continue

        results = actual_games.loc[
            actual_games["date"] == date,
            ["game_pk", "home_team", "away_team", "home_score", "away_score"],
        ]
        picks = picks.merge(results, on=["game_pk", "home_team", "away_team"], how="left")
        picks["game_played"] = picks["home_score"].notna().astype(int)
        picks["actual_winner"] = picks["home_team"].where(
            picks["home_score"] > picks["away_score"], picks["away_team"]
        )
        picks["schedule_backtest_mode"] = mode
        keep = [c for c in game_predictions.GAME_PREDICTION_COLUMNS if c in picks.columns]
        extra = [c for c in ("schedule_backtest_mode",) if c in picks.columns]
        all_picks.append(picks[keep + extra])

    if not all_picks:
        return pd.DataFrame(columns=game_predictions.GAME_PREDICTION_COLUMNS)
    return pd.concat(all_picks, ignore_index=True)


# All EXPLORATORY bullpen-fatigue candidate columns - NOT part of
# game_picks.GAME_PICK_FEATURE_COLUMNS - see
# scripts/train_game_pick_model.py's significance report for whether any
# clear a real bar before going anywhere near the live model.
# home_bullpen_recent_outs/away_bullpen_recent_outs (2026-08-24, "what
# about bullpen rest/readiness") tested a single fixed 2-day window and
# found no signal - real follow-up (2026-08-25, "I want to see if other
# applications of bullpen fatigue are significant... I don't care if
# they're cheap"): a sweep of additional window lengths
# (config.BULLPEN_FATIGUE_CANDIDATE_WINDOWS) plus two genuinely different
# hypotheses - pitchers.compute_bullpen_distinct_relievers (workload
# BREADTH: how many different arms got used, not how many total outs)
# and pitchers.compute_bullpen_back_to_back_relievers (a sharper "which
# SPECIFIC arms are on zero rest" signal, distinct from a team-wide
# workload total).
BULLPEN_FATIGUE_WINDOW_COLUMN_PAIRS = [
    (f"home_bullpen_recent_outs_{d}d", f"away_bullpen_recent_outs_{d}d")
    for d in config.BULLPEN_FATIGUE_CANDIDATE_WINDOWS
]
BULLPEN_FATIGUE_CANDIDATE_COLUMNS = (
    ["home_bullpen_recent_outs", "away_bullpen_recent_outs"]
    + [col for pair in BULLPEN_FATIGUE_WINDOW_COLUMN_PAIRS for col in pair]
    + ["home_bullpen_distinct_relievers", "away_bullpen_distinct_relievers"]
    + ["home_bullpen_back_to_back_relievers", "away_bullpen_back_to_back_relievers"]
)

GAME_PICK_LOG_COLUMNS = (
    ["game_pk", "date", "home_team", "away_team"]
    + game_picks.GAME_PICK_FEATURE_COLUMNS
    + BULLPEN_FATIGUE_CANDIDATE_COLUMNS
    + ["home_win_probability", "Home_Won", "schedule_backtest_mode"]
)


def _merge_team_metric(rows: pd.DataFrame, metric: pd.DataFrame, value_column: str, home_col: str, away_col: str) -> pd.DataFrame:
    """Left-merges a [team, value_column] frame onto `rows` twice - once
    keyed by home_team (as `home_col`), once by away_team (as `away_col`).
    Shared by every bullpen-fatigue candidate merge below so the same
    two-sided join isn't duplicated per metric."""
    rows = rows.merge(
        metric.rename(columns={"team": "home_team", value_column: home_col}), on="home_team", how="left",
    )
    rows = rows.merge(
        metric.rename(columns={"team": "away_team", value_column: away_col}), on="away_team", how="left",
    )
    return rows


def assemble_game_pick_log(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 20,
    schedule_backtest_mode: str | None = None,
    schedule_snapshots_path: str | None = None,
) -> pd.DataFrame:
    """Training log for scripts/train_game_pick_model.py: one row per
    historical game with features + home_win_probability + Home_Won.

    Default ``schedule_backtest_mode`` is ``as_of_snapshot`` (production-
    equivalent). Pass ``actual_starter`` for the labeled retrospective
    diagnostic. Modes are never silently merged; insufficient as-of
    sample size is labeled honestly via ``out.attrs['schedule_sample_report']``.
    """
    season = season or config.SEASON_START.year
    mode = schedule_backtest_mode or config.SCHEDULE_BACKTEST_MODE_DEFAULT

    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return pd.DataFrame(columns=GAME_PICK_LOG_COLUMNS)
    actual_games = derive_historical_schedule_games(persisted)
    snaps = schedule_snapshots.load_schedule_snapshots(schedule_snapshots_path)

    dates = sorted(actual_games["date"].unique())
    if days:
        dates = dates[-days:]

    all_rows = []
    as_of_n = 0
    actual_n = 0
    missing_n = 0
    for date in dates:
        history = persisted[persisted["game_date"] < date]
        if history.empty:
            continue

        todays_games, sched_meta = resolve_backtest_schedule_for_date(
            date,
            mode=mode,
            actual_starter_games=actual_games,
            schedule_snapshots_df=snaps,
        )
        schedule_snapshots.assert_no_actual_starter_substitution(sched_meta)
        if mode == schedule_snapshots.BACKTEST_MODE_AS_OF:
            as_of_n += int(sched_meta.get("n_games", 0) or 0)
            missing_n += int(sched_meta.get("n_missing_snapshot", 0) or 0)
        elif mode == schedule_snapshots.BACKTEST_MODE_ACTUAL:
            actual_n += int(sched_meta.get("n_games", 0) or 0)
        if todays_games.empty:
            continue

        outputs = pipeline.compute_outputs(history)
        features = game_picks.build_game_features(outputs["confidence"], outputs["pave"], todays_games)
        win_probabilities = game_picks.compute_game_win_probabilities(
            outputs["confidence"], outputs["pave"], todays_games
        )
        rows = features.merge(
            win_probabilities[["game_pk", "home_win_probability"]], on="game_pk", how="left"
        )

        # Exploratory candidate (see GAME_PICK_LOG_COLUMNS's own comment) -
        # game_pk is Statcast's own real game id here (this module's own
        # convention, not data.assign_game_ids - see module docstring).
        # `events` is required by build_pitcher_events_with_role/
        # completed_events - always present on real persisted Statcast, but
        # missing from this module's own minimal schedule/score-only
        # synthetic fixtures (derive_historical_schedule_games never needed
        # it) - degrades to null rather than a KeyError, same "missing
        # optional input becomes an honest null" precedent used elsewhere.
        if "events" in history.columns:
            data_with_game_id = history.rename(columns={"game_pk": "game_id"})
            roles = data.label_pitcher_roles(data_with_game_id)
            pdf_with_role = pipeline.build_pitcher_events_with_role(data_with_game_id, roles)

            bullpen_workload = pitchers.compute_bullpen_recent_workload(pdf_with_role)
            rows = _merge_team_metric(
                rows, bullpen_workload, "Bullpen_Recent_Outs", "home_bullpen_recent_outs", "away_bullpen_recent_outs"
            )

            for window, (home_col, away_col) in zip(
                config.BULLPEN_FATIGUE_CANDIDATE_WINDOWS, BULLPEN_FATIGUE_WINDOW_COLUMN_PAIRS
            ):
                workload_at_window = pitchers.compute_bullpen_recent_workload(pdf_with_role, recent_days=window)
                rows = _merge_team_metric(rows, workload_at_window, "Bullpen_Recent_Outs", home_col, away_col)

            distinct_relievers = pitchers.compute_bullpen_distinct_relievers(pdf_with_role)
            rows = _merge_team_metric(
                rows, distinct_relievers, "Bullpen_Distinct_Relievers",
                "home_bullpen_distinct_relievers", "away_bullpen_distinct_relievers",
            )

            back_to_back = pitchers.compute_bullpen_back_to_back_relievers(pdf_with_role)
            rows = _merge_team_metric(
                rows, back_to_back, "Bullpen_Back_To_Back_Relievers",
                "home_bullpen_back_to_back_relievers", "away_bullpen_back_to_back_relievers",
            )
        else:
            for column in BULLPEN_FATIGUE_CANDIDATE_COLUMNS:
                rows[column] = pd.NA

        results = actual_games.loc[
            actual_games["date"] == date,
            ["game_pk", "home_score", "away_score"],
        ]
        rows = rows.merge(results, on="game_pk", how="left")
        # Missing finals are no_game / unresolved — never coerce to Home_Won=0
        # (which would code an away win). Leave NA so trainers dropna.
        played = rows["home_score"].notna() & rows["away_score"].notna()
        rows["Home_Won"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
        rows.loc[played, "Home_Won"] = (
            rows.loc[played, "home_score"] > rows.loc[played, "away_score"]
        ).astype("Int64")
        rows["schedule_backtest_mode"] = mode

        all_rows.append(rows[GAME_PICK_LOG_COLUMNS])

    sample_report = schedule_snapshots.production_equivalent_sample_report(
        as_of_n if mode == schedule_snapshots.BACKTEST_MODE_AS_OF else 0,
        actual_starter_n_games=actual_n if mode == schedule_snapshots.BACKTEST_MODE_ACTUAL else 0,
        missing_snapshot_n_games=missing_n,
    )
    if mode == schedule_snapshots.BACKTEST_MODE_AS_OF and not sample_report["production_equivalent_sample_sufficient"]:
        print(
            f"WARNING: as-of schedule sample insufficient for production-equivalent "
            f"evaluation ({sample_report['as_of_snapshot_n_games']} < "
            f"{sample_report['production_equivalent_min_games']}). "
            f"Do not merge actual-starter diagnostic results. "
            f"label={sample_report['label']}"
        )

    if not all_rows:
        empty = pd.DataFrame(columns=GAME_PICK_LOG_COLUMNS)
        empty.attrs["schedule_sample_report"] = sample_report
        return empty
    out = pd.concat(all_rows, ignore_index=True)
    out.attrs["schedule_sample_report"] = sample_report
    return out


def evaluate_schedule_backtest_modes_separately(
    as_of_log: pd.DataFrame | None,
    actual_starter_log: pd.DataFrame | None,
) -> dict:
    """Score as-of and actual-starter logs separately; never merge rows."""
    def _summarize(frame: pd.DataFrame | None, mode: str) -> dict:
        if frame is None or frame.empty:
            return {"mode": mode, "n_games": 0, "brier_score": float("nan")}
        if "schedule_backtest_mode" in frame.columns:
            unexpected = set(frame["schedule_backtest_mode"].dropna().unique()) - {mode}
            if unexpected:
                raise AssertionError(
                    f"evaluate_schedule_backtest_modes_separately received mixed modes "
                    f"in {mode} frame: {sorted(unexpected)}"
                )
        y = frame.get("Home_Won")
        p = frame.get("home_win_probability")
        if y is None or p is None:
            return {"mode": mode, "n_games": int(len(frame)), "brier_score": float("nan")}
        mask = y.notna() & p.notna()
        if not mask.any():
            return {"mode": mode, "n_games": int(len(frame)), "brier_score": float("nan")}
        brier = float(((p[mask] - y[mask]) ** 2).mean())
        return {"mode": mode, "n_games": int(mask.sum()), "brier_score": brier}

    as_of_summary = _summarize(as_of_log, schedule_snapshots.BACKTEST_MODE_AS_OF)
    actual_summary = _summarize(actual_starter_log, schedule_snapshots.BACKTEST_MODE_ACTUAL)
    return {
        "as_of_snapshot": as_of_summary,
        "actual_starter": actual_summary,
        "modes_merged": False,
        "sample_report": schedule_snapshots.production_equivalent_sample_report(
            as_of_summary["n_games"],
            actual_starter_n_games=actual_summary["n_games"],
        ),
    }


def reconstruct_historical_game_picks_from_persisted(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 20,
    model_version: str = config.GAME_PICK_MODEL_VERSION,
    schedule_backtest_mode: str | None = None,
    schedule_snapshots_path: str | None = None,
) -> pd.DataFrame:
    """Like reconstruct_historical_game_picks, but recomputes confidence.csv/
    pave.csv fresh from persisted Statcast (pipeline.compute_outputs) for
    each replayed date, instead of replaying old git-committed snapshots.

    This answers a different question than the git-history version. Old
    commits' confidence.csv/pave.csv were written by whatever code was live
    on that day, so a column added since (e.g. Power_A_PLUS) simply isn't
    there and gets skipped over - that's the right tool for "how did the
    model as of some past commit perform." This function instead reruns
    pipeline.compute_outputs - the exact function pipeline.run() calls
    every day - against each date's as-of-that-date slice of persisted
    Statcast, so every replayed date reflects whatever signals are live in
    the code RIGHT NOW. That's the right tool for "how does the model I'd
    ship today perform," which is what's needed to evaluate a change before
    waiting for it to accumulate its own live picks (append-only logs mean
    today's already-logged picks can't be rewritten retroactively - see
    predictions.py/game_predictions.py module docstrings).

    Schedule mode defaults to ``as_of_snapshot``; pass ``actual_starter``
    for the labeled diagnostic.

    Recomputes the full season-to-date pipeline once per replayed date, so
    this is meaningfully slower than the git-history version - keep `days`
    modest for interactive use.

    `model_version` defaults to config.GAME_PICK_MODEL_VERSION (not
    LEGACY_MODEL_VERSION) because every replayed date uses today's actual
    live logic end to end - there's no "old logic" here to distinguish from."""
    season = season or config.SEASON_START.year
    mode = schedule_backtest_mode or config.SCHEDULE_BACKTEST_MODE_DEFAULT

    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return pd.DataFrame(columns=game_predictions.GAME_PREDICTION_COLUMNS)
    actual_games = derive_historical_schedule_games(persisted)
    snaps = schedule_snapshots.load_schedule_snapshots(schedule_snapshots_path)

    dates = sorted(actual_games["date"].unique())
    if days:
        dates = dates[-days:]

    all_picks = []
    for date in dates:
        history = persisted[persisted["game_date"] < date]
        if history.empty:
            continue

        todays_games, sched_meta = resolve_backtest_schedule_for_date(
            date,
            mode=mode,
            actual_starter_games=actual_games,
            schedule_snapshots_df=snaps,
        )
        schedule_snapshots.assert_no_actual_starter_substitution(sched_meta)
        if todays_games.empty:
            continue

        outputs = pipeline.compute_outputs(history)
        win_probabilities = game_picks.compute_game_win_probabilities(
            outputs["confidence"], outputs["pave"], todays_games
        )
        # This function's whole purpose is "what would today's live code
        # produce" (see its own docstring) - pipeline.run() applies this
        # same rescaling, so skipping it here would misrepresent what's
        # actually live.
        win_probabilities = game_picks.apply_calibration(win_probabilities)
        picks = game_predictions.select_game_picks(win_probabilities, date, model_version=model_version)
        if picks.empty:
            continue

        results = actual_games.loc[
            actual_games["date"] == date,
            ["game_pk", "home_team", "away_team", "home_score", "away_score"],
        ]
        picks = picks.merge(results, on=["game_pk", "home_team", "away_team"], how="left")
        picks["game_played"] = picks["home_score"].notna().astype(int)
        picks["actual_winner"] = picks["home_team"].where(
            picks["home_score"] > picks["away_score"], picks["away_team"]
        )
        picks["schedule_backtest_mode"] = mode
        keep = [c for c in game_predictions.GAME_PREDICTION_COLUMNS if c in picks.columns]
        extra = [c for c in ("schedule_backtest_mode",) if c in picks.columns]
        all_picks.append(picks[keep + extra])

    if not all_picks:
        return pd.DataFrame(columns=game_predictions.GAME_PREDICTION_COLUMNS)
    return pd.concat(all_picks, ignore_index=True)
