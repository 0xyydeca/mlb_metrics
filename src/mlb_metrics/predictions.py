"""Daily pick logging and outcome resolution - the core of Phase B.

Nothing in the original script ever checked whether WAVE/Game_Hit_Probability
actually predicted hits. This module maintains an append-only log of
(date, game_pk, player, predicted probability, realized outcome) that
evaluation.py scores: `select_picks` turns a computed hitters table into
that day's ranked, qualified picks; `append_predictions` logs them *before*
the game is played; `resolve_predictions` fills in whether the pick actually
got a hit once that game's outcome data is available.

Natural key for new rows: (date, game_pk, key_mlbam, metric). Legacy CSV
rows may have a null game_pk; resolution then falls back to the old
(date, key_mlbam) label so already-logged history stays readable.
"""

import os

import pandas as pd

from mlb_metrics import config, helpers

PREDICTION_COLUMNS = [
    "date", "game_pk", "key_mlbam", "name", "rank", "predicted_probability", "metric",
    "probability", "Matchup_Hit_Probability", "Model_Hit_Probability", "actual_hit", "at_bats", "model_version",
]

# Tag applied (via the migration guards in append_predictions/resolve_predictions)
# to any row logged before the model_version column existed, or reconstructed
# by a historical replay (see git_backtest.py) - distinguishes "we don't know
# which logic produced this" from a real current-version live pick.
LEGACY_MODEL_VERSION = "legacy"

# The set of probability-like signals select_picks jointly gates on (each
# must clear min_probability), whichever of them happen to be present on the
# `hitters` table passed in - see select_picks's docstring.
JOINT_PROBABILITY_GATE_COLUMNS = ["probability", "Game_Hit_Probability", "Matchup_Hit_Probability"]

# Dedup / identity key for the predictions log once game_pk exists.
PREDICTION_KEY_COLUMNS = ["date", "game_pk", "key_mlbam", "metric"]


def _ensure_game_pk_column(df: pd.DataFrame) -> pd.DataFrame:
    """Migrate a frame written before game_pk existed - null, not a guess."""
    if "game_pk" not in df.columns:
        df = df.copy()
        df["game_pk"] = pd.NA
    return df


def _diversify_second_pick(ranked: pd.DataFrame, rank_column: str, margin: float) -> pd.DataFrame:
    """Quant-analytics item #4, slice 2: `ranked` is already sorted best-
    to-worst by `rank_column`. If the #1 and #2 rows share a real
    `game_pk`, look further down `ranked` (in order) for the first row
    from a DIFFERENT game whose `rank_column` value is within `margin` of
    the original #2's own value - if one exists, move it up into the #2
    slot (the original #2 is demoted to right after it; everyone else
    keeps their relative order). `#1` is never touched. Returns `ranked`
    unchanged if there's nothing to diversify (fewer than 2 rows, #1/#2
    already in different games, either game_pk is null/unknown, or no
    close-enough different-game alternative exists further down the
    list - never sacrifices real expected value to force a diversification
    that isn't nearly free)."""
    if len(ranked) < 2:
        return ranked

    top_game = ranked.iloc[0]["game_pk"]
    second_game = ranked.iloc[1]["game_pk"]
    if pd.isna(top_game) or pd.isna(second_game) or top_game != second_game:
        return ranked

    second_score = ranked.iloc[1][rank_column]
    candidate_position = None
    for position in range(2, len(ranked)):
        row = ranked.iloc[position]
        if pd.notna(row["game_pk"]) and row["game_pk"] != top_game and row[rank_column] >= second_score - margin:
            candidate_position = position
            break
    if candidate_position is None:
        return ranked

    order = list(range(len(ranked)))
    order.remove(candidate_position)
    order.insert(1, candidate_position)
    return ranked.iloc[order].reset_index(drop=True)


def select_picks(
    hitters: pd.DataFrame,
    date,
    top_n: int = config.BACKTEST_TOP_N,
    min_plate_appearances: int = config.BACKTEST_MIN_PLATE_APPEARANCES,
    metric: str = "Game_Hit_Probability",
    rank_metric: str | None = None,
    min_probability: float = config.HITTER_MIN_PROBABILITY,
    model_shortlist_size: int = config.HITTER_MODEL_SHORTLIST_SIZE,
    max_avg_batting_order: float = config.LINEUP_TOP_HALF_MAX_SLOT,
    min_start_rate: float = config.LINEUP_MIN_START_RATE,
    max_days_since_last_game: int = config.HITTER_MAX_DAYS_SINCE_LAST_GAME,
    teams_playing_today: set[str] | None = None,
    model_version: str = config.HITTER_MODEL_VERSION,
    same_game_diversification_margin: float = config.SAME_GAME_DIVERSIFICATION_MARGIN,
) -> pd.DataFrame:
    """Rank a computed hitters table (the wave.csv-equivalent output of
    hitters.assemble_hitters, optionally already merged with per-game
    matchup / model columns) by `rank_metric` (defaults to `metric`) and
    return the top `top_n` qualified picks for `date`, in PREDICTION_COLUMNS
    shape with `actual_hit` left null (unresolved). `predicted_probability`
    and the logged `metric` name always come from `metric`, regardless of
    which column was used to rank - `rank_metric` only changes *which*
    qualified hitters get chosen, not what probability gets reported/scored.

    When `hitters` carries `game_pk` (live matchup path / hitter schedule),
    each returned row is keyed to that specific contest. New logged rows
    must have a non-null game_pk; git-history replays without schedule data
    still produce null game_pk and are tagged as legacy-readable.

    `max_avg_batting_order`/`min_start_rate` only take effect if `hitters`
    has avg_batting_order/start_rate columns (see hitters.assemble_hitters's
    optional `lineup_consistency` param) - absent columns mean a no-op, so
    old wave.csv snapshots in git history (which predate this feature) are
    unaffected. A null avg_batting_order (never started for their current
    team) fails the comparison and is correctly excluded, not treated as 0.

    `min_probability` requires EVERY column in JOINT_PROBABILITY_GATE_COLUMNS
    that's actually present on `hitters` to clear this bar (see
    config.HITTER_MIN_PROBABILITY) - a hitter can look good on one signal
    while being unreliable on another (see config.py for what each
    divergence means). On a normal day that's `probability` and
    `Game_Hit_Probability`; on a day `Matchup_Hit_Probability` has been
    merged in too (see pipeline.run), a good matchup is required just as
    much as the other two. Column-gated like the lineup qualifiers, so a
    missing column is simply skipped, not a required 0.

    If `Model_Hit_Probability` is present on `hitters` (see pipeline.run),
    it's used as a BROAD QUALITY FILTER, not the final ranker: the
    qualified pool is narrowed to its top `model_shortlist_size` (or fewer,
    never padded, if fewer survive the qualifiers above) by
    `Model_Hit_Probability`, and `rank_metric` then decides the final order
    among THAT shortlist - same as it would across the whole qualified pool
    on a day the model isn't available. This REPLACES the
    JOINT_PROBABILITY_GATE_COLUMNS gate entirely rather than being added as
    a fourth required column - it's a learned function OF `probability`/
    `Game_Hit_Probability`/`Matchup_Hit_Probability` (plus more raw
    ingredients, see dfs_ml.HITTER_FEATURE_COLUMNS), so requiring all four
    to independently clear the same bar would be circular, unlike the
    original three-column gate where each genuinely captures a different
    failure mode. A null `Model_Hit_Probability` sorts last (pandas'
    default) and is excluded exactly like a real low score, same "null
    loses" precedent as `avg_batting_order`/`Last_Game_Date` below.

    This is a deliberate reversal of an earlier design (v3,
    `HITTER_MODEL_VERSION`) that let the model rank the WHOLE pool
    directly, gated on a probability threshold
    (`HITTER_MIN_MODEL_PROBABILITY`, now removed) - real live feedback
    surfaced a day that dropped a hitter batting high in the lineup in
    favor of the model's single favorite, since the model doesn't see
    lineup-order/everyday-player signals directly the way
    `Approach`/`Matchup_Approach` implicitly do (via the
    `avg_batting_order`/`start_rate` qualifiers below). Keeping the model
    as a broad shortlist gate still kills pure hot-streak outliers (the
    model's original purpose) while handing the final call back to the
    heuristic signal that captures what the model doesn't.

    `max_days_since_last_game` excludes a hitter whose most recent completed
    game (`hitters.compute_last_game_dates`'s `Last_Game_Date`) is more than
    this many days before `date` - column-gated like the lineup qualifiers
    (a missing `Last_Game_Date` column is a no-op, so old wave.csv snapshots
    from before this feature existed are unaffected), and a NaT (a batter
    with zero completed events at all) correctly fails the comparison, same
    "null loses, isn't filled to a value that would wrongly pass" precedent
    `avg_batting_order` already established.

    `teams_playing_today`, if given, additionally requires a batter's team
    to be in the set - unlike the lineup qualifiers this isn't column-gated
    (whether a team is playing today isn't a property of the hitters table
    itself); defaults to None, i.e. off.

    `model_version` is stamped onto every returned row (see
    config.HITTER_MODEL_VERSION) so evaluation.py/the dashboard can segment
    accuracy stats by which selection logic actually produced a given pick -
    without this, a qualifier/ranking change never visibly moves the
    dashboard's stats until the (much larger) pre-change history stops
    dominating the aggregate.

    `probability` and `Matchup_Hit_Probability` are logged alongside
    `predicted_probability` (which stays Game_Hit_Probability - a real,
    calibratable per-game rate, needed as-is for Brier/log-loss scoring) so
    evaluation.py's Beat the Streak "recommended" gate can blend all three
    signals instead of thresholding Game_Hit_Probability alone (see
    evaluation._combined_probability). `Matchup_Hit_Probability` is NaN
    whenever `hitters` doesn't carry it (no schedule/matchup data that
    day, or a historical wave.csv-only replay - see git_backtest.py).

    `same_game_diversification_margin` (quant-analytics item #4, slice 2)
    is a #2-pick-only tie-break: column-gated on a real `game_pk` being
    present on `hitters` (see pipeline.run) - a no-op whenever it's
    absent, same convention as the lineup/model-shortlist qualifiers
    above. See `_diversify_second_pick`'s own docstring for the real
    rule. 0.0 (the live default) is the exact null hypothesis - today's
    unmodified single-column ranking, bit-for-bit.
    """
    qualified = hitters[(hitters["PA_L"] + hitters["PA_R"]) >= min_plate_appearances].copy()
    if "avg_batting_order" in qualified.columns:
        qualified = qualified[qualified["avg_batting_order"] <= max_avg_batting_order]
    if "start_rate" in qualified.columns:
        qualified = qualified[qualified["start_rate"] >= min_start_rate]
    if "Last_Game_Date" in qualified.columns:
        days_since_last_game = (pd.Timestamp(date) - qualified["Last_Game_Date"]).dt.days
        qualified = qualified[days_since_last_game <= max_days_since_last_game]
    if teams_playing_today is not None:
        qualified = qualified[qualified["team"].isin(teams_playing_today)]
    if "Model_Hit_Probability" in qualified.columns:
        # Broad quality filter, not the final ranker: top model_shortlist_size
        # (or fewer, never padded, if the qualified pool is smaller) by
        # Model_Hit_Probability - excludes anyone the model doesn't rate at
        # all, but rank_metric (below) still decides the final order among
        # survivors. Replaces (not adds to) JOINT_PROBABILITY_GATE_COLUMNS,
        # same reasoning as the old threshold gate this replaced. A NaN
        # Model_Hit_Probability sorts last (pandas' default na_position) and
        # is therefore excluded exactly like a real low score.
        qualified = qualified.sort_values("Model_Hit_Probability", ascending=False).head(model_shortlist_size)
    else:
        for gate_column in JOINT_PROBABILITY_GATE_COLUMNS:
            if gate_column in qualified.columns:
                qualified = qualified[qualified[gate_column] >= min_probability]

    ranked = qualified.sort_values(rank_metric or metric, ascending=False).reset_index(drop=True)
    if "game_pk" in ranked.columns and same_game_diversification_margin > 0:
        ranked = _diversify_second_pick(ranked, rank_metric or metric, same_game_diversification_margin)
    picks = ranked.head(top_n).reset_index(drop=True)

    picks["rank"] = picks.index + 1
    picks["date"] = pd.Timestamp(date)
    picks["name"] = picks["name_first"].fillna("").astype(str) + " " + picks["name_last"].fillna("").astype(str)
    picks["predicted_probability"] = picks[metric]
    picks["metric"] = metric
    if "game_pk" not in picks.columns:
        picks["game_pk"] = pd.NA
    for optional_column in ("probability", "Matchup_Hit_Probability", "Model_Hit_Probability"):
        if optional_column not in picks.columns:
            picks[optional_column] = pd.NA
    picks["actual_hit"] = pd.NA
    picks["at_bats"] = pd.NA
    picks["model_version"] = model_version

    return picks[PREDICTION_COLUMNS]


def append_predictions(picks: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Append `picks` to the predictions log at `log_path`, deduping on
    (date, game_pk, key_mlbam, metric) so re-running a day's pipeline - or a
    `picks` batch that already contains duplicates itself, e.g. from git
    history replaying the same date via more than one commit - doesn't create
    duplicate log entries. Existing rows (including already-resolved
    actual_hit values) always win over a re-logged pick for the same key.

    Legacy rows may have a null `game_pk` (migration fills NA, not a guess).
    New live rows must carry a non-null game_pk. Pandas treats NA as equal
    in drop_duplicates, so legacy (date, NA, key_mlbam, metric) keys still
    dedupe the way the old (date, key_mlbam, metric) key did.

    That per-key dedup alone isn't enough when a rerun's TOP-N candidate
    SET changes for a date that's already logged, though - e.g. a mid-day
    code/model deploy between two same-day pipeline runs (real incident:
    2026-08-04, a Daily Update run before a merge landing v3-model-primary
    logged one set of hitters, a second run after the merge logged a
    mostly-disjoint set). Since the two runs' picks don't share key_mlbam
    values, per-key dedup leaves BOTH sets sitting in the log side by
    side, and evaluation.py's _combined_probability (which decides the
    actual displayed "recommended" pick) has no way to know one set is
    stale - it just picks whichever row scores higher, which is exactly
    backwards if the old run happens to score better on the old heuristic
    signals. So: for any date in the new `picks` batch that has NOT yet
    resolved any row in the existing log (every existing row for that
    date still has a null at_bats, matching resolve_predictions's own
    "still pending" definition), the ENTIRE date's existing rows are
    dropped before the new batch is appended - a full resupersede, not a
    per-player upsert. A date with even one resolved row is never touched
    this way; only the per-key dedup below applies there, preserving
    already-resolved backtest history exactly as before."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    picks = _ensure_game_pk_column(picks)
    picks["date"] = pd.to_datetime(picks["date"])

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["date"])
        if "model_version" not in existing.columns:
            existing["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
        existing = _ensure_game_pk_column(existing)
        existing["date"] = pd.to_datetime(existing["date"])
        fresh_dates = set(picks["date"])
        resolved_dates = set(existing.loc[existing["at_bats"].notna(), "date"])
        supersede_dates = fresh_dates - resolved_dates
        existing = existing[~existing["date"].isin(supersede_dates)]
        combined = pd.concat([picks, existing], ignore_index=True)
    else:
        combined = picks

    combined = combined.drop_duplicates(subset=PREDICTION_KEY_COLUMNS, keep="last")
    combined = combined.sort_values(["date", "rank"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def resolve_predictions(log_path: str, completed_events_by_date: pd.DataFrame) -> pd.DataFrame:
    """Fill in `at_bats`/`actual_hit` for any still-pending rows in the
    predictions log whose date is covered by `completed_events_by_date`
    (columns: game_date, batter, events, and game_pk when available -
    e.g. the persisted raw Statcast data).

    When both the log row and the events carry a non-null `game_pk`,
    outcomes are labeled per (game_pk, batter) - a doubleheader miss in
    game one and hit in game two resolve independently. Legacy rows with a
    null game_pk still resolve on (date, batter), preserving pre-migration
    history readability.

    A row is "still pending" if its `at_bats` is null - not
    `actual_hit`, since a batter can be fully resolved with zero at-bats
    (rained out, DNP, etc: at_bats=0, actual_hit stays null because there's
    no hit/miss to score) which must stay distinguishable from a date we
    simply haven't seen outcome data for yet. A row only gets resolved once
    its date is <= the latest date present anywhere in
    `completed_events_by_date` - not just once *that batter* appears in it -
    so a confirmed zero-at-bats day is never mistaken for "not checked yet".
    Already-resolved rows (at_bats already set) are left untouched."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    log = pd.read_csv(log_path, parse_dates=["date"])
    if log.empty:
        return log
    if "at_bats" not in log.columns:
        log["at_bats"] = pd.NA  # migrate a log written before at_bats existed
    if "model_version" not in log.columns:
        log["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
    log = _ensure_game_pk_column(log)

    events = completed_events_by_date.copy()
    events["had_hit"] = helpers.is_hit(events["events"])
    known_through = events["game_date"].max() if len(events) else pd.NaT

    still_pending = log["at_bats"].isna()
    knowable = pd.notna(known_through) & (log["date"] <= known_through)
    resolvable = still_pending & knowable

    has_event_game_pk = "game_pk" in events.columns
    game_keyed = resolvable & log["game_pk"].notna() if has_event_game_pk else pd.Series(False, index=log.index)
    date_keyed = resolvable & ~game_keyed

    log["resolved_at_bats"] = pd.NA
    log["resolved_hit"] = pd.NA

    if game_keyed.any():
        per_batter_game = (
            events.groupby(["game_pk", "batter"])
            .agg(resolved_at_bats=("events", "size"), resolved_hit=("had_hit", "max"))
            .reset_index()
            .rename(columns={"batter": "key_mlbam"})
        )
        # reset_index so merge results reattach by original log index -
        # never assign via positional .to_numpy() after a join that could
        # reorder or fan out duplicates.
        left_game = log.loc[game_keyed, ["game_pk", "key_mlbam"]].reset_index()
        merged_game = left_game.merge(per_batter_game, on=["game_pk", "key_mlbam"], how="left")
        log.loc[merged_game["index"], "resolved_at_bats"] = merged_game["resolved_at_bats"].to_numpy()
        log.loc[merged_game["index"], "resolved_hit"] = merged_game["resolved_hit"].to_numpy()

    if date_keyed.any():
        per_batter_day = (
            events.groupby(["game_date", "batter"])
            .agg(resolved_at_bats=("events", "size"), resolved_hit=("had_hit", "max"))
            .reset_index()
            .rename(columns={"game_date": "date", "batter": "key_mlbam"})
        )
        left_day = log.loc[date_keyed, ["date", "key_mlbam"]].reset_index()
        merged_day = left_day.merge(per_batter_day, on=["date", "key_mlbam"], how="left")
        log.loc[merged_day["index"], "resolved_at_bats"] = merged_day["resolved_at_bats"].to_numpy()
        log.loc[merged_day["index"], "resolved_hit"] = merged_day["resolved_hit"].to_numpy()

    resolved_at_bats = pd.to_numeric(log["resolved_at_bats"], errors="coerce").fillna(0)
    log.loc[resolvable, "at_bats"] = resolved_at_bats[resolvable]

    got_hit = resolvable & (resolved_at_bats > 0) & (pd.to_numeric(log["resolved_hit"], errors="coerce") == 1)
    got_out = resolvable & (resolved_at_bats > 0) & (pd.to_numeric(log["resolved_hit"], errors="coerce") != 1)
    log.loc[got_hit, "actual_hit"] = 1
    log.loc[got_out, "actual_hit"] = 0
    # resolvable & resolved_at_bats == 0 (no_game): at_bats is now 0, but
    # actual_hit is intentionally left null - there's no hit/miss to record.

    log = log.drop(columns=["resolved_at_bats", "resolved_hit"])
    log.to_csv(log_path, index=False)
    return log
