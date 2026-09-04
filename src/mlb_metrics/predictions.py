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

New rows also carry prediction/model provenance (see PROVENANCE_COLUMNS) so
a logged pick records which code SHA / model artifact / selection metric
produced it. Legacy CSV rows migrate with null/"legacy" provenance defaults
without data loss.
"""

import os
from datetime import datetime, timezone

import pandas as pd

from mlb_metrics import config, helpers, ml_models

PREDICTION_COLUMNS = [
    "date", "game_pk", "key_mlbam", "name", "rank", "predicted_probability", "metric",
    "probability", "Matchup_Hit_Probability", "Model_Hit_Probability",
    "Game_Hit_Probability", "Final_Hit_Probability",
    "P_Appear", "Expected_PA_hat", "P_Hit_Given_Appearance",
    "shadow_rank", "shadow_model_artifact_id", "selection_mode",
    "actual_hit", "at_bats", "model_version",
    # Provenance - see PROVENANCE_COLUMNS. Kept in PREDICTION_COLUMNS so
    # append/resolve/export always round-trip the full schema.
    "prediction_timestamp_utc", "prediction_code_sha", "model_artifact_id",
    "training_data_cutoff", "feature_schema_hash", "selection_logic_version",
    "selection_metric", "selection_score", "probability_source",
    "fallback_used", "fallback_reason",
    "prediction_snapshot_type", "lineup_status", "starter_status",
]

PROVENANCE_COLUMNS = [
    "prediction_timestamp_utc", "prediction_code_sha", "model_artifact_id",
    "training_data_cutoff", "feature_schema_hash", "selection_logic_version",
    "selection_metric", "selection_score", "probability_source",
    "fallback_used", "fallback_reason",
    "prediction_snapshot_type", "lineup_status", "starter_status",
]

# Authoritative probability column name used in live mode (and logged as
# diagnostics in shadow mode).
FINAL_HIT_PROBABILITY = "Final_Hit_Probability"

# Tag applied (via the migration guards in append_predictions/resolve_predictions)
# to any row logged before the model_version column existed, or reconstructed
# by a historical replay (see git_backtest.py) - distinguishes "we don't know
# which logic produced this" from a real current-version live pick.
LEGACY_MODEL_VERSION = "legacy"

# Defaults applied when migrating a CSV written before provenance columns
# existed - null/"legacy"/False rather than invented training metadata.
_PROVENANCE_MIGRATION_DEFAULTS = {
    "prediction_timestamp_utc": pd.NA,
    "prediction_code_sha": "legacy",
    "model_artifact_id": pd.NA,
    "training_data_cutoff": pd.NA,
    "feature_schema_hash": pd.NA,
    "selection_logic_version": "legacy",
    "selection_metric": pd.NA,
    "selection_score": pd.NA,
    "probability_source": pd.NA,
    "fallback_used": pd.NA,
    "fallback_reason": pd.NA,
    "prediction_snapshot_type": "legacy",
    "lineup_status": "legacy",
    "starter_status": "legacy",
}

_FINAL_HIT_MIGRATION_DEFAULTS = {
    "Game_Hit_Probability": pd.NA,
    "Final_Hit_Probability": pd.NA,
    "P_Appear": pd.NA,
    "Expected_PA_hat": pd.NA,
    "P_Hit_Given_Appearance": pd.NA,
    "shadow_rank": pd.NA,
    "shadow_model_artifact_id": pd.NA,
    "selection_mode": "legacy",
}

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


def _ensure_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Migrate a frame written before provenance / Final_Hit columns existed."""
    df = df.copy()
    for column, default in _PROVENANCE_MIGRATION_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
    for column, default in _FINAL_HIT_MIGRATION_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
    return df


def _infer_starter_status(row_frame: pd.DataFrame) -> pd.Series:
    """probable / missing / confirmed from optional schedule columns."""
    if "starter_status" in row_frame.columns:
        return row_frame["starter_status"]
    if "probable_pitcher_key_mlbam" in row_frame.columns:
        return row_frame["probable_pitcher_key_mlbam"].notna().map({True: "probable", False: "missing"})
    if "Matchup_Hit_Probability" in row_frame.columns:
        # Matchup ran (schedule present) but starter id wasn't carried onto
        # the pick row - still a probable-starter slate, not a confirmed
        # lineup announcement.
        return pd.Series("probable", index=row_frame.index)
    return pd.Series("missing", index=row_frame.index)


def _infer_lineup_status(row_frame: pd.DataFrame) -> pd.Series:
    if "lineup_status" in row_frame.columns:
        return row_frame["lineup_status"]
    # Historical batting-order consistency is not a confirmed today's lineup.
    return pd.Series("unconfirmed", index=row_frame.index)


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


def _apply_base_qualifiers(
    hitters: pd.DataFrame,
    date,
    *,
    min_plate_appearances: int,
    max_avg_batting_order: float,
    min_start_rate: float,
    max_days_since_last_game: int,
    teams_playing_today: set[str] | None,
) -> pd.DataFrame:
    """Shared PA / lineup / recency / slate filters (all modes)."""
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
    return qualified


def _stamp_pick_identity(picks: pd.DataFrame, date, metric: str) -> pd.DataFrame:
    picks = picks.copy()
    picks["rank"] = picks.index + 1
    picks["date"] = pd.Timestamp(date)
    picks["name"] = picks["name_first"].fillna("").astype(str) + " " + picks["name_last"].fillna("").astype(str)
    picks["metric"] = metric
    if "game_pk" not in picks.columns:
        picks["game_pk"] = pd.NA
    for optional_column in (
        "probability", "Matchup_Hit_Probability", "Model_Hit_Probability",
        "Game_Hit_Probability", FINAL_HIT_PROBABILITY,
        "P_Appear", "Expected_PA_hat", "P_Hit_Given_Appearance",
        "shadow_rank", "shadow_model_artifact_id",
    ):
        if optional_column not in picks.columns:
            picks[optional_column] = pd.NA
    picks["actual_hit"] = pd.NA
    picks["at_bats"] = pd.NA
    return picks


def _stamp_provenance(
    picks: pd.DataFrame,
    *,
    model_version: str,
    used_rank_metric: str,
    probability_source: str,
    selection_mode: str,
    model_status: dict | None,
    fallback_used: bool | None,
    fallback_reason: str | None,
    prediction_snapshot_type: str,
) -> pd.DataFrame:
    picks = picks.copy()
    picks["model_version"] = model_version
    if model_status is not None:
        provenance = ml_models.provenance_fields_from_model_status(model_status)
    else:
        provenance = {
            "model_artifact_id": pd.NA,
            "training_data_cutoff": pd.NA,
            "feature_schema_hash": pd.NA,
            "fallback_used": False,
            "fallback_reason": None,
        }
    if fallback_used is not None:
        provenance["fallback_used"] = bool(fallback_used)
        provenance["fallback_reason"] = fallback_reason

    picks["prediction_timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    picks["prediction_code_sha"] = ml_models.resolve_code_sha()
    picks["model_artifact_id"] = provenance["model_artifact_id"]
    picks["training_data_cutoff"] = provenance["training_data_cutoff"]
    picks["feature_schema_hash"] = provenance["feature_schema_hash"]
    picks["selection_logic_version"] = model_version
    picks["selection_metric"] = used_rank_metric
    picks["selection_score"] = picks[used_rank_metric] if used_rank_metric in picks.columns else pd.NA
    picks["probability_source"] = probability_source
    picks["fallback_used"] = provenance["fallback_used"]
    picks["fallback_reason"] = provenance["fallback_reason"]
    picks["prediction_snapshot_type"] = prediction_snapshot_type
    picks["selection_mode"] = selection_mode
    picks["lineup_status"] = _infer_lineup_status(picks)
    picks["starter_status"] = _infer_starter_status(picks)
    return picks


def _attach_shadow_diagnostics(
    picks: pd.DataFrame,
    hitters: pd.DataFrame,
    *,
    shadow_model_artifact_id=None,
) -> pd.DataFrame:
    """Copy Final_Hit_Probability components + shadow_rank onto official picks.

    Does not change rank / predicted_probability / selection_metric.
    """
    picks = picks.copy()
    shadow_cols = [
        FINAL_HIT_PROBABILITY, "P_Appear", "Expected_PA_hat", "P_Hit_Given_Appearance",
        "shadow_rank",
    ]
    keys = [c for c in ("key_mlbam", "game_pk") if c in picks.columns and c in hitters.columns]
    if not keys or FINAL_HIT_PROBABILITY not in hitters.columns:
        for col in shadow_cols:
            if col not in picks.columns:
                picks[col] = pd.NA
        picks["shadow_model_artifact_id"] = shadow_model_artifact_id if shadow_model_artifact_id is not None else pd.NA
        return picks

    src = hitters[keys + [c for c in shadow_cols if c in hitters.columns]].drop_duplicates(keys)
    merged = picks.drop(columns=[c for c in shadow_cols if c in picks.columns], errors="ignore")
    merged = merged.merge(src, on=keys, how="left")
    merged["shadow_model_artifact_id"] = shadow_model_artifact_id if shadow_model_artifact_id is not None else pd.NA
    for col in shadow_cols:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged


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
    model_status: dict | None = None,
    fallback_used: bool | None = None,
    fallback_reason: str | None = None,
    prediction_snapshot_type: str = "morning",
    selection_mode: str | None = None,
    force_live: bool = False,
    shadow_model_status: dict | None = None,
) -> pd.DataFrame:
    """Rank a computed hitters table and return top ``top_n`` qualified picks.

    ``selection_mode`` (``legacy`` | ``shadow`` | ``live``, default from
    ``config.HITTER_SELECTION_MODE`` via the promotion-gate resolver):

    - **legacy**: current production behavior (optional Model_Hit shortlist,
      rank by ``rank_metric``, log ``metric`` as ``predicted_probability``).
    - **shadow**: identical official picks to legacy, plus Final_Hit_Probability
      components / shadow_rank / shadow artifact id on each logged row.
    - **live**: ``Final_Hit_Probability`` is the sole quantity for ranking,
      thresholding, and logged ``predicted_probability`` /
      ``selection_score``. No ML top-10 shortlist; older signals stay as
      diagnostics only. Bumps ``model_version`` to
      ``HITTER_MODEL_VERSION_LIVE``.

    Live requires ``Final_Hit_Probability`` on ``hitters``. If missing, falls
    back to legacy/shadow behavior with ``fallback_used`` /
    ``fallback_reason`` recorded (never silent).

    In legacy/shadow, ``rank_metric`` may differ from ``metric`` (historical
    Matchup_Approach rank + Game_Hit_Probability log). In live they are
    forced equal to ``Final_Hit_Probability``.
    """
    from mlb_metrics import hitter_probability_model as hpm

    configured = selection_mode if selection_mode is not None else config.HITTER_SELECTION_MODE
    mode, mode_meta = hpm.resolve_hitter_selection_mode(configured, force_live=force_live)

    # Live without Final_Hit_Probability → explicit fallback, never silent.
    live_fallback = False
    live_fallback_reason = None
    if mode == "live" and (
        FINAL_HIT_PROBABILITY not in hitters.columns
        or hitters[FINAL_HIT_PROBABILITY].isna().all()
    ):
        live_fallback = True
        live_fallback_reason = "missing_final_hit_probability"
        mode = "shadow" if configured in ("live", "shadow") else "legacy"
        if fallback_used is None:
            fallback_used = True
            fallback_reason = live_fallback_reason
        elif not fallback_used:
            fallback_used = True
            fallback_reason = live_fallback_reason

    if mode == "live":
        return _select_picks_live(
            hitters, date,
            top_n=top_n,
            min_plate_appearances=min_plate_appearances,
            min_probability=min_probability,
            max_avg_batting_order=max_avg_batting_order,
            min_start_rate=min_start_rate,
            max_days_since_last_game=max_days_since_last_game,
            teams_playing_today=teams_playing_today,
            same_game_diversification_margin=same_game_diversification_margin,
            model_status=model_status if model_status is not None else shadow_model_status,
            fallback_used=fallback_used if fallback_used is not None else False,
            fallback_reason=fallback_reason,
            prediction_snapshot_type=prediction_snapshot_type,
            selection_mode="live",
        )

    picks = _select_picks_legacy(
        hitters, date,
        top_n=top_n,
        min_plate_appearances=min_plate_appearances,
        metric=metric,
        rank_metric=rank_metric,
        min_probability=min_probability,
        model_shortlist_size=model_shortlist_size,
        max_avg_batting_order=max_avg_batting_order,
        min_start_rate=min_start_rate,
        max_days_since_last_game=max_days_since_last_game,
        teams_playing_today=teams_playing_today,
        model_version=model_version,
        same_game_diversification_margin=same_game_diversification_margin,
        model_status=model_status,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        prediction_snapshot_type=prediction_snapshot_type,
        selection_mode=mode,
    )

    if mode == "shadow":
        artifact = None
        if shadow_model_status:
            artifact = shadow_model_status.get("artifact_id")
        elif model_status and model_status.get("model_type") == hpm.MODEL_TYPE:
            artifact = model_status.get("artifact_id")
        picks = _attach_shadow_diagnostics(
            picks, hitters, shadow_model_artifact_id=artifact,
        )
        # Mode-resolution fallback (e.g. live→shadow) must remain visible.
        if mode_meta.get("fallback_used") and not bool(picks["fallback_used"].iloc[0] if len(picks) else False):
            picks["fallback_used"] = True
            picks["fallback_reason"] = mode_meta.get("fallback_reason")
        elif live_fallback:
            picks["fallback_used"] = True
            picks["fallback_reason"] = live_fallback_reason

    return picks[PREDICTION_COLUMNS]


def _select_picks_live(
    hitters: pd.DataFrame,
    date,
    *,
    top_n: int,
    min_plate_appearances: int,
    min_probability: float,
    max_avg_batting_order: float,
    min_start_rate: float,
    max_days_since_last_game: int,
    teams_playing_today: set[str] | None,
    same_game_diversification_margin: float,
    model_status: dict | None,
    fallback_used: bool | None,
    fallback_reason: str | None,
    prediction_snapshot_type: str,
    selection_mode: str,
) -> pd.DataFrame:
    """Live: Final_Hit_Probability is the only ranking / threshold / log score."""
    authoritative = FINAL_HIT_PROBABILITY
    qualified = _apply_base_qualifiers(
        hitters, date,
        min_plate_appearances=min_plate_appearances,
        max_avg_batting_order=max_avg_batting_order,
        min_start_rate=min_start_rate,
        max_days_since_last_game=max_days_since_last_game,
        teams_playing_today=teams_playing_today,
    )
    # No Model_Hit_Probability shortlist. Threshold on Final_Hit_Probability only.
    qualified = qualified[qualified[authoritative].astype(float) >= min_probability]
    ranked = qualified.sort_values(authoritative, ascending=False).reset_index(drop=True)
    if "game_pk" in ranked.columns and same_game_diversification_margin > 0:
        ranked = _diversify_second_pick(ranked, authoritative, same_game_diversification_margin)
    picks = ranked.head(top_n).reset_index(drop=True)

    picks = _stamp_pick_identity(picks, date, metric=authoritative)
    picks["predicted_probability"] = picks[authoritative].astype(float)
    # Diagnostics only — never used for ranking/thresholding in live mode.
    if "Game_Hit_Probability" not in picks.columns or picks["Game_Hit_Probability"].isna().all():
        if "Game_Hit_Probability" in hitters.columns:
            # Already on picks via head(); ensure column exists
            pass
    picks["shadow_rank"] = pd.NA
    picks["shadow_model_artifact_id"] = pd.NA
    picks = _stamp_provenance(
        picks,
        model_version=config.HITTER_MODEL_VERSION_LIVE,
        used_rank_metric=authoritative,
        probability_source=authoritative,
        selection_mode=selection_mode,
        model_status=model_status,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        prediction_snapshot_type=prediction_snapshot_type,
    )
    # Hard invariants: one column owns rank, threshold, log, and score.
    picks["selection_metric"] = authoritative
    picks["selection_score"] = picks[authoritative].astype(float)
    picks["predicted_probability"] = picks[authoritative].astype(float)
    picks["probability_source"] = authoritative
    return picks[PREDICTION_COLUMNS]


def _select_picks_legacy(
    hitters: pd.DataFrame,
    date,
    *,
    top_n: int,
    min_plate_appearances: int,
    metric: str,
    rank_metric: str | None,
    min_probability: float,
    model_shortlist_size: int,
    max_avg_batting_order: float,
    min_start_rate: float,
    max_days_since_last_game: int,
    teams_playing_today: set[str] | None,
    model_version: str,
    same_game_diversification_margin: float,
    model_status: dict | None,
    fallback_used: bool | None,
    fallback_reason: str | None,
    prediction_snapshot_type: str,
    selection_mode: str,
) -> pd.DataFrame:
    """Preserve v4 production selection (shortlist + rank_metric ≠ metric OK)."""
    used_rank_metric = rank_metric or metric
    qualified = _apply_base_qualifiers(
        hitters, date,
        min_plate_appearances=min_plate_appearances,
        max_avg_batting_order=max_avg_batting_order,
        min_start_rate=min_start_rate,
        max_days_since_last_game=max_days_since_last_game,
        teams_playing_today=teams_playing_today,
    )
    if "Model_Hit_Probability" in qualified.columns:
        qualified = qualified.sort_values("Model_Hit_Probability", ascending=False).head(model_shortlist_size)
    else:
        for gate_column in JOINT_PROBABILITY_GATE_COLUMNS:
            if gate_column in qualified.columns:
                qualified = qualified[qualified[gate_column] >= min_probability]

    ranked = qualified.sort_values(used_rank_metric, ascending=False).reset_index(drop=True)
    if "game_pk" in ranked.columns and same_game_diversification_margin > 0:
        ranked = _diversify_second_pick(ranked, used_rank_metric, same_game_diversification_margin)
    picks = ranked.head(top_n).reset_index(drop=True)

    picks = _stamp_pick_identity(picks, date, metric=metric)
    picks["predicted_probability"] = picks[metric]
    if "Game_Hit_Probability" not in picks.columns or picks["Game_Hit_Probability"].isna().all():
        if metric == "Game_Hit_Probability":
            picks["Game_Hit_Probability"] = picks["predicted_probability"]
    picks = _stamp_provenance(
        picks,
        model_version=model_version,
        used_rank_metric=used_rank_metric,
        probability_source=metric,
        selection_mode=selection_mode,
        model_status=model_status,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        prediction_snapshot_type=prediction_snapshot_type,
    )
    return picks

def append_prediction_audit(
    picks: pd.DataFrame,
    audit_path: str | None = None,
    *,
    published_log_path: str | None = None,
) -> None:
    """Append-only immutable audit of every prediction batch (morning / lineup_lock)."""
    if audit_path is None:
        if published_log_path:
            root, ext = os.path.splitext(published_log_path)
            audit_path = f"{root}_audit{ext}"
        else:
            audit_path = config.PREDICTIONS_AUDIT_PATH
    frame = picks.copy()
    if frame.empty:
        return
    os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
    if os.path.exists(audit_path):
        existing = pd.read_csv(audit_path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame
    combined.to_csv(audit_path, index=False)


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
    per-player upsert.

    Lineup-lock refreshes are narrower: when the fresh batch carries
    non-null ``game_pk`` values, unresolved rows for those
    ``(date, game_pk)`` keys are superseded even if *other* games on the
    same calendar date are already resolved. Started/resolved rows
    (``at_bats`` not null) are never rewritten. Every batch is also
    appended to the immutable audit log (``PREDICTIONS_AUDIT_PATH``)."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    picks = _ensure_game_pk_column(picks)
    picks = _ensure_provenance_columns(picks)
    picks["date"] = pd.to_datetime(picks["date"])
    append_prediction_audit(picks, published_log_path=log_path)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["date"])
        if "model_version" not in existing.columns:
            existing["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
        existing = _ensure_game_pk_column(existing)
        existing = _ensure_provenance_columns(existing)
        existing["date"] = pd.to_datetime(existing["date"])

        # Full-date supersede when nothing on that date has resolved yet
        # (morning re-run / mid-day deploy case).
        fresh_dates = set(picks["date"])
        resolved_dates = set(existing.loc[existing["at_bats"].notna(), "date"])
        supersede_dates = fresh_dates - resolved_dates
        existing = existing[~existing["date"].isin(supersede_dates)]

        # Per-game supersede for lineup-lock: drop unresolved rows whose
        # (date, game_pk) appears in the fresh batch, even when other games
        # on the same date are already resolved. Null game_pk rows keep
        # the date-level rule only.
        existing["game_pk"] = pd.to_numeric(existing["game_pk"], errors="coerce")
        picks["game_pk"] = pd.to_numeric(picks["game_pk"], errors="coerce")
        fresh_games = picks.loc[picks["game_pk"].notna(), ["date", "game_pk"]].drop_duplicates()
        if not fresh_games.empty:
            existing = existing.merge(
                fresh_games.assign(_fresh_game=1),
                on=["date", "game_pk"],
                how="left",
            )
            is_resolved = existing["at_bats"].notna()
            drop_mask = existing["_fresh_game"].eq(1) & ~is_resolved
            existing = existing.loc[~drop_mask].drop(columns=["_fresh_game"], errors="ignore")

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
    log = _ensure_provenance_columns(log)

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
