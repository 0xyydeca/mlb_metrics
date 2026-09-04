"""As-of schedule / probable-starter snapshots for production-equivalent
game-model backtests.

Live pipeline runs see only the probable starters announced by the
prediction timestamp. Retrospective reconstruction that substitutes the
**actual** starter is an upper-bound diagnostic, not a production-
equivalent backtest. This module persists append-only schedule snapshots
and selects the latest snapshot available at or before a prediction
timestamp.

Backtest modes (``config.SCHEDULE_BACKTEST_MODES``)::

- ``as_of_snapshot`` — production-equivalent; primary
- ``actual_starter`` — retrospective upper-bound diagnostic (labeled)
- ``missing_snapshot`` — explicitly unavailable; never silently substituted

Never merge ``actual_starter`` rows into an ``as_of_snapshot`` evaluation
set. Until enough forward snapshots accumulate, report production-
equivalent sample size honestly.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from mlb_metrics import config

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEDULE_SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "captured_at_utc",
    "prediction_target_date",
    "game_pk",
    "game_datetime",
    "home_team",
    "away_team",
    "home_probable_pitcher_key_mlbam",
    "away_probable_pitcher_key_mlbam",
    "starter_announcement_status",
    "game_status",
    "source",
]

STARTER_STATUS_BOTH = "both_announced"
STARTER_STATUS_HOME_ONLY = "home_only"
STARTER_STATUS_AWAY_ONLY = "away_only"
STARTER_STATUS_MISSING = "missing"

SOURCE_STATSAPI = "statsapi_schedule"
SOURCE_FIXTURE = "fixture"
SOURCE_MANUAL = "manual"
SOURCE_ACTUAL_STARTER_DIAGNOSTIC = "actual_starter_diagnostic"

BACKTEST_MODE_AS_OF = "as_of_snapshot"
BACKTEST_MODE_ACTUAL = "actual_starter"
BACKTEST_MODE_MISSING = "missing_snapshot"

AS_OF_SCHEDULE_COLUMNS = [
    "game_pk", "date", "home_team", "away_team",
    "home_probable_pitcher_key_mlbam", "away_probable_pitcher_key_mlbam",
    "status", "home_score", "away_score", "game_datetime",
    "schedule_snapshot_id", "schedule_captured_at_utc",
    "starter_announcement_status", "schedule_backtest_mode",
    "schedule_source",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_utc_ts(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def starter_announcement_status(home_pitcher, away_pitcher) -> str:
    home_ok = pd.notna(home_pitcher)
    away_ok = pd.notna(away_pitcher)
    if home_ok and away_ok:
        return STARTER_STATUS_BOTH
    if home_ok:
        return STARTER_STATUS_HOME_ONLY
    if away_ok:
        return STARTER_STATUS_AWAY_ONLY
    return STARTER_STATUS_MISSING


def make_snapshot_id(row: dict | pd.Series) -> str:
    payload = "|".join(
        str(row.get(k))
        for k in (
            "captured_at_utc",
            "prediction_target_date",
            "game_pk",
            "home_probable_pitcher_key_mlbam",
            "away_probable_pitcher_key_mlbam",
            "game_status",
            "source",
        )
    )
    return "sched_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def empty_snapshot_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEDULE_SNAPSHOT_COLUMNS)


def normalize_snapshot_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy() if df is not None else empty_snapshot_frame()
    for col in SCHEDULE_SNAPSHOT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if not out.empty:
        out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
        out["home_probable_pitcher_key_mlbam"] = pd.to_numeric(
            out["home_probable_pitcher_key_mlbam"], errors="coerce"
        )
        out["away_probable_pitcher_key_mlbam"] = pd.to_numeric(
            out["away_probable_pitcher_key_mlbam"], errors="coerce"
        )
        if "prediction_target_date" in out.columns:
            out["prediction_target_date"] = pd.to_datetime(
                out["prediction_target_date"], errors="coerce"
            ).dt.normalize()
    return out[SCHEDULE_SNAPSHOT_COLUMNS]


def snapshots_from_schedule_games(
    schedule_games: pd.DataFrame,
    *,
    captured_at_utc: str | None = None,
    prediction_target_date=None,
    source: str = SOURCE_STATSAPI,
) -> pd.DataFrame:
    """Build snapshot rows from ``schedule.normalize_schedule_games`` output."""
    if schedule_games is None or schedule_games.empty:
        return empty_snapshot_frame()

    captured = captured_at_utc or utc_now_iso()
    rows = []
    for _, game in schedule_games.iterrows():
        home_p = game.get("home_probable_pitcher_key_mlbam")
        away_p = game.get("away_probable_pitcher_key_mlbam")
        target = prediction_target_date
        if target is None:
            target = game.get("date")
        row = {
            "captured_at_utc": captured,
            "prediction_target_date": pd.Timestamp(target).normalize() if pd.notna(target) else pd.NaT,
            "game_pk": game.get("game_pk"),
            "game_datetime": game.get("game_datetime"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "home_probable_pitcher_key_mlbam": home_p,
            "away_probable_pitcher_key_mlbam": away_p,
            "starter_announcement_status": starter_announcement_status(home_p, away_p),
            "game_status": game.get("status") or game.get("game_status"),
            "source": source,
        }
        row["snapshot_id"] = make_snapshot_id(row)
        rows.append(row)
    return normalize_snapshot_frame(pd.DataFrame(rows))


def load_schedule_snapshots(path: str | None = None) -> pd.DataFrame:
    path = path or config.SCHEDULE_SNAPSHOTS_PATH
    if not path or not os.path.exists(path):
        return empty_snapshot_frame()
    return normalize_snapshot_frame(pd.read_csv(path))


def append_schedule_snapshots(
    snapshots: pd.DataFrame,
    path: str | None = None,
) -> pd.DataFrame:
    """Append-only persistence. Never overwrites prior captures."""
    path = path or config.SCHEDULE_SNAPSHOTS_PATH
    frame = normalize_snapshot_frame(snapshots)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        existing = load_schedule_snapshots(path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame
    # Deduplicate exact snapshot_id only - same game at a later capture stays.
    if not combined.empty and "snapshot_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["snapshot_id"], keep="first")
    combined = normalize_snapshot_frame(combined)
    combined.to_csv(path, index=False)
    return combined


def persist_schedule_from_live_games(
    schedule_games: pd.DataFrame,
    *,
    captured_at_utc: str | None = None,
    prediction_target_date=None,
    source: str = SOURCE_STATSAPI,
    path: str | None = None,
) -> pd.DataFrame:
    snaps = snapshots_from_schedule_games(
        schedule_games,
        captured_at_utc=captured_at_utc,
        prediction_target_date=prediction_target_date,
        source=source,
    )
    if snaps.empty:
        return snaps
    return append_schedule_snapshots(snaps, path=path)


# ---------------------------------------------------------------------------
# As-of selection
# ---------------------------------------------------------------------------


def select_as_of_snapshots(
    snapshots: pd.DataFrame,
    prediction_timestamp,
    *,
    prediction_target_date=None,
    game_pks: list | None = None,
) -> pd.DataFrame:
    """Latest schedule snapshot per game_pk with ``captured_at <= prediction_ts``.

    Snapshots captured **after** the prediction timestamp are never used.
    When ``prediction_target_date`` is set, only rows for that slate date
    are considered. Returns one row per game_pk (or empty).
    """
    frame = normalize_snapshot_frame(snapshots)
    if frame.empty:
        return empty_snapshot_frame()

    pred_ts = _to_utc_ts(prediction_timestamp)
    if pd.isna(pred_ts):
        raise ValueError("prediction_timestamp must be a parseable UTC timestamp")

    work = frame.copy()
    work["_ts"] = work["captured_at_utc"].map(_to_utc_ts)
    work = work[work["_ts"].notna() & (work["_ts"] <= pred_ts)]
    if prediction_target_date is not None:
        target = pd.Timestamp(prediction_target_date).normalize()
        work = work[work["prediction_target_date"] == target]
    if game_pks is not None:
        pk_set = set(pd.to_numeric(pd.Series(list(game_pks)), errors="coerce").dropna().astype(int))
        work = work[work["game_pk"].isin(pk_set)]
    if work.empty:
        return empty_snapshot_frame()

    work = work.sort_values(["game_pk", "_ts"])
    latest = work.drop_duplicates(subset=["game_pk"], keep="last")
    return normalize_snapshot_frame(latest.drop(columns=["_ts"], errors="ignore"))


def as_of_snapshots_to_schedule_games(
    as_of_snapshots: pd.DataFrame,
    *,
    scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project as-of snapshots into the ``normalize_schedule_games`` shape.

    Scores (for resolution) may be joined separately; they are not part of
    the pregame snapshot and must not invent starters.
    """
    frame = normalize_snapshot_frame(as_of_snapshots)
    if frame.empty:
        return pd.DataFrame(columns=AS_OF_SCHEDULE_COLUMNS)

    out = pd.DataFrame({
        "game_pk": frame["game_pk"],
        "date": frame["prediction_target_date"],
        "home_team": frame["home_team"],
        "away_team": frame["away_team"],
        "home_probable_pitcher_key_mlbam": frame["home_probable_pitcher_key_mlbam"],
        "away_probable_pitcher_key_mlbam": frame["away_probable_pitcher_key_mlbam"],
        "status": frame["game_status"],
        "home_score": pd.NA,
        "away_score": pd.NA,
        "game_datetime": frame["game_datetime"],
        "schedule_snapshot_id": frame["snapshot_id"],
        "schedule_captured_at_utc": frame["captured_at_utc"],
        "starter_announcement_status": frame["starter_announcement_status"],
        "schedule_backtest_mode": BACKTEST_MODE_AS_OF,
        "schedule_source": frame["source"],
    })
    if scores is not None and not scores.empty and "game_pk" in scores.columns:
        score_keep = ["game_pk"]
        for col in ("home_score", "away_score"):
            if col in scores.columns:
                score_keep.append(col)
        scored = scores[score_keep].drop_duplicates("game_pk")
        out = out.drop(columns=[c for c in ("home_score", "away_score") if c in out.columns], errors="ignore")
        out = out.merge(scored, on="game_pk", how="left")
        for col in ("home_score", "away_score"):
            if col not in out.columns:
                out[col] = pd.NA
    return out


# ---------------------------------------------------------------------------
# Backtest mode resolution
# ---------------------------------------------------------------------------


def resolve_schedule_for_backtest(
    mode: str,
    *,
    prediction_timestamp=None,
    prediction_target_date=None,
    schedule_snapshots: pd.DataFrame | None = None,
    actual_starter_games: pd.DataFrame | None = None,
    game_pks: list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Resolve the schedule frame for a backtest prediction timestamp.

    Returns ``(schedule_games, meta)``. Modes never silently cross-substitute:

    - ``as_of_snapshot``: only games with a snapshot at/before the timestamp
    - ``actual_starter``: labeled diagnostic from Statcast actual starters
    - ``missing_snapshot``: empty frame with explicit unavailable reason
    """
    mode = (mode or config.SCHEDULE_BACKTEST_MODE_DEFAULT).strip()
    meta: dict[str, Any] = {
        "schedule_backtest_mode": mode,
        "prediction_timestamp": str(prediction_timestamp) if prediction_timestamp is not None else None,
        "prediction_target_date": str(prediction_target_date) if prediction_target_date is not None else None,
        "n_games": 0,
        "n_missing_snapshot": 0,
        "substituted_actual_starter": False,
        "production_equivalent": False,
        "diagnostic_only": False,
        "reason": None,
    }

    if mode not in config.SCHEDULE_BACKTEST_MODES:
        meta["reason"] = f"invalid_mode:{mode}"
        return pd.DataFrame(columns=AS_OF_SCHEDULE_COLUMNS), meta

    if mode == BACKTEST_MODE_MISSING:
        meta["reason"] = "explicitly_unavailable"
        meta["diagnostic_only"] = False
        return pd.DataFrame(columns=AS_OF_SCHEDULE_COLUMNS), meta

    if mode == BACKTEST_MODE_ACTUAL:
        if actual_starter_games is None or actual_starter_games.empty:
            meta["reason"] = "no_actual_starter_games"
            meta["diagnostic_only"] = True
            return pd.DataFrame(columns=AS_OF_SCHEDULE_COLUMNS), meta
        out = actual_starter_games.copy()
        if prediction_target_date is not None and "date" in out.columns:
            target = pd.Timestamp(prediction_target_date).normalize()
            out = out[pd.to_datetime(out["date"]).dt.normalize() == target]
        if game_pks is not None:
            pk_set = set(pd.to_numeric(pd.Series(list(game_pks)), errors="coerce").dropna().astype(int))
            out = out[out["game_pk"].isin(pk_set)]
        out = out.copy()
        out["schedule_backtest_mode"] = BACKTEST_MODE_ACTUAL
        out["schedule_source"] = SOURCE_ACTUAL_STARTER_DIAGNOSTIC
        out["starter_announcement_status"] = [
            starter_announcement_status(h, a)
            for h, a in zip(
                out.get("home_probable_pitcher_key_mlbam", pd.Series(dtype=float)),
                out.get("away_probable_pitcher_key_mlbam", pd.Series(dtype=float)),
            )
        ]
        if "schedule_snapshot_id" not in out.columns:
            out["schedule_snapshot_id"] = pd.NA
        if "schedule_captured_at_utc" not in out.columns:
            out["schedule_captured_at_utc"] = pd.NA
        meta["n_games"] = int(len(out))
        meta["diagnostic_only"] = True
        meta["production_equivalent"] = False
        meta["reason"] = "actual_starter_diagnostic"
        return out, meta

    # as_of_snapshot (primary)
    if prediction_timestamp is None:
        meta["reason"] = "missing_prediction_timestamp"
        return pd.DataFrame(columns=AS_OF_SCHEDULE_COLUMNS), meta

    snaps = select_as_of_snapshots(
        schedule_snapshots if schedule_snapshots is not None else empty_snapshot_frame(),
        prediction_timestamp,
        prediction_target_date=prediction_target_date,
        game_pks=game_pks,
    )
    scores = None
    if actual_starter_games is not None and not actual_starter_games.empty:
        # Scores only - never starters - for optional resolution join.
        score_cols = [c for c in ("game_pk", "home_score", "away_score", "status") if c in actual_starter_games.columns]
        scores = actual_starter_games[score_cols]
    out = as_of_snapshots_to_schedule_games(snaps, scores=scores)

    expected_pks = set()
    if game_pks is not None:
        expected_pks = set(pd.to_numeric(pd.Series(list(game_pks)), errors="coerce").dropna().astype(int))
    elif actual_starter_games is not None and not actual_starter_games.empty:
        scoped = actual_starter_games
        if prediction_target_date is not None and "date" in scoped.columns:
            target = pd.Timestamp(prediction_target_date).normalize()
            scoped = scoped[pd.to_datetime(scoped["date"]).dt.normalize() == target]
        expected_pks = set(pd.to_numeric(scoped["game_pk"], errors="coerce").dropna().astype(int))

    found = set(pd.to_numeric(out["game_pk"], errors="coerce").dropna().astype(int)) if not out.empty else set()
    missing = expected_pks - found
    meta["n_games"] = int(len(out))
    meta["n_missing_snapshot"] = int(len(missing))
    meta["missing_game_pks"] = sorted(int(x) for x in missing)
    meta["substituted_actual_starter"] = False
    meta["production_equivalent"] = True
    meta["diagnostic_only"] = False
    if out.empty:
        meta["reason"] = "no_as_of_snapshot_available"
    else:
        meta["reason"] = "as_of_snapshot"
    return out, meta


def assert_no_actual_starter_substitution(meta: dict) -> None:
    if meta.get("substituted_actual_starter"):
        raise AssertionError("actual starter was substituted into an as-of backtest")
    if meta.get("schedule_backtest_mode") == BACKTEST_MODE_AS_OF and meta.get("diagnostic_only"):
        raise AssertionError("as_of_snapshot mode incorrectly marked diagnostic_only")


# ---------------------------------------------------------------------------
# Starter-change statistics
# ---------------------------------------------------------------------------


def _pitcher_equal(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return int(a) == int(b)


def starter_change_statistics(
    as_of_schedule: pd.DataFrame,
    actual_starter_games: pd.DataFrame,
) -> dict:
    """Compare as-of probable starters to actual starters (diagnostic join)."""
    empty = {
        "n_compared_games": 0,
        "probable_matched_actual": 0,
        "probable_matched_actual_rate": float("nan"),
        "starter_changed_before_game": 0,
        "starter_changed_before_game_rate": float("nan"),
        "probable_starter_missing": 0,
        "probable_starter_missing_rate": float("nan"),
        "home_changed": 0,
        "away_changed": 0,
        "both_sides_changed": 0,
    }
    if as_of_schedule is None or as_of_schedule.empty or actual_starter_games is None or actual_starter_games.empty:
        return empty

    as_of = as_of_schedule.copy()
    actual = actual_starter_games.copy()
    actual = actual.rename(columns={
        "home_probable_pitcher_key_mlbam": "home_actual_pitcher_key_mlbam",
        "away_probable_pitcher_key_mlbam": "away_actual_pitcher_key_mlbam",
    })
    keep = ["game_pk", "home_actual_pitcher_key_mlbam", "away_actual_pitcher_key_mlbam"]
    merged = as_of.merge(actual[keep].drop_duplicates("game_pk"), on="game_pk", how="inner")
    if merged.empty:
        return empty

    matched = changed = missing = home_changed = away_changed = both_changed = 0
    for _, row in merged.iterrows():
        home_p = row.get("home_probable_pitcher_key_mlbam")
        away_p = row.get("away_probable_pitcher_key_mlbam")
        home_a = row.get("home_actual_pitcher_key_mlbam")
        away_a = row.get("away_actual_pitcher_key_mlbam")
        home_miss = pd.isna(home_p)
        away_miss = pd.isna(away_p)
        if home_miss or away_miss:
            missing += 1
        home_match = (not home_miss) and _pitcher_equal(home_p, home_a)
        away_match = (not away_miss) and _pitcher_equal(away_p, away_a)
        h_chg = (not home_miss) and (not pd.isna(home_a)) and (not _pitcher_equal(home_p, home_a))
        a_chg = (not away_miss) and (not pd.isna(away_a)) and (not _pitcher_equal(away_p, away_a))
        if h_chg:
            home_changed += 1
        if a_chg:
            away_changed += 1
        if h_chg and a_chg:
            both_changed += 1
        if home_match and away_match:
            matched += 1
        elif h_chg or a_chg:
            changed += 1

    n = int(len(merged))
    return {
        "n_compared_games": n,
        "probable_matched_actual": matched,
        "probable_matched_actual_rate": matched / n if n else float("nan"),
        "starter_changed_before_game": changed,
        "starter_changed_before_game_rate": changed / n if n else float("nan"),
        "probable_starter_missing": missing,
        "probable_starter_missing_rate": missing / n if n else float("nan"),
        "home_changed": home_changed,
        "away_changed": away_changed,
        "both_sides_changed": both_changed,
    }


def starter_change_probability_effects(
    as_of_probabilities: pd.DataFrame,
    actual_starter_probabilities: pd.DataFrame,
    *,
    probability_col: str = "home_win_probability",
    label_col: str | None = "Home_Won",
    change_flags: pd.DataFrame | None = None,
) -> dict:
    """Effect of starter changes on probability and accuracy.

    ``as_of_probabilities`` / ``actual_starter_probabilities`` must share
    ``game_pk``. Modes are compared, never merged into one score.
    """
    result = {
        "n_paired_games": 0,
        "mean_abs_probability_delta": float("nan"),
        "mean_probability_delta": float("nan"),
        "as_of_brier": float("nan"),
        "actual_starter_brier": float("nan"),
        "brier_delta_actual_minus_as_of": float("nan"),
        "changed_subset_mean_abs_delta": float("nan"),
        "matched_subset_mean_abs_delta": float("nan"),
        "modes_merged": False,
    }
    if as_of_probabilities is None or as_of_probabilities.empty:
        return result
    if actual_starter_probabilities is None or actual_starter_probabilities.empty:
        return result

    a = as_of_probabilities[["game_pk", probability_col] + (
        [label_col] if label_col and label_col in as_of_probabilities.columns else []
    )].copy()
    b = actual_starter_probabilities[["game_pk", probability_col]].copy()
    b = b.rename(columns={probability_col: f"{probability_col}_actual"})
    paired = a.merge(b, on="game_pk", how="inner")
    if paired.empty:
        return result

    p_asof = pd.to_numeric(paired[probability_col], errors="coerce")
    p_act = pd.to_numeric(paired[f"{probability_col}_actual"], errors="coerce")
    delta = p_act - p_asof
    result["n_paired_games"] = int(len(paired))
    result["mean_abs_probability_delta"] = float(delta.abs().mean())
    result["mean_probability_delta"] = float(delta.mean())

    if label_col and label_col in paired.columns:
        y = pd.to_numeric(paired[label_col], errors="coerce")
        mask = y.notna() & p_asof.notna() & p_act.notna()
        if mask.any():
            yv, pa, pb = y[mask], p_asof[mask], p_act[mask]
            brier_a = float(np.mean((pa - yv) ** 2))
            brier_b = float(np.mean((pb - yv) ** 2))
            result["as_of_brier"] = brier_a
            result["actual_starter_brier"] = brier_b
            result["brier_delta_actual_minus_as_of"] = brier_b - brier_a

    if change_flags is not None and not change_flags.empty and "game_pk" in change_flags.columns:
        flags = change_flags[["game_pk", "starter_changed"]].drop_duplicates("game_pk")
        with_flags = paired.merge(flags, on="game_pk", how="left")
        changed = with_flags["starter_changed"] == True  # noqa: E712
        matched = with_flags["starter_changed"] == False  # noqa: E712
        d = pd.to_numeric(with_flags[f"{probability_col}_actual"], errors="coerce") - pd.to_numeric(
            with_flags[probability_col], errors="coerce"
        )
        if changed.any():
            result["changed_subset_mean_abs_delta"] = float(d[changed].abs().mean())
        if matched.any():
            result["matched_subset_mean_abs_delta"] = float(d[matched].abs().mean())

    return result


def build_starter_change_flag_frame(
    as_of_schedule: pd.DataFrame,
    actual_starter_games: pd.DataFrame,
) -> pd.DataFrame:
    stats_rows = []
    if as_of_schedule is None or as_of_schedule.empty or actual_starter_games is None:
        return pd.DataFrame(columns=["game_pk", "starter_changed", "probable_missing"])
    actual = actual_starter_games.rename(columns={
        "home_probable_pitcher_key_mlbam": "home_actual_pitcher_key_mlbam",
        "away_probable_pitcher_key_mlbam": "away_actual_pitcher_key_mlbam",
    })
    merged = as_of_schedule.merge(
        actual[["game_pk", "home_actual_pitcher_key_mlbam", "away_actual_pitcher_key_mlbam"]].drop_duplicates("game_pk"),
        on="game_pk",
        how="left",
    )
    for _, row in merged.iterrows():
        home_p, away_p = row.get("home_probable_pitcher_key_mlbam"), row.get("away_probable_pitcher_key_mlbam")
        home_a, away_a = row.get("home_actual_pitcher_key_mlbam"), row.get("away_actual_pitcher_key_mlbam")
        missing = pd.isna(home_p) or pd.isna(away_p)
        changed = (
            (pd.notna(home_p) and pd.notna(home_a) and not _pitcher_equal(home_p, home_a))
            or (pd.notna(away_p) and pd.notna(away_a) and not _pitcher_equal(away_p, away_a))
        )
        stats_rows.append({
            "game_pk": row["game_pk"],
            "starter_changed": bool(changed),
            "probable_missing": bool(missing),
        })
    return pd.DataFrame(stats_rows)


def production_equivalent_sample_report(
    as_of_n_games: int,
    *,
    actual_starter_n_games: int = 0,
    missing_snapshot_n_games: int = 0,
    min_games: int | None = None,
) -> dict:
    """Honest labeling of production-equivalent sample size.

    Never claims sufficiency by merging actual-starter diagnostics.
    """
    min_games = int(config.SCHEDULE_AS_OF_MIN_PRODUCTION_GAMES if min_games is None else min_games)
    sufficient = int(as_of_n_games) >= min_games
    return {
        "as_of_snapshot_n_games": int(as_of_n_games),
        "actual_starter_diagnostic_n_games": int(actual_starter_n_games),
        "missing_snapshot_n_games": int(missing_snapshot_n_games),
        "production_equivalent_min_games": min_games,
        "production_equivalent_sample_sufficient": sufficient,
        "modes_merged": False,
        "label": (
            "production_equivalent"
            if sufficient
            else "production_equivalent_sample_insufficient_do_not_merge_actual_starter"
        ),
    }


def default_morning_prediction_timestamp(target_date) -> pd.Timestamp:
    """Conservative morning prediction time in UTC for ``target_date``.

    Uses config.SCHEDULE_BACKTEST_MORNING_HOUR_UTC on the target calendar
    date (UTC). Callers with a real captured pipeline time should pass that
    instead.
    """
    day = pd.Timestamp(target_date).normalize()
    hour = int(config.SCHEDULE_BACKTEST_MORNING_HOUR_UTC)
    return pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hour, tz="UTC")
