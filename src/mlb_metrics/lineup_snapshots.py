"""Confirmed-lineup snapshots: fetch, persist, and apply to hitter inference.

Output adapter schema (one row per player per game)::

    game_pk, team, opponent, key_mlbam, batting_order, is_confirmed_starter,
    lineup_status, fetched_at_utc, game_datetime, source

Immutable audit snapshots are appended; a separate latest view keeps only
the newest pregame row per (game_pk, team, key_mlbam).

API parsing is gated by ``config.LINEUP_API_SCHEMA_CONFIRMED``. Until Stage A
confirms the live Stats API field paths (see
``scripts/debug_statsapi_lineups.py``), fetch returns empty and callers rely
on injected / fixture snapshot frames. Tests use **normalized** snapshot
fixtures (this module's output schema), not assumed raw API JSON.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mlb_metrics import config, schedule

SNAPSHOT_COLUMNS = [
    "game_pk", "team", "opponent", "key_mlbam", "batting_order",
    "is_confirmed_starter", "lineup_status", "fetched_at_utc",
    "game_datetime", "source", "snapshot_id", "as_of_date",
]

LINEUP_STATUS_CONFIRMED = "confirmed"
LINEUP_STATUS_UNCONFIRMED = "unconfirmed"
LINEUP_STATUS_SCRATCHED = "scratched"
SOURCE_STATSAPI = "statsapi_schedule_lineups"
SOURCE_FIXTURE = "fixture"
SOURCE_MANUAL = "manual"

STARTER_MAX_SLOT = 9


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_snapshot_id(rows: pd.DataFrame, fetched_at_utc: str) -> str:
    payload = rows.sort_values(
        [c for c in ("game_pk", "team", "key_mlbam") if c in rows.columns]
    ).to_csv(index=False)
    digest = hashlib.sha1(f"{fetched_at_utc}\n{payload}".encode("utf-8")).hexdigest()[:12]
    return f"lineup_{digest}"


def empty_snapshot_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def normalize_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame into the adapter schema (missing cols -> NA)."""
    out = df.copy() if df is not None else empty_snapshot_frame()
    for col in SNAPSHOT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if not out.empty:
        out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
        out["key_mlbam"] = pd.to_numeric(out["key_mlbam"], errors="coerce")
        out["batting_order"] = pd.to_numeric(out["batting_order"], errors="coerce")
        out["is_confirmed_starter"] = out["is_confirmed_starter"].astype("boolean")
    return out[SNAPSHOT_COLUMNS]


# ---------------------------------------------------------------------------
# Provisional Stats API parsing (disabled until Stage A confirms)
# ---------------------------------------------------------------------------


def parse_statsapi_schedule_lineups(raw: dict, *, fetched_at_utc: str | None = None) -> pd.DataFrame:
    """Parse a schedule+lineups hydrate response into snapshot rows.

    Field paths follow the common MLB Stats API ``lineups`` hydrate shape
    and are still **provisional** until
    ``config.LINEUP_API_SCHEMA_CONFIRMED`` is flipped after Stage A. When
    confirmation is false, callers should not use this for live traffic.
    """
    fetched_at_utc = fetched_at_utc or utc_now_iso()
    rows = []
    for day in raw.get("dates") or []:
        as_of = day.get("date")
        for game in day.get("games") or []:
            game_pk = game.get("gamePk")
            game_datetime = game.get("gameDate")
            teams = game.get("teams") or {}
            home = (teams.get("home") or {}).get("team") or {}
            away = (teams.get("away") or {}).get("team") or {}
            home_abbrev = schedule.TEAM_ID_TO_ABBREV.get(home.get("id"))
            away_abbrev = schedule.TEAM_ID_TO_ABBREV.get(away.get("id"))
            if not game_pk or not home_abbrev or not away_abbrev:
                continue

            lineups = game.get("lineups") or {}
            # Common shapes: homePlayers/awayPlayers OR nested under team side.
            home_players = lineups.get("homePlayers") or lineups.get("home") or []
            away_players = lineups.get("awayPlayers") or lineups.get("away") or []
            if isinstance(home_players, dict):
                home_players = home_players.get("players") or []
            if isinstance(away_players, dict):
                away_players = away_players.get("players") or []

            confirmed = bool(home_players or away_players)
            status = LINEUP_STATUS_CONFIRMED if confirmed else LINEUP_STATUS_UNCONFIRMED

            def _emit(players, team, opponent):
                for p in players or []:
                    person = p.get("id") or (p.get("person") or {}).get("id")
                    if person is None:
                        continue
                    raw_order = p.get("battingOrder")
                    if raw_order is None:
                        raw_order = p.get("batting_order")
                    # MLB often encodes order as 100,200,...,900
                    order = None
                    if raw_order is not None:
                        try:
                            order_i = int(raw_order)
                            order = order_i // 100 if order_i >= 100 else order_i
                        except (TypeError, ValueError):
                            order = None
                    is_starter = bool(order is not None and 1 <= int(order) <= STARTER_MAX_SLOT)
                    rows.append({
                        "game_pk": game_pk,
                        "team": team,
                        "opponent": opponent,
                        "key_mlbam": int(person),
                        "batting_order": order if is_starter else pd.NA,
                        "is_confirmed_starter": is_starter and status == LINEUP_STATUS_CONFIRMED,
                        "lineup_status": status if is_starter else (
                            LINEUP_STATUS_CONFIRMED if confirmed else LINEUP_STATUS_UNCONFIRMED
                        ),
                        "fetched_at_utc": fetched_at_utc,
                        "game_datetime": game_datetime,
                        "source": SOURCE_STATSAPI,
                        "as_of_date": as_of,
                    })

            _emit(home_players, home_abbrev, away_abbrev)
            _emit(away_players, away_abbrev, home_abbrev)

            # Unconfirmed games: still emit a marker row? Prefer empty — callers
            # keep historical appearance model when no confirmed starters exist.

    if not rows:
        return empty_snapshot_frame()
    frame = pd.DataFrame(rows)
    frame["snapshot_id"] = make_snapshot_id(frame, fetched_at_utc)
    return normalize_snapshot_frame(frame)


def fetch_lineup_snapshots(date, *, force: bool = False) -> pd.DataFrame:
    """Fetch today's lineup snapshots from Stats API, or empty if unconfirmed schema."""
    if not config.LINEUP_API_SCHEMA_CONFIRMED and not force:
        return empty_snapshot_frame()
    import statsapi

    raw = statsapi.get(
        "schedule",
        {"sportId": 1, "date": str(date), "hydrate": "lineups,probablePitcher,team"},
    )
    return parse_statsapi_schedule_lineups(raw)


# ---------------------------------------------------------------------------
# Persistence: immutable audit + latest view
# ---------------------------------------------------------------------------


def append_snapshot_audit(snapshots: pd.DataFrame, audit_path: str | None = None) -> pd.DataFrame:
    """Append-only immutable audit log of every snapshot fetch."""
    path = audit_path or config.LINEUP_SNAPSHOT_AUDIT_PATH
    frame = normalize_snapshot_frame(snapshots)
    if frame.empty:
        if os.path.exists(path):
            return pd.read_csv(path)
        return frame
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame
    # Never rewrite history — keep every fetch. Dedupe only exact duplicates.
    combined = combined.drop_duplicates(keep="first")
    combined.to_csv(path, index=False)
    return combined


def publish_latest_snapshots(snapshots: pd.DataFrame, latest_path: str | None = None) -> pd.DataFrame:
    """Upsert latest pregame snapshot per (game_pk, team, key_mlbam)."""
    path = latest_path or config.LINEUP_SNAPSHOT_LATEST_PATH
    frame = normalize_snapshot_frame(snapshots)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        existing = normalize_snapshot_frame(pd.read_csv(path))
    else:
        existing = empty_snapshot_frame()
    if frame.empty:
        return existing
    combined = pd.concat([existing, frame], ignore_index=True)
    # Latest by fetched_at_utc
    combined["_ts"] = pd.to_datetime(combined["fetched_at_utc"], utc=True, errors="coerce")
    combined = combined.sort_values("_ts")
    latest = combined.drop_duplicates(subset=["game_pk", "team", "key_mlbam"], keep="last")
    latest = latest.drop(columns=["_ts"], errors="ignore")
    latest.to_csv(path, index=False)
    return normalize_snapshot_frame(latest)


def persist_snapshots(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = append_snapshot_audit(snapshots)
    latest = publish_latest_snapshots(snapshots)
    return audit, latest


def snapshot_content_fingerprint(snapshots: pd.DataFrame) -> str:
    """Fingerprint of lineup content (ignores fetched_at / snapshot_id)."""
    cols = ["game_pk", "team", "key_mlbam", "batting_order", "is_confirmed_starter", "lineup_status"]
    frame = normalize_snapshot_frame(snapshots)
    if frame.empty:
        return "empty"
    subset = frame[cols].sort_values(cols).fillna(-1)
    return hashlib.sha1(subset.to_csv(index=False).encode("utf-8")).hexdigest()


def changed_game_pks(previous: pd.DataFrame, current: pd.DataFrame) -> set:
    """Game PKs whose confirmed lineup content changed (or are new)."""
    prev = normalize_snapshot_frame(previous)
    cur = normalize_snapshot_frame(current)
    if cur.empty:
        return set()
    if prev.empty:
        return set(pd.to_numeric(cur["game_pk"], errors="coerce").dropna().astype(int))

    changed = set()
    all_games = set(prev["game_pk"].dropna().astype(int)) | set(cur["game_pk"].dropna().astype(int))
    for gpk in all_games:
        a = prev[prev["game_pk"] == gpk]
        b = cur[cur["game_pk"] == gpk]
        if snapshot_content_fingerprint(a) != snapshot_content_fingerprint(b):
            changed.add(int(gpk))
    return changed


# ---------------------------------------------------------------------------
# Apply confirmed lineup to hitter inference pool
# ---------------------------------------------------------------------------


def confirmed_appearance_probability(
    is_confirmed_starter: bool,
    *,
    scratch_risk: float | None = None,
) -> float:
    """P(Appear) for a confirmed starter.

    Scratch-risk treatment: confirmed starters are not probability 1.0 —
    a small documented scratch risk remains until first pitch
    (``config.LINEUP_CONFIRMED_SCRATCH_RISK``, default 2%).
    """
    risk = float(config.LINEUP_CONFIRMED_SCRATCH_RISK if scratch_risk is None else scratch_risk)
    risk = float(np.clip(risk, 0.0, 0.5))
    if not is_confirmed_starter:
        return float("nan")
    return float(np.clip(1.0 - risk, 0.0, 1.0))


def annotate_scratches(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Mark previously confirmed starters missing from the new confirmed set.

    Returns ``current`` rows plus scratch marker rows (``lineup_status=
    scratched``, ``is_confirmed_starter=False``) for late scratches.
    """
    prev = normalize_snapshot_frame(previous)
    cur = normalize_snapshot_frame(current)
    if prev.empty:
        return cur
    prev_starters = prev[prev["is_confirmed_starter"] == True]  # noqa: E712
    cur_starters = cur[cur["is_confirmed_starter"] == True] if not cur.empty else empty_snapshot_frame()
    if prev_starters.empty:
        return cur

    keys = ["game_pk", "team", "key_mlbam"]
    merged = prev_starters.merge(
        cur_starters[keys].drop_duplicates() if not cur_starters.empty else pd.DataFrame(columns=keys),
        on=keys,
        how="left",
        indicator=True,
    )
    missing = merged[merged["_merge"] == "left_only"]
    if missing.empty:
        return cur

    scratched = missing.drop(columns=["_merge"], errors="ignore").copy()
    scratched["is_confirmed_starter"] = False
    scratched["lineup_status"] = LINEUP_STATUS_SCRATCHED
    scratched["batting_order"] = pd.NA
    scratched["fetched_at_utc"] = utc_now_iso() if cur.empty else cur["fetched_at_utc"].iloc[0]
    scratched["source"] = scratched.get("source", SOURCE_MANUAL)
    if "snapshot_id" in scratched.columns:
        scratched["snapshot_id"] = make_snapshot_id(
            pd.concat([cur, scratched], ignore_index=True) if not cur.empty else scratched,
            str(scratched["fetched_at_utc"].iloc[0]),
        )
    return normalize_snapshot_frame(pd.concat([cur, scratched], ignore_index=True))


def apply_confirmed_lineup_to_pool(
    pick_pool: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    league_priors: dict | pd.Series | None = None,
) -> pd.DataFrame:
    """Merge confirmed lineup into the hitter pick pool.

    - Confirmed starters get ``P_Appear`` via scratch-risk treatment and
      ``batting_order`` / ``lineup_status`` from the snapshot.
    - Games without confirmed snapshots keep historical appearance features.
    - Confirmed players missing from the pool are added with league priors
      (never dropped).
    - Scratched players keep pool membership but get near-zero ``P_Appear``.
    """
    snaps = normalize_snapshot_frame(snapshots)
    if pick_pool is None or pick_pool.empty:
        base = pd.DataFrame()
    else:
        base = pick_pool.copy()

    if snaps.empty:
        if not base.empty and "lineup_status" not in base.columns:
            base["lineup_status"] = LINEUP_STATUS_UNCONFIRMED
        return base

    confirmed = snaps[snaps["is_confirmed_starter"] == True].copy()  # noqa: E712
    scratched = snaps[snaps["lineup_status"] == LINEUP_STATUS_SCRATCHED].copy()

    if confirmed.empty and scratched.empty:
        if not base.empty and "lineup_status" not in base.columns:
            base["lineup_status"] = LINEUP_STATUS_UNCONFIRMED
        return base

    # Add missing confirmed players with priors
    if not confirmed.empty and not base.empty and {"game_pk", "key_mlbam"}.issubset(base.columns):
        from mlb_metrics import hitter_training_data

        confirmed_for_add = confirmed.copy()
        if "date" not in confirmed_for_add.columns and "as_of_date" in confirmed_for_add.columns:
            confirmed_for_add["date"] = confirmed_for_add["as_of_date"]
        if "date" not in base.columns and "date" in confirmed_for_add.columns:
            # add_confirmed_lineup_candidates requires date on keys; live
            # pick pools often only have game_pk.
            base = base.copy()
            base["date"] = pd.to_datetime(confirmed_for_add["date"].iloc[0])
        priors = league_priors
        if priors is None:
            priors = {}
            for col in ("WAVE", "probability", "Game_Hit_Probability", "Park_Factor"):
                if col in base.columns:
                    priors[col] = float(pd.to_numeric(base[col], errors="coerce").median())
            if "Park_Factor" not in priors:
                priors["Park_Factor"] = 1.0
        if "date" not in confirmed_for_add.columns and "date" in base.columns:
            # Map date from any existing pool row for the same game_pk.
            date_by_game = base[["game_pk", "date"]].drop_duplicates()
            confirmed_for_add = confirmed_for_add.merge(date_by_game, on="game_pk", how="left")
        if "date" not in confirmed_for_add.columns:
            confirmed_for_add["date"] = pd.Timestamp("today").normalize()
        confirmed_for_add["date"] = pd.to_datetime(confirmed_for_add["date"])
        if "date" in base.columns:
            base["date"] = pd.to_datetime(base["date"])
        base = hitter_training_data.add_confirmed_lineup_candidates(base, confirmed_for_add, priors)
    elif not confirmed.empty and base.empty:
        base = confirmed.copy()
        for col, val in (league_priors or {}).items():
            if col not in base.columns:
                base[col] = val

    if base.empty:
        return base

    # Merge confirmed batting order / status onto pool
    if not confirmed.empty:
        merge_cols = ["game_pk", "key_mlbam", "batting_order", "is_confirmed_starter", "lineup_status"]
        merge_cols = [c for c in merge_cols if c in confirmed.columns]
        overlay = confirmed[merge_cols].drop_duplicates(["game_pk", "key_mlbam"])
        for col in ("batting_order", "is_confirmed_starter", "lineup_status"):
            if col in base.columns:
                base = base.drop(columns=[col], errors="ignore")
        keys = [c for c in ("game_pk", "key_mlbam") if c in base.columns]
        base = base.merge(overlay, on=keys, how="left", suffixes=("", "_lineup"))

    # Scratches: override status / appearance for previously confirmed players
    if not scratched.empty and {"game_pk", "key_mlbam"}.issubset(base.columns):
        scratch_keys = scratched[["game_pk", "key_mlbam"]].drop_duplicates()
        scratch_keys = scratch_keys.assign(_scratched=True)
        base = base.merge(scratch_keys, on=["game_pk", "key_mlbam"], how="left")
        mask = base["_scratched"].eq(True)
        if "lineup_status" not in base.columns:
            base["lineup_status"] = LINEUP_STATUS_UNCONFIRMED
        base.loc[mask, "lineup_status"] = LINEUP_STATUS_SCRATCHED
        if "is_confirmed_starter" not in base.columns:
            base["is_confirmed_starter"] = False
        base.loc[mask, "is_confirmed_starter"] = False
        base = base.drop(columns=["_scratched"], errors="ignore")

    # Appearance probability for confirmed starters / scratches
    appear = []
    for _, row in base.iterrows():
        status = row.get("lineup_status")
        if status == LINEUP_STATUS_SCRATCHED:
            appear.append(0.0)
        elif bool(row.get("is_confirmed_starter")) is True:
            appear.append(confirmed_appearance_probability(True))
        else:
            existing = row.get("P_Appear", np.nan)
            appear.append(existing if pd.notna(existing) else np.nan)
    base["P_Appear"] = appear

    # Confirmed batting order feeds opportunity features
    if "batting_order" in base.columns:
        base["confirmed_batting_order"] = base["batting_order"]
        mask = base.get("is_confirmed_starter", pd.Series(False, index=base.index)) == True  # noqa: E712
        if "avg_batting_order" in base.columns:
            base.loc[mask, "avg_batting_order"] = base.loc[mask, "batting_order"]
        else:
            base["avg_batting_order"] = base["batting_order"]

    if "lineup_status" in base.columns:
        base["lineup_status"] = base["lineup_status"].fillna(LINEUP_STATUS_UNCONFIRMED)
    else:
        base["lineup_status"] = LINEUP_STATUS_UNCONFIRMED
    return base


def games_in_upcoming_window(
    schedule_df: pd.DataFrame,
    *,
    now_utc: pd.Timestamp | None = None,
    window_hours: float | None = None,
) -> pd.DataFrame:
    """Unstarted games whose start falls within the upcoming window."""
    if schedule_df is None or schedule_df.empty or "game_datetime" not in schedule_df.columns:
        return schedule_df.iloc[0:0] if schedule_df is not None else pd.DataFrame()
    now = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    window_hours = float(config.LINEUP_LOCK_WINDOW_HOURS if window_hours is None else window_hours)
    horizon = now + pd.Timedelta(hours=window_hours)

    df = schedule_df.copy()
    dt = pd.to_datetime(df["game_datetime"], utc=True, errors="coerce")
    mask = dt.notna() & (dt > now) & (dt <= horizon)
    return df.loc[mask].copy()


def game_has_started(game_datetime, *, now_utc: pd.Timestamp | None = None) -> bool:
    if game_datetime is None or pd.isna(game_datetime):
        return False
    now = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    dt = pd.to_datetime(game_datetime, utc=True, errors="coerce")
    if pd.isna(dt):
        return False
    return bool(dt <= now)


def filter_snapshots_to_unstarted(
    snapshots: pd.DataFrame,
    *,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Drop snapshot rows for games that have already started."""
    frame = normalize_snapshot_frame(snapshots)
    if frame.empty:
        return frame
    keep = []
    for _, row in frame.iterrows():
        keep.append(not game_has_started(row.get("game_datetime"), now_utc=now_utc))
    return frame.loc[keep].reset_index(drop=True)
