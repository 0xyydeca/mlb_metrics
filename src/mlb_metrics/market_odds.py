"""Real ESPN moneyline odds ingestion and timestamped snapshot persistence.

Provider rows carry sportsbook, capture time, game start, moneylines, and
de-vigged home probability. Snapshots are append-only. Closing line means
the latest valid pregame snapshot strictly before first pitch — not the
morning Daily Update capture.

Matching ESPN events to MLB ``game_pk`` uses date, start time, slate
ordering, and a persisted provider-event map. Ambiguous doubleheaders are
rejected and logged (``source_status=ambiguous_match``), never guessed.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import requests

from mlb_metrics import config

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"

# The real crosswalk slice 1's dispatch found (GitHub Actions run
# 32516808493, 2026-08-21): ESPN's "ARI"/"CHW" vs. this project's own
# schedule.TEAM_ID_TO_ABBREV "AZ"/"CWS". All other 28 real abbreviations
# ESPN returned matched schedule.TEAM_ID_TO_ABBREV's values exactly.
ESPN_TEAM_ABBREV_FIXUPS = {"ARI": "AZ", "CHW": "CWS"}

PROVIDER_OUTPUT_COLUMNS = [
    "provider_event_id",
    "sportsbook",
    "captured_at_utc",
    "game_datetime",
    "date",
    "home_team",
    "away_team",
    "home_moneyline",
    "away_moneyline",
    "market_home_win_probability",
    "source_status",
]

ODDS_SNAPSHOT_COLUMNS = PROVIDER_OUTPUT_COLUMNS + [
    "game_pk",
    "match_method",
    "snapshot_id",
    "snapshot_role",
]

SNAPSHOT_ROLE_OPENING = "opening"
SNAPSHOT_ROLE_MORNING = "morning"
SNAPSHOT_ROLE_LINEUP_LOCK = "lineup_lock"
SNAPSHOT_ROLE_CLOSING = "closing"
SNAPSHOT_ROLE_RAW = "raw"
# Continuous intraday captures (every ~30 min). Not a prediction-time prior
# role — used for closing selection and audit, never relabeled as morning.
SNAPSHOT_ROLE_INTRADAY = "intraday"

SOURCE_OK = "ok"
SOURCE_MISSING_ODDS = "missing_odds"
SOURCE_AMBIGUOUS_MATCH = "ambiguous_match"
SOURCE_UNMATCHED = "unmatched"
SOURCE_POST_START = "post_start"
SOURCE_HISTORICAL_POSTGAME = "historical_postgame_fetch"

PREDICTION_TIME_ROLES = frozenset({
    SNAPSHOT_ROLE_OPENING,
    SNAPSHOT_ROLE_MORNING,
    SNAPSHOT_ROLE_LINEUP_LOCK,
})
INVALID_PREDICTION_TIME_STATUSES = frozenset({
    SOURCE_MISSING_ODDS,
    SOURCE_AMBIGUOUS_MATCH,
    SOURCE_UNMATCHED,
    SOURCE_POST_START,
    SOURCE_HISTORICAL_POSTGAME,
})

# Backward-compatible columns previously returned by fetch_market_home_win_probabilities.
LEGACY_MARKET_COLUMNS = [
    "home_team", "away_team", "market_home_win_probability", "market_provider",
    "home_moneyline", "away_moneyline",
]


def moneyline_to_implied_probability(moneyline: float) -> float:
    """Standard American-odds-to-implied-probability conversion."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    return 100 / (moneyline + 100)


def devig(home_implied: float, away_implied: float) -> float:
    """Standard proportional de-vig - normalizes both sides' implied
    probabilities to sum to 1.0, removing the bookmaker's overround."""
    return home_implied / (home_implied + away_implied)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _apply_abbrev_fixup(abbrev):
    return ESPN_TEAM_ABBREV_FIXUPS.get(abbrev, abbrev)


def fetch_scoreboard(date: str) -> dict:
    resp = requests.get(SCOREBOARD_URL, params={"dates": date}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_summary(event_id: str) -> dict:
    resp = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_pickcenter_row(summary_json: dict):
    """Pulls the odds row for config.MARKET_ODDS_PREFERRED_PROVIDER out of
    a summary response's ``pickcenter`` array, falling back to the first
    available row. Returns (provider_name, home_moneyline, away_moneyline)
    or None if pickcenter is missing/empty."""
    pickcenter = summary_json.get("pickcenter") or []
    if not pickcenter:
        return None
    row = next(
        (r for r in pickcenter if (r.get("provider") or {}).get("name") == config.MARKET_ODDS_PREFERRED_PROVIDER),
        pickcenter[0],
    )
    away = row.get("awayTeamOdds") or {}
    home = row.get("homeTeamOdds") or {}
    provider_name = (row.get("provider") or {}).get("name")
    return provider_name, home.get("moneyLine"), away.get("moneyLine")


def _event_game_datetime(event: dict) -> str | None:
    """ESPN scoreboard event start time (ISO). Prefer competition date."""
    for competition in event.get("competitions") or []:
        dt = competition.get("date") or competition.get("startDate")
        if dt:
            return dt
    return event.get("date") or event.get("startDate")


def _extract_provider_row(
    event: dict,
    summary_json: dict,
    *,
    captured_at_utc: str | None = None,
    as_of_date=None,
) -> dict | None:
    """Combine one scoreboard event + summary into the provider output schema."""
    home_abbrev = None
    away_abbrev = None
    for competition in event.get("competitions", []):
        for competitor in competition.get("competitors", []):
            team = competitor.get("team") or {}
            abbrev = _apply_abbrev_fixup(team.get("abbreviation"))
            if competitor.get("homeAway") == "home":
                home_abbrev = abbrev
            elif competitor.get("homeAway") == "away":
                away_abbrev = abbrev

    event_id = event.get("id")
    game_datetime = _event_game_datetime(event)
    captured = captured_at_utc or utc_now_iso()
    date_val = as_of_date
    if date_val is None and game_datetime:
        try:
            date_val = pd.Timestamp(game_datetime, tz="UTC").tz_convert(None).normalize()
        except Exception:
            date_val = pd.NaT
    elif date_val is not None:
        date_val = pd.Timestamp(date_val)

    if home_abbrev is None or away_abbrev is None:
        return None

    parsed = _parse_pickcenter_row(summary_json)
    if parsed is None:
        return {
            "provider_event_id": str(event_id) if event_id is not None else pd.NA,
            "sportsbook": pd.NA,
            "captured_at_utc": captured,
            "game_datetime": game_datetime,
            "date": date_val,
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_moneyline": pd.NA,
            "away_moneyline": pd.NA,
            "market_home_win_probability": pd.NA,
            "source_status": SOURCE_MISSING_ODDS,
        }
    provider_name, home_ml, away_ml = parsed
    if home_ml is None or away_ml is None:
        return {
            "provider_event_id": str(event_id) if event_id is not None else pd.NA,
            "sportsbook": provider_name,
            "captured_at_utc": captured,
            "game_datetime": game_datetime,
            "date": date_val,
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_moneyline": home_ml if home_ml is not None else pd.NA,
            "away_moneyline": away_ml if away_ml is not None else pd.NA,
            "market_home_win_probability": pd.NA,
            "source_status": SOURCE_MISSING_ODDS,
        }

    home_implied = moneyline_to_implied_probability(home_ml)
    away_implied = moneyline_to_implied_probability(away_ml)
    return {
        "provider_event_id": str(event_id),
        "sportsbook": provider_name,
        "captured_at_utc": captured,
        "game_datetime": game_datetime,
        "date": date_val,
        "home_team": home_abbrev,
        "away_team": away_abbrev,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "market_home_win_probability": devig(home_implied, away_implied),
        "source_status": SOURCE_OK,
    }


def _extract_market_row(event: dict, summary_json: dict):
    """Legacy helper used by existing unit tests.

    Returns the older compact dict (``market_provider`` naming) or None when
    odds are missing. Prefer ``_extract_provider_row`` for new code.
    """
    row = _extract_provider_row(event, summary_json)
    if row is None or row.get("source_status") != SOURCE_OK:
        return None
    return {
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "market_home_win_probability": row["market_home_win_probability"],
        "market_provider": row["sportsbook"],
        "home_moneyline": row["home_moneyline"],
        "away_moneyline": row["away_moneyline"],
    }


def empty_provider_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PROVIDER_OUTPUT_COLUMNS)


def empty_snapshot_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ODDS_SNAPSHOT_COLUMNS)


def make_snapshot_id(row: dict | pd.Series) -> str:
    payload = "|".join(
        str(row.get(k))
        for k in (
            "provider_event_id", "sportsbook", "captured_at_utc",
            "home_moneyline", "away_moneyline", "game_pk",
        )
    )
    return "odds_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def normalize_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if df is not None else empty_snapshot_frame()
    for col in ODDS_SNAPSHOT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if not out.empty:
        out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
        out["home_moneyline"] = pd.to_numeric(out["home_moneyline"], errors="coerce")
        out["away_moneyline"] = pd.to_numeric(out["away_moneyline"], errors="coerce")
        out["market_home_win_probability"] = pd.to_numeric(
            out["market_home_win_probability"], errors="coerce"
        )
    return out[ODDS_SNAPSHOT_COLUMNS]


# ---------------------------------------------------------------------------
# Safe ESPN event ↔ MLB game_pk matching
# ---------------------------------------------------------------------------


def load_event_map(path: str | None = None) -> pd.DataFrame:
    path = path or config.MARKET_ODDS_EVENT_MAP_PATH
    cols = ["provider_event_id", "game_pk", "home_team", "away_team", "date", "match_method"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    frame = pd.read_csv(path)
    for col in cols:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame["game_pk"] = pd.to_numeric(frame["game_pk"], errors="coerce")
    return frame[cols]


def save_event_map(mapping: pd.DataFrame, path: str | None = None) -> None:
    path = path or config.MARKET_ODDS_EVENT_MAP_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame = mapping.copy()
    if os.path.exists(path):
        existing = load_event_map(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.drop_duplicates(subset=["provider_event_id"], keep="last")
    frame.to_csv(path, index=False)


def _to_utc_ts(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def match_provider_events_to_games(
    provider_rows: pd.DataFrame,
    schedule_games: pd.DataFrame,
    *,
    event_map: pd.DataFrame | None = None,
    time_tolerance_minutes: float | None = None,
    ambiguous_log: list | None = None,
) -> pd.DataFrame:
    """Attach ``game_pk`` / ``match_method`` / ``source_status`` to provider rows.

    Ambiguous matches are rejected (``source_status=ambiguous_match``,
    ``game_pk`` null) and optionally appended to ``ambiguous_log``.
    """
    rows = provider_rows.copy() if provider_rows is not None else empty_provider_frame()
    if rows.empty:
        return normalize_snapshot_frame(
            rows.assign(
                game_pk=pd.NA, match_method=pd.NA,
                snapshot_id=pd.NA, snapshot_role=SNAPSHOT_ROLE_RAW,
            )
        )

    sched = schedule_games.copy() if schedule_games is not None else pd.DataFrame()
    tol = float(
        config.MARKET_ODDS_MATCH_TIME_TOLERANCE_MINUTES
        if time_tolerance_minutes is None
        else time_tolerance_minutes
    )
    event_map = event_map if event_map is not None else load_event_map()
    map_by_event = {}
    if not event_map.empty:
        for _, m in event_map.iterrows():
            if pd.notna(m.get("provider_event_id")) and pd.notna(m.get("game_pk")):
                map_by_event[str(m["provider_event_id"])] = (int(m["game_pk"]), "provider_map")

    if not sched.empty:
        sched = sched.copy()
        sched["game_pk"] = pd.to_numeric(sched["game_pk"], errors="coerce")
        sched["_start"] = (
            sched["game_datetime"].map(_to_utc_ts) if "game_datetime" in sched.columns else pd.NaT
        )
        if "date" in sched.columns:
            sched["date"] = pd.to_datetime(sched["date"]).dt.normalize()

    out_rows = []
    for _, prow in rows.iterrows():
        record = prow.to_dict()
        event_id = str(record.get("provider_event_id")) if pd.notna(record.get("provider_event_id")) else None
        status = record.get("source_status") or SOURCE_OK

        if event_id and event_id in map_by_event:
            gpk, method = map_by_event[event_id]
            record["game_pk"] = gpk
            record["match_method"] = method
            record["source_status"] = status if status != SOURCE_OK else SOURCE_OK
            out_rows.append(record)
            continue

        if sched.empty or not {"home_team", "away_team"}.issubset(sched.columns):
            record["game_pk"] = pd.NA
            record["match_method"] = pd.NA
            record["source_status"] = SOURCE_UNMATCHED if status == SOURCE_OK else status
            out_rows.append(record)
            continue

        candidates = sched[
            (sched["home_team"] == record["home_team"])
            & (sched["away_team"] == record["away_team"])
        ]
        if "date" in candidates.columns and pd.notna(record.get("date")):
            day = pd.Timestamp(record["date"]).normalize()
            candidates = candidates[candidates["date"] == day]

        if len(candidates) == 0:
            record["game_pk"] = pd.NA
            record["match_method"] = pd.NA
            record["source_status"] = SOURCE_UNMATCHED if status == SOURCE_OK else status
            out_rows.append(record)
            continue

        if len(candidates) == 1:
            record["game_pk"] = int(candidates.iloc[0]["game_pk"])
            record["match_method"] = "unique_matchup"
            out_rows.append(record)
            continue

        # Doubleheader / multi-game same matchup: use start time.
        event_start = _to_utc_ts(record.get("game_datetime"))
        if pd.notna(event_start) and candidates["_start"].notna().any():
            deltas = (candidates["_start"] - event_start).abs()
            within = candidates[deltas <= pd.Timedelta(minutes=tol)]
            if len(within) == 1:
                record["game_pk"] = int(within.iloc[0]["game_pk"])
                record["match_method"] = "start_time"
                out_rows.append(record)
                continue
            if len(within) > 1:
                if ambiguous_log is not None:
                    ambiguous_log.append({
                        "provider_event_id": event_id,
                        "home_team": record.get("home_team"),
                        "away_team": record.get("away_team"),
                        "reason": "multiple_start_time_matches",
                        "candidate_game_pks": list(within["game_pk"].astype(int)),
                    })
                record["game_pk"] = pd.NA
                record["match_method"] = pd.NA
                record["source_status"] = SOURCE_AMBIGUOUS_MATCH
                out_rows.append(record)
                continue

        # Ordered pairing when counts match and starts uniquely order.
        same_matchup_provider = rows[
            (rows["home_team"] == record["home_team"])
            & (rows["away_team"] == record["away_team"])
        ]
        if (
            len(same_matchup_provider) == len(candidates)
            and candidates["_start"].notna().all()
            and same_matchup_provider["game_datetime"].map(_to_utc_ts).notna().all()
        ):
            prov_ordered = same_matchup_provider.copy()
            prov_ordered["_start"] = prov_ordered["game_datetime"].map(_to_utc_ts)
            prov_ordered = prov_ordered.sort_values("_start")
            cand_ordered = candidates.sort_values("_start")
            if (
                prov_ordered["_start"].is_unique
                and cand_ordered["_start"].is_unique
                and event_id is not None
            ):
                idx = list(prov_ordered["provider_event_id"].astype(str)).index(event_id)
                record["game_pk"] = int(cand_ordered.iloc[idx]["game_pk"])
                record["match_method"] = "game_order"
                out_rows.append(record)
                continue

        if ambiguous_log is not None:
            ambiguous_log.append({
                "provider_event_id": event_id,
                "home_team": record.get("home_team"),
                "away_team": record.get("away_team"),
                "reason": "ambiguous_doubleheader",
                "candidate_game_pks": list(candidates["game_pk"].astype(int)),
            })
        record["game_pk"] = pd.NA
        record["match_method"] = pd.NA
        record["source_status"] = SOURCE_AMBIGUOUS_MATCH
        out_rows.append(record)

    frame = pd.DataFrame(out_rows)
    frame["snapshot_role"] = SNAPSHOT_ROLE_RAW
    frame["snapshot_id"] = [
        make_snapshot_id(r) if pd.notna(r.get("provider_event_id")) else pd.NA
        for _, r in frame.iterrows()
    ]
    return normalize_snapshot_frame(frame)


# ---------------------------------------------------------------------------
# Persistence + standard snapshot roles
# ---------------------------------------------------------------------------


def append_odds_snapshots(snapshots: pd.DataFrame, path: str | None = None) -> pd.DataFrame:
    """Append-only immutable odds snapshot table."""
    path = path or config.MARKET_ODDS_SNAPSHOTS_PATH
    frame = normalize_snapshot_frame(snapshots)
    if frame.empty:
        if os.path.exists(path):
            return normalize_snapshot_frame(pd.read_csv(path))
        return frame
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        existing = normalize_snapshot_frame(pd.read_csv(path))
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame
    combined = combined.drop_duplicates(subset=["snapshot_id"], keep="first")
    combined.to_csv(path, index=False)
    return combined


def mark_post_start_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Flag rows captured at/after first pitch (not valid as closing)."""
    frame = normalize_snapshot_frame(snapshots)
    if frame.empty:
        return frame
    out = frame.copy()
    for idx, row in out.iterrows():
        start = _to_utc_ts(row.get("game_datetime"))
        captured = _to_utc_ts(row.get("captured_at_utc"))
        if pd.notna(start) and pd.notna(captured) and captured >= start:
            if row.get("source_status") == SOURCE_OK:
                out.at[idx, "source_status"] = SOURCE_POST_START
    return out


def coerce_requested_snapshot_role(
    snapshots: pd.DataFrame,
    requested_role: str,
) -> pd.DataFrame:
    """Never relabel post-start / post-completion captures as morning (etc.).

    Historical odds fetched after a game was completed must not be stored
    as a prediction-time ``morning`` / ``lineup_lock`` / ``opening`` snapshot.
    Those rows keep their prices but are forced to ``raw`` with
    ``historical_postgame_fetch`` (or keep ``post_start``).
    """
    frame = mark_post_start_snapshots(normalize_snapshot_frame(snapshots))
    if frame.empty:
        return frame
    out = frame.copy()
    roles = []
    statuses = []
    for _, row in out.iterrows():
        start = _to_utc_ts(row.get("game_datetime"))
        captured = _to_utc_ts(row.get("captured_at_utc"))
        status = row.get("source_status") or SOURCE_OK
        role = requested_role
        if requested_role in PREDICTION_TIME_ROLES:
            post_start = (
                pd.notna(start)
                and pd.notna(captured)
                and captured >= start
            )
            # Also refuse morning labels when the calendar game date is
            # strictly before the capture calendar day (historical backfill).
            game_day = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.notna(game_day):
                game_day = pd.Timestamp(game_day).tz_localize(None).normalize()
            capture_day = (
                captured.tz_convert(None).normalize()
                if pd.notna(captured)
                else pd.NaT
            )
            historical_day = (
                pd.notna(game_day)
                and pd.notna(capture_day)
                and capture_day > game_day
            )
            if post_start or historical_day:
                role = SNAPSHOT_ROLE_RAW
                if status == SOURCE_OK or status == SOURCE_POST_START:
                    status = (
                        SOURCE_POST_START if post_start else SOURCE_HISTORICAL_POSTGAME
                    )
                elif status not in INVALID_PREDICTION_TIME_STATUSES:
                    status = SOURCE_HISTORICAL_POSTGAME
        roles.append(role)
        statuses.append(status)
    out["snapshot_role"] = roles
    out["source_status"] = statuses
    return out


def filter_valid_prediction_time_snapshots(
    snapshots: pd.DataFrame,
    *,
    required_role: str | None = None,
) -> pd.DataFrame:
    """Rows eligible as prediction-time market priors for model training.

    Rejects missing ``game_pk``, unmatched/ambiguous/post-start/historical
    statuses, captures at/after first pitch, and roles inconsistent with
    capture time. Closing is never returned.
    """
    frame = normalize_snapshot_frame(snapshots)
    if frame.empty:
        return frame
    frame = mark_post_start_snapshots(frame)
    ok = frame[
        frame["game_pk"].notna()
        & frame["market_home_win_probability"].notna()
        & (frame["source_status"] == SOURCE_OK)
        & frame["snapshot_role"].isin(list(PREDICTION_TIME_ROLES))
    ].copy()
    if required_role is not None:
        if required_role == "closing":
            raise ValueError("closing cannot be a prediction-time market prior")
        ok = ok[ok["snapshot_role"] == required_role]
    if ok.empty:
        return ok

    keep = []
    for idx, row in ok.iterrows():
        start = _to_utc_ts(row.get("game_datetime"))
        captured = _to_utc_ts(row.get("captured_at_utc"))
        if pd.isna(captured):
            continue
        if pd.notna(start) and captured >= start:
            continue
        game_day = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.notna(game_day):
            game_day = pd.Timestamp(game_day).tz_localize(None).normalize()
            capture_day = captured.tz_convert(None).normalize()
            if capture_day > game_day:
                continue
        keep.append(idx)
    return ok.loc[keep] if keep else ok.iloc[0:0]


def select_opening_snapshot(snapshots: pd.DataFrame, game_pk) -> pd.Series | None:
    """Earliest observed ok snapshot for ``game_pk``."""
    frame = normalize_snapshot_frame(snapshots)
    scoped = frame[
        (frame["game_pk"] == game_pk)
        & (frame["source_status"] == SOURCE_OK)
        & frame["market_home_win_probability"].notna()
    ]
    if scoped.empty:
        return None
    scoped = scoped.copy()
    scoped["_ts"] = scoped["captured_at_utc"].map(_to_utc_ts)
    scoped = scoped.sort_values("_ts")
    return scoped.iloc[0].drop(labels=["_ts"], errors="ignore")


def select_closing_snapshot(snapshots: pd.DataFrame, game_pk, game_datetime=None) -> pd.Series | None:
    """Latest valid snapshot strictly before game start.

    Never uses a post-start capture. ``game_datetime`` may be supplied
    explicitly; otherwise the max non-null ``game_datetime`` on the
    game's snapshot rows is used.
    """
    frame = normalize_snapshot_frame(snapshots)
    scoped = frame[frame["game_pk"] == game_pk].copy()
    if scoped.empty:
        return None
    start = _to_utc_ts(game_datetime) if game_datetime is not None else pd.NaT
    if pd.isna(start):
        starts = scoped["game_datetime"].map(_to_utc_ts).dropna()
        if starts.empty:
            return None
        start = starts.max()

    scoped["_ts"] = scoped["captured_at_utc"].map(_to_utc_ts)
    valid = scoped[
        (scoped["source_status"] == SOURCE_OK)
        & scoped["market_home_win_probability"].notna()
        & scoped["_ts"].notna()
        & (scoped["_ts"] < start)
    ]
    if valid.empty:
        return None
    valid = valid.sort_values("_ts")
    return valid.iloc[-1].drop(labels=["_ts"], errors="ignore")


def select_role_snapshot(snapshots: pd.DataFrame, game_pk, role: str) -> pd.Series | None:
    """Latest valid prediction-time snapshot tagged with ``snapshot_role``.

    Uses ``filter_valid_prediction_time_snapshots`` so unmatched / post-start /
    historically backfilled rows never enter training or inference priors.
    """
    if role == "closing":
        raise ValueError("use select_closing_snapshot for closing")
    frame = filter_valid_prediction_time_snapshots(snapshots, required_role=role)
    scoped = frame[frame["game_pk"] == game_pk]
    if scoped.empty:
        return None
    scoped = scoped.copy()
    scoped["_ts"] = scoped["captured_at_utc"].map(_to_utc_ts)
    scoped = scoped.sort_values("_ts")
    return scoped.iloc[-1].drop(labels=["_ts"], errors="ignore")


def standard_snapshots_for_game(snapshots: pd.DataFrame, game_pk, game_datetime=None) -> dict:
    """Return opening / morning / lineup_lock / closing Series (or None)."""
    return {
        SNAPSHOT_ROLE_OPENING: select_opening_snapshot(snapshots, game_pk),
        SNAPSHOT_ROLE_MORNING: select_role_snapshot(snapshots, game_pk, SNAPSHOT_ROLE_MORNING),
        SNAPSHOT_ROLE_LINEUP_LOCK: select_role_snapshot(snapshots, game_pk, SNAPSHOT_ROLE_LINEUP_LOCK),
        SNAPSHOT_ROLE_CLOSING: select_closing_snapshot(snapshots, game_pk, game_datetime=game_datetime),
    }


# ---------------------------------------------------------------------------
# Fetch + persist pipeline helpers
# ---------------------------------------------------------------------------


def fetch_provider_odds(date, *, captured_at_utc: str | None = None) -> pd.DataFrame:
    """Fetch ESPN provider rows for ``date`` (no game_pk matching yet)."""
    date_str = pd.Timestamp(date).strftime("%Y%m%d")
    as_of = pd.Timestamp(date).normalize()
    captured = captured_at_utc or utc_now_iso()
    scoreboard = fetch_scoreboard(date_str)

    rows = []
    for event in scoreboard.get("events", []):
        event_id = event.get("id")
        if event_id is None:
            continue
        try:
            summary = fetch_summary(event_id)
            row = _extract_provider_row(
                event, summary, captured_at_utc=captured, as_of_date=as_of,
            )
        except Exception as exc:
            print(
                f"WARNING: failed to fetch/parse real ESPN odds for event {event_id} on {date_str} "
                f"({exc}); skipping that game."
            )
            continue
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows, columns=PROVIDER_OUTPUT_COLUMNS) if rows else empty_provider_frame()


def fetch_and_persist_odds_snapshots(
    date,
    schedule_games: pd.DataFrame | None = None,
    *,
    snapshot_role: str = SNAPSHOT_ROLE_RAW,
    captured_at_utc: str | None = None,
    snapshots_path: str | None = None,
    event_map_path: str | None = None,
) -> pd.DataFrame:
    """Fetch, match, tag role, append to the immutable snapshot log.

    Returns the matched snapshot frame for this capture (including
    unmatched/ambiguous rows). Successful unique matches update the
    provider-event map for future runs.
    """
    provider = fetch_provider_odds(date, captured_at_utc=captured_at_utc)
    if schedule_games is None:
        from mlb_metrics import schedule as schedule_mod
        try:
            schedule_games = schedule_mod.fetch_todays_games(date)
        except Exception as exc:
            print(f"WARNING: schedule fetch for odds matching failed ({exc})")
            schedule_games = pd.DataFrame()

    ambiguous_log: list = []
    matched = match_provider_events_to_games(
        provider,
        schedule_games,
        event_map=load_event_map(event_map_path),
        ambiguous_log=ambiguous_log,
    )
    for entry in ambiguous_log:
        print(
            f"WARNING: ambiguous odds match rejected for provider_event_id="
            f"{entry.get('provider_event_id')} {entry.get('home_team')} vs "
            f"{entry.get('away_team')} candidates={entry.get('candidate_game_pks')} "
            f"reason={entry.get('reason')}"
        )

    matched = mark_post_start_snapshots(matched)
    matched = coerce_requested_snapshot_role(matched, snapshot_role)
    matched = matched.copy()
    matched["snapshot_id"] = [make_snapshot_id(r) for _, r in matched.iterrows()]

    ok_map = matched[
        matched["game_pk"].notna()
        & matched["provider_event_id"].notna()
        & matched["source_status"].isin([SOURCE_OK, SOURCE_POST_START, SOURCE_HISTORICAL_POSTGAME])
    ][["provider_event_id", "game_pk", "home_team", "away_team", "date", "match_method"]]
    if not ok_map.empty:
        save_event_map(ok_map, event_map_path)

    append_odds_snapshots(matched, snapshots_path)
    return matched


REQUIRED_LIVE_MARKET_COLUMNS = (
    "game_pk",
    "snapshot_id",
    "captured_at_utc",
    "snapshot_role",
    "source_status",
    "market_home_win_probability",
    "home_moneyline",
    "away_moneyline",
)

LIVE_RECOMMENDATION_ROLES = frozenset({
    SNAPSHOT_ROLE_MORNING,
    SNAPSHOT_ROLE_LINEUP_LOCK,
})


def load_odds_snapshots(path: str | None = None) -> pd.DataFrame:
    path = path or config.MARKET_ODDS_SNAPSHOTS_PATH
    if not path or not os.path.exists(path):
        return empty_snapshot_frame()
    return normalize_snapshot_frame(pd.read_csv(path))


def market_for_live_recommendations(
    snapshots: pd.DataFrame,
    *,
    required_game_pks: Sequence | None = None,
) -> pd.DataFrame:
    """Snapshot-backed market frame for live bet sizing.

    Requires exact ``game_pk`` match, ``SOURCE_OK``,
    ``captured_at_utc < game_datetime``, and role in
    {morning, lineup_lock}. Rejects legacy team-only frames that lack
    snapshot provenance columns.
    """
    if snapshots is None or (isinstance(snapshots, pd.DataFrame) and snapshots.empty):
        return empty_snapshot_frame()

    missing_cols = [c for c in REQUIRED_LIVE_MARKET_COLUMNS if c not in snapshots.columns]
    if missing_cols:
        raise ValueError(
            "Legacy team-only market frame rejected for live betting; "
            f"missing columns: {missing_cols}"
        )

    # Prediction-time filter already enforces SOURCE_OK + pre-start + role set.
    valid = filter_valid_prediction_time_snapshots(snapshots)
    valid = valid[valid["snapshot_role"].isin(list(LIVE_RECOMMENDATION_ROLES))].copy()
    if valid.empty:
        return valid

    valid["_ts"] = valid["captured_at_utc"].map(_to_utc_ts)
    valid = valid.sort_values("_ts").drop_duplicates(subset=["game_pk"], keep="last")
    valid = valid.drop(columns=["_ts"], errors="ignore")

    if required_game_pks is not None:
        needed = {int(pk) for pk in required_game_pks if pd.notna(pk)}
        have = {int(pk) for pk in valid["game_pk"].dropna().tolist()}
        missing = sorted(needed - have)
        if missing:
            raise ValueError(
                "No valid prediction-time market snapshot for game_pk(s): "
                f"{missing}"
            )
        valid = valid[valid["game_pk"].isin(list(needed))].copy()

    return valid.reset_index(drop=True)


def snapshots_for_predictions(
    matched_snapshots: pd.DataFrame,
    *,
    require_ok: bool = True,
) -> pd.DataFrame:
    """Projection used when attaching odds to game picks (one row per game_pk)."""
    frame = normalize_snapshot_frame(matched_snapshots)
    if frame.empty:
        return frame
    if require_ok:
        frame = frame[frame["source_status"] == SOURCE_OK]
    frame = frame[frame["game_pk"].notna() & frame["market_home_win_probability"].notna()]
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["_ts"] = frame["captured_at_utc"].map(_to_utc_ts)
    frame = frame.sort_values("_ts").drop_duplicates(subset=["game_pk"], keep="last")
    return frame.drop(columns=["_ts"], errors="ignore")


def fetch_market_home_win_probabilities(date) -> pd.DataFrame:
    """Backward-compatible fetch used by existing callers/tests.

    Returns de-vigged probabilities keyed by (home_team, away_team) when
    matching is unique; doubleheader collisions are dropped (same honest
    failure mode as before). Prefer ``fetch_and_persist_odds_snapshots``
    for production paths that need ``game_pk`` and snapshot IDs.
    """
    provider = fetch_provider_odds(date)
    ok = provider[provider["source_status"] == SOURCE_OK].copy()
    if ok.empty:
        return pd.DataFrame(columns=LEGACY_MARKET_COLUMNS)

    dupe_keys = ok.duplicated(subset=["home_team", "away_team"], keep=False)
    if dupe_keys.any():
        dropped = ok.loc[dupe_keys, ["home_team", "away_team"]].drop_duplicates()
        print(
            f"WARNING: dropping ambiguous same-day matchup(s) from legacy market fetch "
            f"{dropped.to_dict('records')} — use fetch_and_persist_odds_snapshots for DH-safe matching."
        )
        ok = ok.loc[~dupe_keys]

    out = ok.rename(columns={"sportsbook": "market_provider"})
    return out[LEGACY_MARKET_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Closing-line value helpers (bet-side)
# ---------------------------------------------------------------------------


def implied_probability_for_side(home_moneyline, away_moneyline, side: str) -> float:
    """Vigged implied probability for ``side`` in {home, away}."""
    if side == "home":
        return float(moneyline_to_implied_probability(float(home_moneyline)))
    if side == "away":
        return float(moneyline_to_implied_probability(float(away_moneyline)))
    raise ValueError(f"side must be 'home' or 'away', got {side!r}")


def probability_clv(bet_implied: float, closing_implied: float) -> float:
    """Conventional probability CLV: closing − bet-time implied for the bet side.

    Positive means the market moved toward the bet (favorable CLV).
    """
    return float(closing_implied) - float(bet_implied)


def moneyline_clv(bet_moneyline: float, closing_moneyline: float) -> float:
    """Raw American-odds difference (closing − bet). Meaningful when same sign."""
    return float(closing_moneyline) - float(bet_moneyline)
