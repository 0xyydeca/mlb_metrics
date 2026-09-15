"""MLB game ↔ Polymarket contract registry and mapping.

Never assumes Yes/long means the home team. Ambiguous matches are
quarantined rather than silently chosen.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from mlb_metrics import config
from mlb_metrics.venues.base import VenueMarket
from mlb_metrics.venues.polymarket_us import normalize_team_abbr

REGISTRY_COLUMNS = [
    "schema_version",
    "venue_id",
    "event_id",
    "event_slug",
    "market_id",
    "market_slug",
    "market_type",
    "provider_game_id",
    "game_pk",
    "home_team",
    "away_team",
    "scheduled_start_utc",
    "trading_status",
    "active",
    "closed",
    "line",
    "rules_hash",
    "rules_text",
    "long_team",
    "short_team",
    "home_outcome_id",
    "away_outcome_id",
    "mapping_status",
    "mapping_evidence",
    "first_observed_at_utc",
    "updated_at_utc",
]

MAPPING_MAPPED = "mapped"
MAPPING_UNMATCHED = "unmatched"
MAPPING_AMBIGUOUS = "ambiguous"
MAPPING_INCOMPLETE = "incomplete_teams"


def _ts_utc(value) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def empty_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def load_schedule_snapshots_for_mapping(path: str | None = None) -> pd.DataFrame:
    """Compact game rows from persisted schedule snapshots (may be stale)."""
    from mlb_metrics import schedule_snapshots

    snaps = schedule_snapshots.load_schedule_snapshots(path)
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=["game_pk", "home_team", "away_team", "game_datetime", "date"])
    cols = [c for c in ["game_pk", "home_team", "away_team", "game_datetime", "date"] if c in snaps.columns]
    return snaps[cols].drop_duplicates(subset=["game_pk"], keep="last")


def fetch_live_schedule_for_mapping(
    *,
    start_date=None,
    lookahead_days: int | None = None,
) -> pd.DataFrame:
    """Fetch MLB StatsAPI games for today_local()..+lookahead (both DH games)."""
    import datetime as dt

    from mlb_metrics import schedule

    start = start_date or schedule.today_local()
    if not isinstance(start, dt.date):
        start = pd.Timestamp(start).date()
    days = int(
        config.POLYMARKET_SCHEDULE_LOOKAHEAD_DAYS
        if lookahead_days is None
        else lookahead_days
    )
    frames = []
    for offset in range(0, max(0, days) + 1):
        day = start + dt.timedelta(days=offset)
        try:
            part = schedule.fetch_todays_games(day)
        except Exception as exc:  # noqa: BLE001 - mapping continues with partial window
            print(f"WARNING: StatsAPI schedule fetch failed for {day}: {type(exc).__name__}: {exc}")
            continue
        if part is None or part.empty:
            continue
        frames.append(part[["game_pk", "home_team", "away_team", "game_datetime", "date"]].copy())
    if not frames:
        return pd.DataFrame(columns=["game_pk", "home_team", "away_team", "game_datetime", "date"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["game_pk"], keep="last")


def load_mapping_schedule(
    *,
    schedule_snapshots_path: str | None = None,
    start_date=None,
    lookahead_days: int | None = None,
    prefer_live: bool = True,
) -> pd.DataFrame:
    """Schedule used to map Polymarket moneylines to ``game_pk``.

    Live StatsAPI rows are preferred. Snapshot rows fill gaps for games not
    yet/no longer in the live window. Same matchup on a different day remains
    distinct via ``game_datetime`` + time tolerance.
    """
    live = (
        fetch_live_schedule_for_mapping(start_date=start_date, lookahead_days=lookahead_days)
        if prefer_live
        else pd.DataFrame(columns=["game_pk", "home_team", "away_team", "game_datetime", "date"])
    )
    snaps = load_schedule_snapshots_for_mapping(schedule_snapshots_path)
    if live.empty and snaps.empty:
        return pd.DataFrame(columns=["game_pk", "home_team", "away_team", "game_datetime", "date"])
    if live.empty:
        return snaps
    if snaps.empty:
        return live
    # Prefer live rows for overlapping game_pk values.
    snap_only = snaps[~snaps["game_pk"].isin(set(live["game_pk"].dropna()))]
    return pd.concat([live, snap_only], ignore_index=True).drop_duplicates(
        subset=["game_pk"], keep="first"
    )


def load_registry(path: str | None = None) -> pd.DataFrame:
    path = path or config.POLYMARKET_CONTRACT_REGISTRY_PATH
    if not path or not os.path.exists(path):
        return empty_registry_frame()
    frame = pd.read_csv(path)
    for col in REGISTRY_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[REGISTRY_COLUMNS]


def save_registry(frame: pd.DataFrame, path: str | None = None) -> str:
    path = path or config.POLYMARKET_CONTRACT_REGISTRY_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = frame.copy()
    for col in REGISTRY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[REGISTRY_COLUMNS]
    out.to_csv(path, index=False)
    return path


def long_short_teams(market: VenueMarket) -> tuple[str | None, str | None]:
    long_team = short_team = None
    for outcome in market.outcomes:
        if outcome.is_long:
            long_team = outcome.team_abbr
        else:
            short_team = outcome.team_abbr
    return long_team, short_team


def outcome_id_for_team(market: VenueMarket, team_abbr: str | None) -> str | None:
    team = normalize_team_abbr(team_abbr)
    if not team:
        return None
    for outcome in market.outcomes:
        if outcome.team_abbr == team:
            return outcome.outcome_id
    return None


def match_market_to_schedule(
    market: VenueMarket,
    schedule_games: pd.DataFrame,
    *,
    time_tolerance_minutes: int | None = None,
) -> dict[str, Any]:
    """Match a moneyline market to MLB schedule rows by teams + start time."""
    tol = int(
        config.POLYMARKET_MATCH_TIME_TOLERANCE_MINUTES
        if time_tolerance_minutes is None
        else time_tolerance_minutes
    )
    home = normalize_team_abbr(market.home_team)
    away = normalize_team_abbr(market.away_team)
    start = _ts_utc(market.scheduled_start_utc)
    if not home or not away:
        return {
            "mapping_status": MAPPING_INCOMPLETE,
            "game_pk": None,
            "mapping_evidence": json.dumps({"reason": "missing_home_or_away"}),
        }
    if schedule_games is None or schedule_games.empty:
        return {
            "mapping_status": MAPPING_UNMATCHED,
            "game_pk": None,
            "mapping_evidence": json.dumps({"reason": "empty_schedule", "home": home, "away": away}),
        }

    games = schedule_games.copy()
    games["home_team"] = games["home_team"].map(normalize_team_abbr)
    games["away_team"] = games["away_team"].map(normalize_team_abbr)
    scoped = games[(games["home_team"] == home) & (games["away_team"] == away)].copy()
    if "game_datetime" not in scoped.columns or pd.isna(start):
        if len(scoped) == 1:
            gpk = scoped.iloc[0].get("game_pk")
            return {
                "mapping_status": MAPPING_MAPPED,
                "game_pk": int(gpk) if pd.notna(gpk) else None,
                "mapping_evidence": json.dumps(
                    {"reason": "unique_matchup_without_time", "home": home, "away": away}
                ),
            }
        if len(scoped) == 0:
            return {
                "mapping_status": MAPPING_UNMATCHED,
                "game_pk": None,
                "mapping_evidence": json.dumps({"reason": "no_matchup", "home": home, "away": away}),
            }
        return {
            "mapping_status": MAPPING_AMBIGUOUS,
            "game_pk": None,
            "mapping_evidence": json.dumps(
                {
                    "reason": "multiple_matchups_without_usable_start",
                    "candidates": [int(x) for x in scoped["game_pk"].dropna().astype(int).tolist()],
                }
            ),
        }

    scoped["_start"] = scoped["game_datetime"].map(_ts_utc)
    scoped = scoped[scoped["_start"].notna()].copy()
    if scoped.empty:
        return {
            "mapping_status": MAPPING_UNMATCHED,
            "game_pk": None,
            "mapping_evidence": json.dumps({"reason": "no_datetimes", "home": home, "away": away}),
        }
    scoped["_delta_min"] = (scoped["_start"] - start).abs().dt.total_seconds() / 60.0
    within = scoped[scoped["_delta_min"] <= float(tol)].copy()
    if within.empty:
        nearest = scoped.sort_values("_delta_min").head(3)
        return {
            "mapping_status": MAPPING_UNMATCHED,
            "game_pk": None,
            "mapping_evidence": json.dumps(
                {
                    "reason": "outside_time_tolerance",
                    "home": home,
                    "away": away,
                    "scheduled_start_utc": market.scheduled_start_utc,
                    "tolerance_minutes": tol,
                    "nearest_candidates": [
                        {
                            "game_pk": int(row["game_pk"]) if pd.notna(row["game_pk"]) else None,
                            "game_datetime": str(row["game_datetime"]),
                            "delta_min": float(row["_delta_min"]),
                        }
                        for _, row in nearest.iterrows()
                    ],
                }
            ),
        }
    if len(within) > 1:
        return {
            "mapping_status": MAPPING_AMBIGUOUS,
            "game_pk": None,
            "mapping_evidence": json.dumps(
                {
                    "reason": "multiple_within_tolerance",
                    "candidates": [
                        {
                            "game_pk": int(row["game_pk"]),
                            "delta_min": float(row["_delta_min"]),
                        }
                        for _, row in within.iterrows()
                    ],
                }
            ),
        }
    row = within.iloc[0]
    return {
        "mapping_status": MAPPING_MAPPED,
        "game_pk": int(row["game_pk"]),
        "mapping_evidence": json.dumps(
            {
                "reason": "unique_teams_and_start",
                "home": home,
                "away": away,
                "delta_min": float(row["_delta_min"]),
                "tolerance_minutes": tol,
            }
        ),
    }


def registry_row_from_market(
    market: VenueMarket,
    *,
    mapping: dict[str, Any],
    observed_at_utc: str,
    previous: pd.Series | None = None,
) -> dict[str, Any]:
    long_team, short_team = long_short_teams(market)
    first_observed = observed_at_utc
    if previous is not None and pd.notna(previous.get("first_observed_at_utc")):
        first_observed = str(previous.get("first_observed_at_utc"))
    return {
        "schema_version": config.POLYMARKET_SCHEMA_VERSION,
        "venue_id": market.venue_id,
        "event_id": market.event_id,
        "event_slug": market.event_slug,
        "market_id": market.market_id,
        "market_slug": market.market_slug,
        "market_type": market.market_type,
        "provider_game_id": market.provider_game_id,
        "game_pk": mapping.get("game_pk"),
        "home_team": market.home_team,
        "away_team": market.away_team,
        "scheduled_start_utc": market.scheduled_start_utc,
        "trading_status": market.trading_status,
        "active": market.active,
        "closed": market.closed,
        "line": market.line,
        "rules_hash": market.rules_hash,
        "rules_text": market.rules_text,
        "long_team": long_team,
        "short_team": short_team,
        "home_outcome_id": outcome_id_for_team(market, market.home_team),
        "away_outcome_id": outcome_id_for_team(market, market.away_team),
        "mapping_status": mapping.get("mapping_status"),
        "mapping_evidence": mapping.get("mapping_evidence"),
        "first_observed_at_utc": first_observed,
        "updated_at_utc": observed_at_utc,
    }


def upsert_registry(
    existing: pd.DataFrame,
    markets: list[VenueMarket],
    schedule_games: pd.DataFrame,
    *,
    observed_at_utc: str,
) -> pd.DataFrame:
    current = empty_registry_frame() if existing is None or existing.empty else existing.copy()
    by_key = {}
    if not current.empty:
        for _, row in current.iterrows():
            by_key[(str(row["venue_id"]), str(row["market_id"]))] = row

    rows = []
    seen = set()
    for market in markets:
        key = (market.venue_id, market.market_id)
        seen.add(key)
        mapping = match_market_to_schedule(market, schedule_games)
        prev = by_key.get(key)
        rows.append(
            registry_row_from_market(
                market,
                mapping=mapping,
                observed_at_utc=observed_at_utc,
                previous=prev,
            )
        )
    # Preserve previously seen contracts not in this capture.
    for key, row in by_key.items():
        if key not in seen:
            rows.append(row.to_dict())
    out = pd.DataFrame(rows)
    if out.empty:
        return empty_registry_frame()
    out = out.drop_duplicates(subset=["venue_id", "market_id"], keep="first")
    return out[REGISTRY_COLUMNS]


def coverage_summary(registry: pd.DataFrame) -> dict[str, Any]:
    if registry is None or registry.empty:
        return {
            "n_contracts": 0,
            "n_mapped": 0,
            "n_unmatched": 0,
            "n_ambiguous": 0,
            "n_incomplete": 0,
            "mapped_rate": None,
        }
    status = registry["mapping_status"].fillna("").astype(str)
    n = int(len(registry))
    n_mapped = int((status == MAPPING_MAPPED).sum())
    return {
        "n_contracts": n,
        "n_mapped": n_mapped,
        "n_unmatched": int((status == MAPPING_UNMATCHED).sum()),
        "n_ambiguous": int((status == MAPPING_AMBIGUOUS).sum()),
        "n_incomplete": int((status == MAPPING_INCOMPLETE).sum()),
        "mapped_rate": (n_mapped / n) if n else None,
    }
