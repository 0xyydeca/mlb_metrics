"""NFL game ↔ Polymarket contract registry (namespaced; separate from MLB).

Does not write to MLB ``data/polymarket/registry/contracts.csv``.
Native identity is ``sport_game_key`` = ``nfl:{game_id}`` — never ``game_pk``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from mlb_metrics import config
from mlb_metrics.sports.base import namespaced_game_key
from mlb_metrics.sports.nfl import NflSportAdapter
from mlb_metrics.venues.base import VenueMarket
from mlb_metrics.venues.polymarket_us import normalize_team_abbr

NFL_REGISTRY_COLUMNS = [
    "schema_version",
    "sport_id",
    "venue_id",
    "event_id",
    "event_slug",
    "market_id",
    "market_slug",
    "market_type",
    "provider_game_id",
    "sport_game_key",
    "native_game_id",
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


def empty_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=NFL_REGISTRY_COLUMNS)


def _ts_utc(value) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def match_market_to_nfl_schedule(
    market: VenueMarket,
    schedule_games: pd.DataFrame,
    *,
    tolerance_minutes: int | None = None,
) -> dict[str, Any]:
    tol = int(
        config.POLYMARKET_MATCH_TIME_TOLERANCE_MINUTES
        if tolerance_minutes is None
        else tolerance_minutes
    )
    home = normalize_team_abbr(market.home_team) if market.home_team else None
    away = normalize_team_abbr(market.away_team) if market.away_team else None
    if not home or not away:
        return {
            "mapping_status": MAPPING_INCOMPLETE,
            "sport_game_key": None,
            "native_game_id": None,
            "mapping_evidence": json.dumps({"reason": "incomplete_teams"}),
        }
    if schedule_games is None or schedule_games.empty:
        return {
            "mapping_status": MAPPING_UNMATCHED,
            "sport_game_key": None,
            "native_game_id": None,
            "mapping_evidence": json.dumps({"reason": "empty_schedule"}),
        }

    start = _ts_utc(market.scheduled_start_utc)
    candidates = schedule_games[
        (schedule_games["home_team"].astype(str).str.upper() == home)
        & (schedule_games["away_team"].astype(str).str.upper() == away)
    ]
    if candidates.empty:
        # Try swapped home/away (listing orientation)
        candidates = schedule_games[
            (schedule_games["home_team"].astype(str).str.upper() == away)
            & (schedule_games["away_team"].astype(str).str.upper() == home)
        ]
    if candidates.empty:
        return {
            "mapping_status": MAPPING_UNMATCHED,
            "sport_game_key": None,
            "native_game_id": None,
            "mapping_evidence": json.dumps({"reason": "no_team_match", "home": home, "away": away}),
        }

    if start is pd.NaT or candidates["game_datetime"].isna().all():
        if len(candidates) == 1:
            row = candidates.iloc[0]
            return {
                "mapping_status": MAPPING_MAPPED,
                "sport_game_key": row["sport_game_key"],
                "native_game_id": row.get("native_game_id"),
                "mapping_evidence": json.dumps({"reason": "unique_teams_no_start"}),
            }
        return {
            "mapping_status": MAPPING_AMBIGUOUS,
            "sport_game_key": None,
            "native_game_id": None,
            "mapping_evidence": json.dumps({"reason": "multiple_team_matches_no_start"}),
        }

    deltas = []
    for _, row in candidates.iterrows():
        gdt = _ts_utc(row.get("game_datetime"))
        if gdt is pd.NaT:
            continue
        delta_min = abs((gdt - start).total_seconds()) / 60.0
        if delta_min <= tol:
            deltas.append((delta_min, row))
    if not deltas:
        return {
            "mapping_status": MAPPING_UNMATCHED,
            "sport_game_key": None,
            "native_game_id": None,
            "mapping_evidence": json.dumps({"reason": "no_start_within_tolerance", "tol_min": tol}),
        }
    deltas.sort(key=lambda x: x[0])
    if len(deltas) > 1 and abs(deltas[0][0] - deltas[1][0]) < 1e-9:
        return {
            "mapping_status": MAPPING_AMBIGUOUS,
            "sport_game_key": None,
            "native_game_id": None,
            "mapping_evidence": json.dumps({"reason": "tied_start_deltas"}),
        }
    best = deltas[0][1]
    return {
        "mapping_status": MAPPING_MAPPED,
        "sport_game_key": best["sport_game_key"],
        "native_game_id": best.get("native_game_id"),
        "mapping_evidence": json.dumps(
            {
                "reason": "unique_teams_and_start",
                "home": home,
                "away": away,
                "delta_min": deltas[0][0],
                "tolerance_minutes": tol,
            }
        ),
    }


def upsert_nfl_registry(
    existing: pd.DataFrame,
    markets: list[VenueMarket],
    schedule_games: pd.DataFrame,
    *,
    observed_at_utc: str,
) -> pd.DataFrame:
    rows = []
    existing = existing if existing is not None else empty_registry_frame()
    by_id = {}
    if not existing.empty and "market_id" in existing.columns:
        for _, row in existing.iterrows():
            by_id[str(row["market_id"])] = row.to_dict()

    for market in markets:
        matched = match_market_to_nfl_schedule(market, schedule_games)
        long_team = None
        short_team = None
        home_oid = None
        away_oid = None
        for o in market.outcomes:
            if o.is_long:
                long_team = o.team_abbr
            else:
                short_team = o.team_abbr
            if o.team_abbr and market.home_team and normalize_team_abbr(o.team_abbr) == normalize_team_abbr(
                market.home_team
            ):
                home_oid = o.outcome_id
            if o.team_abbr and market.away_team and normalize_team_abbr(o.team_abbr) == normalize_team_abbr(
                market.away_team
            ):
                away_oid = o.outcome_id

        prev = by_id.get(str(market.market_id))
        first_obs = prev.get("first_observed_at_utc") if prev else observed_at_utc
        rows.append(
            {
                "schema_version": config.POLYMARKET_SCHEMA_VERSION,
                "sport_id": "nfl",
                "venue_id": market.venue_id,
                "event_id": market.event_id,
                "event_slug": market.event_slug,
                "market_id": market.market_id,
                "market_slug": market.market_slug,
                "market_type": market.market_type,
                "provider_game_id": market.provider_game_id,
                "sport_game_key": matched.get("sport_game_key"),
                "native_game_id": matched.get("native_game_id"),
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
                "home_outcome_id": home_oid,
                "away_outcome_id": away_oid,
                "mapping_status": matched["mapping_status"],
                "mapping_evidence": matched["mapping_evidence"],
                "first_observed_at_utc": first_obs or observed_at_utc,
                "updated_at_utc": observed_at_utc,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return existing if not existing.empty else empty_registry_frame()
    for col in NFL_REGISTRY_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[NFL_REGISTRY_COLUMNS]
    if existing.empty:
        return frame
    combined = pd.concat([existing, frame], ignore_index=True)
    combined = combined.drop_duplicates(subset=["market_id"], keep="last")
    return combined.reset_index(drop=True)


def save_registry(frame: pd.DataFrame, path: str | None = None) -> str:
    path = path or config.POLYMARKET_NFL_CONTRACT_REGISTRY_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_registry(path: str | None = None) -> pd.DataFrame:
    path = path or config.POLYMARKET_NFL_CONTRACT_REGISTRY_PATH
    if not os.path.exists(path):
        return empty_registry_frame()
    return pd.read_csv(path)


def load_nfl_schedule_for_mapping(*, lookahead_days: int = 7) -> pd.DataFrame:
    return NflSportAdapter().load_schedule_for_mapping(lookahead_days=lookahead_days)
