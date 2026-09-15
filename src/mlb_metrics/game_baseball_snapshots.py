"""Game-level baseball input snapshots for decision coverage.

Combines probable starters (schedule) and confirmed lineups (when announced)
into one row per ``game_pk``. Historical use must filter by
``fetched_at_utc`` / observation time so later API revisions cannot silently
rewrite earlier decisions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mlb_metrics import config, lineup_snapshots, schedule

GAME_SNAPSHOT_COLUMNS = [
    "game_pk",
    "date",
    "game_datetime",
    "home_team",
    "away_team",
    "status",
    "home_probable_pitcher_key_mlbam",
    "away_probable_pitcher_key_mlbam",
    "home_starter_status",
    "away_starter_status",
    "home_lineup_status",
    "away_lineup_status",
    "home_confirmed_starter_count",
    "away_confirmed_starter_count",
    "lineup_batting_order_source",
    "usable_for_game_winner",
    "pass_reason",
    "fetched_at_utc",
    "source",
    "as_of_limitation",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_game_snapshot_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=GAME_SNAPSHOT_COLUMNS)


def normalize_game_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if df is not None else empty_game_snapshot_frame()
    for col in GAME_SNAPSHOT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if not out.empty:
        out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
        out["usable_for_game_winner"] = out["usable_for_game_winner"].astype("boolean")
    return out[GAME_SNAPSHOT_COLUMNS]


def _starter_status(key_mlbam) -> str:
    if key_mlbam is None or (isinstance(key_mlbam, float) and pd.isna(key_mlbam)):
        return "missing_probable"
    try:
        if pd.isna(key_mlbam):
            return "missing_probable"
    except (TypeError, ValueError):
        pass
    return "probable"


def build_game_baseball_snapshots(
    schedule_df: pd.DataFrame,
    lineup_df: pd.DataFrame | None = None,
    *,
    fetched_at_utc: str | None = None,
    require_confirmed_lineups: bool = False,
) -> pd.DataFrame:
    """Build one baseball-input row per game from schedule + optional lineups."""
    fetched_at_utc = fetched_at_utc or utc_now_iso()
    if schedule_df is None or schedule_df.empty:
        return empty_game_snapshot_frame()

    lineups = lineup_snapshots.normalize_snapshot_frame(
        lineup_df if lineup_df is not None else lineup_snapshots.empty_snapshot_frame()
    )
    rows: list[dict[str, Any]] = []
    for _, game in schedule_df.iterrows():
        gpk = game.get("game_pk")
        if pd.isna(gpk):
            continue
        gpk_i = int(gpk)
        home = game.get("home_team")
        away = game.get("away_team")
        home_prob = game.get("home_probable_pitcher_key_mlbam")
        away_prob = game.get("away_probable_pitcher_key_mlbam")
        home_starter = _starter_status(home_prob)
        away_starter = _starter_status(away_prob)

        game_lu = lineups[lineups["game_pk"] == gpk_i] if not lineups.empty else lineups
        home_lu = game_lu[game_lu["team"] == home] if not game_lu.empty else game_lu
        away_lu = game_lu[game_lu["team"] == away] if not game_lu.empty else game_lu
        home_confirmed = (
            home_lu[home_lu["is_confirmed_starter"] == True]  # noqa: E712
            if not home_lu.empty
            else home_lu
        )
        away_confirmed = (
            away_lu[away_lu["is_confirmed_starter"] == True]  # noqa: E712
            if not away_lu.empty
            else away_lu
        )
        home_n = int(len(home_confirmed))
        away_n = int(len(away_confirmed))
        home_lineup = (
            lineup_snapshots.LINEUP_STATUS_CONFIRMED
            if home_n > 0
            else lineup_snapshots.LINEUP_STATUS_UNCONFIRMED
        )
        away_lineup = (
            lineup_snapshots.LINEUP_STATUS_CONFIRMED
            if away_n > 0
            else lineup_snapshots.LINEUP_STATUS_UNCONFIRMED
        )
        order_sources = []
        if not home_confirmed.empty and "batting_order_source" in home_confirmed.columns:
            order_sources.extend(
                [str(x) for x in home_confirmed["batting_order_source"].dropna().unique()]
            )
        if not away_confirmed.empty and "batting_order_source" in away_confirmed.columns:
            order_sources.extend(
                [str(x) for x in away_confirmed["batting_order_source"].dropna().unique()]
            )
        order_source = ",".join(sorted(set(order_sources))) if order_sources else None

        reasons: list[str] = []
        status = str(game.get("status") or "")
        status_l = status.lower()
        if any(tok in status_l for tok in ("postpon", "cancel", "suspend")):
            reasons.append(f"schedule_status:{status}")
        if home_starter == "missing_probable":
            reasons.append("missing_home_probable_pitcher")
        if away_starter == "missing_probable":
            reasons.append("missing_away_probable_pitcher")
        if require_confirmed_lineups:
            if home_lineup != lineup_snapshots.LINEUP_STATUS_CONFIRMED:
                reasons.append("missing_home_confirmed_lineup")
            if away_lineup != lineup_snapshots.LINEUP_STATUS_CONFIRMED:
                reasons.append("missing_away_confirmed_lineup")

        usable = len(reasons) == 0 and home_starter == "probable" and away_starter == "probable"
        if require_confirmed_lineups:
            usable = usable and home_n > 0 and away_n > 0
        # Default game-winner path: both probable pitchers required; lineups optional.
        if not require_confirmed_lineups:
            usable = home_starter == "probable" and away_starter == "probable" and not any(
                r.startswith("schedule_status:") for r in reasons
            )
            if not usable and not reasons:
                reasons.append("incomplete_starters")

        rows.append(
            {
                "game_pk": gpk_i,
                "date": game.get("date"),
                "game_datetime": game.get("game_datetime"),
                "home_team": home,
                "away_team": away,
                "status": status or None,
                "home_probable_pitcher_key_mlbam": home_prob,
                "away_probable_pitcher_key_mlbam": away_prob,
                "home_starter_status": home_starter,
                "away_starter_status": away_starter,
                "home_lineup_status": home_lineup,
                "away_lineup_status": away_lineup,
                "home_confirmed_starter_count": home_n,
                "away_confirmed_starter_count": away_n,
                "lineup_batting_order_source": order_source,
                "usable_for_game_winner": bool(usable),
                "pass_reason": None if usable else ",".join(reasons) or "unusable",
                "fetched_at_utc": fetched_at_utc,
                "source": "schedule_plus_lineups",
                "as_of_limitation": (
                    None
                    if not order_source
                    else "announced_lineup_list_order_may_diverge_from_final_boxscore"
                ),
            }
        )
    return normalize_game_snapshot_frame(pd.DataFrame(rows))


def fetch_game_baseball_snapshots(date, *, force_lineups: bool = False) -> pd.DataFrame:
    """Fetch schedule + lineup hydrates for a date and build game snapshots."""
    fetched_at = utc_now_iso()
    schedule_df = schedule.fetch_todays_games(date)
    lineups = lineup_snapshots.fetch_lineup_snapshots(date, force=force_lineups)
    return build_game_baseball_snapshots(
        schedule_df,
        lineups,
        fetched_at_utc=fetched_at,
        require_confirmed_lineups=False,
    )


def filter_snapshots_as_of(
    snapshots: pd.DataFrame,
    *,
    as_of_utc: str,
) -> pd.DataFrame:
    """Keep only snapshots observed at or before ``as_of_utc``."""
    frame = normalize_game_snapshot_frame(snapshots)
    if frame.empty:
        return frame
    cutoff = pd.Timestamp(as_of_utc, tz="UTC")
    ts = pd.to_datetime(frame["fetched_at_utc"], utc=True, errors="coerce")
    return frame.loc[ts.notna() & (ts <= cutoff)].reset_index(drop=True)


def persist_game_snapshots(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_path = config.GAME_BASEBALL_SNAPSHOT_AUDIT_PATH
    latest_path = config.GAME_BASEBALL_SNAPSHOT_LATEST_PATH
    frame = normalize_game_snapshot_frame(snapshots)
    os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(latest_path) or ".", exist_ok=True)
    if frame.empty:
        audit = pd.read_csv(audit_path) if os.path.exists(audit_path) else frame
        latest = pd.read_csv(latest_path) if os.path.exists(latest_path) else frame
        return normalize_game_snapshot_frame(audit), normalize_game_snapshot_frame(latest)
    if os.path.exists(audit_path):
        audit = pd.concat([pd.read_csv(audit_path), frame], ignore_index=True)
    else:
        audit = frame
    audit = audit.drop_duplicates(keep="first")
    audit.to_csv(audit_path, index=False)
    existing = (
        normalize_game_snapshot_frame(pd.read_csv(latest_path))
        if os.path.exists(latest_path)
        else empty_game_snapshot_frame()
    )
    combined = pd.concat([existing, frame], ignore_index=True)
    combined["_ts"] = pd.to_datetime(combined["fetched_at_utc"], utc=True, errors="coerce")
    latest = combined.sort_values("_ts").drop_duplicates(subset=["game_pk"], keep="last")
    latest = latest.drop(columns=["_ts"], errors="ignore")
    latest.to_csv(latest_path, index=False)
    return normalize_game_snapshot_frame(audit), normalize_game_snapshot_frame(latest)


def coverage_waterfall(
    *,
    schedule_games: pd.DataFrame,
    registry: pd.DataFrame,
    quote_health_contracts: list[dict[str, Any]] | None,
    baseball_snapshots: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """all games → mapped → usable prices → usable baseball → reasons removed."""
    n_games = int(len(schedule_games)) if schedule_games is not None else 0
    mapped = registry[registry["mapping_status"] == "mapped"] if registry is not None else pd.DataFrame()
    n_mapped = int(len(mapped))
    quote_rows = quote_health_contracts or []
    by_market = {str(r["market_id"]): r for r in quote_rows}
    n_usable_price = 0
    price_pass: dict[str, int] = {}
    for _, row in mapped.iterrows():
        mid = str(row["market_id"])
        q = by_market.get(mid)
        if q is None:
            price_pass["missing_quote"] = price_pass.get("missing_quote", 0) + 1
            continue
        if q.get("pass_reason"):
            reason = str(q["pass_reason"])
            price_pass[reason] = price_pass.get(reason, 0) + 1
            continue
        n_usable_price += 1

    baseball = (
        normalize_game_snapshot_frame(baseball_snapshots)
        if baseball_snapshots is not None
        else empty_game_snapshot_frame()
    )
    n_usable_baseball = int(baseball["usable_for_game_winner"].fillna(False).sum()) if not baseball.empty else 0
    baseball_pass: dict[str, int] = {}
    if not baseball.empty:
        for reason, count in baseball.loc[
            ~baseball["usable_for_game_winner"].fillna(False), "pass_reason"
        ].value_counts(dropna=False).items():
            baseball_pass[str(reason)] = int(count)

    return {
        "n_schedule_games": n_games,
        "n_mapped_contracts": n_mapped,
        "n_usable_prices": n_usable_price,
        "n_usable_baseball_snapshots": n_usable_baseball,
        "removed_price_reasons": price_pass,
        "removed_baseball_reasons": baseball_pass,
        "notes": [
            "Mapped count is registry coverage, not live tradable books.",
            "Usable prices require a non-stale eligible book for that market_id.",
        ],
    }


def trace_contract_decision_inputs(
    *,
    registry_row: dict[str, Any] | pd.Series,
    quote_row: dict[str, Any] | None,
    market: Any | None,
    book: Any | None,
    baseball_row: dict[str, Any] | pd.Series | None,
    fee_version: str | None,
) -> dict[str, Any]:
    """Trace one contract through mapping → book → fees → baseball inputs."""
    reg = dict(registry_row) if not isinstance(registry_row, dict) else registry_row
    bb = None if baseball_row is None else (
        dict(baseball_row) if not isinstance(baseball_row, dict) else baseball_row
    )
    long_team = reg.get("long_team")
    home_team = reg.get("home_team")
    orientation_ok = True
    orientation_note = "long_team_recorded_separately_from_home_team"
    if market is not None:
        from mlb_metrics.venues.polymarket_us import executable_buy_price_for_team

        home_buy = (
            executable_buy_price_for_team(market, book, home_team)
            if book is not None and home_team
            else None
        )
    else:
        home_buy = None
    pass_reasons = []
    if reg.get("mapping_status") != "mapped":
        pass_reasons.append(f"mapping:{reg.get('mapping_status')}")
    if quote_row is None:
        pass_reasons.append("missing_quote")
    elif quote_row.get("pass_reason"):
        pass_reasons.append(str(quote_row["pass_reason"]))
    if bb is not None and not bool(bb.get("usable_for_game_winner")):
        pass_reasons.append(f"baseball:{bb.get('pass_reason')}")
    elif bb is None:
        pass_reasons.append("missing_baseball_snapshot")
    return {
        "venue_id": reg.get("venue_id"),
        "market_id": reg.get("market_id"),
        "market_slug": reg.get("market_slug"),
        "game_pk": reg.get("game_pk"),
        "home_team": home_team,
        "away_team": reg.get("away_team"),
        "long_team": long_team,
        "short_team": reg.get("short_team"),
        "outcome_orientation_note": orientation_note,
        "orientation_assumes_long_is_home": False,
        "orientation_ok": orientation_ok,
        "rules_hash": reg.get("rules_hash"),
        "rules_text": reg.get("rules_text"),
        "fee_version": fee_version,
        "price_unit": "usd_share_cost_0_to_1",
        "quantity_unit": "contracts",
        "quote": quote_row,
        "home_executable_buy": home_buy,
        "baseball": bb,
        "actionable": len(pass_reasons) == 0,
        "pass_reasons": pass_reasons,
        "evidence_json": json.dumps(
            {
                "mapping_status": reg.get("mapping_status"),
                "mapping_evidence": reg.get("mapping_evidence"),
            }
        ),
    }
