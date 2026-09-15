"""Append-only Polymarket quote store (order books + eligibility).

Quotes are stored as date-partitioned Parquet under
``POLYMARKET_QUOTE_STORE_DIR``. A small CSV index supports fast health checks
without scanning every partition. Quote refreshes are not committed to git.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mlb_metrics import config
from mlb_metrics.venues.base import MarketBookQuote

QUOTE_COLUMNS = [
    "schema_version",
    "venue_id",
    "market_id",
    "market_slug",
    "request_time_utc",
    "receive_time_utc",
    "first_observed_at_utc",
    "source_transact_time_utc",
    "market_state",
    "best_bid",
    "best_bid_size",
    "best_ask",
    "best_ask_size",
    "bids_json",
    "asks_json",
    "display_current_px",
    "last_trade_px",
    "fee_version",
    "fee_coefficient",
    "tick_size",
    "min_trade_qty",
    "suspended",
    "raw_response_hash",
    "data_class",
    "eligible",
    "eligibility_reason",
]

INDEX_COLUMNS = [
    "venue_id",
    "market_id",
    "market_slug",
    "receive_time_utc",
    "best_ask",
    "best_bid",
    "eligible",
    "eligibility_reason",
    "partition_date",
    "raw_response_hash",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def quote_store_dir(path: str | None = None) -> str:
    return path or config.POLYMARKET_QUOTE_STORE_DIR


def index_path(store_dir: str | None = None) -> str:
    return os.path.join(quote_store_dir(store_dir), "quote_index.csv")


def partition_path(receive_time_utc: str, store_dir: str | None = None) -> str:
    day = str(pd.Timestamp(receive_time_utc, tz="UTC").date())
    return os.path.join(quote_store_dir(store_dir), f"dt={day}", "quotes.parquet")


def empty_quote_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=QUOTE_COLUMNS)


def quote_to_row(quote: MarketBookQuote, *, first_observed_at_utc: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": config.POLYMARKET_SCHEMA_VERSION,
        "venue_id": quote.venue_id,
        "market_id": quote.market_id,
        "market_slug": quote.market_slug,
        "request_time_utc": quote.request_time_utc,
        "receive_time_utc": quote.receive_time_utc,
        "first_observed_at_utc": first_observed_at_utc or quote.receive_time_utc,
        "source_transact_time_utc": quote.source_transact_time_utc,
        "market_state": quote.market_state,
        "best_bid": quote.best_bid,
        "best_bid_size": quote.best_bid_size,
        "best_ask": quote.best_ask,
        "best_ask_size": quote.best_ask_size,
        "bids_json": json.dumps(quote.bids),
        "asks_json": json.dumps(quote.asks),
        "display_current_px": quote.display_current_px,
        "last_trade_px": quote.last_trade_px,
        "fee_version": quote.fee_version,
        "fee_coefficient": quote.fee_coefficient,
        "tick_size": quote.tick_size,
        "min_trade_qty": quote.min_trade_qty,
        "suspended": quote.suspended,
        "raw_response_hash": quote.raw_response_hash,
        "data_class": quote.data_class,
        "eligible": quote.eligible,
        "eligibility_reason": quote.eligibility_reason,
    }


def append_quotes(
    quotes: list[MarketBookQuote],
    *,
    store_dir: str | None = None,
) -> dict[str, Any]:
    """Persist quotes idempotently by raw_response_hash within each partition."""
    if not quotes:
        return {"n_received": 0, "n_written": 0, "partitions": []}
    root = quote_store_dir(store_dir)
    os.makedirs(root, exist_ok=True)
    frame = pd.DataFrame([quote_to_row(q) for q in quotes])
    written = 0
    partitions: list[str] = []
    for receive_time, part in frame.groupby(frame["receive_time_utc"].map(
        lambda x: str(pd.Timestamp(x, tz="UTC").date())
    ), sort=False):
        path = os.path.join(root, f"dt={receive_time}", "quotes.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            prev = pd.read_parquet(path)
            combined = pd.concat([prev, part], ignore_index=True)
            combined = combined.drop_duplicates(subset=["raw_response_hash"], keep="first")
            n_new = int(len(combined) - len(prev))
            if n_new > 0:
                combined.to_parquet(path, index=False)
                written += n_new
        else:
            part = part.drop_duplicates(subset=["raw_response_hash"], keep="first")
            part.to_parquet(path, index=False)
            written += len(part)
        partitions.append(path)
    _update_index(frame, store_dir=root)
    return {"n_received": int(len(quotes)), "n_written": int(written), "partitions": partitions}


def _update_index(frame: pd.DataFrame, *, store_dir: str) -> None:
    idx_path = index_path(store_dir)
    rows = frame.copy()
    rows["partition_date"] = rows["receive_time_utc"].map(
        lambda x: str(pd.Timestamp(x, tz="UTC").date())
    )
    rows = rows[INDEX_COLUMNS]
    if os.path.exists(idx_path):
        prev = pd.read_csv(idx_path)
        rows = pd.concat([prev, rows], ignore_index=True)
        rows = rows.drop_duplicates(subset=["raw_response_hash"], keep="last")
    rows.to_csv(idx_path, index=False)


def load_quote_index(store_dir: str | None = None) -> pd.DataFrame:
    path = index_path(store_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.read_csv(path)


def latest_quote_age_seconds(
    store_dir: str | None = None,
    *,
    now_utc: str | None = None,
) -> dict[str, Any]:
    idx = load_quote_index(store_dir)
    now = pd.Timestamp(now_utc or _utc_now_iso(), tz="UTC")
    if idx.empty:
        return {
            "n_index_rows": 0,
            "latest_receive_time_utc": None,
            "age_seconds": None,
            "stale": True,
            "max_age_seconds": float(config.POLYMARKET_QUOTE_MAX_AGE_SECONDS),
            "reason": "no_quotes",
        }
    latest = pd.Timestamp(idx["receive_time_utc"].max(), tz="UTC")
    age = float((now - latest).total_seconds())
    max_age = float(config.POLYMARKET_QUOTE_MAX_AGE_SECONDS)
    return {
        "n_index_rows": int(len(idx)),
        "latest_receive_time_utc": latest.isoformat().replace("+00:00", "Z"),
        "age_seconds": age,
        "stale": age > max_age,
        "max_age_seconds": max_age,
        "reason": "ok" if age <= max_age else "stale",
    }


def latest_quotes_by_market(store_dir: str | None = None) -> pd.DataFrame:
    """Latest index row per market_id (by receive_time_utc)."""
    idx = load_quote_index(store_dir)
    if idx.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    frame = idx.copy()
    frame["_ts"] = pd.to_datetime(frame["receive_time_utc"], utc=True, errors="coerce")
    frame = frame.sort_values("_ts")
    latest = frame.drop_duplicates(subset=["market_id"], keep="last")
    return latest.drop(columns=["_ts"], errors="ignore")


def per_contract_quote_health(
    store_dir: str | None = None,
    *,
    market_ids: list[str] | None = None,
    now_utc: str | None = None,
    max_age_seconds: float | None = None,
) -> dict[str, Any]:
    """Freshness and eligibility for each contract (not just the global latest row)."""
    now = pd.Timestamp(now_utc or _utc_now_iso(), tz="UTC")
    max_age = float(
        config.POLYMARKET_QUOTE_MAX_AGE_SECONDS if max_age_seconds is None else max_age_seconds
    )
    latest = latest_quotes_by_market(store_dir)
    if market_ids is not None:
        wanted = {str(x) for x in market_ids}
        present = set(latest["market_id"].astype(str)) if not latest.empty else set()
        missing = sorted(wanted - present)
    else:
        missing = []
        wanted = set(latest["market_id"].astype(str)) if not latest.empty else set()

    rows = []
    n_fresh = n_stale = n_ineligible = 0
    for _, row in latest.iterrows():
        mid = str(row["market_id"])
        if market_ids is not None and mid not in wanted:
            continue
        recv = pd.Timestamp(row["receive_time_utc"], tz="UTC")
        age = float((now - recv).total_seconds())
        stale = age > max_age
        eligible = bool(row.get("eligible"))
        reason = None
        if stale:
            reason = "stale_quote"
            n_stale += 1
        elif not eligible:
            reason = str(row.get("eligibility_reason") or "ineligible")
            n_ineligible += 1
        else:
            n_fresh += 1
        rows.append(
            {
                "market_id": mid,
                "market_slug": row.get("market_slug"),
                "receive_time_utc": row.get("receive_time_utc"),
                "age_seconds": age,
                "stale": stale,
                "eligible": eligible,
                "best_ask": row.get("best_ask"),
                "best_bid": row.get("best_bid"),
                "pass_reason": reason,
            }
        )
    for mid in missing:
        rows.append(
            {
                "market_id": mid,
                "market_slug": None,
                "receive_time_utc": None,
                "age_seconds": None,
                "stale": True,
                "eligible": False,
                "best_ask": None,
                "best_bid": None,
                "pass_reason": "missing_quote",
            }
        )
    return {
        "n_markets_checked": int(len(wanted) if market_ids is not None else len(rows)),
        "n_fresh_eligible": int(n_fresh),
        "n_stale": int(n_stale + len(missing)),
        "n_ineligible": int(n_ineligible),
        "n_missing": int(len(missing)),
        "max_age_seconds": max_age,
        "contracts": rows,
    }


def actionable_suppressed(
    *,
    quote_health: dict[str, Any] | None = None,
    collector_heartbeat_utc: str | None = None,
    now_utc: str | None = None,
    max_heartbeat_age_seconds: float | None = None,
) -> dict[str, Any]:
    """Suppress actionable output after sleep/outage/stale collector inputs."""
    now = pd.Timestamp(now_utc or _utc_now_iso(), tz="UTC")
    max_hb = float(
        config.POLYMARKET_QUOTE_MAX_AGE_SECONDS
        if max_heartbeat_age_seconds is None
        else max_heartbeat_age_seconds
    )
    reasons: list[str] = []
    if quote_health and quote_health.get("stale"):
        reasons.append(f"global_quotes_{quote_health.get('reason') or 'stale'}")
    if collector_heartbeat_utc:
        hb = pd.Timestamp(collector_heartbeat_utc, tz="UTC")
        age = float((now - hb).total_seconds())
        if age > max_hb:
            reasons.append("collector_heartbeat_stale")
    elif collector_heartbeat_utc is None and quote_health and quote_health.get("reason") == "no_quotes":
        reasons.append("no_collector_heartbeat")
    return {
        "actionable": len(reasons) == 0,
        "suppress_reasons": reasons,
    }
