"""Append-only Polymarket paper ledger (no order placement).

Simulates size-limited taker purchases against recorded order-book levels.
A displayed last price or a limit that was merely touched is not a fill.
Canceled contracts are not treated as $0 losses or automatic refunds.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mlb_metrics import config
from mlb_metrics.venues.base import FeeSchedule
from mlb_metrics.venues.polymarket_us import select_fee_schedule

DECISION_COLUMNS = [
    "decision_id",
    "policy_version",
    "venue_id",
    "market_id",
    "market_slug",
    "game_pk",
    "side_team",
    "is_long",
    "decision_time_utc",
    "quote_receive_time_utc",
    "quote_request_time_utc",
    "fee_version",
    "action",
    "pass_reason",
    "requested_qty",
    "fill_assumption",
    "delay_seconds",
    "adverse_ticks",
    "tick_size",
    "filled_qty",
    "unfilled_qty",
    "avg_fill_price",
    "acquisition_cost",
    "fees_paid",
    "fills_json",
    "model_name",
    "model_probability",
    "market_mid_probability",
    "executable_buy",
    "probability_source",
    "outcome_unknown_at_decision",
]

POSITION_COLUMNS = [
    "decision_id",
    "status",
    "settlement_payout_per_contract",
    "settlement_rule",
    "settled_at_utc",
    "proceeds",
    "net_pnl",
    "open_exposure",
    "acquisition_cost",
    "filled_qty",
    "winning_team",
    "revision_of",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def decision_id(
    *,
    policy_version: str,
    market_id: str,
    game_pk: Any,
    decision_time_utc: str,
    side_team: str,
    fill_assumption: str,
) -> str:
    payload = "|".join(
        [
            str(policy_version),
            str(market_id),
            str(game_pk),
            str(decision_time_utc),
            str(side_team),
            str(fill_assumption),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def walk_asks_for_buy(
    asks: list[dict[str, Any]] | None,
    requested_qty: float,
    *,
    adverse_ticks: int = 0,
    tick_size: float = 0.01,
) -> tuple[list[dict[str, float]], float]:
    """Fill a buy by walking recorded asks. Empty/missing size is unfilled.

    Adverse ticks shift every ask up (worse for the buyer). Limit-touch is
    not modeled: only resting displayed size at recorded prices can fill.
    """
    remaining = float(requested_qty)
    fills: list[dict[str, float]] = []
    if remaining <= 0:
        return fills, 0.0
    tick = float(tick_size or 0.01)
    bump = int(adverse_ticks) * tick
    for level in asks or []:
        if remaining <= 0:
            break
        try:
            px = float(level.get("price"))
            size = float(level.get("size"))
        except (TypeError, ValueError):
            continue
        if not (0.0 < px < 1.0) or size <= 0:
            continue
        take = min(remaining, size)
        fills.append({"price": px + bump, "qty": take})
        remaining -= take
    return fills, remaining


def fill_cost(
    fills: list[dict[str, float]],
    fee: FeeSchedule,
) -> dict[str, float]:
    notional = 0.0
    fees = 0.0
    qty = 0.0
    for fill in fills:
        px = float(fill["price"])
        q = float(fill["qty"])
        notional += px * q
        fees += fee.taker_fee(px, q)
        qty += q
    avg = (notional / qty) if qty else float("nan")
    return {
        "filled_qty": qty,
        "avg_fill_price": avg,
        "acquisition_notional": notional,
        "fees_paid": fees,
        "acquisition_cost": notional + fees,
    }


def market_mid_probability(
    *,
    best_bid: float | None,
    best_ask: float | None,
    home_is_long: bool,
) -> float | None:
    """Book mid for the long instrument, converted to P(home wins).

    Separate from the executable ask used for costs. Crossed/missing books
    return None.
    """
    if best_bid is None or best_ask is None:
        return None
    bid = float(best_bid)
    ask = float(best_ask)
    if not (0.0 < bid < 1.0 and 0.0 < ask < 1.0):
        return None
    if bid > ask:
        return None
    mid_long = 0.5 * (bid + ask)
    return mid_long if home_is_long else (1.0 - mid_long)


def simulate_paper_purchase(
    *,
    asks: list[dict[str, Any]] | None,
    requested_qty: float,
    fee: FeeSchedule,
    delay_seconds: int = 0,
    adverse_ticks: int = 0,
    tick_size: float = 0.01,
    display_price: float | None = None,
) -> dict[str, Any]:
    """Hypothetical taker buy. Display prices never fill.

    ``delay_seconds`` is recorded for provenance. Callers that model manual
    entry delay must select the book observed at ``decision_time + delay``
    (see ``quote_store.quote_as_of`` / ``purchase_after_manual_delay``) before
    calling this function — delay alone does not invent a later book.
    """
    del display_price  # explicit: last/display is not a fill source
    fills, unfilled = walk_asks_for_buy(
        asks,
        requested_qty,
        adverse_ticks=adverse_ticks,
        tick_size=tick_size,
    )
    cost = fill_cost(fills, fee)
    if cost["filled_qty"] <= 0:
        status = "unfilled"
    elif unfilled > 1e-12:
        status = "partial"
    else:
        status = "filled"
    return {
        "status": status,
        "delay_seconds": int(delay_seconds),
        "adverse_ticks": int(adverse_ticks),
        "fills": fills,
        "unfilled_qty": float(unfilled),
        **cost,
    }


def purchase_after_manual_delay(
    *,
    asks_at_decision: list[dict[str, Any]] | None,
    asks_after_delay: list[dict[str, Any]] | None,
    requested_qty: float,
    fee: FeeSchedule,
    delay_seconds: int,
    adverse_ticks: int = 0,
    tick_size: float = 0.01,
) -> dict[str, Any]:
    """Fill against the delayed book when present; else fail closed (unfilled).

    If ``delay_seconds > 0`` and no later book exists, do not silently reuse
    the decision-time book — that would understate manual-entry risk.
    """
    if int(delay_seconds) <= 0:
        return simulate_paper_purchase(
            asks=asks_at_decision,
            requested_qty=requested_qty,
            fee=fee,
            delay_seconds=0,
            adverse_ticks=adverse_ticks,
            tick_size=tick_size,
        )
    if not asks_after_delay:
        return {
            "status": "unfilled",
            "delay_seconds": int(delay_seconds),
            "adverse_ticks": int(adverse_ticks),
            "fills": [],
            "unfilled_qty": float(requested_qty),
            "filled_qty": 0.0,
            "avg_fill_price": float("nan"),
            "acquisition_notional": 0.0,
            "fees_paid": 0.0,
            "acquisition_cost": 0.0,
            "pass_reason": "missing_quote_after_manual_delay",
        }
    return simulate_paper_purchase(
        asks=asks_after_delay,
        requested_qty=requested_qty,
        fee=fee,
        delay_seconds=delay_seconds,
        adverse_ticks=adverse_ticks,
        tick_size=tick_size,
    )


def settle_position(
    *,
    filled_qty: float,
    acquisition_cost: float,
    selected_team: str,
    winning_team: str | None,
    canceled: bool = False,
    settlement_px: float | None = None,
    settlement_rule: str = "official_result_binary",
) -> dict[str, Any]:
    """Settle a paper buy of ``selected_team`` Yes shares.

    Canceled / unknown official results stay unresolved. Intermediate
    ``settlement_px`` is used only when the venue provides it.
    """
    if filled_qty <= 0:
        return {
            "status": "unfilled",
            "settlement_payout_per_contract": None,
            "settlement_rule": settlement_rule,
            "proceeds": 0.0,
            "net_pnl": 0.0,
            "open_exposure": 0.0,
        }
    if canceled:
        return {
            "status": "canceled",
            "settlement_payout_per_contract": None,
            "settlement_rule": "canceled_requires_contract_rules",
            "proceeds": None,
            "net_pnl": None,
            "open_exposure": float(acquisition_cost),
        }
    if settlement_px is not None:
        px = float(settlement_px)
        proceeds = px * float(filled_qty)
        return {
            "status": "settled",
            "settlement_payout_per_contract": px,
            "settlement_rule": "venue_settlement_px",
            "proceeds": proceeds,
            "net_pnl": proceeds - float(acquisition_cost),
            "open_exposure": 0.0,
        }
    if winning_team is None or winning_team == "":
        return {
            "status": "open",
            "settlement_payout_per_contract": None,
            "settlement_rule": settlement_rule,
            "proceeds": None,
            "net_pnl": None,
            "open_exposure": float(acquisition_cost),
        }
    won = str(winning_team).upper() == str(selected_team).upper()
    px = 1.0 if won else 0.0
    proceeds = px * float(filled_qty)
    return {
        "status": "settled",
        "settlement_payout_per_contract": px,
        "settlement_rule": settlement_rule,
        "proceeds": proceeds,
        "net_pnl": proceeds - float(acquisition_cost),
        "open_exposure": 0.0,
    }


def settled_roi(positions: pd.DataFrame, decisions: pd.DataFrame | None = None) -> dict[str, Any]:
    """ROI = settled net / settled acquisition. Open exposure excluded."""
    if positions is None or positions.empty:
        return {
            "n_settled": 0,
            "settled_acquisition": 0.0,
            "settled_net": 0.0,
            "roi": float("nan"),
            "open_exposure": 0.0,
            "n_open": 0,
            "n_canceled": 0,
            "n_unfilled": 0,
        }
    frame = positions.copy()
    if "acquisition_cost" not in frame.columns or frame["acquisition_cost"].isna().all():
        if decisions is not None and not decisions.empty and "decision_id" in frame.columns:
            acq = decisions[["decision_id", "acquisition_cost"]].drop_duplicates("decision_id")
            frame = frame.merge(acq, on="decision_id", how="left", suffixes=("", "_dec"))
            if "acquisition_cost_dec" in frame.columns:
                frame["acquisition_cost"] = frame["acquisition_cost"].fillna(frame["acquisition_cost_dec"])
    settled = frame[frame["status"] == "settled"]
    open_rows = frame[frame["status"] == "open"]
    canceled = frame[frame["status"] == "canceled"]
    unfilled = frame[frame["status"] == "unfilled"]
    acq = (
        float(pd.to_numeric(settled.get("acquisition_cost"), errors="coerce").fillna(0).sum())
        if not settled.empty and "acquisition_cost" in settled.columns
        else 0.0
    )
    if settled.empty:
        net = 0.0
    else:
        net = float(pd.to_numeric(settled["net_pnl"], errors="coerce").fillna(0).sum())
    open_exp = 0.0
    if not open_rows.empty and "open_exposure" in open_rows.columns:
        open_exp = float(pd.to_numeric(open_rows["open_exposure"], errors="coerce").fillna(0).sum())
    roi = (net / acq) if acq > 0 else float("nan")
    return {
        "n_settled": int(len(settled)),
        "settled_acquisition": acq,
        "settled_net": net,
        "roi": roi,
        "open_exposure": open_exp,
        "n_open": int(len(open_rows)),
        "n_canceled": int(len(canceled)),
        "n_unfilled": int(len(unfilled)),
    }


def build_position_row(
    *,
    decision_id_value: str,
    settlement: dict[str, Any],
    acquisition_cost: float,
    filled_qty: float,
    winning_team: str | None = None,
    settled_at_utc: str | None = None,
    revision_of: str | None = None,
) -> dict[str, Any]:
    return {
        "decision_id": decision_id_value,
        "status": settlement.get("status"),
        "settlement_payout_per_contract": settlement.get("settlement_payout_per_contract"),
        "settlement_rule": settlement.get("settlement_rule"),
        "settled_at_utc": settled_at_utc,
        "proceeds": settlement.get("proceeds"),
        "net_pnl": settlement.get("net_pnl"),
        "open_exposure": settlement.get("open_exposure"),
        "acquisition_cost": float(acquisition_cost),
        "filled_qty": float(filled_qty),
        "winning_team": winning_team,
        "revision_of": revision_of,
    }


def equity_drawdown(net_by_order: list[float]) -> dict[str, float]:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    streak = 0
    max_streak = 0
    for pnl in net_by_order:
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "ending_equity": equity,
        "max_drawdown": max_dd,
        "max_loss_streak": int(max_streak),
    }


def append_decisions(rows: list[dict[str, Any]], path: str | None = None) -> pd.DataFrame:
    path = path or config.POLYMARKET_PAPER_DECISIONS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame = pd.DataFrame(rows)
    for col in DECISION_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[DECISION_COLUMNS]
    if os.path.exists(path):
        prev = pd.read_csv(path)
        combined = pd.concat([prev, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["decision_id"], keep="first")
    else:
        combined = frame.drop_duplicates(subset=["decision_id"], keep="first")
    combined.to_csv(path, index=False)
    return combined


def append_positions(rows: list[dict[str, Any]], path: str | None = None) -> pd.DataFrame:
    path = path or config.POLYMARKET_PAPER_POSITIONS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame = pd.DataFrame(rows)
    for col in POSITION_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[POSITION_COLUMNS]
    if os.path.exists(path):
        prev = pd.read_csv(path)
        combined = pd.concat([prev, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["decision_id"], keep="last")
    else:
        combined = frame
    combined.to_csv(path, index=False)
    return combined


def build_decision_row(
    *,
    policy_version: str,
    venue_id: str,
    market_id: str,
    market_slug: str,
    game_pk: Any,
    side_team: str,
    is_long: bool,
    decision_time_utc: str,
    quote_receive_time_utc: str,
    quote_request_time_utc: str | None,
    fee: FeeSchedule,
    action: str,
    pass_reason: str | None,
    requested_qty: float,
    fill_assumption: str,
    purchase: dict[str, Any] | None,
    model_name: str,
    model_probability: float | None,
    market_mid_probability_value: float | None,
    executable_buy: float | None,
    probability_source: str,
    delay_seconds: int = 0,
    adverse_ticks: int = 0,
    tick_size: float = 0.01,
) -> dict[str, Any]:
    did = decision_id(
        policy_version=policy_version,
        market_id=str(market_id),
        game_pk=game_pk,
        decision_time_utc=decision_time_utc,
        side_team=str(side_team),
        fill_assumption=fill_assumption,
    )
    purchase = purchase or {}
    return {
        "decision_id": did,
        "policy_version": policy_version,
        "venue_id": venue_id,
        "market_id": str(market_id),
        "market_slug": market_slug,
        "game_pk": game_pk,
        "side_team": side_team,
        "is_long": bool(is_long),
        "decision_time_utc": decision_time_utc,
        "quote_receive_time_utc": quote_receive_time_utc,
        "quote_request_time_utc": quote_request_time_utc,
        "fee_version": fee.fee_version,
        "action": action,
        "pass_reason": pass_reason,
        "requested_qty": float(requested_qty),
        "fill_assumption": fill_assumption,
        "delay_seconds": int(delay_seconds),
        "adverse_ticks": int(adverse_ticks),
        "tick_size": float(tick_size),
        "filled_qty": purchase.get("filled_qty", 0.0),
        "unfilled_qty": purchase.get("unfilled_qty", requested_qty),
        "avg_fill_price": purchase.get("avg_fill_price"),
        "acquisition_cost": purchase.get("acquisition_cost", 0.0),
        "fees_paid": purchase.get("fees_paid", 0.0),
        "fills_json": json.dumps(purchase.get("fills") or []),
        "model_name": model_name,
        "model_probability": model_probability,
        "market_mid_probability": market_mid_probability_value,
        "executable_buy": executable_buy,
        "probability_source": probability_source,
        "outcome_unknown_at_decision": True,
    }


def fee_as_of(as_of_utc: str) -> FeeSchedule:
    return select_fee_schedule(as_of_utc=as_of_utc)
