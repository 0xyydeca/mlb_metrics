"""Production paper ledger pipeline (no order placement).

Builds candidates and passes from mapped contracts + recorded books,
applies manual-entry delay by selecting as-of quotes, and settles open
positions from official MLB results. Fixtures stay out of production paths.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from mlb_metrics import (
    config,
    decision_board,
    game_baseball_snapshots,
    market_contracts,
    paper_ledger,
    quote_store,
)
from mlb_metrics.venues.polymarket_us import select_fee_schedule


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _add_seconds(iso_utc: str, seconds: int) -> str:
    ts = pd.Timestamp(iso_utc, tz="UTC") + timedelta(seconds=int(seconds))
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def asks_for_side(*, is_long: bool, quote_row: pd.Series | None) -> list[dict[str, float]]:
    """Executable buy levels for long Yes or short Yes (1 - bid) approximation.

    Long side walks recorded asks. Short side walks inverted bids
    (buy short ≈ sell long) using ``price = 1 - bid`` and bid sizes.
    """
    if quote_row is None:
        return []
    if is_long:
        return quote_store.asks_from_quote_row(quote_row)
    bids = quote_store.bids_from_quote_row(quote_row)
    return [{"price": 1.0 - float(b["price"]), "size": float(b["size"])} for b in bids if 0.0 < float(b["price"]) < 1.0]


def evaluate_mapped_side(
    *,
    reg: pd.Series,
    side_team: str,
    decision_time_utc: str,
    delay_seconds: int,
    adverse_ticks: int,
    requested_qty: float,
    policy_version: str,
    evidence_pass: bool,
    baseball_row: pd.Series | None,
    quotes_for_market: pd.DataFrame | None,
    model_probability: float | None,
    probability_source: str,
    model_name: str,
    allow_exploratory_fills: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return (decision_row, optional open position_row)."""
    market_id = str(reg["market_id"])
    long_team = str(reg.get("long_team") or "")
    is_long = side_team == long_team
    fee = select_fee_schedule(as_of_utc=decision_time_utc)
    fill_assumption = f"walk_asks_delay_{int(delay_seconds)}_adv_{int(adverse_ticks)}"

    quote_decision = quote_store.quote_as_of(
        market_id, as_of_utc=decision_time_utc, quotes=quotes_for_market
    )
    exec_time = _add_seconds(decision_time_utc, delay_seconds)
    quote_exec = quote_store.quote_as_of(
        market_id, as_of_utc=exec_time, quotes=quotes_for_market
    )

    pass_reasons: list[str] = []
    if reg.get("mapping_status") != "mapped":
        pass_reasons.append(f"mapping:{reg.get('mapping_status')}")
    if not evidence_pass:
        pass_reasons.append("evidence_gate_failed_or_unavailable")
    start = reg.get("scheduled_start_utc")
    if start and pd.notna(start) and pd.Timestamp(start, tz="UTC") <= pd.Timestamp(decision_time_utc, tz="UTC"):
        pass_reasons.append("game_started")
    if baseball_row is None:
        pass_reasons.append("missing_baseball_snapshot")
    elif not bool(baseball_row.get("usable_for_game_winner")):
        pass_reasons.append(f"baseball:{baseball_row.get('pass_reason')}")
    elif str(baseball_row.get("home_starter_status")) == "missing_probable" or str(
        baseball_row.get("away_starter_status")
    ) == "missing_probable":
        pass_reasons.append("missing_probable_starter")

    if quote_decision is None:
        pass_reasons.append("missing_quote")
    else:
        if quote_decision.get("eligible") is False:
            pass_reasons.append(
                f"ineligible:{quote_decision.get('eligibility_reason') or 'book'}"
            )
        recv = quote_decision.get("receive_time_utc")
        if recv is not None and pd.notna(recv):
            age = float(
                (
                    pd.Timestamp(decision_time_utc, tz="UTC")
                    - pd.Timestamp(recv, tz="UTC")
                ).total_seconds()
            )
            if age > float(config.POLYMARKET_QUOTE_MAX_AGE_SECONDS):
                pass_reasons.append("stale_quote")

    asks_decision = asks_for_side(is_long=is_long, quote_row=quote_decision)
    asks_exec = asks_for_side(is_long=is_long, quote_row=quote_exec)
    if int(delay_seconds) > 0:
        # Fail closed unless a strictly later observation exists after the delay.
        same_obs = (
            quote_decision is not None
            and quote_exec is not None
            and str(quote_exec.get("receive_time_utc"))
            == str(quote_decision.get("receive_time_utc"))
        )
        if quote_exec is None or same_obs:
            asks_exec = []
            if "missing_quote_after_manual_delay" not in pass_reasons:
                pass_reasons.append("missing_quote_after_manual_delay")
    if not asks_decision and "missing_quote" not in pass_reasons:
        pass_reasons.append("missing_book_depth")

    executable = None
    if asks_exec:
        executable = float(asks_exec[0]["price"])
    elif asks_decision:
        executable = float(asks_decision[0]["price"])

    mid = None
    if quote_decision is not None:
        bid = quote_decision.get("best_bid")
        ask = quote_decision.get("best_ask")
        if bid is not None and ask is not None and pd.notna(bid) and pd.notna(ask):
            mid_home = paper_ledger.market_mid_probability(
                best_bid=float(bid),
                best_ask=float(ask),
                home_is_long=str(reg.get("long_team")) == str(reg.get("home_team")),
            )
            if mid_home is not None:
                mid = mid_home if side_team == str(reg.get("home_team")) else (1.0 - mid_home)

    # Max price gate when we have a model probability.
    max_px = None
    if model_probability is not None:
        max_px = decision_board.max_acceptable_price(
            float(model_probability),
            fee,
            min_edge=min(config.GAME_RESIDUAL_EDGE_THRESHOLD_GRID),
            qty=float(requested_qty),
        )
        if executable is not None and (
            max_px is None or float(executable) > float(max_px) + 1e-12
        ):
            pass_reasons.append("executable_above_max_acceptable_price")

    # Evidence / input failures always pass. Exploratory fills only when
    # explicitly enabled AND no hard suppress reasons beyond evidence itself.
    hard = [r for r in pass_reasons if r != "evidence_gate_failed_or_unavailable"]
    can_buy = len(hard) == 0 and (
        evidence_pass or allow_exploratory_fills
    ) and bool(asks_decision or asks_exec)

    purchase = None
    action = "pass"
    if can_buy:
        purchase = paper_ledger.purchase_after_manual_delay(
            asks_at_decision=asks_decision,
            asks_after_delay=asks_exec if delay_seconds > 0 else asks_decision,
            requested_qty=requested_qty,
            fee=fee,
            delay_seconds=delay_seconds,
            adverse_ticks=adverse_ticks,
        )
        if purchase.get("pass_reason"):
            pass_reasons.append(str(purchase["pass_reason"]))
            action = "pass"
            purchase = None
        elif purchase["filled_qty"] <= 0:
            pass_reasons.append("unfilled_no_liquidity")
            action = "pass"
        else:
            action = "buy"
            if not evidence_pass:
                # Label exploratory paper buys distinctly.
                fill_assumption = f"exploratory_{fill_assumption}"
    elif not pass_reasons:
        pass_reasons.append("suppressed")

    decision = paper_ledger.build_decision_row(
        policy_version=policy_version,
        venue_id=str(reg.get("venue_id") or config.POLYMARKET_VENUE_SELECTED),
        market_id=market_id,
        market_slug=str(reg.get("market_slug") or ""),
        game_pk=reg.get("game_pk"),
        side_team=side_team,
        is_long=is_long,
        decision_time_utc=decision_time_utc,
        quote_receive_time_utc="",
        quote_request_time_utc=None,
        fee=fee,
        action=action,
        pass_reason=";".join(pass_reasons) if pass_reasons and action == "pass" else None,
        requested_qty=requested_qty,
        fill_assumption=fill_assumption,
        purchase=purchase,
        model_name=model_name,
        model_probability=model_probability,
        market_mid_probability_value=mid,
        executable_buy=executable,
        probability_source=probability_source,
        delay_seconds=delay_seconds,
        adverse_ticks=adverse_ticks,
    )
    q_used = quote_exec if quote_exec is not None else quote_decision
    if q_used is not None:
        decision["quote_receive_time_utc"] = q_used.get("receive_time_utc")
        decision["quote_request_time_utc"] = q_used.get("request_time_utc")

    position = None
    if action == "buy" and purchase and purchase["filled_qty"] > 0:
        settlement = paper_ledger.settle_position(
            filled_qty=purchase["filled_qty"],
            acquisition_cost=purchase["acquisition_cost"],
            selected_team=side_team,
            winning_team=None,
        )
        position = paper_ledger.build_position_row(
            decision_id_value=decision["decision_id"],
            settlement=settlement,
            acquisition_cost=purchase["acquisition_cost"],
            filled_qty=purchase["filled_qty"],
        )
    return decision, position


def run_paper_decision_cycle(
    *,
    registry: pd.DataFrame | None = None,
    baseball: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    decision_time_utc: str | None = None,
    delay_seconds: int | None = None,
    adverse_ticks: int | None = None,
    requested_qty: float | None = None,
    store_dir: str | None = None,
    allow_exploratory_fills: bool = False,
    decisions_path: str | None = None,
    positions_path: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate every mapped side; append decisions/positions (idempotent)."""
    assert config.BETTING_MODE == "disabled"
    decision_time_utc = decision_time_utc or utc_now_iso()
    delay_seconds = (
        int(config.POLYMARKET_CONSERVATIVE_DELAY_SECONDS)
        if delay_seconds is None
        else int(delay_seconds)
    )
    adverse_ticks = (
        int(config.POLYMARKET_CONSERVATIVE_ADVERSE_TICKS)
        if adverse_ticks is None
        else int(adverse_ticks)
    )
    requested_qty = float(
        config.POLYMARKET_PAPER_ONE_SHARE if requested_qty is None else requested_qty
    )
    gate = decision_board.load_evidence_gate()
    policy_version = str(gate.get("policy_version") or "polymarket_paper_v1")
    registry = registry if registry is not None else market_contracts.load_registry()
    mapped = registry[registry["mapping_status"] == "mapped"] if not registry.empty else registry

    baseball_by: dict[int, pd.Series] = {}
    if baseball is not None and not baseball.empty:
        bb = game_baseball_snapshots.filter_snapshots_as_of(
            baseball, as_of_utc=decision_time_utc
        )
        for _, row in bb.iterrows():
            if pd.notna(row.get("game_pk")):
                baseball_by[int(row["game_pk"])] = row

    pred_by: dict[int, pd.Series] = {}
    if predictions is not None and not predictions.empty and "game_pk" in predictions.columns:
        for gpk, part in predictions.groupby("game_pk"):
            pred_by[int(gpk)] = part.iloc[-1]

    decisions: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    for _, reg in mapped.iterrows():
        gpk = int(reg["game_pk"]) if pd.notna(reg.get("game_pk")) else None
        quotes = (
            quote_store.load_quotes_for_market(str(reg["market_id"]), store_dir=store_dir)
            if gpk is not None
            else None
        )
        bb = baseball_by.get(gpk) if gpk is not None else None
        pred = pred_by.get(gpk) if gpk is not None else None
        model_p = None
        prob_source = "none"
        model_name = "none"
        if pred is not None:
            home = str(reg.get("home_team"))
            away = str(reg.get("away_team"))
            home_p = None
            if "residual_home_win_probability" in pred.index and pd.notna(
                pred.get("residual_home_win_probability")
            ):
                home_p = float(pred["residual_home_win_probability"])
                prob_source = str(pred.get("probability_source") or "residual")
            elif "predicted_probability" in pred.index and pd.notna(pred.get("predicted_probability")):
                winner = str(pred.get("predicted_winner") or "")
                if winner == home:
                    home_p = float(pred["predicted_probability"])
                elif winner == away:
                    home_p = 1.0 - float(pred["predicted_probability"])
                prob_source = str(pred.get("probability_source") or "game_pick_export")
            model_name = str(pred.get("model_version") or "shadow_export")
            for side in (home, away):
                side_p = None if home_p is None else (home_p if side == home else 1.0 - home_p)
                decision, position = evaluate_mapped_side(
                    reg=reg,
                    side_team=side,
                    decision_time_utc=decision_time_utc,
                    delay_seconds=delay_seconds,
                    adverse_ticks=adverse_ticks,
                    requested_qty=requested_qty,
                    policy_version=policy_version,
                    evidence_pass=bool(gate["evidence_pass"]),
                    baseball_row=bb,
                    quotes_for_market=quotes,
                    model_probability=side_p,
                    probability_source=prob_source,
                    model_name=model_name,
                    allow_exploratory_fills=allow_exploratory_fills,
                )
                decisions.append(decision)
                if position is not None:
                    positions.append(position)
        else:
            for side in (str(reg.get("home_team")), str(reg.get("away_team"))):
                if not side or side == "nan":
                    continue
                decision, position = evaluate_mapped_side(
                    reg=reg,
                    side_team=side,
                    decision_time_utc=decision_time_utc,
                    delay_seconds=delay_seconds,
                    adverse_ticks=adverse_ticks,
                    requested_qty=requested_qty,
                    policy_version=policy_version,
                    evidence_pass=bool(gate["evidence_pass"]),
                    baseball_row=bb,
                    quotes_for_market=quotes,
                    model_probability=None,
                    probability_source="missing_prediction",
                    model_name="none",
                    allow_exploratory_fills=allow_exploratory_fills,
                )
                decisions.append(decision)
                if position is not None:
                    positions.append(position)

    decisions_path = decisions_path or config.POLYMARKET_PAPER_DECISIONS_PATH
    positions_path = positions_path or config.POLYMARKET_PAPER_POSITIONS_PATH
    if write and decisions:
        paper_ledger.append_decisions(decisions, path=decisions_path)
    if write and positions:
        paper_ledger.append_positions(positions, path=positions_path)
    n_buy = sum(1 for d in decisions if d.get("action") == "buy")
    n_pass = sum(1 for d in decisions if d.get("action") == "pass")
    return {
        "decision_time_utc": decision_time_utc,
        "n_decisions": len(decisions),
        "n_buy": n_buy,
        "n_pass": n_pass,
        "n_open_positions_written": len(positions),
        "evidence_pass": bool(gate["evidence_pass"]),
        "allow_exploratory_fills": allow_exploratory_fills,
        "delay_seconds": delay_seconds,
        "adverse_ticks": adverse_ticks,
        "decisions_path": decisions_path if write else None,
        "positions_path": positions_path if write else None,
        "betting_mode": config.BETTING_MODE,
    }


def settle_open_positions(
    *,
    results: pd.DataFrame,
    decisions_path: str | None = None,
    positions_path: str | None = None,
    settled_at_utc: str | None = None,
) -> dict[str, Any]:
    """Settle open paper positions using official Final game results.

    ``results`` must include ``game_pk``, ``status``, ``home_team``, ``away_team``,
    ``home_score``, ``away_score``. Canceled/postponed games stay open unless
    explicitly marked canceled (never auto $0).
    """
    decisions_path = decisions_path or config.POLYMARKET_PAPER_DECISIONS_PATH
    positions_path = positions_path or config.POLYMARKET_PAPER_POSITIONS_PATH
    settled_at_utc = settled_at_utc or utc_now_iso()
    if not os.path.exists(decisions_path):
        return {"n_settled": 0, "n_still_open": 0, "reason": "decisions_missing"}
    if not os.path.exists(positions_path):
        return {"n_settled": 0, "n_still_open": 0, "reason": "positions_missing"}
    decisions = pd.read_csv(decisions_path)
    positions = pd.read_csv(positions_path)
    if positions.empty:
        return {"n_settled": 0, "n_still_open": 0, "reason": "no_positions"}

    winners: dict[int, str] = {}
    canceled: set[int] = set()
    for _, row in results.iterrows():
        if pd.isna(row.get("game_pk")):
            continue
        gpk = int(row["game_pk"])
        status = str(row.get("status") or "")
        if status in ("Postponed", "Cancelled", "Canceled"):
            canceled.add(gpk)
            continue
        if status != "Final":
            continue
        try:
            hs = float(row["home_score"])
            aws = float(row["away_score"])
        except (TypeError, ValueError, KeyError):
            continue
        if hs > aws:
            winners[gpk] = str(row["home_team"])
        elif aws > hs:
            winners[gpk] = str(row["away_team"])

    updates: list[dict[str, Any]] = []
    n_settled = 0
    n_canceled = 0
    for _, pos in positions.iterrows():
        if str(pos.get("status")) != "open":
            continue
        did = pos["decision_id"]
        dec_rows = decisions[decisions["decision_id"] == did]
        if dec_rows.empty:
            continue
        dec = dec_rows.iloc[0]
        gpk = int(dec["game_pk"]) if pd.notna(dec.get("game_pk")) else None
        if gpk is None:
            continue
        acq = float(pos.get("acquisition_cost") if pd.notna(pos.get("acquisition_cost")) else dec.get("acquisition_cost") or 0.0)
        filled = float(pos.get("filled_qty") if pd.notna(pos.get("filled_qty")) else dec.get("filled_qty") or 0.0)
        if gpk in canceled:
            settlement = paper_ledger.settle_position(
                filled_qty=filled,
                acquisition_cost=acq,
                selected_team=str(dec["side_team"]),
                winning_team=None,
                canceled=True,
            )
            n_canceled += 1
            updates.append(
                paper_ledger.build_position_row(
                    decision_id_value=str(did),
                    settlement=settlement,
                    acquisition_cost=acq,
                    filled_qty=filled,
                    settled_at_utc=settled_at_utc,
                )
            )
            continue
        if gpk not in winners:
            continue
        settlement = paper_ledger.settle_position(
            filled_qty=filled,
            acquisition_cost=acq,
            selected_team=str(dec["side_team"]),
            winning_team=winners[gpk],
        )
        n_settled += 1
        updates.append(
            paper_ledger.build_position_row(
                decision_id_value=str(did),
                settlement=settlement,
                acquisition_cost=acq,
                filled_qty=filled,
                winning_team=winners[gpk],
                settled_at_utc=settled_at_utc,
            )
        )
    if updates:
        paper_ledger.append_positions(updates, path=positions_path)
    positions2 = pd.read_csv(positions_path) if os.path.exists(positions_path) else pd.DataFrame()
    roi = paper_ledger.settled_roi(positions2, decisions)
    return {
        "n_settled": n_settled,
        "n_canceled_marked": n_canceled,
        "n_still_open": int((positions2["status"] == "open").sum()) if not positions2.empty else 0,
        "roi": roi,
        "settled_at_utc": settled_at_utc,
    }


def trace_observed_contract(
    *,
    market_id: str | None = None,
    store_dir: str | None = None,
    registry_path: str | None = None,
    baseball_path: str | None = None,
    delay_seconds: int | None = None,
) -> dict[str, Any]:
    """Trace one real mapped contract through registry → quote → fees → paper."""
    registry = market_contracts.load_registry(registry_path)
    mapped = registry[registry["mapping_status"] == "mapped"]
    if mapped.empty:
        return {"ok": False, "missing": "no_mapped_contracts"}
    if market_id:
        rows = mapped[mapped["market_id"].astype(str) == str(market_id)]
        if rows.empty:
            return {"ok": False, "missing": f"market_id_not_mapped:{market_id}"}
        reg = rows.iloc[0]
    else:
        # Prefer a contract that has parquet depth.
        reg = mapped.iloc[0]
        for _, candidate in mapped.iterrows():
            q = quote_store.load_quotes_for_market(str(candidate["market_id"]), store_dir=store_dir)
            if not q.empty and quote_store.asks_from_quote_row(q.iloc[-1]):
                reg = candidate
                break

    mid = str(reg["market_id"])
    quotes = quote_store.load_quotes_for_market(mid, store_dir=store_dir)
    if quotes.empty:
        return {
            "ok": False,
            "missing": "no_stored_quotes_for_market",
            "market_id": mid,
            "game_pk": reg.get("game_pk"),
            "note": "Capture books before tracing; mapping alone is not enough.",
        }
    quote = quotes.iloc[-1]
    asks = quote_store.asks_from_quote_row(quote)
    if not asks:
        return {
            "ok": False,
            "missing": "quote_present_but_no_ask_depth",
            "market_id": mid,
            "eligibility_reason": quote.get("eligibility_reason"),
            "collector_vs_liquidity": "liquidity_or_empty_book"
            if quote.get("eligible") is False
            else "unknown",
        }

    decision_time = str(quote.get("receive_time_utc"))
    delay = int(
        config.POLYMARKET_CONSERVATIVE_DELAY_SECONDS if delay_seconds is None else delay_seconds
    )
    fee = select_fee_schedule(as_of_utc=decision_time)
    quote_delayed = quote_store.quote_as_of(
        mid, as_of_utc=_add_seconds(decision_time, delay), quotes=quotes
    )
    asks_delayed: list[dict[str, float]] = []
    if delay == 0:
        asks_delayed = asks
    elif (
        quote_delayed is not None
        and str(quote_delayed.get("receive_time_utc")) != str(quote.get("receive_time_utc"))
    ):
        asks_delayed = quote_store.asks_from_quote_row(quote_delayed)
    purchase = paper_ledger.purchase_after_manual_delay(
        asks_at_decision=asks,
        asks_after_delay=asks_delayed,
        requested_qty=1.0,
        fee=fee,
        delay_seconds=delay,
        adverse_ticks=int(config.POLYMARKET_CONSERVATIVE_ADVERSE_TICKS),
    )

    baseball_row = None
    bb_path = baseball_path or config.GAME_BASEBALL_SNAPSHOT_LATEST_PATH
    if os.path.exists(bb_path) and pd.notna(reg.get("game_pk")):
        bb = game_baseball_snapshots.normalize_game_snapshot_frame(pd.read_csv(bb_path))
        hit = bb[bb["game_pk"] == int(reg["game_pk"])]
        if not hit.empty:
            baseball_row = hit.iloc[0]

    gate = decision_board.load_evidence_gate()
    return {
        "ok": True,
        "venue_id": reg.get("venue_id"),
        "venue_label": "polymarket_us_provisional_research",
        "market_id": mid,
        "market_slug": reg.get("market_slug"),
        "event_slug": reg.get("event_slug"),
        "contract_url": config.POLYMARKET_US_EVENT_URL_TEMPLATE.format(
            event_slug=reg.get("event_slug") or ""
        ),
        "game_pk": reg.get("game_pk"),
        "home_team": reg.get("home_team"),
        "away_team": reg.get("away_team"),
        "long_team": reg.get("long_team"),
        "short_team": reg.get("short_team"),
        "rules_hash": reg.get("rules_hash"),
        "rules_text_excerpt": str(reg.get("rules_text") or "")[:240],
        "scheduled_start_utc": reg.get("scheduled_start_utc"),
        "quote_receive_time_utc": quote.get("receive_time_utc"),
        "quote_request_time_utc": quote.get("request_time_utc"),
        "best_bid": quote.get("best_bid"),
        "best_ask": quote.get("best_ask"),
        "best_ask_size": quote.get("best_ask_size"),
        "fee_version": fee.fee_version,
        "taker_theta": fee.taker_theta,
        "paper_purchase": {
            k: purchase[k]
            for k in (
                "status",
                "delay_seconds",
                "adverse_ticks",
                "filled_qty",
                "unfilled_qty",
                "avg_fill_price",
                "fees_paid",
                "acquisition_cost",
                "fills",
                "pass_reason",
            )
            if k in purchase
        },
        "baseball": None
        if baseball_row is None
        else {
            "usable_for_game_winner": bool(baseball_row.get("usable_for_game_winner")),
            "home_starter_status": baseball_row.get("home_starter_status"),
            "away_starter_status": baseball_row.get("away_starter_status"),
            "home_lineup_status": baseball_row.get("home_lineup_status"),
            "away_lineup_status": baseball_row.get("away_lineup_status"),
            "fetched_at_utc": baseball_row.get("fetched_at_utc"),
            "pass_reason": baseball_row.get("pass_reason"),
        },
        "evidence_verdict": gate.get("verdict"),
        "validation_status": gate.get("validation_status"),
        "actionable_suppressed": True,
        "suppress_note": "BETTING_MODE=disabled; evidence gates have not passed",
        "missing_for_labeled_eval": [
            item
            for item, present in [
                ("persisted_baseball_snapshot", baseball_row is not None),
                ("ask_depth", bool(asks)),
                ("evidence_gate_pass", bool(gate.get("evidence_pass"))),
            ]
            if not present
        ],
    }


def write_trace_report(trace: dict[str, Any], path: str | None = None) -> str:
    path = path or os.path.join("reports", "polymarket", "contract_trace_latest.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path
