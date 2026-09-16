"""Polymarket manual decision board: candidates, max price, explicit states.

Builds a dashboard export from registry + quotes + optional baseball /
shadow predictions. Never places orders. A high win probability alone
never yields a recommendation — evidence gates, freshness, and price
limits must all pass.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mlb_metrics import config, paper_ledger, quote_store
from mlb_metrics.venues.base import FeeSchedule
from mlb_metrics.venues.polymarket_us import select_fee_schedule

STATE_PAPER = config.POLYMARKET_DECISION_STATE_PAPER_CANDIDATE
STATE_INSUFFICIENT = config.POLYMARKET_DECISION_STATE_PASS_INSUFFICIENT
STATE_PRICE = config.POLYMARKET_DECISION_STATE_PASS_PRICE
STATE_STALE = config.POLYMARKET_DECISION_STATE_PASS_STALE
STATE_MISSING = config.POLYMARKET_DECISION_STATE_PASS_MISSING
STATE_STARTED = config.POLYMARKET_DECISION_STATE_GAME_STARTED

BOARD_COLUMNS = [
    "game_pk",
    "home_team",
    "away_team",
    "side_team",
    "is_long",
    "venue_id",
    "market_id",
    "market_slug",
    "event_slug",
    "contract_url",
    "scheduled_start_utc",
    "model_probability",
    "market_mid_probability",
    "executable_buy",
    "executable_qty",
    "est_taker_fee_per_contract",
    "total_acquisition_cost_per_contract",
    "expected_net_value_per_contract",
    "max_acceptable_price",
    "min_edge",
    "quote_age_seconds",
    "quote_receive_time_utc",
    "quote_stale",
    "home_starter_status",
    "away_starter_status",
    "home_lineup_status",
    "away_lineup_status",
    "model_version",
    "probability_source",
    "policy_version",
    "policy_hash",
    "validation_status",
    "evidence_verdict",
    "decision_state",
    "actionable",
    "stake_guidance_enabled",
    "permitted_exposure_units",
    "uncertainty_note",
    "pass_reason",
    "rules_hash",
    "fee_version",
    "board_generated_at_utc",
    "demo_fixture",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def acquisition_cost_per_contract(price: float, fee: FeeSchedule, qty: float = 1.0) -> float:
    """Price + price-dependent taker fee for ``qty`` contracts, per contract."""
    q = float(qty)
    if q <= 0:
        return float("nan")
    return float(price) + fee.taker_fee(float(price), q) / q


def expected_net_value(model_probability: float, price: float, fee: FeeSchedule, qty: float = 1.0) -> float:
    """E[payout] - acquisition cost, per contract (binary $1 settlement)."""
    return float(model_probability) - acquisition_cost_per_contract(price, fee, qty)


def max_acceptable_price(
    model_probability: float,
    fee: FeeSchedule,
    *,
    min_edge: float = 0.01,
    qty: float = 1.0,
    lo: float = 0.01,
    hi: float = 0.99,
) -> float | None:
    """Highest buy price where EV after fees is at least ``min_edge``.

    Solves ``p + fee(p) <= model_probability - min_edge`` by bisection because
    the taker fee depends on p. Returns None when no positive price works.
    """
    target = float(model_probability) - float(min_edge)
    if target <= 0:
        return None

    def ok(p: float) -> bool:
        return acquisition_cost_per_contract(p, fee, qty) <= target + 1e-12

    if not ok(lo):
        return None
    if ok(hi):
        return float(hi)
    a, b = float(lo), float(hi)
    for _ in range(48):
        mid = 0.5 * (a + b)
        if ok(mid):
            a = mid
        else:
            b = mid
    return float(a)


def load_evidence_gate(
    evaluation_path: str | None = None,
    policy_path: str | None = None,
) -> dict[str, Any]:
    evaluation_path = evaluation_path or config.POLYMARKET_PAPER_EVALUATION_REPORT_PATH
    policy_path = policy_path or config.POLYMARKET_FROZEN_POLICY_PATH
    evaluation: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    if evaluation_path and os.path.exists(evaluation_path):
        with open(evaluation_path, encoding="utf-8") as f:
            evaluation = json.load(f)
    if policy_path and os.path.exists(policy_path):
        with open(policy_path, encoding="utf-8") as f:
            policy = json.load(f)
    verdict = evaluation.get("verdict") or "insufficient_evidence"
    validation_status = evaluation.get("validation_status") or "insufficient_data"
    gates_ok = (
        verdict == "edge_supported"
        and validation_status == "validated_passed"
        and config.BETTING_MODE != "disabled"  # still fail-closed while disabled
    )
    # Owner decision: betting mode disabled until gates pass AND personal
    # limits configured. Software gate alone never enables live stakes.
    evidence_pass = verdict == "edge_supported" and validation_status == "validated_passed"
    return {
        "evaluation": evaluation,
        "policy": policy,
        "verdict": verdict,
        "validation_status": validation_status,
        "evidence_pass": bool(evidence_pass),
        "gates_ok_for_actionable": False,  # fail-closed while BETTING_MODE disabled
        "betting_mode": config.BETTING_MODE,
        "fail_reasons": list((evaluation.get("gates") or {}).get("gate_fail_reasons") or []),
        "policy_version": policy.get("policy_version"),
        "policy_hash": policy.get("policy_hash"),
        "min_edge": float(
            ((policy.get("entry") or {}).get("edge_vs_executable_ask_min"))
            or min(config.GAME_RESIDUAL_EDGE_THRESHOLD_GRID)
        ),
        "size_contracts": float(
            ((policy.get("entry") or {}).get("size_contracts"))
            or config.POLYMARKET_PAPER_ONE_SHARE
        ),
    }


def classify_decision_state(
    *,
    evidence_pass: bool,
    game_started: bool,
    quote_stale: bool,
    missing_inputs: bool,
    executable_buy: float | None,
    max_price: float | None,
    model_probability: float | None,
) -> tuple[str, str | None, bool]:
    """Return (state, pass_reason, actionable). High p alone never recommends."""
    if game_started:
        return STATE_STARTED, "game_started", False
    if not evidence_pass:
        return STATE_INSUFFICIENT, "evidence_gate_failed_or_unavailable", False
    if missing_inputs or model_probability is None or executable_buy is None:
        return STATE_MISSING, "missing_quote_or_model_or_baseball", False
    if quote_stale:
        return STATE_STALE, "stale_quote", False
    if max_price is None or float(executable_buy) > float(max_price) + 1e-12:
        return STATE_PRICE, "executable_above_max_acceptable_price", False
    return STATE_PAPER, None, True


def uncertainty_note(
    *,
    model_probability: float | None,
    probability_source: str | None,
    evidence_pass: bool,
    n_eligible_dates: int | None,
) -> str:
    parts = [
        "No invented confidence score.",
        "Uncertainty is conveyed by validation status, sample size, quote age, and calibration reports.",
    ]
    if not evidence_pass:
        parts.append("Registered evidence gates have not passed; treat probabilities as non-actionable.")
    if probability_source == "market_only_fallback":
        parts.append("Probability is market-only fallback, not an independent residual-model estimate.")
    if model_probability is not None and abs(float(model_probability) - 0.5) < 0.03:
        parts.append("Model probability is near 0.5; edge vs market is unlikely after fees.")
    if n_eligible_dates is not None and n_eligible_dates < int(config.POLYMARKET_MIN_ELIGIBLE_DATES):
        parts.append(
            f"Polymarket labeled history ({n_eligible_dates} dates) is below the structural floor "
            f"({config.POLYMARKET_MIN_ELIGIBLE_DATES})."
        )
    return " ".join(parts)


def _latest_quotes_frame(store_dir: str | None = None) -> pd.DataFrame:
    """Latest index quotes enriched with sizes from parquet when present."""
    latest = quote_store.latest_quotes_by_market(store_dir)
    if latest.empty:
        return latest
    store = quote_store.quote_store_dir(store_dir)
    size_by_hash: dict[str, dict[str, Any]] = {}
    for part_date in sorted({str(x) for x in latest.get("partition_date", pd.Series(dtype=str)).dropna().unique()}):
        path = os.path.join(store, f"dt={part_date}", "quotes.parquet")
        if not os.path.exists(path):
            continue
        try:
            part = pd.read_parquet(path, columns=["raw_response_hash", "best_ask_size", "best_bid_size"])
        except Exception:
            continue
        for _, prow in part.iterrows():
            size_by_hash[str(prow["raw_response_hash"])] = {
                "best_ask_size": prow.get("best_ask_size"),
                "best_bid_size": prow.get("best_bid_size"),
            }
    if not size_by_hash:
        return latest
    enriched = latest.copy()
    ask_sizes = []
    bid_sizes = []
    for _, row in enriched.iterrows():
        sizes = size_by_hash.get(str(row.get("raw_response_hash")), {})
        ask_sizes.append(sizes.get("best_ask_size"))
        bid_sizes.append(sizes.get("best_bid_size"))
    enriched["best_ask_size"] = ask_sizes
    enriched["best_bid_size"] = bid_sizes
    return enriched


def build_decision_board(
    *,
    registry: pd.DataFrame,
    quotes: pd.DataFrame | None = None,
    baseball: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    now_utc: str | None = None,
    gate: dict[str, Any] | None = None,
    demo_fixture: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble one row per mapped side (home and away) for dashboard export."""
    now = now_utc or utc_now_iso()
    gate = gate or load_evidence_gate()
    fee = select_fee_schedule(as_of_utc=now)
    quotes = quotes if quotes is not None else _latest_quotes_frame()
    quote_by_id = {}
    if quotes is not None and not quotes.empty:
        for _, q in quotes.iterrows():
            quote_by_id[str(q["market_id"])] = q

    baseball_by_gpk: dict[int, pd.Series] = {}
    if baseball is not None and not baseball.empty and "game_pk" in baseball.columns:
        for _, row in baseball.iterrows():
            if pd.notna(row.get("game_pk")):
                baseball_by_gpk[int(row["game_pk"])] = row

    pred_by_gpk: dict[int, pd.Series] = {}
    if predictions is not None and not predictions.empty and "game_pk" in predictions.columns:
        frame = predictions.copy()
        if "date" in frame.columns:
            frame = frame.sort_values("date")
        for gpk, part in frame.groupby("game_pk"):
            pred_by_gpk[int(gpk)] = part.iloc[-1]

    now_ts = pd.Timestamp(now, tz="UTC")
    rows: list[dict[str, Any]] = []
    mapped = registry[registry.get("mapping_status") == "mapped"] if not registry.empty else registry
    n_eligible = (gate.get("evaluation") or {}).get("data_coverage", {}).get(
        "n_polymarket_labeled_eligible_dates"
    )

    for _, reg in mapped.iterrows():
        market_id = str(reg["market_id"])
        gpk = int(reg["game_pk"]) if pd.notna(reg.get("game_pk")) else None
        start = reg.get("scheduled_start_utc")
        started = False
        if start and pd.notna(start):
            try:
                started = pd.Timestamp(start, tz="UTC") <= now_ts
            except Exception:
                started = False
        q = quote_by_id.get(market_id)
        age = None
        stale = True
        receive = None
        best_ask = None
        best_bid = None
        if q is not None:
            receive = q.get("receive_time_utc")
            if receive is not None and pd.notna(receive):
                age = float((now_ts - pd.Timestamp(receive, tz="UTC")).total_seconds())
                stale = age > float(config.POLYMARKET_QUOTE_MAX_AGE_SECONDS)
            best_ask = q.get("best_ask") if pd.notna(q.get("best_ask")) else None
            best_bid = q.get("best_bid") if pd.notna(q.get("best_bid")) else None
            if q.get("eligible") is False:
                stale = True

        bb = baseball_by_gpk.get(gpk) if gpk is not None else None
        pred = pred_by_gpk.get(gpk) if gpk is not None else None

        home = reg.get("home_team")
        away = reg.get("away_team")
        long_team = reg.get("long_team")
        event_slug = reg.get("event_slug") or ""
        contract_url = config.POLYMARKET_US_EVENT_URL_TEMPLATE.format(event_slug=event_slug)

        # Two sides: home and away Yes costs from long/short orientation.
        for side in (home, away):
            if side is None or (isinstance(side, float) and pd.isna(side)):
                continue
            side = str(side)
            is_long = str(long_team) == side
            if best_ask is None and best_bid is None:
                executable = None
                qty = None
            elif is_long:
                executable = float(best_ask) if best_ask is not None else None
                ask_sz = q.get("best_ask_size") if q is not None else None
                qty = float(ask_sz) if ask_sz is not None and pd.notna(ask_sz) else None
            else:
                executable = (1.0 - float(best_bid)) if best_bid is not None else None
                bid_sz = q.get("best_bid_size") if q is not None else None
                qty = float(bid_sz) if bid_sz is not None and pd.notna(bid_sz) else None

            # Model probability for this side.
            model_p = None
            prob_source = None
            model_version = None
            if pred is not None:
                # Prefer residual home win; convert for away.
                home_p = None
                for col in (
                    "residual_home_win_probability",
                    "home_win_probability",
                    "predicted_probability",
                ):
                    if col in pred.index and pd.notna(pred.get(col)):
                        # predicted_probability on game_picks is for predicted_winner
                        if col == "predicted_probability":
                            winner = pred.get("predicted_winner")
                            if str(winner) == str(home):
                                home_p = float(pred[col])
                            elif str(winner) == str(away):
                                home_p = 1.0 - float(pred[col])
                        else:
                            home_p = float(pred[col])
                        break
                if home_p is not None:
                    model_p = home_p if side == str(home) else (1.0 - home_p)
                prob_source = pred.get("probability_source") or "game_pick_export"
                model_version = pred.get("model_version") or pred.get("game_prediction_mode") or "shadow_export"
            if model_p is None and best_bid is not None and best_ask is not None:
                # Market mid as non-independent display only — never actionable alone.
                mid = paper_ledger.market_mid_probability(
                    best_bid=float(best_bid),
                    best_ask=float(best_ask),
                    home_is_long=str(long_team) == str(home),
                )
                if mid is not None:
                    model_p = mid if side == str(home) else (1.0 - mid)
                    prob_source = "market_mid_display_only"
                    model_version = "none"

            market_mid = None
            if best_bid is not None and best_ask is not None:
                mid_home = paper_ledger.market_mid_probability(
                    best_bid=float(best_bid),
                    best_ask=float(best_ask),
                    home_is_long=str(long_team) == str(home),
                )
                if mid_home is not None:
                    market_mid = mid_home if side == str(home) else (1.0 - mid_home)

            fee_amt = None
            total_cost = None
            env = None
            max_px = None
            if model_p is not None and executable is not None:
                fee_amt = fee.taker_fee(float(executable), 1.0)
                total_cost = acquisition_cost_per_contract(float(executable), fee, 1.0)
                env = expected_net_value(float(model_p), float(executable), fee, 1.0)
                max_px = max_acceptable_price(
                    float(model_p),
                    fee,
                    min_edge=float(gate["min_edge"]),
                    qty=float(gate["size_contracts"]),
                )

            missing = (
                executable is None
                or model_p is None
                or gpk is None
                or (bb is None)
                or (
                    bb is not None
                    and (
                        str(bb.get("home_starter_status")) == "missing_probable"
                        or str(bb.get("away_starter_status")) == "missing_probable"
                    )
                )
                or prob_source
                in (
                    "market_mid_display_only",
                    "market_only_fallback",
                )
            )
            # Evidence gate failure dominates actionability.
            state, reason, actionable = classify_decision_state(
                evidence_pass=bool(gate["evidence_pass"]),
                game_started=started,
                quote_stale=bool(stale),
                missing_inputs=bool(missing),
                executable_buy=executable,
                max_price=max_px,
                model_probability=model_p,
            )
            # While BETTING_MODE is disabled, never mark stake guidance on.
            stake_guidance = False
            actionable_manual = False  # requires evidence + personal limits in UI

            rows.append(
                {
                    "game_pk": gpk,
                    "home_team": home,
                    "away_team": away,
                    "side_team": side,
                    "is_long": is_long,
                    "venue_id": reg.get("venue_id") or config.POLYMARKET_VENUE_SELECTED,
                    "market_id": market_id,
                    "market_slug": reg.get("market_slug"),
                    "event_slug": event_slug,
                    "contract_url": contract_url,
                    "scheduled_start_utc": start,
                    "model_probability": model_p,
                    "market_mid_probability": market_mid,
                    "executable_buy": executable,
                    "executable_qty": qty,
                    "est_taker_fee_per_contract": fee_amt,
                    "total_acquisition_cost_per_contract": total_cost,
                    "expected_net_value_per_contract": env,
                    "max_acceptable_price": max_px,
                    "min_edge": gate["min_edge"],
                    "quote_age_seconds": age,
                    "quote_receive_time_utc": receive,
                    "quote_stale": stale,
                    "home_starter_status": None if bb is None else bb.get("home_starter_status"),
                    "away_starter_status": None if bb is None else bb.get("away_starter_status"),
                    "home_lineup_status": None if bb is None else bb.get("home_lineup_status"),
                    "away_lineup_status": None if bb is None else bb.get("away_lineup_status"),
                    "model_version": model_version,
                    "probability_source": prob_source,
                    "policy_version": gate.get("policy_version"),
                    "policy_hash": gate.get("policy_hash"),
                    "validation_status": gate.get("validation_status"),
                    "evidence_verdict": gate.get("verdict"),
                    "decision_state": state,
                    "actionable": actionable_manual,
                    "stake_guidance_enabled": stake_guidance,
                    "permitted_exposure_units": config.POLYMARKET_PAPER_DEFAULT_PER_BET_UNITS,
                    "uncertainty_note": uncertainty_note(
                        model_probability=model_p,
                        probability_source=prob_source,
                        evidence_pass=bool(gate["evidence_pass"]),
                        n_eligible_dates=n_eligible,
                    ),
                    "pass_reason": reason,
                    "rules_hash": reg.get("rules_hash"),
                    "fee_version": fee.fee_version,
                    "board_generated_at_utc": now,
                    "demo_fixture": bool(demo_fixture),
                }
            )

    board = pd.DataFrame(rows)
    for col in BOARD_COLUMNS:
        if col not in board.columns:
            board[col] = pd.NA
    board = board[BOARD_COLUMNS] if not board.empty else pd.DataFrame(columns=BOARD_COLUMNS)
    meta = {
        "generated_at_utc": now,
        "venue_label": "polymarket_us_provisional_research",
        "evidence_verdict": gate.get("verdict"),
        "validation_status": gate.get("validation_status"),
        "evidence_pass": gate.get("evidence_pass"),
        "betting_mode": config.BETTING_MODE,
        "game_prediction_mode": config.GAME_PREDICTION_MODE,
        "gate_fail_reasons": gate.get("fail_reasons"),
        "policy_version": gate.get("policy_version"),
        "policy_hash": gate.get("policy_hash"),
        "n_board_rows": int(len(board)),
        "n_paper_candidate": int((board["decision_state"] == STATE_PAPER).sum()) if not board.empty else 0,
        "n_actionable": int(board["actionable"].fillna(False).sum()) if not board.empty else 0,
        "paper_unit_label": config.POLYMARKET_PAPER_UNIT_LABEL,
        "personal_limits_required_for_stake_guidance": True,
        "orders_automated": False,
        "notes": [
            "High win probability alone never triggers a recommendation.",
            "Stake guidance stays off until personal bankroll/max-loss limits are configured.",
            "Recheck public quotes before any manual purchase.",
            "Software can display Pass states usefully when the correct answer is no bet.",
        ],
    }
    return board, meta


def write_decision_board(
    board: pd.DataFrame,
    meta: dict[str, Any],
    *,
    board_path: str | None = None,
    meta_path: str | None = None,
) -> dict[str, str]:
    board_path = board_path or config.POLYMARKET_DECISION_BOARD_PATH
    meta_path = meta_path or config.POLYMARKET_DECISION_BOARD_META_PATH
    os.makedirs(os.path.dirname(board_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    board.to_csv(board_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return {"board_path": board_path, "meta_path": meta_path}


def exposure_check(
    *,
    proposed_units: float,
    existing_positions: list[dict[str, Any]],
    game_pk: Any,
    team: str,
    per_bet_cap: float,
    daily_cap: float,
    same_game_cap: float,
    same_team_cap: float,
    decision_day: str,
) -> dict[str, Any]:
    """Reject stake guidance that would exceed configured exposure caps."""
    reasons = []
    if proposed_units > per_bet_cap + 1e-12:
        reasons.append("exceeds_per_bet_cap")
    day_units = sum(
        float(p.get("units") or 0)
        for p in existing_positions
        if str(p.get("decision_day")) == str(decision_day) and p.get("status") != "canceled"
    )
    if day_units + proposed_units > daily_cap + 1e-12:
        reasons.append("exceeds_daily_cap")
    game_units = sum(
        float(p.get("units") or 0)
        for p in existing_positions
        if str(p.get("game_pk")) == str(game_pk) and p.get("status") != "canceled"
    )
    if game_units + proposed_units > same_game_cap + 1e-12:
        reasons.append("exceeds_same_game_cap")
    team_units = sum(
        float(p.get("units") or 0)
        for p in existing_positions
        if str(p.get("side_team")) == str(team) and p.get("status") != "canceled"
    )
    if team_units + proposed_units > same_team_cap + 1e-12:
        reasons.append("exceeds_same_team_cap")
    # Never increase stakes to recover losses — caller must not pass a boost.
    return {
        "allowed": len(reasons) == 0,
        "reasons": reasons,
        "proposed_units": float(proposed_units),
    }
