"""Gated manual Polymarket pilot: readiness, pauses, real fills, reconcile.

Separates software correctness from evidence for a limited real-money pilot.
Never places orders. Never lowers evidence thresholds. Never promotes
market-only fallbacks as independent models. Never increases stakes to
recover losses.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mlb_metrics import config, decision_board, paper_ledger

REAL_FILL_COLUMNS = [
    "fill_id",
    "recorded_at_utc",
    "decision_day",
    "market_id",
    "market_slug",
    "game_pk",
    "side_team",
    "purchase_price",
    "qty",
    "fees_paid",
    "status",
    "settlement_px",
    "settled_net",
    "notes",
    "stream",  # always "real_manual"
]

PAUSE_REASONS = {
    "operational_failure": "Collector/export/health operational failure",
    "loss_limit_exceeded": "Settled real loss reached or exceeded max affordable loss",
    "evidence_deterioration": "Registered evidence verdict no longer supports pilot",
    "owner_pause": "Owner requested pause",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def personal_limits_complete(limits: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Require dedicated bankroll + loss + exposure caps (not income-derived).

    Paper-unit limits (``currency=normalized_paper_unit``) may complete this
    check for gated paper ops. They never imply USD stake guidance or a
    real-money pilot by themselves.
    """
    missing: list[str] = []
    if not limits:
        return False, [
            "bankroll",
            "max_affordable_loss",
            "per_bet_exposure_limit",
            "daily_exposure_limit",
            "correlated_position_limits",
        ]
    required = {
        "bankroll": limits.get("bankroll"),
        "max_affordable_loss": limits.get("max_affordable_loss", limits.get("max_loss")),
        "per_bet_exposure_limit": limits.get("per_bet_exposure_limit", limits.get("per_bet")),
        "daily_exposure_limit": limits.get("daily_exposure_limit", limits.get("daily")),
        "same_game_exposure_limit": limits.get(
            "same_game_exposure_limit", limits.get("same_game")
        ),
        "same_team_exposure_limit": limits.get(
            "same_team_exposure_limit", limits.get("same_team")
        ),
    }
    for key, val in required.items():
        try:
            if val is None or float(val) <= 0:
                missing.append(key)
        except (TypeError, ValueError):
            missing.append(key)
    bankroll = required.get("bankroll")
    max_loss = required.get("max_affordable_loss")
    try:
        if bankroll is not None and max_loss is not None and float(max_loss) > float(bankroll):
            missing.append("max_affordable_loss_exceeds_bankroll")
    except (TypeError, ValueError):
        pass
    return len(missing) == 0, missing


def agent_paper_risk_limits() -> dict[str, Any]:
    """Owner-delegated paper-unit risk policy (not personal USD)."""
    return dict(config.POLYMARKET_AGENT_PAPER_RISK_LIMITS)


def write_risk_limits(
    limits: dict[str, Any] | None = None,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    path = path or config.POLYMARKET_PILOT_RISK_LIMITS_PATH
    payload = dict(limits or agent_paper_risk_limits())
    payload["saved_at_utc"] = utc_now_iso()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return payload


def load_risk_limits(path: str | None = None) -> dict[str, Any] | None:
    path = path or config.POLYMARKET_PILOT_RISK_LIMITS_PATH
    return _load_json(path)


def verify_candidate_and_market(
    evaluation: dict[str, Any] | None,
    protocol: dict[str, Any] | None,
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evidence must refer to the registered protocol and pilot market/candidates."""
    reasons: list[str] = []
    protocol_id = (protocol or {}).get("protocol_id") or (evaluation or {}).get("protocol_id")
    if protocol_id != config.POLYMARKET_PILOT_REQUIRED_PROTOCOL_ID:
        reasons.append(
            f"protocol_mismatch:expected={config.POLYMARKET_PILOT_REQUIRED_PROTOCOL_ID}"
            f":got={protocol_id!r}"
        )
    market = (
        (protocol or {}).get("market_id")
        or (evaluation or {}).get("market_id")
        or config.POLYMARKET_PILOT_MARKET_ID
    )
    if market != config.POLYMARKET_PILOT_MARKET_ID:
        reasons.append(f"market_mismatch:expected={config.POLYMARKET_PILOT_MARKET_ID}:got={market!r}")
    candidates = list(
        (protocol or {}).get("candidates")
        or (evaluation or {}).get("candidates")
        or (policy or {}).get("candidates")
        or []
    )
    if candidates:
        unexpected = [c for c in candidates if c not in config.POLYMARKET_PILOT_CANDIDATE_IDS]
        if unexpected:
            reasons.append(f"unexpected_candidates:{unexpected}")
        # Market-only fallback must not be listed as an independent model candidate.
        for banned in ("market_only_fallback", "market_mid_display_only"):
            if banned in candidates:
                reasons.append(f"banned_independent_candidate:{banned}")
    else:
        reasons.append("candidates_missing_from_registered_artifacts")
    policy_hash_eval = (evaluation or {}).get("policy_hash")
    policy_hash = (policy or {}).get("policy_hash")
    if policy is None:
        reasons.append("frozen_policy_missing")
    elif policy_hash_eval and policy_hash and policy_hash_eval != policy_hash:
        reasons.append("policy_hash_mismatch_evaluation_vs_frozen")
    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "protocol_id": protocol_id,
        "market_id": market,
        "candidates": candidates,
        "policy_hash": policy_hash or policy_hash_eval,
    }


def assess_software_readiness() -> dict[str, Any]:
    """Engineering path works for Pass / no-bet inspection (not an edge claim)."""
    checks = []
    checks.append(
        {
            "name": "modes_fail_closed",
            "passed": config.GAME_PREDICTION_MODE == "shadow"
            and config.BETTING_MODE == "disabled",
            "detail": f"GAME_PREDICTION_MODE={config.GAME_PREDICTION_MODE!r} "
            f"BETTING_MODE={config.BETTING_MODE!r}",
        }
    )
    checks.append(
        {
            "name": "protocol_registered",
            "passed": os.path.exists(config.POLYMARKET_PAPER_PROTOCOL_PATH),
            "detail": config.POLYMARKET_PAPER_PROTOCOL_PATH,
        }
    )
    checks.append(
        {
            "name": "evaluation_report_present",
            "passed": os.path.exists(config.POLYMARKET_PAPER_EVALUATION_REPORT_PATH),
            "detail": config.POLYMARKET_PAPER_EVALUATION_REPORT_PATH,
        }
    )
    checks.append(
        {
            "name": "decision_board_export_present",
            "passed": os.path.exists(config.POLYMARKET_DECISION_BOARD_PATH),
            "detail": config.POLYMARKET_DECISION_BOARD_PATH,
        }
    )
    checks.append(
        {
            "name": "orders_not_automated",
            "passed": True,
            "detail": "manual_pilot never places orders",
        }
    )
    passed = all(c["passed"] for c in checks)
    return {
        "conclusion": "software_operates_for_inspection" if passed else "software_incomplete",
        "passed": passed,
        "checks": checks,
    }


def assess_evidence_for_pilot(
    *,
    evaluation: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    personal_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Whether registered evidence supports a limited real-money pilot."""
    evaluation = evaluation if evaluation is not None else _load_json(
        config.POLYMARKET_PAPER_EVALUATION_REPORT_PATH
    )
    protocol = protocol if protocol is not None else _load_json(config.POLYMARKET_PAPER_PROTOCOL_PATH)
    policy = policy if policy is not None else _load_json(config.POLYMARKET_FROZEN_POLICY_PATH)

    no_bet_reasons: list[str] = []
    identity = verify_candidate_and_market(evaluation, protocol, policy)
    no_bet_reasons.extend(identity["reasons"])

    if evaluation is None:
        no_bet_reasons.append("evaluation_report_missing")
        verdict = "missing"
        validation_status = "missing"
    else:
        verdict = evaluation.get("verdict") or "insufficient_evidence"
        validation_status = evaluation.get("validation_status") or "insufficient_data"
        if verdict != "edge_supported":
            no_bet_reasons.append(f"evidence_verdict:{verdict}")
        if validation_status != "validated_passed":
            no_bet_reasons.append(f"validation_status:{validation_status}")
        for reason in list((evaluation.get("gates") or {}).get("gate_fail_reasons") or []):
            no_bet_reasons.append(f"gate:{reason}")
        n_dates = (evaluation.get("data_coverage") or {}).get(
            "n_polymarket_labeled_eligible_dates"
        )
        if n_dates is None or int(n_dates) < int(config.POLYMARKET_MIN_ELIGIBLE_DATES):
            no_bet_reasons.append(
                f"insufficient_labeled_dates:{n_dates}<{config.POLYMARKET_MIN_ELIGIBLE_DATES}"
            )

    if config.BETTING_MODE == "disabled":
        no_bet_reasons.append("BETTING_MODE=disabled")
    if config.GAME_PREDICTION_MODE != "live":
        no_bet_reasons.append(f"GAME_PREDICTION_MODE={config.GAME_PREDICTION_MODE!r}_not_live")

    limits_ok, limits_missing = personal_limits_complete(personal_limits)
    if not limits_ok:
        no_bet_reasons.append(f"personal_limits_incomplete:{','.join(limits_missing)}")
    elif str((personal_limits or {}).get("currency") or "").startswith("normalized_paper"):
        # Paper caps satisfy the limit schema for ops, but are not USD stakes.
        no_bet_reasons.append("risk_limits_are_paper_units_not_usd_bankroll")
        if (personal_limits or {}).get("real_money_usd_stakes_enabled") is True:
            no_bet_reasons.append("paper_limits_incorrectly_marked_as_usd")

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for r in no_bet_reasons:
        if r not in seen:
            seen.add(r)
            ordered.append(r)

    # Real-money pilot support requires zero no-bet reasons, including the
    # paper-units marker (USD bankroll must replace paper defaults).
    supports = len(ordered) == 0
    remaining = []
    if not supports:
        remaining = [
            "Accumulate ≥70 Polymarket-labeled eligible dates with nested evaluation passing.",
            "Frozen policy present and hash-matched to evaluation.",
            "Evidence verdict edge_supported and validation_status validated_passed.",
            "Replace paper-unit caps with explicit owner USD bankroll/max-loss if/when a pilot is considered.",
            "Owner confirms venue; only then consider enabling BETTING_MODE (still manual orders).",
            "Do not lower thresholds or treat market-only fallback as an independent model.",
        ]
    currency = (personal_limits or {}).get("currency") or "unset"
    return {
        "conclusion": "limited_real_money_pilot_supported" if supports else "pilot_not_supported",
        "supported": supports,
        "verdict": verdict,
        "validation_status": validation_status,
        "identity": identity,
        "no_bet_reasons": ordered,
        "remaining_before_pilot": remaining,
        "personal_limits_complete": limits_ok,
        "risk_limits_currency": currency,
        "stake_guidance_mode": (
            "usd"
            if (supports and limits_ok and currency == "USD")
            else "normalized_paper_units"
        ),
    }


def load_pause_state(path: str | None = None) -> dict[str, Any]:
    path = path or config.POLYMARKET_PILOT_PAUSE_STATE_PATH
    state = _load_json(path)
    if not state:
        return {
            "paused": False,
            "reasons": [],
            "paused_at_utc": None,
            "note": "Pauses do not erase existing exposure.",
            "existing_exposure_preserved": True,
        }
    return state


def write_pause_state(state: dict[str, Any], path: str | None = None) -> str:
    path = path or config.POLYMARKET_PILOT_PAUSE_STATE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = dict(state)
    state["existing_exposure_preserved"] = True
    state["note"] = "Pauses do not erase existing exposure."
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def evaluate_pause_conditions(
    *,
    operational_failure: bool = False,
    settled_real_net: float | None = None,
    max_affordable_loss: float | None = None,
    evidence_supported: bool = False,
    owner_pause: bool = False,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predefined pauses. Existing exposure is never erased by a pause."""
    reasons: list[str] = []
    if operational_failure:
        reasons.append("operational_failure")
    if (
        max_affordable_loss is not None
        and settled_real_net is not None
        and float(settled_real_net) <= -abs(float(max_affordable_loss))
    ):
        reasons.append("loss_limit_exceeded")
    if not evidence_supported:
        reasons.append("evidence_deterioration")
    if owner_pause:
        reasons.append("owner_pause")
    prev = previous or load_pause_state()
    paused = bool(reasons) or bool(prev.get("paused") and "owner_pause" in (prev.get("reasons") or []))
    # If only evidence_deterioration and software still works, still pause new bets.
    return {
        "paused": bool(reasons),
        "reasons": reasons,
        "reason_labels": [PAUSE_REASONS.get(r, r) for r in reasons],
        "paused_at_utc": utc_now_iso() if reasons else prev.get("paused_at_utc"),
        "existing_exposure_preserved": True,
        "new_bets_allowed": False if reasons else True,
        "previous_paused": bool(prev.get("paused")),
    }


def empty_real_fills() -> pd.DataFrame:
    return pd.DataFrame(columns=REAL_FILL_COLUMNS)


def load_real_fills(path: str | None = None) -> pd.DataFrame:
    path = path or config.POLYMARKET_PILOT_REAL_FILLS_PATH
    if not os.path.exists(path):
        return empty_real_fills()
    df = pd.read_csv(path)
    for col in REAL_FILL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[REAL_FILL_COLUMNS]


def record_real_fill(
    fill: dict[str, Any],
    *,
    path: str | None = None,
    pause_state: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    existing: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Append a manual real fill. Rejects when paused or exposure caps exceeded.

    Recording a fill that already happened is allowed even when the pilot is
    paused for *new* guidance — but new recommended exposure is blocked by
    ``guidance_allowed``. This function blocks duplicate open legs and caps.
    """
    path = path or config.POLYMARKET_PILOT_REAL_FILLS_PATH
    pause_state = pause_state if pause_state is not None else load_pause_state()
    frame = existing if existing is not None else load_real_fills(path)
    row = {
        "fill_id": fill.get("fill_id") or f"rf_{utc_now_iso().replace(':', '').replace('-', '')}",
        "recorded_at_utc": fill.get("recorded_at_utc") or utc_now_iso(),
        "decision_day": fill.get("decision_day") or str(fill.get("recorded_at_utc") or utc_now_iso())[:10],
        "market_id": str(fill.get("market_id") or ""),
        "market_slug": fill.get("market_slug"),
        "game_pk": fill.get("game_pk"),
        "side_team": str(fill.get("side_team") or "").upper(),
        "purchase_price": float(fill["purchase_price"]) if fill.get("purchase_price") is not None else None,
        "qty": float(fill.get("qty") or 0),
        "fees_paid": float(fill.get("fees_paid") or 0),
        "status": fill.get("status") or "open",
        "settlement_px": fill.get("settlement_px"),
        "settled_net": fill.get("settled_net"),
        "notes": fill.get("notes") or "",
        "stream": "real_manual",
    }
    if not row["market_id"] or not row["side_team"]:
        return {"accepted": False, "reasons": ["market_id_and_side_required"], "row": row}

    if row["status"] == "open" and pause_state.get("paused") and fill.get("force_record_during_pause") is not True:
        # Allow documenting an already-executed fill during pause only if flagged.
        return {
            "accepted": False,
            "reasons": ["pilot_paused_new_open_fills_blocked"] + list(pause_state.get("reasons") or []),
            "row": row,
        }

    positions = [
        {
            "units": float(r.qty),
            "game_pk": r.game_pk,
            "side_team": r.side_team,
            "decision_day": r.decision_day,
            "status": r.status,
        }
        for r in frame.itertuples(index=False)
    ]
    caps = limits or {}
    exp = decision_board.exposure_check(
        proposed_units=float(row["qty"]),
        existing_positions=positions,
        game_pk=row["game_pk"],
        team=row["side_team"],
        per_bet_cap=float(caps.get("per_bet_exposure_limit") or caps.get("per_bet") or config.POLYMARKET_PAPER_DEFAULT_PER_BET_UNITS),
        daily_cap=float(caps.get("daily_exposure_limit") or caps.get("daily") or config.POLYMARKET_PAPER_DEFAULT_DAILY_UNIT_CAP),
        same_game_cap=float(
            caps.get("same_game_exposure_limit")
            or caps.get("same_game")
            or config.POLYMARKET_PAPER_DEFAULT_SAME_GAME_UNIT_CAP
        ),
        same_team_cap=float(
            caps.get("same_team_exposure_limit")
            or caps.get("same_team")
            or config.POLYMARKET_PAPER_DEFAULT_SAME_TEAM_UNIT_CAP
        ),
        decision_day=str(row["decision_day"]),
    )
    if not exp["allowed"] and row["status"] == "open":
        return {"accepted": False, "reasons": exp["reasons"], "row": row}

    dup = frame[
        (frame["market_id"].astype(str) == row["market_id"])
        & (frame["side_team"].astype(str) == row["side_team"])
        & (frame["status"].astype(str) == "open")
    ]
    if row["status"] == "open" and not dup.empty:
        return {"accepted": False, "reasons": ["duplicate_open_market_side"], "row": row}

    if row["status"] == "settled" and row.get("settlement_px") is not None and row.get("purchase_price") is not None:
        proceeds = float(row["settlement_px"]) * float(row["qty"])
        cost = float(row["purchase_price"]) * float(row["qty"]) + float(row["fees_paid"] or 0)
        row["settled_net"] = proceeds - cost

    out = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return {"accepted": True, "reasons": [], "row": row, "path": path, "n_fills": int(len(out))}


def summarize_real_fills(fills: pd.DataFrame | None = None) -> dict[str, Any]:
    fills = fills if fills is not None else load_real_fills()
    if fills is None or fills.empty:
        return {
            "n_fills": 0,
            "n_open": 0,
            "n_settled": 0,
            "open_exposure": 0.0,
            "settled_net": 0.0,
            "stream": "real_manual",
        }
    open_rows = fills[fills["status"].astype(str) == "open"]
    settled = fills[fills["status"].astype(str) == "settled"]
    open_exposure = 0.0
    for _, r in open_rows.iterrows():
        px = float(r["purchase_price"]) if pd.notna(r.get("purchase_price")) else 0.0
        qty = float(r["qty"]) if pd.notna(r.get("qty")) else 0.0
        fees = float(r["fees_paid"]) if pd.notna(r.get("fees_paid")) else 0.0
        open_exposure += px * qty + fees
    settled_net = float(pd.to_numeric(settled["settled_net"], errors="coerce").fillna(0).sum()) if not settled.empty else 0.0
    return {
        "n_fills": int(len(fills)),
        "n_open": int(len(open_rows)),
        "n_settled": int(len(settled)),
        "open_exposure": open_exposure,
        "settled_net": settled_net,
        "stream": "real_manual",
    }


def reconcile_real_vs_paper(
    *,
    real_fills: pd.DataFrame | None = None,
    paper_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep real and paper result streams separate; report both."""
    real = summarize_real_fills(real_fills)
    paper = paper_summary or {}
    return {
        "generated_at_utc": utc_now_iso(),
        "streams_separate": True,
        "real_manual": real,
        "paper_ledger": {
            "source": "paper_ledger_or_evaluation_fixture",
            "settled_net": paper.get("settled_net"),
            "open_exposure": paper.get("open_exposure"),
            "roi": paper.get("roi"),
            "n_settled": paper.get("n_settled"),
            "note": "Paper results are not real-money P&L.",
        },
        "comparison_note": (
            "Do not merge streams. Real fills are owner-recorded; paper fills are simulated."
        ),
        "edge_claimed": False,
    }


def guidance_allowed(
    *,
    evidence: dict[str, Any],
    pause: dict[str, Any],
    quote_stale: bool,
    executable_buy: float | None,
    max_acceptable_price: float | None,
    missing_inputs: bool,
    exposure: dict[str, Any],
    probability_source: str | None,
    board_age_seconds: float | None = None,
) -> dict[str, Any]:
    """Recheck gates before any actionable / stake guidance."""
    reasons: list[str] = []
    if not evidence.get("supported"):
        reasons.append("missing_or_failed_evidence")
        reasons.extend(list(evidence.get("no_bet_reasons") or [])[:8])
    if pause.get("paused"):
        reasons.append("pilot_paused")
        reasons.extend(list(pause.get("reasons") or []))
    if missing_inputs:
        reasons.append("missing_inputs")
    if quote_stale:
        reasons.append("stale_quote")
    if board_age_seconds is not None and board_age_seconds > float(
        config.POLYMARKET_PILOT_BOARD_MAX_AGE_SECONDS
    ):
        reasons.append("stale_board_export")
    if executable_buy is None or max_acceptable_price is None:
        reasons.append("price_or_max_missing")
    elif float(executable_buy) > float(max_acceptable_price) + 1e-12:
        reasons.append("price_above_max_acceptable")
    if not exposure.get("allowed", True):
        reasons.append("excess_exposure")
        reasons.extend(list(exposure.get("reasons") or []))
    if probability_source in ("market_only_fallback", "market_mid_display_only"):
        reasons.append("market_only_not_independent_model")
    if config.BETTING_MODE == "disabled":
        reasons.append("BETTING_MODE=disabled")
    return {
        "allowed": len(reasons) == 0,
        "reasons": reasons,
        "actionable": False if reasons else True,
        "orders_automated": False,
    }


def classify_with_pilot_gates(
    *,
    base_state: str,
    base_reason: str | None,
    base_actionable: bool,
    pause: dict[str, Any],
    exposure_allowed: bool,
    exposure_reasons: list[str] | None = None,
) -> tuple[str, str | None, bool]:
    """Layer pause/exposure on top of decision_board states; never actionable for real money here."""
    if pause.get("paused"):
        return (
            config.POLYMARKET_DECISION_STATE_PASS_PAUSED,
            ",".join(pause.get("reasons") or ["pilot_paused"]),
            False,
        )
    if not exposure_allowed:
        return (
            config.POLYMARKET_DECISION_STATE_PASS_EXPOSURE,
            ",".join(exposure_reasons or ["excess_exposure"]),
            False,
        )
    return base_state, base_reason, False  # real-money actionable stays false until evidence+limits


def register_exposure_review_protocol(path: str | None = None) -> dict[str, Any]:
    """Rules for any later increase in exposure — register before raising stakes."""
    protocol = {
        "protocol_id": config.POLYMARKET_PILOT_EXPOSURE_REVIEW_PROTOCOL_ID,
        "registered_at_utc": config.POLYMARKET_PILOT_EXPOSURE_REVIEW_REGISTERED_UTC,
        "market_id": config.POLYMARKET_PILOT_MARKET_ID,
        "rules": {
            "never_increase_to_recover_losses": True,
            "never_increase_to_meet_income_deadline": True,
            "require_passing_evidence_at_review": True,
            "require_unchanged_or_improved_drawdown_vs_registered_baseline": True,
            "require_owner_reconfirmation_of_limits": True,
            "pauses_do_not_erase_exposure": True,
            "automated_orders_remain_absent": True,
        },
        "checkpoints_before_any_increase": [
            "Re-run write_pilot_readiness_report.py with current evaluation.",
            "Confirm verdict=edge_supported and validation_status=validated_passed.",
            "Confirm personal limits still within max affordable loss.",
            "Document proposed new caps and register a new review hash before applying.",
            "Keep paper and real ledgers separate in the review packet.",
        ],
        "edge_claimed": False,
    }
    protocol["protocol_hash"] = _sha(protocol)
    path = path or config.POLYMARKET_PILOT_EXPOSURE_REVIEW_PROTOCOL_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
        f.write("\n")
    return protocol


def build_pilot_readiness_report(
    *,
    personal_limits: dict[str, Any] | None = None,
    operational_failure: bool = False,
    owner_pause: bool = False,
) -> dict[str, Any]:
    software = assess_software_readiness()
    # Prefer explicit limits; else load saved; else owner-delegated paper defaults.
    if personal_limits is None:
        personal_limits = load_risk_limits() or agent_paper_risk_limits()
    evidence = assess_evidence_for_pilot(personal_limits=personal_limits)
    real = summarize_real_fills()
    limits_ok, _ = personal_limits_complete(personal_limits)
    max_loss = None
    if personal_limits:
        max_loss = personal_limits.get("max_affordable_loss", personal_limits.get("max_loss"))
    pause = evaluate_pause_conditions(
        operational_failure=operational_failure,
        settled_real_net=real.get("settled_net"),
        max_affordable_loss=float(max_loss) if max_loss is not None and limits_ok else None,
        evidence_supported=bool(evidence.get("supported")),
        owner_pause=owner_pause,
    )
    evaluation = _load_json(config.POLYMARKET_PAPER_EVALUATION_REPORT_PATH) or {}
    paper_fixture = (evaluation.get("paper_ledger_fixture") or {}).get("roi") or {}
    reconcile = reconcile_real_vs_paper(paper_summary=paper_fixture)

    report = {
        "generated_at_utc": utc_now_iso(),
        "market_id": config.POLYMARKET_PILOT_MARKET_ID,
        "protocol_id": config.POLYMARKET_PILOT_REQUIRED_PROTOCOL_ID,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "software": software,
        "evidence_for_limited_real_money_pilot": evidence,
        "risk_limits": personal_limits,
        "pause": pause,
        "reconcile": reconcile,
        "orders_automated": False,
        "edge_claimed": False,
        "stake_guidance": {
            "mode": evidence.get("stake_guidance_mode"),
            "paper_unit_label": config.POLYMARKET_PAPER_UNIT_LABEL,
            "personal_limits_required": True,
            "income_target_not_used": True,
            "real_money_usd_stakes_enabled": bool(
                (personal_limits or {}).get("real_money_usd_stakes_enabled")
            ),
        },
        "dual_conclusions": {
            "software_operates_correctly": bool(software.get("passed")),
            "evidence_supports_limited_real_money_pilot": bool(evidence.get("supported")),
        },
        "pilot_authorized": False,  # never auto-authorize from this report alone
    }
    # Pilot authorized only if both conclusions true AND not paused — still False while modes disabled.
    report["pilot_authorized"] = (
        report["dual_conclusions"]["software_operates_correctly"]
        and report["dual_conclusions"]["evidence_supports_limited_real_money_pilot"]
        and not pause.get("paused")
        and config.BETTING_MODE != "disabled"
        and str((personal_limits or {}).get("currency") or "") == "USD"
    )
    return report


def write_pilot_readiness_report(
    report: dict[str, Any] | None = None,
    *,
    path: str | None = None,
    personal_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = report or build_pilot_readiness_report(personal_limits=personal_limits)
    path = path or config.POLYMARKET_PILOT_READINESS_REPORT_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    pause_path = write_pause_state(report["pause"])
    recon_path = config.POLYMARKET_PILOT_RECONCILE_REPORT_PATH
    os.makedirs(os.path.dirname(recon_path) or ".", exist_ok=True)
    with open(recon_path, "w", encoding="utf-8") as f:
        json.dump(report["reconcile"], f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    report["_paths"] = {
        "readiness": path,
        "pause": pause_path,
        "reconcile": recon_path,
    }
    return report


def fee_as_of(as_of_utc: str | None = None):
    return paper_ledger.fee_as_of(as_of_utc)
