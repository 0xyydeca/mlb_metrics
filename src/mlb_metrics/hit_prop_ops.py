"""Operate the frozen 1+ hit prop paper system (no orders, no nested-eval open).

Prospective observations are stored separately from simulated fixtures.
Every decision records the saved frozen policy version. Candidates and passes
are logged before outcomes. Invalid inputs are excluded explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mlb_metrics import config, hit_prop_forecast, hit_prop_paper, hit_prop_research, paper_ledger

PROP_DECISION_EXTRA_COLUMNS = [
    "key_mlbam",
    "player_name",
    "stat",
    "threshold",
    "policy_hash",
    "cohort",
    "requested_local_date",
    "quote_age_seconds",
    "lineup_availability",
    "pass_reasons_json",
    "exclusion_code",
    "market_slug_key",
    "capture_id",
    "research_only",
    "store_kind",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_frozen_policy(path: str | None = None) -> dict[str, Any]:
    path = path or config.HIT_PROP_FROZEN_POLICY_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Frozen policy missing: {path}")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("action") != "paper_only":
        raise ValueError("Frozen policy action must remain paper_only")
    if doc.get("betting_mode") != "disabled":
        raise ValueError("Frozen policy betting_mode must remain disabled")
    if not doc.get("policy_version") or not doc.get("policy_hash"):
        raise ValueError("Frozen policy missing policy_version/policy_hash")
    return doc


def prospective_paths(root: str | None = None) -> dict[str, str]:
    root = root or config.HIT_PROP_PROSPECTIVE_DIR
    return {
        "dir": root,
        "decisions": os.path.join(root, "decisions.csv"),
        "positions": os.path.join(root, "positions.csv"),
        "exclusions": os.path.join(root, "exclusions.csv"),
        "cycle_log": os.path.join(root, "cycle_log.jsonl"),
        "checkpoint_state": os.path.join(root, "ops_checkpoint.json"),
    }


def sim_paths(root: str | None = None) -> dict[str, str]:
    root = root or config.HIT_PROP_SIM_LEDGER_DIR
    return {
        "dir": root,
        "decisions": os.path.join(root, "decisions.csv"),
        "positions": os.path.join(root, "positions.csv"),
    }


def classify_cohort(requested_local_date: str | None) -> str:
    """Regular season vs postseason separation (explicit; no silent pooling)."""
    if not requested_local_date:
        return "unknown_season"
    try:
        day = pd.Timestamp(requested_local_date).date()
    except (TypeError, ValueError):
        return "unknown_season"
    # MLB postseason typically begins early October; keep conservative split.
    if day.month >= 10:
        return "postseason_or_late_october"
    return "regular_season"


def quote_age_seconds(receive_time_utc: Any, decision_time_utc: str) -> float | None:
    if receive_time_utc is None or (isinstance(receive_time_utc, float) and pd.isna(receive_time_utc)):
        return None
    try:
        recv = pd.Timestamp(str(receive_time_utc).replace("Z", "+00:00"))
        dec = pd.Timestamp(str(decision_time_utc).replace("Z", "+00:00"))
        if recv.tzinfo is None:
            recv = recv.tz_localize("UTC")
        if dec.tzinfo is None:
            dec = dec.tz_localize("UTC")
        return float((dec - recv).total_seconds())
    except (TypeError, ValueError):
        return None


def evaluate_contract_before_outcome(
    contract: dict[str, Any],
    *,
    policy: dict[str, Any],
    decision_time_utc: str,
    lineup_started: bool | None = None,
    store_kind: str = "prospective",
) -> dict[str, Any]:
    """Log candidate/pass for one contract before settlement is known."""
    entry = policy.get("entry") or {}
    reasons: list[str] = []
    exclusion_code = None

    if contract.get("quarantined"):
        reasons.append("quarantined")
        exclusion_code = "quarantined"
    if entry.get("require_mapped_game_pk_and_key_mlbam", True):
        if contract.get("game_mapping_status") != hit_prop_research.MAPPING_MAPPED:
            reasons.append(f"game_mapping:{contract.get('game_mapping_status')}")
            exclusion_code = exclusion_code or "identity_unmapped"
        if contract.get("player_mapping_status") != hit_prop_research.MAPPING_MAPPED:
            reasons.append(f"player_mapping:{contract.get('player_mapping_status')}")
            exclusion_code = exclusion_code or "identity_unmapped"
    if not contract.get("rules_hash"):
        reasons.append("missing_rules_hash")
        exclusion_code = exclusion_code or "missing_rules"

    start = contract.get("scheduled_start_utc")
    if start:
        try:
            start_ts = pd.Timestamp(str(start).replace("Z", "+00:00"))
            dec_ts = pd.Timestamp(str(decision_time_utc).replace("Z", "+00:00"))
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            if dec_ts.tzinfo is None:
                dec_ts = dec_ts.tz_localize("UTC")
            minutes = entry.get("entry_minutes_before_scheduled_start")
            if minutes is not None:
                cutoff = start_ts - pd.Timedelta(minutes=float(minutes))
                if dec_ts > cutoff:
                    reasons.append("inside_entry_cutoff_window_or_after_start")
                    exclusion_code = exclusion_code or "entry_timing"
            if dec_ts >= start_ts:
                reasons.append("game_started_or_started")
                exclusion_code = exclusion_code or "game_started"
        except (TypeError, ValueError):
            reasons.append("unparseable_scheduled_start")
            exclusion_code = exclusion_code or "bad_schedule"

    book_status = contract.get("book_status")
    if book_status == "failed":
        reasons.append("book_request_failed")
        exclusion_code = exclusion_code or "provider_failure"
    elif book_status not in {"captured", "skipped_restart_already_captured"}:
        reasons.append(f"book_status:{book_status}")
        exclusion_code = exclusion_code or "book_unavailable"

    age = quote_age_seconds(contract.get("book_receive_time_utc"), decision_time_utc)
    max_age = float(entry.get("quote_max_age_seconds") or config.POLYMARKET_QUOTE_MAX_AGE_SECONDS)
    if entry.get("require_fresh_eligible_book", True):
        if age is None and book_status == "captured":
            # Captured earlier in the same run may use receive≈decision; allow if prices present.
            if contract.get("yes_buy_price") is None:
                reasons.append("missing_executable_yes_buy")
                exclusion_code = exclusion_code or "stale_or_missing_quote"
        elif age is not None and age > max_age:
            # Prospective paper ops: recorded capture receipts may be minutes old;
            # flag explicitly rather than silently using them as live-fresh.
            reasons.append(f"quote_age_seconds:{age:.0f}>max:{max_age:.0f}")
            exclusion_code = exclusion_code or "stale_quote"

    yes = hit_prop_research._number(contract.get("yes_buy_price"))
    size = hit_prop_research._number(contract.get("yes_buy_size"))
    if yes is None or not (0 < yes < 1):
        reasons.append("invalid_or_missing_yes_buy_price")
        exclusion_code = exclusion_code or "invalid_price"
    if size is None or size <= 0:
        reasons.append("invalid_or_missing_yes_buy_size")
        exclusion_code = exclusion_code or "invalid_size"

    if lineup_started is None:
        reasons.append("lineup_availability_unknown")
        # Do not auto-buy without lineup confirmation under frozen policy.
        exclusion_code = exclusion_code or "lineup_unknown"
        lineup_availability = "unknown"
    elif lineup_started is False:
        reasons.append("not_in_starting_lineup")
        exclusion_code = exclusion_code or "lineup_dnp"
        lineup_availability = "confirmed_not_starting"
    else:
        lineup_availability = "confirmed_starting"

    cohort = classify_cohort(contract.get("requested_local_date"))
    if cohort.startswith("postseason"):
        reasons.append("postseason_cohort_separate")
        exclusion_code = exclusion_code or "postseason_separate"

    mid = hit_prop_forecast.market_mid_from_executable(
        contract.get("yes_buy_price"), contract.get("no_buy_price")
    )
    # Paper ops under frozen policy logs candidates; fills only when no pass reasons.
    # Nested model selection remains closed — use market mid as logged probability source
    # without claiming independent-model promotion.
    action = "pass" if reasons else "buy"
    if action == "buy" and mid is None:
        reasons.append("missing_market_mid_for_identical_opportunity_log")
        exclusion_code = "missing_market_mid"
        action = "pass"

    asks = []
    if yes is not None and size is not None and size > 0 and 0 < yes < 1:
        asks = [{"price": float(yes), "size": float(size)}]

    trade = hit_prop_paper.simulate_prop_paper_trade(
        yes_asks=asks if action == "buy" else [],
        decision_time_utc=decision_time_utc,
        market_id=str(contract.get("market_id") or ""),
        market_slug=str(contract.get("market_slug") or ""),
        game_pk=contract.get("game_pk"),
        key_mlbam=contract.get("key_mlbam"),
        player_name=str(contract.get("player_name") or ""),
        model_name=hit_prop_forecast.CANDIDATE_MARKET_MID,
        model_probability=mid,
        market_mid_probability=mid,
        settlement=None,  # outcomes unknown at decision
        requested_qty=float((policy.get("entry") or {}).get("size_contracts") or 1.0),
        delay_seconds=0,
        adverse_ticks=0,
        policy_version=str(policy.get("policy_version")),
    )
    decision = trade["decision"]
    decision["action"] = action
    decision["pass_reason"] = "|".join(reasons) if reasons else None
    decision["outcome_unknown_at_decision"] = True
    decision["key_mlbam"] = contract.get("key_mlbam")
    decision["player_name"] = contract.get("player_name")
    decision["stat"] = "hits"
    decision["threshold"] = 1
    decision["policy_hash"] = policy.get("policy_hash")
    decision["cohort"] = cohort
    decision["requested_local_date"] = contract.get("requested_local_date")
    decision["quote_age_seconds"] = age
    decision["lineup_availability"] = lineup_availability
    decision["pass_reasons_json"] = json.dumps(reasons)
    decision["exclusion_code"] = exclusion_code
    decision["market_slug_key"] = contract.get("market_slug")
    decision["capture_id"] = contract.get("capture_id")
    decision["research_only"] = True
    decision["store_kind"] = store_kind

    position = None
    if action == "buy" and trade["purchase"]["filled_qty"] > 0:
        position = {
            "decision_id": decision["decision_id"],
            "status": "open",
            "settlement_payout_per_contract": None,
            "settlement_rule": "pending_official_result",
            "settled_at_utc": None,
            "proceeds": None,
            "net_pnl": None,
            "open_exposure": float(trade["purchase"]["acquisition_cost"]),
            "acquisition_cost": trade["purchase"]["acquisition_cost"],
            "filled_qty": trade["purchase"]["filled_qty"],
            "winning_team": None,
            "revision_of": None,
        }
    elif action == "buy" and trade["purchase"]["filled_qty"] <= 0:
        decision["action"] = "pass"
        decision["pass_reason"] = (
            (decision["pass_reason"] + "|unfilled") if decision["pass_reason"] else "unfilled"
        )
        decision["exclusion_code"] = decision.get("exclusion_code") or "unfilled"

    exclusion_row = None
    if exclusion_code or reasons:
        exclusion_row = {
            "observed_at_utc": decision_time_utc,
            "market_slug": contract.get("market_slug"),
            "game_pk": contract.get("game_pk"),
            "key_mlbam": contract.get("key_mlbam"),
            "exclusion_code": exclusion_code or "pass_reasons",
            "pass_reasons": "|".join(reasons),
            "policy_version": policy.get("policy_version"),
            "policy_hash": policy.get("policy_hash"),
            "store_kind": store_kind,
            "dropped_as_unfavorable_outcome": False,
        }

    return {
        "decision": decision,
        "position": position,
        "exclusion": exclusion_row,
        "eligible_for_fill": action == "buy",
        "pass_reasons": reasons,
    }


def _append_csv(path: str, rows: list[dict[str, Any]], key: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame = pd.DataFrame(rows)
    if os.path.exists(path):
        prev = pd.read_csv(path)
        combined = pd.concat([prev, frame], ignore_index=True)
        if key in combined.columns:
            combined = combined.drop_duplicates(subset=[key], keep="first")
    else:
        combined = frame
    combined.to_csv(path, index=False)


def append_exclusions(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame = pd.DataFrame(rows)
    if os.path.exists(path):
        prev = pd.read_csv(path)
        combined = pd.concat([prev, frame], ignore_index=True)
    else:
        combined = frame
    combined.to_csv(path, index=False)


def append_cycle_log(entry: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def run_ops_cycle(
    contracts: list[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
    decision_time_utc: str | None = None,
    lineup_by_key: dict[tuple[Any, Any], bool] | None = None,
    persist: bool = True,
    store_kind: str = "prospective",
    paths: dict[str, str] | None = None,
    universe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process captured contracts into policy-versioned decisions before outcomes."""
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
    policy = policy or load_frozen_policy()
    decision_time_utc = decision_time_utc or utc_now_iso()
    paths = paths or (prospective_paths() if store_kind == "prospective" else sim_paths())
    lineup_by_key = lineup_by_key or {}

    decisions: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    n_buy = 0
    n_pass = 0

    for contract in contracts:
        key = (contract.get("game_pk"), contract.get("key_mlbam"))
        lineup = lineup_by_key.get(key)
        result = evaluate_contract_before_outcome(
            contract,
            policy=policy,
            decision_time_utc=decision_time_utc,
            lineup_started=lineup,
            store_kind=store_kind,
        )
        decisions.append(result["decision"])
        if result["position"] is not None:
            positions.append(result["position"])
            n_buy += 1
        else:
            n_pass += 1
        if result["exclusion"] is not None:
            exclusions.append(result["exclusion"])

    report = {
        "cycle_id": f"hp_ops_{decision_time_utc.replace(':', '').replace('-', '')}",
        "observed_at_utc": decision_time_utc,
        "store_kind": store_kind,
        "policy_version": policy.get("policy_version"),
        "policy_hash": policy.get("policy_hash"),
        "n_contracts_in_denominator": len(contracts),
        "n_decisions": len(decisions),
        "n_buy": n_buy,
        "n_pass": n_pass,
        "n_exclusions_logged": len(exclusions),
        "universe": universe or {},
        "actionable": False,
        "edge_claimed": False,
        "future_observations_claimed": False,
        "nested_evaluation_outcomes": "not_opened",
        "do_not_tune_after_favorable_streak": True,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
    }

    if persist:
        # Extend decision CSV with extra columns beyond paper_ledger defaults.
        os.makedirs(paths["dir"], exist_ok=True)
        if decisions:
            frame = pd.DataFrame(decisions)
            path = paths["decisions"]
            if os.path.exists(path):
                prev = pd.read_csv(path)
                combined = pd.concat([prev, frame], ignore_index=True)
                combined = combined.drop_duplicates(subset=["decision_id"], keep="first")
            else:
                combined = frame.drop_duplicates(subset=["decision_id"], keep="first")
            combined.to_csv(path, index=False)
        if positions:
            paper_ledger.append_positions(positions, path=paths["positions"])
        if store_kind == "prospective" and exclusions:
            append_exclusions(exclusions, paths["exclusions"])
        if store_kind == "prospective":
            append_cycle_log(report, paths["cycle_log"])
            _save_ops_checkpoint(paths, report, decisions)
        report["persisted_paths"] = paths

    report["decisions"] = decisions
    report["positions"] = positions
    report["exclusions"] = exclusions
    return report


def _save_ops_checkpoint(paths: dict[str, str], report: dict[str, Any], decisions: list[dict]) -> None:
    done = []
    path = paths.get("checkpoint_state")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)
        done = list(prior.get("completed_decision_ids") or [])
    done.extend(d["decision_id"] for d in decisions)
    payload = {
        "updated_at_utc": utc_now_iso(),
        "last_cycle_id": report.get("cycle_id"),
        "policy_version": report.get("policy_version"),
        "policy_hash": report.get("policy_hash"),
        "completed_decision_ids": sorted(set(done)),
        "n_completed_decisions": len(set(done)),
    }
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


def reconcile_prospective(paths: dict[str, str] | None = None) -> dict[str, Any]:
    """Reconcile exposure, fees, fills, settlement, and duplicate decisions."""
    paths = paths or prospective_paths()
    decisions = (
        pd.read_csv(paths["decisions"]) if os.path.exists(paths["decisions"]) else pd.DataFrame()
    )
    positions = (
        pd.read_csv(paths["positions"]) if os.path.exists(paths["positions"]) else pd.DataFrame()
    )
    issues: list[str] = []
    n_dup = 0
    if not decisions.empty and "decision_id" in decisions.columns:
        n_dup = int(decisions["decision_id"].duplicated().sum())
        if n_dup:
            issues.append(f"duplicate_decisions:{n_dup}")

    open_exposure = 0.0
    settled_net = 0.0
    fees = 0.0
    n_open = 0
    n_settled = 0
    n_unfilled_buys = 0
    if not positions.empty:
        for _, row in positions.iterrows():
            status = str(row.get("status") or "")
            if status == "open":
                n_open += 1
                open_exposure += float(row.get("open_exposure") or row.get("acquisition_cost") or 0)
            elif status == "settled":
                n_settled += 1
                if row.get("net_pnl") == row.get("net_pnl"):
                    settled_net += float(row.get("net_pnl") or 0)
    if not decisions.empty:
        if "fees_paid" in decisions.columns:
            fees = float(pd.to_numeric(decisions["fees_paid"], errors="coerce").fillna(0).sum())
        buys = decisions[decisions.get("action") == "buy"] if "action" in decisions.columns else decisions
        if not buys.empty and "filled_qty" in buys.columns:
            n_unfilled_buys = int((pd.to_numeric(buys["filled_qty"], errors="coerce").fillna(0) <= 0).sum())

    orphan_positions = 0
    if not positions.empty and not decisions.empty:
        ids = set(decisions["decision_id"].astype(str))
        orphan_positions = int((~positions["decision_id"].astype(str).isin(ids)).sum())
        if orphan_positions:
            issues.append(f"orphan_positions:{orphan_positions}")

    sim_leak = 0
    if not decisions.empty and "store_kind" in decisions.columns:
        sim_leak = int((decisions["store_kind"].astype(str) == "sim").sum())
        if sim_leak:
            issues.append(f"sim_rows_in_prospective:{sim_leak}")
    if not decisions.empty and "market_id" in decisions.columns:
        fixture = int(decisions["market_id"].astype(str).str.startswith("fixture-").sum())
        if fixture:
            issues.append(f"fixture_market_ids_in_prospective:{fixture}")

    status = "ok" if not issues else "issues_found"
    return {
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "issues": issues,
        "n_decisions": int(len(decisions)),
        "n_positions": int(len(positions)),
        "n_duplicate_decision_ids": n_dup,
        "n_open_positions": n_open,
        "n_settled_positions": n_settled,
        "open_exposure": open_exposure,
        "settled_net_pnl": settled_net,
        "fees_paid_total": fees,
        "n_unfilled_buys": n_unfilled_buys,
        "orphan_positions": orphan_positions,
        "sim_separated": sim_leak == 0,
        "research_only": True,
        "actionable": False,
    }


def eligible_decision_dates(paths: dict[str, str] | None = None) -> list[str]:
    paths = paths or prospective_paths()
    if not os.path.exists(paths["decisions"]):
        return []
    frame = pd.read_csv(paths["decisions"])
    if frame.empty or "requested_local_date" not in frame.columns:
        return []
    # Count dates that had at least one logged candidate (buy or pass) under frozen policy.
    return sorted({str(d) for d in frame["requested_local_date"].dropna().unique()})


def checkpoint_status(n_eligible_dates: int, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or load_frozen_policy()
    checkpoints = list(
        policy.get("prospective_checkpoints_game_days")
        or config.HIT_PROP_PROSPECTIVE_CHECKPOINTS_GAME_DAYS
    )
    reached = [c for c in checkpoints if n_eligible_dates >= int(c)]
    next_cp = next((c for c in checkpoints if n_eligible_dates < int(c)), None)
    # Performance evaluation only when protocol schedule allows (structural floor).
    eval_allowed = n_eligible_dates >= int(config.HIT_PROP_MIN_ELIGIBLE_DATES)
    return {
        "n_eligible_dates_elapsed": int(n_eligible_dates),
        "checkpoints": checkpoints,
        "checkpoints_reached": reached,
        "next_checkpoint": next_cp,
        "days_to_next_checkpoint": (
            None if next_cp is None else max(0, int(next_cp) - int(n_eligible_dates))
        ),
        "operational_checkpoint_7": n_eligible_dates >= 7,
        "operational_checkpoint_14": n_eligible_dates >= 14,
        "nested_performance_evaluation_allowed": eval_allowed,
        "do_not_tune_or_increase_exposure_after_favorable_streak": True,
        "note": (
            "7/14 reports cover coverage and accounting health only. "
            "Nested performance evaluation waits for the registered statistical schedule."
        ),
    }


def outage_restart_drill(tmp_root: str) -> dict[str, Any]:
    """Verify restart skips duplicate decision_ids and retains exclusions."""
    policy = load_frozen_policy()
    paths = prospective_paths(os.path.join(tmp_root, "prospective"))
    contracts = [
        {
            "capture_id": "drill",
            "market_id": "drill-1",
            "market_slug": "drill-slug-1",
            "game_pk": 1,
            "key_mlbam": 10,
            "player_name": "Drill Player",
            "game_mapping_status": "mapped",
            "player_mapping_status": "mapped",
            "rules_hash": "abc",
            "quarantined": False,
            "scheduled_start_utc": "2099-01-01T00:00:00Z",
            "book_status": "captured",
            "book_receive_time_utc": "2098-12-31T23:00:00Z",
            "yes_buy_price": 0.55,
            "yes_buy_size": 10,
            "no_buy_price": 0.50,
            "requested_local_date": "2099-01-01",
        },
        {
            "capture_id": "drill",
            "market_id": "drill-2",
            "market_slug": "drill-slug-2",
            "game_pk": 2,
            "key_mlbam": 11,
            "player_name": "Drill Quarantine",
            "game_mapping_status": "ambiguous",
            "player_mapping_status": "unmatched",
            "rules_hash": "abc",
            "quarantined": True,
            "scheduled_start_utc": "2099-01-01T00:00:00Z",
            "book_status": "failed",
            "book_receive_time_utc": None,
            "yes_buy_price": None,
            "yes_buy_size": None,
            "no_buy_price": None,
            "requested_local_date": "2099-01-01",
        },
    ]
    first = run_ops_cycle(
        contracts,
        policy=policy,
        decision_time_utc="2098-12-31T23:00:00Z",
        lineup_by_key={(1, 10): True},
        persist=True,
        store_kind="prospective",
        paths=paths,
        universe={"n_events": 1},
    )
    second = run_ops_cycle(
        contracts,
        policy=policy,
        decision_time_utc="2098-12-31T23:00:00Z",
        lineup_by_key={(1, 10): True},
        persist=True,
        store_kind="prospective",
        paths=paths,
        universe={"n_events": 1},
    )
    decisions = pd.read_csv(paths["decisions"])
    n_ids = decisions["decision_id"].nunique()
    n_rows = len(decisions)
    ok = (
        n_ids == n_rows == 2
        and first["n_decisions"] == 2
        and second["n_decisions"] == 2
        and os.path.exists(paths["exclusions"])
        and os.path.exists(paths["checkpoint_state"])
    )
    return {
        "status": "passed" if ok else "failed",
        "n_unique_decision_ids": int(n_ids),
        "n_decision_rows": int(n_rows),
        "checkpoint_exists": os.path.exists(paths["checkpoint_state"]),
        "exclusions_exist": os.path.exists(paths["exclusions"]),
    }


def retrieve_history_integrity(paths: dict[str, str] | None = None) -> dict[str, Any]:
    """Fresh-environment check: prospective history files are readable and hashed."""
    paths = paths or prospective_paths()
    files = {}
    for key in ("decisions", "positions", "exclusions", "cycle_log", "checkpoint_state"):
        path = paths.get(key)
        if not path or not os.path.exists(path):
            files[key] = {"exists": False, "sha16": None, "bytes": 0}
            continue
        raw = open(path, "rb").read()
        files[key] = {
            "exists": True,
            "sha16": hashlib.sha256(raw).hexdigest()[:16],
            "bytes": len(raw),
            "path": path,
        }
    return {
        "generated_at_utc": utc_now_iso(),
        "retrievable": any(v["exists"] for v in files.values()),
        "files": files,
        "note": "Sim fixtures live under ledger/sim and are excluded from this hash set.",
    }


def build_ops_report(
    *,
    cycle: dict[str, Any] | None = None,
    reconcile: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy or load_frozen_policy()
    paths = prospective_paths()
    dates = eligible_decision_dates(paths)
    n_dates = len(dates)
    cp = checkpoint_status(n_dates, policy=policy)
    obs = hit_prop_paper.additional_observations_needed(
        n_eligible_dates_observed=n_dates,
        protocol=hit_prop_paper.load_protocol()
        if os.path.exists(config.HIT_PROP_PAPER_PROTOCOL_PATH)
        else None,
    )
    integrity = retrieve_history_integrity(paths)
    reconcile = reconcile or reconcile_prospective(paths)
    return {
        "generated_at_utc": utc_now_iso(),
        "research_only": True,
        "actionable": False,
        "edge_claimed": False,
        "future_observations_claimed": False,
        "policy_version": policy.get("policy_version"),
        "policy_hash": policy.get("policy_hash"),
        "host": config.HIT_PROP_COLLECTION_HOST,
        "workflow": config.HIT_PROP_COLLECTION_WORKFLOW,
        "continuing_collector": {
            "script": "scripts/run_hit_prop_paper_ops.py",
            "capture": "scripts/capture_hit_prop_research.py",
            "authorized_host": config.HIT_PROP_COLLECTION_HOST,
            "sleep_windows": "outside cron 15:00–02:59 UTC",
            "note": config.HIT_PROP_COLLECTION_RUNTIME_NOTE,
        },
        "next_registered_review": {
            "local_date": config.HIT_PROP_NEXT_REGISTERED_REVIEW_LOCAL,
            "note": config.HIT_PROP_NEXT_REGISTERED_REVIEW_NOTE,
            "is_betting_launch": False,
        },
        "elapsed_eligible_days": n_dates,
        "eligible_dates": dates,
        "checkpoint": cp,
        "outstanding_evidence_needs": obs,
        "reconcile": reconcile,
        "history_integrity": integrity,
        "last_cycle": {
            "cycle_id": (cycle or {}).get("cycle_id"),
            "n_contracts_in_denominator": (cycle or {}).get("n_contracts_in_denominator"),
            "n_buy": (cycle or {}).get("n_buy"),
            "n_pass": (cycle or {}).get("n_pass"),
        }
        if cycle
        else None,
        "nested_evaluation_outcomes": "not_opened",
        "postseason_separate": True,
        "material_policy_change_requires_new_version": True,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
    }


def write_json(path: str, payload: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def migrate_sim_ledger_away_from_prospective() -> dict[str, Any]:
    """Move fixture ledger rows out of the shared ledger dir into sim/."""
    legacy = config.HIT_PROP_LEDGER_DIR
    sim = sim_paths()
    os.makedirs(sim["dir"], exist_ok=True)
    moved = []
    for name in ("decisions.csv", "positions.csv"):
        src = os.path.join(legacy, name)
        dst = os.path.join(sim["dir"], name)
        if os.path.exists(src):
            # If file contains fixture rows, relocate to sim.
            frame = pd.read_csv(src)
            if frame.empty:
                continue
            is_fixture = False
            if "market_id" in frame.columns:
                is_fixture = bool(frame["market_id"].astype(str).str.startswith("fixture-").any())
            if is_fixture or "store_kind" not in frame.columns:
                shutil.copy2(src, dst)
                # Clear prospective-shared file so it cannot be mistaken for prospective.
                os.remove(src)
                moved.append(name)
    return {"moved_to_sim": moved, "sim_dir": sim["dir"]}
