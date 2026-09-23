"""Operate frozen 1+ hit paper system on authorized host (no orders).

Loads the saved frozen policy, logs candidates/passes before outcomes,
reconciles the prospective ledger, writes ops/checkpoint reports, and
documents continuing collector + next registered review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mlb_metrics import config, hit_prop_ops, hit_prop_research, schedule


def _load_latest_contracts(date_iso: str | None = None) -> tuple[list[dict], dict]:
    store = hit_prop_research.research_store_dir()
    path = os.path.join(store, "contracts.csv")
    universe = {}
    if not os.path.exists(path):
        return [], {"note": "no_contracts_csv"}
    frame = pd.read_csv(path)
    if date_iso and "requested_local_date" in frame.columns:
        frame = frame[frame["requested_local_date"].astype(str) == date_iso]
    # Prefer latest capture_id for the date when present.
    if not frame.empty and "capture_id" in frame.columns:
        latest = frame["capture_id"].astype(str).iloc[-1]
        frame = frame[frame["capture_id"].astype(str) == latest]
    contracts = frame.to_dict(orient="records")
    universe = {
        "source": path,
        "n_rows_loaded": len(contracts),
        "requested_local_date_filter": date_iso,
        "complete_denominator_claimed": False,
        "note": (
            "Denominator is the loaded capture slice for this cycle; "
            "full-day universe remains in capture reports."
        ),
    }
    return contracts, universe


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None, help="Local date YYYY-MM-DD")
    parser.add_argument("--skip-cycle", action="store_true")
    parser.add_argument("--drill", action="store_true", help="Run outage/restart drill only")
    args = parser.parse_args()

    policy = hit_prop_ops.load_frozen_policy()
    hit_prop_ops.migrate_sim_ledger_away_from_prospective()

    if args.drill:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            drill = hit_prop_ops.outage_restart_drill(tmp)
        print(json.dumps({"drill": drill}, indent=2))
        return 0 if drill.get("status") == "passed" else 1

    date_iso = args.date or schedule.today_local().isoformat()
    cycle = None
    if not args.skip_cycle:
        contracts, universe = _load_latest_contracts(date_iso)
        # Attach capture universe if a matching capture JSON exists.
        store = hit_prop_research.research_store_dir()
        cap_dir = os.path.join(store, "captures")
        if os.path.isdir(cap_dir) and contracts:
            caps = sorted(Path(cap_dir).glob("*.json"))
            if caps:
                try:
                    payload = json.loads(caps[-1].read_text(encoding="utf-8"))
                    if payload.get("universe"):
                        universe = {
                            **universe,
                            **payload["universe"],
                            "capture_id": payload.get("capture_id"),
                        }
                except (OSError, json.JSONDecodeError):
                    pass
        cycle = hit_prop_ops.run_ops_cycle(
            contracts,
            policy=policy,
            persist=True,
            store_kind="prospective",
            universe=universe,
        )

    reconcile = hit_prop_ops.reconcile_prospective()
    ops = hit_prop_ops.build_ops_report(cycle=cycle, reconcile=reconcile, policy=policy)
    hit_prop_ops.write_json(config.HIT_PROP_OPS_REPORT_PATH, ops)
    checkpoint = {
        "generated_at_utc": hit_prop_ops.utc_now_iso(),
        "policy_version": policy.get("policy_version"),
        "policy_hash": policy.get("policy_hash"),
        **ops["checkpoint"],
        "elapsed_eligible_days": ops["elapsed_eligible_days"],
        "outstanding_evidence_needs": ops["outstanding_evidence_needs"],
        "coverage_and_accounting_only": True,
        "nested_performance_evaluation_allowed": ops["checkpoint"][
            "nested_performance_evaluation_allowed"
        ],
        "next_registered_review": ops["next_registered_review"],
        "research_only": True,
        "actionable": False,
        "edge_claimed": False,
    }
    hit_prop_ops.write_json(config.HIT_PROP_CHECKPOINT_REPORT_PATH, checkpoint)
    hit_prop_ops.write_json(config.HIT_PROP_RECONCILE_REPORT_PATH, reconcile)

    print(
        json.dumps(
            {
                "ops_report": config.HIT_PROP_OPS_REPORT_PATH,
                "checkpoint_report": config.HIT_PROP_CHECKPOINT_REPORT_PATH,
                "reconcile_report": config.HIT_PROP_RECONCILE_REPORT_PATH,
                "policy_version": policy.get("policy_version"),
                "elapsed_eligible_days": ops["elapsed_eligible_days"],
                "n_buy": (cycle or {}).get("n_buy"),
                "n_pass": (cycle or {}).get("n_pass"),
                "reconcile_status": reconcile.get("status"),
                "next_review": ops["next_registered_review"]["local_date"],
                "actionable": False,
                "edge_claimed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
