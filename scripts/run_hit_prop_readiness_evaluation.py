"""Formal readiness evaluation for frozen 1+ hit prop paper system.

Does NOT place orders. Does NOT flip BETTING_MODE. Does NOT lower thresholds.
Software tests are not evidence of edge.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_metrics import config, hit_prop_paper, hit_prop_readiness


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"

    report = hit_prop_readiness.build_readiness_report(allow_scoring=True)
    path = hit_prop_readiness.write_readiness_report(report)

    # Keep the paper evaluation artifact aligned with the formal verdict.
    eval_mirror = {
        "generated_at_utc": report["generated_at_utc"],
        "protocol_id": report["protocol_id"],
        "protocol_hash": report["protocol_hash"],
        "policy_version": report["policy_version"],
        "policy_hash": report["policy_hash"],
        "validation_status": report["validation_status"],
        "verdict": report["verdict"],
        "gates": report["gates"],
        "probability_report": report["probability_report"],
        "strategy_report": report["strategy_report"],
        "additional_observations_needed": report["additional_observations_needed"],
        "structural_and_precision": report["structural_and_precision"],
        "pre_score_audit_status": report["pre_score_audit"]["status"],
        "readiness_report_path": path,
        "readiness_report_hash": report.get("report_hash"),
        "research_only": True,
        "actionable": False,
        "edge_claimed": False,
        "future_observations_claimed": False,
        "betting_mode_required": "disabled",
        "modes": report["modes"],
        "software_tests_are_not_evidence_of_edge": True,
        "nested_evaluation_outcomes": (
            "not_opened"
            if report["scoring_blocked_reasons"]
            else "scored_under_registered_gates"
        ),
    }
    hit_prop_paper.write_evaluation_report(eval_mirror)

    print(
        json.dumps(
            {
                "readiness_report": path,
                "verdict": report["verdict"],
                "validation_status": report["validation_status"],
                "n_eligible_dates": report["denominators"]["n_eligible_independent_dates"],
                "additional_dates_to_floor": report["additional_observations_needed"][
                    "additional_dates_to_structural_floor"
                ],
                "next_checkpoint": report["next_permitted_checkpoint"][
                    "next_operational_or_structural_checkpoint_dates"
                ],
                "next_review": report["next_permitted_checkpoint"][
                    "next_registered_review_local"
                ],
                "scoring_attempted": report["scoring_attempted"],
                "betting_mode": config.BETTING_MODE,
                "edge_claimed": False,
            },
            indent=2,
        )
    )
    # Exit 0 even on insufficient/unsupported — those are valid registered results.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
