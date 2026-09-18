"""Write dual-conclusion pilot readiness report (no order placement).

Usage:
    PYTHONPATH=src python scripts/write_pilot_readiness_report.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, manual_pilot


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
    # Owner delegated risk-limit selection: persist agent paper-unit defaults.
    # These are not a personal USD bankroll and do not authorize a real-money pilot.
    limits = manual_pilot.write_risk_limits(manual_pilot.agent_paper_risk_limits())
    report = manual_pilot.write_pilot_readiness_report(personal_limits=limits)
    manual_pilot.register_exposure_review_protocol()
    paths = report.get("_paths") or {}
    print(f"Wrote {paths.get('readiness')}")
    print(f"Wrote {config.POLYMARKET_PILOT_RISK_LIMITS_PATH}")
    print(
        "paper_limits=",
        {
            "bankroll": limits["bankroll"],
            "max_affordable_loss": limits["max_affordable_loss"],
            "per_bet": limits["per_bet_exposure_limit"],
            "daily": limits["daily_exposure_limit"],
            "same_game": limits["same_game_exposure_limit"],
            "same_team": limits["same_team_exposure_limit"],
            "currency": limits["currency"],
        },
    )
    print(
        "software=",
        report["dual_conclusions"]["software_operates_correctly"],
        "evidence_pilot=",
        report["dual_conclusions"]["evidence_supports_limited_real_money_pilot"],
        "pilot_authorized=",
        report["pilot_authorized"],
    )
    print("no_bet_reasons:")
    for r in (report["evidence_for_limited_real_money_pilot"].get("no_bet_reasons") or [])[:12]:
        print(f"  - {r}")
    print("remaining_before_pilot:")
    for r in report["evidence_for_limited_real_money_pilot"].get("remaining_before_pilot") or []:
        print(f"  - {r}")
    # Also copy readiness into docs/data for the dashboard.
    docs_path = "docs/data/polymarket_pilot_readiness.json"
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    slim = {k: v for k, v in report.items() if not str(k).startswith("_")}
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    docs_limits = "docs/data/polymarket_pilot_risk_limits.json"
    with open(docs_limits, "w", encoding="utf-8") as f:
        json.dump(limits, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(f"Copied {docs_path}")
    print(f"Copied {docs_limits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
