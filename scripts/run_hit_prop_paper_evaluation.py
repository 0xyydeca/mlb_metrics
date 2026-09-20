"""Run hit-prop paper evaluation (insufficient_data expected; no nested outcomes).

Does NOT place orders. Does NOT flip BETTING_MODE / GAME_PREDICTION_MODE.
September 25 is a paper-system review, not a betting launch.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, hit_prop_paper


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"

    report = hit_prop_paper.build_evaluation_report(open_nested_outcomes=False)
    path = hit_prop_paper.write_evaluation_report(report)
    print(
        json.dumps(
            {
                "report": path,
                "validation_status": report["validation_status"],
                "verdict": report["verdict"],
                "n_collection_dates": report["collection_progress"]["n_dates"],
                "additional_dates_to_floor": report["additional_observations_needed"][
                    "additional_dates_to_structural_floor"
                ],
                "nested_evaluation_outcomes": report["nested_evaluation_outcomes"],
                "edge_claimed": False,
                "betting_mode": config.BETTING_MODE,
                "example": config.HIT_PROP_FORECAST_EXAMPLE_PATH,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
