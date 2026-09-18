"""Register exposure-increase review checkpoints (before any stake raise).

Usage:
    PYTHONPATH=src python scripts/register_pilot_exposure_review.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, manual_pilot


def main() -> int:
    assert config.BETTING_MODE == "disabled"
    protocol = manual_pilot.register_exposure_review_protocol()
    print(f"Wrote {config.POLYMARKET_PILOT_EXPOSURE_REVIEW_PROTOCOL_PATH}")
    print(f"protocol_hash={protocol['protocol_hash']}")
    print("never_increase_to_recover_losses=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
