"""Append simulated hit-prop paper ledger fixtures (NOT prospective history).

Writes under HIT_PROP_SIM_LEDGER_DIR only. Does not place orders.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_metrics import config, hit_prop_ops, hit_prop_paper, paper_ledger


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"

    example = hit_prop_paper.build_forecast_settlement_example()
    hit_prop_paper.write_forecast_settlement_example()

    paths = hit_prop_ops.sim_paths()
    os.makedirs(paths["dir"], exist_ok=True)

    trade = hit_prop_paper.simulate_prop_paper_trade(
        yes_asks=[{"price": 0.60, "size": 5.0}],
        decision_time_utc="2026-09-19T20:00:00Z",
        market_id="fixture-prop-ledger-1",
        market_slug="fixture-ledger-ausmar-gte1",
        game_pk=823976,
        key_mlbam=668885,
        player_name="Austin Martin",
        model_name="contract_rule_adjusted_hitter_hit_model",
        model_probability=(example.get("forecasts") or {})
        .get("baseball_contract_proxy", {})
        .get("contract_yes_probability"),
        market_mid_probability=(example.get("forecasts") or {}).get(
            "market_mid_from_executable"
        ),
        settlement=(example.get("settlement_classes") or {}).get("starter_with_hit"),
        requested_qty=1.0,
    )
    trade["decision"]["store_kind"] = "sim"
    trade["decision"]["research_only"] = True
    # Sim store: write decisions with store_kind tagged.
    frame = __import__("pandas").DataFrame([trade["decision"]])
    if os.path.exists(paths["decisions"]):
        prev = __import__("pandas").read_csv(paths["decisions"])
        frame = __import__("pandas").concat([prev, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["decision_id"], keep="first")
    frame.to_csv(paths["decisions"], index=False)

    position_row = {
        "decision_id": trade["decision"]["decision_id"],
        "status": trade["position"]["status"],
        "settlement_payout_per_contract": trade["position"].get(
            "settlement_payout_per_contract"
        ),
        "settlement_rule": trade["position"].get("settlement_rule"),
        "settled_at_utc": "2026-09-20T06:00:00Z",
        "proceeds": trade["position"].get("proceeds"),
        "net_pnl": trade["position"].get("net_pnl"),
        "open_exposure": trade["position"].get("open_exposure"),
        "acquisition_cost": trade["purchase"].get("acquisition_cost"),
        "filled_qty": trade["purchase"].get("filled_qty"),
        "winning_team": "YES",
        "revision_of": None,
    }
    paper_ledger.append_positions([position_row], path=paths["positions"])

    # Ensure legacy shared ledger is not treated as prospective.
    hit_prop_ops.migrate_sim_ledger_away_from_prospective()

    print(
        json.dumps(
            {
                "store_kind": "sim",
                "ledger_dir": paths["dir"],
                "decisions": paths["decisions"],
                "positions": paths["positions"],
                "example": config.HIT_PROP_FORECAST_EXAMPLE_PATH,
                "prospective_dir": config.HIT_PROP_PROSPECTIVE_DIR,
                "actionable": False,
                "edge_claimed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
