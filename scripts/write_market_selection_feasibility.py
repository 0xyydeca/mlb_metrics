"""Write market-selection feasibility report (after protocol registration).

Uses official API docs, schedule availability, and in-repo capabilities.
Does not run profitable backtests. Does not purchase data.

Usage:
    PYTHONPATH=src python scripts/register_market_selection_protocol.py
    PYTHONPATH=src python scripts/write_market_selection_feasibility.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
    assert os.path.exists(config.MARKET_SELECTION_PROTOCOL_PATH), (
        "Register market selection protocol before writing feasibility outcomes"
    )
    with open(config.MARKET_SELECTION_PROTOCOL_PATH, encoding="utf-8") as f:
        protocol = json.load(f)

    # Feasibility statuses from documentation + repo capabilities only.
    # Observed live sample sizes are filled if a prior NFL capture report exists.
    nfl_coverage = {}
    if os.path.exists(config.POLYMARKET_NFL_COVERAGE_REPORT_PATH):
        with open(config.POLYMARKET_NFL_COVERAGE_REPORT_PATH, encoding="utf-8") as f:
            nfl_coverage = json.load(f)

    report = {
        "generated_at_utc": __import__("mlb_metrics.polymarket_research", fromlist=["utc_now_iso"]).utc_now_iso(),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_hash": protocol.get("protocol_hash"),
        "sources": [
            "https://docs.polymarket.us/api-reference/sports/overview",
            "https://docs.polymarket.us/data-guide/sports-data",
            "https://docs.polymarket.us/fees",
            "https://docs.polymarket.us/faqs/sports-faqs",
            "In-repo: nfl_data.py, nfl_game_predictions.py, sports/* adapters, venues/polymarket_us.py",
        ],
        "candidates": {
            "nfl_pregame_moneyline": {
                "status": "feasible",
                "contract_availability": (
                    "Polymarket US documents league slug nfl via GET /v2/leagues/nfl/events; "
                    "moneyline type SPORTS_MARKET_TYPE_MONEYLINE."
                ),
                "settlement_complexity": (
                    "Binary game winner similar to MLB; independent games; "
                    "cancel/postpone rules still require contract text (not assumed)."
                ),
                "historical_labels": (
                    "In-repo nflreadpy schedules/results via nfl_data.fetch_schedules; "
                    "existing nfl_game_predictions pipeline with game_id."
                ),
                "prices_fees_collection": (
                    "Same US fee schedule machinery as MLB; books via /v1/markets/{slug}/book. "
                    "Liquidity not assumed; empty books remain liquidity failures."
                ),
                "opportunity_frequency": (
                    "Roughly Thu/Sun/Mon game days (~2–3 independent dates/week). "
                    "One NFL season alone is unlikely to reach the 70-date structural floor."
                ),
                "reusable_code": "High — venue adapter, quote_store, paper_ledger, NFL predictions.",
                "sport_specific_work_remaining": (
                    "Mapping QA, paper protocol for NFL, prediction↔Polymarket prior join, "
                    "settlement exceptions, denser near-kickoff capture."
                ),
                "estimated_engineering_days": "2–4 for capture/mapping; 5–10 before nested eval readiness",
                "observed_sample": {
                    "n_markets_last_capture": nfl_coverage.get("n_markets"),
                    "n_mapped_last_capture": nfl_coverage.get("n_mapped"),
                    "n_books_last_capture": nfl_coverage.get("n_books_captured"),
                    "note": "Null until capture_polymarket_nfl.py has been run.",
                },
                "unknowns_that_could_reverse": [
                    "Persistently thin or absent NFL moneyline books on provisional US venue",
                    "Owner actually uses international venue with different contracts",
                    "Settlement rules for postponed/neutral-site games diverge materially",
                ],
            },
            "nba_pregame_moneyline": {
                "status": "uncertain",
                "contract_availability": "League slug nba documented on Polymarket US sports API.",
                "settlement_complexity": "Binary winners; overtime included typically — verify per contract.",
                "historical_labels": (
                    "No first-party NBA prediction/schedule stack in this repo yet."
                ),
                "prices_fees_collection": "Same venue book/fee stack; liquidity unknown until sampled.",
                "opportunity_frequency": (
                    "Near-daily Oct–Apr — better calendar span toward a 70-date floor than NFL."
                ),
                "reusable_code": "Medium — venue/quotes/paper only; no NBA features/models.",
                "sport_specific_work_remaining": (
                    "Schedule adapter, team abbreviations, prediction targets, labels, protocol."
                ),
                "estimated_engineering_days": "5–10 before useful capture+labels; longer before models",
                "unknowns_that_could_reverse": [
                    "2026–27 tip-off timing vs collection start",
                    "Missing public schedule/label pipeline quality",
                ],
            },
            "nhl_pregame_moneyline": {
                "status": "uncertain",
                "contract_availability": "League slug nhl documented on Polymarket US sports API.",
                "settlement_complexity": (
                    "Regulation vs include-OT/SO can differ by contract — higher rule risk than NFL/NBA."
                ),
                "historical_labels": "No NHL stack in this repo.",
                "prices_fees_collection": "Same venue stack; liquidity unknown.",
                "opportunity_frequency": "Near-daily in season; similar calendar advantage to NBA.",
                "reusable_code": "Medium — venue/quotes/paper only.",
                "sport_specific_work_remaining": "Full sport adapter + settlement-rule handling.",
                "estimated_engineering_days": "5–12 before useful labeled capture",
                "unknowns_that_could_reverse": [
                    "OT/SO settlement ambiguity",
                    "Book coverage gaps",
                ],
            },
        },
        "selection": {
            "selected_id": "nfl_pregame_moneyline",
            "decision": "proceed_minimal_capture_only",
            "rationale": (
                "NFL is the only candidate with both documented Polymarket US league events "
                "and an existing in-repo schedule/prediction identity (game_id). Selection is "
                "for research infrastructure reuse and immediate public-data capture — not "
                "because any backtest showed profit. NBA/NHL remain uncertain pending sport "
                "adapters; they may later beat NFL on date density."
            ),
            "not_selected_alternatives": ["nba_pregame_moneyline", "nhl_pregame_moneyline"],
            "no_expansion_would_be_correct_if": (
                "NFL books are systematically missing/ineligible on the provisional US venue, "
                "or the owner confirms a different venue that does not list NFL moneylines."
            ),
            "gates": {
                "model_development": "blocked_until_nfl_paper_protocol_and_validation",
                "real_money": "blocked",
                "mlb_collection": "continues_unchanged",
            },
        },
        "evidence_collection_requirements_if_nfl": {
            "register_nfl_paper_protocol_before_nested_outcomes": True,
            "structural_floor_dates": config.POLYMARKET_MIN_ELIGIBLE_DATES,
            "note": (
                "One NFL season of game-days is unlikely to meet the 70-date floor; "
                "multi-season collection or an explicit NFL date-block redesign would be required. "
                "That does not block starting capture and mapping tests."
            ),
        },
        "future_observations_claimed": False,
        "edge_claimed": False,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
    }

    path = config.MARKET_SELECTION_FEASIBILITY_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(f"Wrote {path}")
    print(f"selected={report['selection']['selected_id']} decision={report['selection']['decision']}")
    print("edge_claimed=False; betting remains disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
