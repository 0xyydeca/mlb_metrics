"""Paper pipeline: delayed books, settlement, lineup invalidation, traces."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, paper_ledger, paper_pipeline, quote_store
from mlb_metrics.venues import polymarket_us as pm_us


FIXTURES_PM = Path(__file__).resolve().parent / "fixtures" / "polymarket"


def _store_two_quotes(tmp_path: Path) -> str:
    store = tmp_path / "quotes"
    fee = pm_us.select_fee_schedule(as_of_utc="2026-09-15T12:00:00Z")
    book_payload = json.loads((FIXTURES_PM / "mlb_moneyline_book.json").read_text())
    q1 = pm_us.parse_book_response(
        book_payload,
        market_id="m-delay",
        market_slug="slug-delay",
        request_time_utc="2026-09-15T12:00:00Z",
        receive_time_utc="2026-09-15T12:00:00Z",
        fee=fee,
    )
    worse = json.loads(json.dumps(book_payload))
    for o in worse["marketData"]["offers"]:
        o["px"]["value"] = f"{float(o['px']['value']) + 0.03:.4f}"
    q2 = pm_us.parse_book_response(
        worse,
        market_id="m-delay",
        market_slug="slug-delay",
        request_time_utc="2026-09-15T12:01:00Z",
        receive_time_utc="2026-09-15T12:01:00Z",
        fee=fee,
    )
    quote_store.append_quotes([q1, q2], store_dir=str(store))
    return str(store)


def test_manual_delay_uses_later_book_not_decision_book(tmp_path):
    store = _store_two_quotes(tmp_path)
    quotes = quote_store.load_quotes_for_market("m-delay", store_dir=store)
    early = quote_store.quote_as_of("m-delay", as_of_utc="2026-09-15T12:00:00Z", quotes=quotes)
    delayed = quote_store.quote_as_of("m-delay", as_of_utc="2026-09-15T12:01:00Z", quotes=quotes)
    assert early is not None and delayed is not None
    asks0 = quote_store.asks_from_quote_row(early)
    asks60 = quote_store.asks_from_quote_row(delayed)
    assert asks60[0]["price"] > asks0[0]["price"]
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    purchase = paper_ledger.purchase_after_manual_delay(
        asks_at_decision=asks0,
        asks_after_delay=asks60,
        requested_qty=1.0,
        fee=fee,
        delay_seconds=60,
    )
    assert purchase["avg_fill_price"] == pytest.approx(asks60[0]["price"])


def test_trace_fails_closed_when_delay_lacks_newer_book(tmp_path):
    store = tmp_path / "quotes"
    fee = pm_us.select_fee_schedule(as_of_utc="2026-09-15T12:00:00Z")
    book_payload = json.loads((FIXTURES_PM / "mlb_moneyline_book.json").read_text())
    q1 = pm_us.parse_book_response(
        book_payload,
        market_id="only-one",
        market_slug="slug-one",
        request_time_utc="2026-09-15T12:00:00Z",
        receive_time_utc="2026-09-15T12:00:00Z",
        fee=fee,
    )
    quote_store.append_quotes([q1], store_dir=str(store))
    registry = pd.DataFrame(
        [
            {
                "market_id": "only-one",
                "market_slug": "slug-one",
                "game_pk": 1,
                "home_team": "COL",
                "away_team": "SD",
                "long_team": "SD",
                "short_team": "COL",
                "venue_id": "polymarket_us",
                "mapping_status": "mapped",
                "scheduled_start_utc": "2099-01-01T00:00:00Z",
                "event_slug": "demo",
                "rules_hash": "abc",
                "rules_text": "demo rules",
            }
        ]
    )
    reg_path = tmp_path / "reg.csv"
    registry.to_csv(reg_path, index=False)
    trace = paper_pipeline.trace_observed_contract(
        market_id="only-one",
        store_dir=str(store),
        registry_path=str(reg_path),
        delay_seconds=60,
    )
    assert trace["ok"] is True
    assert trace["paper_purchase"]["status"] == "unfilled"
    assert trace["paper_purchase"].get("pass_reason") == "missing_quote_after_manual_delay" or trace[
        "paper_purchase"
    ]["filled_qty"] == 0.0


def test_missing_delayed_book_does_not_reuse_decision_book():
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    purchase = paper_ledger.purchase_after_manual_delay(
        asks_at_decision=[{"price": 0.40, "size": 5.0}],
        asks_after_delay=[],
        requested_qty=1.0,
        fee=fee,
        delay_seconds=60,
    )
    assert purchase["status"] == "unfilled"
    assert purchase["pass_reason"] == "missing_quote_after_manual_delay"


def test_lineup_change_after_cutoff_cannot_rewrite_earlier_snapshot():
    early = pd.DataFrame(
        [
            {
                "game_pk": 99,
                "date": "2026-09-15",
                "game_datetime": "2026-09-15T23:00:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
                "status": "Scheduled",
                "home_probable_pitcher_key_mlbam": 1,
                "away_probable_pitcher_key_mlbam": 2,
                "home_starter_status": "probable",
                "away_starter_status": "probable",
                "home_lineup_status": "unconfirmed",
                "away_lineup_status": "unconfirmed",
                "home_confirmed_starter_count": 0,
                "away_confirmed_starter_count": 0,
                "lineup_batting_order_source": "list_index",
                "usable_for_game_winner": True,
                "pass_reason": None,
                "fetched_at_utc": "2026-09-15T12:00:00Z",
                "source": "test",
                "as_of_limitation": None,
            }
        ]
    )
    late = early.copy()
    late["fetched_at_utc"] = "2026-09-15T18:00:00Z"
    late["home_lineup_status"] = "confirmed"
    late["home_confirmed_starter_count"] = 9
    combined = pd.concat([early, late], ignore_index=True)
    from mlb_metrics import game_baseball_snapshots

    kept = game_baseball_snapshots.filter_snapshots_as_of(
        combined, as_of_utc="2026-09-15T13:00:00Z"
    )
    assert len(kept) == 1
    assert kept.iloc[0]["home_lineup_status"] == "unconfirmed"


def test_settle_open_positions_from_final_results(tmp_path):
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    purchase = paper_ledger.simulate_paper_purchase(
        asks=[{"price": 0.5, "size": 1.0}],
        requested_qty=1.0,
        fee=fee,
    )
    decision = paper_ledger.build_decision_row(
        policy_version="t",
        venue_id="polymarket_us",
        market_id="m",
        market_slug="s",
        game_pk=42,
        side_team="NYY",
        is_long=True,
        decision_time_utc="2026-09-15T12:00:00Z",
        quote_receive_time_utc="2026-09-15T12:00:00Z",
        quote_request_time_utc="2026-09-15T12:00:00Z",
        fee=fee,
        action="buy",
        pass_reason=None,
        requested_qty=1.0,
        fill_assumption="walk_asks",
        purchase=purchase,
        model_name="t",
        model_probability=0.6,
        market_mid_probability_value=0.5,
        executable_buy=0.5,
        probability_source="test",
    )
    open_pos = paper_ledger.build_position_row(
        decision_id_value=decision["decision_id"],
        settlement=paper_ledger.settle_position(
            filled_qty=1.0,
            acquisition_cost=purchase["acquisition_cost"],
            selected_team="NYY",
            winning_team=None,
        ),
        acquisition_cost=purchase["acquisition_cost"],
        filled_qty=1.0,
    )
    dpath = tmp_path / "decisions.csv"
    ppath = tmp_path / "positions.csv"
    paper_ledger.append_decisions([decision], path=str(dpath))
    paper_ledger.append_positions([open_pos], path=str(ppath))
    results = pd.DataFrame(
        [
            {
                "game_pk": 42,
                "status": "Final",
                "home_team": "NYY",
                "away_team": "BOS",
                "home_score": 5,
                "away_score": 3,
            }
        ]
    )
    out = paper_pipeline.settle_open_positions(
        results=results,
        decisions_path=str(dpath),
        positions_path=str(ppath),
        settled_at_utc="2026-09-16T06:00:00Z",
    )
    assert out["n_settled"] == 1
    positions = pd.read_csv(ppath)
    assert positions.iloc[0]["status"] == "settled"
    assert float(positions.iloc[0]["proceeds"]) == pytest.approx(1.0)
    roi = paper_ledger.settled_roi(positions, pd.read_csv(dpath))
    assert roi["n_settled"] == 1
    assert roi["settled_net"] == pytest.approx(1.0 - purchase["acquisition_cost"])


def test_postponed_settlement_is_not_auto_zero(tmp_path):
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    purchase = paper_ledger.simulate_paper_purchase(
        asks=[{"price": 0.4, "size": 1.0}], requested_qty=1.0, fee=fee
    )
    decision = paper_ledger.build_decision_row(
        policy_version="t",
        venue_id="polymarket_us",
        market_id="m2",
        market_slug="s2",
        game_pk=7,
        side_team="LAD",
        is_long=True,
        decision_time_utc="2026-09-15T12:00:00Z",
        quote_receive_time_utc="2026-09-15T12:00:00Z",
        quote_request_time_utc=None,
        fee=fee,
        action="buy",
        pass_reason=None,
        requested_qty=1.0,
        fill_assumption="walk_asks",
        purchase=purchase,
        model_name="t",
        model_probability=0.55,
        market_mid_probability_value=0.5,
        executable_buy=0.4,
        probability_source="test",
    )
    open_pos = paper_ledger.build_position_row(
        decision_id_value=decision["decision_id"],
        settlement=paper_ledger.settle_position(
            filled_qty=1.0,
            acquisition_cost=purchase["acquisition_cost"],
            selected_team="LAD",
            winning_team=None,
        ),
        acquisition_cost=purchase["acquisition_cost"],
        filled_qty=1.0,
    )
    dpath = tmp_path / "d.csv"
    ppath = tmp_path / "p.csv"
    paper_ledger.append_decisions([decision], path=str(dpath))
    paper_ledger.append_positions([open_pos], path=str(ppath))
    out = paper_pipeline.settle_open_positions(
        results=pd.DataFrame(
            [
                {
                    "game_pk": 7,
                    "status": "Postponed",
                    "home_team": "LAD",
                    "away_team": "SF",
                    "home_score": None,
                    "away_score": None,
                }
            ]
        ),
        decisions_path=str(dpath),
        positions_path=str(ppath),
    )
    assert out["n_canceled_marked"] == 1
    positions = pd.read_csv(ppath)
    assert positions.iloc[0]["status"] == "canceled"
    assert positions.iloc[0]["net_pnl"] != positions.iloc[0]["net_pnl"] or pd.isna(
        positions.iloc[0]["net_pnl"]
    )  # NaN — not auto zero


def test_evidence_gate_suppresses_production_buys(tmp_path):
    store = _store_two_quotes(tmp_path)
    registry = pd.DataFrame(
        [
            {
                "market_id": "m-delay",
                "market_slug": "slug-delay",
                "game_pk": 1,
                "home_team": "COL",
                "away_team": "SD",
                "long_team": "SD",
                "venue_id": "polymarket_us",
                "mapping_status": "mapped",
                "scheduled_start_utc": "2099-01-01T00:00:00Z",
                "event_slug": "demo",
            }
        ]
    )
    baseball = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "date": "2099-01-01",
                "game_datetime": "2099-01-01T00:00:00Z",
                "home_team": "COL",
                "away_team": "SD",
                "status": "Scheduled",
                "home_probable_pitcher_key_mlbam": 1,
                "away_probable_pitcher_key_mlbam": 2,
                "home_starter_status": "probable",
                "away_starter_status": "probable",
                "home_lineup_status": "confirmed",
                "away_lineup_status": "confirmed",
                "home_confirmed_starter_count": 9,
                "away_confirmed_starter_count": 9,
                "lineup_batting_order_source": "list_index",
                "usable_for_game_winner": True,
                "pass_reason": None,
                "fetched_at_utc": "2026-09-15T12:00:00Z",
                "source": "test",
                "as_of_limitation": None,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "home_team": "COL",
                "away_team": "SD",
                "predicted_winner": "SD",
                "predicted_probability": 0.70,
            }
        ]
    )
    summary = paper_pipeline.run_paper_decision_cycle(
        registry=registry,
        baseball=baseball,
        predictions=predictions,
        decision_time_utc="2026-09-15T12:00:00Z",
        delay_seconds=0,
        adverse_ticks=0,
        store_dir=store,
        allow_exploratory_fills=False,
        decisions_path=str(tmp_path / "dec.csv"),
        positions_path=str(tmp_path / "pos.csv"),
        write=True,
    )
    assert summary["n_buy"] == 0
    assert summary["n_pass"] == 2
    assert config.BETTING_MODE == "disabled"
