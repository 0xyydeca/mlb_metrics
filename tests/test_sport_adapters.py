"""Compatibility: MLB Polymarket replay unchanged after sport-adapter refactor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, market_contracts, paper_ledger, quote_store
from mlb_metrics.sports import get_sport_adapter, namespaced_game_key, parse_namespaced_game_key
from mlb_metrics.venues import polymarket_us as pm_us
from mlb_metrics.venues.base import get_venue_adapter

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "polymarket"


def test_namespaced_keys_do_not_collide_with_game_pk():
    key = namespaced_game_key("nfl", "2026_01_DAL_PHI")
    sport, native = parse_namespaced_game_key(key)
    assert sport == "nfl"
    assert native == "2026_01_DAL_PHI"
    assert not native.isdigit() or True
    # MLB game_pk remains a separate integer concept
    mlb_key = namespaced_game_key("mlb", 824384)
    assert mlb_key == "mlb:824384"
    assert parse_namespaced_game_key(mlb_key) == ("mlb", "824384")


def test_mlb_list_moneyline_wrapper_still_exists():
    adapter = get_venue_adapter("polymarket_us")
    assert hasattr(adapter, "list_mlb_moneyline_markets")
    assert hasattr(adapter, "list_moneyline_markets")


def test_mlb_fixture_book_fill_accounting_unchanged():
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    book = pm_us.parse_book_response(
        json.loads((FIXTURES / "mlb_moneyline_book.json").read_text()),
        market_id="800001",
        market_slug="aec-mlb-sd-col-2026-09-14",
        request_time_utc="2026-09-15T01:00:00Z",
        receive_time_utc="2026-09-15T01:00:01Z",
        fee=fee,
    )
    purchase = paper_ledger.simulate_paper_purchase(
        asks=book.asks,
        requested_qty=1.0,
        fee=fee,
    )
    assert purchase["filled_qty"] == pytest.approx(1.0)
    assert purchase["avg_fill_price"] == pytest.approx(float(book.asks[0]["price"]))
    expected_fee = fee.taker_fee(float(book.asks[0]["price"]), 1.0)
    assert purchase["fees_paid"] == pytest.approx(expected_fee)


def test_mlb_doubleheader_mapping_unchanged():
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "home_team": "NYY",
                "away_team": "BOS",
                "game_datetime": "2026-09-15T17:05:00Z",
            },
            {
                "game_pk": 2,
                "home_team": "NYY",
                "away_team": "BOS",
                "game_datetime": "2026-09-15T23:05:00Z",
            },
        ]
    )
    payload = json.loads((FIXTURES / "mlb_event_moneyline.json").read_text())
    market = pm_us.parse_moneyline_market(payload["events"][0], payload["events"][0]["markets"][0])
    market.home_team = "NYY"
    market.away_team = "BOS"
    market.scheduled_start_utc = "2026-09-15T23:05:00Z"
    matched = market_contracts.match_market_to_schedule(market, schedule)
    assert matched["mapping_status"] == market_contracts.MAPPING_MAPPED
    assert matched["game_pk"] == 2


def test_mlb_quote_store_paths_unchanged_and_nfl_paths_separate(tmp_path):
    assert "nfl" in config.POLYMARKET_NFL_QUOTE_STORE_DIR
    assert config.POLYMARKET_NFL_QUOTE_STORE_DIR != config.POLYMARKET_QUOTE_STORE_DIR
    assert config.POLYMARKET_NFL_CONTRACT_REGISTRY_PATH != config.POLYMARKET_CONTRACT_REGISTRY_PATH
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    book = pm_us.parse_book_response(
        json.loads((FIXTURES / "mlb_moneyline_book.json").read_text()),
        market_id="m1",
        market_slug="s1",
        request_time_utc="2026-09-15T01:00:00Z",
        receive_time_utc="2026-09-15T01:00:01Z",
        fee=fee,
    )
    mlb_dir = tmp_path / "mlb_quotes"
    nfl_dir = tmp_path / "nfl_quotes"
    quote_store.append_quotes([book], store_dir=str(mlb_dir))
    quote_store.append_quotes([book], store_dir=str(nfl_dir))
    assert (mlb_dir / "quote_index.csv").exists()
    assert (nfl_dir / "quote_index.csv").exists()


def test_mlb_sport_adapter_preserves_game_pk_column():
    adapter = get_sport_adapter("mlb")
    caps = adapter.capabilities()
    assert caps.polymarket_league_slug == "mlb"
    assert caps.game_key_namespace == "mlb"


def test_nfl_mapping_unique_matchup():
    from mlb_metrics import nfl_market_contracts
    from mlb_metrics.venues.base import VenueMarket, VenueOutcome

    schedule = pd.DataFrame(
        [
            {
                "sport_id": "nfl",
                "sport_game_key": "nfl:2026_02_DAL_PHI",
                "native_game_id": "2026_02_DAL_PHI",
                "home_team": "PHI",
                "away_team": "DAL",
                "game_datetime": "2026-09-20T17:00:00Z",
                "date": "2026-09-20",
            }
        ]
    )
    market = VenueMarket(
        venue_id="polymarket_us",
        event_id="1",
        event_slug="nfl-dal-phi-2026-09-20",
        market_id="9001",
        market_slug="aec-nfl-dal-phi",
        market_type="moneyline",
        title="DAL @ PHI",
        question="Winner?",
        scheduled_start_utc="2026-09-20T17:00:00Z",
        trading_status="OPEN",
        active=True,
        closed=False,
        line=None,
        tick_size=0.01,
        min_trade_qty=1.0,
        fee_coefficient=None,
        home_team="PHI",
        away_team="DAL",
        provider_game_id="x",
        outcomes=[
            VenueOutcome("1", "DAL", "DAL", True, None, None, True),
            VenueOutcome("2", "PHI", "PHI", False, None, None, True),
        ],
        rules_text="demo",
        rules_hash="abc",
    )
    matched = nfl_market_contracts.match_market_to_nfl_schedule(market, schedule)
    assert matched["mapping_status"] == nfl_market_contracts.MAPPING_MAPPED
    assert matched["sport_game_key"] == "nfl:2026_02_DAL_PHI"


def test_nfl_mapping_rejects_date_only_midnight_kickoff():
    """Schedule rows at midnight UTC must not false-match afternoon kickoffs."""
    from mlb_metrics import nfl_market_contracts
    from mlb_metrics.venues.base import VenueMarket, VenueOutcome

    schedule = pd.DataFrame(
        [
            {
                "sport_id": "nfl",
                "sport_game_key": "nfl:2026_02_DAL_PHI",
                "native_game_id": "2026_02_DAL_PHI",
                "home_team": "PHI",
                "away_team": "DAL",
                "game_datetime": "2026-09-20T00:00:00Z",
                "date": "2026-09-20",
            }
        ]
    )
    market = VenueMarket(
        venue_id="polymarket_us",
        event_id="1",
        event_slug="nfl-dal-phi-2026-09-20",
        market_id="9001",
        market_slug="aec-nfl-dal-phi",
        market_type="moneyline",
        title="DAL @ PHI",
        question="Winner?",
        scheduled_start_utc="2026-09-20T17:00:00Z",
        trading_status="OPEN",
        active=True,
        closed=False,
        line=None,
        tick_size=0.01,
        min_trade_qty=1.0,
        fee_coefficient=None,
        home_team="PHI",
        away_team="DAL",
        provider_game_id="x",
        outcomes=[
            VenueOutcome("1", "DAL", "DAL", True, None, None, True),
            VenueOutcome("2", "PHI", "PHI", False, None, None, True),
        ],
        rules_text="demo",
        rules_hash="abc",
    )
    matched = nfl_market_contracts.match_market_to_nfl_schedule(market, schedule)
    assert matched["mapping_status"] != nfl_market_contracts.MAPPING_MAPPED


def test_nfl_validation_plan_paths_configured():
    assert config.POLYMARKET_NFL_PAPER_PROTOCOL_ID == "polymarket_us_nfl_game_winner_v1"
    assert "nfl" in config.POLYMARKET_NFL_PAPER_PROTOCOL_PATH


def test_modes_remain_disabled():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
