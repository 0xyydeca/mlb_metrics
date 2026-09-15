"""Polymarket US adapter, registry mapping, and quote store tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, market_contracts, quote_store
from mlb_metrics.venues import get_venue_adapter
from mlb_metrics.venues.base import UnsupportedVenueError
from mlb_metrics.venues import polymarket_international as pm_intl
from mlb_metrics.venues import polymarket_us as pm_us

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "polymarket"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_selected_venue_is_polymarket_us_and_modes_unchanged():
    assert config.POLYMARKET_VENUE_SELECTED == "polymarket_us"
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
    adapter = get_venue_adapter()
    assert adapter.venue_id == "polymarket_us"
    assert adapter.capabilities().order_placement is False


def test_international_adapter_is_explicitly_unsupported():
    adapter = pm_intl.PolymarketInternationalAdapter()
    with pytest.raises(UnsupportedVenueError):
        adapter.list_mlb_moneyline_markets()
    with pytest.raises(UnsupportedVenueError):
        adapter.fetch_market_book("any")


def test_parse_moneyline_does_not_assume_long_is_home():
    payload = _load("mlb_event_moneyline.json")
    event = payload["events"][0]
    market = event["markets"][0]
    parsed = pm_us.parse_moneyline_market(event, market)
    assert parsed.market_type == "moneyline"
    assert parsed.home_team == "COL"
    assert parsed.away_team == "SD"
    long_team = next(o.team_abbr for o in parsed.outcomes if o.is_long)
    short_team = next(o.team_abbr for o in parsed.outcomes if not o.is_long)
    assert long_team == "SD"
    assert short_team == "COL"
    assert long_team != parsed.home_team


def test_fee_schedule_versions_and_taker_fee_formula():
    early = pm_us.select_fee_schedule(as_of_utc="2026-08-01T00:00:00Z")
    assert early.taker_theta == pytest.approx(0.06)
    assert early.taker_fee(0.5, 1000) == pytest.approx(15.0)
    later = pm_us.select_fee_schedule(as_of_utc="2026-09-17T04:00:00Z")
    assert later.taker_theta == pytest.approx(0.0695)


def test_parse_book_and_executable_buy_for_home_short_side():
    payload = _load("mlb_event_moneyline.json")
    event = payload["events"][0]
    market = pm_us.parse_moneyline_market(event, event["markets"][0])
    book_payload = _load("mlb_moneyline_book.json")
    book = pm_us.parse_book_response(
        book_payload,
        market_id=market.market_id,
        market_slug=market.market_slug,
        request_time_utc="2026-09-15T01:00:00Z",
        receive_time_utc="2026-09-15T01:00:01Z",
        fee=pm_us.select_fee_schedule(as_of_utc="2026-09-15T01:00:01Z"),
    )
    assert book.data_class == "order_book"
    assert book.best_bid is not None and book.best_ask is not None
    assert book.best_ask_size is not None
    home_buy = pm_us.executable_buy_price_for_team(market, book, market.home_team)
    assert home_buy["is_long"] is False
    assert home_buy["method"] == "one_minus_book_best_bid_short"
    assert home_buy["executable_buy"] == pytest.approx(1.0 - book.best_bid)
    away_buy = pm_us.executable_buy_price_for_team(market, book, market.away_team)
    assert away_buy["is_long"] is True
    assert away_buy["executable_buy"] == pytest.approx(book.best_ask)


def test_match_market_unique_ambiguous_and_unmatched():
    payload = _load("mlb_event_moneyline.json")
    event = payload["events"][0]
    market = pm_us.parse_moneyline_market(event, event["markets"][0])
    schedule = pd.DataFrame([
        {
            "game_pk": 111,
            "home_team": "COL",
            "away_team": "SD",
            "game_datetime": market.scheduled_start_utc,
        }
    ])
    ok = market_contracts.match_market_to_schedule(market, schedule)
    assert ok["mapping_status"] == market_contracts.MAPPING_MAPPED
    assert ok["game_pk"] == 111

    amb = market_contracts.match_market_to_schedule(
        market,
        pd.DataFrame([
            {"game_pk": 1, "home_team": "COL", "away_team": "SD", "game_datetime": market.scheduled_start_utc},
            {"game_pk": 2, "home_team": "COL", "away_team": "SD", "game_datetime": market.scheduled_start_utc},
        ]),
    )
    assert amb["mapping_status"] == market_contracts.MAPPING_AMBIGUOUS

    miss = market_contracts.match_market_to_schedule(
        market,
        pd.DataFrame([
            {"game_pk": 3, "home_team": "NYY", "away_team": "BOS", "game_datetime": market.scheduled_start_utc},
        ]),
    )
    assert miss["mapping_status"] == market_contracts.MAPPING_UNMATCHED


def test_registry_upsert_preserves_first_observed(tmp_path):
    payload = _load("mlb_event_moneyline.json")
    event = payload["events"][0]
    market = pm_us.parse_moneyline_market(event, event["markets"][0])
    schedule = pd.DataFrame([
        {
            "game_pk": 777,
            "home_team": market.home_team,
            "away_team": market.away_team,
            "game_datetime": market.scheduled_start_utc,
        }
    ])
    first = market_contracts.upsert_registry(
        market_contracts.empty_registry_frame(),
        [market],
        schedule,
        observed_at_utc="2026-09-14T12:00:00Z",
    )
    assert first.iloc[0]["first_observed_at_utc"] == "2026-09-14T12:00:00Z"
    second = market_contracts.upsert_registry(
        first,
        [market],
        schedule,
        observed_at_utc="2026-09-15T12:00:00Z",
    )
    assert second.iloc[0]["first_observed_at_utc"] == "2026-09-14T12:00:00Z"
    assert second.iloc[0]["updated_at_utc"] == "2026-09-15T12:00:00Z"
    path = tmp_path / "contracts.csv"
    market_contracts.save_registry(second, str(path))
    loaded = market_contracts.load_registry(str(path))
    assert len(loaded) == 1
    assert loaded.iloc[0]["game_pk"] == 777


def test_quote_store_idempotent_append(tmp_path):
    payload = _load("mlb_event_moneyline.json")
    event = payload["events"][0]
    market = pm_us.parse_moneyline_market(event, event["markets"][0])
    book_payload = _load("mlb_moneyline_book.json")
    book = pm_us.parse_book_response(
        book_payload,
        market_id=market.market_id,
        market_slug=market.market_slug,
        request_time_utc="2026-09-15T01:00:00Z",
        receive_time_utc="2026-09-15T01:00:01Z",
        fee=pm_us.select_fee_schedule(as_of_utc="2026-09-15T01:00:01Z"),
    )
    store = tmp_path / "quotes"
    a = quote_store.append_quotes([book], store_dir=str(store))
    b = quote_store.append_quotes([book], store_dir=str(store))
    assert a["n_written"] == 1
    assert b["n_written"] == 0
    idx = quote_store.load_quote_index(str(store))
    assert len(idx) == 1
    health = quote_store.latest_quote_age_seconds(
        str(store), now_utc="2026-09-15T01:00:10Z",
    )
    assert health["stale"] is False
    stale = quote_store.latest_quote_age_seconds(
        str(store), now_utc="2026-09-15T01:05:00Z",
    )
    assert stale["stale"] is True


def test_adapter_list_markets_uses_injected_http():
    payload = _load("mlb_event_moneyline.json")

    class _Resp:
        def __init__(self, data):
            self._data = json.dumps(data).encode("utf-8")

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(req, timeout=30):
        if "/v2/leagues/" in req.get_full_url():
            return _Resp(payload)
        raise AssertionError(req.get_full_url())

    adapter = pm_us.PolymarketUSAdapter(opener=opener)
    markets = adapter.list_mlb_moneyline_markets(limit=5)
    assert len(markets) == 1
    assert markets[0].market_slug.startswith("aec-mlb-")


def test_load_mapping_schedule_prefers_live_over_snapshots(monkeypatch):
    live = pd.DataFrame([
        {
            "game_pk": 200,
            "home_team": "CIN",
            "away_team": "LAD",
            "game_datetime": "2026-09-15T22:40:00Z",
            "date": pd.Timestamp("2026-09-15"),
        }
    ])
    snaps = pd.DataFrame([
        {
            "game_pk": 100,
            "home_team": "CIN",
            "away_team": "LAD",
            "game_datetime": "2026-09-14T22:40:00Z",
            "date": pd.Timestamp("2026-09-14"),
        },
        {
            "game_pk": 200,
            "home_team": "CIN",
            "away_team": "LAD",
            "game_datetime": "2026-09-15T20:00:00Z",
            "date": pd.Timestamp("2026-09-15"),
        },
    ])
    monkeypatch.setattr(
        market_contracts,
        "fetch_live_schedule_for_mapping",
        lambda **kwargs: live,
    )
    monkeypatch.setattr(
        market_contracts,
        "load_schedule_snapshots_for_mapping",
        lambda path=None: snaps,
    )
    out = market_contracts.load_mapping_schedule(prefer_live=True)
    assert set(out["game_pk"]) == {100, 200}
    row200 = out[out["game_pk"] == 200].iloc[0]
    assert row200["game_datetime"] == "2026-09-15T22:40:00Z"


def test_live_schedule_maps_next_day_moneyline(monkeypatch):
    payload = _load("mlb_event_moneyline.json")
    event = payload["events"][0]
    market = pm_us.parse_moneyline_market(event, event["markets"][0])
    market.home_team = "CIN"
    market.away_team = "LAD"
    market.scheduled_start_utc = "2026-09-15T22:40:00Z"
    live = pd.DataFrame([
        {
            "game_pk": 824466,
            "home_team": "CIN",
            "away_team": "LAD",
            "game_datetime": "2026-09-15T22:40:00Z",
            "date": pd.Timestamp("2026-09-15"),
        }
    ])
    monkeypatch.setattr(
        market_contracts,
        "fetch_live_schedule_for_mapping",
        lambda **kwargs: live,
    )
    monkeypatch.setattr(
        market_contracts,
        "load_schedule_snapshots_for_mapping",
        lambda path=None: pd.DataFrame(
            [{
                "game_pk": 824465,
                "home_team": "CIN",
                "away_team": "LAD",
                "game_datetime": "2026-09-14T22:40:00Z",
                "date": pd.Timestamp("2026-09-14"),
            }]
        ),
    )
    schedule = market_contracts.load_mapping_schedule(prefer_live=True)
    matched = market_contracts.match_market_to_schedule(market, schedule)
    assert matched["mapping_status"] == market_contracts.MAPPING_MAPPED
    assert matched["game_pk"] == 824466


def test_polymarket_capture_workflow_yaml():
    text = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "polymarket_capture.yml"
    ).read_text(encoding="utf-8")
    assert "capture_polymarket.py" in text
    assert "Does NOT place orders" in text
    assert "GAME_PREDICTION_MODE" in text
    assert "BETTING_MODE" in text
    assert "upload-artifact@v4" in text
    assert "27,57" in text
