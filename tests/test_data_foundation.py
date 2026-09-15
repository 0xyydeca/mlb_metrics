"""Data-foundation tests: Polymarket capture health, lineups, coverage waterfall."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, game_baseball_snapshots, lineup_snapshots, market_contracts, quote_store
from mlb_metrics.venues import polymarket_us as pm_us
from mlb_metrics.venues.base import MarketBookQuote

FIXTURES_PM = Path(__file__).resolve().parent / "fixtures" / "polymarket"
FIXTURES_LU = Path(__file__).resolve().parent / "fixtures" / "lineups"


def _load_pm(name: str):
    return json.loads((FIXTURES_PM / name).read_text(encoding="utf-8"))


def test_fee_boundary_before_and_after_us_theta_change():
    before = pm_us.select_fee_schedule(as_of_utc="2026-09-17T03:58:59Z")
    after = pm_us.select_fee_schedule(as_of_utc="2026-09-17T03:59:00Z")
    assert before.taker_theta == pytest.approx(0.06)
    assert after.taker_theta == pytest.approx(0.0695)
    assert before.fee_version != after.fee_version


def test_doubleheader_same_matchup_maps_by_start_time():
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
    payload = _load_pm("mlb_event_moneyline.json")
    market = pm_us.parse_moneyline_market(payload["events"][0], payload["events"][0]["markets"][0])
    market.home_team = "NYY"
    market.away_team = "BOS"
    market.scheduled_start_utc = "2026-09-15T23:05:00Z"
    matched = market_contracts.match_market_to_schedule(market, schedule)
    assert matched["mapping_status"] == market_contracts.MAPPING_MAPPED
    assert matched["game_pk"] == 2


def test_postponed_status_marks_baseball_snapshot_unusable():
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 10,
                "date": "2026-09-15",
                "game_datetime": "2026-09-15T23:00:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
                "status": "Postponed",
                "home_probable_pitcher_key_mlbam": 1,
                "away_probable_pitcher_key_mlbam": 2,
            }
        ]
    )
    snaps = game_baseball_snapshots.build_game_baseball_snapshots(schedule)
    assert bool(snaps.iloc[0]["usable_for_game_winner"]) is False
    assert "schedule_status:Postponed" in str(snaps.iloc[0]["pass_reason"])


def test_missing_book_stays_missing_and_stale_suppresses_actionable(tmp_path):
    health = quote_store.latest_quote_age_seconds(str(tmp_path / "quotes"))
    assert health["reason"] == "no_quotes"
    assert health["stale"] is True
    suppressed = quote_store.actionable_suppressed(
        quote_health=health,
        collector_heartbeat_utc="2026-09-15T00:00:00Z",
        now_utc="2026-09-15T00:05:00Z",
        max_heartbeat_age_seconds=30,
    )
    assert suppressed["actionable"] is False
    assert "collector_heartbeat_stale" in suppressed["suppress_reasons"]


def test_quote_append_restart_dedupes_and_per_contract_freshness(tmp_path):
    store = tmp_path / "quotes"
    book_payload = _load_pm("mlb_moneyline_book.json")
    fee = pm_us.select_fee_schedule(as_of_utc="2026-09-15T01:00:01Z")
    q1 = pm_us.parse_book_response(
        book_payload,
        market_id="800001",
        market_slug="aec-mlb-sd-col-2026-09-14",
        request_time_utc="2026-09-15T01:00:00Z",
        receive_time_utc="2026-09-15T01:00:01Z",
        fee=fee,
    )
    first = quote_store.append_quotes([q1], store_dir=str(store))
    second = quote_store.append_quotes([q1], store_dir=str(store))
    assert first["n_written"] == 1
    assert second["n_written"] == 0
    per = quote_store.per_contract_quote_health(
        str(store),
        market_ids=["800001", "999999"],
        now_utc="2026-09-15T01:00:10Z",
    )
    assert per["n_fresh_eligible"] == 1
    assert per["n_missing"] == 1
    assert any(c["pass_reason"] == "missing_quote" for c in per["contracts"])


def test_empty_book_is_liquidity_not_http_failure():
    fee = pm_us.select_fee_schedule(as_of_utc="2026-09-15T01:00:01Z")
    book = pm_us.parse_book_response(
        {"marketData": {"bids": [], "offers": [], "state": "MARKET_STATE_OPEN"}},
        market_id="1",
        market_slug="x",
        request_time_utc="2026-09-15T01:00:00Z",
        receive_time_utc="2026-09-15T01:00:01Z",
        fee=fee,
    )
    assert book.eligible is False
    assert book.eligibility_reason == "missing_ask"


def test_wrong_outcome_orientation_does_not_assume_long_is_home():
    payload = _load_pm("mlb_event_moneyline.json")
    market = pm_us.parse_moneyline_market(payload["events"][0], payload["events"][0]["markets"][0])
    book = pm_us.parse_book_response(
        _load_pm("mlb_moneyline_book.json"),
        market_id=market.market_id,
        market_slug=market.market_slug,
        request_time_utc="2026-09-15T01:00:00Z",
        receive_time_utc="2026-09-15T01:00:01Z",
        fee=pm_us.select_fee_schedule(as_of_utc="2026-09-15T01:00:01Z"),
    )
    home_buy = pm_us.executable_buy_price_for_team(market, book, market.home_team)
    away_buy = pm_us.executable_buy_price_for_team(market, book, market.away_team)
    assert home_buy["is_long"] is False
    assert away_buy["is_long"] is True
    assert home_buy["executable_buy"] != away_buy["executable_buy"]


def test_parse_verified_schedule_lineups_fixture_uses_list_index_order():
    assert config.LINEUP_API_SCHEMA_CONFIRMED is True
    raw = json.loads((FIXTURES_LU / "statsapi_schedule_lineups_824465.json").read_text())
    frame = lineup_snapshots.parse_statsapi_schedule_lineups(
        raw, fetched_at_utc="2026-09-14T23:00:00Z"
    )
    assert not frame.empty
    assert set(frame["game_pk"].astype(int)) == {824465}
    home = frame[frame["team"] == "CIN"].sort_values("batting_order")
    # Fixture homePlayers[0].id == 667472 (list index 1); final boxscore slot 1 differed.
    assert int(home.iloc[0]["key_mlbam"]) == 667472
    assert int(home.iloc[0]["batting_order"]) == 1
    assert home.iloc[0]["batting_order_source"] == config.LINEUP_BATTING_ORDER_SOURCE
    assert bool(home.iloc[0]["is_confirmed_starter"]) is True


def test_as_of_filter_drops_later_baseball_snapshots():
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 55,
                "date": "2026-09-15",
                "game_datetime": "2026-09-15T23:00:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
                "status": "Scheduled",
                "home_probable_pitcher_key_mlbam": 11,
                "away_probable_pitcher_key_mlbam": 22,
            }
        ]
    )
    early = game_baseball_snapshots.build_game_baseball_snapshots(
        schedule, fetched_at_utc="2026-09-15T12:00:00Z"
    )
    late = game_baseball_snapshots.build_game_baseball_snapshots(
        schedule, fetched_at_utc="2026-09-15T18:00:00Z"
    )
    combined = pd.concat([early, late], ignore_index=True)
    kept = game_baseball_snapshots.filter_snapshots_as_of(
        combined, as_of_utc="2026-09-15T13:00:00Z"
    )
    assert len(kept) == 1
    assert kept.iloc[0]["fetched_at_utc"] == "2026-09-15T12:00:00Z"


def test_coverage_waterfall_distinguishes_mapping_from_prices():
    registry = pd.DataFrame(
        [
            {"market_id": "1", "mapping_status": "mapped"},
            {"market_id": "2", "mapping_status": "mapped"},
            {"market_id": "3", "mapping_status": "unmatched"},
        ]
    )
    quote_contracts = [
        {
            "market_id": "1",
            "pass_reason": None,
            "eligible": True,
            "stale": False,
        },
        {
            "market_id": "2",
            "pass_reason": "stale_quote",
            "eligible": True,
            "stale": True,
        },
    ]
    waterfall = game_baseball_snapshots.coverage_waterfall(
        schedule_games=pd.DataFrame({"game_pk": [10, 11, 12]}),
        registry=registry,
        quote_health_contracts=quote_contracts,
        baseball_snapshots=None,
    )
    assert waterfall["n_schedule_games"] == 3
    assert waterfall["n_mapped_contracts"] == 2
    assert waterfall["n_usable_prices"] == 1
    assert waterfall["removed_price_reasons"]["stale_quote"] == 1


def test_trace_contract_requires_quote_and_baseball():
    reg = {
        "venue_id": "polymarket_us",
        "market_id": "1",
        "market_slug": "x",
        "game_pk": 55,
        "home_team": "NYY",
        "away_team": "BOS",
        "long_team": "BOS",
        "short_team": "NYY",
        "mapping_status": "mapped",
        "mapping_evidence": "{}",
        "rules_hash": "abc",
        "rules_text": "official result",
    }
    trace = game_baseball_snapshots.trace_contract_decision_inputs(
        registry_row=reg,
        quote_row=None,
        market=None,
        book=None,
        baseball_row=None,
        fee_version=None,
    )
    assert trace["actionable"] is False
    assert "missing_quote" in trace["pass_reasons"]
    assert "missing_baseball_snapshot" in trace["pass_reasons"]
    assert trace["orientation_assumes_long_is_home"] is False


def test_http_retries_on_429(monkeypatch):
    calls = {"n": 0}

    class _Body:
        def read(self):
            return b"slow"

        def close(self):
            return None

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"events":[]}'

    def opener(req, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            import urllib.error

            raise urllib.error.HTTPError(
                req.get_full_url(), 429, "rate", hdrs=None, fp=_Body()
            )
        return _Resp()

    adapter = pm_us.PolymarketUSAdapter(
        opener=opener,
        max_retries=3,
        retry_backoff_seconds=0.0,
        min_interval_seconds=0.0,
    )
    assert adapter.list_league_events() == []
    assert calls["n"] == 3
