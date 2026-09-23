"""Research-only hit-prop fixtures: identity, rules, outages, no promotion."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mlb_metrics import config, hit_prop_research as research
from mlb_metrics.venues.polymarket_us import PolymarketUSAdapter


def market(slug="one-hit", **overrides):
    return {
        "slug": slug,
        "id": "55",
        "sportsMarketType": "baseball_player_hits",
        "line": 1,
        "active": True,
        "closed": False,
        "metadata": {"playerId": 123, "playerName": "Fixture Player"},
        "description": (
            "Starting lineup and plate appearance required; otherwise LFMP. "
            "Extra innings included. If postponed and not rescheduled within two days, LFMP."
        ),
        "marketSides": [
            {"long": True, "description": "Yes"},
            {"long": False, "description": "No"},
        ],
        **overrides,
    }


class Adapter:
    def __init__(self, markets=None, fail_book=False, event_extra=None):
        self.markets = markets if markets is not None else [market()]
        self.fail_book = fail_book
        self.calls = []
        self.event_extra = event_extra or {}

    def list_league_events(self, **kwargs):
        # UTC next day is still September 18 in Phoenix.
        return [
            {"slug": "mlb-bos-nyy-2026-09-18", "startTime": "2026-09-19T02:00:00Z", "gameId": 111},
            {"slug": "past", "startTime": "2026-09-18T01:00:00Z"},
            {"slug": "wrong-day", "startTime": "2026-09-19T08:00:00Z"},
        ]

    def fetch_event_details(self, slug):
        assert slug == "mlb-bos-nyy-2026-09-18"
        event = {
            "slug": slug,
            "id": "ev1",
            "gameId": 111,
            "startTime": "2026-09-19T02:00:00Z",
            "teams": [
                {"displayAbbreviation": "BOS", "ordering": "away"},
                {"displayAbbreviation": "NYY", "ordering": "home"},
            ],
            "markets": self.markets,
        }
        event.update(self.event_extra)
        return {
            "event": event,
            "request_time_utc": "2026-09-18T18:00:00Z",
            "receive_time_utc": "2026-09-18T18:00:01Z",
        }

    def fetch_market_book(self, slug, **kwargs):
        self.calls.append(slug)
        if self.fail_book:
            raise RuntimeError("fixture outage")
        return SimpleNamespace(
            eligible=True,
            suspended=False,
            best_bid=0.67,
            best_ask=0.75,
            best_bid_size=20,
            best_ask_size=10,
            receive_time_utc="2026-09-18T18:00:02Z",
            to_dict=lambda: {"best_bid": 0.67, "best_ask": 0.75},
        )


def run(adapter, **kwargs):
    return research.capture(
        adapter,
        date=dt.date(2026, 9, 18),
        event_limit=1,
        book_limit=kwargs.get("book_limit", 1),
        now=dt.datetime(2026, 9, 18, 18, tzinfo=dt.timezone.utc),
        schedule_games=kwargs.get("schedule_games", pd.DataFrame()),
        moneyline_registry=kwargs.get("moneyline_registry", pd.DataFrame()),
        player_candidates=kwargs.get("player_candidates", pd.DataFrame()),
        provider_player_map=kwargs.get("provider_player_map", {}),
        persist_store=kwargs.get("persist_store", False),
        store_dir=kwargs.get("store_dir"),
        resume=kwargs.get("resume", True),
    )


def test_exact_stat_threshold_and_state_only():
    rows = [
        market(),
        market(),
        market("two-hits", line=2),
        market("hr", sportsMarketType="baseball_player_home_runs"),
        market("closed", closed=True),
        market("hidden", hidden=True),
        market("invalid", line="nan"),
        market("unknown", active=None),
    ]
    assert [r["slug"] for r in research.one_hit_markets({"markets": rows})] == ["one-hit"]


def test_phoenix_date_identity_and_side_costs():
    report = run(Adapter())
    assert report["n_future_events_on_requested_date_in_response"] == 1
    row = report["contracts"][0]
    assert row["provider_player_id"] == 123
    assert row["key_mlbam"] is None
    assert row["provider_game_id"] == 111
    assert row["game_pk"] is None
    assert row["yes_buy_price"] == 0.75
    assert row["no_buy_price"] == pytest.approx(0.33)
    assert row["no_buy_size"] == 20
    assert row["actionable"] is False and row["model_probability"] is None
    assert "plate appearance" in row["rules_text"].lower()
    assert row["requires_starting_lineup"] is True
    assert row["requires_plate_appearance"] is True
    assert row["postponement_window_days"] == 2


def test_request_budget_and_duplicate_contracts():
    adapter = Adapter([market("a"), market("a"), market("b")])
    report = run(adapter)
    assert adapter.calls == ["a"]
    assert len(report["contracts"]) == 2
    assert report["contracts"][1]["book_status"] == "not_requested_budget"


def test_failure_is_retained_without_recommendation():
    report = run(Adapter(fail_book=True))
    assert report["status"] == "partial"
    assert report["errors"][0]["stage"] == "book"
    assert report["contracts"][0]["yes_buy_price"] is None
    assert report["contracts"][0]["actionable"] is False


def test_unknown_orientation_is_not_inferred_from_team():
    adapter = Adapter([market(marketSides=[{"long": True, "description": "Team"}])])
    report = run(adapter)
    assert not adapter.calls
    assert report["contracts"][0]["book_status"] == "unverified_outcome_orientation"


def test_failed_discovery_has_persistable_status():
    adapter = Adapter()

    def fail(**kwargs):
        raise RuntimeError("discovery unavailable")

    adapter.list_league_events = fail
    report = run(adapter)
    assert report["status"] == "failed" and report["errors"]
    assert report["actionable"] is False


@pytest.mark.parametrize("bid,ask,size", [(0.8, 0.7, 10), (0.5, 0.5, 10), (0.3, 1.5, float("nan"))])
def test_invalid_prices_and_missing_size_are_not_buy_quotes(bid, ask, size):
    adapter = Adapter()
    book = adapter.fetch_market_book("fixture")
    book.best_bid, book.best_ask, book.best_ask_size = bid, ask, size
    book.to_dict = lambda: {"best_ask_size": size}
    adapter.fetch_market_book = lambda *args, **kwargs: book
    report = run(adapter)
    assert report["contracts"][0]["yes_buy_price"] is None
    json.dumps(report, allow_nan=False)


def test_event_detail_endpoint_and_malformed_response(monkeypatch):
    adapter = PolymarketUSAdapter()
    paths = []

    def get(path):
        paths.append(path)
        return {"event": {"slug": "game"}}

    monkeypatch.setattr(adapter, "_http_get_json", get)
    result = adapter.fetch_event_details("game")
    assert paths == ["/v1/events/slug/game"]
    assert result["event"]["slug"] == "game"
    assert result["request_time_utc"] and result["receive_time_utc"]
    monkeypatch.setattr(adapter, "_http_get_json", lambda path: {})
    with pytest.raises(ValueError, match="event object"):
        adapter.fetch_event_details("game")


def test_game_pk_maps_from_moneyline_registry_and_quarantines_wrong_day():
    registry = pd.DataFrame(
        [
            {
                "event_slug": "mlb-bos-nyy-2026-09-18",
                "game_pk": 999001,
                "mapping_status": "mapped",
                "provider_game_id": 111,
                "home_team": "NYY",
                "away_team": "BOS",
            }
        ]
    )
    report = run(Adapter(), moneyline_registry=registry)
    row = report["contracts"][0]
    assert row["game_pk"] == 999001
    assert row["game_mapping_status"] == research.MAPPING_MAPPED

    # Wrong local day quarantine
    bad = research.match_event_to_game(
        {
            "slug": "mlb-bos-nyy-2026-09-19",
            "startTime": "2026-09-19T20:00:00Z",
            "teams": [
                {"displayAbbreviation": "BOS", "ordering": "away"},
                {"displayAbbreviation": "NYY", "ordering": "home"},
            ],
        },
        pd.DataFrame(),
        requested_local_date=dt.date(2026, 9, 18),
    )
    assert bad["mapping_status"] == research.MAPPING_WRONG_DAY


def test_doubleheader_ambiguous_quarantine():
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "home_team": "NYY",
                "away_team": "BOS",
                "game_datetime": "2026-09-18T17:05:00Z",
                "date": "2026-09-18",
            },
            {
                "game_pk": 2,
                "home_team": "NYY",
                "away_team": "BOS",
                "game_datetime": "2026-09-18T23:05:00Z",
                "date": "2026-09-18",
            },
        ]
    )
    matched = research.match_event_to_game(
        {
            "slug": "mlb-bos-nyy-2026-09-18",
            "startTime": "2026-09-18T20:00:00Z",  # between both; outside 45m of either
            "teams": [
                {"displayAbbreviation": "BOS", "ordering": "away"},
                {"displayAbbreviation": "NYY", "ordering": "home"},
            ],
        },
        schedule,
        requested_local_date=dt.date(2026, 9, 18),
    )
    # Outside tolerance → unmatched; with start matching both within tolerance would be ambiguous/DH
    assert matched["mapping_status"] in {
        research.MAPPING_UNMATCHED,
        research.MAPPING_DOUBLEHEADER,
        research.MAPPING_AMBIGUOUS,
    }

    # Explicit multi-matchup without usable unique start → quarantine
    matched2 = research.match_event_to_game(
        {
            "slug": "mlb-bos-nyy-2026-09-18",
            "startTime": None,
            "teams": [
                {"displayAbbreviation": "BOS", "ordering": "away"},
                {"displayAbbreviation": "NYY", "ordering": "home"},
            ],
        },
        schedule.drop(columns=["game_datetime"]),
        requested_local_date=dt.date(2026, 9, 18),
    )
    assert matched2["mapping_status"] in {
        research.MAPPING_DOUBLEHEADER,
        research.MAPPING_AMBIGUOUS,
    }
    assert matched2["game_pk"] is None


def test_player_name_unique_ambiguous_and_provider_conflict():
    candidates = pd.DataFrame(
        [
            {"key_mlbam": 10, "name_first": "Fixture", "name_last": "Player"},
            {"key_mlbam": 11, "name_first": "Other", "name_last": "Name"},
            {"key_mlbam": 12, "name_first": "José", "name_last": "Ramírez"},
            {"key_mlbam": 13, "name_first": "Jose", "name_last": "Ramirez"},
        ]
    )
    ok = research.match_player_to_key_mlbam("Fixture Player", candidate_players=candidates)
    assert ok["mapping_status"] == research.MAPPING_MAPPED
    assert ok["key_mlbam"] == 10

    amb = research.match_player_to_key_mlbam("Jose Ramirez", candidate_players=candidates)
    assert amb["mapping_status"] == research.MAPPING_AMBIGUOUS
    assert amb["key_mlbam"] is None

    conflict = research.match_player_to_key_mlbam(
        "Fixture Player",
        provider_player_id=123,
        candidate_players=candidates,
        prior_provider_map={"123": 999},
    )
    assert conflict["mapping_status"] == research.MAPPING_PROVIDER_CONFLICT


def test_walk_only_pa_and_missing_outcomes_and_zero_hits():
    rules = research.parse_contract_rules(
        "Must be in the starting lineup and record a plate appearance; otherwise last fair market price."
    )
    walk = research.classify_contract_outcome(
        started=True,
        plate_appearances=2,
        at_bats=0,
        hits=0,
        game_status="Final",
        rules=rules,
    )
    assert walk["binary_yes"] is False
    assert walk["walk_only_appearance"] is True
    assert walk["settlement_class"] == "binary_no"

    missing = research.classify_contract_outcome(
        started=True,
        plate_appearances=1,
        at_bats=1,
        hits=None,
        game_status="Final",
        rules=rules,
    )
    assert missing["settlement_class"] == "unknown_pending"
    assert missing["binary_yes"] is None

    dnp = research.classify_contract_outcome(
        started=False,
        plate_appearances=0,
        at_bats=0,
        hits=0,
        game_status="Final",
        rules=rules,
    )
    assert dnp["settlement_class"] == "last_fair_market_price"

    zero_ab_unknown = research.classify_contract_outcome(
        started=None,
        plate_appearances=None,
        at_bats=0,
        hits=0,
        game_status="Final",
        rules=rules,
    )
    assert zero_ab_unknown["settlement_class"] == "unknown_pending"


def test_restart_skips_completed_markets(tmp_path):
    store = str(tmp_path / "store")
    adapter = Adapter([market("m1"), market("m2")])
    first = run(adapter, book_limit=1, persist_store=True, store_dir=store, resume=True)
    assert first["n_book_requests"] == 1
    assert (tmp_path / "store" / "contracts.csv").exists()
    adapter2 = Adapter([market("m1"), market("m2")])
    second = run(adapter2, book_limit=2, persist_store=True, store_dir=store, resume=True)
    # m1 skipped via checkpoint; m2 newly requested
    assert "m1" not in adapter2.calls or second["contracts"][0]["book_status"] == "skipped_restart_already_captured"
    assert any(c["market_slug"] == "m2" and c["book_status"] == "captured" for c in second["contracts"])


def test_modes_remain_fail_closed():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
    assert config.HIT_PROP_PAPER_PROTOCOL_ID.startswith("polymarket_us_mlb_hitter_hits")


def test_traded_or_wrong_team_quarantine():
    candidates = pd.DataFrame(
        [
            {"key_mlbam": 10, "name_first": "Fixture", "name_last": "Player", "team": "LAD"},
            {"key_mlbam": 11, "name_first": "Other", "name_last": "Name", "team": "NYY"},
        ]
    )
    wrong = research.match_player_to_key_mlbam(
        "Fixture Player",
        candidate_players=candidates,
        home_team="NYY",
        away_team="BOS",
    )
    assert wrong["mapping_status"] == research.MAPPING_TRADED_OR_WRONG_TEAM
    assert wrong["key_mlbam"] is None

    contract_mismatch = research.match_player_to_key_mlbam(
        "Other Name",
        candidate_players=candidates,
        home_team="NYY",
        away_team="BOS",
        contract_team="LAD",
    )
    assert contract_mismatch["mapping_status"] == research.MAPPING_TRADED_OR_WRONG_TEAM

    ok = research.match_player_to_key_mlbam(
        "Other Name",
        candidate_players=candidates,
        home_team="NYY",
        away_team="BOS",
        contract_team="NYY",
    )
    assert ok["mapping_status"] == research.MAPPING_MAPPED
    assert ok["key_mlbam"] == 11


def test_rules_version_and_nonbinary_flag():
    rules = research.parse_contract_rules(
        "Must be in the starting lineup and record a plate appearance; otherwise last fair market price. "
        "Extra innings included. Shortened official games settle."
    )
    assert rules["rules_version"] == research.HIT_PROP_RULES_VERSION
    assert rules["nonbinary_settlement_possible"] is True
    assert rules["settlement_on_non_participation"] == "last_fair_market_price"

    report = run(
        Adapter(),
        player_candidates=pd.DataFrame(
            [{"key_mlbam": 10, "name_first": "Fixture", "name_last": "Player", "team": "NYY"}]
        ),
        moneyline_registry=pd.DataFrame(
            [
                {
                    "event_slug": "mlb-bos-nyy-2026-09-18",
                    "game_pk": 999001,
                    "mapping_status": "mapped",
                    "provider_game_id": 111,
                    "home_team": "NYY",
                    "away_team": "BOS",
                }
            ]
        ),
    )
    row = report["contracts"][0]
    assert row["rules_version"] == research.HIT_PROP_RULES_VERSION
    assert row["nonbinary_settlement_possible"] is True
    assert row["key_mlbam"] == 10
    assert row["game_pk"] == 999001
    assert row["research_only"] is True
    assert row["actionable"] is False


def test_host_ops_and_example_reports(tmp_path):
    host_path = tmp_path / "host.json"
    host = research.write_host_ops_report(path=str(host_path))
    assert host["host"] == config.HIT_PROP_COLLECTION_HOST
    assert host["paid_services_purchased"] is False
    assert host["paid_budget_authorized_usd"] is None
    assert host["missing_host"] is None
    assert "asleep" in host["sleep_behavior"].lower()
    assert host_path.exists()

    report = run(
        Adapter(),
        moneyline_registry=pd.DataFrame(
            [
                {
                    "event_slug": "mlb-bos-nyy-2026-09-18",
                    "game_pk": 999001,
                    "mapping_status": "mapped",
                    "home_team": "NYY",
                    "away_team": "BOS",
                }
            ]
        ),
        player_candidates=pd.DataFrame(
            [{"key_mlbam": 10, "name_first": "Fixture", "name_last": "Player", "team": "NYY"}]
        ),
    )
    example_path = tmp_path / "example.json"
    example = research.write_contract_example(report, path=str(example_path))
    assert example is not None
    assert example["identity_mapping"]["game"]["game_pk"] == 999001
    assert example["identity_mapping"]["player"]["key_mlbam"] == 10
    assert example["contract"]["rules_version"] == research.HIT_PROP_RULES_VERSION
    assert example["contract"]["book_status"] == "captured"
    assert example["contract"]["yes_buy_price"] == 0.75
    assert example["parsed_rules"]["nonbinary_settlement_possible"] is True
    assert example_path.exists()


def test_workflow_includes_hit_prop_fail_soft_step():
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "polymarket_capture.yml"
    text = path.read_text(encoding="utf-8")
    assert "capture_hit_prop_research.py" in text
    assert "Hit-prop research capture" in text
    assert "hit_prop_host_ops_latest.json" in text
    assert "run_hit_prop_paper_ops.py" in text
    assert "hit_prop_ops_latest.json" in text
    assert "run_hit_prop_readiness_evaluation.py" in text
