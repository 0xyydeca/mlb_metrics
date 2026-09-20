"""Research-only MLB 1+ hitter-hit prop inventory and identity mapping.

Never scores models, places orders, or enables betting. Provider player/game
IDs are not MLB IDs until explicitly mapped. Contract participation and
settlement semantics are parsed from exact rules text and must be verified
before any evaluation.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import unicodedata
from typing import Any

import pandas as pd

from mlb_metrics import config, market_contracts, schedule
from mlb_metrics.venues.polymarket_us import normalize_team_abbr, rules_hash

MAPPING_MAPPED = "mapped"
MAPPING_UNMATCHED = "unmatched"
MAPPING_AMBIGUOUS = "ambiguous"
MAPPING_WRONG_DAY = "wrong_day"
MAPPING_DOUBLEHEADER = "doubleheader_ambiguous"
MAPPING_PROVIDER_CONFLICT = "provider_id_conflict"
MAPPING_TRADED_OR_WRONG_TEAM = "traded_or_wrong_team"
MAPPING_INCOMPLETE = "incomplete"

QUARANTINE_STATUSES = {
    MAPPING_AMBIGUOUS,
    MAPPING_WRONG_DAY,
    MAPPING_DOUBLEHEADER,
    MAPPING_PROVIDER_CONFLICT,
    MAPPING_TRADED_OR_WRONG_TEAM,
    MAPPING_INCOMPLETE,
    MAPPING_UNMATCHED,
}

# Version of the parsed-rules schema (independent of per-contract rules_hash).
HIT_PROP_RULES_VERSION = "hit_prop_rules_v1"

HIT_PROP_REGISTRY_COLUMNS = [
    "schema_version",
    "venue_id",
    "capture_id",
    "observed_at_utc",
    "requested_local_date",
    "event_id",
    "event_slug",
    "market_id",
    "market_slug",
    "question",
    "scheduled_start_utc",
    "provider_game_id",
    "game_pk",
    "game_mapping_status",
    "game_mapping_evidence",
    "provider_player_id",
    "key_mlbam",
    "player_name",
    "player_mapping_status",
    "player_mapping_evidence",
    "home_team",
    "away_team",
    "stat",
    "threshold",
    "rules_text",
    "rules_hash",
    "rules_version",
    "requires_starting_lineup",
    "requires_plate_appearance",
    "settlement_on_non_participation",
    "postponement_window_days",
    "includes_extra_innings",
    "shortened_official_game_settles",
    "nonbinary_settlement_possible",
    "fee_coefficient",
    "book_status",
    "book_request_time_utc",
    "book_receive_time_utc",
    "yes_buy_price",
    "yes_buy_size",
    "no_buy_price",
    "no_buy_size",
    "actionable",
    "research_only",
    "quarantined",
    "no_bet_reasons",
]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _timestamp(value: Any) -> dt.datetime | None:
    try:
        text = str(value).replace("Z", "+00:00")
        result = dt.datetime.fromisoformat(text)
        if result.tzinfo is None:
            # Polymarket timestamps are UTC; treat naive ISO as UTC.
            result = result.replace(tzinfo=dt.timezone.utc)
        return result
    except ValueError:
        return None


def _json_safe(value):
    """Persist unavailable numeric fields as null, never nonstandard JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_person_name(value: Any) -> str:
    """ASCII-folded lowercase first+last tokens for matching (not display)."""
    text = str(value or "").strip()
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in folded if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-zA-Z\s\-']", " ", ascii_only).lower()
    parts = [p for p in re.split(r"[\s\-]+", cleaned) if p and p not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(parts)


def parse_contract_rules(rules_text: str | None) -> dict[str, Any]:
    """Extract participation/settlement flags from exact contract text.

    Does not invent missing rules. Unknowns stay explicit.
    """
    text = str(rules_text or "")
    lower = text.lower()
    requires_start = "starting lineup" in lower or "must be in the starting lineup" in lower
    requires_pa = "plate appearance" in lower
    lfmp = "last fair market price" in lower or "lfmp" in lower
    extra = "extra innings" in lower
    shortened = "shortened" in lower and "official" in lower
    window = None
    m = re.search(
        r"within\s+(\w+)\s+days?(?:\s+of\s+the\s+originally\s+scheduled(?:\s+date)?)?",
        lower,
    )
    if m:
        word = m.group(1)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
        if word.isdigit():
            window = int(word)
        else:
            window = words.get(word)
    return {
        "requires_starting_lineup": requires_start,
        "requires_plate_appearance": requires_pa,
        "settlement_on_non_participation": (
            "last_fair_market_price" if lfmp and (requires_start or requires_pa) else "unspecified"
        ),
        "postponement_window_days": window,
        "includes_extra_innings": extra,
        "shortened_official_game_settles": shortened,
        "nonbinary_settlement_possible": bool(lfmp),
        "rules_text_present": bool(text.strip()),
        "rules_version": HIT_PROP_RULES_VERSION,
        "notes": [
            "A plate appearance is not necessarily an at-bat; walk-only appearances participate under PA rules.",
            "Zero hits with a qualifying PA settles No under ordinary binary settlement.",
            "Non-participation (not starting / no PA) settles to last fair market price when rules say so.",
            "Unknown outcomes must stay distinct from confirmed zero at-bats.",
            "Last fair market price is a nonbinary settlement class relative to Yes/No.",
        ],
    }


def classify_contract_outcome(
    *,
    started: bool | None,
    plate_appearances: int | None,
    at_bats: int | None,
    hits: int | None,
    game_status: str | None,
    rules: dict[str, Any],
    threshold: int = 1,
) -> dict[str, Any]:
    """Map baseball participation stats to contract settlement classes.

    Returns research labels only — never a trading recommendation.
    """
    status = str(game_status or "").lower()
    if status in {"postponed", "cancelled", "canceled", "suspended"}:
        return {
            "settlement_class": "last_fair_market_price_or_unresolved",
            "reason": "game_postponed_or_suspended",
            "binary_yes": None,
        }
    if started is None or plate_appearances is None:
        return {
            "settlement_class": "unknown_pending",
            "reason": "missing_participation_or_pa",
            "binary_yes": None,
        }
    if rules.get("requires_starting_lineup") and not started:
        return {
            "settlement_class": "last_fair_market_price",
            "reason": "not_in_starting_lineup",
            "binary_yes": None,
        }
    if rules.get("requires_plate_appearance") and int(plate_appearances) <= 0:
        return {
            "settlement_class": "last_fair_market_price",
            "reason": "no_plate_appearance",
            "binary_yes": None,
        }
    # Qualifying participation: PA can be walk-only (AB=0, PA>0).
    if hits is None:
        return {
            "settlement_class": "unknown_pending",
            "reason": "missing_hits",
            "binary_yes": None,
            "walk_only_appearance": bool(
                at_bats is not None and int(at_bats) == 0 and int(plate_appearances) > 0
            ),
        }
    yes = int(hits) >= int(threshold)
    return {
        "settlement_class": "binary_yes" if yes else "binary_no",
        "reason": "qualifying_pa_with_known_hits",
        "binary_yes": yes,
        "walk_only_appearance": bool(
            at_bats is not None and int(at_bats) == 0 and int(plate_appearances) > 0
        ),
        "note": "Confirmed zero at-bats with PA>0 is walk/HBP/sac participation, not missing data.",
    }


def teams_from_event(event: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (home, away) using ordering when present, else event slug away-home."""
    from mlb_metrics.venues.polymarket_us import home_away_from_event_teams

    home, away = home_away_from_event_teams(list(event.get("teams") or []))
    if home and away:
        return home, away
    slug = str(event.get("slug") or "")
    # mlb-chc-cin-2026-09-18 → away CHC, home CIN (Polymarket convention).
    m = re.match(r"^mlb-([a-z0-9]+)-([a-z0-9]+)-\d{4}-\d{2}-\d{2}$", slug.lower())
    if m:
        return normalize_team_abbr(m.group(2)), normalize_team_abbr(m.group(1))
    return None, None


def match_event_to_game(
    event: dict[str, Any],
    schedule_games: pd.DataFrame,
    *,
    moneyline_registry: pd.DataFrame | None = None,
    requested_local_date: dt.date | None = None,
) -> dict[str, Any]:
    """Map event → game_pk; quarantine DH / wrong-day / conflicts."""
    slug = str(event.get("slug") or "")
    provider_game_id = event.get("gameId")
    start = _timestamp(event.get("startTime") or event.get("gameStartTime"))
    home, away = teams_from_event(event)

    # Prefer already-verified moneyline registry mapping for the same event.
    if moneyline_registry is not None and not moneyline_registry.empty and slug:
        hit = moneyline_registry[
            (moneyline_registry.get("event_slug") == slug)
            & (moneyline_registry.get("mapping_status") == "mapped")
        ] if "event_slug" in moneyline_registry.columns else moneyline_registry.iloc[0:0]
        if not hit.empty and pd.notna(hit.iloc[0].get("game_pk")):
            gpk = int(hit.iloc[0]["game_pk"])
            # Provider ID conflict check against other mapped rows.
            if provider_game_id is not None and "provider_game_id" in hit.columns:
                other = moneyline_registry[
                    (moneyline_registry["provider_game_id"].astype(str) == str(provider_game_id))
                    & (moneyline_registry["mapping_status"] == "mapped")
                    & (moneyline_registry["game_pk"].notna())
                ].copy()
                other_gpks = {
                    int(x)
                    for x in pd.to_numeric(other["game_pk"], errors="coerce").dropna().astype(int).tolist()
                    if int(x) != gpk
                }
                if other_gpks:
                    return {
                        "mapping_status": MAPPING_PROVIDER_CONFLICT,
                        "game_pk": None,
                        "home_team": home,
                        "away_team": away,
                        "mapping_evidence": json.dumps(
                            {
                                "reason": "provider_game_id_maps_to_multiple_game_pk",
                                "provider_game_id": provider_game_id,
                                "candidates": sorted(other_gpks | {gpk}),
                            }
                        ),
                    }
            return {
                "mapping_status": MAPPING_MAPPED,
                "game_pk": gpk,
                "home_team": home or hit.iloc[0].get("home_team"),
                "away_team": away or hit.iloc[0].get("away_team"),
                "mapping_evidence": json.dumps(
                    {"reason": "moneyline_registry_event_slug", "event_slug": slug, "game_pk": gpk}
                ),
            }

    if not home or not away:
        return {
            "mapping_status": MAPPING_INCOMPLETE,
            "game_pk": None,
            "home_team": home,
            "away_team": away,
            "mapping_evidence": json.dumps({"reason": "missing_home_away", "slug": slug}),
        }

    # Wrong-day quarantine relative to requested Phoenix local date.
    if requested_local_date is not None and start is not None:
        local_day = start.astimezone(schedule.LOCAL_TIMEZONE).date()
        if local_day != requested_local_date:
            return {
                "mapping_status": MAPPING_WRONG_DAY,
                "game_pk": None,
                "home_team": home,
                "away_team": away,
                "mapping_evidence": json.dumps(
                    {
                        "reason": "start_local_date_mismatch",
                        "requested_local_date": requested_local_date.isoformat(),
                        "event_local_date": local_day.isoformat(),
                    }
                ),
            }

    if schedule_games is None or schedule_games.empty:
        return {
            "mapping_status": MAPPING_UNMATCHED,
            "game_pk": None,
            "home_team": home,
            "away_team": away,
            "mapping_evidence": json.dumps({"reason": "empty_schedule"}),
        }

    class _M:
        pass

    market = _M()
    market.home_team = home
    market.away_team = away
    market.scheduled_start_utc = start.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z") if start else None
    matched = market_contracts.match_market_to_schedule(market, schedule_games)
    status = matched.get("mapping_status")
    evidence = matched.get("mapping_evidence")
    # Upgrade multi-matchup without time to explicit doubleheader quarantine.
    if status == market_contracts.MAPPING_AMBIGUOUS:
        try:
            payload = json.loads(evidence) if evidence else {}
        except json.JSONDecodeError:
            payload = {}
        if "doubleheader" in str(payload.get("reason", "")).lower() or len(payload.get("candidates") or []) > 1:
            status = MAPPING_DOUBLEHEADER
    return {
        "mapping_status": status,
        "game_pk": matched.get("game_pk"),
        "home_team": home,
        "away_team": away,
        "mapping_evidence": evidence,
    }


def match_player_to_key_mlbam(
    player_name: str | None,
    *,
    provider_player_id: Any = None,
    candidate_players: pd.DataFrame | None = None,
    prior_provider_map: dict[str, int] | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    contract_team: str | None = None,
) -> dict[str, Any]:
    """Map display name → key_mlbam; quarantine ambiguous / conflicting / traded IDs."""
    prior_provider_map = prior_provider_map or {}
    norm = normalize_person_name(player_name)
    if not norm:
        return {
            "mapping_status": MAPPING_INCOMPLETE,
            "key_mlbam": None,
            "mapping_evidence": json.dumps({"reason": "missing_player_name"}),
        }

    # Prior verified provider map (conflict if disagrees with unique name match later).
    prior_key = None
    if provider_player_id is not None and str(provider_player_id) in prior_provider_map:
        prior_key = int(prior_provider_map[str(provider_player_id)])

    game_teams = {
        t for t in (normalize_team_abbr(home_team), normalize_team_abbr(away_team)) if t
    }
    contract_team_n = normalize_team_abbr(contract_team)
    if contract_team_n and game_teams and contract_team_n not in game_teams:
        return {
            "mapping_status": MAPPING_TRADED_OR_WRONG_TEAM,
            "key_mlbam": None,
            "mapping_evidence": json.dumps(
                {
                    "reason": "contract_team_not_in_game",
                    "contract_team": contract_team_n,
                    "game_teams": sorted(game_teams),
                }
            ),
        }

    if candidate_players is None or candidate_players.empty:
        if prior_key is not None:
            return {
                "mapping_status": MAPPING_MAPPED,
                "key_mlbam": prior_key,
                "mapping_evidence": json.dumps(
                    {"reason": "prior_provider_player_map", "provider_player_id": provider_player_id}
                ),
            }
        return {
            "mapping_status": MAPPING_UNMATCHED,
            "key_mlbam": None,
            "mapping_evidence": json.dumps({"reason": "no_candidate_roster", "name": norm}),
        }

    frame = candidate_players.copy()
    if "name_norm" not in frame.columns:
        if {"name_first", "name_last"}.issubset(frame.columns):
            frame["name_norm"] = (
                frame["name_first"].fillna("").astype(str) + " " + frame["name_last"].fillna("").astype(str)
            ).map(normalize_person_name)
        elif "player_name" in frame.columns:
            frame["name_norm"] = frame["player_name"].map(normalize_person_name)
        else:
            return {
                "mapping_status": MAPPING_INCOMPLETE,
                "key_mlbam": None,
                "mapping_evidence": json.dumps({"reason": "candidate_frame_missing_name_columns"}),
            }

    # Prefer candidates on the game's two clubs when team is available.
    scoped = frame
    if game_teams and "team" in frame.columns:
        team_norm = frame["team"].map(normalize_team_abbr)
        on_game = frame[team_norm.isin(game_teams)]
        if not on_game.empty:
            scoped = on_game

    hits = scoped[scoped["name_norm"] == norm]
    # If scoped-to-game miss but global unique name exists on another club → traded/wrong team.
    global_hits = frame[frame["name_norm"] == norm]
    keys = (
        sorted({int(x) for x in hits["key_mlbam"].dropna().astype(int).tolist()})
        if "key_mlbam" in hits.columns
        else []
    )
    global_keys = (
        sorted({int(x) for x in global_hits["key_mlbam"].dropna().astype(int).tolist()})
        if "key_mlbam" in global_hits.columns
        else []
    )
    if not keys and global_keys:
        teams_seen = []
        if "team" in global_hits.columns:
            teams_seen = sorted(
                {t for t in global_hits["team"].map(normalize_team_abbr).dropna().tolist() if t}
            )
        return {
            "mapping_status": MAPPING_TRADED_OR_WRONG_TEAM,
            "key_mlbam": None,
            "mapping_evidence": json.dumps(
                {
                    "reason": "name_matched_outside_game_teams",
                    "name": norm,
                    "candidate_teams": teams_seen,
                    "game_teams": sorted(game_teams),
                    "candidate_keys": global_keys,
                }
            ),
        }
    if len(keys) > 1:
        return {
            "mapping_status": MAPPING_AMBIGUOUS,
            "key_mlbam": None,
            "mapping_evidence": json.dumps({"reason": "ambiguous_name", "name": norm, "candidates": keys}),
        }
    if len(keys) == 1:
        key = keys[0]
        if prior_key is not None and prior_key != key:
            return {
                "mapping_status": MAPPING_PROVIDER_CONFLICT,
                "key_mlbam": None,
                "mapping_evidence": json.dumps(
                    {
                        "reason": "provider_player_id_conflicts_with_name_match",
                        "provider_player_id": provider_player_id,
                        "prior_key_mlbam": prior_key,
                        "name_key_mlbam": key,
                    }
                ),
            }
        return {
            "mapping_status": MAPPING_MAPPED,
            "key_mlbam": key,
            "mapping_evidence": json.dumps({"reason": "unique_normalized_name", "name": norm, "key_mlbam": key}),
        }
    if prior_key is not None:
        return {
            "mapping_status": MAPPING_MAPPED,
            "key_mlbam": prior_key,
            "mapping_evidence": json.dumps(
                {"reason": "prior_provider_player_map_without_roster_hit", "provider_player_id": provider_player_id}
            ),
        }
    # Last-name-only unique fallback within game-scoped candidates.
    last = norm.split()[-1] if norm else ""
    if last:
        last_hits = scoped[scoped["name_norm"].str.endswith(" " + last) | (scoped["name_norm"] == last)]
        last_keys = sorted({int(x) for x in last_hits["key_mlbam"].dropna().astype(int).tolist()})
        if len(last_keys) > 1:
            return {
                "mapping_status": MAPPING_AMBIGUOUS,
                "key_mlbam": None,
                "mapping_evidence": json.dumps(
                    {"reason": "ambiguous_last_name", "last": last, "candidates": last_keys}
                ),
            }
    return {
        "mapping_status": MAPPING_UNMATCHED,
        "key_mlbam": None,
        "mapping_evidence": json.dumps({"reason": "name_not_in_candidates", "name": norm}),
    }


def contract_team_from_event(event: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    """Resolve market metadata.teamId to a display abbreviation when possible."""
    team_id = metadata.get("teamId")
    if team_id is None:
        return None
    for team in event.get("teams") or []:
        if str(team.get("id")) == str(team_id):
            return normalize_team_abbr(team.get("displayAbbreviation") or team.get("abbreviation"))
    return None


def one_hit_markets(event: dict) -> list[dict]:
    """Keep explicit 1+ hits only; no inference from marketing titles."""
    rows = []
    seen = set()
    for market in event.get("markets") or []:
        if (
            market.get("sportsMarketType") != "baseball_player_hits"
            or _number(market.get("line")) != 1
            or market.get("active") is not True
            or market.get("closed") is not False
            or market.get("hidden") is True
        ):
            continue
        slug = market.get("slug")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        rows.append(market)
    return sorted(rows, key=lambda m: m["slug"])


def load_player_candidates(path: str | None = None) -> pd.DataFrame:
    """Name/key candidates from published hitter log (research join aid only)."""
    path = path or "data/predictions/hitter_hit_log.csv"
    empty_cols = ["key_mlbam", "name_first", "name_last", "name_norm", "team"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=empty_cols)
    frame = pd.read_csv(path)
    cols = [c for c in ("key_mlbam", "name_first", "name_last", "team", "date") if c in frame.columns]
    if not {"key_mlbam", "name_first", "name_last"}.issubset(cols):
        return pd.DataFrame(columns=empty_cols)
    work = frame[cols].dropna(subset=["key_mlbam"]).copy()
    if "date" in work.columns:
        work = work.sort_values("date")
    keep = [c for c in ("key_mlbam", "name_first", "name_last", "team") if c in work.columns]
    out = work[keep].drop_duplicates(subset=["key_mlbam"], keep="last")
    out["name_norm"] = (
        out["name_first"].fillna("").astype(str) + " " + out["name_last"].fillna("").astype(str)
    ).map(normalize_person_name)
    return out


def write_host_ops_report(path: str | None = None) -> dict[str, Any]:
    """Document authorized collection host availability, sleep, and costs."""
    path = path or config.HIT_PROP_HOST_REPORT_PATH
    payload = {
        "generated_at_utc": utc_now_iso(),
        "research_only": True,
        "actionable": False,
        "edge_claimed": False,
        "host": config.HIT_PROP_COLLECTION_HOST,
        "workflow": config.HIT_PROP_COLLECTION_WORKFLOW,
        "runtime_note": config.HIT_PROP_COLLECTION_RUNTIME_NOTE,
        "paid_services_purchased": False,
        "paid_budget_authorized_usd": None,
        "missing_host": None,
        "schedule_windows_utc": [
            "cron 27,57 15-23 * * * (MLB afternoon/evening)",
            "cron 27,57 0-2 * * * (late West-coast finishes)",
        ],
        "sleep_behavior": (
            "Outside scheduled cron windows the GitHub Actions host is asleep; "
            "no prop capture runs until the next scheduled or workflow_dispatch trigger. "
            "Local optional runs fill gaps when the owner explicitly executes the script."
        ),
        "outage_behavior": (
            "Provider/API failures are retained in capture JSON errors and collection "
            "status reports. Failures do not invent quotes, do not place orders, and "
            "do not flip GAME_PREDICTION_MODE / BETTING_MODE. Resume uses "
            "capture_checkpoint.json to skip already-captured market slugs."
        ),
        "operating_costs": {
            "runner": "ubuntu-latest free-tier minutes",
            "additional_paid_services": "none authorized",
            "note": "Do not purchase paid runners/hosts without an explicit USD budget.",
        },
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "venue_label": "provisional_polymarket_us_unconfirmed",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return payload


def write_contract_example(
    report: dict[str, Any],
    *,
    path: str | None = None,
) -> dict[str, Any] | None:
    """Persist one mapped contract with rules + optional book for acceptance demos."""
    path = path or config.HIT_PROP_EXAMPLE_TRACE_PATH
    contracts = report.get("contracts") or []
    preferred = [
        c
        for c in contracts
        if c.get("game_mapping_status") == MAPPING_MAPPED
        and c.get("player_mapping_status") == MAPPING_MAPPED
        and c.get("rules_text")
    ]
    if not preferred:
        preferred = [c for c in contracts if c.get("rules_text")]
    if not preferred:
        return None
    # Prefer a row with a captured book when available.
    preferred.sort(key=lambda c: 0 if c.get("book_status") == "captured" else 1)
    row = preferred[0]
    rules = parse_contract_rules(row.get("rules_text"))
    settlement_examples = {
        "starter_with_hit": classify_contract_outcome(
            started=True,
            plate_appearances=3,
            at_bats=3,
            hits=1,
            game_status="Final",
            rules=rules,
        ),
        "walk_only_pa_zero_hits": classify_contract_outcome(
            started=True,
            plate_appearances=1,
            at_bats=0,
            hits=0,
            game_status="Final",
            rules=rules,
        ),
        "did_not_start": classify_contract_outcome(
            started=False,
            plate_appearances=0,
            at_bats=0,
            hits=0,
            game_status="Final",
            rules=rules,
        ),
        "missing_hits": classify_contract_outcome(
            started=True,
            plate_appearances=2,
            at_bats=2,
            hits=None,
            game_status="Final",
            rules=rules,
        ),
        "postponed": classify_contract_outcome(
            started=True,
            plate_appearances=1,
            at_bats=1,
            hits=0,
            game_status="Postponed",
            rules=rules,
        ),
    }
    payload = {
        "generated_at_utc": utc_now_iso(),
        "capture_id": report.get("capture_id"),
        "research_only": True,
        "actionable": False,
        "edge_claimed": False,
        "venue_id": report.get("venue_id") or "polymarket_us",
        "venue_label": "provisional_polymarket_us_unconfirmed",
        "contract": {
            "event_slug": row.get("event_slug"),
            "market_slug": row.get("market_slug"),
            "market_id": row.get("market_id"),
            "question": row.get("question"),
            "player_name": row.get("player_name"),
            "provider_game_id": row.get("provider_game_id"),
            "provider_player_id": row.get("provider_player_id"),
            "scheduled_start_utc": row.get("scheduled_start_utc"),
            "rules_text": row.get("rules_text"),
            "rules_hash": row.get("rules_hash"),
            "rules_version": row.get("rules_version") or HIT_PROP_RULES_VERSION,
            "fee_coefficient": row.get("fee_coefficient"),
            "book_status": row.get("book_status"),
            "book_request_time_utc": row.get("book_request_time_utc"),
            "book_receive_time_utc": row.get("book_receive_time_utc"),
            "yes_buy_price": row.get("yes_buy_price"),
            "yes_buy_size": row.get("yes_buy_size"),
            "no_buy_price": row.get("no_buy_price"),
            "no_buy_size": row.get("no_buy_size"),
        },
        "identity_mapping": {
            "game": {
                "game_pk": row.get("game_pk"),
                "mapping_status": row.get("game_mapping_status"),
                "mapping_evidence": row.get("game_mapping_evidence"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
            },
            "player": {
                "key_mlbam": row.get("key_mlbam"),
                "mapping_status": row.get("player_mapping_status"),
                "mapping_evidence": row.get("player_mapping_evidence"),
            },
        },
        "parsed_rules": rules,
        "settlement_examples": settlement_examples,
        "remaining_blockers": report.get("remaining_blockers") or [],
        "universe": report.get("universe") or {},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return payload


def load_provider_player_map(path: str | None = None) -> dict[str, int]:
    path = path or config.HIT_PROP_PROVIDER_PLAYER_MAP_PATH
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in (raw.get("provider_player_id_to_key_mlbam") or {}).items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def save_provider_player_map(mapping: dict[str, int], path: str | None = None) -> str:
    path = path or config.HIT_PROP_PROVIDER_PLAYER_MAP_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "updated_at_utc": utc_now_iso(),
        "note": "Research-only verified provider_player_id → key_mlbam. Conflicts quarantine.",
        "provider_player_id_to_key_mlbam": {str(k): int(v) for k, v in mapping.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def research_store_dir(path: str | None = None) -> str:
    return path or config.HIT_PROP_RESEARCH_STORE_DIR


def load_checkpoint(store_dir: str | None = None) -> dict[str, Any]:
    path = os.path.join(research_store_dir(store_dir), "capture_checkpoint.json")
    if not os.path.exists(path):
        return {
            "schema_version": config.HIT_PROP_RESEARCH_SCHEMA_VERSION,
            "completed_event_slugs": [],
            "completed_market_slugs": [],
            "n_book_requests_total": 0,
            "last_capture_id": None,
        }
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint: dict[str, Any], store_dir: str | None = None) -> str:
    store = research_store_dir(store_dir)
    os.makedirs(store, exist_ok=True)
    path = os.path.join(store, "capture_checkpoint.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def append_registry_rows(rows: list[dict[str, Any]], store_dir: str | None = None) -> str:
    store = research_store_dir(store_dir)
    os.makedirs(store, exist_ok=True)
    path = os.path.join(store, "contracts.csv")
    frame = pd.DataFrame(rows)
    for col in HIT_PROP_REGISTRY_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[HIT_PROP_REGISTRY_COLUMNS]
    if os.path.exists(path):
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def capture(
    adapter,
    *,
    date: dt.date,
    event_limit: int,
    book_limit: int,
    now: dt.datetime | None = None,
    schedule_games: pd.DataFrame | None = None,
    moneyline_registry: pd.DataFrame | None = None,
    player_candidates: pd.DataFrame | None = None,
    provider_player_map: dict[str, int] | None = None,
    persist_store: bool = False,
    store_dir: str | None = None,
    resume: bool = True,
    capture_id: str | None = None,
    define_universe: bool = True,
) -> dict:
    """Capture 1+ hit props with optional identity mapping and durable history.

    ``event_limit`` / ``book_limit`` remain hard request budgets. Universe
    denominators are always counted even when only a subset is fetched.
    """
    if event_limit < 1 or book_limit < 0:
        raise ValueError("event_limit must be positive; book_limit nonnegative")
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    capture_id = capture_id or f"hp_{now.strftime('%Y%m%dT%H%M%SZ')}"
    checkpoint = load_checkpoint(store_dir) if persist_store and resume else {
        "completed_event_slugs": [],
        "completed_market_slugs": [],
        "n_book_requests_total": 0,
    }
    done_events = set(checkpoint.get("completed_event_slugs") or [])
    done_markets = set(checkpoint.get("completed_market_slugs") or [])

    if schedule_games is None:
        try:
            schedule_games = market_contracts.load_mapping_schedule(
                start_date=date - dt.timedelta(days=1),
                lookahead_days=max(2, int(config.POLYMARKET_SCHEDULE_LOOKAHEAD_DAYS)),
            )
        except Exception:
            schedule_games = pd.DataFrame()
    if moneyline_registry is None:
        try:
            moneyline_registry = market_contracts.load_registry()
        except Exception:
            moneyline_registry = pd.DataFrame()
    if player_candidates is None:
        player_candidates = load_player_candidates()
    if provider_player_map is None:
        provider_player_map = load_provider_player_map()

    report = {
        "schema_version": config.HIT_PROP_RESEARCH_SCHEMA_VERSION,
        "venue_id": "polymarket_us",
        "requested_local_date": date.isoformat(),
        "capture_id": capture_id,
        "capture_started_at_utc": now.astimezone(dt.timezone.utc).isoformat(),
        "scope": "research_only_hit_prop_inventory",
        "event_limit": event_limit,
        "book_limit": book_limit,
        "actionable": False,
        "model_probability": None,
        "edge_claimed": False,
        "errors": [],
        "event_snapshots": [],
        "contracts": [],
        "universe": {},
        "exclusions": [],
        "remaining_blockers": [],
    }
    try:
        events = adapter.list_league_events(league="mlb", limit=200)
    except Exception as exc:
        report["errors"].append({"stage": "league", "error": str(exc)})
        report["status"] = "failed"
        report["remaining_blockers"] = ["league_event_discovery_failed"]
        return report

    eligible = []
    excluded_past = 0
    excluded_wrong_day = 0
    for event in events:
        start = _timestamp(event.get("startTime"))
        if start is None or not event.get("slug"):
            report["exclusions"].append({"slug": event.get("slug"), "reason": "missing_start_or_slug"})
            continue
        if start <= now:
            excluded_past += 1
            continue
        local_day = start.astimezone(schedule.LOCAL_TIMEZONE).date()
        if local_day != date:
            excluded_wrong_day += 1
            continue
        eligible.append((start, event["slug"]))

    universe_slugs = sorted({slug for _, slug in eligible})
    report["n_events_in_league_response"] = len(events)
    report["n_future_events_on_requested_date_in_response"] = len(universe_slugs)
    report["universe"] = {
        "definition": (
            f"All active future MLB Polymarket US events with start local-date "
            f"{date.isoformat()} ({schedule.LOCAL_TIMEZONE.key}) from the league list "
            f"response; 1+ baseball_player_hits markets only."
        ),
        "event_slugs": universe_slugs,
        "n_events": len(universe_slugs),
        "n_excluded_already_started": excluded_past,
        "n_excluded_wrong_local_day": excluded_wrong_day,
        "complete_denominator_claimed": define_universe and len(events) > 0,
        "note": (
            "League list may omit props; event-detail fetch is required for contracts. "
            "event_limit/book_limit bound requests — incomplete fetch is explicit, not silent."
        ),
    }

    attempts = 0
    persisted_rows: list[dict[str, Any]] = []
    # Prefer not-yet-completed events for restart recovery.
    ordered = sorted(set(eligible))
    ordered = [x for x in ordered if x[1] not in done_events] + [x for x in ordered if x[1] in done_events]
    for _, slug in ordered[:event_limit]:
        try:
            envelope = adapter.fetch_event_details(slug)
            event = envelope["event"]
            if event.get("slug") != slug:
                raise ValueError("Event slug differs from requested event")
            # Attach per-request timestamps when adapter provides them.
            report["event_snapshots"].append(
                {
                    **envelope,
                    "capture_id": capture_id,
                    "observed_at_utc": envelope.get("receive_time_utc") or utc_now_iso(),
                }
            )
        except Exception as exc:
            report["errors"].append({"stage": "event", "slug": slug, "error": str(exc)})
            continue

        game_map = match_event_to_game(
            event,
            schedule_games,
            moneyline_registry=moneyline_registry,
            requested_local_date=date,
        )
        for market in one_hit_markets(event):
            metadata = market.get("metadata") or {}
            rules = market.get("description") or ""
            rules_parsed = parse_contract_rules(rules)
            contract_team = contract_team_from_event(event, metadata)
            player_map = match_player_to_key_mlbam(
                metadata.get("playerName"),
                provider_player_id=metadata.get("playerId"),
                candidate_players=player_candidates,
                prior_provider_map=provider_player_map,
                home_team=game_map.get("home_team"),
                away_team=game_map.get("away_team"),
                contract_team=contract_team,
            )
            quarantined = (
                game_map["mapping_status"] in QUARANTINE_STATUSES
                or player_map["mapping_status"] in QUARANTINE_STATUSES
            )
            no_bet = [
                "research_only",
                "unvalidated_prop_model",
            ]
            if game_map["mapping_status"] != MAPPING_MAPPED:
                no_bet.append(f"game_mapping:{game_map['mapping_status']}")
            if player_map["mapping_status"] != MAPPING_MAPPED:
                no_bet.append(f"player_mapping:{player_map['mapping_status']}")
            if not rules_parsed["rules_text_present"]:
                no_bet.append("missing_contract_rules_text")
            else:
                # Rules present but evaluation still gated until labeled study.
                no_bet.append("participation_settlement_parsed_not_yet_evaluated")

            observed = utc_now_iso()
            row = {
                "capture_id": capture_id,
                "observed_at_utc": observed,
                "requested_local_date": date.isoformat(),
                "event_id": str(event.get("id") or ""),
                "event_slug": slug,
                "market_slug": market["slug"],
                "market_id": str(market.get("id") or ""),
                "question": market.get("question"),
                "scheduled_start_utc": market.get("gameStartTime") or event.get("startTime"),
                "provider_game_id": event.get("gameId"),
                "game_pk": game_map.get("game_pk"),
                "game_mapping_status": game_map.get("mapping_status"),
                "game_mapping_evidence": game_map.get("mapping_evidence"),
                "home_team": game_map.get("home_team"),
                "away_team": game_map.get("away_team"),
                "provider_player_id": metadata.get("playerId"),
                "key_mlbam": player_map.get("key_mlbam"),
                "player_name": metadata.get("playerName"),
                "player_mapping_status": player_map.get("mapping_status"),
                "player_mapping_evidence": player_map.get("mapping_evidence"),
                "stat": "hits",
                "threshold": 1,
                "rules_text": rules,
                "rules_hash": rules_hash(rules),
                "rules_version": rules_parsed.get("rules_version") or HIT_PROP_RULES_VERSION,
                **{k: rules_parsed[k] for k in (
                    "requires_starting_lineup",
                    "requires_plate_appearance",
                    "settlement_on_non_participation",
                    "postponement_window_days",
                    "includes_extra_innings",
                    "shortened_official_game_settles",
                    "nonbinary_settlement_possible",
                )},
                "fee_coefficient": _number(market.get("feeCoefficient")),
                "actionable": False,
                "research_only": True,
                "model_probability": None,
                "quarantined": quarantined,
                "no_bet_reasons": no_bet,
                "book_status": "not_requested_budget",
                "book": None,
                "book_request_time_utc": None,
                "book_receive_time_utc": None,
                "yes_buy_price": None,
                "yes_buy_size": None,
                "no_buy_price": None,
                "no_buy_size": None,
            }

            sides = market.get("marketSides") or []
            oriented = (
                any(s.get("long") is True and str(s.get("description")).lower() == "yes" for s in sides)
                and any(s.get("long") is False and str(s.get("description")).lower() == "no" for s in sides)
            )
            market_slug = market["slug"]
            skip_book = market_slug in done_markets and resume and persist_store
            if not oriented:
                row["book_status"] = "unverified_outcome_orientation"
                row["no_bet_reasons"].append("unverified_outcome_orientation")
            elif skip_book:
                row["book_status"] = "skipped_restart_already_captured"
            elif attempts < book_limit:
                attempts += 1
                req_t = utc_now_iso()
                row["book_request_time_utc"] = req_t
                try:
                    book = adapter.fetch_market_book(
                        market_slug,
                        market_id=row["market_id"],
                        fee_coefficient=row["fee_coefficient"],
                        tick_size=_number(market.get("orderPriceMinTickSize")),
                        min_trade_qty=_number(market.get("minimumTradeQty")),
                    )
                    row["book"] = _json_safe(book.to_dict())
                    row["book_status"] = "captured"
                    row["book_receive_time_utc"] = getattr(book, "receive_time_utc", None) or utc_now_iso()
                    bid, ask = _number(book.best_bid), _number(book.best_ask)
                    bid_size, ask_size = _number(book.best_bid_size), _number(book.best_ask_size)
                    crossed = bid is not None and ask is not None and bid >= ask
                    if book.eligible and not book.suspended and not crossed:
                        if ask is not None and 0 < ask < 1 and ask_size is not None and ask_size > 0:
                            row["yes_buy_price"], row["yes_buy_size"] = ask, ask_size
                        if bid is not None and 0 < bid < 1 and bid_size is not None and bid_size > 0:
                            row["no_buy_price"] = 1 - bid
                            row["no_buy_size"] = bid_size
                    else:
                        row["no_bet_reasons"].append(
                            "crossed_or_locked_book" if crossed else (book.eligibility_reason or "book_unavailable")
                        )
                    done_markets.add(market_slug)
                except Exception as exc:
                    row["book_status"] = "failed"
                    report["errors"].append({"stage": "book", "slug": market_slug, "error": str(exc)})
            report["contracts"].append(row)
            persisted_rows.append({**row, "schema_version": config.HIT_PROP_RESEARCH_SCHEMA_VERSION, "venue_id": "polymarket_us", "no_bet_reasons": "|".join(row["no_bet_reasons"])})

        done_events.add(slug)

    report["n_book_requests"] = attempts
    report["n_books_captured"] = sum(r["book_status"] == "captured" for r in report["contracts"])
    report["n_mapped_game_pk"] = sum(r.get("game_mapping_status") == MAPPING_MAPPED for r in report["contracts"])
    report["n_mapped_key_mlbam"] = sum(r.get("player_mapping_status") == MAPPING_MAPPED for r in report["contracts"])
    report["n_quarantined"] = sum(bool(r.get("quarantined")) for r in report["contracts"])
    report["status"] = (
        "partial"
        if report["errors"]
        else ("captured" if report["contracts"] else "no_eligible_contracts")
    )
    report["universe"]["n_contracts_in_fetched_events"] = len(report["contracts"])
    report["universe"]["n_events_fetched"] = len({r["event_slug"] for r in report["contracts"]})
    report["universe"]["n_events_not_fetched_due_to_budget"] = max(
        0, len(universe_slugs) - report["universe"]["n_events_fetched"]
    )

    blockers = []
    if report["universe"]["n_events_not_fetched_due_to_budget"] > 0:
        blockers.append("event_or_book_request_budget_incomplete_universe")
    if report["n_mapped_key_mlbam"] < len(report["contracts"]):
        blockers.append("player_identity_mapping_incomplete_or_quarantined")
    if report["n_mapped_game_pk"] < len(report["contracts"]):
        blockers.append("game_pk_mapping_incomplete_or_quarantined")
    blockers.extend(
        [
            "no_registered_prop_evaluation_outcomes_yet",
            "hitter_positive_ab_rates_not_valid_contract_probabilities",
            "BETTING_MODE_disabled_orders_absent",
            "venue_us_vs_international_unconfirmed",
        ]
    )
    report["remaining_blockers"] = blockers

    if persist_store:
        append_registry_rows(persisted_rows, store_dir=store_dir)
        # Persist raw report for audit.
        store = research_store_dir(store_dir)
        os.makedirs(os.path.join(store, "captures"), exist_ok=True)
        out_path = os.path.join(store, "captures", f"{capture_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(report), f, indent=2, allow_nan=False)
            f.write("\n")
        checkpoint.update(
            {
                "schema_version": config.HIT_PROP_RESEARCH_SCHEMA_VERSION,
                "completed_event_slugs": sorted(done_events),
                "completed_market_slugs": sorted(done_markets),
                "n_book_requests_total": int(checkpoint.get("n_book_requests_total") or 0) + attempts,
                "last_capture_id": capture_id,
                "updated_at_utc": utc_now_iso(),
            }
        )
        save_checkpoint(checkpoint, store_dir=store_dir)
        report["persisted_capture_path"] = out_path

    return report
