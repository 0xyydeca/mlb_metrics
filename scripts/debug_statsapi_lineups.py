"""Stage A: inspect the real MLB Stats API lineup-related response shapes.

Prints a compact sanitized schema summary for schedule / lineup hydrates and
optional boxscore payloads. Does **not** build production parsers against an
assumed shape — run this via the Debug statsapi lineups GitHub Actions
workflow (real network access), then only after the live shape is confirmed
should fixtures be saved and production parsing finalized.

Usage:
    python scripts/debug_statsapi_lineups.py [YYYY-MM-DD]
    python scripts/debug_statsapi_lineups.py 2026-09-03 --game-pk 776001
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return type(value).__name__


def _schema_summary(obj: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 4) -> list[str]:
    """Compact path -> type lines; truncates deep trees and list samples."""
    lines: list[str] = []
    if depth > max_depth:
        lines.append(f"{prefix}: <max-depth>")
        return lines
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            val = obj[key]
            lines.append(f"{path}: {_type_name(val)}")
            if isinstance(val, (dict, list)) and depth < max_depth:
                lines.extend(_schema_summary(val, prefix=path, depth=depth + 1, max_depth=max_depth))
    elif isinstance(obj, list):
        lines.append(f"{prefix}: list[{len(obj)}]")
        if obj:
            lines.append(f"{prefix}[0]: {_type_name(obj[0])}")
            if isinstance(obj[0], (dict, list)):
                lines.extend(_schema_summary(obj[0], prefix=f"{prefix}[0]", depth=depth + 1, max_depth=max_depth))
    return lines


def _sanitize(obj: Any) -> Any:
    """Drop bulky / PII-ish fields for printed samples; keep structural keys."""
    drop_keys = {
        "copyright", "link", "content", "venue", "broadcasts", "weather",
        "review", "flags", "alerts", "ticketLink", "calendarEventID",
    }
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in drop_keys:
                continue
            if k in ("note", "description") and isinstance(v, str) and len(v) > 80:
                out[k] = v[:77] + "..."
            else:
                out[k] = _sanitize(v)
        return out
    if isinstance(obj, list):
        # Keep at most two elements for samples
        return [_sanitize(x) for x in obj[:2]]
    return obj


def _looks_like_lineup_related(key: str) -> bool:
    key_l = key.lower()
    return any(tok in key_l for tok in (
        "lineup", "batter", "batting", "order", "player", "roster", "boxscore",
    ))


def _collect_interesting_paths(obj: Any, prefix: str = "", found: list | None = None) -> list[str]:
    found = found if found is not None else []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if _looks_like_lineup_related(str(k)):
                found.append(f"{path}: {_type_name(v)}")
            _collect_interesting_paths(v, path, found)
    elif isinstance(obj, list) and obj and isinstance(obj[0], (dict, list)):
        _collect_interesting_paths(obj[0], f"{prefix}[0]", found)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: today per statsapi)")
    parser.add_argument("--game-pk", type=int, default=None, help="Optional gamePk for boxscore probe")
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    try:
        import statsapi
    except ImportError:
        print("MLB-StatsAPI is not installed. pip install MLB-StatsAPI", file=sys.stderr)
        sys.exit(1)

    print(f"statsapi version: {getattr(statsapi, '__version__', 'unknown')}")
    print("NOTE: Stage A only — do not wire production parsing until this shape is confirmed.")
    print("Target fields of interest: game_pk, player MLBAM id, batting order,")
    print("lineup confirmation status, game start time.\n")

    # Probe 1: schedule with lineup-oriented hydrates
    hydrates = [
        "lineups",
        "lineups,probablePitcher,team",
        "probablePitcher,team,lineups",
    ]
    for hydrate in hydrates:
        print(f"=== schedule hydrate={hydrate!r} ===")
        params = {"sportId": 1, "hydrate": hydrate}
        if args.date:
            params["date"] = args.date
        try:
            raw = statsapi.get("schedule", params)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        print(f"  top-level keys: {sorted(raw.keys())}")
        dates = raw.get("dates") or []
        print(f"  n_dates: {len(dates)}")
        if not dates:
            continue
        games = dates[0].get("games") or []
        print(f"  n_games on {dates[0].get('date')}: {len(games)}")
        if not games:
            continue
        game0 = games[0]
        print(f"  game[0] keys: {sorted(game0.keys())}")
        interesting = _collect_interesting_paths(game0)
        print("  lineup-related paths:")
        for line in interesting[:40]:
            print(f"    {line}")
        if len(interesting) > 40:
            print(f"    ... ({len(interesting) - 40} more)")
        print("  schema summary (sanitized sample game):")
        for line in _schema_summary(_sanitize(game0), max_depth=args.max_depth)[:80]:
            print(f"    {line}")
        # Capture a game_pk for boxscore if not provided
        if args.game_pk is None:
            args.game_pk = game0.get("gamePk") or game0.get("game_pk")

    # Probe 2: boxscore for one game
    if args.game_pk:
        print(f"\n=== boxscore_data(gamePk={args.game_pk}) ===")
        try:
            box = statsapi.boxscore_data(args.game_pk)
            print(f"  top-level keys: {sorted(box.keys()) if isinstance(box, dict) else type(box)}")
            if isinstance(box, dict):
                interesting = _collect_interesting_paths(box)
                print("  lineup-related paths:")
                for line in interesting[:50]:
                    print(f"    {line}")
                print("  schema summary (sanitized):")
                for line in _schema_summary(_sanitize(box), max_depth=min(3, args.max_depth))[:60]:
                    print(f"    {line}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")

        print(f"\n=== statsapi.get('game', {{'gamePk': {args.game_pk}}}) optional ===")
        try:
            game = statsapi.get("game", {"gamePk": args.game_pk})
            print(f"  top-level keys: {sorted(game.keys()) if isinstance(game, dict) else type(game)}")
            if isinstance(game, dict):
                interesting = _collect_interesting_paths(game)
                print("  lineup-related paths (first 40):")
                for line in interesting[:40]:
                    print(f"    {line}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\n=== Stage A checklist ===")
    print("Confirm presence of:")
    print("  [ ] gamePk / game_pk")
    print("  [ ] player id (MLBAM)")
    print("  [ ] batting order field")
    print("  [ ] confirmed vs probable/unconfirmed status signal")
    print("  [ ] gameDate / start time")
    print("Then save a real JSON fixture and unlock production parsing")
    print("(config.LINEUP_API_SCHEMA_CONFIRMED) with the verified field paths.")


if __name__ == "__main__":
    main()
