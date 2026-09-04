"""Lineup-lock prediction refresh.

Runs on a short interval after the morning Daily Update. Immediately exits
when no unstarted game falls within ``config.LINEUP_LOCK_WINDOW_HOURS``.
Only games whose confirmed lineup content newly appeared or changed are
recomputed; games that have already started are never updated.

Usage:
    python scripts/run_lineup_lock_update.py
    python scripts/run_lineup_lock_update.py --as-of-date 2026-09-03
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, lineup_snapshots, pipeline, schedule


def _log_run(record: dict) -> None:
    path = config.LINEUP_LOCK_SHADOW_DECISIONS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame = pd.DataFrame([record])
    if os.path.exists(path):
        prev = pd.read_csv(path)
        frame = pd.concat([prev, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def run_lineup_lock(
    as_of_date: datetime.date,
    *,
    window_hours: float | None = None,
    now_utc: pd.Timestamp | None = None,
    lineup_snapshot_frame: pd.DataFrame | None = None,
    raw_dir: str = "data/raw",
    output_dir: str = "docs/data",
    predictions_dir: str = "data/predictions",
    persist_raw: bool = False,
) -> dict:
    now = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    window_hours = float(config.LINEUP_LOCK_WINDOW_HOURS if window_hours is None else window_hours)

    try:
        schedule_games = schedule.fetch_todays_games(as_of_date)
    except Exception as exc:
        print(f"lineup-lock: schedule fetch failed ({exc}); exiting.")
        _log_run({
            "fetched_at_utc": lineup_snapshots.utc_now_iso(),
            "as_of_date": str(as_of_date),
            "status": "schedule_fetch_failed",
            "error": str(exc),
            "changed_game_pks": "",
        })
        return {"status": "schedule_fetch_failed", "changed_game_pks": set()}

    upcoming = lineup_snapshots.games_in_upcoming_window(
        schedule_games, now_utc=now, window_hours=window_hours,
    )
    if upcoming.empty:
        print(
            f"lineup-lock: no unstarted games within {window_hours}h window; exiting."
        )
        _log_run({
            "fetched_at_utc": lineup_snapshots.utc_now_iso(),
            "as_of_date": str(as_of_date),
            "status": "no_games_in_window",
            "changed_game_pks": "",
            "window_hours": window_hours,
        })
        return {"status": "no_games_in_window", "changed_game_pks": set()}

    previous_latest = lineup_snapshots.empty_snapshot_frame()
    latest_path = config.LINEUP_SNAPSHOT_LATEST_PATH
    if os.path.exists(latest_path):
        previous_latest = lineup_snapshots.normalize_snapshot_frame(pd.read_csv(latest_path))

    if lineup_snapshot_frame is not None:
        current = lineup_snapshots.normalize_snapshot_frame(lineup_snapshot_frame)
    else:
        try:
            current = lineup_snapshots.fetch_lineup_snapshots(as_of_date)
        except Exception as exc:
            print(f"lineup-lock: lineup fetch failed ({exc}); exiting.")
            _log_run({
                "fetched_at_utc": lineup_snapshots.utc_now_iso(),
                "as_of_date": str(as_of_date),
                "status": "lineup_fetch_failed",
                "error": str(exc),
                "changed_game_pks": "",
            })
            return {"status": "lineup_fetch_failed", "changed_game_pks": set()}

    # Restrict to unstarted games; annotate late scratches vs previous latest.
    current = lineup_snapshots.filter_snapshots_to_unstarted(current, now_utc=now)
    current = lineup_snapshots.annotate_scratches(previous_latest, current)
    if not current.empty:
        lineup_snapshots.persist_snapshots(current)

    upcoming_pks = set(pd.to_numeric(upcoming["game_pk"], errors="coerce").dropna().astype(int))
    prev_in_window = previous_latest[previous_latest["game_pk"].isin(upcoming_pks)] if not previous_latest.empty else previous_latest
    cur_in_window = current[current["game_pk"].isin(upcoming_pks)] if not current.empty else current
    changed = lineup_snapshots.changed_game_pks(prev_in_window, cur_in_window)
    # Only recompute games still unstarted and in the window.
    changed &= upcoming_pks
    started = {
        int(pk) for pk in upcoming_pks
        if lineup_snapshots.game_has_started(
            upcoming.loc[upcoming["game_pk"] == pk, "game_datetime"].iloc[0]
            if (upcoming["game_pk"] == pk).any() else None,
            now_utc=now,
        )
    }
    changed -= started

    if not changed:
        print("lineup-lock: no newly confirmed/changed lineups in window; exiting.")
        _log_run({
            "fetched_at_utc": lineup_snapshots.utc_now_iso(),
            "as_of_date": str(as_of_date),
            "status": "no_changes",
            "changed_game_pks": "",
            "window_hours": window_hours,
            "upcoming_game_count": len(upcoming_pks),
        })
        return {"status": "no_changes", "changed_game_pks": set()}

    print(f"lineup-lock: recomputing game_pks={sorted(changed)}")
    pipeline.run(
        as_of_date,
        raw_dir=raw_dir,
        output_dir=output_dir,
        predictions_dir=predictions_dir,
        persist_raw=persist_raw,
        log_predictions=True,
        prediction_snapshot_type="lineup_lock",
        lineup_snapshot_frame=current,
        game_pk_filter=changed,
    )
    _log_run({
        "fetched_at_utc": lineup_snapshots.utc_now_iso(),
        "as_of_date": str(as_of_date),
        "status": "recomputed",
        "changed_game_pks": ",".join(str(x) for x in sorted(changed)),
        "window_hours": window_hours,
        "upcoming_game_count": len(upcoming_pks),
    })
    return {"status": "recomputed", "changed_game_pks": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lineup-lock prediction refresh.")
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--window-hours", type=float, default=None)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="docs/data")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions")
    parser.add_argument(
        "--persist-raw",
        action="store_true",
        help="Persist Statcast pull (default off — morning run already did).",
    )
    args = parser.parse_args()
    as_of_date = (
        datetime.date.fromisoformat(args.as_of_date) if args.as_of_date else schedule.today_local()
    )
    result = run_lineup_lock(
        as_of_date,
        window_hours=args.window_hours,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        predictions_dir=args.predictions_dir,
        persist_raw=args.persist_raw,
    )
    print(f"lineup-lock done: {result['status']}")


if __name__ == "__main__":
    main()
