"""Daily bet-sizing report — hard-gated behind live betting readiness.

This script is intentionally fail-closed. It does NOT place bets and must
not print actionable "Real bet recommendations" unless:

- ``resolve_betting_mode()`` effective mode is exactly ``live``
- the betting promotion report exists and every required check passes
- a loadable residual artifact exists whose artifact_id matches the report
- picks use residual (non-legacy / non-fallback) probabilities
- market data is matched, pre-start, and not stale

When ``BETTING_MODE`` is ``disabled`` (the default) or resolves to
``shadow``, this script hard-exits without generating real-money advice.

Usage (will exit nonzero under current defaults):
    python scripts/recommend_bets.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import (
    config, game_evaluation, game_predictions, game_residual_model,
    market_odds, schedule,
)


def _print_confidence_banner(log_path: str) -> None:
    if not os.path.exists(log_path):
        print("No game-predictions log found yet - nothing to base a confidence check on.")
        return
    log = pd.read_csv(log_path, parse_dates=["date"])
    _, summary = game_evaluation.build_game_picks_export(log)
    n = int(summary.loc[0, "n_beat_closing_line_compared"]) if not summary.empty else 0
    rate = summary.loc[0, "beat_closing_line_rate"] if not summary.empty else float("nan")
    rate_str = f"{rate:.1%}" if pd.notna(rate) else "n/a"
    print("=" * 72, flush=True)
    print(
        f"Real beat_closing_line_rate so far: {rate_str} "
        f"(n={n} real market-compared games)",
        flush=True,
    )
    if n < config.KELLY_MIN_GAMES_FOR_CONFIDENCE:
        print(
            f"WARNING: n={n} is well below a real statistically meaningful sample "
            f"(config.KELLY_MIN_GAMES_FOR_CONFIDENCE={config.KELLY_MIN_GAMES_FOR_CONFIDENCE}).",
            flush=True,
        )
    print("=" * 72, flush=True)


def assert_live_betting_cli_allowed() -> tuple[str, dict]:
    """Hard-refuse unless betting mode resolves to live with a passed gate."""
    mode, meta = game_residual_model.resolve_betting_mode()
    if mode != "live":
        raise SystemExit(
            f"Refusing recommend_bets.py: effective BETTING_MODE is '{mode}' "
            f"(configured={meta.get('configured')}, "
            f"fallback={meta.get('fallback_reason')}). "
            "No real-money recommendations are generated while betting is "
            "disabled or shadow-only. READY FOR LIVE BETTING: NO"
        )

    ok, details = game_residual_model.evaluate_betting_promotion_gate(
        _load_betting_report()
    )
    if not ok:
        raise SystemExit(
            f"Refusing recommend_bets.py: betting promotion gate failed "
            f"({details.get('reason')}). READY FOR LIVE BETTING: NO"
        )

    art = game_residual_model.load_residual_model()
    if art is None:
        raise SystemExit(
            "Refusing recommend_bets.py: residual model artifact missing. "
            "READY FOR LIVE BETTING: NO"
        )
    report = _load_betting_report() or {}
    report_artifact = report.get("artifact_id")
    model_artifact = (art.metadata or {}).get("artifact_id")
    if not model_artifact or report_artifact != model_artifact:
        raise SystemExit(
            "Refusing recommend_bets.py: residual artifact_id does not match "
            f"validation report (model={model_artifact!r}, report={report_artifact!r}). "
            "READY FOR LIVE BETTING: NO"
        )
    return mode, meta


def _load_betting_report() -> dict | None:
    path = config.BETTING_PROMOTION_GATE_REPORT_PATH
    if not path or not os.path.exists(path):
        # Fall back to nested residual report which also carries the gate.
        alt = config.GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH
        if not alt or not os.path.exists(alt):
            return None
        path = alt
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_target_date_picks(log_path: str, target_date: pd.Timestamp) -> pd.DataFrame:
    if not os.path.exists(log_path):
        raise SystemExit(f"{log_path} does not exist - run the daily pipeline first.")

    log = pd.read_csv(log_path, parse_dates=["date"])
    day = log[(log["date"] == target_date) & (log["metric"] == "GamePick_Win_Probability")]
    if day.empty:
        raise SystemExit(
            f"No game picks logged for {target_date.date()} - run the daily pipeline for "
            f"this date first rather than betting on a stale or guessed date."
        )

    pending = day[day["game_played"].isna()]
    if pending.empty:
        raise SystemExit(
            f"Every logged game pick for {target_date.date()} is already resolved - nothing to bet on."
        )
    return pending


def _real_game_statuses(target_date) -> dict:
    games = schedule.fetch_todays_games(target_date)
    return dict(zip(games["game_pk"], games["status"]))


def _refuse_legacy_or_fallback_probabilities(picks: pd.DataFrame) -> None:
    if "probability_source" in picks.columns:
        bad = picks[
            ~picks["probability_source"].astype(str).str.contains("residual", case=False, na=False)
        ]
        if not bad.empty:
            raise SystemExit(
                "Refusing recommend_bets.py: picks use legacy/heuristic/fallback "
                "probabilities, not the residual model. READY FOR LIVE BETTING: NO"
            )
    # Schema uses ``fallback_used`` (not ``model_fallback_used``).
    if "fallback_used" in picks.columns and picks["fallback_used"].fillna(False).astype(bool).any():
        raise SystemExit(
            "Refusing recommend_bets.py: fallback_used=True on one or more picks. "
            "READY FOR LIVE BETTING: NO"
        )


def _refuse_invalid_market(market: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    if market is None or market.empty:
        raise SystemExit(
            "Refusing recommend_bets.py: no market data available. "
            "READY FOR LIVE BETTING: NO"
        )
    missing = [c for c in market_odds.REQUIRED_LIVE_MARKET_COLUMNS if c not in market.columns]
    if missing:
        raise SystemExit(
            "Refusing recommend_bets.py: legacy team-only market frame rejected "
            f"(missing {missing}). READY FOR LIVE BETTING: NO"
        )
    if (market["source_status"] != market_odds.SOURCE_OK).any():
        raise SystemExit(
            "Refusing recommend_bets.py: market rows are unmatched/ambiguous/"
            "post-start. READY FOR LIVE BETTING: NO"
        )
    if market["game_pk"].isna().any():
        raise SystemExit(
            "Refusing recommend_bets.py: market rows missing game_pk. "
            "READY FOR LIVE BETTING: NO"
        )
    roles = set(market["snapshot_role"].astype(str))
    if not roles.issubset(market_odds.LIVE_RECOMMENDATION_ROLES):
        raise SystemExit(
            "Refusing recommend_bets.py: market snapshot_role must be morning "
            f"or lineup_lock (got {sorted(roles)}). READY FOR LIVE BETTING: NO"
        )
    # Exact game_pk coverage for every pending pick.
    needed = {int(pk) for pk in picks["game_pk"].dropna().tolist()}
    have = {int(pk) for pk in market["game_pk"].dropna().tolist()}
    missing_pks = sorted(needed - have)
    if missing_pks:
        raise SystemExit(
            "Refusing recommend_bets.py: no exact game_pk market match for "
            f"{missing_pks}. READY FOR LIVE BETTING: NO"
        )
    return market


def _load_snapshot_backed_market(picks: pd.DataFrame) -> pd.DataFrame:
    """Load persisted odds snapshots; never use legacy team-only fetch."""
    snaps = market_odds.load_odds_snapshots()
    try:
        return market_odds.market_for_live_recommendations(
            snaps,
            required_game_pks=picks["game_pk"].tolist(),
        )
    except ValueError as exc:
        raise SystemExit(
            f"Refusing recommend_bets.py: {exc}. READY FOR LIVE BETTING: NO"
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-path", default="data/predictions/game_predictions.csv")
    parser.add_argument("--date", type=datetime.date.fromisoformat, default=schedule.today_local())
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--kelly-fraction", type=float, default=config.KELLY_FRACTION_MULTIPLIER)
    parser.add_argument("--min-edge", type=float, default=config.KELLY_MIN_EDGE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Fail closed BEFORE any recommendation table is computed.
    assert_live_betting_cli_allowed()

    _print_confidence_banner(args.log_path)

    target_date = pd.Timestamp(args.date)
    todays_picks = _load_target_date_picks(args.log_path, target_date)
    _refuse_legacy_or_fallback_probabilities(todays_picks)

    real_statuses = _real_game_statuses(args.date)
    scheduled_mask = todays_picks["game_pk"].map(real_statuses) == "Scheduled"
    not_scheduled = todays_picks[~scheduled_mask]
    if not not_scheduled.empty:
        print(
            f"Skipping {len(not_scheduled)} game(s) whose real MLB Stats API status isn't "
            f"'Scheduled' (already started/finished, or not found for {target_date.date()}) - "
            f"never bet on those."
        )
    todays_picks = todays_picks[scheduled_mask]
    if todays_picks.empty:
        print(f"No real still-scheduled games left to evaluate for {target_date.date()}.")
        return

    market = _load_snapshot_backed_market(todays_picks)
    market = _refuse_invalid_market(market, todays_picks)

    recommendations = game_predictions.advise_bets(
        todays_picks, market, args.kelly_fraction, args.min_edge,
    )
    if recommendations.empty:
        print(
            f"No qualifying edge found for {target_date.date()} - no bets recommended. "
            f"This is a real, expected outcome (most days should have none), not a bug."
        )
        return

    recommendations["units"] = recommendations["kelly_stake_fraction"] / config.UNIT_SIZE_FRACTION
    if args.bankroll is not None:
        recommendations["recommended_stake_dollars"] = (
            recommendations["kelly_stake_fraction"] * args.bankroll
        )

    # Live mode only reaches here; still label honestly.
    print(f"\nLive-gated bet sizing table for {target_date.date()} (manual placement only):")
    print(recommendations.to_string(index=False))

    if args.out:
        recommendations.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
