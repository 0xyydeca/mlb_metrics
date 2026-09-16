"""MLB sport adapter — preserves game_pk / StatsAPI semantics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlb_metrics import market_contracts, schedule
from mlb_metrics.sports.base import (
    SPORT_SCHEDULE_COLUMNS,
    SportCapabilities,
    namespaced_game_key,
)


class MlbSportAdapter:
    sport_id = "mlb"

    def capabilities(self) -> SportCapabilities:
        return SportCapabilities(
            sport_id="mlb",
            schedule=True,
            official_results=True,
            participant_features=True,
            polymarket_league_slug="mlb",
            game_key_namespace="mlb",
            notes="Native identity remains integer game_pk; sport_game_key is mlb:{game_pk}.",
        )

    def load_schedule_for_mapping(
        self,
        *,
        lookahead_days: int = 3,
        as_of_local=None,
        prefer_live: bool = True,
        schedule_snapshots_path: str | None = None,
    ) -> pd.DataFrame:
        """Reuse existing MLB mapping schedule (live + snapshots)."""
        games = market_contracts.load_mapping_schedule(
            schedule_snapshots_path=schedule_snapshots_path,
            lookahead_days=lookahead_days,
            prefer_live=prefer_live,
        )
        if as_of_local is not None and not games.empty and "date" in games.columns:
            # Optional filter; default mapping uses full lookahead window.
            pass
        if games.empty:
            return pd.DataFrame(columns=SPORT_SCHEDULE_COLUMNS + ["game_pk"])
        out = games.copy()
        out["sport_id"] = "mlb"
        out["sport_game_key"] = out["game_pk"].map(lambda x: namespaced_game_key("mlb", int(x)))
        for col in ("status", "home_score", "away_score"):
            if col not in out.columns:
                out[col] = pd.NA
        cols = list(dict.fromkeys(SPORT_SCHEDULE_COLUMNS + ["game_pk"]))
        for col in cols:
            if col not in out.columns:
                out[col] = pd.NA
        return out[cols]

    def official_winners(self, results: pd.DataFrame) -> dict[str, str]:
        winners: dict[str, str] = {}
        if results is None or results.empty:
            return winners
        for _, row in results.iterrows():
            if pd.isna(row.get("game_pk")):
                continue
            status = str(row.get("status") or "")
            if status != "Final":
                continue
            try:
                hs = float(row["home_score"])
                aws = float(row["away_score"])
            except (TypeError, ValueError, KeyError):
                continue
            gpk = int(row["game_pk"])
            key = namespaced_game_key("mlb", gpk)
            if hs > aws:
                winners[key] = str(row["home_team"])
            elif aws > hs:
                winners[key] = str(row["away_team"])
        return winners

    def fetch_results_for_local_date(self, local_date: Any) -> pd.DataFrame:
        return schedule.fetch_game_results(local_date)
