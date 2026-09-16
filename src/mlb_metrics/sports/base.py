"""Cross-sport schedule / identity contracts (no venue I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd


class UnsupportedSportError(NotImplementedError):
    """Raised when a sport adapter is not implemented."""


@dataclass(frozen=True)
class SportCapabilities:
    sport_id: str
    schedule: bool
    official_results: bool
    participant_features: bool
    polymarket_league_slug: str | None
    game_key_namespace: str
    notes: str = ""


# Canonical columns for sport schedule frames used by contract mapping.
SPORT_SCHEDULE_COLUMNS = [
    "sport_id",
    "sport_game_key",  # namespaced for non-MLB; MLB may also set game_pk
    "home_team",
    "away_team",
    "game_datetime",
    "date",
    "status",
    "home_score",
    "away_score",
]


def namespaced_game_key(sport_id: str, native_id: Any) -> str:
    """Build a sport-scoped identity. Never coerce foreign ids into game_pk."""
    return f"{sport_id}:{native_id}"


def parse_namespaced_game_key(key: str) -> tuple[str, str]:
    if ":" not in str(key):
        raise ValueError(f"Expected namespaced sport_game_key, got {key!r}")
    sport, native = str(key).split(":", 1)
    return sport, native


class SportScheduleAdapter(Protocol):
    sport_id: str

    def capabilities(self) -> SportCapabilities: ...

    def load_schedule_for_mapping(
        self,
        *,
        lookahead_days: int = 3,
        as_of_local=None,
    ) -> pd.DataFrame:
        """Return rows with at least home/away/start and sport_game_key."""
        ...

    def official_winners(self, results: pd.DataFrame) -> dict[str, str]:
        """Map sport_game_key -> winning team abbr for Final games only."""
        ...


def get_sport_adapter(sport_id: str) -> SportScheduleAdapter:
    from mlb_metrics.sports import mlb as mlb_mod
    from mlb_metrics.sports import nfl as nfl_mod

    sid = str(sport_id).lower()
    if sid == "mlb":
        return mlb_mod.MlbSportAdapter()
    if sid == "nfl":
        return nfl_mod.NflSportAdapter()
    raise UnsupportedSportError(f"Unknown sport_id={sport_id!r}")
