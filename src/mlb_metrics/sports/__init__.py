"""Sport-specific adapters for Polymarket research (schedules, identities, settlement).

Venue discovery, quotes, fees, and paper fill math stay in venues/quote_store/
paper_ledger. Sport adapters supply schedule rows and official winners only.
MLB keeps ``game_pk`` / ``key_mlbam``. Other sports use namespaced ``sport_game_key``.
"""

from __future__ import annotations

from mlb_metrics.sports.base import (
    SportCapabilities,
    SportScheduleAdapter,
    UnsupportedSportError,
    get_sport_adapter,
    namespaced_game_key,
    parse_namespaced_game_key,
)
from mlb_metrics.sports.mlb import MlbSportAdapter
from mlb_metrics.sports.nfl import NflSportAdapter

__all__ = [
    "SportCapabilities",
    "SportScheduleAdapter",
    "UnsupportedSportError",
    "get_sport_adapter",
    "namespaced_game_key",
    "parse_namespaced_game_key",
    "MlbSportAdapter",
    "NflSportAdapter",
]
