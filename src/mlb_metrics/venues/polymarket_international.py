"""Polymarket international venue — explicit unsupported stub.

Public international market data exists, but this project has not verified
the owner's account venue or international contract semantics. Do not infer
US results apply here.
"""

from __future__ import annotations

from mlb_metrics.venues.base import UnsupportedVenueError, VenueCapabilities

VENUE_ID = "polymarket_international"


class PolymarketInternationalAdapter:
    venue_id = VENUE_ID

    def capabilities(self) -> VenueCapabilities:
        return VenueCapabilities(
            venue_id=VENUE_ID,
            public_market_data=False,
            order_book=False,
            historical_display_prices=False,
            fee_schedule=False,
            order_placement=False,
        )

    def list_mlb_moneyline_markets(self, *, limit: int = 200):
        raise UnsupportedVenueError(
            "polymarket_international is not implemented until the owner's "
            "venue is confirmed and international contract/fee semantics are tested"
        )

    def fetch_market_book(self, market_slug: str, **kwargs):
        raise UnsupportedVenueError(
            "polymarket_international order books are not supported yet"
        )

    def fee_schedule_as_of(self, as_of_utc: str | None = None):
        raise UnsupportedVenueError(
            "polymarket_international fee schedules are not supported yet"
        )
