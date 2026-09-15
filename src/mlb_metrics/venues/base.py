"""Common Polymarket / venue interfaces for read-only market research.

No order placement, wallet, or account APIs live here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence


class UnsupportedVenueError(NotImplementedError):
    """Raised when a venue adapter is intentionally not implemented yet."""


@dataclass(frozen=True)
class VenueCapabilities:
    venue_id: str
    public_market_data: bool
    order_book: bool
    historical_display_prices: bool
    fee_schedule: bool
    order_placement: bool = False


@dataclass(frozen=True)
class FeeSchedule:
    fee_version: str
    effective_from_utc: str
    taker_theta: float
    maker_rebate_theta: float
    notes: str = ""

    def taker_fee(self, price: float, contracts: float = 1.0) -> float:
        """Fee = theta * C * p * (1-p); banker's rounding left to callers."""
        p = float(price)
        c = float(contracts)
        if not (0.0 < p < 1.0) or c <= 0:
            return 0.0
        return float(self.taker_theta) * c * p * (1.0 - p)


@dataclass
class VenueOutcome:
    outcome_id: str
    label: str
    team_abbr: str | None
    is_long: bool
    display_price: float | None
    side_quote: float | None
    tradable: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass
class VenueMarket:
    venue_id: str
    event_id: str
    event_slug: str
    market_id: str
    market_slug: str
    market_type: str
    title: str
    question: str
    scheduled_start_utc: str | None
    trading_status: str
    active: bool
    closed: bool
    line: float | None
    tick_size: float | None
    min_trade_qty: float | None
    fee_coefficient: float | None
    home_team: str | None
    away_team: str | None
    provider_game_id: str | None
    outcomes: list[VenueOutcome]
    rules_text: str
    rules_hash: str
    raw_event: dict[str, Any] = field(default_factory=dict, repr=False)
    raw_market: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "event_id": self.event_id,
            "event_slug": self.event_slug,
            "market_id": self.market_id,
            "market_slug": self.market_slug,
            "market_type": self.market_type,
            "title": self.title,
            "question": self.question,
            "scheduled_start_utc": self.scheduled_start_utc,
            "trading_status": self.trading_status,
            "active": self.active,
            "closed": self.closed,
            "line": self.line,
            "tick_size": self.tick_size,
            "min_trade_qty": self.min_trade_qty,
            "fee_coefficient": self.fee_coefficient,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "provider_game_id": self.provider_game_id,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "rules_text": self.rules_text,
            "rules_hash": self.rules_hash,
        }


@dataclass
class MarketBookQuote:
    venue_id: str
    market_id: str
    market_slug: str
    request_time_utc: str
    receive_time_utc: str
    source_transact_time_utc: str | None
    market_state: str | None
    best_bid: float | None
    best_bid_size: float | None
    best_ask: float | None
    best_ask_size: float | None
    bids: list[dict[str, float]]
    asks: list[dict[str, float]]
    display_current_px: float | None
    last_trade_px: float | None
    fee_version: str
    fee_coefficient: float | None
    tick_size: float | None
    min_trade_qty: float | None
    suspended: bool
    raw_response_hash: str
    data_class: str = "order_book"
    eligible: bool = True
    eligibility_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VenueAdapter(Protocol):
    venue_id: str

    def capabilities(self) -> VenueCapabilities: ...

    def list_mlb_moneyline_markets(self, *, limit: int = 200) -> list[VenueMarket]: ...

    def fetch_market_book(self, market_slug: str, **kwargs: Any) -> MarketBookQuote: ...

    def fee_schedule_as_of(self, as_of_utc: str | None = None) -> FeeSchedule: ...


def get_venue_adapter(venue_id: str | None = None) -> VenueAdapter:
    from mlb_metrics import config
    from mlb_metrics.venues import polymarket_international, polymarket_us

    selected = venue_id or config.POLYMARKET_VENUE_SELECTED
    if selected == polymarket_us.VENUE_ID:
        return polymarket_us.PolymarketUSAdapter()
    if selected == polymarket_international.VENUE_ID:
        return polymarket_international.PolymarketInternationalAdapter()
    raise UnsupportedVenueError(f"Unknown venue_id={selected!r}")
