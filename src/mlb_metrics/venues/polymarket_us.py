"""Polymarket US public gateway adapter (read-only).

Uses ``https://gateway.polymarket.us`` league/events and market book endpoints.
Does not place orders or require trading credentials.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Sequence

from mlb_metrics import config
from mlb_metrics.venues.base import (
    FeeSchedule,
    MarketBookQuote,
    VenueCapabilities,
    VenueMarket,
    VenueOutcome,
)

VENUE_ID = "polymarket_us"
MONEYLINE_TYPE_V2 = "SPORTS_MARKET_TYPE_MONEYLINE"
USER_AGENT = "mlb_metrics-polymarket-research/1.0 (+read-only; no trading)"
# Price unit: USD cost per Yes share in [0, 1]. Size unit: contracts.


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _amount_pair(entry: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(entry, dict):
        return None, None
    return _to_float(entry.get("px")), _to_float(entry.get("qty"))


def rules_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def select_fee_schedule(
    schedules: Sequence[dict[str, Any]] | None = None,
    *,
    as_of_utc: str | None = None,
) -> FeeSchedule:
    rows = list(schedules or config.POLYMARKET_US_FEE_SCHEDULES)
    as_of = as_of_utc or _utc_now_iso()
    as_of_ts = pd_timestamp(as_of)
    chosen = rows[0]
    for row in sorted(rows, key=lambda r: r["effective_from_utc"]):
        if pd_timestamp(row["effective_from_utc"]) <= as_of_ts:
            chosen = row
    return FeeSchedule(
        fee_version=str(chosen["fee_version"]),
        effective_from_utc=str(chosen["effective_from_utc"]),
        taker_theta=float(chosen["taker_theta"]),
        maker_rebate_theta=float(chosen["maker_rebate_theta"]),
        notes=str(chosen.get("notes") or ""),
    )


def pd_timestamp(value: str):
    # Local helper avoids importing pandas at module import for tiny scripts;
    # keep behavior aligned with the rest of the project.
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def normalize_team_abbr(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def home_away_from_event_teams(teams: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    home = away = None
    for team in teams or []:
        abbr = normalize_team_abbr(team.get("displayAbbreviation") or team.get("abbreviation"))
        ordering = str(team.get("ordering") or "").lower()
        if ordering == "home":
            home = abbr
        elif ordering == "away":
            away = abbr
    if home and away:
        return home, away
    # Fallback: do not guess home/away from list order alone.
    return home, away


def parse_moneyline_market(event: dict[str, Any], market: dict[str, Any]) -> VenueMarket:
    teams = list(event.get("teams") or [])
    home, away = home_away_from_event_teams(teams)
    if (not home or not away) and len(teams) >= 2:
        # Secondary: marketSides team.ordering is authoritative when present.
        for side in market.get("marketSides") or []:
            team = side.get("team") or {}
            abbr = normalize_team_abbr(team.get("displayAbbreviation") or team.get("abbreviation"))
            ordering = str(team.get("ordering") or "").lower()
            if ordering == "home":
                home = abbr
            elif ordering == "away":
                away = abbr

    outcomes_labels = [str(x) for x in _parse_json_list(market.get("outcomes"))]
    outcome_prices = [_to_float(x) for x in _parse_json_list(market.get("outcomePrices"))]
    sides = list(market.get("marketSides") or [])
    outcomes: list[VenueOutcome] = []
    if sides:
        for side in sides:
            team = side.get("team") or {}
            outcomes.append(
                VenueOutcome(
                    outcome_id=str(side.get("id")),
                    label=str(side.get("description") or team.get("name") or ""),
                    team_abbr=normalize_team_abbr(
                        team.get("displayAbbreviation") or team.get("abbreviation")
                    ),
                    is_long=bool(side.get("long")),
                    display_price=_to_float(side.get("price")),
                    side_quote=_to_float(side.get("quote")),
                    tradable=bool(side.get("tradable", True)),
                    raw=side,
                )
            )
    else:
        for idx, label in enumerate(outcomes_labels):
            outcomes.append(
                VenueOutcome(
                    outcome_id=f"{market.get('id')}:{idx}",
                    label=label,
                    team_abbr=None,
                    is_long=idx == 0,
                    display_price=outcome_prices[idx] if idx < len(outcome_prices) else None,
                    side_quote=None,
                    tradable=True,
                    raw={},
                )
            )

    rules_text = str(market.get("description") or market.get("question") or market.get("title") or "")
    return VenueMarket(
        venue_id=VENUE_ID,
        event_id=str(event.get("id")),
        event_slug=str(event.get("slug") or ""),
        market_id=str(market.get("id")),
        market_slug=str(market.get("slug") or ""),
        market_type="moneyline",
        title=str(market.get("title") or event.get("title") or ""),
        question=str(market.get("question") or ""),
        scheduled_start_utc=str(
            market.get("gameStartTime") or event.get("startTime") or event.get("startDate") or ""
        )
        or None,
        trading_status=str(market.get("status") or ("closed" if market.get("closed") else "open")),
        active=bool(market.get("active")),
        closed=bool(market.get("closed")),
        line=_to_float(market.get("line")),
        tick_size=_to_float(market.get("orderPriceMinTickSize")),
        min_trade_qty=_to_float(market.get("minimumTradeQty")),
        fee_coefficient=_to_float(market.get("feeCoefficient")),
        home_team=home,
        away_team=away,
        provider_game_id=str(event.get("gameId")) if event.get("gameId") is not None else None,
        outcomes=outcomes,
        rules_text=rules_text,
        rules_hash=rules_hash(rules_text),
        raw_event=event,
        raw_market=market,
    )


def parse_book_response(
    payload: dict[str, Any],
    *,
    market_id: str,
    market_slug: str,
    request_time_utc: str,
    receive_time_utc: str,
    fee: FeeSchedule,
    fee_coefficient: float | None = None,
    tick_size: float | None = None,
    min_trade_qty: float | None = None,
) -> MarketBookQuote:
    raw = json.dumps(payload, sort_keys=True, default=str)
    md = payload.get("marketData") if isinstance(payload, dict) else None
    if not isinstance(md, dict):
        md = payload if isinstance(payload, dict) else {}
    bids_raw = list(md.get("bids") or [])
    asks_raw = list(md.get("offers") or md.get("asks") or [])
    bids = []
    asks = []
    for entry in bids_raw:
        px, qty = _amount_pair(entry)
        if px is None:
            continue
        bids.append({"price": px, "size": qty if qty is not None else float("nan")})
    for entry in asks_raw:
        px, qty = _amount_pair(entry)
        if px is None:
            continue
        asks.append({"price": px, "size": qty if qty is not None else float("nan")})
    best_bid = bids[0]["price"] if bids else None
    best_bid_size = bids[0]["size"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    best_ask_size = asks[0]["size"] if asks else None
    stats = md.get("stats") if isinstance(md.get("stats"), dict) else {}
    state = md.get("state")
    suspended = state not in (None, "MARKET_STATE_OPEN", "MARKET_STATE_PREOPEN")
    eligible = bool(best_ask is not None and not suspended)
    reason = None
    if suspended:
        reason = f"market_state:{state}"
    elif best_ask is None:
        reason = "missing_ask"
    return MarketBookQuote(
        venue_id=VENUE_ID,
        market_id=str(market_id),
        market_slug=str(market_slug),
        request_time_utc=request_time_utc,
        receive_time_utc=receive_time_utc,
        source_transact_time_utc=str(md.get("transactTime")) if md.get("transactTime") else None,
        market_state=str(state) if state is not None else None,
        best_bid=best_bid,
        best_bid_size=best_bid_size,
        best_ask=best_ask,
        best_ask_size=best_ask_size,
        bids=bids,
        asks=asks,
        display_current_px=_to_float((stats or {}).get("currentPx")),
        last_trade_px=_to_float((stats or {}).get("lastTradePx")),
        fee_version=fee.fee_version,
        fee_coefficient=fee_coefficient,
        tick_size=tick_size,
        min_trade_qty=min_trade_qty,
        suspended=suspended,
        raw_response_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        data_class="order_book",
        eligible=eligible,
        eligibility_reason=reason,
    )


def executable_buy_price_for_team(
    market: VenueMarket,
    book: MarketBookQuote,
    team_abbr: str,
) -> dict[str, Any]:
    """Estimate executable buy cost for a team moneyline outcome.

    Long side: use best ask on the shared instrument book.
    Short side: use ``1 - best_bid`` when book exists (documented complement),
    else the side quote from market metadata. Display prices alone are never
    treated as verified fills.
    """
    team = normalize_team_abbr(team_abbr)
    outcome = next((o for o in market.outcomes if o.team_abbr == team), None)
    if outcome is None:
        return {
            "team": team,
            "executable_buy": None,
            "method": None,
            "eligible": False,
            "reason": "team_not_on_market",
        }
    if outcome.is_long:
        px = book.best_ask
        method = "book_best_ask_long"
    else:
        if book.best_bid is not None:
            px = 1.0 - float(book.best_bid)
            method = "one_minus_book_best_bid_short"
        else:
            px = outcome.side_quote
            method = "side_quote_short_fallback"
    eligible = px is not None and book.eligible and outcome.tradable
    reason = None if eligible else (book.eligibility_reason or "missing_executable_price")
    fee = select_fee_schedule(as_of_utc=book.receive_time_utc)
    fee_amt = fee.taker_fee(px, 1.0) if px is not None else None
    return {
        "team": team,
        "outcome_id": outcome.outcome_id,
        "is_long": outcome.is_long,
        "executable_buy": px,
        "display_price": outcome.display_price,
        "method": method,
        "eligible": bool(eligible),
        "reason": reason,
        "fee_version": fee.fee_version,
        "est_taker_fee_per_contract": fee_amt,
        "total_acquisition_cost_per_contract": (
            None if px is None or fee_amt is None else float(px) + float(fee_amt)
        ),
    }


class PolymarketUSAdapter:
    venue_id = VENUE_ID

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        opener=None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        min_interval_seconds: float | None = None,
    ):
        self.base_url = (base_url or config.POLYMARKET_US_API_BASE_URL).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self._opener = opener
        self.max_retries = int(
            config.POLYMARKET_HTTP_MAX_RETRIES if max_retries is None else max_retries
        )
        self.retry_backoff_seconds = float(
            config.POLYMARKET_HTTP_RETRY_BACKOFF_SECONDS
            if retry_backoff_seconds is None
            else retry_backoff_seconds
        )
        self.min_interval_seconds = float(
            config.POLYMARKET_HTTP_MIN_INTERVAL_SECONDS
            if min_interval_seconds is None
            else min_interval_seconds
        )
        self._last_request_monotonic: float | None = None

    def capabilities(self) -> VenueCapabilities:
        return VenueCapabilities(
            venue_id=VENUE_ID,
            public_market_data=True,
            order_book=True,
            historical_display_prices=True,
            fee_schedule=True,
            order_placement=False,
        )

    def fee_schedule_as_of(self, as_of_utc: str | None = None) -> FeeSchedule:
        return select_fee_schedule(as_of_utc=as_of_utc)

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.monotonic()
        if self._last_request_monotonic is not None:
            elapsed = now - self._last_request_monotonic
            wait = self.min_interval_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_monotonic = time.monotonic()

    def _http_get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                if self._opener is not None:
                    with self._opener(req, timeout=self.timeout_seconds) as resp:
                        return json.loads(resp.read().decode("utf-8"))
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_exc = RuntimeError(f"Polymarket US HTTP {exc.code} for {url}: {body[:300]}")
                # Retry rate limits and transient gateway errors only.
                if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                    raise last_exc from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_exc = RuntimeError(f"Polymarket US request failed for {url}: {exc}")
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        assert last_exc is not None
        raise last_exc

    def list_league_events(
        self,
        *,
        league: str | None = None,
        limit: int = 200,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict[str, Any]]:
        slug = league or config.POLYMARKET_US_LEAGUE_SLUG
        payload = self._http_get_json(
            f"/v2/leagues/{slug}/events",
            {
                "limit": int(limit),
                "active": str(bool(active)).lower(),
                "closed": str(bool(closed)).lower(),
            },
        )
        events = payload.get("events") if isinstance(payload, dict) else None
        return list(events or [])

    def list_mlb_moneyline_markets(self, *, limit: int = 200) -> list[VenueMarket]:
        markets: list[VenueMarket] = []
        for event in self.list_league_events(limit=limit):
            for market in event.get("markets") or []:
                type_v2 = market.get("sportsMarketTypeV2")
                market_type = str(market.get("marketType") or "").lower()
                if type_v2 != MONEYLINE_TYPE_V2 and market_type != "moneyline":
                    continue
                sports_type = str(market.get("sportsMarketType") or "")
                if "first_five" in sports_type:
                    continue
                markets.append(parse_moneyline_market(event, market))
        return markets

    def fetch_market_book(
        self,
        market_slug: str,
        *,
        market_id: str | None = None,
        fee_coefficient: float | None = None,
        tick_size: float | None = None,
        min_trade_qty: float | None = None,
        as_of_utc: str | None = None,
    ) -> MarketBookQuote:
        request_time = _utc_now_iso()
        payload = self._http_get_json(f"/v1/markets/{urllib.parse.quote(market_slug)}/book")
        receive_time = _utc_now_iso()
        fee = self.fee_schedule_as_of(as_of_utc or receive_time)
        return parse_book_response(
            payload,
            market_id=market_id or market_slug,
            market_slug=market_slug,
            request_time_utc=request_time,
            receive_time_utc=receive_time,
            fee=fee,
            fee_coefficient=fee_coefficient,
            tick_size=tick_size,
            min_trade_qty=min_trade_qty,
        )

    def fetch_market_settlement(self, market_slug: str) -> dict[str, Any]:
        """Official settlement price when available (docs: GET .../settlement).

        Returns ``{"slug", "settlement", "request_time_utc", "receive_time_utc"}``.
        404 / unsettled markets raise RuntimeError — callers must keep positions open.
        """
        request_time = _utc_now_iso()
        payload = self._http_get_json(
            f"/v1/markets/{urllib.parse.quote(market_slug)}/settlement"
        )
        receive_time = _utc_now_iso()
        if not isinstance(payload, dict) or "settlement" not in payload:
            raise RuntimeError(f"Unexpected settlement payload for {market_slug}: {payload!r}")
        return {
            "slug": payload.get("slug") or market_slug,
            "settlement": float(payload["settlement"]),
            "request_time_utc": request_time,
            "receive_time_utc": receive_time,
            "source": "polymarket_us_settlement_api",
        }
