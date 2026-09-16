"""Local dashboard host with optional quote recheck (no order placement).

Serves docs/ and exposes:
  GET /api/decision-meta
  GET /api/recheck-quote?market_slug=...&market_id=...

Usage:
    PYTHONPATH=src python scripts/serve_decision_dashboard.py
    # open http://127.0.0.1:8765/polymarket.html
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config
from mlb_metrics.venues import get_venue_adapter
from mlb_metrics.venues.polymarket_us import select_fee_schedule


ROOT = os.path.join(os.path.dirname(__file__), "..", "docs")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/decision-meta":
            path = os.path.join(
                os.path.dirname(__file__),
                "..",
                config.POLYMARKET_DECISION_BOARD_META_PATH,
            )
            if not os.path.exists(path):
                return self._json(404, {"error": "meta_missing", "path": path})
            with open(path, encoding="utf-8") as f:
                return self._json(200, json.load(f))
        if parsed.path == "/api/recheck-quote":
            assert config.BETTING_MODE == "disabled"
            qs = parse_qs(parsed.query)
            slug = (qs.get("market_slug") or [None])[0]
            market_id = (qs.get("market_id") or [None])[0]
            if not slug:
                return self._json(400, {"error": "market_slug_required"})
            try:
                adapter = get_venue_adapter(config.POLYMARKET_VENUE_SELECTED)
                if adapter.capabilities().order_placement:
                    return self._json(500, {"error": "refusing_order_capable_venue"})
                book = adapter.fetch_market_book(slug, market_id=market_id or slug)
                fee = select_fee_schedule(as_of_utc=book.receive_time_utc)
                return self._json(
                    200,
                    {
                        "market_slug": slug,
                        "market_id": book.market_id,
                        "best_bid": book.best_bid,
                        "best_ask": book.best_ask,
                        "best_ask_size": book.best_ask_size,
                        "best_bid_size": book.best_bid_size,
                        "eligible": book.eligible,
                        "eligibility_reason": book.eligibility_reason,
                        "receive_time_utc": book.receive_time_utc,
                        "request_time_utc": book.request_time_utc,
                        "fee_version": fee.fee_version,
                        "orders_placed": False,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                return self._json(
                    502,
                    {"error": "recheck_failed", "error_type": type(exc).__name__, "detail": str(exc)[:300]},
                )
        return super().do_GET()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> int:
    host = os.environ.get("DECISION_DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("DECISION_DASH_PORT", "8765"))
    print(f"Serving {ROOT} at http://{host}:{port}/polymarket.html")
    print("Read-only recheck API at /api/recheck-quote — no orders.")
    print(f"BETTING_MODE={config.BETTING_MODE!r} GAME_PREDICTION_MODE={config.GAME_PREDICTION_MODE!r}")
    httpd = ThreadingHTTPServer((host, port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
