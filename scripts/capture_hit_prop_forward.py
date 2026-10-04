"""Capture independent research forecasts and delayed books; no outcomes/orders.

Writes only new immutable files beneath --output-dir. Input CSV/model bytes are
archived before inference. Historical-date/backdating options are not supported.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mlb_metrics import config, hit_prop_forward as forward
from mlb_metrics.venues.polymarket_us import PolymarketUSAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=config.HIT_PROP_FORWARD_DIR)
    parser.add_argument('--contracts', default='data/polymarket/research/hit_props/contracts.csv')
    parser.add_argument('--wave', default='docs/data/wave.csv')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if config.GAME_PREDICTION_MODE != 'shadow' or config.BETTING_MODE != 'disabled':
        raise ValueError('Research modes must remain disabled/shadow')
    if args.audit_only:
        report = forward.audit_store(args.output_dir)
    else:
        adapter = PolymarketUSAdapter(timeout_seconds=config.HIT_PROP_FORWARD_REQUEST_TIMEOUT, max_retries=0)
        try:
            contracts = Path(args.contracts).read_bytes()
            wave = Path(args.wave).read_bytes()
            model = Path(config.HIT_PROP_FORWARD_MODEL_PATH).read_bytes()
        except OSError as exc:
            from uuid import uuid4
            report = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                      'outcomes_opened': False, 'real_money_ready': False}
            run = Path(args.output_dir) / 'runs' / (forward.utc_now().strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8])
            forward.publish_once(run / 'report.json', forward.encode(report))
        else:
            report = forward.capture(root=args.output_dir, contracts_bytes=contracts,
                                     wave_bytes=wave, model_bytes=model, adapter=adapter)
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(report['status'] in {'failed', 'integrity_failure'})


if __name__ == '__main__':
    raise SystemExit(main())
