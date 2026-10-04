"""Append-only, independent forward research forecasts. No orders or scoring.

Pin exact input bytes before inference. Keep first forecasts, failed captures and
missing fills. Historical model training is uncertified; this stream never
increments formal eligible dates or reads registered evaluation outcomes.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import joblib
import pandas as pd

from mlb_metrics import config, schedule, hit_prop_development as development
from mlb_metrics import hit_prop_market_benchmark as benchmark
from mlb_metrics.venues.polymarket_us import normalize_team_abbr, rules_hash


def utc_now():
    return pd.Timestamp.now(tz='UTC')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()


def publish_once(path, raw):
    """Atomically expose fsynced bytes without replacing an existing record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic create-if-absent, even with concurrent writers.
    finally:
        os.unlink(temporary)


def blob(root, raw):
    digest = sha(raw)
    path = Path(root) / 'blobs' / digest
    try:
        publish_once(path, raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError('Content-addressed snapshot mismatch')
    return digest


def identity(contract):
    key = benchmark.identity(contract)
    if key is None or not isinstance(contract.get('market_id'), str) or not contract['market_id']:
        raise ValueError('invalid_contract_identity')
    return list(key) + [contract['market_id']]


def clock_date(now):
    return now.tz_convert(schedule.LOCAL_TIMEZONE).date().isoformat()


def candidate_reason(contract, wave, now):
    try:
        identity(contract)
    except ValueError as exc:
        return str(exc)
    start = benchmark.timestamp(contract.get('scheduled_start_utc'))
    if contract.get('requested_local_date') != clock_date(now):
        return 'not_current_local_date'
    if start is None or now >= start - pd.Timedelta(minutes=config.HIT_PROP_FORWARD_ENTRY_MINUTES):
        return 'inside_entry_cutoff_or_invalid_start'
    if (contract.get('venue_id') != 'polymarket_us'
        or str(contract.get('quarantined')).lower() != 'false'
        or contract.get('game_mapping_status') != 'mapped'
        or contract.get('player_mapping_status') != 'mapped'):
        return 'unverified_identity'
    if (contract.get('game_type_source') != 'mlb_statsapi_game_pk'
        or contract.get('game_type') not in {'R', 'F', 'D', 'L', 'W'}):
        return 'unverified_cohort'
    if (contract.get('stat') != 'hits' or benchmark.number(contract.get('threshold')) != 1
        or str(contract.get('requires_starting_lineup')).lower() != 'true'
        or str(contract.get('requires_plate_appearance')).lower() != 'true'
        or contract.get('settlement_on_non_participation') != 'last_fair_market_price'):
        return 'unsupported_contract'
    text = contract.get('rules_text')
    if not isinstance(text, str) or not text or rules_hash(text) != contract.get('rules_hash'):
        return 'rules_hash_mismatch'
    if benchmark.number(contract.get('fee_coefficient')) != config.HIT_PROP_BENCHMARK_TAKER_THETA:
        return 'unverified_fee_schedule'
    if now < benchmark.timestamp(config.HIT_PROP_BENCHMARK_FEE_EFFECTIVE_UTC):
        return 'unverified_fee_date'
    player = wave[pd.to_numeric(wave.key_mlbam, errors='coerce') == float(contract['key_mlbam'])]
    if len(player) != 1:
        return 'missing_or_duplicate_player_features'
    team = normalize_team_abbr(player.iloc[0].get('team'))
    teams = [normalize_team_abbr(contract.get(k)) for k in ('home_team', 'away_team')]
    if not team or not all(teams) or teams[0] == teams[1] or team not in teams:
        return 'feature_team_mismatch'
    last = pd.to_datetime(player.iloc[0].get('Last_Game_Date'), errors='coerce', utc=True)
    date = pd.to_datetime(contract.get('requested_local_date'), errors='coerce', utc=True)
    if pd.isna(last) or pd.isna(date) or last.normalize() >= date.normalize():
        return 'missing_or_same_day_history_cutoff'
    return None


def feature_row(contract, wave):
    row = wave[pd.to_numeric(wave.key_mlbam, errors='coerce') == float(contract['key_mlbam'])].iloc[0]
    result = {key: row.get(key) for key in config.HIT_PROP_DEV_FEATURES}
    result['Days_Rest'] = (pd.Timestamp(contract['requested_local_date']) - pd.Timestamp(row.Last_Game_Date)).days
    result['is_home'] = int(normalize_team_abbr(row.team) == normalize_team_abbr(contract['home_team']))
    return development._features(pd.DataFrame([result]))


def quote_reason(book, contract, now, *, earliest=None):
    request = benchmark.timestamp(book.get('request_time_utc'))
    received = benchmark.timestamp(book.get('receive_time_utc'))
    start = benchmark.timestamp(contract.get('scheduled_start_utc'))
    if request is None or received is None or start is None or not request <= received <= now < start:
        return 'invalid_or_poststart_quote_time'
    if earliest is not None and (request < earliest or (received-earliest).total_seconds() > config.HIT_PROP_BENCHMARK_QUOTE_AGE):
        return 'delayed_quote_outside_registered_window'
    if (now-received).total_seconds() > config.HIT_PROP_BENCHMARK_QUOTE_AGE:
        return 'stale_quote'
    if any(str(book.get(key)) != str(contract.get(key)) for key in ('venue_id', 'market_id', 'market_slug')):
        return 'quote_identity_mismatch'
    if book.get('eligible') is not True or book.get('suspended') is not False:
        return 'ineligible_book'
    bid, ask, size = map(benchmark.number, (book.get('best_bid'), book.get('best_ask'), book.get('best_ask_size')))
    if bid is None or ask is None or size is None or not 0 < bid < ask < 1 or size < config.HIT_PROP_BENCHMARK_QUANTITY:
        return 'invalid_crossed_or_insufficient_book'
    if benchmark.number(book.get('fee_coefficient')) != config.HIT_PROP_BENCHMARK_TAKER_THETA:
        return 'quote_fee_mismatch'
    minimum = book.get('min_trade_qty')
    if minimum is not None and (benchmark.number(minimum) is None or float(minimum) > config.HIT_PROP_BENCHMARK_QUANTITY):
        return 'unsupported_minimum_quantity'
    return None


def capture(*, root, contracts_bytes, wave_bytes, model_bytes, adapter, clock=utc_now, sleep=time.sleep):
    """One bounded cycle; callers cannot assign a retrospective CLI timestamp."""
    root = Path(root)
    run_id = clock().strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
    run = root / 'runs' / run_id
    run.mkdir(parents=True, exist_ok=False)
    report = {'run_id': run_id, 'version': config.HIT_PROP_FORWARD_VERSION,
              'started_at_utc': clock().isoformat(), 'status': 'started',
              'forecasts_saved': 0, 'executable_quotes_matched': 0,
              'exclusions': [], 'research_only': True, 'real_money_ready': False,
              'formal_evaluation_eligible_dates': 0, 'outcomes_opened': False}
    publish_once(run / 'started.json', encode(report))
    try:
        if sha(model_bytes) != config.HIT_PROP_FORWARD_MODEL_SHA256:
            raise ValueError('Pinned model hash mismatch; no automatic model replacement')
        registration = {'version': config.HIT_PROP_FORWARD_VERSION,
                        'model_sha256': config.HIT_PROP_FORWARD_MODEL_SHA256,
                        'features': list(config.HIT_PROP_DEV_FEATURES),
                        'entry_minutes': config.HIT_PROP_FORWARD_ENTRY_MINUTES,
                        'delay_seconds': config.HIT_PROP_BENCHMARK_DELAY_SECONDS,
                        'quote_age_seconds': config.HIT_PROP_BENCHMARK_QUOTE_AGE,
                        'quantity': config.HIT_PROP_BENCHMARK_QUANTITY,
                        'fee_theta': config.HIT_PROP_BENCHMARK_TAKER_THETA,
                        'fee_effective_utc': config.HIT_PROP_BENCHMARK_FEE_EFFECTIVE_UTC,
                        'max_contracts_per_cycle': config.HIT_PROP_FORWARD_MAX_CONTRACTS,
                        'selection': 'first eligible forecast; slug order; all omissions recorded',
                        'formal_holdout_assignment': 'unassigned_research_only',
                        'settlement_scoring': 'blocked_until_registered_evaluation_gates_pass'}
        registered = encode(registration)
        try:
            publish_once(root / 'registration.json', registered)
        except FileExistsError:
            if (root / 'registration.json').read_bytes() != registered:
                raise ValueError('Research registration changed; use a reviewed version and separate store')
        inputs = {name: blob(root, raw) for name, raw in
                  [('contracts', contracts_bytes), ('wave', wave_bytes), ('model', model_bytes)]}
        observed = clock().isoformat()
        model = joblib.load(io.BytesIO(model_bytes))
        if model.get('model_version') != config.HIT_PROP_DEV_MODEL_VERSION or tuple(model.get('features', ())) != tuple(config.HIT_PROP_DEV_FEATURES):
            raise ValueError('Artifact version/features mismatch')
        if model.get('production_serving_allowed') is not False or model.get('research_only') is not True:
            raise ValueError('Expected explicitly research-only artifact')
        wave = pd.read_csv(io.BytesIO(wave_bytes))
        contracts = pd.read_csv(io.BytesIO(contracts_bytes), dtype={'market_id': str})
        input_count = len(contracts)
        if 'requested_local_date' in contracts:
            contracts = contracts[contracts.requested_local_date == clock_date(clock())]
        report['historical_or_other_date_rows_not_replayed'] = input_count - len(contracts)
        if not contracts.empty and 'capture_id' in contracts:
            contracts = contracts[contracts.capture_id == contracts.capture_id.max()]
        report['loaded_capture_rows'] = len(contracts)
        trained = pd.to_datetime(model.get('training_end_date'), utc=True, errors='coerce')
        if pd.isna(trained) or trained.normalize() >= clock().normalize():
            raise ValueError('Invalid or non-past training end date')
        # This universe snapshot is immutable even if no rows become forecasts.
        manifest = {'version': config.HIT_PROP_FORWARD_VERSION, 'input_hashes': inputs,
                    'inputs_observed_at_utc': observed, 'model_version': model['model_version'],
                    'features': list(config.HIT_PROP_DEV_FEATURES),
                    'selection': 'first pregame forecast per exact contract; sorted slug; bounded request budget; no outcome-based selection',
                    'quantity': config.HIT_PROP_BENCHMARK_QUANTITY,
                    'delay_seconds': config.HIT_PROP_BENCHMARK_DELAY_SECONDS,
                    'max_quote_age_seconds': config.HIT_PROP_BENCHMARK_QUOTE_AGE,
                    'training_end_date': model['training_end_date'],
                    'historical_training_availability_certified': False,
                    'formal_holdout_assignment': 'unassigned_research_only',
                    'outcome_scoring_allowed': False}
        manifest['registration_sha256'] = blob(root, registered)
        manifest_hash = blob(root, encode(manifest))
        publish_once(run / 'manifest.json', encode(manifest))
        pending = []
        used = 0
        # Preserve every captured universe row, reject ambiguous duplicates.
        counts = contracts.market_slug.value_counts().to_dict() if 'market_slug' in contracts else {}
        for contract in contracts.sort_values('market_slug').to_dict('records'):
            reason = 'ambiguous_contract_rows' if counts.get(contract['market_slug'], 0) != 1 else candidate_reason(contract, wave, clock())
            key = None
            if not reason:
                key = sha(encode([config.HIT_PROP_FORWARD_VERSION, identity(contract)]))
                if (root / 'forecasts' / f'{key}.json').exists():
                    reason = 'first_forecast_already_saved'
                elif used >= config.HIT_PROP_FORWARD_MAX_CONTRACTS:
                    reason = 'request_budget_omission'
            if reason:
                report['exclusions'].append({'market_slug': contract.get('market_slug'), 'reason': reason})
                continue
            used += 1
            try:
                book = adapter.fetch_market_book(contract['market_slug'], market_id=contract['market_id'], fee_coefficient=float(contract['fee_coefficient'])).to_dict()
                book_hash = blob(root, encode(book))
                reason = quote_reason(book, contract, clock())
                if reason:
                    raise ValueError(reason)
                features = feature_row(contract, wave)
                features_raw = features.to_csv(index=False).encode()
                feature_hash = blob(root, features_raw)
                # Inference consumes the archived representation, not mutable inputs.
                probs = development.predict_states(model['model'], pd.read_csv(io.BytesIO(features_raw)))
                development.payout_bounds(probs)  # Reject invalid output distributions.
                now = clock()
                if now >= benchmark.timestamp(contract['scheduled_start_utc']) - pd.Timedelta(minutes=config.HIT_PROP_FORWARD_ENTRY_MINUTES):
                    raise ValueError('prediction_finished_inside_entry_cutoff')
                if quote_reason(book, contract, now):
                    raise ValueError('prediction_time_quote_expired')
                record = {'forecast_id': key, 'run_id': run_id, 'manifest_sha256': manifest_hash,
                          'contract': {k: contract[k] for k in ('venue_id', 'market_id', 'market_slug', 'game_pk', 'key_mlbam', 'rules_hash', 'rules_text', 'scheduled_start_utc', 'game_type', 'requested_local_date')},
                          'forecast_time_utc': now.isoformat(), 'artifact_sha256': inputs['model'],
                          'input_hashes': inputs, 'feature_matrix_sha256': feature_hash,
                          'prediction_book_sha256': book_hash, 'inputs_observed_at_utc': observed,
                          'probabilities': dict(zip(development.CLASS_NAMES, probs[0].tolist())),
                          'probability_source': 'independent_three_state_model',
                          'inference_inputs_observed_before_prediction': True,
                          'prediction_time_availability_certified': False,
                          'historical_training_availability_certified': False,
                          'formal_holdout_assignment': 'unassigned_research_only',
                          'cohort': 'regular_season' if contract['game_type'] == 'R' else 'postseason',
                          'research_only': True, 'actionable': False}
                publish_once(root / 'forecasts' / f'{key}.json', encode(record))
                receipt = {'forecast_id': key, 'forecast_sha256': sha(encode(record)), 'saved_at_utc': clock().isoformat()}
                publish_once(root / 'receipts' / f'{key}.json', encode(receipt))
                report['forecasts_saved'] += 1
                pending.append((record, contract))
            except Exception as exc:
                report['exclusions'].append({'market_slug': contract['market_slug'], 'reason': f'{type(exc).__name__}: {exc}'})
        for record, contract in pending:
            result = {'forecast_id': record['forecast_id'], 'status': 'missing_execution_quote', 'hypothetical_fill_only': True}
            try:
                due = benchmark.timestamp(record['forecast_time_utc']) + pd.Timedelta(seconds=config.HIT_PROP_BENCHMARK_DELAY_SECONDS)
                remaining = (due-clock()).total_seconds()
                if remaining > 0:
                    sleep(min(remaining, config.HIT_PROP_BENCHMARK_DELAY_SECONDS))
                if clock() > due + pd.Timedelta(seconds=config.HIT_PROP_BENCHMARK_QUOTE_AGE):
                    raise ValueError('delayed_window_missed')
                book = adapter.fetch_market_book(contract['market_slug'], market_id=contract['market_id'], fee_coefficient=float(contract['fee_coefficient'])).to_dict()
                result['book_sha256'] = blob(root, encode(book))
                reason = quote_reason(book, contract, clock(), earliest=due)
                if reason:
                    raise ValueError(reason)
                qty = config.HIT_PROP_BENCHMARK_QUANTITY
                fee = benchmark.standard_fee_cap(book['best_ask'], qty)
                result.update(status='executable_quote_matched', quantity=qty, ask=book['best_ask'],
                              fee_cap=fee, acquisition_cost=qty*book['best_ask']+fee,
                              quote_received_at_utc=book['receive_time_utc'])
                report['executable_quotes_matched'] += 1
            except Exception as exc:
                result['reason'] = f'{type(exc).__name__}: {exc}'
            publish_once(root / 'execution' / f"{record['forecast_id']}.json", encode(result))
        report['status'] = 'captured' if report['forecasts_saved'] else 'no_eligible_forecasts'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    report['finished_at_utc'] = clock().isoformat()
    publish_once(run / 'report.json', encode(report))
    return report


def audit_store(root):
    """Reopen saved bytes independently; coverage only, never outcome scoring."""
    root = Path(root)
    report = {'status': 'insufficient_evidence', 'forecasts': 0,
              'intact_pregame_forecasts': 0, 'execution_quotes_matched': 0,
              'integrity_errors': [], 'missing_execution': 0, 'cohort_counts': {},
              'formal_eligible_dates': 0, 'outcomes_opened': False,
              'realized_returns': None, 'real_money_ready': False,
              'remaining_gates': ['historical_training_provenance_uncertified',
                                  'immutable_formal_holdout_assignment_missing',
                                  'registered_sample_and_precision_requirements',
                                  'verified_actual_contract_settlements_required']}
    report.update(capture_runs=0, failed_runs=0, incomplete_runs=0, exclusion_reason_counts={})
    for run in sorted((root / 'runs').glob('*')):
        if not run.is_dir():
            continue
        report['capture_runs'] += 1
        try:
            cycle = json.loads((run / 'report.json').read_bytes())
            report['failed_runs'] += int(cycle.get('status') == 'failed')
            for exclusion in cycle.get('exclusions', []):
                reason = exclusion['reason']
                report['exclusion_reason_counts'][reason] = report['exclusion_reason_counts'].get(reason, 0) + 1
        except (OSError, ValueError, KeyError):
            report['incomplete_runs'] += 1
    for path in sorted((root / 'forecasts').glob('*.json')):
        report['forecasts'] += 1
        try:
            raw = path.read_bytes()
            f = json.loads(raw)
            key = f['forecast_id']
            if key != path.stem or key != sha(encode([config.HIT_PROP_FORWARD_VERSION, identity(f['contract'])])):
                raise ValueError('Forecast identity hash mismatch')
            receipt = json.loads((root / 'receipts' / f'{key}.json').read_bytes())
            if receipt['forecast_sha256'] != sha(raw) or receipt['forecast_id'] != key:
                raise ValueError('Receipt hash mismatch')
            def read_blob(digest):
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('Invalid snapshot digest')
                data = (root / 'blobs' / digest).read_bytes()
                if sha(data) != digest:
                    raise ValueError('Snapshot hash mismatch')
                return data
            for digest in [f['manifest_sha256'], f['feature_matrix_sha256'], f['prediction_book_sha256'], *f['input_hashes'].values()]:
                read_blob(digest)
            manifest = json.loads(read_blob(f['manifest_sha256']))
            registration_bytes = read_blob(manifest['registration_sha256'])
            if registration_bytes != (root / 'registration.json').read_bytes():
                raise ValueError('Research registration mismatch')
            if manifest['input_hashes'] != f['input_hashes'] or f['artifact_sha256'] != config.HIT_PROP_FORWARD_MODEL_SHA256:
                raise ValueError('Manifest/model mismatch')
            observed, predicted, saved, start = map(benchmark.timestamp, (f['inputs_observed_at_utc'], f['forecast_time_utc'], receipt['saved_at_utc'], f['contract']['scheduled_start_utc']))
            if any(x is None for x in (observed, predicted, saved, start)) or not observed <= predicted <= saved < start:
                raise ValueError('Invalid pregame persistence timing')
            first_book = json.loads(read_blob(f['prediction_book_sha256']))
            # The compact forecast contract omits pricing metadata; restore recorded contract inputs.
            contracts = pd.read_csv(io.BytesIO(read_blob(f['input_hashes']['contracts'])), dtype={'market_id': str})
            candidates = contracts[(contracts.market_slug == f['contract']['market_slug']) & (contracts.rules_hash == f['contract']['rules_hash'])]
            if candidates.empty:
                raise ValueError('Contract missing from captured universe')
            contract = candidates.iloc[-1].to_dict()
            if identity(contract) != identity(f['contract']):
                raise ValueError('Captured contract mismatch')
            reason = quote_reason(first_book, contract, predicted)
            if reason:
                raise ValueError(reason)
            development.payout_bounds([[f['probabilities'][k] for k in development.CLASS_NAMES]])
            report['intact_pregame_forecasts'] += 1
            cohort = f['cohort']
            report['cohort_counts'][cohort] = report['cohort_counts'].get(cohort, 0) + 1
            execution_path = root / 'execution' / f'{key}.json'
            if not execution_path.exists():
                report['missing_execution'] += 1
                continue
            execution = json.loads(execution_path.read_bytes())
            if execution['forecast_id'] != key:
                raise ValueError('Execution forecast mismatch')
            if execution['status'] == 'executable_quote_matched':
                book = json.loads(read_blob(execution['book_sha256']))
                due = predicted + pd.Timedelta(seconds=config.HIT_PROP_BENCHMARK_DELAY_SECONDS)
                received = benchmark.timestamp(book.get('receive_time_utc'))
                if received is None:
                    raise ValueError('Missing execution time')
                reason = quote_reason(book, contract, received, earliest=due)
                if reason:
                    raise ValueError(reason)
                qty = config.HIT_PROP_BENCHMARK_QUANTITY
                fee = benchmark.standard_fee_cap(book['best_ask'], qty)
                if (execution['quantity'] != qty or execution['ask'] != book['best_ask']
                    or execution['fee_cap'] != fee or execution['acquisition_cost'] != qty*book['best_ask']+fee):
                    raise ValueError('Execution accounting mismatch')
                report['execution_quotes_matched'] += 1
            else:
                report['missing_execution'] += 1
        except Exception as exc:
            report['integrity_errors'].append({'path': path.name, 'error': f'{type(exc).__name__}: {exc}'})
    if report['integrity_errors']:
        report['status'] = 'integrity_failure'
    return report
