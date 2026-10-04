"""Receipt timing must apply to prior dates, corrections and doubleheaders."""
import pandas as pd
import pytest
from mlb_metrics.hitter_training_data import audit_history_availability, slice_history_before_game

CUTOFF = '2026-06-20T12:00:00Z'


def frame(completed='2026-06-19T23:00:00Z', observed='2026-06-20T01:00:00Z'):
    return pd.DataFrame({'game_pk': [1], 'game_date': ['2026-06-19'],
                         'game_completed_at': [completed], 'source_observed_at_utc': [observed]})


@pytest.mark.parametrize('column', ['game_completed_at', 'source_observed_at_utc'])
@pytest.mark.parametrize('value', [None, '', 'bad', '2026-06-19T23:00:00', CUTOFF, '2026-06-21T00:00:00Z'])
def test_strict_rejects_unknown_naive_equal_or_future(column, value):
    rows = frame()
    rows[column] = value
    assert slice_history_before_game(rows, '2026-06-20', prediction_timestamp_utc=CUTOFF,
                                     require_observed_history=True).empty


@pytest.mark.parametrize('column', ['game_completed_at', 'source_observed_at_utc'])
def test_missing_column_fails_closed(column):
    assert not audit_history_availability(frame().drop(columns=column), CUTOFF).timestamps_eligible.any()


def test_valid_offsets_and_doubleheader():
    rows = frame('2026-06-20T03:00:00-07:00', '2026-06-20T04:00:00-07:00')
    rows['game_date'] = '2026-06-20'
    assert len(slice_history_before_game(rows, '2026-06-20', target_game_pk=2,
               prediction_timestamp_utc=CUTOFF, require_observed_history=True)) == 1
    assert slice_history_before_game(rows, '2026-06-20', target_game_pk=1,
               prediction_timestamp_utc=CUTOFF, require_observed_history=True).empty
    assert slice_history_before_game(rows, '2026-06-20', require_observed_history=True).empty


@pytest.mark.parametrize('strict', [True, False])
def test_target_excluded_even_if_rescheduled_prior_date(strict):
    assert slice_history_before_game(frame(), '2026-06-20', target_game_pk=1,
               prediction_timestamp_utc=CUTOFF, require_observed_history=strict).empty


@pytest.mark.parametrize('cutoff', [None, 'bad', '2026-06-20'])
def test_explicit_audit_cutoff_required(cutoff):
    with pytest.raises((ValueError, TypeError)):
        audit_history_availability(frame(), cutoff)


def test_legacy_reconstruction_not_receipt_evidence():
    rows = frame().drop(columns=['source_observed_at_utc'])
    assert len(slice_history_before_game(rows, '2026-06-20')) == 1
    assert slice_history_before_game(rows, '2026-06-20', require_observed_history=True).empty


def test_reconstructed_features_never_claim_certification():
    from mlb_metrics.hitter_training_data import _attach_pregame_features
    candidates = pd.DataFrame({'key_mlbam': [42], 'game_pk': [7]})
    wave = pd.DataFrame({'key_mlbam': [42], 'Last_Game_Date': ['2026-06-19']})
    result = _attach_pregame_features(candidates, pd.DataFrame(), wave, '2026-06-20', CUTOFF)
    assert result.feature_availability_status.tolist() == ['reconstructed_unverified']
    assert result.prediction_time_availability_certified.tolist() == [False]


@pytest.mark.parametrize('existing', ['output', 'coverage'])
def test_strict_cli_rejects_legacy_output_before_loading(tmp_path, monkeypatch, existing):
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/build_hitter_opportunity_log.py'
    spec = importlib.util.spec_from_file_location('timing_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output, coverage = tmp_path / 'rows.csv', tmp_path / 'coverage.csv'
    selected = output if existing == 'output' else coverage
    selected.write_text('preserve me')
    monkeypatch.setattr(sys, 'argv', ['builder', '--require-observed-history', '--output', str(output), '--coverage-output', str(coverage)])
    monkeypatch.setattr(module.hitter_training_data, 'assemble_hitter_opportunity_dataset', lambda *a, **k: pytest.fail('must reject before loading'))
    with pytest.raises(ValueError, match='new output and coverage'):
        module.main()
    assert selected.read_text() == 'preserve me'
