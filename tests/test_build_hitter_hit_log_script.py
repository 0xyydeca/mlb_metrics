"""End-to-end smoke test for scripts/build_hitter_hit_log.py: a synthetic
persisted Statcast history is used to write/append hitter_hit_log.csv,
checking schema, --days truncation, dedupe-on-rerun (same (date,
game_pk, key_mlbam) rows get replaced by the fresher recompute, not
duplicated),
and no-data resilience - mirroring test_build_dfs_rankings_script.py's
exact pattern."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from mlb_metrics import data, dfs_ml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_hitter_hit_log.py"


def _load_build_hitter_hit_log_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_hitter_hit_log", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _game_rows(game_pk, date, events, pitcher=99, batter=1, home_team="NYY", away_team="BOS"):
    rows = []
    away_runs = 0
    for i, e in enumerate(events):
        pre = away_runs
        if e in ("home_run", "single"):
            away_runs += 1
        rows.append({
            "game_pk": game_pk, "game_date": date, "pitcher": pitcher, "batter": batter,
            "events": e, "p_throws": "R", "inning_topbot": "Top",
            "home_team": home_team, "away_team": away_team,
            "at_bat_number": i + 1, "pitch_number": 1,
            "home_score": 0, "away_score": pre,
            "post_home_score": 0, "post_away_score": away_runs,
            "bat_score": pre, "post_bat_score": away_runs,
        })
    return rows


def _multi_game_statcast(n_games=6, gap_days=5):
    events = ["strikeout"] * 5 + ["field_out"] * 6 + ["walk"] * 3 + ["single"] * 4 + ["double"] * 1 + ["home_run"] * 1
    rows = []
    for i in range(n_games):
        date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=i * gap_days)
        rows.extend(_game_rows(i + 1, date, events))
    return pd.DataFrame(rows)


def _no_network_name_register(monkeypatch):
    monkeypatch.setattr(
        data, "get_name_register",
        lambda: pd.DataFrame(columns=["key_mlbam", "key_bbref", "name_first", "name_last"]),
    )


def test_build_hitter_hit_log_full_backfill_writes_expected_schema(tmp_path, monkeypatch):
    _no_network_name_register(monkeypatch)
    module = _load_build_hitter_hit_log_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)
    output = tmp_path / "hitter_hit_log.csv"

    sys.argv = [
        "build_hitter_hit_log.py", "--raw-dir", str(raw_dir), "--season", "2026",
        "--output", str(output),
    ]
    module.main()

    result = pd.read_csv(output)
    assert not result.empty
    expected_cols = {
        "date", "game_pk", "key_mlbam", "name_first", "name_last", "team",
        *dfs_ml.HITTER_FEATURE_COLUMNS, "Total_PA", "Days_Rest", "Umpire_Factor", "Got_Hit",
    }
    assert set(result.columns) == expected_cols


def test_build_hitter_hit_log_days_truncates_recomputed_rows(tmp_path, monkeypatch):
    _no_network_name_register(monkeypatch)
    module = _load_build_hitter_hit_log_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)
    output = tmp_path / "hitter_hit_log.csv"

    sys.argv = [
        "build_hitter_hit_log.py", "--raw-dir", str(raw_dir), "--season", "2026",
        "--output", str(output), "--days", "1",
    ]
    module.main()

    result = pd.read_csv(output)
    assert result["date"].nunique() <= 1


def test_build_hitter_hit_log_rerun_dedupes_and_preserves_older_dates(tmp_path, monkeypatch):
    _no_network_name_register(monkeypatch)
    module = _load_build_hitter_hit_log_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)
    output = tmp_path / "hitter_hit_log.csv"

    sys.argv = [
        "build_hitter_hit_log.py", "--raw-dir", str(raw_dir), "--season", "2026",
        "--output", str(output),
    ]
    module.main()
    first_run = pd.read_csv(output, parse_dates=["date"])

    # Re-running with the SAME persisted data must not duplicate rows -
    # every (date, game_pk, key_mlbam) pair recomputes identically and the
    # dedupe keeps exactly one row per pair.
    module.main()
    second_run = pd.read_csv(output, parse_dates=["date"])

    assert len(second_run) == len(first_run)
    assert not second_run.duplicated(subset=["date", "game_pk", "key_mlbam"]).any()

    # A --days-limited rerun still preserves the older, untouched dates
    # already on disk rather than truncating the file down to just the
    # recomputed window.
    sys.argv = [
        "build_hitter_hit_log.py", "--raw-dir", str(raw_dir), "--season", "2026",
        "--output", str(output), "--days", "1",
    ]
    module.main()
    third_run = pd.read_csv(output, parse_dates=["date"])

    assert third_run["date"].nunique() == first_run["date"].nunique()
    assert len(third_run) == len(first_run)


def test_build_hitter_hit_log_no_data_and_no_existing_file_writes_nothing(tmp_path, monkeypatch):
    _no_network_name_register(monkeypatch)
    module = _load_build_hitter_hit_log_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "hitter_hit_log.csv"

    sys.argv = [
        "build_hitter_hit_log.py", "--raw-dir", str(raw_dir), "--season", "2026",
        "--output", str(output),
    ]
    module.main()

    assert not output.exists()
