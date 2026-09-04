"""No-lookahead hitter-opportunity dataset: candidate universe, labels,
coverage, and leakage guards. Does not train or wire a live model.
"""

import inspect
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, data, dfs_ml, hitter_training_data, lineup, pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_network_name_register(monkeypatch):
    monkeypatch.setattr(
        data, "get_name_register",
        lambda: pd.DataFrame(columns=["key_mlbam", "key_bbref", "name_first", "name_last"]),
    )


def _game_rows(game_pk, date, events, pitcher=99, batter=1, home_team="NYY", away_team="BOS", inning_topbot="Top"):
    rows = []
    runs = 0
    for i, e in enumerate(events):
        pre = runs
        if e in ("home_run", "single"):
            runs += 1
        is_top = inning_topbot == "Top"
        rows.append({
            "game_pk": game_pk, "game_date": date, "pitcher": pitcher, "batter": batter,
            "events": e, "p_throws": "R", "inning_topbot": inning_topbot,
            "home_team": home_team, "away_team": away_team,
            "at_bat_number": i + 1, "pitch_number": 1,
            "home_score": 0 if is_top else pre, "away_score": pre if is_top else 0,
            "post_home_score": 0 if is_top else runs, "post_away_score": runs if is_top else 0,
            "bat_score": pre, "post_bat_score": runs,
        })
    return rows


STANDARD_EVENTS = ["strikeout"] * 5 + ["field_out"] * 6 + ["walk"] * 3 + ["single"] * 4 + ["double"] * 1 + ["home_run"] * 1


def _multi_game_statcast(n_games=6, gap_days=5, batter=1):
    rows = []
    for i in range(n_games):
        date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=i * gap_days)
        rows.extend(_game_rows(i + 1, date, STANDARD_EVENTS, batter=batter))
    return pd.DataFrame(rows)


def _clone_batter_as(df, source_batter, new_batter, game_pks=None):
    subset = df[df["batter"] == source_batter]
    if game_pks is not None:
        subset = subset[subset["game_pk"].isin(game_pks)]
    extra = []
    for _game_pk, g in subset.groupby("game_pk"):
        clone = g.copy()
        clone["batter"] = new_batter
        clone["at_bat_number"] = clone["at_bat_number"] + int(g["at_bat_number"].max())
        extra.append(clone)
    if not extra:
        return df
    return pd.concat([df, *extra], ignore_index=True)


def test_feature_columns_exclude_target_labels():
    assert "Batting_Order" not in hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS
    assert "Started" not in hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS
    assert "Appeared" not in hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS
    assert "Got_Hit" not in hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS
    assert "Hits" not in hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS
    assert "Plate_Appearances" not in hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS
    assert "No_Game" not in hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS
    assert "Batting_Order" in hitter_training_data.OPPORTUNITY_LABEL_COLUMNS
    assert set(hitter_training_data.OPPORTUNITY_LABEL_COLUMNS).isdisjoint(
        hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS
    )
    assert not any(c.startswith("Actual_") for c in hitter_training_data.OPPORTUNITY_FEATURE_COLUMNS)


def test_latest_known_team_uses_game_pk_not_future_rows():
    rows = (
        _game_rows(1, pd.Timestamp("2026-05-01"), STANDARD_EVENTS, batter=1, away_team="BOS", home_team="NYY")
        + _game_rows(2, pd.Timestamp("2026-05-10"), STANDARD_EVENTS, batter=1, away_team="BOS", home_team="NYY")
        + _game_rows(
            3, pd.Timestamp("2026-05-20"), STANDARD_EVENTS, batter=1,
            away_team="TBR", home_team="NYY", inning_topbot="Bot",
        )
    )
    history = pd.DataFrame(rows)
    history = history[history["game_pk"] < 3]
    latest = hitter_training_data.latest_known_batter_team(
        hitter_training_data.batter_team_appearances(history)
    ).set_index("key_mlbam")
    assert latest.loc[1, "team"] == "BOS"


def test_candidate_universe_requires_latest_team_and_lookback(monkeypatch):
    monkeypatch.setattr(config, "HITTER_OPPORTUNITY_LOOKBACK_TEAM_GAMES", 1)
    monkeypatch.setattr(config, "HITTER_OPPORTUNITY_LOOKBACK_CALENDAR_DAYS", 1)

    rows = (
        _game_rows(1, pd.Timestamp("2026-05-01"), STANDARD_EVENTS, batter=1)
        + _game_rows(2, pd.Timestamp("2026-05-06"), STANDARD_EVENTS, batter=2)
    )
    history = pd.DataFrame(rows)
    schedule = pd.DataFrame([
        {"date": pd.Timestamp("2026-05-11"), "game_pk": 3, "team": "BOS", "opponent": "NYY", "is_home": False},
    ])
    candidates = hitter_training_data.build_pregame_candidate_universe(
        history, schedule, pd.Timestamp("2026-05-11"), lookback_games=1, lookback_days=1,
    )
    assert set(candidates["key_mlbam"]) == {2}
    assert (candidates["candidate_source"] == hitter_training_data.CANDIDATE_SOURCE_PREGAME_LOOKBACK).all()


def test_candidate_universe_ignores_actual_lineup_frame():
    history = _multi_game_statcast(n_games=3)
    schedule = pd.DataFrame([
        {"date": pd.Timestamp("2026-05-16"), "game_pk": 99, "team": "BOS", "opponent": "NYY", "is_home": False},
    ])
    source = inspect.getsource(hitter_training_data.build_pregame_candidate_universe)
    assert "assign_batting_order" not in source
    assert "batting_order" not in source
    candidates = hitter_training_data.build_pregame_candidate_universe(
        history, schedule, pd.Timestamp("2026-05-16"),
    )
    assert set(candidates["key_mlbam"]) == {1}


def test_labels_started_null_order_when_not_started_and_keep_pa_counts():
    date = pd.Timestamp("2026-06-01")
    rows = []
    for slot, batter in enumerate(range(101, 111), start=1):
        rows.extend(_game_rows(
            50, date, ["single"] if slot == 1 else ["strikeout"],
            batter=batter, pitcher=9,
        ))
        rows[-1]["at_bat_number"] = slot
    events = pd.DataFrame(rows)
    labels = hitter_training_data.compute_game_hitter_labels(events).set_index("key_mlbam")

    assert labels.loc[101, "Started"] == 1
    assert labels.loc[101, "Batting_Order"] == 1
    assert labels.loc[101, "Appeared"] == 1
    assert labels.loc[101, "Hits"] == 1
    assert labels.loc[101, "Got_Hit"] == 1
    assert labels.loc[101, "Plate_Appearances"] == 1
    assert labels.loc[101, "Official_At_Bats"] == 1

    assert labels.loc[110, "Started"] == 0
    assert pd.isna(labels.loc[110, "Batting_Order"])
    assert labels.loc[110, "Appeared"] == 1
    assert labels.loc[110, "No_Game"] == 0

    walk_rows = pd.DataFrame(_game_rows(51, date, ["walk", "strikeout"], batter=201))
    walk_labels = hitter_training_data.compute_game_hitter_labels(walk_rows).set_index("key_mlbam")
    assert walk_labels.loc[201, "Plate_Appearances"] == 2
    assert walk_labels.loc[201, "Official_At_Bats"] == 1


def test_dnp_candidates_are_kept_as_negative_examples(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = _multi_game_statcast(n_games=6)
    history_pks = set(df["game_pk"].unique()) - {6}
    df = _clone_batter_as(df, 1, 2, game_pks=history_pks)
    df.to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    rows, _coverage = hitter_training_data.assemble_hitter_opportunity_dataset(
        str(raw_dir), season=2026, days=None,
    )
    last = rows[(rows["game_pk"] == 6) & (rows["team"] == "BOS")].set_index("key_mlbam")
    assert 2 in last.index
    assert last.loc[2, "Appeared"] == 0
    assert last.loc[2, "Started"] == 0
    assert last.loc[2, "No_Game"] == 1
    assert pd.isna(last.loc[2, "Batting_Order"])
    assert last.loc[2, "Plate_Appearances"] == 0
    assert last.loc[2, "Got_Hit"] == 0
    assert 1 in last.index
    assert last.loc[1, "Appeared"] == 1
    assert last.loc[1, "No_Game"] == 0


def test_callup_without_pregame_history_is_omitted_from_candidates(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = _multi_game_statcast(n_games=6)
    last = df[df["game_pk"] == 6].copy()
    last["batter"] = 999
    last["at_bat_number"] = last["at_bat_number"] + 50
    df = pd.concat([df, last], ignore_index=True)
    df.to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    rows, coverage = hitter_training_data.assemble_hitter_opportunity_dataset(
        str(raw_dir), season=2026, days=None,
    )
    last_rows = rows[rows["game_pk"] == 6]
    assert 999 not in set(last_rows["key_mlbam"])
    last_cov = coverage[(coverage["game_pk"] == 6) & (coverage["team"] == "BOS")].iloc[0]
    assert last_cov["n_omitted_no_pregame_history"] >= 1
    assert last_cov["n_omitted_appeared"] >= 1
    assert last_cov["appearance_coverage"] < 1


def test_doubleheader_keeps_separate_game_pks_and_dnp_on_both(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _multi_game_statcast(n_games=5).to_dict("records")
    dh_date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=5 * 5)
    miss = _game_rows(101, dh_date, ["strikeout"] * 5 + ["field_out"] * 10 + ["walk"] * 5, pitcher=201)
    miss[-1]["home_score"] = 1
    miss[-1]["post_home_score"] = 1
    rows.extend(miss)
    rows.extend(_game_rows(102, dh_date, STANDARD_EVENTS, pitcher=202))
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    result = hitter_training_data.assemble_hitter_opportunity_log(str(raw_dir), season=2026)
    dh = result[result["date"] == dh_date]
    assert set(dh["game_pk"]) == {101, 102}
    batter = dh[dh["key_mlbam"] == 1].set_index("game_pk")
    assert len(batter) == 2
    assert batter.loc[101, "Got_Hit"] == 0
    assert batter.loc[102, "Got_Hit"] == 1
    assert not result.duplicated(subset=hitter_training_data.OPPORTUNITY_KEY_COLUMNS).any()


def test_assemble_is_deterministic_and_asserts_unique_keys(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    first = hitter_training_data.assemble_hitter_opportunity_log(str(raw_dir), season=2026)
    second = hitter_training_data.assemble_hitter_opportunity_log(str(raw_dir), season=2026)
    pd.testing.assert_frame_equal(first, second)
    hitter_training_data.assert_unique_opportunity_keys(first)

    duped = pd.concat([first, first], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate"):
        hitter_training_data.assert_unique_opportunity_keys(duped)


def test_target_game_rows_absent_from_feature_history(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = _multi_game_statcast(n_games=6)
    df.to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    rows = hitter_training_data.assemble_hitter_opportunity_log(str(raw_dir), season=2026)
    assert not rows.empty
    for _, row in rows.iterrows():
        history = hitter_training_data.slice_history_before_game(
            df, row["as_of_date"], target_game_pk=row["game_pk"],
        )
        assert row["game_pk"] not in set(history["game_pk"])
        expected = pipeline.compute_outputs(history)["wave"].set_index("key_mlbam")
        key = row["key_mlbam"]
        if key in expected.index and pd.notna(row["WAVE"]):
            assert row["WAVE"] == pytest.approx(float(expected.loc[key, "WAVE"]))
        assert row["Days_Rest"] >= 1 or pd.isna(row["Days_Rest"])


def _with_completed_at(df, game_pk, ts):
    out = df.copy()
    out["game_completed_at"] = pd.Series([pd.NaT] * len(out), dtype="datetime64[ns, UTC]")
    out.loc[out["game_pk"] == game_pk, "game_completed_at"] = ts
    return out


def test_actual_batting_order_is_label_not_pregame_feature(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = _multi_game_statcast(n_games=6)
    df = _clone_batter_as(df, 1, 2)
    df.to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    rows = hitter_training_data.assemble_hitter_opportunity_log(str(raw_dir), season=2026)
    assert "Batting_Order" in rows.columns
    assert "avg_batting_order" in rows.columns
    last = rows[(rows["game_pk"] == 6) & (rows["key_mlbam"] == 1)].iloc[0]
    assert last["Batting_Order"] == 1
    source = inspect.getsource(hitter_training_data._attach_pregame_features)
    assert "Batting_Order" not in source


def test_morning_snapshot_game_two_features_ignore_game_one(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _multi_game_statcast(n_games=5).to_dict("records")
    dh_date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=5 * 5)
    boom = _game_rows(101, dh_date, ["home_run"] * 20, pitcher=201)
    boom[-1]["home_score"] = 1
    boom[-1]["post_home_score"] = 1
    rows.extend(boom)
    rows.extend(_game_rows(102, dh_date, STANDARD_EVENTS, pitcher=202))
    df = _with_completed_at(
        pd.DataFrame(rows), 101, pd.Timestamp("2026-05-26 16:00:00", tz="UTC"),
    )
    df.to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    result = hitter_training_data.assemble_hitter_opportunity_log(str(raw_dir), season=2026)
    g2 = result[(result["game_pk"] == 102) & (result["key_mlbam"] == 1)].iloc[0]
    history = df[df["game_date"] < dh_date].drop(columns=["game_completed_at"])
    morning_wave = pipeline.compute_outputs(history)["wave"].set_index("key_mlbam")
    leaked_wave = pipeline.compute_outputs(
        df[df["game_pk"] != 102].drop(columns=["game_completed_at"])
    )["wave"].set_index("key_mlbam")

    assert g2["WAVE"] == pytest.approx(float(morning_wave.loc[1, "WAVE"]))
    assert leaked_wave.loc[1, "WAVE"] != pytest.approx(float(morning_wave.loc[1, "WAVE"]))
    assert g2["Days_Rest"] >= 1


def test_game_two_may_use_game_one_only_after_configured_completion(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _multi_game_statcast(n_games=5).to_dict("records")
    dh_date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=5 * 5)
    boom = _game_rows(101, dh_date, ["home_run"] * 20, pitcher=201)
    boom[-1]["home_score"] = 1
    boom[-1]["post_home_score"] = 1
    rows.extend(boom)
    rows.extend(_game_rows(102, dh_date, STANDARD_EVENTS, pitcher=202))
    df = _with_completed_at(
        pd.DataFrame(rows), 101, pd.Timestamp("2026-05-26 16:00:00", tz="UTC"),
    )
    df.to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    too_early = hitter_training_data.assemble_hitter_opportunity_log(
        str(raw_dir), season=2026,
        prediction_timestamp_utc=pd.Timestamp("2026-05-26 15:00:00", tz="UTC"),
    )
    after = hitter_training_data.assemble_hitter_opportunity_log(
        str(raw_dir), season=2026,
        prediction_timestamp_utc=pd.Timestamp("2026-05-26 18:00:00", tz="UTC"),
    )
    history = df[df["game_date"] < dh_date].drop(columns=["game_completed_at"])
    morning_wave = float(pipeline.compute_outputs(history)["wave"].set_index("key_mlbam").loc[1, "WAVE"])
    with_g1 = float(
        pipeline.compute_outputs(df[df["game_pk"] != 102].drop(columns=["game_completed_at"]))["wave"]
        .set_index("key_mlbam").loc[1, "WAVE"]
    )

    early_g2 = too_early[(too_early["game_pk"] == 102) & (too_early["key_mlbam"] == 1)].iloc[0]
    late_g2 = after[(after["game_pk"] == 102) & (after["key_mlbam"] == 1)].iloc[0]
    late_g1 = after[(after["game_pk"] == 101) & (after["key_mlbam"] == 1)].iloc[0]

    assert early_g2["WAVE"] == pytest.approx(morning_wave)
    assert late_g2["WAVE"] == pytest.approx(with_g1)
    assert late_g1["WAVE"] == pytest.approx(morning_wave)


def test_same_day_debut_does_not_improve_game_two_coverage(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _multi_game_statcast(n_games=5).to_dict("records")
    dh_date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=5 * 5)
    g1 = _game_rows(101, dh_date, STANDARD_EVENTS, pitcher=201, batter=888)
    g1[-1]["home_score"] = 1
    g1[-1]["post_home_score"] = 1
    g1_regular = _game_rows(101, dh_date, ["strikeout"] * 20, pitcher=201, batter=1)
    for r in g1_regular:
        r["at_bat_number"] += 50
    rows.extend(g1)
    rows.extend(g1_regular)
    rows.extend(_game_rows(102, dh_date, STANDARD_EVENTS, pitcher=202, batter=888))
    g2_regular = _game_rows(102, dh_date, STANDARD_EVENTS, pitcher=202, batter=1)
    for r in g2_regular:
        r["at_bat_number"] += 50
    rows.extend(g2_regular)
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    result, coverage = hitter_training_data.assemble_hitter_opportunity_dataset(str(raw_dir), season=2026)
    dh = result[result["date"] == dh_date]
    assert 888 not in set(dh["key_mlbam"])
    g2_cov = coverage[(coverage["game_pk"] == 102) & (coverage["team"] == "BOS")].iloc[0]
    assert g2_cov["n_omitted_no_pregame_history"] >= 1


def test_future_team_assignment_is_not_used(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    history = _multi_game_statcast(n_games=5)
    last_date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=5 * 5)
    nyy_half = _game_rows(
        6, last_date, STANDARD_EVENTS, pitcher=77, batter=1,
        home_team="NYY", away_team="BOS", inning_topbot="Bot",
    )
    bos_half = _game_rows(
        6, last_date, STANDARD_EVENTS, pitcher=88, batter=5,
        home_team="NYY", away_team="BOS", inning_topbot="Top",
    )
    for r in nyy_half:
        r["at_bat_number"] += 20
    df = pd.concat([history, pd.DataFrame(bos_half + nyy_half)], ignore_index=True)
    df.to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    rows, coverage = hitter_training_data.assemble_hitter_opportunity_dataset(str(raw_dir), season=2026)
    last = rows[rows["game_pk"] == 6]
    nyy = last[(last["team"] == "NYY") & (last["key_mlbam"] == 1)]
    bos = last[(last["team"] == "BOS") & (last["key_mlbam"] == 1)]
    assert nyy.empty
    assert not bos.empty
    assert bos.iloc[0]["Appeared"] == 0
    nyy_cov = coverage[(coverage["game_pk"] == 6) & (coverage["team"] == "NYY")].iloc[0]
    assert nyy_cov["n_omitted_appeared"] >= 1


def test_historical_assemble_never_calls_confirmed_lineup_path():
    source = inspect.getsource(hitter_training_data.assemble_hitter_opportunity_dataset)
    assert "add_confirmed_lineup_candidates" not in source
    assert hitter_training_data.CANDIDATE_SOURCE_CONFIRMED_LINEUP not in source


def test_confirmed_lineup_path_adds_unseen_player_with_league_priors():
    historical = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-06-01"), "game_pk": 10, "team": "BOS",
            "opponent": "NYY", "key_mlbam": 1, "WAVE": 0.3,
            "candidate_source": hitter_training_data.CANDIDATE_SOURCE_PREGAME_LOOKBACK,
        }
    ])
    confirmed = pd.DataFrame([
        {"date": pd.Timestamp("2026-06-01"), "game_pk": 10, "team": "BOS", "key_mlbam": 1},
        {"date": pd.Timestamp("2026-06-01"), "game_pk": 10, "team": "BOS", "key_mlbam": 42},
    ])
    priors = pd.Series({"WAVE": 0.25, "start_rate": 0.0})
    out = hitter_training_data.add_confirmed_lineup_candidates(historical, confirmed, priors)
    added = out[out["key_mlbam"] == 42].iloc[0]
    assert added["candidate_source"] == hitter_training_data.CANDIDATE_SOURCE_CONFIRMED_LINEUP
    assert added["WAVE"] == 0.25
    assert (out["key_mlbam"] == 1).sum() == 1


def test_slice_history_morning_excludes_same_date_even_with_completion_time():
    dh_date = pd.Timestamp("2026-05-26")
    df = pd.DataFrame(
        _game_rows(1, pd.Timestamp("2026-05-20"), STANDARD_EVENTS)
        + _game_rows(101, dh_date, STANDARD_EVENTS)
        + _game_rows(102, dh_date, STANDARD_EVENTS)
    )
    df = _with_completed_at(df, 101, pd.Timestamp("2026-05-26 16:00:00", tz="UTC"))
    morning = hitter_training_data.slice_history_before_game(df, dh_date, target_game_pk=102)
    assert set(morning["game_pk"]) == {1}

    after = hitter_training_data.slice_history_before_game(
        df, dh_date, target_game_pk=102,
        prediction_timestamp_utc=pd.Timestamp("2026-05-26 18:00:00", tz="UTC"),
    )
    assert set(after["game_pk"]) == {1, 101}
    assert 102 not in set(after["game_pk"])


def test_summarize_opportunity_log_reports_coverage_and_labels():
    rows = pd.DataFrame({
        "date": [pd.Timestamp("2026-05-01")] * 3,
        "game_pk": [1, 1, 1],
        "key_mlbam": [1, 2, 3],
        "Started": [1, 0, 0],
        "Appeared": [1, 0, 1],
        "Got_Hit": [1, 0, 0],
        "No_Game": [0, 1, 0],
        "candidate_source": [hitter_training_data.CANDIDATE_SOURCE_PREGAME_LOOKBACK] * 3,
    })
    coverage = pd.DataFrame({
        "date": [pd.Timestamp("2026-05-01")],
        "game_pk": [1],
        "team": ["BOS"],
        "n_candidates": [3],
        "n_actual_starters": [2],
        "n_starters_in_candidates": [1],
        "starter_coverage": [0.5],
        "n_actual_appeared": [2],
        "n_appeared_in_candidates": [2],
        "appearance_coverage": [1.0],
        "n_omitted_appeared": [0],
        "n_omitted_no_pregame_history": [0],
    })
    summary = hitter_training_data.summarize_opportunity_log(rows, coverage)
    assert summary["starter_coverage"] == pytest.approx(0.5)
    assert summary["appearance_coverage"] == pytest.approx(1.0)
    assert summary["No_Game_count"] == 1
    assert summary["Got_Hit_rate_among_appeared"] == pytest.approx(0.5)


def _load_script(name: str):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_script_is_idempotent_and_asserts_unique_keys(tmp_path):
    module = _load_script("build_hitter_opportunity_log.py")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)
    output = tmp_path / "hitter_opportunity_log.csv"

    sys.argv = [
        "build_hitter_opportunity_log.py", "--raw-dir", str(raw_dir), "--season", "2026",
        "--output", str(output),
    ]
    module.main()
    first = pd.read_csv(output, parse_dates=["date"])
    expected_feature_subset = set(dfs_ml.HITTER_FEATURE_COLUMNS)
    assert expected_feature_subset.issubset(set(first.columns))
    assert {"Started", "Appeared", "Batting_Order", "No_Game", "Got_Hit", "Hits"}.issubset(set(first.columns))
    assert first["candidate_source"].eq(hitter_training_data.CANDIDATE_SOURCE_PREGAME_LOOKBACK).all()

    module.main()
    second = pd.read_csv(output, parse_dates=["date"])
    assert len(second) == len(first)
    assert not second.duplicated(subset=["date", "game_pk", "key_mlbam"]).any()

    coverage = pd.read_csv(tmp_path / "hitter_opportunity_log_coverage.csv")
    assert "starter_coverage" in coverage.columns
    assert "n_omitted_no_pregame_history" in coverage.columns


def test_report_script_prints_coverage(tmp_path, capsys):
    module = _load_script("report_hitter_opportunity_log.py")
    log = tmp_path / "hitter_opportunity_log.csv"
    cov = tmp_path / "hitter_opportunity_log_coverage.csv"
    pd.DataFrame({
        "date": [pd.Timestamp("2026-05-10")],
        "game_pk": [1],
        "key_mlbam": [1],
        "Started": [1],
        "Appeared": [1],
        "Got_Hit": [0],
        "No_Game": [0],
        "candidate_source": [hitter_training_data.CANDIDATE_SOURCE_PREGAME_LOOKBACK],
    }).to_csv(log, index=False)
    pd.DataFrame({
        "date": [pd.Timestamp("2026-05-10")],
        "game_pk": [1],
        "team": ["BOS"],
        "n_candidates": [1],
        "n_actual_starters": [1],
        "n_starters_in_candidates": [1],
        "starter_coverage": [1.0],
        "n_actual_appeared": [1],
        "n_appeared_in_candidates": [1],
        "appearance_coverage": [1.0],
        "n_omitted_appeared": [0],
        "n_omitted_no_pregame_history": [0],
    }).to_csv(cov, index=False)

    sys.argv = ["report_hitter_opportunity_log.py", "--log", str(log), "--coverage", str(cov)]
    module.main()
    printed = capsys.readouterr().out
    assert "starter coverage" in printed
    assert "omitted with no pregame history" in printed
    assert "No_Game" in printed


def test_starter_max_matches_lineup_constant():
    assert lineup.STARTER_MAX_BATTING_ORDER == 9
