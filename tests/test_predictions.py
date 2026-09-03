import pandas as pd
import pytest

from mlb_metrics import config, predictions


def _hitters(rows):
    """rows: list of (key_mlbam, pa_l, pa_r, game_hit_prob). probability
    mirrors game_hit_prob (a reasonable synthetic stand-in) so these
    fixtures clear select_picks' joint HITTER_MIN_PROBABILITY gate on both
    columns unless a test is specifically exercising that gate."""
    return pd.DataFrame(
        [
            {
                "key_mlbam": key, "name_first": f"F{key}", "name_last": f"L{key}", "team": "NYY",
                "PA_L": pa_l, "PA_R": pa_r,
                "probability_L": 0, "probability_R": 0, "probability": ghp,
                "Game_Hit_Probability": ghp, "Consistency": 0, "Approach": ghp * ghp, "Expected_Bases": 0,
            }
            for key, pa_l, pa_r, ghp in rows
        ]
    )


def test_select_picks_applies_plate_appearance_qualifier_and_ranks():
    hitters = _hitters([
        (1, 0, 5, 0.99),   # unqualified: only 5 PA, would otherwise rank first
        (2, 10, 20, 0.80),
        (3, 15, 15, 0.70),
        (4, 0, 40, 0.60),
    ])

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=2, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [2, 3]
    assert list(picks["rank"]) == [1, 2]
    assert (picks["date"] == pd.Timestamp("2026-06-20")).all()
    assert list(picks["predicted_probability"]) == [0.80, 0.70]
    assert (picks["metric"] == "Game_Hit_Probability").all()
    assert picks["actual_hit"].isna().all()
    assert picks["at_bats"].isna().all()
    assert picks.loc[0, "name"] == "F2 L2"
    assert (picks["model_version"] == config.HITTER_MODEL_VERSION).all()


def test_select_picks_logs_probability_and_defaults_matchup_hit_probability_to_na():
    hitters = _hitters([(1, 0, 40, 0.9)])

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30)

    assert picks.iloc[0]["probability"] == 0.9  # passed through from the hitters table
    assert pd.isna(picks.iloc[0]["Matchup_Hit_Probability"])  # not merged into `hitters` this call


def test_select_picks_logs_matchup_hit_probability_when_present():
    hitters = _hitters([(1, 0, 40, 0.9)])
    hitters["Matchup_Hit_Probability"] = 0.75

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30)

    assert picks.iloc[0]["Matchup_Hit_Probability"] == 0.75


def test_select_picks_model_version_is_configurable():
    hitters = _hitters([(1, 0, 40, 0.9)])

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30, model_version="test-v")

    assert picks.loc[0, "model_version"] == "test-v"


def test_append_predictions_migrates_a_log_written_before_model_version_existed(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    legacy_log = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9,
         "metric": "Game_Hit_Probability", "actual_hit": 1, "at_bats": 1},
    ])
    legacy_log.to_csv(log_path, index=False)

    new_pick = predictions.select_picks(_hitters([(2, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    combined = predictions.append_predictions(new_pick, log_path)

    row_18 = combined[combined["date"] == "2026-06-18"].iloc[0]
    assert row_18["model_version"] == predictions.LEGACY_MODEL_VERSION
    row_19 = combined[combined["date"] == "2026-06-19"].iloc[0]
    assert row_19["model_version"] == config.HITTER_MODEL_VERSION


def test_select_picks_applies_lineup_qualifiers_when_columns_present():
    hitters = _hitters([
        (1, 0, 40, 0.90),  # top-half slot, consistent starter -> qualifies
        (2, 0, 40, 0.85),  # bats bottom of the order -> excluded
        (3, 0, 40, 0.80),  # bats top-half, but inconsistent (low start_rate) -> excluded
        (4, 0, 40, 0.70),  # never started at all (null avg) -> excluded, not treated as slot 0
    ])
    hitters["avg_batting_order"] = [2.0, 7.0, 2.0, float("nan")]
    hitters["start_rate"] = [0.9, 0.9, 0.2, 0.0]

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_lineup_qualifiers_are_noop_without_columns():
    # A batter who would fail the lineup qualifiers if they applied must
    # still be picked when the columns simply aren't present (old wave.csv
    # snapshots from before this feature existed).
    hitters = _hitters([(1, 0, 40, 0.90)])
    assert "avg_batting_order" not in hitters.columns

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_excludes_hitter_who_has_not_played_recently():
    hitters = _hitters([
        (1, 0, 40, 0.90),  # played 2 days before the pick date -> qualifies
        (2, 0, 40, 0.85),  # played 10 days before the pick date -> excluded
    ])
    hitters["Last_Game_Date"] = [pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-10")]

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_recency_gate_is_noop_without_last_game_date_column():
    # A batter who would fail the recency gate if it applied must still be
    # picked when the column simply isn't present (old wave.csv snapshots
    # from before this feature existed).
    hitters = _hitters([(1, 0, 40, 0.90)])
    assert "Last_Game_Date" not in hitters.columns

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_max_days_since_last_game_is_configurable():
    hitters = _hitters([(1, 0, 40, 0.90)])
    hitters["Last_Game_Date"] = [pd.Timestamp("2026-06-10")]  # 10 days before the pick date

    excluded = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)
    assert excluded.empty

    included = predictions.select_picks(
        hitters, "2026-06-20", top_n=5, min_plate_appearances=30, max_days_since_last_game=10
    )
    assert list(included["key_mlbam"]) == [1]


def test_select_picks_requires_both_probability_and_game_hit_probability_above_threshold():
    hitters = _hitters([
        (1, 0, 40, 0.90),  # both signals strong -> qualifies
        (2, 0, 40, 0.85),  # high Game_Hit_Probability but a divergent, low probability -> excluded
        (3, 0, 40, 0.75),  # high probability but a divergent, low Game_Hit_Probability -> excluded
    ])
    hitters.loc[hitters["key_mlbam"] == 2, "probability"] = 0.5
    hitters.loc[hitters["key_mlbam"] == 3, "Game_Hit_Probability"] = 0.5

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_min_probability_is_configurable():
    hitters = _hitters([(1, 0, 40, 0.65)])  # would fail the default 0.7 gate

    excluded = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)
    assert excluded.empty

    included = predictions.select_picks(
        hitters, "2026-06-20", top_n=5, min_plate_appearances=30, min_probability=0.6
    )
    assert list(included["key_mlbam"]) == [1]


def test_select_picks_model_shortlist_replaces_joint_gate_not_adds_to_it():
    # Player 1 would FAIL the old three-column JOINT_PROBABILITY_GATE_COLUMNS
    # gate (probability/Game_Hit_Probability both far below HITTER_MIN_PROBABILITY),
    # but has a strong Model_Hit_Probability. 11 total rows (>model_shortlist_size,
    # so the shortlist step is genuinely engaged, not a no-op) and top_n
    # covering the whole shortlist, so player 1's presence proves the model
    # shortlist REPLACES the old gate rather than requiring both, independent
    # of where Approach ranks them within the shortlist.
    rows = [(1, 0, 40, 0.2)] + [(i, 0, 40, 0.9) for i in range(2, 12)]
    hitters = _hitters(rows)
    hitters["Model_Hit_Probability"] = [0.99] + [0.5] * 10  # player 1 clears the shortlist easily

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=10, min_plate_appearances=30, rank_metric="Approach"
    )

    assert 1 in list(picks["key_mlbam"])


def test_select_picks_model_shortlist_excludes_heuristic_favorite_outside_model_top_n():
    # 11 qualified hitters. Player 99 has the single best Approach by far,
    # but the WORST Model_Hit_Probability (last of 11, outside the
    # default top-10 shortlist) - must be excluded even though it would win
    # a pure-heuristic ranking, proving the model shortlist is a real gate.
    rows = [(i, 0, 40, 0.5) for i in range(1, 11)] + [(99, 0, 40, 0.95)]
    hitters = _hitters(rows)
    hitters["Model_Hit_Probability"] = [0.60 + i * 0.01 for i in range(10)] + [0.1]

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=5, min_plate_appearances=30, rank_metric="Approach"
    )

    assert 99 not in list(picks["key_mlbam"])


def test_select_picks_model_shortlist_ranks_survivors_by_heuristic_not_model():
    # 11 qualified hitters (the 11th has the worst model score and gets
    # shortlisted out). Among the 10 survivors, player 1 has the HIGHEST
    # Model_Hit_Probability but a LOWER Approach than player 2 - the final
    # pick must be player 2, proving the two-stage design genuinely
    # re-ranks by the heuristic rather than just keeping the model's order.
    rows = [(1, 0, 40, 0.60), (2, 0, 40, 0.90)] + [(i, 0, 40, 0.50) for i in range(3, 12)]
    hitters = _hitters(rows)
    hitters["Model_Hit_Probability"] = [0.95, 0.80] + [0.70] * 8 + [0.10]

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30, rank_metric="Approach"
    )

    assert list(picks["key_mlbam"]) == [2]


def test_select_picks_model_shortlist_size_is_configurable():
    hitters = _hitters([(1, 0, 40, 0.9), (2, 0, 40, 0.8), (3, 0, 40, 0.7), (4, 0, 40, 0.6), (5, 0, 40, 0.5)])
    hitters["Model_Hit_Probability"] = [0.9, 0.8, 0.7, 0.6, 0.5]

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=5, min_plate_appearances=30,
        rank_metric="Approach", model_shortlist_size=2,
    )

    assert list(picks["key_mlbam"]) == [1, 2]  # only the top-2-by-model ever eligible


def test_select_picks_model_shortlist_smaller_pool_uses_whole_pool_unpadded():
    hitters = _hitters([(1, 0, 40, 0.6), (2, 0, 40, 0.9), (3, 0, 40, 0.7)])
    hitters["Model_Hit_Probability"] = [0.5, 0.6, 0.55]  # default shortlist_size=10 > 3 rows, no truncation

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=5, min_plate_appearances=30, rank_metric="Approach"
    )

    assert list(picks["key_mlbam"]) == [2, 3, 1]  # ranked by Approach (0.81, 0.49, 0.36), all 3 present


def test_select_picks_model_shortlist_null_model_probability_sorts_last_and_is_excluded():
    # A null Model_Hit_Probability must sort LAST (not be treated as 0 or
    # excluded specially) - with 11 qualified rows and the default
    # shortlist size of 10, the one null-scored hitter is the one that gets
    # truncated out.
    rows = [(1, 0, 40, 0.9)] + [(i, 0, 40, 0.9) for i in range(2, 12)]
    hitters = _hitters(rows)
    hitters["Model_Hit_Probability"] = [float("nan")] + [0.5] * 10

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=10, min_plate_appearances=30, rank_metric="Approach"
    )

    assert 1 not in list(picks["key_mlbam"])
    assert len(picks) == 10


def test_select_picks_model_hit_probability_absent_leaves_old_gate_unchanged():
    # Regression test: without a Model_Hit_Probability column at all, the
    # original three-column joint gate behaves exactly as before.
    hitters = _hitters([
        (1, 0, 40, 0.90),  # both signals strong -> qualifies
        (2, 0, 40, 0.85),  # divergent low probability -> excluded
    ])
    hitters.loc[hitters["key_mlbam"] == 2, "probability"] = 0.5

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_model_hit_probability_logged_and_defaults_to_na():
    with_model = _hitters([(1, 0, 40, 0.9)])
    with_model["Model_Hit_Probability"] = 0.8
    picks_with = predictions.select_picks(with_model, "2026-06-20", top_n=1, min_plate_appearances=30)
    assert picks_with.iloc[0]["Model_Hit_Probability"] == 0.8

    without_model = _hitters([(1, 0, 40, 0.9)])
    picks_without = predictions.select_picks(without_model, "2026-06-20", top_n=1, min_plate_appearances=30)
    assert pd.isna(picks_without.iloc[0]["Model_Hit_Probability"])


def _hitters_with_game_pk(rows):
    """rows: list of (key_mlbam, ghp, game_pk). Same shape as _hitters
    (min_probability=0 assumed by callers - see below) plus a game_pk
    column for quant-analytics item #4, slice 2's diversification tests."""
    hitters = _hitters([(key, 0, 40, ghp) for key, ghp, _ in rows])
    hitters["game_pk"] = [game_pk for _, _, game_pk in rows]
    return hitters


# Shared fixture for the diversification tests below: #1 (key=1, ghp=0.90)
# and #2 (key=2, ghp=0.85) share game_pk=100; key=3 (ghp=0.80, game_pk=200)
# is a close different-game alternative to #2 (0.85-0.80=0.05); key=4
# (ghp=0.60, game_pk=300) is a distant, irrelevant different-game candidate.
_DIVERSIFICATION_ROWS = [(1, 0.90, 100), (2, 0.85, 100), (3, 0.80, 200), (4, 0.60, 300)]


def test_select_picks_diversification_default_margin_is_a_noop():
    hitters = _hitters_with_game_pk(_DIVERSIFICATION_ROWS)

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=4, min_plate_appearances=30, min_probability=0)

    # 0.0 (the live default) reproduces today's exact single-column
    # ranking, even though game_pk is present and #1/#2 share a game.
    assert list(picks["key_mlbam"]) == [1, 2, 3, 4]
    assert list(picks["rank"]) == [1, 2, 3, 4]


def test_select_picks_diversification_swaps_in_a_close_different_game_candidate():
    hitters = _hitters_with_game_pk(_DIVERSIFICATION_ROWS)

    # margin=0.06 clears the real 0.05 gap between #2 (0.85, game 100) and
    # key=3 (0.80, game 200) - key=3 should be promoted into the #2 slot,
    # demoting the original #2 (key=2) to #3. key=4 (game 300, ghp=0.60)
    # is untouched either way - too far down to matter here.
    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=4, min_plate_appearances=30, min_probability=0,
        same_game_diversification_margin=0.06,
    )

    assert list(picks["key_mlbam"]) == [1, 3, 2, 4]
    assert list(picks["rank"]) == [1, 2, 3, 4]
    # #1 is untouched - still the real best candidate, its own probability unchanged.
    assert picks.iloc[0]["key_mlbam"] == 1
    assert picks.iloc[0]["predicted_probability"] == 0.90


def test_select_picks_diversification_does_not_swap_when_no_close_alternative_exists():
    hitters = _hitters_with_game_pk(_DIVERSIFICATION_ROWS)

    # margin=0.03 does NOT clear the real 0.05 gap - no real expected
    # value should be sacrificed to force a diversification that isn't
    # nearly free, so the original ranking survives unchanged.
    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=4, min_plate_appearances=30, min_probability=0,
        same_game_diversification_margin=0.03,
    )

    assert list(picks["key_mlbam"]) == [1, 2, 3, 4]


def test_select_picks_diversification_is_a_noop_without_game_pk_column():
    hitters = _hitters([(1, 0, 40, 0.90), (2, 0, 40, 0.85), (3, 0, 40, 0.80), (4, 0, 40, 0.60)])

    # No game_pk column at all (every historical wave.csv-only replay,
    # e.g. git_backtest.py) - column-gated, same convention as
    # avg_batting_order/Model_Hit_Probability above, even with a real
    # nonzero margin passed in.
    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=4, min_plate_appearances=30, min_probability=0,
        same_game_diversification_margin=0.06,
    )

    assert list(picks["key_mlbam"]) == [1, 2, 3, 4]


def test_select_picks_rank_metric_chooses_differently_than_metric_but_reports_metric():
    # Player 2 has the higher Game_Hit_Probability, but player 1 has the
    # higher Approach (Game_Hit_Probability * probability) once ranked on a
    # combined signal - rank_metric picks among qualified hitters by
    # whichever column it names, while predicted_probability/metric always
    # still reflect `metric` (Game_Hit_Probability here), not rank_metric.
    hitters = _hitters([(1, 0, 40, 0.80), (2, 0, 40, 0.82)])
    hitters.loc[hitters["key_mlbam"] == 1, "probability"] = 0.95
    hitters.loc[hitters["key_mlbam"] == 1, "Approach"] = 0.80 * 0.95
    hitters.loc[hitters["key_mlbam"] == 2, "probability"] = 0.72
    hitters.loc[hitters["key_mlbam"] == 2, "Approach"] = 0.82 * 0.72

    by_metric = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30)
    assert list(by_metric["key_mlbam"]) == [2]

    by_rank_metric = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30, rank_metric="Approach"
    )
    assert list(by_rank_metric["key_mlbam"]) == [1]
    assert by_rank_metric.iloc[0]["predicted_probability"] == 0.80  # still Game_Hit_Probability
    assert by_rank_metric.iloc[0]["metric"] == "Game_Hit_Probability"


def test_select_picks_filters_by_teams_playing_today():
    hitters = _hitters([
        (1, 0, 40, 0.90),
        (2, 0, 40, 0.80),
    ])
    hitters.loc[hitters["key_mlbam"] == 1, "team"] = "NYY"
    hitters.loc[hitters["key_mlbam"] == 2, "team"] = "BOS"

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=5, min_plate_appearances=30, teams_playing_today={"BOS"}
    )

    assert list(picks["key_mlbam"]) == [2]


def test_append_predictions_dedupes_and_prefers_existing_row(tmp_path):
    log_path = str(tmp_path / "predictions.csv")

    day1 = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    predictions.append_predictions(day1, log_path)

    # Simulate resolution: mark the 06-19 pick as a hit directly in the log
    # (real resolve_predictions always sets at_bats alongside actual_hit -
    # matched here since append_predictions's same-date-supersede logic
    # keys off at_bats, not actual_hit, to detect an already-resolved date).
    resolved = pd.read_csv(log_path, parse_dates=["date"])
    resolved.loc[0, "actual_hit"] = 1
    resolved.loc[0, "at_bats"] = 1
    resolved.to_csv(log_path, index=False)

    # Re-logging the same (date, key_mlbam, metric) should NOT clobber the
    # already-resolved actual_hit, and a genuinely new day's pick should
    # simply be added.
    day1_again = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    day2 = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-06-20", top_n=1, min_plate_appearances=30)
    combined = predictions.append_predictions(pd.concat([day1_again, day2]), log_path)

    assert len(combined) == 2
    row_19 = combined[combined["date"] == "2026-06-19"].iloc[0]
    assert row_19["actual_hit"] == 1
    row_20 = combined[combined["date"] == "2026-06-20"].iloc[0]
    assert pd.isna(row_20["actual_hit"])


def test_append_predictions_supersedes_a_still_pending_dates_stale_rows(tmp_path):
    # Real incident, 2026-08-04: two Daily Update runs landed either side
    # of a mid-day merge, and select_picks' top-N candidate SET changed
    # between them (a different rank_metric tier) - the second run's
    # hitters didn't share key_mlbam with the first run's, so per-key
    # dedup alone left both runs' picks sitting in the log side by side,
    # and evaluation.py's _combined_probability (which decides the
    # displayed "recommended" pick) picked the stale run's hitter over the
    # fresh one. Since nothing for that date has resolved yet, the fresh
    # batch must fully replace the stale one, not merely upsert by player.
    log_path = str(tmp_path / "predictions.csv")

    stale_pick = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-08-04", top_n=1, min_plate_appearances=30)
    predictions.append_predictions(stale_pick, log_path)

    fresh_pick = predictions.select_picks(_hitters([(2, 0, 40, 0.9)]), "2026-08-04", top_n=1, min_plate_appearances=30)
    combined = predictions.append_predictions(fresh_pick, log_path)

    assert list(combined["key_mlbam"]) == [2]


def test_append_predictions_does_not_supersede_a_dates_rows_once_any_resolved(tmp_path):
    # Once even one row for a date has resolved (real games happened),
    # a same-date rerun must fall back to the old safe per-key upsert -
    # never bulk-dropping a date's already-resolved backtest history.
    log_path = str(tmp_path / "predictions.csv")

    stale_pick = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-08-04", top_n=1, min_plate_appearances=30)
    predictions.append_predictions(stale_pick, log_path)

    resolved = pd.read_csv(log_path, parse_dates=["date"])
    resolved.loc[0, "actual_hit"] = 1
    resolved.loc[0, "at_bats"] = 1
    resolved.to_csv(log_path, index=False)

    fresh_pick = predictions.select_picks(_hitters([(2, 0, 40, 0.9)]), "2026-08-04", top_n=1, min_plate_appearances=30)
    combined = predictions.append_predictions(fresh_pick, log_path)

    assert set(combined["key_mlbam"]) == {1, 2}
    row_1 = combined[combined["key_mlbam"] == 1].iloc[0]
    assert row_1["actual_hit"] == 1


def test_append_predictions_dedupes_within_a_single_fresh_batch(tmp_path):
    """Regression test: a `picks` batch that already contains duplicate
    (date, key_mlbam, metric) rows - e.g. from git_backtest reconstructing
    the same date via two different commits - must be deduped even on the
    very first write, when there's no existing log to merge against yet."""
    log_path = str(tmp_path / "predictions.csv")

    day = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    duplicated_batch = pd.concat([day, day.copy()], ignore_index=True)

    combined = predictions.append_predictions(duplicated_batch, log_path)

    assert len(combined) == 1
    logged = pd.read_csv(log_path, parse_dates=["date"])
    assert len(logged) == 1


def test_resolve_predictions_fills_pending_and_leaves_resolved_rows_alone(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    log = pd.DataFrame([
        # Already resolved (at_bats already set) - must be left alone even
        # though completed_events below would compute a *different* result
        # for it (a miss instead of the stored hit), proving it's skipped.
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9, "metric": "Game_Hit_Probability", "actual_hit": 1, "at_bats": 1},
        {"date": "2026-06-19", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.8, "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
        {"date": "2026-06-19", "key_mlbam": 2, "name": "B", "rank": 2, "predicted_probability": 0.7, "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
        # Batter 3 had zero at-bats on 06-19 (not in completed_events at all
        # for that date) - should resolve to a confirmed no_game, not stay pending.
        {"date": "2026-06-19", "key_mlbam": 3, "name": "C", "rank": 3, "predicted_probability": 0.6, "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
        # 06-20 is beyond the outcome data's coverage - must stay pending.
        {"date": "2026-06-20", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.85, "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
    ])
    log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-18"), "batter": 1, "events": "field_out"},  # would be a miss - must be ignored
        {"game_date": pd.Timestamp("2026-06-19"), "batter": 1, "events": "single"},
        {"game_date": pd.Timestamp("2026-06-19"), "batter": 2, "events": "field_out"},
        # note: no rows at all for batter 3 on 06-19, and 06-20 doesn't appear anywhere.
    ])

    result = predictions.resolve_predictions(log_path, completed_events).set_index(["date", "key_mlbam"])

    row_18 = result.loc[(pd.Timestamp("2026-06-18"), 1)]
    assert row_18["actual_hit"] == 1 and row_18["at_bats"] == 1  # untouched, not recomputed

    row_19_1 = result.loc[(pd.Timestamp("2026-06-19"), 1)]
    assert row_19_1["actual_hit"] == 1 and row_19_1["at_bats"] == 1  # single -> hit

    row_19_2 = result.loc[(pd.Timestamp("2026-06-19"), 2)]
    assert row_19_2["actual_hit"] == 0 and row_19_2["at_bats"] == 1  # field_out -> miss

    row_19_3 = result.loc[(pd.Timestamp("2026-06-19"), 3)]
    assert pd.isna(row_19_3["actual_hit"]) and row_19_3["at_bats"] == 0  # confirmed no_game

    row_20 = result.loc[(pd.Timestamp("2026-06-20"), 1)]
    assert pd.isna(row_20["actual_hit"]) and pd.isna(row_20["at_bats"])  # still genuinely pending


def test_resolve_predictions_missing_log_returns_empty():
    result = predictions.resolve_predictions("/nonexistent/path.csv", pd.DataFrame(columns=["game_date", "batter", "events"]))
    assert result.empty


def test_resolve_predictions_migrates_a_log_written_before_at_bats_existed(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    # No at_bats column at all - simulates a log written by an older version.
    legacy_log = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9, "metric": "Game_Hit_Probability", "actual_hit": None},
    ])
    legacy_log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-18"), "batter": 1, "events": "single"},
    ])

    result = predictions.resolve_predictions(log_path, completed_events)

    assert result.loc[0, "actual_hit"] == 1
    assert result.loc[0, "at_bats"] == 1


def test_resolve_predictions_migrates_a_log_written_before_model_version_existed(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    legacy_log = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9,
         "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
    ])
    legacy_log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-18"), "batter": 1, "events": "single"},
    ])

    result = predictions.resolve_predictions(log_path, completed_events)

    assert result.loc[0, "model_version"] == predictions.LEGACY_MODEL_VERSION


def test_append_predictions_migrates_a_log_written_before_game_pk_existed(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    legacy_log = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9,
         "metric": "Game_Hit_Probability", "actual_hit": 1, "at_bats": 1, "model_version": "legacy"},
    ])
    legacy_log.to_csv(log_path, index=False)

    new_pick = predictions.select_picks(_hitters([(2, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    new_pick = new_pick.copy()
    new_pick["game_pk"] = 824651
    combined = predictions.append_predictions(new_pick, log_path)

    row_18 = combined[combined["date"] == "2026-06-18"].iloc[0]
    assert pd.isna(row_18["game_pk"])
    row_19 = combined[combined["date"] == "2026-06-19"].iloc[0]
    assert row_19["game_pk"] == 824651
    assert "game_pk" in predictions.PREDICTION_COLUMNS


def test_append_predictions_legacy_null_game_pk_still_dedupes(tmp_path):
    # Pandas treats NA as equal in drop_duplicates, so two legacy rows with
    # the same (date, NA, key_mlbam, metric) still collapse the way the old
    # (date, key_mlbam, metric) key did.
    log_path = str(tmp_path / "predictions.csv")
    legacy = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9,
         "metric": "Game_Hit_Probability", "probability": 0.9, "Matchup_Hit_Probability": pd.NA,
         "Model_Hit_Probability": pd.NA, "actual_hit": 1, "at_bats": 1, "model_version": "legacy"},
    ])
    legacy.to_csv(log_path, index=False)

    again = legacy.copy()
    again["predicted_probability"] = 0.5  # would overwrite if keep=last preferred new
    combined = predictions.append_predictions(again, log_path)

    assert len(combined) == 1
    assert combined.iloc[0]["actual_hit"] == 1  # existing resolved row wins (keep=last on concat[picks, existing])


def test_resolve_predictions_uses_game_pk_when_present(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    log = pd.DataFrame([
        {"date": "2026-06-19", "game_pk": 101, "key_mlbam": 1, "name": "A", "rank": 1,
         "predicted_probability": 0.8, "metric": "Game_Hit_Probability",
         "probability": 0.8, "Matchup_Hit_Probability": pd.NA, "Model_Hit_Probability": pd.NA,
         "actual_hit": None, "at_bats": None, "model_version": "v1"},
        {"date": "2026-06-19", "game_pk": 102, "key_mlbam": 1, "name": "A", "rank": 2,
         "predicted_probability": 0.7, "metric": "Game_Hit_Probability",
         "probability": 0.7, "Matchup_Hit_Probability": pd.NA, "Model_Hit_Probability": pd.NA,
         "actual_hit": None, "at_bats": None, "model_version": "v1"},
    ])
    log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-19"), "game_pk": 101, "batter": 1, "events": "field_out"},
        {"game_date": pd.Timestamp("2026-06-19"), "game_pk": 102, "batter": 1, "events": "single"},
    ])

    result = predictions.resolve_predictions(log_path, completed_events).set_index("game_pk")
    assert result.loc[101, "actual_hit"] == 0
    assert result.loc[102, "actual_hit"] == 1


def test_resolve_predictions_legacy_null_game_pk_still_resolves_by_date(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    log = pd.DataFrame([
        {"date": "2026-06-19", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.8,
         "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
    ])
    log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-19"), "game_pk": 101, "batter": 1, "events": "field_out"},
        {"game_date": pd.Timestamp("2026-06-19"), "game_pk": 102, "batter": 1, "events": "single"},
    ])

    result = predictions.resolve_predictions(log_path, completed_events)
    # Date-level legacy label: any hit that day -> hit.
    assert result.loc[0, "actual_hit"] == 1
    assert pd.isna(result.loc[0, "game_pk"])


def test_select_picks_carries_game_pk_when_present():
    hitters = _hitters([(1, 0, 40, 0.9)])
    hitters["game_pk"] = 824651

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30)

    assert picks.iloc[0]["game_pk"] == 824651
    assert list(picks.columns) == predictions.PREDICTION_COLUMNS


def test_select_picks_stamps_selection_and_probability_provenance():
    hitters = _hitters([(1, 0, 40, 0.9), (2, 0, 40, 0.8)])
    hitters["Matchup_Approach"] = [0.5, 0.7]

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30, rank_metric="Matchup_Approach"
    )

    row = picks.iloc[0]
    assert row["key_mlbam"] == 2
    assert row["selection_metric"] == "Matchup_Approach"
    assert row["selection_score"] == pytest.approx(0.7)
    assert row["probability_source"] == "Game_Hit_Probability"
    assert row["predicted_probability"] == pytest.approx(0.8)
    assert row["selection_logic_version"] == config.HITTER_MODEL_VERSION
    assert row["prediction_snapshot_type"] == "morning"
    assert row["lineup_status"] == "unconfirmed"
    assert row["fallback_used"] == False  # noqa: E712
    assert pd.isna(row["fallback_reason"]) or row["fallback_reason"] is None
    assert row["prediction_code_sha"]  # non-empty


def test_select_picks_records_heuristic_fallback_from_model_load_failure():
    hitters = _hitters([(1, 0, 40, 0.9)])
    status = {
        "loaded": False,
        "fallback_used": True,
        "fallback_reason": "missing_artifact",
        "artifact_id": None,
        "training_data_cutoff": None,
        "feature_schema_hash": None,
    }

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30,
        model_status=status, fallback_used=True, fallback_reason="missing_artifact",
    )

    assert picks.iloc[0]["fallback_used"] == True  # noqa: E712
    assert picks.iloc[0]["fallback_reason"] == "missing_artifact"
    assert pd.isna(picks.iloc[0]["model_artifact_id"])


def test_select_picks_records_model_artifact_metadata_when_loaded():
    hitters = _hitters([(1, 0, 40, 0.9)])
    hitters["Model_Hit_Probability"] = 0.85
    status = {
        "loaded": True,
        "fallback_used": False,
        "fallback_reason": None,
        "artifact_id": "abc123",
        "training_data_cutoff": "2026-07-01",
        "feature_schema_hash": "deadbeef",
    }

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30, model_status=status,
    )

    assert picks.iloc[0]["model_artifact_id"] == "abc123"
    assert picks.iloc[0]["training_data_cutoff"] == "2026-07-01"
    assert picks.iloc[0]["feature_schema_hash"] == "deadbeef"
    assert picks.iloc[0]["fallback_used"] == False  # noqa: E712


def test_append_predictions_migrates_provenance_columns_on_legacy_csv(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    legacy = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9,
         "metric": "Game_Hit_Probability", "actual_hit": 1, "at_bats": 1, "model_version": "legacy"},
    ])
    legacy.to_csv(log_path, index=False)

    new_pick = predictions.select_picks(_hitters([(2, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    combined = predictions.append_predictions(new_pick, log_path)

    assert set(predictions.PROVENANCE_COLUMNS).issubset(combined.columns)
    row_18 = combined[combined["date"] == "2026-06-18"].iloc[0]
    assert row_18["prediction_code_sha"] == "legacy"
    assert row_18["selection_logic_version"] == "legacy"
    assert row_18["prediction_snapshot_type"] == "legacy"
    row_19 = combined[combined["date"] == "2026-06-19"].iloc[0]
    assert row_19["prediction_snapshot_type"] == "morning"
    assert pd.isna(row_19["actual_hit"])
