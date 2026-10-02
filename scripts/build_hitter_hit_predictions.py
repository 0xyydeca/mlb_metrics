"""Build docs/data/hitter_hit_predictions.csv: today's PA-qualified hitters
ranked by dfs_ml's trained hit-probability model
(config.HITTER_HIT_PROBABILITY_MODEL_PATH, scripts/train_hitter_hit_model.py),
for the dashboard's Beat the Streak "Model Odds" subtab.

Purely informational - this is an ADDITIONAL, independent view alongside
the official Beat the Streak picks. It does not gate, filter, or replace
anything in the live pick-selection path.

Excludes hitters who haven't played within config.HITTER_MAX_DAYS_SINCE_
LAST_GAME days (same gate predictions.select_picks/build_dfs_rankings.py
already apply) - a season-long injured-list stay can leave season-to-date
rates looking strong even though the hitter hasn't actually played in
weeks, and this tab has no other way to notice that.

Writes a dated status file and an empty current export when inputs are unavailable,
so a previous slate is never silently presented as today's predictions.

Usage:
    python scripts/build_hitter_hit_predictions.py
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, dfs_ml, matchup, schedule


OUTPUT_COLUMNS = ["key_mlbam", "game_pk", "name_first", "name_last", "team", "opponent", "is_home", "Model_Hit_Probability", "forecast_date", "generated_at_utc"]


def publish(args, result=None, *, status):
    os.makedirs(args.data_dir, exist_ok=True)
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = pd.DataFrame(columns=OUTPUT_COLUMNS) if result is None else result.copy()
    result["forecast_date"] = args.as_of_date.isoformat()
    result["generated_at_utc"] = generated
    result.reindex(columns=OUTPUT_COLUMNS).to_csv(os.path.join(args.data_dir, "hitter_hit_predictions.csv"), index=False)
    with open(os.path.join(args.data_dir, "hitter_hit_predictions_status.json"), "w") as handle:
        json.dump({"status": status, "forecast_date": args.as_of_date.isoformat(), "generated_at_utc": generated, "n_rows": len(result), "research_only": True}, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--as-of-date", type=datetime.date.fromisoformat, default=schedule.today_local())
    args = parser.parse_args()

    wave_path = os.path.join(args.data_dir, "wave.csv")
    pave_path = os.path.join(args.data_dir, "pave.csv")
    confidence_path = os.path.join(args.data_dir, "confidence.csv")
    if not (os.path.exists(wave_path) and os.path.exists(pave_path) and os.path.exists(confidence_path)):
        print(f"No wave.csv/pave.csv/confidence.csv in {args.data_dir} - run scripts/wave.py first.")
        publish(args, status="missing_daily_inputs")
        return

    wave = pd.read_csv(wave_path)
    # pd.read_csv doesn't auto-parse dates - needed as a real datetime below
    # for the days-since-last-game recency check, not a plain string.
    wave["Last_Game_Date"] = pd.to_datetime(wave["Last_Game_Date"])
    pave = pd.read_csv(pave_path)
    confidence = pd.read_csv(confidence_path)

    try:
        schedule_df = schedule.fetch_hitter_schedule(args.as_of_date)
    except Exception as exc:
        print(f"WARNING: schedule unavailable ({exc}).")
        publish(args, status="schedule_unavailable")
        return

    if schedule_df.empty:
        publish(args, status="no_scheduled_games")
        return

    matchup_probability = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df)
    hitter_features = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability)
    hitter_features = hitter_features.merge(
        wave[["key_mlbam", "name_first", "name_last", "team", "Last_Game_Date"]], on="key_mlbam", how="left"
    )
    hitter_features = hitter_features.merge(schedule_df[["game_pk", "team", "opponent"]].drop_duplicates(), on=["game_pk", "team"], how="left", validate="many_to_one")

    hitter_features["Total_PA"] = hitter_features["PA_L"] + hitter_features["PA_R"]
    qualified = hitter_features[hitter_features["Total_PA"] >= config.BACKTEST_MIN_PLATE_APPEARANCES].copy()

    # Exclude hitters who haven't played recently (e.g. a season-long
    # injured-list stay) even though their season rates still qualify them
    # on PA alone - same gate predictions.select_picks/build_dfs_rankings.py
    # already apply, reused via the same config constant rather than
    # duplicated. A hitter with no completed event at all (Last_Game_Date
    # NaT) is kept, not excluded - can't be stale if they've never played,
    # though in practice the PA floor above already means every remaining
    # row has a real game history. Without this, a long-injured hitter
    # whose season-long WAVE/Game_Hit_Probability still looks strong could
    # top the Model Odds tab despite not having played in weeks.
    days_since_last_game = (pd.Timestamp(args.as_of_date) - qualified["Last_Game_Date"]).dt.days
    qualified = qualified[
        qualified["Last_Game_Date"].isna() | (days_since_last_game <= config.HITTER_MAX_DAYS_SINCE_LAST_GAME)
    ].drop(columns=["Last_Game_Date"])

    if qualified.empty:
        publish(args, status="no_qualified_hitters")
        return
    predictions = dfs_ml.predict_hitter_hit_probability(qualified)
    if predictions.empty:
        publish(args, status="model_unavailable")
        return

    result = qualified[["key_mlbam", "game_pk", "name_first", "name_last", "team", "opponent", "is_home"]].merge(
        predictions, on=["game_pk", "key_mlbam"], how="inner", validate="one_to_one"
    )
    result = result.sort_values("Model_Hit_Probability", ascending=False)

    publish(args, result, status="generated")
    print(f"Wrote hitter_hit_predictions.csv ({len(result)} rows) for {args.as_of_date}.")


if __name__ == "__main__":
    main()
