"""Train a separate development artifact using already-inspected pre-protocol dates.

Reads explicit files only. Writes exclusively to a new --output-dir. Never
updates production models, published predictions, registered policies or ledgers.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import pandas as pd
from mlb_metrics import config, hit_prop_development as dev


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "schedule", "boxscore-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "model_version": config.HIT_PROP_DEV_MODEL_VERSION,
              "implementation_sha256": sha(dev.__file__), "config_sha256": sha(config.__file__),
              "research_only": True, "production_serving_allowed": False, "real_money_ready": False,
              "prediction_time_availability_certified": False,
              "features": list(config.HIT_PROP_DEV_FEATURES),
              "development_end_exclusive": config.HIT_PROP_DEV_END_EXCLUSIVE,
              "limitations": ["Already-inspected development dates, not a pristine holdout",
                  "Historical morning candidate universe differs from venue-listed, confirmed-lineup opportunities",
                  "Starter-dependent matchup features excluded; historical rate features still use reconstructed past batting order",
                  "Official labels obtained after outcomes; not pregame lineup snapshots",
                  "Historical timestamps are reconstructed date cutoffs; UTC midnight does not certify prior-day game completion or provider availability",
                  "No matched executable historical prices, fees, liquidity, returns or profitability estimate",
                  "Postseason and interrupted-game nonbinary settlement are not modeled",
                  "Last-fair-market-price distribution remains unknown; only payout bounds available"]}
    (out/'report.json').write_text(json.dumps(report | {"status": "started"}, indent=2))
    try:
        report.update(input_sha256=sha(args.input), schedule_sha256=sha(args.schedule))
        raw = pd.read_csv(args.input)
        frame, excluded, hashes = dev.prepare_development_frame(raw, json.loads(Path(args.schedule).read_text()), args.boxscore_dir)
        excluded.to_csv(out/'exclusions.csv', index=False)
        frame.to_csv(out/'official_labeled_development.csv', index=False)
        (out/'boxscore_hashes.json').write_text(json.dumps(hashes, sort_keys=True, indent=2))
        report.update(n_input_rows=len(raw), n_included_rows=len(frame), n_excluded_rows=len(excluded),
            exclusion_reasons=excluded.label_reason.value_counts().to_dict(),
            class_counts={dev.CLASS_NAMES[int(k)]:int(v) for k,v in frame.target_state.value_counts().items()})
        if {'Started','Plate_Appearances','Hits'} <= set(raw):
            old = raw[dev.KEYS].copy()
            qualified = raw.Started.eq(1) & raw.Plate_Appearances.gt(0)
            old['old_state'] = 0
            old.loc[qualified, 'old_state'] = 1
            old.loc[qualified & raw.Hits.gt(0), 'old_state'] = 2
            aligned = frame.merge(old, on=dev.KEYS, validate='one_to_one')
            report['n_corrected_state_labels'] = int(aligned.target_state.ne(aligned.old_state).sum())
            aligned[aligned.target_state.ne(aligned.old_state)][dev.KEYS+['old_state','target_state','label_reason']].to_csv(out/'label_corrections.csv',index=False)
        predictions, evaluation = dev.chronological_development(frame)
        report['evaluation'] = evaluation
        report['status'] = evaluation['status']
        if not predictions.empty:
            predictions.to_csv(out/'forward_development_predictions.csv',index=False)
            artifact = out/'development_only.joblib'
            joblib.dump({'model':dev.fit_model(frame), 'features':config.HIT_PROP_DEV_FEATURES,
                         'model_version':config.HIT_PROP_DEV_MODEL_VERSION, 'research_only':True,
                         'production_serving_allowed':False,'prediction_time_availability_certified':False,
                         'training_end_date':frame.date.max(),
                         'source_hash':report['input_sha256']},artifact)
            report['artifact_sha256'] = sha(artifact)
            report['artifact_path'] = str(artifact)
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(report,indent=2,allow_nan=False))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        raise


if __name__ == '__main__':
    main()
