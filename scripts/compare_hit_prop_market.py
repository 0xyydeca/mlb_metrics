"""Audit and compare forecasts against saved quotes without opening outcomes.

Reads explicit inputs and writes only to a new output directory. No captures,
training, policy changes, orders, formal outcome scoring or production writes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from mlb_metrics.hit_prop_market_benchmark import compare_forecasts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forecasts',required=True)
    p.add_argument('--quotes',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    report={'status':'started','real_money_ready':False,'outcomes_opened':False}
    try:
        for label,path in [('forecasts',args.forecasts),('quotes',args.quotes)]:
            report[label+'_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        rows,result=compare_forecasts(pd.read_csv(args.forecasts),pd.read_csv(args.quotes))
        report.update(result)
        rows.to_csv(out/'comparisons.csv',index=False)
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report,indent=2))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        raise


if __name__=='__main__':main()
