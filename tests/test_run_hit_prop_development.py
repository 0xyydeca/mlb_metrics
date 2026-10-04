import json
from pathlib import Path
import subprocess
import sys


def test_development_cli_records_failure_and_never_overwrites(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/run_hit_prop_development.py'
    output=tmp_path/'run'
    command=[sys.executable,str(script),'--input',str(tmp_path/'missing.csv'),
             '--schedule',str(tmp_path/'missing.json'),'--boxscore-dir',str(tmp_path), '--output-dir',str(output)]
    first=subprocess.run(command,capture_output=True)
    assert first.returncode != 0
    original=(output/'report.json').read_bytes()
    report=json.loads(original)
    assert report['status']=='failed' and not report['real_money_ready']
    second=subprocess.run(command,capture_output=True)
    assert second.returncode != 0 and (output/'report.json').read_bytes()==original
