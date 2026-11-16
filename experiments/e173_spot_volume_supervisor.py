"""Durable E173 model/finish stages; never sends mail or changes live models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
os.chdir(REPO)
ROOT=REPO/'results/reports/crypto/e173_spot_volume'
RUNNER=REPO/'experiments/e173_spot_volume_comparison.py'
MODELS=['ridge','lgb_ktz','mlp_ktz']


def run(*args):
    subprocess.run([sys.executable,'-u',str(RUNNER),*args],check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=MODELS+['finish']);a=ap.parse_args()
    try:
        manifest=json.loads((ROOT/'feature_manifest.json').read_text())
        for path,expected in manifest['code_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, 'Runner changed after feature manifest: '+path
        assert (ROOT/'BUILD_COMPLETE.json').exists(), 'Feature build incomplete'
        if a.stage!='finish':
            run('fit','--model',a.stage,'--n-jobs','4')
            run('score','--model',a.stage)
        else:
            since=time.time()
            while not all((ROOT/f'scores_{m}.parquet').exists() for m in MODELS):
                failures=list(ROOT.glob('FAILED_*.json'))
                if failures:raise RuntimeError('Model stage failed: '+str(failures))
                if time.time()-since>48*3600:raise TimeoutError('E173 model stages exceeded48h')
                time.sleep(30)
            run('report')
    except Exception as e:
        (ROOT/f'FAILED_{a.stage}.json').write_text(json.dumps(dict(stage=a.stage,error=repr(e),time=time.time()),indent=2))
        raise

if __name__=='__main__':main()
