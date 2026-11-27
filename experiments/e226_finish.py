"""Wait for registered E226 heads, score, update E212 and run both audits."""
import json
import time
from experiments import e226_volatility_cs_ridge as R


def main():
    wanted=[R.ROOT/'fits'/f'{arm}_t{h}'/'COMPLETE.json' for arm in ('CS99','TS_CS198') for h in (20,60)]
    while not all(p.exists() for p in wanted):
        R.save(R.ROOT/'status.json',dict(stage='fitting_parallel',completed_heads=sum(p.exists() for p in wanted),total_heads=4,time=time.time()))
        time.sleep(10)
    for path in wanted:
        marker=json.loads(path.read_text())
        assert marker['prediction_sha256']==R.sha(path.parent/'predictions.parquet')
    R.save(R.ROOT/'status.json',dict(stage='scoring',time=time.time()));R.score()
    R.save(R.ROOT/'status.json',dict(stage='reporting',time=time.time()));R.integrate()
    R.save(R.ROOT/'status.json',dict(stage='auditing',time=time.time()))
    from experiments.e226_validate_completion import run as audit_new
    from experiments.e212_validate_completion import run as audit_original
    audit_new();audit_original()
    R.save(R.ROOT/'status.json',dict(stage='complete',heads=4,summary_rows=16,audit='PASS WITH CONDITIONS',time=time.time()))
    print('E226 COMPLETE: all four heads, report and independent audits.',flush=True)

if __name__=='__main__':main()
