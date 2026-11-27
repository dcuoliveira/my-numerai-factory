"""Finish E230 after all four registered CPU fits have published their markers."""
import json
import subprocess
import time
from experiments import e230_dataset as D
from experiments import e212_volatility99_tsfm as S


def main():
    started=time.time()
    while True:
        complete=[a for a in D.ARMS if (D.ROOT/'fits'/a/'COMPLETE.json').exists()]
        if len(complete)==4:break
        progress={}
        for arm in D.ARMS:
            path=D.ROOT/'fits'/arm/'progress.json'
            if path.exists():progress[arm]=json.loads(path.read_text())
            if arm not in complete:
                state=subprocess.check_output(['systemctl','--user','show','nmr-e230_'+arm,'-p','ActiveState','--value'],text=True).strip()
                if state not in ['active','activating']:
                    S.save(D.ROOT/'status.json',dict(stage='fit_failed',arm=arm,unit_state=state,time=time.time()))
                    raise RuntimeError('Incomplete fit stopped: '+arm+' '+state)
        S.save(D.ROOT/'status.json',dict(stage='fitting',completed_arms=complete,per_asset_progress=progress,elapsed_wait_seconds=time.time()-started,time=time.time()))
        time.sleep(15)
    from experiments.e230_results import finish
    finish()

if __name__=='__main__':main()
