"""E212 sequential separate-storage CPU fitting, with explicit stream claims.

The main supervisor reads external_head_claims.json. Claims remain until outputs
are returned, even if SSH fails, to prevent a second fitting job racing them.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from experiments.e212_westeros_bridge import remote,save,digest,ROOT,REMOTE,SSH

TASKS=[(m,h) for m in ['feature_ridge','timestral','chronos2','t0'] for h in [20,60]]
PREFIX='export XDG_RUNTIME_DIR=/run/user/$(id -u); '
CANONICAL='/home/danielco/Downloads/my-numerai-agent'


def sync_outputs(model,horizon,done=False):
    name=f'{model}_t{horizon}'
    # Block files are only visible after their atomic COMPLETE marker.
    code="""from pathlib import Path
p=Path(ROOT)
for d in sorted((p/'blocks'/NAME).glob('f*')):
 if (d/'COMPLETE.json').exists():
  for f in d.iterdir():
   if f.is_file() and '.tmp' not in f.name and '.partial' not in f.name:print(f.relative_to(p))
if (p/'fits'/NAME/'WESTEROS_RETURN_READY.json').exists():
 for f in (p/'fits'/NAME).rglob('*'):
  if f.is_file():print(f.relative_to(p))
""".replace('ROOT',repr(REMOTE+'/results/reports/crypto/e212_volatility99_tsfm')).replace('NAME',repr(name))
    archive=None
    if done:
        archive_code="""import pathlib,tarfile,json,hashlib
p=pathlib.Path(ROOT); name=NAME
out=p/'archives';out.mkdir(exist_ok=True);dest=out/(name+'.tar.gz');receipt=out/(name+'.json')
if not receipt.exists():
 tmp=dest.with_suffix('.tmp')
 with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
  for d in [p/'blocks'/name,p/'cache/blocks'/name]:
   if d.exists():tar.add(d,arcname=str(d.relative_to(p)))
 tmp.replace(dest)
 h=hashlib.sha256()
 with open(dest,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 receipt.write_text(json.dumps(dict(file=str(dest.relative_to(p)),sha256=h.hexdigest(),contains=['blocks/'+name,'cache/blocks/'+name])))
print(receipt.read_text())
""".replace('ROOT',repr(REMOTE+'/results/reports/crypto/e212_volatility99_tsfm')).replace('NAME',repr(name))
        result=subprocess.check_output(SSH+['python3 -c '+shlex.quote(archive_code)],text=True,timeout=600)
        archive=json.loads(result)
    files=remote('python3 -c '+shlex.quote(code)).splitlines()
    if archive:files.extend([archive['file'],'archives/'+name+'.json'])
    files=[f for f in files if not (ROOT/f).exists()]
    if not files:return
    staging=ROOT/'westeros_fit_incoming';staging.mkdir(exist_ok=True)
    # rsync verifies each file's transmitted checksum; publish only the verified
    # completed task after predictions, model parameters and block outputs arrive.
    subprocess.run(['rsync','-az','--files-from=-','-e','ssh -o BatchMode=yes',
        'westeros:'+REMOTE+'/results/reports/crypto/e212_volatility99_tsfm/',str(staging)+'/'],
        input='\n'.join(files)+'\n',text=True,check=True,timeout=600)
    fitdir=staging/'fits'/name
    if (fitdir/'predictions.sha256.json').exists():
        expected=json.loads((fitdir/'predictions.sha256.json').read_text())['sha256']
        prediction=fitdir/'predictions.parquet'
        if not prediction.exists():prediction=ROOT/'fits'/name/'predictions.parquet'
        assert digest(prediction)==expected,'Prediction transfer hash mismatch'
    if archive:
        target=staging/archive['file']
        if not target.exists():target=ROOT/archive['file']
        assert digest(target)==archive['sha256'],'Parameter archive checksum mismatch'
    ordered=sorted(files,key=lambda x:x.endswith('COMPLETE.json'))
    for name in ordered:
        source=staging/name;destination=ROOT/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.exists():source.replace(destination)


def run():
    claims=ROOT/'external_head_claims.json'
    for model,horizon in TASKS:
        name=f'{model}_t{horizon}';job='e212_head_'+name
        if (ROOT/'fits'/name/'COMPLETE.json').exists():continue
        # The hashes recorded before transfer protect H, coverage and config.
        ready=REMOTE+'/results/reports/crypto/e212_volatility99_tsfm/westeros_fit_ready.json'
        while True:
            state_ready=remote('test ! -f '+ready+' || cat '+ready).strip()
            if state_ready and model in json.loads(state_ready)['models']:break
            time.sleep(20)
        unit='nmr-e212_westeros_head_'+name
        state=remote(PREFIX+'systemctl --user is-active '+unit+' || true').strip()
        completed=REMOTE+'/results/reports/crypto/e212_volatility99_tsfm/fits/'+name+'/WESTEROS_RETURN_READY.json'
        if state not in ['active','activating'] and not remote('test ! -f '+completed+' || echo complete').strip():
            script=REMOTE+'/experiments/e212_westeros_fit_worker.sh'
            remote(PREFIX+'systemctl --user reset-failed '+unit+' 2>/dev/null || true; '+
                'systemd-run --user --unit='+unit+' --property=MemoryMax=32G '+
                'taskset -c 0-3 /bin/bash '+script+' '+model+' '+str(horizon))
        while True:
            state=remote(PREFIX+'systemctl --user show '+unit+' -p ActiveState -p Result')
            done=bool(remote('test ! -f '+completed+' || echo complete').strip())
            sync_outputs(model,horizon,done)
            save(ROOT/'westeros_fit_progress.json',dict(model=model,target=horizon,remote_state=state,complete=done,time=time.time()))
            if done:
                receipt=json.loads(remote('cat '+REMOTE+'/results/reports/crypto/e212_volatility99_tsfm/fits/'+name+'/predictions.sha256.json'))
                assert digest(ROOT/'fits'/name/'predictions.parquet')==receipt['sha256']
                current=json.loads(claims.read_text());current['jobs']=[x for x in current['jobs'] if x!=job];save(claims,current)
                print('RETURNED',name,flush=True);break
            if 'ActiveState=failed' in state or 'ActiveState=inactive' in state:
                raise RuntimeError('Remote fitting did not complete: '+name+' '+state)
            time.sleep(30)
    save(ROOT/'westeros_fit_COMPLETE.json',dict(time=time.time()))


if __name__=='__main__':
    try:run()
    except Exception as error:
        save(ROOT/'westeros_fit_FAILED.json',dict(error=str(error),time=time.time(),claims_retained=True));raise
