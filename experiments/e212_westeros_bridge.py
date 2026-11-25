"""E212 Timestral remote-only extraction bridge; no shared filesystem assumed.

Launch with the same managed extraction unit names on hamilton, so the existing
E212 supervisor sees these streams as active instead of duplicating their work.
The frozen extractor itself remains unchanged. Only complete, fingerprint- and
hash-verified per-asset pairs become visible to the main study.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO/'results/reports/crypto/e212_volatility99_tsfm'
REMOTE = '/home/danielco/Downloads/my-numerai-factory'
SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','westeros']


def remote(command):
    return subprocess.check_output(SSH+[command],text=True,timeout=40)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{socket.gethostname()}_{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def validate_manifest(expected, actual):
    if actual['fingerprint']!=expected['fingerprint']:
        raise RuntimeError('Remote extractor differs from frozen stream; no cache import permitted')
    if actual['package_versions']!=expected['package_versions']:
        raise RuntimeError('Remote model package versions differ')
    if actual['device']!='cuda:0':raise RuntimeError('GPU extraction required')


def import_pairs(directory,source,expected):
    # Compute remote hashes only for stable complete pairs; never import progress/temp files.
    code="""import pathlib,json,hashlib
p=pathlib.Path(SOURCE);result={}
for meta in p.glob('asset_*.json'):
 if not meta.stem.removeprefix('asset_').isdigit():continue
 data=meta.with_suffix('.npz')
 if not data.exists():continue
 for f in [meta,data]:result[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
print(json.dumps(result))
""".replace('SOURCE',repr(source))
    hashes=json.loads(remote('python3 -c '+shlex.quote(code)))
    wanted={n:h for n,h in hashes.items() if not (directory/n).exists()}
    if not wanted:return len(hashes)//2
    staging=directory/'westeros_incoming';staging.mkdir(parents=True,exist_ok=True)
    # Fetch both halves of each pair even when recovering an interrupted import.
    assets={n.split('.')[0] for n in wanted}
    names=sorted(n for n in hashes if n.split('.')[0] in assets)
    subprocess.run(['rsync','-az','--files-from=-','-e','ssh -o BatchMode=yes',
                    'westeros:'+source+'/',str(staging)+'/'],input='\n'.join(names)+'\n',text=True,check=True,timeout=180)
    import numpy as np
    import pandas as pd
    panel=pd.read_parquet(ROOT/'panel.parquet',columns=['row','ucid'])
    identity=panel.set_index('row').ucid
    for name in names:
        if digest(staging/name)!=hashes[name]:raise RuntimeError('Transfer hash mismatch: '+name)
    for asset in sorted(assets):
        data=staging/(asset+'.npz');meta=staging/(asset+'.json')
        info=json.loads(meta.read_text())
        assert info['fingerprint']==expected['fingerprint'] and info['host']=='westeros'
        with np.load(data) as z:
            rows=z['rows'];states=z['states']
            assert states.shape==(len(rows),expected['width']) and len(rows)==info['rows']
            assert len(np.unique(rows))==len(rows) and np.isfinite(states).all()
            assert states.dtype==np.float16 and identity.loc[rows].eq(int(asset[6:])).all()
        for f in [data,meta]:
            dest=directory/f.name
            if dest.exists():
                if digest(dest)!=hashes[f.name]:raise RuntimeError('Existing cache differs: '+f.name)
                f.unlink()
            else:f.replace(dest)
    return len(hashes)//2


def run(h):
    directory=ROOT/f'cache/t{h}/embeddings/timestral'
    source=REMOTE+f'/results/reports/crypto/e212_volatility99_tsfm/cache/t{h}/embeddings/timestral'
    expected=json.loads((directory/'extractor.json').read_text())
    unit=f'nmr-e212_westeros_timestral_t{h}'
    prefix='export XDG_RUNTIME_DIR=/run/user/$(id -u); '
    active=remote(prefix+'systemctl --user is-active '+unit+' || true').strip()
    if active!='active':
        # The remote GPU lock serializes the two horizons, including smoke approval.
        command=(prefix+'systemctl --user reset-failed '+unit+' 2>/dev/null || true; '+
                 'systemd-run --user --unit='+unit+' --property=CPUQuota=200% --property=MemoryMax=24G '+
                 'taskset -c 0,1 /bin/bash '+REMOTE+'/experiments/e212_westeros_worker.sh '+str(h))
        remote(command)
    approved=False
    while True:
        snapshot=remote(prefix+'systemctl --user show '+unit+' -p ActiveState -p Result')
        smoke=remote('test ! -f '+source+'/SMOKE_COMPLETE.json || cat '+source+'/SMOKE_COMPLETE.json')
        if smoke and not approved:
            record=json.loads(smoke);actual=json.loads(remote('cat '+source+'/extractor.json'))
            validate_manifest(expected,actual)
            assert record['passed'] and record['host']=='westeros'
            audit=json.loads(remote('cat '+source+'/audit.json'))
            assert audit['passed'] and audit['batch_verified']
            save(directory/'extractor_westeros.json',actual)
            save(directory/'audit_westeros_0_cuda_0.json',audit)
            remote('printf approved > '+source+'/WESTEROS_APPROVED.json')
            approved=True
        imported=import_pairs(directory,source,expected) if approved else 0
        complete=remote('test ! -f '+source+'/WESTEROS_COMPLETE || echo complete').strip()=='complete'
        save(directory/'westeros_bridge.json',dict(host='westeros',unit=unit,approved=approved,
             completed_assets=imported,complete=complete,remote_state=snapshot,time=time.time()))
        print(h,'westeros',imported,'complete',complete,flush=True)
        if complete:
            # Import all pairs before releasing this managed stream reservation.
            p=__import__('pandas').read_parquet(ROOT/'panel.parquet',columns=['ucid','covered'])
            wanted=p.groupby('ucid').covered.any()
            assert imported==int(wanted.sum()),'Remote stream not complete'
            return
        if 'ActiveState=failed' in snapshot or 'ActiveState=inactive' in snapshot:
            raise RuntimeError('Westeros worker stopped before completion: '+snapshot)
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',required=True,type=int,choices=[20,60])
    target=p.parse_args().target
    try:
        run(target)
    except Exception as error:
        save(ROOT/f'cache/t{target}/embeddings/timestral/westeros_bridge_FAILED.json',
             dict(error=str(error),time=time.time()))
        try:remote('export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user stop nmr-e212_westeros_timestral_t'+str(target))
        finally:raise
