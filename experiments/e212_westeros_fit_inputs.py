"""Transfer E212 frozen H to independent storage and release each verified model."""
import json
from pathlib import Path
import shlex
import subprocess
import time
from experiments.e212_westeros_bridge import ROOT,REMOTE,remote,save


def run():
    manifest=Path('/tmp/e212_fit_H_hashes.json')
    while not manifest.exists():time.sleep(5)
    groups=json.loads(manifest.read_text())
    save(ROOT/'westeros_fit_H_hashes.json',groups)
    for model,hashes in groups.items():
        print('TRANSFER',model,flush=True)
        subprocess.run(['rsync','-azS','--files-from=-','-e','ssh -o BatchMode=yes','./',
            'westeros:'+REMOTE+'/'],input='\n'.join(hashes)+'\n',text=True,check=True)
        code="""import hashlib,json,pathlib
repo=pathlib.Path(REPO)
hashes=json.loads(HASHES)
for name,expected in hashes.items():
 h=hashlib.sha256()
 with open(repo/name,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 assert h.hexdigest()==expected,name
p=repo/'results/reports/crypto/e212_volatility99_tsfm/westeros_fit_ready.json'
info=json.loads(p.read_text())
if MODEL not in info['models']:info['models'].append(MODEL)
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(info));tmp.replace(p)
print('Hash-verified model ready:',MODEL)
""".replace('REPO',repr(REMOTE)).replace('HASHES',repr(json.dumps(hashes))).replace('MODEL',repr(model))
        print(remote('python3 -c '+shlex.quote(code)),flush=True)
    save(ROOT/'westeros_fit_inputs_COMPLETE.json',dict(models=list(groups),time=time.time()))


if __name__=='__main__':
    try:run()
    except Exception as error:
        save(ROOT/'westeros_fit_inputs_FAILED.json',dict(error=str(error),time=time.time()));raise
