"""E218: independent native KTZ heads on the screenshot's seven explicit C inputs."""
import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import shutil
import smtplib
import socket
import subprocess
import sys
import time
import traceback
from email.message import EmailMessage
from email.utils import getaddresses, make_msgid

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from src.analysis import asset_ktz_states as K
from src.cluster import budget, hosts, manager
from src.live.submit import rank_normalise

ROOT = REPO/'results/reports/crypto/e218_asset_ktz_states'
SCR = Path('/scratch/danielco/e218_asset_ktz_states')
SOURCE = REPO/'results/reports/crypto/e206_momentum3_tsfm'
SOURCE_SCR = Path('/scratch/danielco/e206_momentum3_tsfm')
PY = '/home/danielco/miniconda3/bin/python3'
FAMILIES = list(K.ALIASES)
save, parquet = E.save, E.parquet


def plan():
    return json.loads((ROOT/'plan.json').read_text())


def prepare():
    if (ROOT/'PREPARED.json').exists(): return
    ROOT.mkdir(parents=True, exist_ok=True); SCR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(SOURCE/'ranked_results.csv')
    wanted = {'chronos2':['rolling','cross_section'], 't0':['rolling','cross_section'],
              'timestral':['robust','cross_section'], 'tabpfnts':['asset']}
    table = table[table.universe.eq('common') & table.side.eq('neutral') & table.design.eq('C') &
                  table.apply(lambda r:r.standardization in wanted.get(r.model,[]),axis=1)].sort_values(['model','standardization'])
    assert len(table)==7 and table.tag.is_unique
    configs=[]
    for i,row in enumerate(table.itertuples()):
        directory=SOURCE/'fits'/row.tag
        choice=json.loads((directory/'validation_selection.json').read_text())
        parameters=json.loads((directory/'ridge_parameters.json').read_text())
        configs.append(dict(id=f'c{i:02d}',model=row.model,norm=row.standardization,source=row.tag,
             window=int(choice['window']),vocabulary=parameters['design']['vocabulary'],
             source_parameters_hash=E.sha(directory/'ridge_parameters.json')))
    save(ROOT/'plan.json',configs)
    save(ROOT/'candidates.json',{family:K.candidates(family) for family in FAMILIES})
    p=pd.read_parquet(SOURCE/'panel.parquet')
    assert np.array_equal(p.row,np.arange(len(p))) and not p[['ucid','date']].duplicated().any()
    for mask in ['train','validation','final_fit']:
        assert np.isfinite(p.loc[p[mask] & np.load(SOURCE_SCR/'common_rows.npy'),E.TARGET]).all()
    assert p.date.max()<pd.Timestamp('2026-02-20')
    parquet(ROOT/'panel.parquet',p)
    save(ROOT/'PREPARED.json',dict(experiment='E218',configs=configs,source_panel_hash=E.sha(SOURCE/'panel.parquet'),
         training='Original E206 target20 splits and scalers/vocabulary/selected windows; fresh separate native KTZ estimator per asset. Four frozen grid candidates, validation MSE, independent final refit.',
         evaluation='Common matched asset/date keys, same297-date E206 score period;7 ridge references plus21 new portfolios, all2..5 subsets. Same-history ensemble selection exploratory.',
         mlp='Native sklearn initialization/optimizer; identity network inference saved as mathematically exact affine composition after numerical parity check.',
         resource_share=.60, default_share=budget.SHARE, source_vintage_limit='Inherits E205/E206 vendor identity/vintage and checkpoint pretraining limitations; no live or actual-price-return claims.'))


def build(config_id):
    c=next(c for c in plan() if c['id']==config_id)
    root=ROOT/'inputs'/config_id; scratch=SCR/'inputs'/config_id
    if (root/'COMPLETE.json').exists(): return
    root.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    for name in ['common_rows.npy','embeddings']:
        if not (scratch/name).exists():(scratch/name).symlink_to(SOURCE_SCR/name,target_is_directory=name=='embeddings')
    E.ROOT,E.SCR=SOURCE,scratch
    p=pd.read_parquet(ROOT/'panel.parquet')
    assets=p.ucid.astype(str).to_numpy(); steps=p.step.to_numpy()
    dimensions={}
    for phase in ['selection','final']:
        h=E.normalized(c['model'],c['norm'],phase,c['window'])
        required=(int(p.step.max())+1)*len(c['vocabulary'])*h.shape[1]*4
        if shutil.disk_usage(scratch).free < required+2_000_000_000:
            raise RuntimeError('Insufficient scratch capacity for exact C matrix')
        valid=np.isfinite(h).all(axis=1)
        phi=K.materialize(h,p.ucid.to_numpy(),p.step.to_numpy(),valid,c['vocabulary'],int(p.step.max()+1),scratch/f'{phase}.npy')
        np.save(scratch/f'{phase}_valid.npy',valid)
        dimensions[phase]=list(phi.shape)
        # Independent block-sum check on a few snapshot pairs.
        for a,b in [(int(p.step.max())-1,int(p.step.max())),(300,301)]:
            expected=0.
            for asset in c['vocabulary']:
                ia=np.flatnonzero((assets==str(asset)) & (steps==a) & valid)
                ib=np.flatnonzero((assets==str(asset)) & (steps==b) & valid)
                if len(ia) and len(ib): expected+=np.dot(h[ia[0]].astype(float),h[ib[0]].astype(float))
            np.testing.assert_allclose(np.dot(phi[a].astype(float),phi[b].astype(float)),expected,rtol=1e-5,atol=1e-5)
        del phi,h
    # Keep reference scalers/manifest, release temporary normalized flat arrays.
    for f in (scratch/'normalizations').glob('*/*/*/H.npy'):f.unlink()
    save(root/'COMPLETE.json',dict(config=c,dimensions=dimensions,time=time.time()))


def fit(config_id,family):
    c=next(c for c in plan() if c['id']==config_id);tag=f'{config_id}_{family}'
    root=ROOT/'fits'/tag
    if (root/'COMPLETE.json').exists(): return
    root.mkdir(parents=True,exist_ok=True)
    p=pd.read_parquet(ROOT/'panel.parquet');common=np.load(SOURCE_SCR/'common_rows.npy')
    scratch=SCR/'inputs'/config_id
    xs=np.load(scratch/'selection.npy',mmap_mode='r');xf=np.load(scratch/'final.npy',mmap_mode='r')
    vs=np.load(scratch/'selection_valid.npy');vf=np.load(scratch/'final_valid.npy')
    ridge=pd.read_parquet(SOURCE/'fits'/c['source']/'predictions.parquet')
    allowed=set(ridge.loc[ridge.prediction.notna(),'row'])
    out=p.loc[p.test,['row','ucid','symbol','date',E.TARGET,'meta_model']].copy();out['prediction']=np.nan
    records=[];failures=[];grid=json.loads((ROOT/'candidates.json').read_text())[family]
    for asset,g in p.groupby('ucid',sort=True):
        idx=g.index.to_numpy()
        tr=idx[common[idx]&p.train.to_numpy()[idx]&vs[idx]]
        va=idx[common[idx]&p.validation.to_numpy()[idx]&vs[idx]]
        ft=idx[common[idx]&p.final_fit.to_numpy()[idx]&vf[idx]]
        te=idx[p.row.iloc[idx].isin(allowed).to_numpy() & vf[idx]]
        if len(tr)<60 or len(va)<20 or not len(te):continue
        record_path=root/'assets'/f'{asset}.json';pred_path=root/'assets'/f'{asset}.parquet'
        if record_path.exists() and pred_path.exists():
            data=pd.read_parquet(pred_path);record=json.loads(record_path.read_text())
        else:
            started=time.time()
            try:
                prediction,record=K.fit_asset(family,grid,xs[p.step.iloc[tr]],p.loc[tr,E.TARGET].to_numpy(),
                    xs[p.step.iloc[va]],p.loc[va,E.TARGET].to_numpy(),xf[p.step.iloc[ft]],p.loc[ft,E.TARGET].to_numpy(),
                    xf[p.step.iloc[te]],int(os.getenv('NMR_SLOT_CORES','4')),
                    SCR/'models'/tag/f'asset_{asset}.joblib')
                record.update(asset=str(asset),training_rows=p.row.iloc[tr].tolist(),validation_rows=p.row.iloc[va].tolist(),final_rows=p.row.iloc[ft].tolist())
                data=pd.DataFrame(dict(row=p.row.iloc[te].to_numpy(),prediction=prediction))
                parquet(pred_path,data);save(record_path,record)
            except Exception as error:
                failure=dict(asset=str(asset),error=str(error),traceback=traceback.format_exc(),seconds=time.time()-started)
                save(root/'asset_failures'/f'{asset}.json',failure);failures.append(failure);continue
        mapping=data.set_index('row').prediction
        selected=out.row.isin(mapping.index);out.loc[selected,'prediction']=out.loc[selected,'row'].map(mapping)
        records.append(record)
        save(root/'progress.json',dict(completed_assets=len(records),failed_assets=len(failures),latest_asset=str(asset),latest_seconds=record.get('seconds'),time=time.time()))
        print('ASSET',tag,asset,record.get('seconds'),flush=True)
    if not records: raise RuntimeError('No successful assets for '+tag)
    parquet(root/'predictions.parquet',out)
    if failures:raise RuntimeError(f'{len(failures)} asset fits failed; completed assets are checkpointed for retry')
    save(root/'COMPLETE.json',dict(task=dict(tag=tag,model=family),assets=len(records),failed_assets=len(failures),predictions=int(out.prediction.notna().sum()),time=time.time()))


def email(event,pdf=None):
    receipt=ROOT/f'email_{event}.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('smtp_accepted'):return
    from src.live.env import load_env
    from src.live.notify import _config
    load_env(required=());config=_config()
    if config is None:raise RuntimeError('Notification email is not configured')
    message=EmailMessage();message['From']=config['user'];message['To']=config['to'];message['Message-ID']=make_msgid()
    message['Subject']=f'[numerai-crypto] E218 {event} — per-asset KTZ on seven TSFM state inputs'
    message.set_content('E218 '+event+'. Seven frozen E206 input configurations;21 fresh per-asset KTZ portfolios plus7 ridge references. '
        'One-off CPU share60%, restored on stop/completion. Full matched historical performance and ensemble search; no live submissions. '+
        ('The completed reporting PDF is attached.' if pdf else 'Launch confirmed; configuration preparation/training is now scheduled.'))
    if pdf:message.add_attachment(Path(pdf).read_bytes(),maintype='application',subtype='pdf',filename=Path(pdf).name)
    if len(message.as_bytes())>24_000_000:raise RuntimeError('PDF exceeds email attachment limit')
    recipients=[address for _,address in getaddresses([config['to']])]
    with smtplib.SMTP(config['host'],config['port'],timeout=60) as server:
        server.starttls();server.login(config['user'],config['password'])
        refused=server.send_message(message,from_addr=config['user'],to_addrs=recipients)
        if refused:raise RuntimeError('Notification recipient refused')
    save(receipt,dict(smtp_accepted=True,time=time.time(),message_id=message['Message-ID'],pdf_hash=E.sha(pdf) if pdf else None))


def finish():
    from src.analysis.e218_ktz_report import analyze
    if not (ROOT/'ANALYSIS_COMPLETE.json').exists():analyze(ROOT,SCR,SOURCE,plan())
    email('completed',ROOT/'E218_asset_ktz_states.pdf')
    save(ROOT/'COMPLETE.json',dict(status='success',new_models=21,ridge_controls=7,report=str(ROOT/'E218_asset_ktz_states.pdf'),time=time.time()))


def supervise():
    prepare()
    file=ROOT/'supervisor.json'
    state=json.loads(file.read_text()) if file.exists() else dict(owned={},attempts={},launches=[])
    def interrupt(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    save(ROOT/'resource_lease.json',dict(active_share=.60,default_share=budget.SHARE,status='active',global_default_changed=False))
    try:
        while not (ROOT/'COMPLETE.json').exists():
            probes=hosts.reachable(hosts.probe_all(timeout=3));snapshots=manager.status(names=[h.name for h in probes],timeout=3)
            active,_=E.unit_states(snapshots)
            for unit in list(state['owned']):
                if unit not in active and state['owned'][unit]['host'] in snapshots:del state['owned'][unit]
            jobs=[]
            for c in plan():
                if not (ROOT/'inputs'/c['id']/'COMPLETE.json').exists():
                    jobs.append(budget.Job(f'e218_build_{c["id"]}',f'{PY} -u experiments/e218_asset_ktz_states.py build --config {c["id"]}',cores=4,ram_gb=16))
                else:
                    for family in FAMILIES:
                        if not (ROOT/'fits'/f'{c["id"]}_{family}'/'COMPLETE.json').exists():
                            jobs.append(budget.Job(f'e218_{c["id"]}_{family}',f'{PY} -u experiments/e218_asset_ktz_states.py fit --config {c["id"]} --family {family}',cores=4,ram_gb=24))
            if not jobs:
                save(ROOT/'status.json',dict(stage='scoring_and_reporting',time=time.time()));finish();break
            pending=[]
            for job in jobs:
                if 'nmr-'+job.name in active or 'nmr-'+job.name in state['owned']:continue
                if state['attempts'].get(job.name,0)>=3:raise RuntimeError('Three attempts exhausted: '+job.name)
                pending.append(job)
            placements,unplaced=budget.distribute(pending,probes,share=.60)
            for placement in placements:
                subprocess.run(['ssh','-o','BatchMode=yes',placement.host,'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user reset-failed '+placement.unit],capture_output=True,timeout=15)
                ok,message=manager.launch(placement)
                if ok:
                    record=dict(unit=placement.unit,host=placement.host,cpus=list(placement.cpus))
                    state['owned'][placement.unit]=record;state['launches'].append(record)
                    state['attempts'][placement.job.name]=state['attempts'].get(placement.job.name,0)+1
                    save(file,state)
                print('LAUNCH',ok,message,flush=True)
            state['updated']=time.time();state['waiting']=[dict(name=j.name,reason=why) for j,why in unplaced];save(file,state)
            save(ROOT/'status.json',dict(stage='building_inputs_and_fitting',prepared_configs=len(list((ROOT/'inputs').glob('*/COMPLETE.json'))),completed_models=len(list((ROOT/'fits').glob('*/COMPLETE.json'))),time=time.time()))
            if state['launches']:
                try:email('started')
                except Exception as error:save(ROOT/'email_start_error.json',dict(error=str(error),time=time.time()))
            for c in plan():
                if all((ROOT/'fits'/f'{c["id"]}_{f}'/'COMPLETE.json').exists() for f in FAMILIES):
                    for cache in (SCR/'inputs'/c['id']).glob('*.npy'):
                        if cache.name!='common_rows.npy':cache.unlink()
            time.sleep(30)
    except Exception as error:
        save(ROOT/'FAILED.json',dict(error=str(error),traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        for host in {v['host'] for v in state['launches']}:
            manager.stop(host,list({v['unit'] for v in state['launches'] if v['host']==host}),timeout=20)
        save(ROOT/'resource_lease.json',dict(active_share=budget.SHARE,default_share=budget.SHARE,status='restored',global_default_changed=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['supervise','prepare','build','fit','finish'])
    ap.add_argument('--config');ap.add_argument('--family',choices=FAMILIES);args=ap.parse_args()
    if args.command=='build':build(args.config)
    elif args.command=='fit':fit(args.config,args.family)
    else:globals()[args.command]()
