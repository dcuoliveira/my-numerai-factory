"""E220: four JA H inputs, H-only versus H+own momentum, native per-asset KTZ heads."""
import os,sys,json,time,shutil,signal,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'experiments')]
from experiments import e218_asset_ktz_states as B
from src.analysis import asset_ktz_states as K
from src.analysis.wide_tabnet import WideTabNetSkRegressor
from src.cluster import budget,hosts,manager
E=B.E
ROOT=REPO/'results/reports/crypto/e220_ja_heads';SCR=Path('/scratch/danielco/e220_ja_heads')
OLD=B.ROOT;OLD_SCR=B.SCR
FAMILIES=['mlp_ktz','lgbm_ktz','tabnet_ktz']
ORIGINAL_WRAPPER=K.wrapper


def wrapper(family,cores=1):
    if family!='tabnet_ktz':return ORIGINAL_WRAPPER(family,cores)
    from src.prediction.models import MODELS
    w=MODELS['tabnet_ktz'](task='regression',seed=0,n_jobs=cores)
    old=w.ModelClass.steps[-1][1]
    w.ModelClass.steps[-1]=('net',WideTabNetSkRegressor(**old.get_params()))
    return w


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);SCR.mkdir(parents=True,exist_ok=True)
    if (ROOT/'PREPARED.json').exists():return
    configs=[c for c in json.loads((OLD/'plan.json').read_text()) if c['id'] in ['c01','c02','c03','c05']]
    assert len(configs)==4
    E.save(ROOT/'plan.json',configs)
    shutil.copy2(OLD/'panel.parquet',ROOT/'panel.parquet')
    grids={f:K.candidates(f) for f in FAMILIES if f!='tabnet_ktz'}
    grid=wrapper('tabnet_ktz').param_grid
    first={k:list(v)[0] for k,v in grid[0].items()}
    choices=[K.native(first)]
    for point in ParameterSampler(grid,n_iter=20,random_state=220):
        point=K.native(point)
        if point not in choices:choices.append(point)
        if len(choices)==4:break
    grids['tabnet_ktz']=choices;E.save(ROOT/'candidates.json',grids)
    E.save(ROOT/'PREPARED.json',dict(experiment='E220',source_panel_hash=E.sha(OLD/'panel.parquet'),
      inputs='Four JA design-C H configurations; combined variant appends own asset 33 momentum features, not other assets raw features',
      fitting='Same E218 target20 splits, label wall, four frozen candidates/family; per-asset validation MSE then independent final fit',
      tabnet='Unchanged KTZ axes, sparse identity storage; batch capped64 and at greatest power of2 <= available rows; virtual batch capped at effective batch. All epochs train; no validation early stopping.',
      reuse='Eight H-only sklearn MLP/LightGBM streams reuse identical E218 fits; no new residualization or ensembles requested',
      scope='24 portfolios,16 fresh fits,8 references. Historical research; inherited vintage/pretraining limits; no live submission'))


def inputs(c):
    root=ROOT/'inputs'/c['id'];scratch=SCR/'inputs'/c['id']
    if (root/'COMPLETE.json').exists():return
    root.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    source=OLD_SCR/'inputs'/c['id']
    names=['selection.npy','final.npy','selection_valid.npy','final_valid.npy']
    if all((source/n).exists() for n in names):
        for n in names:
            destination=scratch/n;temporary=destination.with_suffix('.copying')
            shutil.copyfile(source/n,temporary)
            assert E.sha(temporary)==E.sha(source/n)
            temporary.replace(destination)
        E.save(root/'COMPLETE.json',dict(source=str(source),verified_hashes={n:E.sha(scratch/n) for n in names}))
    else:
        B.ROOT,B.SCR=ROOT,SCR;B.build(c['id'])


def fit(config_id,variant,family):
    c=next(v for v in json.loads((ROOT/'plan.json').read_text()) if v['id']==config_id)
    tag=f'{config_id}_{variant}_{family}';root=ROOT/'fits'/tag
    if (root/'COMPLETE.json').exists():return
    root.mkdir(parents=True,exist_ok=True)
    inputs(c);K.wrapper=wrapper
    p=pd.read_parquet(ROOT/'panel.parquet');common=np.load(B.SOURCE_SCR/'common_rows.npy')
    directory=SCR/'inputs'/config_id
    xs=np.load(directory/'selection.npy',mmap_mode='r');xf=np.load(directory/'final.npy',mmap_mode='r')
    vs=np.load(directory/'selection_valid.npy');vf=np.load(directory/'final_valid.npy')
    ridge=pd.read_parquet(B.SOURCE/'fits'/c['source']/'predictions.parquet');allowed=set(ridge.loc[ridge.prediction.notna(),'row'])
    features=[name for name in p if name.startswith('feature_')];assert len(features)==33
    out=p.loc[p.test,['row','ucid','symbol','date',E.TARGET,'meta_model']].copy();out['prediction']=np.nan
    grid=json.loads((ROOT/'candidates.json').read_text())[family];records=[];failures=[]
    def matrix(cache,idx):
        x=np.asarray(cache[p.step.iloc[idx]],dtype=np.float32)
        if variant=='Hmom':
            mom=p.loc[idx,features].to_numpy(np.float32);assert np.isfinite(mom).all()
            x=np.column_stack([x,mom])
        return x
    for asset,g in p.groupby('ucid',sort=True):
        idx=g.index.to_numpy()
        tr=idx[common[idx]&p.train.to_numpy()[idx]&vs[idx]];va=idx[common[idx]&p.validation.to_numpy()[idx]&vs[idx]]
        ft=idx[common[idx]&p.final_fit.to_numpy()[idx]&vf[idx]];te=idx[p.row.iloc[idx].isin(allowed).to_numpy()&vf[idx]]
        if len(tr)<60 or len(va)<20 or not len(te):continue
        record_path=root/'assets'/f'{asset}.json';pred_path=root/'assets'/f'{asset}.parquet'
        if record_path.exists() and pred_path.exists():record=json.loads(record_path.read_text());data=pd.read_parquet(pred_path)
        else:
            try:
                pred,record=K.fit_asset(family,grid,matrix(xs,tr),p.loc[tr,E.TARGET].to_numpy(),matrix(xs,va),p.loc[va,E.TARGET].to_numpy(),matrix(xf,ft),p.loc[ft,E.TARGET].to_numpy(),matrix(xf,te),int(os.getenv('NMR_SLOT_CORES','4')),SCR/'models'/tag/f'{asset}.joblib')
                record.update(asset=str(asset),training_rows=p.row.iloc[tr].tolist(),validation_rows=p.row.iloc[va].tolist(),final_rows=p.row.iloc[ft].tolist())
                data=pd.DataFrame(dict(row=p.row.iloc[te].to_numpy(),prediction=pred));E.parquet(pred_path,data);E.save(record_path,record)
            except Exception as error:
                E.save(root/'asset_failures'/f'{asset}.json',dict(error=str(error),traceback=traceback.format_exc()));failures.append(str(asset));continue
        mapping=data.set_index('row').prediction;selected=out.row.isin(mapping.index);out.loc[selected,'prediction']=out.loc[selected,'row'].map(mapping)
        records.append(record);E.save(root/'progress.json',dict(completed_assets=len(records),failed_assets=len(failures),latest_asset=str(asset),latest_seconds=record['seconds'],time=time.time()))
        print('ASSET',tag,asset,record['seconds'],flush=True)
    if failures or not records:raise RuntimeError(f'Unfinished failed assets: {failures}; successful assets checkpointed')
    E.parquet(root/'predictions.parquet',out)
    E.save(root/'COMPLETE.json',dict(task=dict(tag=tag,model=family),assets=len(records),predictions=int(out.prediction.notna().sum()),variant=variant,time=time.time()))


def supervise():
    prepare();state_path=ROOT/'supervisor.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(attempts={},owned={})
    def interrupt(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        while True:
            probes=hosts.reachable(hosts.probe_all(timeout=3));snapshots=manager.status(names=[h.name for h in probes],timeout=3);active,_=E.unit_states(snapshots)
            pending=[];complete=0
            for c in json.loads((ROOT/'plan.json').read_text()):
                for variant in ['H','Hmom']:
                    for family in FAMILIES:
                        tag=f'{c["id"]}_{variant}_{family}';destination=ROOT/'fits'/tag
                        if variant=='H' and family!='tabnet_ktz':
                            source=OLD/'fits'/f'{c["id"]}_{family}'
                            if (source/'COMPLETE.json').exists():
                                destination.mkdir(parents=True,exist_ok=True)
                                if not (destination/'COMPLETE.json').exists():
                                    shutil.copy2(source/'predictions.parquet',destination/'predictions.parquet')
                                    E.save(destination/'COMPLETE.json',dict(task=dict(tag=tag,model=family),reference=str(source),source_hash=E.sha(source/'predictions.parquet')))
                            if (destination/'COMPLETE.json').exists():complete+=1
                            continue
                        if (destination/'COMPLETE.json').exists():complete+=1;continue
                        name='e220_'+tag;unit='nmr-'+name
                        if unit in active:continue
                        prior=state['owned'].get(unit)
                        if prior and prior not in snapshots:continue
                        if state['attempts'].get(name,0)>=3:raise RuntimeError('Three attempts exhausted: '+name)
                        pending.append(budget.Job(name,f'{B.PY} -u experiments/e220_ja_heads.py fit --config {c["id"]} --variant {variant} --family {family}',cores=4,ram_gb=48))
            # One input writer per config: its first TabNet-H job prepares the cache.
            pending=[j for j in pending if '_H_tabnet_ktz' in j.name or (ROOT/'inputs'/j.name.split('_')[1]/'COMPLETE.json').exists()]
            placements,waiting=budget.distribute(pending,probes,share=.30)
            reserved={}
            for host,rows in snapshots.items():
                reserved[host]=sum((8 if 'e220_ja_heads_supervisor' in row['unit'] else 16 if 'e218_asset_ktz_supervisor' in row['unit'] else 48 if 'nmr-e220_' in row['unit'] else 24 if 'nmr-e218_' in row['unit'] else 0) for row in rows if row['state'].startswith('active'))
            for placement in placements:
                host=next(h for h in probes if h.name==placement.host)
                if reserved.get(host.name,0)+placement.job.ram_gb>budget.ram_cap_gb(host,.30):continue
                reserved[host.name]=reserved.get(host.name,0)+placement.job.ram_gb
                import subprocess
                subprocess.run(['ssh','-o','BatchMode=yes',placement.host,'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user reset-failed '+placement.unit],capture_output=True,timeout=15)
                ok,message=manager.launch(placement);print('LAUNCH',ok,message,flush=True)
                if ok:state['attempts'][placement.job.name]=state['attempts'].get(placement.job.name,0)+1;state['owned'][placement.unit]=placement.host;E.save(state_path,state)
            E.save(ROOT/'status.json',dict(completed_portfolios=complete,expected=24,waiting=[j.name for j,_ in waiting],time=time.time()))
            if complete==24:
                from src.analysis.e220_report import report
                report(ROOT,B.SOURCE);E.save(ROOT/'COMPLETE.json',dict(portfolios=24,time=time.time()));return
            time.sleep(30)
    except Exception as error:
        E.save(ROOT/'FAILED.json',dict(error=str(error),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        for host in set(state['owned'].values()):
            manager.stop(host,[unit for unit,owner in state['owned'].items() if owner==host],timeout=20)



if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('command',choices=['supervise','fit']);a.add_argument('--config');a.add_argument('--variant',choices=['H','Hmom']);a.add_argument('--family',choices=FAMILIES);args=a.parse_args()
    if args.command=='fit':fit(args.config,args.variant,args.family)
    else:supervise()
