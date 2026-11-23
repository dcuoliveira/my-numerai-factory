"""E207 full-history walk-forward extension of E206; frozen embeddings reused."""
import os,sys,time,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
ORIGINAL=REPO/'results/reports/crypto/e206_momentum3_tsfm'
ROOT=REPO/'results/reports/crypto/e207_fullsample'
SCR=Path('/scratch/danielco/e207_fullsample')
PY='/home/danielco/miniconda3/bin/python3'

def main():
    ROOT.mkdir(parents=True,exist_ok=True);SCR.mkdir(parents=True,exist_ok=True)
    E.save(ROOT/'status.json',dict(status='queued_after_E206',evaluation='full-sample walk-forward'))
    while not (ORIGINAL/'COMPLETE.json').exists():
        if (ORIGINAL/'FAILED.json').exists():raise RuntimeError('E206 requires recovery before full-sample extension')
        time.sleep(30)
    p=pd.read_parquet(ORIGINAL/'panel.parquet');dates=np.sort(p.date.unique())
    source=Path('/scratch/danielco/e206_momentum3_tsfm');common=np.load(source/'common_rows.npy')
    folds=[]
    for start in range(200,len(dates),130):
        origin=pd.Timestamp(dates[start]);end=origin-pd.Timedelta(days=35)
        past=dates[dates<=end.to_datetime64()]
        if len(past)<64:continue
        valstart=pd.Timestamp(past[-63]);innerend=valstart-pd.Timedelta(days=35)
        lower=pd.Timestamp(dates[max(0,start-504)])
        q=p.copy();q['train']=(q.date>=lower)&(q.date<=innerend)
        q['validation']=(q.date>=valstart)&(q.date<=end)
        q['final_fit']=(q.date>=lower)&(q.date<=end)
        q['test']=q.date.isin(dates[start:start+130])
        tr=q.loc[common&q.train].groupby('ucid').size();va=q.loc[common&q.validation].groupby('ucid').size()
        if not len(tr.index[(tr>=60)&(va.reindex(tr.index,fill_value=0)>=20)]):continue
        f=len(folds);r=ROOT/'folds'/f'f{f:02d}';s=SCR/f'f{f:02d}';r.mkdir(parents=True,exist_ok=True);s.mkdir(exist_ok=True)
        E.parquet(r/'panel.parquet',q)
        for name in ['PREPARED.json','EMBEDDINGS_READY.json']:
            (r/name).write_bytes((ORIGINAL/name).read_bytes())
        for name in ['embeddings','common_rows.npy']:
            if not (s/name).exists():(s/name).symlink_to(source/name,target_is_directory=name=='embeddings')
        env=dict(os.environ,E206_SCOPE_ROOT=str(r),E206_SCOPE_SCR=str(s),E206_SCOPE_ID='e207',E206_SCOPE_RUN='r01',E206_SCOPE_FOLD=str(f))
        E.save(r/'grid.json',[dict(id='e207_'+c['id'].removeprefix('e206_'),**{k:v for k,v in c.items() if k!='id'}) for c in json.loads((ORIGINAL/'grid.json').read_text())])
        E.save(r/'walkforward.json',dict(origin=str(origin),training_lower=str(lower),inner_last=str(innerend),validation_first=str(valstart),last_resolved_label=str(end)))
        subprocess.run([PY,str(REPO/'experiments/e206_momentum3_tsfm.py'),'supervise'],env=env,check=True)
        subprocess.run([PY,'-c','from experiments import e206_momentum3_tsfm as E; E.momentum_baseline()'],env=env,cwd=REPO,check=True)
        folds.append(r)
        E.save(ROOT/'progress.json',dict(completed_folds=len(folds),latest_origin=str(origin)))
    if not folds:raise RuntimeError('No eligible walk-forward blocks')
    E.ROOT=ROOT;E.SCR=SCR
    for name in ['PREPARED.json','execution_returns.parquet']:(ROOT/name).write_bytes((ORIGINAL/name).read_bytes())
    grid=json.loads((folds[0]/'grid.json').read_text());E.save(ROOT/'grid.json',grid)
    p['test']=True;E.parquet(ROOT/'panel.parquet',p)
    baseline=p[['row','symbol','ucid','date',E.TARGET,'meta_model']].copy()
    baseline['prediction']=np.nan;baseline['baseline']=.5
    baseline['baseline_rank']=baseline.groupby('date').baseline.transform(lambda s:E.rank_normalise(s.to_numpy()))
    parts=pd.concat([pd.read_parquet(r/'momentum_baseline.parquet') for r in folds],ignore_index=True).set_index('row')
    assert parts.index.is_unique
    baseline=baseline.set_index('row');baseline.update(parts);E.parquet(ROOT/'momentum_baseline.parquet',baseline.reset_index())
    for c in grid:
        frames=[];choices=[]
        for r in folds:
            d=r/'fits'/c['id']
            if not (d/'COMPLETE.json').exists():raise RuntimeError('Missing full-sample configuration: '+str(d))
            frames.append(pd.read_parquet(d/'predictions.parquet'));choices.append(json.loads((d/'validation_selection.json').read_text()))
        forecast=pd.concat(frames,ignore_index=True);assert not forecast.row.duplicated().any()
        d=ROOT/'fits'/c['id'];E.parquet(d/'predictions.parquet',forecast)
        E.save(d/'validation_selection.json',dict(mean_asset_mse=float(np.mean([x['mean_asset_mse'] for x in choices])),window='per-fold',descriptive_only=True))
        E.save(d/'COMPLETE.json',dict(task=dict(tag=c['id'],model=c['model']),folds=len(folds)))
    os.environ['E206_SCOPE_ID']='e207'
    os.environ['E206_SCOPE_FULLSAMPLE']='1'
    E.save(ROOT/'status.json',dict(status='scoring_full_sample',folds=len(folds)))
    E.score()
    page=ROOT/'E206_momentum3_tsfm.html'
    if page.exists():
        document=page.read_text().replace('E206 — Momentum3 frozen representations','E207 — Full-sample walk-forward portfolios').replace('Validation selects parameters; the entire test grid is reported.','Each forecast block selects parameters on its resolved past. All available forecast dates form one full-sample backtest; initial warm-up dates hold cash.')
        final_page=ROOT/'E207_fullsample.html';final_page.write_text(document)
        try:subprocess.run(['ssh','-o','BatchMode=yes','lovelace','python3',str(REPO/'src/analysis/html_pdf.py'),str(final_page)],check=True,timeout=300)
        except Exception as error:E.save(ROOT/'pdf_error_fullsample.json',dict(error=str(error)))
    E.save(ROOT/'status.json',dict(status='complete',folds=len(folds),report=str(ROOT/'E207_fullsample.html')))
    winner=ROOT/'validation_selected_configuration.json'
    if winner.exists():winner.rename(ROOT/'descriptive_validation_summary.json')
    E.save(ROOT/'evaluation_scope.json',dict(experiment='E207',full_sample=True,folds=len(folds),
        dates=[str(dates[0]),str(dates[-1])],protocol='Full-history rolling forecasts; tuning/scalers only on preceding resolved labels,35-calendar-day purges; up to504 prior panel dates;63-date validation;130-date prediction blocks. No reserved final test.',
        warmup='Initial rows without eligible asset models remain explicitly missing; portfolio holds cash when fewer than10 forecasts exist. No in-sample predictions substituted.',
        global_rankings='Descriptive full-sample results, not configuration selection for past portfolio returns'))
if __name__=='__main__':main()
