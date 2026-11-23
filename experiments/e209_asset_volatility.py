"""E209: E205's independent asset regressions on selected volatility formulas.

Uses E205's unchanged fitting, baseline and scoring functions, with explicit
per-dataset feature coordinates. No pooled fits or cross-sectional inputs.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
import fcntl
import html
import json
import socket
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
import e205_asset_momentum as B
from src.analysis import asset_momentum as M
from src.data_mani import volatility_proxies as V
from src.data_mani.build_volatility_replacement import SPECS
from src.analysis.cross_target import newey_west
from e150_finish_scoring import matrix_scores

ROOT = REPO/'results/reports/crypto/e209_asset_volatility'
MOMENTUM = REPO/'results/reports/crypto/e205_asset_momentum'
WINDOWS = tuple(V.WINDOWS)
FORMULAS = {f'vol_{i+1}_{s[1]}_{s[2]}_{s[3]}_{s[5]}': s for i, s in enumerate(SPECS)}
GROUPS = {name: 11 for name in FORMULAS}
GROUPS.update(selected_volatility9=9, momentum132=132, momentum132_volatility9=141)
TASKS = [(model, name) for model in ['ridge', 'mlp_torch'] for name in GROUPS]
save, parquet, sha = B.save, B.parquet, B.sha
CODE = [Path(__file__), REPO/'experiments/e205_asset_momentum.py',
        REPO/'src/analysis/asset_momentum.py', REPO/'src/data_mani/volatility_proxies.py',
        REPO/'src/data_mani/build_volatility_replacement.py', B.CANDIDATES,
        REPO/'src/prediction/torch_mlp.py', REPO/'experiments/e150_finish_scoring.py',
        REPO/'src/live/submit.py', REPO/'src/analysis/cross_target.py']


def feature_frame(values):
    return pd.DataFrame(values, columns=[f'feature_ts_{i}' for i in range(values.shape[1])], dtype=np.float32)


def evaluate_asset(dates, bars, benchmark):
    """Keep selected calendar/estimator/side; window sweeps use TS bins.

    Selected9 preserves original raw/percentile/bin view *before* CS binning.
    At a fixed source date, all transforms use strictly earlier observations.
    """
    grids, selected = {}, np.full((len(dates), 9), np.nan)
    for i, (name, spec) in enumerate(FORMULAS.items()):
        _, family, calendar, method, window, side, view = spec
        settings = [(calendar, method, w, side) for w in WINDOWS]
        values = V.build_asset(dates, bars, benchmark if family == 'idiosyncratic' else {}, family, settings)
        col = int(side == 'up')
        ts = []
        for w in WINDOWS:
            raw, hist = values[calendar, method, w]
            ts.append(V.bins(hist[:, col]))
            if w == window:
                selected[:, i] = raw[:, col] if view == 'raw' else (
                    V.bins(hist[:, col]) if view == 'bins' else hist[:, col])
        grids[name] = np.column_stack(ts)
    return grids, selected


def prepare():
    B.ROOT = ROOT
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'PREPARED.json').exists():
        return B.verify()
    source_files = [B.SOURCE/'train.parquet', B.SOURCE/'historical_meta_models.parquet', B.PRICES,
                    MOMENTUM/'keys.parquet', MOMENTUM/'registration.json', MOMENTUM/'PREPARED.json']
    source_files += [MOMENTUM/'inputs'/f'{name}_ts.parquet' for name in M.FAMILIES]
    plan = dict(experiment='E209', registered='2026-11-23', revision=1,
        groups=GROUPS, selected_specs=SPECS, windows=WINDOWS, n_portfolios=len(TASKS),
        candidates=json.loads(B.CANDIDATES.read_text()), seeds=B.SEEDS,
        models=['ridge', 'mlp_torch'],
        processing='9 selected estimator/calendar/side specifications. Selected9 retains raw, prior '
                   'percentile or prior-bin views before cross-sectional binning. Each of nine '
                   'formula datasets uses all11 windows with prior-history TS bins. '
                   'Momentum132 reuses all12x11 E205 TS-only inputs; combined has141 inputs.',
        selection='No outer-score selection. All nine previously selected volatility measures '
                  'were chosen on this development history; rankings remain exploratory.',
        fitting='Unchanged E205 independent asset temporal-MSE tuning: ridge100 alphas; '
                'identity Torch MLP80 frozen candidates, exact sklearn weights AND biases, '
                '200 max epochs, loss tol1e-4/patience10, five seeds; no pooled fitting.',
        folds='outer130/purge22/test130; inner68/purge22/validation40; min40/20/80 rows',
        coverage='Original panel keys and candidate NaNs retained. Common causal intersection '
                 'of all formula/window, selected9 and momentum features plus E205 identity/OHLC gate; '
                 'BTC and ETH excluded from common candidate cohort to avoid self-factor idiosyncratic variance. '
                 'Per-arm source masks retained separately. No coverage predictors.',
        source='Frozen E157 prices, permanent-ID verified E205 cohort, completed bars <=D-1; '
               'daily and weekday selected calendars. Strict grid, no filling of missing bars; '
               'weekday rollback at most3 calendar days from D. Prior252/min126 excludes current.',
        controls='Matched covered per-asset historical target mean, and full-universe historical '
                 'mean fallback. Momentum and combined trained freshly on identical common keys.',
        routing='Identical E205 deterministic rank-slot routing; exact uncovered ranks preserved.',
        ensembles='Three prespecified equal-rank portfolios over nine formula groups: ridge, '
                  'Torch, both. Never mix momentum/combined/selected9 into these ensembles.',
        comparisons='selected9 vs momentum132 and combined141 vs momentum132 by model; '
                    'all portfolios vs matched asset mean. Payout=0.1CORR+historical-meta proxyMMC, '
                    'HAC20/block20 bootstrap2000, seeds/blocks, cumulative score/drawdown.',
        claim='Standalone per-asset research and marginal value over momentum only; '
              'no marginal native22 claim, no actual investment returns or live changes.',
        limitation='Current vendor snapshots cannot establish historical availability. '
                   'Development only; all target dates>=2026-02-20 excluded at source read.',
        code_hashes={str(p.relative_to(REPO)):sha(p) for p in CODE},
        input_hashes={str(p.relative_to(REPO)):sha(p) for p in source_files})
    if (ROOT/'registration.json').exists():
        previous=B.verify()
        assert previous['input_hashes']==plan['input_hashes'], 'Source snapshot changed during preparation'
    save(ROOT/'registration.json', plan)
    base = pd.read_parquet(B.SOURCE/'train.parquet', columns=['date','ucid',B.TARGET],
                           filters=[('date','<',B.CUTOFF)]).reset_index()
    base = base.sort_values(['date','symbol']).reset_index(drop=True)
    assert base.date.max() < B.CUTOFF and not base.duplicated(['ucid','date']).any()
    old = pd.read_parquet(MOMENTUM/'keys.parquet')
    pd.testing.assert_frame_equal(base[['symbol','ucid','date',B.TARGET]],
                                  old[['symbol','ucid','date',B.TARGET]])
    meta = pd.read_parquet(B.SOURCE/'historical_meta_models.parquet',
                            filters=[('date','<',B.CUTOFF)]).reset_index()
    base = base.merge(meta[['symbol','date','prediction']].rename(columns={'prediction':'meta_model'}),
                      on=['symbol','date'], how='left', validate='1:1')
    base['row'] = np.arange(len(base))
    prices = pd.read_parquet(B.PRICES, filters=[('date','<',B.CUTOFF)])
    _, _, cohort = M.price_context(base, prices)
    identity = prices.groupby('symbol').ucid.nunique().eq(1)
    source = prices[prices.symbol.map(identity)].rename(columns={'date':'bar_date'})
    # Reject UCID reuse/multiple symbols, rather than arbitrarily picking a ticker.
    valid_ids = source.groupby('ucid').symbol.nunique().eq(1)
    bars = {str(i): V.clean_bars(g) for i,g in source.groupby('ucid') if valid_ids.loc[i]}
    benchmarks = {i:bars[str(i)].close for i in (1,1027)}
    inputs = {name:np.full((len(base),11),np.nan,dtype=np.float32) for name in FORMULAS}
    selected = np.full((len(base),9),np.nan,dtype=np.float32)
    audit_rows = []
    eligible = base[cohort & ~base.ucid.astype(str).isin(['1','1027'])]
    for n,(ucid,g) in enumerate(eligible.groupby(eligible.ucid.astype(str)),1):
        if ucid not in bars:
            continue
        # Use all original dates for this asset, not future cohort membership, as the calculation grid.
        rows = base.index[base.ucid.astype(str).eq(ucid)]
        dates = pd.DatetimeIndex(base.loc[rows,'date'])
        cache = ROOT/'asset_inputs'/f'{ucid}.parquet'
        audit_path = ROOT/'asset_inputs'/f'{ucid}_prefix.json'
        if cache.exists() and audit_path.exists():
            stored = pd.read_parquet(cache)
            np.testing.assert_array_equal(stored.row,rows)
            values = stored.drop(columns='row').to_numpy()
            grids = {name:values[:,j*11:(j+1)*11] for j,name in enumerate(FORMULAS)}
            nine = values[:,-9:]
            audit_rows.extend(json.loads(audit_path.read_text()))
        else:
            grids,nine = evaluate_asset(dates,bars[ucid],benchmarks)
            checks=[]
            for cut in pd.to_datetime(['2022-12-30','2024-12-31','2025-12-31']):
                keep=dates<=cut
                if not keep.any():
                    continue
                replay,rnine=evaluate_asset(dates[keep],bars[ucid].loc[bars[ucid].index<cut],
                       {i:s.loc[s.index<cut] for i,s in benchmarks.items()})
                np.testing.assert_allclose(nine[keep],rnine,rtol=1e-9,atol=1e-12,equal_nan=True)
                for name in grids:
                    np.testing.assert_array_equal(grids[name][keep],replay[name])
                checks.append(dict(ucid=ucid,cutoff=str(cut.date()),formula_windows=99,selected=9,changed=0))
            data=pd.DataFrame(np.column_stack([*grids.values(),nine]))
            data.insert(0,'row',rows)
            data.columns=data.columns.astype(str)
            parquet(cache,data)
            save(audit_path,checks);audit_rows.extend(checks)
        for name in inputs:
            inputs[name][rows]=grids[name]
        selected[rows]=nine
        if n%10==0:
            print('SOURCE',n,'assets',flush=True)
    momentum=np.column_stack([pd.read_parquet(MOMENTUM/'inputs'/f'{f}_ts.parquet').to_numpy()
                              for f in M.FAMILIES])
    inputs.update(selected_volatility9=selected,momentum132=momentum,
                  momentum132_volatility9=np.column_stack([momentum,selected]))
    common=cohort & ~base.ucid.astype(str).isin(['1','1027'])
    masks={name:cohort & np.isfinite(value).all(axis=1) & ~base.ucid.astype(str).isin(['1','1027'])
           for name,value in inputs.items()}
    for mask in masks.values():
        common &= mask
    assert common.any(), 'No common source-covered rows'
    base['covered']=common
    mask_table=base[['row','symbol','ucid','date']].copy()
    dataset_hashes={}
    for name,values in inputs.items():
        mask_table[name]=masks[name]
        values=values.copy();values[~common]=np.nan
        frame=pd.concat([base[['row','symbol','ucid','date',B.TARGET,'covered']],feature_frame(values)],axis=1)
        path=ROOT/'datasets'/f'{name}.parquet';parquet(path,frame)
        dataset_hashes[name]=sha(path)
    parquet(ROOT/'source_masks.parquet',mask_table)
    parquet(ROOT/'keys.parquet',base)
    save(ROOT/'date_grid.json',list(map(str,np.sort(base.date.unique()))))
    pd.DataFrame(audit_rows).to_csv(ROOT/'source_prefix_audits.csv',index=False)
    base.groupby('date').covered.agg(panel_rows='size',covered_rows='sum',covered_share='mean').to_csv(ROOT/'coverage_by_date.csv')
    save(ROOT/'PREPARED.json',dict(rows=len(base),covered=int(common.sum()),assets=int(base.ucid.nunique()),
        eligible_assets=int(base.loc[common].ucid.nunique()),datasets=len(inputs),feature_counts=GROUPS,
        dataset_hashes=dataset_hashes,keys_sha256=sha(ROOT/'keys.parquet'),prefix_checks=len(audit_rows),finished=time.time()))
    print('PREPARED',json.loads((ROOT/'PREPARED.json').read_text()),flush=True)
    return plan


def run_task(task):
    B.ROOT = ROOT
    # E205's unchanged fitter enumerates feature_ts_<coordinate>. Configure the
    # coordinates in this worker only; each process executes one task at a time.
    B.M.WINDOWS=list(range(GROUPS[task[1]]))
    return B.run_task(task)


def score_tasks():
    B.ROOT = ROOT
    keys=pd.read_parquet(ROOT/'keys.parquet');fallback=pd.read_parquet(ROOT/'baseline.parquet')
    rows=[]
    for i,task in enumerate(TASKS,1):
        path=ROOT/'daily'/f'{task[0]}__{task[1]}.parquet'
        if path.exists():
            r=pd.read_parquet(path).to_dict('records')
        else:
            r=B.score_one(task,keys,fallback)
        for row in r:
            row['window']='selected' if task[1] not in FORMULAS else 'all11'
            row['rep']='selected_pre_cs_views' if 'volatility9' in task[1] else 'ts_bins'
        rows.extend(r);print('SCORE',i,'/',len(TASKS),flush=True)
    return pd.DataFrame(rows),keys,fallback


def add_ensembles(daily,keys,fallback):
    rows=[]
    for model in ['ridge','mlp_torch','both']:
        selected=[t for t in TASKS if t[1] in FORMULAS and (model=='both' or t[0]==model)]
        total=None;coverage=None
        for m,f in selected:
            p=pd.read_parquet(ROOT/'portfolios'/f'{m}__{f}.parquet').set_index('row')
            if coverage is None: coverage=p.covered
            else: pd.testing.assert_series_equal(coverage,p.covered)
            x=p[['routed','control_routed']]
            total=x if total is None else total.add(x)
        total/=len(selected);total['covered']=coverage
        full=fallback.merge(keys[['row','symbol',B.TARGET,'meta_model']],on='row',validate='1:1').merge(total.reset_index(),on='row',validate='1:1')
        name=f'{model}__VOL9_FORMULAS__fixed_equal_rank';portfolios=[]
        for date,g in full.groupby('date'):
            g=g.sort_values('symbol').copy();mask=g.covered.to_numpy()
            for col in ['routed','control_routed']:
                values=g[col].to_numpy().copy();values[~mask]=np.nan
                g[col]=M.route_slots(g.baseline_rank.to_numpy(),values)
            portfolios.append(g[['row','date','block','routed','control_routed']])
            g=g[np.isfinite(g[B.TARGET])&np.isfinite(g.meta_model)]
            if len(g)<10: continue
            c,mm=matrix_scores(g[['routed','control_routed','baseline_rank']].to_numpy(),g[B.TARGET].to_numpy(),g.meta_model.to_numpy())
            for j,arm in enumerate(['signal','asset_mean','baseline']):
                cv=float(c[j]) if np.isfinite(c[j]) else 0.
                rows.append(dict(task=name,model=model,family='VOL9_FORMULAS',rep='equal_rank',window='all11',
                    seed='ensemble',date=date,block=int(g.block.iloc[0]),sample='routed',arm=arm,
                    corr=cv,mmc=float(mm[j]),payout=.1*cv+float(mm[j]),n_coins=len(g)))
        parquet(ROOT/'ensembles'/f'{name}.parquet',pd.concat(portfolios,ignore_index=True))
    return pd.concat([daily,pd.DataFrame(rows)],ignore_index=True)


def summarize(daily):
    table=[];deltas=[]
    for (task,sample,seed),g in daily.groupby(['task','sample','seed']):
        s=g[g.arm=='signal'].sort_values('date')
        if s.empty: continue
        y=s.payout.to_numpy();curve=np.cumsum(y)
        table.append(dict(task=task,sample=sample,seed=seed,corr=s['corr'].mean(),mmc=s.mmc.mean(),
            payout=y.mean(),sharpe_payout=y.mean()/y.std(ddof=1),maxdd_payout=float((np.maximum.accumulate(np.r_[0,curve])[1:]-curve).max()),
            n_dates=len(s),coins=s.n_coins.mean()))
        references={r:g[g.arm==r] for r in ['asset_mean','baseline']}
        if '__momentum132_volatility9' in task or '__selected_volatility9' in task:
            model=task.split('__')[0]
            references['matched_momentum132']=daily[(daily.task==f'{model}__momentum132')&(daily['sample']==sample)&(daily.seed==seed)&(daily.arm=='signal')]
        for ref,r in references.items():
            p=s.merge(r,on='date',suffixes=('_signal','_control'),validate='1:1')
            if p.empty: continue
            assert len(p)==len(s), 'Paired date grids differ'
            for metric in ['corr','mmc','payout']:
                diff=(p[f'{metric}_signal']-p[f'{metric}_control']).to_numpy()
                mean,t=newey_west(diff,lag=20)
                ci=B.bootstrap_mean(diff) if seed=='ensemble' else [None,None]
                deltas.append(dict(task=task,sample=sample,seed=seed,reference=ref,metric=metric,
                    delta=mean,hac20_t=t,ci_low=ci[0],ci_high=ci[1],n=len(diff)))
    return pd.DataFrame(table).sort_values('payout',ascending=False),pd.DataFrame(deltas)


def audit_outputs():
    B.ROOT = ROOT
    B.verify();keys=pd.read_parquet(ROOT/'keys.parquet');fallback=pd.read_parquet(ROOT/'baseline.parquet')
    checks=[];coverage=None
    for model,name in TASKS:
        f=pd.read_parquet(ROOT/'datasets'/f'{name}.parquet')
        pd.testing.assert_frame_equal(f[['row','ucid','date',B.TARGET,'covered']],keys[['row','ucid','date',B.TARGET,'covered']])
        features=f.filter(regex='^feature_ts_')
        assert features.shape[1]==GROUPS[name]
        assert features.loc[~f.covered].isna().all().all()
        assert np.isfinite(features.loc[f.covered]).all().all()
        p=pd.read_parquet(ROOT/'portfolios'/f'{model}__{name}.parquet').merge(fallback[['row','baseline_rank']],on='row',validate='1:1')
        if coverage is None: coverage=p[['row','covered']].sort_values('row').reset_index(drop=True)
        else: pd.testing.assert_frame_equal(coverage,p[['row','covered']].sort_values('row').reset_index(drop=True))
        assert np.isfinite(p[['routed','control_routed','baseline_rank']]).all().all()
        assert (p.loc[~p.covered,'routed']==p.loc[~p.covered,'baseline_rank']).all()
        for _,g in p.groupby('date'):
            for col in ['routed','control_routed']:
                np.testing.assert_array_equal(np.sort(g.loc[g.covered,col]),np.sort(g.loc[g.covered,'baseline_rank']))
        checks.append(dict(task=f'{model}__{name}',rows=len(p),covered=int(p.covered.sum()),pass_checks=True))
    prefixes=pd.read_csv(ROOT/'source_prefix_audits.csv');assert prefixes.changed.eq(0).all()
    audit=dict(outcome='PASS WITH CONDITIONS',checks=checks,prefix_rebuilds=len(prefixes),
        conditions=['Current vendor histories and retrospective identity filters cannot prove historical availability.',
            'Standalone per-asset research; marginal comparison applies only to the matched momentum132 model.',
            'Volatility specifications selected on this same development history; no fresh signal-confirmation holdout.',
            'No live deployment or investment-return claim.'])
    save(ROOT/'audit.json',audit);return audit


def volatility_information_set(task):
    """Readable input-set labels for the full-universe portfolio table."""
    group = task.split('__')[1]
    if group == 'VOL9_FORMULAS':
        return 'All nine volatility formula groups (11 windows each)'
    if group == 'selected_volatility9':
        return 'Selected nine volatility measures (selected windows/views)'
    if group == 'momentum132':
        return 'None (132 momentum inputs only)'
    if group == 'momentum132_volatility9':
        return 'Selected nine volatility measures + 132 momentum inputs'
    _, family, calendar, method, _, side, _ = FORMULAS[group]
    names = {'idiosyncratic': 'Idiosyncratic', 'semivariance': 'Semivariance',
             'rogers_satchell': 'Rogers–Satchell', 'parkinson': 'Parkinson',
             'yang_zhang': 'Yang–Zhang'}
    methods = {'btc_eth': 'BTC + ETH', 'btc': 'BTC', 'side_count_log': 'side-count log',
               'gap_added': 'gap added', 'standard': 'standard', 'ewma': 'EWMA'}
    direction = f', {side}side' if side in ('down', 'up') else ''
    return f'{names[family]}{direction} ({calendar}; {methods[method]}; 11 windows)'


def report(table,deltas,daily,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for ax,model in zip(axes,['ridge','mlp_torch']):
        for group in ['selected_volatility9','momentum132','momentum132_volatility9']:
            p=daily[(daily.task==f'{model}__{group}')&(daily['sample']=='routed')&(daily.seed=='ensemble')&(daily.arm=='signal')].sort_values('date')
            ax.plot(p.date,p.payout.cumsum(),label=group)
        p=daily[(daily.task==f'{model}__VOL9_FORMULAS__fixed_equal_rank')&(daily.arm=='signal')].sort_values('date')
        ax.plot(p.date,p.payout.cumsum(),label='nine formula equal-rank ensemble')
        ax.set_title(model);ax.set_ylabel('Cumulative payout score');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'cumulative_payout.png',dpi=150);plt.close(fig)
    primary=table[(table.seed=='ensemble')&(table['sample']=='routed')].copy()
    primary.insert(1, 'Volatility information set', primary.task.map(volatility_information_set))
    comparison=deltas[(deltas.seed=='ensemble')&(deltas['sample']=='routed')&(deltas.metric=='payout')]
    body='<h1>E209 — Per-asset volatility models and matched momentum comparison</h1>'
    body+='<p>Separate ridge and sklearn-initialised Torch identity MLP for every asset, dataset and fold. '
    body+='Identity MLPs are affine functions; this is the unchanged E205 optimisation comparison, not a nonlinear model test. '
    body+='The nine selected volatility measures retain their original calendars, estimators and pre-cross-sectional views. '
    body+='Nine formula models each jointly use eleven prior-history-binned windows. Selected9, momentum132 and combined141 '
    body+='are fitted freshly on exactly the same common eligible rows. No one-hot or cross-sectional input processing. '
    body+='Five MLP predictions averaged before ranking. Fixed formula ensembles have equal rank weights. '
    body+='Payout=0.1 CORR+historical-meta proxy MMC; drawdown describes payout scores, not investment returns. '
    body+='Source reconstruction uses today’s frozen vendor snapshot. Previously selected specifications and all rankings '
    body+='are development evidence; the final target holdout remains closed.</p>'
    body+='<h2>All full-universe portfolios — sorted by payout</h2>'+primary.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    body+='<img style="width:100%" src="cumulative_payout.png">'
    body+='<h2>Paired payout differences</h2>'+comparison.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    body+='<p>HAC lag20 and circular block20 bootstrap, 2,000 draws. Intervals are unadjusted; '
    body+='formula rankings are exploratory. Matched momentum comparisons use identical common covered and fit-eligible rows.</p>'
    body+='<h2>Coverage and data construction</h2><pre>'+html.escape((ROOT/'PREPARED.json').read_text())+'</pre>'
    body+='<h2>Missing-value and routing audit</h2><pre>'+html.escape(json.dumps(audit,indent=2))+'</pre>'
    body+='<h2>Covered-only portfolios and individual seeds</h2>'+table.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    body+='<h2>All paired comparisons</h2>'+deltas.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    body+='<h2>Per-block results</h2>'+daily.groupby(['task','sample','seed','arm','block'])[['corr','mmc','payout']].mean().reset_index().to_html(index=False,float_format=lambda x:f'{x:.6f}')
    body+='<h2>Frozen design and input provenance</h2><pre>'+html.escape((ROOT/'registration.json').read_text())+'</pre>'
    page=ROOT/'E209_asset_volatility.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:8px;border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px}pre{white-space:pre-wrap}</style></head><body>'+body+'</body></html>')
    from src.analysis.html_pdf import to_pdf
    to_pdf(str(page),'lovelace')
    pdf=page.with_suffix('.pdf')
    import fitz
    with fitz.open(pdf) as document:
        assert len(document)>2 and 'E209' in document[0].get_text()
        pages=len(document)
    save(ROOT/'COMPLETE.json',dict(experiment='E209',finished=time.time(),portfolios=len(TASKS),fixed_ensembles=3,
        pdf=str(pdf),pdf_sha256=sha(pdf),pages=pages,audit=audit['outcome']))


def finish():
    B.ROOT = ROOT
    B.verify()
    assert all((ROOT/'fits'/f'{m}__{f}'/'COMPLETE.json').exists() for m,f in TASKS)
    daily,keys,fallback=score_tasks();daily=add_ensembles(daily,keys,fallback)
    parquet(ROOT/'scores_with_fixed_ensembles.parquet',daily)
    table,deltas=summarize(daily)
    table.to_csv(ROOT/'table.csv',index=False);deltas.to_csv(ROOT/'paired_deltas.csv',index=False)
    table[(table['sample']=='routed')&(table.seed=='ensemble')&~table.task.str.contains('fixed_equal_rank')].rename(columns={'task':'tag'}).to_csv(ROOT/'all_model_window_rows.csv',index=False)
    daily.groupby(['task','sample','seed','arm','block'])[['corr','mmc','payout']].mean().to_csv(ROOT/'by_block.csv')
    audit=audit_outputs();report(table,deltas,daily,audit)


def email_report():
    import smtplib
    from email.message import EmailMessage
    from email.utils import getaddresses, make_msgid
    from src.live.env import load_env
    from src.live.notify import _config
    receipt=ROOT/'email_delivery.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('smtp_accepted'): return
    complete=json.loads((ROOT/'COMPLETE.json').read_text());pdf=Path(complete['pdf'])
    assert sha(pdf)==complete['pdf_sha256']
    load_env(required=());config=_config()
    if config is None: raise RuntimeError('SMTP configuration unavailable')
    message=EmailMessage()
    message['Subject']='[numerai-crypto] E209 completed — per-asset volatility PDF report'
    message['From']=config['user'];message['To']=config['to'];message['Message-ID']=make_msgid()
    message.set_content('E209 has completed. Attached is the PDF covering the selected volatility '
                        'features, nine formula/lookback models, matched momentum comparison and '
                        'combined models. These are exploratory development results with no live changes.')
    message.add_attachment(pdf.read_bytes(),maintype='application',subtype='pdf',filename=pdf.name)
    if len(message.as_bytes())>24_000_000:
        raise RuntimeError('Report exceeds attachment size limit; split PDF before sending')
    recipients=[address for _,address in getaddresses([config['to']])]
    with smtplib.SMTP(config['host'],config['port'],timeout=120) as server:
        server.starttls();server.login(config['user'],config['password'])
        refused=server.send_message(message,from_addr=config['user'],to_addrs=recipients)
        assert not refused, 'SMTP refused report recipient'
        save(receipt,dict(smtp_accepted=True,time=time.time(),pdf_sha256=sha(pdf),
                         recipient=recipients,message_id=message['Message-ID']))
    print('Report email accepted',flush=True)


def run(workers):
    started=time.time()
    save(ROOT/'launch.json',dict(host=socket.gethostname(),pid=os.getpid(),start=started,workers=workers))
    prepare();B.baseline()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_task,t):t for t in TASKS}
        for future in as_completed(futures):
            task=future.result()
            n=sum((ROOT/'fits'/f'{m}__{f}'/'COMPLETE.json').exists() for m,f in TASKS)
            save(ROOT/'progress.json',dict(completed_tasks=n,total=len(TASKS),latest=task,updated=time.time(),elapsed_seconds=time.time()-started))
            print('TASK',n,'/',len(TASKS),task,flush=True)
    finish();email_report()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','run','finish','email'])
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    with open(ROOT/'runner.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for attempt in range(1,4):
            try:
                if args.command=='run':run(args.workers)
                elif args.command=='prepare':prepare()
                elif args.command=='finish':finish();email_report()
                else:email_report()
                (ROOT/'FAILED.json').unlink(missing_ok=True)
                break
            except Exception as exc:
                save(ROOT/'FAILED.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time(),attempt=attempt))
                if attempt==3: raise
                print('RETRY',attempt,str(exc),flush=True)
                time.sleep(15)
