"""Registered E173 fixed-config spot-volume ablation, matched 9/4 venues + Yahoo."""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
import numpy as np
import pandas as pd
from src.data_mani.spot_volume_comparison import (FOUR, SIGNALS, CONTINUOUS, asset_signals,
                                                 cross_section, arm_columns, prepare_fold)
import e133_histfe_factorial as H
import e135_route_histfe as Q
import e131_analysis as E

ROOT = REPO/'results/reports/crypto/e173_spot_volume'
WORK = Path('/scratch/danielco/volumealpha_e152')
SOURCE = WORK/'venue_observations.parquet'
BASE = WORK/'base_train.parquet'
MODELS = ['ridge','lgb_ktz','mlp_ktz']


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj, indent=2, default=str)); tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def panel():
    p=pd.read_parquet(BASE).reset_index()
    p['date']=pd.to_datetime(p.date)
    p['ucid']=p.ucid.astype(str)
    return p[p.date<=Q.LAST].sort_values(['date','symbol']).reset_index(drop=True)


def gate_panel(p, raw):
    p=p[['symbol','ucid','date']].merge(raw,on=['ucid','date'],how='left',validate='1:1')
    gate=pd.read_parquet(WORK/'identity_gate.parquet')
    gate['date']=pd.to_datetime(gate.date).dt.tz_localize(None)
    gate['ucid']=gate.ucid.astype(str)
    p=p.merge(gate,on=['ucid','date'],how='left',validate='1:1')
    cols=[c for c in p if c not in ['symbol','ucid','date','identity_stat']]
    p.loc[~p.identity_stat.le(.18),cols]=np.nan
    return cross_section(p.drop(columns='identity_stat'))


def build_venue(v):
    chunks=[]
    for ucid,g in v.groupby('ucid',sort=False):
        z=asset_signals(g);z['ucid']=str(ucid);chunks.append(z)
    return pd.concat(chunks,ignore_index=True)


def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'BUILD_COMPLETE.json').exists(): return
    p=panel(); base=[c for c in p if c.startswith('feature_')];assert len(base)==22
    # Prove the frozen baseline, labels and original training panel are unchanged.
    old,oldcols,_=H.load()
    old=old[['symbol','date',H.T20]+base].sort_values(['date','symbol']).reset_index(drop=True)
    chk=p[p.date<H.TEST_START][old.columns].reset_index(drop=True)
    chk[base]=chk[base].fillna(.5)
    pd.testing.assert_frame_equal(chk,old,check_dtype=False,check_exact=True)
    keys=pd.read_parquet(Q.KEYS);keys['date']=pd.to_datetime(keys.date)
    common=p.merge(keys[['symbol','date']],on=['symbol','date'],how='inner')
    assert len(common)==len(keys)==468000
    check=common[['symbol','date',H.T20]].merge(keys[['symbol','date',H.T20]],on=['symbol','date'],suffixes=('','_frozen'),validate='1:1')
    np.testing.assert_equal(check[H.T20].to_numpy(),check[H.T20+'_frozen'].to_numpy())
    v=pd.read_parquet(SOURCE)
    v['date']=pd.to_datetime(v.date).dt.tz_localize(None)
    available=pd.to_datetime(v.available_at_inferred,utc=True).dt.tz_localize(None)
    assert (available==v.date+pd.Timedelta(days=2)).all()
    v['ucid']=v.ucid.astype(str)
    v=v[v.price_crosscheck_pass & (v.date<=Q.LAST-pd.Timedelta(days=2))].copy()
    assert not v.duplicated(['ucid','venue','pair','date']).any()
    feeds={};audit=[]
    for feed, vv in [('nine',v),('four',v[v.venue.isin(FOUR)])]:
        print('Building',feed,len(vv),flush=True)
        raw=build_venue(vv); f=gate_panel(p,raw)
        for cutoff in ['2022-12-30','2024-12-31','2025-12-31']:
            # Real-source prefix rebuild on 12 diverse assets; no suffix can
            # alter a matched rolling window or prediction-date CS transform.
            ids=sorted(vv.ucid.unique())[:12]
            sample=vv[vv.ucid.isin(ids)]
            cut=pd.Timestamp(cutoff)
            full=build_venue(sample)
            pre=build_venue(sample[sample.date<=cut-pd.Timedelta(days=2)])
            a=full[full.date<=cut].sort_values(['ucid','date']).reset_index(drop=True)
            b=pre.set_index(['ucid','date']).reindex(pd.MultiIndex.from_frame(a[['ucid','date']])).reset_index()
            b=b[a.columns]
            pd.testing.assert_frame_equal(a,b,check_exact=True)
            audit.append(dict(feed=feed,cutoff=cutoff,assets=len(ids),rows=len(a),changed=0))
        feeds[feed]=f
    y=pd.read_parquet('data/crypto/spot/ohlcv.parquet')
    y=y[y.source.eq('yahoo')].copy();y['date']=pd.to_datetime(y.date);y['ucid']=y.ucid.astype(str)
    # Yahoo cryptocurrency Volume is already reported in USD: never multiply by price.
    yc=[]
    for u,g in y.groupby('ucid',sort=False):
        g=g.sort_values('date').drop_duplicates('date').set_index('date')
        grid=pd.date_range(g.index.min(),g.index.max(),freq='D')
        volume=g.volume.reindex(grid).where(lambda z:z>=0)
        yc.append(pd.DataFrame({'ucid':str(u),'date':grid+pd.Timedelta(days=2),
                               'yahoo_liquidity':volume.rolling(20,min_periods=20).median().values}))
    yy=p[['symbol','ucid','date']].merge(pd.concat(yc),on=['ucid','date'],how='left',validate='1:1')
    yy=yy.merge(feeds['nine'][['symbol','date','relative_return','feature_sv_relative_return']],on=['symbol','date'],validate='1:1')
    yy['yahoo_liquidity_return']=yy.yahoo_liquidity*yy.relative_return
    for c in ['yahoo_liquidity','yahoo_liquidity_return']:
        yy['feature_sv_'+c]=yy.groupby('date')[c].rank(pct=True)
    feeds['yahoo']=yy
    cols={k:[c for c in f if c.startswith('feature_')] for k,f in feeds.items()}
    for feed,f in feeds.items():
        f.to_parquet(ROOT/f'features_{feed}.parquet',index=False,compression='zstd')
    p.to_parquet(ROOT/'base_panel.parquet',index=False,compression='zstd')
    meta=pd.read_parquet(WORK/'base_historical_meta_models.parquet')
    meta.to_parquet(ROOT/'historical_meta_models.parquet',index=False,compression='zstd')
    keys.to_parquet(ROOT/'keys.parquet',index=False,compression='zstd')
    A=arm_columns(base)
    arms={'nine':A,'four':A,'yahoo':{'A0':base,'A1':A['A1'],
          'Y_LIQ':A['A1']+['feature_sv_yahoo_liquidity','feature_sv_yahoo_liquidity_return']}}
    manifest=dict(experiment='E173',registered='2026-11-14',target=H.T20,baseline='E133/E135 frozen raw22',
        fixed=H.FIXED,seeds=Q.SEEDS,arms=arms,features=cols,four_venues=FOUR,nine_venues=sorted(v.venue.unique()),
        timing='UTC prediction-day midnight; exact bar D-2; no stale carry or filling; inferred availability, not verified vintages',
        volume_surprise='log(sum pair-volume latest / median prior20 sums), identical pairs complete all21 calendar days',
        persistent='log(mean latest3 summed base volume / mean preceding14), identical complete pairs all17 days',
        relative_return='one-day USD arithmetic return using same pairs at both endpoints; subtract eligible-panel cross-sectional median',
        breadth='distinct eligible venues with V>1.5*prior20 venue median / venue count; stable 21day pairs within venue; >=2 venues',
        liquidity='median latest20 calendar days summed measured-USD spot volume; identical complete pairs all20 days',
        interactions='raw surprise * raw relative return; raw USD liquidity * raw relative return; then per-date percentile ranks',
        ranks='average-tie percentile rank over observed panel assets, preserve NaN',continuous='surprise, CS median-deviation, volume_return; train-only .01/.99 clip and mean/std scaling',
        missing='NaN retained in feature files; neutral .5 ranks and train-standardized zero continuous at estimator boundary; no indicators',
        yahoo='Yahoo-only source rows, own uninterrupted20day calendar history; reported USD Volume used directly, never close*Volume; separate Y_LIQ vs A1',
        train_protocol='130 training dates /22purged/130forecast dates; exact frozen12 E133/E135 blocks; block11 reused test, not virgin',
        settings='fixed all arms; no retuning, blends, model selection or stake changes',
        coverage='all468000 evaluation keys fixed across arms; separate 9vs4 A5-all-inputs finite common cohort; Yahoo paired common liquidity cohort',
        baseline_return_reuse='no equivalent shipped one-day relative-return input in raw22, so added once in A1',
        identity='E152 inferred unique ticker-to-ucid map and causal expanding volatility identity gate <=.18; REST venues survivor-biased',
        caveats='retrospective reconstructed sources can be revised; no verified vintage or live alpha claim; many comparisons exploratory',
        source_hashes={str(x):digest(x) for x in [SOURCE,BASE,WORK/'identity_gate.parquet',Path('data/crypto/spot/ohlcv.parquet')]},
        code_hashes={str(x):digest(x) for x in [Path(__file__),Path('src/data_mani/spot_volume_comparison.py')]})
    write_json(ROOT/'feature_manifest.json',manifest);write_json(ROOT/'point_in_time_audit.json',audit)
    coverage=[];block_coverage=[]
    for feed,f in feeds.items():
        z=f.merge(keys[['symbol','date','block']] if 'block' in keys else keys[['symbol','date']],on=['symbol','date'])
        for c in cols[feed]:
            coverage.append(dict(feed=feed,feature=c,coverage=float(z[c].notna().mean()),n=int(z[c].notna().sum())))
            for block,g in z.groupby('block'):
                block_coverage.append(dict(feed=feed,feature=c,block=int(block),coverage=float(g[c].notna().mean()),n=int(g[c].notna().sum())))
    pd.DataFrame(coverage).to_csv(ROOT/'feature_coverage.csv',index=False)
    pd.DataFrame(block_coverage).to_csv(ROOT/'feature_coverage_by_block.csv',index=False)
    write_json(ROOT/'BUILD_COMPLETE.json',dict(time=time.time(),rows=len(p),validation_rows=len(keys),audit_rows=sum(a['rows'] for a in audit)))


def load(feed):
    p=pd.read_parquet(ROOT/'base_panel.parquet')
    f=pd.read_parquet(ROOT/f'features_{feed}.parquet')
    return p.merge(f[['symbol','date']+[c for c in f if c.startswith('feature_')]],on=['symbol','date'],how='left',validate='1:1')


def splits(df):
    dates=np.array(sorted(df.loc[df.date<H.TEST_START,'date'].unique()))
    k=pd.read_parquet(ROOT/'keys.parquet')
    te=pd.to_datetime(k.loc[k.kind.eq('test'),'date']).unique()
    return H.blocks(dates)+[(11,dates[-(H.PURGE+H.TRAIN):-H.PURGE],te)]


def fit(model,njobs):
    m=json.loads((ROOT/'feature_manifest.json').read_text())
    for feed in ['nine','four','yahoo']:
        df=load(feed); B=splits(df)
        for arm,cols in m['arms'][feed].items():
            if feed!='nine' and arm in ['A0','A1'] and feed=='yahoo': continue # identical nine A1; A0 common all
            if feed=='four' and arm=='A0': continue
            for seed in Q.SEEDS[model]:
                dest=ROOT/'preds'/model/f'{feed}_{arm}_s{seed}.parquet'
                if dest.exists():continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                if arm=='A0':
                    # Reuse the existing baseline predictions and exact seed/config.
                    pred=Q.preds(model,f'B0_s{seed}')
                    k=pd.read_parquet(ROOT/'keys.parquet')
                    o=k[['symbol','date','kind']].merge(pred,on=['symbol','date'],validate='1:1')
                    if 'block' not in o:
                        blockmap={pd.Timestamp(d):i for i,_,te in B for d in te}
                        o['block']=o.date.map(blockmap)
                    assert len(o)==468000 and o.prediction.notna().all()
                    parts=[o[['symbol','date','prediction','block']]];configs=[]
                else:
                    parts=[];configs=[]
                    for i,tr,te in B:
                        a=df[df.date.isin(tr)];b=df[df.date.isin(te)]
                        X,Z,bounds=prepare_fold(a,b,cols)
                        pred=H.fit_predict(model,H.FIXED[model],seed,njobs,X,a[H.T20].to_numpy(),Z)
                        assert np.isfinite(pred).all()
                        o=b[['symbol','date']].copy();o['prediction']=pred;o['block']=i;parts.append(o)
                        configs.append(dict(block=i,train_first=str(min(tr)),train_last=str(max(tr)),test_first=str(min(te)),test_last=str(max(te)),
                                            train_rows=len(a),predict_rows=len(b),clipping=bounds,spread=float(np.std(pred))))
                out=pd.concat(parts,ignore_index=True)
                assert len(out)==468000
                # Numerai score is rank-based; preserve float64 predictions for reproducibility.
                tmp=dest.with_suffix('.partial');out.to_parquet(tmp,index=False,compression='zstd');tmp.replace(dest)
                write_json(dest.with_suffix('.json'),dict(model=model,feed=feed,arm=arm,seed=seed,columns=cols,params=H.FIXED[model],folds=configs))
                print('FIT',model,feed,arm,seed,flush=True)
    write_json(ROOT/f'FIT_COMPLETE_{model}.json',dict(time=time.time()))


def prediction_path(model,feed,arm,seed):
    if arm=='A0':feed='nine'
    if feed=='yahoo' and arm=='A1':feed='nine'
    return ROOT/'preds'/model/f'{feed}_{arm}_s{seed}.parquet'


def batch_metrics(g, columns):
    """Call the unchanged Numerai evaluator with a matrix of prediction columns."""
    from numerai_tools.scoring import numerai_corr, correlation_contribution
    corr=numerai_corr(g[columns],g[H.T20]).fillna(0.)
    ranks=g[columns].rank(method='average')
    spearman=ranks.corrwith(g[H.T20].rank(method='average')).fillna(0.)
    mmc=pd.Series(np.nan,index=columns)
    valid=g.meta_model.notna()
    if valid.sum()>=10:
        h=g.loc[valid].set_index('symbol')
        mmc=correlation_contribution(h[columns],h.meta_model,h[H.T20].copy())
    return corr,spearman,mmc


def score(model):
    m=json.loads((ROOT/'feature_manifest.json').read_text())
    p=pd.read_parquet(ROOT/'base_panel.parquet')[['symbol','date',H.T20]]
    meta=pd.read_parquet(ROOT/'historical_meta_models.parquet')[['symbol','date','prediction']].rename(columns={'prediction':'meta_model'})
    meta['date']=pd.to_datetime(meta.date)
    keys=pd.read_parquet(ROOT/'keys.parquet')[['symbol','date','block']]
    p=keys.merge(p,on=['symbol','date'],validate='1:1').merge(meta,on=['symbol','date'],how='left',validate='1:1')
    fs={k:pd.read_parquet(ROOT/f'features_{k}.parquet') for k in ['nine','four','yahoo']}
    masks=[]
    for feed in ['nine','four']:
        cols=[c for c in m['arms'][feed]['A5'] if c.startswith('feature_sv')]
        masks.append(fs[feed].set_index(['symbol','date'])[cols].notna().all(axis=1))
    common=(masks[0]&masks[1]).rename('common94').reset_index()
    ycols=['feature_sv_yahoo_liquidity','feature_sv_yahoo_liquidity_return']
    ymask=fs['yahoo'].set_index(['symbol','date'])[ycols].notna().all(axis=1)
    ncols=['feature_sv_relative_return','feature_sv_liquidity','feature_sv_liquidity_return']
    nmask=fs['nine'].set_index(['symbol','date'])[ncols].notna().all(axis=1)
    common=common.merge((ymask&nmask).rename('common_yahoo').reset_index(),on=['symbol','date'])
    p=p.merge(common,on=['symbol','date'],validate='1:1')
    descriptions={}; arrays=[]
    for feed,arms in m['arms'].items():
        for arm in arms:
            for seed in Q.SEEDS[model]:
                name=f'p_{feed}_{arm}_s{seed}'
                f=pd.read_parquet(prediction_path(model,feed,arm,seed))
                f=keys[['symbol','date']].merge(f[['symbol','date','prediction']],on=['symbol','date'],how='left',validate='1:1')
                assert len(f)==468000 and f.prediction.notna().all()
                arrays.append(f.prediction.rename(name));descriptions[name]=(feed,arm,seed)
    p=pd.concat([p.reset_index(drop=True),pd.concat(arrays,axis=1)],axis=1)
    assert p.symbol.notna().all()
    rows=[];checks=[]
    for scope in ['all','common94','common_yahoo']:
        columns=[c for c,(feed,arm,seed) in descriptions.items() if not (scope=='common94' and feed=='yahoo') and not (scope=='common_yahoo' and feed=='four')]
        d=p if scope=='all' else p[p[scope]]
        for j,(date,g) in enumerate(d.groupby('date',sort=True)):
            if len(g)<10:continue
            corr,spearman,mmc=batch_metrics(g,columns)
            # Real-date scalar-vs-batch identity checks, including meta-covered dates.
            if j in [0,300,800,1200,1500]:
                for c in [columns[0],columns[len(columns)//2],columns[-1]]:
                    h=g[['symbol',H.T20,'meta_model',c]].rename(columns={c:'prediction'})
                    scalar_c=E.corr_one(h);scalar_m=E.contrib_one(h.set_index('symbol'),'meta_model')
                    scalar_s=float(h.prediction.corr(h[H.T20],method='spearman'))
                    for a,b in [(corr[c],scalar_c),(spearman[c],scalar_s),(mmc[c],scalar_m)]:
                        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,equal_nan=True)
                    checks.append(dict(scope=scope,date=str(date),column=c,corr_delta=float(corr[c]-scalar_c),mmc_delta=float(mmc[c]-scalar_m)))
            for c in columns:
                feed,arm,seed=descriptions[c]
                rows.append(dict(model=model,feed=feed,arm=arm,seed=seed,scope=scope,date=date,block=int(g.block.iloc[0]),
                                 corr=float(corr[c]),spearman=float(spearman[c]),mmc=float(mmc[c]),payout=float(.1*corr[c]+mmc[c]),n=len(g)))
            if j%200==0:print('SCORE',model,scope,j,'dates',flush=True)
    dest=ROOT/f'scores_{model}.parquet';tmp=dest.with_suffix('.partial')
    pd.DataFrame(rows).to_parquet(tmp,index=False,compression='zstd');tmp.replace(dest)
    write_json(ROOT/f'evaluator_batch_identity_{model}.json',checks)


def report():
    S=pd.concat([pd.read_parquet(ROOT/f'scores_{m}.parquet') for m in MODELS],ignore_index=True)
    S['period']=np.where(S.block.eq(11),'test','dev')
    S.to_parquet(ROOT/'scores_by_date_seed.parquet',index=False,compression='zstd')
    group=['model','feed','arm','scope','period']
    daily=S.groupby(group+['date','block'],as_index=False)[['corr','spearman','mmc','payout','n']].mean()
    levels=daily.groupby(group,as_index=False).agg(validation_score=('corr','mean'),spearman=('spearman','mean'),mmc=('mmc','mean'),payout=('payout','mean'),dates=('date','size'),mean_assets=('n','mean'))
    levels.to_csv(ROOT/'validation_levels.csv',index=False)
    S.groupby(group+['seed'],as_index=False)[['corr','spearman','mmc','payout']].mean().to_csv(ROOT/'levels_by_seed.csv',index=False)
    daily.groupby(group+['block'],as_index=False)[['corr','spearman','mmc','payout','n']].mean().to_csv(ROOT/'levels_by_block.csv',index=False)
    pairs=[]; seedrows=[]; blockrows=[]
    for (model,feed,scope,period),g in S.groupby(['model','feed','scope','period']):
        arms=sorted(g.arm.unique())
        comparisons=[(a,'A0','vs_baseline') for a in arms if a!='A0']
        comparisons += [(a,arms[j-1],'incremental') for j,a in enumerate(arms) if j>1]
        if feed=='yahoo':comparisons += [('Y_LIQ','A1','yahoo_liquidity')]
        for arm,ref,kind in comparisons:
            x=g[g.arm.eq(arm)].merge(g[g.arm.eq(ref)],on=['seed','date','block'],suffixes=('','_ref'),validate='1:1')
            for metric in ['corr','spearman','mmc','payout']:
                x['delta']=x[metric]-x[metric+'_ref']
                bydate=x.groupby('date').delta.mean()
                mean,se,t=E.newey_west(bydate.to_numpy(),20);lo,hi=E.block_bootstrap_ci(bydate.to_numpy(),block=20,reps=2000,seed=173)
                ss=x.groupby('seed').delta.mean();bb=x.groupby('block').delta.mean()
                row=dict(model=model,feed=feed,scope=scope,period=period,arm=arm,reference=ref,comparison=kind,metric=metric,
                         delta=mean,nw_se=se,nw_t=t,bootstrap_lo=lo,bootstrap_hi=hi,dates=len(bydate),positive_seeds=int((ss>0).sum()),seeds=len(ss),positive_blocks=int((bb>0).sum()),blocks=len(bb))
                pairs.append(row)
                for seed,value in ss.items():seedrows.append(dict(row,seed=int(seed),seed_delta=value))
                for block,value in bb.items():blockrows.append(dict(row,block=int(block),block_delta=value))
    paired=pd.DataFrame(pairs);paired.to_csv(ROOT/'paired_deltas.csv',index=False)
    pd.DataFrame(seedrows).to_csv(ROOT/'paired_deltas_by_seed.csv',index=False)
    pd.DataFrame(blockrows).to_csv(ROOT/'paired_deltas_by_block.csv',index=False)
    # Direct nine-minus-four comparison on exactly the same covered rows.
    cross=[]
    a=S[S.scope.eq('common94') & S.feed.eq('nine')]
    b=S[S.scope.eq('common94') & S.feed.eq('four')]
    x=a.merge(b,on=['model','arm','seed','date','block','period'],suffixes=('','_four'),validate='1:1')
    for (model,arm,period),g in x.groupby(['model','arm','period']):
        for metric in ['corr','spearman','payout']:
            d=(g[metric]-g[metric+'_four']).groupby(g.date).mean();mean,se,t=E.newey_west(d,20)
            lo,hi=E.block_bootstrap_ci(d.to_numpy(),block=20,reps=2000,seed=173)
            cross.append(dict(model=model,arm=arm,period=period,metric=metric,nine_minus_four=mean,nw_t=t,bootstrap_lo=lo,bootstrap_hi=hi,dates=len(d)))
    pd.DataFrame(cross).to_csv(ROOT/'nine_vs_four_common.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for ax,model in zip(axes,MODELS):
        d=daily[daily.model.eq(model)&daily.feed.eq('nine')&daily.scope.eq('all')]
        for arm,g in d.groupby('arm'):ax.plot(g.date,g['corr'].cumsum(),label=arm)
        ax.set_title(model);ax.set_ylabel('Cumulative CORR');ax.legend(ncol=6)
    fig.tight_layout();fig.savefig(ROOT/'cumulative_corr.png',dpi=150);plt.close(fig)
    def table(d):return d.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
    primary=paired[(paired.feed=='nine')&(paired.scope=='all')&(paired.period=='dev')&(paired.metric=='corr')&(paired.comparison=='incremental')]
    leaders=primary.sort_values('delta',ascending=False).groupby('model',sort=False).head(1)
    body='<html><head><meta charset="utf-8"><style>body{font:13px Arial;color:#18202a;margin:25px}table{border-collapse:collapse;font-size:10px;width:100%;margin:12px 0}td,th{padding:5px;border-bottom:1px solid #ddd;text-align:right}h2{margin-top:28px}img{max-width:100%}</style></head><body>'
    body+='<h1>E173: Do matched spot-volume features improve forward-return prediction?</h1>'
    body+='<p>Exploratory fixed-config ablation. The validation score is mean daily Numerai CORR against target_binned_return_20. Spearman is reported separately. MMC and payout use the repository historical meta-model proxy. Conclusions concern this reconstructed panel; no deployment or official Numerai score is claimed.</p>'
    body+='<h2>Largest incremental development gains by model</h2>'+table(leaders[['model','arm','reference','delta','nw_t','bootstrap_lo','bootstrap_hi','positive_blocks','blocks','positive_seeds','seeds']])
    conclusions=[]
    for r in leaders.itertuples():
        test=paired[(paired.model==r.model)&(paired.feed=='nine')&(paired.scope=='all')&(paired.period=='test')&(paired.arm==r.arm)&(paired.reference==r.reference)&(paired.comparison=='incremental')&(paired.metric=='corr')]
        testdelta=float(test.delta.iloc[0]) if len(test) else np.nan
        stable=(r.delta>0 and testdelta>0 and r.positive_blocks>r.blocks/2 and r.positive_seeds>r.seeds/2)
        conclusions.append(f'{r.model}: largest incremental development change is {r.arm} versus {r.reference}, {r.delta:+.5f} CORR; test {testdelta:+.5f}. Positive in {r.positive_blocks}/{r.blocks} development blocks and {r.positive_seeds}/{r.seeds} seeds. '+('Direction holds in a majority of blocks/seeds and on test; remains exploratory.' if stable else 'Does not show a consistent gain across development blocks, seeds and test.'))
    body+='<p>'+('</p><p>'.join(html.escape(x) for x in conclusions))+'</p>'
    body+='<p>These leaders are selected on development data. Read their test, block and seed deltas below before interpreting them as improvements. A group that wins one block or seed only is unstable; multiplicity is not corrected, so all positive findings remain exploratory.</p>'
    body+='<h2>Design and timing</h2><p>Reuse E133/E135 frozen raw22 baseline, original target bins, 12 chronological folds, 130 training /22 purge /130 prediction dates, seeds0–4 for LightGBM and identity-activation MLP, deterministic ridge seed0. Baseline saved predictions reused after exact raw-input/target identity check. Fixed estimator settings across arms; no retuning. 468,000 validation keys; no coverage-based training deletion. Block11 is a reused test period, not virgin. Missing signals stay NaN in saved feature files; models see neutral rank .5 or zero for standardized continuous inputs. Clipping at training-only1/99 percentiles, training-only mean/std. All-date scores use the same rows in every arm. Common94 uses an identical complete A5 cohort for both venue versions and all arms. Dates with fewer than10 eligible assets are excluded from that secondary comparison only.</p>'
    body+='<p>Nine venues: Binance, Gate, KuCoin, Bitget, Bybit, Coinbase, OKX, MEXC, Hyperliquid. Four: Binance, Coinbase, OKX, KuCoin. Only spot bars, no derivatives or Yahoo mixed into venue totals. Exact D−2 bar joins follow stored inferred availability; historical publication timestamps and revisions are unverified. Existing causal identity gate and source price sanity checks retained. REST listings and source identities are retrospective and survivor-biased.</p>'
    body+='<h2>Feature construction and arms</h2><p>A1 adds one-day USD return minus cross-sectional median (no equivalent raw22 return feature). A2 adds log latest base-volume/prior20 median and log latest3 mean/preceding14 mean, plus clipped continuous surprise and its median-deviation. A3 adds raw surprise×relative return (rank and clipped continuous). A4 adds venue surge breadth (>1.5×prior20 median; minimum2 venues) and distinct venue count. A5 adds median latest20 USD volume and its raw interaction with relative return. All derived signals additionally percentile-ranked over observed panel assets per prediction day, preserving missingness. Every rolling ratio holds its eligible venue-pair set fixed across its full comparison window. Zero denominator or infinite log becomes missing; no epsilon substitution.</p>'
    body+='<p>Yahoo test adds only its liquidity and liquidity×the same nine-venue relative return to A1. Yahoo Volume is already USD; its20-day history is rolled independently using Yahoo-only rows, never stitched with venue data or multiplied by close. It has the same2-day lag.</p>'
    body+='<h2>Validation levels: fixed full panel</h2>'+table(levels[levels.scope.eq('all')])
    body+='<h2>Paired score deltas: fixed full panel</h2>'+table(paired[paired.scope.eq('all')&paired.metric.isin(['corr','spearman'])].drop(columns=['nw_se']))
    body+='<h2>Nine versus four venues on common coverage</h2>'+table(pd.DataFrame(cross))
    body+='<h2>Yahoo comparison on common coverage</h2>'+table(paired[paired.scope.eq('common_yahoo')&paired.feed.eq('yahoo')&paired.metric.isin(['corr','spearman'])])
    body+='<h2>Feature coverage on validation keys</h2>'+table(pd.read_csv(ROOT/'feature_coverage.csv'))
    body+='<h2>Cumulative validation CORR</h2><img src="data:image/png;base64,'+base64.b64encode((ROOT/'cumulative_corr.png').read_bytes()).decode()+'">'
    body+='<h2>Chronological block deltas</h2>'+table(pd.DataFrame(blockrows).query("scope == 'all' and metric == 'corr'")[['model','feed','arm','reference','comparison','period','block','block_delta']])
    body+='<h2>Seed deltas</h2>'+table(pd.DataFrame(seedrows).query("scope == 'all' and metric == 'corr'")[['model','feed','arm','reference','comparison','period','seed','seed_delta']])
    body+='<h2>Audit and saved artifacts</h2><p>Real-source prefix rebuilds at three cutoffs on12 assets per venue version are exact, with zero differing cells. Full-panel cross-sectional ranks use only that date. All hashes, mathematical definitions, feature columns and source limitations are in feature_manifest.json; per-fold training ranges, clipping bounds and spreads are saved next to each validation prediction file. Statistical deltas pair the same dates and seed first, then average seeds per date; Newey–West lag20 and moving-block20/bootstrap2000 intervals account for overlapping targets, but do not remove model-selection or reconstruction bias. The fixed-budget design requested here omits retuning and coverage/tenure controls; no adoption claim is justified without further validation.</p></body></html>'
    path=ROOT/'E173_spot_volume_comparison_EXPLORATORY.html';path.write_text(body)
    subprocess.run(['ssh','-o','BatchMode=yes','lovelace','/home/danielco/miniconda3/bin/python3',str(REPO/'src/analysis/html_pdf.py'),str(path)],check=True)
    pdf=path.with_suffix('.pdf');assert pdf.exists() and pdf.stat().st_size>10000
    write_json(ROOT/'COMPLETE.json',dict(time=time.time(),pdf=str(pdf),scores=len(S)))
    print('COMPLETE',pdf,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['build','fit','score','report']);ap.add_argument('--model',choices=MODELS);ap.add_argument('--n-jobs',type=int,default=4);a=ap.parse_args()
    if a.phase=='build':build()
    elif a.phase=='fit':fit(a.model,a.n_jobs)
    elif a.phase=='score':score(a.model)
    else:report()

if __name__=='__main__':main()
