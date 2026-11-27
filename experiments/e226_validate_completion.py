"""Independent reconstruction audit of E226 dataset, heads and report additions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from experiments import e226_volatility_cs_ridge as R
from experiments import e212_volatility99_tsfm as S
from experiments.e211_cross_target_tsfm import fold_panel
from src.prediction import meta_resid as MR
from src.live.submit import rank_normalise
from src.analysis.asset_momentum import route_slots


def run():
    manifest=json.loads((R.DATA/'manifest.json').read_text())
    plan=json.loads((R.B.SOURCE/'registration.json').read_text())
    assert plan['selected_specs']==[list(s) for s in R.B.GROUPS.values()]
    for rel in ('src/data_mani/volatility_proxies.py','src/data_mani/build_volatility_replacement.py'):
        assert R.sha(R.REPO/rel)==plan['code_hashes'][rel],rel
    assert manifest['price_sha256']==R.sha(manifest['price_path'])
    for name,h in manifest['output_hashes'].items():assert R.sha(R.DATA/name)==h,name
    assert manifest['builder_sha256']==R.sha(R.__file__)
    base=pd.read_parquet(R.BASE/'train.parquet');new=pd.read_parquet(R.DATA/'train.parquet')
    pd.testing.assert_frame_equal(new[base.columns],base)
    names=manifest['feature_sets']['CS99'];both=manifest['feature_sets']['TS_CS198'];assert len(names)==99 and len(both)==198
    assert np.isin(new[both].to_numpy()[np.isfinite(new[both].to_numpy())],[0,.25,.5,.75,1]).all()
    raw=pd.read_parquet(R.ROOT/'raw_volatility.parquet');pd.testing.assert_index_equal(raw.index,base.index)
    counts=raw.notna().groupby(level='date').sum().reindex(raw.index.get_level_values('date')).to_numpy()
    np.testing.assert_array_equal(np.isfinite(new[names]),np.isfinite(raw)&(counts>=2))
    pd.testing.assert_frame_equal(pd.read_parquet(R.DATA/'historical_meta_models.parquet'),pd.read_parquet(R.BASE/'historical_meta_models.parquet'))
    # Independently calculate every CS feature on three complete panel dates.
    unique=np.sort(new.index.get_level_values('date').unique());cs_dates=[]
    for d in unique[[len(unique)//4,len(unique)//2,-1]]:
        frame=raw.xs(d,level='date');expected=new.xs(d,level='date')
        for name in names:
            finite=frame[name].notna();x=frame.loc[finite,name]
            ranks=x.rank(method='average');pct=(ranks-.5)/len(x)
            values=np.select([pct<=.05,pct<=.25,pct<=.75,pct<=.95],[0,.25,.5,.75],default=1).astype(np.float32)
            np.testing.assert_array_equal(values,expected.loc[x.index,name])
        cs_dates.append(str(d))
    panel=pd.read_parquet(S.ROOT/'panel.parquet');common=np.load(S.SCR/'common_rows.npy')
    newfeatures=new.reset_index().set_index(['symbol','date'])
    checked=0;max_error=0.;fits=[]
    for arm in ('CS99','TS_CS198'):
        features=manifest['feature_sets'][arm]
        p=panel.drop(columns=[f for f in features if f in panel]).merge(newfeatures[features].reset_index(),on=['symbol','date'],how='left',sort=False,validate='1:1')
        np.testing.assert_array_equal(p.row,panel.row)
        for h in (20,60):
            name=f'{arm}_t{h}';pred=pd.read_parquet(R.ROOT/'fits'/name/'predictions.parquet').set_index('row')
            paths=sorted((R.ROOT/'blocks'/name).glob('*/protocol.json'))
            assert len(paths)==11
            for path in paths:
                original=S.ROOT/'blocks'/f'feature_ridge_t{h}'/path.parent.name/'protocol.json'
                assert json.loads(path.read_text())==json.loads(original.read_text())
                pars=json.loads((path.parent/'parameters.json').read_text())
                oldpars=json.loads((S.SCR/'blocks'/f'feature_ridge_t{h}'/path.parent.name/'parameters.json').read_text())
                assert {r['asset']:r['training_rows'] for r in pars}=={r['asset']:r['training_rows'] for r in oldpars}
            for path in [paths[0],paths[len(paths)//2],paths[-1]]:
                protocol=json.loads(path.read_text());start=int(p.loc[p.date.eq(pd.Timestamp(protocol['origin'])),'step'].iloc[0]);q,protocol=fold_panel(p,start,h)
                params=sorted((path.parent/'ridge').glob('asset_*.npz'))
                for saved in params[:2]:
                    asset=saved.stem.removeprefix('asset_');g=q[q.ucid.astype(str).eq(asset)]
                    with np.load(saved) as z:
                        tr=g.index[common[g.index]&q.final_fit.to_numpy()[g.index]&np.isfinite(q[f'target_binned_return_{h}'].to_numpy()[g.index])].to_numpy()
                        te=g.index[common[g.index]&q.test.to_numpy()[g.index]].to_numpy()
                        np.testing.assert_array_equal(z['training_rows'],tr)
                        x=q.loc[tr,features].to_numpy(float);center=x.mean(0);scale=x.std(0);scale[scale<R.N.FLOOR]=1
                        np.testing.assert_array_equal(center,z['center']);np.testing.assert_array_equal(scale,z['scale'])
                        assert float(z['alpha']) in R.N.ALPHAS
                        xx=(x-center)/scale;v=(q.loc[te,features].to_numpy(float)-center)/scale
                        k=v@xx.T;kc=k-z['kernel_train_mean'][None,:]-k.mean(1)[:,None]+z['kernel_grand_mean']
                        forecast=(kc@z['dual_weights']+z['intercept']).ravel()
                        actual=pred.prediction.reindex(q.loc[te,'row']).to_numpy()
                        np.testing.assert_allclose(forecast,actual,atol=1e-12,rtol=1e-11)
                        max_error=max(max_error,float(np.max(np.abs(forecast-actual))))
                        # Replay validation selection over all33 alphas, independent of saved final weights.
                        tt=g.index[common[g.index]&q.train.to_numpy()[g.index]&np.isfinite(q[f'target_binned_return_{h}'].to_numpy()[g.index])]
                        va=g.index[common[g.index]&q.validation.to_numpy()[g.index]&np.isfinite(q[f'target_binned_return_{h}'].to_numpy()[g.index])]
                        xx=q.loc[tt,features].to_numpy(float);center=xx.mean(0);scale=xx.std(0);scale[scale<R.N.FLOOR]=1
                        vv=(q.loc[va,features].to_numpy(float)-center)/scale;xx=(xx-center)/scale
                        trial,_=R.N.dual_ridge(xx@xx.T,vv@xx.T,q.loc[tt,f'target_binned_return_{h}'].to_numpy())
                        losses=((trial-q.loc[va,f'target_binned_return_{h}'].to_numpy()[:,None])**2).mean(0)
                        assert float(R.N.ALPHAS[losses.argmin()])==float(z['alpha'])
                        checked+=1
            fits.append(name)
    forecasts=pd.read_parquet(R.ROOT/'standalone_forecasts.parquet');routed=pd.read_parquet(R.ROOT/'routed_forecasts.parquet');coverage=pd.read_parquet(S.ROOT/'coverage.parquet')
    configs=json.loads((R.ROOT/'trials.json').read_text());streamnames=[c['members'][0] for c in configs]
    meta=pd.read_parquet(R.BASE/'historical_meta_models.parquet').reset_index()
    for d in pd.read_parquet(S.ROOT/'score_dates.parquet').date.iloc[[0,234,-1]]:
        idx=np.flatnonzero(panel.date.eq(d));covered=coverage.causal_common.to_numpy()[idx];rows=idx[covered]
        lagged,snapshot=MR.lagged_meta(meta,panel.loc[rows,'symbol'],d)
        assert snapshot is None or snapshot<=d-pd.Timedelta(days=5)
        for j in range(4):
            native,_=MR.residualise(forecasts.loc[rows,streamnames[j]].to_numpy(),lagged)
            np.testing.assert_allclose(native,forecasts.loc[rows,streamnames[j+4]],atol=1e-12,rtol=1e-12)
        br=rank_normalise(coverage.baseline.to_numpy()[idx])
        for name in streamnames:
            expected=route_slots(br,np.where(covered,forecasts.loc[idx,name],np.nan))
            np.testing.assert_array_equal(expected,routed.loc[idx,name])
    complete=json.loads((R.ROOT/'COMPLETE.json').read_text())
    for path,h in complete['original_numerical_hashes'].items():assert R.sha(path)==h,path
    table=pd.read_parquet(S.ROOT/'report_comparison_trials.parquet')
    assert len(table)==27520 and not table.duplicated(['id','evaluation_target']).any()
    for h in (20,60):
        rows=table[(table.evaluation_target==h)&table.kind.eq('standalone')]
        assert len(rows)==24 and (rows.model=='feature_ridge').sum()==12
        assert set(rows.loc[rows.model.eq('feature_ridge'),'information_set'])=={'TS99','CS99','TS_CS198'}
    report=S.ROOT/'E212_volatility99_tsfm.pdf';assert R.sha(report)==complete['report_sha256']
    result=dict(outcome='PASS WITH CONDITIONS',dataset_exact_TS_keys_targets=True,independent_CS_replay_dates=cs_dates,fit_names=fits,all44_fold_protocols_and_asset_training_counts_matched=True,independent_scaler_prediction_and33_alpha_replays=checked,max_prediction_replay_error=max_error,independent_native_residual_and_routing_replays=True,original_E212_forecasts_trials_hashes_unchanged=True,report_standalones_per_target=24,report_ridge_variants_per_target=12,pdf_sha256=R.sha(report),conditions=json.loads((R.ROOT/'audit.json').read_text())['conditions'])
    # Add verified task identities for the repository's checkpoint indexer.
    for name in fits:
        marker=R.ROOT/'fits'/name/'COMPLETE.json'
        info=json.loads(marker.read_text());info['task']=dict(tag=name,model='ridge')
        R.save(marker,info)
    R.save(R.ROOT/'independent_completion_audit.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
