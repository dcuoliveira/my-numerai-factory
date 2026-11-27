"""Independent E212 completion audit against native scoring and routing helpers."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from numerai_tools.scoring import numerai_corr,correlation_contribution
from experiments import e212_volatility99_tsfm as S
from src.prediction import meta_resid as MR


def run():
    root=S.ROOT;complete=json.loads((root/'COMPLETE.json').read_text());c=S.verify()
    assert c['models']==['chronos2','t0','timestral']
    readiness=json.loads((root/'EMBEDDINGS_READY.json').read_text())
    assert S.sha(root/'cache/common_rows.npy')==readiness['mask_sha256']
    protocols=0
    for path in (root/'blocks').glob('*/*/protocol.json'):
        protocol=json.loads(path.read_text());wall=pd.Timedelta(days=S.WALLS[int(protocol['target'])])
        assert pd.Timestamp(protocol['last_resolved'])+wall<=pd.Timestamp(protocol['origin'])
        assert pd.Timestamp(protocol['training_last'])+wall<=pd.Timestamp(protocol['validation_first'])
        protocols+=1
    assert protocols==88
    assert complete['actual_rows']==27504
    table=pd.read_parquet(root/'all_trials.parquet');configs=json.loads((root/'trials.json').read_text())
    dates=pd.read_parquet(root/'score_dates.parquet');panel=pd.read_parquet(root/'panel.parquet')
    coverage=pd.read_parquet(root/'coverage.parquet');forecasts=pd.read_parquet(root/'standalone_forecasts.parquet')
    routed=pd.read_parquet(root/'cache/routed_standalone.parquet')
    assert table.shape[0]==27504 and len(configs)==13752
    assert not table.duplicated(['id','evaluation_target']).any()
    assert set(table.evaluation_target)=={20,60} and table.status.eq('success').all()
    assert table.dates.nunique()==1 and int(table.dates.iloc[0])==len(dates)
    assert dates.assets.min()>=10 and panel.row.equals(coverage.row) and panel.row.equals(forecasts.row)
    assert panel.row.equals(routed.row)
    names=[c['members'][0] for c in configs if c['kind']=='standalone'];assert len(names)==16
    for horizon in [20,60]:
        part=table[table.evaluation_target==horizon]
        assert len(part)==13752 and (part.kind=='standalone').sum()==16
        assert part.payout.is_monotonic_decreasing
    index={v['id']:i for i,v in enumerate(configs)}
    arrays={h:np.load(root/f'cache/daily_scores/t{h}/corr_mmc.npy',mmap_mode='r') for h in [20,60]}
    for array in arrays.values():assert array.shape==(len(dates),len(configs),2) and np.isfinite(array).all()
    candidates=np.ascontiguousarray(forecasts[names].to_numpy(float))
    assert np.isfinite(routed[names].to_numpy()).all()
    meta=pd.read_parquet(S.REPO/'data'/c['dataset']/'historical_meta_models.parquet').reset_index()
    combos=S.definitions(16);weights=np.zeros((16,len(combos)))
    for j,members in enumerate(combos):weights[list(members),j]=1/len(members)
    chosen=set()
    for horizon in [20,60]:
        chosen.update(table[(table.evaluation_target==horizon)&(table.kind=='ensemble')].head(3).id)
    max_corr_error=0.;max_mmc_error=0.;checks=[]
    for day in sorted({0,len(dates)//2,len(dates)-1}):
        date=dates.date.iloc[day]
        rows=np.flatnonzero((panel.date==date).to_numpy()&coverage.causal_common.to_numpy())
        evaluation=coverage.evaluation_available.to_numpy()[rows];evalrows=rows[evaluation]
        assert len(evalrows)==dates.assets.iloc[day]
        values=np.ascontiguousarray(candidates[rows]);ranks=(rankdata(values,axis=0,method='average')-.5)/len(rows)
        lagged,snapshot=MR.lagged_meta(meta,panel.symbol.iloc[rows],date)
        for j in range(8):
            native,_=MR.residualise(values[:,j],lagged)
            np.testing.assert_allclose(native,values[:,j+8],rtol=1e-12,atol=1e-12)
        full=np.flatnonzero(panel.date.eq(date).to_numpy());baseline=S.rank_normalise(coverage.baseline.to_numpy()[full])
        covered=coverage.causal_common.to_numpy()[full]
        for name in names:
            actual=routed[name].to_numpy()[full]
            expected=S.route_slots(baseline,np.where(covered,forecasts[name].to_numpy()[full],np.nan))
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(actual[~covered],baseline[~covered])
            np.testing.assert_array_equal(np.sort(actual[covered]),np.sort(baseline[covered]))
        selection={configs[j]['id']:values[:,j] for j in [0,3,7,8,15]}
        for identity in chosen:
            ci=index[identity];combo=(ci-16)//2;first=combo//256*256;last=min(first+256,len(combos))
            blend=ranks@weights[:,first:last]
            if configs[ci]['post_residualized']:
                transformed,_=S.residual_matrix(blend,lagged)
                native,_=MR.residualise(blend[:,combo-first],lagged)
                np.testing.assert_allclose(native,transformed[:,combo-first],rtol=1e-12,atol=1e-12)
                prediction=transformed[:,combo-first]
            else:prediction=blend[:,combo-first]
            selection[identity]=prediction
        pred=pd.DataFrame({k:v[evaluation] for k,v in selection.items()},index=evalrows)
        for horizon in [20,60]:
            target=pd.Series(panel[f'target_binned_return_{horizon}'].to_numpy()[evalrows],index=evalrows)
            benchmark=pd.Series(panel.meta_model.to_numpy()[evalrows],index=evalrows)
            corr=numerai_corr(pred,target).fillna(0.)
            mmc=correlation_contribution(pred,benchmark,target)
            for identity in selection:
                expected=arrays[horizon][day,index[identity]]
                error=np.abs(np.array([corr[identity],mmc[identity]])-expected)
                max_corr_error=max(max_corr_error,float(error[0]));max_mmc_error=max(max_mmc_error,float(error[1]))
                np.testing.assert_allclose([corr[identity],mmc[identity]],expected,rtol=2e-6,atol=2e-7)
            checks.append(dict(date=str(date),target=horizon,trials=len(selection),assets=len(evalrows),snapshot=str(snapshot)))
    for horizon in [20,60]:
        part=table[table.evaluation_target==horizon]
        for identity in list(chosen)+[configs[0]['id']]:
            series=arrays[horizon][:,index[identity],:]
            metrics=S.payout_metrics(series[:,0],series[:,1]);row=part[part.id==identity].iloc[0]
            for key,value in metrics.items():np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-12,equal_nan=True)
        expected_corr=np.corrcoef((.1*arrays[horizon][:,:16,0]+arrays[horizon][:,:16,1]),rowvar=False)
        order=[index[x] for x in part[part.kind=='standalone'].id]
        actual=pd.read_csv(root/f'daily_payout_correlation_t{horizon}.csv',index_col=0).to_numpy()
        np.testing.assert_allclose(actual,expected_corr[np.ix_(order,order)],rtol=1e-12,atol=1e-12)
        for name in ['payout_heatmap','all_standalones','top3_plus_standalones']:
            assert (root/f'{name}_t{horizon}.png').stat().st_size>10000
    pdf=root/'E212_volatility99_tsfm.pdf';assert S.sha(pdf)==complete['report_sha256']
    text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
    assert 'Evaluation target20' in text and 'Evaluation target60' in text and text.count('Top THREE')>=2
    assert 'Frozen protocol' not in text and 'Frozen configuration' not in text
    outcome=dict(outcome='PASS WITH CONDITIONS',native_score_checks=checks,max_native_corr_error=max_corr_error,
        max_native_mmc_error=max_mmc_error,verified_fold_protocols=protocols,trial_rows=len(table),scored_dates=len(dates),scoreable_asset_dates=int(dates.assets.sum()),
        routing='Full routed matrix finite; native routing and both exact rank-slot invariants replayed on first/middle/last scored dates',
        residualization='Native D-5 standalone and ensemble post transforms replayed on identical causal universes',
        plots='Heatmaps reconstructed from daily payout arrays; selected trial summaries reproduced; all six plot files present',
        pdf_sha256=complete['report_sha256'],conditions=c['limitations'])
    S.save(root/'independent_completion_audit.json',outcome)
    print(json.dumps({k:v for k,v in outcome.items() if k not in ['conditions','native_score_checks']},indent=2))


if __name__=='__main__':run()
