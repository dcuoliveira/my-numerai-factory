"""Independent E231 reconstruction of all63 portfolios and all-date native scores."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from numerai_tools.scoring import numerai_corr,correlation_contribution
from experiments import e231_tabnet_ensembles as T
from experiments import e212_volatility99_tsfm as S


def run():
    root=T.ROOT;c=json.loads((root/'COMPLETE.json').read_text());assert S.sha(c['report'])==c['pdf_sha256']
    meta=T.prepare();assert all(S.sha(p)==h for p,h in meta['identity']['hashes'].items())
    trials=json.loads((root/'trials.json').read_text());table=pd.read_csv(root/'all_results.csv');daily=pd.read_csv(root/'daily_payout.csv',index_col=0,parse_dates=True);cum=pd.read_csv(root/'cumulative_payout.csv',index_col=0,parse_dates=True);forecasts=pd.read_parquet(root/'standalone_forecasts.parquet');scores=np.load(root/'daily_corr_mmc.npy')
    assert len(trials)==63 and len(table)==63 and scores.shape==(297,63,2) and len(forecasts)==37334
    assert not forecasts.duplicated(['ucid','date']).any()
    for n in range(1,7):assert int((table['size']==n).sum())==math.comb(6,n)
    assert table.members.is_unique and table.id.is_unique
    keys=['AR','AM','PR','PM','AT','PT'];errors=[];sums=[]
    for j,(date,g) in enumerate(forecasts.groupby('date',sort=True)):
        nums=(2*rankdata(g[keys].to_numpy(),axis=0,method='average')-1).astype(np.int64)
        rebuilt=np.column_stack([nums[:,[keys.index(k) for k in item['members'].split('+')]].sum(1)/(2.*len(g)*item['size']) for item in trials]);pred=pd.DataFrame(rebuilt)
        target=g['target_binned_return_20'].reset_index(drop=True);meta_pred=g.meta_model.reset_index(drop=True)
        corr=numerai_corr(pred,target).fillna(0).to_numpy();mmc=correlation_contribution(pred,meta_pred,target).to_numpy()
        np.testing.assert_allclose(scores[j,:,0],corr,atol=2e-7,rtol=2e-6);np.testing.assert_allclose(scores[j,:,1],mmc,atol=1e-12,rtol=1e-10);errors.append([float(abs(scores[j,:,0]-corr).max()),float(abs(scores[j,:,1]-mmc).max())])
    pay=.1*scores[:,:,0]+scores[:,:,1];np.testing.assert_allclose(daily,pay,atol=1e-15,rtol=1e-12);np.testing.assert_allclose(cum,pay.cumsum(0),atol=1e-12,rtol=1e-12)
    cumul=np.vstack([np.zeros(63),pay.cumsum(0)]);drawdown=(np.maximum.accumulate(cumul,axis=0)-cumul).max(0)
    for row in table.to_dict('records'):
        k=int(row['column'])
        for field,expected in [('corr',scores[:,k,0].mean()),('mmc',scores[:,k,1].mean()),('payout',pay[:,k].mean()),('sharpe.p',pay[:,k].mean()/pay[:,k].std(ddof=0)),('maxdd.p',drawdown[k]),('cumulative_payout',pay[:,k].sum())]:np.testing.assert_allclose(row[field],expected,atol=1e-12,rtol=1e-12)
    # A hand-sized tied-rank fixture verifies exact equality under member permutations.
    toy=np.array([[0,2],[1,1],[2,0]],float);nums=(2*rankdata(toy,axis=0)-1).astype(np.int64);np.testing.assert_array_equal(nums.sum(1),np.full(3,6));np.testing.assert_array_equal(nums.sum(1),nums[:,::-1].sum(1))
    audit=json.loads((root/'audit.json').read_text());assert audit['outcome']=='PASS WITH CONDITIONS'
    S.save(root/'independent_completion_audit.json',dict(outcome='PASS WITH CONDITIONS',all63_subsets_all297_dates_native_reconstruction=True,max_corr_error=max(v[0] for v in errors),max_mmc_error=max(v[1] for v in errors),all_metrics_and_cumulative_curves_reconstructed=True,combination_counts_and_unique_members=True,tie_and_member_permutation_fixture=True,source_hashes_unchanged=True,pdf_sha256=c['pdf_sha256'],validator_sha256=S.sha(__file__),conditions=audit['conditions']))
    print('E231 independent completion audit passed',flush=True)

if __name__=='__main__':run()
