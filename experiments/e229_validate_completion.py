import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from numerai_tools.scoring import numerai_corr,correlation_contribution
from experiments import e212_volatility99_tsfm as S
r=S.REPO/'results/reports/crypto/e229_momentum_volatility_ensembles';s=S.REPO/'results/reports/crypto/e228_momentum_volatility_correlation'
t=pd.read_csv(r/'all_results.csv');a=np.load(r/'daily_corr_mmc.npy');d=pd.read_parquet(r/'daily_payout.parquet');b=pd.read_parquet(s/'common_keys.parquet');f=pd.read_parquet(s/'selected_forecasts.parquet');mapping=pd.read_csv(r/'member_mapping.csv')
assert len(t)==1023 and len(t[t.kind.eq('Ensemble')])==1013
assert len(set(t.members))==1023
for n in range(2,11):
 import math
 assert (t['size'].eq(n)).sum()==math.comb(10,n)
assert a.shape==(297,1023,2)
ordered=t.sort_values('column')
np.testing.assert_array_equal(d.columns[1:],ordered.id)
pay=.1*a[:,:,0]+a[:,:,1];np.testing.assert_allclose(d.iloc[:,1:],pay,atol=0,rtol=0)
np.testing.assert_allclose(t.payout,pay.mean(0)[t.column],atol=1e-15)
np.testing.assert_allclose(t['sharpe.p'],(pay.mean(0)/pay.std(0))[t.column],atol=1e-14)
cum=np.vstack([np.zeros(1023),pay.cumsum(0)]);dd=(np.maximum.accumulate(cum,axis=0)-cum).max(0)
np.testing.assert_allclose(t['maxdd.p'],dd[t.column],atol=1e-13)
plot=pd.read_csv(r/'plot_cumulative_payout.csv',index_col=0)
np.testing.assert_allclose(plot,d.set_index('date')[plot.columns].cumsum(),atol=1e-13)
checks=[]
# Independent native scorer reconstructs the selected winner from source ranks on EVERY date.
winner=t.iloc[0];cols=winner.members.split('+');vals=[]
for date,idx in b.groupby('date',sort=True).indices.items():
 raw=f.loc[idx,cols].to_numpy();ranks=rankdata(raw,axis=0,method='average');pred=(2*ranks-1).astype(np.int64).sum(1)/(2*len(idx)*len(cols))
 target=b.target_binned_return_20.iloc[idx].reset_index(drop=True);meta=b.meta_model.iloc[idx].reset_index(drop=True)
 nc=float(numerai_corr(pd.DataFrame({'blend':pred}),target).iloc[0]);nm=float(correlation_contribution(pd.DataFrame({'blend':pred}),meta,target).iloc[0]);vals.append([nc,nm])
np.testing.assert_allclose(vals,a[:,int(winner.column),:],atol=2e-7,rtol=2e-6)
manifest=json.loads((r/'source_manifest.json').read_text());assert all(S.sha(p)==h for p,h in manifest['input_hashes'].items())
S.save(r/'independent_completion_audit.json',dict(outcome='PASS WITH CONDITIONS',count_by_size_exact=True,all_metrics_reconstructed=True,plot_series_reconstructed=True,top_winner_all_297_dates_native_parity=True,max_native_error=float(np.max(np.abs(np.array(vals)-a[:,int(winner.column),:]))),source_hashes_unchanged=True,conditions=json.loads((r/'audit.json').read_text())['conditions']))
print(t.head(3)[['members','payout']]);print('independent audit passed')
