"""Independent E230 publication metrics and all-date native scoring replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json
import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr,correlation_contribution
from experiments import e230_dataset as D
from experiments import e212_volatility99_tsfm as S


def run():
    r=D.ROOT;complete=json.loads((r/'COMPLETE.json').read_text());assert complete['status']=='success';assert S.sha(complete['report'])==complete['report_sha256']
    a=json.loads((r/'audit.json').read_text());assert a['outcome']=='PASS WITH CONDITIONS'
    assert S.sha(r/'dataset_audit.json')==a['independent_dataset_audit_sha256']
    prepared=json.loads((r/'PREPARED.json').read_text())
    for path,h in {**prepared['input_hashes'],**prepared['output_hashes']}.items():assert S.sha(path)==h,path
    manifest=json.loads((r/'scoring_manifest.json').read_text());assert all(S.sha(p)==h for p,h in manifest['source_hashes'].items())
    names=manifest['model_order'];array=np.load(r/'daily_corr_mmc.npy');daily=pd.read_csv(r/'daily_payout.csv',index_col=0,parse_dates=True);cumulative=pd.read_csv(r/'cumulative_payout.csv',index_col=0,parse_dates=True);t=pd.read_csv(r/'results.csv').set_index('model')
    assert array.shape==(297,8,2) and len(t)==8 and all(arm in t.index for arm in D.ARMS)
    np.testing.assert_array_equal(daily.columns,names);pay=.1*array[:,:,0]+array[:,:,1];np.testing.assert_allclose(daily,pay,atol=1e-15);np.testing.assert_allclose(cumulative,pay.cumsum(0),atol=1e-12)
    c=np.vstack([np.zeros(len(names)),pay.cumsum(0)]);drawdown=(np.maximum.accumulate(c,axis=0)-c).max(0)
    for j,name in enumerate(names):
        for col,value in [('corr',array[:,j,0].mean()),('mmc',array[:,j,1].mean()),('payout',pay[:,j].mean()),('sharpe.p',pay[:,j].mean()/pay[:,j].std(ddof=0)),('maxdd.p',drawdown[j]),('cumulative_payout',pay[:,j].sum())]:np.testing.assert_allclose(t.loc[name,col],value,atol=1e-12,rtol=1e-12)
    forecasts=pd.read_parquet(r/'scored_forecasts.parquet');assert len(forecasts)==37334 and not forecasts.duplicated(['ucid','date']).any()
    errors=[]
    for j,(date,g) in enumerate(forecasts.groupby('date',sort=True)):
        predictions=g[names].reset_index(drop=True);target=g[D.TARGET].reset_index(drop=True);meta=g.meta_model.reset_index(drop=True)
        nc=numerai_corr(predictions,target).fillna(0).to_numpy();nm=correlation_contribution(predictions,meta,target).to_numpy()
        np.testing.assert_allclose(array[j,:,0],nc,atol=2e-7,rtol=2e-6);np.testing.assert_allclose(array[j,:,1],nm,atol=1e-12,rtol=1e-10);errors.append([float(np.max(np.abs(array[j,:,0]-nc))),float(np.max(np.abs(array[j,:,1]-nm)))])
    report_text=__import__('subprocess').check_output(['pdftotext',complete['report'],'-'],text=True)
    assert 'E230' in report_text and '134' in report_text and 'Frozen protocol' not in report_text
    S.save(r/'independent_completion_audit.json',dict(outcome='PASS WITH CONDITIONS',all8_streams_all297_dates_native_parity=True,max_corr_error=max(v[0] for v in errors),max_mmc_error=max(v[1] for v in errors),all_report_metrics_and_cumulative_series_reconstructed=True,all37334_keys_unique=True,source_dataset_forecast_hashes_unchanged=True,pdf_sha256=complete['report_sha256'],validator_sha256=S.sha(__file__),conditions=prepared['conditions']))
    print('E230 completion independent audit passed',flush=True)

if __name__=='__main__':run()
