"""E209 selected definitions, causal reconstruction and variable-width E205 adapter."""
import numpy as np
import pandas as pd
import pytest
from experiments import e209_asset_volatility as E
from src.analysis import asset_momentum as M
from src.data_mani.build_volatility_replacement import estimate_asset


def prices(seed=209):
    rng=np.random.default_rng(seed)
    index=pd.date_range('2019-01-01',periods=1200)
    c=np.exp(np.cumsum(rng.normal(0,.02,len(index))))
    o=c*np.exp(rng.normal(0,.01,len(index)))
    return pd.DataFrame(dict(open=o,close=c,high=np.maximum(o,c)*1.02,low=np.minimum(o,c)*.98),index=index)


def test_selected_nine_match_existing_pre_cross_section_definitions_and_prefix():
    bars=prices();benchmark={1:prices(1).close,1027:prices(1027).close}
    dates=pd.bdate_range('2020-01-01','2022-04-14')
    grids,nine=E.evaluate_asset(dates,bars,benchmark)
    expected=estimate_asset(dates,bars,benchmark)
    np.testing.assert_allclose(nine,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
    assert len(grids)==9 and all(x.shape==(len(dates),11) for x in grids.values())
    cut=pd.Timestamp('2021-08-02');keep=dates<=cut
    prefix,pnine=E.evaluate_asset(dates[keep],bars.loc[bars.index<cut],
                                 {i:s.loc[s.index<cut] for i,s in benchmark.items()})
    np.testing.assert_array_equal(pnine,nine[keep])
    for name,value in grids.items():
        np.testing.assert_array_equal(prefix[name],value[keep])
        assert set(np.unique(value[np.isfinite(value)])) <= {0.,.25,.5,.75,1.}


@pytest.mark.parametrize('width',[9,11,132,141])
def test_adapter_fits_joint_inputs_and_excludes_query_labels(tmp_path,monkeypatch,width):
    rng=np.random.default_rng(width)
    dates=pd.bdate_range('2020-01-01',periods=282).to_numpy()
    frame=pd.DataFrame([(a,d) for a in range(2) for d in dates],columns=['ucid','date'])
    frame['row']=np.arange(len(frame));frame['covered']=True
    frame[E.B.TARGET]=rng.uniform(0,1,len(frame))
    frame=pd.concat([frame,E.feature_frame(rng.normal(size=(len(frame),width)))],axis=1)
    monkeypatch.setattr(E.B.M,'WINDOWS',list(range(width)))
    monkeypatch.setattr(E.B,'ROOT',tmp_path/'a')
    first=pd.read_parquet(E.B.fit_fold('ridge','test',0,dates[:130],dates[152:],frame,[.01,1.,100.]))
    frame.loc[frame.date.isin(dates[152:]),E.B.TARGET]+=10000
    monkeypatch.setattr(E.B,'ROOT',tmp_path/'b')
    second=pd.read_parquet(E.B.fit_fold('ridge','test',0,dates[:130],dates[152:],frame,[.01,1.,100.]))
    pd.testing.assert_frame_equal(first,second)
    assert len(first)==260


def test_selected_formula_membership_and_feature_widths():
    assert len(E.FORMULAS)==9 and len(E.TASKS)==24
    assert E.GROUPS['momentum132_volatility9']==141
    assert E.GROUPS['selected_volatility9']==9
    assert len(set(E.TASKS))==24


def test_141_input_torch_matches_scalar_reference():
    pytest.importorskip('torch')
    from src.tests.test_asset_momentum import scalar_torch
    rng=np.random.default_rng(141)
    x=rng.normal(size=(2,20,141));y=rng.normal(size=(2,20));q=rng.normal(size=(2,4,141))
    x[1,:3]=np.nan
    params=dict(hidden_layer_sizes=[4],alpha=.1,learning_rate_init=.001,activation='identity',batch_size=1024)
    pred,epochs=M.torch_candidates(x,y,q,[params],seed=2,max_iter=8,tol=0.)
    for i in range(2):
        expected,stopped=scalar_torch(x[i],y[i],q[i],params,2,8,0.)
        np.testing.assert_allclose(pred[i,:,0],expected,rtol=1e-9,atol=1e-10)
        assert epochs[i,0]==stopped


def test_full_scoring_audit_ensembles_and_momentum_pairs(tmp_path,monkeypatch):
    monkeypatch.setattr(E,'ROOT',tmp_path)
    monkeypatch.setattr(E.B,'ROOT',tmp_path)
    monkeypatch.setattr(E.B,'verify',lambda:{})
    reported=[]
    monkeypatch.setattr(E,'report',lambda *args:reported.append(args))
    rng=np.random.default_rng(209)
    keys=pd.DataFrame([(d,a) for d in pd.bdate_range('2024-01-01',periods=4) for a in range(12)],columns=['date','ucid'])
    keys['row']=np.arange(len(keys));keys['symbol']=keys.ucid.map(lambda a:f'A{a:02d}')
    keys[E.B.TARGET]=rng.uniform(0,1,len(keys)).astype(np.float32)
    keys['meta_model']=rng.uniform(0,1,len(keys));keys['covered']=keys.ucid.lt(10)
    E.parquet(tmp_path/'keys.parquet',keys)
    fallback=keys[['row','date']].copy();fallback['baseline']=rng.normal(size=len(keys))
    fallback['baseline_rank']=fallback.groupby('date').baseline.transform(lambda s:E.B.rank_normalise(s.to_numpy()))
    fallback['block']=0;E.parquet(tmp_path/'baseline.parquet',fallback)
    for name,width in E.GROUPS.items():
        values=rng.normal(size=(len(keys),width));values[~keys.covered]=np.nan
        frame=pd.concat([keys[['row','ucid','date',E.B.TARGET,'covered']],E.feature_frame(values)],axis=1)
        E.parquet(tmp_path/'datasets'/f'{name}.parquet',frame)
        for model in ['ridge','mlp_torch']:
            parts=[]
            for seed in ([0] if model=='ridge' else E.B.SEEDS):
                p=keys.loc[keys.covered,['row']].copy()
                p['prediction']=rng.normal(size=len(p));p['asset_mean']=p.row.mod(12)/12
                p['seed']=seed;p['block']=0;parts.append(p)
            directory=tmp_path/'fits'/f'{model}__{name}'
            E.parquet(directory/'block0.parquet',pd.concat(parts,ignore_index=True))
            E.save(directory/'COMPLETE.json',dict(task=dict(tag=f'{model}__{name}',model=model)))
    pd.DataFrame([dict(ucid=1,changed=0)]).to_csv(tmp_path/'source_prefix_audits.csv',index=False)
    E.finish()
    assert len(reported)==1
    table=pd.read_csv(tmp_path/'table.csv')
    assert table[table.seed=='ensemble'].task.nunique()==27
    deltas=pd.read_csv(tmp_path/'paired_deltas.csv')
    paired=deltas[(deltas.reference=='matched_momentum132')&(deltas.seed=='ensemble')&(deltas['sample']=='routed')&(deltas.metric=='payout')]
    assert len(paired)==4
    audit=__import__('json').loads((tmp_path/'audit.json').read_text())
    assert audit['outcome']=='PASS WITH CONDITIONS' and len(audit['checks'])==24


def test_completion_email_attaches_pdf_and_does_not_resend(tmp_path,monkeypatch):
    import smtplib
    from src.live import env,notify
    monkeypatch.setattr(E,'ROOT',tmp_path)
    pdf=tmp_path/'report.pdf';pdf.write_bytes(b'%PDF synthetic report')
    E.save(tmp_path/'COMPLETE.json',dict(pdf=str(pdf),pdf_sha256=E.sha(pdf)))
    monkeypatch.setattr(env,'load_env',lambda **kwargs:None)
    monkeypatch.setattr(notify,'_config',lambda:dict(host='test',port=587,user='sender@example.test',password='test',to='recipient@example.test'))
    messages=[]
    class Server:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def starttls(self):pass
        def login(self,*args):pass
        def send_message(self,message,**kwargs):
            messages.append(message);return {}
    monkeypatch.setattr(smtplib,'SMTP',Server)
    E.email_report();E.email_report()
    assert len(messages)==1
    attachments=list(messages[0].iter_attachments())
    assert len(attachments)==1 and attachments[0].get_payload(decode=True)==pdf.read_bytes()
