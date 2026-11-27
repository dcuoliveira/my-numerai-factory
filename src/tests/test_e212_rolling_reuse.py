"""Numerical equivalence of label-independent rolling H reuse across folds."""
import numpy as np
import pandas as pd
import pytest
from experiments import e212_volatility99_tsfm as S


def test_cached_heads_match_uncached_across_blocks_and_label_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(S,'ROOT',tmp_path/'root');monkeypatch.setattr(S,'SCR',tmp_path/'cache')
    S.ROOT.mkdir();S.SCR.mkdir()
    rng=np.random.default_rng(212)
    dates=pd.bdate_range('2020-01-01',periods=620)
    p=pd.DataFrame([(d,str(a)) for d in dates for a in [1,2]],columns=['date','ucid'])
    p['row']=np.arange(len(p));p['step']=np.repeat(np.arange(len(dates)),2);p['covered']=True
    p['target_binned_return_20']=rng.uniform(size=len(p))
    directory=S.extraction_dir(20,'chronos2');directory.mkdir(parents=True)
    h=rng.normal(size=(len(p),6)).astype(np.float16)
    np.save(directory/'H_rows.npy',h);np.save(directory/'valid_rows.npy',np.ones(len(p),bool))
    np.save(S.SCR/'common_rows.npy',np.ones(len(p),bool))
    memo={}
    for start in [430,520]:
        q,_=S.fold_panel(p,start,20)
        for phase in ['selection','final']:
            reference,records,params,info=S.solve_head(q,20,'chronos2','rolling',63,'selection',S.ROOT,S.SCR/f'ref_{start}')
            if phase=='final':reference,*_=S.solve_head(q,20,'chronos2','rolling',63,'final',S.ROOT,S.SCR/f'ref_final_{start}',params)
            actual,*_=S.solve_head(q,20,'chronos2','rolling',63,phase,S.ROOT,S.SCR/f'memo_{start}_{phase}',params if phase=='final' else None,representation_cache=memo)
            np.testing.assert_array_equal(actual,reference)
        # Future label changes cannot enter normalization or its cache identity.
        q.loc[q.test,'target_binned_return_20']+=1000
        actual,*_=S.solve_head(q,20,'chronos2','rolling',63,'final',S.ROOT,S.SCR/'future',params,representation_cache=memo)
        np.testing.assert_array_equal(actual,reference)
    assert len(memo['normalizations'])==1
    assert not next(iter(memo['normalizations'].values()))[0].flags.writeable
    # A changed underlying H file must not silently reuse the old transform.
    np.save(directory/'H_rows.npy',h*2)
    with pytest.raises(RuntimeError,match='identity changed'):
        S.solve_head(q,20,'chronos2','rolling',63,'selection',S.ROOT,S.SCR/'changed',representation_cache=memo)
