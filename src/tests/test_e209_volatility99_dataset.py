"""Reusable volatility99 schema, missingness and generic training guard."""
import json
import numpy as np
import pandas as pd
import pytest
from src.data_mani import build_e209_volatility99 as D
from src.data_mani import utils


def test_99_names_capture_distinct_formula_calendars_and_windows():
    names=D.feature_names()
    assert len(names)==99 and len(set(names))==99
    assert all(n.startswith('feature_vol_') for n in names)
    assert set(D.WINDOWS)=={5,10,15,20,30,40,50,60,90,120,240}
    assert sum('parkinson_daily_' in n for n in names)==11
    assert sum('parkinson_weekday_' in n for n in names)==11
    assert sum('semivariance_' in n and '_down_' in n for n in names)==11


def test_partial_coverage_keeps_observed_zero_and_does_not_use_momentum():
    keys=pd.DataFrame(dict(row=[0,1,2],symbol=['a','b','c'],ucid=[1,2,3],date=pd.to_datetime(['2024-01-01']*3)))
    values=np.zeros((3,99),np.float32);values[1,2]=np.nan;values[2]=np.nan
    coverage,availability=D.coverage_tables(keys,values,[True,True,False])
    assert coverage.covered.tolist()==[True,False,False]
    assert availability.iloc[1].sum()==98
    assert values[1,0]==0 and np.isnan(values[1,2])
    assert coverage[list(D.GROUPS)[1]].tolist()==[True,True,False]
    assert not any('momentum' in c for c in coverage)


def test_source_cache_identity_and_schema_are_checked():
    f=pd.DataFrame(np.zeros((2,108),np.float32),columns=list(map(str,range(108))))
    f.insert(0,'row',[5,9])
    assert D.verify_cache(f,[5,9]).shape==(2,99)
    with pytest.raises(AssertionError):D.verify_cache(f,[9,5])
    f['0']=.1
    with pytest.raises(ValueError,match='Invalid TS-bin'):D.verify_cache(f,[5,9])


def test_generic_loader_cannot_neutral_fill_candidate_dataset(tmp_path,monkeypatch):
    monkeypatch.setattr(utils,'DATA_DIR',str(tmp_path))
    root=tmp_path/'crypto'/'vol99';root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps(dict(missing_value_policy='coverage_required')))
    monkeypatch.setattr(utils,'_load_crypto',lambda *a:pytest.fail('must reject before reading/filling'))
    with pytest.raises(ValueError,match='coverage-aware training'):
        utils.load_tournament_data('crypto',version='crypto/vol99',verbose=False)


def test_ordinary_dataset_loading_still_uses_existing_fill(tmp_path,monkeypatch):
    monkeypatch.setattr(utils,'DATA_DIR',str(tmp_path))
    dates=pd.date_range('2024-01-01',periods=10)
    f=pd.DataFrame(dict(symbol=['a']*10,date=dates,feature_test=[np.nan]*10,target_binned_return_20=[.5]*10))
    monkeypatch.setattr(utils,'_load_crypto',lambda *a:(f.copy(),['feature_test']))
    loaded,features=utils.load_tournament_data('crypto',version='crypto/ordinary',embargo=0,verbose=False)
    assert features==['feature_test'] and loaded.feature_test.eq(.5).all()


def test_source_metadata_keeps_retrieval_time_unknown(tmp_path):
    prices=pd.DataFrame(dict(ucid=['7','7'],symbol=['A','A'],source=['vendor','vendor'],
        provider_ticker=['A-USD','A-USD'],date=pd.to_datetime(['2020-01-01','2020-01-02'])))
    identity,provenance=D.source_metadata(prices,tmp_path/'inputs'/'prices.parquet')
    assert identity.bars.iloc[0]==2 and identity.ucid.iloc[0]=='7'
    assert provenance['retrieved_at'] is None and provenance['provider_dataset_version'] is None
