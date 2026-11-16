import numpy as np
import pandas as pd
from src.data_mani.spot_volume_comparison import asset_signals, matched_windows, prepare_fold


def test_changing_pair_coverage_cannot_create_surge():
    x=np.column_stack([np.ones(24),np.r_[np.full(23,np.nan),100000.]])
    w,ok=matched_windows(x,21)
    assert (w[-1]==1).all()
    assert ok[-1].tolist()==[True,False]


def test_calendar_gaps_and_two_venue_breadth():
    rows=[]
    for venue in ['a','b']:
        for i,date in enumerate(pd.date_range('2024-01-01',periods=23)):
            rows.append(dict(date=date,venue=venue,pair='X/USD',base_volume=20 if i==22 else 10,
                             dollar_volume=200 if i==22 else 100,close_usd=10.))
    g=pd.DataFrame(rows);z=asset_signals(g)
    assert np.isclose(z.surprise.iloc[-1],np.log(2))
    assert z.breadth.iloc[-1]==1 and z.venue_count.iloc[-1]==2
    assert z.date.iloc[-1]==pd.Timestamp('2024-01-25')
    one=asset_signals(g[g.venue.eq('a')])
    assert pd.isna(one.breadth.iloc[-1]) and one.venue_count.iloc[-1]==1
    gap=asset_signals(g[~g.date.eq(pd.Timestamp('2024-01-10'))])
    assert pd.isna(gap.surprise.iloc[-1])


def test_clipping_thresholds_never_use_validation_data():
    tr=pd.DataFrame({'feature_svc_surprise':np.arange(100.),'feature_sv_surprise':.5})
    te=pd.DataFrame({'feature_svc_surprise':[1e9,np.nan],'feature_sv_surprise':[.2,np.nan]})
    a,b,bounds=prepare_fold(tr,te,list(tr))
    assert bounds['feature_svc_surprise']['upper']==98.01
    assert b[1].tolist()==[0.,.5]
    assert b[0,0]<2
