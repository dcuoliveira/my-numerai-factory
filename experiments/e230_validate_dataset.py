"""Independent E230 source-column, mask and label-wall reconstruction."""
import json
import numpy as np
import pandas as pd
from experiments import e230_dataset as D
from experiments import e212_volatility99_tsfm as S


def run():
    meta=json.loads((D.ROOT/'PREPARED.json').read_text());stored=pd.read_parquet(D.DATA/'train.parquet').reset_index();coverage=pd.read_parquet(D.DATA/'coverage.parquet').reset_index();catalog=json.loads((D.DATA/'features.json').read_text())['feature_sets'];checks=[]
    assert len(catalog['all'])==134 and len(set(catalog['all']))==134
    assert len(stored)==474496 and not stored.duplicated(['ucid','date']).any()
    def align(frame):
        q=frame.reset_index();q.ucid=q.ucid.astype(str)
        return q.set_index(['ucid','date']).reindex(pd.MultiIndex.from_frame(stored.assign(ucid=stored.ucid.astype(str))[['ucid','date']]))
    for root,group in [(D.MOM,'momentum33'),(D.VOL,'volatility99')]:
        q=align(pd.read_parquet(root/'train.parquet'))
        np.testing.assert_array_equal(q.symbol,stored.symbol);np.testing.assert_array_equal(q[D.TARGET],stored[D.TARGET]);np.testing.assert_array_equal(q[catalog[group]].to_numpy(),stored[catalog[group]].to_numpy());checks.append(dict(group=group,features=len(catalog[group]),rows=len(stored),exact_values_and_nan_locations=True,keys_symbols_labels_exact=True))
    q=align(pd.read_parquet(D.NATIVE));np.testing.assert_array_equal(q.symbol,stored.symbol);np.testing.assert_array_equal(q[D.TARGET],stored[D.TARGET]);np.testing.assert_array_equal(q[catalog['native2']].to_numpy(),stored[catalog['native2']].to_numpy())
    cm=align(pd.read_parquet(D.MOM/'coverage.parquet'));cv=align(pd.read_parquet(D.VOL/'coverage.parquet'))
    expected=np.isfinite(stored[catalog['all']]).all(axis=1)&cm.covered.to_numpy(bool)&cv.source_eligible.to_numpy(bool)
    np.testing.assert_array_equal(expected,coverage.covered)
    p=pd.read_parquet(D.ROOT/'panel.parquet');i=dict(np.load(D.ROOT/'indices.npz'))
    train=(p.date<=pd.Timestamp('2023-11-27'));val=p.date.between('2024-01-01','2024-11-27');final=p.date<=pd.Timestamp('2024-11-27');test=p.date.between('2025-01-01','2026-02-19')
    complete=p.covered&np.isfinite(p[D.TARGET]);counts=p.assign(tr=train&complete,va=val&complete).groupby('ucid')[['tr','va']].sum();assets=counts.index[(counts.tr>=60)&(counts.va>=20)].tolist();assert set(assets)==set(meta['assets'])
    for phase,mask in [('train',train),('validation',val),('final_fit',final)]:np.testing.assert_array_equal(i[phase],np.flatnonzero(mask&complete&p.ucid.isin(assets)))
    np.testing.assert_array_equal(i['query'],np.flatnonzero(test&p.covered&p.ucid.isin(assets)))
    old=pd.read_parquet(D.E228/'common_keys.parquet');left=p.loc[i['score'],['ucid','date','symbol',D.TARGET,'meta_model']].reset_index(drop=True);right=old[['ucid','date','symbol',D.TARGET,'meta_model']].reset_index(drop=True);left.ucid=left.ucid.astype(str);right.ucid=right.ucid.astype(str);pd.testing.assert_frame_equal(left,right,check_dtype=False)
    assert all(S.sha(path)==h for path,h in meta['input_hashes'].items());assert all(S.sha(path)==h for path,h in meta['output_hashes'].items())
    S.save(D.ROOT/'dataset_audit.json',dict(outcome='PASS WITH CONDITIONS',source_column_reconstructions=checks,native2_exact=True,all474496_base_keys_targets_preserved=True,coverage_and_nan_masks_reconstructed=True,all134_source_columns_exact=True,all_split_masks_and_asset_eligibility_reconstructed=True,all37334_e228_keys_symbols_targets_meta_exact=True,source_hashes_unchanged=True,conditions=meta['conditions'],audit_sha256=S.sha(__file__)))
    print('Dataset independent audit passed',len(stored),len(assets),flush=True)

if __name__=='__main__':run()
