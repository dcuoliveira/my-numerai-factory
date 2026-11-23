"""E215: canonical CU_SA slot forecasts and frozen E214 FIXED_s0.75."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from src.analysis import slots
from experiments.e206_momentum3_tsfm import TARGET, matrix_scores
from experiments.e210_tsfm_ensembles import digest
from src.live.submit import rank_normalise

ROOT = REPO/'results/reports/crypto/e215_book_fixed_ensembles'
NAMES = ['CU_SA_IA_BO_02', 'CU_SA_IA_BO_MA_R_01', 'CU_SA_IA_BO_MA_01', 'FIXED_s0.75']


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ROOT.mkdir(parents=True, exist_ok=True)
    source = REPO/'results/reports/crypto/e214_meta_variants/variant_predictions.parquet'
    panel = pd.read_parquet(source, columns=['row', 'symbol', 'date', TARGET, 'meta_model', 'FIXED_s0.75'])
    initial = len(panel)
    provenance = {str(source): dict(sha256=digest(source), transform='E214 stored strength0.75, lag>=5-day meta')}
    for name in NAMES[:3]:
        model, run, note = slots.slot_run(name)
        path = REPO/'results/forecast/crypto'/model/(run+'.csv')
        frame = pd.read_csv(path, comment='#', usecols=['symbol', 'date', 'prediction'], parse_dates=['date'])
        assert not frame.duplicated(['symbol', 'date']).any()
        panel = panel.merge(frame.rename(columns={'prediction': name}), on=['symbol', 'date'], how='inner', validate='1:1')
        provenance[str(path)] = dict(slot=name, model=model, run=run, sha256=digest(path), note=note)
    panel = panel.dropna(subset=NAMES+[TARGET, 'meta_model']).sort_values(['date', 'symbol']).reset_index(drop=True)
    assert np.isfinite(panel[NAMES+[TARGET, 'meta_model']].to_numpy()).all()
    assert panel.row.is_unique and panel.date.max() < pd.Timestamp('2026-02-20')
    panel.to_parquet(ROOT/'matched_predictions.parquet', index=False)
    definitions = [c for size in range(1, 5) for c in itertools.combinations(range(4), size)]
    assert len(definitions) == 15
    weights = np.zeros((4, 15))
    for j, members in enumerate(definitions): weights[list(members), j] = 1/len(members)
    daily, saved = [], []
    for date, idx in panel.groupby('date', sort=True).indices.items():
        if len(idx) < 10: continue
        raw = panel.loc[idx, NAMES].to_numpy()
        ranks = np.column_stack([rank_normalise(raw[:, i]) for i in range(4)])
        predictions = ranks @ weights
        predictions[:, :4] = raw
        c, m = matrix_scores(predictions, panel.loc[idx, TARGET].to_numpy(), panel.loc[idx, 'meta_model'].to_numpy())
        # Verify batched scoring against each scalar forecast, including ensembles.
        for j in range(15):
            cc, mm = matrix_scores(predictions[:, j, None], panel.loc[idx, TARGET].to_numpy(), panel.loc[idx, 'meta_model'].to_numpy())
            np.testing.assert_allclose([c[j], m[j]], [cc[0], mm[0]], rtol=1e-10, atol=1e-12)
        daily.extend(dict(date=date, id=j, n=len(idx), corr=c[j], mmc=m[j], payout=.1*c[j]+m[j]) for j in range(15))
        saved.append(predictions)
    scores = pd.DataFrame(daily)
    scores.to_parquet(ROOT/'daily_scores.parquet', index=False)
    np.savez_compressed(ROOT/'trial_predictions.npz', rows=panel.row.to_numpy(), predictions=np.vstack(saved))
    results = []
    for j, members in enumerate(definitions):
        s = scores[scores.id.eq(j)].sort_values('date'); p=s.payout.to_numpy(); cum=np.r_[0., p.cumsum()]
        results.append(dict(tag=f'e215_{j:02d}', id=j, size=len(members), members=' + '.join(NAMES[i] for i in members),
                            dates=len(s), mean_assets=s.n.mean(), corr=s['corr'].mean(), mmc=s.mmc.mean(), payout=p.mean(),
                            **{'sharpe.p':p.mean()/p.std(ddof=0) if p.std(ddof=0)>0 else np.nan,
                               'maxdd.p':(np.maximum.accumulate(cum)-cum).max()}))
    table=pd.DataFrame(results).sort_values(['payout','id'],ascending=[False,True])
    assert len(table)==15 and table.payout.is_monotonic_decreasing
    if len(panel)==initial:
        old=pd.read_csv(REPO/'results/reports/crypto/e214_meta_variants/standalone_variants.csv').set_index('members').loc['FIXED_s0.75']
        new=table.loc[table.members.eq('FIXED_s0.75')].iloc[0]
        np.testing.assert_allclose(new[['corr','mmc','payout','sharpe.p','maxdd.p']].to_numpy(float),
                                   old[['corr','mmc','payout','sharpe.p','maxdd.p']].to_numpy(float),rtol=1e-9,atol=1e-10)
    table.to_csv(ROOT/'performance.csv',index=False)
    table[table['size']>1].rename(columns={'dates':'n_dates'}).to_csv(ROOT/'all_model_window_rows.csv',index=False)
    for row in table[table['size']>1].itertuples():
        d=ROOT/'fits'/row.tag;d.mkdir(parents=True,exist_ok=True)
        (d/'COMPLETE.json').write_text(json.dumps(dict(task=dict(tag=row.tag,model='rank_ensemble'),members=row.members)))
    fig, ax=plt.subplots(figsize=(16,10),layout='constrained')
    palette=plt.get_cmap('turbo')(np.linspace(.02,.98,len(table)))
    for color,row in zip(palette,table.itertuples()):
        s=scores[scores.id.eq(row.id)].sort_values('date')
        ax.plot(s.date,s.payout.cumsum(),label=row.members,color=color,
                linewidth=2.3 if row.size==1 else 1.4,linestyle='--' if row.size==1 else '-')
    ax.set(title='Three CU_SA slots + FIXED_s0.75 — all standalone and ensemble payouts',xlabel='Date',ylabel='Cumulative payout score')
    ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),fontsize=7,ncol=1)
    fig.savefig(ROOT/'cumulative_payout.png',dpi=170);plt.close(fig)
    dates=panel.date.agg(['min','max'])
    doc=('<html><head><meta charset="utf-8"></head><body><h1>E215 — CU_SA book and FIXED_s0.75 ensembles</h1>'
         f'<p>Four standalone controls and all11 equal-rank ensembles (six pairs, four triples, one four-member blend). '
         f'Matched historical target20 sample: {len(panel):,} rows, {panel.date.nunique()} dates, '
         f'{dates["min"].date()} to {dates["max"].date()}. All models use identical shared keys; '
         'canonical slot forecast transformations and E214 FIXED_s0.75 retained, no second residualization. '
         'Ensembles average within-date member ranks and are freshly scored. Payout=0.1CORR+historical-meta proxyMMC; '
         'sharpe.p is unannualized mean/population std; maxdd.p is additive cumulative payout drawdown. '
         'Historical backtest forecasts; same-history selection exploratory, no live leaderboard, independent confirmation or investment-return claim.</p>')
    doc+=table.drop(columns=['tag','id']).to_html(index=False,float_format=lambda x:f'{x:.6f}')
    doc+='<img style="width:100%" src="cumulative_payout.png"></body></html>'
    page=ROOT/'E215_book_fixed_ensembles.html';page.write_text(doc)
    (ROOT/'manifest.json').write_text(json.dumps(dict(source_provenance=provenance,source_fixed_rows=initial,matched_rows=len(panel),
        definitions=[dict(id=i,members=c) for i,c in enumerate(definitions)],
        audit=dict(outcome='PASS WITH CONDITIONS',scalar_batch_parity=True,matched_common_keys=True,
                   conditions=['Prior E206/E213/E214 configuration/strength selection; same-history ensemble selection.',
                               'Canonical historical slot backtests, not certified live parity or current round forecasts.',
                               'Common-covered standalone payout comparison; no native-feature marginal claim.'])),indent=2))
    subprocess.run(['ssh','-o','ControlPath=none','-o','BatchMode=yes','lovelace','python3',str(REPO/'src/analysis/html_pdf.py'),str(page)],check=True,timeout=300)
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(status='success',ensembles=11,controls=4,report=str(page.with_suffix('.pdf')))))
    print(table[['members','payout','sharpe.p','maxdd.p']].head(5).to_string(index=False))


if __name__=='__main__': main()
