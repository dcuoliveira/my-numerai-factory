"""E214: causal meta residualization of the selected E213 blend and components."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments.e206_momentum3_tsfm import TARGET, matrix_scores
from experiments.e210_tsfm_ensembles import digest
from src.prediction import meta_resid as MR
from src.live.submit import rank_normalise

ROOT = REPO/'results/reports/crypto/e214_meta_variants'
SOURCE = REPO/'results/reports/crypto/e213_design_c_ensembles'
SCR = Path('/scratch/danielco/e214_meta_variants')
STRENGTHS = [0., .25, .5, .75, 1.]


def main():
    ROOT.mkdir(parents=True, exist_ok=True); SCR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    best = pd.read_csv(SOURCE/'ensemble_results.csv').set_index('tag').loc['e213_129']
    frozen_members = [1, 2, 3, 6]
    definition = next(x for x in manifest['definitions'] if x['id'] == 'e213_129')
    assert definition['indices'] == frozen_members
    cache = np.load(SOURCE/'common_predictions.npz')
    rows, previous = cache['rows'], cache['predictions']
    baseline = pd.read_parquet(REPO/'results/reports/crypto/e206_momentum3_tsfm/momentum_baseline.parquet')
    assert baseline.row.is_unique and len(np.unique(rows)) == len(rows)
    panel = baseline.set_index('row').reindex(rows).reset_index()
    assert panel.date.notna().all()
    raw = previous[:, frozen_members+[129]]
    del previous
    base_names = ['CR', 'TC', 'TR', 'SC', 'FIXED']
    names = [f'{name}_s{strength:g}' for name in base_names for strength in STRENGTHS]
    metadata = pd.read_parquet(REPO/'data/crypto/v2.0/historical_meta_models.parquet').reset_index()[['symbol', 'date', 'prediction']]
    variants = np.empty((len(panel), 25), float)
    snapshots = []
    groups = list(panel.groupby('date', sort=True).indices.items())
    reference_scores = np.load(SOURCE/'daily_scores.npz')
    max_replay = 0.
    for d, (date, idx) in enumerate(groups):
        lagged, snapshot = MR.lagged_meta(metadata, panel.loc[idx, 'symbol'], date)
        assert snapshot is None or snapshot <= date-pd.Timedelta(days=5)
        for component in range(5):
            for s, strength in enumerate(STRENGTHS):
                variants[idx, component*5+s], beta = MR.residualise(raw[idx, component], lagged, strength)
        corr, mmc = matrix_scores(variants[idx, 20, None], panel.loc[idx, TARGET].to_numpy(), panel.loc[idx, 'meta_model'].to_numpy())
        np.testing.assert_allclose([corr[0], mmc[0]], [reference_scores['corr'][d, 129], reference_scores['mmc'][d, 129]], atol=1e-8, rtol=1e-6)
        max_replay = max(max_replay, abs(corr[0]-reference_scores['corr'][d, 129]), abs(mmc[0]-reference_scores['mmc'][d, 129]))
        snapshots.append(dict(date=date, snapshot=snapshot, n=len(idx), meta_covered=int(np.isfinite(lagged).sum()), beta_fixed=beta))
    del raw
    frame = panel[['row', 'symbol', 'date', TARGET, 'meta_model']].join(pd.DataFrame(variants, columns=names))
    frame.to_parquet(ROOT/'variant_predictions.parquet', index=False)
    pd.DataFrame(snapshots).to_parquet(ROOT/'meta_snapshots.parquet', index=False)
    combos = [c for size in range(1, 6) for c in itertools.combinations(range(25), size)]
    assert len(combos) == 68405
    weights = np.zeros((25, len(combos)))
    for i, members in enumerate(combos): weights[list(members), i] = 1/len(members)
    scores = np.lib.format.open_memmap(SCR/'scores.npy', mode='w+', dtype=float, shape=(len(groups), len(combos), 2))
    for d, (date, idx) in enumerate(groups):
        ranks = np.column_stack([rank_normalise(variants[idx, j]) for j in range(25)])
        for start in range(0, len(combos), 256):
            stop = min(start+256, len(combos))
            pred = ranks @ weights[:, start:stop]
            if start < 25:
                pred[:, :min(stop,25)-start] = variants[idx, start:min(stop,25)]
            c, m = matrix_scores(pred, panel.loc[idx, TARGET].to_numpy(), panel.loc[idx, 'meta_model'].to_numpy())
            scores[d, start:stop, 0], scores[d, start:stop, 1] = c, m
        if d % 10 == 0:
            scores.flush()
            (ROOT/'progress.json').write_text(json.dumps(dict(dates=d+1, total_dates=len(groups), trials=len(combos))))
            print('SCORED', d+1, '/', len(groups), flush=True)
    scores.flush()
    payout = .1*scores[:, :, 0]+scores[:, :, 1]
    results = []
    for j, members in enumerate(combos):
        p = payout[:, j]; cum = np.r_[0., p.cumsum()]; sd = p.std(ddof=0)
        results.append(dict(id=j, size=len(members), members=' + '.join(names[i] for i in members), corr=scores[:, j, 0].mean(),
                            mmc=scores[:, j, 1].mean(), payout=p.mean(), **{'sharpe.p': p.mean()/sd if sd else np.nan,
                            'maxdd.p': (np.maximum.accumulate(cum)-cum).max()}, dates=len(groups)))
    table = pd.DataFrame(results).sort_values(['payout', 'id'], ascending=[False, True])
    table.to_csv(ROOT/'all_trials.csv', index=False)
    controls, ensembles = table[table['size'] == 1], table[table['size'] > 1]
    controls.to_csv(ROOT/'standalone_variants.csv', index=False)
    indexed = controls.copy(); indexed['tag'] = indexed.members
    indexed.rename(columns={'dates':'n_dates'}).to_csv(ROOT/'all_model_window_rows.csv',index=False)
    for row in controls.itertuples():
        directory=ROOT/'fits'/row.members;directory.mkdir(parents=True,exist_ok=True)
        (directory/'COMPLETE.json').write_text(json.dumps(dict(task=dict(tag=row.members,model='meta_resid'))))
    (ROOT/'definitions.json').write_text(json.dumps([dict(id=i, members=members) for i, members in enumerate(combos)]))
    report(table, payout, [d for d, _ in groups])
    (ROOT/'audit.json').write_text(json.dumps(dict(outcome='PASS WITH CONDITIONS', raw_fixed_replay_error=max_replay,
        source_hashes={name:digest(SOURCE/name) for name in ['common_predictions.npz','manifest.json']},
        temporal_rule='Native meta residualization, lag>=5 calendar days; source common rows retained.',
        limits=['Same-history strength and ensemble selection is exploratory.', 'Inherited vintage/pretraining limits; common-covered standalone payout research, no live/native marginal/price-return claim.']),indent=2))
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(status='success', variants=25, ensembles=68380, dates=len(groups),
        report=str(ROOT/'E214_meta_variants.pdf')),indent=2))
    print(table.head(5).to_string(index=False), flush=True)


def report(table, payout, dates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = []
    controls = table[table['size'] == 1]
    for title, rows, name in [
        ('Fixed ensemble — residualization strengths', controls[controls.id.between(20,24)], 'fixed_strengths'),
        ('Four component models — raw and residualized', controls[controls.id < 20], 'component_strengths'),
        ('Top three blends plus all standalone variants', pd.concat([table[table['size'] > 1].head(3), controls]), 'top3_and_variants')]:
        fig, ax = plt.subplots(figsize=(16, 10), layout='constrained')
        for row in rows.itertuples():
            ax.plot(dates, payout[:, row.id].cumsum(), label=row.members, linewidth=2.5 if row.size > 1 else 1.1,
                    linestyle='-' if row.size > 1 else '--')
        ax.set(title=title, xlabel='Date', ylabel='Cumulative payout score')
        ax.grid(alpha=.2); ax.legend(loc='upper center', bbox_to_anchor=(.5,-.10), fontsize=7, ncol=2)
        filename=name+'.png'; fig.savefig(ROOT/filename,dpi=160); plt.close(fig); plots.append(filename)
    text=('<html><head><meta charset="utf-8"></head><body><h1>E214 — Meta residualization variants and blends</h1>'
          '<p>CR=Chronos rolling; TC=t0 cross_section; TR=t0 rolling; SC=Timestral cross_section; '
          'FIXED=equal-rank CR+TC+TR+SC. All components use design C. s indicates residualization strength. '
          'Meta snapshots lag at least5 calendar days; native missing-meta/pass-through behavior retained. '
          'All 297 common dates; no new fits. Payout=0.1CORR+historical-meta proxyMMC; sharpe.p unannualized population mean/std; '
          'maxdd.p additive cumulative payout drawdown. All68380 blends average prediction ranks equally and are freshly scored. '
          'Same-history strength/membership selection is exploratory. No investment-return claim.</p><h2>All25 standalone variants</h2>')
    text += controls.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    text += '<h2>Top50 new blends</h2>'+table[table['size']>1].head(50).to_html(index=False,float_format=lambda x:f'{x:.6f}')
    text += '<p>Complete68405-row comparison: all_trials.csv. Daily CORR/MMC arrays and every combination definition retained.</p>'
    for plot in plots: text += f'<img style="width:100%" src="{plot}">'
    page=ROOT/'E214_meta_variants.html';page.write_text(text+'</body></html>')
    subprocess.run(['ssh','-o','ControlPath=none','-o','BatchMode=yes','lovelace','python3',str(REPO/'src/analysis/html_pdf.py'),str(page)],check=True,timeout=300)


if __name__ == '__main__':
    main()
