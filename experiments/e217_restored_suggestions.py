"""Registered E217: eight E206 design-C streams, all 247 equal-rank subsets."""
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments.e210_tsfm_ensembles import digest
from experiments.e206_momentum3_tsfm import TARGET, matrix_scores
from src.live.submit import rank_normalise
from src.prediction import meta_resid as MR
from src.analysis.tsfm_recommended_report import RECOMMENDED

SOURCE = REPO/'results/reports/crypto/e206_momentum3_tsfm'
ROOT = REPO/'results/reports/crypto/e217_restored_suggestions'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = pd.read_csv(SOURCE/'ranked_results.csv')
    selected = all_rows.loc[all_rows.universe.eq('common') & all_rows.side.eq('neutral') &
                           ((all_rows.design.eq('C') & (all_rows.standardization.eq(all_rows.model.map(RECOMMENDED)) |
                            (all_rows.standardization.eq('cross_section') & all_rows.model.isin(['chronos2','t0','timestral'])))) | all_rows.model.eq('momentum'))].copy()
    selected = selected.sort_values(['model', 'standardization']).reset_index(drop=True)
    assert len(selected) == 8 and selected.tag.is_unique
    labels = [f'{r.model}/{r.standardization}' for r in selected.itertuples()]
    base = pd.read_parquet(SOURCE/'momentum_baseline.parquet').sort_values(['date', 'ucid']).reset_index(drop=True)
    mask_dir = Path('/scratch/danielco/e206_momentum3_tsfm/portfolios')/selected.iloc[0].tag/'common'
    mask = pd.read_parquet(mask_dir, columns=['row', 'eligible'])
    assert mask.row.is_unique
    covered = base.row.isin(mask.loc[mask.eligible, 'row']).to_numpy()
    raw, references, hashes = [], [], {}
    for row in selected.itertuples():
        path = SOURCE/'momentum_baseline.parquet' if row.model=='momentum' else SOURCE/'fits'/row.tag/'predictions.parquet'
        frame = pd.read_parquet(path, columns=['row', 'prediction'])
        assert frame.row.is_unique
        raw.append(frame.set_index('row').prediction.reindex(base.row).to_numpy())
        references.append(pd.read_parquet(SOURCE/'scores'/f'{row.tag}_common.parquet').set_index('date'))
        hashes[str(path)] = digest(path)
    raw = np.column_stack(raw)
    assert np.isfinite(raw[covered]).all()
    definitions = [(i,) for i in range(8)] + [c for size in range(2, 9) for c in itertools.combinations(range(8), size)]
    assert len(definitions) == 255 and len(set(definitions)) == 255
    weights = np.zeros((8, 255))
    for j, members in enumerate(definitions):
        weights[list(members), j] = 1/len(members)
    metadata = pd.read_parquet(REPO/'data/crypto/v2.0/historical_meta_models.parquet').reset_index()[['date','symbol','prediction']]
    snapshots=[]
    daily, forecast_parts = [], []
    replay_error = 0.
    for date, indices in base.groupby('date', sort=True).indices.items():
        rows = indices[covered[indices]]
        if not len(rows):
            continue
        ranks = np.column_stack([rank_normalise(raw[rows, i]) for i in range(8)])
        predictions = ranks @ weights
        predictions[:, :8] = raw[rows]
        lagged,snapshot=MR.lagged_meta(metadata,base.loc[rows,'symbol'],date)
        assert snapshot is None or snapshot<=date-pd.Timedelta(days=5)
        residual=np.column_stack([MR.residualise(predictions[:,j],lagged,.5)[0] for j in range(8,255)])
        predictions=np.column_stack([predictions,residual])
        snapshots.append(dict(date=date,snapshot=snapshot,n=len(rows),meta_covered=int(np.isfinite(lagged).sum())))
        valid = np.isfinite(base.loc[rows, TARGET]) & np.isfinite(base.loc[rows, 'meta_model'])
        if valid.sum() < 10:
            continue
        corr, mmc = matrix_scores(predictions[np.asarray(valid)], base.loc[rows[valid], TARGET].to_numpy(),
                                  base.loc[rows[valid], 'meta_model'].to_numpy())
        for i in range(8):
            ref = references[i].loc[date]
            assert int(valid.sum()) == ref['n']
            expected = ref[['corr', 'mmc']].to_numpy(float)
            actual = np.array([corr[i], mmc[i]])
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
            replay_error = max(replay_error, float(np.abs(actual-expected).max()))
        daily.append((date, corr, mmc, int(valid.sum())))
        forecast_parts.append((rows, predictions))
        if len(daily)%50==0:print("SCORED",len(daily),flush=True)
    dates = pd.DatetimeIndex([x[0] for x in daily])
    for ref in references:
        assert dates.equals(pd.DatetimeIndex(ref.index))
    corr = np.stack([x[1] for x in daily]); mmc = np.stack([x[2] for x in daily])
    pay = .1*corr+mmc
    trials = []
    expanded = [(c,0.) for c in definitions]+[(c,.5) for c in definitions[8:]]
    for j, (members,strength) in enumerate(expanded):
        cumulative = np.r_[0., pay[:, j].cumsum()]
        trials.append(dict(tag=f'e217_{j:03d}', size=len(members), residual_strength=strength, members=' + '.join(labels[i] for i in members),
                           dates=len(dates), corr=corr[:, j].mean(), mmc=mmc[:, j].mean(), payout=pay[:, j].mean(),
                           **{'sharpe.p': pay[:, j].mean()/pay[:, j].std(ddof=0) if pay[:, j].std(ddof=0) > 0 else np.nan,
                              'maxdd.p': (np.maximum.accumulate(cumulative)-cumulative).max()}))
    table = pd.DataFrame(trials).sort_values(['payout', 'tag'], ascending=[False, True])
    ensembles = table.loc[table['size'] > 1]
    assert len(ensembles) == 494 and ensembles.payout.is_monotonic_decreasing
    ensembles.to_csv(ROOT/'ensemble_results.csv', index=False)
    table.to_csv(ROOT/'consolidated_results.csv', index=False)
    ensembles.rename(columns={'dates': 'n_dates'}).to_csv(ROOT/'all_model_window_rows.csv', index=False)
    np.savez_compressed(ROOT/'daily_scores.npz', dates=dates.to_numpy(), corr=corr, mmc=mmc, payout=pay,
                        n=np.array([x[3] for x in daily]))
    np.savez_compressed(ROOT/'common_predictions.npz', rows=base.row.to_numpy()[np.concatenate([x[0] for x in forecast_parts])],
                        predictions=np.vstack([x[1] for x in forecast_parts]))
    pd.DataFrame(dict(row=base.row, covered=covered)).to_parquet(ROOT/'coverage.parquet', index=False)
    pd.DataFrame(snapshots).to_parquet(ROOT/'meta_snapshots.parquet',index=False)
    standalone = pd.DataFrame(pay[:, :8], index=dates, columns=labels)
    correlation = standalone.corr()
    correlation.to_csv(ROOT/'daily_payout_correlation.csv')
    fig, ax = plt.subplots(figsize=(12, 10), layout='constrained')
    im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(8), labels, rotation=45, ha='right')
    ax.set_yticks(range(8), labels)
    for i in range(8):
        for j in range(8):
            value = correlation.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if abs(value) > .6 else 'black')
    ax.set_title('Expanded suggestions — correlation of daily payouts (not cumulative)')
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    heatmap = 'E217_design_C_daily_payout_correlation.png'
    fig.savefig(ROOT/heatmap, dpi=170); plt.close(fig)
    top = ensembles.head(3)
    fig, ax = plt.subplots(figsize=(16, 10), layout='constrained')
    for row in pd.concat([top, table[table['size'] == 1]]).itertuples():
        j = int(row.tag.rsplit('_', 1)[1])
        ax.plot(dates, pay[:, j].cumsum(), label=row.members+(f' [resid={row.residual_strength:g}]' if row.size>1 else ''),
                linewidth=2.6 if row.size > 1 else 1.2, linestyle='-' if row.size > 1 else '--')
    ax.set(title='Expanded suggestions — top three ensemble views and eight standalone models', xlabel='Date', ylabel='Cumulative payout score')
    ax.grid(alpha=.2); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.10), fontsize=8, ncol=1)
    plot = 'E217_design_C_top3_cumulative_payout.png'
    fig.savefig(ROOT/plot, dpi=170); plt.close(fig)
    selected=selected.set_index('tag')
    controls=table[table['size']==1].copy()
    suggestions=[]
    for i,source_row in enumerate(selected.itertuples()):
        result=table.loc[table.tag.eq(f'e217_{i:03d}')].iloc[0]
        record={k:getattr(source_row,k) for k in ['model','standardization','design','mean_coverage','rank_ic','mse','max_drawdown']}
        record.update({k:result[k] for k in ['corr','mmc','payout','sharpe.p','maxdd.p']});record['score_dates']=len(dates)
        suggestions.append(record)
    suggestions=pd.DataFrame(suggestions).sort_values('payout',ascending=False)
    suggestions=suggestions[['model','standardization','design','score_dates','mean_coverage','corr','mmc','payout','rank_ic','mse','sharpe.p','max_drawdown','maxdd.p']]
    suggestions.to_csv(ROOT/'suggestions.csv',index=False)
    fig,ax=plt.subplots(figsize=(14,8),layout='constrained')
    for i,label in enumerate(labels):ax.plot(dates,pay[:,i].cumsum(),label=label)
    ax.set(title='Expanded suggestions — cumulative payout',xlabel='Date',ylabel='Cumulative payout score');ax.grid(alpha=.2);ax.legend(fontsize=9)
    standalone_plot='E217_suggestions_cumulative_payout.png';fig.savefig(ROOT/standalone_plot,dpi=170);plt.close(fig)
    import html
    page=SOURCE/'E206_momentum3_tsfm.html'
    backup=Path('/home/danielco/.codex/attachments/31682d1f-633e-4fcb-a70c-c031309557e8/E206_momentum3_tsfm_backup.pdf')
    shutil.copy2(backup,ROOT/'E206_user_backup.pdf')
    previous=ROOT/'E206_before_restore.pdf'
    if not previous.exists():shutil.copy2(page.with_suffix('.pdf'),previous)
    document='<html><head><meta charset="utf-8"></head><body><h1>E206 — Momentum3 frozen representations</h1>'
    document+='<p>Restored from the user-supplied backup, expanded by E217. Original recommended configurations plus cross_section for Chronos, t0 and Timestral; TabPFN remains asset. Design C and original momentum baseline. Same common assets and297 dates; payout=0.1CORR+historical-meta proxyMMC, sharpe.p unannualized. Original max_drawdown is the inherited source-price neutral-return measure; maxdd.p is additive payout-score drawdown. The known source-price anomaly limits interpretation of max_drawdown.</p>'
    document+='<h2>Recommended configurations and momentum baseline — common assets</h2>'+suggestions.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    document+=f'<img style="width:100%" src="{standalone_plot}"><h2>Correlation of daily payouts</h2><img style="width:90%" src="{heatmap}">'
    document+='<h2>All247 ensembles — raw and output-residualized</h2><p>Equal member-rank average, then residualize the ensemble output at strength0.5 against the latest meta snapshot at least5 calendar days earlier. Both views retained and freshly scored. No component residualization is added. Rankings on this same history are exploratory, not independent confirmation.</p>'
    document+=ensembles.drop(columns='tag').to_html(index=False,float_format=lambda x:f'{x:.6f}')
    document+=f'<h2>Top three ensemble views plus eight standalones</h2><img style="width:100%" src="{plot}">'
    document+='<h2>Frozen protocol</h2><pre>'+html.escape((SOURCE/'PREPARED.json').read_text())+'</pre></body></html>'
    page.write_text(document)
    for name in [heatmap,plot,standalone_plot]:shutil.copy2(ROOT/name,SOURCE/name)
    for row in ensembles.itertuples():
        directory = ROOT/'fits'/row.tag; directory.mkdir(parents=True, exist_ok=True)
        (directory/'COMPLETE.json').write_text(json.dumps(dict(task=dict(tag=row.tag, model='rank_ensemble'), members=row.members)))
    manifest = dict(experiment='E217', definitions=[dict(id=f'e217_{i:03d}', indices=c,residual_strength=strength) for i, (c,strength) in enumerate(expanded)],
                    source_members=selected.reset_index().to_dict('records'), backup_hash=digest(backup), meta_source_hash=digest(REPO/'data/crypto/v2.0/historical_meta_models.parquet'), source_hashes=hashes, coverage_hash=digest(ROOT/'coverage.parquet'),
                    standalone_max_error=replay_error, top_three=top.to_dict('records'),
                    audit=dict(outcome='PASS WITH CONDITIONS', matched_common_keys=True,
                               conditions=['Inherited source vintage/pretraining overlap limits.', 'Same-history membership/configuration selection is exploratory.',
                                           'Common-covered standalone payout research; no native-feature marginal or live deployment claim.']))
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    subprocess.run(['ssh', '-o', 'ControlPath=none', '-o', 'BatchMode=yes', 'lovelace', 'python3',
                    str(REPO/'src/analysis/html_pdf.py'), str(page)], check=True, timeout=300)
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(status='success', ensembles=247, ensemble_views=494, controls=8, dates=len(dates), report=str(page.with_suffix('.pdf')))))
    print(top[['members', 'payout', 'sharpe.p']].to_string(index=False))


if __name__ == '__main__':
    main()
