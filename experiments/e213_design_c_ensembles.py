"""Registered E213: eight E206 design-C streams, all 247 equal-rank subsets."""
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

SOURCE = REPO/'results/reports/crypto/e206_momentum3_tsfm'
ROOT = REPO/'results/reports/crypto/e213_design_c_ensembles'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = pd.read_csv(SOURCE/'ranked_results.csv')
    selected = all_rows.loc[all_rows.universe.eq('common') & all_rows.side.eq('neutral') &
                           all_rows.design.eq('C') & all_rows.standardization.isin(['rolling', 'cross_section'])].copy()
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
        path = SOURCE/'fits'/row.tag/'predictions.parquet'
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
    daily, forecast_parts = [], []
    replay_error = 0.
    for date, indices in base.groupby('date', sort=True).indices.items():
        rows = indices[covered[indices]]
        if not len(rows):
            continue
        ranks = np.column_stack([rank_normalise(raw[rows, i]) for i in range(8)])
        predictions = ranks @ weights
        predictions[:, :8] = raw[rows]
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
    dates = pd.DatetimeIndex([x[0] for x in daily])
    for ref in references:
        assert dates.equals(pd.DatetimeIndex(ref.index))
    corr = np.stack([x[1] for x in daily]); mmc = np.stack([x[2] for x in daily])
    pay = .1*corr+mmc
    trials = []
    for j, members in enumerate(definitions):
        cumulative = np.r_[0., pay[:, j].cumsum()]
        trials.append(dict(tag=f'e213_{j:03d}', size=len(members), members=' + '.join(labels[i] for i in members),
                           dates=len(dates), corr=corr[:, j].mean(), mmc=mmc[:, j].mean(), payout=pay[:, j].mean(),
                           **{'sharpe.p': pay[:, j].mean()/pay[:, j].std(ddof=0) if pay[:, j].std(ddof=0) > 0 else np.nan,
                              'maxdd.p': (np.maximum.accumulate(cumulative)-cumulative).max()}))
    table = pd.DataFrame(trials).sort_values(['payout', 'tag'], ascending=[False, True])
    ensembles = table.loc[table['size'] > 1]
    assert len(ensembles) == 247 and ensembles.payout.is_monotonic_decreasing
    ensembles.to_csv(ROOT/'ensemble_results.csv', index=False)
    table.to_csv(ROOT/'consolidated_results.csv', index=False)
    ensembles.rename(columns={'dates': 'n_dates'}).to_csv(ROOT/'all_model_window_rows.csv', index=False)
    np.savez_compressed(ROOT/'daily_scores.npz', dates=dates.to_numpy(), corr=corr, mmc=mmc, payout=pay,
                        n=np.array([x[3] for x in daily]))
    np.savez_compressed(ROOT/'common_predictions.npz', rows=base.row.to_numpy()[np.concatenate([x[0] for x in forecast_parts])],
                        predictions=np.vstack([x[1] for x in forecast_parts]))
    pd.DataFrame(dict(row=base.row, covered=covered)).to_parquet(ROOT/'coverage.parquet', index=False)
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
    ax.set_title('Design C — correlation of daily payouts (not cumulative)')
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    heatmap = 'E213_design_C_daily_payout_correlation.png'
    fig.savefig(ROOT/heatmap, dpi=170); plt.close(fig)
    top = ensembles.head(3)
    fig, ax = plt.subplots(figsize=(16, 10), layout='constrained')
    for row in pd.concat([top, table[table['size'] == 1]]).itertuples():
        j = int(row.tag.rsplit('_', 1)[1])
        ax.plot(dates, pay[:, j].cumsum(), label=row.members,
                linewidth=2.6 if row.size > 1 else 1.2, linestyle='-' if row.size > 1 else '--')
    ax.set(title='Design C — top three ensembles and eight standalone models', xlabel='Date', ylabel='Cumulative payout score')
    ax.grid(alpha=.2); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.10), fontsize=8, ncol=1)
    plot = 'E213_design_C_top3_cumulative_payout.png'
    fig.savefig(ROOT/plot, dpi=170); plt.close(fig)
    section = (f'<h2>E213 — Design C daily payout correlation</h2><p>Pearson correlation of daily payouts on the same {len(dates)} dates, not of cumulative curves.</p>'
               f'<img style="width:100%" src="{heatmap}"><h2>Design C — all 247 ensembles</h2>'
               '<p>All 2–8-member subsets; equal-weight member prediction ranks, freshly scored. Sorted by mean payout. '
               'Top-three selection on this same history is exploratory. No investment-return or independent confirmation claim.</p>'+
               ensembles.drop(columns='tag').to_html(index=False, float_format=lambda x: f'{x:.6f}')+
               f'<h2>Top three ensembles plus all eight design-C standalones</h2><img style="width:100%" src="{plot}">')
    page = SOURCE/'E206_standardization_comparison.html'
    document = re.sub(r'<!-- E213 START -->.*?<!-- E213 END -->', '', page.read_text(), flags=re.S)
    page.write_text(document.replace('</body>', '<!-- E213 START -->'+section+'<!-- E213 END --></body>'))
    for name in [heatmap, plot]:
        shutil.copy2(ROOT/name, SOURCE/name)
    for row in ensembles.itertuples():
        directory = ROOT/'fits'/row.tag; directory.mkdir(parents=True, exist_ok=True)
        (directory/'COMPLETE.json').write_text(json.dumps(dict(task=dict(tag=row.tag, model='rank_ensemble'), members=row.members)))
    manifest = dict(experiment='E213', definitions=[dict(id=f'e213_{i:03d}', indices=c) for i, c in enumerate(definitions)],
                    source_members=selected.to_dict('records'), source_hashes=hashes, coverage_hash=digest(ROOT/'coverage.parquet'),
                    standalone_max_error=replay_error, top_three=top.to_dict('records'),
                    audit=dict(outcome='PASS WITH CONDITIONS', matched_common_keys=True,
                               conditions=['Inherited source vintage/pretraining overlap limits.', 'Same-history membership/configuration selection is exploratory.',
                                           'Common-covered standalone payout research; no native-feature marginal or live deployment claim.']))
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    subprocess.run(['ssh', '-o', 'ControlPath=none', '-o', 'BatchMode=yes', 'lovelace', 'python3',
                    str(REPO/'src/analysis/html_pdf.py'), str(page)], check=True, timeout=300)
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(status='success', ensembles=247, controls=8, dates=len(dates), report=str(page.with_suffix('.pdf')))))
    print(top[['members', 'payout', 'sharpe.p']].to_string(index=False))


if __name__ == '__main__':
    main()
