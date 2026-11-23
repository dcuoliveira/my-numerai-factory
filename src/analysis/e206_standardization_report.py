"""Compare stored E206 rolling and cross-sectional configurations; no new fits."""
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]/'results/reports/crypto/e206_momentum3_tsfm'


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    table = pd.read_csv(ROOT/'ranked_results.csv')
    table = table.loc[table.universe.eq('common') & table.side.eq('neutral') &
                      table.standardization.isin(['rolling', 'cross_section'])].copy()
    assert len(table) == 24 and not table.tag.duplicated().any()
    frames = {row.tag: pd.read_parquet(ROOT/'scores'/f'{row.tag}_common.parquet').set_index('date')
              for row in table.itertuples()}
    dates = next(iter(frames.values())).index
    for frame in frames.values():
        assert frame.index.is_unique
        dates = dates.intersection(frame.dropna(subset=['corr', 'mmc']).index)
    dates = dates.sort_values()
    reference = next(iter(frames.values())).loc[dates, 'n'].to_numpy()
    daily = {}
    for tag, frame in frames.items():
        frame = frame.loc[dates]
        np.testing.assert_array_equal(frame.n.to_numpy(), reference)
        pay = .1*frame['corr']+frame.mmc
        daily[tag] = pay
        mask = table.tag.eq(tag)
        table.loc[mask, 'corr'] = frame['corr'].mean()
        table.loc[mask, 'mmc'] = frame.mmc.mean()
        table.loc[mask, 'payout'] = pay.mean()
        table.loc[mask, 'sharpe.p'] = pay.mean()/pay.std(ddof=0) if pay.std(ddof=0) > 0 else np.nan
        cumulative = np.r_[0., pay.cumsum()]
        table.loc[mask, 'maxdd.p'] = (np.maximum.accumulate(cumulative)-cumulative).max()
        table.loc[mask, 'score_dates'] = len(dates)
    table['score_dates'] = table.score_dates.astype(int)
    table = table.sort_values('payout', ascending=False, kind='stable')
    columns = ['model', 'standardization', 'design', 'score_dates', 'mean_coverage',
               'corr', 'mmc', 'payout', 'rank_ic', 'mse', 'sharpe.p', 'maxdd.p']
    table[columns].to_csv(ROOT/'E206_standardization_comparison.csv', index=False)
    payout = pd.DataFrame(daily).loc[:, table.tag]
    payout.to_csv(ROOT/'E206_standardization_daily_payout.csv', index_label='date')
    labels = {row.tag: f'{row.model} | {row.standardization} | {row.design}' for row in table.itertuples()}
    plots = []
    for grouped in [False, True]:
        fig, ax = plt.subplots(figsize=(15, 9), layout='constrained')
        palette = plt.get_cmap('turbo')(np.linspace(.03, .97, len(table)))
        colors = {'rolling': '#1565c0', 'cross_section': '#e66b15'}
        styles = {'chronos2': '-', 't0': '--', 'timestral': '-.', 'tabpfnts': ':'}
        for i, row in enumerate(table.itertuples()):
            ax.plot(payout.index, payout[row.tag].cumsum(), label=labels[row.tag],
                    color=colors[row.standardization] if grouped else palette[i],
                    linestyle=styles[row.model] if grouped else '-',
                    linewidth={'A': .9, 'B': 1.4, 'C': 2.2}[row.design])
        ax.set(title='E206 cumulative payout — '+('color grouped by standardization' if grouped else 'all 24 configurations'),
               xlabel='Date', ylabel='Cumulative payout score (0.1 CORR + proxy MMC)')
        ax.grid(alpha=.2)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3, fontsize=8)
        name = 'E206_standardization_'+('grouped' if grouped else 'all')+'.png'
        fig.savefig(ROOT/name, dpi=180)
        plt.close(fig)
        plots.append(name)
    document = '<html><head><meta charset="utf-8"></head><body><h1>E206 — Rolling vs cross-sectional standardization</h1>'
    document += (f'<p>All 24 configurations: four foundation models × two standardizations × three designs. '
                 f'Identical common assets across {len(dates)} shared dates; rows sorted by mean payout descending. '
                 'Payout = 0.1 CORR + historical-meta proxy MMC; sharpe.p = mean payout / population standard deviation, '
                 'unannualized; maxdd.p = maximum decline of additive cumulative payout from its running peak, including zero. '
                 'These are payout scores, not investment returns. Descriptive comparison on previously inspected E206 history.</p>')
    document += table[columns].to_html(index=False, float_format=lambda x: f'{x:.6f}')
    document += f'<h2>All time series — individual colors</h2><img style="width:100%" src="{plots[0]}">'
    document += ('<h2>All time series — colors grouped by standardization</h2>'
                 '<p>Blue = rolling; orange = cross_section. Line styles identify models; '
                 'line thickness increases from design A to B to C.</p>'
                 f'<img style="width:100%" src="{plots[1]}"></body></html>')
    page = ROOT/'E206_standardization_comparison.html'
    page.write_text(document)
    assert 'Frozen protocol' not in document and document.count('<table') == 1
    assert table.payout.is_monotonic_decreasing
    subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lovelace', 'python3',
                    str(Path(__file__).with_name('html_pdf.py')), str(page)], check=True, timeout=300)
    print(f'Verified {len(table)} rows, {len(dates)} dates, two 24-series plots: {page.with_suffix(".pdf")}')


if __name__ == '__main__':
    build()
