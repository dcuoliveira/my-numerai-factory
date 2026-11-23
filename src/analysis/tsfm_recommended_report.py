"""Render the recommended TSFM designs against stored common-asset controls."""
from pathlib import Path

import numpy as np
import pandas as pd


RECOMMENDED = {'chronos2': 'rolling', 't0': 'rolling',
               'tabpfnts': 'asset', 'timestral': 'robust'}


def render(root, table):
    """Build a matched-date summary and plots without fitting or rescoring."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    recommended = (table.design.eq('C') &
                   table.standardization.eq(table.model.map(RECOMMENDED)))
    selected = table.loc[table.universe.eq('common') & table.side.eq('neutral') &
                         (recommended | table.model.eq('momentum'))].copy()
    assert set(selected.model) == set(RECOMMENDED) | {'momentum'}
    assert not selected.model.duplicated().any()
    frames = {}
    labels = {}
    for row in selected.itertuples():
        frame = pd.read_parquet(root/'scores'/f'{row.tag}_common.parquet').set_index('date')
        assert frame.index.is_unique
        frames[row.tag] = frame[['corr', 'mmc', 'n']].dropna()
        labels[row.tag] = ('Momentum ridge baseline' if row.model == 'momentum'
                           else f'{row.model} | {row.standardization} | C')
    dates = next(iter(frames.values())).index
    for frame in frames.values():
        dates = dates.intersection(frame.index)
    dates = dates.sort_values()
    assert len(dates), 'No shared score dates'
    reference_n = next(iter(frames.values())).loc[dates, 'n'].to_numpy()
    payouts = {}
    for tag, frame in frames.items():
        frame = frame.loc[dates]
        np.testing.assert_array_equal(frame.n.to_numpy(), reference_n)
        mask = selected.tag.eq(tag)
        selected.loc[mask, 'corr'] = frame['corr'].mean()
        selected.loc[mask, 'mmc'] = frame.mmc.mean()
        selected.loc[mask, 'payout'] = (.1*frame['corr'] + frame.mmc).mean()
        selected.loc[mask, 'score_dates'] = len(dates)
        payouts[tag] = .1*frame['corr'] + frame.mmc
        volatility = payouts[tag].std(ddof=0)
        selected.loc[mask, 'sharpe.p'] = payouts[tag].mean()/volatility if volatility > 0 else np.nan
    selected['score_dates'] = selected.score_dates.astype(int)
    selected = selected.drop(columns=['sharpe'])
    selected = selected.sort_values('payout', ascending=False, kind='stable')
    selected.to_csv(root/'recommended_common_results.csv', index=False)
    daily = pd.DataFrame(payouts).loc[:, selected.tag]
    daily.columns = [labels[tag] for tag in selected.tag]
    daily.to_csv(root/'recommended_common_daily_payout.csv', index_label='date')
    correlation = daily.corr()
    correlation.to_csv(root/'recommended_common_payout_correlation.csv')

    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    daily.cumsum().plot(ax=ax, linewidth=1.8)
    ax.set(title='Cumulative payout — common assets and dates',
           xlabel='Date', ylabel='Cumulative payout score (0.1 CORR + MMC)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.savefig(root/'recommended_common_cumulative_payout.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(daily.columns)), daily.columns, rotation=35, ha='right')
    ax.set_yticks(range(len(daily.columns)), daily.columns)
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            value = correlation.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color='white' if abs(value) > .6 else 'black')
    ax.set_title('Correlation of daily payouts — common assets and dates')
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    fig.savefig(root/'recommended_common_payout_correlation.png', dpi=180)
    plt.close(fig)

    columns = ['model', 'standardization', 'design', 'score_dates', 'mean_coverage',
               'corr', 'mmc', 'payout', 'rank_ic', 'mse', 'sharpe.p', 'max_drawdown']
    return ('<h2>Recommended configurations and momentum baseline — common assets</h2>'
            '<p>Design C with each model’s recommended standardization; sorted by payout descending. '
            'The stored momentum control is an independent per-asset ridge on the 33 raw momentum features. '
            'All payout curves use identical common assets and shared score dates. '
            'sharpe.p is mean payout divided by its population standard deviation, without annualization. '
            'Drawdown still refers to the neutral return portfolio; payout is a Numerai score, '
            'not investment return. Correlations are Pearson correlations of daily payouts, '
            'not cumulative curves.</p>' +
            selected[columns].to_html(index=False, float_format=lambda x: f'{x:.6f}') +
            '<img style="width:100%" src="recommended_common_cumulative_payout.png">'
            '<img style="width:85%" src="recommended_common_payout_correlation.png">')
