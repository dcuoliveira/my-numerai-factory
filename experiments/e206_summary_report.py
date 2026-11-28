"""E206 summary: payout-ranked table and cumulative payout from the stored per-date scores.

Reads results/reports/crypto/e206_momentum3_tsfm/scores/*_common.parquet (72 TSFM configurations +
the momentum feature-ridge baseline, all on E206's common test sample), rebuilds payout per date as
0.1*corr + mmc, summarises once, and adds a paired Newey-West (lag 20) payout delta against the
momentum baseline. Writes E206_summary.{html,pdf} beside the study's other reports.

    python3 experiments/e206_summary_report.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.analysis.cross_target import newey_west
from src.prediction.metrics import summarize

ROOT = REPO/'results/reports/crypto/e206_momentum3_tsfm'
BASE = 'momentum_baseline'


def load():
    rows = {}
    for f in sorted((ROOT/'scores').glob('*_common.parquet')):
        name = f.name.replace('_common.parquet', '')
        name = BASE if name == BASE else name.split('momentum3_')[1].removesuffix('_r01')
        d = pd.read_parquet(f, columns=['date', 'corr', 'mmc']).set_index('date')
        rows[name] = d
    dates = sorted(set.intersection(*(set(d.index) for d in rows.values())))
    return {k: v.loc[dates] for k, v in rows.items()}, pd.DatetimeIndex(dates)


def table(series):
    base_pay = 0.1*series[BASE]['corr'] + series[BASE]['mmc']
    out = []
    for name, d in series.items():
        pay = 0.1*d['corr'] + d['mmc']
        s = summarize(pay)
        model, norm, design = (name.split('_', 1)[0], *name.split('_', 1)[1].rsplit('_', 1)) if name != BASE else ('momentum', 'feature ridge', '-')
        delta = pay - base_pay
        nw = newey_west(delta.to_numpy()) if name != BASE else None
        t = nw['t'] if isinstance(nw, dict) else (nw[1] if isinstance(nw, tuple) else nw)
        out.append(dict(model=model, standardization=norm, design=design, corr=d['corr'].mean(), mmc=d['mmc'].mean(),
                        payout=s['mean'], **{'sharpe.p': s['sharpe'], 'maxdd.p': s['max_drawdown']},
                        **{'burn(d)': int((pay < 0).sum()), 'dpay': delta.mean() if name != BASE else np.nan,
                           't': t if name != BASE else np.nan, 'n': len(pay)}, key=name))
    return pd.DataFrame(out).sort_values('payout', ascending=False).reset_index(drop=True)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.analysis.html_pdf import to_pdf
    series, dates = load(); t = table(series)
    t.drop(columns='key').to_csv(ROOT/'E206_summary_table.csv', index=False)
    best_per_model = t[t.model != 'momentum'].groupby('model').head(1)
    picks = list(dict.fromkeys(t[t.model != 'momentum'].head(5).key.tolist() + best_per_model.key.tolist())) + [BASE]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for k in picks:
        d = series[k]; pay = (0.1*d['corr'] + d['mmc']).cumsum()
        ax.plot(dates, pay, lw=2.4 if k == BASE else 1.3, ls='--' if k == BASE else '-', color='k' if k == BASE else None,
                label=('momentum feature ridge (baseline)' if k == BASE else k) + f'  {pay.iloc[-1]/len(pay):.4f}')
    ax.set_title('E206 cumulative payout (0.1 CORR + MMC), common test sample'); ax.grid(alpha=.25)
    ax.legend(fontsize=8, title='mean payout per date', title_fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(ROOT/'E206_summary_cumulative_payout.png', dpi=110); plt.close(fig)
    fmt = lambda v: f'{v:.5f}' if isinstance(v, float) else v
    show = t.drop(columns='key').copy(); show['t'] = show['t'].map(lambda v: '' if pd.isna(v) else f'{v:+.2f}')
    show['dpay'] = show['dpay'].map(lambda v: '' if pd.isna(v) else f'{v:+.5f}')
    body = ('<h1>E206: momentum3 frozen TSFM states, per-asset ridge</h1>'
            '<p>Pre-head hidden states H of Chronos-2, t0, Timestral and TabPFN-TS on <b>crypto/v2.0-e205-ts-momentum3</b> '
            '(target20 + 33 time-series momentum bins, 35-day label wall). Each configuration = model × H standardization × '
            'design (A own H, B distribution design, C concatenation across a fixed asset universe), one dual-ridge head per asset. '
            f'All {len(t)} rows share one common test sample: {len(dates)} dates, {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}. '
            'Payout is rebuilt per date as 0.1 CORR + MMC (historical-meta proxy), then summarised; '
            '<b>dpay / t</b> is the paired payout difference against the momentum feature ridge baseline, Newey-West lag 20.</p>')
    body += '<h2>Cumulative payout</h2><p>Top five configurations, the best of each model, and the baseline.</p><img style="width:100%" src="E206_summary_cumulative_payout.png">'
    body += '<h2>All configurations, sorted by payout</h2>' + show.to_html(index=False, float_format=fmt)
    body += ('<h2>Limitations</h2><ul><li>Development history; 72 configurations compared, t-statistics unadjusted for the search.</li>'
             '<li>Pretrained checkpoints may overlap the test period.</li><li>MMC is against the historical meta-model proxy.</li></ul>')
    page = ROOT/'E206_summary.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:8.5px;border-collapse:collapse}'
                    'td,th{border:1px solid #ccc;padding:2px 4px}</style></head><body>' + body + '</body></html>')
    to_pdf(str(page), 'lovelace')
    pd.set_option('display.width', 200)
    print(t.drop(columns=['key']).head(15).to_string(index=False, float_format=lambda v: f'{v:.5f}'))
    print(t[t.model == 'momentum'].drop(columns='key').to_string(index=False, float_format=lambda v: f'{v:.5f}'))
    print('rank of baseline:', int(t.index[t.model == 'momentum'][0]) + 1, 'of', len(t))


if __name__ == '__main__':
    main()
