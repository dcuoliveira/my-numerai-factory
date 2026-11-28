"""Momentum3 TSFM hidden-state summary: four sets of ridge forecasts on one common sample.

Sets (input x head), all on crypto/v2.0-e205-ts-momentum3 (E206's frozen panel):
  * target state, per-asset  -- E206: target-row H, one dual ridge per asset (designs A/B/C), + momentum ridge
  * target state, pooled     -- E234: target-row H, one ridge over the whole panel (A/B), + momentum ridge pooled
  * feature stats, per-asset -- E235: mean/std/skew of the 33 feature-row states, per asset (A/B/C)
  * feature stats, pooled    -- E235: same input, one ridge over the panel (A/B)
Every set present is re-scored from its predictions with E206's scorer on ONE common sample: E206's
common test rows that every included forecast covers. Payout is rebuilt per date as 0.1*corr + mmc and
summarised once; dpay/t is the paired Newey-West (lag 20) difference against the per-asset momentum
ridge. Writes momentum_new_tsfm_hs_summary.{html,pdf} beside E206's other reports.

    python3 experiments/e206_summary_report.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from src.analysis.cross_target import newey_west
from src.prediction.metrics import summarize

CRYPTO = REPO/'results/reports/crypto'
ROOT = CRYPTO/'e206_momentum3_tsfm'
BASE = 'momentum_baseline'
TARGET_SET, FEATURE_SET = 'target state', 'feature stats'
SETS = [  # (input, head, fits folder, how to name a fit folder)
    (TARGET_SET, 'per-asset', ROOT/'fits', lambda n: n.split('momentum3_')[1].removesuffix('_r01')),
    (TARGET_SET, 'pooled', CRYPTO/'e234_e206_pooled_ridge'/'fits', lambda n: n),
    (FEATURE_SET, 'per-asset', CRYPTO/'e235_mom3_featstats_hs'/'per_asset'/'fits', lambda n: n.split('momentum3_')[1].removesuffix('_r01')),
    (FEATURE_SET, 'pooled', CRYPTO/'e235_mom3_featstats_hs'/'pooled'/'fits', lambda n: n),
]
REF = (TARGET_SET, 'per-asset', BASE)
SHORT = {(TARGET_SET, 'per-asset'): 'T/asset', (TARGET_SET, 'pooled'): 'T/pooled',
         (FEATURE_SET, 'per-asset'): 'F/asset', (FEATURE_SET, 'pooled'): 'F/pooled'}


def predictions():
    out = {}
    base = pd.read_parquet(ROOT/'momentum_baseline.parquet')
    out[REF] = base[['row', 'prediction']]
    for inp, head, folder, rename in SETS:
        if not folder.exists():
            continue
        for d in sorted(folder.iterdir()):
            if (d/'COMPLETE.json').exists() and (d/'predictions.parquet').exists():
                out[(inp, head, rename(d.name) if d.name != BASE else BASE)] = pd.read_parquet(d/'predictions.parquet', columns=['row', 'prediction'])
    return out, base


def scores(preds, base):
    """Common rows across every forecast; each forecast scored there with E206's scorer (cached by row-set hash)."""
    from e234_e206_pooled_ridge import daily_scores
    common = set(base.loc[np.isfinite(base.prediction), 'row'])
    for f in preds.values():
        common &= set(f.loc[np.isfinite(f.prediction), 'row'])
    tag = hashlib.sha256(np.sort(np.fromiter(common, np.int64)).tobytes()).hexdigest()[:12]
    cache = ROOT/'combined_scores'/tag; cache.mkdir(parents=True, exist_ok=True)
    series = {}
    for key, f in preds.items():
        path = cache/('__'.join(key).replace(' ', '_') + '.parquet')
        if not path.exists():
            daily_scores(f, common, base).to_parquet(path)
        series[key] = pd.read_parquet(path).set_index('date')[['corr', 'mmc']]
    dates = sorted(set.intersection(*(set(s.index) for s in series.values())))
    return {k: v.loc[dates] for k, v in series.items()}, pd.DatetimeIndex(dates), len(common)


def payout(d):
    return 0.1*d['corr'] + d['mmc']


def parts(name):
    if name == BASE:
        return 'momentum', 'feature ridge', '-'
    model, rest = name.split('_', 1); norm, design = rest.rsplit('_', 1)
    return model, norm, design


def table(series):
    base_pay = payout(series[REF]); out = []
    for key, d in series.items():
        inp, head, name = key; pay = payout(d); s = summarize(pay); delta = pay - base_pay; ref = key == REF
        out.append(dict(input=inp, head=head, **dict(zip(['model', 'standardization', 'design'], parts(name))),
                        corr=d['corr'].mean(), mmc=d['mmc'].mean(), payout=s['mean'],
                        **{'sharpe.p': s['sharpe'], 'maxdd.p': s['max_drawdown'], 'burn(d)': int((pay < 0).sum()),
                           'dpay': np.nan if ref else delta.mean(), 't': np.nan if ref else newey_west(delta.to_numpy())[1], 'n': len(pay)},
                        key=key))
    return pd.DataFrame(out).sort_values('payout', ascending=False).reset_index(drop=True)


def paired(series, a, b, label):
    """Same model/standardization/design under set a minus set b."""
    out = []
    for (inp, head, name), d in series.items():
        if (inp, head) == a and name != BASE and (*b, name) in series:
            delta = payout(d) - payout(series[(*b, name)]); m, t = newey_west(delta.to_numpy())
            out.append(dict(configuration=name, **{SHORT[a]: payout(d).mean(), SHORT[b]: payout(series[(*b, name)]).mean(), label: m, 't': t}))
    return pd.DataFrame(out).sort_values(SHORT[a], ascending=False) if out else pd.DataFrame()


def plot(series, dates, keys, title, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for k in keys:
        pay = payout(series[k]).cumsum(); is_base = k[2] == BASE
        label = (f'{SHORT[k[:2]]} momentum feature ridge' if is_base else f'{SHORT[k[:2]]}: {k[2]}') + f'  {pay.iloc[-1]/len(pay):.4f}'
        ax.plot(dates, pay, lw=2.4 if is_base else 1.3, ls=('--' if k == REF else ':') if is_base else '-',
                color='k' if is_base else None, label=label)
    ax.set_title(title); ax.grid(alpha=.25); ax.legend(fontsize=8, title='mean payout per date', title_fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    import matplotlib
    matplotlib.use('Agg')
    from src.analysis.html_pdf import to_pdf
    preds, base = predictions(); series, dates, n_rows = scores(preds, base); t = table(series)
    t.drop(columns='key').to_csv(ROOT/'momentum_new_tsfm_hs_summary_table.csv', index=False)
    present = [s[:2] for s in SETS if any(k[:2] == s[:2] and k[2] != BASE for k in series)]
    fmt = lambda v: f'{v:.5f}' if isinstance(v, float) else v
    body = ('<h1>Momentum3 TSFM hidden states: target-row state vs feature-row statistics, per-asset vs pooled heads</h1>'
            '<p>Pretrained Chronos-2, t0, Timestral and TabPFN-TS read <b>crypto/v2.0-e205-ts-momentum3</b> (target20 plus 33 time-series '
            'momentum bins, 35-day label wall) one asset at a time; a ridge head maps their hidden states to the 20-day target. '
            'Configuration = model × H standardization × design (A own input; B own input plus leave-one-out cross-asset distribution; '
            'C inputs of every asset concatenated in fixed slots). Four sets:</p><ul>'
            '<li><b>target state, per-asset</b> (E206): the target row\'s pre-head state; one dual ridge per asset.</li>'
            '<li><b>target state, pooled</b> (E234): same input; one ridge fitted on the whole asset panel at once (all asset-date '
            'rows in one batch, no asset ID).</li>'
            '<li><b>feature stats, per-asset</b> (E235): instead of the target row, the mean, std and skew over the 33 feature rows\' '
            'states (per hidden dimension, same forward pass); one ridge per asset. Chronos-2, t0, Timestral (TabPFN-TS has no feature rows).</li>'
            '<li><b>feature stats, pooled</b> (E235): same input, one ridge over the whole panel.</li></ul>'
            '<p>Design C has no pooled version: its input is the same vector for every asset on a date, so one pooled model would give '
            'every asset the same prediction. Alpha (and the rolling window) by validation MSE, fresh final fit, in every set. '
            f'All {len(t)} rows are re-scored on one common sample: {len(dates)} dates, {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, '
            f'{n_rows:,} asset-dates (E206\'s common test rows that every forecast covers). Payout = 0.1 CORR + MMC (historical-meta proxy), '
            'rebuilt per date; <b>dpay / t</b> = paired payout difference against the per-asset momentum feature ridge, Newey-West lag 20. '
            f'Sets present: {", ".join(SHORT[s] for s in present)} (T = target state, F = feature stats).</p>')
    best_keys = []
    for s in present:
        sub = t[(t.input == s[0]) & (t['head'] == s[1]) & (t.model != 'momentum')]
        keys = list(dict.fromkeys(sub.head(5).key.tolist() + sub.groupby('model').head(1).key.tolist()))
        keys += [k for k in [(*s, BASE), REF] if k in series and k not in keys]
        fname = f'summary_cum_{SHORT[s].replace("/", "_")}.png'
        plot(series, dates, keys, f'{s[0]}, {s[1]} ridge: cumulative payout', ROOT/fname)
        body += f'<h2>Cumulative payout: {s[0]}, {s[1]}</h2><p>Top five, best of each model, momentum ridge.</p><img style="width:100%" src="{fname}">'
        best_keys.append(sub.key.iloc[0])
    plot(series, dates, best_keys + [k for k in [REF, (TARGET_SET, 'pooled', BASE)] if k in series], 'Best configuration of each set', ROOT/'summary_cum_best_by_set.png')
    body += '<h2>Best configuration of each set</h2><img style="width:100%" src="summary_cum_best_by_set.png">'
    best = t[t.model != 'momentum'].groupby(['input', 'head']).head(1).drop(columns='key')
    body += '<h2>Best of each set</h2>' + best.to_html(index=False, float_format=fmt)
    show = t.drop(columns='key').copy()
    show['t'] = show['t'].map(lambda v: '' if pd.isna(v) else f'{v:+.2f}'); show['dpay'] = show['dpay'].map(lambda v: '' if pd.isna(v) else f'{v:+.5f}')
    body += '<h2>All configurations, sorted by payout</h2>' + show.to_html(index=False, float_format=fmt)
    pfmt = lambda v: f'{v:+.5f}' if abs(v) < 1 else f'{v:+.2f}'
    for a, b, label, title in [((TARGET_SET, 'pooled'), (TARGET_SET, 'per-asset'), 'pooled − per-asset', 'Target state: pooled minus per-asset'),
                               ((FEATURE_SET, 'pooled'), (FEATURE_SET, 'per-asset'), 'pooled − per-asset', 'Feature stats: pooled minus per-asset'),
                               ((FEATURE_SET, 'per-asset'), (TARGET_SET, 'per-asset'), 'F − T', 'Per-asset: feature stats minus target state'),
                               ((FEATURE_SET, 'pooled'), (TARGET_SET, 'pooled'), 'F − T', 'Pooled: feature stats minus target state')]:
        pt = paired(series, a, b, label)
        if len(pt):
            body += f'<h2>{title}, same configuration (Newey-West lag 20)</h2>' + pt.to_html(index=False, float_format=pfmt)
    body += ('<h2>Limitations</h2><ul><li>Development history; many configurations compared, t-statistics unadjusted for the search.</li>'
             '<li>Pretrained checkpoints may overlap the test period.</li><li>MMC is against the historical meta-model proxy.</li>'
             '<li>Pooled heads use a wider alpha grid (to 1e8) than per-asset heads (to 1e4) because the pooled Gram matrix is far larger.</li></ul>')
    page = ROOT/'momentum_new_tsfm_hs_summary.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:8px;border-collapse:collapse}'
                    'td,th{border:1px solid #ccc;padding:2px 4px}</style></head><body>' + body + '</body></html>')
    to_pdf(str(page), 'lovelace')
    pd.set_option('display.width', 240)
    print('common rows', n_rows, 'dates', len(dates))
    print(best.to_string(index=False, float_format=lambda v: f'{v:.5f}'))
    print(t.drop(columns='key').head(25).to_string(index=False, float_format=lambda v: f'{v:.5f}'))


if __name__ == '__main__':
    main()
