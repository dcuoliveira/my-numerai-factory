"""One table of everything tested on the reversal dataset (E219 -> crypto/v2.0-e219-ts-reversal187).

Two blocks, because they do not share a sample:
  A  E219 per-asset reversal formulas (ridge / Torch MLP, routed full universe, 468 dates),
     read from results/reports/crypto/e219_asset_reversal/table.csv and paired_deltas.csv.
  B  TSFM states on reversal187: every finished E221 portfolio, every E222/E223 Chronos-2
     T/F/TF cell and the E223 ensembles, re-scored here on ONE common sample (the asset/date
     rows where every included prediction is finite), so rows in B are paired.
E224 and the unfinished E221 portfolios are named as pending, not dropped silently.

    python3 experiments/reversal187_summary_report.py      # run on lovelace (PDF printing)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e221_reversal187_tsfm as S
from experiments import e223_e224_featmean_norms as M
from src.analysis.cross_target import newey_west
from src.analysis.e218_ktz_report import metrics
from e150_finish_scoring import matrix_scores

R = REPO/'results/reports/crypto'
OUT = R/'reversal187_summary'
E219 = R/'e219_asset_reversal'
E221 = R/'e221_reversal187_tsfm'


def block_a():
    t = pd.read_csv(E219/'table.csv')
    t = t[(t['sample'] == 'routed') & (t.seed == 'ensemble')].copy()
    d = pd.read_csv(E219/'paired_deltas.csv')
    d = d[(d['sample'] == 'routed') & (d.seed == 'ensemble') & (d.reference == 'asset_mean') & (d.metric == 'payout')]
    t = t.merge(d[['task', 'delta', 'hac20_t']], on='task', how='left')
    learner = t.task.str.split('__').str[0].map({'ridge': 'ridge', 'mlp_torch': 'mlp_torch (5-seed avg)', 'both': 'ridge+mlp'})
    feat = t.task.str.split('__', n=1).str[1].str.replace('REV17__fixed_equal_rank', 'ensemble of 17 formulas (equal rank)').str.replace('momentum132', 'momentum132 (matched momentum control, not reversal)')
    return pd.DataFrame(dict(block='A', study='E219', what=feat, inputs='own-history reversal (11 windows)', learner=learner,
                             norm='TS bins', corr=t['corr'], mmc=t.mmc, payout=t.payout, **{'sharpe.p': t.p_sharpe, 'maxdd.p': t.maxdd_payout},
                             n_dates=t.n_dates, assets=t.coins, d_pay=t.delta, t=t.hac20_t, reference='per-asset historical mean'))


def block_b():
    p = S.pd.read_parquet(E221/'panel.parquet')
    test = p.loc[p.test, ['row', 'ucid', 'date', S.TARGET, 'meta_model']].sort_values(['date', 'ucid']).reset_index(drop=True)
    cols, meta = [], []
    pending = [t for t in S.portfolios() if not (E221/'fits'/t/'COMPLETE.json').exists()]
    for tag in S.portfolios():
        if tag in pending:
            continue
        cols.append(pd.read_parquet(E221/'fits'/tag/'predictions.parquet').set_index('row').prediction)
        tsfm, inputs, learner = (['-', 'X'] + [tag[2:]]) if tag.startswith('X_') else [tag.split('_')[0], tag.split('_')[1], tag.split('_', 2)[2]]
        meta.append(dict(study='E221', what=f'{tsfm} {inputs}' if tsfm != '-' else 'reversal187 only (X control)',
                         inputs={'H': 'TSFM state', 'HX': 'TSFM state + reversal187', 'X': 'reversal187'}[inputs],
                         learner=learner, norm=S.NORMS.get(tsfm, '-'), key=tag))
    for n in M.NORMS:
        for a in M.ARMS:
            f = M.fit_dir('chronos2', n, a)/'predictions.parquet'
            if f.exists():
                cols.append(pd.read_parquet(f).set_index('row').prediction)
                meta.append(dict(study='E222' if n == 'rolling' else 'E223', what=f'chronos2 {M.LABEL[a]}',
                                 inputs='TSFM state', learner='ridge', norm=n, key=f'{n}/{a}'))
    raw = np.column_stack([c.reindex(test.row).to_numpy(float) for c in cols])
    ok = np.isfinite(raw).all(1) & np.isfinite(test[S.TARGET]).to_numpy() & np.isfinite(test.meta_model).to_numpy()
    keys = [m['key'] for m in meta]
    # E223 ensembles, rebuilt on this sample from their member cells
    ens = pd.read_csv(M.REPORT/'ensembles_chronos2.csv')
    cells = [k for k in keys if '/' in k]
    groups = {'all 18 Chronos-2 cells': cells}
    for a in M.ARMS:
        groups[f'all 6 Chronos-2 {a}'] = [k for k in cells if k.endswith('/'+a)]
    for _, e in ens[ens.kind != 'fixed'].sort_values('payout', ascending=False).head(3).iterrows():
        groups[e.ensemble] = [s.strip() for s in e.ensemble.split('+')]
    split = pd.read_csv(M.REPORT/'ensembles_split_chronos2.csv').iloc[0]
    groups[split.chosen + ' (split-half pick)'] = [s.strip() for s in split.chosen.split('+')]
    dates, scores, counts = [], [], []
    for date, idx in test.groupby('date', sort=True).indices.items():
        rows = idx[ok[idx]]
        if len(rows) < 10:
            continue
        X = raw[rows]; Rk = pd.DataFrame(X).rank(pct=True).to_numpy()
        E = np.column_stack([Rk[:, [keys.index(k) for k in g]].mean(1) for g in groups.values()])
        c, m = matrix_scores(np.column_stack([X, E]), test.loc[rows, S.TARGET].to_numpy(), test.loc[rows, 'meta_model'].to_numpy())
        dates.append(date); scores.append(np.column_stack([c, m])); counts.append(len(rows))
    scores = np.stack(scores); pay = .1*scores[:, :, 0]+scores[:, :, 1]
    for g, mem in groups.items():
        meta.append(dict(study='E223', what=f'ensemble: {g}', inputs='TSFM state', learner=f'rank avg of {len(mem)} ridge cells',
                         norm='mixed' if len({m.split("/")[0] for m in mem}) > 1 else mem[0].split('/')[0], key=g))
    ref = keys.index('chronos2_H_ridge')
    dt = [newey_west(pay[:, j]-pay[:, ref], 20) for j in range(pay.shape[1])]
    out = pd.DataFrame(meta).drop(columns='key')
    out = pd.DataFrame(dict(block='B', **{c: out[c] for c in out}, **metrics(scores), n_dates=len(dates),
                            assets=round(float(np.mean(counts)), 1), d_pay=[d for d, _ in dt], t=[t for _, t in dt],
                            reference='E221 chronos2 H ridge (rolling)'))
    return out, pending, (str(dates[0])[:10], str(dates[-1])[:10]), int(ok.sum())


def main():
    from src.analysis.html_pdf import to_pdf
    OUT.mkdir(parents=True, exist_ok=True)
    a = block_a(); b, pending, (d0, d1), nkeys = block_b()
    every = pd.concat([a, b], ignore_index=True).sort_values('payout', ascending=False)
    every.insert(0, 'rank', range(1, len(every)+1))
    every.to_csv(OUT/'reversal187_everything.csv', index=False)
    fmt = lambda v: f'{v:.4f}' if isinstance(v, float) else v
    show = ['rank', 'block', 'study', 'what', 'inputs', 'learner', 'norm', 'payout', 'corr', 'mmc', 'sharpe.p', 'maxdd.p',
            'n_dates', 'assets', 'd_pay', 't']
    html = lambda df: df[show].to_html(index=False, float_format=fmt, na_rep='')
    blk = lambda k: every[every.block == k].assign(rank=lambda d: range(1, len(d)+1))
    body = (
        '<h1>Everything tested on the reversal dataset</h1>'
        '<p>Dataset <code>crypto/v2.0-e219-ts-reversal187</code>: 17 per-asset reversal formulas &times; 11 windows, '
        'own-history TS bins (E219), coverage_required missing-value policy. Payout = 0.1 CORR + MMC. Every table is '
        'sorted by payout, descending.</p>'
        '<p><b>Two blocks, two samples.</b> <b>A</b> (E219): per-asset ridge / Torch MLP on each reversal formula, routed '
        'onto the full ~300-coin universe, 468 dates, MMC against the official meta model; &Delta; and t against the '
        'per-asset historical-mean control. <b>B</b> (E221, E222, E223): TSFM hidden states built from reversal187, '
        f're-scored here on one common sample, {d0} to {d1}, {nkeys:,} asset/date rows (about 128 assets per date), '
        'historical-meta proxy MMC; &Delta; and t (Newey-West lag 20, paired) against E221\'s Chronos-2 H ridge, the default '
        'representation. Rows are paired within a block, NOT across blocks: the big table puts both in one ranking for '
        'overview only, and a block-A row and a block-B row differ in universe, dates and MMC.</p>'
        f'<p><b>Not yet included:</b> {len(pending)} E221 portfolios still fitting ({", ".join(pending)}), and E224 (t0 and '
        'Timestral feature-row states), still extracting. E219 per-seed MLP rows and its covered-sample rows are left out; '
        'the 5-seed averages and routed sample are shown.</p>'
        '<p>Glossary. T = target-row TSFM state, F = mean of the 187 feature-row states, TF = both. H = design-C TSFM state, '
        'HX = H plus the asset\'s 187 reversal columns, X = reversal187 alone. Ensembles are equal-weight per-date rank '
        'averages; the in-sample screen picks are upper bounds, the split-half pick was chosen on the first half of dates.</p>'
        '<h2>All results</h2>' + html(every) +
        '<h2>Block A: E219 reversal formulas (468 dates, routed universe)</h2>' + html(blk('A')) +
        '<h2>Block B: TSFM states on reversal187 (common sample)</h2>' + html(blk('B')) +
        '<h2>Limitations</h2><ul><li>Many comparisons, t-statistics unadjusted.</li>'
        '<li>Block B is a development-history test window; pretrained checkpoints may overlap it; no live path.</li>'
        '<li>The E223 ensemble picks were screened on this same window.</li></ul>')
    page = OUT/'reversal187_everything.html'
    page.write_text('<html><head><meta charset="utf-8"><style>@page{size:A4 landscape;margin:10mm}body{font:10px Arial}'
                    'table{font-size:7.5px;border-collapse:collapse;margin-bottom:8px}td,th{border:1px solid #ccc;padding:2px 3px}'
                    '</style></head><body>' + body + '</body></html>')
    to_pdf(str(page), 'lovelace')
    print(every[show].head(25).to_string(index=False))
    print(page.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
