"""E223 / E224: E222's T vs F vs TF ridge across H standardizations and across TSFMs.

E223 -- Chronos-2, E222's cache unchanged, standardizations pooled, asset, robust,
        cross_section, l2 (rolling is E222 itself).
E224 -- t0 and Timestral: same one-pass read-out (target-row state + mean of the
        187 covariate-row states, src/data_mani/e224_featmean_embeddings.py),
        all six standardizations including rolling.

Everything else is E222: E221's frozen panel, per-asset dual ridge on design C,
33 alphas by validation MSE, fresh final fit, E206 target20 splits, rolling
window 63/126/252 by validation MSE. One common causal sample per TSFM (shared
by its 3 arms x 6 standardizations). Arms: T target row, F covariate-row mean,
TF concatenation.

    python3 experiments/e223_e224_featmean_norms.py extract --model t0      # GPU worker
    python3 experiments/e223_e224_featmean_norms.py all --study E223|E224 --workers 3
    python3 experiments/e223_e224_featmean_norms.py report                  # combined PDF
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from experiments import e221_reversal187_tsfm as S
from experiments import e222_c2_featmean_ridge as C
from src.analysis import tsfm_momentum_study as N
from src.analysis.cross_target import newey_west
from src.analysis.e218_ktz_report import metrics
from e150_finish_scoring import matrix_scores

TARGET = S.TARGET
NORMS = ['rolling', 'pooled', 'asset', 'robust', 'cross_section', 'l2']
ARMS = ['T', 'F', 'TF']
LABEL = {'T': 'T (target row)', 'F': 'F (feature-row mean)', 'TF': 'TF (concatenated)'}
STUDY = {'E223': dict(root=REPO/'results/reports/crypto/e223_c2_featmean_norms', tsfms=['chronos2'],
                      norms=[n for n in NORMS if n != 'rolling']),
         'E224': dict(root=REPO/'results/reports/crypto/e224_t0ts_featmean_norms', tsfms=['t0', 'timestral'], norms=NORMS)}
REPORT = REPO/'results/reports/crypto/e222_featmean_combined'
SCR224 = Path('/scratch/danielco/e224_t0ts_featmean_norms')
WIDTH = {'chronos2': 768, 't0': None, 'timestral': None}
save, parquet, sha = E.save, E.parquet, E.sha


def scr(tsfm):
    return C.SCR if tsfm == 'chronos2' else SCR224/tsfm


def arm_dir(tsfm, arm):
    if tsfm == 'chronos2':
        return {'T': 'c2tgt', 'F': 'c2fmean', 'TF': 'c2cat'}[arm]
    return arm


def fit_dir(tsfm, norm, arm):
    if tsfm == 'chronos2' and norm == 'rolling':
        return C.ROOT/'fits'/arm_dir(tsfm, arm)
    study = 'E223' if tsfm == 'chronos2' else 'E224'
    return STUDY[study]['root']/'fits'/f'{tsfm}_{norm}_{arm}'


def bind(tsfm):
    E.ROOT, E.SCR, E.TARGET = S.ROOT, scr(tsfm), TARGET
    S.SCR = scr(tsfm)


# ----------------------------------------------------------------- extraction
def extract(model, smoke=False):
    env = dict(os.environ, TSFM_EMBED_ROOT=str(S.ROOT), TSFM_EMBED_SCR=str(scr(model)), TSFM_EMBED_TARGET=TARGET,
               TSFM_EMBED_WALL=str(S.WALL), E206_BATCH='4')
    cmd = [E.VENVS[model], '-u', '-m', 'src.data_mani.e224_featmean_embeddings', '--model', model, '--device', 'cuda:0']
    subprocess.run(cmd + (['--smoke'] if smoke else []), cwd=REPO, check=True, env=env)


def extraction_complete(p, tsfm):
    if tsfm == 'chronos2':
        return True
    wanted = p.groupby('ucid').covered.any(); d = scr(tsfm)/'embeddings'/tsfm
    return all((d/f'asset_{a}.npz').exists() for a in wanted.index[wanted])


def collect(tsfm):
    """Split the 2d-wide cache into T, F, TF; parity-check T against E221; per-TSFM common mask."""
    d = scr(tsfm)
    if tsfm == 'chronos2' or (d/'EMBEDDINGS_READY.json').exists():
        return
    p = C.panel(); raw = d/'embeddings'/tsfm; manifest = json.loads((raw/'extractor.json').read_text())
    w = manifest['target_width']; assert manifest['width'] == 2*w and manifest['target'] == TARGET
    outs = {}
    for arm, width in [('T', w), ('F', w), ('TF', 2*w)]:
        (d/'embeddings'/arm).mkdir(parents=True, exist_ok=True)
        outs[arm] = np.lib.format.open_memmap(d/'embeddings'/arm/'H_rows.partial.npy', mode='w+', dtype='float16', shape=(len(p), width))
        outs[arm][:] = np.nan
    ref_dir = S.SCR/'embeddings'/tsfm; diffs = []; compared = 0
    for f in sorted(raw.glob('asset_*.npz')):
        assert json.loads(f.with_suffix('.json').read_text())['fingerprint'] == manifest['fingerprint']
        with np.load(f) as z:
            rows, st = z['rows'], z['states']
        if not len(rows):
            continue
        outs['T'][rows] = st[:, :w]; outs['F'][rows] = st[:, w:]; outs['TF'][rows] = st
        if (ref_dir/f.name).exists():
            with np.load(ref_dir/f.name) as r:
                assert np.array_equal(r['rows'], rows), f.name
                diffs.append(float(np.abs(r['states'].astype(np.float32)-st[:, :w].astype(np.float32)).max())); compared += len(rows)
    for arm in ARMS:
        outs[arm].flush(); del outs[arm]
        (d/'embeddings'/arm/'H_rows.partial.npy').replace(d/'embeddings'/arm/'H_rows.npy')
    parity = dict(rows_compared=compared, assets_compared=len(diffs), max_abs_diff=max(diffs) if diffs else None)
    assert diffs and parity['max_abs_diff'] < 0.05, parity
    common = p.covered.to_numpy().copy()
    for arm in ARMS:
        valid = np.isfinite(np.load(d/'embeddings'/arm/'H_rows.npy', mmap_mode='r')).all(axis=1)
        np.save(d/'embeddings'/arm/'valid_rows.npy', valid); common &= valid
    common = N.common_history(p.ucid.to_numpy(), p.date.to_numpy(), common, 32)
    common &= pd.Series(common).groupby(p.date).transform('sum').ge(2).to_numpy()
    np.save(d/'common_rows.npy', common)
    save(d/'EMBEDDINGS_READY.json', dict(tsfm=tsfm, width=w, common_rows=int(common.sum()), t_parity_vs_e221=parity,
                                         mask_sha256=sha(d/'common_rows.npy'), time=time.time()))


# ---------------------------------------------------------------------- ridge
def kernel(tsfm, arm, norm, phase, window, p):
    a = arm_dir(tsfm, arm)
    kern, valid, info = E.build_design(a, norm, phase, window, 'C', p)
    shutil.rmtree(scr(tsfm)/'normalizations'/a/norm/f'{phase}_{window}', ignore_errors=True)   # quota; regenerable
    return kern, valid, info


def fit(job):
    tsfm, norm, arm = job; out = fit_dir(tsfm, norm, arm)
    if (out/'COMPLETE.json').exists():
        return job
    os.environ.setdefault('OMP_NUM_THREADS', '3')
    bind(tsfm); p = C.panel(); choices = []
    for window in (N.ROLLING if norm == 'rolling' else [126]):
        kern, valid, info = kernel(tsfm, arm, norm, 'selection', window, p)
        _, records = S.ridge_portfolio(p, {'selection': (kern, valid)}, None, ('selection',))
        choices.append((float(np.mean([r['mse'] for r in records.values()])), window, info)); del kern
    score, window, info = min(choices, key=lambda c: (c[0], c[1]))
    kernels = {ph: kernel(tsfm, arm, norm, ph, window, p)[:2] for ph in ['selection', 'final']}
    preds, records = S.ridge_portfolio(p, kernels, None)
    res = p.loc[p.test, ['row', 'ucid', 'symbol', 'date', TARGET, 'meta_model']].copy()
    res['prediction'] = preds['final'][p.test.to_numpy()]
    out.mkdir(parents=True, exist_ok=True); parquet(out/'predictions.parquet', res)
    save(out/'COMPLETE.json', dict(tsfm=tsfm, norm=norm, arm=arm, window=window if norm == 'rolling' else None,
         mean_asset_mse=score, assets=len(records), vocabulary=len(info['vocabulary']), time=time.time()))
    return job


def run_all(study, workers):
    cfg = STUDY[study]; root = cfg['root']; root.mkdir(parents=True, exist_ok=True); p = C.panel()
    for tsfm in cfg['tsfms']:
        while not extraction_complete(p, tsfm):
            save(root/'status.json', dict(stage='waiting_for_extraction_'+tsfm, time=time.time())); time.sleep(300)
        save(root/'status.json', dict(stage='collect_'+tsfm, time=time.time())); collect(tsfm)
    jobs = [(t, n, a) for t in cfg['tsfms'] for n in cfg['norms'] for a in ARMS]
    done = 0
    with ProcessPoolExecutor(workers) as pool:
        for _ in pool.map(fit, jobs):
            done += 1; save(root/'status.json', dict(stage='fitting', done=done, expected=len(jobs), time=time.time()))
    report()
    save(root/'COMPLETE.json', dict(experiment=study, fits=len(jobs), time=time.time()))


# --------------------------------------------------------------------- report
def score_tsfm(tsfm):
    """Daily CORR/MMC for every finished (norm, arm) of one TSFM on its common test sample."""
    p = C.panel()
    test = p.loc[p.test, ['row', 'ucid', 'date', TARGET, 'meta_model']].sort_values(['date', 'ucid']).reset_index(drop=True)
    tags = [(n, a) for n in NORMS for a in ARMS if (fit_dir(tsfm, n, a)/'COMPLETE.json').exists()]
    if not tags:
        return None
    raw = np.column_stack([pd.read_parquet(fit_dir(tsfm, n, a)/'predictions.parquet').set_index('row').prediction
                           .reindex(test.row).to_numpy(float) for n, a in tags])
    ok = np.isfinite(raw).all(1) & np.isfinite(test[TARGET]).to_numpy() & np.isfinite(test.meta_model).to_numpy()
    dates, scores, counts, days = [], [], [], []
    for date, idx in test.groupby('date', sort=True).indices.items():
        rows = idx[ok[idx]]
        if len(rows) < 10:
            continue
        y, mm = test.loc[rows, TARGET].to_numpy(), test.loc[rows, 'meta_model'].to_numpy()
        c, m = matrix_scores(raw[rows], y, mm)
        dates.append(date); scores.append(np.column_stack([c, m])); counts.append(len(rows))
        days.append((pd.DataFrame(raw[rows]).rank(pct=True).to_numpy(), y, mm))
    scores = np.stack(scores)
    info = [json.loads((fit_dir(tsfm, n, a)/'COMPLETE.json').read_text()) for n, a in tags]
    table = pd.DataFrame(dict(tsfm=tsfm, norm=[n for n, _ in tags], arm=[LABEL[a] for _, a in tags],
                              window=[i.get('window') or '-' for i in info], n_dates=len(dates),
                              mean_assets=round(float(np.mean(counts)), 1), **metrics(scores)))
    pay = .1*scores[:, :, 0]+scores[:, :, 1]; deltas = []
    for n in NORMS:
        for a, b in [('F', 'T'), ('TF', 'T'), ('TF', 'F')]:
            if (n, a) in tags and (n, b) in tags:
                ia, ib = tags.index((n, a)), tags.index((n, b)); row = dict(tsfm=tsfm, norm=n, comparison=f'{a} - {b}')
                for k, s in [('corr', scores[:, ia, 0]-scores[:, ib, 0]), ('mmc', scores[:, ia, 1]-scores[:, ib, 1]), ('payout', pay[:, ia]-pay[:, ib])]:
                    row[f'd_{k}'], row[f't_{k}'] = newey_west(s, 20)
                deltas.append(row)
    for a in ARMS:      # each standardization against E222's recommended rolling, same arm
        for n in NORMS[1:]:
            if (n, a) in tags and ('rolling', a) in tags:
                ia, ib = tags.index((n, a)), tags.index(('rolling', a)); row = dict(tsfm=tsfm, norm=n, comparison=f'{a}: {n} - rolling')
                for k, s in [('corr', scores[:, ia, 0]-scores[:, ib, 0]), ('mmc', scores[:, ia, 1]-scores[:, ib, 1]), ('payout', pay[:, ia]-pay[:, ib])]:
                    row[f'd_{k}'], row[f't_{k}'] = newey_west(s, 20)
                deltas.append(row)
    r = dict(table=table, deltas=pd.DataFrame(deltas), dates=np.asarray(dates), pay=pay, tags=tags, keys=int(ok.sum()),
             scores=scores)
    r.update(ensembles(tsfm, r, days))
    return r


def ensembles(tsfm, r, days):
    """Equal-weight per-date rank averages of the finished cells of one TSFM, on the same common sample.

    fixed: pre-specified families (all T / all F / all TF / all cells, per-standardization T+F rank blend
    against TF feature concatenation). screen: every 2- and 3-member blend, ranked on the full window
    (in-sample) and, for honesty, chosen on the first half of dates and scored on the second."""
    from itertools import combinations
    tags, k = r['tags'], len(r['tags'])
    name = lambda c: ' + '.join(f'{tags[i][0]}/{tags[i][1]}' for i in c)
    fixed = {f'all {a}': [i for i, t in enumerate(tags) if t[1] == a] for a in ARMS}
    fixed[f'all {k} cells'] = list(range(k))
    for n in NORMS:
        if (n, 'T') in tags and (n, 'F') in tags:
            fixed[f'{n}: T+F rank blend'] = [tags.index((n, 'T')), tags.index((n, 'F'))]
    fixed = {kk: v for kk, v in fixed.items() if len(v) >= 2}
    screen = [c for m in (2, 3) for c in combinations(range(k), m)]
    combos = [tuple(v) for v in fixed.values()] + screen
    W = np.zeros((k, len(combos)))
    for j, c in enumerate(combos):
        W[list(c), j] = 1/len(c)
    sc = np.stack([np.column_stack(matrix_scores(R @ W, y, mm)) for R, y, mm in days])
    pay = .1*sc[:, :, 0]+sc[:, :, 1]; nf = len(fixed)
    single = r['pay']; best = int(single.mean(0).argmax()); bname = f'{tags[best][0]}/{tags[best][1]}'
    def rows(ix, labels, kind):
        mt = metrics(sc[:, ix]); out = pd.DataFrame(dict(kind=kind, ensemble=labels, members=[len(combos[j]) for j in ix], **mt))
        dt = [newey_west(pay[:, j]-single[:, best], 20) for j in ix]
        out['d_pay vs best single'] = [d for d, _ in dt]; out['t'] = [t for _, t in dt]
        return out
    fixed_t = rows(list(range(nf)), list(fixed), 'fixed')
    order = nf + np.argsort(-pay[:, nf:].mean(0))[:10]
    top_t = rows(list(order), [name(combos[j]) for j in order], 'screen (in-sample)')
    h = len(r['dates'])//2; h1, h2 = slice(0, h), slice(h, None)
    je = nf + int(pay[h1, nf:].mean(0).argmax()); js = int(single[h1].mean(0).argmax())
    d, t = newey_west(pay[h2, je]-single[h2, js], 20)
    oos = pd.DataFrame([dict(pick='best 2-3 blend on H1', chosen=name(combos[je]), H1_payout=pay[h1, je].mean(), H2_payout=pay[h2, je].mean()),
                        dict(pick='best single cell on H1', chosen=f'{tags[js][0]}/{tags[js][1]}', H1_payout=single[h1, js].mean(), H2_payout=single[h2, js].mean()),
                        dict(pick='H2: blend - single', chosen=f'NW t {t:+.2f}', H1_payout=np.nan, H2_payout=d)])
    rk = np.mean([np.corrcoef(R, rowvar=False) for R, _, _ in days], axis=0)
    return dict(ens_fixed=fixed_t, ens_top=top_t, ens_oos=oos, ens_best=bname, ens_split=(str(r['dates'][h-1])[:10], str(r['dates'][h])[:10]),
                pred_corr=pd.DataFrame(rk, index=[f'{a}/{b}' for a, b in tags], columns=[f'{a}/{b}' for a, b in tags]))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.analysis.html_pdf import to_pdf
    REPORT.mkdir(parents=True, exist_ok=True)
    have = {t: score_tsfm(t) for t in ['chronos2', 't0', 'timestral']}
    have = {t: r for t, r in have.items() if r is not None}
    e224 = any(t in have for t in ['t0', 'timestral'])
    version = 'v3' if e224 else 'v2'
    fmt = lambda v: f'{v:.5f}' if isinstance(v, float) else v
    cols = ['norm', 'arm', 'window', 'corr', 'mmc', 'payout', 'sharpe.p', 'maxdd.p']
    body = ('<h1>E222 / E223 / E224: TSFM feature-row states alongside the target-row state</h1>'
            '<p>Each TSFM sees one asset as one group: the target20 series (35-day label wall) plus the 187 reversal187 '
            'series as covariates. <b>T</b> is the target row\'s pre-head state (what E221 uses), <b>F</b> the mean of the '
            '187 covariate rows\' states at the same output position from the same forward pass, <b>TF</b> both side by side. '
            'Per-asset dual ridge on design C, alpha by validation MSE, fresh final fit, E206 target20 splits. '
            '<b>E222</b>: Chronos-2, rolling standardization (window 63/126/252 by validation MSE). '
            '<b>E223</b>: the same Chronos-2 states under pooled, asset, robust, cross_section and l2. '
            + ('<b>E224</b>: t0 and Timestral, all six standardizations. ' if e224 else '') +
            'Within a TSFM every arm and standardization is scored on one common sample. Payout = 0.1 CORR + historical-meta proxy MMC.</p>'
            '<p>Standardizations: rolling = per-asset trailing z-score; pooled = one train-fitted z-score; asset = per-asset '
            'train-fitted z-score; robust = train median/IQR; cross_section = per-date z-score across assets; l2 = each row '
            'scaled to unit norm.</p>')
    summary = []
    for tsfm, r in have.items():
        t = r['table']
        best = t.sort_values('payout', ascending=False).iloc[0]
        summary.append(dict(tsfm=tsfm, fits=len(t), best=f"{best.norm} / {best.arm}", best_payout=float(best.payout)))
    body += '<h2>Summary</h2>' + pd.DataFrame(summary).to_html(index=False, float_format=fmt)
    for tsfm, r in have.items():
        parity = '' if tsfm == 'chronos2' else ''
        body += (f'<h2>{tsfm}</h2><p>{len(r["dates"])} test dates, {r["keys"]:,} asset/date rows, identical across the '
                 f'{len(r["tags"])} fits below.</p>')
        body += r['table'].sort_values('payout', ascending=False)[cols].to_html(index=False, float_format=fmt)
        body += '<h3>Paired differences (Newey-West lag 20)</h3>' + r['deltas'].drop(columns='tsfm').to_html(index=False, float_format=fmt)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        for ax, a in zip(axes, ARMS):
            for j, (n, aa) in enumerate(r['tags']):
                if aa == a:
                    ax.plot(r['dates'], r['pay'][:, j].cumsum(), lw=2.2 if n == 'rolling' else 1.1, label=n)
            ax.set_title(f'{tsfm} {LABEL[a]}'); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.tight_layout(); png = REPORT/f'cumulative_{tsfm}.png'; fig.savefig(png, dpi=100); plt.close(fig)
        body += f'<h3>Cumulative payout by standardization</h3><img style="width:100%" src="{png.name}">'
        a0, a1 = r['ens_split']
        body += (f'<h3>Ensembles</h3><p>Equal-weight average of per-date ranks, scored on the same {len(r["dates"])} dates. '
                 f'Deltas are paired Newey-West (lag 20) against the best single cell, <b>{r["ens_best"]}</b>. '
                 f'The 2-3 member screen covers every combination of the {len(r["tags"])} cells; its top rows are picked on the '
                 f'very window they are scored on, so read them as an upper bound. The split check picks on dates up to {a0} '
                 f'and scores from {a1}.</p>')
        body += '<h4>Pre-specified blends</h4>' + r['ens_fixed'].drop(columns='kind').to_html(index=False, float_format=fmt)
        body += '<h4>Top 10 of the 2-3 member screen (in-sample)</h4>' + r['ens_top'].drop(columns='kind').to_html(index=False, float_format=fmt)
        body += '<h4>Split-half check</h4>' + r['ens_oos'].to_html(index=False, float_format=fmt, na_rep='')
        fig, ax = plt.subplots(figsize=(8, 7)); pc = r['pred_corr']
        im = ax.imshow(pc.to_numpy(), vmin=pc.to_numpy().min(), vmax=1, cmap='viridis')
        ax.set_xticks(range(len(pc)), pc.columns, rotation=90, fontsize=7); ax.set_yticks(range(len(pc)), pc.index, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=.8); ax.set_title(f'{tsfm}: mean per-date rank correlation of predictions')
        fig.tight_layout(); png = REPORT/f'predcorr_{tsfm}.png'; fig.savefig(png, dpi=100); plt.close(fig)
        body += f'<h4>Prediction correlation</h4><img style="width:70%" src="{png.name}">'
    body += ('<h2>Limitations</h2><ul><li>Development history; ridge only; many comparisons, t-statistics unadjusted.</li>'
             '<li>TF shares one ridge penalty across both blocks.</li>'
             '<li>Pretrained checkpoints may overlap the test period; no live path.</li></ul>')
    page = REPORT/f'E222_featmean_report_{version}.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:8.5px;border-collapse:'
                    'collapse;margin-bottom:8px}td,th{border:1px solid #ccc;padding:3px}</style></head><body>'+body+'</body></html>')
    to_pdf(str(page), 'lovelace')
    for tsfm, r in have.items():
        r['table'].to_csv(REPORT/f'table_{tsfm}.csv', index=False); r['deltas'].to_csv(REPORT/f'deltas_{tsfm}.csv', index=False)
        pd.concat([r['ens_fixed'], r['ens_top']]).to_csv(REPORT/f'ensembles_{tsfm}.csv', index=False)
        r['ens_oos'].to_csv(REPORT/f'ensembles_split_{tsfm}.csv', index=False); r['pred_corr'].to_csv(REPORT/f'predcorr_{tsfm}.csv')
    return page.with_suffix('.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['extract', 'all', 'report', 'fit'])
    ap.add_argument('--model', choices=['t0', 'timestral']); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--study', choices=list(STUDY)); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--tsfm'); ap.add_argument('--norm'); ap.add_argument('--arm')
    a = ap.parse_args()
    if a.stage == 'extract':
        extract(a.model, a.smoke)
    elif a.stage == 'all':
        run_all(a.study, a.workers)
    elif a.stage == 'fit':
        fit((a.tsfm, a.norm, a.arm))
    else:
        print(report())
