"""E234: E206 rows with ONE pooled ridge over the asset panel instead of one ridge per asset.

Same inputs as E206 (crypto/v2.0-e205-ts-momentum3; E206's frozen panel, splits, common causal rows and
normalized TSFM states H of Chronos-2, t0, Timestral, TabPFN-TS), same alpha selection by validation
MSE and fresh final fit, but every asset-date row of the panel enters a single ridge fit (one batch,
no asset ID). Rows:

  * design A (own H) and design B (own H + leave-one-out cross-asset distribution) under all six
    H standardizations, 4 models -> 48 pooled fits (rolling window 63/126/252 by validation MSE);
  * the momentum feature ridge (33 momentum bins, train-standardized) pooled -> 1 fit.

Design C has no pooled analogue: its input (all assets' H concatenated in fixed slots) is the same
vector for every asset on a date, so a single pooled model gives every asset the same prediction and
cannot rank. Those 24 rows are reported as not applicable.

Pooled ridge is solved in the primal: chunked float64 X'X on centered training rows, one eigen-
decomposition, every alpha of logspace(-4, 8, 49) from it (wider than E206's 1e-4..1e4 because the
pooled Gram matrix is ~10^3 times larger; boundary hits are recorded). Scored on E206's common
test rows with E206's own scorer so the per-asset and pooled rows share one sample.

    python3 experiments/e234_e206_pooled_ridge.py fit --workers 2
    python3 experiments/e234_e206_pooled_ridge.py score
    python3 experiments/e234_e206_pooled_ridge.py all --workers 2
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from src.analysis import tsfm_momentum_study as N

E206 = REPO/'results/reports/crypto/e206_momentum3_tsfm'
E206_SCR = Path('/scratch/danielco/e206_momentum3_tsfm')
ROOT = REPO/'results/reports/crypto/e234_e206_pooled_ridge'
SCR = Path('/scratch/danielco/e234_e206_pooled_ridge')
TARGET = E.TARGET
ALPHAS = np.logspace(-4, 8, 49)
DESIGNS = ['A', 'B']
BASE = 'momentum_baseline'
save, parquet = E.save, E.parquet


def panel():
    return pd.read_parquet(E206/'panel.parquet')


def normalized(model, norm, phase, window, p):
    """E206's cached normalized H when present; otherwise the same N.normalize, cached under E234."""
    for root in [E206_SCR, SCR]:
        path = root/'normalizations'/model/norm/f'{phase}_{window}'/'H.npy'
        if path.exists():
            return np.load(path, mmap_mode='r')
    common = np.load(E206_SCR/'common_rows.npy')
    h = np.load(E206_SCR/'embeddings'/model/'H_rows.npy', mmap_mode='r')
    valid = np.load(E206_SCR/'embeddings'/model/'valid_rows.npy') & p.covered.to_numpy()
    fit = common & p['train' if phase == 'selection' else 'final_fit'].to_numpy() & np.isfinite(p[TARGET])
    z, _ = N.normalize(h, p.ucid.to_numpy(), p.date.to_numpy(), valid, fit, norm, window)
    path = SCR/'normalizations'/model/norm/f'{phase}_{window}'/'H.npy'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.tmp.npy'), z); path.with_suffix('.tmp.npy').replace(path)
    return np.load(path, mmap_mode='r')


def design(model, norm, phase, window, kind, p):
    h = normalized(model, norm, phase, window, p)
    valid = np.isfinite(h).all(axis=1)
    if kind == 'B':
        x, _ = N.distribution_design(h, p.date.to_numpy(), valid, norm)
        return x, valid & np.isfinite(x).all(axis=1)
    return h, valid


def gram(x, rows, chunk=20000):
    """Column means, centered X'X and the column-std keep mask, in float64, chunked over rows."""
    n, q = len(rows), x.shape[1]
    s = np.zeros(q); ss = np.zeros(q)
    for i in range(0, n, chunk):
        b = np.asarray(x[rows[i:i+chunk]], np.float64); s += b.sum(0); ss += (b*b).sum(0)
    mean = s/n; keep = np.sqrt(np.maximum(ss/n - mean*mean, 0)) > N.FLOOR
    g = np.zeros((keep.sum(), keep.sum()))
    for i in range(0, n, chunk):
        b = np.asarray(x[rows[i:i+chunk]], np.float64)[:, keep] - mean[keep]; g += b.T @ b
    return mean, keep, g


def ridge_path(x, fit_rows, y, query_rows, alphas):
    """Pooled ridge for every alpha via one eigendecomposition; returns [n_query, n_alpha] predictions."""
    mean, keep, g = gram(x, fit_rows)
    ybar = y.mean(); xty = np.zeros(keep.sum())
    for i in range(0, len(fit_rows), 20000):
        b = np.asarray(x[fit_rows[i:i+20000]], np.float64)[:, keep] - mean[keep]; xty += b.T @ (y[i:i+20000] - ybar)
    s, v = np.linalg.eigh(g); proj = v.T @ xty
    coefs = v @ (proj[:, None]/(s[:, None] + np.asarray(alphas)[None, :]))      # [q, n_alpha]
    out = np.empty((len(query_rows), len(alphas)))
    for i in range(0, len(query_rows), 20000):
        b = np.asarray(x[query_rows[i:i+20000]], np.float64)[:, keep] - mean[keep]; out[i:i+20000] = b @ coefs + ybar
    return out, dict(columns=int(x.shape[1]), kept_columns=int(keep.sum()), n_fit=len(fit_rows))


def rows_for(p, valid, phase):
    common = np.load(E206_SCR/'common_rows.npy'); y = p[TARGET].to_numpy()
    fit = np.flatnonzero(common & p['train' if phase == 'selection' else 'final_fit'].to_numpy() & np.isfinite(y) & valid)
    query = np.flatnonzero(p['validation' if phase == 'selection' else 'test'].to_numpy() & valid)
    return fit, query


def fit_config(job):
    model, norm, kind = job
    tag = f'{model}_{norm}_{kind}'; dest = ROOT/'fits'/tag
    if (dest/'COMPLETE.json').exists():
        return tag
    p = panel(); y = p[TARGET].to_numpy(); choices = []
    for window in (N.ROLLING if norm == 'rolling' else [126]):
        x, valid = design(model, norm, 'selection', window, kind, p)
        fit, query = rows_for(p, valid, 'selection')
        pred, info = ridge_path(x, fit, y[fit], query, ALPHAS)
        good = np.isfinite(y[query]); loss = ((pred[good] - y[query][good, None])**2).mean(0)
        j = int(np.argmin(loss)); choices.append(dict(window=window, alpha=float(ALPHAS[j]), mse=float(loss[j]),
                                                     boundary=bool(j in (0, len(ALPHAS)-1)), **info))
        del x
    best = min(choices, key=lambda c: (c['mse'], c['window']))
    x, valid = design(model, norm, 'final', best['window'], kind, p)
    fit, query = rows_for(p, valid, 'final')
    pred, info = ridge_path(x, fit, y[fit], query, [best['alpha']])
    out = p.loc[p.test, ['row', 'symbol', 'ucid', 'date', TARGET, 'meta_model']].copy()
    full = np.full(len(p), np.nan); full[query] = pred[:, 0]; out['prediction'] = full[p.test.to_numpy()]
    parquet(dest/'predictions.parquet', out)
    save(dest/'selection.json', dict(selected=best, candidates=choices, final=info))
    save(dest/'COMPLETE.json', dict(tag=tag, predictions=int(np.isfinite(out.prediction).sum()), time=time.time()))
    print('FIT COMPLETE', tag, best, flush=True)
    return tag


def fit_baseline():
    dest = ROOT/'fits'/BASE
    if (dest/'COMPLETE.json').exists():
        return
    p = panel(); y = p[TARGET].to_numpy(); feats = [c for c in p if c.startswith('feature_')]
    common = np.load(E206_SCR/'common_rows.npy'); finite = np.isfinite(p[feats].to_numpy(float)).all(axis=1)

    def scaled(fit_mask):
        x = p[feats].to_numpy(np.float64); mu = x[fit_mask].mean(0); sd = x[fit_mask].std(0); sd[sd < N.FLOOR] = 1
        return (x - mu)/sd
    tr = np.flatnonzero(common & p.train.to_numpy() & np.isfinite(y) & finite)
    va = np.flatnonzero(common & p.validation.to_numpy() & np.isfinite(y) & finite)
    pred, _ = ridge_path(scaled(np.isin(np.arange(len(p)), tr)), tr, y[tr], va, ALPHAS)
    loss = ((pred - y[va, None])**2).mean(0); j = int(np.argmin(loss))
    ft = np.flatnonzero(common & p.final_fit.to_numpy() & np.isfinite(y) & finite)
    te = np.flatnonzero(p.test.to_numpy() & p.covered.to_numpy() & finite)
    pred, info = ridge_path(scaled(np.isin(np.arange(len(p)), ft)), ft, y[ft], te, [ALPHAS[j]])
    out = p.loc[p.test, ['row', 'symbol', 'ucid', 'date', TARGET, 'meta_model']].copy()
    full = np.full(len(p), np.nan); full[te] = pred[:, 0]; out['prediction'] = full[p.test.to_numpy()]
    parquet(dest/'predictions.parquet', out)
    save(dest/'selection.json', dict(alpha=float(ALPHAS[j]), mse=float(loss[j]), boundary=bool(j in (0, len(ALPHAS)-1)), final=info))
    save(dest/'COMPLETE.json', dict(tag=BASE, time=time.time()))
    print('FIT COMPLETE', BASE, flush=True)


def daily_scores(frame, rows, base):
    """E206's scorer, restricted to E206's common test rows."""
    g = base.drop(columns=['prediction']).merge(frame[['row', 'prediction']], on='row', how='left', validate='1:1')
    out = []
    for d, day in g.groupby('date', sort=True):
        day = day.sort_values('ucid'); valid = day.row.isin(rows).to_numpy() & np.isfinite(day.prediction.to_numpy())
        finite = valid & np.isfinite(day[TARGET].to_numpy()) & np.isfinite(day.meta_model.to_numpy())
        if finite.sum() < 10:
            continue
        c, m = E.matrix_scores(day.prediction.to_numpy()[finite, None], day[TARGET].to_numpy()[finite], day.meta_model.to_numpy()[finite])
        out.append(dict(date=d, n=int(finite.sum()), corr=float(c[0]) if np.isfinite(c[0]) else 0., mmc=float(m[0]),
                        missing_on_common=int(day.row.isin(rows).sum() - valid.sum())))
    return pd.DataFrame(out)


def score():
    base = pd.read_parquet(E206/'momentum_baseline.parquet')
    grid = json.loads((E206/'grid.json').read_text())
    common = set(base.loc[np.isfinite(base.prediction), 'row'])
    for c in grid:
        f = pd.read_parquet(E206/'fits'/c['id']/'predictions.parquet'); common &= set(f.loc[np.isfinite(f.prediction), 'row'])
    assert len(common) == json.loads((E206/'common_test_coverage.json').read_text())['rows'], 'E206 common rows not reproduced'
    # Parity: E206's stored per-asset scores reproduced by this scorer.
    ref = pd.read_parquet(E206/'scores'/f'{grid[0]["id"]}_common.parquet')
    mine = daily_scores(pd.read_parquet(E206/'fits'/grid[0]['id']/'predictions.parquet'), common, base)
    parity = float(np.abs(ref.set_index('date')[['corr', 'mmc']] - mine.set_index('date')[['corr', 'mmc']]).max().max())
    assert parity < 1e-6, parity   # cross-host BLAS noise ~1e-8
    (ROOT/'scores').mkdir(parents=True, exist_ok=True)
    for d in sorted((ROOT/'fits').iterdir()):
        if (d/'COMPLETE.json').exists():
            s = daily_scores(pd.read_parquet(d/'predictions.parquet'), common, base)
            s.to_parquet(ROOT/'scores'/f'{d.name}_common.parquet')
            print('SCORE', d.name, round(float((0.1*s['corr'] + s.mmc).mean()), 5), 'missing', int(s.missing_on_common.sum()), flush=True)
    save(ROOT/'SCORED.json', dict(common_rows=len(common), parity_max_abs=parity, time=time.time()))


def run_fits(workers):
    ROOT.mkdir(parents=True, exist_ok=True); SCR.mkdir(parents=True, exist_ok=True)
    fit_baseline()
    jobs = [(m, n, k) for m in N.MODELS for n in N.NORMS for k in DESIGNS]
    with ProcessPoolExecutor(workers) as pool:
        for i, tag in enumerate(pool.map(fit_config, jobs), 1):
            save(ROOT/'status.json', dict(stage='fitting', done=i, expected=len(jobs), last=tag, time=time.time()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['fit', 'score', 'all'])
    ap.add_argument('--workers', type=int, default=2); a = ap.parse_args()
    if a.stage in ('fit', 'all'):
        run_fits(a.workers)
    if a.stage in ('score', 'all'):
        score(); save(ROOT/'COMPLETE.json', dict(experiment='E234', time=time.time()))
