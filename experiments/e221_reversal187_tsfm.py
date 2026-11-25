"""E221: frozen TSFM states on reversal187; per-asset ridge/LightGBM/MLP/TabNet heads.

Chronos-2, t0, TabPFN-TS and Timestral pre-head states H are extracted per asset
from crypto/v2.0-e219-ts-reversal187 (target20 context, 35-day label wall) with
the unchanged E212 extractor. H gets the recommended standardization
(Chronos-2/t0 rolling, TabPFN-TS asset, Timestral robust) and design C (fixed
training-asset vocabulary, own asset once). Two prediction experiments:

  H   design-C H only
  HX  design-C H plus the predicted asset's own 187 reversal187 columns

each fitted from scratch per asset with ridge, lgbm_ktz, mlp_ktz and tabnet_ktz,
plus the matched X-only (reversal187 alone) control per learner. 36 portfolios
on one common causal sample; E206 target20 splits; validation-MSE selection.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_key, '1')
import argparse
import fcntl
import html
import json
import shlex
import signal
import smtplib
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from experiments import e212_volatility99_tsfm as G
from src.analysis import tsfm_momentum_study as N
from src.analysis import asset_ktz_states as K
from src.analysis.asset_momentum import route_slots
from src.analysis.cross_target import newey_west
from src.analysis.e218_ktz_report import metrics
from src.cluster import budget, hosts, manager
from src.live.submit import rank_normalise
from e150_finish_scoring import matrix_scores

ROOT = REPO/'results/reports/crypto/e221_reversal187_tsfm'
SCR = Path('/scratch/danielco/e221_reversal187_tsfm')
DATASET = 'crypto/v2.0-e219-ts-reversal187'
TARGET = 'target_binned_return_20'
WALL = 35
MODELS = ['chronos2', 't0', 'tabpfnts', 'timestral']
NORMS = {'chronos2': 'rolling', 't0': 'rolling', 'tabpfnts': 'asset', 'timestral': 'robust'}
FAMILIES = ['ridge', 'lgbm_ktz', 'mlp_ktz', 'tabnet_ktz']
VARIANTS = ['H', 'HX']
E220_CANDIDATES = REPO/'results/reports/crypto/e220_ja_heads/candidates.json'
PY = '/home/danielco/miniconda3/bin/python3'
# User-authorized 2026-11-25: double the CPU share on the two idle-GPU boxes so E221's
# GPU jobs are not starved by E218 workers; every other host keeps the default share.
DOUBLED_HOSTS = {'deepfive': 2*budget.SHARE, 'deepsix': 2*budget.SHARE}
# User-authorized 2026-11-25 (temporary, extraction stage only): both cards on those two
# hosts when free; a card stays blocked if reserved by any active nmr unit or in use.
DOUBLED_GPU_SLOTS = 2
save, parquet, sha = E.save, E.parquet, E.sha
CODE = [Path(__file__), REPO/'src/data_mani/e212_signal_embeddings.py', REPO/'experiments/e206_momentum3_tsfm.py',
        REPO/'src/analysis/tsfm_momentum_study.py', REPO/'src/analysis/asset_ktz_states.py',
        REPO/'src/analysis/wide_tabnet.py', REPO/'src/live/submit.py', REPO/'src/analysis/asset_momentum.py',
        REPO/'experiments/e150_finish_scoring.py']


def portfolios():
    """36 tags: <model>_<variant>_<family> and X_<family>."""
    tags = [f'{m}_{v}_{f}' for m in MODELS for v in VARIANTS for f in FAMILIES]
    return tags + [f'X_{f}' for f in FAMILIES]


def bind():
    """Point the reused E206 numerical helpers at this study's own paths."""
    E.ROOT, E.SCR, E.TARGET = ROOT, SCR, TARGET


# ---------------------------------------------------------------- preparation
def prepare():
    if (ROOT/'PREPARED.json').exists():
        return verify()
    data = REPO/'data'/DATASET
    manifest = json.loads((data/'manifest.json').read_text())
    for name, digest in manifest['output_hashes'].items():
        assert sha(data/name) == digest, name
    names = manifest['features']
    frame = pd.read_parquet(data/'train.parquet').reset_index().sort_values(['date', 'ucid']).reset_index(drop=True)
    assert [c for c in frame if c.startswith('feature_')] == names and len(names) == 187
    assert not frame.duplicated(['ucid', 'date']).any() and frame.date.max() < pd.Timestamp('2026-02-20')
    cover = pd.read_parquet(data/'coverage.parquet').reset_index()[['ucid', 'date', 'covered']]
    frame = frame.merge(cover, on=['ucid', 'date'], how='left', validate='1:1', sort=False)
    assert frame.covered.notna().all() and np.isfinite(frame.loc[frame.covered, names]).all().all()
    meta = pd.read_parquet(data/'historical_meta_models.parquet').reset_index().rename(columns={'prediction': 'meta_model'})
    frame = frame.merge(meta, on=['symbol', 'date'], how='left', validate='1:1', sort=False)
    frame['row'] = np.arange(len(frame))
    dates, assets = np.sort(frame.date.unique()), np.sort(frame.ucid.unique())
    frame['step'] = np.searchsorted(dates, frame.date)
    frame['asset_position'] = pd.Index(assets).get_indexer(frame.ucid)
    for name, mask in zip(['train', 'validation', 'final_fit', 'test'], N.split_masks(frame.date)):
        frame[name] = mask
    ROOT.mkdir(parents=True, exist_ok=True); SCR.mkdir(parents=True, exist_ok=True)
    parquet(ROOT/'panel.parquet', frame)
    candidates = json.loads(E220_CANDIDATES.read_text())
    grids = {'lgbm_ktz': K.candidates('lgbm_ktz'), 'mlp_ktz': K.candidates('mlp_ktz'),
             'tabnet_ktz': candidates['tabnet_ktz']}
    assert grids['lgbm_ktz'] == candidates['lgbm_ktz'] and grids['mlp_ktz'] == candidates['mlp_ktz']
    save(ROOT/'candidates.json', grids)
    yy = frame[TARGET].to_numpy()
    origins = 0
    for _, g in frame.groupby('ucid'):
        y = np.full(len(dates), np.nan); y[g.step] = np.where(g.covered, yy[g.index], np.nan)
        prefix = np.r_[0, np.cumsum(np.isfinite(y))]
        wall = np.searchsorted(dates, g.date.to_numpy()-np.timedelta64(WALL, 'D'), side='right')-1
        hi = np.maximum(wall+1, 0); lo = np.maximum(0, hi-256)
        origins += int(((prefix[hi]-prefix[lo] >= 32) & g.covered.to_numpy()).sum())
    config = dict(experiment='E221', dataset=DATASET, models=MODELS, normalizations=NORMS, design='C',
        rolling_windows=N.ROLLING, window_rule='rolling window by equal-weight per-asset validation MSE of the H ridge head',
        families=FAMILIES, variants=dict(H='design-C H', HX='design-C H + own 187 reversal187 columns',
                                         X='own 187 reversal187 columns only (matched control)'),
        training_target=TARGET, label_wall_days=WALL, context=256, min_resolved=32,
        splits='train<=2023-11-27; validation 2024-01-01..2024-11-27; final fit<=2024-11-27; test 2025-01-01..2026-02-19',
        alpha_grid=N.ALPHAS.tolist(), ktz='four frozen candidates per family (E220 set), validation MSE, fresh final refit',
        portfolios=portfolios(), code_hashes={str(p.relative_to(REPO)): sha(p) for p in CODE},
        input_hashes={str(data/n): sha(data/n) for n in ['train.parquet', 'coverage.parquet',
                                                          'historical_meta_models.parquet', 'manifest.json']})
    save(ROOT/'config.json', config)
    save(ROOT/'PREPARED.json', dict(experiment='E221', dataset=DATASET, features=names, rows=len(frame),
         assets=assets.tolist(), dates=list(map(str, dates)), panel_hash=sha(ROOT/'panel.parquet'),
         source_covered_rows=int(frame.covered.sum()), eligible_origins_per_model=origins))
    print('PREPARED rows', len(frame), 'covered', int(frame.covered.sum()), 'origins/model', origins, flush=True)
    return config


def verify():
    c = json.loads((ROOT/'config.json').read_text())
    for path, digest in c['code_hashes'].items():
        if sha(REPO/path) != digest:
            raise RuntimeError('Frozen study code changed: '+path)
    for path, digest in c['input_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen dataset input changed: '+path)
    assert sha(ROOT/'panel.parquet') == json.loads((ROOT/'PREPARED.json').read_text())['panel_hash']
    return c


# ----------------------------------------------------------------- extraction
def embed_dir(model):
    return SCR/'embeddings'/model


def extraction_complete(panel, model):
    wanted = panel.groupby('ucid').covered.any()
    d = embed_dir(model)
    return all((d/f'asset_{a}.npz').exists() and (d/f'asset_{a}.json').exists() for a in wanted.index[wanted])


def worker_command(model, smoke):
    env = [f'TSFM_EMBED_ROOT={ROOT}', f'TSFM_EMBED_SCR={SCR}', f'TSFM_EMBED_TARGET={TARGET}',
           f'TSFM_EMBED_WALL={WALL}', 'E206_BATCH=4']
    args = ['env', *env, E.VENVS[model], '-u', '-m', 'src.data_mani.e212_signal_embeddings', '--model', model,
            '--device', 'cuda:0'] + (['--smoke'] if smoke else [])
    return shlex.join(args)


def collect():
    """Row-indexed H per model and the common causal mask shared by all 36 portfolios."""
    c = verify(); p = pd.read_parquet(ROOT/'panel.parquet'); common = p.covered.to_numpy().copy()
    for model in MODELS:
        d = embed_dir(model); manifest = json.loads((d/'extractor.json').read_text())
        assert manifest['target'] == TARGET and manifest['features'] == json.loads((ROOT/'PREPARED.json').read_text())['features']
        path = d/'H_rows.npy'
        if not path.exists():
            tmp = d/'H_rows.partial.npy'
            x = np.lib.format.open_memmap(tmp, mode='w+', dtype='float16', shape=(len(p), manifest['width'])); x[:] = np.nan
            for asset in d.glob('asset_*.npz'):
                info = json.loads(asset.with_suffix('.json').read_text())
                assert info['fingerprint'] == manifest['fingerprint']
                with np.load(asset) as z:
                    assert p.loc[z['rows'], 'ucid'].astype(str).eq(asset.stem.removeprefix('asset_')).all()
                    x[z['rows']] = z['states']
            x.flush(); del x; tmp.replace(path)
        valid = np.isfinite(np.load(path, mmap_mode='r')).all(axis=1)
        np.save(d/'valid_rows.npy', valid); common &= valid
    common = N.common_history(p.ucid.to_numpy(), p.date.to_numpy(), common, 32)
    common &= pd.Series(common).groupby(p.date).transform('sum').ge(2).to_numpy()
    tmp = SCR/'common_rows.partial.npy'; np.save(tmp, common); tmp.replace(SCR/'common_rows.npy')
    parquet(ROOT/'embedding_coverage.parquet', p[['row', 'ucid', 'symbol', 'date', 'covered']].assign(common_causal=common))
    save(ROOT/'EMBEDDINGS_READY.json', dict(models=MODELS, common_rows=int(common.sum()),
                                            mask_sha256=sha(SCR/'common_rows.npy'), time=time.time()))


# ---------------------------------------------------------------------- heads
def masks(p, valid=None):
    """Index sets per asset; identical rules for every learner and variant."""
    common = np.load(SCR/'common_rows.npy')
    ok = common if valid is None else common & valid
    y = np.isfinite(p[TARGET].to_numpy())
    out = {}
    for asset, g in p.groupby('ucid', sort=True):
        i = g.index.to_numpy()
        tr = i[ok[i] & p.train.to_numpy()[i] & y[i]]
        va = i[ok[i] & p.validation.to_numpy()[i] & y[i]]
        ft = i[ok[i] & p.final_fit.to_numpy()[i] & y[i]]
        te = i[ok[i] & p.test.to_numpy()[i]]
        if len(tr) >= 60 and len(va) >= 20 and len(te):
            out[asset] = (tr, va, ft, te)
    return out


def standardized(x, fit_rows, rows):
    mean = x[fit_rows].mean(0); scale = x[fit_rows].std(0); scale[scale < N.FLOOR] = 1
    return (x[rows]-mean)/scale


def ridge_portfolio(p, kernels, X, phases=('selection', 'final')):
    """Per-asset dual ridge on kernel K_C (+ standardized own X). kernels: phase -> (K, valid) or None."""
    preds = {}; records = {}
    for phase in phases:
        kern, valid = kernels[phase] if kernels else (None, None)
        sets = masks(p, valid)
        result = np.full(len(p), np.nan)
        for asset, (tr, va, ft, te) in sets.items():
            if phase == 'final' and str(asset) not in records:
                continue
            fi, qi = (tr, va) if phase == 'selection' else (ft, te)
            ktt = np.zeros((len(fi), len(fi))); kqt = np.zeros((len(qi), len(fi)))
            if kern is not None:
                fs, qs = p.step.to_numpy()[fi], p.step.to_numpy()[qi]
                ktt += kern[np.ix_(fs, fs)]; kqt += kern[np.ix_(qs, fs)]
            if X is not None:
                both = np.r_[fi, qi]; z = standardized(X, fi, both)
                zt, zq = z[:len(fi)], z[len(fi):]
                ktt += zt@zt.T; kqt += zq@zt.T
            alphas = N.ALPHAS if phase == 'selection' else [records[str(asset)]['alpha']]
            pred, _ = N.dual_ridge(ktt, kqt, p.loc[fi, TARGET].to_numpy(), alphas)
            if phase == 'selection':
                loss = ((pred-p.loc[qi, TARGET].to_numpy()[:, None])**2).mean(0); j = int(loss.argmin())
                records[str(asset)] = dict(alpha=float(alphas[j]), mse=float(loss[j]), n_train=len(fi))
            else:
                j = 0; records[str(asset)]['n_final'] = len(fi)
            result[qi] = pred[:, j]
        preds[phase] = result
    return preds, records


def kernel(model, phase, window):
    """Design-C kernel on the recommended standardization (E206 build_design)."""
    bind()
    p = pd.read_parquet(ROOT/'panel.parquet')
    kern, valid, info = E.build_design(model, NORMS[model], phase, window, 'C', p)
    return kern, valid, info


def write_predictions(tag, p, pred, extra):
    root = ROOT/'fits'/tag; root.mkdir(parents=True, exist_ok=True)
    out = p.loc[p.test, ['row', 'ucid', 'symbol', 'date', TARGET, 'meta_model']].copy()
    out['prediction'] = pred[p.test.to_numpy()]
    parquet(root/'predictions.parquet', out)
    save(root/'COMPLETE.json', dict(tag=tag, predictions=int(np.isfinite(out.prediction).sum()), time=time.time(), **extra))


def fit_ridge(model):
    """Window choice (rolling only), then H and HX ridge on the chosen window; model='X' fits the control."""
    verify(); p = pd.read_parquet(ROOT/'panel.parquet')
    features = [c for c in p if c.startswith('feature_')]; X = p[features].to_numpy(float)
    if model == 'X':
        if (ROOT/'fits'/'X_ridge'/'COMPLETE.json').exists():
            return
        preds, records = ridge_portfolio(p, None, X)
        write_predictions('X_ridge', p, preds['final'], dict(assets=len(records), records=records))
        return
    if all((ROOT/'fits'/f'{model}_{v}_ridge'/'COMPLETE.json').exists() for v in VARIANTS):
        return
    choices = []
    for window in (N.ROLLING if NORMS[model] == 'rolling' else [126]):
        kern, valid, info = kernel(model, 'selection', window)
        preds, records = ridge_portfolio(p, {'selection': (kern, valid)}, None, ('selection',))
        choices.append((float(np.mean([r['mse'] for r in records.values()])), window, info))
    score, window, info = min(choices, key=lambda c: (c[0], c[1]))
    save(ROOT/'windows'/f'{model}.json', dict(model=model, norm=NORMS[model], window=window, mean_asset_mse=score,
         candidates=[dict(window=w, mse=s) for s, w, _ in choices], vocabulary=info['vocabulary']))
    kernels = {phase: kernel(model, phase, window)[:2] for phase in ['selection', 'final']}
    for variant in VARIANTS:
        preds, records = ridge_portfolio(p, kernels, X if variant == 'HX' else None)
        write_predictions(f'{model}_{variant}_ridge', p, preds['final'], dict(window=window, assets=len(records), records=records))


def build_c(model):
    """Explicit design-C matrices [date, vocabulary slot x hidden] for the KTZ learners."""
    done = ROOT/'inputs'/model/'COMPLETE.json'
    if done.exists():
        return
    verify(); bind()
    choice = json.loads((ROOT/'windows'/f'{model}.json').read_text())
    p = pd.read_parquet(ROOT/'panel.parquet'); dims = {}
    for phase in ['selection', 'final']:
        h = E.normalized(model, NORMS[model], phase, choice['window'])
        valid = np.isfinite(h).all(axis=1)
        phi = K.materialize(h, p.ucid.to_numpy(), p.step.to_numpy(), valid, choice['vocabulary'],
                            int(p.step.max()+1), SCR/'inputs'/model/f'{phase}.npy')
        np.save(SCR/'inputs'/model/f'{phase}_valid.npy', valid); dims[phase] = list(phi.shape)
        del phi, h
    for f in (SCR/'normalizations'/model).glob('*/*/H.npy'):
        f.unlink()       # regenerable; scalers and manifests stay
    save(done, dict(model=model, window=choice['window'], dimensions=dims, time=time.time()))


ORIGINAL_WRAPPER = K.wrapper


def wrapper(family, cores=1):
    """E220's TabNet accommodation (sparse identities, batch capped by rows); others unchanged."""
    if family != 'tabnet_ktz':
        return ORIGINAL_WRAPPER(family, cores)
    from src.prediction.models import MODELS as REGISTRY
    from src.analysis.wide_tabnet import WideTabNetSkRegressor
    w = REGISTRY['tabnet_ktz'](task='regression', seed=0, n_jobs=cores)
    old = w.ModelClass.steps[-1][1]
    w.ModelClass.steps[-1] = ('net', WideTabNetSkRegressor(**old.get_params()))
    return w


def fit_ktz(model, variant, family):
    tag = f'X_{family}' if model == 'X' else f'{model}_{variant}_{family}'
    root = ROOT/'fits'/tag
    if (root/'COMPLETE.json').exists():
        return
    verify(); root.mkdir(parents=True, exist_ok=True); K.wrapper = wrapper
    p = pd.read_parquet(ROOT/'panel.parquet')
    features = [c for c in p if c.startswith('feature_')]
    grid = json.loads((ROOT/'candidates.json').read_text())[family]
    if model == 'X':
        xs = xf = None; vs = vf = None
    else:
        d = SCR/'inputs'/model
        xs, xf = np.load(d/'selection.npy', mmap_mode='r'), np.load(d/'final.npy', mmap_mode='r')
        vs, vf = np.load(d/'selection_valid.npy'), np.load(d/'final_valid.npy')

    def matrix(cache, idx):
        parts = [] if cache is None else [np.asarray(cache[p.step.to_numpy()[idx]], dtype=np.float32)]
        if variant in ('HX', 'X'):
            parts.append(p.loc[idx, features].to_numpy(np.float32))
        return np.column_stack(parts)
    sel, fin = masks(p, vs), masks(p, vf)
    result = np.full(len(p), np.nan); records = 0; failures = []
    for asset, (tr, va, _, _) in sel.items():
        if asset not in fin:
            continue
        _, _, ft, te = fin[asset]
        rec, pp = root/'assets'/f'{asset}.json', root/'assets'/f'{asset}.parquet'
        if rec.exists() and pp.exists():
            data = pd.read_parquet(pp)
        else:
            try:
                pred, record = K.fit_asset(family, grid, matrix(xs, tr), p.loc[tr, TARGET].to_numpy(),
                    matrix(xs, va), p.loc[va, TARGET].to_numpy(), matrix(xf, ft), p.loc[ft, TARGET].to_numpy(),
                    matrix(xf, te), int(os.getenv('NMR_SLOT_CORES', '4')), SCR/'models'/tag/f'{asset}.joblib')
                record.update(asset=str(asset))
                data = pd.DataFrame(dict(row=te, prediction=pred)); parquet(pp, data); save(rec, record)
            except Exception as error:
                save(root/'asset_failures'/f'{asset}.json', dict(error=str(error), traceback=traceback.format_exc()))
                failures.append(str(asset)); continue
        result[data.row.to_numpy()] = data.prediction.to_numpy(); records += 1
        save(root/'progress.json', dict(completed_assets=records, total=len(sel), failed=len(failures), time=time.time()))
        print('ASSET', tag, asset, flush=True)
    if failures or not records:
        raise RuntimeError(f'{tag}: failed assets {failures}; completed assets are checkpointed')
    write_predictions(tag, p, result, dict(assets=records, family=family, variant=variant))


def fit(model, variant, family):
    if family == 'ridge':
        fit_ridge(model)
    else:
        fit_ktz(model, variant, family)


# -------------------------------------------------------------------- scoring
def score():
    verify(); p = pd.read_parquet(ROOT/'panel.parquet'); tags = portfolios()
    test = p.loc[p.test, ['row', 'ucid', 'symbol', 'date', TARGET, 'meta_model']].sort_values(['date', 'ucid']).reset_index(drop=True)
    raw = np.column_stack([pd.read_parquet(ROOT/'fits'/t/'predictions.parquet').set_index('row').prediction
                           .reindex(test.row).to_numpy(float) for t in tags])
    # Full-universe baseline: X-only ridge, else the asset's final-fit target mean, else 0.5.
    mean = p.loc[p.final_fit].groupby('ucid')[TARGET].mean()
    base = np.where(np.isfinite(raw[:, tags.index('X_ridge')]), raw[:, tags.index('X_ridge')],
                    test.ucid.map(mean).fillna(.5).to_numpy())
    test['baseline_rank'] = pd.Series(base).groupby(test.date).transform(lambda s: rank_normalise(s.to_numpy())).to_numpy()
    valid = np.isfinite(raw).all(1) & np.isfinite(test[TARGET]).to_numpy() & np.isfinite(test.meta_model).to_numpy()
    routed = np.empty_like(raw); dates = []; scores = []; counts = []
    for date, idx in test.groupby('date', sort=True).indices.items():
        for j in range(len(tags)):
            routed[idx, j] = route_slots(test.baseline_rank.to_numpy()[idx], raw[idx, j])
        rows = idx[valid[idx]]
        if len(rows) < 10:
            continue
        c, m = matrix_scores(raw[rows], test.loc[rows, TARGET].to_numpy(), test.loc[rows, 'meta_model'].to_numpy())
        dates.append(date); scores.append(np.column_stack([c, m])); counts.append(len(rows))
    assert np.isfinite(routed).all()
    scores = np.stack(scores); dates = np.asarray(dates)
    np.savez_compressed(ROOT/'forecasts.npz', row=test.row.to_numpy(), tags=np.array(tags), raw=raw, routed=routed, common=valid)
    np.savez_compressed(ROOT/'daily_scores.npz', date=dates, tags=np.array(tags), scores=scores)
    m = metrics(scores)
    table = pd.DataFrame(dict(tag=tags, n_dates=len(dates), mean_assets=np.mean(counts), **m))
    table['tsfm'] = [t.split('_')[0] if not t.startswith('X_') else '-' for t in tags]
    table['inputs'] = ['X' if t.startswith('X_') else t.split('_')[1] for t in tags]
    table['learner'] = [t.split('_', 2)[-1] if not t.startswith('X_') else t[2:] for t in tags]
    table['norm'] = table.tsfm.map(NORMS).fillna('-')
    table = table.sort_values('payout', ascending=False)
    table.to_csv(ROOT/'table.csv', index=False)
    pay = .1*scores[:, :, 0]+scores[:, :, 1]
    deltas = []
    for model in MODELS:
        for family in FAMILIES:
            for a, b, label in [(f'{model}_HX_{family}', f'X_{family}', 'HX-X (H over reversal)'),
                                (f'{model}_HX_{family}', f'{model}_H_{family}', 'HX-H (reversal over H)'),
                                (f'{model}_H_{family}', f'X_{family}', 'H-X')]:
                ia, ib = tags.index(a), tags.index(b)
                row = dict(tsfm=model, learner=family, comparison=label)
                for k, series in [('corr', scores[:, ia, 0]-scores[:, ib, 0]), ('mmc', scores[:, ia, 1]-scores[:, ib, 1]),
                                  ('payout', pay[:, ia]-pay[:, ib])]:
                    row[f'd_{k}'], row[f't_{k}'] = newey_west(series, 20)
                deltas.append(row)
    deltas = pd.DataFrame(deltas); deltas.to_csv(ROOT/'paired_deltas.csv', index=False)
    report(table, deltas, dates, pay, tags, len(dates), int(valid.sum()))
    save(ROOT/'COMPLETE.json', dict(experiment='E221', portfolios=len(tags), dates=len(dates), common_keys=int(valid.sum()),
                                    pdf=str(ROOT/'E221_reversal187_tsfm.pdf'), time=time.time()))


def report(table, deltas, dates, pay, tags, n_dates, n_keys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.analysis.html_pdf import to_pdf
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for ax, family in zip(axes.flat, FAMILIES):
        for j, t in enumerate(tags):
            if t.endswith(family):
                style = '-' if '_HX_' in t else ('--' if '_H_' in t else ':')
                ax.plot(dates, pay[:, j].cumsum(), style, lw=2.5 if t.startswith('X_') else 1.2, label=t.replace('_'+family, ''))
        ax.set_title(family); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('E221 cumulative payout (solid HX, dashed H, dotted X-only)'); fig.tight_layout()
    fig.savefig(ROOT/'cumulative_payout.png', dpi=110); plt.close(fig)
    corr = np.corrcoef(pay.T)
    fig, ax = plt.subplots(figsize=(13, 11)); im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(tags)), tags, rotation=90, fontsize=6); ax.set_yticks(range(len(tags)), tags, fontsize=6)
    fig.colorbar(im); ax.set_title('Daily payout correlation'); fig.tight_layout()
    fig.savefig(ROOT/'payout_correlation.png', dpi=110); plt.close(fig)
    fmt = lambda v: f'{v:.5f}' if isinstance(v, float) else v
    cols = ['tag', 'tsfm', 'norm', 'inputs', 'learner', 'corr', 'mmc', 'payout', 'sharpe.p', 'maxdd.p']
    body = (f'<h1>E221: TSFM states on reversal187, per-asset ridge / LightGBM / MLP / TabNet</h1>'
            f'<p>Chronos-2 and t0 (rolling), TabPFN-TS (asset) and Timestral (robust) pre-head states extracted from '
            f'crypto/v2.0-e219-ts-reversal187 on the target20 context with a 35-day label wall, design C. '
            f'H = design-C states only; HX = H plus the asset\'s own 187 reversal columns; X = reversal187 alone '
            f'(matched control). Every learner is fitted from scratch per asset (validation-MSE selection, fresh final fit). '
            f'Common sample: {n_dates} test dates, {n_keys:,} asset/date rows, identical for all 36 portfolios. '
            f'Payout = 0.1 CORR + historical-meta proxy MMC.</p>')
    body += '<h2>All portfolios, by payout</h2>' + table[cols].to_html(index=False, float_format=fmt)
    body += ('<h2>Paired differences (Newey-West lag 20)</h2><p>HX-X is the marginal value of the TSFM states over '
             'reversal187; HX-H the marginal value of reversal187 over the states. Unadjusted for the 48 comparisons.</p>')
    body += deltas.to_html(index=False, float_format=fmt)
    body += '<h2>Cumulative payout</h2><img style="width:100%" src="cumulative_payout.png">'
    body += '<h2>Daily payout correlation</h2><img style="width:90%" src="payout_correlation.png">'
    body += ('<h2>Limitations</h2><ul><li>Development history; the dataset and reversal formulas were chosen on it.</li>'
             '<li>Vendor snapshot cannot prove historical availability; pretrained checkpoints may overlap the test period.</li>'
             '<li>No live path; no investment-return claim.</li></ul>')
    page = ROOT/'E221_reversal187_tsfm.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:8px;border-collapse:'
                    'collapse}td,th{border:1px solid #ccc;padding:3px}</style></head><body>'+body+'</body></html>')
    to_pdf(str(page), 'lovelace')


def email_report():
    receipt = ROOT/'email_delivery.json'
    if receipt.exists():
        return
    from src.live.env import load_env
    from src.live.notify import _config
    load_env(required=()); config = _config(); assert config
    pdf = ROOT/'E221_reversal187_tsfm.pdf'
    msg = EmailMessage(); msg['From'] = config['user']; msg['To'] = config['to']
    msg['Subject'] = '[numerai-crypto] E221 reversal187 TSFM states, ridge/LightGBM/MLP/TabNet — completed PDF'
    msg.set_content('E221 is complete. Attached: 36 per-asset portfolios (4 TSFMs x H/HX x 4 learners + 4 reversal-only controls) '
                    'on one common sample, with paired Newey-West deltas. Development research only.')
    msg.add_attachment(pdf.read_bytes(), maintype='application', subtype='pdf', filename=pdf.name)
    with smtplib.SMTP(config['host'], config['port'], timeout=120) as server:
        server.starttls(); server.login(config['user'], config['password'])
        assert not server.send_message(msg)
    save(receipt, dict(smtp_accepted=True, recipient=config['to'], pdf_sha256=sha(pdf), time=time.time()))


# ----------------------------------------------------------------- supervisor
def jobs_now(p, config):
    jobs = []; stage = 'gpu_smokes'
    if not (ROOT/'EMBEDDINGS_READY.json').exists():
        smokes = [m for m in MODELS if not (embed_dir(m)/'SMOKE_COMPLETE.json').exists()]
        if smokes:
            return [budget.Job(f'e221_gpu_smoke_{m}', worker_command(m, True), cores=2, ram_gb=16, gpus=1) for m in smokes], stage
        stage = 'gpu_extracting'
        pending = [m for m in MODELS if not extraction_complete(p, m)]
        if pending:
            return [budget.Job(f'e221_gpu_extract_{m}', worker_command(m, False), cores=2, ram_gb=24, gpus=1) for m in pending], stage
        collect()
    stage = 'fitting'
    run = lambda *a: shlex.join([PY, '-u', 'experiments/e221_reversal187_tsfm.py', *a])
    for model in MODELS:
        if not all((ROOT/'fits'/f'{model}_{v}_ridge'/'COMPLETE.json').exists() for v in VARIANTS):
            jobs.append(budget.Job(f'e221_ridge_{model}', run('fit', '--model', model, '--family', 'ridge'), cores=4, ram_gb=40))
            continue
        if not (ROOT/'inputs'/model/'COMPLETE.json').exists():
            jobs.append(budget.Job(f'e221_build_{model}', run('build', '--model', model), cores=2, ram_gb=40))
            continue
        for variant in VARIANTS:
            for family in FAMILIES[1:]:
                if not (ROOT/'fits'/f'{model}_{variant}_{family}'/'COMPLETE.json').exists():
                    jobs.append(budget.Job(f'e221_{model}_{variant}_{family}',
                                           run('fit', '--model', model, '--variant', variant, '--family', family), cores=4, ram_gb=32))
    for family in FAMILIES:
        if not (ROOT/'fits'/f'X_{family}'/'COMPLETE.json').exists():
            jobs.append(budget.Job(f'e221_X_{family}', run('fit', '--model', 'X', '--variant', 'X', '--family', family), cores=4, ram_gb=12))
    if not jobs and not (ROOT/'COMPLETE.json').exists():
        stage = 'scoring'
        jobs = [budget.Job('e221_score', run('score'), cores=4, ram_gb=24)]
    return jobs, stage


def admission(probes, owned, active, snapshots):
    """E212 admission, except that our own card on a DOUBLED_HOSTS box no longer blocks its second card."""
    from dataclasses import replace
    occupied = {r['host'] for u, r in owned.items() if u in active and r.get('gpus')}
    result = []
    for host in probes:
        blocked, _, seen = snapshots.get(host.name, (set(), [], set()))
        busy = set(host.busy_gpus) | blocked | (set(range(host.gpus))-seen)
        if host.name in occupied:
            if host.name in DOUBLED_HOSTS:
                busy |= {g for u, r in owned.items() if u in active and r['host'] == host.name for g in r.get('gpus', [])}
            else:
                busy = set(range(host.gpus))
        result.append(replace(host, busy_gpus=tuple(sorted(busy))))
    return result


def recover_locks():
    """Remove a per-asset extraction lock only when its owner process is confirmed gone."""
    for path in SCR.glob('embeddings/*/*.lock'):
        try:
            owner = json.loads(path.read_text()); pid = int(owner['pid'])
        except (OSError, ValueError, KeyError):
            continue
        try:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4', owner['host'],
                                     f'ps -p {pid} -o args='], capture_output=True, text=True, timeout=8)
            if result.returncode in [0, 1] and 'src.data_mani.e212_signal_embeddings' not in result.stdout:
                path.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            continue


def place(pending, admitted):
    """Doubled share on DOUBLED_HOSTS first; whatever does not fit there goes to the rest at the default."""
    placements, unplaced = [], list(pending)
    for name, share in DOUBLED_HOSTS.items():
        pool = [h for h in admitted if h.name == name]
        if pool and unplaced:
            got, left = budget.distribute([j for j in unplaced], pool, share=share, gpu_slots=DOUBLED_GPU_SLOTS)
            placements += got; unplaced = [j for j, _ in left]
    rest = [h for h in admitted if h.name not in DOUBLED_HOSTS]
    got, left = budget.distribute(unplaced, rest, share=budget.SHARE, gpu_slots=1)
    return placements+got, left


def supervise():
    config = prepare(); config = verify(); p = pd.read_parquet(ROOT/'panel.parquet')
    path = ROOT/'supervisor.json'
    state = json.loads(path.read_text()) if path.exists() else dict(owned={}, attempts={}, launches=[])
    def interrupt(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while not (ROOT/'COMPLETE.json').exists():
            probes = [h for h in hosts.reachable(hosts.probe_all(timeout=4)) if h.name not in {'westeros', 'imeusp'}]
            active, _ = E.unit_states(manager.status(names=[h.name for h in probes], timeout=4))
            for unit in list(state['owned']):
                if unit not in active:
                    confirmed = G.confirm_unit(state['owned'][unit]['host'], unit)
                    if confirmed is False:
                        del state['owned'][unit]
                    else:
                        active.add(unit)
            recover_locks()
            gpu_hosts = [h for h in probes if h.gpus]
            with ThreadPoolExecutor(max_workers=max(1, len(gpu_hosts))) as pool:
                snapshots = {n: (b, r, s) for n, b, r, s in pool.map(G.gpu_snapshot, [h.name for h in gpu_hosts])}
            jobs, stage = jobs_now(p, config)
            pending = []
            for job in jobs:
                unit = 'nmr-'+job.name
                if unit in active or unit in state['owned']:
                    continue
                if state['attempts'].get(job.name, 0) >= 3:
                    raise RuntimeError('Three attempts exhausted: '+job.name+f'; see logs/cluster/{job.name}.out')
                pending.append(job)
            admitted = admission(probes, state['owned'], active, snapshots)
            placements, unplaced = place(pending, admitted)
            for placement in placements:
                subprocess.run(['ssh', '-o', 'BatchMode=yes', placement.host,
                                'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user reset-failed '+placement.unit],
                               capture_output=True, timeout=15)
                ok, message = manager.launch(placement)
                if ok:
                    record = dict(host=placement.host, unit=placement.unit, cpus=list(placement.cpus),
                                  gpus=list(placement.gpus), time=time.time())
                    state['owned'][placement.unit] = record; state['launches'].append(record)
                    state['attempts'][placement.job.name] = state['attempts'].get(placement.job.name, 0)+1
                print('LAUNCH', ok, message, flush=True)
            done = [t for t in portfolios() if (ROOT/'fits'/t/'COMPLETE.json').exists()]
            state.update(stage=stage, updated=time.time(), waiting=[dict(job=j.name, reason=r) for j, r in unplaced])
            save(path, state)
            save(ROOT/'status.json', dict(stage=stage, completed_portfolios=len(done), expected=36,
                 active=sorted(state['owned']), waiting=state['waiting'], time=time.time()))
            time.sleep(30)
        email_report()
    except Exception as exc:
        save(ROOT/'FAILED.json', dict(error=str(exc), traceback=traceback.format_exc(), time=time.time())); raise
    finally:
        for host in {r['host'] for r in state['owned'].values()}:
            manager.stop(host, [u for u, r in state['owned'].items() if r['host'] == host and u.startswith('nmr-e221_')], timeout=20)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'supervise', 'fit', 'build', 'score', 'email'])
    ap.add_argument('--model', choices=MODELS+['X']); ap.add_argument('--variant', choices=VARIANTS+['X'])
    ap.add_argument('--family', choices=FAMILIES)
    a = ap.parse_args(); ROOT.mkdir(parents=True, exist_ok=True)
    if a.command == 'fit':
        fit(a.model, a.variant, a.family)
    elif a.command == 'build':
        build_c(a.model)
    elif a.command == 'score':
        score()
    elif a.command == 'email':
        email_report()
    elif a.command == 'prepare':
        prepare()
    else:
        with open(ROOT/'supervisor.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            supervise()
