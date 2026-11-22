"""E205: independent per-asset momentum tuning, portfolio aggregation/backtest.

No mail, deployment, or held-out return reads. Run is resumable and writes
durable preparation, per-fold, per-task and report checkpoints.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
import hashlib
import html
import json
import socket
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from src.analysis import asset_momentum as M
from src.analysis.cross_target import newey_west
from src.live.submit import rank_normalise
from e150_finish_scoring import matrix_scores

ROOT = REPO/'results/reports/crypto/e205_asset_momentum'
SOURCE = REPO/'data/crypto/v2.0-e156-frozen'
PRICES = REPO/'results/reports/crypto/e157_volatility_reverse_engineering/inputs/prices.parquet'
CANDIDATES = REPO/'experiments/e205_momentum_candidates.json'
TARGET = 'target_binned_return_20'
CUTOFF = pd.Timestamp('2026-02-20')
SEEDS = list(range(5))
CODE = [Path(__file__), REPO/'src/analysis/asset_momentum.py', CANDIDATES,
        REPO/'src/prediction/torch_mlp.py',
        REPO/'experiments/e150_finish_scoring.py', REPO/'src/live/submit.py']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    tmp.replace(path)


def parquet(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def folds(dates):
    starts = list(range(172, len(dates)-129, 130))
    if len(starts) != 11 or starts[-1]+130 != len(dates):
        raise ValueError('Development date grid differs from E163')
    return [(i, dates[s-152:s-22], dates[s:s+130]) for i, s in enumerate(starts)]


def verify():
    plan = json.loads((ROOT/'registration.json').read_text())
    for name, expected in plan['code_hashes'].items():
        if sha(REPO/name) != expected:
            raise RuntimeError('Registered code changed: '+name)
    prepared = ROOT/'PREPARED.json'
    if prepared.exists():
        expected = json.loads(prepared.read_text())['keys_sha256']
        if sha(ROOT/'keys.parquet') != expected:
            raise RuntimeError('Registered asset/date/target keys changed')
    return plan


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'PREPARED.json').exists():
        return verify()
    inputs = [SOURCE/'train.parquet', SOURCE/'historical_meta_models.parquet', PRICES]
    plan = dict(experiment='E205', registered='2026-11-22', revision=2,
                families=M.FAMILIES, windows=M.WINDOWS, effective_window='nominal - 1',
                processing='TS bins only; no cross-sectional signal inputs, no asset dummies',
                models=['ridge', 'mlp_torch'], seeds=SEEDS, n_portfolios=24,
                candidates=json.loads(CANDIDATES.read_text()),
                min_inner=40, min_validation=20, min_outer=80,
                folds='outer130/purge22/test130; inner68/purge22/validation40',
                objective='independent per-asset temporal MSE; no cross-asset tuning',
                inputs='One dataset per formula; its11 TS-binned lookback variants are joint model inputs',
                mlp='identity; exact sklearn NumPy RandomState weights and biases; '
                    'half-MSE + alpha*sum(weight**2)/(2*n_asset); Adam; max200, tol1e-4, patience10; '
                    'separate scaler/parameters/moments/stopping for every asset x candidate; '
                    'full-batch because each asset has at most130 rows',
                aggregation='mean expected-target predictions across five seeds, then cross-asset rank; '
                            'fixed equal-rank ensembles across12 formula portfolios per model and across all24',
                coverage='common finite TS across all12x11 inputs and E163 identity/OHLC cohort; '
                         'all original keys retained and candidate NaNs stored; fit only covered rows; '
                         'source cutoff D-1, max4 calendar days stale; no coverage predictors',
                control='per-asset historical target mean on identical training and evaluation keys',
                fallback='per-asset historical mean on all available preceding resolved labels; '
                         'unseen assets use fixed0.5; same full-universe rank slots for candidate/control',
                claim='standalone momentum-over-asset-level comparison, not marginal native22 attribution',
                limitation='Downloaded vendor price histories and unique-identity filters do not prove '
                           'historical source availability. Development-only; no live deployment claim.',
                return_holdout='dates >=2026-02-20 excluded at parquet read',
                code_hashes={str(p.relative_to(REPO)): sha(p) for p in CODE},
                input_hashes={str(p.relative_to(REPO)): sha(p) for p in inputs})
    save(ROOT/'registration.json', plan)
    base = pd.read_parquet(SOURCE/'train.parquet', columns=['date', 'ucid', TARGET],
                           filters=[('date', '<', CUTOFF)]).reset_index()
    base = base.sort_values(['date', 'symbol']).reset_index(drop=True)
    assert base.date.max() < CUTOFF and not base.duplicated(['ucid', 'date']).any()
    meta = pd.read_parquet(SOURCE/'historical_meta_models.parquet',
                           filters=[('date', '<', CUTOFF)]).reset_index()
    base = base.merge(meta[['symbol', 'date', 'prediction']].rename(columns={'prediction': 'meta_model'}),
                      how='left', on=['symbol', 'date'], validate='1:1')
    base['row'] = np.arange(len(base))
    prices = pd.read_parquet(PRICES, filters=[('date', '<', CUTOFF)])
    close, columns, cohort = M.price_context(base, prices)
    common = cohort.copy()
    # Construction uses each asset's own price/time history only.
    for family in M.FAMILIES:
        path = ROOT/'inputs'/f'{family}_ts.parquet'
        if path.exists():
            f = pd.read_parquet(path)
        else:
            values, audits = {}, []
            for w in M.WINDOWS:
                raw = M.signal(close, family, w-1)
                ts = pd.DataFrame(M.prior_bins(raw.to_numpy()), index=raw.index, columns=raw.columns)
                values[f'feature_ts_{w}'] = M.attach(base, ts, columns)
                if w in [20, 240]:
                    for cutoff in ['2022-12-30', '2024-12-31', '2025-12-31']:
                        cut = pd.Timestamp(cutoff)
                        short_raw = M.signal(close.loc[:cut], family, w-1)
                        short_ts = M.prior_bins(short_raw.to_numpy())
                        np.testing.assert_allclose(raw.loc[:cut].to_numpy(), short_raw.to_numpy(),
                                                   rtol=1e-9, atol=1e-12, equal_nan=True)
                        np.testing.assert_array_equal(ts.loc[:cut].to_numpy(), short_ts)
                        audits.append(dict(family=family, window=w, cutoff=cutoff, raw_changed=0, ts_changed=0))
            f = pd.DataFrame(values, dtype=np.float32)
            parquet(path, f)
            save(ROOT/'audits'/f'{family}_prefix.json', audits)
        assert len(f) == len(base)
        common &= np.isfinite(f.to_numpy()).all(axis=1)
        print('PREPARE TS', family, 'common rows so far', int(common.sum()), flush=True)
    base['covered'] = common
    dataset_hashes = {}
    for family in M.FAMILIES:
        f = pd.read_parquet(ROOT/'inputs'/f'{family}_ts.parquet')
        f.loc[~common, :] = np.nan
        assert f.loc[~common].isna().all().all()
        dataset = pd.concat([base[['row', 'symbol', 'ucid', 'date', TARGET, 'covered']], f], axis=1)
        path = ROOT/'datasets'/f'{family}.parquet'
        parquet(path, dataset)
        dataset_hashes[family] = sha(path)
    parquet(ROOT/'keys.parquet', base)
    dates = np.sort(base.date.unique())
    save(ROOT/'date_grid.json', list(map(str, dates)))
    pd.DataFrame([dict(date=d, rows=len(g), covered=int(g.covered.sum()),
                       fraction=float(g.covered.mean())) for d, g in base.groupby('date')]).to_csv(
                           ROOT/'coverage_by_date.csv', index=False)
    save(ROOT/'PREPARED.json', dict(rows=len(base), covered=int(common.sum()),
         assets=int(base.ucid.nunique()), eligible_assets=int(base.loc[common].ucid.nunique()),
         datasets=12, features_per_dataset=11, folds=len(folds(dates)), dataset_hashes=dataset_hashes,
         keys_sha256=sha(ROOT/'keys.parquet'), finished=time.time()))
    print('PREPARED', json.loads((ROOT/'PREPARED.json').read_text()), flush=True)
    return plan


def baseline():
    """Full-universe per-asset intercept fallback; never a pooled panel model."""
    if (ROOT/'baseline.parquet').exists():
        return
    verify()
    p = pd.read_parquet(ROOT/'keys.parquet')
    dates = pd.to_datetime(json.loads((ROOT/'date_grid.json').read_text())).to_numpy()
    output, picks = [], []
    for block, tr, te in folds(dates):
        train = p[p.date.isin(tr) & np.isfinite(p[TARGET])]
        test = p[p.date.isin(te)]
        means = train.groupby(train.ucid.astype(str))[TARGET].mean()
        values = test.ucid.astype(str).map(means)
        piece = test[['row', 'date']].copy()
        piece['baseline'] = values.fillna(.5).to_numpy()
        piece['baseline_rank'] = piece.groupby('date').baseline.transform(lambda s: rank_normalise(s.to_numpy()))
        piece['block'] = block
        output.append(piece)
        picks.append(dict(block=block, train=len(train), query=len(test), unseen_rows=int(values.isna().sum()),
                          trained_assets=len(means), latest_train_date=str(train.date.max())))
    parquet(ROOT/'baseline.parquet', pd.concat(output, ignore_index=True))
    save(ROOT/'baseline_picks.json', picks)


def dense(frame, values, dates, assets):
    rows = pd.Index(assets).get_indexer(frame.ucid.astype(str))
    cols = pd.Index(dates).get_indexer(frame.date)
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError('Unknown training keys')
    v = np.asarray(values, float)
    out = np.full((len(assets), len(dates)) + v.shape[1:], np.nan)
    out[rows, cols] = v
    return out


def fit_fold(model, family, block, tr, te, frame, params):
    directory = ROOT/'tasks'/f'{model}__{family}'
    final = directory/f'block{block}.parquet'
    pickpath = directory/f'block{block}_picks.parquet'
    if final.exists() and pickpath.exists():
        return final
    features = [f'feature_ts_{w}' for w in M.WINDOWS]
    covered = frame[frame.covered & np.isfinite(frame[features]).all(axis=1)]
    train = covered[covered.date.isin(tr) & np.isfinite(covered[TARGET])]
    counts = train.groupby(train.ucid.astype(str)).agg(
        n=('date', 'size'), ni=('date', lambda s: int(s.isin(tr[:68]).sum())),
        nv=('date', lambda s: int(s.isin(tr[-40:]).sum())))
    assets = counts.index[(counts.n >= 80) & (counts.ni >= 40) & (counts.nv >= 20)].to_numpy()
    test = covered[covered.date.isin(te) & covered.ucid.astype(str).isin(assets)].copy()
    if not len(assets) or not len(test):
        parquet(final, pd.DataFrame(columns=['row', 'prediction', 'asset_mean', 'seed', 'block']))
        parquet(pickpath, pd.DataFrame(columns=['ucid', 'candidate', 'inner_mse']))
        return final
    train = train[train.ucid.astype(str).isin(assets)]
    inner = train[train.date.isin(tr[:68])]
    val = train[train.date.isin(tr[-40:])]
    xi = dense(inner, inner[features], tr[:68], assets)
    yi = dense(inner, inner[TARGET], tr[:68], assets)
    xv = dense(val, val[features], tr[-40:], assets)
    yv = dense(val, val[TARGET], tr[-40:], assets)
    if model == 'ridge':
        validation = M.ridge_candidates(xi, yi, xv, params)
        epochs = np.zeros((len(assets), len(params)), int)
    else:
        validation, epochs = M.torch_candidates(xi, yi, xv, params, seed=0)
    mask = np.isfinite(yv) & np.isfinite(xv).all(axis=2)
    mse = np.where(mask[:, :, None], (validation-yv[:, :, None])**2, 0).sum(axis=1)/mask.sum(axis=1)[:, None]
    if not np.isfinite(mse).all():
        raise FloatingPointError('Nonfinite inner per-asset MSE')
    choice = mse.argmin(axis=1)
    xt = dense(train, train[features], tr, assets)
    yt = dense(train, train[TARGET], tr, assets)
    xq = dense(test, test[features], te, assets)
    asset_mean = np.nanmean(yt, axis=1)
    asset_index = pd.Index(assets).get_indexer(test.ucid.astype(str))
    date_index = pd.Index(te).get_indexer(test.date)
    predictions, diagnostics = [], []
    for seed in ([0] if model == 'ridge' else SEEDS):
        out = np.full(xq.shape[:2], np.nan)
        fit_epochs = np.zeros(len(assets), int)
        if model == 'ridge':
            allpred = M.ridge_candidates(xt, yt, xq, params)
            out = allpred[np.arange(len(assets)), :, choice]
        else:
            for i in np.unique(choice):
                selected = choice == i
                pred, e = M.torch_candidates(xt[selected], yt[selected], xq[selected], [params[i]], seed=seed)
                out[selected] = pred[:, :, 0]
                fit_epochs[selected] = e[:, 0]
        piece = test[['row']].copy()
        piece['prediction'] = out[asset_index, date_index]
        piece['asset_mean'] = asset_mean[asset_index]
        piece['seed'], piece['block'] = seed, block
        assert np.isfinite(piece[['prediction', 'asset_mean']]).all().all()
        predictions.append(piece)
        diagnostics.extend([dict(ucid=str(a), seed=seed, candidate=int(choice[j]),
              inner_mse=float(mse[j, choice[j]]), search_epochs=int(epochs[j, choice[j]]),
              refit_epochs=int(fit_epochs[j]), n_train=int(counts.loc[a, 'n']),
              n_inner=int(counts.loc[a, 'ni']), n_val=int(counts.loc[a, 'nv'])) for j, a in enumerate(assets)])
    parquet(final, pd.concat(predictions, ignore_index=True))
    parquet(pickpath, pd.DataFrame(diagnostics))
    print('FOLD', model, family, block, 'assets', len(assets), 'rows', len(test), flush=True)
    return final


def run_task(task):
    model, family = task
    plan = verify()
    key = f'{model}__{family}'
    done = ROOT/'tasks'/key/'COMPLETE.json'
    if done.exists():
        return key
    prepared = json.loads((ROOT/'PREPARED.json').read_text())
    path = ROOT/'datasets'/f'{family}.parquet'
    if sha(path) != prepared['dataset_hashes'][family]:
        raise RuntimeError('Dataset artifact changed')
    frame = pd.read_parquet(path)
    dates = pd.to_datetime(json.loads((ROOT/'date_grid.json').read_text())).to_numpy()
    paths = [fit_fold(model, family, b, tr, te, frame, plan['candidates'][model])
             for b, tr, te in folds(dates)]
    sizes = [len(pd.read_parquet(p, columns=['row'])) for p in paths]
    save(done, dict(task=key, host=socket.gethostname(), finished=time.time(), rows=sum(sizes), folds=len(paths)))
    return key


def score_one(task, keys, fallback):
    model, family = task
    taskid = f'{model}__{family}'
    frames = [pd.read_parquet(p) for p in sorted((ROOT/'tasks'/taskid).glob('block[0-9]*.parquet'))
              if '_picks' not in p.name]
    raw = pd.concat(frames, ignore_index=True)
    assert not raw.duplicated(['row', 'seed']).any()
    # Five-seed expected-target ensemble before per-date ranking is primary.
    mean = raw.groupby(['row', 'block'], as_index=False)[['prediction', 'asset_mean']].mean()
    rows, portfolios = [], []
    for label, predictions in [('ensemble', mean)] + [(str(s), raw[raw.seed == s]) for s in sorted(raw.seed.unique())]:
        full = fallback.merge(keys[['row', 'symbol', 'ucid', TARGET, 'meta_model']], on='row', validate='1:1')
        full = full.merge(predictions[['row', 'prediction', 'asset_mean']], on='row', how='left', validate='1:1')
        for date, g in full.groupby('date', sort=True):
            g = g.sort_values('symbol')
            prediction, control = g.prediction.to_numpy(), g.asset_mean.to_numpy()
            candidate_mask = np.isfinite(prediction)
            assert np.array_equal(candidate_mask, np.isfinite(control))
            routed = M.route_slots(g.baseline_rank.to_numpy(), prediction)
            control_routed = M.route_slots(g.baseline_rank.to_numpy(), control)
            if label == 'ensemble':
                piece = g[['row', 'date', 'block']].copy()
                piece['routed'], piece['control_routed'] = routed, control_routed
                piece['covered'] = candidate_mask
                piece['covered_rank'] = np.nan
                if candidate_mask.any():
                    piece.loc[candidate_mask, 'covered_rank'] = rank_normalise(prediction[candidate_mask])
                portfolios.append(piece)
            for sample in ['covered', 'routed']:
                mask = candidate_mask.copy() if sample == 'covered' else np.ones(len(g), bool)
                mask &= np.isfinite(g[TARGET].to_numpy()) & np.isfinite(g.meta_model.to_numpy())
                if mask.sum() < 10:
                    continue
                values = np.c_[prediction[mask], control[mask]] if sample == 'covered' else np.c_[routed[mask], control_routed[mask], g.baseline_rank.to_numpy()[mask]]
                corr, mmc = matrix_scores(values, g[TARGET].to_numpy()[mask], g.meta_model.to_numpy()[mask])
                for j, name in enumerate(['signal', 'asset_mean'] + ([] if sample == 'covered' else ['baseline'])):
                    # Numerai convention for a constant forecast: zero CORR.
                    c = 0. if not np.isfinite(corr[j]) else float(corr[j])
                    rows.append(dict(task=taskid, model=model, family=family, rep='ts', window='all11',
                         seed=label, date=date, block=int(g.block.iloc[0]), sample=sample, arm=name,
                         corr=c, mmc=float(mmc[j]), payout=.1*c+float(mmc[j]),
                         n_coins=int(mask.sum()), n_covered=int(candidate_mask.sum()), n_universe=len(g)))
    parquet(ROOT/'portfolios'/f'{taskid}.parquet', pd.concat(portfolios, ignore_index=True))
    parquet(ROOT/'daily'/f'{taskid}.parquet', pd.DataFrame(rows))
    return rows


def bootstrap_mean(values, reps=2000, block=20):
    x = np.asarray(values, float)
    rng = np.random.default_rng(205)
    starts = rng.integers(0, len(x), (reps, int(np.ceil(len(x)/block))))
    indices = (starts[:, :, None]+np.arange(block)) % len(x)
    means = x[indices.reshape(reps, -1)[:, :len(x)]].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def finish(tasks):
    verify()
    keys = pd.read_parquet(ROOT/'keys.parquet')
    fallback = pd.read_parquet(ROOT/'baseline.parquet')
    rows = []
    for i, task in enumerate(tasks):
        path = ROOT/'daily'/f'{task[0]}__{task[1]}.parquet'
        if path.exists():
            rows.extend(pd.read_parquet(path).to_dict('records'))
        else:
            rows.extend(score_one(task, keys, fallback))
        print('SCORE', i+1, '/', len(tasks), flush=True)
    daily = pd.DataFrame(rows)
    parquet(ROOT/'scores_by_date_seed.parquet', daily)
    # Rank ensembles never use outer scores for membership or weights.
    ensemble_rows = []
    for model in ['ridge', 'mlp_torch', 'ALL24']:
        selected = [t for t in tasks if model == 'ALL24' or t[0] == model]
        paths = [ROOT/'portfolios'/f'{m}__{f}.parquet' for m, f in selected]
        totals = None
        coverage = None
        for path in paths:
            p = pd.read_parquet(path, columns=['row', 'routed', 'control_routed', 'covered']).set_index('row')
            if coverage is None:
                coverage = p.covered
            else:
                pd.testing.assert_series_equal(coverage, p.covered)
            p = p[['routed', 'control_routed']]
            totals = p if totals is None else totals.add(p)
        totals /= len(paths)
        totals['covered'] = coverage
        full = fallback.merge(keys[['row', 'symbol', TARGET, 'meta_model']], on='row', validate='1:1').merge(
            totals.reset_index(), on='row', validate='1:1')
        name = f'{model}__ALL12__fixed_equal_rank'
        portfolio = []
        for date, g in full.groupby('date'):
            g = g.sort_values('symbol').copy()
            mask = g.covered.to_numpy()
            for col in ['routed', 'control_routed']:
                value = g[col].to_numpy().copy()
                value[~mask] = np.nan
                g[col] = M.route_slots(g.baseline_rank.to_numpy(), value)
            portfolio.append(g[['row', 'date', 'block', 'routed', 'control_routed']])
            g = g[np.isfinite(g[TARGET]) & np.isfinite(g.meta_model)]
            if len(g) < 10:
                continue
            values = g[['routed', 'control_routed', 'baseline_rank']].to_numpy()
            c, mm = matrix_scores(values, g[TARGET].to_numpy(), g.meta_model.to_numpy())
            for j, arm in enumerate(['signal', 'asset_mean', 'baseline']):
                ensemble_rows.append(dict(task=name, model=model, family='ALL12', rep='equal_rank', window='all11',
                    seed='ensemble', date=date, block=int(g.block.iloc[0]), sample='routed', arm=arm,
                    corr=float(c[j]) if np.isfinite(c[j]) else 0., mmc=float(mm[j]),
                    payout=.1*(float(c[j]) if np.isfinite(c[j]) else 0.)+float(mm[j]), n_coins=len(g)))
        parquet(ROOT/'ensembles'/f'{name}.parquet', pd.concat(portfolio, ignore_index=True))
    daily = pd.concat([daily, pd.DataFrame(ensemble_rows)], ignore_index=True)
    parquet(ROOT/'scores_with_fixed_ensembles.parquet', daily)
    table, deltas = [], []
    for (task, sample, seed), g in daily.groupby(['task', 'sample', 'seed']):
        signal = g[g.arm == 'signal'].sort_values('date')
        if signal.empty:
            continue
        y = signal.payout.to_numpy()
        curve = np.cumsum(y)
        dd = np.maximum.accumulate(np.r_[0, curve])[1:] - curve
        table.append(dict(task=task, sample=sample, seed=seed, corr=signal['corr'].mean(),
                          mmc=signal.mmc.mean(), payout=y.mean(), payout_std=y.std(ddof=1),
                          maxdd_payout=float(dd.max()), n_dates=len(signal), coins=signal.n_coins.mean()))
        for reference in ['asset_mean', 'baseline']:
            other = g[g.arm == reference]
            paired = signal.merge(other, on='date', suffixes=('_signal', '_control'), validate='1:1')
            if paired.empty:
                continue
            for metric in ['corr', 'mmc', 'payout']:
                diff = (paired[f'{metric}_signal']-paired[f'{metric}_control']).to_numpy()
                mean, t = newey_west(diff, lag=20)
                ci = bootstrap_mean(diff) if seed == 'ensemble' else [None, None]
                deltas.append(dict(task=task, sample=sample, seed=seed, reference=reference, metric=metric,
                                   delta=mean, hac20_t=t, ci_low=ci[0], ci_high=ci[1], n=len(diff)))
    table = pd.DataFrame(table).sort_values(['sample', 'seed', 'mmc'], ascending=[True, True, False])
    table.to_csv(ROOT/'table.csv', index=False)
    pd.DataFrame(deltas).to_csv(ROOT/'paired_deltas.csv', index=False)
    daily.groupby(['task', 'sample', 'seed', 'arm', 'block'])[['corr', 'mmc', 'payout']].mean().to_csv(ROOT/'by_block.csv')
    make_report(table, pd.DataFrame(deltas), daily)


def make_report(table, deltas, daily):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary = table[table.seed == 'ensemble']
    routed = primary[primary['sample'] == 'routed']
    fig, ax = plt.subplots(figsize=(12, 5))
    for model in ['ridge', 'mlp_torch', 'ALL24']:
        name = f'{model}__ALL12__fixed_equal_rank'
        p = daily[(daily.task == name) & (daily.arm == 'signal')].sort_values('date')
        ax.plot(p.date, p.payout.cumsum(), label=model)
    ax.set_title('E205 — fixed equal-rank formula portfolios')
    ax.set_ylabel('Cumulative payout score'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT/'fixed_ensemble_payout.png', dpi=150); plt.close(fig)
    audit = dict(outcome='PASS WITH CONDITIONS',
        checks=['scalar and batched Torch reference tests; independent asset losses and scalers',
                'original panel retained, missing candidate values stored as NaN, covered-only fits',
                'strict source prefix rebuilds for all12 families and20/240 windows',
                'exact covered rank-slot sets/uncovered ranks checked on every emitted portfolio',
                'finite per-asset-mean full-universe fallback and identical candidate/control coverage'],
        conditions=['Historical source availability is not demonstrated by current vendor snapshots.',
                    'Standalone momentum research; no marginal native-feature or live-parity claim.',
                    'All window/formula comparisons are development-only, many exploratory comparisons.'])
    save(ROOT/'audit.json', audit)
    body = '<h1>E205 — Per-asset momentum portfolios</h1>'
    body += '<p>Independent per-asset ridge and sklearn-initialised Torch identity MLPs. '
    body += 'Temporal inner-MSE selection; outer predictions only. Five MLP seeds averaged before ranking. '
    body += 'Twelve formula datasets, each with eleven TS-binned lookbacks as joint inputs; no one-hot or cross-sectional signal processing. '
    body += 'Payout=0.1 CORR + historical-meta proxy MMC. Drawdown is a payout-score decline, not investment loss. '
    body += 'The final return holdout remains closed. Outer rankings are exploratory.</p>'
    body += '<h2>Fixed portfolios, defined before scoring</h2>'+routed[routed.task.str.contains('fixed_equal_rank')].to_html(index=False)
    body += '<img style="width:100%" src="fixed_ensemble_payout.png">'
    body += '<h2>All per-signal portfolios — seed ensemble, sorted by proxy MMC</h2>'+primary.to_html(index=False)
    body += '<h2>Paired portfolio deltas</h2>'+deltas[deltas.seed == 'ensemble'].to_html(index=False)
    body += '<h2>Audit and limitations</h2><pre>'+html.escape(json.dumps(audit, indent=2))+'</pre>'
    body += '<h2>Frozen study design</h2><pre>'+html.escape((ROOT/'registration.json').read_text())+'</pre>'
    page = ROOT/'E205_asset_momentum.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:8px;'
                    'border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px}pre{white-space:pre-wrap}'
                    '</style></head><body>'+body+'</body></html>')
    try:
        from src.analysis.html_pdf import to_pdf
        to_pdf(str(page), 'lovelace')
        pdf = str(page.with_suffix('.pdf'))
    except Exception as exc:
        save(ROOT/'pdf_error.json', dict(error=str(exc)))
        pdf = None
    save(ROOT/'COMPLETE.json', dict(experiment='E205', completed=time.time(), portfolios=24,
         fixed_ensembles=3, html=str(page), pdf=pdf, audit=audit['outcome']))


def run(workers):
    started = time.time()
    save(ROOT/'launch.json', dict(host=socket.gethostname(), pid=os.getpid(), start=started, workers=workers))
    try:
        prepare(); baseline()
        tasks = [(m, f) for m in ['ridge', 'mlp_torch'] for f in M.FAMILIES]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_task, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                task = future.result()
                save(ROOT/'progress.json', dict(completed_tasks=i, total=len(tasks), latest=task,
                     elapsed_seconds=time.time()-started, updated=time.time()))
                print('TASK', i, '/', len(tasks), task, flush=True)
        finish(tasks)
    except Exception as exc:
        save(ROOT/'FAILED.json', dict(error=str(exc), traceback=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'finish'])
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'run':
        run(args.workers)
    else:
        finish([(m, f) for m in ['ridge', 'mlp_torch'] for f in M.FAMILIES])
