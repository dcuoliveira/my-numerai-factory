"""E219: E205/E209-style independent per-asset models on reversal metrics.

Seventeen reversal formula datasets (src/data_mani/reversal_metrics.py), each
jointly holding its eleven own-history-binned (TS) windows, plus the matched
E205 momentum132 control fitted freshly on the same rows. Unchanged E205
per-asset fitting, fallback, routing and scoring. No pooled fits, asset
dummies or cross-sectional inputs.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
import fcntl
import html
import json
import socket
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
import e205_asset_momentum as B
from src.analysis import asset_momentum as M
from src.analysis.cross_target import newey_west
from src.data_mani import reversal_metrics as R
from e150_finish_scoring import matrix_scores

warnings.filterwarnings('ignore', category=RuntimeWarning)
ROOT = REPO/'results/reports/crypto/e219_asset_reversal'
B.ROOT = ROOT
MOMENTUM = REPO/'results/reports/crypto/e205_asset_momentum'
FORMULAS = list(R.FAMILIES)
GROUPS = {name: 11 for name in FORMULAS}
GROUPS['momentum132'] = 132
TASKS = [(model, name) for model in ['ridge', 'mlp_torch'] for name in GROUPS]
ENSEMBLE = 'REV17__fixed_equal_rank'
FACTOR_IDS = ('1', '1027')
AUDIT_SLOTS = (3, 7, 10)            # windows 5, 20, 120 (swing k 4, 14, 60)
AUDIT_CUTS = ('2022-12-30', '2024-12-31', '2025-12-31')
save, parquet, sha = B.save, B.parquet, B.sha
CODE = [Path(__file__), REPO/'experiments/e205_asset_momentum.py',
        REPO/'src/analysis/asset_momentum.py', REPO/'src/data_mani/reversal_metrics.py',
        B.CANDIDATES, REPO/'src/prediction/torch_mlp.py', REPO/'experiments/e150_finish_scoring.py',
        REPO/'src/live/submit.py', REPO/'src/analysis/cross_target.py']


def feature_frame(values):
    return pd.DataFrame(values, columns=[f'feature_ts_{i}' for i in range(values.shape[1])], dtype=np.float32)


def factors(bars):
    return {'btc': bars['close']['BTC'], 'eth': bars['close']['ETH']}


def family_inputs(base, bars, cols, family):
    """Eleven TS-binned windows of one family attached at D-1; raw and TS frames kept for audits."""
    out = np.full((len(base), 11), np.nan, dtype=np.float32)
    frames = {}
    for slot in range(11):
        raw = R.signal(bars, family, slot, factors(bars))
        ts = pd.DataFrame(M.prior_bins(raw.to_numpy()), index=raw.index, columns=raw.columns)
        out[:, slot] = M.attach(base, ts, cols)
        frames[slot] = (raw, ts)
    return out, frames


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'PREPARED.json').exists():
        return B.verify()
    source_files = [B.SOURCE/'train.parquet', B.SOURCE/'historical_meta_models.parquet', B.PRICES,
                    MOMENTUM/'keys.parquet', MOMENTUM/'PREPARED.json']
    source_files += [MOMENTUM/'inputs'/f'{name}_ts.parquet' for name in M.FAMILIES]
    plan = dict(experiment='E219', registered='2026-11-25', revision=1,
        groups=GROUPS, families=R.FAMILIES, windows=R.WINDOWS, swing_k=R.SWING_K,
        lookbacks=dict(beta=R.BETA_WINDOW, volume_base=R.VOLUME_BASE, trend=R.TREND_WINDOW,
                       swing_search=R.SWING_SEARCH),
        n_portfolios=len(TASKS), candidates=json.loads(B.CANDIDATES.read_text()), seeds=B.SEEDS,
        models=['ridge', 'mlp_torch'],
        fitting='Unchanged E205 independent per-asset temporal-MSE tuning; ridge100 alphas; identity '
                'Torch MLP80 frozen candidates, sklearn weights and biases, five refit seeds.',
        folds='outer130/purge22/test130; inner68/purge22/validation40; min40/20/80 rows',
        coverage='Original keys and candidate NaNs kept. Common causal intersection of all 17x11 '
                 'reversal inputs, momentum132 and the E205 identity/OHLC cohort; BTC and ETH excluded '
                 '(they are the residual factors). Per-arm source masks kept. No coverage predictors.',
        controls='Matched per-asset historical target mean; full-universe historical-mean fallback; '
                 'momentum132 refitted on identical common rows.',
        ensembles='Three prespecified equal-rank portfolios over the 17 reversal formulas: ridge, Torch, both.',
        claim='Standalone per-asset development research; marginal value over momentum132 only.',
        limitation='Current vendor snapshots cannot prove historical availability. Development only; '
                   'target dates >=2026-02-20 excluded at source read.',
        code_hashes={str(p.relative_to(REPO)): sha(p) for p in CODE},
        input_hashes={str(p.relative_to(REPO)): sha(p) for p in source_files})
    if (ROOT/'registration.json').exists():
        previous = B.verify()
        assert previous['input_hashes'] == plan['input_hashes'], 'Source snapshot changed during preparation'
    save(ROOT/'registration.json', plan)
    base = pd.read_parquet(B.SOURCE/'train.parquet', columns=['date', 'ucid', B.TARGET],
                           filters=[('date', '<', B.CUTOFF)]).reset_index()
    base = base.sort_values(['date', 'symbol']).reset_index(drop=True)
    assert base.date.max() < B.CUTOFF and not base.duplicated(['ucid', 'date']).any()
    old = pd.read_parquet(MOMENTUM/'keys.parquet')
    pd.testing.assert_frame_equal(base[['symbol', 'ucid', 'date', B.TARGET]],
                                  old[['symbol', 'ucid', 'date', B.TARGET]])
    meta = pd.read_parquet(B.SOURCE/'historical_meta_models.parquet',
                           filters=[('date', '<', B.CUTOFF)]).reset_index()
    base = base.merge(meta[['symbol', 'date', 'prediction']].rename(columns={'prediction': 'meta_model'}),
                      on=['symbol', 'date'], how='left', validate='1:1')
    base['row'] = np.arange(len(base))
    prices = pd.read_parquet(B.PRICES, filters=[('date', '<', B.CUTOFF)])
    close, cols, cohort = M.price_context(base, prices)
    bars = R.wide_bars(prices)
    if not bars['close'].equals(close):
        raise RuntimeError('Wide bars differ from the E163 price context')
    truncated = {cut: R.wide_bars(prices[prices.date <= pd.Timestamp(cut)]) for cut in AUDIT_CUTS}
    inputs, audits = {}, []
    for family in FORMULAS:
        path = ROOT/'inputs'/f'{family}_ts.parquet'
        if path.exists():
            values = pd.read_parquet(path).to_numpy()
        else:
            values, frames = family_inputs(base, bars, cols, family)
            for cut, short in truncated.items():
                for slot in AUDIT_SLOTS:
                    raw, ts = frames[slot]
                    # Coins with no bar before the cutoff are absent from the truncated
                    # pivot; their full-history values up to the cutoff must be all NaN.
                    sraw = R.signal(short, family, slot, factors(short)).reindex(columns=raw.columns)
                    sts = M.prior_bins(sraw.to_numpy())
                    np.testing.assert_allclose(raw.loc[:cut].to_numpy(), sraw.to_numpy(),
                                               rtol=1e-9, atol=1e-12, equal_nan=True)
                    np.testing.assert_array_equal(ts.loc[:cut].to_numpy(), sts)
                    audits.append(dict(family=family, slot=slot, window=R.WINDOWS[slot], cutoff=cut, changed=0))
            parquet(path, feature_frame(values))
        inputs[family] = values
        print('PREPARE', family, 'finite rows', int(np.isfinite(values).all(axis=1).sum()), flush=True)
    if audits:
        pd.DataFrame(audits).to_csv(ROOT/'source_prefix_audits.csv', index=False)
    inputs['momentum132'] = np.column_stack([pd.read_parquet(MOMENTUM/'inputs'/f'{f}_ts.parquet').to_numpy()
                                             for f in M.FAMILIES])
    eligible = cohort & ~base.ucid.astype(str).isin(FACTOR_IDS).to_numpy()
    masks = {name: eligible & np.isfinite(v).all(axis=1) for name, v in inputs.items()}
    common = eligible.copy()
    for mask in masks.values():
        common &= mask
    assert common.any(), 'No common source-covered rows'
    base['covered'] = common
    mask_table = base[['row', 'symbol', 'ucid', 'date']].copy()
    dataset_hashes = {}
    for name, values in inputs.items():
        mask_table[name] = masks[name]
        values = values.astype(np.float32).copy()
        values[~common] = np.nan
        frame = pd.concat([base[['row', 'symbol', 'ucid', 'date', B.TARGET, 'covered']], feature_frame(values)], axis=1)
        path = ROOT/'datasets'/f'{name}.parquet'
        parquet(path, frame)
        dataset_hashes[name] = sha(path)
    parquet(ROOT/'source_masks.parquet', mask_table)
    parquet(ROOT/'keys.parquet', base)
    save(ROOT/'date_grid.json', list(map(str, np.sort(base.date.unique()))))
    base.groupby('date').covered.agg(panel_rows='size', covered_rows='sum', covered_share='mean').to_csv(
        ROOT/'coverage_by_date.csv')
    save(ROOT/'PREPARED.json', dict(rows=len(base), covered=int(common.sum()), assets=int(base.ucid.nunique()),
         eligible_assets=int(base.loc[common].ucid.nunique()), datasets=len(inputs), feature_counts=GROUPS,
         per_arm_rows={k: int(v.sum()) for k, v in masks.items()}, dataset_hashes=dataset_hashes,
         keys_sha256=sha(ROOT/'keys.parquet'), prefix_checks=len(audits), finished=time.time()))
    print('PREPARED', json.loads((ROOT/'PREPARED.json').read_text()), flush=True)
    return plan


def run_task(task):
    B.ROOT = ROOT
    B.M.WINDOWS = list(range(GROUPS[task[1]]))
    return B.run_task(task)


def score_tasks():
    keys = pd.read_parquet(ROOT/'keys.parquet')
    fallback = pd.read_parquet(ROOT/'baseline.parquet')
    rows = []
    for i, task in enumerate(TASKS, 1):
        path = ROOT/'daily'/f'{task[0]}__{task[1]}.parquet'
        r = pd.read_parquet(path).to_dict('records') if path.exists() else B.score_one(task, keys, fallback)
        rows.extend(r)
        print('SCORE', i, '/', len(TASKS), flush=True)
    return pd.DataFrame(rows), keys, fallback


def add_ensembles(daily, keys, fallback):
    rows = []
    for model in ['ridge', 'mlp_torch', 'both']:
        selected = [t for t in TASKS if t[1] in FORMULAS and (model == 'both' or t[0] == model)]
        total, coverage = None, None
        for m, f in selected:
            p = pd.read_parquet(ROOT/'portfolios'/f'{m}__{f}.parquet').set_index('row')
            if coverage is None:
                coverage = p.covered
            else:
                pd.testing.assert_series_equal(coverage, p.covered)
            x = p[['routed', 'control_routed']]
            total = x if total is None else total.add(x)
        total /= len(selected)
        total['covered'] = coverage
        full = fallback.merge(keys[['row', 'symbol', B.TARGET, 'meta_model']], on='row', validate='1:1').merge(
            total.reset_index(), on='row', validate='1:1')
        name = f'{model}__{ENSEMBLE}'
        portfolios = []
        for date, g in full.groupby('date'):
            g = g.sort_values('symbol').copy()
            mask = g.covered.to_numpy()
            for col in ['routed', 'control_routed']:
                values = g[col].to_numpy().copy()
                values[~mask] = np.nan
                g[col] = M.route_slots(g.baseline_rank.to_numpy(), values)
            portfolios.append(g[['row', 'date', 'block', 'routed', 'control_routed']])
            g = g[np.isfinite(g[B.TARGET]) & np.isfinite(g.meta_model)]
            if len(g) < 10:
                continue
            c, mm = matrix_scores(g[['routed', 'control_routed', 'baseline_rank']].to_numpy(),
                                  g[B.TARGET].to_numpy(), g.meta_model.to_numpy())
            for j, arm in enumerate(['signal', 'asset_mean', 'baseline']):
                cv = float(c[j]) if np.isfinite(c[j]) else 0.
                rows.append(dict(task=name, model=model, family='REV17', rep='equal_rank', window='all11',
                                 seed='ensemble', date=date, block=int(g.block.iloc[0]), sample='routed', arm=arm,
                                 corr=cv, mmc=float(mm[j]), payout=.1*cv+float(mm[j]), n_coins=len(g)))
        parquet(ROOT/'ensembles'/f'{name}.parquet', pd.concat(portfolios, ignore_index=True))
    return pd.concat([daily, pd.DataFrame(rows)], ignore_index=True)


def summarize(daily):
    table, deltas = [], []
    for (task, sample, seed), g in daily.groupby(['task', 'sample', 'seed']):
        s = g[g.arm == 'signal'].sort_values('date')
        if s.empty:
            continue
        y = s.payout.to_numpy()
        curve = np.cumsum(y)
        dd = float((np.maximum.accumulate(np.r_[0, curve])[1:] - curve).max())
        c = s['corr'].to_numpy()
        table.append(dict(task=task, sample=sample, seed=seed, corr=c.mean(), mmc=s.mmc.mean(),
                          c_sharpe=c.mean()/c.std(), payout=y.mean(), p_sharpe=y.mean()/y.std(),
                          maxdd_payout=dd, calmar=y.mean()/dd if dd > 0 else np.nan,
                          burn_days=int((y < 0).sum()), n_dates=len(s), coins=s.n_coins.mean()))
        model = task.split('__')[0]
        references = {r: g[g.arm == r] for r in ['asset_mean', 'baseline']}
        if not task.endswith('__momentum132') and model in ('ridge', 'mlp_torch'):
            references['matched_momentum132'] = daily[(daily.task == f'{model}__momentum132') & (daily['sample'] == sample)
                                                      & (daily.seed == seed) & (daily.arm == 'signal')]
        for ref, r in references.items():
            p = s.merge(r, on='date', suffixes=('_signal', '_control'), validate='1:1')
            if p.empty:
                continue
            assert len(p) == len(s), 'Paired date grids differ'
            for metric in ['corr', 'mmc', 'payout']:
                diff = (p[f'{metric}_signal'] - p[f'{metric}_control']).to_numpy()
                mean, t = newey_west(diff, lag=20)
                ci = B.bootstrap_mean(diff) if seed == 'ensemble' else [None, None]
                deltas.append(dict(task=task, sample=sample, seed=seed, reference=ref, metric=metric,
                                   delta=mean, hac20_t=t, ci_low=ci[0], ci_high=ci[1], n=len(diff)))
    return pd.DataFrame(table).sort_values('payout', ascending=False), pd.DataFrame(deltas)


def audit_outputs():
    B.verify()
    fallback = pd.read_parquet(ROOT/'baseline.parquet')
    keys = pd.read_parquet(ROOT/'keys.parquet')
    checks, coverage = [], None
    for model, name in TASKS:
        f = pd.read_parquet(ROOT/'datasets'/f'{name}.parquet')
        pd.testing.assert_frame_equal(f[['row', 'ucid', 'date', B.TARGET, 'covered']],
                                      keys[['row', 'ucid', 'date', B.TARGET, 'covered']])
        features = f.filter(regex='^feature_ts_')
        assert features.shape[1] == GROUPS[name]
        assert features.loc[~f.covered].isna().all().all()
        assert np.isfinite(features.loc[f.covered]).all().all()
        p = pd.read_parquet(ROOT/'portfolios'/f'{model}__{name}.parquet').merge(
            fallback[['row', 'baseline_rank']], on='row', validate='1:1')
        cur = p[['row', 'covered']].sort_values('row').reset_index(drop=True)
        if coverage is None:
            coverage = cur
        else:
            pd.testing.assert_frame_equal(coverage, cur)
        assert np.isfinite(p[['routed', 'control_routed', 'baseline_rank']]).all().all()
        assert (p.loc[~p.covered, 'routed'] == p.loc[~p.covered, 'baseline_rank']).all()
        for _, g in p.groupby('date'):
            for col in ['routed', 'control_routed']:
                np.testing.assert_array_equal(np.sort(g.loc[g.covered, col]), np.sort(g.loc[g.covered, 'baseline_rank']))
        checks.append(dict(task=f'{model}__{name}', rows=len(p), covered=int(p.covered.sum()), pass_checks=True))
    prefixes = pd.read_csv(ROOT/'source_prefix_audits.csv')
    assert prefixes.changed.eq(0).all() and len(prefixes) == len(FORMULAS)*len(AUDIT_SLOTS)*len(AUDIT_CUTS)
    audit = dict(outcome='PASS WITH CONDITIONS', checks=checks, prefix_rebuilds=len(prefixes),
                 conditions=['Current vendor histories and retrospective identity filters cannot prove historical availability.',
                             'Standalone per-asset research; marginal comparison only against the matched momentum132 model.',
                             'Formula definitions fixed before fitting; window grid and lookbacks not tuned on outcomes.',
                             'No live deployment or investment-return claim.'])
    save(ROOT/'audit.json', audit)
    return audit


def label(task):
    model, group = task.split('__')[:2]
    if group == 'REV17':
        return 'All 17 reversal formulas, equal-rank'
    if group == 'momentum132':
        return 'Momentum132 control (E205 inputs)'
    return f'{group}: {R.FAMILIES[group]}'


def report(table, deltas, daily, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    routed = daily[(daily['sample'] == 'routed') & (daily.seed == 'ensemble') & (daily.arm == 'signal')]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    cmap = plt.get_cmap('tab20')
    for ax, model in zip(axes, ['ridge', 'mlp_torch']):
        for i, f in enumerate(FORMULAS):
            p = routed[routed.task == f'{model}__{f}'].sort_values('date')
            ax.plot(p.date, p.payout.cumsum(), lw=.8, color=cmap(i % 20), label=f)
        for t, style in [(f'{model}__momentum132', 'k--'), (f'{model}__{ENSEMBLE}', 'k-')]:
            p = routed[routed.task == t].sort_values('date')
            ax.plot(p.date, p.payout.cumsum(), style, lw=2, label=t.split('__')[1])
        am = daily[(daily.task == f'{model}__{ENSEMBLE}') & (daily.arm == 'asset_mean')].sort_values('date')
        ax.plot(am.date, am.payout.cumsum(), ':', color='grey', lw=2, label='asset mean control')
        ax.set_title(model)
        ax.set_ylabel('Cumulative payout score')
        ax.legend(fontsize=6, ncol=3)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT/'cumulative_payout.png', dpi=150)
    plt.close(fig)
    wide = routed[routed.model == 'ridge'].pivot_table(index='date', columns='family', values='payout')
    corr = wide.corr()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iat[i, j]:.2f}', ha='center', va='center', fontsize=5)
    fig.colorbar(im)
    ax.set_title('Daily payout correlation, ridge portfolios (routed)')
    fig.tight_layout()
    fig.savefig(ROOT/'payout_correlation.png', dpi=150)
    plt.close(fig)
    primary = table[(table.seed == 'ensemble') & (table['sample'] == 'routed')].copy()
    primary.insert(1, 'inputs', primary.task.map(label))
    cmp = deltas[(deltas.seed == 'ensemble') & (deltas['sample'] == 'routed') & (deltas.metric == 'payout')]
    cmp = cmp.pivot_table(index='task', columns='reference', values=['delta', 'hac20_t']).round(5)
    cmp.columns = [f'{a}_vs_{b}' for a, b in cmp.columns]
    prepared = json.loads((ROOT/'PREPARED.json').read_text())
    fmt = lambda x: f'{x:.6f}'
    body = '<h1>E219 — Per-asset reversal metrics, 11 TS-binned windows</h1>'
    body += ('<p>One ridge and one sklearn-initialised Torch identity MLP per asset, per dataset and fold '
             '(the unchanged E205/E209 procedure). Each of the 17 reversal datasets holds its formula at 11 '
             f'weekday windows {R.WINDOWS} (swing pivots: half-widths {R.SWING_K}), each converted to a '
             'strictly-prior own-history percentile (252 bars, min 126) and five bins. No cross-sectional '
             'inputs, no asset dummies. Momentum132 (E205) is refitted on the identical rows as the matched '
             'control. Five MLP seeds averaged before ranking; uncovered coins keep their per-asset-mean '
             'fallback rank. Payout = 0.1 CORR + historical-meta proxy MMC; scores, not investment returns. '
             'Identity MLPs are affine, so ridge vs MLP is an optimisation comparison, not a nonlinearity test.</p>')
    body += (f'<p>Panel rows {prepared["rows"]:,}; common covered rows {prepared["covered"]:,} on '
             f'{prepared["eligible_assets"]} assets (BTC and ETH excluded as residual factors). '
             f'{audit["prefix_rebuilds"]} source-prefix rebuilds unchanged. Audit: {audit["outcome"]}.</p>')
    body += '<h2>Full-universe portfolios, sorted by payout</h2>' + primary.to_html(index=False, float_format=fmt)
    body += '<img style="width:100%" src="cumulative_payout.png">'
    body += '<h2>Paired payout deltas (HAC lag 20)</h2>' + cmp.to_html(float_format=lambda x: f'{x:.5f}')
    body += ('<p>Block-20 bootstrap intervals are in paired_deltas.csv. Unadjusted for the 36 comparisons; '
             'family rankings are exploratory.</p>')
    body += '<img style="width:90%" src="payout_correlation.png">'
    body += '<h2>Audit conditions</h2><ul>' + ''.join(f'<li>{html.escape(c)}</li>' for c in audit['conditions']) + '</ul>'
    body += '<h2>Covered-only portfolios and individual seeds</h2>' + table.to_html(index=False, float_format=fmt)
    page = ROOT/'E219_asset_reversal.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:8px;'
                    'border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px}</style></head><body>'
                    + body + '</body></html>')
    from src.analysis.html_pdf import to_pdf
    to_pdf(str(page), 'lovelace')
    pdf = page.with_suffix('.pdf')
    import fitz
    with fitz.open(pdf) as document:
        assert len(document) > 2 and 'E219' in document[0].get_text()
        pages = len(document)
    save(ROOT/'COMPLETE.json', dict(experiment='E219', finished=time.time(), portfolios=len(TASKS), fixed_ensembles=3,
                                    pdf=str(pdf), pdf_sha256=sha(pdf), pages=pages, audit=audit['outcome']))


def finish():
    B.verify()
    assert all((ROOT/'fits'/f'{m}__{f}'/'COMPLETE.json').exists() for m, f in TASKS)
    daily, keys, fallback = score_tasks()
    daily = add_ensembles(daily, keys, fallback)
    parquet(ROOT/'scores_with_fixed_ensembles.parquet', daily)
    table, deltas = summarize(daily)
    table.to_csv(ROOT/'table.csv', index=False)
    deltas.to_csv(ROOT/'paired_deltas.csv', index=False)
    daily.groupby(['task', 'sample', 'seed', 'arm', 'block'])[['corr', 'mmc', 'payout']].mean().to_csv(ROOT/'by_block.csv')
    audit = audit_outputs()
    report(table, deltas, daily, audit)


def email_report():
    from src.live import notify
    receipt = ROOT/'email_delivery.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('smtp_accepted'):
        return
    complete = json.loads((ROOT/'COMPLETE.json').read_text())
    pdf = Path(complete['pdf'])
    assert sha(pdf) == complete['pdf_sha256']
    import smtplib
    from email.message import EmailMessage
    from email.utils import getaddresses, make_msgid
    from src.live.env import load_env
    load_env(required=())
    config = notify._config()
    if config is None:
        raise RuntimeError('SMTP configuration unavailable')
    message = EmailMessage()
    message['Subject'] = '[numerai-crypto] E219 completed — per-asset reversal metrics PDF'
    message['From'], message['To'], message['Message-ID'] = config['user'], config['to'], make_msgid()
    message.set_content('E219 has completed: 17 reversal formula datasets x {ridge, Torch identity MLP}, '
                        'per-asset fits on 11 TS-binned windows, matched momentum132 control. PDF attached. '
                        'Exploratory development results; no live changes.')
    message.add_attachment(pdf.read_bytes(), maintype='application', subtype='pdf', filename=pdf.name)
    recipients = [a for _, a in getaddresses([config['to']])]
    with smtplib.SMTP(config['host'], config['port'], timeout=120) as server:
        server.starttls()
        server.login(config['user'], config['password'])
        refused = server.send_message(message, from_addr=config['user'], to_addrs=recipients)
        assert not refused, 'SMTP refused report recipient'
    save(receipt, dict(smtp_accepted=True, time=time.time(), pdf_sha256=sha(pdf), message_id=message['Message-ID']))
    print('Report email accepted', flush=True)


def run(workers):
    started = time.time()
    save(ROOT/'launch.json', dict(host=socket.gethostname(), pid=os.getpid(), start=started, workers=workers))
    prepare()
    B.baseline()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, t): t for t in TASKS}
        for future in as_completed(futures):
            task = future.result()
            n = sum((ROOT/'fits'/f'{m}__{f}'/'COMPLETE.json').exists() for m, f in TASKS)
            save(ROOT/'progress.json', dict(completed_tasks=n, total=len(TASKS), latest=task,
                                            updated=time.time(), elapsed_seconds=time.time()-started))
            print('TASK', n, '/', len(TASKS), task, flush=True)
    finish()
    email_report()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'finish', 'email'])
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT/'runner.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for attempt in range(1, 4):
            try:
                if args.command == 'run':
                    run(args.workers)
                elif args.command == 'prepare':
                    prepare()
                elif args.command == 'finish':
                    finish()
                    email_report()
                else:
                    email_report()
                (ROOT/'FAILED.json').unlink(missing_ok=True)
                break
            except Exception as exc:
                save(ROOT/'FAILED.json', dict(error=str(exc), traceback=traceback.format_exc(),
                                              time=time.time(), attempt=attempt))
                if attempt == 3:
                    raise
                print('RETRY', attempt, str(exc), flush=True)
                time.sleep(15)
