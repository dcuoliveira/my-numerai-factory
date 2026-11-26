"""E222: does the mean of Chronos-2's FEATURE-row states add to its TARGET-row state?

E221 keeps only the target row's pre-head state; the 187 reversal187 covariate
rows' states are computed and discarded. E222 re-runs the Chronos-2 pass on
E221's frozen panel and keeps both (src/data_mani/e222_featmean_embeddings.py),
then fits per-asset design-C dual ridge, rolling standardization (63/126/252
chosen per arm by validation MSE), E206 target20 splits, on three arms:

  T  -- target-row state (E221's H, parity-checked)              768
  F  -- mean of the 187 feature-row states at the same patch      768
  TF -- T concatenated with F                                     1536

One common causal sample for all three. Paired Newey-West (lag 20) deltas
F-T, TF-T, TF-F on CORR, MMC and payout.

    python3 experiments/e222_c2_featmean_ridge.py extract [--smoke]   # GPU worker
    python3 experiments/e222_c2_featmean_ridge.py all                 # wait, collect, fit, score, PDF, email
"""
import argparse
import json
import os
import shutil
import smtplib
import subprocess
import sys
import time
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from experiments import e221_reversal187_tsfm as S
from src.analysis import tsfm_momentum_study as N
from src.analysis.cross_target import newey_west
from src.analysis.e218_ktz_report import metrics
from e150_finish_scoring import matrix_scores

ROOT = REPO/'results/reports/crypto/e222_c2_featmean_ridge'
SCR = Path('/scratch/danielco/e222_c2_featmean_ridge')
PANEL_ROOT = S.ROOT                       # E221's frozen panel, read only
TARGET = S.TARGET
ARMS = ['c2tgt', 'c2fmean', 'c2cat']
LABEL = {'c2tgt': 'T (target row)', 'c2fmean': 'F (feature-row mean)', 'c2cat': 'TF (concatenated)'}
D = 768
save, parquet, sha = E.save, E.parquet, E.sha


def bind():
    E.ROOT, E.SCR, E.TARGET = PANEL_ROOT, SCR, TARGET
    S.SCR = SCR                           # S.masks / S.ridge_portfolio read common_rows.npy from here


def panel():
    return pd.read_parquet(PANEL_ROOT/'panel.parquet')


def raw_dir():
    return SCR/'embeddings'/'chronos2'


def extract(smoke=False):
    os.environ.update(TSFM_EMBED_ROOT=str(PANEL_ROOT), TSFM_EMBED_SCR=str(SCR), TSFM_EMBED_TARGET=TARGET,
                      TSFM_EMBED_WALL=str(S.WALL), E206_BATCH='4')
    cmd = [E.VENVS['chronos2'], '-u', '-m', 'src.data_mani.e222_featmean_embeddings', '--device', 'cuda:0'] + (['--smoke'] if smoke else [])
    subprocess.run(cmd, cwd=REPO, check=True)


def extraction_complete(p):
    wanted = p.groupby('ucid').covered.any()
    return all((raw_dir()/f'asset_{a}.npz').exists() for a in wanted.index[wanted])


def collect():
    """Split the 1536-wide cache into T, F, TF row matrices; parity-check T against E221; common mask."""
    if (ROOT/'EMBEDDINGS_READY.json').exists():
        return
    p = panel(); manifest = json.loads((raw_dir()/'extractor.json').read_text())
    assert manifest['width'] == 2*D and manifest['target'] == TARGET
    outs = {}
    for arm, width in [('c2tgt', D), ('c2fmean', D), ('c2cat', 2*D)]:
        (SCR/'embeddings'/arm).mkdir(parents=True, exist_ok=True)
        outs[arm] = np.lib.format.open_memmap(SCR/'embeddings'/arm/'H_rows.partial.npy', mode='w+', dtype='float16', shape=(len(p), width))
        outs[arm][:] = np.nan
    e221 = Path('/scratch/danielco/e221_reversal187_tsfm/embeddings/chronos2')
    diffs = []; compared = 0
    for f in sorted(raw_dir().glob('asset_*.npz')):
        assert json.loads(f.with_suffix('.json').read_text())['fingerprint'] == manifest['fingerprint']
        with np.load(f) as z:
            rows, st = z['rows'], z['states']
        if not len(rows):
            continue
        outs['c2tgt'][rows] = st[:, :D]; outs['c2fmean'][rows] = st[:, D:]; outs['c2cat'][rows] = st
        ref = e221/f.name
        if ref.exists():
            with np.load(ref) as r:
                assert np.array_equal(r['rows'], rows), f.name
                diffs.append(float(np.abs(r['states'].astype(np.float32)-st[:, :D].astype(np.float32)).max())); compared += len(rows)
    for arm in ARMS:
        outs[arm].flush(); del outs[arm]
        (SCR/'embeddings'/arm/'H_rows.partial.npy').replace(SCR/'embeddings'/arm/'H_rows.npy')
    parity = dict(rows_compared=compared, assets_compared=len(diffs), max_abs_diff=max(diffs) if diffs else None)
    assert diffs and parity['max_abs_diff'] < 0.05, parity
    common = p.covered.to_numpy().copy()
    for arm in ARMS:
        valid = np.isfinite(np.load(SCR/'embeddings'/arm/'H_rows.npy', mmap_mode='r')).all(axis=1)
        np.save(SCR/'embeddings'/arm/'valid_rows.npy', valid); common &= valid
    common = N.common_history(p.ucid.to_numpy(), p.date.to_numpy(), common, 32)
    common &= pd.Series(common).groupby(p.date).transform('sum').ge(2).to_numpy()
    np.save(SCR/'common_rows.npy', common)
    save(ROOT/'EMBEDDINGS_READY.json', dict(common_rows=int(common.sum()), t_parity_vs_e221=parity,
                                            mask_sha256=sha(SCR/'common_rows.npy'), time=time.time()))


def kernel(arm, phase, window, p):
    kern, valid, info = E.build_design(arm, 'rolling', phase, window, 'C', p)
    shutil.rmtree(SCR/'normalizations'/arm/'rolling'/f'{phase}_{window}', ignore_errors=True)   # quota; regenerable
    return kern, valid, info


def fit(arm):
    if (ROOT/'fits'/arm/'COMPLETE.json').exists():
        return
    bind(); p = panel(); choices = []
    for window in N.ROLLING:
        kern, valid, info = kernel(arm, 'selection', window, p)
        _, records = S.ridge_portfolio(p, {'selection': (kern, valid)}, None, ('selection',))
        choices.append((float(np.mean([r['mse'] for r in records.values()])), window, info)); del kern
    score, window, info = min(choices, key=lambda c: (c[0], c[1]))
    kernels = {phase: kernel(arm, phase, window, p)[:2] for phase in ['selection', 'final']}
    preds, records = S.ridge_portfolio(p, kernels, None)
    out = p.loc[p.test, ['row', 'ucid', 'symbol', 'date', TARGET, 'meta_model']].copy()
    out['prediction'] = preds['final'][p.test.to_numpy()]
    (ROOT/'fits'/arm).mkdir(parents=True, exist_ok=True); parquet(ROOT/'fits'/arm/'predictions.parquet', out)
    save(ROOT/'fits'/arm/'COMPLETE.json', dict(arm=arm, window=window, mean_asset_mse=score, assets=len(records),
         candidates=[dict(window=w, mse=s) for s, w, _ in choices], vocabulary=len(info['vocabulary']), time=time.time()))


def score():
    p = panel()
    test = p.loc[p.test, ['row', 'ucid', 'date', TARGET, 'meta_model']].sort_values(['date', 'ucid']).reset_index(drop=True)
    raw = np.column_stack([pd.read_parquet(ROOT/'fits'/a/'predictions.parquet').set_index('row').prediction
                           .reindex(test.row).to_numpy(float) for a in ARMS])
    ok = np.isfinite(raw).all(1) & np.isfinite(test[TARGET]).to_numpy() & np.isfinite(test.meta_model).to_numpy()
    dates, scores, counts = [], [], []
    for date, idx in test.groupby('date', sort=True).indices.items():
        rows = idx[ok[idx]]
        if len(rows) < 10:
            continue
        c, m = matrix_scores(raw[rows], test.loc[rows, TARGET].to_numpy(), test.loc[rows, 'meta_model'].to_numpy())
        dates.append(date); scores.append(np.column_stack([c, m])); counts.append(len(rows))
    scores = np.stack(scores); dates = np.asarray(dates)
    np.savez_compressed(ROOT/'daily_scores.npz', date=dates, arms=np.array(ARMS), scores=scores)
    windows = {a: json.loads((ROOT/'fits'/a/'COMPLETE.json').read_text())['window'] for a in ARMS}
    table = pd.DataFrame(dict(arm=[LABEL[a] for a in ARMS], width=[D, D, 2*D], window=[windows[a] for a in ARMS],
                              n_dates=len(dates), mean_assets=np.mean(counts), **metrics(scores)))
    table.to_csv(ROOT/'table.csv', index=False)
    pay = .1*scores[:, :, 0]+scores[:, :, 1]; deltas = []
    for a, b in [('c2fmean', 'c2tgt'), ('c2cat', 'c2tgt'), ('c2cat', 'c2fmean')]:
        ia, ib = ARMS.index(a), ARMS.index(b); row = dict(comparison=f'{LABEL[a]} - {LABEL[b]}')
        for k, s in [('corr', scores[:, ia, 0]-scores[:, ib, 0]), ('mmc', scores[:, ia, 1]-scores[:, ib, 1]), ('payout', pay[:, ia]-pay[:, ib])]:
            row[f'd_{k}'], row[f't_{k}'] = newey_west(s, 20)
        deltas.append(row)
    deltas = pd.DataFrame(deltas); deltas.to_csv(ROOT/'paired_deltas.csv', index=False)
    report(table, deltas, dates, pay, len(dates), int(ok.sum()))
    print(table.to_string(index=False)); print(deltas.to_string(index=False))


def report(table, deltas, dates, pay, n_dates, n_keys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.analysis.html_pdf import to_pdf
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, a in enumerate(ARMS):
        ax.plot(dates, pay[:, j].cumsum(), label=LABEL[a])
    ax.set_title('E222 cumulative payout, per-asset design-C ridge'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT/'cumulative_payout.png', dpi=110); plt.close(fig)
    parity = json.loads((ROOT/'EMBEDDINGS_READY.json').read_text())['t_parity_vs_e221']
    fmt = lambda v: f'{v:.5f}' if isinstance(v, float) else v
    body = ('<h1>E222: Chronos-2 feature-row states alongside the target-row state</h1>'
            '<p>Chronos-2 sees each asset as one group: the target20 series (35-day label wall) plus the 187 '
            'reversal187 series as covariates. E221 keeps only the target row\'s pre-head state. Here the same encoder '
            'call also yields the 187 feature rows\' states at the same output patch; F is their mean. '
            'Per-asset dual ridge on design C, rolling standardization with the window (63/126/252) chosen per arm '
            'by validation MSE, alpha by validation MSE, fresh final fit, E206 target20 splits. '
            f'Common sample: {n_dates} test dates, {n_keys:,} asset/date rows, identical for all three arms. '
            f'T reproduces E221\'s states (max |diff| {parity["max_abs_diff"]:.2g} over {parity["rows_compared"]:,} rows). '
            'Payout = 0.1 CORR + historical-meta proxy MMC.</p>')
    body += '<h2>Arms</h2>' + table.to_html(index=False, float_format=fmt)
    body += '<h2>Paired differences (Newey-West lag 20)</h2>' + deltas.to_html(index=False, float_format=fmt)
    body += '<h2>Cumulative payout</h2><img style="width:100%" src="cumulative_payout.png">'
    body += ('<h2>Limitations</h2><ul><li>Development history; one TSFM, one learner.</li>'
             '<li>TF shares one ridge penalty across both blocks.</li>'
             '<li>Pretrained checkpoint may overlap the test period; no live path.</li></ul>')
    page = ROOT/'E222_c2_featmean_ridge.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:9px;border-collapse:'
                    'collapse}td,th{border:1px solid #ccc;padding:3px}</style></head><body>'+body+'</body></html>')
    to_pdf(str(page), 'lovelace')


def email_report():
    if (ROOT/'email_delivery.json').exists():
        return
    from src.live.env import load_env
    from src.live.notify import _config
    load_env(required=()); config = _config(); assert config
    pdf = ROOT/'E222_c2_featmean_ridge.pdf'
    msg = EmailMessage(); msg['From'] = config['user']; msg['To'] = config['to']
    msg['Subject'] = '[numerai-crypto] E222 Chronos-2 feature-row states, ridge — completed PDF'
    msg.set_content('E222 is complete: T vs F vs TF per-asset design-C ridge on one common sample. PDF attached.')
    msg.add_attachment(pdf.read_bytes(), maintype='application', subtype='pdf', filename=pdf.name)
    with smtplib.SMTP(config['host'], config['port'], timeout=120) as server:
        server.starttls(); server.login(config['user'], config['password']); assert not server.send_message(msg)
    save(ROOT/'email_delivery.json', dict(smtp_accepted=True, time=time.time()))


def run_all():
    ROOT.mkdir(parents=True, exist_ok=True); p = panel()
    while not extraction_complete(p):
        save(ROOT/'status.json', dict(stage='waiting_for_extraction', assets=len(list(raw_dir().glob('asset_*.npz'))), time=time.time()))
        time.sleep(300)
    save(ROOT/'status.json', dict(stage='collect', time=time.time())); collect()
    for arm in ARMS:
        save(ROOT/'status.json', dict(stage='fit_'+arm, time=time.time())); fit(arm)
    save(ROOT/'status.json', dict(stage='score', time=time.time())); score(); email_report()
    save(ROOT/'COMPLETE.json', dict(experiment='E222', pdf=str(ROOT/'E222_c2_featmean_ridge.pdf'), time=time.time()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['extract', 'collect', 'fit', 'score', 'all'])
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--arm', choices=ARMS)
    a = ap.parse_args()
    if a.stage == 'extract':
        extract(a.smoke)
    elif a.stage == 'all':
        run_all()
    elif a.stage == 'fit':
        fit(a.arm)
    else:
        globals()[a.stage]()
