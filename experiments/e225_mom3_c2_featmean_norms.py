"""E225: E222 + E223 on momentum3 -- Chronos-2 target-row state vs covariate-row-mean state, six standardizations.

Same design as E222/E223, but the inputs are crypto/v2.0-e205-ts-momentum3 (E206's frozen
panel: target20 + the 33 momentum covariates, 35-day label wall, E206 target20 splits)
instead of E221's reversal187 panel. One Chronos-2 pass with the unchanged E222 extractor
(src/data_mani/e222_featmean_embeddings.py) keeps, per origin, the target-row state T and
the mean of the 33 covariate-row states F at the same output patch. T is compared against
E206's own Chronos-2 cache (recorded, not fatal: E206 used an earlier extractor file).

Per-asset dual ridge on design C, alpha by validation MSE, fresh final fit, under all six
standardizations (rolling with window 63/126/252 by validation MSE; pooled, asset, robust,
cross_section, l2). Arms T, F, TF -> 18 fits on one common causal sample. Paired NW20
F-T, TF-T, TF-F within each standardization, and each standardization minus rolling per arm.

    python3 experiments/e225_mom3_c2_featmean_norms.py prepare
    python3 experiments/e225_mom3_c2_featmean_norms.py extract [--smoke]   # GPU worker
    python3 experiments/e225_mom3_c2_featmean_norms.py all --workers 3     # wait, collect, 18 fits, PDF, email
    python3 experiments/e225_mom3_c2_featmean_norms.py report
"""
import argparse
import json
import os
import shutil
import smtplib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from experiments import e221_reversal187_tsfm as S
from experiments import e222_c2_featmean_ridge as C
from experiments import e223_e224_featmean_norms as F
from src.analysis import tsfm_momentum_study as N

DATASET = 'crypto/v2.0-e205-ts-momentum3'
E206_ROOT = REPO/'results/reports/crypto/e206_momentum3_tsfm'
REF = Path('/scratch/danielco/e206_momentum3_tsfm/embeddings/chronos2')   # E206's target-row cache
ROOT = REPO/'results/reports/crypto/e225_mom3_c2_featmean_norms'
PANEL_ROOT = ROOT/'panel'
SCR = Path('/scratch/danielco/e225_mom3_c2_featmean_norms')
TARGET = S.TARGET
NORMS, ARMS, LABEL = F.NORMS, F.ARMS, F.LABEL
save, sha = E.save, E.sha

# Point the E222/E223 machinery at the momentum3 panel and this study's cache/outputs.
C.PANEL_ROOT, C.SCR, C.ROOT = PANEL_ROOT, SCR, ROOT
S.ROOT, S.SCR = PANEL_ROOT, SCR
F.fit_dir = lambda tsfm, norm, arm: ROOT/'fits'/f'{norm}_{arm}'
assert S.WALL == 35 and json.loads((REF/'extractor.json').read_text())['label_wall_calendar_days'] == 35


def prepare():
    """E206's frozen panel, read only, plus the `dataset` key the E212 extractor's manifest needs."""
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    link = PANEL_ROOT/'panel.parquet'
    if not link.exists():
        link.symlink_to(E206_ROOT/'panel.parquet')
    prepared = json.loads((E206_ROOT/'PREPARED.json').read_text())
    assert sha(link) == prepared['panel_hash'], 'E206 panel changed'
    prepared['dataset'] = DATASET
    prepared['e225_note'] = 'copy of E206 PREPARED.json with dataset added; panel.parquet is a symlink to E206'
    save(PANEL_ROOT/'PREPARED.json', prepared)


def collect():
    """Split the 1536-wide cache into T, F, TF; compare T with E206's cache; common mask."""
    if (ROOT/'EMBEDDINGS_READY.json').exists():
        return
    p = C.panel(); raw = C.raw_dir(); manifest = json.loads((raw/'extractor.json').read_text())
    d = manifest['target_width']; assert manifest['width'] == 2*d and manifest['target'] == TARGET
    outs = {}
    for arm, width in [('c2tgt', d), ('c2fmean', d), ('c2cat', 2*d)]:
        (SCR/'embeddings'/arm).mkdir(parents=True, exist_ok=True)
        outs[arm] = np.lib.format.open_memmap(SCR/'embeddings'/arm/'H_rows.partial.npy', mode='w+', dtype='float16', shape=(len(p), width))
        outs[arm][:] = np.nan
    diffs, compared, row_mismatch = [], 0, []
    for f in sorted(raw.glob('asset_*.npz')):
        assert json.loads(f.with_suffix('.json').read_text())['fingerprint'] == manifest['fingerprint']
        with np.load(f) as z:
            rows, st = z['rows'], z['states']
        if not len(rows):
            continue
        outs['c2tgt'][rows] = st[:, :d]; outs['c2fmean'][rows] = st[:, d:]; outs['c2cat'][rows] = st
        if (REF/f.name).exists():
            with np.load(REF/f.name) as r:
                if not np.array_equal(r['rows'], rows):
                    row_mismatch.append(f.name); continue
                diffs.append(float(np.abs(r['states'].astype(np.float32)-st[:, :d].astype(np.float32)).max())); compared += len(rows)
    for arm in outs:
        outs[arm].flush()
    del outs
    for arm in ['c2tgt', 'c2fmean', 'c2cat']:
        (SCR/'embeddings'/arm/'H_rows.partial.npy').replace(SCR/'embeddings'/arm/'H_rows.npy')
    parity = dict(rows_compared=compared, assets_compared=len(diffs), assets_row_mismatch=len(row_mismatch),
                  max_abs_diff=max(diffs) if diffs else None,
                  median_asset_max_abs_diff=float(np.median(diffs)) if diffs else None)
    common = p.covered.to_numpy().copy()
    for arm in ['c2tgt', 'c2fmean', 'c2cat']:
        valid = np.isfinite(np.load(SCR/'embeddings'/arm/'H_rows.npy', mmap_mode='r')).all(axis=1)
        np.save(SCR/'embeddings'/arm/'valid_rows.npy', valid); common &= valid
    common = N.common_history(p.ucid.to_numpy(), p.date.to_numpy(), common, 32)
    common &= pd.Series(common).groupby(p.date).transform('sum').ge(2).to_numpy()
    np.save(SCR/'common_rows.npy', common)
    save(ROOT/'EMBEDDINGS_READY.json', dict(common_rows=int(common.sum()), t_parity_vs_e206=parity,
                                            mask_sha256=sha(SCR/'common_rows.npy'), time=time.time()))


def run_all(workers):
    ROOT.mkdir(parents=True, exist_ok=True); prepare(); p = C.panel()
    while not C.extraction_complete(p):
        save(ROOT/'status.json', dict(stage='waiting_for_extraction', assets=len(list(C.raw_dir().glob('asset_*.npz'))), time=time.time()))
        time.sleep(300)
    save(ROOT/'status.json', dict(stage='collect', time=time.time())); collect()
    jobs = [('chronos2', n, a) for n in NORMS for a in ARMS]; done = 0
    with ProcessPoolExecutor(workers) as pool:
        for _ in pool.map(F.fit, jobs):
            done += 1; save(ROOT/'status.json', dict(stage='fitting', done=done, expected=len(jobs), time=time.time()))
    save(ROOT/'status.json', dict(stage='report', time=time.time()))
    pdf = report(); email_report(pdf)
    save(ROOT/'COMPLETE.json', dict(experiment='E225', fits=len(jobs), pdf=str(pdf), time=time.time()))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.analysis.html_pdf import to_pdf
    r = F.score_tsfm('chronos2'); t = r['table']
    t.to_csv(ROOT/'table.csv', index=False); r['deltas'].to_csv(ROOT/'paired_deltas.csv', index=False)
    ready = json.loads((ROOT/'EMBEDDINGS_READY.json').read_text()); par = ready['t_parity_vs_e206']
    fmt = lambda v: f'{v:.5f}' if isinstance(v, float) else v
    cols = ['norm', 'arm', 'window', 'corr', 'mmc', 'payout', 'sharpe.p', 'maxdd.p']
    best = t.sort_values('payout', ascending=False).iloc[0]
    body = ('<h1>E225: Chronos-2 covariate-row states alongside the target-row state, momentum3 inputs</h1>'
            '<p>E222 and E223 repeated on <b>crypto/v2.0-e205-ts-momentum3</b> (E206\'s panel). Chronos-2 sees each asset '
            'as one group: the target20 series (35-day label wall) plus the 33 momentum series as covariates. '
            '<b>T</b> is the target row\'s pre-head state, <b>F</b> the mean of the 33 covariate rows\' states at the same '
            'output patch from the same forward pass, <b>TF</b> both side by side. Per-asset dual ridge on design C, '
            'alpha by validation MSE, fresh final fit, E206 target20 splits, under six standardizations of H. '
            f'All 18 fits are scored on one common sample: {len(r["dates"])} test dates, {r["keys"]:,} asset/date rows. '
            'Payout = 0.1 CORR + historical-meta proxy MMC.</p>'
            '<p>Standardizations: rolling = per-asset trailing z-score (window 63/126/252 by validation MSE); pooled = one '
            'train-fitted z-score; asset = per-asset train-fitted z-score; robust = train median/IQR; cross_section = '
            'per-date z-score across assets; l2 = each row scaled to unit norm.</p>'
            f'<p>Best fit by payout: <b>{best.norm} / {best.arm}</b>, {best.payout:.5f}. '
            f'T against E206\'s Chronos-2 cache: max |diff| {par["max_abs_diff"]}, median per-asset max |diff| '
            f'{par["median_asset_max_abs_diff"]} over {par["rows_compared"]:,} rows ({par["assets_row_mismatch"]} assets with '
            'different row sets).</p>')
    body += '<h2>Fits</h2>' + t.sort_values('payout', ascending=False)[cols].to_html(index=False, float_format=fmt)
    body += '<h2>Paired differences (Newey-West lag 20)</h2>' + r['deltas'].drop(columns='tsfm').to_html(index=False, float_format=fmt)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, a in zip(axes, ARMS):
        for j, (n, aa) in enumerate(r['tags']):
            if aa == a:
                ax.plot(r['dates'], r['pay'][:, j].cumsum(), lw=2.2 if n == 'rolling' else 1.1, label=n)
        ax.set_title(LABEL[a]); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT/'cumulative_payout.png', dpi=100); plt.close(fig)
    body += '<h2>Cumulative payout by standardization</h2><img style="width:100%" src="cumulative_payout.png">'
    body += ('<h2>Limitations</h2><ul><li>Development history; ridge only; 18 fits and many comparisons, t-statistics unadjusted.</li>'
             '<li>TF shares one ridge penalty across both blocks.</li>'
             '<li>Pretrained checkpoint may overlap the test period; no live path.</li></ul>')
    page = ROOT/'E225_mom3_c2_featmean_norms.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{font-size:8.5px;border-collapse:'
                    'collapse;margin-bottom:8px}td,th{border:1px solid #ccc;padding:3px}</style></head><body>'+body+'</body></html>')
    to_pdf(str(page), 'lovelace')
    print(t.sort_values('payout', ascending=False)[cols].to_string(index=False))
    return page.with_suffix('.pdf')


def email_report(pdf):
    if (ROOT/'email_delivery.json').exists():
        return
    from src.live.env import load_env
    from src.live.notify import _config
    load_env(required=()); config = _config(); assert config
    msg = EmailMessage(); msg['From'] = config['user']; msg['To'] = config['to']
    msg['Subject'] = '[numerai-crypto] E225 momentum3 Chronos-2 T/F/TF x 6 standardizations — completed PDF'
    msg.set_content('E225 is complete: E222/E223 on crypto/v2.0-e205-ts-momentum3, 18 per-asset design-C ridge fits. PDF attached.')
    msg.add_attachment(pdf.read_bytes(), maintype='application', subtype='pdf', filename=pdf.name)
    with smtplib.SMTP(config['host'], config['port'], timeout=120) as server:
        server.starttls(); server.login(config['user'], config['password']); assert not server.send_message(msg)
    save(ROOT/'email_delivery.json', dict(smtp_accepted=True, time=time.time()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['prepare', 'extract', 'all', 'report'])
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    if a.stage == 'prepare':
        prepare()
    elif a.stage == 'extract':
        prepare(); C.extract(a.smoke)
    elif a.stage == 'all':
        run_all(a.workers)
    else:
        print(report())
