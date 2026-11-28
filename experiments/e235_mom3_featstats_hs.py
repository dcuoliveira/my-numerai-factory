"""E235: momentum3 TSFM rows on FEATURE-row hidden-state statistics, per-asset and pooled-panel heads.

E206 / E234 use the target row's pre-head state H. E235 replaces it with statistics of the 33
covariate (feature) rows' states from the same forward pass: per hidden dimension, the mean, std
(ddof=0) and skew over the 33 feature rows -> S = [mean | std | skew], width 3d. Extractor:
src/data_mani/e235_featstats_embeddings.py (unchanged E212 protocol, 35-day wall) on E206's frozen
momentum3 panel. Models with per-feature rows: Chronos-2, t0, Timestral (TabPFN-TS has none).

Two heads, analogous to E206 and E234, everything else as there:
  * per-asset: E206's own fit_group (dual ridge per asset, designs A/B/C, 6 standardizations,
    rolling window and alpha by validation MSE, fresh final fit) -> 54 fits;
  * pooled: E234's fit_config (one primal ridge over every asset-date row, designs A/B) -> 36 fits.
Design B's leave-one-out distribution block is held compactly in float16 (valid rows only).
Training rows = E206's common causal rows that also have S for all three models.

    python3 experiments/e235_mom3_featstats_hs.py extract --model chronos2      # GPU worker
    python3 experiments/e235_mom3_featstats_hs.py all --workers 2               # wait, collect, 90 fits
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
from scipy.stats import rankdata

os.environ['E206_SCOPE_ID'] = 'e235'
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from experiments import e234_e206_pooled_ridge as P
from src.analysis import tsfm_momentum_study as N

MODELS = ['chronos2', 't0', 'timestral']
PANEL_ROOT = REPO/'results/reports/crypto/e225_mom3_c2_featmean_norms/panel'   # symlink to E206 panel + PREPARED w/ dataset
E206 = REPO/'results/reports/crypto/e206_momentum3_tsfm'
E206_SCR = Path('/scratch/danielco/e206_momentum3_tsfm')
E225_SCR = Path('/scratch/danielco/e225_mom3_c2_featmean_norms')
ROOT = REPO/'results/reports/crypto/e235_mom3_featstats_hs'
SCR = Path('/scratch/danielco/e235_mom3_featstats_hs')
TARGET = E.TARGET
save = E.save

# Point E206's per-asset machinery and E234's pooled machinery at E235's states and outputs.
E.ROOT, E.SCR = ROOT/'per_asset', SCR
P.ROOT, P.E206_SCR, P.SCR = ROOT/'pooled', SCR, SCR


class Rows:
    """Full-panel row view over a compact float16 block holding only valid rows."""
    def __init__(self, data, pos):
        self.data, self.pos = data, pos
        self.shape = (len(pos), data.shape[1])

    def __getitem__(self, idx):
        rows, cols = (idx if isinstance(idx, tuple) else (idx, slice(None)))
        k = self.pos[rows]
        assert (np.asarray(k) >= 0).all(), 'row outside the valid design rows'
        return self.data[k][:, cols] if isinstance(idx, tuple) else self.data[k]


def distribution_compact(h, dates, valid):
    """N.distribution_design's own + LOO mean/std/p25/p50/p75 blocks (non-cross_section kinds), float16, valid rows only."""
    q = h.shape[1]; blocks = 6
    ids = np.flatnonzero(valid); pos = np.full(len(valid), -1, np.int64)
    out = np.full((len(ids), q*blocks), np.nan, np.float16); write = 0; keep = []
    for d in np.unique(dates[ids]):
        idx = ids[dates[ids] == d]
        if len(idx) < 2:
            continue
        x = np.asarray(h[idx], np.float64); n = len(x)
        mean = (x.sum(0)[None, :] - x)/(n-1)
        var = np.maximum((np.square(x).sum(0)[None, :] - x*x)/(n-1) - mean*mean, 0)
        ordered = np.sort(x, axis=0); ranks = rankdata(x, axis=0, method='ordinal').astype(int) - 1
        qs = []
        for pct in [.25, .5, .75]:
            position = (n-2)*pct; lo, hi = int(np.floor(position)), int(np.ceil(position))
            a = ordered[lo + (ranks <= lo), np.arange(q)[None, :]]; b = ordered[hi + (ranks <= hi), np.arange(q)[None, :]]
            qs.append(a + (position-lo)*(b-a))
        out[write:write+n] = np.concatenate([x, mean, np.sqrt(var)] + qs, axis=1); pos[idx] = np.arange(write, write+n)
        write += n; keep.append(idx)
    return Rows(out[:write], pos)


def distribution_compact_cs(h, dates, valid):
    """cross_section kind: own + LOO p25/p50/p75 (LOO mean/std are functions of own z, as in N)."""
    q = h.shape[1]
    ids = np.flatnonzero(valid); pos = np.full(len(valid), -1, np.int64)
    out = np.full((len(ids), q*4), np.nan, np.float16); write = 0
    for d in np.unique(dates[ids]):
        idx = ids[dates[ids] == d]
        if len(idx) < 2:
            continue
        x = np.asarray(h[idx], np.float64); n = len(x)
        ordered = np.sort(x, axis=0); ranks = rankdata(x, axis=0, method='ordinal').astype(int) - 1
        qs = []
        for pct in [.25, .5, .75]:
            position = (n-2)*pct; lo, hi = int(np.floor(position)), int(np.ceil(position))
            a = ordered[lo + (ranks <= lo), np.arange(q)[None, :]]; b = ordered[hi + (ranks <= hi), np.arange(q)[None, :]]
            qs.append(a + (position-lo)*(b-a))
        out[write:write+n] = np.concatenate([x] + qs, axis=1); pos[idx] = np.arange(write, write+n); write += n
    return Rows(out[:write], pos)


def design_B(h, dates, valid, norm):
    x = (distribution_compact_cs if norm == 'cross_section' else distribution_compact)(h, dates, valid)
    v = x.pos >= 0
    return x, valid & v


def e206_build_design(model, norm, phase, window, design, p):
    """E206.build_design with the compact design-B block (A and C unchanged)."""
    if design != 'B':
        return ORIGINAL_BUILD(model, norm, phase, window, design, p)
    h = E.normalized(model, norm, phase, window); valid = np.isfinite(h).all(axis=1)
    x, valid = design_B(h, p.date.to_numpy(), valid, norm)
    return x, valid, dict(blocks='own + LOO distribution (compact float16)')


def pooled_design(model, norm, phase, window, kind, p):
    h = P.normalized(model, norm, phase, window, p); valid = np.isfinite(h).all(axis=1)
    if kind == 'B':
        return design_B(h, p.date.to_numpy(), valid, norm)
    return h, valid


ORIGINAL_BUILD = E.build_design
E.build_design = e206_build_design
P.design = pooled_design


# ----------------------------------------------------------------- extraction
def extract(model, smoke=False):
    env = dict(os.environ, TSFM_EMBED_ROOT=str(PANEL_ROOT), TSFM_EMBED_SCR=str(SCR/'raw'), TSFM_EMBED_TARGET=TARGET,
               TSFM_EMBED_WALL='35', E206_BATCH='4')
    cmd = [E.VENVS[model], '-u', '-m', 'src.data_mani.e235_featstats_embeddings', '--model', model, '--device', 'cuda:0']
    subprocess.run(cmd + (['--smoke'] if smoke else []), cwd=REPO, check=True, env=env)


def panel():
    return pd.read_parquet(E206/'panel.parquet')


def extraction_complete(p, model):
    wanted = p.groupby('ucid').covered.any(); d = SCR/'raw'/'embeddings'/model
    return all((d/f'asset_{a}.npz').exists() for a in wanted.index[wanted])


def collect(model, p):
    d = SCR/'embeddings'/model
    if (d/'COLLECTED.json').exists():
        return
    raw = SCR/'raw'/'embeddings'/model; manifest = json.loads((raw/'extractor.json').read_text())
    w = manifest['target_width']; assert manifest['width'] == 4*w and manifest['target'] == TARGET
    d.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(d/'H_rows.partial.npy', mode='w+', dtype='float16', shape=(len(p), 3*w)); out[:] = np.nan
    ref_dir = (E225_SCR/'raw'/'embeddings'/'chronos2') if model == 'chronos2' else (E206_SCR/'embeddings'/model)
    if model == 'chronos2' and not ref_dir.exists():
        ref_dir = E225_SCR/'embeddings'/'chronos2'
    tdiff, mdiff, compared = [], [], 0
    for f in sorted(raw.glob('asset_*.npz')):
        assert json.loads(f.with_suffix('.json').read_text())['fingerprint'] == manifest['fingerprint']
        with np.load(f) as z:
            rows, st = z['rows'], z['states']
        if not len(rows):
            continue
        out[rows] = st[:, w:]
        if (ref_dir/f.name).exists():
            with np.load(ref_dir/f.name) as r:
                c, ia, ib = np.intersect1d(rows, r['rows'], return_indices=True)
                if len(c):
                    ref = r['states'][ib].astype(np.float32); compared += len(c)
                    tdiff.append(float(np.abs(ref[:, :w] - st[ia, :w].astype(np.float32)).max()))
                    if model == 'chronos2' and ref.shape[1] == 2*w:   # E225 cache: [target | feature mean]
                        mdiff.append(float(np.abs(ref[:, w:] - st[ia, w:2*w].astype(np.float32)).max()))
    out.flush(); del out; (d/'H_rows.partial.npy').replace(d/'H_rows.npy')
    valid = np.isfinite(np.load(d/'H_rows.npy', mmap_mode='r')).all(axis=1); np.save(d/'valid_rows.npy', valid)
    save(d/'COLLECTED.json', dict(width=3*w, valid_rows=int(valid.sum()), reference=str(ref_dir), rows_compared=compared,
                                  target_max_abs_diff=max(tdiff) if tdiff else None,
                                  target_median_asset_max_abs_diff=float(np.median(tdiff)) if tdiff else None,
                                  e225_feature_mean_max_abs_diff=max(mdiff) if mdiff else None, time=time.time()))


def prepare_common(p):
    common = np.load(E206_SCR/'common_rows.npy').copy()
    for m in MODELS:
        common &= np.load(SCR/'embeddings'/m/'valid_rows.npy')
    np.save(SCR/'common_rows.npy', common)
    for head in ['per_asset', 'pooled']:
        (ROOT/head).mkdir(parents=True, exist_ok=True)
        link = ROOT/head/'panel.parquet'
        if not link.exists():
            link.symlink_to(E206/'panel.parquet')
    save(ROOT/'EMBEDDINGS_READY.json', dict(common_rows=int(common.sum()), e206_common_rows=int(np.load(E206_SCR/'common_rows.npy').sum()),
                                            collected={m: json.loads((SCR/'embeddings'/m/'COLLECTED.json').read_text()) for m in MODELS}))


def job(args):
    model, norm = args
    E.fit_group(model, norm)                               # per-asset A, B, C
    for kind in ['A', 'B']:
        P.fit_config((model, norm, kind))                  # pooled A, B
    shutil.rmtree(SCR/'normalizations'/model/norm, ignore_errors=True)
    return f'{model}_{norm}'


def run_all(workers, shard=0, shards=1):
    """Shard 0 collects and prepares; every shard fits jobs[shard::shards]; the last to finish writes COMPLETE.json."""
    ROOT.mkdir(parents=True, exist_ok=True); p = panel()
    while not all(extraction_complete(p, m) for m in MODELS):
        save(ROOT/'status.json', dict(stage='waiting_for_extraction', assets={m: len(list((SCR/'raw'/'embeddings'/m).glob('asset_*.npz')))
                                                                              for m in MODELS}, time=time.time()))
        time.sleep(300)
    if shard == 0:
        for m in MODELS:
            save(ROOT/'status.json', dict(stage='collect', model=m, time=time.time())); collect(m, p)
        if not (ROOT/'EMBEDDINGS_READY.json').exists():
            prepare_common(p)
    while not (ROOT/'EMBEDDINGS_READY.json').exists():
        time.sleep(60)
    jobs = [(m, n) for m in MODELS for n in N.NORMS][shard::shards]
    status = ROOT/(f'status_shard{shard}.json' if shards > 1 else 'status.json')
    with ProcessPoolExecutor(workers) as pool:
        for i, tag in enumerate(pool.map(job, jobs), 1):
            save(status, dict(stage='fitting', shard=shard, shards=shards, done=i, expected=len(jobs), last=tag, time=time.time()))
    n_pa = len(list((ROOT/'per_asset'/'fits').glob('*/COMPLETE.json'))); n_po = len(list((ROOT/'pooled'/'fits').glob('*/COMPLETE.json')))
    if n_pa < 3*len(MODELS)*len(N.NORMS) or n_po < 2*len(MODELS)*len(N.NORMS):
        return
    save(ROOT/'COMPLETE.json', dict(experiment='E235', per_asset=len(list((ROOT/'per_asset'/'fits').glob('*/COMPLETE.json'))),
                                    pooled=len(list((ROOT/'pooled'/'fits').glob('*/COMPLETE.json'))), time=time.time()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['extract', 'all'])
    ap.add_argument('--model', choices=MODELS); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1); a = ap.parse_args()
    if a.stage == 'extract':
        extract(a.model, a.smoke)
    else:
        run_all(a.workers, a.shard, a.shards)
