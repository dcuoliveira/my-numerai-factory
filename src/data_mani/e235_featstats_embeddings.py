"""E235: target-row state plus mean / std / skew of the covariate-row states, one pass per TSFM.

Unchanged E212 extractor (context, 35-day wall, batching, leakage audit, cache protocol) on E206's
frozen momentum3 panel (via E225's panel folder). The only change is the read-out: from the same
forward pass keep, per origin,

    [ target row state | mean | std | skew ]   each of width d, stored as 4*d float16

where mean / std (population, ddof=0) / skew (mean of cubed z-scores; 0 where std < 1e-6) are taken
per hidden dimension over the 33 covariate (feature) rows at the same output patch / token. The
target block is kept only for parity checks; E235 models use [d:] (the three feature statistics).

Row layouts as in E222 (Chronos-2: each sample's variates consecutive, target first) and E224
(t0: B target rows first, then B*F covariate rows sample-major; Timestral: v_all consecutive rows
per sample, target first).

Environment (set by the runner): TSFM_EMBED_ROOT=<panel folder>, TSFM_EMBED_SCR=<E235 scratch>.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data_mani import e212_signal_embeddings as B

EPS = 1e-6


def readout(t, target, cov):
    """target [b, d], cov [b, F, d] -> [b, 4d] = target | mean | std | skew over the F covariate rows."""
    mean = cov.mean(1)
    std = cov.std(1, unbiased=False)
    z = (cov - mean[:, None]) / std.clamp_min(EPS)[:, None]
    skew = t.where(std > EPS, (z**3).mean(1), t.zeros_like(std))
    return t.cat([target, mean, std, skew], 1)


class ChronosFeatureStats(B.ChronosExtractor):
    def states(self, y, x, f, horizon, last_dates):
        t = self.torch
        batch, variates = len(y), 1+x.shape[1]
        context = np.concatenate([y[:, None, :], x], axis=1).reshape(-1, B.CTX)
        future = np.full((context.shape[0], horizon), np.nan, np.float32)
        for b in range(batch):
            future[b*variates+1:(b+1)*variates] = f[b]
        patches = (horizon+15)//16
        with t.inference_mode():
            enc = self.model.encode(context=t.from_numpy(context).to(self.device),
                                    group_ids=t.arange(batch, device=self.device).repeat_interleave(variates),
                                    future_covariates=t.from_numpy(np.nan_to_num(future, nan=0)).to(self.device),
                                    future_covariates_mask=t.from_numpy(np.isfinite(future).astype(np.float32)).to(self.device),
                                    num_output_patches=patches)[0]
            hidden = enc[0] if isinstance(enc, tuple) else enc.last_hidden_state
            h = hidden[:, -patches+(horizon-1)//16].float().reshape(batch, variates, -1)
            return readout(t, h[:, 0], h[:, 1:]).cpu().numpy()


def t0_states(self, y, X, F, h, last_dates):
    from src.data_mani.tsfm_state_adapters.common import autocast, chunks
    t = self.torch; out = []
    hp = -(-h // self.ps); j = (h - 1) // self.ps; nf = X.shape[1]
    for sl in chunks(len(y)):
        b = sl.stop - sl.start
        fut = np.concatenate([X[sl], F[sl]], axis=-1)
        with autocast(t, self.device):
            self.model.predict(y[sl], horizon=h, quantile_levels=[0.5], future_covariates=fut, drop_constant_covariates=False)
        hid = self.cap.x
        assert hid.shape[0] == b * (1 + nf), hid.shape
        tok = hid.shape[1] - hp - 1 + j
        out.append(readout(t, hid[:b, tok].float(), hid[b:, tok].float().reshape(b, nf, -1)).cpu().numpy())
    return np.concatenate(out)


def timestral_states(self, y, X, F, h, last_dates):
    from src.data_mani.tsfm_state_adapters.common import autocast, chunks
    t = self.torch; out = []
    hp = -(-h // self.ps); j = (h - 1) // self.ps
    for sl in chunks(len(y)):
        b = sl.stop - sl.start
        ctx = t.from_numpy(y[sl][:, None, :]).to(self.device)
        mask = t.isfinite(ctx)
        known = t.from_numpy(np.concatenate([X[sl], F[sl]], axis=-1)).to(self.device)
        with t.inference_mode(), autocast(t, self.device):
            rows, m, roles, kf, kfm, n_tgt = self.fc._assemble_rows(ctx, mask, h, None, known)
            v_all = rows.shape[1]
            assert n_tgt == 1 and v_all == 1 + X.shape[1], (n_tgt, v_all)
            gids = t.arange(b, device=self.device).repeat_interleave(v_all)
            self.model.predict_block(rows.reshape(b * v_all, -1), m.reshape(b * v_all, -1), gids, hp, roles, kf, kfm)
        hid = self.cap.x
        g = hid[:, hid.shape[1] - hp + j].float().reshape(b, v_all, -1)
        out.append(readout(t, g[:, 0], g[:, 1:]).cpu().numpy())
    return np.concatenate(out)


ORIGINAL_ADAPTER = B.adapter


def adapter(model, device):
    ext, manifest = ORIGINAL_ADAPTER(model, device)
    if model == 'chronos2':
        ext.__class__ = ChronosFeatureStats
    else:
        ext.states = {'t0': t0_states, 'timestral': timestral_states}[model].__get__(ext)
    d = int(ext.dim); ext.dim = 4*d
    manifest['width'] = 4*d; manifest['target_width'] = d
    manifest['blocks'] = ['target', 'feature_mean', 'feature_std', 'feature_skew']
    manifest['extraction'] += (f' || E235: [:{d}] target row; then mean, std (ddof=0) and skew per hidden dimension over the '
                               f'{len(manifest["features"])} covariate rows at the same output position')
    manifest['e235_extractor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['fingerprint'] = hashlib.sha256(json.dumps(
        {k: manifest[k] for k in ['panel_sha256', 'dataset', 'features', 'target', 'checkpoint', 'extraction', 'context',
                                  'label_wall_calendar_days', 'extractor_sha256', 'e235_extractor_sha256',
                                  'package_versions']}, sort_keys=True).encode()).hexdigest()
    return ext, manifest


B.adapter = adapter

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True, choices=['chronos2', 't0', 'timestral'])
    p.add_argument('--device', default='cuda:0'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    B.run(a.model, a.device, a.smoke)
