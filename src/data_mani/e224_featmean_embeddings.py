"""E224: t0 / Timestral target-row state AND mean of the 187 covariate-row states, one pass.

E222's construction for the other two TSFMs. Unchanged E212 extractor (context,
wall, batching, audit, cache protocol) on E221's frozen panel; the only change is
the read-out: from the same forward pass keep the target row's state (identical
to E221's H) concatenated with the mean over the 187 covariate rows at the same
token. Stored width 2 x d: [:d] target, [d:] covariate mean.

Row layouts (read from the libraries): t0.TimeSeries.from_array stacks the B
target rows first, then B*F covariate rows sample-major; Timestral's
_assemble_rows gives each sample v_all consecutive rows, target first.

Environment (set by the runner): TSFM_EMBED_ROOT=<E221 root>, TSFM_EMBED_SCR=<E224 scratch>.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data_mani import e212_signal_embeddings as B
from src.data_mani.tsfm_state_adapters.common import autocast, chunks


def t0_states(self, y, X, F, h, last_dates):
    t = self.torch; out = []
    hp = -(-h // self.ps); j = (h - 1) // self.ps; nf = X.shape[1]
    for sl in chunks(len(y)):
        b = sl.stop - sl.start
        fut = np.concatenate([X[sl], F[sl]], axis=-1)
        with autocast(t, self.device):
            self.model.predict(y[sl], horizon=h, quantile_levels=[0.5], future_covariates=fut,
                               drop_constant_covariates=False)
        hid = self.cap.x
        assert hid.shape[0] == b * (1 + nf), hid.shape
        tok = hid.shape[1] - hp - 1 + j
        tgt = hid[:b, tok].float()
        cov = hid[b:, tok].float().reshape(b, nf, -1).mean(1)
        out.append(t.cat([tgt, cov], 1).cpu().numpy())
    return np.concatenate(out)


def timestral_states(self, y, X, F, h, last_dates):
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
        tok = hid.shape[1] - hp + j
        g = hid[:, tok].float().reshape(b, v_all, -1)
        out.append(t.cat([g[:, 0], g[:, 1:].mean(1)], 1).cpu().numpy())
    return np.concatenate(out)


READOUT = {'t0': t0_states, 'timestral': timestral_states}
ORIGINAL_ADAPTER = B.adapter


def adapter(model, device):
    assert model in READOUT
    ext, manifest = ORIGINAL_ADAPTER(model, device)
    ext.states = READOUT[model].__get__(ext)
    d = int(ext.dim); ext.dim = 2*d
    manifest['width'] = 2*d; manifest['target_width'] = d
    manifest['extraction'] += (f' || E224: [:{d}] target row (as E221), [{d}:] mean over the '
                               f'{len(manifest["features"])} covariate rows at the same token')
    manifest['e224_extractor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['fingerprint'] = hashlib.sha256(json.dumps(
        {k: manifest[k] for k in ['panel_sha256', 'dataset', 'features', 'target', 'checkpoint', 'extraction', 'context',
                                  'label_wall_calendar_days', 'extractor_sha256', 'e224_extractor_sha256',
                                  'package_versions']}, sort_keys=True).encode()).hexdigest()
    return ext, manifest


B.adapter = adapter

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True, choices=list(READOUT))
    p.add_argument('--device', default='cuda:0'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    B.run(a.model, a.device, a.smoke)
