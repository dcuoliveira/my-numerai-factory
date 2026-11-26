"""E222: Chronos-2 target-row state AND the mean of the 187 feature-row states, one pass.

Runs the unchanged E212 extractor (same context, wall, batching, audit, cache
protocol) on E221's frozen panel, with one difference: from the same encoder
call it keeps, per origin, the target row's pre-head state (identical to E221's
H) concatenated with the mean over the 187 covariate rows' states at the same
output patch. Stored width 2 x d_model: [:d] target, [d:] feature mean.

Environment (set by the E222 runner): TSFM_EMBED_ROOT=<E221 root> for the panel,
TSFM_EMBED_SCR=<E222 scratch> for the cache.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data_mani import e212_signal_embeddings as B


class TargetAndFeatureMean(B.ChronosExtractor):
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
            return t.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).cpu().numpy()


ORIGINAL_ADAPTER = B.adapter


def adapter(model, device):
    assert model == 'chronos2'
    ext, manifest = ORIGINAL_ADAPTER(model, device)
    ext.__class__ = TargetAndFeatureMean
    d = ext.dim
    ext.dim = 2*d
    manifest['width'] = 2*d
    manifest['target_width'] = d
    manifest['extraction'] = (manifest['extraction'] + f' || E222: [:{d}] target row (as E221), '
                              f'[{d}:] mean over the {len(manifest["features"])} covariate rows at the same output patch')
    manifest['e222_extractor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['fingerprint'] = hashlib.sha256(json.dumps(
        {k: manifest[k] for k in ['panel_sha256', 'dataset', 'features', 'target', 'checkpoint', 'extraction', 'context',
                                  'label_wall_calendar_days', 'extractor_sha256', 'e222_extractor_sha256',
                                  'package_versions']}, sort_keys=True).encode()).hexdigest()
    return ext, manifest


B.adapter = adapter

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cuda:0'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    B.run('chronos2', a.device, a.smoke)
