"""
t0 (The Forecasting Company) state adapter -- E196.

Venv /scratch/danielco/tsfm2/venv_tt (torch 2.7.0+cu118, tfc-t0 from the GitHub clone
/scratch/danielco/tsfm2/tfc-t0). Checkpoint theforecastingcompany/t0-gamma (env T0_CKPT overrides).

Input per row: the label series as the target ([B, 256]) and the 22 features as future covariates
over context + horizon ([B, 22, 256 + h]) -- t0's route for anything known through the forecast
date. drop_constant_covariates=False so every row sees the same 22 rows; no trimming, symmetric
averaging, clipping or snapping (those post-process quantiles, not states). One forward pass
(h <= 35 << max_horizon 1024).

State: t0 is a next-patch decoder; the forecast of horizon patch j is read from the token at
context_patches - 1 + j. The state is the transformer output (input of the decoder ResidualBlock)
of the target row at that token, j = (h-1)//patch_size. from_array puts the B target rows first.
"""
import os

import numpy as np

from .common import Capture, autocast, chunks

CKPT = os.environ.get("T0_CKPT", "theforecastingcompany/t0-gamma")


class Extractor:
    name = "t0"

    def __init__(self, device="cuda:0"):
        import torch
        from t0 import T0Forecaster
        self.torch = torch
        self.device = device
        self.model = T0Forecaster.from_pretrained(CKPT).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.ps = self.model.patch_size
        self.cap = Capture(self.model.decoder, pre=True)
        try:
            from huggingface_hub import HfApi
            rev = HfApi().model_info(CKPT).sha
        except Exception:
            rev = "unknown"
        self.checkpoint = "%s@%s" % (CKPT, rev)
        self.dim = int(self.model.config.embed_dim)
        self.notes = "target=label, future covariates=22 features over ctx+h; state=decoder input, target row, token ctx_patches-1+j"

    def states(self, y, X, F, h, last_dates):
        t = self.torch
        out = []
        hp = -(-h // self.ps)
        j = (h - 1) // self.ps
        for sl in chunks(len(y)):
            b = sl.stop - sl.start
            fut = np.concatenate([X[sl], F[sl]], axis=-1)
            with autocast(t, self.device):
                self.model.predict(y[sl], horizon=h, quantile_levels=[0.5], future_covariates=fut,
                                   drop_constant_covariates=False)
            hid = self.cap.x                                # [rows, patches, D]
            assert hid.shape[0] == b * (1 + X.shape[1]), hid.shape
            cp = hid.shape[1] - hp
            out.append(hid[:b, cp - 1 + j].float().cpu().numpy())
        return np.concatenate(out)
