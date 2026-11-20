"""
Timestral (Mistral AI) state adapter -- E197.

Venv /scratch/danielco/tsfm2/venv_tt (torch 2.7.0+cu118, timestral from the GitHub clone
/scratch/danielco/tsfm2/timestral). Checkpoint mistralai/Timestral-209M (env TIMESTRAL_CKPT overrides).

Input per row: the label series as the single target variate (B, 1, 256) and the 22 features as
KNOWN covariates over context + horizon (B, 22, 256 + h): they are known up to D, so they are
known-future relative to the label wall. Rows are assembled by the forecaster's own _assemble_rows
(target forward-fill with mask, NaN covariates masked, left-pad to the patch), then one
model.predict_block call -- a single decode block since h <= 35 << 32 patches.

State: the decoder token of the target row at the horizon patch holding step h, i.e. the input of
model.head (QuantileHead) at token ll + (h-1)//patch_size, ll = context patches. The un-mirrored
pass only: the forecaster's symmetric averaging (a second pass on -context) averages quantiles,
which has no state analogue.
"""
import os

import numpy as np

from .common import Capture, autocast, chunks

CKPT = os.environ.get("TIMESTRAL_CKPT", "mistralai/Timestral-209M")


class Extractor:
    name = "timestral"

    def __init__(self, device="cuda:0"):
        import torch
        from timestral import TimestralForecaster
        self.torch = torch
        self.device = device
        self.fc = TimestralForecaster.from_pretrained(CKPT, device=device)
        self.model = self.fc.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.ps = self.fc.patch_size
        self.dim = self.model.config.d_model
        self.cap = Capture(self.model.head, pre=True)
        try:
            from huggingface_hub import HfApi
            rev = HfApi().model_info(CKPT).sha
        except Exception:
            rev = "unknown"
        self.checkpoint = "%s@%s" % (CKPT, rev)
        self.notes = "target=label (1 variate), known covariates=22 features over ctx+h; state=head input, target row, horizon patch"

    def states(self, y, X, F, h, last_dates):
        t = self.torch
        out = []
        hp = -(-h // self.ps)
        j = (h - 1) // self.ps
        for sl in chunks(len(y)):
            b = sl.stop - sl.start
            ctx = t.from_numpy(y[sl][:, None, :]).to(self.device)
            mask = t.isfinite(ctx)
            known = t.from_numpy(np.concatenate([X[sl], F[sl]], axis=-1)).to(self.device)
            with t.inference_mode(), autocast(t, self.device):
                rows, m, roles, kf, kfm, n_tgt = self.fc._assemble_rows(ctx, mask, h, None, known)
                v_all = rows.shape[1]
                packed = rows.reshape(b * v_all, -1)
                pm = m.reshape(b * v_all, -1)
                gids = t.arange(b, device=self.device).repeat_interleave(v_all)
                self.model.predict_block(packed, pm, gids, hp, roles, kf, kfm)
            hid = self.cap.x                                    # (R, n_dec, D)
            n = hid.shape[1]
            ll = n - hp
            tgt = t.arange(b, device=self.device) * v_all
            out.append(hid[tgt, ll + j].float().cpu().numpy())
        return np.concatenate(out)
