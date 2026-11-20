"""
Toto 2.0 (Datadog) state adapter -- E193.

Venv /scratch/danielco/tsfm2/venv_toto (toto-2 from the GitHub clone /scratch/danielco/tsfm2/toto/toto2).
Checkpoint Datadog/Toto-2.0-313m (env TOTO_CKPT overrides).

Input per row: the label series as the single target variate (B, 1, 256) with its observation mask,
and the 22 features as known_dynamic covariates over context + horizon (B, 22, 256 + h), all in
one series (series_ids 0), so variate attention mixes them. The context is left-padded with NaN ->
masked. model.forecast with decode_block_size=None: one forward pass (the leaderboard setting).

State: Toto is a next-patch decoder; forecast patch j is read from x_out[..., -(block+1):-1][j],
the input of output_head. The state is that tensor at the target variate and j = (h-1)//patch_size.
Runs in fp32: under fp16 autocast every state came out NaN (checked 2026-11-20), the fp32 pass is finite.
"""
import os

import numpy as np

from .common import Capture, chunks

CKPT = os.environ.get("TOTO_CKPT", "Datadog/Toto-2.0-313m")


class Extractor:
    name = "toto"

    def __init__(self, device="cuda:0"):
        import torch
        from toto2 import Toto2Model
        self.torch = torch
        self.device = device
        self.model = Toto2Model.from_pretrained(CKPT).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.ps = self.model.config.patch_size
        self.dim = self.model.config.d_model
        self.cap = Capture(self.model.output_head, pre=True)
        try:
            from huggingface_hub import HfApi
            rev = HfApi().model_info(CKPT).sha
        except Exception:
            rev = "unknown"
        self.checkpoint = "%s@%s" % (CKPT, rev)
        self.notes = "target=label, known_dynamic=22 features over ctx+h, one series; state=output_head input, target variate, patch j"

    def states(self, y, X, F, h, last_dates):
        t = self.torch
        out = []
        j = (h - 1) // self.ps
        for sl in chunks(len(y)):
            b = sl.stop - sl.start
            tgt = t.from_numpy(y[sl][:, None, :]).to(self.device)
            m = t.isfinite(tgt)
            kd = t.from_numpy(np.concatenate([X[sl], F[sl]], axis=-1)).to(self.device)
            kdm = t.isfinite(kd)
            inputs = {"target": t.nan_to_num(tgt), "target_mask": m,
                      "series_ids": t.zeros(b, 1, dtype=t.long, device=self.device),
                      "known_dynamic": t.nan_to_num(kd), "known_dynamic_mask": kdm,
                      "known_dynamic_series_ids": t.zeros(b, kd.shape[1], dtype=t.long, device=self.device)}
            with t.inference_mode():                            # fp32: fp16 autocast overflows to NaN (u-muP scales)
                self.model.forecast(inputs, horizon=h, decode_block_size=None, has_missing_values=True)
            hid = self.cap.x                                    # (B, n_var + 22, block, D)
            out.append(hid[:, 0, j].float().cpu().numpy())
        return np.concatenate(out)
