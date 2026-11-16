"""Chronos-2-Synth (autogluon/chronos-2-synth @ the E143-E155 revision) behind the adapter contract (CONTRACT.md).

Env: /scratch/danielco/chronos2/venv, HF_HOME=/scratch/danielco/hf HF_HUB_OFFLINE=1.
Point = the q50 of the 13-level quantile head. NaN handled natively. fp32; each row is its own group.
This is the same model call as src/data_mani/c2synth_zeroshot.Model, so E154/E155 are reproducible through it.
"""
import numpy as np

CKPT = "autogluon/chronos-2-synth"
REV = "3607918a9fd027d5c465d8213e46b98e2c041cea"
CHUNK = 256


class Forecaster:
    name = "chronos2synth"
    point = "median"
    max_context = 8192
    max_horizon = 1024
    handles_nan = True
    checkpoint = "%s@%s" % (CKPT, REV)

    def __init__(self, device="cuda:0"):
        import torch
        from chronos import Chronos2Pipeline
        self.t, self.device = torch, device
        self.m = Chronos2Pipeline.from_pretrained(CKPT, revision=REV, device_map=device,
                                                  torch_dtype=torch.float32).model.float().eval()

    def forecast(self, ctx, n_steps, last_dates):
        t = self.t
        n_out = int(np.ceil(n_steps / 16))
        out = []
        for i in range(0, len(ctx), CHUNK):
            c = np.ascontiguousarray(ctx[i:i + CHUNK], "float32")
            with t.inference_mode():
                p = self.m(context=t.from_numpy(c).to(self.device), group_ids=t.arange(len(c), device=self.device),
                           num_output_patches=n_out).quantile_preds
            out.append(p[:, 6, :n_steps].float().cpu().numpy())          # quantile index 6 = q0.5
        return np.concatenate(out) if out else np.zeros((0, n_steps), "float32")
