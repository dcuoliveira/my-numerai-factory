"""
TempoPFN state adapter -- E195.

Venv /scratch/danielco/tsfm/envs/tempopfn; model code and loading reuse the zero-shot adapter
src/data_mani/tsfm_adapters/tempopfn.py (AutoML-org/TempoPFN 38M checkpoint, sha256 pinned, fp32
gated-delta-product swap for sm_75, leading padding passed through history_mask).

TempoPFN is univariate: channels of one series are processed independently (batch x channel are
merged before the encoder), so covariates cannot interact with the target. The input is the label
series alone (256 steps ending at the wall); the 22 features are NOT used -- the one deliberate
departure from E189's cov mode, recorded in the study.

State: the encoder output at the prediction position holding step h, i.e. the input of model.mlp
(before final_output_layer) at position h-1 of the prediction_length = h positions. Daily frequency
and start = wall date - 255 days, as in the zero-shot adapter (the early 3-day grid is read as daily).
"""
import numpy as np

from src.data_mani.tsfm_adapters import tempopfn as Z

from .common import Capture


class Extractor:
    name = "tempopfn"

    def __init__(self, device="cuda:0"):
        self.f = Z.Forecaster(device=device, batch_size=256, pad="mask")
        self.cap = Capture(self.f.model.mlp, pre=True)
        self.dim = int(self.f.model.embed_size)
        self.checkpoint = Z.Forecaster.checkpoint
        self.notes = "univariate label series only (TempoPFN has no cross-channel mixing); state=mlp input at horizon position h"

    def states(self, y, X, F, h, last_dates):
        L = y.shape[1]
        out = np.full((len(y), self.dim), np.nan, np.float32)
        finite = np.isfinite(y)
        first = np.where(finite.any(1), finite.argmax(1), L)
        bs = self.f.batch_size
        for i in range(0, len(y), bs):
            idx = np.arange(i, min(len(y), i + bs))
            fill = np.concatenate([idx, np.repeat(idx[:1], bs - len(idx))])
            mask = np.arange(L)[None, :] >= first[fill][:, None]
            starts = [np.datetime64(last_dates[k] - np.timedelta64(L - 1, "D"), "ns") for k in fill]
            self.f._run(y[fill], mask, starts, h)
            hid = self.cap.x                                    # (B*N, h, E), N = 1
            out[idx] = hid[:len(idx), h - 1].float().cpu().numpy()
        return out
