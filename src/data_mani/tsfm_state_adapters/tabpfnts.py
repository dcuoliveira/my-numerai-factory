"""
TabPFN-TS state adapter -- E194.

Venv /scratch/danielco/tsfm/envs/tabpfnts (tabpfn 9.2.0, tabpfn-time-series 1.4.0). Checkpoint: the
time-series fine-tune of TabPFN-3, tabpfn-v3-regressor-v3_20260506_timeseries.ckpt (sha256 pinned in
src/data_mani/tsfm_adapters/tabpfnts3.py), loaded LOCALLY, telemetry off, nothing sent to the cloud.

TabPFN-TS frames a series as tabular regression: one row per past step with a known label, the target
as y, time features as X; forecast rows are later steps with X only. Here, per panel row (ucid, D):
  train  the context steps (<= wall) whose label is resolved: X = [running step index, 22 features at
         that step], y = label;
  test   ONE row, step h after the wall (D itself): X = [its step index, D's 22 features].
So the 22 features enter exactly as covariates (known at D, known-future relative to the wall). The
pipeline's calendar/FFT-seasonal features are left out: the panel grid is irregular early on and
crypto trades every day, so they would carry grid artefacts, not seasonality.

State: TabPFNRegressor.get_embeddings(test row) -- the transformer output of the test token that the
regression head reads. n_estimators = 1 (one preprocessing configuration, random_state 0) so every row
lives in the same embedding coordinates; one TabPFN fit per panel row (no information crosses rows).
"""
import os

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from src.data_mani.tsfm_adapters.tabpfnts3 import CKPT_DEFAULT, CKPT_SHA256, _sha256


class Extractor:
    name = "tabpfnts"

    def __init__(self, device="cuda:0"):
        import torch
        from tabpfn import TabPFNRegressor
        assert _sha256(CKPT_DEFAULT) == CKPT_SHA256
        torch.manual_seed(0)
        self.reg = TabPFNRegressor(model_path=CKPT_DEFAULT, device=device, n_estimators=1, random_state=0,
                                   fit_mode="low_memory", ignore_pretraining_limits=True)
        self.checkpoint = "%s sha256=%s" % (CKPT_DEFAULT, CKPT_SHA256)
        self.dim = 512                     # TabPFN-3 test-token width (checked on every call)
        self.notes = ("per row: TabPFN fit on resolved context steps X=[step index, 22 feats], y=label; "
                      "state=get_embeddings of D's test row, n_estimators=1")

    def states(self, y, X, F, h, last_dates):
        L = y.shape[1]
        idx = np.arange(L, dtype=np.float32)
        out = []
        for k in range(len(y)):
            ok = np.isfinite(y[k])
            Xtr = np.column_stack([idx[ok], X[k][:, ok].T])
            Xte = np.concatenate([[L - 1 + h], F[k][:, -1]])[None, :].astype(np.float32)
            self.reg.fit(Xtr, y[k][ok])
            e = np.asarray(self.reg.get_embeddings(Xte, data_source="test"))
            out.append(e.reshape(-1, e.shape[-1])[0])
        r = np.stack(out).astype(np.float32)
        assert r.shape[1] == self.dim, r.shape
        return r
