"""Mamba4Cast zero-shot adapter (automl/Mamba4Cast, NeurIPS 2024 TSALM workshop).

Runs in its own environment (torch 2.1.1+cu118, mamba-ssm 2.2.2, causal-conv1d 1.4.0, Python 3.10):

    M4C_HOME=/scratch/danielco/tsfm/Mamba4Cast   (code; weights in $M4C_HOME/models/)
    /scratch/danielco/tsfm/envs/mamba4cast/bin/python  with PYTHONPATH=<repo>, CUDA_VISIBLE_DEVICES=<one gpu>

Weights: ``mamba4cast_2l_1024_conv_i5e5.pth``, the single checkpoint the README links (Dropbox). The
model needs a GPU: Mamba2's selective-scan path is Triton/CUDA only.

Inference path reproduced from ``Mamba4Cast_example.ipynb`` and ``src_torch/benchmark/eval_real_dataset.py``
(``multipoint_predict``, the paper's evaluation):

  1. ``SSMModelMulti(scaler='min_max', sub_day=True, **ssm_config)`` with the notebook's ssm_config
     (2 Mamba2 layers, d_model 1024, d_state 128, dilated-conv input projection, layernorm), state dict
     passed through ``adapt_state_dict_keys`` (``linear_layer`` -> ``stage_2_layer.0``), eval mode.
  2. Time features per date, ``compute_time_features_real(subday=True)``:
     [year, month, day, day_of_week+1, day_of_year, hour=0, minute=0]; history dates are
     last_date-(m-1) .. last_date, target dates last_date+1 .. last_date+n_steps (daily).
  3. task = 0 (SINGLE_POINT) for every target step; one forward pass returns all n_steps (multipoint).
  4. Scaling is inside the model (min_max over the row's own history: (x - (min-1e-3)) /
     ((max+1e-3) - (min-1e-3))); the output is mapped back with ``scale_data(..., 'min_max')``.

Output: a single regression value per step (MSE-trained) -> point = "point".

Limits: trained on synthetic contexts of 30..323 points and horizons of 10..60
(``config.batch_ddp.yaml``); the authors' own real-data evaluation feeds up to 512 points
(``MAX_LENGTH = 512``), which is what ``max_context`` is set to. The architecture forecasts any horizon
(one appended task token per step), so ``max_horizon`` is None, but steps beyond 60 are outside
the training range.

NaN rule (the min_max scaler takes max/min over the raw history, so one NaN poisons the whole row):
leading NaN padding is dropped per row, interior NaN are forward-filled, nothing is back-filled.
Rows with fewer than ``min_context`` points (default 30 = training ``min_seq_len``) after that
return NaN, as do rows whose output is non-finite.

Rows are independent: scaling is per row, Mamba2 and the convolutions act along time only, and
there is no cross-batch interaction. The network is causal in time and has no padding mask, so
rows are batched only with rows of the SAME cleaned length (no padding is ever fed), and a row's
output is therefore the same alone or in a batch up to kernel float noise (checked <= 1e-4 rel).
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

_DEFAULT_HOME = "/scratch/danielco/tsfm/Mamba4Cast"
WEIGHTS = "mamba4cast_2l_1024_conv_i5e5.pth"
MAX_CONTEXT = 512   # eval_real_dataset.MAX_LENGTH (training max_seq_len was 323)
MIN_CONTEXT = 30    # config.batch_ddp.yaml min_seq_len

# Mamba4Cast_example.ipynb, cell 1 (identical to config.batch_ddp.yaml ssm_config)
SSM_CONFIG = {
    "bidirectional": False,
    "enc_conv": True,
    "init_dil_conv": True,
    "enc_conv_kernel": 5,
    "init_conv_kernel": 5,
    "init_conv_max_dilation": 3,
    "global_residual": False,
    "in_proj_norm": False,
    "initial_gelu_flag": True,
    "linear_seq": 15,
    "mamba2": True,
    "norm": True,
    "norm_type": "layernorm",
    "num_encoder_layers": 2,
    "d_state": 128,
    "residual": False,
    "token_embed_len": 1024,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def adapt_state_dict_keys(old_state_dict):
    """Mamba4Cast_example.ipynb, cell 2."""
    return {k.replace("linear_layer", "stage_2_layer.0") if "linear_layer" in k else k: v
            for k, v in old_state_dict.items()}


def _time_features(dates: np.ndarray) -> np.ndarray:
    """compute_time_features_real(subday=True): [year, month, day, dow+1, doy, hour, minute]."""
    ts = pd.DatetimeIndex(np.asarray(dates).astype("datetime64[ns]"))
    return np.stack([ts.year, ts.month, ts.day, ts.dayofweek + 1, ts.dayofyear, ts.hour, ts.minute],
                    axis=-1).astype(np.float32)


def _clean_row(row: np.ndarray):
    """Contract NaN rule: drop leading NaN, forward-fill interior NaN, back-fill nothing."""
    ok = np.isfinite(row)
    if not ok.any():
        return None
    first = int(np.argmax(ok))
    return pd.Series(row[first:].astype(np.float64)).ffill().to_numpy()


class Forecaster:
    name = "mamba4cast"
    point = "point"
    max_context = MAX_CONTEXT
    max_horizon = None
    handles_nan = False

    def __init__(self, device: str = "cuda:0", batch_size: int = 256, min_context: int = MIN_CONTEXT,
                 weights_path: str | None = None):
        import torch
        if not str(device).startswith("cuda"):
            raise RuntimeError("Mamba4Cast needs a CUDA device (Mamba2 kernels are Triton/CUDA only)")
        self._torch = torch
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)  # models.py builds its position tables on the 'cuda' default
        self.batch_size = int(batch_size)
        self.min_context = int(min_context)
        torch.manual_seed(0)
        np.random.seed(0)

        home = os.environ.get("M4C_HOME", _DEFAULT_HOME)
        src = os.path.join(home, "src_torch")
        if src not in sys.path:
            sys.path.insert(0, src)
        from training.models import SSMModelMulti  # noqa: E402  (Mamba4Cast's own code)

        self.weights_path = weights_path or os.path.join(home, "models", WEIGHTS)
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                f"{self.weights_path} missing. The README's Dropbox link now serves 'File Deleted' "
                "(checked 2026-11-13); place the official checkpoint there.")
        model = SSMModelMulti(scaler="min_max", sub_day=True, **SSM_CONFIG)
        self._move_tables(model)
        ckpt = torch.load(self.weights_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(adapt_state_dict_keys(sd), strict=True)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.checkpoint = f"{os.path.realpath(self.weights_path)} sha256 {_sha256(self.weights_path)}"

    def _move_tables(self, model):
        """PositionExpansion keeps its sin/cos table as a plain tensor on module-level 'cuda'; pin it
        to our device so a non-default GPU works."""
        for m in model.modules():
            if hasattr(m, "embedding") and isinstance(getattr(m, "embedding"), self._torch.Tensor) \
                    and not isinstance(m, self._torch.nn.Embedding):
                m.embedding = m.embedding.to(self.device)

    # ------------------------------------------------------------------ model call
    def _predict_group(self, hist: np.ndarray, ts: np.ndarray, tt: np.ndarray, n_steps: int) -> np.ndarray:
        """hist [b, m] float64, ts [b, m, 7], tt [b, n_steps, 7] -> [b, n_steps] in original units."""
        torch = self._torch
        out = []
        for i in range(0, len(hist), self.batch_size):
            sl = slice(i, i + self.batch_size)
            b = hist[sl].shape[0]
            x = {"ts": torch.from_numpy(ts[sl]).to(self.device),
                 "history": torch.from_numpy(hist[sl].astype(np.float32)).to(self.device),
                 "target_dates": torch.from_numpy(tt[sl]).to(self.device),
                 "task": torch.zeros(b, n_steps, dtype=torch.int32, device=self.device)}
            with torch.no_grad():
                o = self.model(x, n_steps)
            hi, lo = o["scale"][0].squeeze(-1), o["scale"][1].squeeze(-1)  # [b, 1] each
            res = o["result"].reshape(b, n_steps)
            y = res * (hi - lo) + lo                                       # scale_data(..., 'min_max')
            out.append(y.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------ contract
    def forecast(self, ctx: np.ndarray, n_steps: int, last_dates: np.ndarray) -> np.ndarray:
        ctx = np.asarray(ctx, dtype=np.float32)
        n, L = ctx.shape
        n_steps = int(n_steps)
        last_dates = np.asarray(last_dates).astype("datetime64[D]")
        out = np.full((n, n_steps), np.nan, dtype=np.float32)
        if L > self.max_context:
            ctx = ctx[:, -self.max_context:]

        groups: dict[int, list] = {}
        for k in range(n):
            v = _clean_row(ctx[k])
            if v is None or len(v) < self.min_context:
                continue
            groups.setdefault(len(v), []).append((k, v))

        steps = np.arange(1, n_steps + 1)
        for m, items in groups.items():
            rows = np.array([k for k, _ in items])
            hist = np.stack([v for _, v in items])
            ld = last_dates[rows]
            # dates are per row; cache features per distinct last date
            uniq, inv = np.unique(ld, return_inverse=True)
            ts_u = np.stack([_time_features(d - np.arange(m - 1, -1, -1)) for d in uniq])
            tt_u = np.stack([_time_features(d + steps) for d in uniq])
            pred = self._predict_group(hist, ts_u[inv], tt_u[inv], n_steps)
            pred[~np.isfinite(pred).all(axis=1)] = np.nan
            out[rows] = pred.astype(np.float32)
        return out
