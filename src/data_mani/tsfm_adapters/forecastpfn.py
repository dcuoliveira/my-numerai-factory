"""ForecastPFN zero-shot adapter (abacusai/ForecastPFN, NeurIPS 2023).

Runs in its own environment (TensorFlow 2.9 CPU, Python 3.9):

    FPFN_HOME=/scratch/danielco/tsfm/ForecastPFN  (weights in $FPFN_HOME/saved_weights)
    /scratch/danielco/tsfm/envs/forecastpfn/bin/python  with PYTHONPATH=<repo>

Inference path reproduced from ``benchmark/exp/exp_ForecastPFN.py::_process_tuple`` (the paper's
benchmark), one row at a time but batched:

  1. history x (most recent <= 100 points). If every value is equal, ``x[-1] += 1``.
  2. StandardScaler (mean, population std) fitted on the row's own history; z = (x - mu) / sigma.
  3. local_scale = mean(z[-6:]) + std(z[-6:]) + 1e-4;  h = clip(z / local_scale, 0, 1).
  4. Left-pad h with 0 and the date features with 0 to the fixed history length 100 (year 0 is the
     model's padding mask; value 0 is also what its internal robust scaler treats as missing).
  5. Date features [year, month, day, day_of_week+1, day_of_year]; one model query per target date
     (task = 1, SINGLE_POINT), so step s is a separate batch element with target_ts = last_date + s.
  6. forecast_z = result * scale  (the SavedModel's two outputs, multiplied as the benchmark does),
     then inverse StandardScaler: forecast = forecast_z * sigma + mu.

``variant="benchmark"`` (default) is that path verbatim, including the fact that the benchmark never
multiplies the local_scale of step 3 back in. ``variant="rescaled"`` multiplies it back
(forecast_z * local_scale), as ``src/evaluate_public_datasets/evaluate.py`` does; use it as a
sensitivity check only. Either way the model's output head is a ReLU, so in standardized units the
forecast is >= 0, i.e. every forecast is >= the row's own history mean (when local_scale > 0).

Output: a single regression value per target date -> point = "point".

NaN rule (model cannot take NaN: a 0 means "missing" inside it, but NaN propagates): leading NaN
padding is dropped per row, interior NaN are forward-filled, nothing is back-filled. Rows with fewer
than MIN_CONTEXT = 20 points after that (ForecastPFN's training MIN_CONTEXT, src/training/constants.py)
return NaN, as do rows whose model output is non-finite (e.g. a history whose scaled values are all 0,
which leaves the model's internal robust scaler with nothing to measure).

Rows are independent: all scaling is per row and the network has no cross-batch interaction, so
batching is only for speed (checked: a row alone vs in a batch agrees to float32 noise).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HISTORY_LEN = 100
MIN_CONTEXT = 20
_DEFAULT_HOME = "/scratch/danielco/tsfm/ForecastPFN"


def _sha256_dir(path: str) -> str:
    """sha256 over (relative path, bytes) of every file in a SavedModel directory, sorted."""
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, path).encode())
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def _time_features(dates: np.ndarray) -> np.ndarray:
    """[year, month, day, day_of_week+1, day_of_year] as in exp_ForecastPFN._ForecastPFN_time_features."""
    ts = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    return np.stack([ts.year, ts.month, ts.day, ts.dayofweek + 1, ts.dayofyear], axis=-1).astype(np.int64)


def _clean_row(row: np.ndarray):
    """Contract NaN rule: drop leading NaN, forward-fill interior NaN. Returns (values, n_dropped)."""
    ok = np.isfinite(row)
    if not ok.any():
        return None, len(row)
    first = int(np.argmax(ok))
    v = pd.Series(row[first:].astype(np.float64)).ffill().to_numpy()
    return v, first


class Forecaster:
    name = "forecastpfn"
    point = "point"
    max_context = HISTORY_LEN
    max_horizon = None  # one query per target date; any future date can be asked for
    handles_nan = False

    def __init__(self, device: str = "cpu", variant: str = "benchmark", batch_size: int = 4096,
                 threads: int = 4, model_path: str | None = None):
        if variant not in ("benchmark", "rescaled"):
            raise ValueError(variant)
        self.variant = variant
        self.batch_size = int(batch_size)
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        if not str(device).startswith("cuda"):
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import tensorflow as tf
        import tensorflow_io  # noqa: F401  (registers ops the SavedModel may reference)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(threads)
        except RuntimeError:
            pass
        tf.random.set_seed(0)
        self._tf = tf
        home = os.environ.get("FPFN_HOME", _DEFAULT_HOME)
        self.model_path = model_path or os.path.join(home, "saved_weights")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        sha = _sha256_dir(self.model_path)
        self.checkpoint = f"{os.path.realpath(self.model_path)} (SavedModel dir sha256 {sha})"

    # ------------------------------------------------------------------ model call
    def _call(self, ts, hist, target_ts):
        tf = self._tf
        out_r, out_s = [], []
        for i in range(0, len(hist), self.batch_size):
            sl = slice(i, i + self.batch_size)
            inp = {"ts": tf.constant(ts[sl], dtype=tf.int64),
                   "history": tf.constant(hist[sl], dtype=tf.float32),
                   "target_ts": tf.constant(target_ts[sl], dtype=tf.int64),
                   "task": tf.fill([hist[sl].shape[0]], 1)}
            o = self.model(inp, training=False)
            out_r.append(np.asarray(o["result"]).reshape(-1))
            out_s.append(np.asarray(o["scale"]).reshape(-1))
        return np.concatenate(out_r), np.concatenate(out_s)

    # ------------------------------------------------------------------ contract
    def forecast(self, ctx: np.ndarray, n_steps: int, last_dates: np.ndarray) -> np.ndarray:
        ctx = np.asarray(ctx, dtype=np.float32)
        n, L = ctx.shape
        last_dates = np.asarray(last_dates).astype("datetime64[D]")
        out = np.full((n, n_steps), np.nan, dtype=np.float32)
        steps = np.arange(1, n_steps + 1)

        H, TS, TT, rows, mus, sigmas, lscales = [], [], [], [], [], [], []
        for k in range(n):
            row = ctx[k, -self.max_context:] if L > self.max_context else ctx[k]
            v, first = _clean_row(row)
            if v is None or len(v) < MIN_CONTEXT:
                continue
            m = len(v)
            x = v.copy()
            if np.all(x == x[0]):
                x[-1] += 1
            mu = x.mean()
            sigma = x.std()  # StandardScaler: population std; sklearn maps 0 -> 1
            if sigma == 0:
                sigma = 1.0
            z = (x - mu) / sigma
            local_scale = np.nanmean(z[-6:]) + np.nanstd(z[-6:]) + 1e-4
            h = np.clip(z / local_scale, 0, 1)
            hp = np.zeros(HISTORY_LEN, dtype=np.float32)
            hp[-m:] = h
            hist_dates = last_dates[k] - np.arange(m - 1, -1, -1).astype("timedelta64[D]")
            tsp = np.zeros((HISTORY_LEN, 5), dtype=np.int64)
            tsp[-m:] = _time_features(hist_dates)
            tgt = _time_features(last_dates[k] + steps.astype("timedelta64[D]"))
            H.append(np.repeat(hp[None], n_steps, 0))
            TS.append(np.repeat(tsp[None], n_steps, 0))
            TT.append(tgt[:, None, :])
            rows.append(k)
            mus.append(mu)
            sigmas.append(sigma)
            lscales.append(local_scale)
        if not rows:
            return out
        r, s = self._call(np.concatenate(TS), np.concatenate(H), np.concatenate(TT))
        zpred = (r * s).reshape(len(rows), n_steps).astype(np.float64)
        if self.variant == "rescaled":
            zpred = zpred * np.asarray(lscales)[:, None]
        pred = zpred * np.asarray(sigmas)[:, None] + np.asarray(mus)[:, None]
        out[np.asarray(rows)] = pred.astype(np.float32)
        out[~np.isfinite(out)] = np.nan
        return out
