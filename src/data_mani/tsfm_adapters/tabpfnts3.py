"""Zero-shot TabPFN-TS-3 adapter (contract: CONTRACT.md in this directory).

Model
    Prior Labs' TabPFN-TS pipeline (package ``tabpfn-time-series``) driving the finetuned
    time-series checkpoint of TabPFN-3, ``tabpfn-v3-regressor-v3_20260506_timeseries.ckpt``
    from Hugging Face ``Prior-Labs/tabpfn_3`` (revision 24a16a89d245878b846555110985634aa2e656d7).
    The checkpoint path is passed explicitly as ``model_path``; the package default
    (TabPFN-3.5 since tabpfn-time-series 1.3.0) is never used. LOCAL mode only: nothing is
    sent to the tabpfn-client cloud, and telemetry is switched off before tabpfn is imported.

    Each row is framed by the pipeline as its own tabular regression -- target on running
    index + calendar features + FFT-detected seasonal sin/cos -- and gets its own TabPFN
    fit; the pipeline's default feature set and TabPFN config are used unchanged.
    ``forecast`` returns the 0.5 quantile of TabPFN's predictive distribution.

NaN handling (native)
    The pipeline accepts missing targets itself: rows of the context whose target is NaN
    are dropped before featurisation, while every kept row keeps its TRUE date (so the
    calendar features see the real dates and the forecast dates are last_dates+1..+n_steps
    exactly). Leading NaN padding is therefore simply absent from the fit. Note the
    pipeline's running-index feature is a positional count over the KEPT rows, so interior
    gaps compress it; that is the pipeline's own behaviour and is left untouched.
    ``nan_mode="ffill"`` switches to the generic contract rule (drop leading padding,
    forward-fill interior NaN) for comparison. Rows with fewer than ``MIN_POINTS`` observed
    values return NaN (the pipeline would otherwise zero-fill a series with <= 1 value).

Independence
    Every row is a separate TabPFN fit on its own data (the pipeline loops series one by
    one), so batching rows into one call cannot move information between them. Measured on
    the smoke set: a row alone vs inside the 269-row batch, a repeat call, and 1/4/8 workers
    all agree to 0.0 (bit-identical).

Throughput (RTX 2070, deepsix, 2026-11-13)
    ~2 series/s with one process; the pipeline's own ``num_workers_per_gpu`` (independent
    processes sharing the GPU, identical outputs) lifts it to ~4.8-6.0 series/s at 4 workers
    and ~5.2-7.5 at 8 (L=256..64). Default here: 4. ``compile=True`` (the pipeline default
    on CUDA) gave no speed-up on Turing, moves forecasts by float rounding (~1e-4 abs), and
    runs out of memory as soon as the GPU is shared by >1 worker, so it is off.

Device
    Pin the GPU with CUDA_VISIBLE_DEVICES and pass device="cuda:0": the pipeline's GPU
    worker addresses GPUs by index itself, so any other index is refused here.
"""
from __future__ import annotations

import hashlib
import os

# Must be set before tabpfn / tabpfn_common_utils are imported: no usage telemetry.
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd

CKPT_DEFAULT = (
    "/scratch/danielco/tsfm/weights/tabpfnts3/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"
)
CKPT_SHA256 = "48ca82019fec74f08e15d56a157bbe728d6ec25c221f2ff2d4fd22fd4e09ec6e"
MIN_POINTS = 2
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # pipeline default; 0.5 is read


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Forecaster:
    name = "tabpfnts3"
    point = "median"
    max_context = 32768  # pipeline's TS-3 ship default for max_context_length
    max_horizon = None   # each step is a separate test row of the tabular fit
    handles_nan = True
    checkpoint = f"{CKPT_DEFAULT} sha256={CKPT_SHA256}"

    def __init__(self, device: str = "cuda:0", ckpt: str | None = None,
                 nan_mode: str = "native", compile: bool = False,
                 verify_sha: bool = True, num_workers_per_gpu: int | None = None):
        import torch
        from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

        ckpt = ckpt or os.environ.get("TABPFNTS3_CKPT", CKPT_DEFAULT)
        if verify_sha:
            got = _sha256(ckpt)
            if got != CKPT_SHA256:
                raise RuntimeError(f"checkpoint sha256 {got} != expected {CKPT_SHA256}")
        if nan_mode not in ("native", "ffill"):
            raise ValueError(nan_mode)
        self.nan_mode = nan_mode
        self.handles_nan = nan_mode == "native"
        self.checkpoint = f"{ckpt} sha256={CKPT_SHA256}"
        if num_workers_per_gpu is None:
            num_workers_per_gpu = int(os.environ.get("TABPFNTS3_WORKERS", "4"))
        if device.startswith("cuda"):
            idx = int(device.split(":")[1]) if ":" in device else 0
            if idx != 0:
                raise ValueError("pin the GPU with CUDA_VISIBLE_DEVICES and pass device='cuda:0'")
            torch.cuda.set_device(0)
        else:
            num_workers_per_gpu = 1
        torch.manual_seed(0)
        np.random.seed(0)
        self.pipe = TabPFNTSPipeline(
            max_context_length=self.max_context,
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_output_selection="median",
            tabpfn_model_config={"model_path": ckpt, "device": device},
            num_workers_per_gpu=num_workers_per_gpu,
            compile=compile,
        )
        w = getattr(self.pipe.predictor, "_worker", None)
        if w is not None and hasattr(w, "num_gpus"):  # one GPU per Forecaster, never spread
            w.num_gpus = 1
            w.total_num_workers = w.num_workers_per_gpu

    def _prepare(self, row: np.ndarray) -> np.ndarray | None:
        ok = ~np.isnan(row)
        if ok.sum() < MIN_POINTS:
            return None
        if self.nan_mode == "native":
            return row
        first = int(np.argmax(ok))
        r = pd.Series(row[first:]).ffill().to_numpy()
        return np.concatenate([np.full(first, np.nan), r])

    def forecast(self, ctx: np.ndarray, n_steps: int, last_dates: np.ndarray) -> np.ndarray:
        from tabpfn_time_series import TimeSeriesDataFrame

        ctx = np.asarray(ctx, dtype=np.float32)
        if ctx.ndim != 2:
            raise ValueError("ctx must be [n, L]")
        if self.max_context is not None and ctx.shape[1] > self.max_context:
            ctx = ctx[:, -self.max_context:]
        n, L = ctx.shape
        last_dates = np.asarray(last_dates).astype("datetime64[D]")
        out = np.full((n, n_steps), np.nan, dtype=np.float32)

        offs = np.arange(-(L - 1), 1).astype("timedelta64[D]")
        fut_offs = np.arange(1, n_steps + 1).astype("timedelta64[D]")
        c_id, c_ts, c_y, f_id, f_ts, keep = [], [], [], [], [], []
        for k in range(n):
            r = self._prepare(ctx[k].astype(np.float64))
            if r is None:
                continue
            ts = last_dates[k] + offs
            m = ~np.isnan(r)  # drop NaN rows here; the pipeline would drop them identically
            c_id.append(np.full(m.sum(), k)); c_ts.append(ts[m]); c_y.append(r[m])
            f_id.append(np.full(n_steps, k)); f_ts.append(last_dates[k] + fut_offs)
            keep.append(k)
        if not keep:
            return out
        context = TimeSeriesDataFrame.from_data_frame(pd.DataFrame({
            "item_id": np.concatenate(c_id),
            "timestamp": np.concatenate(c_ts).astype("datetime64[ns]"),
            "target": np.concatenate(c_y),
        }))
        future = TimeSeriesDataFrame.from_data_frame(pd.DataFrame({
            "item_id": np.concatenate(f_id),
            "timestamp": np.concatenate(f_ts).astype("datetime64[ns]"),
            "target": np.nan,
        }))
        pred = self.pipe.predict(context, future, quantiles=QUANTILES)
        q = pred[0.5] if 0.5 in pred.columns else pred["0.5"]
        for k in keep:
            out[k] = q.loc[k].to_numpy(dtype=np.float32)[:n_steps]
        return out
