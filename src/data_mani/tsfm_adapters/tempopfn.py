"""Zero-shot TempoPFN adapter (automl/TempoPFN, 38M checkpoint) -- see CONTRACT.md.

Environment: /scratch/danielco/tsfm/envs/tempopfn (python 3.12, torch 2.6.0+cu124, triton 3.2.0,
flash-linear-attention 0.4.0). The model code is the GitHub clone at TEMPOPFN_ROOT (default
/scratch/danielco/tsfm/TempoPFN); it is put on sys.path at construction time.

Inference recipe (mirrors the repo's own src/gift_eval/predictor.py):
- every row is one univariate series [1, S, 1]; frequency daily; start = last_date - (S - 1) days;
- future_values are a zero tensor of length n_steps -- the forward pass uses them only for their
  length (and for a `future_scaled` output we ignore), so the forecast cannot depend on them
  (verified in the smoke test: zeros vs 1e6-filled futures give identical outputs);
- fp32 throughout (autocast off): the RTX 2070 (sm_75) has no bfloat16;
- output = q0.5 of the 9-quantile head, inverse-scaled by the per-series robust scaler
  (median / IQR of the row's own finite history), so it is in the original units of ctx.

NaN handling: interior NaN are passed natively (the model swaps them for its learned
`nan_embedding`, and the scaler ignores non-finite values). Leading NaN padding is DROPPED per row
before the forward pass, so a short history is seen exactly as the model would see a short series.
Rows are then grouped by their effective length and batched only with rows of identical length;
the encoder is a per-sequence linear RNN and the scaler is per-row, so no information crosses rows
(batch invariance is checked in the smoke test). A row with no finite value returns NaN.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

TEMPOPFN_ROOT = os.environ.get("TEMPOPFN_ROOT", "/scratch/danielco/tsfm/TempoPFN")
CKPT_REPO = "AutoML-org/TempoPFN"
CKPT_FILE = "models/checkpoint_38M.pth"
CKPT_SHA256 = "a520c07e6f4dc6583b25a7129251c81eef15f168003766adf6ae4983db7b575b"
CKPT_REVISION = "f88ff0ab3231415e48e55c9dc9d4f636e7200ea5"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


GDP_BACKEND = os.environ.get("TEMPOPFN_GDP_BACKEND", "fused_recurrent")  # or "naive" (pure torch)


def _fp32_gated_delta_product(q, k, v, g, beta, initial_state=None, output_final_state=False,
                              cu_seqlens=None, num_householder=1, use_qk_l2norm_in_kernel=True,
                              scale=None):
    """fp32 drop-in for fla's chunk_gated_delta_product (whose Triton chunk kernel asserts
    dtype != float32, i.e. needs bf16, which sm_75 does not have).

    Gated DeltaProduct with n Householder steps per token == gated delta rule over the token
    sequence expanded n-fold, with the gate on the first sub-step and the query on the last one
    (exactly fla's own chunk_gated_delta_product_ref). The expanded sequence is run through fla's
    fused_recurrent_gated_delta_rule kernel, which accumulates in fp32 and accepts fp32 inputs;
    TEMPOPFN_GDP_BACKEND=naive uses fla's pure-PyTorch naive_recurrent_gated_delta_product
    instead (slow; for verification).
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    assert cu_seqlens is None
    n = num_householder
    B, T, H, K = q.shape
    if scale is None:
        scale = K ** -0.5
    if GDP_BACKEND == "naive":
        from fla.ops.gated_delta_product.naive import naive_recurrent_gated_delta_product
        if use_qk_l2norm_in_kernel:
            q, k = F.normalize(q.float(), p=2, dim=-1, eps=1e-6), F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
        h0 = initial_state.float().clone() if initial_state is not None else None
        o, ht = naive_recurrent_gated_delta_product(q * scale, k, v, g, beta, scale, None, h0,
                                                    output_final_state, n)
        return o, (ht if output_final_state else None)
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    q_new = q.new_zeros(B, T, n, H, K)
    q_new[:, :, -1] = q
    q_x = rearrange(q_new, "b t n h d -> b (t n) h d")
    g_x = None
    if g is not None:
        g_new = g.new_zeros(B, T, n, H, dtype=torch.float32)
        g_new[:, :, 0] = g
        g_x = rearrange(g_new, "b t n h -> b (t n) h")
    o, ht = fused_recurrent_gated_delta_rule(
        q=q_x.contiguous(), k=k.contiguous(), v=v.contiguous(), g=g_x, beta=beta.contiguous(),
        scale=scale, initial_state=initial_state, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    o = rearrange(o, "b (t n) h d -> b t n h d", n=n)
    return o[:, :, -1].contiguous(), ht


def _import_tempopfn(device):
    """Import TempoPFN's top-level package `src` without clobbering this repo's own `src`.

    TempoPFN uses absolute `from src.x import ...`, which collides with the numerai repo's `src`
    package when the adapter is imported as src.data_mani.tsfm_adapters.tempopfn. We hide the
    repo's `src*` modules, import TempoPFN from TEMPOPFN_ROOT, keep its modules under a
    `_tempopfn_src*` alias, and restore the originals. TempoPFN does no lazy `src` imports on the
    inference path, so the bound module objects keep working after the swap.
    """
    saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, TEMPOPFN_ROOT)
    try:
        # src.utils.utils.device is a module-level global the model uses for some buffers.
        import src.utils.utils as tutils
        tutils.device = device
        from src.data.containers import BatchTimeSeriesContainer
        from src.data.frequency import Frequency
        from src.models.model import TimeSeriesModel
        import src.models.gated_deltaproduct.gated_deltaproduct as gdp_mod
        gdp_mod.chunk_gated_delta_product = _fp32_gated_delta_product
    finally:
        sys.path.remove(TEMPOPFN_ROOT)
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules["_tempopfn_" + k] = sys.modules.pop(k)
        sys.modules.update(saved)
    return BatchTimeSeriesContainer, Frequency, TimeSeriesModel


class Forecaster:
    name = "tempopfn"
    point = "median"
    # The repo evaluates GIFT-Eval with max_context_length=3072; training series were <= 2048 long.
    max_context = 3072
    max_horizon = 1024
    handles_nan = True
    checkpoint = f"{CKPT_REPO}@{CKPT_REVISION}:{CKPT_FILE} sha256={CKPT_SHA256}"

    def __init__(self, device: str = "cuda:0", batch_size: int = 256, pad: str = "mask", verify_sha: bool = True):
        import torch
        import yaml

        from huggingface_hub import hf_hub_download

        BatchTimeSeriesContainer, Frequency, TimeSeriesModel = _import_tempopfn(torch.device(device))

        torch.manual_seed(0)
        self._torch = torch
        self._Container = BatchTimeSeriesContainer
        self._freq = Frequency.D
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        assert pad in ("mask", "nan", "trim"), pad
        self.pad = pad

        path = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILE, revision=CKPT_REVISION)
        if verify_sha and _sha256(path) != CKPT_SHA256:
            raise RuntimeError(f"checkpoint sha256 mismatch for {path}")
        self.checkpoint_path = path
        with open(os.path.join(TEMPOPFN_ROOT, "configs", "example.yaml")) as f:
            cfg = yaml.safe_load(f)["TimeSeriesModel"]
        model = TimeSeriesModel(**cfg).to(self.device)
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self._qidx = list(model.quantiles).index(0.5)

    def _run(self, hist: np.ndarray, mask, starts: list, n_steps: int) -> np.ndarray:
        torch = self._torch
        hv = torch.from_numpy(np.ascontiguousarray(hist, dtype=np.float32)).unsqueeze(-1).to(self.device)
        hm = None if mask is None else torch.from_numpy(mask.astype(np.float32)).to(self.device)
        fv = torch.zeros((hv.shape[0], n_steps, 1), dtype=torch.float32, device=self.device)
        cont = self._Container(history_values=hv, future_values=fv, start=starts,
                               frequency=[self._freq] * hv.shape[0], history_mask=hm)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            out = self.model(cont)
        pred = self.model.scaler.inverse_scale(out["result"].float(), out["scale_statistics"])
        return pred[:, :, 0, self._qidx].cpu().numpy()  # [B, P]

    def forecast(self, ctx: np.ndarray, n_steps: int, last_dates: np.ndarray) -> np.ndarray:
        ctx = np.asarray(ctx, dtype=np.float32)
        if ctx.ndim != 2:
            raise ValueError("ctx must be [n, L]")
        if n_steps > self.max_horizon:
            raise ValueError(f"n_steps {n_steps} > max_horizon {self.max_horizon}")
        if ctx.shape[1] > self.max_context:
            ctx = ctx[:, -self.max_context:]
        n, L = ctx.shape
        last = np.asarray(last_dates).astype("datetime64[D]")
        out = np.full((n, n_steps), np.nan, dtype=np.float32)
        finite = np.isfinite(ctx)
        first = np.where(finite.any(1), finite.argmax(1), L)  # index of first finite value
        ok = np.where(first < L)[0]  # an all-NaN row has nothing to condition on -> NaN

        if self.pad == "trim":  # exact per-row trimming; one shape per distinct length (slow: JIT per shape)
            groups: dict[int, list[int]] = {}
            for k in ok:
                groups.setdefault(L - int(first[k]), []).append(int(k))
            for S, rows in groups.items():
                for i in range(0, len(rows), self.batch_size):
                    idx = rows[i:i + self.batch_size]
                    starts = [np.datetime64(last[k] - np.timedelta64(S - 1, "D"), "ns") for k in idx]
                    out[idx] = self._run(ctx[idx, L - S:], None, starts, n_steps)
            return out

        # Fixed shape [batch_size, L]: the last chunk is filled up with copies of its first row, so
        # the Triton kernels (several have the row count as a constexpr) compile once per (L, n_steps).
        for i in range(0, len(ok), self.batch_size):
            idx = ok[i:i + self.batch_size]
            fill = np.concatenate([idx, np.repeat(idx[:1], self.batch_size - len(idx))])
            hist = ctx[fill]
            mask = None
            if self.pad == "mask":  # leading padding -> history_mask 0 (model's own padding path)
                mask = np.arange(L)[None, :] >= first[fill][:, None]
            starts = [np.datetime64(last[k] - np.timedelta64(L - 1, "D"), "ns") for k in fill]
            out[idx] = self._run(hist, mask, starts, n_steps)[: len(idx)]
        return out
