"""
TabPFN + a new output layer, LoRA fine-tuned on an MMC objective for the
Timmermann-Vulicevic epoch count (study E184).

    # on deepfive, one GPU:
    CUDA_VISIBLE_DEVICES=0 python -m src.analysis.tabpfn_lora_head_mmc --run-name e184

What trains
  * LoRA adapters (rank r) on every attention q/k/v/out projection, backbone frozen
    (E102's adapter set).
  * A NEW layer at the end: Linear(emsize -> 1) on each query row's final hidden
    state, the vector TabPFN's own decoder (`output_projection`) reads. It is added
    as a residual to the bar-distribution mean, and starts at zero, so step 0 is
    exactly zero-shot TabPFN:  pred = bardist_mean(logits) + head(h).
  The new layer reads the hidden state, not the decoded distribution -- E83's head sat
  on the ~5000-bin output and stayed neutral.

Objective: MMC -- 1 - corr(pred orthogonalised on the meta model, target), on the
meta-covered rows of each date (E45's mmc_loss). One episode = one date's
cross-section (context = the 130 dates before the T-20 leak wall, subsampled to
4096; query = date T), as in E102. Training dates are the leak-safe ones
(<= E19's first scored date - 20) on which the real meta model exists; it starts
2024-05-07, so there are few of them (E45's caveat).

Epochs: E = floor(0.2 * P**0.8) + 10, P = every trainable scalar (LoRA + new layer),
early stopping off (the timmermann-mlp-compute skill). One epoch = one pass over the
training dates in a fresh random order.

Eval: per-date walk-forward on E19's scored dates at log-spaced checkpoints (and
the final epoch, the PRE-SPECIFIED result). Each checkpoint writes two forecasts
from one forward: `<run>_ep<N>` (LoRA + layer) and `<run>_lora_ep<N>` (LoRA, layer
dropped); `<run>_zs` is the matched n_estimators=1 zero-shot (adapters disabled),
written once. Intermediate checkpoints are descriptive only -- never pick on them.
Resumable: state is saved every --ckpt-every epochs.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.analysis.tabpfn_head import EMBARGO, CTX_WINDOW, MIN_CTX, load_full, block_logits
from src.analysis.tabpfn_lora import build_lora
from src.analysis.tabpfn_finetune_metric import mmc_loss

FC_MODEL = "tabpfn_head"          # results/forecast/crypto/<FC_MODEL>/<run>.csv (timesfm_score layout)


def tv_epochs(p):
    return max(int(math.floor(0.2 * p ** 0.8)), 1) + 10


def eval_epochs(total):
    pts = {e for e in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if e < total}
    return sorted(pts | {total})


def attach_meta(arr, gd, data_version):
    """Meta-model prediction per row of `arr` (NaN where absent), keyed (ucid, date)."""
    meta = pd.read_parquet("data/crypto/v2.0/historical_meta_models.parquet",
                           columns=["id", "date", "prediction"])
    meta["date"] = pd.to_datetime(meta["date"])
    key = pd.DataFrame({"ucid": arr["U"], "date": pd.to_datetime(gd[arr["E"]])})
    m = key.merge(meta.rename(columns={"id": "ucid"}).assign(ucid=lambda d: d["ucid"].astype(str)),
                  on=["ucid", "date"], how="left")
    assert len(m) == len(key)
    return m["prediction"].to_numpy(np.float32)


class Head:
    """Hook on output_projection's input + a zero-initialised Linear(emsize, 1)."""

    def __init__(self, pm, device):
        import torch
        mods = [m for n, m in pm.named_modules() if n.endswith("output_projection")]
        assert len(mods) == 1, "expected one output_projection, found %d" % len(mods)
        proj = mods[0]
        emsize = proj[0].in_features
        self.lin = torch.nn.Linear(emsize, 1).to(device)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)
        self.h = None
        proj.register_forward_pre_hook(self._grab)

    def _grab(self, _mod, inputs):
        self.h = inputs[0]                       # [Q, B, E]

    def __call__(self, nq):
        h = self.h
        assert h is not None and h.shape[0] == nq, "hidden %s vs %d query rows" % (
            None if h is None else tuple(h.shape), nq)
        return self.lin(h.float()).mean(dim=1).reshape(-1)     # mean over estimator batch


def date_rows(arr, T):
    b = T - EMBARGO
    cm = (arr["E"] >= b - CTX_WINDOW) & (arr["E"] <= b)
    qm = arr["E"] == T
    return cm, qm


def forward(reg, head, arr, T, perf, rng, dev):
    import torch
    cm, qm = date_rows(arr, T)
    qX = arr["X"][qm]
    lg, batch = block_logits(reg, arr["X"][cm], arr["Y"][cm], qX, perf, rng)
    base = batch.raw_space_bardist.mean(lg).reshape(-1)
    return base, head(len(qX)), qm


def save_state(path, pm, head, opt, epoch, rng, log):
    import torch
    from peft import get_peft_model_state_dict
    tmp = path + ".tmp"
    torch.save({"lora": get_peft_model_state_dict(pm), "head": head.lin.state_dict(),
                "opt": opt.state_dict(), "epoch": epoch,
                "rng": rng.bit_generator.state, "log": log}, tmp)
    os.replace(tmp, path)


def run_eval(reg, pm, head, arr, gd, dates, perf, seed, dev, target, tags):
    """One walk-forward pass; tags maps output column -> run name."""
    import torch
    rng = np.random.default_rng(seed)            # same context subsample at every checkpoint
    pm.eval()
    rows = []
    with torch.no_grad():
        for T in dates:
            base, add, qm = forward(reg, head, arr, T, perf, rng, dev)
            rows.append(pd.DataFrame({
                "symbol": arr["U"][qm], "date": gd[T], target: arr["Y"][qm],
                "lora": base.float().cpu().numpy(),
                "head": (base + add).float().cpu().numpy()}))
    pm.train()
    out = pd.concat(rows, ignore_index=True)
    for col, run in tags.items():
        dest = "results/forecast/crypto/%s/%s.csv" % (FC_MODEL, run)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        out[["symbol", "date", target, col]].rename(columns={col: "prediction"}).to_csv(dest, index=False)
        r = subprocess.run([sys.executable, "-m", "src.analysis.timesfm_score",
                            "--model", FC_MODEL, "--run-name", run,
                            "--data-version", "crypto/v2.0-fenccausal"],
                           capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        print("  wrote %s (%d rows); score rc=%d %s" % (dest, len(out), r.returncode,
                                                         " | ".join(tail)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--match-run", default="tabpfn/e19_fenccausal")
    ap.add_argument("--run-name", default="e184")
    ap.add_argument("--state", default="models/tabpfn/e184/state.pt")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=0, help="override TV (smoke only)")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--eval-limit", type=int, default=0, help="cap eval dates (smoke)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    try:
        from tabpfn.utils import PerformanceOptions
    except Exception:
        from tabpfn.finetuning.finetuned_base import PerformanceOptions

    torch.manual_seed(args.seed)
    arr, feats, gd, pos = load_full(args.data_version, args.target)
    arr["M"] = attach_meta(arr, gd, args.data_version)
    scored = sorted(pos[d] for d in pd.to_datetime(pd.read_csv(
        "results/per_period/crypto/%s/%s.csv" % tuple(args.match_run.split("/")))["date"]).values
        if d in pos)
    cutoff_era = min(scored) - EMBARGO
    train_dates = []
    for T in range(CTX_WINDOW + EMBARGO, cutoff_era + 1):
        cm, qm = date_rows(arr, T)
        if cm.sum() >= MIN_CTX and np.isfinite(arr["M"][qm]).sum() >= 3:
            train_dates.append(T)
    assert train_dates, "no leak-safe meta-covered dates"
    n_meta_rows = int(sum(np.isfinite(arr["M"][arr["E"] == T]).sum() for T in train_dates))
    print("E184: %d feats; leak wall %s; %d training dates %s..%s (%d meta-covered query rows)"
          % (len(feats), str(gd[cutoff_era])[:10], len(train_dates),
             str(gd[train_dates[0]])[:10], str(gd[train_dates[-1]])[:10], n_meta_rows), flush=True)

    dev = "cuda:0" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    reg, pm = build_lora(dev, args.r, args.alpha, 0.0)
    head = Head(pm, dev)
    perf = PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False)
    lora_params = [p for p in pm.parameters() if p.requires_grad]
    P = sum(p.numel() for p in lora_params) + sum(p.numel() for p in head.lin.parameters())
    E = args.epochs or tv_epochs(P)
    print("  trainable P = %d (LoRA %d + layer %d); TV epochs E = %d%s; %d steps"
          % (P, P - sum(p.numel() for p in head.lin.parameters()),
             sum(p.numel() for p in head.lin.parameters()), E,
             " (OVERRIDE)" if args.epochs else "", E * len(train_dates)), flush=True)
    opt = torch.optim.Adam([{"params": lora_params, "lr": args.lr},
                            {"params": head.lin.parameters(), "lr": args.head_lr}])
    rng = np.random.default_rng(args.seed)
    start, log = 0, []
    if os.path.exists(args.state):
        from peft import set_peft_model_state_dict
        st = torch.load(args.state, map_location=dev, weights_only=False)
        set_peft_model_state_dict(pm, st["lora"])
        head.lin.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]
        start, log = st["epoch"], st["log"]
        print("  resumed from %s at epoch %d" % (args.state, start), flush=True)
    os.makedirs(os.path.dirname(args.state), exist_ok=True)

    eval_dates = scored[:args.eval_limit] if args.eval_limit else scored
    evals = [] if args.no_eval else eval_epochs(E)
    params = lora_params + list(head.lin.parameters())
    if not args.no_eval and start == 0:
        with pm.disable_adapter():
            run_eval(reg, pm, head, arr, gd, eval_dates, perf, args.seed + 1, dev,
                     args.target, {"lora": args.run_name + "_zs"})
    pm.train()
    for epoch in range(start, E):
        t0, tot, nb = time.time(), 0.0, 0
        for T in [train_dates[i] for i in rng.permutation(len(train_dates))]:
            base, add, qm = forward(reg, head, arr, T, perf, rng, dev)
            meta = arr["M"][qm]
            mask = torch.as_tensor(np.isfinite(meta), device=dev)
            loss = mmc_loss(base + add, torch.as_tensor(arr["Y"][qm], device=dev),
                            torch.as_tensor(np.nan_to_num(meta), device=dev), mask=mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        ep = epoch + 1
        log.append({"epoch": ep, "loss": tot / max(nb, 1), "sec": time.time() - t0})
        print("  epoch %d/%d  mean (1-bmc) %.5f  %.1fs  |head w| %.2e"
              % (ep, E, log[-1]["loss"], log[-1]["sec"], float(head.lin.weight.norm())), flush=True)
        if ep % args.ckpt_every == 0 or ep == E or ep in evals:
            save_state(args.state, pm, head, opt, ep, rng, log)
        if ep in evals:
            run_eval(reg, pm, head, arr, gd, eval_dates, perf, args.seed + 1, dev, args.target,
                     {"head": "%s_ep%d" % (args.run_name, ep),
                      "lora": "%s_lora_ep%d" % (args.run_name, ep)})
    with open(os.path.join(os.path.dirname(args.state), "train_log.json"), "w") as f:
        json.dump({"P": P, "E": E, "train_dates": len(train_dates), "log": log}, f)
    print("done: final = %s_ep%d (pre-specified), matched zero-shot = %s_zs"
          % (args.run_name, E, args.run_name), flush=True)


if __name__ == "__main__":
    main()
