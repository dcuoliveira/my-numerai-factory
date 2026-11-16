"""
LoRA fine-tune of TabPFN with a per-date CROSS-SECTIONAL correlation loss (study E102).

    # on the GPU box, base env (tabpfn 8.5.0 + peft + CUDA torch):
    CUDA_VISIBLE_DEVICES=0 python -m src.analysis.tabpfn_lora --run-name e102_lora

The one fine-tune cell E44/E45/E83 never hit all three requirements of: it is
(a) RANK-ALIGNED -- a differentiable cross-sectional CORR loss (the stand-in for
numerai_corr); (b) CROSS-SECTIONAL -- each training episode is ONE date's full
cross-section of coins (context = that coin-set's history up to the T-20 leak
wall, query = the date-T cross-section), so the loss is listwise over the date,
not pointwise over random rows; and (c) PRIOR-PRESERVING -- only LoRA adapters on
the attention q/k/v/out projections train (rank r), the ~11M backbone frozen, so
the meta-learned prior that full fine-tuning corrupted (E44 -0.027 t-3.02) is kept.

Readout is TabPFN's own bar-distribution mean (what run_forecast uses at eval), so
training optimises exactly the quantity scored. After training the LoRA is MERGED
into the backbone and saved as a normal checkpoint, so eval is the ORDINARY E19
walk-forward via model_path -- E19-vs-E102 isolates the LoRA adaptation.
Leak-safe: episodes only on dates <= 2024-06-27.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.analysis.tabpfn_head import (
    EMBARGO, VENDORED, CTX_WINDOW, load_full, block_logits, episodes_for,
)
from src.analysis.tabpfn_finetune_metric import corr_loss   # 1 - pearson(pred, target)


def build_lora(device, r, alpha, dropout):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from peft import LoraConfig, get_peft_model
    reg = TabPFNRegressor.create_default_for_version(
        version=ModelVersion.V2, model_path=os.path.abspath(VENDORED),
        n_estimators=1, fit_mode="batched", differentiable_input=False,
        ignore_pretraining_limits=True,
        inference_config={"ENABLE_GPU_PREPROCESSING": False}, device=device)
    reg._initialize_model_variables()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=["q_projection", "k_projection",
                                     "v_projection", "out_projection"])
    pm = get_peft_model(reg.models_[0], cfg)
    reg.models_[0] = pm
    pm.to(device)
    return reg, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--match-run", default="tabpfn/e19_fenccausal")
    ap.add_argument("--run-name", default="e102_lora")
    ap.add_argument("--out-ckpt", default="models/tabpfn/e102_lora.ckpt")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    try:
        from tabpfn.utils import PerformanceOptions
    except Exception:
        from tabpfn.finetuning.finetuned_base import PerformanceOptions
    from tabpfn.model_loading import save_tabpfn_model

    arr, feats, gd, pos = load_full(args.data_version, args.target)
    scored = sorted(pos[d] for d in
                    pd.to_datetime(pd.read_csv("results/per_period/crypto/%s/%s.csv"
                                  % tuple(args.match_run.split("/")))["date"]).values
                    if d in pos)
    cutoff_era = min(scored) - EMBARGO
    train_dates = [e for e in range(CTX_WINDOW + EMBARGO, cutoff_era + 1)]
    print("E102 LoRA: %d feats, cutoff era %d (%s), %d train dates"
          % (len(feats), cutoff_era, str(gd[cutoff_era])[:10], len(train_dates)), flush=True)

    dev = "cuda:0" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    reg, pm = build_lora(dev, args.r, args.alpha, args.dropout)
    perf = PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False)
    params = [p for p in pm.parameters() if p.requires_grad]
    print("  LoRA trainable params: %d (%d tensors) of %d total"
          % (sum(p.numel() for p in params), len(params),
             sum(p.numel() for p in pm.parameters())), flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)

    last = None
    for epoch in range(args.epochs):
        picks = rng.choice(train_dates, size=min(args.steps_per_epoch, len(train_dates)),
                           replace=False)
        tot, nb = 0.0, 0
        for cX, cY, qX, qY, qU, T in episodes_for(arr, gd, picks, CTX_WINDOW):
            lg, batch = block_logits(reg, cX, cY, qX, perf, rng)        # [Q, L] with grad -> LoRA
            scores = batch.raw_space_bardist.mean(lg).reshape(-1)       # [Q], differentiable
            loss = corr_loss(scores, torch.as_tensor(qY, device=dev))   # listwise cross-sectional
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss); nb += 1; last = batch
            if args.limit and nb >= args.limit:
                break
        print("  epoch %d/%d  mean (1-corr) %.5f  (%d episodes)"
              % (epoch + 1, args.epochs, tot / max(nb, 1), nb), flush=True)
        if args.limit:
            break

    merged = pm.merge_and_unload()                                      # fold LoRA into the backbone
    reg.models_[0] = merged
    reg.znorm_space_bardist_ = last.znorm_space_bardist
    reg.raw_space_bardist_ = last.raw_space_bardist
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    save_tabpfn_model(reg, os.path.abspath(args.out_ckpt))
    print("saved merged LoRA checkpoint -> %s" % args.out_ckpt, flush=True)
    print("eval: run_forecast crypto tabpfn --data-version %s --run-name %s "
          "--model-options '{\"device\":\"cuda\",\"model_path\":\"%s\",\"n_estimators\":4,"
          "\"max_context\":4096}' (E19 config)" % (args.data_version, args.run_name,
          os.path.abspath(args.out_ckpt)), flush=True)


if __name__ == "__main__":
    main()
