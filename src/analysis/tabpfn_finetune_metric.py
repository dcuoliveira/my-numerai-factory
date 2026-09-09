"""
Fine-tune TabPFN v2 with a METRIC-ALIGNED objective (study E45): CORR or MMC,
instead of E44's library CRPS+MSE. Then the ordinary E19 walk-forward uses the
adapted checkpoint via model_path.

    # CORR objective (subclass of the high-level finetuner):
    CUDA_VISIBLE_DEVICES=1 python -m src.analysis.tabpfn_finetune_metric \
        --objective corr --out-ckpt models/tabpfn/e45_corr.ckpt

    # MMC objective (custom loop; meta aligned per query row):
    CUDA_VISIBLE_DEVICES=1 python -m src.analysis.tabpfn_finetune_metric \
        --objective mmc  --out-ckpt models/tabpfn/e45_mmc.ckpt

E44 fine-tuned with CRPS+MSE and it HURT (CORR 0.145 -> 0.118, t -3.02). Part of
the damage is objective mismatch: MSE compresses the cross-sectional spread the
ranking pays for. E45 fine-tunes toward the PAYABLE metric instead and asks
whether that recovers or beats E19. If a correct objective still loses,
prior-corruption -- not the loss -- is the cause.

The point prediction is TabPFN's bar-distribution MEAN over the query logits
(softmax @ bucket-means; differentiable to the transformer weights). Losses reuse
the repo's own differentiable stand-ins:
  CORR : -pearson(pred, target)                    (per episode)
  MMC  : -pearson(pred |_ meta, target)  on the rows where the meta model exists
Leak wall and the save->model_path round-trip are E44's, unchanged.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.tabpfn_finetune import (
    EMBARGO, VENDORED, eval_cutoff, load_xy, time_split,
)


def corr_loss(pred, target, eps=1e-8):
    """-Pearson(pred, target); pred [.,Q] or [Q], target [Q]. Negative-corr = loss."""
    import torch
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean()
    num = (p * t).sum(dim=-1)
    den = p.norm(dim=-1) * t.norm() + eps
    return (1.0 - (num / den)).mean()


def mmc_loss(pred, target, reference, mask=None, eps=1e-8):
    """-BMC over the meta model: orthogonalise pred against reference, then correlate.

    pred/target/reference are 1-D over the query rows; mask selects rows where the
    meta model is available. Matches src/prediction/nn/losses.BMCLoss.
    """
    import torch
    if mask is not None:
        pred, target, reference = pred[mask], target[mask], reference[mask]
    if pred.numel() < 3:
        return pred.sum() * 0.0                       # too few rows: no gradient
    p = pred - pred.mean()
    r = reference - reference.mean()
    resid = p - r * ((r * p).sum() / ((r * r).sum() + eps))
    t = target - target.mean()
    return 1.0 - ((resid * t).sum() / (resid.norm() * t.norm() + eps))


# --------------------------------------------------------------------------- #
# CORR objective: subclass the high-level finetuner, swap only the loss.
# --------------------------------------------------------------------------- #
def run_corr(Xtr, ytr, Xva, yva, args):
    import torch
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNRegressor
    from tabpfn.model_loading import save_tabpfn_model

    class CorrFinetuned(FinetunedTabPFNRegressor):
        def _forward_with_loss(self, batch):
            _, per_estim_logits, _ = self._training_forward(batch.X_query)
            logits_QBEL = torch.stack(per_estim_logits, dim=2)
            Q, B, E, L = logits_QBEL.shape
            logits_BQL = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L)
            preds_EQ = batch.znorm_space_bardist.mean(logits_BQL)      # [E,Q], diff'able
            y = batch.y_query.to(preds_EQ.device).reshape(-1)         # [Q]
            return corr_loss(preds_EQ, y)

    reg = CorrFinetuned(
        device=args.device, model_version=ModelVersion.V2,
        extra_regressor_kwargs={"model_path": os.path.abspath(VENDORED)},
        epochs=args.epochs, learning_rate=args.lr, weight_decay=0.01,
        n_finetune_ctx_plus_query_samples=args.ctx, finetune_ctx_query_split_ratio=0.2,
        n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=1,
        n_inference_subsample_samples=args.ctx, use_activation_checkpointing=True,
        early_stopping=False, validation_split_ratio=0.0, random_state=args.seed,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    reg.fit(Xtr, ytr, output_dir=Path(args.out_dir))
    save_tabpfn_model(reg.finetuned_estimator_, os.path.abspath(args.out_ckpt))


# --------------------------------------------------------------------------- #
# MMC objective: from-scratch loop, per-date episodes, meta aligned per row.
# --------------------------------------------------------------------------- #
def run_mmc(df_full, feats, cutoff, args):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning.data_util import (
        get_preprocessed_dataset_chunks, meta_dataset_collator)
    from tabpfn.model_loading import save_tabpfn_model
    try:
        from tabpfn.utils import PerformanceOptions
    except Exception:
        from tabpfn.finetuning.finetuned_base import PerformanceOptions

    dev = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    pre = df_full[df_full["date"] <= cutoff]
    preX = np.nan_to_num(pre[feats].to_numpy(np.float32), nan=0.5)
    prey = pre[args.target].to_numpy(np.float32)
    meta_all = pre["meta_model"].to_numpy(np.float32)
    pre_dates = pre["date"].to_numpy()
    # QUERY dates = meta-available pre-cutoff dates. The real meta model only
    # begins 2024-05-07, so leak-safe (<= 2024-06-27) meta rows span ~35 dates.
    # CONTEXT is drawn from ALL prior history (meta not needed for the context),
    # so each episode still has a rich in-context training set.
    meta_cnt = pre.groupby("date")["meta_model"].apply(lambda s: int(s.notna().sum()))
    q_dates = sorted(d for d, c in meta_cnt.items() if c >= 3)
    print("MMC fine-tune: %d meta-available query dates (%s..%s), %d history rows"
          % (len(q_dates), str(q_dates[0])[:10] if q_dates else "-",
             str(q_dates[-1])[:10] if q_dates else "-", len(preX)), flush=True)
    if not q_dates:
        raise SystemExit("no meta-available pre-cutoff dates; cannot fine-tune MMC")
    idx_by_date = {d: np.where(pre_dates == d)[0] for d in q_dates}
    rng = np.random.default_rng(args.seed)

    reg = TabPFNRegressor.create_default_for_version(
        version=ModelVersion.V2, model_path=os.path.abspath(VENDORED),
        n_estimators=1, fit_mode="batched", differentiable_input=False,
        inference_config={"ENABLE_GPU_PREPROCESSING": False}, device=str(dev))
    reg._initialize_model_variables()
    reg.model_.to(dev)
    opt = torch.optim.AdamW(reg.model_.parameters(), lr=args.lr, weight_decay=0.01)
    perf = PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False)

    last_batch = None
    for epoch in range(args.epochs):
        tot, nb = 0.0, 0
        for t in [q_dates[i] for i in rng.permutation(len(q_dates))]:
            qi = idx_by_date[t]
            qX, qy, qmeta = preX[qi], prey[qi], meta_all[qi]
            ctx_pool = np.where(pre_dates < t)[0]                 # history before t
            if len(ctx_pool) < 100:
                continue
            ci = rng.choice(ctx_pool, size=min(args.ctx, len(ctx_pool)), replace=False)
            n_ctx = len(ci)
            epX = np.vstack([preX[ci], qX]).astype(np.float32)
            epy = np.concatenate([prey[ci], qy]).astype(np.float32)
            ds = get_preprocessed_dataset_chunks(
                calling_instance=reg, X_raw=[epX], y_raw=[epy],
                split_fn=lambda X, y, *, stratify=None, n=n_ctx: (X[:n], X[n:], y[:n], y[n:]),
                max_data_size=None, model_type="regressor", equal_split_size=False,
                data_shuffle_seed=int(args.seed), preprocessing_random_state=int(args.seed),
                shuffle=False)
            batch = meta_dataset_collator([ds[0]]); last_batch = batch   # add the batch dim
            qr = batch.X_query_raw
            qr = qr[0] if qr.ndim == 3 else qr                          # -> [Q, F]
            if nb == 0 and epoch == 0:
                assert qr.shape[0] == len(qX), "query size %d vs %d" % (qr.shape[0], len(qX))
                assert np.allclose(np.asarray(qr.detach().cpu(), dtype=np.float32),
                                   qX, atol=1e-3), "query row ORDER changed by preprocessing"
            opt.zero_grad()
            reg.fit_from_preprocessed(batch.X_context, batch.y_context,
                                      batch.cat_indices, batch.configs, performance_options=perf)
            _, logits_list, _ = reg.forward(batch.X_query)
            lg = torch.stack(logits_list, dim=2)
            Q, B, E, L = lg.shape
            lg = lg.permute(1, 2, 0, 3).reshape(B * E, Q, L)
            preds = batch.znorm_space_bardist.mean(lg).mean(dim=0)          # [Q]
            y = torch.as_tensor(qy, device=dev)
            meta = torch.as_tensor(np.nan_to_num(qmeta, nan=0.0), device=dev)
            mask = torch.as_tensor(np.isfinite(qmeta), device=dev)
            loss = mmc_loss(preds, y, meta, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reg.model_.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
            if args.limit and nb >= args.limit:
                break
        print("  epoch %d/%d  mean loss %.5f  (%d episodes)"
              % (epoch + 1, args.epochs, tot / max(nb, 1), nb), flush=True)
        if args.limit:
            break
    if last_batch is None:
        raise SystemExit("no episodes ran")
    reg.znorm_space_bardist_ = last_batch.znorm_space_bardist
    reg.raw_space_bardist_ = last_batch.raw_space_bardist
    save_tabpfn_model(reg, os.path.abspath(args.out_ckpt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["corr", "mmc"], required=True)
    ap.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--cutoff", default="auto")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-embargo", type=int, default=EMBARGO)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--out-dir", default="/scratch/danielco/tabpfn_ft_metric")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap episodes/rows (smoke)")
    args = ap.parse_args()

    small = pd.read_parquet("data/%s/train.parquet" % args.data_version,
                            columns=["ucid", "date", args.target])
    gd, pos, auto_cut = eval_cutoff(small, args.target)
    cutoff = auto_cut if args.cutoff == "auto" else np.datetime64(args.cutoff)
    print("E45 %s  cutoff %s" % (args.objective, cutoff), flush=True)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    if args.objective == "corr":
        X, y, dates, feats = load_xy(args.data_version, args.target, cutoff)
        if args.limit:
            X, y, dates = X[-args.limit:], y[-args.limit:], dates[-args.limit:]
        Xtr, ytr, Xva, yva, tr_last, val_first = time_split(
            X, y, dates, args.val_frac, args.val_embargo)
        print("  train %d rows (<= %s)" % (len(Xtr), tr_last), flush=True)
        run_corr(Xtr, ytr, Xva, yva, args)
    else:
        df = pd.read_parquet("data/%s/train.parquet" % args.data_version)
        feats = [c for c in df.columns if c.startswith("feature_")]
        df = df.dropna(subset=[args.target])
        meta = pd.read_parquet("data/crypto/v2.0/historical_meta_models.parquet",
                               columns=["id", "date", "prediction"]).rename(
            columns={"id": "ucid", "prediction": "meta_model"})
        df["ucid"] = df["ucid"].astype(str); meta["ucid"] = meta["ucid"].astype(str)
        df = df.merge(meta, on=["ucid", "date"], how="left")
        run_mmc(df, feats, cutoff, args)

    print("saved fine-tuned checkpoint -> %s" % args.out_ckpt, flush=True)
    print("next: E19 walk-forward with model_path=%s" % args.out_ckpt, flush=True)


if __name__ == "__main__":
    main()
