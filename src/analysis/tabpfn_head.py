"""
Frozen TabPFN backbone + a trained 5-class head, E[y|x] readout (study E82).

The gentle adaptation E44/E45 lacked. E44 (CRPS+MSE full fine-tune) and E45 (CORR/MMC
objectives) both moved all ~11M transformer weights and HURT -- prior corruption. Here
the backbone is FROZEN (the vendored checkpoint, untouched) and only a NEW HEAD is
trained: a Linear over TabPFN's per-query bar-distribution logits -> 5 classes for the
target bins {0,.25,.5,.75,1} -> softmax -> E[y|x] = sum p_k * bin_k.

The signal is the CONTINUOUS expected value, bounded to [0,1] by construction -- NOT a
hard label. numerai_corr is rank-based, so a hard 5-level output would tie ~245
coins/era onto five values and destroy rank resolution. "Discrete" lives in the head's
structure and the cross-entropy loss, never in the emitted number.

Eval is a leak-safe per-date walk-forward (context = the 130 dates before the T-20 leak
wall, subsampled to 4096; query = the cross-section at T). ONE forward per date yields
BOTH readouts from the same logits:
  e82_head  = softmax(head(logits)) @ bins          (the trained head)
  e82_base  = raw-space bar-distribution mean         (matched n_estimators=1 zero-shot)
Pairing head vs base isolates the HEAD exactly (same context, same forward, only the
readout differs); both are also compared to E19 (the deployed n_est=4 zero-shot arm).
Head training uses the same episode shape on the pre-2024-06-27 leak-safe pool.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.analysis.tabpfn_finetune import EMBARGO, VENDORED

BINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
CTX_WINDOW = 130          # rolling context window (dates), matching E19
CTX_SAMPLE = 4096         # in-context subsample, matching E19
MIN_CTX = 256


def to_class(y):
    return np.clip(np.rint(np.asarray(y) * 4), 0, 4).astype(np.int64)


def load_full(data_version, target, include=None):
    import re
    df = pd.read_parquet("data/%s/train.parquet" % data_version)
    feats = [c for c in df.columns if c.startswith("feature_")]
    if include:
        feats = [c for c in feats if re.search(include, c)]
    df = df.dropna(subset=[target]).copy()
    df[feats] = df[feats].fillna(0.5)
    gd = np.sort(df["date"].unique())
    pos = {d: i for i, d in enumerate(gd)}
    df["era"] = df["date"].map(pos).astype(int)
    df = df.sort_values(["ucid", "era"])
    arr = dict(X=df[feats].to_numpy(np.float32), Y=df[target].to_numpy(np.float32),
               E=df["era"].to_numpy(np.int64), U=df["ucid"].astype(str).to_numpy())
    return arr, feats, gd, pos


def build_frozen(device):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    reg = TabPFNRegressor.create_default_for_version(
        version=ModelVersion.V2, model_path=os.path.abspath(VENDORED),
        n_estimators=1, fit_mode="batched", differentiable_input=False,
        ignore_pretraining_limits=True,                      # wide blocks (onehot 740)
        inference_config={"ENABLE_GPU_PREPROCESSING": False}, device=device)
    reg._initialize_model_variables()
    reg.model_.to(device)
    for p in reg.model_.parameters():
        p.requires_grad = False
    return reg


def block_logits(reg, ctxX, ctxY, qX, perf, rng):
    """Bar-distribution logits [Q, L] for the query, plus the batch (for raw bardist)."""
    import torch
    from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator
    if len(ctxX) > CTX_SAMPLE:
        keep = np.sort(rng.choice(len(ctxX), size=CTX_SAMPLE, replace=False))
        ctxX, ctxY = ctxX[keep], ctxY[keep]
    n = len(ctxX)
    epX = np.vstack([ctxX, qX]).astype(np.float32)
    epY = np.concatenate([ctxY, np.zeros(len(qX), np.float32)]).astype(np.float32)
    ds = get_preprocessed_dataset_chunks(
        calling_instance=reg, X_raw=[epX], y_raw=[epY],
        split_fn=lambda X, y, *, stratify=None, k=n: (X[:k], X[k:], y[:k], y[k:]),
        max_data_size=None, model_type="regressor", equal_split_size=False,
        data_shuffle_seed=0, preprocessing_random_state=0, shuffle=False)
    batch = meta_dataset_collator([ds[0]])
    reg.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices,
                              batch.configs, performance_options=perf)
    _, logits_list, _ = reg.forward(batch.X_query)
    lg = torch.stack(logits_list, dim=2)
    Q, B, E, L = lg.shape
    lg = lg.permute(1, 2, 0, 3).reshape(B * E, Q, L).mean(dim=0)     # [Q, L]
    return lg, batch


def episodes_for(arr, gd, dates, ctx_win):
    """Yield (ctxX, ctxY, qX, qY, date_i) for each date in `dates` with enough history."""
    for T in dates:
        b = T - EMBARGO
        cm = (arr["E"] >= b - ctx_win) & (arr["E"] <= b)
        qm = arr["E"] == T
        if cm.sum() < MIN_CTX or qm.sum() < 3:
            continue
        yield arr["X"][cm], arr["Y"][cm], arr["X"][qm], arr["Y"][qm], arr["U"][qm], T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    ap.add_argument("--feature-include", default=None, help="regex to select a feature block")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--match-run", default="tabpfn/e19_fenccausal")
    ap.add_argument("--run-name", default="e82_head")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap eval dates (smoke)")
    args = ap.parse_args()

    import torch
    try:
        from tabpfn.utils import PerformanceOptions
    except Exception:
        from tabpfn.finetuning.finetuned_base import PerformanceOptions

    arr, feats, gd, pos = load_full(args.data_version, args.target, args.feature_include)
    scored = sorted(pos[d] for d in
                    pd.to_datetime(pd.read_csv("results/per_period/crypto/%s/%s.csv"
                                  % tuple(args.match_run.split("/")))["date"]).values
                    if d in pos)
    cutoff_era = min(scored) - EMBARGO                       # leak wall: 2024-06-27
    cutoff = gd[cutoff_era]
    train_dates = [e for e in range(CTX_WINDOW + EMBARGO, cutoff_era + 1)]
    print("E82 head: %d feats, cutoff era %d (%s), %d scored eval eras, %d train eras"
          % (len(feats), cutoff_era, str(cutoff)[:10], len(scored), len(train_dates)), flush=True)

    dev = "cuda:0" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    reg = build_frozen(dev)
    perf = PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False)
    rng = np.random.default_rng(args.seed)

    # ---- train the head ----
    head, opt = None, None
    ce = torch.nn.CrossEntropyLoss()
    bins_t = torch.tensor(BINS, device=dev)
    for epoch in range(args.epochs):
        picks = rng.choice(train_dates, size=min(args.steps_per_epoch, len(train_dates)),
                           replace=False)
        tot, nb = 0.0, 0
        for cX, cY, qX, qY, _, T in episodes_for(arr, gd, picks, CTX_WINDOW):
            lg, _ = block_logits(reg, cX, cY, qX, perf, rng)          # [Q, L]
            if head is None:
                L = lg.shape[1]
                head = torch.nn.Linear(L, 5).to(dev)
                opt = torch.optim.Adam(head.parameters(), lr=args.lr)
                print("  head Linear(%d, 5), %d params" % (L, sum(p.numel() for p in head.parameters())), flush=True)
            opt.zero_grad()
            probs = torch.softmax(lg, dim=-1).detach()                # distribution over the ~5000 bins
            logit5 = head(probs)                                      # [Q, 5]
            loss = ce(logit5, torch.tensor(to_class(qY), device=dev))
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print("  epoch %d/%d  mean CE %.4f  (%d episodes)" % (epoch + 1, args.epochs, tot / max(nb, 1), nb), flush=True)

    # ---- eval: per-date walk-forward, both readouts ----
    head.eval()
    eval_dates = scored[:args.limit] if args.limit else scored
    rows = []
    with torch.no_grad():
        for i, (cX, cY, qX, qY, qU, T) in enumerate(episodes_for(arr, gd, eval_dates, CTX_WINDOW)):
            lg, batch = block_logits(reg, cX, cY, qX, perf, rng)
            probs = torch.softmax(lg, dim=-1)                         # over the ~5000 bins
            head_pred = (torch.softmax(head(probs), dim=-1) @ bins_t).cpu().numpy()
            base_pred = np.asarray(batch.raw_space_bardist.mean(lg).cpu()).ravel()
            rows.append(pd.DataFrame({"symbol": qU, "date": gd[T], args.target: qY,
                                      "head": head_pred, "base": base_pred}))
            if (i + 1) % 50 == 0:
                print("  eval %d/%d (%s)" % (i + 1, len(eval_dates), str(gd[T])[:10]), flush=True)

    out = pd.concat(rows, ignore_index=True)
    for tag, col in (("", "head"), ("_base", "base")):
        f = out[["symbol", "date", args.target, col]].rename(columns={col: "prediction"})
        run = args.run_name + tag
        dest = "results/forecast/crypto/tabpfn_head/%s.csv" % run
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        f.to_csv(dest, index=False)
        print("wrote %s  (%d rows)" % (dest, len(f)), flush=True)
    hp = "models/tabpfn/%s.pt" % args.run_name
    os.makedirs(os.path.dirname(hp), exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "L": head.in_features}, hp)
    print("saved head %s\nscore: python -m src.analysis.timesfm_score --model tabpfn_head --run-name %s [and %s_base]"
          % (hp, args.run_name, args.run_name), flush=True)


if __name__ == "__main__":
    main()
