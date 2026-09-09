"""
Fine-tune the TabPFN v2 regressor on crypto fenccausal (study E44), then hand the
adapted checkpoint to the ordinary E19 walk-forward via `model_path`.

    # on the GPU box, base env (tabpfn 8.5.0 + CUDA torch):
    CUDA_VISIBLE_DEVICES=1 python -m src.analysis.tabpfn_finetune \
        --out-ckpt models/tabpfn/e44_fenccausal_ft.ckpt

E19 ran TabPFN ZERO-SHOT (frozen weights, in-context) on crypto/v2.0-fenccausal
and scored CORR 0.145 -- the strongest foundation-model arm on the board. This
FINE-TUNES the ~11M-param transformer on the crypto feature distribution, so E44
is E19 with exactly one thing changed: the weights. It is the tabular counterpart
of E41 (fine-tuned TEMPO), run on the framing that already works.

Leak wall: fine-tune only on rows whose date is on or before the eval cutoff
(min scored date - 20-period embargo = 2024-06-27), and hold out a TIME-ORDERED
tail of that pre-cutoff pool as the early-stopping validation set (an embargo gap
between train and val), passed explicitly so TabPFN's own random val split -- which
would interleave val rows among train dates -- never runs. Every context/query
episode the fine-tuner forms is drawn from the pre-cutoff pool, so no eval-window
row can enter either side.

The fine-tuned weights are saved with tabpfn.model_loading.save_tabpfn_model, which
writes the same {state_dict, config, ...} dict the vendored .ckpt uses, so
`TabPFNRegressor(model_path=<out>)` -- and therefore src/prediction/tabpfn_fm.py --
loads it unchanged. Scoring/eval is a separate run_forecast step (see the printed
command), identical to E19 but for --model-options model_path.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

EMBARGO = 20                                    # periods; the eval leak wall
MATCH_RUN = ("tabpfn", "e19_fenccausal")        # defines the scored window
VENDORED = "models/tabpfn/tabpfn-v2-regressor.ckpt"


def eval_cutoff(df, target):
    """min scored date - EMBARGO, on the global grid (= 2024-06-27)."""
    pp = pd.read_csv("results/per_period/crypto/%s/%s.csv" % MATCH_RUN)
    scored = set(pd.to_datetime(pp["date"]).values)
    gd = np.sort(df["date"].unique())
    pos = {d: i for i, d in enumerate(gd)}
    min_scored = min(d for d in scored if d in pos)
    return gd, pos, gd[pos[min_scored] - EMBARGO]


def load_xy(data_version, target, cutoff):
    df = pd.read_parquet("data/%s/train.parquet" % data_version)
    feats = [c for c in df.columns if c.startswith("feature_")]
    pre = (df.dropna(subset=[target])
             .loc[lambda d: d["date"] <= cutoff]
             .sort_values(["date", "ucid"]))
    X = np.nan_to_num(pre[feats].to_numpy(np.float32), nan=0.5)   # crypto neutral bin
    y = pre[target].to_numpy(np.float32)
    return X, y, pre["date"].values, feats


def time_split(X, y, dates, val_frac, embargo):
    """Latest val_frac of dates = validation; earlier dates minus an embargo = train."""
    ud = np.sort(np.unique(dates))
    n_val = max(1, int(len(ud) * val_frac))
    first_val = ud[-n_val]
    train_dates = ud[:max(1, len(ud) - n_val - embargo)]
    tr = np.isin(dates, train_dates)
    va = np.isin(dates, ud[-n_val:])
    return X[tr], y[tr], X[va], y[va], train_dates[-1], first_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--cutoff", default="auto")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-embargo", type=int, default=EMBARGO)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ctx", type=int, default=4096,
                    help="n_finetune_ctx_plus_query_samples (8 GB-safe)")
    ap.add_argument("--time-limit", type=float, default=None, help="seconds cap")
    ap.add_argument("--out-ckpt", default="models/tabpfn/e44_fenccausal_ft.ckpt")
    ap.add_argument("--out-dir", default="/scratch/danielco/tabpfn_ft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="subsample rows (smoke)")
    args = ap.parse_args()

    df = pd.read_parquet("data/%s/train.parquet" % args.data_version,
                         columns=["ucid", "date", args.target])
    gd, pos, auto_cut = eval_cutoff(df, args.target)
    cutoff = auto_cut if args.cutoff == "auto" else np.datetime64(args.cutoff)

    X, y, dates, feats = load_xy(args.data_version, args.target, cutoff)
    if args.limit:                                   # smoke: keep the latest rows
        X, y, dates = X[-args.limit:], y[-args.limit:], dates[-args.limit:]
    Xtr, ytr, Xva, yva, tr_last, val_first = time_split(
        X, y, dates, args.val_frac, args.val_embargo)
    print("E44 fine-tune  cutoff %s  features %d" % (cutoff, len(feats)), flush=True)
    print("  train %d rows (<= %s)   val %d rows (>= %s, embargo %d)"
          % (len(Xtr), tr_last, len(Xva), val_first, args.val_embargo), flush=True)
    assert tr_last < val_first and val_first <= cutoff, "leak in the fine-tune split"

    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNRegressor
    from tabpfn.model_loading import save_tabpfn_model

    reg = FinetunedTabPFNRegressor(
        device=args.device,
        model_version=ModelVersion.V2,               # start from OUR vendored v2, not a V3 download
        extra_regressor_kwargs={"model_path": os.path.abspath(VENDORED)},
        epochs=args.epochs, learning_rate=args.lr, weight_decay=0.01,
        n_finetune_ctx_plus_query_samples=args.ctx,
        finetune_ctx_query_split_ratio=0.2,
        n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=1,
        n_inference_subsample_samples=args.ctx,
        use_activation_checkpointing=True,
        early_stopping=True, early_stopping_patience=args.patience,
        crps_loss_weight=1.0, mse_loss_weight=1.0,
        time_limit=args.time_limit, random_state=args.seed,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    reg.fit(Xtr, ytr, X_val=Xva, y_val=yva, output_dir=Path(args.out_dir))

    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    save_tabpfn_model(reg.finetuned_estimator_, os.path.abspath(args.out_ckpt))
    print("saved fine-tuned checkpoint -> %s" % args.out_ckpt, flush=True)
    print("next: run E19's walk-forward with model_path=%s (see the E44 runner)"
          % args.out_ckpt, flush=True)


if __name__ == "__main__":
    main()
