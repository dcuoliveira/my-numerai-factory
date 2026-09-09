"""
Learning-dynamics report for the torch_mlp loss study (E42 / E43).

The board (report_html) says WHERE each loss lands; this says HOW it got there --
so a payout difference can be attributed to how the loss learns, not just its
final number. Per arm it reads the metrics json's per-refit diagnostics
(n_iter, final vs best loss, pred_std as the collapse proxy, diverged) and the
per-epoch training_dynamics curves, plus the CV train/val corr gap from the
hyperparams csv (the overfitting signal already on disk), and prints one table
per model + a mean-learning-curve sparkline per arm.

    python3 -m src.analysis.loss_dynamics --model torch_mlp_loss --runs e42_mse_raw e42_corr_raw ...
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

METRICS = "results/metrics/crypto/%s/%s.json"
HYPERPARAMS = "results/hyperparams/crypto/%s/%s.csv"
_SPARK = " ▁▂▃▄▅▆▇█"


def _loss_of(meta, run):
    """The loss name, from the recorded model_options, else the run tag."""
    opt = (meta.get("config") or {}).get("model_options") or {}
    if "loss" in opt:
        return opt["loss"]
    parts = run.split("_")
    return parts[1] if len(parts) > 2 else "?"


def _sparkline(series):
    if series is None or len(series) < 2:
        return ""
    s = np.asarray(series, dtype="float64")
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if hi - lo < 1e-12:
        return _SPARK[0] * len(s)
    idx = np.clip(((s - lo) / (hi - lo) * (len(_SPARK) - 1)).round().astype(int),
                  0, len(_SPARK) - 1)
    return "".join(_SPARK[i] for i in idx)


def _mean_curve(meta):
    """Element-wise mean of the per-refit loss curves, truncated to the shortest."""
    td = meta.get("training_dynamics") or []
    curves = [r.get("loss_curve") for r in td if r.get("loss_curve")]
    if not curves:
        return None
    n = min(len(c) for c in curves)
    return np.mean([c[:n] for c in curves], axis=0)


def _cv_gap(model, run):
    """Mean (cv_train_corr - cv_val_corr) across refits -- the overfitting signal."""
    path = HYPERPARAMS % (model, run)
    if not os.path.exists(path):
        return float("nan")
    df = pd.read_csv(path)
    if not {"cv_train_corr", "cv_val_corr"}.issubset(df.columns):
        return float("nan")
    gap = df["cv_train_corr"] - df["cv_val_corr"]
    gap = gap[np.isfinite(gap) & (df["cv_val_corr"] > -1e8)]   # drop the -1e9 sentinel
    return float(gap.mean()) if len(gap) else float("nan")


def _row(model, run):
    path = METRICS % (model, run)
    if not os.path.exists(path):
        return None
    meta = json.load(open(path))
    refits = meta.get("refits") or []

    def _mean(attr):
        vals = [r[attr] for r in refits if r.get(attr) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    corr = (meta.get("corr") or {}).get("mean", float("nan"))
    mmc = (meta.get("mmc") or {}).get("mean", float("nan"))
    diverged = any(bool(r.get("diverged")) for r in refits)
    final, best = _mean("final_loss"), _mean("best_loss")
    return {
        "run": run,
        "loss": _loss_of(meta, run),
        "corr": corr,
        "mmc": mmc,
        "payout": 0.1 * corr + mmc if np.isfinite(corr) and np.isfinite(mmc) else float("nan"),
        "epochs": _mean("n_iter"),
        "final-best": (final - best) if np.isfinite(final) and np.isfinite(best) else float("nan"),
        "pred_std": _mean("pred_std"),
        "cv_gap": _cv_gap(model, run),
        "diverged": diverged,
        "curve": _mean_curve(meta),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="torch_mlp_loss")
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    rows = [r for r in (_row(args.model, run) for run in args.runs) if r is not None]
    if not rows:
        print("no runs found for model %s" % args.model)
        return 1

    print("\nLEARNING DYNAMICS -- %s  (per loss arm)" % args.model)
    print("%-24s %-8s %8s %8s | %7s %10s %8s %7s %5s" % (
        "run", "loss", "corr", "payout",
        "epochs", "final-best", "pred_std", "cv_gap", "diver"))
    print("-" * 104)
    for r in sorted(rows, key=lambda x: (-x["payout"] if np.isfinite(x["payout"]) else 0)):
        print("%-24s %-8s %8.5f %8.5f | %7.1f %10.2e %8.4f %7.4f %5s" % (
            r["run"], r["loss"], r["corr"], r["payout"],
            r["epochs"], r["final-best"], r["pred_std"], r["cv_gap"],
            "YES" if r["diverged"] else ""))
    print("-" * 104)
    print("epochs=mean n_iter to converge | final-best=late-instability gap | "
          "pred_std=collapse proxy (small=constant) | cv_gap=train-val corr (overfit)")
    print("\nmean learning curve per arm (loss vs epoch, min->max shaded):")
    for r in sorted(rows, key=lambda x: x["loss"]):
        spark = _sparkline(r["curve"])
        n = len(r["curve"]) if r["curve"] is not None else 0
        print("  %-8s %s  (%d epochs)" % (r["loss"], spark, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
