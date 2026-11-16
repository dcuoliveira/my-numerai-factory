"""
Stitch a diagnosed walk-forward back into one pseudo-out-of-sample run.

    python -m src.analysis.assemble_walkforward --base torchmlp_default_noreg_500ep_s42_full

src.prediction.diagnose trains ONE refit per call. Run with --scheme full
--save-preds over every refit (tags <base>_r00, <base>_r01, ...), it leaves each
refit's train and held-out predictions at every epoch on disk. This puts them back
together:

  1. the held-out blocks of all refits at one epoch (default: the last), written as
     an ordinary run -- results/{forecast,per_period,metrics}/crypto/<tree>/<run>
     -- so results_table, report_html and paired_delta read it like any other;
  2. the whole walk-forward scored at many epochs: full sample and the standard
     510-day subset (the refits the shared protocol also has), next to the refits'
     mean in-sample (train) CORR and losses from their log.json -- how training
     length moves the pseudo-OOS result. Written to <base>_walkforward/by_epoch.csv.

Scoring is the repo's own: per-day numerai CORR, MMC against the real meta model,
payout = 0.1*CORR + MMC per period, summarised by results_table.row_from_frame.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

ROOT = os.path.join("results", "diag", "crypto")
TARGET = "target_binned_return_20"


def refit_dirs(base):
    dirs = sorted(glob.glob(os.path.join(ROOT, base + "_r[0-9][0-9]")))
    if not dirs:
        raise SystemExit("no refits found for %s_rNN under %s" % (base, ROOT))
    return dirs


def load_refits(base):
    """Per refit: config, log, held-out row index and the per-epoch prediction array."""
    out = []
    for i, d in enumerate(refit_dirs(base)):
        cfg = json.load(open(os.path.join(d, "config.json")))
        log = json.load(open(os.path.join(d, "log.json")))
        rows = pd.read_parquet(os.path.join(d, "val_rows.parquet"))
        preds = np.load(os.path.join(d, "pred_val.npy"), mmap_mode="r")
        out.append({"dir": d, "refit": i, "config": cfg, "log": log, "rows": rows, "preds": preds,
                    "standard": cfg.get("standard_refit") is not None,
                    "last": int(log["snapshots"][-1]["epoch"])})
    return out


def frame_at(refits, epoch):
    """The stitched held-out frame at one epoch, in the forecast layout run_forecast writes."""
    parts = []
    for r in refits:
        p = np.asarray(r["preds"][epoch])
        if np.isnan(p).all():
            raise SystemExit("%s has no epoch %d" % (r["dir"], epoch))
        f = r["rows"].rename(columns={"target": TARGET}).copy()
        f.insert(3, "prediction", p)
        f["refit"] = r["refit"]
        f["standard"] = r["standard"]
        parts.append(f)
    frame = pd.concat(parts, ignore_index=True)
    return frame[["symbol", "date", TARGET, "prediction", "refit", "reference_prediction",
                  "meta_model", "standard"]]


def _score_epoch(task):
    """CORR and MMC per period for one epoch (worker; corr+mmc only, the curve's needs)."""
    base, epoch = task
    from src.prediction.metrics import per_period_corr, per_period_bmc
    frame = frame_at(load_refits(base), epoch)
    table = pd.DataFrame({
        "corr": per_period_corr(frame, "prediction", TARGET, "date"),
        "mmc": per_period_bmc(frame, "prediction", "meta_model", TARGET, "date")})
    standard_days = set(frame.loc[frame["standard"], "date"])
    return epoch, table, standard_days


def _train_side(refits, epoch):
    """Mean over refits of what each refit's log.json measured on its own window."""
    vals = {"train_corr": [], "held_corr": [], "train_data_loss": [], "held_data_loss": []}
    for r in refits:
        snap = next((s for s in r["log"]["snapshots"] if s["epoch"] == epoch), None)
        if snap is None:
            continue
        m = snap.get("metric") or {}
        vals["train_corr"].append(m.get("train_corr", np.nan))
        vals["held_corr"].append(m.get("held_corr", np.nan))
        vals["train_data_loss"].append(snap["train"]["data_loss"])
        vals["held_data_loss"].append(snap["held"]["data_loss"])
    return {k: float(np.nanmean(v)) if v else np.nan for k, v in vals.items()}


def write_run(refits, epoch, tree, run_name):
    """The stitched held-out predictions at `epoch`, as an ordinary results/ run."""
    from src.prediction.metrics import per_period_table, summarize
    frame = frame_at(refits, epoch)
    table = per_period_table(frame, "prediction", TARGET, "date",
                             reference_col="reference_prediction", meta_model_col="meta_model")
    results = {"n_periods": int(table["corr"].notna().sum())}
    for family in ("corr", "bmc", "mmc"):
        results[family] = summarize(table[family])
        if family != "corr":
            results["%s_n_periods" % family] = int(table[family].notna().sum())
    cfg = dict(refits[0]["config"])
    for key in ("refit", "standard_refit", "train_start", "train_end", "held_start", "held_end",
                "train_days", "held_days", "tag"):
        cfg.pop(key, None)
    cfg.update({"model": tree, "tournament": "crypto", "task": "regression",
                "cv_scheme": "rolling", "walk_forward": "full sample, %d refits" % len(refits),
                "epoch": int(epoch), "refits_dirs": [r["dir"] for r in refits],
                "standard_refits": [r["refit"] for r in refits if r["standard"]],
                "assembled_by": "src.analysis.assemble_walkforward"})
    results.update({"config": cfg, "refits": [
        {"refit": r["refit"], "held_start": r["config"]["held_start"],
         "held_end": r["config"]["held_end"], "standard_refit": r["config"].get("standard_refit"),
         "epochs_trained": r["last"]} for r in refits]})
    paths = {}
    for kind in ("forecast", "per_period", "metrics"):
        directory = os.path.join("results", kind, "crypto", tree)
        os.makedirs(directory, exist_ok=True)
        paths[kind] = os.path.join(directory, run_name + (".json" if kind == "metrics" else ".csv"))
    frame.drop(columns=["standard"]).to_csv(paths["forecast"], index=False)
    table.to_csv(paths["per_period"])
    with open(paths["metrics"], "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.analysis.assemble_walkforward",
                                description=__doc__.split("\n")[1])
    p.add_argument("--base", required=True, help="the refits are results/diag/crypto/<base>_rNN")
    p.add_argument("--epoch", default="last", help="epoch for the results/ run (default: last)")
    p.add_argument("--curve", default="0,1,2,3,5,10,15,20,30,50,75,100,150,200,300,400,500",
                   help="epochs to score the whole walk-forward at")
    p.add_argument("--tree", default="torch_scratch", help="model tree under results/")
    p.add_argument("--run-name", default=None, help="default: <base>_ep<EEE>")
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args(argv)
    from src.analysis import results_table as rt

    refits = load_refits(args.base)
    last = min(r["last"] for r in refits)
    epoch = last if args.epoch == "last" else int(args.epoch)
    print("%s: %d refits (%d standard), every refit trained to epoch >= %d" % (
        args.base, len(refits), sum(r["standard"] for r in refits), last))
    for r in refits:
        print("  r%02d %s..%s %3d days%s" % (r["refit"], r["config"]["held_start"],
                                            r["config"]["held_end"], r["config"]["held_days"],
                                            "  [standard %d]" % r["config"]["standard_refit"]
                                            if r["standard"] else ""))

    name = args.run_name or "%s_ep%03d" % (args.base, epoch)
    paths = write_run(refits, epoch, args.tree, name)
    print("\nwrote the epoch-%d pseudo-OOS run: %s" % (epoch, paths["metrics"]))

    epochs = [e for e in (int(x) for x in args.curve.split(",")) if e <= last]
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        scored = sorted(pool.map(_score_epoch, [(args.base, e) for e in epochs]))
    lines, rows = [], []
    head = ("%5s | %26s | %26s | %10s %10s | %9s %9s" % (
        "epoch", "full sample CORR MMC calmar", "standard-510 CORR MMC cal.", "train CORR",
        "held CORR", "train dl", "held dl"))
    lines.append(head)
    for e, table, std_days in scored:
        full = rt.row_from_frame("full", table, target=TARGET)
        std = rt.row_from_frame("std", table.loc[table.index.isin(std_days)], target=TARGET)
        side = _train_side(refits, e)
        rows.append({"epoch": e, **{"full_%s" % k: full[k] for k in
                                    ("n", "CORR", "MMC", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)")},
                     "full_n_corr": int(table["corr"].notna().sum()),
                     **{"std_%s" % k: std[k] for k in
                        ("n", "CORR", "MMC", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)")},
                     **{"refits_mean_%s" % k: v for k, v in side.items()}})
        lines.append("%5d | %+.4f %+.4f %6.3f (n%4d) | %+.4f %+.4f %6.3f (n%3d) | %+10.4f %+10.4f | %.5f %.5f" % (
            e, full["CORR"], full["MMC"] or 0, full["calmar"], full["n"], std["CORR"], std["MMC"] or 0,
            std["calmar"], std["n"], side["train_corr"], side["held_corr"],
            side["train_data_loss"], side["held_data_loss"]))
    out_dir = os.path.join(ROOT, args.base + "_walkforward")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "by_epoch.csv"), index=False)
    note = ("\nfull-sample n counts the periods with MMC (the meta model covers only part of "
            "history); CORR uses every period (n = %d). 'train/held CORR' and 'dl' (data loss) "
            "are the mean over refits of each refit's own window." % rows[-1]["full_n_corr"])
    text = "\n".join(lines) + note
    with open(os.path.join(out_dir, "by_epoch.txt"), "w") as handle:
        handle.write(text + "\n")
    print("\n" + text)
    print("\nwrote %s/by_epoch.csv" % out_dir)


if __name__ == "__main__":
    main()
