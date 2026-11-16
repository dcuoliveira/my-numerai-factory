"""
The procedure level of a diagnosed walk-forward: P1-P5 of the training-diagnostics skill.

    python -m src.analysis.walkforward_diagnostics --base torchmlp_noreg_500ep_wf_s42 \
        --floor-base torchmlp_noreg_500ep_wf_s43 [--html --notes reading.md]

src.prediction.diagnose trains ONE refit per call. Run over every refit (tags
<base>_rNN; --scheme full --save-preds --predict-next) it leaves each refit's log.json
and its predictions at every epoch. This reads them back as one procedure:

  P1  folds x epochs: held-out CORR minus the fold's baseline, stop rule marked
  P2  best epoch per fold -- the range within --flat of its best -- beside the stop rule
  P3  per-fold CORR minus a ridge fitted on the same window and scored on the same block
  P4  median and min-max envelope of the probe-batch gradient norm across folds
  P5  churn at each refit boundary: 1 - the mean within-day rank correlation between
      refit k's model and refit k+1's on block k+1 (pred_next.npy against the next
      refit's pred_val.npy). Its floor is the same quantity between two SEEDS of
      refit k+1 on its own block (--floor-base): the movement reseeding alone causes.

plus the per-fold table (summarised by median [IQR], never the mean), the three folds
to give the full panel (first, median and worst by the gap at the kept epoch), and the
stitched held-out scores at the kept epoch against the ridge baseline.

--metric mmc / payout ranks P1-P3, the table and the picks on that metric instead. The
held-out score is then rescored from pred_val.npy at every epoch on the days the meta
model covers (meta_scores), checked against log.json wherever the log has it; a fold
with no meta-model day drops out of P1-P3 and is named, and one with only some is
scored on those days. P6 then sets each fold's MMC gap beside its CORR gap on the same
days, and the same per calendar month: whether beating the ridge on one is beating it
on the other. --suffix keeps those outputs apart from the CORR ones.

The kept epoch is what the procedure ships: the last one for a --no-stop recipe, the
stop rule's epoch otherwise (--kept overrides). The best epoch is picked on the scored
block, so it judges the stop rule and is never a contender.

Output: results/diag/crypto/<base>_walkforward/{procedure.png, churn.png, folds.csv,
summary.json, baseline.json, baseline_per_period.csv}; with --html also
results/reports/crypto/<base>_walkforward.html.
"""
import argparse
import base64
import concurrent.futures as cf
import glob
import html
import json
import multiprocessing as mp
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.join("results", "diag", "crypto")
REPORTS = os.path.join("results", "reports", "crypto")
NW_LAG = 20
CHURN_EPOCHS = (0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500)
# P4's screen: a fold is flagged when its gradient norm sits outside [1/2, 2] x the
# cross-fold median for the typical epoch, or reaches 10x it at any epoch.
GRAD_TYPICAL, GRAD_PEAK = 2.0, 10.0
LABEL = {"corr": "CORR", "mmc": "MMC", "payout": "payout"}


def base_score(row, metric):
    """The ridge baseline's fold score on `metric` (payout = 0.1 * CORR + MMC)."""
    if metric == "corr":
        return row["corr"]
    if row.get("mmc") is None:
        return np.nan
    # payout lives on the meta-model days, so its CORR term is the fold's CORR on those
    # days where that differs (a partly covered fold), not over the whole block
    return row["mmc"] if metric == "mmc" else 0.1 * row.get("corr_meta", row["corr"]) + row["mmc"]


# ------------------------------------------------------------------ loading

def load_folds(base, metric="corr"):
    """
    Every <base>_rNN: its log, per-epoch held-out metric and gradient, finished or not.
    `score` is the held-out series P1-P3 rank on: CORR by default, MMC or payout for a
    model trained on them (--metric).
    """
    dirs = sorted(glob.glob(os.path.join(ROOT, base + "_r[0-9][0-9]")))
    if not dirs:
        raise SystemExit("no refits found for %s_rNN under %s" % (base, ROOT))
    folds = []
    for d in dirs:
        log = json.load(open(os.path.join(d, "log.json")))
        snaps, cfg = log["snapshots"], log["meta"]
        scored = [s.get("metric") or {} for s in snaps]
        get = lambda key: np.array([m.get(key, np.nan) for m in scored], dtype="float64")
        folds.append({
            "dir": d, "tag": os.path.basename(d), "refit": int(cfg["refit"]), "config": cfg,
            "summary": log.get("summary"), "finished": log.get("summary") is not None,
            "epoch": np.array([s["epoch"] for s in snaps]),
            "held_corr": get("held_corr"), "held_mmc": get("held_mmc"),
            "score": get("held_%s" % metric), "metric": metric,
            "train_corr": get("train_corr"),
            "grad": np.array([s["gradients"]["total"] for s in snaps], dtype="float64")})
    return folds


def at(fold, key, epoch):
    """fold[key] at one epoch, NaN where that epoch was not reached or not scored."""
    hit = np.flatnonzero(fold["epoch"] == epoch)
    return float(fold[key][hit[0]]) if len(hit) else np.nan


def kept_epoch(fold, rule):
    stop = (fold["summary"] or {}).get("stop_rule_epoch")
    if rule == "stop" and stop:
        return int(stop)
    return int(fold["epoch"][-1])


def best_epoch(fold, flat):
    """(best epoch, first and last epoch within `flat` of the best) on the held-out score."""
    corr, ep = fold.get("score", fold["held_corr"]), fold["epoch"]
    ok = np.isfinite(corr)
    best = int(ep[ok][np.argmax(corr[ok])])
    near = ep[ok][corr[ok] >= corr[ok].max() - flat]
    return best, int(near.min()), int(near.max())


def meta_scores(fold_dir):
    """
    Held-out CORR and MMC at every epoch, on the days the meta model covers.

    The run's log.json scores MMC only when most held-out rows have a meta model
    (training_diagnostics: over half of them), so on the full layout r07 -- 37 meta days
    at the end of its block -- has none there, and r00-r06 have no meta day at all. This
    rescores pred_val.npy on the covered days: per day, numerai_tools on a (rows x
    epochs) frame, which is per_period_corr / per_period_bmc for every epoch at once.
    A flat day's CORR is 0, as the log charges it; an undefined MMC day is skipped, as
    the ridge baseline skips it. Epochs the run never reached stay NaN. Cached as the
    fold's meta_scores.npz. None when the fold has no meta-model day.
    """
    from numerai_tools.scoring import correlation_contribution, numerai_corr
    cache = os.path.join(fold_dir, "meta_scores.npz")
    pred_path = os.path.join(fold_dir, "pred_val.npy")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(pred_path):
        z = np.load(cache)
        return {k: z[k] for k in z.files} if int(z["n_days"]) else None
    rows = pd.read_parquet(os.path.join(fold_dir, "val_rows.parquet"))
    covered = rows["meta_model"].notna().to_numpy()
    if not covered.any():
        np.savez(cache, n_days=0)
        return None
    P = np.asarray(np.load(pred_path, mmap_mode="r")[:, covered])     # epochs x covered rows
    # Only the epochs the run reached go to the scorer: one all-NaN column makes
    # numerai_tools drop every row of the day and score nothing.
    reached = np.isfinite(P).all(axis=1)
    sub = rows[covered].reset_index(drop=True)
    corr, mmc = [], []
    for _, idx in sorted(sub.groupby("date").indices.items()):
        frame = pd.DataFrame(P[reached][:, idx].T)                      # rows x reached epochs
        target = sub["target"].iloc[idx].reset_index(drop=True)
        meta = sub["meta_model"].iloc[idx].reset_index(drop=True)
        for out, score in ((corr, lambda: numerai_corr(frame, target)),
                           (mmc, lambda: correlation_contribution(frame, meta, target))):
            try:
                out.append(score().to_numpy(dtype="float64"))
            except Exception:
                out.append(np.full(int(reached.sum()), np.nan))
    C = np.full(P.shape[0], np.nan)
    M = np.full(P.shape[0], np.nan)
    C[reached] = np.nan_to_num(np.vstack(corr), nan=0.0).mean(axis=0)
    by_epoch = np.vstack(mmc).T
    M[reached] = [np.nanmean(col) if np.isfinite(col).any() else np.nan for col in by_epoch]
    result = {"corr": C, "mmc": M, "n_days": np.int64(len(corr)),
              "n_days_block": np.int64(rows["date"].nunique())}
    np.savez(cache, **result)
    return result


def rescore_on_meta_days(folds, metric, jobs=8, tol=1e-9):
    """
    Replace each fold's held-out score with meta_scores(): `score` becomes MMC (or the
    payout 0.1 * CORR + MMC) on the meta-model days, `corr_meta` the CORR on those days.
    A fold with no meta day is marked covered=False and drops out of P1-P3.

    Checked against log.json wherever the log scored the same thing -- MMC on a fold the
    log covered, CORR on a fold whose every day has the meta model -- and refused past
    `tol`, so the rescoring cannot quietly differ from what training recorded.
    """
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as pool:
        scored = list(pool.map(meta_scores, [f["dir"] for f in folds]))
    gate = {"mmc_max_diff": 0.0, "corr_max_diff": 0.0, "mmc_checked": [], "corr_checked": []}
    for f, r in zip(folds, scored):
        nan = np.full(len(f["epoch"]), np.nan)
        if r is None:
            f.update({"covered": False, "meta_days": 0, "held_mmc": nan, "corr_meta": nan, "score": nan})
            continue
        e = f["epoch"]
        mmc, corr = r["mmc"][e], r["corr"][e]
        logged_mmc = f["held_mmc"][np.isfinite(mmc)]
        if len(logged_mmc) and np.isfinite(logged_mmc).all():
            gate["mmc_max_diff"] = max(gate["mmc_max_diff"],
                                       float(np.nanmax(np.abs(mmc[np.isfinite(mmc)] - logged_mmc))))
            gate["mmc_checked"].append(f["tag"])
        if int(r["n_days"]) == int(r["n_days_block"]):
            ok = np.isfinite(corr) & np.isfinite(f["held_corr"])
            gate["corr_max_diff"] = max(gate["corr_max_diff"],
                                        float(np.max(np.abs(corr[ok] - f["held_corr"][ok]))))
            gate["corr_checked"].append(f["tag"])
        f.update({"covered": True, "meta_days": int(r["n_days"]), "block_days": int(r["n_days_block"]),
                  "held_mmc": mmc, "corr_meta": corr,
                  "score": mmc if metric == "mmc" else 0.1 * corr + mmc})
    for key in ("mmc", "corr"):
        if gate["%s_max_diff" % key] > tol:
            raise SystemExit("rescored %s differs from log.json by %g on %s" % (
                key.upper(), gate["%s_max_diff" % key], ", ".join(gate["%s_checked" % key])))
    return gate


def rows_frame(fold, which):
    return pd.read_parquet(os.path.join(fold["dir"], which + "_rows.parquet"))


def preds(fold, which):
    return np.load(os.path.join(fold["dir"], "pred_%s.npy" % which), mmap_mode="r")


# ------------------------------------------------------------------ P3 baseline

def ridge_baseline(folds, out_dir, alpha=1.0, alphas=(1e-4, 1.0, 100.0), refresh=False):
    """
    A ridge on each refit's own (scaled) training window, scored on its held-out block.

    Same panel, same windows, same scaler and the same per-day scorer as the network's
    held_corr (src.prediction.diagnose.refit_parts, per_period_corr with flat days
    charged 0), so the gap is the network against a linear fit of the same information.
    alpha 1 is near-OLS at 39,000 rows; the repo's ridge grid spans 1e-4..1e2, and the
    CORR at each end of it is recorded per fold so the baseline's own sensitivity is on
    file (on fenccausal, alpha 100 runs up to ~0.01 above alpha 1 in a fold).

    Its predictions on each refit's own block and on the NEXT one (with that refit's
    scaler) go to ridge_preds.npz, in the run's own row order (checked against
    val_rows.parquet), so P5 can show what a linear model's refit churn is.
    """
    path = os.path.join(out_dir, "baseline.json")
    per_path = os.path.join(out_dir, "baseline_per_period.csv")
    npz_path = os.path.join(out_dir, "ridge_preds.npz")
    if all(os.path.exists(x) for x in (path, per_path, npz_path)) and not refresh:
        cached = json.load(open(path))
        if cached.get("alpha") == alpha and set(cached["folds"]) >= {str(f["refit"]) for f in folds}:
            return cached, pd.read_csv(per_path, parse_dates=["date"])
    from sklearn.linear_model import Ridge
    from src.prediction import diagnose
    from src.prediction.metrics import per_period_corr, per_period_bmc
    cfg = folds[0]["config"]
    args = diagnose.parse_args([
        "--tag", "ridge_baseline", "--data-version", cfg["data_version"], "--target", cfg["target"],
        "--scheme", cfg["scheme"], "--train-window", str(cfg["train_window"]),
        "--refit-every", str(cfg["refit_every"]), "--embargo", str(cfg["embargo"]),
        "--val-frac", str(cfg["val_frac"]), "--predict-next",
        "--n-refits", str(cfg.get("n_refits", 12))] + (["--meta-only"] if cfg.get("meta_only") else []))
    panel = diagnose.load_panel(args)
    target = cfg["target"]
    out, periods, stored = {"alpha": alpha, "alphas": list(alphas), "folds": {}}, [], {}
    for f in folds:
        args.refit = f["refit"]
        train, held, meta, nxt = diagnose.refit_parts(panel, args)
        if (meta["held_start"], meta["held_end"]) != (f["config"]["held_start"], f["config"]["held_end"]):
            raise SystemExit("baseline block for refit %d is not the run's" % f["refit"])
        if os.path.exists(os.path.join(f["dir"], "val_rows.parquet")):
            _same_rows(pd.DataFrame({"date": held["date"], "symbol": held["symbol"]}),
                       rows_frame(f, "val"), "ridge held rows vs %s" % f["tag"])
        fitted = Ridge(alpha=alpha).fit(train["X"], train["y"])
        stored["held_r%02d" % f["refit"]] = fitted.predict(held["X"])
        if nxt is not None:
            stored["next_r%02d" % f["refit"]] = fitted.predict(nxt["X"])
        row = {}
        frame = pd.DataFrame({"date": held["date"], target: held["y_score"],
                              "meta_model": held["meta"]})
        for a in sorted(set(alphas) | {alpha}):
            frame["prediction"] = Ridge(alpha=a).fit(train["X"], train["y"]).predict(held["X"])
            corr = per_period_corr(frame, "prediction", target, "date").fillna(0.0)
            row["corr_alpha_%g" % a] = float(corr.mean())
            if a == alpha:
                # Every day the meta model covers, however few in the fold: MMC is scored
                # per day, and a fold-level coverage gate here once silently dropped the
                # 37 meta days at the end of a mostly uncovered fold from the MMC table.
                mmc = per_period_bmc(frame, "prediction", "meta_model", target, "date")
                row["corr"], row["mmc"] = float(corr.mean()), float(mmc.mean()) if mmc.notna().any() else None
                periods.append(pd.DataFrame({"date": corr.index, "corr": corr.values,
                                             "mmc": mmc.reindex(corr.index).values,
                                             "refit": f["refit"]}))
        out["folds"][str(f["refit"])] = row
    per = pd.concat(periods, ignore_index=True)
    per["date"] = pd.to_datetime(per["date"])
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    per.to_csv(per_path, index=False)
    np.savez(npz_path, **stored)
    return out, per


def ridge_churn(out_dir, refit):
    """Churn of the ridge baseline from refit-1 to refit on refit's block, if stored."""
    path = os.path.join(out_dir, "ridge_preds.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    a, b = "next_r%02d" % (refit - 1), "held_r%02d" % refit
    return (z[a], z[b]) if a in z.files and b in z.files else None


# ------------------------------------------------------------------ P5 churn

def _day_rows(dates):
    from src.prediction.diagnose import rows_by_day
    return rows_by_day(np.asarray(dates))


def churn(a, b, day_rows):
    """1 - mean within-day rank correlation of two prediction vectors on the same rows."""
    from scipy.stats import rankdata
    vals = []
    for idx in day_rows:
        x, y = np.asarray(a)[idx], np.asarray(b)[idx]
        if np.ptp(x) == 0 or np.ptp(y) == 0 or not (np.isfinite(x).all() and np.isfinite(y).all()):
            vals.append(np.nan)
            continue
        vals.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    vals = np.asarray(vals)
    return (float(1.0 - np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
            int(np.isnan(vals).sum()))


def _same_rows(r1, r2, what):
    a = r1[["date", "symbol"]].reset_index(drop=True)
    b = r2[["date", "symbol"]].reset_index(drop=True)
    if not a.equals(b):
        raise SystemExit("row index mismatch: %s" % what)


def boundary_churn(folds, floor, rule, epochs=CHURN_EPOCHS, out_dir=None):
    """
    Per boundary k -> k+1: churn between the refits at the kept epoch, the seed floor
    on block k+1, and both at a list of epochs (same epoch on both sides).
    """
    by_refit = {f["refit"]: f for f in folds if f["finished"]}
    floor_by = {f["refit"]: f for f in (floor or []) if f["finished"]}
    out = []
    for k in sorted(by_refit):
        a, b = by_refit[k], by_refit.get(k + 1)
        if b is None or not os.path.exists(os.path.join(a["dir"], "pred_next.npy")):
            continue
        nrows, vrows = rows_frame(a, "next"), rows_frame(b, "val")
        _same_rows(nrows, vrows, "%s next vs %s val" % (a["tag"], b["tag"]))
        days = _day_rows(vrows["date"])
        pn, pv = preds(a, "next"), preds(b, "val")
        ka, kb = kept_epoch(a, rule), kept_epoch(b, rule)
        cross, cross_nan = churn(pn[ka], pv[kb], days)
        rec = {"from": k, "to": k + 1, "block": "%s..%s" % (b["config"]["held_start"], b["config"]["held_end"]),
               "epoch_from": ka, "epoch_to": kb, "churn": cross, "churn_undefined_days": cross_nan,
               "floor": None, "ridge": None, "by_epoch": {}}
        pair = ridge_churn(out_dir, k + 1) if out_dir else None
        if pair is not None:
            rec["ridge"], _ = churn(pair[0], pair[1], days)
        fb = floor_by.get(k + 1)
        if fb is not None:
            _same_rows(vrows, rows_frame(fb, "val"), "%s val vs %s val" % (b["tag"], fb["tag"]))
            pf = preds(fb, "val")
            rec["floor"], _ = churn(pv[kb], pf[kept_epoch(fb, rule)], days)
        for e in epochs:
            if e > min(a["epoch"][-1], b["epoch"][-1]):
                continue
            row = {"cross": churn(pn[e], pv[e], days)[0]}
            if fb is not None and e <= fb["epoch"][-1]:
                row["floor"] = churn(pv[e], pf[e], days)[0]
            rec["by_epoch"][int(e)] = row
        out.append(rec)
    return out


# ------------------------------------------------------------------ P4 screen

def gradient_screen(folds):
    """Cross-fold median and min-max of the gradient norm per epoch, and the flagged folds."""
    done = [f for f in folds if f["finished"]]
    n = min(len(f["grad"]) for f in done)
    G = np.vstack([f["grad"][:n] for f in done])
    med = np.median(G, axis=0)
    flags = {}
    for f, g in zip(done, G):
        ratio = g / med
        typical, peak = float(np.median(ratio)), float(ratio.max())
        if not (1 / GRAD_TYPICAL <= typical <= GRAD_TYPICAL) or peak >= GRAD_PEAK:
            flags[f["refit"]] = {"typical_ratio": typical, "peak_ratio": peak}
    return {"epochs": done[0]["epoch"][:n], "median": med, "lo": G.min(0), "hi": G.max(0),
            "per_fold": G, "refits": [f["refit"] for f in done], "flags": flags}


# ------------------------------------------------------------------ per-fold table

def fold_table(folds, floor, baseline, churns, rule, flat, metric="corr"):
    churn_into = {c["from"]: c for c in churns}
    floor_by = {f["refit"]: f for f in (floor or []) if f["finished"] and f.get("covered", True)}
    rows = []
    for f in folds:
        if not f["finished"] or not f.get("covered", True):
            continue
        kept = kept_epoch(f, rule)
        stop = (f["summary"] or {}).get("stop_rule_epoch")
        best, lo, hi = best_epoch(f, flat)
        base = baseline["folds"][str(f["refit"])]
        fb = floor_by.get(f["refit"])
        c = churn_into.get(f["refit"])
        rows.append({
            "refit": f["refit"], "held_start": f["config"]["held_start"],
            "held_end": f["config"]["held_end"], "held_days": f["config"]["held_days"],
            "stop_epoch": stop, "best_epoch": best, "best_lo": lo, "best_hi": hi,
            "kept_epoch": kept,
            "score_best": at(f, "score", best), "score_stop": at(f, "score", stop) if stop else np.nan,
            "score_kept": at(f, "score", kept), "score_epoch0": at(f, "score", 0),
            "corr_kept": at(f, "held_corr", kept), "ridge_corr": base["corr"],
            "train_corr_kept": at(f, "train_corr", kept),
            "ridge_score": base_score(base, metric),
            "gap_kept": at(f, "score", kept) - base_score(base, metric),
            "gap_stop": (at(f, "score", stop) - base_score(base, metric)) if stop else np.nan,
            "mmc_kept": at(f, "held_mmc", kept), "ridge_mmc": base.get("mmc"),
            "floor_best_epoch": best_epoch(fb, flat)[0] if fb else None,
            "floor_score_kept": at(fb, "score", kept_epoch(fb, rule)) if fb else np.nan,
            "churn_next": c["churn"] if c else np.nan,
            "churn_next_floor": c["floor"] if c and c["floor"] is not None else np.nan,
            "churn_next_ridge": c["ridge"] if c and c.get("ridge") is not None else np.nan})
        if "corr_meta" in f:                       # ranked on the meta days: CORR on the same days
            rows[-1].update({
                "meta_days": f["meta_days"], "block_days": f["block_days"],
                "corr_meta_kept": at(f, "corr_meta", kept), "ridge_corr_meta": base.get("corr_meta", np.nan),
                "corr_gap_meta_kept": at(f, "corr_meta", kept) - base.get("corr_meta", np.nan),
                "corr_gap_meta_stop": (at(f, "corr_meta", stop) - base.get("corr_meta", np.nan))
                if stop else np.nan})
    return pd.DataFrame(rows)


def median_iqr(x):
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    return float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))


def pick_folds(table, flags):
    """First, median and worst by the gap at the kept epoch, plus any P4-flagged fold."""
    order = table.sort_values("gap_kept").reset_index(drop=True)
    picks = {"first": int(table["refit"].min()), "worst": int(order["refit"].iloc[0]),
             "median": int(order["refit"].iloc[(len(order) - 1) // 2])}
    for r in flags:
        picks.setdefault("flagged r%02d" % r, int(r))
    return picks


# ------------------------------------------------------------------ stitched scores

def stitched_periods(folds, rule, per_fold_epoch=None):
    """Per-period CORR and MMC of every finished refit's held-out predictions at its kept epoch."""
    from src.prediction.metrics import per_period_corr, per_period_bmc
    parts = []
    for f in folds:
        if not f["finished"]:
            continue
        e = per_fold_epoch(f) if per_fold_epoch else kept_epoch(f, rule)
        rows = rows_frame(f, "val")
        frame = rows.assign(prediction=np.asarray(preds(f, "val")[e]))
        target = f["config"]["target"]
        frame = frame.rename(columns={"target": target})
        corr = per_period_corr(frame, "prediction", target, "date").fillna(0.0)
        mmc = per_period_bmc(frame, "prediction", "meta_model", target, "date")
        parts.append(pd.DataFrame({"date": corr.index, "corr": corr.values,
                                   "mmc": mmc.reindex(corr.index).values, "refit": f["refit"]}))
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def score_rows(series, target):
    """results_table rows for each labelled per-period frame, full sample and meta-covered window."""
    from src.analysis import results_table as rt
    from src.analysis.ablation_report import newey_west
    common = sorted(set.intersection(*[set(s["date"]) for s in series.values()]))
    covered = sorted(set.intersection(*[set(s.loc[s["mmc"].notna(), "date"]) for s in series.values()]))
    out = {"full": [], "covered": [], "n_full": len(common), "n_covered": len(covered)}
    base_label = list(series)[-1]
    frames = {k: v.set_index("date").sort_index() for k, v in series.items()}
    for label, frame in frames.items():
        for window, dates in (("full", common), ("covered", covered)):
            if not dates:
                continue
            sl = frame.loc[dates, ["corr", "mmc"]]
            if window == "full":
                sl = sl[["corr"]]
            row = rt.row_from_frame(label, sl, target=target)
            if label != base_label:
                b = frames[base_label].loc[dates]
                d_corr = sl["corr"].values - b["corr"].values
                row["dCORR"], _, row["t(dCORR)"] = newey_west(d_corr, lag=NW_LAG)
                if window == "covered":
                    pay = 0.1 * sl["corr"].values + sl["mmc"].values
                    bpay = 0.1 * b["corr"].values + b["mmc"].values
                    row["dpay"], _, row["t(dpay)"] = newey_west(pay - bpay, lag=NW_LAG)
            out[window].append(row)
    return out


def gap_agreement(nn, ridge, table):
    """
    Whether beating the ridge on CORR is beating it on MMC: per fold (the table's gaps on
    the meta days), per calendar month, and per day (the correlation of the two daily
    gap series), all at the kept epoch, on the days the meta model covers.
    """
    a = nn.set_index("date")[["corr", "mmc"]]
    b = ridge.set_index("date")[["corr", "mmc"]]
    days = a.index[a["mmc"].notna()].intersection(b.index[b["mmc"].notna()])
    gap = (a.loc[days] - b.loc[days]).rename(columns={"corr": "corr_gap", "mmc": "mmc_gap"})
    months = gap.groupby(gap.index.to_period("M")).agg(
        corr_gap=("corr_gap", "mean"), mmc_gap=("mmc_gap", "mean"), days=("corr_gap", "size"))
    months = months[months["days"] >= 10]
    same = lambda x, y: int((np.sign(x) == np.sign(y)).sum())
    slope, intercept = np.polyfit(gap["corr_gap"].to_numpy(), gap["mmc_gap"].to_numpy(), 1)
    return {"n_days": int(len(gap)), "day_corr": float(gap["corr_gap"].corr(gap["mmc_gap"])),
            "day_fit": {"intercept": float(intercept), "slope": float(slope)},
            "folds": table[["refit", "corr_gap_meta_kept", "gap_kept", "meta_days"]].to_dict("records"),
            "folds_same_sign": same(table["corr_gap_meta_kept"], table["gap_kept"]),
            "n_folds": int(len(table)),
            "months": [{"month": str(m), **r} for m, r in months.to_dict("index").items()],
            "months_same_sign": same(months["corr_gap"], months["mmc_gap"]), "n_months": int(len(months)),
            "corr_gap_mean": float(gap["corr_gap"].mean()), "mmc_gap_mean": float(gap["mmc_gap"].mean())}


def _gauss_rank(x):
    from scipy.stats import norm, rankdata
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def meta_alignment(folds, rule, out_dir):
    """
    How close the network and the ridge each sit to the meta model: the mean per-day
    correlation of their rank-gaussianised predictions with the meta model's (the space
    MMC is computed in), at the kept epoch, on the meta-model days -- and of the
    network's departure from the ridge with it. A departure that leans toward the meta
    model is CORR that MMC does not pay for.
    """
    z = np.load(os.path.join(out_dir, "ridge_preds.npz"))
    vals = {"network": [], "ridge": [], "departure": [], "network_vs_ridge": []}
    for f in folds:
        if not f["finished"] or not f.get("covered"):
            continue
        rows = rows_frame(f, "val")
        cov = rows["meta_model"].notna().to_numpy()
        p = np.asarray(preds(f, "val")[kept_epoch(f, rule)])[cov]
        r = z["held_r%02d" % f["refit"]][cov]
        m = rows.loc[cov, "meta_model"].to_numpy()
        for _, idx in rows[cov].reset_index(drop=True).groupby("date").indices.items():
            gp, gr, gm = _gauss_rank(p[idx]), _gauss_rank(r[idx]), _gauss_rank(m[idx])
            vals["network"].append(np.corrcoef(gp, gm)[0, 1])
            vals["ridge"].append(np.corrcoef(gr, gm)[0, 1])
            vals["network_vs_ridge"].append(np.corrcoef(gp, gr)[0, 1])
            d = gp - gr
            vals["departure"].append(np.corrcoef(d, gm)[0, 1] if d.std() > 0 else np.nan)
    return {k: float(np.nanmean(v)) for k, v in vals.items()}


# ------------------------------------------------------------------ figures

def _legend_below(ax, ncol, y=-0.09):
    """Legend under the axis: these panels fill their plotting area edge to edge."""
    from src.analysis.plot_training_diagnostics import INK2
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", bbox_to_anchor=(0, y), ncol=ncol, fontsize=7.5,
                  frameon=False, labelcolor=INK2, handlelength=1.6, borderaxespad=0)


def _runs(refits):
    """[0, 1, 2, 3, 5] -> 'r00–r03, r05'."""
    out, refits = [], sorted(refits)
    while refits:
        a = b = refits.pop(0)
        while refits and refits[0] == b + 1:
            b = refits.pop(0)
        out.append("r%02d" % a if a == b else "r%02d–r%02d" % (a, b))
    return ", ".join(out)


def plot_agreement(agree, path, metric):
    """P6: each fold's and each month's gap to the ridge on MMC against its gap on CORR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.plot_training_diagnostics import SLOTS, INK, INK2, AXIS, GRID, SURFACE, _frame
    fig, ax = plt.subplots(figsize=(11, 7.2), facecolor=SURFACE,
                           gridspec_kw={"left": 0.09, "right": 0.97, "top": 0.86, "bottom": 0.17})
    mx = np.array([m["corr_gap"] for m in agree["months"]])
    my = np.array([m["mmc_gap"] for m in agree["months"]])
    ax.scatter(mx, my, s=34, color=SLOTS[0], alpha=0.45, edgecolor=SURFACE, linewidth=1.5,
               label="calendar month (%d with ≥ 10 meta days)" % agree["n_months"], zorder=2)
    for f in agree["folds"]:
        ax.scatter([f["corr_gap_meta_kept"]], [f["gap_kept"]], s=110, color=SLOTS[1], edgecolor=SURFACE,
                   linewidth=2, zorder=3, label="fold" if f is agree["folds"][0] else None)
    ax.axhline(0, color=AXIS, linewidth=1); ax.axvline(0, color=AXIS, linewidth=1)
    lim_x = max(np.abs(np.r_[mx, [f["corr_gap_meta_kept"] for f in agree["folds"]]]).max() * 1.15, 1e-3)
    lim_y = max(np.abs(np.r_[my, [f["gap_kept"] for f in agree["folds"]]]).max() * 1.15, 1e-3)
    ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y)
    # fold labels in a column to the left of the folds, spread so none overlap, with leader lines
    gap = 0.075 * 2 * lim_y
    placed = []
    for f in sorted(agree["folds"], key=lambda f: -f["gap_kept"]):
        y = f["gap_kept"] if not placed else min(f["gap_kept"], placed[-1] - gap)
        placed.append(y)
        ax.annotate("r%02d (%d d)" % (f["refit"], f["meta_days"]), xy=(f["corr_gap_meta_kept"], f["gap_kept"]),
                    xytext=(-0.45 * lim_x, y), textcoords="data", fontsize=8.5, color=INK, ha="right",
                    va="center", arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8,
                                             "shrinkA": 3, "shrinkB": 6})
    fit = agree["day_fit"]
    xs = np.array([-lim_x, lim_x])
    ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=INK2, linewidth=1.4, linestyle=(0, (5, 3)),
            label="daily fit: %s gap = %+.4f + %.2f × CORR gap" % (LABEL[metric], fit["intercept"], fit["slope"]),
            zorder=1)
    for x, y, text in ((0.98, 0.97, "beats the ridge on both"), (0.02, 0.03, "behind it on both"),
                       (0.02, 0.97, "MMC only"), (0.98, 0.03, "CORR only")):
        ax.text(x, y, text, transform=ax.transAxes, ha="right" if x > 0.5 else "left",
                va="top" if y > 0.5 else "bottom", fontsize=8.5, color=INK2)
    ax.set_xlabel("held-out CORR − ridge CORR, same days", fontsize=9, color=INK2)
    ax.set_ylabel("held-out %s − ridge %s" % (LABEL[metric], LABEL[metric]), fontsize=9, color=INK2)
    al = agree.get("meta_alignment") or {}
    _frame(ax, "P6 · Does beating the ridge on CORR mean beating it on %s?" % LABEL[metric],
           "kept epoch, meta-model days only · same sign: %d of %d folds, %d of %d months · "
           "per-day correlation of the two gaps %+.2f (n = %d days)" % (
               agree["folds_same_sign"], agree["n_folds"], agree["months_same_sign"],
               agree["n_months"], agree["day_corr"], agree["n_days"]))
    if al:
        fig.text(0.09, 0.018, "Per-day correlation with the meta model (rank-gaussianised, as MMC sees it): "
                 "network %.3f, ridge %.3f, network against ridge %.3f.\nThe network's departure from the "
                 "ridge correlates %+.3f with the meta model." % (
                     al["network"], al["ridge"], al["network_vs_ridge"], al["departure"]),
                 fontsize=8.5, color=INK2)
    ax.grid(color=GRID, linewidth=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3, fontsize=8, frameon=False,
              labelcolor=INK2)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def _batching(cfg):
    if cfg.get("batch") == "days":
        return "%s-day batches" % (cfg.get("days_per_batch") or cfg.get("days"))
    return "%s-row batches" % cfg.get("batch_size")


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    # dataviz reference diverging pair: blue (model above its baseline) <-> red, gray midpoint
    return LinearSegmentedColormap.from_list(
        "blue_red", ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#8f1d1c"][::-1])


def plot_procedure(folds, floor, table, baseline, grads, path, rule, flat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.plot_training_diagnostics import (SLOTS, INK, INK2, MUTED, GRID, AXIS,
                                                        SURFACE, _frame, _legend)
    finished = [f for f in folds if f["finished"]]
    done = [f for f in finished if f.get("covered", True)]
    left = [f["refit"] for f in finished if not f.get("covered", True)]
    partial = {f["refit"]: f["meta_days"] for f in done
               if f.get("meta_days") is not None and f["meta_days"] < f.get("block_days", f["meta_days"])}
    label = {f["refit"]: "r%02d  %s%s" % (f["refit"], f["config"]["held_start"][:7],
                                          " (%d d)" % partial[f["refit"]] if f["refit"] in partial else "")
             for f in done}
    coverage = ""
    if left or partial:
        coverage = " (" + "; ".join(
            ([_runs(left) + " have none and are left out"] if left else []) +
            ["r%02d is scored on its %d meta days" % (r, d) for r, d in partial.items()]) + ")"
    fig = plt.figure(figsize=(15, 15.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.5, wspace=0.18 if not partial else 0.22,
                          left=0.08 if not partial else 0.1, right=0.97, top=0.9, bottom=0.06)

    # P1 -- folds x epochs, CORR minus the fold's ridge, log epoch axis (epoch 0 omitted)
    ax = fig.add_subplot(gs[0, :])
    n_ep = min(len(f["epoch"]) for f in done)
    metric = done[0].get("metric", "corr")
    M = np.vstack([f["score"][1:n_ep] - base_score(baseline["folds"][str(f["refit"])], metric) for f in done])
    lim = float(np.nanpercentile(np.abs(M), 99)) or 0.05
    edges = np.arange(1, n_ep + 1) - 0.5
    mesh = ax.pcolormesh(edges, np.arange(len(done) + 1), M, cmap=_diverging(), vmin=-lim, vmax=lim,
                         shading="flat", rasterized=True)
    ax.set_xscale("log"); ax.set_xlim(0.5, n_ep - 0.5); ax.invert_yaxis()
    ax.set_yticks(np.arange(len(done)) + 0.5); ax.set_yticklabels([label[f["refit"]] for f in done])
    for i, f in enumerate(done):
        stop = (f["summary"] or {}).get("stop_rule_epoch")
        if stop:
            ax.plot([stop], [i + 0.5], marker="|", markersize=14, mew=2.2, color=INK)
        b, _, _ = best_epoch(f, flat)
        ax.plot([max(b, 1)], [i + 0.5], marker="o", markersize=6, mfc="none", mew=1.6, color=INK)
    _frame(ax, "P1 · Held-out %s minus the fold's ridge baseline, every fold × every epoch" % LABEL[metric],
           "blue = above the ridge, red = below, gray = level · ○ best epoch · | stop rule "
           "(where sklearn's rule would have stopped) · log epoch axis, epoch 0 omitted")
    ax.grid(False); ax.set_xlabel("epoch", fontsize=9, color=INK2)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=8, length=0); cb.outline.set_visible(False)
    cb.set_label("%s − ridge" % LABEL[metric], fontsize=8.5, color=INK2)

    # P2 -- best epoch per fold, flat-top range, stop rule; the second seed hollow
    ax = fig.add_subplot(gs[1, 0])
    floor_by = {f["refit"]: f for f in (floor or []) if f["finished"] and f.get("covered", True)}
    for i, f in enumerate(done):
        b, lo, hi = best_epoch(f, flat)
        ax.plot([max(lo, 0.8), max(hi, 0.8)], [i, i], color=SLOTS[0], linewidth=2.4, alpha=0.35,
                solid_capstyle="round")
        ax.plot([max(b, 0.8)], [i], "o", color=SLOTS[0], markersize=8,
                label="best epoch (seed %s)" % f["config"]["seed"] if i == 0 else None)
        stop = (f["summary"] or {}).get("stop_rule_epoch")
        if stop:
            ax.plot([stop], [i], marker="|", markersize=13, mew=2.2, color=SLOTS[1],
                    label="stop rule" if i == 0 else None)
        fb = floor_by.get(f["refit"])
        if fb is not None:
            ax.plot([max(best_epoch(fb, flat)[0], 0.8)], [i], "o", mfc="none", mew=1.6, color=SLOTS[0],
                    markersize=8, label="best epoch (seed %s)" % fb["config"]["seed"] if i == 0 else None)
    ax.set_xscale("log"); ax.set_yticks(range(len(done)))
    ax.set_yticklabels([label[f["refit"]] for f in done]); ax.invert_yaxis()
    ax.set_xlim(0.7, max(f["epoch"][-1] for f in done) * 1.3)
    _frame(ax, "P2 · Best epoch per fold (on %s)" % LABEL[metric], "bar = epochs within %.3f of the "
           "fold's best (a long bar is a flat top: its argmax is noise)" % flat)
    ax.grid(axis="x", color=GRID, linewidth=0.7); ax.grid(axis="y", visible=False)
    ax.set_xlabel("epoch (log)", fontsize=9, color=INK2); _legend_below(ax, 3, y=-0.15)

    # P3 -- the gap at the kept epoch, and what the stop rule would have shipped
    ax = fig.add_subplot(gs[1, 1])
    t = table.set_index("refit").loc[[f["refit"] for f in done]]
    y = np.arange(len(done))
    colors = [SLOTS[0] if g > 0 else SLOTS[7] for g in t["gap_kept"]]
    ax.barh(y, t["gap_kept"], color=colors, height=0.6, label="epoch kept (%s)" % rule)
    ax.plot(t["gap_stop"], y, "D", color=INK, markersize=5, label="at the stop-rule epoch")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([label[r] for r in t.index]); ax.invert_yaxis()
    med, q1, q3 = median_iqr(t["gap_kept"])
    _frame(ax, "P3 · Held-out %s minus the ridge on the same window and block" % LABEL[metric],
           "median %+.4f [IQR %+.4f, %+.4f] · %d of %d folds ≤ 0 at the kept epoch"
           % (med, q1, q3, int((t["gap_kept"] <= 0).sum()), len(t)))
    ax.grid(axis="x", color=GRID, linewidth=0.7); ax.grid(axis="y", visible=False)
    _legend_below(ax, 2)

    # P4 -- gradient envelope; flagged folds drawn on their own
    ax = fig.add_subplot(gs[2, 0])
    _frame(ax, "P4 · Gradient norm on the fixed probe batch, across folds",
           "median and min–max band over %d folds · a fold outside the screen is drawn alone"
           % len(grads["refits"]), log=True)
    x = np.maximum(grads["epochs"], 0.8)
    ax.fill_between(x, grads["lo"], grads["hi"], color=SLOTS[0], alpha=0.18, linewidth=0,
                    label="min–max across folds")
    ax.plot(x, grads["median"], color=SLOTS[0], linewidth=2, label="median")
    for j, (r, flag) in enumerate(grads["flags"].items()):
        g = grads["per_fold"][grads["refits"].index(r)]
        ax.plot(x, g, color=SLOTS[1 + j % 6], linewidth=1.2, label="r%02d (flagged)" % r)
    ax.set_xscale("log"); ax.set_xlabel("epoch (log)", fontsize=9, color=INK2)
    _legend(ax, "upper right")

    # P2/P3 companion -- held-out CORR at best / stop / kept against the ridge, per fold
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t["score_best"], y, "o", color=SLOTS[0], markersize=7, label="network at its best epoch")
    ax.plot(t["score_stop"], y, "D", color=INK, markersize=5, label="network at the stop rule")
    ax.plot(t["score_kept"], y, "s", color=SLOTS[1], markersize=6, label="network at the kept epoch")
    ax.plot(t["ridge_score"], y, "|", color=SLOTS[2], markersize=16, mew=2.6, label="ridge baseline")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([label[r] for r in t.index]); ax.invert_yaxis()
    _frame(ax, "Raw held-out %s per fold" % LABEL[metric], "period difficulty moves every model together; "
           "read P3, not this, to compare folds")
    ax.grid(axis="x", color=GRID, linewidth=0.7); ax.grid(axis="y", visible=False)
    _legend_below(ax, 4)

    cfg = done[0]["config"]
    fig.text(0.08, 0.968, "Walk-forward procedure · %s" % os.path.basename(path).replace("_procedure.png", ""),
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.08, 0.952, "%s · %s · hidden %s %s · %s loss · alpha %g · lr %g %s · %s · init %s · %d "
             "epochs%s · seed %s · %d of %d refits finished" % (
                 cfg["data_version"], cfg["target"], cfg["hidden"], cfg["activation"],
                 {"corr": "−Pearson", "mmc": "−MMC"}.get(cfg.get("loss"), "MSE"), cfg["alpha"],
                 cfg["lr"], cfg["schedule"], _batching(cfg), cfg["init"],
                 cfg["max_epochs"], " (--no-stop)" if cfg.get("no_stop") else "", cfg["seed"],
                 len(finished), len(folds)), fontsize=9, color=INK2)
    if metric != "corr":
        fig.text(0.08, 0.937, "P1–P3 ranked on %s, over the %d folds with meta-model days%s; P4 keeps every fold"
                 % (LABEL[metric], len(done), coverage), fontsize=9, color=INK2)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_churn(churns, path, rule):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.plot_training_diagnostics import SLOTS, INK2, GRID, SURFACE, _frame, _legend
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.18, "left": 0.06, "right": 0.98,
                                          "top": 0.82, "bottom": 0.3})
    ax = axes[0]
    x = np.arange(len(churns))
    ax.bar(x, [c["churn"] for c in churns], color=SLOTS[0], width=0.6, label="refit k → k+1")
    fl = [c["floor"] if c["floor"] is not None else np.nan for c in churns]
    ax.plot(x, fl, "_", color=SLOTS[1], markersize=26, mew=3, label="seed floor (two seeds of refit k+1)")
    rg = [c["ridge"] if c.get("ridge") is not None else np.nan for c in churns]
    if np.isfinite(rg).any():
        ax.plot(x, rg, "_", color=SLOTS[2], markersize=26, mew=3, label="ridge baseline, refit k → k+1")
    ax.set_xticks(x)
    ax.set_xticklabels(["r%02d→r%02d\n%s" % (c["from"], c["to"], c["block"][:7]) for c in churns],
                       fontsize=7.5)
    ax.set_ylim(0, 1)
    _frame(ax, "P5 · Churn at each refit boundary, at the kept epoch (%s)" % rule,
           "1 − mean within-day rank correlation on block k+1 · it is the ranking turnover the "
           "refit swap costs")
    _legend_below(ax, 2, y=-0.2)
    ax = axes[1]
    eps = sorted({e for c in churns for e in c["by_epoch"]})
    for key, color, name in (("cross", SLOTS[0], "refit k → k+1"), ("floor", SLOTS[1], "seed floor")):
        M = np.array([[c["by_epoch"].get(e, {}).get(key, np.nan) for e in eps] for c in churns],
                     dtype="float64")
        if not np.isfinite(M).any():
            continue
        xe = np.maximum(np.array(eps, dtype="float64"), 0.8)
        ax.fill_between(xe, np.nanmin(M, 0), np.nanmax(M, 0), color=color, alpha=0.15, linewidth=0)
        ax.plot(xe, np.nanmedian(M, 0), color=color, linewidth=2, marker="o", markersize=4,
                label="%s (median, min–max band)" % name)
    rg = [c["ridge"] for c in churns if c.get("ridge") is not None]
    if rg:
        ax.axhline(np.median(rg), color=SLOTS[2], linewidth=1.6, linestyle=(0, (4, 3)),
                   label="ridge baseline, refit k → k+1 (median)")
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("epoch (log; same epoch on both sides)",
                                                           fontsize=9, color=INK2)
    _frame(ax, "P5 · Churn by training length", "how far training moves the function between "
           "refits, against what reseeding alone moves it")
    _legend_below(ax, 2, y=-0.17)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ report

def _img(path):
    with open(path, "rb") as handle:
        return "data:image/png;base64," + base64.b64encode(handle.read()).decode()


def _md(text):
    """Paragraphs, '- ' bullets (one nested level, '  - '), '## ' headings, **bold**, `code`."""
    inline = lambda t: re.sub(r"`([^`]+)`", r"<code>\1</code>",
                              re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html.escape(t)))
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.rstrip().split("\n")
        if lines[0].startswith("## "):
            out.append("<h2>%s</h2>" % inline(lines[0][3:]))
            lines = lines[1:]
            if not lines:
                continue
        if all(l.strip().startswith("|") for l in lines):    # a pipe table, header + rule + rows
            cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
            rows = [cells(l) for l in lines if not re.match(r"^\s*\|[\s:|-]+\|\s*$", l)]
            out.append("<div class='scroll'><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>" % (
                "".join("<th>%s</th>" % inline(c) for c in rows[0]),
                "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % inline(c) for c in r) for r in rows[1:])))
            continue
        if not lines[0].startswith("- "):
            out.append("<p>%s</p>" % inline(" ".join(l.strip() for l in lines)))
            continue
        items = []                                   # [[text, [sub, ...]], ...]
        for l in lines:
            if l.startswith("- "):
                items.append([l[2:], []])
            elif l.startswith("  - "):
                items[-1][1].append(l[4:])
            elif items[-1][1]:
                items[-1][1][-1] += " " + l.strip()
            else:
                items[-1][0] += " " + l.strip()
        out.append("<ul>" + "".join(
            "<li>%s%s</li>" % (inline(t), "<ul>%s</ul>" % "".join("<li>%s</li>" % inline(x) for x in sub)
                                if sub else "") for t, sub in items) + "</ul>")
    return "\n".join(out)


def _table(frame, fmt, caption=None):
    head = "".join("<th>%s</th>" % html.escape(str(c)) for c in frame.columns)
    body = []
    for _, r in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = r[c]
            spec = fmt.get(c)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                s = "–"
            elif spec and isinstance(v, (int, float, np.floating, np.integer)):
                s = spec.format(v)
            else:
                s = str(v)
            cells.append("<td>%s</td>" % html.escape(s))
        body.append("<tr>%s</tr>" % "".join(cells))
    cap = "<caption>%s</caption>" % html.escape(caption) if caption else ""
    return "<table>%s<thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (cap, head, "".join(body))


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f3f2ee;
--ink2:#c3c2b7;--muted:#898781;--grid:#383835;--card:#232322}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f3f2ee;--ink2:#c3c2b7;--muted:#898781;--grid:#383835;--card:#232322}
body{background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
Helvetica,Arial,sans-serif;margin:0}
main{max-width:1180px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:34px 0 10px;border-top:1px solid var(--grid);
padding-top:22px}h3{font-size:15px;margin:22px 0 6px}
.sub{color:var(--ink2);margin:0 0 18px}p,li{max-width:900px}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--card);padding:1px 4px;
border-radius:4px}
.fig{background:#fcfcfb;border-radius:8px;padding:6px;margin:10px 0}
.fig img{width:100%;height:auto;display:block}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;margin:8px 0 4px}
caption{text-align:left;color:var(--ink2);font:13px -apple-system,sans-serif;padding:0 0 6px}
th,td{padding:4px 9px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600}td:first-child,th:first-child{text-align:left}
tr.summary td{font-weight:700}
.note{color:var(--ink2);font-size:13.5px}
"""


def write_html(base, summary, table, scores, panels, figs, notes, path):
    cfg = summary["config"]
    metric = summary.get("metric", "corr")
    L = LABEL[metric]
    t = table.copy()
    t["best (flat range)"] = ["%d (%d–%d)" % (b, lo, hi) for b, lo, hi in
                              zip(t["best_epoch"], t["best_lo"], t["best_hi"])]
    t["held-out"] = t["held_start"] + " → " + t["held_end"]
    cols = {"refit": "fold", "held-out": "held-out", "stop_epoch": "stop", "best (flat range)": "best (flat range)",
            "score_best": "%s@best" % L, "score_stop": "%s@stop" % L, "score_kept": "%s@kept" % L,
            "ridge_score": "ridge %s" % L, "gap_kept": "gap@kept", "gap_stop": "gap@stop",
            "train_corr_kept": "train CORR@kept", "mmc_kept": "MMC@kept",
            "churn_next": "churn→next", "churn_next_floor": "seed floor",
            "churn_next_ridge": "ridge churn"}
    if metric != "corr":            # ranked on the meta days: their count, and CORR's gap on the same days
        cols = {k: v for k, v in cols.items() if k not in ("train_corr_kept", "mmc_kept")}
        cols = dict(list(cols.items())[:2] + [("meta_days", "meta days")] + list(cols.items())[2:])
        cols["gap_kept"], cols["gap_stop"] = "%s gap@kept" % L, "%s gap@stop" % L
        at_churn = list(cols).index("churn_next")
        items = list(cols.items())
        cols = dict(items[:at_churn] + [("corr_gap_meta_kept", "CORR gap@kept (same days)")] + items[at_churn:])
    shown = t[list(cols)].rename(columns=cols)
    fmt = {c: "{:+.4f}" for c in ("%s@best" % L, "%s@stop" % L, "%s@kept" % L, "ridge %s" % L,
                                  "gap@kept", "gap@stop", "train CORR@kept", "MMC@kept",
                                  "%s gap@kept" % L, "%s gap@stop" % L, "CORR gap@kept (same days)")}
    fmt.update({"churn→next": "{:.3f}", "seed floor": "{:.3f}", "ridge churn": "{:.3f}",
                "stop": "{:.0f}", "fold": "r{:02.0f}", "meta days": "{:.0f}"})
    s = summary["fold_summary"]
    summary_row = {"fold": "median [IQR]", "held-out": "%d folds" % len(t), "stop": "",
                   "best (flat range)": "%s [%s, %s]" % tuple("%.0f" % v for v in s["best_epoch"]),
                   "%s@best" % L: "", "%s@stop" % L: "", "%s@kept" % L: "%+.4f" % s["score_kept"][0],
                   "ridge %s" % L: "%+.4f" % s["ridge_score"][0],
                   "gap@kept": "%+.4f [%+.4f, %+.4f]" % tuple(s["gap_kept"]),
                   "gap@stop": "%+.4f" % s["gap_stop"][0], "train CORR@kept": "", "MMC@kept": "",
                   "churn→next": "%.3f [%.3f, %.3f]" % tuple(s["churn_next"]),
                   "seed floor": "%.3f" % s["churn_next_floor"][0],
                   "ridge churn": "%.3f" % s["churn_next_ridge"][0]}
    if metric != "corr":
        summary_row["%s@kept" % L] = "%+.4f" % s["score_kept"][0]   # the literal "MMC@kept" above blanked it
        summary_row.update({"meta days": "", "%s gap@kept" % L: summary_row.pop("gap@kept"),
                            "%s gap@stop" % L: summary_row.pop("gap@stop"),
                            "CORR gap@kept (same days)": "%+.4f [%+.4f, %+.4f]" % tuple(s["corr_gap_meta_kept"])})
    fold_html = _table(shown, fmt).replace(
        "</tbody>", "<tr class='summary'>%s</tr></tbody>" % "".join(
            "<td>%s</td>" % html.escape(str(summary_row[c])) for c in shown.columns))
    score_cols = ["run", "n", "CORR", "c.sharpe", "dd(corr)", "dCORR", "t(dCORR)"]
    cov_cols = ["run", "n", "CORR", "MMC", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)",
                "dpay", "t(dpay)"]
    sfmt = {"CORR": "{:+.4f}", "MMC": "{:+.4f}", "payout": "{:+.4f}", "c.sharpe": "{:.3f}",
            "p.sharpe": "{:.3f}", "dd(corr)": "{:.3f}", "dd(pay)": "{:.3f}", "calmar": "{:.3f}",
            "dCORR": "{:+.4f}", "t(dCORR)": "{:+.2f}", "dpay": "{:+.4f}", "t(dpay)": "{:+.2f}",
            "n": "{:.0f}", "burn(d)": "{:.0f}"}
    full = pd.DataFrame(scores["full"]).reindex(columns=score_cols)
    cov = pd.DataFrame(scores["covered"]).reindex(columns=cov_cols)
    coverage_note = ""
    agree_html = ""
    if metric != "corr":
        coverage = summary["metric_coverage"]
        coverage_note = ("<p class='note'>Ranked on <strong>%s</strong>, which exists only where the meta model does. "
                         "P1–P3, the table and the fold picks cover the %d folds with meta-model days%s. P4 and P5 "
                         "do not depend on the metric and keep every fold.</p>" % (
                             L, len(coverage["covered"]),
                             (" (%s has none: left out)" % _runs(coverage["left_out"]) if coverage["left_out"] else "")
                             + "".join("; r%02d is scored on its %d meta days of %d" % (r, d, n)
                                       for r, d, n in coverage["partial"])))
        g = summary["agreement"]
        agree_html = ("<h2>Does beating the ridge on CORR mean beating it on %s? (P6)</h2>"
                      "<p class='note'>Each fold's gap to the ridge on %s against its gap on CORR, over the same "
                      "meta-model days and at the kept epoch, with every calendar month that has at least 10 of "
                      "them. Same sign: %d of %d folds and %d of %d months. The two daily gap series correlate "
                      "%+.2f over %d days, and the daily fit is %s gap = %+.4f + %.2f × CORR gap. Per-day "
                      "correlation with the meta model, in the rank-gaussianised space MMC uses: network %.3f, "
                      "ridge %.3f, network against ridge %.3f, and the network's departure from the ridge "
                      "%+.3f.</p><div class='fig'><img alt='P6 agreement' src='%s'></div>" % (
                          L, L, g["folds_same_sign"], g["n_folds"], g["months_same_sign"], g["n_months"],
                          g["day_corr"], g["n_days"], L, g["day_fit"]["intercept"], g["day_fit"]["slope"],
                          g["meta_alignment"]["network"], g["meta_alignment"]["ridge"],
                          g["meta_alignment"]["network_vs_ridge"], g["meta_alignment"]["departure"],
                          _img(figs["agreement"])))
    panel_html = "".join(
        "<h3>%s fold — r%02d (%s)</h3><div class='fig'><img alt='diagnostic panel of refit %d' src='%s'></div>"
        % (role.capitalize(), r, desc, r, _img(p)) for role, r, desc, p in panels)
    doc = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Walk-forward training report</title>
<style>%s</style></head><body><main>
<h1>Walk-forward training report%s</h1>
<p class="sub"><code>%s</code> · %s · %s · hidden %s %s · %s loss · alpha %g · lr %g (%s) · %s · %s init ·
%d epochs%s · seeds %s · %d refits (%s layout%s) · generated by <code>src.analysis.walkforward_diagnostics</code></p>
%s
%s
<h2>Procedure level — every fold (P1–P4)</h2>
<div class="fig"><img alt="P1 to P4" src="%s"></div>
%s
<h2>Procedure level — cross-fold churn (P5)</h2>
<div class="fig"><img alt="P5 churn" src="%s"></div>
<h2>Per-fold table</h2>
<p class="note">One row per fold. The kept epoch is <strong>%s</strong>. The best epoch is chosen
on the scored block, so it judges the stop rule and is not something to deploy. The summary row is
the median with its [IQR], never the mean. Churn→next compares this refit's model with the next
refit's on the next block. The seed floor is the same quantity between two seeds of the next
refit, and ridge churn is the ridge baseline's own churn across the same boundary.</p>
<div class="scroll">%s</div>
<h2>Run level — first, median and worst fold</h2>
<p class="note">The full single-run panel, chosen by the gap at the kept epoch (P3).</p>
%s
<h2>Scores — the stitched walk-forward at the kept epoch</h2>
<p class="note">Every finished refit's held-out block at its kept epoch, stitched together and scored
the way <code>results_table</code> scores a run. The last row in each table is the ridge baseline,
and d·/t(·) are the paired differences against it (Newey-West, lag %d). The first table covers every
common period and reports CORR only. The meta model publishes only recent history, so MMC, payout
and calmar are on the second table's window alone. Both windows are shared by every row, so calmar
compares like with like within each table.</p>
<div class="scroll">%s</div><div class="scroll">%s</div>
</main></body></html>""" % (
        CSS, " — ranked on %s" % L if metric != "corr" else "", html.escape(base), cfg["data_version"], cfg["target"], cfg["hidden"], cfg["activation"],
        {"corr": "per-day −Pearson", "mmc": "per-day −MMC"}.get(cfg.get("loss"), "0.5·MSE"), cfg["alpha"], cfg["lr"], cfg["schedule"], _batching(cfg), cfg["init"],
        cfg["max_epochs"], " (no early stopping)" if cfg.get("no_stop") else "",
        ", ".join(str(x) for x in summary["seeds"]), summary["n_refits"], cfg.get("scheme", "?"),
        ", meta-model days only" if cfg.get("meta_only") else "",
        _md(notes) if notes else "", coverage_note, _img(figs["procedure"]), agree_html, _img(figs["churn"]),
        summary["kept_rule_text"], fold_html, panel_html, NW_LAG,
        _table(full, sfmt, "Full sample (n = %d periods) — CORR" % scores["n_full"]),
        _table(cov, sfmt, "Meta-model window (n = %d periods) — CORR, MMC, payout" % scores["n_covered"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(doc)
    return path


# ------------------------------------------------------------------ main

def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.analysis.walkforward_diagnostics",
                                description=__doc__.split("\n")[1])
    p.add_argument("--base", required=True, help="the refits are results/diag/crypto/<base>_rNN")
    p.add_argument("--floor-base", default=None, help="a second seed of the same refits (P5 floor)")
    p.add_argument("--kept", choices=("auto", "last", "stop"), default="auto",
                   help="the epoch the procedure ships; auto = last for --no-stop runs, else stop")
    p.add_argument("--flat", type=float, default=0.005, help="P2's flat-top tolerance, in the metric's units")
    p.add_argument("--metric", choices=("corr", "mmc", "payout"), default="corr",
                   help="what P1-P3 and the fold picks rank on: CORR, or MMC / payout for a "
                        "model trained on them (they need the meta model on every held-out day)")
    p.add_argument("--suffix", default="",
                   help="appended to the figure, table, summary and HTML names (e.g. _mmc), so a second "
                        "metric's report does not overwrite the first; the ridge baseline is shared")
    p.add_argument("--jobs", type=int, default=8, help="processes for rescoring on the meta-model days")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--refresh-baseline", action="store_true")
    p.add_argument("--html", action="store_true", help="also write results/reports/crypto/<base>_walkforward.html")
    p.add_argument("--notes", default=None, help="a markdown file of reading to put at the top of the HTML")
    args = p.parse_args(argv)

    folds = load_folds(args.base, args.metric)
    floor = load_folds(args.floor_base, args.metric) if args.floor_base else None
    done = [f for f in folds if f["finished"]]
    if not done:
        raise SystemExit("no finished refit under %s" % args.base)
    rule = args.kept if args.kept != "auto" else ("last" if done[0]["config"].get("no_stop") else "stop")
    out_dir = os.path.join(ROOT, args.base + "_walkforward")
    os.makedirs(out_dir, exist_ok=True)
    missing = [f["tag"] for f in folds if not f["finished"]]
    if missing:
        print("LEFT OUT (unfinished, no summary in log.json): %s" % ", ".join(missing))
    if floor:
        fm = [f["tag"] for f in floor if not f["finished"]]
        if fm:
            print("floor runs LEFT OUT (unfinished): %s" % ", ".join(fm))

    baseline, base_periods = ridge_baseline(folds, out_dir, alpha=args.ridge_alpha,
                                            refresh=args.refresh_baseline)
    gate = None
    if args.metric != "corr":
        gate = rescore_on_meta_days(folds + (floor or []), args.metric, jobs=args.jobs)
        print("rescored on the meta-model days; matches log.json to %.1e (MMC, %d folds) and %.1e "
              "(CORR, %d fully covered folds)" % (gate["mmc_max_diff"], len(gate["mmc_checked"]),
                                                  gate["corr_max_diff"], len(gate["corr_checked"])))
        for refit, g in base_periods[base_periods["mmc"].notna()].groupby("refit"):
            baseline["folds"][str(refit)]["corr_meta"] = float(g["corr"].mean())
        uncovered = [f["refit"] for f in done if not f["covered"]]
        if uncovered:
            print("LEFT OUT of P1-P3 and the table (no meta-model day): %s" % _runs(uncovered))
    churns = boundary_churn(folds, floor, rule, out_dir=out_dir)
    grads = gradient_screen(folds)
    table = fold_table(folds, floor, baseline, churns, rule, args.flat, args.metric)
    picks = pick_folds(table, grads["flags"])
    table.to_csv(os.path.join(out_dir, "folds%s.csv" % args.suffix), index=False)

    fold_summary = {k: median_iqr(table[k]) for k in
                    ("best_epoch", "score_kept", "ridge_score", "corr_kept", "ridge_corr", "gap_kept",
                     "gap_stop", "score_stop",
                     "churn_next", "churn_next_floor", "churn_next_ridge", "stop_epoch",
                     "train_corr_kept") + (("corr_gap_meta_kept",) if args.metric != "corr" else ())}
    target = done[0]["config"]["target"]
    nn = stitched_periods(folds, rule)
    nn_stop = stitched_periods(folds, rule, per_fold_epoch=lambda f: kept_epoch(f, "stop"))
    finished_refits = set(nn["refit"])
    ridge = base_periods[base_periods["refit"].isin(finished_refits)]
    kept_label = "network @%s" % ("last epoch" if rule == "last" else "stop rule")
    series = {kept_label: nn}
    if rule == "last":
        series["network @stop rule"] = nn_stop
    series["ridge baseline"] = ridge
    scores = score_rows(series, target)

    stem = os.path.join(out_dir, args.base + args.suffix)
    figs = {"procedure": plot_procedure(folds, floor, table, baseline, grads, stem + "_procedure.png",
                                        rule, args.flat),
            "churn": plot_churn(churns, stem + "_churn.png", rule)}
    agreement, coverage = None, None
    if args.metric != "corr":
        agreement = gap_agreement(nn, ridge, table)
        agreement["meta_alignment"] = meta_alignment(folds, rule, out_dir)
        figs["agreement"] = plot_agreement(agreement, stem + "_agreement.png", args.metric)
        coverage = {"covered": [f["refit"] for f in done if f["covered"]],
                    "left_out": [f["refit"] for f in done if not f["covered"]],
                    "partial": [(f["refit"], f["meta_days"], f["block_days"]) for f in done
                                if f["covered"] and f["meta_days"] < f["block_days"]]}
    summary = {
        "base": args.base, "floor_base": args.floor_base, "config": done[0]["config"],
        "metric": args.metric,
        "seeds": [done[0]["config"]["seed"]] + ([floor[0]["config"]["seed"]] if floor else []),
        "n_refits": len(folds), "finished": [f["refit"] for f in done], "left_out": missing,
        "kept_rule": rule, "kept_rule_text": "the last epoch (the recipe has no early stopping)"
        if rule == "last" else "the stop rule's epoch",
        "flat": args.flat, "ridge": {"alpha": baseline["alpha"], "alphas": baseline["alphas"],
                                     "by_fold": baseline["folds"]},
        "fold_summary": fold_summary, "picks": picks, "gradient_flags": grads["flags"],
        "churn": churns, "scores": scores, "figures": figs,
        "rescore_gate": gate, "metric_coverage": coverage, "agreement": agreement}
    with open(os.path.join(out_dir, "summary%s.json" % args.suffix), "w") as handle:
        json.dump(summary, handle, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    pd.set_option("display.width", 220)
    show = table[["refit", "held_start", "stop_epoch", "best_epoch", "best_lo", "best_hi", "score_best",
                  "score_stop", "score_kept", "ridge_score", "gap_kept", "gap_stop", "corr_kept", "mmc_kept",
                  "churn_next", "churn_next_floor", "churn_next_ridge", "floor_best_epoch"]
                 + (["meta_days", "corr_gap_meta_kept"] if args.metric != "corr" else [])]
    print("\n%s: %d of %d refits finished; kept epoch = %s\n" % (args.base, len(done), len(folds), rule))
    print(show.to_string(index=False, float_format=lambda v: "%+.4f" % v))
    print("\nmedian [IQR]: " + "  ".join("%s %.4f [%.4f, %.4f]" % (k, *v) for k, v in fold_summary.items()))
    print("ridge CORR by alpha (mean over folds): " + "  ".join(
        "%s %.5f" % (a, np.mean([r["corr_alpha_%g" % a] for r in baseline["folds"].values()]))
        for a in baseline["alphas"]))
    print("gradient flags: %s" % (grads["flags"] or "none"))
    print("picks: %s" % picks)
    for window in ("full", "covered"):
        print("\nstitched scores, %s window (n=%d):" % (window, scores["n_" + window]))
        print(pd.DataFrame(scores[window]).to_string(index=False, float_format=lambda v: "%+.4f" % v))
    if agreement:
        print("\nP6 agreement (kept epoch, meta days): same sign %d of %d folds, %d of %d months; "
              "per-day correlation of the CORR and %s gaps %+.3f over %d days" % (
                  agreement["folds_same_sign"], agreement["n_folds"], agreement["months_same_sign"],
                  agreement["n_months"], LABEL[args.metric], agreement["day_corr"], agreement["n_days"]))
        print("daily fit: %s gap = %+.5f + %.3f x CORR gap; per-day correlation with the meta model: %s" % (
            LABEL[args.metric], agreement["day_fit"]["intercept"], agreement["day_fit"]["slope"],
            ", ".join("%s %.3f" % kv for kv in agreement["meta_alignment"].items())))
        print(pd.DataFrame(agreement["months"]).to_string(index=False, float_format=lambda v: "%+.4f" % v))
    print("\nwrote %s" % ", ".join(figs.values()))

    if args.html:
        roles = {}
        for role, r in picks.items():                     # one panel per fold, every role named
            roles.setdefault(r, []).append(role)
        panels = []
        for r, names in roles.items():
            row = table.set_index("refit").loc[r]
            panels.append((" and ".join(names), r, "%s → %s, gap@kept %+.4f" % (
                row["held_start"], row["held_end"], row["gap_kept"]),
                os.path.join(ROOT, "%s_r%02d" % (args.base, r), "diagnostics.png")))
        notes = open(args.notes).read() if args.notes else None
        path = write_html(args.base, summary, table, scores, panels, figs, notes,
                          os.path.join(REPORTS, args.base + "_walkforward%s.html" % args.suffix))
        print("wrote %s" % path)
    return summary


if __name__ == "__main__":
    main()
