"""
Generic zero-shot TSFM runner for the E154/E155 design, any model behind src/data_mani/tsfm_adapters/CONTRACT.md.

    # in the model's own environment (see each adapter's docstring for its python and env vars):
    PYTHONPATH=<repo> <env python> -m src.data_mani.tsfm_zeroshot run --model tempopfn --arm return_rank --out <dir> [--smoke]

Everything that defines the experiment is imported from src.data_mani.c2synth_zeroshot, unchanged: the rounds
(keys.parquet blocks 9-10 validation, 11 test), the universe (panel symbols per round), the cutoff (data through
C-1), the target interval (days C+3..C+30), the label availability rule (d <= C-32), the eligibility rules (fixed on
the 64-step window, so every model and every context covers the same tokens) and the input windows. Only the
forecast call differs:

  Arm A  return_rank   raw score = sum of the model's point forecast over steps 4..31 (days C+3..C+30) from
                       daily log returns through C-1.
  Arm B  bin_forecast  raw score = the model's point forecast at step h = C - L from the label series ending at L.

Contexts: {64, 128, 256} intersected with the model's max_context. A row the model returns NaN for keeps its
eligibility flag but gets skip_reason "model_nan" and no score (counted in coverage).
"""
import argparse
import importlib
import json
import os
import time

import numpy as np
import pandas as pd

from src.data_mani import c2resid_paths as RP
from src.data_mani import c2synth_zeroshot as Z


def contexts(fc):
    return tuple(c for c in Z.CTXS if fc.max_context is None or c <= fc.max_context)


def a_round(fc, ctxs, days, ucids, r, g):
    C = g["date"].iloc[0]
    day = int(np.searchsorted(days.values, (C - pd.Timedelta(days=RP.LAG)).to_datetime64(), side="right") - 1)
    col = g["ucid"].astype(str).map({u: i for i, u in enumerate(ucids)}).fillna(-1).astype(int).to_numpy()
    out = g[["date", "symbol", "ucid", "split"]].copy()
    w64 = RP.windows(r, col, np.full(len(col), day), Z.ELIG_WIN)
    nfin = np.isfinite(w64).sum(1)
    reason = np.where(col < 0, "no_spot", np.where(nfin < Z.A_MIN, "thin_history", ""))
    elig = reason == ""
    out["n_obs64"], out["eligible"] = nfin, elig
    w28 = RP.windows(r, col, np.full(len(col), day), Z.MOM_WIN)
    out["mom28"] = np.where(np.isfinite(w28).sum(1) >= Z.MOM_MIN, np.nansum(w28, 1), np.nan)
    last = np.full(elig.sum(), np.datetime64(days[day].date(), "D"))
    model_nan = np.zeros(len(col), bool)
    for c in ctxs:
        out["raw_ctx%d" % c] = np.nan
        if elig.any():
            W = RP.windows(r, col[elig], np.full(elig.sum(), day), c)
            f = fc.forecast(W, Z.A_H, last)
            s = f[:, Z.A_SPAN].sum(-1)
            s[~np.isfinite(f[:, Z.A_SPAN]).all(-1)] = np.nan
            out.loc[elig, "raw_ctx%d" % c] = s
            model_nan[np.flatnonzero(elig)[~np.isfinite(s)]] = True
    out["skip_reason"] = np.where(model_nan & elig, "model_nan", reason)
    return out


def b_round(fc, ctxs, cal, syms, Y, g):
    C = g["date"].iloc[0]
    A = C - pd.Timedelta(days=Z.LABEL_LAG)
    a_idx = int(np.searchsorted(cal.values, A.to_datetime64(), side="right") - 1)
    c_idx = (C - cal[0]).days
    col, L = Z.last_label_idx(syms, Y, g, a_idx)
    out = g[["date", "symbol", "ucid", "split"]].copy()
    n = len(col)
    h = np.where(L >= 0, c_idx - L, -1)
    nobs, last, last5 = np.zeros(n, int), np.full(n, np.nan), np.full(n, np.nan)
    for k in range(n):
        if L[k] >= 0:
            nobs[k] = np.isfinite(Y[max(L[k] - Z.ELIG_WIN + 1, 0):L[k] + 1, col[k]]).sum()
            hv = Y[:L[k] + 1, col[k]]
            hv = hv[np.isfinite(hv)]
            last[k], last5[k] = hv[-1], hv[-5:].mean()
    reason = np.where(col < 0, "no_label_series", np.where(L < 0, "no_label_yet",
                      np.where(nobs < Z.B_MIN, "thin_labels", np.where(h > Z.B_HMAX, "stale_gap", ""))))
    elig = reason == ""
    out["last_label_date"] = [cal[i] if i >= 0 else pd.NaT for i in L]
    out["gap_steps"], out["n_obs64"], out["eligible"] = h, nobs, elig
    out["last_label"], out["mean5_label"] = last, last5
    assert (out.loc[elig, "last_label_date"] <= A).all(), "label availability violated"
    model_nan = np.zeros(n, bool)
    for c in ctxs:
        out["raw_ctx%d" % c] = np.nan
        if elig.any():
            ks = np.flatnonzero(elig)
            W = RP.windows(Y, col[ks], L[ks], c)
            hs = h[ks]
            f = fc.forecast(W, int(hs.max()), np.array([np.datetime64(cal[i].date(), "D") for i in L[ks]]))
            v = f[np.arange(len(ks)), hs - 1]
            out.loc[elig, "raw_ctx%d" % c] = v
            model_nan[ks[~np.isfinite(v)]] = True
    out["skip_reason"] = np.where(model_nan & elig, "model_nan", reason)
    return out


def run(model, arm, root, smoke=False, device="cuda:0"):
    fc = importlib.import_module("src.data_mani.tsfm_adapters." + model).Forecaster(device)
    ctxs = contexts(fc)
    sfx = "_smoke" if smoke else ""
    os.makedirs(os.path.join(root, "parts"), exist_ok=True)
    U = Z.rounds()
    dates = sorted(U["date"].unique())
    if smoke:
        dates = [dates[0], dates[len(dates) // 2], dates[-1]]
    cfg = dict(model=model, arm=arm, checkpoint=fc.checkpoint, point=fc.point, handles_nan=fc.handles_nan,
               max_context=fc.max_context, max_horizon=fc.max_horizon, contexts=ctxs, keys=Z.KEYS, panel=Z.PANEL,
               val_blocks=Z.VAL_BLOCKS, test_blocks=Z.TEST_BLOCKS, rounds=len(dates), host=os.uname().nodename,
               job=os.environ.get("INVOCATION_ID", ""), unit=os.environ.get("NMR_UNIT", ""),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""), started=time.strftime("%F %T"),
               score=("sum of point forecast over days C+3..C+30 from daily log returns through C-1" if arm == "return_rank"
                      else "point forecast at step C-L from labels on original dates, label d usable iff d <= C-%d" % Z.LABEL_LAG))
    json.dump(cfg, open(os.path.join(root, "config%s.json" % sfx), "w"), indent=1, default=str)
    if arm == "return_rank":
        days, ucids, r = Z.returns_inputs()
    else:
        cal, syms, Y = Z.label_matrix()
    t0, parts = time.time(), []
    for i, C in enumerate(dates):
        path = os.path.join(root, "parts", "%s%s.parquet" % (pd.Timestamp(C).date(), sfx))
        if os.path.exists(path) and not smoke:
            parts.append(pd.read_parquet(path))
            continue
        g = U[U["date"] == C]
        o = a_round(fc, ctxs, days, ucids, r, g) if arm == "return_rank" else b_round(fc, ctxs, cal, syms, Y, g)
        o.to_parquet(path, index=False)
        parts.append(o)
        if i % 10 == 0 or smoke:
            print("%s/%s %s: %d/%d eligible, %d model_nan (%d/%d rounds, %.1f min)"
                  % (model, arm, pd.Timestamp(C).date(), o["eligible"].sum(), len(o), (o["skip_reason"] == "model_nan").sum(),
                     i + 1, len(dates), (time.time() - t0) / 60), flush=True)
    P = pd.concat(parts, ignore_index=True)
    for c in ctxs:
        P["signal_ctx%d" % c] = P.groupby("date")["raw_ctx%d" % c].transform(Z.ranked)
    P["coverage_flag"] = (P["eligible"] & (P["skip_reason"] != "model_nan")).astype(int)
    P.to_parquet(os.path.join(root, "predictions%s.parquet" % sfx), index=False)
    cov = P.groupby(["split", "date"]).agg(universe=("symbol", "size"), eligible=("coverage_flag", "sum")).reset_index()
    reasons = P[P["coverage_flag"] == 0].groupby(["split", "date", "skip_reason"]).size().unstack(fill_value=0).reset_index()
    cov.merge(reasons, on=["split", "date"], how="left").fillna(0).to_csv(os.path.join(root, "coverage_by_round%s.csv" % sfx), index=False)
    cfg.update(finished=time.strftime("%F %T"), minutes=round((time.time() - t0) / 60, 2), rows=len(P),
               covered_rows=int(P["coverage_flag"].sum()), model_nan_rows=int((P["skip_reason"] == "model_nan").sum()))
    json.dump(cfg, open(os.path.join(root, "config%s.json" % sfx), "w"), indent=1, default=str)
    print("DONE %s/%s: %d rows, %d covered, %.1f min" % (model, arm, len(P), cfg["covered_rows"], cfg["minutes"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=["return_rank", "bin_forecast"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    run(a.model, a.arm, a.out, a.smoke, a.device)


if __name__ == "__main__":
    main()
