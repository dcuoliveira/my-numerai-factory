"""
E154 / E155: two independent zero-shot Chronos-2-Synth forecasting arms on Numerai Crypto.

    # on a GPU box, in the chronos venv (one arm per job; the arms share nothing but this file):
    HF_HOME=/scratch/danielco/hf HF_HUB_OFFLINE=1 PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES=0 \\
        /scratch/danielco/chronos2/venv/bin/python -m src.data_mani.c2synth_zeroshot run --arm return_rank [--smoke]
    ...                                                                     run --arm bin_forecast [--smoke]
    ...                                                                     audit --arm <arm>        # point-in-time check

Shared setup (identical for both arms):
  rounds    keys.parquet blocks 9-10 (validation, 260 rounds) and block 11 (test, 130 rounds), the house split
  universe  the symbols the panel holds on each round (300 per round); ucid from the frozen E145 panel
  model     autogluon/chronos-2-synth @ REV (frozen, fp32, eval). Every token is its own inference group, so
            group attention never mixes tokens; fp32 batch-invariance was measured at ~2e-6 in E145.
  context   64, 128 and 256 steps, all three computed; the choice is made on validation by the evaluator.
            Eligibility is fixed on the 64-step window, so all three contexts cover the same tokens.

Target (E142 probe + c2resid_paths): target_binned_return_20 on round C ~ close(C+2) -> close(C+30), i.e. the
daily log returns of days C+3..C+30. A round's features run through C-1 (the house one-day submit lag).

Arm A, return_rank (E154)
  input   daily spot log returns, clipped to [-1, 1] (c2resid_paths.returns), through day C-1. PREDECLARED.
  horizon 31 steps = days C..C+30. The interval start close(C+2) is not observed at C-1, so it is forecast too.
  score   sum of the q50 forecasts over days C+3..C+30. Chronos-2 has a quantile head (13 levels); its point
          forecast is the q50 median, not a mean, so this is a sum of per-day medians, not the median of the sum.
  eligible >= A_MIN finite returns in the last 64 days.          baseline: mom28 = sum of the last 28 returns.

Arm B, bin_forecast (E155)
  input   the official target_binned_return_20 labels of each token, on their ORIGINAL dates, on a daily calendar
          (weekends and missing dates are NaN, spacing preserved).
  availability  label of date d is usable at round C iff d <= C - LABEL_LAG days (close(d+30) realised by C-2,
          and Numerai serves labels ~31-32 days behind live). Asserted per round.
  horizon from the last available label date L to C: h = (C - L) days; the prediction at step h is kept (q50,
          continuous). h > B_HMAX -> skipped as stale.
  eligible >= B_MIN labels in the 64 days ending at L.           baselines: last label, mean of the last five.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.data_mani import c2resid_paths as RP

CKPT = "autogluon/chronos-2-synth"
REV = "3607918a9fd027d5c465d8213e46b98e2c041cea"          # the E143-E148 revision
KEYS = "results/reports/crypto/e132_feed_fillind/preds/keys.parquet"
PANEL = "data/crypto/v2.0-asof-e145/train.parquet"        # frozen 2026-11-06 copy; labels to 2026-10-06
TARGET = "target_binned_return_20"
VAL_BLOCKS, TEST_BLOCKS = (9, 10), (11,)
CTXS = (64, 128, 256)
ELIG_WIN = 64
A_MIN, A_H, A_SPAN = 32, 31, slice(3, 31)                 # days C..C+30; kept days C+3..C+30
MOM_WIN, MOM_MIN = 28, 20
LABEL_LAG, B_MIN, B_HMAX = 32, 20, 64
QIDX = (2, 6, 10)                                          # q10, q50, q90 of the 13 quantiles
CHUNK = 256
OUT = {"return_rank": "results/reports/crypto/e154_c2synth_return_rank",
       "bin_forecast": "results/reports/crypto/e155_c2synth_bin_forecast"}


class Model:
    def __init__(self, device):
        import torch
        from chronos import Chronos2Pipeline
        self.t, self.device = torch, device
        self.m = Chronos2Pipeline.from_pretrained(CKPT, revision=REV, device_map=device,
                                                  torch_dtype=torch.float32).model.float().eval()

    def q(self, ctx, n_steps):
        """[n, 3, n_steps] q10/q50/q90; row k of ctx is its own group."""
        t = self.t
        n_out = int(np.ceil(n_steps / 16))
        out = []
        for i in range(0, len(ctx), CHUNK):
            c = np.ascontiguousarray(ctx[i:i + CHUNK], "float32")
            with t.inference_mode():
                p = self.m(context=t.from_numpy(c).to(self.device),
                           group_ids=t.arange(len(c), device=self.device), num_output_patches=n_out).quantile_preds
            out.append(p[:, list(QIDX), :n_steps].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3, n_steps), "float32")


def rounds():
    k = pd.read_parquet(KEYS, columns=["symbol", "date", "block", "kind"])
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["block"].isin(VAL_BLOCKS + TEST_BLOCKS)].copy()
    k["split"] = np.where(k["block"].isin(TEST_BLOCKS), "test", "validation")
    p = pd.read_parquet(PANEL, columns=["symbol", "date", "ucid"]).reset_index()[["symbol", "date", "ucid"]]
    p["date"] = pd.to_datetime(p["date"])
    k = k.merge(p, on=["symbol", "date"], how="left", validate="1:1")
    return k.sort_values(["date", "symbol"]).reset_index(drop=True)


# ------------------------------------------------------------------ Arm A
def returns_inputs(cut=None):
    spot = pd.read_parquet(RP.SPOT)
    spot["date"] = pd.to_datetime(spot["date"])
    days, ucids, r = RP.returns(spot, cut)
    return pd.DatetimeIndex(days), list(ucids), r.astype("float32")


def a_round(model, days, ucids, r, g):
    """Arm A rows for the universe g of one round."""
    C = g["date"].iloc[0]
    day = int(np.searchsorted(days.values, (C - pd.Timedelta(days=RP.LAG)).to_datetime64(), side="right") - 1)
    col_of = {u: i for i, u in enumerate(ucids)}
    col = g["ucid"].astype(str).map(col_of).fillna(-1).astype(int).to_numpy()
    out = g[["date", "symbol", "ucid", "split"]].copy()
    out["input_last_day"] = days[day] if day >= 0 else pd.NaT
    w64 = RP.windows(r, col, np.full(len(col), day), ELIG_WIN)
    nfin = np.isfinite(w64).sum(1)
    out["n_obs64"] = nfin
    reason = np.where(col < 0, "no_spot", np.where(nfin < A_MIN, "thin_history", ""))
    out["skip_reason"] = reason
    elig = reason == ""
    out["eligible"] = elig
    w28 = RP.windows(r, col, np.full(len(col), day), MOM_WIN)
    mom = np.nansum(w28, 1)
    out["mom28"] = np.where(np.isfinite(w28).sum(1) >= MOM_MIN, mom, np.nan)
    for c in CTXS:
        for nm in ("raw", "q10", "q90"):
            out["%s_ctx%d" % (nm, c)] = np.nan
        if elig.any():
            W = RP.windows(r, col[elig], np.full(elig.sum(), day), c)
            q = model.q(W, A_H)
            s = q[:, :, A_SPAN].sum(-1)
            out.loc[elig, "q10_ctx%d" % c], out.loc[elig, "raw_ctx%d" % c], out.loc[elig, "q90_ctx%d" % c] = s[:, 0], s[:, 1], s[:, 2]
    return out


# ------------------------------------------------------------------ Arm B
def label_matrix(cut_date=None):
    """(calendar, symbols, Y [days, symbols]) of official labels on their original dates; optional truncation."""
    p = pd.read_parquet(PANEL, columns=["symbol", "date", TARGET]).reset_index()[["symbol", "date", TARGET]]
    p["date"] = pd.to_datetime(p["date"])
    if cut_date is not None:
        p = p[p["date"] <= cut_date]
    cal = pd.date_range(p["date"].min(), p["date"].max(), freq="D")
    Y = p.pivot_table(index="date", columns="symbol", values=TARGET, aggfunc="last").reindex(cal)
    return cal, list(Y.columns), Y.to_numpy("float32")


def last_label_idx(syms, Y, g, a_idx):
    """(column of each token in Y, calendar index of its last label on or before a_idx; -1 if none)."""
    col = g["symbol"].map({s: i for i, s in enumerate(syms)}).fillna(-1).astype(int).to_numpy()
    L = np.full(len(col), -1)
    if a_idx >= 0:
        fin = np.isfinite(Y[:a_idx + 1])
        for k in np.flatnonzero(col >= 0):
            nz = np.flatnonzero(fin[:, col[k]])
            if len(nz):
                L[k] = nz[-1]
    return col, L


def b_round(model, cal, syms, Y, g):
    C = g["date"].iloc[0]
    A = C - pd.Timedelta(days=LABEL_LAG)
    a_idx = int(np.searchsorted(cal.values, A.to_datetime64(), side="right") - 1)
    c_idx = (C - cal[0]).days                     # calendar step index of C (the calendar may end before C)
    col, L = last_label_idx(syms, Y, g, a_idx)
    out = g[["date", "symbol", "ucid", "split"]].copy()
    n = len(col)
    h = np.where(L >= 0, c_idx - L, -1)
    nobs = np.zeros(n, int)
    last5 = np.full(n, np.nan)
    last = np.full(n, np.nan)
    for k in range(n):
        if L[k] >= 0:
            seg = Y[max(L[k] - ELIG_WIN + 1, 0):L[k] + 1, col[k]]
            nobs[k] = np.isfinite(seg).sum()
            hist = Y[:L[k] + 1, col[k]]
            hv = hist[np.isfinite(hist)]
            last[k], last5[k] = hv[-1], hv[-5:].mean()
    reason = np.where(col < 0, "no_label_series", np.where(L < 0, "no_label_yet",
                      np.where(nobs < B_MIN, "thin_labels", np.where(h > B_HMAX, "stale_gap", ""))))
    elig = reason == ""
    out["last_label_date"] = [cal[i] if i >= 0 else pd.NaT for i in L]
    out["gap_steps"] = h
    out["n_obs64"] = nobs
    out["skip_reason"], out["eligible"] = reason, elig
    out["last_label"], out["mean5_label"] = last, last5
    assert (out.loc[elig, "last_label_date"] <= A).all(), "label availability violated"
    for c in CTXS:
        out["raw_ctx%d" % c] = np.nan
        out["q10_ctx%d" % c] = np.nan
        out["q90_ctx%d" % c] = np.nan
        if elig.any():
            ks = np.flatnonzero(elig)
            W = RP.windows(Y, col[ks], L[ks], c)
            hs = h[ks]
            q = model.q(W, int(hs.max()))
            pick = q[np.arange(len(ks)), :, hs - 1]
            out.loc[elig, "q10_ctx%d" % c], out.loc[elig, "raw_ctx%d" % c], out.loc[elig, "q90_ctx%d" % c] = pick[:, 0], pick[:, 1], pick[:, 2]
    return out


# ------------------------------------------------------------------ driver
def ranked(x):
    """Within-round ranks mapped strictly into (0, 1): (rank - 0.5) / n over the non-missing entries."""
    s = pd.Series(x)
    return (s.rank(method="average") - 0.5) / s.notna().sum()


def run(arm, smoke=False, device="cuda:0"):
    root = OUT[arm]
    os.makedirs(os.path.join(root, "parts"), exist_ok=True)
    U = rounds()
    dates = sorted(U["date"].unique())
    if smoke:
        dates = [dates[0], dates[len(dates) // 2], dates[-1]]
    cfg = dict(arm=arm, checkpoint=CKPT, revision=REV, contexts=CTXS, keys=KEYS, panel=PANEL, val_blocks=VAL_BLOCKS,
               test_blocks=TEST_BLOCKS, rounds=len(dates), host=os.uname().nodename, job=os.environ.get("INVOCATION_ID", ""),
               unit=os.environ.get("NMR_UNIT", ""), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               started=time.strftime("%F %T"))
    cfg.update(dict(input="daily spot log returns clipped [-1,1] through C-1", horizon_days="C..C+30", score="sum q50 over C+3..C+30",
                    elig="%d finite returns in last %d days" % (A_MIN, ELIG_WIN), baseline="mom28") if arm == "return_rank" else
               dict(input="target_binned_return_20 labels on original dates, daily calendar", label_lag_days=LABEL_LAG, max_gap=B_HMAX,
                    elig="%d labels in the 64 days ending at L" % B_MIN, baselines=["last_label", "mean5_label"]))
    json.dump(cfg, open(os.path.join(root, "config%s.json" % ("_smoke" if smoke else "")), "w"), indent=1, default=str)
    model = Model(device)
    if arm == "return_rank":
        days, ucids, r = returns_inputs()
    else:
        cal, syms, Y = label_matrix()
    t0, parts = time.time(), []
    for i, C in enumerate(dates):
        path = os.path.join(root, "parts", "%s%s.parquet" % (pd.Timestamp(C).date(), "_smoke" if smoke else ""))
        if os.path.exists(path) and not smoke:
            parts.append(pd.read_parquet(path))
            continue
        g = U[U["date"] == C]
        o = a_round(model, days, ucids, r, g) if arm == "return_rank" else b_round(model, cal, syms, Y, g)
        o.to_parquet(path, index=False)
        parts.append(o)
        if i % 10 == 0 or smoke:
            print("%s %s: %d/%d eligible  (%d/%d rounds, %.1f min)" % (arm, pd.Timestamp(C).date(), o["eligible"].sum(), len(o),
                                                                       i + 1, len(dates), (time.time() - t0) / 60), flush=True)
    P = pd.concat(parts, ignore_index=True)
    for c in CTXS:
        P["signal_ctx%d" % c] = P.groupby("date")["raw_ctx%d" % c].transform(ranked)
    P["coverage_flag"] = P["eligible"].astype(int)
    P.to_parquet(os.path.join(root, "predictions%s.parquet" % ("_smoke" if smoke else "")), index=False)
    cov = P.groupby(["split", "date"]).agg(universe=("symbol", "size"), eligible=("eligible", "sum")).reset_index()
    reasons = P[~P["eligible"]].groupby(["split", "date", "skip_reason"]).size().unstack(fill_value=0).reset_index()
    cov.merge(reasons, on=["split", "date"], how="left").fillna(0).to_csv(os.path.join(root, "coverage_by_round%s.csv" % ("_smoke" if smoke else "")), index=False)
    cfg.update(finished=time.strftime("%F %T"), minutes=round((time.time() - t0) / 60, 2), rows=len(P), eligible_rows=int(P["eligible"].sum()))
    json.dump(cfg, open(os.path.join(root, "config%s.json" % ("_smoke" if smoke else "")), "w"), indent=1, default=str)
    print("DONE %s: %d rows, %d eligible, %.1f min" % (arm, len(P), P["eligible"].sum(), cfg["minutes"]), flush=True)


def audit(arm, n=3):
    """Rebuild one arm's model inputs from data truncated at each round's cutoff and compare with the full-data inputs."""
    U = rounds()
    dates = sorted(U["date"].unique())
    pick = [dates[0], dates[len(dates) // 2], dates[-1]][:n]
    res = []
    for C in pick:
        g = U[U["date"] == C]
        if arm == "return_rank":
            cut = C - pd.Timedelta(days=RP.LAG)
            full, part = returns_inputs(), returns_inputs(cut)
            ws = []
            for days, ucids, r in (full, part):
                day = int(np.searchsorted(days.values, cut.to_datetime64(), side="right") - 1)
                col = g["ucid"].astype(str).map({u: i for i, u in enumerate(ucids)}).fillna(-1).astype(int).to_numpy()
                ws.append(RP.windows(r, col, np.full(len(col), day), max(CTXS)))
        else:
            A = C - pd.Timedelta(days=LABEL_LAG)
            ws, lds = [], []
            for cal, syms, Y in (label_matrix(), label_matrix(cut_date=A)):
                a_idx = int(np.searchsorted(cal.values, A.to_datetime64(), side="right") - 1)
                col, L = last_label_idx(syms, Y, g, a_idx)       # the windows the model reads end at L
                ws.append(RP.windows(Y, col, L, max(CTXS)))
                lds.append([cal[i] if i >= 0 else pd.NaT for i in L])
            assert lds[0] == lds[1], "last available label dates differ"
        a, b = ws
        same = np.isclose(a, b, rtol=0, atol=0) | (np.isnan(a) & np.isnan(b))
        res.append(dict(round=str(pd.Timestamp(C).date()), cells=int(same.size), differ=int((~same).sum())))
        print(res[-1], flush=True)
    os.makedirs(OUT[arm], exist_ok=True)
    json.dump(res, open(os.path.join(OUT[arm], "audit_inputs.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "audit"])
    ap.add_argument("--arm", choices=list(OUT), required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.arm, a.smoke)
    else:
        audit(a.arm)


if __name__ == "__main__":
    main()
