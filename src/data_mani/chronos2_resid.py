"""
E148: zero-shot Chronos-2-Synth FORECASTS of each coin's BTC+ETH beta-residual return path.

    # on deepfive, in the chronos venv:
    HF_HOME=/scratch/danielco/hf HF_HUB_OFFLINE=1 PYTHONPATH=<repo> \\
        /scratch/danielco/chronos2/venv/bin/python -m src.data_mani.chronos2_resid run [--smoke]
    ... audit --dates 2021-12-31 2023-06-30 2026-02-20

Paths come from src.data_mani.c2resid_paths (paths.npz, rows.parquet). Row (coin, D) sees days <= D-1.
Context CTX = 256 days; forecast 31 days (D .. D+30); kept: the sum over days D+3..D+30 (the
target's span, close(D+2) -> close(D+30)) of q10, q50 and q90, and the sum of q50 over D..D+30.

Arms (all zero-shot, frozen autogluon/chronos-2-synth at E143-E145's revision):
  raw  the coin's RAW log-return path alone                      -> does residualising the input matter?
  res  the coin's residual path alone                            -> the baseline C2 arm
  cov  residual path + BTC, ETH and the leave-one-out size-peer index as past covariates (one group
       of 4 series per coin; the covariates' own forecasts are discarded)
  grp  one group per (date, size bucket): every member coin's residual path + BTC + ETH, forecast
       jointly, so group attention reads the peers' paths themselves rather than their average
A target series needs MIN_OBS = 64 finite values in its context, else the row has no forecast.
"""
import argparse
import glob
import os
from time import time

import numpy as np
import pandas as pd

from src.data_mani import c2resid_paths as RP

OUT = RP.OUT
CKPT = "autogluon/chronos-2-synth"
REV = "3607918a9fd027d5c465d8213e46b98e2c041cea"
N_OUT, H = 2, 31                         # 2 output patches of 16 = 32 steps; days D..D+30 kept
SPAN = slice(3, 31)                      # days D+3 .. D+30
QIDX = (2, 6, 10)                        # q10, q50, q90 of the 13 quantiles
MIN_OBS = 64
ARMS = ("raw", "res", "cov", "grp")
CHUNK = {"raw": 256, "res": 256, "cov": 64}
PART = 100                               # dates per part file


class Model:
    def __init__(self, device):
        import torch
        from chronos import Chronos2Pipeline
        self.t, self.device = torch, device
        self.m = Chronos2Pipeline.from_pretrained(CKPT, revision=REV, device_map=device, torch_dtype=torch.float32).model.float().eval()

    def q(self, ctx, groups, take):
        """[len(take), 3, H] q10/q50/q90 of the series `take`."""
        t = self.t
        with t.inference_mode():         # fp32: batch-invariant to ~2e-6 (E145)
            out = self.m(context=t.from_numpy(np.ascontiguousarray(ctx, "float32")).to(self.device),
                         group_ids=t.from_numpy(groups.astype("int64")).to(self.device), num_output_patches=N_OUT).quantile_preds
        return out[t.from_numpy(np.asarray(take, "int64")).to(self.device)][:, list(QIDX), :H].float().cpu().numpy()


def summary(q):
    """[n, 4]: span sums of q10, q50, q90 and the full-horizon q50 sum."""
    s = q[:, :, SPAN].sum(-1)
    return np.c_[s, q[:, 1, :].sum(-1)].astype("float32")


def date_inputs(paths, rows):
    """Windows for the rows of one date: raw, res, peer [n, CTX]; btc, eth [CTX]."""
    r, e, ucids = paths["r"], paths["e"], list(paths["ucids"])
    col, day = rows["col"].to_numpy(), rows["day"].to_numpy()
    d = int(day[0])
    W = dict(raw=RP.windows(r, col, day), res=RP.windows(e, col, day), peer=RP.peer_windows(r, rows))
    bcol = np.array([ucids.index(RP.BTC), ucids.index(RP.ETH)])
    bw = RP.windows(r, bcol, np.array([d, d]))
    W["btc"], W["eth"] = bw[0], bw[1]
    return W


def forecast_date(model, paths, rows, arms=ARMS):
    """{arm: (row ids, [n, 4] summaries)} for the rows of one date."""
    rows = rows[(rows["col"] >= 0) & (rows["day"] >= 0)]
    out = {a: (np.zeros(0, "int64"), np.zeros((0, 4), "float32")) for a in arms}
    if not len(rows) or int(rows["day"].iloc[0]) < RP.CTX // 4:
        return out
    W = date_inputs(paths, rows)
    ids = rows.index.to_numpy()
    for a in arms:
        src = W["raw"] if a == "raw" else W["res"]
        ok = np.isfinite(src).sum(1) >= MIN_OBS
        if a == "grp":
            ok &= rows["bucket"].notna().to_numpy()
        if not ok.any():
            continue
        keep = np.flatnonzero(ok)
        got = []
        if a in ("raw", "res"):
            for i in range(0, len(keep), CHUNK[a]):
                k = keep[i:i + CHUNK[a]]
                got.append(model.q(src[k], np.arange(len(k)), np.arange(len(k))))
        elif a == "cov":
            for i in range(0, len(keep), CHUNK[a]):
                k = keep[i:i + CHUNK[a]]
                ctx = np.stack([np.stack([src[j], W["btc"], W["eth"], W["peer"][j]]) for j in k]).reshape(-1, RP.CTX)
                got.append(model.q(ctx, np.repeat(np.arange(len(k)), 4), np.arange(len(k)) * 4))
        else:
            b = rows["bucket"].to_numpy()[keep].astype(int)
            order, ctx, groups, take = [], [], [], []
            for g, bk in enumerate(np.unique(b)):
                mem = keep[b == bk]
                take += list(range(len(ctx), len(ctx) + len(mem)))
                ctx += [src[j] for j in mem] + [W["btc"], W["eth"]]
                groups += [g] * (len(mem) + 2)
                order += list(mem)
            got.append(model.q(np.stack(ctx), np.array(groups), take))
            keep = np.array(order)
        out[a] = (ids[keep], summary(np.concatenate(got)))
    return out


def load_paths(path=None):
    z = np.load(path or os.path.join(OUT, "paths.npz"), allow_pickle=True)
    return dict(days=z["days"], ucids=z["ucids"], r=z["r"], e=z["e"])


def run(device, smoke=False):
    paths = load_paths()
    rows = pd.read_parquet(os.path.join(OUT, "rows.parquet"))
    by_date = rows.groupby("date").indices
    dates = sorted(by_date)
    model = Model(device)
    if smoke:
        for D in (dates[400], dates[-30]):
            t0 = time()
            got = forecast_date(model, paths, rows.iloc[by_date[D]])
            print("smoke %s: %s  %.1fs" % (str(D)[:10], {a: len(v[0]) for a, v in got.items()}, time() - t0))
            for a, (i, s) in got.items():
                if len(i):
                    print("   %s q50 span sum: median %.4f, sd %.4f; q90-q10 %.3f" % (a, np.median(s[:, 1]), s[:, 1].std(), np.mean(s[:, 2] - s[:, 0])))
        return
    d = os.path.join(OUT, "parts")
    os.makedirs(d, exist_ok=True)
    t0 = time()
    for p in range(0, len(dates), PART):
        f = os.path.join(d, "part_%04d.npz" % (p // PART))
        if os.path.exists(f):
            continue
        acc = {a: ([], []) for a in ARMS}
        for D in dates[p:p + PART]:
            for a, (i, s) in forecast_date(model, paths, rows.iloc[by_date[D]]).items():
                acc[a][0].append(i)
                acc[a][1].append(s)
        np.savez(f + ".tmp.npz", **{"%s_rows" % a: np.concatenate(v[0]) for a, v in acc.items()},
                 **{"%s_s" % a: np.concatenate(v[1]) for a, v in acc.items()})
        os.replace(f + ".tmp.npz", f)
        print("part %d: dates %d-%d, %.1f min" % (p // PART, p, min(p + PART, len(dates)), (time() - t0) / 60), flush=True)
    print("done: %d part files" % len(glob.glob(os.path.join(d, "part_*.npz"))))


def run_audit(device, dates):
    """Rebuild the paths from spot truncated at D-1, forecast date D, compare with the full run."""
    import json
    full = merged()
    model = Model(device)
    spot = pd.read_parquet(RP.SPOT)
    spot["date"] = pd.to_datetime(spot["date"])
    res = []
    for D in dates:
        D = pd.Timestamp(D)
        days, ucids, r, e = RP.build(spot.copy(), cut=D - pd.Timedelta(days=RP.LAG))
        rows = RP.panel_rows(days=days, ucids=ucids)
        rows = rows[rows["date"] == D]
        got = forecast_date(model, dict(days=days, ucids=ucids, r=r, e=e), rows)
        for a, (i, s) in got.items():
            f = full[a].reindex(i)
            diff = np.abs(f.to_numpy() - s)
            res.append(dict(date=str(D.date()), arm=a, rows=len(i), missing_in_full=int(f.isna().any(axis=1).sum()),
                            max_abs_diff=float(np.nanmax(diff)) if len(i) else 0.0, cells=int(diff.size)))
            print(res[-1], flush=True)
    json.dump(res, open(os.path.join(OUT, "audit.json"), "w"), indent=1)


def merged():
    """{arm: DataFrame [row id -> 4 summaries]} from the part files."""
    out = {}
    parts = sorted(glob.glob(os.path.join(OUT, "parts", "part_*.npz")))
    for a in ARMS:
        i = np.concatenate([np.load(p)["%s_rows" % a] for p in parts])
        s = np.concatenate([np.load(p)["%s_s" % a] for p in parts])
        assert len(np.unique(i)) == len(i), a
        out[a] = pd.DataFrame(s, index=i, columns=["q10", "q50", "q90", "q50_all"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "audit"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dates", nargs="*", default=[])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.stage == "run":
        run(a.device, a.smoke)
    else:
        run_audit(a.device, a.dates)


if __name__ == "__main__":
    main()
