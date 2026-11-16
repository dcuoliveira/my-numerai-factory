"""
Chronos-2 zero-shot with the feature cross-section as covariates (study E56).

    # on the GPU box, in the chronos2 venv:
    HF_HOME=/scratch/danielco/hf HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=1 \
        /scratch/danielco/chronos2/venv/bin/python -m src.analysis.chronos2_covariates \
        --run-name e56_fenccausal

The missing cell of the foundation-model arc. Univariate time-series FMs saw only
the target's own history and scored ~0 (TimesFM E38 +0.012, TEMPO E40 -0.017); the
TABULAR FM on the feature cross-section worked (TabPFN E19 CORR 0.145). Chronos-2
(amazon/chronos-2, ~120M, arXiv:2510.15821) is the first time-series FM that ingests
EXOGENOUS COVARIATES via its "group attention", so it can finally use the features.

Per scored era T we make ONE panel forecast over every window-complete symbol, so
Chronos-2's group attention shares information ACROSS the cross-section of symbols,
and each symbol's target is conditioned on ITS feature history:

  * context (df):    every symbol's (target + 44 features) up to the leak wall
                     b = T - EMBARGO. target_binned_return_20 at t is a 20-day
                     forward return, unobservable until t+20, so context targets
                     stop at b. Timestamps are a contiguous OWN-STEP index
                     (cumcount) so each series is regular -- Chronos-2 needs a
                     gapless grid and cannot infer a frequency from the gappy
                     calendar (same own-step convention E38/E40 use).
  * future (fut):    features for eras b+1..T as KNOWN-FUTURE covariates. Features
                     at date t are observed at t (only the TARGET is embargoed), so
                     feeding features up to T lets the CURRENT cross-section
                     condition the forecast without leaking the label.
  * prediction_len 20, and the era-T step (the last horizon step, conditioned on
    the era-T features) is the signal.

Only symbols present on EVERY era of [b..T] enter (last context = b, exactly 20
clean future rows) -- ~86% of the cross-section per era; coverage is printed.
Output is the pipeline's forecast frame, scored by timesfm_score --model chronos2.
"""

import argparse
import os

import numpy as np
import pandas as pd

EMBARGO = 20
HORIZON = 20
MIN_CONTEXT = 32
CKPT = "amazon/chronos-2"


def load_panel(data_version, target):
    df = pd.read_parquet("data/%s/train.parquet" % data_version)
    feats = [c for c in df.columns if c.startswith("feature_")]
    df = df.dropna(subset=[target]).copy()
    df[feats] = df[feats].fillna(0.5)
    gd = np.sort(df["date"].unique())
    pos = {d: i for i, d in enumerate(gd)}
    df["era"] = df["date"].map(pos).astype(int)
    return df.sort_values(["ucid", "era"]), feats, gd, pos


def scored_eras(match_run, pos):
    model, run = match_run.split("/")
    d = pd.to_datetime(pd.read_csv("results/per_period/crypto/%s/%s.csv"
                                   % (model, run))["date"]).values
    return sorted(pos[x] for x in d if x in pos)


def forecast_era(pipe, df, feats, target, T, ctx_len):
    """One panel forecast for scored era T; returns symbol/prediction/target or None."""
    b = T - EMBARGO
    win = list(range(b, T + 1))                              # [b .. T], 21 eras
    g = df[df["era"].isin(win)]
    cnt = g.groupby("ucid")["era"].nunique()
    present = set(cnt[cnt == len(win)].index)                # window-complete symbols
    if not present:
        return None
    sub = df[(df["ucid"].isin(present)) & (df["era"] <= T) & (df["era"] >= b - ctx_len)].copy()
    sub["step"] = sub.groupby("ucid").cumcount()             # contiguous per-series index
    ok = sub[sub["era"] <= b].groupby("ucid").size()
    ok = set(ok[ok >= MIN_CONTEXT].index)
    sub = sub[sub["ucid"].isin(ok)]
    if sub.empty:
        return None
    ctx = (sub[sub["era"] <= b][["ucid", "step", target, *feats]]
           .rename(columns={"ucid": "symbol", target: "target"}).reset_index(drop=True))
    fut = (sub[sub["era"] > b][["ucid", "step", *feats]]
           .rename(columns={"ucid": "symbol"}).reset_index(drop=True))
    out = pipe.predict_df(ctx, future_df=fut, prediction_length=HORIZON,
                          quantile_levels=[0.5], id_column="symbol",
                          timestamp_column="step", target="target")
    ts = out.columns[1]                                      # timestamp column
    out = out.reset_index(drop=True)
    last = out.groupby("symbol")[ts].transform("max")        # era-T horizon step
    sig = out[out[ts] == last][["symbol", "predictions"]].reset_index(drop=True)
    sig["symbol"] = sig["symbol"].astype(str)
    tgt = (df[df["era"] == T][["ucid", target]]
           .rename(columns={"ucid": "symbol", target: "tgt"}).reset_index(drop=True))
    tgt["symbol"] = tgt["symbol"].astype(str)
    return sig.merge(tgt, on="symbol", how="inner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--match-run", default="tabpfn/e19_fenccausal")
    ap.add_argument("--run-name", default="e56_fenccausal")
    ap.add_argument("--ctx-len", type=int, default=252)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from chronos import Chronos2Pipeline

    df, feats, gd, pos = load_panel(args.data_version, args.target)
    eras = scored_eras(args.match_run, pos)
    if args.limit:
        eras = eras[:args.limit]
    print("chronos-2: %d features, %d scored eras, ctx_len %d"
          % (len(feats), len(eras), args.ctx_len), flush=True)

    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(CKPT, device_map=dev, torch_dtype=torch.float32)

    frames, cov, skipped = [], [], 0
    for i, T in enumerate(eras):
        try:
            m = forecast_era(pipe, df, feats, args.target, T, args.ctx_len)
        except Exception as exc:                             # OOM/etc: skip the era, keep going
            torch.cuda.empty_cache()
            print("  era %d/%d SKIPPED: %s" % (i + 1, len(eras), str(exc)[:120]), flush=True)
            skipped += 1
            continue
        if m is None or m.empty:
            continue
        m["date"] = gd[T]
        frames.append(m)
        cov.append(len(m))
        if (i + 1) % 20 == 0:
            print("  era %d/%d (%s)  symbols %d" % (i + 1, len(eras), str(gd[T])[:10], len(m)),
                  flush=True)
    if skipped:
        print("  %d eras skipped" % skipped, flush=True)

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"tgt": args.target, "predictions": "prediction"})
    out = out[["symbol", "date", args.target, "prediction"]]
    dest = "results/forecast/crypto/chronos2/%s.csv" % args.run_name
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False)
    print("wrote %s  (%d rows, mean %d symbols/era)" % (dest, len(out), int(np.mean(cov))), flush=True)
    print("score: python -m src.analysis.timesfm_score --model chronos2 --run-name %s"
          % args.run_name, flush=True)


if __name__ == "__main__":
    main()
