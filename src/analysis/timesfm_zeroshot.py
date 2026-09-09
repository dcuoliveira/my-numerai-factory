"""
Zero-shot univariate inference with TimesFM (Das et al. 2023) on Numerai crypto.

    # in the scratch venv on the GPU box (HF cache on scratch):
    HF_HOME=/scratch/danielco/hf CUDA_VISIBLE_DEVICES=1 \
        /scratch/danielco/tfm/venv/bin/python -m src.analysis.timesfm_zeroshot \
        --run-name e25_target

TimesFM is a UNIVARIATE time-series foundation model: one series' history in, its
continuation out, no per-dataset training. This runs it ONE SYMBOL AT A TIME on
each coin's own `target_binned_return_20` history and uses the forecast as the
Numerai signal. It is deliberately OUTSIDE the run_forecast/MODELS pipeline,
which is row-wise; TimesFM is a per-symbol sequence forecaster, the same reason
the LSTM lives in its own tree.

Model: the ORIGINAL TimesFM 1.0 (200M), loaded from the vendored
`google/timesfm-1.0-200m-pytorch` checkpoint through the v1.2.6 torch backend
(TimesFmTorch) -- no JAX. See the tfm venv on the GPU box.

------------------------------------------------------------------------------
The leak boundary -- the one thing that must be right
------------------------------------------------------------------------------

target_binned_return_20 at date t summarises [t, t+20] and is not observable
until t+20. So a prediction at date t may use ONLY the symbol's targets whose
forward window has CLOSED by t: dates t' with t' + 20 periods <= t. On the global
date grid (business-daily, so a period is one row) that boundary is the date 20
positions before t. Context is the symbol's targets up to that boundary date;
nothing later. TimesFM then forecasts HORIZON of the symbol's own steps and the
signal is the mean of that horizon -- its estimate of where the coin's
forward-return series is heading across the ~20-period gap to t. The series
overlaps ~95% period to period, so the exact step is not critical and the mean is
the robust summary; for a symbol present on every date HORIZON=20 own-steps lands
on t, and for a gappy symbol it overshoots in calendar time (documented, not
hidden).

Output: results/forecast/crypto/timesfm/<run>.csv with symbol,date,target,
prediction -- the shape the pipeline writes, so scoring (a separate step in the
base env) reuses per_period_corr / per_period_bmc.
"""

import argparse
import os

import numpy as np
import pandas as pd

EMBARGO = 20            # global-period embargo == target horizon; the leak wall
CONTEXT_LEN = 512      # TimesFM 1.0's max context
HORIZON = 20           # forecast this many of the symbol's own steps
MIN_CONTEXT = 32       # too little history to forecast anything meaningful
BATCH = 1024           # series per forecast() call


def load_panel(data_version, target):
    df = pd.read_parquet("data/%s/train.parquet" % data_version,
                         columns=["ucid", "date", target])
    return df.dropna(subset=[target]).sort_values(["ucid", "date"])


def scored_dates(match_run):
    """The scored window of an existing arm, so the comparison is on one grid."""
    model, run = match_run.split("/") if "/" in match_run else ("tabpfn", match_run)
    pp = pd.read_csv("results/per_period/crypto/%s/%s.csv" % (model, run))
    return set(pd.to_datetime(pp["date"]).values)


def build_tasks(df, target, scored):
    """One task per (symbol, scored-date) with a leak-free causal context."""
    global_dates = np.sort(df["date"].unique())
    pos = {d: i for i, d in enumerate(global_dates)}
    syms, dates, tgts, ctxs = [], [], [], []
    for sym, g in df.groupby("ucid", sort=False):
        dts = g["date"].values
        ys = g[target].to_numpy(dtype=np.float32)
        for i in range(len(dts)):
            t = dts[i]
            if t not in scored:
                continue
            b_pos = pos[t] - EMBARGO                  # leak wall on the global grid
            if b_pos < 0:
                continue
            ctx = ys[dts <= global_dates[b_pos]]      # resolved history only
            if len(ctx) < MIN_CONTEXT:
                continue
            syms.append(sym); dates.append(t)
            tgts.append(ys[i]); ctxs.append(ctx[-CONTEXT_LEN:])
    return syms, dates, tgts, ctxs


def load_timesfm(device):
    """The original TimesFM 1.0 (200M) via the pytorch checkpoint (no JAX)."""
    import timesfm
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=("gpu" if device == "cuda" else "cpu"),
            per_core_batch_size=32,
            horizon_len=HORIZON, context_len=CONTEXT_LEN,
            num_layers=20, model_dims=1280,
            input_patch_len=32, output_patch_len=128,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )


def forecast_signal(tfm, contexts):
    """Zero-shot forecast each context HORIZON steps; signal = mean of horizon."""
    out = np.empty(len(contexts), dtype=np.float64)
    for start in range(0, len(contexts), BATCH):
        chunk = contexts[start:start + BATCH]
        point, _ = tfm.forecast(inputs=chunk, freq=[0] * len(chunk))
        out[start:start + len(chunk)] = np.asarray(point)[:, :HORIZON].mean(axis=1)
        print("  forecast %d/%d" % (min(start + BATCH, len(contexts)),
                                    len(contexts)), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="crypto/v2.0")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--match-run", default="tabpfn/e19_fenccausal",
                    help="score on this arm's dates for a matched comparison")
    ap.add_argument("--run-name", default="e25_target")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df = load_panel(args.data_version, args.target)
    scored = scored_dates(args.match_run)
    syms, dates, tgts, ctxs = build_tasks(df, args.target, scored)
    print("tasks: %d  (%d symbols, %d scored dates)"
          % (len(syms), len(set(syms)), len(set(dates))), flush=True)

    tfm = load_timesfm(args.device)
    preds = forecast_signal(tfm, ctxs)

    out = pd.DataFrame({"symbol": syms, "date": dates,
                        args.target: tgts, "prediction": preds})
    dest = "results/forecast/crypto/timesfm/%s.csv" % args.run_name
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False)
    print("wrote %s  (%d rows)" % (dest, len(out)), flush=True)


if __name__ == "__main__":
    main()
