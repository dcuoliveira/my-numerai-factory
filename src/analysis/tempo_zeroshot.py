"""
Zero-shot univariate inference with TEMPO (Cao et al., ICLR 2024) on Numerai crypto.

    # in the tempo venv on the GPU box (HF cache on scratch):
    HF_HOME=/scratch/danielco/hf CUDA_VISIBLE_DEVICES=1 \
        /scratch/danielco/tempo/venv/bin/python -m src.analysis.tempo_zeroshot \
        --run-name e40_target

The exact sibling of src.analysis.timesfm_zeroshot (E38): the SAME per-symbol,
univariate, leak-free zero-shot design, with TEMPO swapped in for TimesFM so the
two are a controlled model comparison. TEMPO is a GPT-2-backbone generative
time-series foundation model (~80M) that STL-decomposes each series into
trend/seasonal/residual, RevIN-normalises each, patch-tokenises and reads a soft
prompt pool -- all inside predict(); we owe it only one raw finite 1-D series per
call, most-recent-last. Pretrained zero-shot weights: Melady/TEMPO,
TEMPO-80M_v1.pth (HuggingFace, apache-2.0).

The leak boundary is IDENTICAL to E38 and is the one thing that must be right:
target_binned_return_20 at date t is not observable until t+20, so a prediction
at t may use only the symbol's targets whose forward window has closed by t --
dates t' with t' + 20 periods <= t, i.e. up to the date 20 positions before t on
the global grid. Context is the symbol's targets up to that boundary; TEMPO then
forecasts HORIZON own-steps and the signal is the mean of that horizon.

Two knobs differ from E38, both TEMPO's: CONTEXT_LEN 336 (TEMPO's seq_len; it
truncates longer, pads shorter) rather than 512, and per-series predict() rather
than TimesFM's batched forecast(). Output: the same forecast frame the pipeline
writes, scored by src.analysis.timesfm_score --model tempo in the base env.
"""

import argparse
import os

import numpy as np
import pandas as pd

EMBARGO = 20            # global-period embargo == target horizon; the leak wall
CONTEXT_LEN = 336      # TEMPO 80M seq_len
HORIZON = 20           # forecast this many own-steps; signal = mean
MIN_CONTEXT = 32       # too little history to forecast anything meaningful
CKPT_REPO = "Melady/TEMPO"
CKPT_FILE = "TEMPO-80M_v1.pth"


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


def load_tempo(device):
    import torch
    from tempo.models.TEMPO import TEMPO
    dev = torch.device("cuda:0" if (device == "cuda" and torch.cuda.is_available())
                       else "cpu")
    model = TEMPO.load_pretrained_model(
        device=dev, repo_id=CKPT_REPO, filename=CKPT_FILE,
        cache_dir="/scratch/danielco/tempo/ckpts")
    return model


def forecast_signal(model, contexts):
    """Zero-shot forecast each context HORIZON steps; signal = mean of horizon.

    TEMPO's predict() is single-series, so this loops. Contexts are finite (bin
    targets after dropna), most-recent-last, trimmed to CONTEXT_LEN upstream.
    """
    import torch
    out = np.empty(len(contexts), dtype=np.float64)
    with torch.no_grad():
        for i, ctx in enumerate(contexts):
            yhat = model.predict(np.asarray(ctx, dtype=np.float32),
                                 pred_length=HORIZON)
            out[i] = float(np.asarray(yhat).ravel()[:HORIZON].mean())
            if (i + 1) % 2000 == 0:
                print("  forecast %d/%d" % (i + 1, len(contexts)), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="crypto/v2.0")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--match-run", default="tabpfn/e19_fenccausal",
                    help="score on this arm's dates for a matched comparison")
    ap.add_argument("--run-name", default="e40_target")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="forecast only the first N tasks (timing smoke test)")
    args = ap.parse_args()

    df = load_panel(args.data_version, args.target)
    scored = scored_dates(args.match_run)
    syms, dates, tgts, ctxs = build_tasks(df, args.target, scored)
    if args.limit:
        syms, dates, tgts, ctxs = (syms[:args.limit], dates[:args.limit],
                                   tgts[:args.limit], ctxs[:args.limit])
    print("tasks: %d  (%d symbols, %d scored dates)"
          % (len(syms), len(set(syms)), len(set(dates))), flush=True)

    model = load_tempo(args.device)
    preds = forecast_signal(model, ctxs)

    out = pd.DataFrame({"symbol": syms, "date": dates,
                        args.target: tgts, "prediction": preds})
    dest = "results/forecast/crypto/tempo/%s.csv" % args.run_name
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False)
    print("wrote %s  (%d rows)" % (dest, len(out)), flush=True)


if __name__ == "__main__":
    main()
