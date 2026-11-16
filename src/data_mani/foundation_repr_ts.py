"""
Time-series foundation-model representations of each coin's PRICE path -- rows 8-9.

    # on the GPU box, each in its own venv (neither model is in the main env):
    HF_HOME=/scratch/danielco/hf CUDA_VISIBLE_DEVICES=0 \\
        /scratch/danielco/tfm/venv/bin/python -m src.data_mani.foundation_repr_ts --model timesfm
    HF_HOME=/scratch/danielco/hf HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 \\
        /scratch/danielco/chronos2/venv/bin/python -m src.data_mani.foundation_repr_ts --model chronos2

E38 and E56 fed these models our LABEL history and found nothing. This feeds them
what features-foundation.md row 8 proposes instead: each coin's own log close
from the spot feed (data/crypto/spot), no labels at all. For every panel
(ucid, date D) it forecasts HORIZON daily steps ahead and keeps the 20-step-ahead
quantiles of log price RELATIVE to the last close -- a forecast distribution of
the 20-day return -- and, for Chronos-2, the encoder embedding (row 9; TimesFM
exposes none).

Point in time: the same rule as momentum_features.attach_signals_asof. Row D may
read only closes dated <= D - LAG_DAYS (the bar for day D is not final when
round D is submitted), and the newest of them must be within MAX_STALE_DAYS of D,
else the row is left out (neutral at load). No label is read, so no T-20 wall.

Embeddings are reduced here, not in the builder, to keep the cache small: the
basis is fitted ONCE on the burn-in dates [START, START + BURNIN) and applied to
every date -- past-only for every row any walk-forward reads (build_fmrep checks
the burn-in ends before the first training date).

Imports only numpy/pandas/torch at module scope: it runs in venvs that do not
carry the rest of the repo's stack.
"""

import argparse
import os
from time import time

import numpy as np
import pandas as pd

PANEL = os.path.join("data", "crypto", "v2.0-fenccausal", "train.parquet")
SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
CACHE_DIR = os.path.join("data", "crypto", "fmrep_cache")

START = "2023-05-15"       # = foundation_repr.START
BURNIN = 130               # dates in the embedding PCA burn-in
CONTEXT = 512              # TimesFM 1.0's maximum
HORIZON = 20
MIN_OBS = 64
LAG_DAYS = 1
MAX_STALE_DAYS = 4
N_PCS = 16
QUANTILES = (0.1, 0.5, 0.9)


def burnin_pca(x, dates, burnin_dates, k):
    """
    Project x onto the top-k principal axes of its burn-in rows.

    :param x: (rows, dim)
    :param dates: (rows,) date of each row
    :param burnin_dates: the dates whose rows fit the basis
    :return: (rows, k) float32
    """
    fit = np.isin(dates, burnin_dates)
    if fit.sum() <= k:
        raise ValueError("burn-in holds %d rows, need more than %d" % (fit.sum(), k))
    centre = x[fit].mean(axis=0)
    _, _, vt = np.linalg.svd(x[fit] - centre, full_matrices=False)
    return ((x - centre) @ vt[:k].T).astype("float32")


def spot_histories(spot):
    """{ucid: (dates as int days, log close)} for every coin with a usable close."""
    spot = spot.dropna(subset=["close"])
    spot = spot[spot["close"] > 0].sort_values(["ucid", "date"])
    days = spot["date"].to_numpy().astype("datetime64[D]").astype("int64")
    logs = np.log(spot["close"].to_numpy())
    out = {}
    for ucid, idx in spot.groupby("ucid").indices.items():
        out[ucid] = (days[idx], logs[idx])
    return out


def context_at(history, day):
    """
    The point-in-time context for panel date `day` (int days), or None.

    :return: log closes dated <= day - LAG_DAYS, newest within MAX_STALE_DAYS
    """
    dates, logs = history
    end = np.searchsorted(dates, day - LAG_DAYS, side="right")
    if end < MIN_OBS or day - dates[end - 1] > MAX_STALE_DAYS:
        return None
    return logs[max(0, end - CONTEXT):end]


class TimesFM:
    name = "tsfm"

    def __init__(self, device):
        import timesfm
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=("gpu" if device.startswith("cuda") else "cpu"),
                per_core_batch_size=64, horizon_len=HORIZON, context_len=CONTEXT,
                num_layers=20, model_dims=1280, input_patch_len=32, output_patch_len=128),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"))

    def quantiles(self, contexts):
        _, q = self.model.forecast(inputs=contexts, freq=[0] * len(contexts))
        q = np.asarray(q)[:, HORIZON - 1, :]       # column 0 is the mean, 1..9 = q10..q90
        return q[:, [1, 5, 9]]

    def embed(self, contexts):
        return None


class Chronos2:
    name = "chr"

    def __init__(self, device):
        import torch
        from chronos import Chronos2Pipeline
        self.torch = torch
        self.pipe = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map=device, torch_dtype=torch.float32)

    def quantiles(self, contexts):
        series = [self.torch.tensor(c, dtype=self.torch.float32) for c in contexts]
        q, _ = self.pipe.predict_quantiles(series, prediction_length=HORIZON,
                                           quantile_levels=list(QUANTILES))
        return np.stack([np.asarray(x[0, HORIZON - 1, :].cpu()) for x in q])

    def embed(self, contexts):
        series = [self.torch.tensor(c, dtype=self.torch.float32) for c in contexts]
        embeds, _ = self.pipe.embed(series, context_length=CONTEXT)
        # (1, patches + 2, d) per series -> mean over every token, [REG] included
        return np.stack([np.asarray(e[0].float().mean(dim=0)) for e in embeds])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=("timesfm", "chronos2"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="smoke test: stop after this many dates (no PCA, writes *_smoke)")
    args = parser.parse_args()

    panel = pd.read_parquet(PANEL, columns=["ucid", "date"]).reset_index()
    panel = panel[panel["date"] >= pd.Timestamp(START)]
    histories = spot_histories(pd.read_parquet(SPOT, columns=["ucid", "date", "close"]))
    model = (TimesFM if args.model == "timesfm" else Chronos2)(args.device)
    dates = np.sort(panel["date"].unique())
    if args.max_dates:
        dates = dates[:args.max_dates]
    print("%s: %d panel dates %s .. %s, %d coins with spot"
          % (args.model, len(dates), dates[0], dates[-1], len(histories)), flush=True)

    keys, qs, embs, started = [], [], [], time()
    for i, date in enumerate(dates):
        day = int(np.datetime64(date, "D").astype("int64"))
        ucids, contexts = [], []
        for ucid in panel.loc[panel["date"] == date, "ucid"]:
            history = histories.get(ucid)
            ctx = None if history is None else context_at(history, day)
            if ctx is not None:
                ucids.append(ucid)
                contexts.append(ctx)
        if not contexts:
            continue
        last = np.array([c[-1] for c in contexts])
        qs.append(model.quantiles(contexts) - last[:, None])
        e = model.embed(contexts)
        if e is not None:
            embs.append(e.astype("float32"))
        keys.append(pd.DataFrame({"ucid": ucids, "date": date}))
        if i % 50 == 0:
            print("  %s: %d coins, %.1f min" % (pd.Timestamp(date).date(), len(contexts),
                                                 (time() - started) / 60), flush=True)

    out = pd.concat(keys, ignore_index=True)
    q = np.concatenate(qs)
    for j, level in enumerate(QUANTILES):
        out["%s_q%02d" % (model.name, round(100 * level))] = q[:, j].astype("float32")
    if embs and args.max_dates:
        print("smoke: embedding dim %d, not reduced" % embs[0].shape[1])
    elif embs:
        pcs = burnin_pca(np.concatenate(embs), out["date"].to_numpy(), dates[:BURNIN], N_PCS)
        for j in range(N_PCS):
            out["%s_emb_pc%02d" % (model.name, j)] = pcs[:, j]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "ts_%s%s.parquet" % (args.model, "_smoke" if args.max_dates else ""))
    out.to_parquet(path, index=False)
    print("wrote %s: %d rows x %d cols in %.1f min" % (path, len(out), out.shape[1],
                                                       (time() - started) / 60), flush=True)


if __name__ == "__main__":
    main()
