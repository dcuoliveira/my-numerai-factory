"""
TabPFN representations, generated CAUSALLY over the panel -- the GPU half of E109.

    CUDA_VISIBLE_DEVICES=0 python -m src.data_mani.foundation_repr --pass main --device cuda:0

experiments/features-foundation.md lists what TabPFN v2 could hand a simpler model
(rows 1-7). This writes each of them per (symbol, date) so build_fmrep can merge
them onto a base dataset as ordinary feature columns. One PASS is one TabPFN
configuration:

    main            fenccausal (44)   rows 1-5: mean/median/mode, quantiles, std,
                                      bin probabilities, per-estimator means,
                                      192-d test-token embedding
    head_<block>    one feature block row 6: the mean of TabPFN on tenc, interact,
                                      size, price or raw alone
    ctx030, ctx390  fenccausal (44)   row 7: the mean under a 30- or 390-date
                                      context window instead of 130

------------------------------------------------------------------------------
Causality -- the T-20 wall, by construction
------------------------------------------------------------------------------
The panel is cut into REFRESH-date blocks. For a block starting at date index b
the context is the WINDOW dates ending EMBARGO dates before it:

    periods[b - EMBARGO - WINDOW : b - EMBARGO]   ->   predict periods[b : b + REFRESH]

A 20-day label at date t resolves at t + 20, so every context label is known by
the block's first date, and a row is never in its own context (its date is >= b).
That holds for every row the downstream model trains on as well as every row it
is scored on, which is the leak features-foundation.md section 3 warns about. A
row's representation is at most REFRESH - 1 dates staler than a live round's,
which recomputes daily -- conservative, not optimistic.

Generation starts at START, not at the panel's first date: no downstream
walk-forward reads a feature before its first training date (2024-01, see
build_fmrep's check), and the dates in between are the PCA burn-in. Earlier rows
are absent from the cache and land on the neutral bin at load.

Only TEST-token embeddings are taken; data_source="train" tokens have attended to
their own label.
"""

import argparse
import os
from time import time

import numpy as np
import pandas as pd

from src.data_mani.utils import load_tournament_data
from src.prediction.functions import filter_features, sorted_unique
from src.prediction.tabpfn_fm import TabPFNSkRegressor

TARGET = "target_binned_return_20"
PERIOD, ID = "date", "symbol"
FENC = "crypto/v2.0-fenccausal"
AFPP = "crypto/v2.0-tenc-onehot-rollsel-spread-interact"     # allfeat_pp, 812 columns
CACHE_DIR = os.path.join("data", "crypto", "fmrep_cache")

START = "2023-05-15"
REFRESH = 20
EMBARGO = 20
WINDOW = 130
CHUNK = 10                # dates per predict() call; bounds GPU memory
SEED = 42
N_ESTIMATORS = 4          # E19's settings, so `main` is E19 made causal
MAX_CONTEXT = 4096
TEMPERATURE = 0.9

#: Interval edges between the five target bins {0, .25, .5, .75, 1}.
BIN_EDGES = (0.125, 0.375, 0.625, 0.875)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

# The head blocks are CURUPIRA_05's member regexes (src/live/registry.py), read
# from the allfeat_pp dataset that holds them; raw is fenccausal minus its means.
PASSES = {
    "main": dict(version=FENC, window=WINDOW, full=True),
    "head_tenc": dict(version=AFPP, include=r"^feature_sym_tenc$", window=WINDOW),
    "head_interact": dict(version=AFPP, include=r"_x_|_accel_", window=WINDOW),
    "head_size": dict(version=AFPP, include=r"^feature_market_cap_(avg|ewa)_(20d|60d)$", window=WINDOW),
    "head_price": dict(version=AFPP, include=r"^feature_close_(avg|ewa)_(20d|60d)$", window=WINDOW),
    "head_raw": dict(version=FENC, exclude=r"_symmean$", window=WINDOW),
    "ctx030": dict(version=FENC, window=30),
    "ctx390": dict(version=FENC, window=390),
}


def refresh_blocks(periods, first, refresh=REFRESH, window=WINDOW, embargo=EMBARGO):
    """
    Yield (context_periods, predict_periods), causal by construction.

    :param periods: every panel period, sorted
    :param first: index of the first period to predict
    :return: generator of (array, array); a block whose context would be shorter
             than `window` is not yielded
    """
    for b in range(first, len(periods), refresh):
        end = b - embargo
        if end - window < 0:
            continue
        yield periods[end - window:end], periods[b:b + refresh]


class EstimatorMeans:
    """
    Capture each internal estimator's predictive mean during one predict().

    TabPFNRegressor averages its estimators inside _compute_aggregated_logits and
    keeps nothing per member. Every member's probabilities pass through
    translate_probs_across_borders on the way in, onto the shared z-norm borders,
    so wrapping that one call sees each member exactly once.
    """

    def __init__(self, regressor):
        import tabpfn.regressor as module
        self.module = module
        self.regressor = regressor
        self.original = module.translate_probs_across_borders
        self.means = []

    def __enter__(self):
        def wrapped(*args, **kwargs):
            probs = self.original(*args, **kwargs)
            flat = probs.reshape(-1, probs.shape[-1])
            # The raw-space bar distribution shares the z-norm bucket count, so a
            # member's z-norm probabilities read off its borders give its raw mean.
            mean = self.regressor.raw_space_bardist_.mean(flat.clamp_min(1e-30).log())
            self.means.append(mean.float().cpu().numpy())
            return probs
        self.module.translate_probs_across_borders = wrapped
        return self

    def __exit__(self, *exc):
        self.module.translate_probs_across_borders = self.original


def describe_chunk(regressor, X, quantiles=QUANTILES, embeddings=True):
    """
    Every representation of one chunk of rows.

    :return: (dict of 1-D arrays, (rows, dim) float32 embedding or None)
    """
    import torch

    with EstimatorMeans(regressor) as members:
        full = regressor.predict(X, output_type="full", quantiles=list(quantiles))
    logits, criterion = full["logits"], full["criterion"]
    with torch.no_grad():
        edges = torch.tensor(BIN_EDGES, dtype=logits.dtype, device=logits.device)
        cdf = criterion.cdf(logits, edges).float().cpu().numpy()
        std = criterion.variance(logits).clamp_min(0).sqrt().float().cpu().numpy()
    cdf = np.concatenate([np.zeros((len(cdf), 1)), cdf, np.ones((len(cdf), 1))], axis=1)

    out = {"tp_mean": full["mean"], "tp_median": full["median"], "tp_mode": full["mode"],
           "tp_std": std}
    for q, values in zip(quantiles, full["quantiles"]):
        out["tp_q%02d" % round(100 * q)] = values
    for k, p in enumerate(np.diff(cdf, axis=1).T):
        out["tp_p%d" % k] = p
    for k, mean in enumerate(members.means):
        if len(mean) != len(X):
            raise RuntimeError("estimator %d covered %d of %d rows" % (k, len(mean), len(X)))
        out["tp_est%d" % k] = mean

    if not embeddings:
        return out, None
    embedding = np.asarray(regressor.get_embeddings(X, data_source="test"))
    if embedding.ndim == 3:                       # (estimators, rows, dim)
        embedding = embedding.mean(axis=0)
    return out, embedding.astype("float32")


def run_pass(name, device, start=START):
    """Generate one pass over the panel and write CACHE_DIR/<name>.parquet."""
    cfg = PASSES[name]
    df, features = load_tournament_data(
        tournament="crypto", version=cfg["version"], val_frac=0.3, embargo=EMBARGO,
        verbose=False, target=TARGET)
    features = filter_features(features, cfg.get("include"), cfg.get("exclude"), verbose=False)
    periods = sorted_unique(df[PERIOD].to_numpy())
    first = int(np.searchsorted(periods, np.datetime64(start)))
    print("%s: %s, %d features, window %d, predicting %s .. %s on %s"
          % (name, cfg["version"], len(features), cfg["window"], periods[first],
             periods[-1], device), flush=True)

    frames, started = [], time()
    for n, (context, block) in enumerate(refresh_blocks(periods, first, window=cfg["window"])):
        cmask = df[PERIOD].isin(context)
        estimator = TabPFNSkRegressor(
            n_estimators=N_ESTIMATORS, max_context=MAX_CONTEXT,
            softmax_temperature=TEMPERATURE, device=device, random_state=SEED)
        estimator.fit(df.loc[cmask, features], df.loc[cmask, TARGET])
        for i in range(0, len(block), CHUNK):
            rows = df.loc[df[PERIOD].isin(block[i:i + CHUNK]), [ID, PERIOD] + features]
            X = rows[features].to_numpy(dtype=np.float32)
            frame = rows[[ID, PERIOD]].reset_index(drop=True)
            if cfg.get("full"):
                values, embedding = describe_chunk(estimator._model, X)
                for column, v in values.items():
                    frame[column] = np.asarray(v, dtype="float32")
                emb = pd.DataFrame(embedding.astype("float16"),
                                   columns=["emb_%03d" % j for j in range(embedding.shape[1])])
                frame = pd.concat([frame, emb], axis=1)
            else:
                frame[name] = estimator.predict(X).astype("float32")
            frame["block"] = n
            frames.append(frame)
        if n % 5 == 0:
            print("  block %d: context %s..%s -> %s..%s, %.1f min"
                  % (n, context[0], context[-1], block[0], block[-1], (time() - started) / 60),
                  flush=True)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name + ".parquet")
    out.to_parquet(path, index=False)
    print("wrote %s: %d rows x %d cols, %d blocks, %.1f min"
          % (path, len(out), out.shape[1], out["block"].nunique(), (time() - started) / 60), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pass", dest="passes", default="main",
                        help="comma-separated pass names, or 'all': %s" % ", ".join(PASSES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()
    names = list(PASSES) if args.passes == "all" else args.passes.split(",")
    for name in names:
        if args.skip_done and os.path.exists(os.path.join(CACHE_DIR, name + ".parquet")):
            print("%s on disk -- skip" % name)
            continue
        run_pass(name, args.device)


if __name__ == "__main__":
    main()
