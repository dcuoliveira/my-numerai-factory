"""
Every representation TabPFN v2 exposes, per coin-date, on a finished run's contexts.

    python -m src.analysis.tabpfn_representations --run e19_fenccausal --device cuda:0

The forecast CSVs keep one column, the predictive MEAN. This replays a finished
`tabpfn` walk-forward -- same loader, same refit windows (read back from its
hyperparams table), same seeded context subsample -- and writes everything else
the model computes for those rows (experiments/features-foundation.md, rows 1-5):

    tp_mean, tp_median, tp_mode       point predictions                    (row 1)
    tp_q10 .. tp_q90, tp_std          the predictive distribution          (row 2)
    tp_p0 .. tp_p4                    P(target bin = 0, .25, .5, .75, 1)   (row 3)
    tp_est0 .. tp_est{k-1}            each internal estimator's mean       (row 4)
    emb_000 .. emb_191                test-token embedding, averaged over
                                      the estimators                      (row 5)

Because the contexts are E19's, `tp_mean` must reproduce the run's own
`prediction` column; the script prints the gap, and that is the check that the
other columns describe the same model.

Leak wall: inherited, not rebuilt. Each block's context ends `embargo` periods
before the block's first date, exactly as in the run. Only TEST-token embeddings
are taken -- `data_source="train"` tokens have attended to their own label.
"""

import argparse
import json
import os
from time import time

import numpy as np
import pandas as pd

from src.data_mani.utils import get_spec, load_tournament_data
from src.prediction.functions import filter_features
from src.prediction.tabpfn_fm import TabPFNSkRegressor

#: Interval edges between the five target bins {0, .25, .5, .75, 1}.
BIN_EDGES = (0.125, 0.375, 0.625, 0.875)
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
OUT_DIR = os.path.join("results", "cache", "tabpfn_repr")


def load_run(run, model="tabpfn"):
    """The run's config and its refit windows, as the walk-forward recorded them."""
    with open(os.path.join("results", "metrics", "crypto", model, run + ".json")) as fh:
        config = json.load(fh)["config"]
    refits = pd.read_csv(os.path.join("results", "hyperparams", "crypto", model, run + ".csv"))
    return config, refits


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


def describe_chunk(regressor, X):
    """All representations for one chunk of rows, as a dict of 1-D / 2-D arrays."""
    import torch

    with EstimatorMeans(regressor) as members:
        full = regressor.predict(X, output_type="full", quantiles=list(QUANTILES))
    logits, criterion = full["logits"], full["criterion"]
    with torch.no_grad():
        edges = torch.tensor(BIN_EDGES, dtype=logits.dtype, device=logits.device)
        cdf = criterion.cdf(logits, edges).float().cpu().numpy()
        std = criterion.variance(logits).clamp_min(0).sqrt().float().cpu().numpy()
    cdf = np.concatenate([np.zeros((len(cdf), 1)), cdf, np.ones((len(cdf), 1))], axis=1)

    out = {
        "tp_mean": full["mean"], "tp_median": full["median"], "tp_mode": full["mode"],
        "tp_std": std,
    }
    for q, values in zip(QUANTILES, full["quantiles"]):
        out["tp_q%02d" % round(100 * q)] = values
    for k, p in enumerate(np.diff(cdf, axis=1).T):
        out["tp_p%d" % k] = p
    for k, mean in enumerate(members.means):
        if len(mean) != len(X):
            raise RuntimeError("estimator %d covered %d of %d rows" % (k, len(mean), len(X)))
        out["tp_est%d" % k] = mean

    embedding = np.asarray(regressor.get_embeddings(X, data_source="test"))
    if embedding.ndim == 3:                       # (estimators, rows, dim)
        embedding = embedding.mean(axis=0)
    return out, embedding.astype("float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="e19_fenccausal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-periods", type=int, default=10,
                        help="dates per predict call; bounds GPU memory")
    args = parser.parse_args()

    config, refits = load_run(args.run)
    spec = get_spec("crypto")
    period_col, id_col = spec["period_col"], spec["id_col"]
    target = config["target"]
    options = dict(config["model_options"])
    options.update(config.get("fixed_params") or {})

    df, features = load_tournament_data(
        tournament="crypto", version=config["data_version"],
        feature_set=config.get("feature_set"), era_step=config["era_step"],
        val_frac=config["val_frac"], embargo=config["embargo"], verbose=False,
        target=target)
    features = filter_features(features, config.get("feature_include"),
                               config.get("feature_exclude"), verbose=False)
    print("%s: %d features, %d refits, device %s" % (args.run, len(features), len(refits), args.device))

    frames = []
    started = time()
    for _, refit in refits.iterrows():
        dates = df[period_col]
        train_mask = (dates >= pd.Timestamp(refit["train_start"])) & (dates <= pd.Timestamp(refit["train_end"]))
        block_mask = (dates >= pd.Timestamp(refit["predict_start"])) & (dates <= pd.Timestamp(refit["predict_end"]))
        if train_mask.sum() != refit["train_rows"]:
            raise RuntimeError("refit %d: %d training rows, the run had %d"
                               % (refit["refit"], train_mask.sum(), refit["train_rows"]))

        estimator = TabPFNSkRegressor(
            model_path=options["model_path"], n_estimators=options["n_estimators"],
            max_context=options["max_context"],
            softmax_temperature=options.get("softmax_temperature", 0.9),
            device=args.device, random_state=config["seed"])
        estimator.fit(df.loc[train_mask, features], df.loc[train_mask, target])
        regressor = estimator._model

        block = df.loc[block_mask, [id_col, period_col, target] + features]
        periods = np.sort(block[period_col].unique())
        for i in range(0, len(periods), args.chunk_periods):
            rows = block[block[period_col].isin(periods[i:i + args.chunk_periods])]
            X = rows[features].to_numpy(dtype=np.float32)
            values, embedding = describe_chunk(regressor, X)
            frame = rows[[id_col, period_col, target]].reset_index(drop=True)
            frame["refit"] = int(refit["refit"])
            for name, column in values.items():
                frame[name] = np.asarray(column, dtype="float32")
            emb = pd.DataFrame(embedding, columns=["emb_%03d" % j for j in range(embedding.shape[1])])
            frames.append(pd.concat([frame, emb], axis=1))
        print("refit %d: %d context rows, %d dates, %.1f min elapsed"
              % (refit["refit"], estimator.context_rows_, len(periods), (time() - started) / 60))

    out = pd.concat(frames, ignore_index=True)

    # The check: tp_mean must be the run's own prediction column.
    saved = pd.read_csv(os.path.join("results", "forecast", "crypto", "tabpfn", args.run + ".csv"),
                        parse_dates=[period_col])
    both = out.merge(saved[[id_col, period_col, "prediction"]], on=[id_col, period_col])
    gap = (both["tp_mean"] - both["prediction"]).abs()
    rank = both.groupby(period_col).apply(
        lambda g: g["tp_mean"].rank().corr(g["prediction"].rank()), include_groups=False)
    print("tp_mean vs saved prediction: %d rows matched, max |gap| %.2e, "
          "min per-date rank corr %.6f" % (len(both), gap.max(), rank.min()))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.run + ".parquet")
    out.to_parquet(path, index=False)
    print("wrote %s (%d rows x %d cols) in %.1f min" % (path, len(out), out.shape[1], (time() - started) / 60))


if __name__ == "__main__":
    main()
