"""
E118's source models: the book's own model classes refit walk-forward from START,
with the internals each one exposes kept per (symbol, date).

    python -m src.data_mani.fmx_models --source rf,catboost,lgb,mlp,blocks

Every source runs through the pipeline's own `forecast()` -- the same loader,
purged hyperparameter search inside each 130-date training window, 20-period
embargo and 130-period refit cadence as every arm in results/ -- with validation
starting at foundation_repr.START, so it predicts every date a downstream training
window can read. Two hooks into src.prediction.functions observe it without
changing it: one remembers each refit's training block, the other reads the
refitted estimator when it predicts the block.

    rf        MA_FENC's class/data (rf_ktz, fenccausal)   x22 prediction, x23 per-tree std,
                                                          x24 leaf PCs, x25 leaf support + label variance
    catboost  IAUARETE_01's (catboost_ktz, fenccausal)    x26 prediction, x27 virtual-ensemble std
    lgb       SACI_04's (lgb, raw)                        x28 prediction, x29 stage profile, x30 SHAP summary
    mlp       IA_01's (mlp, fenccausal)                   x31 prediction, x32 hidden-unit PCs, x33 seed std
    blocks    CURUPIRA_05's five block ridges (allfeat_pp) x34 the five block predictions

Causality: every number for a row comes from an estimator searched and fitted on
the 130 dates ending 20 periods before that row's block, and every reduction (leaf
SVD, hidden-unit PCA) is fitted on that same training block, never on the rows it
describes. SACI_04's pinned parameters are NOT reused: they were chosen on history
that overlaps the scored window, so the lgb source is searched per refit like the
others. Live, each source is one refit per 130 rounds plus a predict per round.
"""

import argparse
import os
from time import time

import numpy as np
import pandas as pd

import src.prediction.functions as F
from src.data_mani.foundation_repr import CACHE_DIR, START

N_PCS = 8
SEEDS = (43, 44, 45, 46)

SOURCES = {
    "rf": dict(model="rf_ktz", data="crypto/v2.0-fenccausal", n_iter=40, n_splits=5),
    "catboost": dict(model="catboost_ktz", data="crypto/v2.0-fenccausal", n_iter=12, n_splits=5),
    "lgb": dict(model="lgb", data="crypto/v2.0", n_iter=20, n_splits=3),
    "mlp": dict(model="mlp", data="crypto/v2.0-fenccausal", n_iter=50, n_splits=5),
}
AFPP = "crypto/v2.0-tenc-onehot-rollsel-spread-interact"
BLOCKS = {   # CURUPIRA_05's members (src/live/registry.py)
    "onehot": ("ridge_glmnet", r"^feature_sym_(?!tenc$)"),
    "tenc": ("ridge_glmnet", r"^feature_sym_tenc$"),
    "interact": ("ridge_glmnet", r"_x_|_accel_"),
    "size": ("ridge", r"^feature_market_cap_(avg|ewa)_(20d|60d)$"),
    "price": ("ridge_glmnet", r"^feature_close_(avg|ewa)_(20d|60d)$"),
}


def val_frac_from(data_version, start=START):
    """The val_frac whose validation block starts exactly at `start`."""
    from src.data_mani.utils import local_path
    periods = np.sort(pd.read_parquet(local_path("crypto", data_version, "train.parquet"), columns=["date"])["date"].unique())
    i = int(np.searchsorted(periods, np.datetime64(start)))
    return (len(periods) - i + 0.5) / len(periods)


# --------------------------------------------------------------------------- #
# Extractors: (fitted estimator, training block, query block) -> {column: array}
# --------------------------------------------------------------------------- #
def rf_parts(est, Xt, yt, Xq):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import OneHotEncoder
    trees = np.stack([t.predict(Xq.to_numpy(np.float32)) for t in est.estimators_], axis=1)
    out = {"pred": est.predict(Xq), "tree_std": trees.std(axis=1)}
    lt, lq = est.apply(Xt), est.apply(Xq)
    enc = OneHotEncoder(handle_unknown="ignore").fit(lt)
    svd = TruncatedSVD(N_PCS, random_state=42).fit(enc.transform(lt))
    for j, col in enumerate(svd.transform(enc.transform(lq)).T):
        out["leaf_pc%d" % j] = col
    size = np.stack([t.tree_.n_node_samples[lq[:, k]] for k, t in enumerate(est.estimators_)], axis=1)
    var = np.stack([t.tree_.impurity[lq[:, k]] for k, t in enumerate(est.estimators_)], axis=1)
    out["leaf_support"] = size.shape[1] / (1.0 / size).sum(axis=1)
    out["leaf_var"] = var.mean(axis=1)
    return out


def catboost_parts(est, Xt, yt, Xq):
    ve = np.asarray(est.virtual_ensembles_predict(Xq, prediction_type="VirtEnsembles", virtual_ensembles_count=10))
    ve = ve.reshape(len(Xq), 10, -1)[:, :, 0]
    return {"pred": est.predict(Xq), "ve_std": ve.std(axis=1)}


def lgb_parts(est, Xt, yt, Xq):
    total = est.booster_.current_iteration()
    stages = np.stack([est.predict(Xq, num_iteration=max(1, int(total * f))) for f in (0.25, 0.5, 0.75, 1.0)], axis=1)
    shap = np.abs(np.asarray(est.predict(Xq, pred_contrib=True))[:, :-1])
    return {"pred": stages[:, -1], "early": stages[:, 0], "stage_sd": stages.std(axis=1),
            "shap_norm": np.sqrt((shap ** 2).sum(axis=1)), "shap_top": shap.max(axis=1) / np.clip(shap.sum(axis=1), 1e-12, None)}


def mlp_parts(est, Xt, yt, Xq):
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    net = est.steps[-1][1]
    act = {"relu": lambda z: np.maximum(z, 0), "identity": lambda z: z, "tanh": np.tanh,
           "logistic": lambda z: 1 / (1 + np.exp(-z))}[net.activation]

    def hidden(X):
        z = est[:-1].transform(X) if len(est.steps) > 1 else np.asarray(X)
        for W, b in zip(net.coefs_[:-1], net.intercepts_[:-1]):
            z = act(z @ W + b)
        return z
    pca = PCA(N_PCS, random_state=42).fit(hidden(Xt))
    out = {"pred": est.predict(Xq)}
    for j, col in enumerate(pca.transform(hidden(Xq)).T):
        out["hid_pc%d" % j] = col
    step = est.steps[-1][0]
    preds = [out["pred"]]
    for s in SEEDS:
        twin = clone(est).set_params(**{"%s__random_state" % step: s})
        preds.append(twin.fit(Xt, yt).predict(Xq))
    out["seed_std"] = np.std(preds, axis=0)
    return out


EXTRACT = {"rf": rf_parts, "catboost": catboost_parts, "lgb": lgb_parts, "mlp": mlp_parts}


class Observe:
    """Watch forecast() refit by refit without changing what it computes."""

    def __init__(self, extractor):
        self.extractor, self.frames = extractor, []
        self.df = self.train = None
        self.in_search = False

    def __enter__(self):
        self.orig = (F.walk_forward_fit_and_predict, F.hyper_params_search, F.predict_signal)
        wf, hps, ps = self.orig

        def walk(df, *a, **k):
            self.df = df
            return wf(df, *a, **k)

        def search(*a, **k):
            self.train = (k["X"], k["y"])
            self.in_search = True
            try:
                return hps(*a, **k)
            finally:
                self.in_search = False

        def predict(estimator, X, task):
            pred = ps(estimator, X, task)
            if not self.in_search and self.extractor is not None:
                parts = self.extractor(estimator, self.train[0], self.train[1], X)
                frame = self.df.loc[X.index, ["symbol", "date"]].reset_index(drop=True)
                for name, col in parts.items():
                    frame[name] = np.asarray(col, dtype="float32")
                self.frames.append(frame)
            return pred

        F.walk_forward_fit_and_predict, F.hyper_params_search, F.predict_signal = walk, search, predict
        return self

    def __exit__(self, *exc):
        F.walk_forward_fit_and_predict, F.hyper_params_search, F.predict_signal = self.orig


def run_source(name, n_jobs, smoke=False):
    started = time()
    start = "2026-05-01" if smoke else START
    n_iter = (lambda k: 2) if smoke else (lambda k: k)
    if name == "blocks":
        out = None
        vf = val_frac_from(AFPP, start)
        for block, (model, include) in BLOCKS.items():
            preds, _, _ = F.forecast("crypto", model, data_version=AFPP, val_frac=vf, refit_every=130, embargo=20,
                                     scheme="rolling", train_window=130, n_iter=n_iter(50), n_splits=5, n_jobs=n_jobs,
                                     seed=42, verbose=False, feature_include=include,
                                     target="target_binned_return_20")
            p = preds[["symbol", "date", "prediction"]].rename(columns={"prediction": "blk_" + block})
            out = p if out is None else out.merge(p, on=["symbol", "date"], how="outer")
            print("  blocks/%s: %d rows, %.1f min" % (block, len(p), (time() - started) / 60), flush=True)
    else:
        spec = SOURCES[name]
        vf = val_frac_from(spec["data"], start)
        with Observe(EXTRACT[name]) as obs:
            preds, _, results = F.forecast("crypto", spec["model"], data_version=spec["data"], val_frac=vf,
                                           refit_every=130, embargo=20, scheme="rolling", train_window=130,
                                           n_iter=n_iter(spec["n_iter"]), n_splits=spec["n_splits"], n_jobs=n_jobs,
                                           seed=42, verbose=False, target="target_binned_return_20")
        out = pd.concat(obs.frames, ignore_index=True)
        check = out.merge(preds[["symbol", "date", "prediction"]], on=["symbol", "date"])
        gap = float(np.abs(check["pred"] - check["prediction"]).max())
        print("  %s: %d refits observed, pred matches forecast() to %.1e" % (name, len(obs.frames), gap), flush=True)
    out = out[out["date"] >= pd.Timestamp(start)]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "m_%s%s.parquet" % (name, "_smoke" if smoke else ""))
    out.to_parquet(path, index=False)
    print("wrote %s: %d rows x %d cols, %s .. %s, %.1f min" % (path, len(out), out.shape[1], out["date"].min().date(),
                                                               out["date"].max().date(), (time() - started) / 60), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, help="comma-separated: rf, catboost, lgb, mlp, blocks")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="one late refit, n_iter 2, writes m_<source>_smoke")
    args = parser.parse_args()
    for name in args.source.split(","):
        print("== %s" % name, flush=True)
        run_source(name, args.n_jobs, args.smoke)


if __name__ == "__main__":
    main()
