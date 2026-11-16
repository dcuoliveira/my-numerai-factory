"""
Causal leaf-value features from E128's 6-month random forest (study E129).

    python3 -m src.data_mani.build_rfleaf                 # fit, verify, write every variant
    python3 -m src.data_mani.build_rfleaf --skip-fit      # re-merge from the cached leaf panel

What a "leaf" feature is here
-----------------------------
A forest's leaf IDS are not features a downstream model can use across refits: the
forest is refit every 130 periods, and leaf 7 of tree 12 means something different
in every refit. What does carry over is each leaf's VALUE -- the mean target of the
training rows that landed in it, i.e. that tree's prediction. For every row we take
the 100 trees' leaf values and summarise their DISTRIBUTION, which is
refit-invariant: mean (== the forest's prediction), std (tree disagreement) and the
10/25/50/75/90th percentiles. An eighth column is the forest's prediction
residualised on the 5-day-lagged meta model (E107, s = 0.5), since the source model
is "RF 6m t20 resid". Every column is rank-binned within the date onto Numerai's
{0,.25,.5,.75,1}.

Why it is causal
----------------
The forest is E128's arm w130 re-run through the pipeline's own forecast(): rf_ktz
on crypto/v2.0-tenc-onehot, target_binned_return_20, rolling 130-period window,
refit every 130, embargo 20, the 40-draw purged search at every refit, seed 42. The
validation start is moved back by 8 x 130 periods so the walk-forward covers almost
the whole history, which keeps E128's block boundaries. A row at date d therefore
gets the leaves of a forest trained only on rows dated <= (start of d's block) - 20,
whose 20-day targets had resolved before d. No row ever gets leaves from a forest
that trained on it, so a downstream model's training rows and prediction rows carry
the same kind of out-of-sample feature. Dates before the first 130-period block are
left NaN and become the neutral 0.5 at load.

Verified at build time: the leaf mean reproduces the forest's own predict(), and on
E128's blocks 2-3 the forest reproduces E128's stored forecast bit for bit (2e-16).
Blocks 0 and 1 are not comparable: make_validation_split drops the 20 embargo dates
before the validation start and E128's first two training windows straddle that
gap, while here they are contiguous (rank corr 0.88 / 0.98 to E128's).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from src.data_mani import binning  # noqa: E402
from src.data_mani.utils import local_path  # noqa: E402

SOURCE = "crypto/v2.0-tenc-onehot"
E128_RUN = ("rf_ktz", "e128_w130")
PREFIX = "feature_rfleaf_"
SUMMARIES = ("mean", "std", "q10", "q25", "q50", "q75", "q90", "resid")
BASES = ("v2.0-asof", "v2.0-fenccausal", "v2.0-tenc-onehot-spread-interact",
         "v2.0-tenc-onehot-rollsel-spread-interact")
CACHE = os.path.join("data", "crypto", "rfleaf_cache", "leafvals.parquet")
BLOCK, BLOCKS_BACK = 130, 8


def _forest(estimator):
    if hasattr(estimator, "named_steps"):
        return list(estimator.named_steps.values())[-1]
    return estimator


def fit_leaves(n_jobs, search_jobs, verbose=True, smoke=False):
    """Re-run E128 w130 over the extended window; return per-row tree values."""
    from src.prediction import functions

    captured = []
    original = functions.predict_signal

    def capture(estimator, X, task):
        out = original(estimator, X, task)
        caller = sys._getframe(1)
        if caller.f_code.co_name == "walk_forward_fit_and_predict":
            # Key the trees by the walk-forward's own rows, not by call order.
            loc = caller.f_locals
            keys = loc["df"].loc[loc["block_mask"], [loc["id_col"], loc["period_col"]]]
            assert len(keys) == len(X)
            forest = _forest(estimator)
            Xa = np.asarray(X, dtype=np.float32)
            trees = np.column_stack([t.predict(Xa) for t in forest.estimators_])
            assert np.allclose(trees.mean(axis=1), out, atol=1e-6), "leaf mean != forest predict"
            block = keys.reset_index(drop=True).rename(columns={loc["id_col"]: "symbol", loc["period_col"]: "date"})
            block["prediction"], block["refit"] = np.asarray(out, dtype="float64"), len(captured)
            captured.append(pd.concat([block, pd.DataFrame(
                trees, columns=["tree_%03d" % i for i in range(trees.shape[1])])], axis=1))
        return out

    functions.predict_signal = capture
    try:
        dates = pd.read_parquet(local_path("crypto", SOURCE.split("/")[1], "train.parquet"),
                                columns=["date"])["date"]
        n_periods = pd.to_datetime(dates).nunique()
        n_val = int(round(n_periods * 0.3)) + (1 if smoke else BLOCKS_BACK) * BLOCK
        val_frac = n_val / n_periods
        assert int(round(n_periods * val_frac)) == n_val
        predictions, _, _ = functions.forecast(
            tournament="crypto", model_name="rf_ktz", task="regression", metric="corr",
            data_version=SOURCE, era_step=1, val_frac=val_frac, refit_every=BLOCK,
            embargo=20, scheme="rolling", train_window=BLOCK, n_iter=2 if smoke else 40,
            n_splits=2 if smoke else 5,
            n_jobs=n_jobs, seed=42, verbose=verbose, search_jobs=search_jobs,
            min_train=BLOCK, target="target_binned_return_20", inner_cv="purged")
    finally:
        functions.predict_signal = original

    frame = pd.concat(captured, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    assert len(frame) == len(predictions) and not frame.duplicated(["symbol", "date"]).any()
    return frame, n_periods, val_frac


def verify_against_e128(frame):
    """Blocks 1-3 of E128 must be reproduced exactly; report block 0 separately."""
    model, run = E128_RUN
    e128 = pd.read_csv(os.path.join("results", "forecast", "crypto", model, run + ".csv"),
                       usecols=["symbol", "date", "prediction", "refit"], parse_dates=["date"])
    joined = e128.merge(frame[["symbol", "date", "prediction"]], on=["symbol", "date"],
                        suffixes=("_e128", "_mine"))
    report = {}
    for refit, block in joined.groupby("refit"):
        diff = float(np.abs(block.prediction_e128 - block.prediction_mine).max())
        rho = float(block.groupby("date").apply(
            lambda g: g.prediction_e128.rank().corr(g.prediction_mine.rank())).mean())
        report[int(refit)] = {"rows": int(len(block)), "max_abs_diff": diff, "rank_corr": rho}
    for refit, r in report.items():
        print("E128 block %d: %d rows, max |diff| %.2e, mean within-date rank corr %.6f"
              % (refit, r["rows"], r["max_abs_diff"], r["rank_corr"]))
    # Only E128's blocks 2+ are comparable. make_validation_split drops the 20
    # embargo dates before the validation start, and E128's block 0 AND block 1
    # training windows reach back across that gap (block 1's 130 periods span 150
    # dates). With the start moved back the gap moves with it, so here both
    # windows are contiguous 130-date blocks -- the honest walk-forward -- and
    # cannot equal E128's. Blocks 2 and 3 sit wholly after the gap in both runs.
    later = [r for k, r in report.items() if k >= 2]
    assert later and all(r["max_abs_diff"] < 1e-6 for r in later), \
        "the re-run forest does not reproduce E128 on blocks 2-3"
    return report


def summarise(frame):
    """Tree values -> the 8 refit-invariant columns, binned within each date."""
    from src.prediction import meta_resid

    trees = frame.filter(like="tree_").to_numpy()
    out = frame[["symbol", "date"]].copy()
    out[PREFIX + "mean"] = trees.mean(axis=1)
    out[PREFIX + "std"] = trees.std(axis=1)
    for q in (10, 25, 50, 75, 90):
        out[PREFIX + "q%d" % q] = np.percentile(trees, q, axis=1)

    meta = pd.read_parquet(local_path("crypto", "v2.0", "historical_meta_models.parquet")).reset_index()
    resid = np.full(len(out), np.nan)
    for date, idx in out.groupby("date").indices.items():
        values, _ = meta_resid.lagged_meta(meta, out["symbol"].to_numpy()[idx], date)
        r, _ = meta_resid.residualise(out[PREFIX + "mean"].to_numpy()[idx], values)
        resid[idx] = r
    out[PREFIX + "resid"] = resid
    return out


def write_variants(leaf, bases):
    cols = [PREFIX + s for s in SUMMARIES]
    written = []
    for base in bases:
        src = pd.read_parquet(local_path("crypto", base, "train.parquet"))
        idx_name = src.index.name or "symbol"
        panel = src.reset_index() if "symbol" not in src.columns else src.reset_index(drop=True)
        panel["date"] = pd.to_datetime(panel["date"])
        clash = [c for c in cols if c in panel.columns]
        assert not clash, clash
        merged = panel.merge(leaf, on=["symbol", "date"], how="left")
        assert len(merged) == len(panel)
        # Bin over the PANEL's own cross-section each date, after the merge.
        merged = binning.bin_cross_sectional(merged, cols, "date")
        for c in cols:
            mass = binning.bin_mass(merged[c])
            drift = float(np.max(np.abs(np.asarray(mass) - np.array([.05, .20, .50, .20, .05]))))
            assert drift < 0.01, (base, c, mass)
        name = base + "-rfleaf"
        path = local_path("crypto", name, "train.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        merged.set_index(idx_name).to_parquet(path)
        meta_src = local_path("crypto", base, "historical_meta_models.parquet")
        if not os.path.exists(meta_src):
            meta_src = local_path("crypto", "v2.0", "historical_meta_models.parquet")
        shutil.copy(meta_src, os.path.join(os.path.dirname(path), "historical_meta_models.parquet"))
        cover = float(merged[cols[0]].notna().mean())
        print("wrote crypto/%s: %d rows, %d columns (+%d rfleaf), rfleaf coverage %.1f%%"
              % (name, len(merged), merged.shape[1], len(cols), 100 * cover))
        written.append(name)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--search-jobs", type=int, default=8)
    ap.add_argument("--skip-fit", action="store_true", help="re-merge from the cached leaf panel")
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--smoke", action="store_true",
                    help="2-draw search, one block back, no E128 check, nothing written under data/")
    args = ap.parse_args()

    if args.skip_fit:
        frame = pd.read_parquet(CACHE)
        meta = json.load(open(CACHE.replace(".parquet", ".json")))
        print("cached leaf panel: %d rows (%s)" % (len(frame), meta))
    elif args.smoke:
        frame, n_periods, val_frac = fit_leaves(args.n_jobs, args.search_jobs, smoke=True)
        leaf = summarise(frame)
        print("SMOKE: %d leaf rows over %d dates, blocks %d, val_frac %.4f" % (len(frame), frame.date.nunique(), frame.refit.nunique(), val_frac))
        print(leaf.describe().T.round(4).to_string())
        return
    else:
        frame, n_periods, val_frac = fit_leaves(args.n_jobs, args.search_jobs)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        frame.to_parquet(CACHE, index=False)      # before the check: a failed check must not cost the fit
        report = verify_against_e128(frame)
        json.dump({"n_periods": int(n_periods), "val_frac": val_frac, "first_leaf_date": str(frame.date.min().date()),
                   "blocks": int(frame.refit.nunique()), "e128_check": report},
                  open(CACHE.replace(".parquet", ".json"), "w"), indent=1)
    leaf = summarise(frame)
    write_variants(leaf, [b for b in args.bases.split(",") if b])


if __name__ == "__main__":
    main()
