"""E152: paired, covered-sample return evaluation for verified volume-alpha inputs.

Dev and test are separate commands; test requires a frozen dev-only selection.
Uncovered rows always receive their explicit baseline rank in exported files.
"""
import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler

from src.analysis.ablation_report import newey_west
from src.data_mani.volumealpha_features import PREFIX
from src.data_mani.volumealpha_sources import digest
from src.prediction.functions import purged_period_splits
from src.prediction.metrics import per_period_bmc, per_period_corr, summarize
from src.prediction.models import build_wrapper
from src.prediction.volumealpha import route_slots

MODEL_ALIASES = {"ridge_ktz": "ridge", "gbm_ktz": "lgb_ktz", "mlp_ktz": "mlp_ktz"}
TARGET = "target_binned_return_20"
TRAIN, PURGE, REFIT = 130, 22, 130
TEST_START = "2026-11-13"  # registered forward block, NOT the repeatedly reused 2026-02..08 block


def arm_families(families):
    fundamentals = [k for k in ("size", "surprise", "leverage", "valuation", "beta", "age") if k in families]
    raw = [k for k in families if k not in ("forecasts", "costs") and not k.startswith("scenario_")]
    arms = {"TECHNICAL": ["technical"], "LIQUIDITY": ["liquidity"],
            "FUNDAMENTALS": fundamentals, "EVENTS": ["events"], "CROSS_ASSET": ["cross_asset"],
            "FORECASTS": ["forecasts"], "COSTS": ["costs"], "ALL_RAW": raw,
            "ALL_DERIVED": ["forecasts", "costs"], "FULL": raw + ["forecasts", "costs"]}
    arms["RAW_COMMON"], arms["MODEL_COMMON"] = raw, ["forecasts", "costs"]
    for key in families:
        if key in ("size", "surprise", "leverage", "valuation", "beta", "age"):
            arms["F_" + key.upper()] = [key]
        if key.startswith("scenario_"):
            arms[key.upper()] = [key]
    # Sparse source arms need not pretend unavailable families exist.
    return {a: present for a, groups in arms.items()
            if (present := [g for g in groups if g in families])}


def score_predictions(frame, prediction="prediction"):
    p = per_period_corr(frame, prediction, TARGET, "date").rename("corr").to_frame()
    if "meta_model" in frame:
        # Tiny diagnostic groups cannot support contribution estimates. In
        # particular, demeaning one reference observation gives a zero vector.
        usable = frame.dropna(subset=["meta_model", TARGET, prediction])
        counts = usable.groupby("date")["meta_model"].transform("count")
        distinct = usable.groupby("date")["meta_model"].transform("nunique")
        usable = usable[(counts >= 3) & (distinct >= 2)]
        p = p.join(per_period_bmc(usable, prediction, "meta_model", TARGET, "date").rename("mmc"))
    else:
        p["mmc"] = np.nan
    p["payout"] = 0.1 * p["corr"] + p["mmc"]
    return p


def paired_stats(candidate, baseline, bootstrap=500, seed=42):
    """Date-paired HAC plus 20-date moving-block bootstrap; no unpaired means."""
    d = (candidate - baseline).dropna().to_numpy()
    if not len(d):
        return {"n": 0, "delta": None, "t": None, "win_rate": None, "ci_low": None, "ci_high": None}
    mean, _, t = newey_west(d, lag=20)
    rng = np.random.default_rng(seed)
    width = min(20, len(d))
    samples = []
    for _ in range(bootstrap):
        starts = rng.integers(0, len(d) - width + 1, size=int(np.ceil(len(d) / width)))
        sample = np.concatenate([d[s:s + width] for s in starts])[:len(d)]
        samples.append(sample.mean())
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"n": len(d), "delta": float(mean), "t": float(t) if np.isfinite(t) else None,
            "win_rate": float((d > 0).mean()), "ci_low": float(lo), "ci_high": float(hi)}


def _fit(wrapper, train, cols, candidates, folds, min_rows=100):
    X, y = train[cols].fillna(0.5), train[TARGET]
    if len(train) < min_rows:
        return None, None
    best, best_score = None, -np.inf
    for params in candidates:
        scores = []
        for tr_dates, va_dates in folds:
            tr, va = train["date"].isin(tr_dates), train["date"].isin(va_dates)
            if tr.sum() < min_rows or va.sum() < 10:
                continue
            estimator = clone(wrapper.ModelClass).set_params(**params)
            estimator.fit(X.loc[tr], y.loc[tr])
            f = train.loc[va, ["date", TARGET]].assign(prediction=estimator.predict(X.loc[va]))
            scores.append(per_period_corr(f, "prediction", TARGET, "date").mean())
        score = np.nanmean(scores) if scores else np.nan
        if np.isfinite(score) and score > best_score:
            best, best_score = params, score
    if best is None:
        return None, None
    estimator = clone(wrapper.ModelClass).set_params(**best)
    estimator.fit(X, y)
    return estimator, best


def run(args):
    source, output = Path(args.data), Path(args.output)
    manifest = json.loads((source / "manifest.json").read_text())
    inferred = manifest.get("status") == "BUILT_INFERRED_RECONSTRUCTION"
    if manifest.get("status") != "BUILT_VERIFIED_INPUTS" and not (inferred and getattr(args, "allow_inferred", False)):
        raise ValueError("strict evaluation requires verified inputs, not a reconstructed-source dataset")
    selection_path = output / "selection.json"
    if args.phase == "test" and not selection_path.exists():
        raise ValueError("test evaluation requires a frozen dev-only selection (--select)")
    if args.test_start != TEST_START:
        raise ValueError(f"E152's registered forward-test boundary is {TEST_START}; register a new study to change it")
    # Predicate pushdown keeps forward-test target values out of the dev process.
    filters = [("date", "<", pd.Timestamp(TEST_START))] if args.phase == "dev" else None
    base = pd.read_parquet(source / "train.parquet", filters=filters)
    d = base.reset_index() if "symbol" not in base else base.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    d["ucid"] = d["ucid"].astype(str)
    d = d.sort_values(["date", "ucid"]).reset_index(drop=True)
    meta = pd.read_parquet(source / "historical_meta_models.parquet")
    if "symbol" not in meta:
        meta = meta.reset_index()
    meta = meta[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"]).dt.tz_localize(None)
    d = d.merge(meta, on=["symbol", "date"], how="left", validate="1:1")
    families = manifest["families"]
    arms = arm_families(families)
    fingerprint = {"manifest_sha256": digest(source / "manifest.json"), "test_start": TEST_START,
                   "n_iter": args.n_iter, "n_splits": args.n_splits, "seed_list": args.seeds,
                   "train": TRAIN, "purge": PURGE, "refit": REFIT,
                   "model_aliases": MODEL_ALIASES}
    output.mkdir(parents=True, exist_ok=True)
    design = output / "design.json"
    if design.exists() and json.loads(design.read_text()) != fingerprint:
        raise ValueError("existing evaluation design differs: choose a new output directory")
    design.write_text(json.dumps(fingerprint, indent=2))
    if selection_path.exists():
        sel = json.loads(selection_path.read_text())
        if sel["design"] != fingerprint:
            raise ValueError("dev selection belongs to a different data/design snapshot")
        if sel["families"]:
            arms["PARSIM"] = sel["families"]
    chosen = args.arms or list(arms)
    unknown = set(chosen) - set(arms)
    if unknown:
        raise ValueError(f"unknown or unselected arms: {sorted(unknown)}")
    base_cols = [c for c in d if c.startswith("feature_") and not c.startswith(PREFIX)]
    dates = sorted(d["date"].unique())
    frozen_test = pd.Timestamp(TEST_START)
    rows_summary, blocked = [], []
    for alias in args.models:
        for seed in args.seeds:
            wrapper = build_wrapper(MODEL_ALIASES[alias], task="regression", seed=seed, n_jobs=args.n_jobs)
            for arm in chosen:
                block_cols = sum((families[k] for k in arms[arm]), [])
                cols = base_cols + block_cols
                eligible = d[block_cols].notna().all(axis=1) if block_cols else pd.Series(False, index=d.index)
                if arm in ("RAW_COMMON", "MODEL_COMMON"):
                    common_cols = sum((families[k] for k in arms["FULL"]), [])
                    eligible = d[common_cols].notna().all(axis=1)
                if not eligible.any():
                    blocked.append({"model": alias, "seed": seed, "arm": arm, "reason": "no causal arm coverage"})
                    continue
                pred_frames, fits = [], []
                for block, start in enumerate(range(TRAIN + PURGE, len(dates), REFIT)):
                    val_dates = dates[start:start + REFIT]
                    if args.phase == "dev":
                        val_dates = [v for v in val_dates if pd.Timestamp(v) < frozen_test]
                    else:
                        val_dates = [v for v in val_dates if pd.Timestamp(v) >= frozen_test]
                    if not val_dates:
                        continue
                    # Even if the final block is split by the phase boundary,
                    # its model remains trained at the ORIGINAL block start.
                    tr_dates = dates[start - PURGE - TRAIN:start - PURGE]
                    tr = d["date"].isin(tr_dates) & d[TARGET].notna()
                    va = d["date"].isin(val_dates)
                    split = purged_period_splits(np.asarray(tr_dates), n_splits=args.n_splits,
                                                 embargo=PURGE, scheme="rolling", train_window=TRAIN)
                    folds = [(np.asarray(tr_dates)[a], np.asarray(tr_dates)[b]) for a, b in split]
                    # Exactly the same candidates, date boundaries and seed for
                    # B_ALL, B_COV and the candidate; grid comes from B_ALL.
                    grid = wrapper.fit_grid(d.loc[tr, base_cols].fillna(0.5), d.loc[tr, TARGET]) if hasattr(wrapper, "fit_grid") else wrapper.param_grid
                    candidates = list(ParameterSampler(grid, n_iter=args.n_iter, random_state=seed))
                    b_all, params_all = _fit(wrapper, d.loc[tr], base_cols, candidates, folds)
                    if b_all is None:
                        raise ValueError(f"baseline has no usable training folds at block {block}")
                    matched = tr & eligible
                    b_cov, params_cov = _fit(wrapper, d.loc[matched], base_cols, candidates, folds)
                    candidate, params = _fit(wrapper, d.loc[matched], cols, candidates, folds)
                    # Insufficient coverage is a baseline fallback, not a row
                    # deletion or a secretly skipped difficult time block.
                    f = d.loc[va, ["symbol", "ucid", "date", TARGET, "meta_model"]].copy()
                    f["block"] = block
                    b = b_all.predict(d.loc[va, base_cols].fillna(0.5))
                    c = np.full(len(f), np.nan)
                    bc = c.copy()
                    covered = eligible.loc[va].to_numpy() & (candidate is not None) & (b_cov is not None)
                    if covered.any():
                        c[covered] = candidate.predict(d.loc[va, cols].iloc[np.flatnonzero(covered)].fillna(0.5))
                        bc[covered] = b_cov.predict(d.loc[va, base_cols].iloc[np.flatnonzero(covered)].fillna(0.5))
                    f["covered"], f["candidate_prediction"] = covered, c
                    for key, values in (("market_cap", "feature_market_cap_avg_20d"),
                                        ("liquidity", PREFIX + "logdv_ma5")):
                        f[key + "_group"] = d.loc[va, values].to_numpy()
                    f["prediction"], f["baseline_prediction"] = np.nan, np.nan
                    f["matched_baseline_prediction"] = np.nan
                    for _, ix in f.groupby("date").groups.items():
                        loc = f.index.get_indexer(ix)
                        routed, ranks = route_slots(b[loc], c[loc], covered[loc])
                        matched_routed, _ = route_slots(b[loc], bc[loc], covered[loc])
                        f.loc[ix, "prediction"] = routed
                        f.loc[ix, "baseline_prediction"] = ranks
                        f.loc[ix, "matched_baseline_prediction"] = matched_routed
                    pred_frames.append(f)
                    fits.append({"block": block, "rows_all": int(tr.sum()), "rows_covered": int(matched.sum()),
                                 "baseline_params": params_all, "matched_params": params_cov, "candidate_params": params})
                if not pred_frames:
                    blocked.append({"model": alias, "seed": seed, "arm": arm, "reason": "no dates in phase"})
                    continue
                f = pd.concat(pred_frames, ignore_index=True)
                stem = f"{args.phase}_{alias}_s{seed}_{arm}"
                f.to_parquet(output / (stem + "_predictions.parquet"), index=False)
                (output / (stem + "_fits.json")).write_text(json.dumps(fits, indent=2, default=str))
                actual = score_predictions(f)
                baseline = score_predictions(f, "baseline_prediction")
                matched_scores = score_predictions(f, "matched_baseline_prediction")
                actual.to_csv(output / (stem + "_per_period.csv"))
                arm_summary = []
                for comparison, ref in (("B_ALL", baseline), ("B_COV_ROUTED", matched_scores)):
                    for metric in ("corr", "mmc", "payout"):
                        summary = paired_stats(actual[metric], ref[metric])
                        pay = summarize(actual["payout"])
                        block_delta = (actual[metric] - ref[metric]).groupby(
                                f.groupby("date")["block"].first()).mean().dropna()
                        arm_summary.append(dict(model=alias, seed=seed, arm=arm, comparison=comparison,
                                metric=metric, level=actual[metric].mean(), baseline=ref[metric].mean(),
                                coverage=float(f["covered"].mean()), worst_block_delta=block_delta.min() if len(block_delta) else np.nan,
                                payout_sharpe=pay["sharpe"], payout_drawdown=pay["max_drawdown"],
                                calmar=pay["mean"] / pay["max_drawdown"] if pay["max_drawdown"] else np.nan,
                                burn_days=int((actual["payout"].dropna() < 0).sum()), **summary))
                rows_summary.extend(arm_summary)
                pd.DataFrame(arm_summary).to_csv(output / (stem + "_summary.csv"), index=False)
                subgroup = []
                for grouping in ("market_cap_group", "liquidity_group"):
                    # These groupings use only dated causal bins, never later caps.
                    for value, g in f.dropna(subset=[grouping]).groupby(grouping):
                        a = score_predictions(g)
                        b = score_predictions(g, "baseline_prediction")
                        for metric in ("corr", "mmc", "payout"):
                            subgroup.append(dict(grouping=grouping, value=value, metric=metric,
                                                 **paired_stats(a[metric], b[metric])))
                pd.DataFrame(subgroup).to_csv(output / (stem + "_groups.csv"), index=False)
                print(stem, "scored", len(f), "rows", flush=True)
    parts = sorted(output.glob(args.phase + "_*_s*_*_summary.csv"))
    summary = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else pd.DataFrame(rows_summary)
    summary.to_csv(output / (args.phase + "_summary.csv"), index=False)
    (output / (args.phase + "_unavailable.json")).write_text(json.dumps(blocked, indent=2))
    common_comparisons = []
    for alias in args.models:
        for seed in args.seeds:
            a = output / f"{args.phase}_{alias}_s{seed}_RAW_COMMON_predictions.parquet"
            b = output / f"{args.phase}_{alias}_s{seed}_MODEL_COMMON_predictions.parquet"
            if not a.exists() or not b.exists():
                continue
            raw, model = pd.read_parquet(a), pd.read_parquet(b)
            pd.testing.assert_frame_equal(raw[["date", "ucid", "covered"]], model[["date", "ucid", "covered"]])
            rs, ms = score_predictions(raw), score_predictions(model)
            for metric in ("corr", "mmc", "payout"):
                common_comparisons.append(dict(model=alias, seed=seed, metric=metric,
                      comparison="MODEL_COMMON minus RAW_COMMON", **paired_stats(ms[metric], rs[metric])))
    pd.DataFrame(common_comparisons).to_csv(output / (args.phase + "_raw_vs_model_common.csv"), index=False)
    write_report(output, summary, blocked, manifest, args.phase)


def select(output):
    output = Path(output)
    design = json.loads((output / "design.json").read_text())
    d = pd.read_csv(output / "dev_summary.csv")
    families = {"TECHNICAL": ["technical"], "LIQUIDITY": ["liquidity"],
                "FUNDAMENTALS": ["size", "surprise", "leverage", "valuation", "beta", "age"],
                "EVENTS": ["events"], "CROSS_ASSET": ["cross_asset"],
                "FORECASTS": ["forecasts"], "COSTS": ["costs"]}
    picks = []
    for arm, f in families.items():
        corr = d[(d["arm"] == arm) & (d["metric"] == "corr") & (d["comparison"] == "B_COV_ROUTED")]
        pay = d[(d["arm"] == arm) & (d["metric"] == "payout") & (d["comparison"] == "B_COV_ROUTED")]
        # Require all registered models and seeds; never turn partial output
        # into a dev selection or interpret a missing family as a negative test.
        expected = {(m, s) for m in MODEL_ALIASES for s in design["seed_list"]}
        if set(zip(corr["model"], corr["seed"])) != expected or set(zip(pay["model"], pay["seed"])) != expected:
            continue
        if (corr.groupby("model")["t"].mean() >= 2).sum() >= 2 and (pay.groupby("model")["t"].mean() > -2).all():
            picks.extend(f)
    path = output / "selection.json"
    value = {"design": design, "families": sorted(set(picks)), "selected_on": "dev only",
             "rule": "mean seed HAC t >=2 on CORR for >=2 models; payout t >-2 for all; complete model/seed matrix"}
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("selection is already frozen; register a new study to change it")
    path.write_text(json.dumps(value, indent=2))
    print(json.dumps(value, indent=2))


def write_report(output, summary, blocked, manifest, phase):
    text = ("<html><head><meta charset='utf-8'><title>E152 volume-alpha</title>"
            "<style>body{font:15px system-ui;margin:36px;max-width:1500px}table{border-collapse:collapse}"
            "td,th{padding:6px;border:1px solid #ddd}pre{white-space:pre-wrap}</style></head><body>"
            "<h1>E152 · volume-alpha causal evaluation</h1>"
            f"<p>Phase: {escape(phase)}. Forward-test boundary: {TEST_START}. "
            "Window/purge/refit: 130/22/130 periods. Same search draws, folds, seeds and covered "
            "sample for candidate vs B_COV. B_ALL predicts every coin; routing only rearranges covered rank slots. "
            "MMC is proxy contribution against the historical meta model.</p>"
            "<p>The aliases ridge_ktz / gbm_ktz map to this repo's ridge / lgb_ktz. "
            "mlp_ktz is the repository's identity-activation MLP. No unmodified global scalar, "
            "coverage indicator, or in-sample volume prediction is a production feature.</p>"
            "<p>Volume prediction and execution/capacity proxies do not establish return alpha. "
            "Any claim needs positive paired return results on the registered forward block. "
            "Multiplicity applies across families and model classes. Bootstrap samples 20-date blocks; HAC lag 20.</p>")
    text += summary.to_html(index=False, float_format=lambda x: f"{x:.5g}") if not summary.empty else "<p>No return evaluation is available.</p>"
    text += "<h2>Unavailable arms</h2><pre>" + escape(json.dumps(blocked, indent=2)) + "</pre>"
    text += "<h2>Source and feature manifest</h2><pre>" + escape(json.dumps(manifest, indent=2)) + "</pre></body></html>"
    (Path(output) / (phase + "_report.html")).write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/crypto/v2.0-volumealpha-causal")
    ap.add_argument("--output", default="results/reports/crypto/e152_volumealpha_causal")
    ap.add_argument("--phase", choices=["dev", "test"], default="dev")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--models", choices=list(MODEL_ALIASES), nargs="+", default=list(MODEL_ALIASES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--allow-inferred", action="store_true",
                    help="explicitly evaluate the user-authorized reconstructed source arm; retains provenance limitations in the report")
    args = ap.parse_args()
    if args.select:
        select(args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
