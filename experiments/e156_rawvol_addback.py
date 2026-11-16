"""E156: simulate adding the two shipped raw volatility features to v2.0.

Primary comparison: full v2.0 (22 columns) minus v2.0-exrawvolfeat (20).
Add back 20d and 60d separately as fixed, predeclared attribution arms.
All rows, labels, folds and seeds match. Development only; no macro/feed masks.
"""
import argparse
import base64
import copy
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
import e133_histfe_factorial as H
import e131_analysis as E
from e150_finish_scoring import matrix_scores, fast_bootstrap

ROOT = Path("results/reports/crypto/e156_rawvol_addback")
SNAPSHOT = Path("data/crypto/v2.0-e156-frozen")
REDUCED = Path("data/crypto/v2.0-exrawvolfeat")
RAW = ["feature_volatility_20d", "feature_volatility_60d"]
ARM_NAMES = ["EXRAWVOL", "ADD20", "ADD60", "FULL"]
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
GRID = copy.deepcopy(H.GRID)
GRID["mlp_ktz"] = {"mlp__alpha": [0.01, 0.1, 1.0], "mlp__learning_rate_init": [0.01, 0.05011872336272722]}
H.GRID = GRID


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def feature_arms(columns):
    full = [c for c in columns if c.startswith("feature_")]
    if len(full) != 22 or any(c not in full for c in RAW):
        raise ValueError("Expected precisely the 22 shipped v2.0 columns including both volatility columns")
    reduced = [c for c in full if c not in RAW]
    return dict(EXRAWVOL=reduced, ADD20=reduced + RAW[:1], ADD60=reduced + RAW[1:], FULL=full)


def prepare():
    if (ROOT / "registration.json").exists():
        raise RuntimeError("Study already registered; inputs will not be overwritten")
    ROOT.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    REDUCED.mkdir(parents=True, exist_ok=False)
    for file in ("train.parquet", "historical_meta_models.parquet"):
        shutil.copyfile(Path("data/crypto/v2.0") / file, SNAPSHOT / file)
    base = pd.read_parquet(SNAPSHOT / "train.parquet")
    arms = feature_arms(base.columns)
    reduced = base.drop(columns=RAW)
    pd.testing.assert_frame_equal(base[reduced.columns], reduced, check_exact=True)
    reduced.to_parquet(REDUCED / "train.parquet")
    shutil.copyfile(SNAPSHOT / "historical_meta_models.parquet", REDUCED / "historical_meta_models.parquet")
    dates = pd.to_datetime(base.date)
    folds = H.blocks(np.sort(dates[dates < H.TEST_START].unique()))
    registration = dict(experiment="E156", question="Marginal value of adding shipped raw volatility columns to the other 20 v2.0 features",
                        primary="FULL - EXRAWVOL; positive means adding raw volatility helps",
                        source_version="crypto/v2.0", reduced_version="crypto/v2.0-exrawvolfeat", arms=arms,
                        excluded_features=RAW, target=H.T20, features_copied_exactly=True,
                        rows=len(base), universe="all panel coins; no coverage, macro or external price restrictions",
                        missing="preserve dataset NaNs; identical standard v2.0 neutral fill 0.5 at model input",
                        fixed=H.FIXED, search_grid=GRID, search_candidates_per_fold=6,
                        seeds={m: [0] if m == "ridge" else list(range(5)) for m in MODELS},
                        searched_seeds="same five stochastic seeds as fixed fits; each arm/seed gets six candidates per fold",
                        timing=dict(train_dates=H.TRAIN, purge_dates=H.PURGE, predict_dates=H.BLOCK,
                                    inner_validation_dates=H.INNER_VAL, cutoff=str(H.TEST_START.date()),
                                    development_blocks=len(folds), first_prediction=str(pd.Timestamp(folds[0][2][0]).date()),
                                    last_prediction=str(pd.Timestamp(folds[-1][2][-1]).date())),
                        uncertainty="paired Newey-West lag 20 and moving-block bootstrap length 20, 2000 draws",
                        test="development-only; previously evaluated final block excluded; no untouched-test claim",
                        interpretation="conditional value of the two columns; Bollinger and Sharpe features retain volatility information",
                        inputs={str(p): sha(p) for p in [SNAPSHOT / "train.parquet", SNAPSHOT / "historical_meta_models.parquet",
                                                        REDUCED / "train.parquet"]}, runner_sha256=sha(__file__))
    write_json(ROOT / "registration.json", registration)
    write_json(REDUCED / "manifest.json", registration)
    print(json.dumps(dict(experiment="E156", rows=len(base), feature_counts={k: len(v) for k, v in arms.items()},
                          prediction_blocks=len(folds))), flush=True)


def verify():
    registration = json.loads((ROOT / "registration.json").read_text())
    assert registration["runner_sha256"] == sha(__file__), "Frozen runner changed"
    for path, digest in registration["inputs"].items():
        assert sha(path) == digest, "Frozen input changed: " + path
    return registration


def load():
    frame = pd.read_parquet(SNAPSHOT / "train.parquet", filters=[("date", "<", H.TEST_START)]).reset_index()
    frame["date"] = pd.to_datetime(frame.date)
    arms = feature_arms(frame.columns)
    meta = pd.read_parquet(SNAPSHOT / "historical_meta_models.parquet")
    if "symbol" not in meta.columns:
        meta = meta.reset_index()
    meta["date"] = pd.to_datetime(meta.date)
    frame = frame.merge(meta[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"}),
                        on=["symbol", "date"], how="left", validate="1:1")
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    frame[arms["FULL"]] = frame[arms["FULL"]].fillna(0.5)
    assert np.isfinite(frame[arms["FULL"]].to_numpy()).all()
    assert frame.date.max() < H.TEST_START
    return frame, arms


def fit(model, arm, n_jobs):
    registration = verify()
    frame, arms = load()
    features = arms[arm]
    blocks = H.blocks(np.sort(frame.date.unique()))
    status = ROOT / "batch_status" / (model + "_" + arm + ".json")
    write_json(status, dict(state="running", model=model, arm=arm, started=time.time()))
    try:
        for searched in (False, True):
            for seed in registration["seeds"][model]:
                tag = ("tune" if searched else "") + arm
                path = ROOT / "preds" / model / (tag + "_s" + str(seed) + ".parquet")
                if path.exists():
                    continue
                parts, picks = [], []
                for block, train, test in blocks:
                    training = frame[frame.date.isin(train)]
                    evaluation = frame[frame.date.isin(test)]
                    params = H.pick(model, training, features, train, seed, n_jobs) if searched else H.FIXED[model]
                    if params is None:
                        raise RuntimeError("All search candidates failed to produce a finite validation score")
                    predicted = H.fit_predict(model, params, seed, n_jobs,
                                              training[features].to_numpy(np.float32), training[H.T20].to_numpy(),
                                              evaluation[features].to_numpy(np.float32))
                    assert np.isfinite(predicted).all()
                    part = evaluation[["symbol", "date"]].copy()
                    part["prediction"], part["block"] = predicted, block
                    parts.append(part)
                    picks.append(dict(block=block, parameters=params, train_rows=len(training), predict_rows=len(evaluation),
                                      prediction_std=float(np.std(predicted)), unique_predictions=int(len(np.unique(predicted)))))
                    print(model, tag, "seed", seed, "block", block, "done", flush=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".partial")
                pd.concat(parts, ignore_index=True).to_parquet(temporary, index=False)
                temporary.replace(path)
                write_json(path.with_suffix(".picks.json"), picks)
        write_json(status, dict(state="complete", model=model, arm=arm, completed=time.time()))
    except Exception as error:
        write_json(status, dict(state="failed", model=model, arm=arm, error=str(error)))
        raise


def describe_delta(left, right, metric):
    common = left[["date", "block", metric]].merge(right[["date", metric]], on="date", suffixes=("", "_base"), validate="1:1").dropna()
    delta = common[metric] - common[metric + "_base"]
    mean, se, t = E.newey_west(delta.to_numpy())
    lo, hi = fast_bootstrap(delta.to_numpy())
    by_block = delta.groupby(common.block).mean()
    return dict(delta=mean, se=se, t=t, lo=lo, hi=hi, win=float((delta > 0).mean()), n=len(delta),
                worst_block=float(by_block.min()), worst_block_id=int(by_block.idxmin()), delta_drawdown=E.max_dd(delta))


def finish():
    registration = verify()
    statuses = [ROOT / "batch_status" / (m + "_" + a + ".json") for m in MODELS for a in ARM_NAMES]
    while True:
        states = [json.loads(p.read_text())["state"] if p.exists() else "pending" for p in statuses]
        if "failed" in states:
            raise RuntimeError("Training batch failed: " + str([str(p) for p, s in zip(statuses, states) if s == "failed"]))
        if all(s == "complete" for s in states):
            break
        print("Completed training batches", states.count("complete"), "/", len(states), flush=True)
        time.sleep(20)
    frame, arms = load()
    dates = np.sort(frame.date.unique())
    scored_dates = np.concatenate([test for _, _, test in H.blocks(dates)])
    keys = frame[frame.date.isin(scored_dates)][["symbol", "date", H.T20, "meta_model"]].reset_index(drop=True)
    multi = pd.MultiIndex.from_frame(keys[["symbol", "date"]])
    boundaries = np.r_[0, np.flatnonzero(keys.date.to_numpy()[1:] != keys.date.to_numpy()[:-1]) + 1, len(keys)]
    all_scores, parity = [], []
    for model in MODELS:
        for seed in registration["seeds"][model]:
            columns, metadata = [], []
            for searched in (False, True):
                for arm in ARM_NAMES:
                    tag = ("tune" if searched else "") + arm
                    p = pd.read_parquet(ROOT / "preds" / model / (tag + "_s" + str(seed) + ".parquet"))
                    p["date"] = pd.to_datetime(p.date)
                    value = p.set_index(["symbol", "date"])["prediction"].reindex(multi).to_numpy()
                    assert np.isfinite(value).all(), "Full-universe predictions required"
                    columns.append(value)
                    metadata.append(tag)
            matrix = np.column_stack(columns)
            for i, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                part = keys.iloc[start:stop]
                c, m = matrix_scores(matrix[start:stop], part[H.T20].to_numpy(), part.meta_model.to_numpy(float))
                if i in (0, 715, len(boundaries) - 2):
                    for col, tag in enumerate(metadata):
                        ref = part.assign(prediction=matrix[start:stop, col]).set_index("symbol")
                        expected = [E.corr_one(ref), E.contrib_one(ref, "meta_model") if ref.meta_model.notna().sum() >= 10 else np.nan]
                        np.testing.assert_allclose([c[col], m[col]], expected, atol=1e-12, rtol=1e-11, equal_nan=True)
                    parity.append(dict(model=model, seed=seed, date=str(part.date.iloc[0].date()), columns=len(metadata)))
                for col, tag in enumerate(metadata):
                    all_scores.append(dict(model=model, arm=tag, seed=seed, date=part.date.iloc[0], block=i // H.BLOCK,
                                           corr=c[col], mmc=m[col], payout=.1 * c[col] + m[col]))
            print("Scored", model, "seed", seed, flush=True)
    S = pd.DataFrame(all_scores)
    S.to_parquet(ROOT / "scores_by_date_seed.parquet", index=False)
    A = S.groupby(["model", "arm", "date", "block"], as_index=False)[["corr", "mmc", "payout"]].mean()
    paired, seeds, levels = [], [], []
    for model in MODELS:
        for prefix in ("", "tune"):
            for arm in ("ADD20", "ADD60", "FULL"):
                for metric in ("corr", "mmc", "payout"):
                    left = A[A.model.eq(model) & A.arm.eq(prefix + arm)]
                    right = A[A.model.eq(model) & A.arm.eq(prefix + "EXRAWVOL")]
                    paired.append(dict(model=model, arm=prefix + arm, vs=prefix + "EXRAWVOL", metric=metric,
                                       **describe_delta(left, right, metric)))
                    for seed in registration["seeds"][model]:
                        subset = S[S.model.eq(model) & S.seed.eq(seed)]
                        seeds.append(dict(model=model, arm=prefix + arm, vs=prefix + "EXRAWVOL", metric=metric, seed=seed,
                                          **describe_delta(subset[subset.arm.eq(prefix + arm)], subset[subset.arm.eq(prefix + "EXRAWVOL")], metric)))
    for (model, arm), part in A.groupby(["model", "arm"]):
        payout = part.dropna(subset=["payout"])
        dd = E.max_dd(payout.payout)
        levels.append(dict(model=model, arm=arm, corr=part['corr'].mean(), mmc=payout.mmc.mean(), payout=payout.payout.mean(),
                           corr_sharpe=part['corr'].mean() / part['corr'].std(ddof=0),
                           payout_sharpe=payout.payout.mean() / payout.payout.std(ddof=0),
                           corr_drawdown=E.max_dd(part['corr']), payout_drawdown=dd,
                           calmar=payout.payout.mean() / dd if dd > 0 else np.nan, burn=int((payout.payout < 0).sum()),
                           n=len(part), n_pay=len(payout)))
    D, L = pd.DataFrame(paired), pd.DataFrame(levels)
    D.to_csv(ROOT / "paired_deltas.csv", index=False)
    L.to_csv(ROOT / "levels.csv", index=False)
    pd.DataFrame(seeds).to_csv(ROOT / "paired_by_seed.csv", index=False)
    A.groupby(["model", "arm", "block"], as_index=False)[["corr", "mmc", "payout"]].mean().to_csv(ROOT / "by_block.csv", index=False)
    write_json(ROOT / "scoring_parity.json", parity)
    report(A, D, L, registration)
    write_json(ROOT / "COMPLETE.json", dict(experiment="E156", completed=time.time(), prediction_rows=len(keys),
                                           score_rows=len(S), report="E156_does_adding_raw_volatility_help_DEVELOPMENT.pdf"))


def report(A, D, L, registration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    head = D[D.arm.eq("tuneFULL") & D.metric.eq("corr")].copy()
    head["verdict"] = np.where(head.t >= 2, "PROMISING: positive on development", np.where(head.t <= -2, "REJECT on development", "NO IMPROVEMENT detected"))
    head.to_csv(ROOT / "headline.csv", index=False)
    body = '<h1>E156: What do the two shipped volatility features add?</h1><p>Primary comparison: full v2.0 (22 features) minus v2.0-exrawvolfeat (20 features). Positive differences mean adding the two volatility columns helps.</p>'
    body += head[["model", "delta", "se", "t", "lo", "hi", "win", "worst_block", "n", "verdict"]].to_html(index=False, float_format=lambda x: f"{x:.5g}")
    body += '<p>Development-only experiment. The already evaluated 2026-02-20 onward block was excluded. This cannot establish an untouched-test result. No macro features or reconstructed external price coverage are used.</p>'
    body += '<p>Removed columns: feature_volatility_20d and feature_volatility_60d. The other 20 shipped features, including Sharpe ratios and Bollinger bands, remain available. This measures conditional predictive value, not the value of all possible volatility information.</p>'
    body += '<p>All arms use identical full-universe rows, labels, 130-date training windows, 22-date purges and 130-date prediction blocks. Fixed configurations and equal-budget six-candidate inner searches are reported; both use five matched stochastic seeds (ridge: seed 0). Standard v2.0 missing-input fill is 0.5 in every arm, with source missing values preserved in the dataset. The causal own-history and macro-volatility feature blocks are absent.</p>'
    body += '<p>Paired differences use Newey-West lag 20 uncertainty and 2,000 moving-block bootstrap draws of length 20. MMC is a proxy against the historical meta model, available on fewer dates than CORR. Payout is formed per date as 0.1 CORR + proxy MMC. Multiple source/metric/model comparisons are exploratory.</p>'
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        part = A[A.model.eq(model)].pivot(index="date", columns="arm", values="corr")
        for prefix, label in (("", "Fixed"), ("tune", "Searched")):
            delta = part[prefix + "FULL"] - part[prefix + "EXRAWVOL"]
            ax.plot(delta.index, delta.cumsum(), label=label)
        ax.axhline(0, color="gray", linewidth=.6)
        ax.set_title(model)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Cumulative paired CORR difference")
    axes[-1].legend()
    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)
    figure = figdir / "raw_volatility_addback.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    body += '<h2>Stability over development dates</h2><p>Adding both columns minus the same model without them. These are accumulated correlation differences, not investment returns.</p><img style="width:100%" src="data:image/png;base64,' + base64.b64encode(figure.read_bytes()).decode() + '">'
    for title, table in (("Paired add-back results", D), ("Levels and downside", L),
                         ("By-seed stability", pd.read_csv(ROOT / "paired_by_seed.csv")),
                         ("By-block results", pd.read_csv(ROOT / "by_block.csv"))):
        body += '<h2>' + title + '</h2>' + table.to_html(index=False, float_format=lambda x: f"{x:.5g}")
    body += '<h2>Frozen design and inputs</h2><pre>' + html.escape(json.dumps(registration, indent=2)) + '</pre>'
    css = 'body{font:11px Arial;color:#18202a}table{border-collapse:collapse;width:100%;font-size:9px}td,th{padding:4px;border:1px solid #ddd;overflow-wrap:anywhere}h2{break-before:page}tr{break-inside:avoid}pre{white-space:pre-wrap}'
    path = ROOT / "E156_does_adding_raw_volatility_help_DEVELOPMENT.html"
    path.write_text('<!doctype html><html><head><meta charset="utf-8"><style>' + css + '</style></head><body>' + body + '</body></html>')
    subprocess.run([sys.executable, "-m", "src.analysis.html_pdf", str(path)], check=True)


def main():
    ap = argparse.ArgumentParser()
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--fit", action="store_true")
    action.add_argument("--finish", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arm", choices=ARM_NAMES)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    if args.prepare:
        prepare()
    elif args.finish:
        finish()
    else:
        if not args.model or not args.arm:
            ap.error("--fit requires --model and --arm")
        fit(args.model, args.arm, args.n_jobs)


if __name__ == "__main__":
    main()
