"""Finish frozen E150 predictions with batched, reference-checked scoring.

No models, features, selection rules or holdout dates are changed. Identical
prediction files share scoring work; all published metrics retain arm/seed IDs.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e150_macrovol_evaluation as M

ROOT = Path("results/reports/crypto/e150_macrovol_observed_dev")
DATA = "data/crypto/v2.0-vol-causal-macrovol-retrospective"


def configure():
    M.ROOT, M.PRED, M.DIR = str(ROOT), str(ROOT / "preds"), DATA
    M.THIN = M.thin_blocks()


def matrix_scores(predictions, targets, meta):
    """Columnwise Numerai CORR and historical-meta MMC, matching numerai_tools."""
    n = len(targets)
    assert np.isfinite(predictions).all() and np.isfinite(targets).all()
    z = ndtri((rankdata(predictions, axis=0, method="average") - 0.5) / n)
    powered = np.sign(z) * np.abs(z) ** 1.5
    y = targets - targets.mean()
    y = np.sign(y) * np.abs(y) ** 1.5
    # The library transforms targets in their stored precision (float32),
    # then Pearson correlation centers in float64.
    y = y.astype(float)
    y = y - y.mean()
    powered -= powered.mean(axis=0)
    denom = np.sqrt(np.sum(powered ** 2, axis=0) * np.sum(y ** 2))
    corr = np.divide(y @ powered, denom, out=np.full(predictions.shape[1], np.nan), where=denom > 0)
    valid = np.isfinite(meta)
    mmc = np.full(predictions.shape[1], np.nan)
    if valid.sum() >= 10:
        nn = int(valid.sum())
        p = ndtri((rankdata(predictions[valid], axis=0, method="average") - 0.5) / nn)
        m = ndtri((rankdata(meta[valid], method="average") - 0.5) / nn)
        residual = p - np.outer(m, (p.T @ m) / (m @ m))
        target = targets[valid].copy()
        if (target >= 0).all() and (target <= 1).all():
            target *= 4
        target -= target.mean()
        mmc = (target @ residual) / nn
    return corr, mmc


def score_batch(task):
    model, seed, jobs = task
    configure()
    cache = ROOT / "scoring_batches" / (model + "_s" + str(seed) + ".parquet")
    if cache.exists():
        return str(cache)
    k = pd.read_parquet(M.Q.KEYS, columns=["symbol", "date", M.H.T20, "meta_model", "block", "kind"])
    k["date"] = pd.to_datetime(k["date"])
    k = k[k.kind.eq("dev") & ~k.block.isin(M.THIN)].sort_values(["date", "symbol"]).reset_index(drop=True)
    baseline = M.b_all(model, seed)
    frame = k.merge(baseline, on=["symbol", "date"], how="left", validate="1:1")
    assert frame.b0.notna().all()
    key = pd.MultiIndex.from_frame(k[["symbol", "date"]])
    columns, aliases, unique = [], [], {}
    for job in jobs:
        arm = job[1]
        path = ROOT / "preds" / model / (arm + "_s" + str(seed) + ".parquet")
        identity = ("baseline",) if arm == "B_ALL" else (os.stat(path).st_dev, os.stat(path).st_ino)
        if identity not in unique:
            unique[identity] = len(columns)
            if arm == "B_ALL":
                values = np.full(len(k), np.nan)
            else:
                p = pd.read_parquet(path, columns=["symbol", "date", "prediction"])
                p["date"] = pd.to_datetime(p["date"])
                values = p.set_index(["symbol", "date"])["prediction"].reindex(key).to_numpy(float)
            columns.append(values)
        aliases.append((arm, unique[identity]))
    candidates = np.column_stack(columns)
    rows = []
    validation = []
    boundaries = np.r_[0, np.flatnonzero(k.date.to_numpy()[1:] != k.date.to_numpy()[:-1]) + 1, len(k)]
    for counter, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        part = k.iloc[start:stop]
        base = rankdata(frame.b0.to_numpy()[start:stop], method="average") / (stop - start)
        c = candidates[start:stop]
        x = np.repeat(base[:, None], c.shape[1], axis=1)
        coverage = np.isfinite(c)
        for col in range(c.shape[1]):
            indices = np.flatnonzero(coverage[:, col])
            x[indices[np.argsort(c[indices, col], kind="stable")], col] = np.sort(base[indices])
        assert np.array_equal(x[~coverage], np.repeat(base[:, None], c.shape[1], axis=1)[~coverage])
        corr, mmc = matrix_scores(x, part[M.H.T20].to_numpy(), part.meta_model.to_numpy(float))
        # Real-data parity checks cover early, middle and meta-model-era dates.
        if counter in (0, (len(boundaries) - 1) // 2, len(boundaries) - 2):
            errors = []
            for col in range(x.shape[1]):
                reference = part.assign(prediction=x[:, col]).set_index("symbol")
                rc = M.E.corr_one(reference)
                rm = M.E.contrib_one(reference, "meta_model") if reference.meta_model.notna().sum() >= 10 else np.nan
                np.testing.assert_allclose([corr[col], mmc[col]], [rc, rm], rtol=1e-11, atol=1e-12, equal_nan=True)
                errors.extend([abs(corr[col] - rc), abs(mmc[col] - rm)])
            validation.append(dict(date=str(part.date.iloc[0].date()), max_error=float(np.nanmax(errors)), columns=x.shape[1]))
        covered = coverage.mean(axis=0)
        for arm, col in aliases:
            rows.append(dict(model=model, arm=arm, seed=seed, date=part.date.iloc[0], block=int(part.block.iloc[0]),
                             kind="dev", corr=corr[col], mmc=mmc[col], payout=0.1 * corr[col] + mmc[col], covered=covered[col]))
    cache.parent.mkdir(exist_ok=True)
    temporary = cache.with_suffix(".partial")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(cache)
    cache.with_suffix(".validation.json").write_text(json.dumps(validation, indent=2))
    return str(cache)


def fast_bootstrap(x, block=M.E.BLK, reps=M.E.NBOOT, seed=131):
    """Same random block starts as the frozen scorer, using block sums."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2 * block:
        return np.nan, np.nan
    k = int(np.ceil(n / block))
    starts = np.random.default_rng(seed).integers(0, n - block + 1, size=(reps, k))
    cumulative = np.r_[0., np.cumsum(x)]
    sums = cumulative[block:] - cumulative[:-block]
    last = n - (k - 1) * block
    partial = cumulative[last:] - cumulative[:-last]
    means = (sums[starts[:, :-1]].sum(axis=1) + partial[starts[:, -1]]) / n
    return tuple(np.percentile(means, [2.5, 97.5]))


PAIR_CACHE = {}


def fast_paired(A, a, b, model, scope, metric):
    identity = id(A)
    # A cache entry holds a reference as well, so object IDs cannot be reused.
    if identity not in PAIR_CACHE:
        PAIR_CACHE[identity] = (A, {(m, arm, kind): part.set_index("date")
                                   for (m, arm, kind), part in A.groupby(["model", "arm", "kind"])})
    data = PAIR_CACHE[identity][1]
    left, right = data.get((model, a, scope)), data.get((model, b, scope))
    if left is None or right is None:
        return None
    z = left[[metric, "block"]].join(right[[metric]], rsuffix="_b", how="inner").dropna()
    delta = z[metric] - z[metric + "_b"]
    if len(delta) < 5:
        return None
    mean, se, t = M.E.newey_west(delta.values)
    lo, hi = fast_bootstrap(delta.values)
    worst = delta.groupby(z.block).mean()
    return dict(model=model, arm=a, vs=b, scope=scope, metric=metric, delta=float(mean), se=float(se), t=float(t),
                lo=float(lo), hi=float(hi), win=float((delta > 0).mean()), worst_block=float(worst.min()),
                worst_block_id=int(worst.idxmin()), dd=M.E.max_dd(delta.values), n=len(delta))


def tables(S):
    M.paired = fast_paired
    A, D, L = M.summarise(S, ["dev"])
    D.to_csv(ROOT / "paired_deltas.csv", index=False)
    L.to_csv(ROOT / "levels.csv", index=False)
    print("Headline tables written", flush=True)
    rows = []
    for source in M.SOURCES:
        regimes = A.merge(M.regimes(source), on="date", how="left")
        for regime in ("high", "mid", "low"):
            subset = regimes[regimes.regime.eq(regime)]
            for a, b in M.PAIRS:
                for model in M.MODELS:
                    row = fast_paired(subset, a, b, model, "dev", "corr")
                    if row:
                        rows.append(dict(row, regime=regime, macro_source=source))
            PAIR_CACHE.pop(id(subset), None)
        print("Regime tables", source, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "regime_deltas.csv", index=False)
    seed_rows = []
    for (model, seed), part in S[S.model.ne("ridge")].groupby(["model", "seed"]):
        subset = M.per_date(part)
        for a, b in [("ALL", "VC_COV"), ("VC_COV", "B_COV")] + [("F_" + k.upper(), "VC_COV") for k in M.SOURCES]:
            row = fast_paired(subset, a, b, model, "dev", "corr")
            if row:
                seed_rows.append(dict(row, seed=int(seed)))
        PAIR_CACHE.pop(id(subset), None)
    pd.DataFrame(seed_rows).to_csv(ROOT / "seed_deltas.csv", index=False)
    cov = pd.read_parquet(Path(DATA) / "train.parquet", columns=["date"] + M.MASK).reset_index()
    cov["date"] = pd.to_datetime(cov.date)
    cov["covered"] = cov[M.MASK].all(axis=1)
    keys = pd.read_parquet(M.Q.KEYS, columns=["symbol", "date", "kind"])
    keys["date"] = pd.to_datetime(keys.date)
    joined = keys[keys.kind.eq("dev")].merge(cov, on=["symbol", "date"], how="left", validate="1:1")
    joined.groupby([joined.date.dt.year, "kind"])[M.MASK + ["covered"]].mean().round(4).reset_index().to_csv(ROOT / "coverage_by_year.csv", index=False)
    # Confirm the completed selection calculation against the archived decision.
    selection = json.loads((ROOT / "parsim.json").read_text())
    for source, detail in selection["detail"].items():
        arm = "tuneF_" + source.upper()
        for model in M.MODELS:
            for metric, key in (("corr", "corr_t"), ("payout", "payout_t")):
                row = fast_paired(A, arm, arm + "__VC_COV", model, "dev", metric)
                assert round(row["t"], 2) == detail[key][model], (source, model, metric, row["t"], detail[key][model])
    print("Archived selection statistics reproduced", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    configure()
    arms = ["B_ALL"] + M.FIT_ARMS
    matched = [a + suffix for a in arms if a not in ("B_ALL", "B_COV", "VC_COV")
               for suffix in ("__B_COV", "__VC_COV")]
    jobs = M.jobs(arms + matched, ["dev"]) + M.jobs(arms[1:] + matched, ["dev"], tuned=True)
    groups = {}
    for job in jobs:
        groups.setdefault((job[0], job[2]), []).append(job)
    tasks = [(model, seed, group) for (model, seed), group in groups.items()]
    paths = []
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        pending = {pool.submit(score_batch, task): task[:2] for task in tasks}
        for future in as_completed(pending):
            paths.append(future.result())
            print("scored", pending[future], len(paths), "/", len(tasks), flush=True)
    S = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    S = S.sort_values(["model", "arm", "seed", "date"]).reset_index(drop=True)
    S.to_parquet(ROOT / "scores_by_date_seed.parquet", index=False)
    tables(S)
    import e150_observed_finish as report
    report.report()
    completion = dict(scope="exploratory development only", completed=time.time(), scoring_seconds=time.time() - started,
                      prediction_jobs=len(jobs), batches=len(tasks), rows=len(S),
                      scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      validation="numerai_tools real-date parity <= 1e-12; original selection t-statistics reproduced")
    (ROOT / "COMPLETE.json").write_text(json.dumps(completion, indent=2))
    print("COMPLETE", json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
