#!/usr/bin/env python3
"""
E104 -- day-indexed minibatches vs row-shuffled minibatches, one pinned model.

WHAT IT RUNS
  The from-scratch torch MLP with sklearn's EXACT initial weights that tracks
  mlp/e33_fenc_a64_32_16_identity_lre3_al100: identity, (64,32,16), alpha 100,
  lr 1e-3, Adam, 0.5*MSE, crypto/v2.0-fenccausal (44f), rolling 130/130, embargo
  20, val_frac 0.3, target_binned_return_20. The ONLY thing varied is how a
  training window's 130 days x ~300 stocks are cut into minibatches:

    row      shuffle the ROWS, slice 1024 at a time. The baseline; at seed 42 it
             must reproduce torch_scratch/e33repl_scratch_SKINIT_* bit for bit.
    day3     shuffle the DAYS; a batch is EVERY stock of k consecutive shuffled
             days, k = round(1024 / mean stocks per day) = 3 -- size-matched.
    day1     the same with one day per batch: the pure cross-section batch.

  A batch of whole days has no fixed row count, because the number of stocks per
  day is not a constant of the panel (241..300). Two things follow from that and
  are decided here rather than left to fall out of the arithmetic:

    LOSS  each DAY in a batch weighs 1/k and its stocks share that equally, so a
          thin day is not out-voted by a broad one -- the weighting the per-day
          metric itself uses. With equal counts it IS the plain row mean.
    L2    sklearn's penalty is 0.5*alpha*|W|^2 / rows_in_batch, so alpha's
          effective strength is alpha/rows and would move with breadth and with
          k (alpha/300 is 3.4x alpha/1024). day3 and day1 pin it to alpha/1024,
          the configured batch size.

  Three controls separate "the batch is made of days" from "the penalty moved":
  rowfix (rows, penalty pinned), day3lit and day1lit (days, penalty divided by
  the rows actually in the batch -- the literal port). Two more separate it from
  "the batch is smaller": row900 and row300 are ROW batches at the day arms' own
  size, and so at their step count, with the same pinned penalty.

MEASURED AGAINST
  row, paired per period, Newey-West lag 20. This arm is numerically DEGENERATE
  -- alpha 100 shrinks the net linear map to ~1e-15 and seeds range from a
  constant predictor to CORR 0.13 -- so ONE seed says nothing. 10 seeds per arm,
  and the test is on the seed-AVERAGED per-period delta.

WHAT COUNTS AS A RESULT
  |t| >= 2 on the seed-mean payout delta of day3 vs row. That delta splits
  exactly into (rowfix - row), the penalty pin, plus (day3 - rowfix), the batch
  composition at an identical objective. Anything less is a null.

Env knobs: E104_SEEDS (42..51), E104_JOBS (24), E104_ONLY (arms, comma list),
E104_FORECAST_SEEDS (42; a forecast is 11 MB), E104_FORCE=1 (rerun finished
arms), DRY_RUN=1, NO_MAIL=1. Modes: `selftest`, `report`; default runs + reports.
INTERPRETER: this file is run with /home/danielco/miniconda3/bin/python3.
"""
import os, sys, json, math, time

# One thread per process, set before numpy/torch load: the pool supplies the
# parallelism, and a pinned thread count is what makes a rerun bit-reproducible.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import concurrent.futures as cf, multiprocessing as mp
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO); os.makedirs("logs", exist_ok=True)

EXPERIMENT, PREFIX, TREE = "E104", "e104", "torch_scratch"
DATA, TGT, NFEAT = "crypto/v2.0-fenccausal", "target_binned_return_20", 44
WIN, REFIT, EMB, VF = 130, 130, 20, 0.3
HID, ALPHA, LR, BATCH, MAXIT, TOL, PAT = (64, 32, 16), 100.0, 1e-3, 1024, 200, 1e-4, 10
BASELINE_REF = "e33repl_scratch_SKINIT_fenc_a64_32_16_identity_lre3_al100"

# arm -> (what is shuffled, batch size, what the L2 term is divided by). The batch
# size is in DAYS for a day arm and in ROWS for a row arm (None = BATCH).
ARMS = {
    "row":     ("row", None,   "batch"),
    "rowfix":  ("row", None,   "ref"),
    "day3":    ("day", "auto", "ref"),
    "day1":    ("day", 1,      "ref"),
    "day3lit": ("day", "auto", "batch"),
    "day1lit": ("day", 1,      "batch"),
    # Added once day1 came back constant on every seed. A day batch differs from
    # `row` in THREE ways -- what a batch is made of, how many rows it holds, and
    # so how many optimiser steps an epoch takes. These hold the last two at the
    # day arms' values (900 rows / 44 steps, 300 rows / 130 steps), leaving the
    # composition as the only difference.
    "row900":  ("row", 900,    "ref"),
    "row300":  ("row", 300,    "ref"),
}
PAIRS = [
    ("day3", "row",       "HEADLINE  day batches vs the baseline"),
    ("rowfix", "row",     "  = the L2 pin alone (rows unchanged)"),
    ("day3", "rowfix",    "  + whole-day batches at the same objective"),
    ("day3", "row900",    "COMPOSITION ONLY: days vs rows, same size and steps"),
    ("day1", "row300",    "COMPOSITION ONLY: one day vs 300 rows, same steps"),
    ("row300", "rowfix",  "BATCH SIZE ONLY: 300 vs 1024 rows, same objective"),
    ("row900", "rowfix",  "BATCH SIZE ONLY: 900 vs 1024 rows, same objective"),
    ("day1", "rowfix",    "one day per batch vs rows, same objective"),
    ("day1", "day3",      "one day vs three days per batch"),
    ("day3lit", "row",    "literal port, 3 days (L2 / rows in batch)"),
    ("day1lit", "row",    "literal port, 1 day  (L2 / rows in batch)"),
]
SEEDS = [int(s) for s in os.environ.get(
    "E104_SEEDS", ",".join(str(s) for s in range(42, 52))).split(",")]
JOBS = int(os.environ.get("E104_JOBS", "24"))
ONLY = [a for a in os.environ.get("E104_ONLY", "").split(",") if a]
FC_SEEDS = [int(s) for s in os.environ.get("E104_FORECAST_SEEDS", "42").split(",") if s]
FORCE = os.environ.get("E104_FORCE") == "1"
DRY = os.environ.get("DRY_RUN") == "1"
NOMAIL = os.environ.get("NO_MAIL") == "1"
# Names a second pass over a subset (the 100-seed one is "s100") so its report
# and figure land beside the full study's instead of on top of them.
TAG = os.environ.get("E104_TAG", "")
SELECTED = [a for a in ARMS if not ONLY or a in ONLY]


def run_name(arm, seed):
    return "%s_%s_s%d" % (PREFIX, arm, seed)


def path(kind, name):
    return os.path.join("results", kind, "crypto", TREE,
                        name + (".json" if kind == "metrics" else ".csv"))


def _torch():
    import torch
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    return torch


# --------------------------------------------------------------------------- #
# The batch scheme under test
# --------------------------------------------------------------------------- #

def rows_by_day(dates):
    """
    Group a training window's rows by day.

    :param dates: the date of every training row, in row order
    :return: one integer array per day, chronological; arrays are RAGGED --
             nothing here assumes two days hold the same number of stocks
    :rtype: [np.ndarray]
    """
    codes, uniq = pd.factorize(dates, sort=True)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(uniq))
    return np.split(order, np.cumsum(counts)[:-1])


def day_batches(day_rows, day_order, days_per_batch):
    """
    Cut one epoch into batches of WHOLE days.

    The index being shuffled is the day; the elements of a day are its stocks,
    and a day is never split across batches. Row counts therefore differ from
    batch to batch whenever breadth does, and the last batch holds whatever days
    are left over.

    :param day_rows: rows_by_day() output
    :param day_order: a permutation of range(len(day_rows))
    :param days_per_batch: k
    :return: yields (row indices, per-row weights, number of days). The weights
             sum to 1 and give each DAY 1/k, shared equally among its stocks.
    """
    for start in range(0, len(day_order), days_per_batch):
        days = day_order[start:start + days_per_batch]
        sizes = np.array([len(day_rows[d]) for d in days])
        weights = np.repeat(1.0 / (len(days) * sizes), sizes)
        yield np.concatenate([day_rows[d] for d in days]), weights, len(days)


def auto_days(day_rows):
    """Days per batch that brings a day batch closest to BATCH rows."""
    return max(1, int(round(BATCH / np.mean([len(r) for r in day_rows]))))


# --------------------------------------------------------------------------- #
# The pinned model
# --------------------------------------------------------------------------- #

def build_net(n_in, seed):
    """
    The identity stack, initialised with sklearn's own draws.

    _init_coef order: one RandomState(seed), and per layer a (fan_in, fan_out)
    weight then a bias, both uniform(-b, b) with b = sqrt(6 / (in + out)). torch
    stores a weight as (out, in), hence the transpose.
    """
    torch = _torch(); from torch import nn
    dims = [n_in] + list(HID) + [1]
    net = nn.Sequential(*[nn.Linear(dims[i], dims[i + 1]).double()
                          for i in range(len(dims) - 1)])   # identity: no activation
    rs = np.random.RandomState(int(seed))
    with torch.no_grad():
        for lin in net:
            b = math.sqrt(6.0 / (lin.in_features + lin.out_features))
            lin.weight.copy_(torch.tensor(
                rs.uniform(-b, b, (lin.in_features, lin.out_features)).T))
            lin.bias.copy_(torch.tensor(rs.uniform(-b, b, lin.out_features)))
    return net


def train(net, X, y, day_rows, scheme, days_per_batch, l2_norm, seed, max_iter=MAXIT):
    """
    sklearn's MLPRegressor loop, with the batch scheme as a parameter.

    The `row`/`batch` path is written to evaluate the same expressions in the
    same order as the run it must reproduce, so that gate is exact rather than
    approximate.
    """
    torch = _torch(); from torch import nn
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    weights = [m.weight for m in net if isinstance(m, nn.Linear)]
    rng = np.random.RandomState(int(seed))
    n = len(X); best, bad = np.inf, 0
    bs = max(1, min(int((days_per_batch if scheme == "row" else None) or BATCH), n))
    epochs, steps, sizes = 0, 0, []
    for _ in range(int(max_iter)):
        if scheme == "row":
            order = rng.permutation(n)
            batches = ((order[s:s + bs], None, None) for s in range(0, n, bs))
        else:
            batches = day_batches(day_rows, rng.permutation(len(day_rows)), days_per_batch)
        tot, seen, steps, sizes = 0.0, 0, 0, []
        for rows, day_w, n_days in batches:
            idx = torch.as_tensor(rows, dtype=torch.long)
            opt.zero_grad(set_to_none=True); out = net(Xt[idx])
            if day_w is None:
                data = 0.5 * ((yt[idx] - out) ** 2).mean()
            else:
                data = 0.5 * (torch.as_tensor(day_w).reshape(-1, 1) * (yt[idx] - out) ** 2).sum()
            # "batch" is sklearn's rule and ties alpha's strength to the row
            # count; "ref" pins it to the configured batch size.
            denom = len(idx) if l2_norm == "batch" else BATCH
            loss = data + (0.5 * ALPHA) * sum((w * w).sum() for w in weights) / denom
            loss.backward(); opt.step()
            unit = len(idx) if n_days is None else n_days
            tot += float(loss.detach()) * unit; seen += unit
            steps += 1; sizes.append(len(idx))
        epochs += 1
        ep = tot / max(seen, 1)
        bad = bad + 1 if ep > best - TOL else 0
        if ep < best:
            best = ep
        if bad > PAT:
            break
    net.eval()
    with torch.no_grad():
        total = None                      # the net linear map this stack composes to
        for lin in net:
            total = lin.weight if total is None else lin.weight @ total
        diag = {"epochs": epochs, "steps_per_epoch": steps, "best_loss": float(best),
                "rows_per_batch_min": int(min(sizes)), "rows_per_batch_max": int(max(sizes)),
                "pred_std": float(net(Xt).std()), "w_total_norm": float(total.norm())}
    return net, diag


# --------------------------------------------------------------------------- #
# Data: loaded once per worker
# --------------------------------------------------------------------------- #

_CACHE = {}


def blocks():
    """The four walk-forward blocks, scaled, each training window grouped by day."""
    if "blocks" in _CACHE:
        return _CACHE["blocks"]
    from src.data_mani.utils import (
        load_tournament_data, load_reference, attach_reference, load_meta_model,
        attach_meta_model, REFERENCE_COL, META_MODEL_COL)
    from src.prediction.models import ScaleContinuousOnly
    df, feats = load_tournament_data(tournament="crypto", version=DATA, era_step=1,
                                     val_frac=VF, embargo=EMB, verbose=False, target=TGT)
    if len(feats) != NFEAT:
        raise SystemExit("%s loads %d features, expected %d" % (DATA, len(feats), NFEAT))
    ref, rcol = load_reference("crypto", DATA, verbose=False)
    df, _ = attach_reference(df, ref, "crypto", rcol)
    meta, _ = load_meta_model("crypto", DATA, verbose=False)
    df, _ = attach_meta_model(df, meta, "crypto")
    allp = sorted(pd.unique(df["date"])); pos = {p: i for i, p in enumerate(allp)}
    valp = sorted(pd.unique(df.loc[df["split"] == "validation", "date"]))
    out = []
    for ri, start in enumerate(range(0, len(valp), REFIT)):
        block = valp[start:start + REFIT]
        te = max(0, pos[block[0]] - EMB); tp = allp[max(0, te - WIN):te]
        # The leakage boundary, checked rather than assumed: regrouping rows
        # into days must not be able to move a training row past the embargo.
        if pos[block[0]] - pos[tp[-1]] <= EMB:
            raise SystemExit("refit %d: training reaches inside the embargo" % ri)
        tm, bm = df["date"].isin(tp), df["date"].isin(block)
        Xtr = df.loc[tm, feats].to_numpy("float64")
        scaler = ScaleContinuousOnly().fit(Xtr)
        dates = df.loc[tm, "date"].to_numpy()
        out.append({
            "refit": ri, "Xtr": scaler.transform(Xtr),
            "ytr": df.loc[tm, TGT].to_numpy("float64"),
            "dates": dates, "day_rows": rows_by_day(dates),
            "Xbl": scaler.transform(df.loc[bm, feats].to_numpy("float64")),
            "frame": pd.DataFrame({
                "symbol": df.loc[bm, "symbol"].to_numpy(),
                "date": df.loc[bm, "date"].to_numpy(),
                TGT: df.loc[bm, TGT].to_numpy(),
                "refit": ri,
                REFERENCE_COL: df.loc[bm, REFERENCE_COL].to_numpy(),
                META_MODEL_COL: df.loc[bm, META_MODEL_COL].to_numpy()}),
            "train_start": str(tp[0])[:10], "train_end": str(tp[-1])[:10],
            "predict_start": str(block[0])[:10], "predict_end": str(block[-1])[:10],
        })
    _CACHE["blocks"] = out
    return out


def run_one(task):
    """One (arm, seed): four refits, scored by the repo's own metrics, written out."""
    arm, seed = task
    name = run_name(arm, seed)
    if os.path.exists(path("metrics", name)) and not FORCE:
        return name, "skipped", None
    started = time.time()
    try:
        from src.data_mani.utils import REFERENCE_COL, META_MODEL_COL
        from src.prediction.metrics import per_period_table, summarize_predictions
        torch = _torch()
        scheme, dpb, l2_norm = ARMS[arm]
        frames, refits = [], []
        for b in blocks():
            k = auto_days(b["day_rows"]) if dpb == "auto" else dpb   # days, or rows for a row arm
            net, diag = train(build_net(b["Xtr"].shape[1], seed), b["Xtr"], b["ytr"],
                              b["day_rows"], scheme, k, l2_norm, seed)
            k = k if scheme == "day" else None
            frame = b["frame"].copy()
            frame.insert(3, "prediction", net(torch.tensor(b["Xbl"])).detach().numpy().ravel())
            frames.append(frame)
            counts = [len(r) for r in b["day_rows"]]
            diag.update({"refit": b["refit"], "days_per_batch": k,
                         "train_rows": int(len(b["Xtr"])), "train_periods": len(counts),
                         "stocks_per_day_min": int(min(counts)),
                         "stocks_per_day_max": int(max(counts)),
                         "train_start": b["train_start"], "train_end": b["train_end"],
                         "predict_start": b["predict_start"], "predict_end": b["predict_end"]})
            refits.append(diag)
        pred = pd.concat(frames, ignore_index=True)
        kw = dict(reference_col=REFERENCE_COL, meta_model_col=META_MODEL_COL)
        table = per_period_table(pred, "prediction", TGT, "date", **kw)
        results = summarize_predictions(pred, "prediction", TGT, "date", **kw)
        scorable = int((pred.groupby("date")["prediction"].std() > 0).sum())
        results.update({
            "config": {
                "tournament": "crypto", "model": TREE, "task": "regression",
                "data_version": DATA, "n_features": NFEAT, "target": TGT,
                "cv_scheme": "rolling", "train_window": WIN, "refit_every": REFIT,
                "embargo": EMB, "val_frac": VF, "era_step": 1, "seed": seed,
                "init": "sklearn _init_coef", "batch_scheme": scheme,
                "days_per_batch": refits[0]["days_per_batch"], "l2_norm": l2_norm,
                "fixed_params": {"hidden_layer_sizes": list(HID), "activation": "identity",
                                 "alpha": ALPHA, "learning_rate_init": LR,
                                 "batch_size": BATCH}},
            "refits": refits, "skipped_blocks": [], "scorable_periods": scorable,
            "runtime_minutes": round((time.time() - started) / 60.0, 3)})
        for kind in ("per_period", "metrics", "forecast"):
            os.makedirs(os.path.dirname(path(kind, name)), exist_ok=True)
        table.to_csv(path("per_period", name))
        if seed in FC_SEEDS:
            pred.to_csv(path("forecast", name), index=False)
        with open(path("metrics", name), "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True, default=str)
        return name, "ok", (results["corr"]["mean"], scorable, results["runtime_minutes"])
    except Exception as error:                      # one arm must not cost the others
        return name, "FAILED %r" % (error,), None


# --------------------------------------------------------------------------- #
# Self-test: the sampler's invariants, on real windows and on a ragged one
# --------------------------------------------------------------------------- #

def _check_epoch(dates, day_rows, k):
    seen, sizes = [], []
    order = np.random.RandomState(0).permutation(len(day_rows))
    for rows, weights, n_days in day_batches(day_rows, order, k):
        days = pd.unique(dates[rows])
        assert len(days) == n_days <= k, "a batch holds exactly its k days"
        assert len(rows) == int(np.isin(dates, days).sum()), "and EVERY stock of them"
        assert abs(weights.sum() - 1.0) < 1e-12, "weights sum to one"
        for day in days:
            assert abs(weights[dates[rows] == day].sum() - 1.0 / n_days) < 1e-12, \
                "each day weighs 1/k whatever its breadth"
        seen.append(rows); sizes.append(len(rows))
    seen = np.concatenate(seen)
    assert len(seen) == len(dates) == len(np.unique(seen)), "every row once per epoch"
    return sizes


def selftest():
    torch = _torch()
    for b in blocks():
        for k in (1, auto_days(b["day_rows"]), 4):
            sizes = _check_epoch(b["dates"], b["day_rows"], k)
        print("refit %d: %d days x %d..%d stocks, auto k=%d -> %d batches of %d..%d rows  OK" % (
            b["refit"], len(b["day_rows"]), min(len(r) for r in b["day_rows"]),
            max(len(r) for r in b["day_rows"]), auto_days(b["day_rows"]),
            len(_check_epoch(b["dates"], b["day_rows"], auto_days(b["day_rows"]))),
            min(_check_epoch(b["dates"], b["day_rows"], auto_days(b["day_rows"]))),
            max(_check_epoch(b["dates"], b["day_rows"], auto_days(b["day_rows"])))))
    # These four windows happen to hold exactly 300 stocks every day, so breadth
    # is thinned by hand to the panel's real range -- plus a 1-stock and a
    # 7-stock day -- to prove the sampler and the loop on a genuinely ragged one.
    b = blocks()[0]; rng = np.random.RandomState(1)
    keep = np.ones(len(b["dates"]), dtype=bool)
    for i, rows in enumerate(b["day_rows"]):
        drop = len(rows) - 1 if i == 5 else len(rows) - 7 if i == 9 else rng.randint(0, 60)
        keep[rng.choice(rows, size=drop, replace=False)] = False
    dates, X, y = b["dates"][keep], b["Xtr"][keep], b["ytr"][keep]
    ragged = rows_by_day(dates)
    counts = sorted(len(r) for r in ragged)
    for k in (1, 3, 4):
        sizes = _check_epoch(dates, ragged, k)
    print("ragged window: stocks/day %d..%d (two thinnest %s), k=4 batches of %d..%d rows  OK" % (
        counts[0], counts[-1], counts[:2], min(sizes), max(sizes)))
    for l2_norm in ("ref", "batch"):
        _, diag = train(build_net(X.shape[1], 42), X, y, ragged, "day", 3, l2_norm, 42, max_iter=2)
        assert math.isfinite(diag["best_loss"]) and diag["rows_per_batch_min"] < diag["rows_per_batch_max"]
        print("ragged train, L2/%s: 2 epochs, %d steps, rows/batch %d..%d, loss %.5f  OK" % (
            l2_norm, diag["steps_per_epoch"], diag["rows_per_batch_min"],
            diag["rows_per_batch_max"], diag["best_loss"]))
    # With equal counts the day-balanced loss must BE the row mean.
    rows, weights, _ = next(day_batches(b["day_rows"], np.arange(len(b["day_rows"])), 3))
    err = torch.tensor(rng.randn(len(rows)))
    gap = abs(float((torch.as_tensor(weights) * err ** 2).sum() - (err ** 2).mean()))
    assert gap < 1e-12
    print("equal counts: day-balanced loss - row mean = %.1e  OK" % gap)
    print("SELFTEST PASS")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def report():
    from src.analysis import results_table as rt
    from src.analysis.ablation_report import newey_west
    lines = []; say = lines.append
    runs, missing = {}, []
    for arm in SELECTED:
        for seed in SEEDS:
            name = run_name(arm, seed)
            row = rt.row_for("crypto", TREE, "%s/%s" % (TREE, name))
            if row is None:
                missing.append(name); continue
            periods = pd.read_csv(path("per_period", name), parse_dates=["date"]).set_index("date")
            runs[(arm, seed)] = {
                "row": row, "meta": json.load(open(path("metrics", name))),
                "series": pd.DataFrame({"corr": rt.charged_corr(periods), "mmc": periods["mmc"],
                                        "payout": rt.payout_series(periods)})}
    say("%s -- day-indexed minibatches vs row-shuffled minibatches" % EXPERIMENT)
    say("model   torch from scratch, sklearn init, identity (64,32,16), alpha %g, lr %g, batch %d"
        % (ALPHA, LR, BATCH))
    say("data    %s (%df), %s, rolling %d/%d, embargo %d" % (DATA, NFEAT, TGT, WIN, REFIT, EMB))
    say("runs    %d of %d on disk, arms %s, seeds %d..%d (%d)" % (
        len(runs), len(SELECTED) * len(SEEDS), " ".join(SELECTED), min(SEEDS), max(SEEDS),
        len(SEEDS)))
    if missing:
        say("MISSING (not in any table below): %s" % " ".join(missing))

    ref = path("per_period", BASELINE_REF)
    if ("row", 42) in runs and os.path.exists(ref):
        a = pd.read_csv(ref, parse_dates=["date"]).set_index("date")["corr"]
        b = pd.read_csv(path("per_period", run_name("row", 42)),
                        parse_dates=["date"]).set_index("date")["corr"]
        gap = float((a - b).abs().max())
        say("\nBASELINE GATE  row s42 vs %s/%s: max |d corr| over %d periods = %.1e  %s"
            % (TREE, BASELINE_REF, len(a), gap,
               "PASS" if gap < 1e-9 else "FAIL -- row is not the run it claims to be"))

    def stat(run, key):
        return float(np.mean([r[key] for r in run["meta"]["refits"]]))

    say("\n== every run (n = periods; scor = periods with a non-constant prediction) ==")
    say("%-18s %4s %4s %9s %9s %9s %8s %8s %7s %7s %5s %6s %8s %9s" % (
        "run", "n", "scor", "CORR", "MMC", "payout", "c.sharpe", "p.sharpe", "dd(pay)",
        "calmar", "burn", "epochs", "steps/ep", "|W_total|"))
    for (arm, seed), run in runs.items():
        r = run["row"]
        calmar = "  const" if r["calmar"] == float("inf") else "%7.3f" % r["calmar"]
        say("%-18s %4d %4d %+9.5f %+9.5f %+9.5f %8.2f %8.2f %7.2f %s %5d %6.0f %8.0f %9.1e" % (
            run_name(arm, seed), r["n"], run["meta"]["scorable_periods"], r["CORR"], r["MMC"],
            r["payout"], r["c.sharpe"] or 0.0, r["p.sharpe"] or 0.0, r["dd(pay)"], calmar,
            r["burn(d)"], stat(run, "epochs"), stat(run, "steps_per_epoch"),
            stat(run, "w_total_norm")))

    say("\n== per arm, across seeds (const = seeds that predict a constant in EVERY period;")
    say("   part = in some. calmar is the median over seeds that are not const, n=510 each) ==")
    say("   steps = optimiser steps before the stop; |W| = median norm of the net linear map.")
    say("%-8s %5s %5s %4s | %9s %9s %8s %9s %9s | %9s %9s %8s %7s | %6s %8s %7s %6s %8s" % (
        "arm", "seeds", "const", "part", "CORR mean", "median", "sd", "min", "max",
        "MMC mean", "pay mean", "p.sharpe", "calmar", "epochs", "steps/ep", "rows/b",
        "steps", "|W|"))
    for arm in SELECTED:
        got = [run for (a, _), run in runs.items() if a == arm]
        if not got:
            continue
        corr = np.array([g["row"]["CORR"] for g in got])
        n_scor = np.array([g["meta"]["scorable_periods"] for g in got])
        n = np.array([g["row"]["n"] for g in got])
        calm = [g["row"]["calmar"] for g in got if g["row"]["calmar"] != float("inf")]
        steps = np.mean([np.mean([r["epochs"] * r["steps_per_epoch"] for r in g["meta"]["refits"]])
                         for g in got])
        w_norm = np.median([r["w_total_norm"] for g in got for r in g["meta"]["refits"]])
        say("%-8s %5d %5d %4d | %+9.5f %+9.5f %8.5f %+9.5f %+9.5f | %+9.5f %+9.5f %8.2f %7.3f | %6.0f %8.0f %7.0f %6.0f %8.1e" % (
            arm, len(got), int((n_scor == 0).sum()), int(((n_scor > 0) & (n_scor < n)).sum()),
            corr.mean(), np.median(corr), corr.std(), corr.min(), corr.max(),
            np.mean([g["row"]["MMC"] for g in got]), np.mean([g["row"]["payout"] for g in got]),
            # a constant predictor has no sharpe (0/0); average the seeds that do
            np.nanmean([g["row"]["p.sharpe"] if g["row"]["calmar"] != float("inf")
                        else np.nan for g in got] + [np.nan]) if calm else float("nan"),
            np.median(calm) if calm else float("nan"),
            np.mean([stat(g, "epochs") for g in got]),
            np.mean([stat(g, "steps_per_epoch") for g in got]),
            np.mean([stat(g, "rows_per_batch_max") for g in got]), steps, w_norm))

    # The outcome is close to a three-state lottery per seed -- works, constant,
    # or sign-flipped -- so the per-seed view is the one to read first.
    for metric, key in (("CORR", "CORR"), ("payout", "payout"), ("calmar", "calmar")):
        say("\n== %s by seed (const = a constant predictor in every period) ==" % metric)
        say("%-6s " % "seed" + " ".join("%9s" % a for a in SELECTED))
        for seed in SEEDS:
            cells = []
            for arm in SELECTED:
                run = runs.get((arm, seed))
                if run is None:
                    cells.append("%9s" % "-")
                elif run["meta"]["scorable_periods"] == 0:
                    cells.append("%9s" % "const")
                else:
                    cells.append(("%9.3f" if key == "calmar" else "%+9.5f") % run["row"][key])
            say("%-6d " % seed + " ".join(cells))

    say("\n== paired, per period, Newey-West lag 20, on the SEED-MEAN delta (arm - base) ==")
    say("   t = over TIME, the %d seeds taken as given.  seed t = over SEEDS (%d df), each"
        % (len(SEEDS), len(SEEDS) - 1))
    say("   seed's mean delta one observation -- the bar a claim about the SCHEME must clear.")
    say("   wins = seeds whose own mean delta is positive; t range = the per-seed NW t")
    say("%-9s %-8s %-7s %5s %11s %10s %7s %7s %5s %15s  %s" % (
        "arm", "base", "metric", "seeds", "delta", "NW se", "t", "seed t", "wins", "t range",
        "what it isolates"))
    for arm, base, what in PAIRS:
        common = [s for s in SEEDS if (arm, s) in runs and (base, s) in runs]
        if not common:
            continue
        for metric in ("corr", "mmc", "payout"):
            delta = pd.DataFrame({s: runs[(arm, s)]["series"][metric]
                                  - runs[(base, s)]["series"][metric] for s in common})
            mean, se, t = newey_west(delta.mean(axis=1).dropna(), lag=20)
            each = [newey_west(delta[s].dropna(), lag=20)[2] for s in common]
            each = [e for e in each if e == e]
            by_seed = delta.mean()
            spread = by_seed.std(ddof=1) / math.sqrt(len(by_seed)) if len(by_seed) > 1 else 0.0
            seed_t = by_seed.mean() / spread if spread > 0 else float("nan")
            say("%-9s %-8s %-7s %5d %+11.6f %10.6f %+7.2f %+7.2f %2d/%-2d %+7.2f..%+6.2f  %s" % (
                arm, base, metric, len(common), mean, se, t, seed_t,
                int((delta.mean() > 0).sum()), len(common),
                min(each) if each else float("nan"), max(each) if each else float("nan"),
                what if metric == "corr" else ""))
    say("\n510 periods at ~95% overlap is ~26 effective windows; |t| >= 2 is the house bar.")
    say("calmar and burn are comparable here because every row is n=510 on one window.")
    text = "\n".join(lines)
    with open("logs/%s_report%s.txt" % (PREFIX, "_" + TAG if TAG else ""), "w") as handle:
        handle.write(text + "\n")
    print(text)
    return text


# --------------------------------------------------------------------------- #
# Boxplots across seeds
# --------------------------------------------------------------------------- #

# Categorical slots 1-3 of the dataviz reference palette, in order; validated
# all-pairs on the light surface (CVD dE 9.2, normal-vision dE 24.0).
ARM_COLOR = {"row": "#2a78d6", "row900": "#eb6834", "day3": "#1baf7a"}
ARM_LABEL = {"row": "row\n1024 rows\n(baseline)",
             "row900": "row900\n900 rows\n(same size as day3)",
             "day3": "day3\n3 whole days\n(900 rows)"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
METRICS = [("CORR", "CORR", "mean per-period CORR"), ("MMC", "MMC", "mean per-period MMC"),
           ("dd_pay", "dd(pay)", "max drawdown of cumulative payout")]


def seed_table():
    """
    One row per (arm, seed), on the seeds EVERY selected arm has.

    Read through results_table.row_for, so CORR charges a constant period the 0
    it earned and dd(pay) is the same number the reports print.
    """
    from src.analysis import results_table as rt
    rows = []
    for arm in SELECTED:
        for seed in SEEDS:
            name = run_name(arm, seed)
            row = rt.row_for("crypto", TREE, "%s/%s" % (TREE, name))
            if row is None:
                continue
            scorable = json.load(open(path("metrics", name)))["scorable_periods"]
            rows.append({"arm": arm, "seed": seed, "n": row["n"], "scorable": scorable,
                         "const": scorable == 0, "CORR": row["CORR"], "MMC": row["MMC"],
                         "payout": row["payout"], "dd_pay": row["dd(pay)"]})
    table = pd.DataFrame(rows)
    shared = set.intersection(*[set(table.loc[table["arm"] == a, "seed"]) for a in SELECTED])
    return table[table["seed"].isin(shared)].sort_values(["arm", "seed"]).reset_index(drop=True)


def seed_stats(table):
    """The numbers behind the boxplots: per arm, then paired by seed against day3."""
    from scipy.stats import wilcoxon
    lines = []; say = lines.append
    seeds = sorted(table["seed"].unique())
    say("%s boxplot statistics -- %d seeds (%d..%d), the same seeds in every arm, n=510 each"
        % (EXPERIMENT, len(seeds), seeds[0], seeds[-1]))
    say("\n== per arm (const = a constant predictor in every period; dd(pay) over the rest) ==")
    say("%-7s %5s %5s %4s | %24s | %24s | %24s" % (
        "arm", "seeds", "const", "neg", "CORR  q25 / median / q75", "MMC  q25 / median / q75",
        "dd(pay)  q25 / med / q75"))
    for arm in [a for a in ARM_COLOR if a in SELECTED]:
        part = table[table["arm"] == arm]; live = part[~part["const"]]
        q = lambda s: tuple(np.percentile(s, [25, 50, 75]))
        say("%-7s %5d %5d %4d | %+7.4f /%+7.4f /%+7.4f | %+7.4f /%+7.4f /%+7.4f | %6.2f / %6.2f / %6.2f" % (
            (arm, len(part), int(part["const"].sum()), int((part["CORR"] < 0).sum()))
            + q(part["CORR"]) + q(part["MMC"]) + q(live["dd_pay"])))
    wide = table.pivot(index="seed", columns="arm")
    say("\n== paired by seed: day3 minus base (wilcoxon = signed-rank, two-sided) ==")
    say("%-7s %-8s %5s %11s %11s %14s %8s %10s" % (
        "base", "metric", "seeds", "mean", "median", "up/same/down", "paired t", "wilcoxon p"))
    for base in [a for a in SELECTED if a != "day3"]:
        for key, label, _ in METRICS:
            d = (wide[key]["day3"] - wide[key][base]).dropna()
            se = d.std(ddof=1) / math.sqrt(len(d))
            moved = d[d != 0]
            say("%-7s %-8s %5d %+11.6f %+11.6f %5d/%3d/%4d %+8.2f %10.3f" % (
                base, label, len(d), d.mean(), d.median(), int((d > 0).sum()),
                int((d == 0).sum()), int((d < 0).sum()), d.mean() / se if se > 0 else float("nan"),
                wilcoxon(moved).pvalue if len(moved) > 5 else float("nan")))
    state = lambda t: np.where(t["const"], "const", np.where(t["CORR"] < 0, "negative", "positive"))
    for base in [a for a in SELECTED if a != "day3"]:
        a = table[table["arm"] == base].set_index("seed"); b = table[table["arm"] == "day3"].set_index("seed")
        cross = pd.crosstab(pd.Series(state(a), index=a.index, name=base),
                            pd.Series(state(b), index=b.index, name="day3"))
        say("\n== where each seed lands: rows = %s, columns = day3 ==" % base)
        say(cross.to_string())
    return "\n".join(lines)


def boxplot():
    """Distributions across seeds (top) and the same seeds paired against day3 (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = seed_table()
    tag = "_" + TAG if TAG else ""
    table.to_csv("logs/%s_seed_metrics%s.csv" % (PREFIX, tag), index=False)
    text = seed_stats(table)
    with open("logs/%s_boxplot_stats%s.txt" % (PREFIX, tag), "w") as handle:
        handle.write(text + "\n")
    print(text)

    seeds = sorted(table["seed"].unique())
    # One horizontal offset per SEED, reused in every box, so a seed sits at the
    # same place within each arm's cloud.
    jitter = dict(zip(seeds, np.random.RandomState(0).uniform(-0.19, 0.19, len(seeds))))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS,
                         "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": MUTED,
                         "text.color": INK, "axes.labelcolor": INK2})
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.4), dpi=160, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.5, "wspace": 0.24})

    def frame(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.7); ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", length=0)

    def box(ax, values, at, color):
        parts = ax.boxplot([values], positions=[at], widths=0.5, whis=(5, 95), showfliers=False,
                           patch_artist=True, manage_ticks=False, zorder=3)
        parts["boxes"][0].set(facecolor=color, alpha=0.14, edgecolor="none")
        parts["medians"][0].set(color=color, linewidth=2.2)
        for item in parts["whiskers"] + parts["caps"]:
            item.set(color=color, linewidth=1.0)

    arms = [a for a in ARM_COLOR if a in SELECTED]      # baseline, its twin, the day scheme
    for col, (key, label, what) in enumerate(METRICS):
        ax = axes[0, col]; frame(ax)
        log = key == "dd_pay"
        if log:
            ax.set_yscale("log")
        for i, arm in enumerate(arms):
            part = table[table["arm"] == arm]; live = part[~part["const"]]; dead = part[part["const"]]
            color = ARM_COLOR[arm]
            # A constant predictor's drawdown is 0 because it never bet; it has no
            # place on a log axis or in the box, and gets its own strip instead.
            inbox = live if log else part
            box(ax, inbox[key].to_numpy(), i, color)
            ax.scatter(i + live["seed"].map(jitter), live[key], s=17, color=color, alpha=0.6,
                       edgecolor=SURFACE, linewidth=0.5, zorder=4)
            median = float(np.median(inbox[key]))
            ax.annotate(("%.2f" if log else "%+.3f") % median, (i + 0.29, median), fontsize=8.5,
                        color=INK, va="center", ha="left", zorder=5)
            if len(dead):
                y = 0.021 if log else 0.0
                ax.scatter(i + dead["seed"].map(jitter), np.full(len(dead), y), s=17,
                           facecolor=SURFACE, edgecolor=MUTED, linewidth=0.9, zorder=4)
        if log:
            ax.set_ylim(0.015, 60)
            ax.axhline(0.03, color=AXIS, linewidth=0.8)
            ax.text(-0.46, 0.021, "constant", fontsize=7.5, color=MUTED, va="center", ha="right")
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=8.5)
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=INK, pad=20)
        ax.text(0, 1.035, what + (" · log scale, bigger is worse" if log else ""),
                transform=ax.transAxes, fontsize=8.5, color=INK2)

    wide = table.pivot(index="seed", columns="arm")
    bases = [a for a in arms if a != "day3"]
    for col, (key, label, _) in enumerate(METRICS):
        ax = axes[1, col]; frame(ax)
        ax.set_yscale("symlog", linthresh=0.01 if key != "dd_pay" else 0.1)
        ax.axhline(0, color=AXIS, linewidth=0.9, zorder=2)
        ticks = []
        for i, base in enumerate(bases):
            delta = (wide[key]["day3"] - wide[key][base]).dropna()
            both_dead = (wide["const"]["day3"] & wide["const"][base]).reindex(delta.index).fillna(False)
            box(ax, delta.to_numpy(), i, INK2)
            x = i + delta.index.map(jitter)
            ax.scatter(x[~both_dead], delta[~both_dead], s=15, color=INK2, alpha=0.55,
                       edgecolor=SURFACE, linewidth=0.5, zorder=4)
            ax.scatter(x[both_dead], delta[both_dead], s=15, facecolor=SURFACE, edgecolor=MUTED,
                       linewidth=0.9, zorder=4)
            # The median goes in the tick label: beside the box it lands on the
            # neighbouring cloud once there are a hundred points in it.
            ticks.append("day3 − %s\nmedian %+.4f\n%d seeds up · %d down" % (
                base, delta.median(), int((delta > 0).sum()), int((delta < 0).sum())))
        ax.set_xlim(-0.6, len(bases) - 0.4)
        ax.set_xticks(range(len(bases)))
        ax.set_xticklabels(ticks, fontsize=8.5)
        ax.set_title("%s, paired by seed" % label, loc="left", fontsize=11, fontweight="bold",
                     color=INK, pad=20)
        ax.text(0, 1.035, "day3 minus the row arm, same seed · symmetric log scale",
                transform=ax.transAxes, fontsize=8.5, color=INK2)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=ARM_COLOR[a],
                          label=a) for a in arms]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=SURFACE,
                              markeredgecolor=MUTED, label="constant predictor"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.3, columnspacing=1.4)
    dead = " · ".join("%s %d" % (a, int(table.loc[table["arm"] == a, "const"].sum())) for a in arms)
    fig.suptitle("Shuffled-day batches vs shuffled-row batches, %d seeds" % len(seeds),
                 x=0.035, y=0.985, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.035, 0.948, "torch MLP, sklearn init, identity (64,32,16), alpha 100 · fenccausal 44f · "
             "rolling 130/130 · 510 periods per run · seeds %d–%d, identical in every arm"
             % (seeds[0], seeds[-1]), fontsize=9.5, color=INK2)
    fig.text(0.035, 0.012, "Box = quartiles, line = median, whiskers = 5th–95th percentile; one dot per "
             "seed. Constant predictors (hollow) score CORR = MMC = 0 and are in those boxes; they have "
             "no drawdown and are\nkept out of the dd(pay) box. Constant on every period: %s of %d seeds."
             % (dead, len(seeds)), fontsize=8.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.145)
    os.makedirs("results/plots/crypto", exist_ok=True)
    out = "results/plots/crypto/%s_seed_boxplots%s.png" % (PREFIX, tag)
    fig.savefig(out, facecolor=SURFACE)
    print("\nwrote %s" % out)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "selftest":
        return selftest()
    if mode == "report":
        return report()
    if mode == "boxplot":
        return boxplot()
    arms = [a for a in ARMS if not ONLY or a in ONLY]
    # one-day arms take ~3x the steps per epoch, so they go to the pool first
    arms.sort(key=lambda a: 0 if ARMS[a][1] == 1 else 1)
    tasks = [(a, s) for a in arms for s in SEEDS]
    print("%s: %d arms x %d seeds = %d runs, %d at a time" % (
        EXPERIMENT, len(arms), len(SEEDS), len(tasks), JOBS))
    if DRY:
        for arm, seed in tasks:
            print("  %-18s %s" % (run_name(arm, seed), ARMS[arm]))
        return
    started, failed = time.time(), []
    with cf.ProcessPoolExecutor(max_workers=JOBS, mp_context=mp.get_context("spawn")) as pool:
        for i, (name, status, info) in enumerate(pool.map(run_one, tasks), 1):
            if status.startswith("FAILED"):
                failed.append(name)
            print("[%2d/%d] %-18s %s%s" % (i, len(tasks), name, status,
                  "" if info is None else "  corr %+.5f  scorable %d  %.1f min" % info), flush=True)
    text = report()
    tail = "\n%d runs in %.1f min, %d failed%s" % (
        len(tasks), (time.time() - started) / 60.0, len(failed),
        ": " + " ".join(failed) if failed else "")
    print(tail)
    if not NOMAIL:
        try:
            from src.live.notify import send_text
            send_text("[crypto] %s %s" % (EXPERIMENT, "FAILED arms" if failed else "finished"),
                      text + tail)
        except Exception as error:                  # SMTP must never decide the exit status
            print("note: mail failed (%r), run unaffected" % (error,))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
