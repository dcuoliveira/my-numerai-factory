#!/usr/bin/env python3
"""
E105 -- learning-rate schedulers on the day3 batch scheme.

WHAT IT RUNS
  E104's day3 arm, unchanged -- the from-scratch torch MLP with sklearn's exact
  initial weights (identity (64,32,16), alpha 100, Adam, 0.5*MSE, fenccausal 44f,
  rolling 130/130, embargo 20), a batch being every stock of 3 shuffled days with
  the L2 term pinned to alpha/1024 -- and varies ONLY the learning rate over
  training:

    const      1e-3 throughout. The baseline; gated bit for bit against
               torch_scratch/e104_day3_s* on every seed.
    step       StepLR, x0.5 every 15 epochs
    exp        ExponentialLR, gamma 0.955 (6% of the peak by epoch 60)
    cosine     CosineAnnealingLR, T_max 60, down to 1e-5
    warmcos    5 epochs of linear warm-up from 10%, then cosine to 1e-5 by 60
    restarts   CosineAnnealingWarmRestarts, T_0 15
    onecycle   OneCycleLR, peak 1e-3, stepped per batch (it cycles beta1 too)
    plateau    ReduceLROnPlateau on the epoch loss, x0.5, patience 3, threshold
               1e-3 (at the stop rule's 1e-4 it never fires -- see make_scheduler)

  Every scheduler is torch's own class and peaks at the baseline's 1e-3. The
  horizon is 60 epochs because constant-lr training stops at ~59: a schedule laid
  over the 200-epoch cap would still be at ~80% of its peak when the run ends.

  Every schedule also lowers the AVERAGE learning rate, so two more arms hold the
  rate constant at a lower level -- lr5e4 and lr25e5. They separate "the shape of
  the schedule" from "less learning rate", the way row900 separated what a batch
  is made of from how big it is in E104.

MEASURED AGAINST
  const, on the same 100 seeds (42..141), paired by seed. This arm is numerically
  degenerate: its outcome is close to a three-state lottery per seed (works /
  constant / sign-flipped), so the distribution across seeds is the result and a
  single seed is not.

WHAT COUNTS AS A RESULT
  A scheduler whose paired-by-seed difference from const clears |t| >= 2 on
  payout AND is not matched by a constant-lr control. Nine arms are tested, so a
  lone |t| near 2 is the expected false positive, not a finding.

Env knobs: E105_SEEDS (42..141), E105_JOBS (30), E105_ONLY (arms), E105_FORCE=1,
DRY_RUN=1, NO_MAIL=1. Modes: `gate`, `report`, `boxplot`; default runs, then all
three. INTERPRETER: this file is run with /home/danielco/miniconda3/bin/python3.
"""
import os, sys, json, math, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e104_daybatch_torch_mlp_crypto as e104    # pins the thread count, chdirs to the repo

import concurrent.futures as cf, multiprocessing as mp
import numpy as np, pandas as pd

EXPERIMENT, PREFIX, TREE = "E105", "e105", e104.TREE
LR, HORIZON, FLOOR, DAYS = e104.LR, 60, 1e-5, 3
TGT, ALPHA, BATCH, MAXIT, TOL, PAT = e104.TGT, e104.ALPHA, e104.BATCH, e104.MAXIT, e104.TOL, e104.PAT

# arm -> (base lr, scheduler or None, group, how it reads on an axis)
ARMS = {
    "const":    (1e-3,   None,       "baseline",  "constant 1e-3"),
    "lr5e4":    (5e-4,   None,       "constant",  "constant 5e-4"),
    "lr25e5":   (2.5e-4, None,       "constant",  "constant 2.5e-4"),
    "step":     (LR,     "step",     "scheduler", "StepLR  ×0.5 / 15 ep"),
    "exp":      (LR,     "exp",      "scheduler", "ExponentialLR  γ 0.955"),
    "cosine":   (LR,     "cosine",   "scheduler", "CosineAnnealingLR"),
    "warmcos":  (LR,     "warmcos",  "scheduler", "warm-up + cosine"),
    "restarts": (LR,     "restarts", "scheduler", "cosine warm restarts"),
    "onecycle": (LR,     "onecycle", "scheduler", "OneCycleLR"),
    "plateau":  (LR,     "plateau",  "scheduler", "ReduceLROnPlateau"),
}
SEEDS = [int(s) for s in os.environ.get(
    "E105_SEEDS", ",".join(str(s) for s in range(42, 142))).split(",")]
JOBS = int(os.environ.get("E105_JOBS", "30"))
ONLY = [a for a in os.environ.get("E105_ONLY", "").split(",") if a]
FORCE = os.environ.get("E105_FORCE") == "1"
DRY = os.environ.get("DRY_RUN") == "1"
NOMAIL = os.environ.get("NO_MAIL") == "1"
SELECTED = [a for a in ARMS if not ONLY or a in ONLY]


def run_name(arm, seed):
    return "%s_%s_s%d" % (PREFIX, arm, seed)


def path(kind, name):
    return os.path.join("results", kind, "crypto", TREE,
                        name + (".json" if kind == "metrics" else ".csv"))


def make_scheduler(name, opt, steps_per_epoch):
    """
    One of torch's schedulers on `opt`.

    :return: (scheduler, when it steps, how many times it may step). The cap is
             what holds a schedule at its end value: CosineAnnealingLR climbs
             back up past T_max and OneCycleLR raises past total_steps.
    """
    import torch.optim.lr_scheduler as S
    if name == "step":
        return S.StepLR(opt, step_size=15, gamma=0.5), "epoch", None
    if name == "exp":
        return S.ExponentialLR(opt, gamma=0.955), "epoch", None
    if name == "cosine":
        return S.CosineAnnealingLR(opt, T_max=HORIZON, eta_min=FLOOR), "epoch", HORIZON
    if name == "warmcos":
        return S.SequentialLR(opt, [
            S.LinearLR(opt, start_factor=0.1, total_iters=5),
            S.CosineAnnealingLR(opt, T_max=HORIZON - 5, eta_min=FLOOR)], milestones=[5]), \
            "epoch", HORIZON
    if name == "restarts":
        return S.CosineAnnealingWarmRestarts(opt, T_0=15, eta_min=FLOOR), "epoch", None
    if name == "onecycle":
        total = HORIZON * steps_per_epoch
        return S.OneCycleLR(opt, max_lr=LR, total_steps=total), "batch", total - 1
    if name == "plateau":
        # 10x the stop rule's tolerance, and it has to be. The stop rule compares an
        # epoch with the RUNNING best; this class compares it with the last ACCEPTED
        # best. Past epoch ~40 the loss still creeps down 2-9e-5 an epoch, which the
        # stop rule counts as a stall and this, at 1e-4, counts as a gain every third
        # epoch -- so it never fired and the arm was the baseline bit for bit.
        return S.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, threshold=10 * TOL,
                                   threshold_mode="abs", min_lr=FLOOR), "plateau", None
    raise ValueError("unknown scheduler %r" % (name,))


def train(net, X, y, day_rows, seed, base_lr, sched):
    """
    E104's day3 loop with a scheduler hooked in.

    With sched=None and base_lr=LR nothing below differs from e104.train's
    day/"ref" path in any value it computes -- the gate checks that, it is not
    taken on trust.
    """
    torch = e104._torch(); from torch import nn
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8)
    weights = [m.weight for m in net if isinstance(m, nn.Linear)]
    rng = np.random.RandomState(int(seed)); best, bad = np.inf, 0
    per_epoch = int(math.ceil(len(day_rows) / float(DAYS)))
    scheduler, when, cap = make_scheduler(sched, opt, per_epoch) if sched else (None, None, None)
    epochs, steps, ticks, lr_sum = 0, 0, 0, 0.0
    for _ in range(int(MAXIT)):
        tot, seen = 0.0, 0
        for rows, day_w, n_days in e104.day_batches(day_rows, rng.permutation(len(day_rows)), DAYS):
            idx = torch.as_tensor(rows, dtype=torch.long)
            opt.zero_grad(set_to_none=True); out = net(Xt[idx])
            data = 0.5 * (torch.as_tensor(day_w).reshape(-1, 1) * (yt[idx] - out) ** 2).sum()
            loss = data + (0.5 * ALPHA) * sum((w * w).sum() for w in weights) / BATCH
            loss.backward(); lr_sum += opt.param_groups[0]["lr"]; opt.step()
            tot += float(loss.detach()) * n_days; seen += n_days; steps += 1
            if when == "batch" and ticks < cap:
                scheduler.step(); ticks += 1
        epochs += 1
        ep = tot / max(seen, 1)
        if when == "epoch" and (cap is None or ticks < cap):
            scheduler.step(); ticks += 1
        elif when == "plateau":
            scheduler.step(ep)
        bad = bad + 1 if ep > best - TOL else 0
        if ep < best:
            best = ep
        if bad > PAT:
            break
    net.eval()
    with torch.no_grad():
        total = None
        for lin in net:
            total = lin.weight if total is None else lin.weight @ total
        diag = {"epochs": epochs, "steps": steps, "lr_sum": lr_sum,
                "lr_last": float(opt.param_groups[0]["lr"]), "best_loss": float(best),
                "pred_std": float(net(Xt).std()), "w_total_norm": float(total.norm())}
    return net, diag


def run_one(task):
    """One (arm, seed): four refits, scored once, written in the standard layout."""
    arm, seed = task
    name = run_name(arm, seed)
    if os.path.exists(path("metrics", name)) and not FORCE:
        return name, "skipped", None
    started = time.time()
    try:
        from src.data_mani.utils import REFERENCE_COL, META_MODEL_COL
        from src.prediction.metrics import per_period_table, summarize
        torch = e104._torch()
        base_lr, sched, group, _ = ARMS[arm]
        frames, refits = [], []
        for b in e104.blocks():
            if e104.auto_days(b["day_rows"]) != DAYS:
                raise SystemExit("refit %d: day3 is no longer 3 days a batch" % b["refit"])
            net, diag = train(e104.build_net(b["Xtr"].shape[1], seed), b["Xtr"], b["ytr"],
                              b["day_rows"], seed, base_lr, sched)
            frame = b["frame"].copy()
            frame.insert(3, "prediction", net(torch.tensor(b["Xbl"])).detach().numpy().ravel())
            frames.append(frame)
            diag.update({"refit": b["refit"], "train_start": b["train_start"],
                         "train_end": b["train_end"], "predict_start": b["predict_start"],
                         "predict_end": b["predict_end"]})
            refits.append(diag)
        pred = pd.concat(frames, ignore_index=True)
        table = per_period_table(pred, "prediction", TGT, "date", reference_col=REFERENCE_COL,
                                 meta_model_col=META_MODEL_COL)
        # summarize_predictions would score every period a second time to reach
        # these same series; summarising the table's own columns is identical
        # (checked to 1e-15) and halves the scoring cost of a 1,000-run study.
        results = {"n_periods": int(table["corr"].notna().sum())}
        for family in ("corr", "bmc", "mmc"):
            results[family] = summarize(table[family])
            if family != "corr":
                results["%s_n_periods" % family] = int(table[family].notna().sum())
        by_day = pred.groupby("date")["prediction"]
        results.update({
            "config": {
                "tournament": "crypto", "model": TREE, "task": "regression",
                "data_version": e104.DATA, "n_features": e104.NFEAT, "target": TGT,
                "cv_scheme": "rolling", "train_window": e104.WIN, "refit_every": e104.REFIT,
                "embargo": e104.EMB, "val_frac": e104.VF, "era_step": 1, "seed": seed,
                "init": "sklearn _init_coef", "batch_scheme": "day", "days_per_batch": DAYS,
                "l2_norm": "ref", "lr_schedule": sched or "constant", "lr_group": group,
                "fixed_params": {"hidden_layer_sizes": list(e104.HID), "activation": "identity",
                                 "alpha": ALPHA, "learning_rate_init": base_lr,
                                 "batch_size": BATCH}},
            "refits": refits, "skipped_blocks": [],
            "scorable_periods": int((by_day.std() > 0).sum()),
            "distinct_per_day": float(by_day.nunique().median()),
            "runtime_minutes": round((time.time() - started) / 60.0, 3)})
        for kind in ("per_period", "metrics"):
            os.makedirs(os.path.dirname(path(kind, name)), exist_ok=True)
        table.to_csv(path("per_period", name))
        with open(path("metrics", name), "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True, default=str)
        return name, "ok", (results["corr"]["mean"], results["scorable_periods"],
                            results["runtime_minutes"])
    except Exception as error:                      # one arm must not cost the others
        return name, "FAILED %r" % (error,), None


# --------------------------------------------------------------------------- #
# Gate, tables, figure
# --------------------------------------------------------------------------- #

def gate():
    """const must BE e104's day3, period for period, on every seed both have."""
    worst, checked, missing = 0.0, 0, 0
    for seed in SEEDS:
        mine, theirs = path("per_period", run_name("const", seed)), \
            path("per_period", "e104_day3_s%d" % seed)
        if not (os.path.exists(mine) and os.path.exists(theirs)):
            missing += 1; continue
        a, b = pd.read_csv(mine), pd.read_csv(theirs)
        for column in ("corr", "mmc"):
            gap = (a[column] - b[column]).abs().max()
            same_nan = bool((a[column].isna() == b[column].isna()).all())
            worst = max(worst, 0.0 if gap != gap else float(gap))
            if not same_nan:
                worst = float("inf")
        checked += 1
    text = "BASELINE GATE  const vs torch_scratch/e104_day3: %d seeds checked, max |d| over " \
           "corr and mmc = %.1e  %s%s" % (
               checked, worst, "PASS" if checked and worst < 1e-12 else "FAIL",
               "" if not missing else "  (%d seeds not on disk in both)" % missing)
    print(text)
    return text


def seed_table():
    """One row per (arm, seed), on the seeds every selected arm has."""
    from src.analysis import results_table as rt
    rows = []
    for arm in SELECTED:
        for seed in SEEDS:
            name = run_name(arm, seed)
            row = rt.row_for("crypto", TREE, "%s/%s" % (TREE, name))
            if row is None:
                continue
            meta = json.load(open(path("metrics", name)))
            mean = lambda key: float(np.mean([r[key] for r in meta["refits"]]))
            rows.append({
                "arm": arm, "seed": seed, "n": row["n"], "const": meta["scorable_periods"] == 0,
                "CORR": row["CORR"], "MMC": row["MMC"], "payout": row["payout"],
                "dd_pay": row["dd(pay)"], "calmar": row["calmar"], "p_sharpe": row["p.sharpe"],
                "epochs": mean("epochs"), "steps": mean("steps"), "lr_sum": mean("lr_sum"),
                "lr_last": mean("lr_last"), "distinct": meta["distinct_per_day"],
                "w_norm": float(np.median([r["w_total_norm"] for r in meta["refits"]]))})
    table = pd.DataFrame(rows)
    shared = set.intersection(*[set(table.loc[table["arm"] == a, "seed"]) for a in SELECTED])
    return table[table["seed"].isin(shared)].sort_values(["arm", "seed"]).reset_index(drop=True)


def payout_series(arm, seed):
    from src.analysis import results_table as rt
    periods = pd.read_csv(path("per_period", run_name(arm, seed)),
                          parse_dates=["date"]).set_index("date")
    return pd.DataFrame({"corr": rt.charged_corr(periods), "mmc": periods["mmc"],
                         "payout": rt.payout_series(periods)})


def report():
    from scipy.stats import wilcoxon
    from src.analysis.ablation_report import newey_west
    table = seed_table()
    seeds = sorted(table["seed"].unique())
    arms = [a for a in SELECTED if a in set(table["arm"])]
    lines = []; say = lines.append
    say("%s -- learning-rate schedulers on the day3 batch scheme" % EXPERIMENT)
    say("model   torch from scratch, sklearn init, identity (64,32,16), alpha %g, batch = 3 "
        "shuffled days, L2 / %d" % (ALPHA, BATCH))
    say("data    %s (%df), %s, rolling %d/%d, embargo %d, n=510 per run" % (
        e104.DATA, e104.NFEAT, TGT, e104.WIN, e104.REFIT, e104.EMB))
    say("runs    %d on disk; %d seeds (%d..%d) shared by every arm: %s" % (
        len(table), len(seeds), seeds[0], seeds[-1], " ".join(arms)))
    say("\n" + gate())

    q = lambda s: tuple(np.percentile(s, [25, 50, 75]))
    say("\n== per arm, across seeds (const = a constant predictor in EVERY period) ==")
    say("   dd(pay) and calmar are medians over the seeds that are not constant; n=510 each.")
    say("%-9s %-22s %5s %5s %4s %6s | %25s %8s | %8s %8s %8s | %7s %7s" % (
        "arm", "learning rate", "seeds", "const", "neg", ">=.10", "CORR  q25 / median / q75",
        "mean", "MMC med", "MMC mean", "pay mean", "dd(pay)", "calmar"))
    for arm in arms:
        part = table[table["arm"] == arm]; live = part[~part["const"]]
        say("%-9s %-22s %5d %5d %4d %6d | %+8.4f /%+8.4f /%+8.4f %+8.5f | %+8.4f %+8.5f %+8.5f | %7.2f %7.3f" % (
            (arm, ARMS[arm][3], len(part), int(part["const"].sum()), int((part["CORR"] < 0).sum()),
             int((part["CORR"] >= 0.10).sum())) + q(part["CORR"])
            + (part["CORR"].mean(), part["MMC"].median(), part["MMC"].mean(),
               part["payout"].mean(), live["dd_pay"].median() if len(live) else float("nan"),
               live["calmar"].median() if len(live) else float("nan"))))

    say("\n== what the schedule did to training (means over seeds and refits; |W| and distinct")
    say("   are medians). sum lr = the learning rate summed over every optimiser step;")
    say("   distinct = different prediction values among a day's 300 stocks ==")
    say("%-9s %7s %7s %9s %10s %10s %9s" % (
        "arm", "epochs", "steps", "sum lr", "last lr", "|W|", "distinct"))
    for arm in arms:
        part = table[table["arm"] == arm]
        say("%-9s %7.0f %7.0f %9.3f %10.1e %10.1e %9.0f" % (
            arm, part["epochs"].mean(), part["steps"].mean(), part["lr_sum"].mean(),
            part["lr_last"].median(), part["w_norm"].median(), part["distinct"].median()))

    if "const" not in arms:                         # nothing to pair against
        text = "\n".join(lines + ["\nconst is not among the selected arms: no paired tables."])
        print(text)
        return text
    wide = table.pivot(index="seed", columns="arm")
    say("\n== paired by seed against const (t on %d df; wilcoxon signed-rank, two-sided; NW t =" % (
        len(seeds) - 1))
    say("   Newey-West lag 20 on the seed-mean per-period delta, i.e. over TIME not seeds) ==")
    say("%-9s %-8s %11s %11s %14s %8s %9s %7s" % (
        "arm", "metric", "mean", "median", "up/same/down", "seed t", "wilcoxon", "NW t"))
    series = {}
    for arm in arms:
        series[arm] = {s: payout_series(arm, s) for s in seeds}
    for arm in [a for a in arms if a != "const"]:
        for key, label in (("CORR", "corr"), ("MMC", "mmc"), ("payout", "payout"), ("dd_pay", None)):
            d = (wide[key][arm] - wide[key]["const"]).dropna()
            se = d.std(ddof=1) / math.sqrt(len(d)); moved = d[d != 0]
            nw = float("nan")
            if label:
                delta = pd.DataFrame({s: series[arm][s][label] - series["const"][s][label]
                                      for s in seeds})
                nw = newey_west(delta.mean(axis=1).dropna(), lag=20)[2]
            say("%-9s %-8s %+11.6f %+11.6f %5d/%3d/%4d %+8.2f %9.3f %+7.2f" % (
                arm, "dd(pay)" if key == "dd_pay" else key, d.mean(), d.median(),
                int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()),
                d.mean() / se if se > 0 else float("nan"),
                wilcoxon(moved).pvalue if len(moved) > 5 else float("nan"), nw))

    # A schedule also lowers the average rate, so beating const is not enough: the
    # claim is about its SHAPE only if it also beats the best plain lower rate.
    controls = [a for a in arms if ARMS[a][2] == "constant"]
    if controls:
        best = max(controls, key=lambda a: table.loc[table["arm"] == a, "payout"].mean())
        say("\n== schedulers paired by seed against %s, the better constant-rate control ==" % best)
        say("%-9s %-8s %11s %11s %14s %8s %9s" % (
            "arm", "metric", "mean", "median", "up/same/down", "seed t", "wilcoxon"))
        for arm in [a for a in arms if ARMS[a][2] == "scheduler"]:
            for key in ("CORR", "payout"):
                d = (wide[key][arm] - wide[key][best]).dropna()
                se = d.std(ddof=1) / math.sqrt(len(d)); moved = d[d != 0]
                say("%-9s %-8s %+11.6f %+11.6f %5d/%3d/%4d %+8.2f %9.3f" % (
                    arm, key, d.mean(), d.median(), int((d > 0).sum()), int((d == 0).sum()),
                    int((d < 0).sum()), d.mean() / se if se > 0 else float("nan"),
                    wilcoxon(moved).pvalue if len(moved) > 5 else float("nan")))

    state = lambda t: np.where(t["const"], "constant", np.where(t["CORR"] < 0, "negative", "positive"))
    base = table[table["arm"] == "const"].set_index("seed")
    say("\n== where each seed lands, against where const put it (rows of 3 = const's state) ==")
    say("%-9s %6s | %27s | %27s | %27s" % (
        "arm", "same", "const was POSITIVE -> p/c/n", "const was CONSTANT -> p/c/n",
        "const was NEGATIVE -> p/c/n"))
    for arm in [a for a in arms if a != "const"]:
        other = table[table["arm"] == arm].set_index("seed")
        cross = pd.crosstab(pd.Series(state(base), index=base.index),
                            pd.Series(state(other), index=other.index))
        cell = lambda r, c: int(cross.loc[r, c]) if r in cross.index and c in cross.columns else 0
        trio = lambda r: "%8d /%4d /%4d      " % tuple(cell(r, c) for c in ("positive", "constant", "negative"))
        say("%-9s %6d | %27s | %27s | %27s" % (
            arm, sum(cell(s, s) for s in ("positive", "constant", "negative")),
            trio("positive"), trio("constant"), trio("negative")))
    say("\n%d arms are compared with const, so one |t| near 2 is the expected false positive." % (
        len(arms) - 1))
    text = "\n".join(lines)
    table.to_csv("logs/%s_seed_metrics.csv" % PREFIX, index=False)
    with open("logs/%s_report.txt" % PREFIX, "w") as handle:
        handle.write(text + "\n")
    print(text)
    return text


GROUP_COLOR = {"baseline": "#2a78d6", "constant": "#eb6834", "scheduler": "#1baf7a"}
GROUP_LABEL = {"baseline": "constant 1e-3 (baseline)", "constant": "constant, lower rate",
               "scheduler": "scheduler, peak 1e-3"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = e104.INK, e104.INK2, e104.MUTED, e104.GRID, e104.AXIS, e104.SURFACE
PANELS = [("CORR", "CORR", "mean per-period CORR", "%+.3f"),
          ("MMC", "MMC", "mean per-period MMC", "%+.3f"),
          ("dd_pay", "dd(pay)", "max drawdown of cumulative payout · log scale, right is worse", "%.2f")]


def boxplot():
    """One horizontal box per arm and metric, every seed drawn, read against the baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = seed_table()
    seeds = sorted(table["seed"].unique())
    arms = [a for a in SELECTED if a in set(table["arm"])]
    jitter = dict(zip(seeds, np.random.RandomState(0).uniform(-0.27, 0.27, len(seeds))))
    # Groups are set apart by a gap: baseline, the lower constant rates, the schedulers.
    slot, at, previous = {}, 0.0, None
    for arm in arms:
        at += 1.0 if ARMS[arm][2] == previous or previous is None else 1.55
        slot[arm] = at; previous = ARMS[arm][2]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS,
                         "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": INK2,
                         "text.color": INK, "axes.labelcolor": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 0.62 * len(arms) + 3.7), dpi=160,
                             facecolor=SURFACE, sharey=True, gridspec_kw={"wspace": 0.17})
    for ax, (key, label, what, fmt) in zip(axes, PANELS):
        log = key == "dd_pay"
        # No panel fill: the figure already is the surface, and an opaque patch
        # would paint over the median column of the panel to its left.
        ax.set_facecolor("none"); ax.grid(axis="x", color=GRID, linewidth=0.7); ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", length=0)
        if log:
            ax.set_xscale("log")
        reference = table[(table["arm"] == "const") & (~table["const"] | (not log))][key].median()
        ax.axvline(reference, color=GROUP_COLOR["baseline"], linewidth=1.0, alpha=0.55, zorder=2)
        for arm in arms:
            part = table[table["arm"] == arm]; live = part[~part["const"]]; dead = part[part["const"]]
            color, y = GROUP_COLOR[ARMS[arm][2]], slot[arm]
            inbox = live if log else part
            if len(inbox):
                parts = ax.boxplot([inbox[key].to_numpy()], positions=[y], widths=0.62, vert=False,
                                   whis=(5, 95), showfliers=False, patch_artist=True,
                                   manage_ticks=False, zorder=3)
                parts["boxes"][0].set(facecolor=color, alpha=0.16, edgecolor="none")
                parts["medians"][0].set(color=color, linewidth=2.2)
                for item in parts["whiskers"] + parts["caps"]:
                    item.set(color=color, linewidth=1.0)
            ax.scatter(live[key], y + live["seed"].map(jitter), s=9, color=color, alpha=0.5,
                       edgecolor=SURFACE, linewidth=0.3, zorder=4)
            if len(dead):
                ax.scatter(np.full(len(dead), 0.021 if log else 0.0), y + dead["seed"].map(jitter),
                           s=9, facecolor=SURFACE, edgecolor=MUTED, linewidth=0.7, zorder=4)
            median = float(np.median(inbox[key])) if len(inbox) else float("nan")
            ax.text(1.005, y, "n/a" if median != median else fmt % median, fontsize=8.5, color=INK,
                    transform=ax.get_yaxis_transform(), va="center", ha="left")
        if log:
            ax.set_xlim(0.015, 60)
            ax.axvline(0.03, color=AXIS, linewidth=0.8)
            ax.set_xticks([0.021, 0.1, 1, 10])
            ax.set_xticklabels(["const.", "0.1", "1", "10"])
            ax.minorticks_off()
        ax.text(1.005, min(slot.values()) - 0.78, "median", fontsize=8, color=MUTED,
                transform=ax.get_yaxis_transform(), va="center", ha="left")
        ax.set_ylim(max(slot.values()) + 0.75, min(slot.values()) - 0.62)
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
        ax.text(0, 1.018, what, transform=ax.transAxes, fontsize=8.5, color=INK2)
    axes[0].set_yticks([slot[a] for a in arms])
    axes[0].set_yticklabels(["%s\n%s" % (a, ARMS[a][3]) for a in arms], fontsize=8.5)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=7, color=GROUP_COLOR[g],
                          label=GROUP_LABEL[g]) for g in ("baseline", "constant", "scheduler")
               if any(ARMS[a][2] == g for a in arms)]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=SURFACE,
                              markeredgecolor=MUTED, label="constant predictor"))
    handles.append(plt.Line2D([], [], color=GROUP_COLOR["baseline"], linewidth=1.0, alpha=0.55,
                              label="baseline median"))
    height = fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.022, 1 - 0.78 / height),
               ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.4,
               columnspacing=1.6)
    fig.suptitle("Learning-rate schedulers on the day3 batch scheme, %d seeds" % len(seeds),
                 x=0.03, y=1 - 0.22 / height, ha="left", va="top", fontsize=14.5,
                 fontweight="bold", color=INK)
    dead = " · ".join("%s %d" % (a, int(table.loc[table["arm"] == a, "const"].sum())) for a in arms)
    fig.text(0.03, 1 - 0.66 / height, "torch MLP, sklearn init, identity (64,32,16), alpha 100, batch = 3 "
             "shuffled days · fenccausal 44f · rolling 130/130 · 510 periods per run · seeds "
             "%d–%d, identical in every arm" % (seeds[0], seeds[-1]), fontsize=9.5, color=INK2)
    fig.text(0.03, 0.012, "Box = quartiles, line = median, whiskers = 5th–95th percentile; one dot "
             "per seed. Constant predictors (hollow) score CORR = MMC = 0 and are in those boxes; "
             "they have no drawdown and are kept out of the dd(pay) box.\nConstant on every "
             "period, of %d seeds: %s." % (len(seeds), dead), fontsize=8.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.115, right=0.955, top=1 - 1.85 / height, bottom=0.95 / height)
    os.makedirs("results/plots/crypto", exist_ok=True)
    out = "results/plots/crypto/%s_seed_boxplots.png" % PREFIX
    fig.savefig(out, facecolor=SURFACE)
    print("wrote %s" % out)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "gate":
        return gate()
    if mode == "report":
        return report()
    if mode == "boxplot":
        return boxplot()
    tasks = [(a, s) for s in SEEDS for a in SELECTED]       # seed-major: every arm fills evenly
    print("%s: %d arms x %d seeds = %d runs, %d at a time" % (
        EXPERIMENT, len(SELECTED), len(SEEDS), len(tasks), JOBS))
    if DRY:
        for arm in SELECTED:
            print("  %-9s lr %-8g %-9s %s" % (arm, ARMS[arm][0], ARMS[arm][1] or "-", ARMS[arm][3]))
        return
    started, failed = time.time(), []
    with cf.ProcessPoolExecutor(max_workers=JOBS, mp_context=mp.get_context("spawn")) as pool:
        for i, (name, status, info) in enumerate(pool.map(run_one, tasks), 1):
            if status.startswith("FAILED"):
                failed.append(name)
            print("[%4d/%d] %-20s %s%s" % (i, len(tasks), name, status,
                  "" if info is None else "  corr %+.5f  scorable %d  %.1f min" % info), flush=True)
    text = report()
    try:
        boxplot()
    except Exception as error:                      # a figure must not cost the tables
        print("note: boxplot failed (%r)" % (error,))
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
