#!/usr/bin/env python3
"""
E116 -- how often to refit the fenccausal ridge: every period, 1, 3 or 6 months,
for three of E115's lambda rules, at the data lag a live round actually has.

    python3 experiments/e116_ridge_lambda_cadence_crypto.py              # 4 cadences in parallel, then the report
    python3 experiments/e116_ridge_lambda_cadence_crypto.py --cadence 22 # one cadence's arms, no report
    python3 experiments/e116_ridge_lambda_cadence_crypto.py --report     # the report, from what is on disk
    SMOKE=1  ...                                                         # --era-step 4 scout, e116scout_*
    DRY_RUN=1 ...                                                        # load, print each cadence's blocks, exit
    NO_MAIL=1 ...                                                        # skip the email

WHAT IT ASKS
  E115 refit monthly and trailed the 6-monthly house ridge (e109_ridge_fenc) by
  ~0.02 payout -- but the two differed in cadence, lambda grid AND selection rule
  at once, so the gap could not be attributed. This study holds the grid and
  the rule fixed and moves ONLY the refit cadence, for three rules:

    tscv_corr     lambda by in-window purged CV on -CORR (the pipeline's rule)
    preq_payout   lambda by prequential -(0.1 corr + mmc), real meta model
    tscv_mse      lambda by in-window purged CV on MSE

  at four cadences, in periods (the panel is weekday-only, one period = one
  live round):

    r1    every period -- the fastest a live round can refit: train.parquet
          gains exactly one resolved date per round
    r22   one month   (utils.default_refit_every(months=1))
    r65   three months
    r130  six months -- registry.FIT's live cadence

  The repo's early cadence sweep (quoted in src/live/refit.py) found an
  inverted U on raw ridge: monthly 0.127 < quarterly 0.145 < SIX-MONTHLY 0.153 >
  yearly 0.144. That is the prior; this asks whether it survives a lambda that
  is re-chosen at every refit.

WHAT IT RUNS
  crypto/v2.0-fenccausal (44f), target_binned_return_20, rolling 130-period
  window, 5 inner folds, the E115 grid (51 alphas, 1e-3..1e7), val_frac 0.3 ->
  the same 510 scored dates as E115 and e109_ridge_fenc.

  EMBARGO 21, not 20. Live, train.parquet ends 22 periods before the round's
  date (30 calendar days, 32 over a weekend; 45 of 52 recorded rounds, the rest
  were stale-data rounds). The walk-forward trains through position
  start - embargo - 1, so embargo 20 leaves a 21-period gap -- one period more
  than live. At a monthly cadence that only touched each block's first day; at
  r1 it would touch every day. 21 reproduces the live gap. The loader drops 21
  periods before the scored block instead of 20, which leaves the scored dates
  unchanged.

  One lambda_walk_forward per cadence, narrowed to the three arms (only what
  they read is computed). The cadences run as four child processes in parallel,
  BLAS capped at 4 threads each, then one report reads all twelve arms.

WHAT WOULD COUNT AS A RESULT
  Per arm, each cadence paired against r130 (the live cadence), Newey-West lag
  20 over the 510 common dates, on CORR, MMC and payout; |t| >= 2 is the bar.
  Plus an embargo check: r22 here against E115's monthly arm (same everything,
  embargo 20) -- what the one-period optimism was worth.

ARTEFACTS
  results/{forecast,per_period,hyperparams,metrics}/crypto/ridge/e116_{tscv_corr,preq_payout,tscv_mse}_r{1,22,65,130}
  results/reports/crypto/e116/{lambda_path,fixed_lambda_curve}_r<N>.csv, fit_r<N>.json, table.csv
  logs/e116_r<N>.log, logs/e116_report.txt (+ email, success and failure)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from e115_ridge_prequential_crypto import (  # noqa: E402
    lambda_path,
    load,
    mail,
    parse_alphas,
    path_stats,
    write_arm,
)
from src.analysis.ablation_report import newey_west  # noqa: E402
from src.analysis.results_table import payout_series, render, row_from_frame  # noqa: E402
from src.data_mani.utils import get_spec  # noqa: E402
from src.prediction.functions import sorted_unique  # noqa: E402
from src.prediction.lambda_selection import (  # noqa: E402
    design_spectrum,
    lambda_walk_forward,
    refit_blocks,
)

TOURNAMENT, MODEL_TREE = "crypto", "ridge"
ARMS = [("tscv", "corr"), ("prequential", "payout"), ("tscv", "mse")]
SHORT = {"prequential": "preq", "tscv": "tscv"}
CADENCES = [(1, "every period"), (22, "1 month"), (65, "3 months"), (130, "6 months")]
BASELINE_CADENCE = 130
TRAIN_WINDOW, N_SPLITS = 130, 5
CONTEXT = "e109_ridge_fenc"          # 6-monthly house ridge, grid 1e-4..1e2
SMOKE = os.environ.get("SMOKE") == "1"
THREADS = {k: "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cadence", type=int, action="append",
                    help="fit only this cadence (periods); repeatable; no report")
    ap.add_argument("--report", action="store_true", help="report from disk only")
    ap.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    ap.add_argument("--expect-features", type=int, default=44)
    ap.add_argument("--alphas", default="1e-3:1e7:51")
    ap.add_argument("--embargo", type=int, default=21,
                    help="periods; 21 reproduces the live 22-period train/live gap")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--era-step", type=int, default=4 if SMOKE else 1)
    ap.add_argument("--prefix", default="e116scout_" if SMOKE else "e116_")
    return ap.parse_args()


def arm_key(rule, loss):
    return "{}_{}".format(SHORT[rule], loss)


def run_name(prefix, rule, loss, cadence):
    return "{}{}_r{}".format(prefix, arm_key(rule, loss), cadence)


def report_dir(args):
    return os.path.join("results", "reports", TOURNAMENT,
                        "e116_scout" if args.era_step != 1 else "e116")


def load_data(args):
    spec = get_spec(TOURNAMENT)
    loaded = load(SimpleNamespace(data_version=args.data_version, era_step=args.era_step,
                                  val_frac=args.val_frac,
                                  expect_features=args.expect_features),
                  spec["target"], args.embargo)
    return spec, loaded


def fit_cadence(args, cadence):
    """Fit one cadence's three arms and write them, its lambda path and its curve."""
    started = time.time()
    spec, (df, features, reference, meta) = load_data(args)
    period_col, id_col, target_col = spec["period_col"], spec["id_col"], spec["target"]
    alphas = parse_alphas(args.alphas)
    print("E116 r{} | {} | rolling {} / refit {} / embargo {} | {} alphas | era_step {}"
          .format(cadence, args.data_version, TRAIN_WINDOW, cadence, args.embargo,
                  len(alphas), args.era_step), flush=True)

    run = lambda_walk_forward(
        df, features, target_col, period_col, id_col, alphas,
        refit_every=cadence, embargo=args.embargo, scheme="rolling",
        train_window=TRAIN_WINDOW, n_splits=N_SPLITS, arms=ARMS,
        extra_cols=(reference[0], meta[0]), meta_col=meta[0], verbose=False)
    minutes = (time.time() - started) / 60.0
    warm = [b for b in run["blocks"] if not b["scored"]]
    scored = [b for b in run["blocks"] if b["scored"]]
    print("r{}: walk-forward done in {:.1f} min, {} warm-up + {} scored blocks".format(
        cadence, minutes, len(warm), len(scored)), flush=True)

    train_periods = sorted_unique(df.loc[df["split"] == "train", period_col].to_numpy())
    low, high = design_spectrum(
        df.loc[df[period_col].isin(train_periods[-TRAIN_WINDOW:]), features])

    for rule, loss in ARMS:
        predictions, records = run["arms"][(rule, loss)]
        config = {
            "tournament": TOURNAMENT, "model": MODEL_TREE, "task": "regression",
            "target": target_col, "metric": loss, "data_version": args.data_version,
            "n_features": len(features), "era_step": args.era_step,
            "val_frac": args.val_frac, "feature_include": None, "feature_exclude": None,
            "refit_every": cadence, "embargo": args.embargo, "cv_scheme": "rolling",
            "train_window": TRAIN_WINDOW, "n_iter": len(alphas), "n_splits": N_SPLITS,
            "inner_cv": "purged" if rule == "tscv" else None, "seed": None,
            "search_type": "grid", "fixed_params": None, "bmc_reference": "benchmark",
            "min_train": 0, "lambda_rule": rule, "lambda_loss": loss,
            "alpha_grid": [float(a) for a in alphas], "warmup_blocks": len(warm),
            "first_oos_period": warm[0]["start_period"] if warm else None,
            "design_eigen_span": [low, high],
            # The train/live gap this embargo reproduces: training ends at
            # start - embargo - 1, live train.parquet ends 22 periods back.
            "live_gap_periods": args.embargo + 1,
            "study": "E116",
        }
        write_arm(run_name(args.prefix, rule, loss, cadence), predictions, records,
                  run["skipped"], config, minutes, target_col, period_col, reference, meta)

    out = report_dir(args)
    os.makedirs(out, exist_ok=True)
    lambda_path(run, alphas).to_csv(os.path.join(out, "lambda_path_r%d.csv" % cadence),
                                    index=False)
    scored_periods = sorted_unique(
        df.loc[df["split"] == "validation", period_col].to_numpy())
    curve = pd.DataFrame({"alpha": alphas})
    for loss in ("corr", "payout"):
        losses = run["losses"][loss]
        curve[loss] = (-losses.loc[losses.index.isin(scored_periods)]).mean(axis=0).to_numpy()
    curve.to_csv(os.path.join(out, "fixed_lambda_curve_r%d.csv" % cadence), index=False)
    with open(os.path.join(out, "fit_r%d.json" % cadence), "w") as handle:
        json.dump({"cadence": cadence, "embargo": args.embargo, "minutes": minutes,
                   "warmup_blocks": len(warm), "scored_blocks": len(scored),
                   "design_eigen_span": [low, high]}, handle, indent=2)


def dry_run(args):
    spec, (df, _, _, _) = load_data(args)
    period_col = spec["period_col"]
    all_periods = sorted_unique(df[period_col].to_numpy())
    val = sorted_unique(df.loc[df["split"] == "validation", period_col].to_numpy())
    for cadence, label in CADENCES:
        blocks, skipped = refit_blocks(all_periods, val, cadence, args.embargo,
                                       "rolling", TRAIN_WINDOW)
        warm = [b for b in blocks if not b["scored"]]
        scored = [b for b in blocks if b["scored"]]
        print("r{:<4} {:<13} {:>4} warm-up + {:>3} scored blocks, first scored {} "
              "trains to {} | {} skipped".format(
                  cadence, label, len(warm), len(scored),
                  all_periods[scored[0]["start"]], all_periods[scored[0]["train_end"] - 1],
                  len(skipped)))
        for rule, loss in ARMS:
            print("      would write ridge/" + run_name(args.prefix, rule, loss, cadence))


def launch(args):
    """Fit the four cadences as parallel children; return {cadence: exit code}."""
    passthrough = ["--data-version", args.data_version, "--alphas", args.alphas,
                   "--embargo", str(args.embargo), "--val-frac", str(args.val_frac),
                   "--era-step", str(args.era_step), "--prefix", args.prefix,
                   "--expect-features", str(args.expect_features)]
    env = dict(os.environ, **THREADS)
    children = {}
    for cadence, _ in CADENCES:
        log = os.path.join("logs", "{}r{}.log".format(args.prefix, cadence))
        handle = open(log, "w")
        children[cadence] = (subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__), "--cadence", str(cadence)]
            + passthrough, stdout=handle, stderr=subprocess.STDOUT, env=env), handle, log)
        print("launched r{} -> {}".format(cadence, log), flush=True)
    codes = {}
    for cadence, (process, handle, log) in children.items():
        codes[cadence] = process.wait()
        handle.close()
        print("r{} exited {} ({})".format(cadence, codes[cadence], log), flush=True)
    return codes


def paired(a, b):
    """(mean, t, n) of a - b over their common dates, Newey-West lag 20."""
    common = a.dropna().index.intersection(b.dropna().index)
    mean, _, t = newey_west((a.loc[common] - b.loc[common]).to_numpy())
    return mean, t, len(common)


def report(args, codes=None):
    out = report_dir(args)
    lines = []
    say = lines.append
    say("E116 -- refit cadence x lambda rule, ridge on fenccausal (embargo {}, live gap "
        "{} periods)".format(args.embargo, args.embargo + 1))
    if codes:
        failed = {c: code for c, code in codes.items() if code != 0}
        if failed:
            say("FAILED cadences (exit codes): {} -- see logs/{}r<N>.log".format(
                failed, args.prefix))

    tables, missing = {}, []
    for rule, loss in ARMS:
        for cadence, _ in CADENCES:
            path = os.path.join("results", "per_period", TOURNAMENT, MODEL_TREE,
                                run_name(args.prefix, rule, loss, cadence) + ".csv")
            if os.path.exists(path):
                tables[(rule, loss, cadence)] = pd.read_csv(path, index_col=0,
                                                            parse_dates=True)
            else:
                missing.append(run_name(args.prefix, rule, loss, cadence))
    if missing:
        say("MISSING arms, left out of everything below: " + ", ".join(missing))
    if not tables:
        return "\n".join(lines + ["nothing to report"])

    common = None
    for table in tables.values():
        common = table.index if common is None else common.intersection(table.index)

    rows = []
    for (rule, loss, cadence), table in tables.items():
        rows.append(row_from_frame(run_name(args.prefix, rule, loss, cadence),
                                   table.loc[common], target="target_binned_return_20"))
    context = {}
    if args.era_step == 1:
        for label, run in [("e115_" + arm_key(r, l), "e115_" + arm_key(r, l)) for r, l in ARMS] \
                + [(CONTEXT, CONTEXT)]:
            path = os.path.join("results", "per_period", TOURNAMENT, MODEL_TREE, run + ".csv")
            if not os.path.exists(path):
                say("context {} not on disk -- left out".format(run))
                continue
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            if len(common.intersection(frame.index)) != len(common):
                say("context {} does not cover the common window -- left out".format(run))
                continue
            context[label] = frame
            tag = " (ctx, r22 emb 20)" if run.startswith("e115_") else " (ctx, r130 grid<=100)"
            rows.append(row_from_frame(run + tag, frame.loc[common],
                                       target="target_binned_return_20"))
    say("")
    say("Level and risk, common window n={} (calmar and burn comparable at equal n):"
        .format(len(common)))
    say(render(rows, sort_by=None))
    pd.DataFrame(rows).to_csv(os.path.join(out, "table.csv"), index=False)
    say("")

    say("Payout (calmar) by cadence:")
    say("{:<14}".format("arm") + "".join("{:>18}".format("r%d %s" % (c, lab))
                                         for c, lab in CADENCES))
    for rule, loss in ARMS:
        cells = []
        for cadence, _ in CADENCES:
            row = next((r for r in rows if r["run"] == run_name(args.prefix, rule, loss,
                                                                cadence)), None)
            cells.append("{:>18}".format("-" if row is None else
                                         "%.5f (%.3f)" % (row["payout"], row["calmar"])))
        say("{:<14}".format(arm_key(rule, loss)) + "".join(cells))
    say("")

    say("Paired, Newey-West lag 20 (|t| >= 2 is the bar):")
    say("{:<40} {:>10} {:>6} {:>10} {:>6} {:>10} {:>6} {:>5}".format(
        "comparison", "dCORR", "t", "dMMC", "t", "dpayout", "t", "n"))

    def line(label, a, b):
        cells = [label]
        for column in ("corr", "mmc", "payout"):
            x = payout_series(a) if column == "payout" else a[column]
            y = payout_series(b) if column == "payout" else b[column]
            mean, t, n = paired(x, y)
            cells += [mean, t]
        if all(c == 0 for c in cells[1::2]):
            # Two arms that picked the same lambda in every block are one series.
            say("{:<40} identical predictions on all {} dates".format(label, n))
            return
        say("{:<40} {:>+10.5f} {:>+6.2f} {:>+10.5f} {:>+6.2f} {:>+10.5f} {:>+6.2f} {:>5}"
            .format(*cells, n))

    say("-- each cadence against r{} (the live cadence), per arm".format(BASELINE_CADENCE))
    for rule, loss in ARMS:
        base = tables.get((rule, loss, BASELINE_CADENCE))
        for cadence, _ in CADENCES:
            if cadence == BASELINE_CADENCE or base is None \
                    or (rule, loss, cadence) not in tables:
                continue
            line("{} r{} - r{}".format(arm_key(rule, loss), cadence, BASELINE_CADENCE),
                 tables[(rule, loss, cadence)].loc[common], base.loc[common])
    say("-- the rules against tscv_corr, at each cadence")
    for cadence, _ in CADENCES:
        ref = tables.get(("tscv", "corr", cadence))
        for rule, loss in ARMS[1:]:
            if ref is not None and (rule, loss, cadence) in tables:
                line("r{} {} - tscv_corr".format(cadence, arm_key(rule, loss)),
                     tables[(rule, loss, cadence)].loc[common], ref.loc[common])
    if context:
        say("-- embargo check: r22 at embargo {} against E115's r22 at embargo 20"
            .format(args.embargo))
        for rule, loss in ARMS:
            label = "e115_" + arm_key(rule, loss)
            if label in context and (rule, loss, 22) in tables:
                line("{} r22 - e115 (emb 20)".format(arm_key(rule, loss)),
                     tables[(rule, loss, 22)].loc[common], context[label].loc[common])
        if CONTEXT in context:
            say("-- against the 6-monthly house ridge (grid stops at 100: context only)")
            for rule, loss in ARMS:
                if (rule, loss, BASELINE_CADENCE) in tables:
                    line("{} r{} - {}".format(arm_key(rule, loss), BASELINE_CADENCE, CONTEXT),
                         tables[(rule, loss, BASELINE_CADENCE)].loc[common],
                         context[CONTEXT].loc[common])
    say("")

    alphas = parse_alphas(args.alphas)
    say("Selected alpha per arm and cadence (evid = periods the first own-loss pick "
        "stood on; fb = blocks that took the corr pick for want of a meta model):")
    say("{:<22} {:>7} {:>11} {:>9} {:>9} {:>9} {:>6} {:>9} {:>4}".format(
        "arm", "blocks", "median", "IQR(dec)", "distinct", "switches", "edge",
        "evid(1st)", "fb"))
    for rule, loss in ARMS:
        for cadence, _ in CADENCES:
            path = os.path.join("results", "hyperparams", TOURNAMENT, MODEL_TREE,
                                run_name(args.prefix, rule, loss, cadence) + ".csv")
            if not os.path.exists(path):
                continue
            h = pd.read_csv(path)
            s = path_stats(h.rename(columns={"alpha": "a"}), "a", alphas)
            own = h[h["selection_loss"] == loss]
            say("{:<22} {:>7} {:>11.4g} {:>9.2f} {:>9} {:>9} {:>6} {:>9} {:>4}".format(
                "{}_r{}".format(arm_key(rule, loss), cadence), s["n"], s["median"],
                s["iqr_decades"], s["distinct"], s["switches"], s["edge"],
                int(own["selection_periods"].iloc[0]) if len(own) else 0,
                int(len(h) - len(own))))
    say("")

    say("Ex-post FIXED-alpha score per cadence (oracle, not an arm):")
    say("{:<14} {:>12} {:>10} {:>12} {:>10} {:>10} {:>10}".format(
        "cadence", "CORR alpha", "CORR", "payout alpha", "payout", "OLS pay", "limit pay"))
    for cadence, label in CADENCES:
        path = os.path.join(out, "fixed_lambda_curve_r%d.csv" % cadence)
        if not os.path.exists(path):
            continue
        f = pd.read_csv(path)
        bc, bp = f["corr"].idxmax(), f["payout"].idxmax()
        say("{:<14} {:>12.4g} {:>10.5f} {:>12.4g} {:>10.5f} {:>10.5f} {:>10.5f}".format(
            "r%d %s" % (cadence, label), f["alpha"][bc], f["corr"][bc], f["alpha"][bp],
            f["payout"][bp], f["payout"].iloc[0], f["payout"].iloc[-1]))
    say("")
    for cadence, _ in CADENCES:
        path = os.path.join(out, "fit_r%d.json" % cadence)
        if os.path.exists(path):
            info = json.load(open(path))
            say("r{}: {} warm-up + {} scored blocks, walk-forward {:.1f} min".format(
                cadence, info["warmup_blocks"], info["scored_blocks"], info["minutes"]))

    text = "\n".join(lines)
    print("\n" + text, flush=True)
    if args.era_step == 1:
        with open(os.path.join("logs", "e116_report.txt"), "w") as handle:
            handle.write(text + "\n")
    return text


def main():
    args = parse_args()
    if os.environ.get("DRY_RUN") == "1":
        dry_run(args)
        return None
    if args.cadence:
        for cadence in args.cadence:
            fit_cadence(args, cadence)
        return None
    if args.report:
        return report(args)
    started = time.time()
    codes = launch(args)
    text = report(args, codes)
    return text + "\nwall clock {:.1f} min".format((time.time() - started) / 60.0)


if __name__ == "__main__":
    child = "--cadence" in sys.argv
    try:
        out = main()
    except SystemExit:
        raise
    except Exception:
        body = traceback.format_exc()
        print(body, flush=True)
        if not (child or SMOKE or os.environ.get("DRY_RUN") == "1"):
            mail("[crypto] E116 FAILED -- ridge refit cadence x lambda rule", body)
        sys.exit(1)
    if isinstance(out, str) and not SMOKE:
        mail("[crypto] E116 finished -- ridge refit cadence x lambda rule", out)
