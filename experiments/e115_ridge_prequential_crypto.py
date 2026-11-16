#!/usr/bin/env python3
"""
E115 -- ridge on fenccausal, refit monthly: lambda chosen by PREQUENTIAL loss
against lambda chosen by TIME-SERIES CV.

    python3 experiments/e115_ridge_prequential_crypto.py              # full run
    SMOKE=1  python3 experiments/e115_ridge_prequential_crypto.py     # --era-step 4 scout, e115scout_*
    DRY_RUN=1 python3 experiments/e115_ridge_prequential_crypto.py    # load, print the block layout, exit
    NO_MAIL=1 ...                                                     # skip the email

WHAT IT ASKS
  Ridge's one hyperparameter is lambda. Should it be chosen on the forecasts the
  model has ACTUALLY made out of sample (prequential), or on folds carved out of
  the current training window (time-series CV, the pipeline's own rule)?

    prequential  at refit b, every lambda's mean per-period loss over all of its
                 own out-of-sample forecasts whose target has resolved by b
                 (period position < b's train_end), since the warm-up began.
                 Choose the argmin.
    tscv         at refit b, every lambda's mean held-out loss over 5 purged
                 chronological folds inside b's 130-period training window
                 (functions.purged_period_splits, embargo 20 -- the inner CV
                 hyper_params_search runs). Choose the argmin.

  Each rule is run with three losses, because "loss" decides what the rule rewards:
    corr    minus the per-period numerai_corr (what the pipeline's search
            maximises)
    mse     per-period mean squared error (ridge's own loss; near-blind to rank)
    payout  minus the per-period payout, 0.1 * corr + mmc against the REAL meta
            model, built as results_table.payout_series builds it -- the utility
            Numerai actually pays. The meta model starts 2024-05-07, only 17
            periods before the first scored block's training ends, so a payout
            choice stands on THIN evidence early: prequential has 17 resolved
            periods at block 1 (+~22 a month), and TSCV votes only with the
            inner folds that have a meta model until the window is covered
            (~block 6). Periods without one do not vote; the report prints how
            many periods each pick stood on ("evidence").
  so 2 rules x 3 losses = 6 arms, all from ONE pass.

WHAT IT RUNS
  crypto/v2.0-fenccausal (44f = raw 22 + 22 causal per-symbol means),
  target_binned_return_20, rolling 130-period window, refit every 22 periods
  (one month, utils.default_refit_every), embargo 20, val_frac 0.3 -> 510 scored
  periods, the same scored window as every fenccausal ridge on the board.

  One fixed alpha grid on sklearn's scale (default 51 points, 1e-3..1e7, five per
  decade). It has to bracket the centred X'X spectrum, which the run records: on
  fenccausal that is ~0.02..2e4 per 130-period window, so 1e-3 is least squares
  and 1e7 is the X'y limit, where CORR stops moving. A 1e0 floor was tried first
  and the scout picked it as an edge. At every refit the ridge is fitted for EVERY alpha on that refit's
  training window (one SVD, src.prediction.lambda_selection.ridge_path) and every
  alpha predicts the block. The two rules then pick a column of that matrix, so
  the six arms share every fit and differ ONLY in which lambda each block used.
  The stride is carried back through the training history (warm-up blocks:
  predicted, never scored) so the prequential rule has resolved out-of-sample
  forecasts to read from the first scored block on.

WHAT WOULD COUNT AS A RESULT
  Paired preq - tscv per loss, Newey-West lag 20 over the 510 common periods, on
  CORR, MMC and payout; |t| >= 2 is the house bar. With ~26 effective windows a
  null is the expected outcome -- and if the fixed-lambda CORR curve below is
  flat, NO rule can matter and the null is structural, not a weak test.

ARTEFACTS
  results/{forecast,per_period,hyperparams,metrics}/crypto/ridge/e115_{preq,tscv}_{corr,mse,payout}
  results/reports/crypto/e115/lambda_path.csv       every scored block's pick, per arm
  results/reports/crypto/e115/fixed_lambda_curve.csv ex-post CORR (and payout) of each FIXED alpha
  logs/e115_report.txt (+ email, success and failure)
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

from src.analysis.ablation_report import newey_west  # noqa: E402
from src.analysis.results_table import payout_series, render, row_from_frame  # noqa: E402
from src.data_mani.utils import (  # noqa: E402
    attach_meta_model,
    attach_reference,
    default_embargo,
    default_refit_every,
    get_spec,
    load_meta_model,
    load_reference,
    load_tournament_data,
)
from src.prediction.functions import evaluate, sorted_unique  # noqa: E402
from src.prediction.lambda_selection import (  # noqa: E402
    LOSSES,
    RULES,
    design_spectrum,
    lambda_walk_forward,
    refit_blocks,
)
from src.prediction.metrics import per_period_table, summarize  # noqa: E402
from src.run_forecast import write_outputs  # noqa: E402

TOURNAMENT = "crypto"
MODEL_TREE = "ridge"
SHORT = {"prequential": "preq", "tscv": "tscv"}
REPORT_DIR = os.path.join("results", "reports", TOURNAMENT, "e115")
REPORT_TXT = os.path.join("logs", "e115_report.txt")
# The house fenccausal ridge (E109's no-block arm): 130/130 refit, the fixed
# 1e-4..1e2 grid. A different cadence AND grid, so it is context, not a control.
CONTEXT_RUN = "e109_ridge_fenc"


def parse_alphas(text):
    """'lo:hi:n' -> n log-spaced alphas from lo to hi, or a comma list."""
    if ":" in text:
        lo, hi, n = text.split(":")
        return np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(n))
    return np.array(sorted(float(v) for v in text.split(",")))


def parse_args():
    smoke = os.environ.get("SMOKE") == "1"
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    ap.add_argument("--expect-features", type=int, default=44,
                    help="refuse to run unless the dataset has exactly this many "
                         "features (0 disables the gate)")
    ap.add_argument("--alphas", default="1e-3:1e7:51",
                    help="'lo:hi:n' log-spaced, or a comma list; sklearn's scale")
    ap.add_argument("--train-window", type=int, default=130)
    ap.add_argument("--refit-every", type=int,
                    default=default_refit_every(TOURNAMENT, months=1))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--era-step", type=int, default=4 if smoke else 1)
    ap.add_argument("--losses", default=",".join(LOSSES))
    ap.add_argument("--prefix", default="e115scout_" if smoke else "e115_")
    return ap.parse_args()


def arm_name(prefix, rule, loss):
    return "{}{}_{}".format(prefix, SHORT[rule], loss)


def paired(a, b):
    """(mean, t) of a - b on the common periods, Newey-West lag 20."""
    common = a.index.intersection(b.index)
    mean, _, t = newey_west((a.loc[common] - b.loc[common]).to_numpy())
    return mean, t, len(common)


def load(args, target_col, embargo):
    df, features = load_tournament_data(
        tournament=TOURNAMENT, version=args.data_version, era_step=args.era_step,
        val_frac=args.val_frac, embargo=embargo, verbose=True, target=target_col)
    if args.expect_features and len(features) != args.expect_features:
        raise SystemExit("feature gate: {} has {} features, expected {}. Refusing "
                         "to attach these numbers to E115.".format(
                             args.data_version, len(features), args.expect_features))
    reference, reference_col = load_reference(TOURNAMENT, args.data_version, True)
    df, coverage = attach_reference(df, reference, TOURNAMENT, reference_col)
    meta_frame, meta_model_col = load_meta_model(TOURNAMENT, args.data_version, True)
    df, meta_coverage = attach_meta_model(df, meta_frame, TOURNAMENT)
    print("reference '{}' covers {:.1%}, meta model '{}' covers {:.1%} of rows".format(
        reference_col, coverage, meta_model_col, meta_coverage))
    return df, features, (reference_col, coverage), (meta_model_col, meta_coverage)


def write_arm(name, predictions, records, skipped, config, minutes, target_col,
              period_col, reference, meta):
    """Score one arm and write the four standard files; return its per-period table."""
    (reference_col, coverage), (meta_model_col, meta_coverage) = reference, meta
    results = evaluate(predictions, target_col=target_col, period_col=period_col,
                       reference_col=reference_col, meta_model_col=meta_model_col)
    periods_table = per_period_table(
        predictions, pred_col="prediction", target_col=target_col,
        period_col=period_col, reference_col=reference_col,
        meta_model_col=meta_model_col)
    results["pnl"] = summarize(periods_table["pnl"])
    results["reference"] = {"column": reference_col, "row_coverage": coverage,
                            "kind": "benchmark", "proxy_fit_periods": None}
    results["meta_model"] = {"column": meta_model_col, "row_coverage": meta_coverage}
    results["refits"] = records
    results["skipped_blocks"] = skipped
    results["config"] = config
    results["runtime_minutes"] = round(minutes, 3)
    paths = write_outputs(predictions, periods_table, results, TOURNAMENT,
                          MODEL_TREE, name)
    print("{:<26} CORR {:.5f}  MMC {:.5f}  -> {}".format(
        name, results["corr"]["mean"],
        (results.get("mmc") or {}).get("mean", float("nan")), paths["per_period"]))
    return periods_table


def lambda_path(run, alphas):
    """One row per scored block: each arm's alpha, side by side."""
    frames = []
    for (rule, loss), (_, records) in run["arms"].items():
        frames.append(pd.DataFrame({
            "refit": [r["refit"] for r in records],
            "predict_start": [r["predict_start"] for r in records],
            "{}_{}".format(SHORT[rule], loss): [r["best_params"]["alpha"] for r in records],
        }).set_index(["refit", "predict_start"]))
    return pd.concat(frames, axis=1).reset_index()


def path_stats(path, column, alphas):
    """How a selected alpha moved: median, spread, switches, edge hits."""
    a = path[column].to_numpy()
    log = np.log10(a)
    return {
        "median": float(np.median(a)),
        "iqr_decades": float(np.subtract(*np.percentile(log, [75, 25]))),
        "distinct": int(len(np.unique(a))),
        "switches": int((a[1:] != a[:-1]).sum()),
        "edge": int(((a == alphas[0]) | (a == alphas[-1])).sum()),
        "n": int(len(a)),
    }


def main():
    args = parse_args()
    started = time.time()
    spec = get_spec(TOURNAMENT)
    period_col, id_col, target_col = spec["period_col"], spec["id_col"], spec["target"]
    embargo = default_embargo(TOURNAMENT, target_col)
    alphas = parse_alphas(args.alphas)
    losses = [s for s in args.losses.split(",") if s]

    print("E115 | {} | rolling {} / refit {} / embargo {} | {} alphas {:.3g}..{:.3g} | "
          "losses {} | era_step {}".format(
              args.data_version, args.train_window, args.refit_every, embargo,
              len(alphas), alphas[0], alphas[-1], losses, args.era_step))

    df, features, (reference_col, coverage), (meta_model_col, meta_coverage) = load(
        args, target_col, embargo)

    if os.environ.get("DRY_RUN") == "1":
        all_periods = sorted_unique(df[period_col].to_numpy())
        val = sorted_unique(df.loc[df["split"] == "validation", period_col].to_numpy())
        blocks, skipped = refit_blocks(all_periods, val, args.refit_every, embargo,
                                       "rolling", args.train_window)
        warm = [b for b in blocks if not b["scored"]]
        scored = [b for b in blocks if b["scored"]]
        print("{} periods, {} validation | {} warm-up blocks from {} | {} scored "
              "blocks from {} | {} skipped".format(
                  len(all_periods), len(val), len(warm),
                  all_periods[warm[0]["start"]] if warm else "-", len(scored),
                  all_periods[scored[0]["start"]], len(skipped)))
        for rule in RULES:
            for loss in losses:
                print("  would write ridge/" + arm_name(args.prefix, rule, loss))
        return 0

    # The span a useful grid must bracket, on the last full window before the
    # scored block: alpha << low is OLS, alpha >> high only rescales X'y.
    train_periods = sorted_unique(df.loc[df["split"] == "train", period_col].to_numpy())
    window = df[period_col].isin(train_periods[-args.train_window:])
    low, high = design_spectrum(df.loc[window, features])
    print("design eigenvalue span, last training window: {:.3g} .. {:.3g}".format(
        low, high))

    run = lambda_walk_forward(
        df, features, target_col, period_col, id_col, alphas,
        refit_every=args.refit_every, embargo=embargo, scheme="rolling",
        train_window=args.train_window, n_splits=args.n_splits, losses=losses,
        extra_cols=(reference_col, meta_model_col), meta_col=meta_model_col)
    fit_minutes = (time.time() - started) / 60.0
    warm = [b for b in run["blocks"] if not b["scored"]]
    print("walk-forward done in {:.1f} min: {} warm-up + {} scored blocks".format(
        fit_minutes, len(warm), len(run["blocks"]) - len(warm)))

    tables = {}
    for (rule, loss), (predictions, records) in run["arms"].items():
        name = arm_name(args.prefix, rule, loss)
        config = {
            "tournament": TOURNAMENT, "model": MODEL_TREE, "task": "regression",
            "target": target_col, "metric": loss,
            "data_version": args.data_version, "n_features": len(features),
            "era_step": args.era_step, "val_frac": args.val_frac,
            "feature_include": None, "feature_exclude": None,
            "refit_every": args.refit_every, "embargo": embargo,
            "cv_scheme": "rolling", "train_window": args.train_window,
            "n_iter": len(alphas), "n_splits": args.n_splits,
            "inner_cv": "purged" if rule == "tscv" else None,
            "seed": None, "search_type": "grid", "fixed_params": None,
            "bmc_reference": "benchmark", "min_train": 0,
            # What makes these runs different from every other ridge here.
            "lambda_rule": rule, "lambda_loss": loss,
            "alpha_grid": [float(a) for a in alphas],
            "warmup_blocks": len(warm),
            "first_oos_period": warm[0]["start_period"] if warm else None,
            "design_eigen_span": [low, high],
            "study": "E115",
        }
        tables[(rule, loss)] = write_arm(
            name, predictions, records, run["skipped"], config, fit_minutes,
            target_col, period_col, (reference_col, coverage),
            (meta_model_col, meta_coverage))

    # ---------------------------------------------------------------- report
    os.makedirs(REPORT_DIR, exist_ok=True)
    lines = []
    say = lines.append
    say("E115 -- ridge lambda: prequential loss vs time-series CV")
    say("{} ({} features), rolling {} / refit {} / embargo {}, {} alphas {:.3g}..{:.3g}, "
        "era_step {}".format(args.data_version, len(features), args.train_window,
                             args.refit_every, embargo, len(alphas), alphas[0],
                             alphas[-1], args.era_step))
    say("prequential history starts {} ({} warm-up blocks); design eigenvalues "
        "{:.3g}..{:.3g}".format(warm[0]["start_period"] if warm else "-", len(warm),
                                low, high))
    say("")

    common = None
    for table in tables.values():
        common = table.index if common is None else common.intersection(table.index)
    rows = [row_from_frame(arm_name(args.prefix, rule, loss), table.loc[common],
                           target=target_col)
            for (rule, loss), table in tables.items()]
    context_path = os.path.join("results", "per_period", TOURNAMENT, MODEL_TREE,
                                CONTEXT_RUN + ".csv")
    context = None
    if os.path.exists(context_path) and args.era_step == 1:
        context = pd.read_csv(context_path, index_col=0, parse_dates=True)
        shared = common.intersection(context.index)
        if len(shared) == len(common):
            rows.append(row_from_frame(CONTEXT_RUN + " (ctx)", context.loc[common],
                                       target=target_col))
        else:
            say("context {} covers {} of {} periods -- left out of the table".format(
                CONTEXT_RUN, len(shared), len(common)))
            context = None
    else:
        say("context {} left out ({})".format(
            CONTEXT_RUN, "scout run" if args.era_step != 1 else "not on disk"))
    say("Level and risk, common window n={} (calmar and burn comparable at equal n):"
        .format(len(common)))
    say(render(rows, sort_by=None))
    say("")

    say("Paired, Newey-West lag 20 (|t| >= 2 is the bar):")
    say("{:<26} {:>10} {:>6} {:>10} {:>6} {:>10} {:>6} {:>5}".format(
        "comparison", "dCORR", "t", "dMMC", "t", "dpayout", "t", "n"))

    def delta_line(label, a, b):
        out = [label]
        for column in ("corr", "mmc", "payout"):
            x = payout_series(a) if column == "payout" else a[column]
            y = payout_series(b) if column == "payout" else b[column]
            mean, t, n = paired(x.dropna(), y.dropna())
            out += [mean, t]
        return "{:<26} {:>+10.5f} {:>+6.2f} {:>+10.5f} {:>+6.2f} {:>+10.5f} {:>+6.2f} {:>5}".format(
            *out, n)

    for loss in losses:
        say(delta_line("preq - tscv   [{}]".format(loss),
                       tables[("prequential", loss)].loc[common],
                       tables[("tscv", loss)].loc[common]))
    # Within a rule, every pair of losses: does WHAT the rule rewards matter?
    pairs = [(a, b) for a, b in (("corr", "mse"), ("payout", "corr"), ("payout", "mse"))
             if a in losses and b in losses]
    for rule in RULES:
        for a, b in pairs:
            say(delta_line("{} - {} [{}]".format(a, b, SHORT[rule]),
                           tables[(rule, a)].loc[common],
                           tables[(rule, b)].loc[common]))
    if context is not None:
        for (rule, loss), table in tables.items():
            say(delta_line("{}_{} - ctx".format(SHORT[rule], loss),
                           table.loc[common], context.loc[common]))
    say("")

    path = lambda_path(run, alphas)
    path.to_csv(os.path.join(REPORT_DIR, "lambda_path.csv"), index=False)
    # Evidence = the periods each pick stood on (resolved out-of-sample periods
    # for prequential, inner validation periods for tscv; for payout, only the
    # ones with a meta model).
    evidence, fallback = {}, {}
    for (rule, loss), (_, records) in run["arms"].items():
        key = "{}_{}".format(SHORT[rule], loss)
        own = [r for r in records if r["selection_loss"] == loss]
        evidence[key] = [r["selection_periods"] for r in own] or [0]
        fallback[key] = len(records) - len(own)
    say("Selected alpha per scored block ({} blocks). evid = periods the pick stood "
        "on, over the blocks chosen on the arm's own loss; fb = blocks that took the "
        "corr pick for want of any meta model:".format(len(path)))
    say("{:<14} {:>12} {:>12} {:>9} {:>9} {:>6} {:>10} {:>10} {:>4}".format(
        "arm", "median", "IQR(dec)", "distinct", "switches", "edge", "evid(1st)",
        "evid(min)", "fb"))
    for column in [c for c in path.columns if c not in ("refit", "predict_start")]:
        s = path_stats(path, column, alphas)
        say("{:<14} {:>12.4g} {:>12.2f} {:>9} {:>9} {:>6} {:>10} {:>10} {:>4}".format(
            column, s["median"], s["iqr_decades"], s["distinct"], s["switches"],
            s["edge"], evidence[column][0], min(evidence[column]), fallback[column]))
    for loss in losses:
        same = (path["preq_" + loss] == path["tscv_" + loss]).mean()
        say("blocks where preq and tscv agree [{}]: {:.0%}".format(loss, same))
    say("")

    # Ex post, NOT an arm: what each FIXED alpha would have scored on the scored
    # periods. A flat curve means no selection rule can matter.
    curves = [loss for loss in ("corr", "payout") if loss in losses]
    if curves:
        fixed = pd.DataFrame({"alpha": alphas})
        for loss in curves:
            scored_periods = run["losses"][loss].index.intersection(common)
            fixed[loss] = (-run["losses"][loss].loc[scored_periods]).mean(axis=0).to_numpy()
        fixed.to_csv(os.path.join(REPORT_DIR, "fixed_lambda_curve.csv"), index=False)
        say("Ex-post FIXED-alpha score on the scored periods (oracle, not an arm):")
        for loss in curves:
            best = fixed[loss].idxmax()
            say("  {:<6} best alpha {:.4g} -> {:.5f}; worst {:.5f}; OLS end {:.5f}; "
                "limit end {:.5f}".format(loss, fixed["alpha"][best], fixed[loss][best],
                                          fixed[loss].min(), fixed[loss].iloc[0],
                                          fixed[loss].iloc[-1]))
        say("  {:>16} ".format("alpha") + "".join("{:>10}".format(c) for c in curves))
        for q in np.linspace(0, len(alphas) - 1, 9).round().astype(int):
            say("  {:>16.4g} ".format(alphas[q])
                + "".join("{:>10.5f}".format(fixed[c][q]) for c in curves))
    say("")
    say("run time {:.1f} min".format((time.time() - started) / 60.0))

    text = "\n".join(lines)
    print("\n" + text)
    if args.era_step == 1:
        with open(REPORT_TXT, "w") as handle:
            handle.write(text + "\n")
    return text


def mail(subject, body):
    if os.environ.get("NO_MAIL") == "1":
        return
    try:
        from src.live.notify import send_text
        send_text(subject, body)
    except Exception as error:   # never let SMTP decide the run's exit status
        print("mail failed: {}".format(error))


if __name__ == "__main__":
    try:
        out = main()
    except SystemExit:
        raise
    except Exception:
        body = traceback.format_exc()
        print(body)
        if os.environ.get("SMOKE") != "1" and os.environ.get("DRY_RUN") != "1":
            mail("[crypto] E115 FAILED -- ridge lambda prequential vs tscv", body)
        sys.exit(1)
    if isinstance(out, str) and os.environ.get("SMOKE") != "1":
        mail("[crypto] E115 finished -- ridge lambda prequential vs tscv", out)
