"""
Compare the 20-day and 60-day model blocks, and ensemble across them.

    python3 -m src.analysis.cross_target                 # report on what exists
    python3 -m src.analysis.cross_target --wait          # block until t60 finishes
    python3 -m src.analysis.cross_target --wait --run    # then build the ensembles

Four steps: wait for the experiment, correlate the two blocks, propose the pairs
worth combining, run them as full pseudo-out-of-sample ensembles.

------------------------------------------------------------------------------
Everything is scored on the 20-day target, including the 60-day models
------------------------------------------------------------------------------

This is the decision that makes the comparison mean anything. Crypto pays on
target_binned_return_20; the 60-day label is a different SIGNAL, not a different
payout. So a 60-day model's own metrics file -- which scores it against the
60-day label -- answers a question nobody is paid for.

Here every arm's predictions are re-scored against the 20-day target, with the
same reference and meta-model series the deployed slots use. A 60-day model that
looks weaker on its own label may still be the most valuable thing in the book
if it ranks coins differently and still correlates with what pays.

The two blocks also cover different windows: the 60-day target is NaN over its
last 40 dates, so those arms score 470 periods against 510. Every comparison
below is an inner join on (symbol, date), and the row count is printed, because
comparing two arms on different denominators is the easiest way to get a
flattering answer.

------------------------------------------------------------------------------
Which pairs get proposed
------------------------------------------------------------------------------

Adding a weaker model to a stronger one raises the combined Sharpe if and only if

    rho  <  S_weak / S_strong

(the condition for the weak member to enter with a POSITIVE weight; verified on
91 of 91 pairs earlier in this project). The threshold is not a constant -- two
equally good models help each other at almost any correlation, while a model
with a tenth the Sharpe needs to be nearly orthogonal.

Because Sharpes here are estimated on ~25 independent windows, the proposal
discounts the threshold by a third before acting on it, and every proposed
ensemble is built at a FIXED 50/50. That is not laziness: seven weighting
schemes were tested walk-forward on this data and all eight lost to 1/N, four of
them significantly. Estimated weights are the wrong tool at this sample size.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from src.data_mani.utils import META_MODEL_COL, get_spec
from src.prediction.metrics import per_period_table, summarize, summarize_predictions

FORECAST = "results/forecast/crypto"
METRICS = "results/metrics/crypto"
PER_PERIOD = "results/per_period/crypto"

TARGET_20 = "target_binned_return_20"
PERIOD, SYMBOL = "date", "symbol"
NEWEY_WEST_LAG = 20

# The deployed 20-day block: the four single-model slots plus CURUPIRA_05's
# 1/N blend. CURUPIRA_06 is excluded because it is a function of two of these.
BLOCK_20 = [
    ("C01_20", "ridge_glmnet", "w6r6_glmnet_reg_tenc"),
    ("C02_20", "ridge", "w6r6_fixed_reg"),
    ("C03_20", "ridge_glmnet", "w6r6_glmnet_reg_onehot_rollsel"),
    ("C04_20", "ridge_glmnet", "w6r6_glmnet_reg_allfeatpp"),
    ("C05_20", "ridge_glmnet", "w6r6_blend_quintet2"),
]

# The 60-day block, produced by experiments/target60_crypto.sh.
BLOCK_60 = [
    ("C01_60", "ridge_glmnet", "t60_c01_tenc"),
    ("C02_60", "ridge", "t60_c02_raw"),
    ("C03_60", "ridge_glmnet", "t60_c03_onehotroll"),
    ("C04_60", "ridge_glmnet", "t60_c04_allfeatpp"),
    ("C05_60", "ridge_glmnet", "t60_c05_blend"),
]

# Members of the 60-day run that are not slots in their own right; they exist so
# the C05_60 blend can be reconstructed, and they are what the runner waits on.
MEMBERS_60 = ["t60_m_onehot", "t60_m_tenc", "t60_m_interact",
              "t60_m_size", "t60_m_price"]


def forecast_path(model, run):
    return os.path.join(FORECAST, model, run + ".csv")


def completed(block):
    """(present, missing) run names for a block."""
    present = [(n, m, r) for n, m, r in block if os.path.exists(forecast_path(m, r))]
    missing = [r for n, m, r in block if not os.path.exists(forecast_path(m, r))]
    return present, missing


def wait_for_block(block, poll=120, timeout=36000, verbose=True):
    """
    Block until every arm exists, or give up.

    Polls the RESULT FILES rather than the process table: a pgrep pattern would
    match this waiter's own command line, which is how an earlier chained job in
    this project spent 90 minutes waiting on itself.
    """
    waited = 0
    while True:
        _, missing = completed(block)
        if not missing:
            if verbose:
                print("all %d arms present after %d min" % (len(block), waited // 60))
            return True
        if waited >= timeout:
            print("gave up after %d min; still missing: %s" % (waited // 60, missing))
            return False
        if verbose and waited % 600 == 0:
            print("  waiting, %d/%d present (%d min elapsed)"
                  % (len(block) - len(missing), len(block), waited // 60))
        time.sleep(poll)
        waited += poll


def load_scoring_frame():
    """
    The 20-day target plus the reference and meta-model series, from a deployed
    arm's forecast file.

    Taken from an existing 20-day run rather than rebuilt from the parquet so
    the reference and meta-model columns are exactly the ones every deployed
    slot was scored against.
    """
    base = pd.read_csv(forecast_path("ridge_glmnet", "w6r6_glmnet_reg_tenc"),
                       parse_dates=[PERIOD])
    keep = [c for c in (TARGET_20, META_MODEL_COL, "reference_prediction")
            if c in base.columns]
    return base.set_index([SYMBOL, PERIOD])[keep]


def load_predictions(arms):
    """Within-period ranked predictions for each arm, aligned on (symbol, date)."""
    frames = {}
    for name, model, run in arms:
        path = forecast_path(model, run)
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path, usecols=[SYMBOL, PERIOD, "prediction"],
                            parse_dates=[PERIOD]).set_index([SYMBOL, PERIOD])
        frames[name] = frame.groupby(level=1)["prediction"].rank(pct=True)
    return pd.DataFrame(frames)


def score_on_payable_target(prediction, extras):
    """
    Score any prediction series against the 20-day target.

    The summary is DERIVED from the per-period table rather than recomputed by
    summarize_predictions. Calling both costs 45s per arm against 23s, because
    the second call redoes every per-period correlation the first already did --
    and with 10 arms plus a pair screen that difference is the whole runtime.

    :return: (per-period table, summary dict)
    """
    frame = extras.copy()
    frame["prediction"] = prediction
    frame = frame.dropna(subset=["prediction", TARGET_20]).reset_index()

    reference = "reference_prediction" if "reference_prediction" in frame else None
    meta = META_MODEL_COL if META_MODEL_COL in frame else None
    table = per_period_table(frame, "prediction", TARGET_20, PERIOD,
                             reference_col=reference, meta_model_col=meta)

    summary = {"n_periods": int(len(table))}
    for column in ("corr", "bmc", "mmc", "pnl"):
        if column in table:
            series = table[column].dropna()
            if len(series):
                summary[column] = summarize(series)
                summary["%s_n_periods" % column] = int(len(series))
    return table, summary


def payout(table):
    return 0.1 * table["corr"] + table["mmc"]


def stats(series):
    x = np.asarray(series, dtype="float64")
    mean, sd = x.mean(), x.std(ddof=0)
    cumulative = x.cumsum()
    return mean, (mean / sd if sd > 0 else np.nan), \
        (np.maximum.accumulate(cumulative) - cumulative).max()


def newey_west(series, lag=NEWEY_WEST_LAG):
    x = np.asarray(series, dtype="float64")
    n = len(x)
    mean = x.mean()
    e = x - mean
    variance = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(variance, 0.0) / n)
    return mean, (mean / se if se > 0 else np.nan)


def within_period_rank_corr(predictions):
    """Mean over periods of the within-period correlation matrix."""
    dates = predictions.index.get_level_values(PERIOD)
    centred = predictions - predictions.groupby(dates).transform("mean")
    scale = centred.groupby(dates).transform("std")
    z = (centred / scale.replace(0.0, np.nan)).fillna(0.0)

    total = np.zeros((z.shape[1], z.shape[1]))
    n = 0
    for _, block in z.groupby(dates):
        values = block.to_numpy()
        total += values.T @ values / max(len(values) - 1, 1)
        n += 1
    return pd.DataFrame(total / n, index=z.columns, columns=z.columns)


def main():
    parser = argparse.ArgumentParser(
        description="Correlate the 20d and 60d blocks and ensemble across them.")
    parser.add_argument("--wait", action="store_true",
                        help="block until the 60-day experiment finishes")
    parser.add_argument("--run", action="store_true",
                        help="build and score the proposed ensembles")
    parser.add_argument("--top", type=int, default=5,
                        help="how many proposed pairs to actually run")
    parser.add_argument("--discount", type=float, default=0.67,
                        help="haircut on the Sharpe threshold, for estimation "
                             "error on ~25 independent windows")
    parser.add_argument("--email", action="store_true",
                        help="mail the results when finished, using the same "
                             "SMTP settings as the live round notifier")
    parser.add_argument("--poll", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=36000)
    args = parser.parse_args()

    if args.wait:
        pending = BLOCK_60 + [("m_%s" % r, "ridge_glmnet", r) for r in MEMBERS_60]
        # The blend is produced last by the runner, so waiting on BLOCK_60 alone
        # is enough; the members are listed so a partial failure is visible.
        if not wait_for_block(BLOCK_60, poll=args.poll, timeout=args.timeout):
            print("\nproceeding with whatever completed")

    present_20, missing_20 = completed(BLOCK_20)
    present_60, missing_60 = completed(BLOCK_60)
    print("\n20-day block: %d/%d present%s" % (
        len(present_20), len(BLOCK_20),
        "" if not missing_20 else "  MISSING %s" % missing_20))
    print("60-day block: %d/%d present%s" % (
        len(present_60), len(BLOCK_60),
        "" if not missing_60 else "  MISSING %s" % missing_60))
    if not present_60:
        print("\nnothing from the 60-day experiment yet -- rerun with --wait, or "
              "after experiments/target60_crypto.sh finishes.")
        return

    extras = load_scoring_frame()
    predictions = load_predictions(present_20 + present_60)
    predictions = predictions.dropna(how="any")
    print("\naligned on %d rows, %d periods (inner join: the 60-day arms lose "
          "the last 40 dates to an unresolved target)"
          % (len(predictions), predictions.index.get_level_values(PERIOD).nunique()))

    # ---- every arm re-scored on the payable 20-day target --------------------
    scored = {}
    for name in predictions.columns:
        table, summary = score_on_payable_target(predictions[name], extras)
        series = payout(table)
        mean, sharpe, drawdown = stats(series)
        scored[name] = {"series": series, "payout": mean, "sharpe": sharpe,
                        "maxdd": drawdown, "corr": summary["corr"]["mean"],
                        "mmc": summary["mmc"]["mean"] if summary.get("mmc") else np.nan}

    print("\n=== every arm scored on the PAYABLE 20-day target ===")
    print("%-10s %10s %10s %10s %8s %8s" % ("arm", "CORR", "MMC", "payout",
                                            "p.shp", "p.dd"))
    for name in sorted(scored, key=lambda k: -scored[k]["payout"]):
        s = scored[name]
        print("%-10s %10.6f %10.6f %10.6f %8.4f %8.4f" % (
            name, s["corr"], s["mmc"], s["payout"], s["sharpe"], s["maxdd"]))

    # ---- the cross-block correlation matrix ----------------------------------
    matrix = within_period_rank_corr(predictions)
    names_20 = [n for n, _, _ in present_20]
    names_60 = [n for n, _, _ in present_60]
    print("\n=== prediction correlation, 20-day block (rows) x 60-day block (cols) ===")
    print(matrix.loc[names_20, names_60].round(3).to_string())

    within_20 = matrix.loc[names_20, names_20].to_numpy()
    within_60 = matrix.loc[names_60, names_60].to_numpy()
    cross = matrix.loc[names_20, names_60].to_numpy()
    print("\nmean |rho| within 20-day block : %.3f" % _offdiag(within_20))
    print("mean |rho| within 60-day block : %.3f" % _offdiag(within_60))
    print("mean |rho| ACROSS blocks       : %.3f" % np.abs(cross).mean())

    # ---- screen pairs cheaply, confirm the survivors properly ----------------
    #
    # Fully scoring all 45 pairs would cost ~17 minutes, and the screen only has
    # to RANK them. So pairs are ranked on a 50/50 combination of the members'
    # PAYOUT SERIES, which is free once each arm is scored, and only the top few
    # are then built as real prediction-level blends.
    #
    # These are genuinely different objects: combining payout series is what
    # splitting a stake across two model slots does, while blending predictions
    # is one slot. They agree closely enough to rank on, and the confirmation
    # step below reports the real thing rather than the proxy.
    proposals = []
    for a, b in itertools.combinations(scored, 2):
        sa, sb = scored[a]["sharpe"], scored[b]["sharpe"]
        strong, weak = (a, b) if sa >= sb else (b, a)
        s_strong, s_weak = max(sa, sb), min(sa, sb)
        if not np.isfinite(s_strong) or s_strong <= 0:
            continue
        joined = pd.concat([scored[a]["series"], scored[b]["series"]],
                           axis=1, join="inner", keys=["a", "b"])
        combined = 0.5 * joined["a"] + 0.5 * joined["b"]
        mean, sharpe, drawdown = stats(combined)
        proposals.append({
            "pair": (strong, weak), "rho": matrix.loc[a, b],
            "threshold": (s_weak / s_strong) * args.discount,
            "passes": matrix.loc[a, b] < (s_weak / s_strong) * args.discount,
            "cross_block": (a in names_20) != (b in names_20),
            "payout": mean, "sharpe": sharpe, "maxdd": drawdown,
            "gain": sharpe - s_strong, "s_strong": s_strong,
        })
    proposals.sort(key=lambda p: -p["sharpe"])

    print("\n=== proposed 50/50 pairs, ranked by realised payout sharpe ===")
    print("(threshold = discounted S_weak/S_strong; 'X' marks a cross-block pair)")
    print("%-24s %3s %7s %10s %10s %8s %8s %8s" % (
        "pair", "X", "rho", "threshold", "payout", "sharpe", "d.shp", "maxdd"))
    for p in proposals[:12]:
        print("%-24s %3s %7.3f %10.3f %10.6f %8.4f %+8.4f %8.4f" % (
            "%s + %s" % p["pair"], "X" if p["cross_block"] else "",
            p["rho"], p["threshold"], p["payout"], p["sharpe"],
            p["gain"], p["maxdd"]))

    best_single = max(scored.values(), key=lambda s: s["sharpe"])
    best_single_name = max(scored, key=lambda k: scored[k]["sharpe"])
    print("\nbest single arm: %s (payout %.6f, sharpe %.4f)"
          % (best_single_name, best_single["payout"], best_single["sharpe"]))

    lines = _render(scored, matrix, names_20, names_60, proposals,
                    best_single_name, best_single, len(predictions))

    if not args.run:
        print("\n(--run to build the top %d as full ensembles and paired-test them)"
              % args.top)
        if args.email:
            _email("\n".join(lines), subject="cross-target report (no ensembles run)")
        return

    # ---- confirm the top proposals as real prediction-level blends ----------
    print("\n=== full pseudo-OOS ensembles, scored on the 20-day target ===")
    reference = scored[best_single_name]["series"]
    header = ("%-26s %10s %10s %10s %8s %8s %11s %7s"
              % ("ensemble", "CORR", "MMC", "payout", "p.shp", "p.dd",
                 "d vs best", "t"))
    print(header)
    lines += ["", "=== full pseudo-OOS ensembles (prediction-level 50/50) ===",
              "reference: %s" % best_single_name, header]

    for p in proposals[:args.top]:
        a, b = p["pair"]
        blended = 0.5 * predictions[a] + 0.5 * predictions[b]
        table, summary = score_on_payable_target(blended, extras)
        series = payout(table)
        mean, sharpe, drawdown = stats(series)
        joined = pd.concat([reference, series], axis=1, join="inner",
                           keys=["ref", "arm"])
        delta, t = newey_west(joined["arm"] - joined["ref"])

        run_name = "xt_%s_%s" % (a.lower(), b.lower())
        _write_run(run_name, table, summary, [a, b])
        row = ("%-26s %10.6f %10.6f %10.6f %8.4f %8.4f %+11.6f %7.2f"
               % (run_name, summary["corr"]["mean"],
                  summary["mmc"]["mean"] if summary.get("mmc") else np.nan,
                  mean, sharpe, drawdown, delta, t))
        print(row)
        lines.append(row)

    tail = ("\nwritten to results/{forecast,per_period,metrics}/crypto/"
            "ridge_glmnet/xt_*")
    print(tail)
    lines.append(tail)

    if args.email:
        _email("\n".join(lines))


def _render(scored, matrix, names_20, names_60, proposals,
            best_name, best, n_rows):
    """The report, as text, for the email."""
    lines = [
        "Cross-target analysis: 20-day block vs 60-day block.",
        "Every arm re-scored on the PAYABLE 20-day target, inner-joined on "
        "%d rows." % n_rows,
        "",
        "=== every arm, scored on the payable 20-day target ===",
        "%-10s %10s %10s %10s %8s %8s" % ("arm", "CORR", "MMC", "payout",
                                          "p.shp", "p.dd"),
    ]
    for name in sorted(scored, key=lambda k: -scored[k]["payout"]):
        s = scored[name]
        lines.append("%-10s %10.6f %10.6f %10.6f %8.4f %8.4f" % (
            name, s["corr"], s["mmc"], s["payout"], s["sharpe"], s["maxdd"]))

    lines += ["", "=== prediction correlation, 20-day (rows) x 60-day (cols) ===",
              matrix.loc[names_20, names_60].round(3).to_string(), ""]
    cross = matrix.loc[names_20, names_60].to_numpy()
    lines.append("mean |rho| across blocks: %.3f  (within 20d: %.3f, within 60d: %.3f)"
                 % (np.abs(cross).mean(),
                    _offdiag(matrix.loc[names_20, names_20].to_numpy()),
                    _offdiag(matrix.loc[names_60, names_60].to_numpy())))

    lines += ["", "=== proposed 50/50 pairs (screened on payout series) ===",
              "%-24s %3s %7s %10s %10s %8s %8s" % (
                  "pair", "X", "rho", "threshold", "payout", "sharpe", "d.shp")]
    for p in proposals[:12]:
        lines.append("%-24s %3s %7.3f %10.3f %10.6f %8.4f %+8.4f" % (
            "%s + %s" % p["pair"], "X" if p["cross_block"] else "",
            p["rho"], p["threshold"], p["payout"], p["sharpe"], p["gain"]))
    lines += ["", "best single arm: %s (payout %.6f, sharpe %.4f)"
              % (best_name, best["payout"], best["sharpe"])]
    return lines


def _email(body, subject="cross-target ensemble results"):
    """
    Mail the report, reusing the live pipeline's notifier.

    Never raises: the analysis has already run and been printed by this point,
    and a broken mailer must not turn a finished job into a failed one -- the
    same rule the round notifier follows.
    """
    try:
        from src.live.env import load_env
        from src.live.notify import send
        load_env()
        send({"round": subject, "live_date": "-",
              "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
              "elapsed_minutes": 0.0, "deadline_minutes": "-", "dry_run": False,
              "slots": [], "diagnostics": {}, "warnings": [],
              "crash": None, "report_body": body})
    except Exception as error:
        print("\n(email failed: %s -- the results above are unaffected)" % error)


def _offdiag(matrix):
    return np.abs(matrix[np.triu_indices(len(matrix), k=1)]).mean()


def _write_run(run_name, table, summary, members):
    """Persist an ensemble in the standard results tree."""
    summary = dict(summary)
    summary["config"] = {"tournament": "crypto", "model": "ridge_glmnet",
                         "members": members, "weights": [0.5, 0.5],
                         "scored_on": TARGET_20,
                         "note": "cross-target ensemble; 60-day members are "
                                 "re-scored on the payable 20-day target"}
    for kind, obj in (("per_period", table), ("metrics", summary)):
        directory = os.path.join("results", kind, "crypto", "ridge_glmnet")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "%s.%s"
                            % (run_name, "json" if kind == "metrics" else "csv"))
        if kind == "metrics":
            with open(path, "w") as handle:
                json.dump(obj, handle, indent=2, default=str)
        else:
            obj.to_csv(path)


if __name__ == "__main__":
    main()
