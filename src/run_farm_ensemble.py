"""
Combine several completed runs with the Factor-Adjusted Regularized Model of
Lee & Lee (2025), weights re-estimated WALK-FORWARD.

    python3 -m src.run_farm_ensemble crypto ridge_glmnet \
        ridge_glmnet/e17_featdiff9 lgb/e05_cumlag7 mlp/e08_mlp_onehot_rollsel \
        gen_ridge/e26_pos gen_ridge/e26_neg \
        --method farm1 --beta-method l2boost --run-name e27_decorr5_farm1_l2boost

This is the FARM sibling of src.run_weighted_ensemble. That module chooses a
convex weight vector over the members' scores; this one treats the simple average
as a common factor and regresses its error on the members' idiosyncratic
components (src/prediction/farm.py). The `--method equal` arm reproduces the plain
1/N rank average exactly, so it is the baseline every FARM arm is measured against,
and it is written through the SAME scoring path as run_weighted_ensemble and
run_ensemble -- so a FARM combination is directly comparable with a fixed-weight
blend, a walk-forward-weighted blend and any single run, and shows up in the
experiment database like any other arm.

The lag is the target's resolution lag, not the CV embargo -- see the module
docstring in src/prediction/farm.py. Members enter as per-period ranks and are
RE-RANKED on the common panel here, so the simple average is a clean permutation
mean and the no-intercept FARM fit is unbiased.
"""

import argparse
import json
import os
from time import time

from src.data_mani.utils import META_MODEL_COL, TOURNAMENTS, get_spec
from src.prediction.farm import (BETA_METHODS, METHODS, walk_forward_combination,
                                 LAG, UPDATE_EVERY, MIN_HISTORY, ROLL_PERIODS)
from src.prediction.metrics import per_period_table, summarize, summarize_predictions
from src.run_weighted_ensemble import load_members


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine completed runs with walk-forward FARM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tournament", choices=sorted(TOURNAMENTS))
    parser.add_argument("model", help="model directory results are written to")
    parser.add_argument("members", nargs="+",
                        help="member runs as 'model_dir/run_name' or 'run_name'")
    parser.add_argument("--method", default="farm1", choices=sorted(METHODS),
                        help="equal (1/N baseline), farm1/2/3, gr, or ds")
    parser.add_argument("--beta-method", default="ridge", choices=sorted(BETA_METHODS),
                        help="estimator for the farm* deviation regression")
    parser.add_argument("--update-every", type=int, default=UPDATE_EVERY,
                        help="periods between coefficient updates")
    parser.add_argument("--lag", type=int, default=LAG,
                        help="periods a fit must wait for the target to resolve")
    parser.add_argument("--min-history", type=int, default=MIN_HISTORY,
                        help="resolved periods before leaving the equal-weight cold start")
    parser.add_argument("--roll", type=int, default=ROLL_PERIODS,
                        help="rolling estimation window, in periods")
    parser.add_argument("-n", "--run-name", type=str, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time()
    spec = get_spec(args.tournament)
    period_col, id_col, target_col = spec["period_col"], spec["id_col"], spec["target"]

    ranked, extras = load_members(args.tournament, args.model, args.members,
                                  period_col, id_col, target_col)

    # Re-rank on the COMMON panel so every member is a clean per-period permutation
    # and the simple average is an unbiased centre -- see farm.walk_forward_combination.
    period_index = ranked.index.get_level_values(period_col)
    ranked = ranked.groupby(period_index).rank(pct=True)

    if not args.quiet:
        print("aligned on %d rows, %d periods, %d members" % (
            len(ranked), period_index.nunique(), len(args.members)))

    prediction, records = walk_forward_combination(
        ranked.to_numpy(), extras[target_col].to_numpy(), period_index.to_numpy(),
        method=args.method, beta_method=args.beta_method, lag=args.lag,
        update_every=args.update_every, min_history=args.min_history, roll=args.roll)

    if not args.quiet:
        print("method=%s  beta=%s  update_every=%d  lag=%d  min_history=%d  roll=%d"
              % (args.method, args.beta_method, args.update_every, args.lag,
                 args.min_history, args.roll))
        cold = sum(1 for r in records if r["scheme"].startswith("equal"))
        r_choices = [r["r"] for r in records if "r" in r]
        print("   %d update(s), %d cold-start%s%s" % (
            len(records), cold, "" if cold == 1 else "s",
            "" if not r_choices else "  mean r=%.2f" % (sum(r_choices) / len(r_choices))))

    frame = extras.copy()
    frame["prediction"] = prediction
    frame = frame.reset_index()

    reference_col = ("reference_prediction"
                     if "reference_prediction" in frame.columns else None)
    meta_col = META_MODEL_COL if META_MODEL_COL in frame.columns else None

    results = summarize_predictions(frame, "prediction", target_col, period_col,
                                    reference_col=reference_col,
                                    meta_model_col=meta_col)
    table = per_period_table(frame, "prediction", target_col, period_col,
                             reference_col=reference_col, meta_model_col=meta_col)
    results["pnl"] = summarize(table["pnl"])
    results["config"] = {
        "tournament": args.tournament, "model": args.model,
        # See run_ensemble: the config must carry the label too, so the row
        # survives its forecast being reclaimed.
        "target": target_col,
        "members": args.members, "method": args.method,
        "beta_method": args.beta_method, "update_every": args.update_every,
        "lag": args.lag, "min_history": args.min_history, "roll": args.roll,
    }
    results["farm_updates"] = records

    run_name = args.run_name or "farm_%s_%s" % (args.method, args.beta_method)
    paths = {}
    for kind in ("forecast", "per_period", "metrics"):
        directory = os.path.join("results", kind, args.tournament, args.model)
        os.makedirs(directory, exist_ok=True)
        paths[kind] = os.path.join(
            directory, "%s.%s" % (run_name, "json" if kind == "metrics" else "csv"))
    frame.to_csv(paths["forecast"], index=False)
    # index_label matches run_forecast and run_weighted_ensemble: the per-period
    # files must carry their date index or no paired comparison can join them.
    table.to_csv(paths["per_period"], index_label=period_col)
    with open(paths["metrics"], "w") as handle:
        json.dump(results, handle, indent=2, default=str)

    if not args.quiet:
        counts = {"corr": "n_periods", "bmc": "bmc_n_periods", "mmc": "mmc_n_periods"}
        for key, count_key in counts.items():
            block = results.get(key)
            if block:
                print("%-5s mean %9.6f  sharpe %7.4f  max_dd %7.5f  n=%s" % (
                    key.upper(), block["mean"], block["sharpe"],
                    block["max_drawdown"], results.get(count_key, "?")))
        print("wrote %s  (%.1fs)" % (paths["metrics"], time() - started))


if __name__ == "__main__":
    main()
