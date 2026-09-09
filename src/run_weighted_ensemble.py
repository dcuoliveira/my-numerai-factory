"""
Ensemble several completed runs with weights re-estimated WALK-FORWARD.

    python3 -m src.run_weighted_ensemble crypto ridge_glmnet \
        ridge_glmnet/w6r6_alone_onehot ridge_glmnet/w6r6_alone_tenc \
        ridge_glmnet/w6r6_alone_interact ridge/w6r6_grp_size_fixed \
        ridge_glmnet/w6r6_grp_price_glmnet \
        --scheme caruana --run-name w6r6_wblend_caruana

src.run_ensemble applies ONE fixed weight vector over the whole history. This
applies a weight vector that is re-estimated as history accumulates, using only
what was knowable at the time -- which is the only way a weighting scheme can be
compared with 1/N honestly.

------------------------------------------------------------------------------
The lag, and why it is not the embargo
------------------------------------------------------------------------------

To weight members by how they have been scoring, the scores must have RESOLVED.
A 20-day forward target opened at period p is not known until p + 20, so weights
applied at period t may only read member scores from periods <= t - 20.

This is the same arithmetic as the CV embargo but a different mechanism, and
conflating them is how a weighting study turns into a leakage study. The
embargo separates training rows from validation rows within one fit. This lag
separates the OUTCOME of a prediction from the decision that uses it. Drop it
and every scheme below scores beautifully, for the same reason the whole-sample
target encoding scored 0.2536 against a causal 0.0557.

------------------------------------------------------------------------------
Cold start
------------------------------------------------------------------------------

At the first update point there is no resolved history at all, so the weights
are 1/N by necessity. `--min-history` extends that: below it the estimate would
be noise dressed as a decision, and equal weights are used instead. Every block
records which scheme actually produced its weights, so a run that fell back to
equal for most of its life cannot be mistaken for one that did not.
"""

import argparse
import json
import os
from time import time

import numpy as np
import pandas as pd

from src.data_mani.utils import META_MODEL_COL, TOURNAMENTS, get_spec
from src.prediction.metrics import per_period_table, summarize, summarize_predictions
from src.prediction.weighting import PREDICTION_BASED, SCHEMES, estimate

# Rows the prediction-based schemes may regress on. The full history is 153,000
# rows by the end; the schemes are linear and converge long before that, and
# capping keeps a single run in seconds rather than minutes.
MAX_REGRESSION_ROWS = 60000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble completed runs with walk-forward weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tournament", choices=sorted(TOURNAMENTS))
    parser.add_argument("model", help="model directory results are written to")
    parser.add_argument("members", nargs="+",
                        help="member runs as 'model_dir/run_name' or 'run_name'")
    parser.add_argument("--scheme", default="equal", choices=sorted(SCHEMES))
    parser.add_argument("--update-every", type=int, default=22,
                        help="periods between weight updates (22 ~ one month "
                             "of crypto dates)")
    parser.add_argument("--lag", type=int, default=20,
                        help="periods a prediction needs to resolve; weights at "
                             "t may only read scores through t - lag")
    parser.add_argument("--min-history", type=int, default=60,
                        help="resolved periods required before leaving 1/N")
    parser.add_argument("--window", type=int, default=0,
                        help="rolling estimation window in periods; 0 = expanding")
    parser.add_argument("--objective", default="sharpe", choices=["sharpe", "mean"],
                        help="what the greedy and subset schemes maximise")
    parser.add_argument("--shrinkage", type=float, default=0.2,
                        help="shrunk_tangency only: trust placed in the covariance")
    parser.add_argument("--subset-size", type=int, default=3,
                        help="complete_subset only: members per subset")
    parser.add_argument("--keep-fraction", type=float, default=0.5,
                        help="complete_subset only: fraction of scored subsets "
                             "kept; 1.0 keeps every subset and reduces to 1/N")
    parser.add_argument("--gr-unconstrained", action="store_true",
                        help="granger_ramanathan only: drop the sum-to-one and "
                             "non-negativity constraints. This is the variant "
                             "that produced sign-flipping weights across "
                             "subsamples -- included so that is measurable, not "
                             "because it is recommended")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="stacking_nnls only: ridge penalty on the weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-n", "--run-name", type=str, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def load_members(tournament, model, members, period_col, id_col, target_col):
    """Ranked member predictions aligned on (id, period), plus the extras."""
    frames, extras = [], None
    for name in members:
        member_model, run = name.split("/", 1) if "/" in name else (model, name)
        path = os.path.join("results", "forecast", tournament, member_model,
                            "%s.csv" % run)
        if not os.path.exists(path):
            raise SystemExit("member run not found: %s" % path)

        member = pd.read_csv(
            path, parse_dates=[period_col] if period_col == "date" else None)
        member = member.set_index([id_col, period_col])
        ranked = member.groupby(level=1)["prediction"].rank(pct=True)
        frames.append(ranked.rename(name))

        if extras is None:
            keep = [c for c in (target_col, META_MODEL_COL, "reference_prediction")
                    if c in member.columns]
            extras = member[keep]

    joined = pd.concat(frames, axis=1, join="inner")
    return joined, extras.loc[joined.index]


def member_scores(ranked, target, period_col):
    """
    Each member's own per-period score, the input the score-based schemes read.

    Spearman between the member's ranks and the target within each period. This
    stands in for CORR rather than calling numerai_corr: the members are already
    rank-transformed, the schemes only ever compare periods against each other,
    and it keeps the update loop cheap enough to run every 22 periods.
    """
    frame = ranked.copy()
    frame["_target"] = target.to_numpy()
    periods = frame.index.get_level_values(period_col)
    return frame.groupby(periods).apply(
        lambda block: block[ranked.columns].corrwith(block["_target"], method="spearman"))


def walk_forward_weights(ranked, extras, target_col, period_col, args):
    """
    Weights for every period, re-estimated on the schedule and never reading
    beyond the resolution lag.

    :return: (weights frame indexed by period, list of update records)
    """
    periods = np.array(sorted(ranked.index.get_level_values(period_col).unique()))
    position = {p: i for i, p in enumerate(periods)}
    names = list(ranked.columns)

    scores = member_scores(ranked, extras[target_col], period_col)
    scores = scores.reindex(periods)

    row_periods = ranked.index.get_level_values(period_col)
    row_position = np.array([position[p] for p in row_periods])

    weights = np.zeros((len(periods), len(names)))
    records = []
    current = np.ones(len(names)) / len(names)
    scheme_in_force = "equal (cold start)"

    for start in range(0, len(periods), args.update_every):
        # Everything at or after `start` is the future. The lag removes the
        # additional block whose targets have not resolved yet.
        usable = max(0, start - args.lag)
        lo = 0 if args.window <= 0 else max(0, usable - args.window)

        if usable - lo >= args.min_history:
            history = scores.iloc[lo:usable].dropna(how="any")
            try:
                if args.scheme in PREDICTION_BASED:
                    mask = (row_position >= lo) & (row_position < usable)
                    block = ranked.loc[mask]
                    y = extras.loc[mask, target_col]
                    if len(block) > MAX_REGRESSION_ROWS:
                        take = np.linspace(0, len(block) - 1, MAX_REGRESSION_ROWS)
                        block = block.iloc[take.astype(int)]
                        y = y.iloc[take.astype(int)]
                    current = estimate(args.scheme, predictions=block.to_numpy(),
                                       target=y.to_numpy(), alpha=args.ridge_alpha,
                                       constrained=not args.gr_unconstrained)
                elif len(history) >= args.min_history:
                    current = estimate(
                        args.scheme, scores=history.to_numpy(),
                        objective=args.objective, shrinkage=args.shrinkage,
                        subset_size=args.subset_size,
                        keep_fraction=args.keep_fraction, seed=args.seed)
                scheme_in_force = args.scheme
            except Exception as error:
                # A scheme that cannot be estimated must not lose the round; the
                # record says so rather than the run failing.
                scheme_in_force = "equal (%s failed: %s)" % (args.scheme, error)
                current = np.ones(len(names)) / len(names)

        stop = min(start + args.update_every, len(periods))
        weights[start:stop] = current
        records.append({
            "from_period": str(periods[start]), "scheme": scheme_in_force,
            "history_periods": int(max(0, usable - lo)),
            "weights": {n: round(float(w), 4) for n, w in zip(names, current)},
        })

    return pd.DataFrame(weights, index=periods, columns=names), records


def main():
    args = parse_args()
    started = time()
    spec = get_spec(args.tournament)
    period_col, id_col, target_col = spec["period_col"], spec["id_col"], spec["target"]

    ranked, extras = load_members(args.tournament, args.model, args.members,
                                  period_col, id_col, target_col)
    if not args.quiet:
        print("aligned on %d rows, %d periods, %d members" % (
            len(ranked), ranked.index.get_level_values(period_col).nunique(),
            len(args.members)))

    weights, records = walk_forward_weights(ranked, extras, target_col,
                                            period_col, args)
    if not args.quiet:
        print("scheme=%s  update_every=%d  lag=%d  min_history=%d  window=%s"
              % (args.scheme, args.update_every, args.lag, args.min_history,
                 args.window or "expanding"))
        for record in records:
            print("   from %s  %-28s %s" % (
                record["from_period"][:10], record["scheme"],
                " ".join("%s:%.2f" % (k.split("/")[-1][:14], v)
                         for k, v in record["weights"].items())))

    # Apply each period's weights to that period's rows.
    row_periods = ranked.index.get_level_values(period_col)
    aligned = weights.reindex(row_periods).to_numpy()
    blend = (ranked.to_numpy() * aligned).sum(axis=1)

    frame = extras.copy()
    frame["prediction"] = blend
    frame = frame.reset_index()

    # Same scoring path as src.run_ensemble, so a walk-forward-weighted blend is
    # directly comparable with a fixed-weight one and with any single run.
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
        "members": args.members, "scheme": args.scheme,
        "update_every": args.update_every, "lag": args.lag,
        "min_history": args.min_history, "window": args.window,
        "objective": args.objective, "shrinkage": args.shrinkage,
        "subset_size": args.subset_size, "keep_fraction": args.keep_fraction,
        "gr_unconstrained": args.gr_unconstrained,
        "ridge_alpha": args.ridge_alpha, "seed": args.seed,
    }
    results["weight_updates"] = records

    run_name = args.run_name or "wens_%s" % args.scheme
    paths = {}
    for kind, obj in (("forecast", frame), ("per_period", table), ("metrics", results)):
        directory = os.path.join("results", kind, args.tournament, args.model)
        os.makedirs(directory, exist_ok=True)
        paths[kind] = os.path.join(
            directory, "%s.%s" % (run_name, "json" if kind == "metrics" else "csv"))
    frame.to_csv(paths["forecast"], index=False)
    # per_period_table() indexes by period, and that index IS the date column.
    # Dropping it here (index=False) makes these files unjoinable to any other
    # run: weighting_report pairs arms date by date, and a paired comparison is
    # the only way a difference between near-identical blends is measurable at
    # all. run_forecast writes the same table WITH its index, for the same
    # reason -- the two writers must agree.
    table.to_csv(paths["per_period"], index_label=period_col)
    with open(paths["metrics"], "w") as handle:
        json.dump(results, handle, indent=2, default=str)

    if not args.quiet:
        # Period counts live alongside the blocks, not inside them: summarize()
        # returns only the four statistics, and the coverage counts are separate
        # top-level keys because bmc and mmc cover fewer periods than corr.
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
