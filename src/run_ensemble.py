"""
Combine several completed runs into an ensemble and score it like any other run.

    python3 -m src.run_ensemble crypto ridge reg_corr_win3m reg_corr_win24m reg_corr_3month \
        --run-name ens_3m_24m_exp

Members must already have been run -- this reads their forecast CSVs rather than
refitting anything, so an ensemble is cheap no matter how expensive its members
were.

How the combination works: predictions are ranked to [0, 1] *within each period*
and then averaged. Averaging raw predictions would be wrong here, because the
members are not on a common scale -- a regression on the binned target and a
classifier's predict_proba live in different ranges, and even two ridges with
different penalties have different spreads. Numerai scores by rank anyway, so
ranking first is both the correct normalisation and a no-op with respect to how
any single member would have been scored.

Outputs land in the same tree and the same four files as src.run_forecast, so an
ensemble is directly comparable with its members.
"""

import argparse
import json
import os
from time import time

import pandas as pd

from src.data_mani.utils import META_MODEL_COL, TOURNAMENTS, get_spec
from src.prediction.metrics import per_period_table, summarize, summarize_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble several completed runs and score the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tournament", choices=sorted(TOURNAMENTS))
    parser.add_argument("model", help="model directory the member runs live in")
    parser.add_argument("members", nargs="+", help="run names to combine")
    parser.add_argument("-w", "--weights", type=str, default=None,
                        help="comma-separated weights, one per member. Defaults "
                             "to equal weight. Normalised to sum to 1")
    parser.add_argument("--target", type=str, default=None,
                        help="target column carried by the member forecasts. "
                             "Defaults to the tournament's, which is wrong for "
                             "members trained on a different label: a 60-day "
                             "run's forecast file carries "
                             "target_binned_return_60 and nothing else, so the "
                             "default lookup raises KeyError")
    parser.add_argument("-n", "--run-name", type=str, default=None,
                        help="output stem, defaults to ens_<member count>")
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def load_members(tournament, model, members, period_col, id_col, target_col):
    """
    Load each member's predictions and align them on (id, period).

    An inner join is deliberate: if the members scored different windows, only
    the overlap is defensible to ensemble, and the row count tells you at once
    whether they were run on the same footing.
    """
    frames, extras = [], None
    for name in members:
        # A member may name its own directory as "model/run". Ensembles across
        # wrappers are the interesting case -- a fixed-alpha ridge and a
        # glmnet-path ridge write into separate trees, and the whole point of
        # blending them is that they are different estimators.
        if "/" in name:
            member_model, run = name.split("/", 1)
        else:
            member_model, run = model, name
        path = os.path.join("results", "forecast", tournament, member_model, "%s.csv" % run)
        if not os.path.exists(path):
            raise SystemExit("member run not found: %s" % path)

        member = pd.read_csv(path, parse_dates=[period_col] if period_col == "date" else None)
        member = member.set_index([id_col, period_col])

        # Rank within period, so members on different scales combine sensibly.
        ranked = member.groupby(level=1)["prediction"].rank(pct=True)
        frames.append(ranked.rename(name))

        # Target and reference columns are identical across members; keep one copy.
        if extras is None:
            keep = [c for c in (target_col, META_MODEL_COL, "reference_prediction")
                    if c in member.columns]
            extras = member[keep]

    joined = pd.concat(frames, axis=1, join="inner")
    return joined, extras.loc[joined.index]


def main():
    args = parse_args()
    verbose = not args.quiet
    spec = get_spec(args.tournament)
    period_col, id_col = spec["period_col"], spec["id_col"]
    target_col = args.target or spec["target"]

    weights = [1.0] * len(args.members)
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(args.members):
            raise SystemExit("got %d weights for %d members" % (len(weights), len(args.members)))
    total = sum(weights)
    weights = [w / total for w in weights]

    started = time()
    ranked, extras = load_members(
        args.tournament, args.model, args.members, period_col, id_col, target_col
    )

    if verbose:
        for name, w in zip(args.members, weights):
            print("  %-22s weight %.3f" % (name, w))
        print("aligned on %d rows, %d periods" % (
            len(ranked), ranked.index.get_level_values(1).nunique()))

    # Weighted average of the per-period ranks.
    blend = sum(ranked[name] * w for name, w in zip(args.members, weights))

    frame = extras.copy()
    frame["prediction"] = blend
    frame = frame.reset_index()

    reference_col = "reference_prediction" if "reference_prediction" in frame.columns else None
    meta_col = META_MODEL_COL if META_MODEL_COL in frame.columns else None

    results = summarize_predictions(
        frame, "prediction", target_col, period_col,
        reference_col=reference_col, meta_model_col=meta_col,
    )
    table = per_period_table(
        frame, "prediction", target_col, period_col,
        reference_col=reference_col, meta_model_col=meta_col,
    )
    results["pnl"] = summarize(table["pnl"])
    results["config"] = {
        "tournament": args.tournament, "model": args.model,
        # The forecast header is the primary record of the scored label, but a
        # reclaimed forecast takes it with it -- so write the target here too.
        "target": target_col,
        "members": args.members, "weights": weights,
        "combination": "per-period rank average",
    }
    results["runtime_minutes"] = round((time() - started) / 60.0, 3)

    run_name = args.run_name or "ens_%d" % len(args.members)
    paths = {}
    for kind, obj in (("forecast", frame), ("per_period", table), ("metrics", results)):
        directory = os.path.join("results", kind, args.tournament, args.model)
        os.makedirs(directory, exist_ok=True)
        paths[kind] = os.path.join(
            directory, "%s.%s" % (run_name, "json" if kind == "metrics" else "csv"))

    frame.to_csv(paths["forecast"], index=False)
    table.to_csv(paths["per_period"])
    with open(paths["metrics"], "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)

    if verbose:
        for fam in ("corr", "bmc", "mmc", "pnl"):
            s = results.get(fam)
            if s:
                print("%-5s mean %10.6f  sharpe %8.4f  max_dd %8.5f" % (
                    fam.upper(), s["mean"], s["sharpe"], s["max_drawdown"]))
        for kind, path in paths.items():
            print("%-11s -> %s" % (kind, path))


if __name__ == "__main__":
    main()
