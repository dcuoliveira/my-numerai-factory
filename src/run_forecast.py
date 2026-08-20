"""
Run one Numerai / Crypto model evaluation.

One invocation is exactly one run, fully determined by the arguments: a single
tournament, model, task, metric and CV setup. There is no looping over
tournaments or models inside the runner -- comparing setups means invoking it
again with different arguments.

    python -m src.run_forecast crypto lgb
    python -m src.run_forecast crypto ridge_glmnet --task classification
    python -m src.run_forecast numerai lgb --metric bmc_sharpe --era-step 4
    python -m src.run_forecast crypto ridge_glmnet --search-type bayes

Outputs land in two files, mirroring the results tree of the pipeline this is
adapted from:

    results/forecast/<tournament>/<model>/<run_name>.csv    per-row predictions
    results/metrics/<tournament>/<model>/<run_name>.json    summary + config
"""

import argparse
import json
import os
from time import time

import numpy as np
import pandas as pd

from src.data_mani.utils import (
    TOURNAMENTS,
    default_embargo,
    default_refit_every,
    get_spec,
)
from src.prediction import search_spaces
from src.prediction.functions import METRICS, forecast
from src.prediction.models import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one model on one Numerai tournament.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("tournament", type=str, choices=sorted(TOURNAMENTS),
                        help="which tournament to run")
    parser.add_argument("model", type=str, choices=sorted(MODELS),
                        help="which model to evaluate")

    parser.add_argument("-t", "--task", type=str, default="regression",
                        choices=["regression", "classification"],
                        help="regression predicts the binned target directly; "
                             "classification binarises it at 0.5 and uses the "
                             "positive-class probability as the signal")
    parser.add_argument("-m", "--metric", type=str, default="corr",
                        choices=list(METRICS),
                        help="objective the hyperparameter search maximises. All "
                             "metrics are reported regardless. Note that the "
                             "drawdown objectives are degenerate on their own -- "
                             "a weak, near-flat signal has little to give back -- "
                             "so use them to compare against a mean- or "
                             "sharpe-selected run, not as a default")
    parser.add_argument("--metric2", type=str, default=None,
                        choices=list(METRICS),
                        help="optional second objective. The search then "
                             "maximises  weight*metric + (1-weight)*metric2")
    parser.add_argument("--metric-weight", type=float, default=0.5,
                        help="weight on the first metric when --metric2 is set. "
                             "NOTE the two components are not rescaled, and "
                             "their natural magnitudes differ a lot (corr ~0.015 "
                             "on numerai vs sharpe ~1.0), so a nominal 50/50 "
                             "split can be 99%% one metric in practice. Blending "
                             "within a family (corr+bmc) behaves far better than "
                             "across families (corr+sharpe)")
    parser.add_argument("--bmc-reference", type=str, default="benchmark",
                        choices=["benchmark", "meta_proxy"],
                        help="what BMC scores against. 'benchmark' is the "
                             "official series (v53_lgbm_ender20 on numerai, the "
                             "published meta model on crypto), which covers only "
                             "part of history. 'meta_proxy' is a Ridge fit to the "
                             "meta model's per-period rank, predicted everywhere, "
                             "so BMC gets full coverage")
    parser.add_argument("--proxy-alpha", type=float, default=1.0,
                        help="Ridge penalty for the meta-model proxy")

    # Search
    parser.add_argument("-i", "--n-iter", type=int, default=50,
                        help="hyperparameter draws per refit")
    parser.add_argument("-s", "--n-splits", type=int, default=5,
                        help="inner cross-validation folds")
    parser.add_argument("-j", "--n-jobs", type=int, default=-1,
                        help="threads given to the model itself")
    parser.add_argument("--fixed-params", type=str, default=None,
                        help="JSON dict of estimator parameters to PIN, e.g. "
                             "'{\"n_estimators\": 4327, \"max_depth\": 8}'. The "
                             "hyperparameter search is then a single candidate: "
                             "nothing is selected, --n-iter is forced to 1, and "
                             "the wrapper's fit_grid() hook is bypassed so a "
                             "pinned min_child_samples is not silently rewritten "
                             "from the block size. Use it to vary one thing (a "
                             "cadence, a feature set) with the model held fixed")
    parser.add_argument("--search-type", type=str, default="random",
                        choices=["random", "bayes"],
                        help="how the hyperparameter space is explored. 'random' "
                             "draws independently; 'bayes' fits a surrogate to "
                             "the scores so far and proposes where to look next "
                             "(scikit-optimize). NOTE the two do not search the "
                             "same space -- bayes turns each explicit grid into "
                             "a continuous interval -- so a difference between "
                             "them is a difference between strategies, not a "
                             "cleaner optimum over one grid. It also splits the "
                             "budget across sub-grids: the boosters have six, "
                             "one per max_depth, so --n-iter 50 gives each "
                             "surrogate only ~8 draws. Raise --n-iter to 150+ "
                             "for boosters, or use it on ridge / ridge_glmnet, "
                             "whose grids are single dicts")
    parser.add_argument("--bayes-init-points", type=int, default=0,
                        help="random warm-up draws per sub-grid before the "
                             "surrogate is first fitted. 0 auto-scales it to the "
                             "budget, which is the point -- skopt's own default "
                             "of 10 would silently consume the entire per-grid "
                             "budget of a booster run and reduce it to random "
                             "search. Ignored unless --search-type bayes")
    parser.add_argument("--search-jobs", type=int, default=1,
                        help="processes evaluating hyperparameter candidates in "
                             "parallel. Keep at 1 for numerai with a large "
                             "feature set, where each worker forks a full copy "
                             "of the data. Raise it for crypto, where the data "
                             "is small and this is roughly 4x faster")
    parser.add_argument("-S", "--seed", type=int, default=42,
                        help="random seed, applied to both search and estimator")

    # Walk-forward
    parser.add_argument("--refit-every", type=str, default="month",
                        help="how often to refit, in PERIODS -- or a cadence in "
                             "months such as 'month', '3month', '0.5month', "
                             "which is converted using the tournament's period "
                             "length. A month is 22 dates on crypto but 4 eras "
                             "on numerai, so the month form is portable where a "
                             "raw number is not")
    parser.add_argument("--embargo", type=str, default="auto",
                        help="periods discarded between train and validation. "
                             "'auto' derives it from the target horizon: a 20-day "
                             "target needs 4 periods on numerai (weekly eras) but "
                             "20 on crypto (daily dates)")
    parser.add_argument("--min-train", type=str, default="0",
                        help="burn-in floor: minimum training PERIODS (or a "
                             "months form like '12month') before the "
                             "walk-forward will predict a block, and before an "
                             "inner CV fold counts. Blocks below it are skipped "
                             "and never scored. Default 0 means no floor, in "
                             "which case how much history the first block gets "
                             "is just a consequence of --val-frac")
    parser.add_argument("--cv-scheme", type=str, default="expanding",
                        choices=["expanding", "rolling"],
                        help="expanding grows training from the start of history; "
                             "rolling keeps a fixed window and drops old periods")
    parser.add_argument("--train-window", type=str, default=None,
                        help="history length for --cv-scheme rolling, in PERIODS "
                             "or a months form like '12month'. Required when "
                             "rolling. This is the dial that varies how much data "
                             "the model actually trains on -- unlike --min-train, "
                             "which only drops blocks that fall short")

    # Data
    parser.add_argument("--val-frac", type=float, default=0.3,
                        help="fraction of periods held out for validation. "
                             "crypto only -- numerai ships its own split")
    parser.add_argument("--era-step", type=int, default=1,
                        help="keep every Nth period. Use 4 for a fast numerai "
                             "scout run")
    parser.add_argument("--feature-set", type=str, default=None,
                        help="numerai only: small | medium | quantum | all")
    parser.add_argument("--data-version", type=str, default=None,
                        help="dataset version, defaults to the tournament's")
    parser.add_argument("--target", type=str, default=None,
                        help="target column; default is the tournament's. Crypto "
                             "also ships target_binned_return_60, whose horizon "
                             "sets the embargo automatically (60 periods, not 20)")
    parser.add_argument("--feature-include", type=str, default=None,
                        help="regex: keep only feature columns matching it. "
                             "Applied before --feature-exclude. Used for "
                             "block-alone ablations, e.g. '_rm[0-9]+$'")
    parser.add_argument("--feature-exclude", type=str, default=None,
                        help="regex: drop feature columns matching it. Used for "
                             "leave-one-out ablations, e.g. '^feature_sym_' to "
                             "remove the ticker one-hot")

    parser.add_argument("-n", "--run-name", type=str, default=None,
                        help="output file stem, defaults to <task>_<metric>_<scheme>")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress progress output")

    return parser.parse_args()


def resolve_periods(args, value):
    """
    Turn a period count or a months form ("12month", "0.5month") into an int.

    Shared by --refit-every and --min-train, because both are durations and both
    hit the same portability trap: the same calendar span is 22 periods on
    crypto and 4 on numerai.
    """
    value = str(value).strip().lower()
    if value.endswith("month"):
        months = value[:-len("month")]
        return default_refit_every(args.tournament, float(months) if months else 1.0)
    return int(value)


def resolve_refit_every(args):
    """
    Turn --refit-every into a period count.

    Accepts a plain integer, or a months form like "month" / "3month" / "0.5month".
    The months form is the portable one: the same calendar cadence is 22 periods
    on crypto and 4 on numerai, so a raw number silently means something
    different in each tournament -- the same trap --embargo avoids.
    """
    value = str(args.refit_every).strip().lower()
    if value.endswith("month"):
        months = value[:-len("month")]
        months = float(months) if months else 1.0
        return default_refit_every(args.tournament, months)
    return int(value)


def resolve_embargo(args):
    """
    Turn --embargo into an int, deriving it from the target horizon if 'auto'.

    Derived from the target ACTUALLY BEING MODELLED, not the tournament default.
    Crypto ships both a 20-day and a 60-day label; the 60-day one overlaps 98%
    between consecutive daily periods against 95% for the 20-day one, so reading
    the default here would embargo 20 periods where 60 are needed and leak three
    times over -- silently, and with a better-looking score as the only symptom.
    """
    if args.embargo == "auto":
        spec = get_spec(args.tournament)
        return default_embargo(args.tournament, args.target or spec["target"])
    return int(args.embargo)


def check_args(args):
    """Fail early on argument combinations that cannot work."""
    # Rolling with no window would otherwise silently behave like expanding.
    if args.cv_scheme == "rolling" and not args.train_window:
        raise SystemExit(
            "--train-window is required when --cv-scheme rolling. Units are "
            "periods: about 52 per year on numerai, 252 per year on crypto."
        )

    # Two layers of parallelism multiply. Leaving both wide open spawns
    # search_jobs processes that each grab every core, which thrashes badly
    # enough to look like a hang.
    if args.search_jobs > 1 and args.n_jobs in (-1, 0):
        raise SystemExit(
            "--search-jobs {} with --n-jobs {} oversubscribes the machine: each "
            "of the {} search processes would also take every core. Set --n-jobs "
            "to about (cores / search-jobs), e.g. --search-jobs 4 --n-jobs {}.".format(
                args.search_jobs, args.n_jobs, args.search_jobs,
                max(1, (os.cpu_count() or 4) // args.search_jobs),
            )
        )

    # A missing optional dependency must fail before the data load, not after
    # it. How the budget then divides across sub-grids is reported by the
    # walk-forward itself, which knows the real grid.
    if args.search_type == "bayes":
        try:
            import skopt                                        # noqa: F401
        except ImportError:
            raise SystemExit(search_spaces.INSTALL_HINT)
    elif args.bayes_init_points:
        print("note: --bayes-init-points is ignored without --search-type bayes")

    # Warn rather than pretend these did something.
    if args.tournament == "numerai" and args.val_frac != 0.3:
        print("note: --val-frac is ignored for numerai, which ships its own "
              "train/validation split")
    if args.tournament == "crypto" and args.feature_set is not None:
        print("note: --feature-set is ignored for crypto, which has one fixed "
              "set of 22 features")


def hyperparams_table(results):
    """
    Flatten the per-refit records into one row per walk-forward step.

    This is the hyperparameter history: what the search picked, when it applied,
    what it was picked from, and how it scored. One row per refit, so a report
    can plot how a chosen parameter drifts as the training window grows.

    That drift is the interesting part for ridge_glmnet, whose alpha path is
    rebuilt from each refit's training block -- the selected alpha moves both
    because the data changed and because the path itself moved. The
    `<param>_grid_min` / `_grid_max` columns are what let you tell those apart:
    an alpha pinned to the edge of its path means the path, not the data, was
    the binding constraint.

    :param results: the results dict from forecast()
    :type results: dict
    :return: one row per refit
    :rtype: pd.DataFrame
    """
    rows = []
    for refit in results.get("refits", []):
        row = {k: v for k, v in refit.items()
               if k not in ("best_params", "searched_bounds")}

        # The chosen values, one column each.
        for name, value in refit.get("best_params", {}).items():
            row[name] = value

        # And the span each was chosen from, where it was an explicit list.
        # Under --search-type bayes a *_grid_size of 0 is not missing data: it
        # marks a continuous dimension, which has no countable values. That is
        # the one place the two search types read differently here.
        for name, bounds in refit.get("searched_bounds", {}).items():
            row["{}_grid_min".format(name)] = bounds["min"]
            row["{}_grid_max".format(name)] = bounds["max"]
            row["{}_grid_size".format(name)] = bounds["size"]

        rows.append(row)

    return pd.DataFrame(rows)


def write_outputs(predictions, periods_table, results, tournament, model, run_name):
    """
    Write the three output files a report is built from.

        forecast/    per-row out-of-sample predictions. Joining these across
                     runs on (id, period) is what gives the cross-model
                     prediction correlation matrix.
        per_period/  one row per period: corr, bmc, mmc, pnl. Cumulative curves
                     are .cumsum() on these.
        hyperparams/ one row per walk-forward step: the parameters the search
                     chose, when they applied, and the grid they came from.
        metrics/     summaries, CV train/validation scores, and the full config.

    makedirs with exist_ok, not mkdir: the parent chain may not exist, and
    parallel runs would otherwise race each other creating it.
    """
    paths = {}
    for kind in ("forecast", "per_period", "hyperparams", "metrics"):
        directory = os.path.join("results", kind, tournament, model)
        os.makedirs(directory, exist_ok=True)
        suffix = "json" if kind == "metrics" else "csv"
        paths[kind] = os.path.join(directory, "{}.{}".format(run_name, suffix))

    predictions.to_csv(paths["forecast"], index=False)
    periods_table.to_csv(paths["per_period"])
    hyperparams_table(results).to_csv(paths["hyperparams"], index=False)
    with open(paths["metrics"], "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)

    return paths


def print_summary(results):
    """Print the headline numbers in the same shape as the example notebooks."""
    print("\n" + "=" * 62)
    for family in ("corr", "bmc", "mmc"):
        summary = results.get(family)
        if not summary:
            print("{:<5} not computed (series unavailable)".format(family.upper()))
            continue
        # MMC is usually scored over far fewer periods than CORR, so print the
        # count on the same line rather than leaving the mean to stand alone.
        n = results.get("{}_n_periods".format(family), results.get("n_periods"))
        print("{:<5} mean {:>10.6f}   std {:>9.6f}   sharpe {:>8.4f}   "
              "max_dd {:>8.5f}   n={}".format(
                  family.upper(), summary["mean"], summary["std"],
                  summary["sharpe"], summary["max_drawdown"], n))

    pnl = results.get("pnl")
    if pnl:
        print("{:<5} mean {:>10.6f}   std {:>9.6f}   sharpe {:>8.4f}   "
              "max_dd {:>8.5f}".format(
                  "PNL", pnl["mean"], pnl["std"], pnl["sharpe"],
                  pnl["max_drawdown"]))

    reference = results.get("reference", {})
    meta = results.get("meta_model", {})
    print("bmc reference : {} ({}), {:.1%} row coverage".format(
        reference.get("column"), reference.get("kind"),
        reference.get("row_coverage", 0.0)))
    print("mmc meta model: {}, {:.1%} row coverage".format(
        meta.get("column"), meta.get("row_coverage", 0.0)))
    print("=" * 62)


def main():
    args = parse_args()
    check_args(args)

    verbose = not args.quiet
    embargo = resolve_embargo(args)
    refit_every = resolve_refit_every(args)
    min_train = resolve_periods(args, args.min_train)
    train_window = resolve_periods(args, args.train_window) if args.train_window else None

    run_name = args.run_name or "{}_{}_{}".format(
        args.task, args.metric, args.cv_scheme
    )

    started = time()
    fixed_params = None
    if args.fixed_params:
        try:
            fixed_params = json.loads(args.fixed_params)
        except ValueError as exc:
            raise SystemExit("--fixed-params is not valid JSON: %s" % exc)
        if not isinstance(fixed_params, dict) or not fixed_params:
            raise SystemExit("--fixed-params must be a non-empty JSON object")
        print("pinned %d parameter(s), search disabled: %s" % (
            len(fixed_params), ", ".join(sorted(fixed_params))))

    predictions, periods_table, results = forecast(
        tournament=args.tournament,
        model_name=args.model,
        task=args.task,
        metric=args.metric,
        data_version=args.data_version,
        feature_set=args.feature_set,
        target=args.target,
        feature_include=args.feature_include,
        feature_exclude=args.feature_exclude,
        era_step=args.era_step,
        val_frac=args.val_frac,
        refit_every=refit_every,
        embargo=embargo,
        scheme=args.cv_scheme,
        train_window=train_window,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        seed=args.seed,
        verbose=verbose,
        search_jobs=args.search_jobs,
        search_type=args.search_type,
        bayes_init_points=args.bayes_init_points,
        fixed_params=fixed_params,
        metric2=args.metric2,
        metric_weight=args.metric_weight,
        bmc_reference=args.bmc_reference,
        proxy_alpha=args.proxy_alpha,
        min_train=min_train,
    )

    elapsed = (time() - started) / 60.0
    results["runtime_minutes"] = round(elapsed, 3)

    paths = write_outputs(
        predictions, periods_table, results, args.tournament, args.model, run_name
    )

    if verbose:
        print_summary(results)
        for kind, path in paths.items():
            print("{:<11} -> {}".format(kind, path))
        print("total run time = {} min".format(np.round(elapsed, 2)))


if __name__ == "__main__":
    main()
