"""
Run one LSTM forecast: `python -m src.run_nn_forecast <tournament> [flags]`.

The sequence-model counterpart of run_forecast.py, and it keeps that module's
contract: argparse only, no config files, one invocation is exactly one run fully
determined by its arguments, and the resolved config is written into the metrics
json so any result can be reproduced from the file itself.

What is deliberately NOT here
-----------------------------
`--n-iter`, `--n-splits`, `--search-type`, `--search-jobs`, `--bayes-init-points`.
There is no hyperparameter search. The network's hyperparameters are fixed per run
(iwtsf's approach) and the budget goes into seeds instead: `--max-trials` networks
are trained on one fixed split, the best `--top-frac` are kept, and their
predictions are averaged. On a noisy correlation objective which initialisation you
drew matters as much as most architectural dials, so that trade is a deliberate one.
Passing any of the absent flags is an error that says this rather than being
silently ignored.

`--task` is also absent. The head is always a classifier over the target's bins, so
there is no regression/classification switch to make; the continuous signal comes
from the head's expectation. See prediction/nn/model.py.

Outputs land in the same four trees as the tabular pipeline, under the model name
`lstm`, so results/{forecast,per_period,hyperparams,metrics}/<tournament>/lstm/ can
be read by the existing analysis scripts with no changes.
"""

import argparse
import os
from time import time

from src.data_mani.utils import TOURNAMENTS
from src.prediction.nn.losses import LOSSES
from src.run_forecast import print_summary, write_outputs

#: Flags that exist on run_forecast.py and cannot mean anything here. Listed in the
#: help text so someone adapting a run_forecast command line is told why their flag
#: vanished, rather than just getting argparse's "unrecognized arguments".
ABSENT_FLAGS = {
    "--n-iter": "there is no hyperparameter search; use --max-trials for seeds",
    "--n-splits": "there is no inner CV; early stopping uses --val-frac",
    "--search-type": "there is no hyperparameter search",
    "--search-jobs": "there is no hyperparameter search",
    "--bayes-init-points": "there is no hyperparameter search",
    "--metric": "--metric selects a SCORING metric; use --loss for the objective",
    "--metric2": "use --loss mixture with --bmc-weight / --corr-weight",
    "--metric-weight": "use --bmc-weight / --corr-weight (independent, not convex)",
    "--task": "the head is always a classifier over the target's bins",
    "--n-jobs": "the network trains on one device; see --device",
}


def parse_args(argv=None):
    """
    Parse the command line.

    :param argv: argument list, or None to read sys.argv
    :type argv: [str] or None
    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    epilog = "flags from run_forecast.py that do not apply here:\n" + "\n".join(
        "  {:<20} {}".format(flag, why) for flag, why in sorted(ABSENT_FLAGS.items())
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.run_nn_forecast",
        description="Train and back-test the LSTM sequence model.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("tournament", type=str, choices=sorted(TOURNAMENTS),
                        help="which tournament to run")

    # ---- data ---------------------------------------------------------------
    parser.add_argument("--target", type=str, default=None,
                        help="target column; default is the tournament's own. "
                             "crypto also ships target_binned_return_60")
    parser.add_argument("--data-version", type=str, default=None,
                        help="dataset version, e.g. crypto/v2.0-tenc")
    parser.add_argument("--feature-set", type=str, default=None,
                        help="numerai only: small|medium|quantum|all")
    parser.add_argument("--feature-include", type=str, default=None,
                        help="regex of feature columns to keep")
    parser.add_argument("--feature-exclude", type=str, default=None,
                        help="regex of feature columns to drop, applied after "
                             "--feature-include")
    parser.add_argument("--era-step", type=int, default=1,
                        help="keep every Nth period. BREAKS sequence contiguity -- "
                             "smoke runs only")
    parser.add_argument("--val-frac", type=float, default=0.3,
                        help="fraction of periods held out for the walk-forward, and "
                             "the fraction of each training block used for early "
                             "stopping")

    # ---- walk-forward -------------------------------------------------------
    parser.add_argument("--refit-every", type=int, default=130,
                        help="validation periods predicted per refit. 130 matches "
                             "the live protocol in src/live/registry.py")
    parser.add_argument("--embargo", type=str, default="auto",
                        help="periods between training targets and a predicted "
                             "block; 'auto' derives it from the target horizon")
    parser.add_argument("--cv-scheme", type=str, default="expanding",
                        choices=("expanding", "rolling"),
                        help="grow the training block from the start of history, or "
                             "roll a fixed window")
    parser.add_argument("--train-window", type=int, default=None,
                        help="training periods when --cv-scheme rolling; required")
    parser.add_argument("--min-train", type=int, default=0,
                        help="skip a block whose training set is shorter than this")

    # ---- network ------------------------------------------------------------
    parser.add_argument("--hidden", type=int, default=60,
                        help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="stacked LSTM layers")
    parser.add_argument("--embed-dim", type=int, default=50,
                        help="symbol embedding width; 0 disables embeddings")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="dropout on the LSTM input and output")
    parser.add_argument("--dropout-type", type=str, default="regular",
                        choices=("regular", "variational"),
                        help="'variational' shares one mask across all timesteps")
    parser.add_argument("--no-layer-norm", action="store_true",
                        help="disable the LayerNorm after the LSTM")

    # ---- sequence -----------------------------------------------------------
    parser.add_argument("--seq-length", type=int, default=84,
                        help="window length in periods. Longer windows are only "
                             "available to symbols with long unbroken histories -- "
                             "see the coverage report")
    parser.add_argument("--overlapping", action="store_true",
                        help="step training windows by 1 period instead of by "
                             "--seq-length; many more windows, heavily correlated")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="windows per batch. The loss is pooled over the whole "
                             "batch, so this also sets the correlation's sample size")
    parser.add_argument("--stop-frac", type=float, default=None,
                        help="fraction of each WALK-FORWARD training block used for "
                             "early stopping. Defaults to --val-frac, which is what "
                             "used to size it; 0 carves no slice at all and fits on "
                             "the whole block (the slice plus its embargo costs 45%% "
                             "of a 130-period block)")
    parser.add_argument("--train-buffer", type=int, default=20,
                        help="extra periods a run needs beyond --seq-length to be "
                             "used for training; iwtsf's slack")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="do not shuffle training windows")

    # ---- optimisation -------------------------------------------------------
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Adam learning rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="maximum epochs per network")
    parser.add_argument("--patience", type=int, default=20,
                        help="epochs without improvement before stopping")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="gradient-norm cap; 0 disables clipping")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="L2 penalty. The tabular MLP that outscores this model "
                             "wins on its L2 alone, so this is not a minor dial")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=("adam", "adamw"),
                        help="'adam' couples the L2 penalty to the adaptive step "
                             "(sklearn's convention), 'adamw' decouples it")
    parser.add_argument("--warm-start", action="store_true",
                        help="carry weights between walk-forward blocks. Off by "
                             "default: it turns the back-test into one continuous "
                             "training run")

    # ---- loss ---------------------------------------------------------------
    parser.add_argument("--loss", type=str, default="corr", choices=sorted(LOSSES),
                        help="training objective")
    parser.add_argument("--bmc-weight", type=float, default=1.0,
                        help="BMC multiplier for --loss mixture")
    parser.add_argument("--corr-weight", type=float, default=0.1,
                        help="CORR multiplier for --loss mixture. Independent "
                             "multipliers, not a convex blend")
    parser.add_argument("--bmc-reference", type=str, default="meta_proxy",
                        choices=("meta_proxy", "benchmark"),
                        help="what BMC contributes over. Defaults to the proxy "
                             "because the real reference covers only part of history "
                             "and a training loss needs gradient everywhere")
    parser.add_argument("--proxy-alpha", type=float, default=1.0,
                        help="Ridge penalty for the meta proxy")

    # ---- seeds --------------------------------------------------------------
    parser.add_argument("--max-trials", type=int, default=100,
                        help="networks trained during seed selection")
    parser.add_argument("--top-frac", type=float, default=0.1,
                        help="fraction of seeds kept and ensembled")
    parser.add_argument("-S", "--seed", type=int, default=42,
                        help="meta-seed for the trial draw, so the seed set is "
                             "itself reproducible")

    # ---- run ----------------------------------------------------------------
    parser.add_argument("--device", type=str, default=None,
                        help="torch device; default auto-detects CUDA")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any seed-selection checkpoint and start over")
    parser.add_argument("--export-embeddings", action="store_true",
                        help="write each block's learned symbol embedding matrix to "
                             "models/embeddings/<run_name>.parquet, for "
                             "src.data_mani.build_symbol_embedding to turn into "
                             "features for other models")
    parser.add_argument("-n", "--run-name", type=str, default=None,
                        help="output file stem; defaults to lstm_<loss>_<scheme>")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress progress output")

    return parser.parse_args(argv)


def check_args(args):
    """
    Fail early on argument combinations that cannot work.

    :param args: parsed arguments
    :type args: argparse.Namespace
    :raises SystemExit: with an explanation
    """
    if args.cv_scheme == "rolling" and not args.train_window:
        raise SystemExit(
            "--train-window is required when --cv-scheme rolling. Units are "
            "periods: about 52 per year on numerai, 252 per year on crypto."
        )

    if args.loss == "mixture" and args.bmc_weight == 0.0 and args.corr_weight == 0.0:
        raise SystemExit(
            "--bmc-weight and --corr-weight are both 0, so the loss is constant "
            "and nothing would be learned."
        )

    if not 0.0 < args.top_frac <= 1.0:
        raise SystemExit("--top-frac must be in (0, 1]")

    if args.max_trials < 1:
        raise SystemExit("--max-trials must be at least 1")

    if args.embed_dim and args.embed_dim < 0:
        raise SystemExit("--embed-dim cannot be negative; use 0 to disable embeddings")

    # There is nothing to export from a network built without an embedding table,
    # and discovering that after the run finishes would be infuriating.
    if args.export_embeddings and not args.embed_dim:
        raise SystemExit(
            "--export-embeddings needs --embed-dim > 0; with embed_dim=0 the "
            "network has no symbol embedding to export."
        )

    # torch is optional (see setup.sh), so say what to do rather than letting an
    # ImportError surface from three modules deep.
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "this pipeline needs torch, which is an optional dependency: {}\n"
            "Install it with:  bash setup.sh   (torch is in the OPTIONAL set)".format(
                error
            )
        )

    if args.tournament == "crypto" and args.feature_set is not None:
        print("note: --feature-set is ignored for crypto, which has one fixed set "
              "of 22 features")
    if args.tournament == "numerai" and args.val_frac != 0.3:
        print("note: numerai ships its own train/validation split, so --val-frac "
              "only controls the early-stopping slice inside each training block")


def build_config(args):
    """
    Flatten the arguments into the config dict the pipeline reads.

    One dict rather than twenty parameters, because it is also what gets recorded
    in the metrics json -- so the thing that configured the run and the thing that
    documents it cannot disagree.

    :param args: parsed arguments
    :type args: argparse.Namespace
    :return: the run config
    :rtype: dict
    """
    run_name = args.run_name or "lstm_{}_{}".format(args.loss, args.cv_scheme)

    checkpoint_path = None
    if not args.no_resume:
        # Beside the outputs, keyed by run name, so two runs cannot inherit each
        # other's seed scores.
        checkpoint_path = os.path.join(
            "results", "metrics", args.tournament, "lstm",
            "{}.seeds.json".format(run_name),
        )

    embedding_path = None
    if args.export_embeddings:
        embedding_path = os.path.join(
            "models", "embeddings", "{}.parquet".format(run_name)
        )

    return {
        "run_name": run_name,
        "embedding_path": embedding_path,
        "target": args.target,
        "data_version": args.data_version,
        "feature_set": args.feature_set,
        "feature_include": args.feature_include,
        "feature_exclude": args.feature_exclude,
        "era_step": args.era_step,
        "val_frac": args.val_frac,
        "refit_every": args.refit_every,
        # "auto" is resolved inside prepare_panel, from the target being modelled.
        "embargo": None if args.embargo == "auto" else int(args.embargo),
        "cv_scheme": args.cv_scheme,
        "train_window": args.train_window,
        "min_train": args.min_train,
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "embed_dim": args.embed_dim,
        "dropout": args.dropout,
        "dropout_type": args.dropout_type,
        "layer_norm": not args.no_layer_norm,
        "seq_length": args.seq_length,
        "overlapping": args.overlapping,
        "batch_size": args.batch_size,
        "train_buffer": args.train_buffer,
        "stop_frac": args.stop_frac,
        "shuffle": not args.no_shuffle,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "warm_start": args.warm_start,
        "loss": args.loss,
        "bmc_weight": args.bmc_weight,
        "corr_weight": args.corr_weight,
        "bmc_reference": args.bmc_reference,
        "proxy_alpha": args.proxy_alpha,
        "max_trials": args.max_trials,
        "top_frac": args.top_frac,
        "seed": args.seed,
        "device": args.device,
        "checkpoint_path": checkpoint_path,
    }


def print_seed_summary(results):
    """
    Report the seed phase, including how much spread there was.

    The spread is the interesting number: if the best and median seeds score nearly
    the same then the ensemble is buying little and --max-trials can come down; if
    they differ a lot, run-to-run variance is large and any single-seed comparison
    against another model is not meaningful.

    :param results: the results dict from nn_forecast()
    :type results: dict
    """
    selection = results.get("seed_selection")
    if not selection:
        return

    scores = sorted(selection["val_scores"].values())
    if not scores:
        return

    middle = scores[len(scores) // 2]
    print("\nseed selection: {} trials, kept {} ({:.0%})".format(
        selection["n_trials"], len(selection["selected"]), selection["top_frac"]))
    print("  val loss  best {:+.4f} | median {:+.4f} | worst {:+.4f}".format(
        scores[0], middle, scores[-1]))
    print("  selected  {}".format(selection["selected"]))

    coverage = selection.get("train_coverage") or {}
    if coverage:
        print("  windows   {} from {} runs ({} too short), {} of {} symbols".format(
            coverage.get("n_windows"), coverage.get("n_runs"),
            coverage.get("n_runs_too_short"),
            coverage.get("n_symbols_contributing"), coverage.get("n_symbols_seen")))


def main(argv=None):
    """
    Entry point.

    :param argv: argument list, or None to read sys.argv
    :type argv: [str] or None
    :return: process exit code
    :rtype: int
    """
    args = parse_args(argv)
    check_args(args)

    # Imported here, after check_args has verified torch is installed, so a machine
    # without it gets the actionable message instead of an ImportError traceback.
    from src.prediction.nn.pipeline import nn_forecast

    config = build_config(args)
    verbose = not args.quiet

    if verbose:
        print("=" * 62)
        print("LSTM forecast: {} | loss {} | seq {} | {} seeds -> top {:.0%}".format(
            args.tournament, args.loss, args.seq_length, args.max_trials,
            args.top_frac))
        print("=" * 62)

    started = time()
    predictions, periods_table, results = nn_forecast(
        args.tournament, config, verbose=verbose
    )
    results["runtime_minutes"] = round((time() - started) / 60.0, 2)

    paths = write_outputs(
        predictions, periods_table, results, args.tournament, "lstm",
        config["run_name"],
    )

    if verbose:
        print_summary(results)
        print_seed_summary(results)
        print("\nwrote:")
        for kind, path in sorted(paths.items()):
            print("  {:<12} {}".format(kind, path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
