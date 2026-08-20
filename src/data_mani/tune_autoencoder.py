"""
Tune the autoencoder's hyperparameters, then hand the winner to the builder.

    python3 -m src.data_mani.tune_autoencoder --n-trials 24
    python3 -m src.data_mani.build_autoencoder --hidden 128,64 --dropout 0.05 ...

This does not write a dataset. It searches, reports, and prints the exact
build_autoencoder command that reproduces its winner -- the same separation the
rest of the repo keeps between deciding something and shipping it.

------------------------------------------------------------------------------
What is NOT searched, and why that matters more than what is
------------------------------------------------------------------------------

The objective here is held-out RECONSTRUCTION. Two of the builder's knobs are
monotone in that objective, so putting them in the search would not tune them --
it would just read the monotonicity back out and call it a result:

  --code-dim    a wider bottleneck always reconstructs better, all the way to
                code_dim = n_features where the answer is the identity. Searching
                it returns the largest value on offer, every time, and says
                nothing about whether the extra columns help a forecast. The
                runner sweeps k as separate ARMS (ae_k4 / ae_k16) scored on the
                thing that actually matters, which is the only honest way to
                choose it.

  --swap-noise  denoising deliberately TRADES reconstruction for a code that does
                not lean on any single input. The network is trained on corrupted
                rows and scored on clean ones, so more noise is strictly worse on
                this objective by construction. Searching it drives it to zero
                and quietly deletes the regularisation. The ae_nonoise arm
                settles it downstream instead.

So both are held FIXED at whatever the build will use, and the search covers the
four knobs that have a real interior optimum on held-out data: the layer widths,
dropout, the learning rate, and the batch size.

This is the same discipline as `--mode global` and `--fit-scope all` elsewhere in
data_mani: name the thing the metric cannot see, rather than letting the metric
answer a question it is not fit to answer.

------------------------------------------------------------------------------
And what reconstruction R^2 does not tell you
------------------------------------------------------------------------------

Fidelity is not usefulness. A code that reconstructs the 22 inputs perfectly may
carry nothing a ridge can use. This search finds the best autoencoder AS AN
AUTOENCODER; whether that one helps a forecast is settled by
experiments/autoencoder_crypto.sh on a paired per-period delta, and a tuned arm
that still loses to pca_plus_raw would be a real answer, not a failed search.

What the search did settle is narrower and worth recording. At the first-guess
architecture the autoencoder reconstructed WORSE than the linear PCA control
(0.904 against 0.932), which reads as "this panel is linear, drop torch". Twenty
trials later the same objective on the same slice gives 0.954. The linearity
reading was an artefact of a network four times too narrow.

------------------------------------------------------------------------------
Where the trials are allowed to look
------------------------------------------------------------------------------

Everything happens inside the pipeline's TRAIN block -- the periods before
`fit_boundary`, exactly as build_autoencoder defines it. Nothing here ever sees
the validation block a run will be scored on, so selecting on these numbers
cannot inflate a backtest.

Inside that window, three disjoint chronological slices:

    [--------- inner fit ---------][ early stop ][ tune score ]

  inner fit    what each trial trains on
  early stop   what each trial's early stopping watches, so a bad learning rate
               is not rewarded for stopping late
  tune score   what the trial is RANKED on -- untouched by both of the above

The final build then refits on the whole train block with its own two-way split,
which is the ordinary refit-after-selection move. The hyperparameters were chosen
on rows that final fit trains on; what they were never chosen on is the
validation block, and that is the property that has to hold.

------------------------------------------------------------------------------
Cost
------------------------------------------------------------------------------

Measured, wall-clock here is GRADIENT STEPS, not rows -- the per-step cost barely
moves with the slice size, because indexing and Python overhead dominate a
network this small. So subsampling rows would buy no speed at all, and every
trial trains on the whole inner-fit slice. The budget is a step count.

What the per-step cost DOES move with is the architecture the trial draws:
about 0.03s at 32-16 and roughly five times that at 256-128, which is a 40k
parameter network rather than a 4k one. So a step budget buys a predictable
amount of OPTIMISATION and an unpredictable amount of TIME. Budget generously
and expect the total to land toward the upper end of the range printed at
startup.

Equal steps rather than equal epochs, because "epochs" means different amounts of
optimisation at different batch sizes, and batch size is one of the things being
searched -- a budget stated in epochs would hand small batches four times the
compute and then report that small batches win. Patience scales with each trial's
epoch count for the same reason.

What this protocol does have is a convergence bias: within a fixed budget it
mildly favours configurations that get there fast, which overlaps with what a
high learning rate buys. Two things blunt it. Every trial reports whether it ran
out of steps or out of patience, and the tuner refuses to let a budget-bound
winner pass silently. And the final build then trains the winner to actual
convergence -- early stopping against a 600-epoch ceiling -- so the number you
end up comparing is not a budget-limited one.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

from src.data_mani.build_autoencoder import (
    align_features,
    column_mean_r2,
    fit_boundary,
    load_backend,
)
from src.data_mani.utils import (
    default_embargo,
    download_if_missing,
    fill_missing_features,
    get_api,
    get_spec,
    local_path,
)

#: Where a study's full history lands, next to the other result trees.
RESULTS_DIR = os.path.join("results", "tuning", "crypto")

#: Layer widths on offer. The decoder mirrors whichever is chosen.
HIDDEN_CHOICES = ["32,16", "64,32", "128,64", "256,128", "128,64,32"]

#: Rows per gradient step. Small batches take more steps per epoch, so this
#: interacts with the epoch cap -- which is exactly why it is searched rather
#: than assumed.
BATCH_CHOICES = [1024, 2048, 4096, 8192]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Search autoencoder hyperparameters on held-out reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-version", default="crypto/v2.0")

    # ---- held fixed, not searched (see the module docstring) ---- #
    parser.add_argument("--code-dim", type=int, default=8,
                        help="held FIXED: monotone in the objective, so the "
                             "runner sweeps it as arms instead")
    parser.add_argument("--swap-noise", type=float, default=0.15,
                        help="held FIXED: denoising trades reconstruction away "
                             "by construction, so searching it drives it to 0")

    # ---- the window the trials may look at ---- #
    parser.add_argument("--val-frac", type=float, default=0.3,
                        help="must match the --val-frac the runs will use")
    parser.add_argument("--embargo", default="auto",
                        help="periods between train and validation, or auto")
    parser.add_argument("--stop-frac", type=float, default=0.15,
                        help="early-stopping slice, as a fraction of the fit window")
    parser.add_argument("--tune-frac", type=float, default=0.15,
                        help="ranking slice, held out from both of the above")

    # ---- the budget ---- #
    parser.add_argument("--n-trials", type=int, default=24,
                        help="TPE needs a handful of random starts before it "
                             "is doing anything but sampling")
    parser.add_argument("--timeout", type=float, default=None,
                        help="seconds; stop starting new trials after this")
    parser.add_argument("--tune-steps", type=int, default=5000,
                        help="gradient steps per trial, equalised across batch "
                             "sizes; 0.03-0.15s each depending on the width the "
                             "trial draws, so budget generously")
    parser.add_argument("--patience-frac", type=float, default=0.15,
                        help="early-stopping patience as a fraction of a "
                             "trial's own epoch budget, so batch size does not "
                             "change how forgiving it is")

    # ---- run ---- #
    parser.add_argument("--study-name", default="autoencoder",
                        help="names the json under results/tuning/crypto/")
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds the sampler, the row sample and every trial")
    parser.add_argument("--device", default=None,
                        help="torch device, or auto-detect")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="hide the per-trial lines")
    return parser.parse_args(argv)


def load_optuna():
    """
    Import optuna, failing with the fix rather than a traceback.

    Optional for the same reason torch is: nothing in the build or the live path
    needs it, and a tuning run is something you do once.

    :return: the optuna module
    :rtype: module
    """
    try:
        import optuna
    except ImportError as error:
        raise SystemExit(
            "this needs optuna, which is an optional dependency (%s).\n"
            "Install it with:  pip install optuna" % error)
    return optuna


def three_way_split(periods, stop_frac, tune_frac):
    """
    Cut the fit window into inner-fit, early-stop and tune-score slices.

    Chronological and disjoint. The ranking slice is last, so a configuration is
    judged on the most recent data available to it -- while still sitting
    entirely inside the pipeline's train block, which is what keeps this from
    touching the thing a run is scored on.

    :param periods: the fit window's periods, in order
    :type periods: np.ndarray
    :param stop_frac: early-stopping slice, as a fraction of the window
    :type stop_frac: float
    :param tune_frac: ranking slice, as a fraction of the window
    :type tune_frac: float
    :return: (inner_periods, stop_periods, tune_periods)
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    :raises ValueError: if nothing would be left to train on
    """
    total = len(periods)
    n_tune = int(round(total * tune_frac))
    n_stop = int(round(total * stop_frac))

    if n_tune < 1 or n_stop < 1:
        raise ValueError(
            "--tune-frac and --stop-frac must each leave at least one period "
            "of the {} in the fit window".format(total))
    if total - n_tune - n_stop < 2:
        raise ValueError(
            "nothing left to train on after --stop-frac={} and --tune-frac={}"
            .format(stop_frac, tune_frac))

    tune = periods[total - n_tune:]
    stop = periods[total - n_tune - n_stop:total - n_tune]
    inner = periods[:total - n_tune - n_stop]
    return inner, stop, tune


def suggest_config(trial, args, n_fit_rows):
    """
    One point in the search space, as a flat config the backend can consume.

    code_dim and swap_noise come from args, not from the trial: both are monotone
    in held-out reconstruction, so searching them would read the monotonicity
    back out rather than tune anything. See the module docstring.

    The epoch count is DERIVED, not searched -- it is whatever spends the trial's
    step budget at the batch size the trial drew, so every trial gets the same
    optimisation and roughly the same wall clock.

    :param trial: the optuna trial
    :param n_fit_rows: rows in the inner-fit slice, to convert steps to epochs
    :type n_fit_rows: int
    :return: the config dict train_autoencoder expects
    :rtype: dict
    """
    from src.prediction.nn.autoencoder import steps_per_epoch

    hidden = trial.suggest_categorical("hidden", HIDDEN_CHOICES)
    batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)

    epochs = max(1, int(round(
        args.tune_steps / steps_per_epoch(n_fit_rows, batch_size))))

    return {
        "code_dim": args.code_dim,
        "swap_noise": args.swap_noise,
        "hidden": tuple(int(h) for h in hidden.split(",")),
        "dropout": trial.suggest_float("dropout", 0.0, 0.30, step=0.05),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": max(5, int(round(args.patience_frac * epochs))),
        "max_grad_norm": 1.0,
        "seed": args.seed,
    }


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.quiet
    optuna = load_optuna()
    backend = load_backend("ae")
    spec = get_spec("crypto")
    api = get_api("crypto")

    # ---- source, identical to the builder's ------------------------------- #
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    feature_columns = [c for c in df.columns if c.startswith("feature_")]
    df = fill_missing_features(df, feature_columns, spec["neutral_value"], verbose)
    values = align_features(df, feature_columns)

    embargo = (default_embargo("crypto", spec["target"])
               if args.embargo == "auto" else int(args.embargo))
    periods = np.sort(df["date"].unique())
    boundary = fit_boundary(periods, args.val_frac, embargo, "train")
    fit_periods = periods[:boundary]

    inner_periods, stop_periods, tune_periods = three_way_split(
        fit_periods, args.stop_frac, args.tune_frac)

    def slice_of(chosen):
        return values[df["date"].isin(chosen).to_numpy()]

    inner = slice_of(inner_periods)
    stop = slice_of(stop_periods)
    tune = slice_of(tune_periods)

    # The baseline every R^2 is measured against comes from the training rows,
    # so the baseline gets no held-out advantage the network was denied.
    baseline_means = inner.mean(axis=0)

    print("fit window: %d of %d periods, ending %s (val_frac=%.2f, embargo=%d)" % (
        len(fit_periods), len(periods), pd.Timestamp(fit_periods[-1]).date(),
        args.val_frac, embargo))
    print("  inner fit   %6d rows over %4d periods" % (len(inner), len(inner_periods)))
    print("  early stop  %6d rows over %4d periods" % (len(stop), len(stop_periods)))
    print("  tune score  %6d rows over %4d periods, ending %s" % (
        len(tune), len(tune_periods), pd.Timestamp(tune_periods[-1]).date()))
    print("held fixed, not searched: code_dim=%d  swap_noise=%.2f" % (
        args.code_dim, args.swap_noise))
    # A step is 0.03s on the narrowest architecture in the space and about five
    # times that on the widest, so a single-number estimate would be wrong for
    # most of the search. Quote the range and let the per-trial lines below
    # report what it actually cost.
    print("budget: %d trials x %d gradient steps (%.0f-%.0f min in total, "
          "depending on the widths drawn)\n" % (
              args.n_trials, args.tune_steps,
              args.n_trials * args.tune_steps * 0.03 / 60,
              args.n_trials * args.tune_steps * 0.15 / 60))

    device = backend.resolve_device(args.device)

    def objective(trial):
        config = suggest_config(trial, args, len(inner))
        result = backend.train_autoencoder(
            inner, stop, config, device, verbose=False)

        r2 = column_mean_r2(
            tune,
            backend.reconstruct_values(
                result["model"], tune, device, config["batch_size"]),
            baseline_means)

        # Recorded so a budget-bound winner is visible afterwards rather than
        # being taken at face value.
        trial.set_user_attr("hit_epoch_cap", result["hit_epoch_cap"])
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("epochs_run", result["epochs_run"])
        trial.set_user_attr("stop_mse", result["best_val_mse"])

        if verbose:
            print("  trial %2d  r2 %.4f  hidden %-10s dropout %.2f  lr %.5f  "
                  "batch %4d  epochs %3d/%3d%s" % (
                      trial.number, r2, trial.params["hidden"],
                      trial.params["dropout"], trial.params["lr"],
                      trial.params["batch_size"], result["epochs_run"],
                      config["epochs"],
                      "  (BUDGET-BOUND)" if result["hit_epoch_cap"] else ""))
        return r2

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    started = datetime.datetime.now()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    elapsed = (datetime.datetime.now() - started).total_seconds()

    best = study.best_trial
    print("\nbest of %d trials: r2 %.4f  (%.0f min)" % (
        len(study.trials), best.value, elapsed / 60))
    for name, value in sorted(best.params.items()):
        print("    %-12s %s" % (name, value))

    capped = sum(t.user_attrs.get("hit_epoch_cap", False) for t in study.trials)
    if best.user_attrs.get("hit_epoch_cap"):
        print("WARN  the winner spent its whole step budget without patience "
              "running out, so it is BUDGET-BOUND: it was still improving when "
              "the trial ended, and a slower-converging config may well beat it "
              "given room. Re-run with a larger --tune-steps before trusting "
              "this ranking.")
    print("%d of %d trials were budget-bound at %d steps" % (
        capped, len(study.trials), args.tune_steps))

    spread = [t.value for t in study.trials if t.value is not None]
    if spread:
        print("r2 across trials: best %.4f  median %.4f  worst %.4f" % (
            max(spread), float(np.median(spread)), min(spread)))
        print("  (a narrow spread means these knobs are not what is limiting "
              "this autoencoder)")

    # ---- persist, and hand the winner over -------------------------------- #
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "%s.json" % args.study_name)
    with open(path, "w") as handle:
        json.dump({
            "study_name": args.study_name,
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv[1:] if argv is None else list(argv),
            "source_version": args.source_version,
            "objective": "held-out reconstruction R^2 on the tune slice",
            "fixed": {"code_dim": args.code_dim, "swap_noise": args.swap_noise},
            "searched": ["hidden", "dropout", "lr", "batch_size"],
            "budget": {"n_trials": args.n_trials, "tune_steps": args.tune_steps,
                       "patience_frac": args.patience_frac},
            "slices": {"inner_rows": int(len(inner)), "stop_rows": int(len(stop)),
                       "tune_rows": int(len(tune)),
                       "tune_ends": str(pd.Timestamp(tune_periods[-1]).date())},
            "best_value": best.value,
            "best_params": best.params,
            "best_user_attrs": best.user_attrs,
            "n_capped": int(capped),
            "elapsed_seconds": elapsed,
            "trials": [
                {"number": t.number, "value": t.value, "params": t.params,
                 "user_attrs": t.user_attrs} for t in study.trials
            ],
        }, handle, indent=2, sort_keys=True, default=str)
    print("\nstudy -> %s" % path)

    print("\nbuild the winner with:")
    print("  python3 -m src.data_mani.build_autoencoder \\\n"
          "      --source-version %s --code-dim %d --swap-noise %.2f \\\n"
          "      --hidden %s --dropout %.2f --lr %.5f --batch-size %d \\\n"
          "      --val-frac %.2f --out-suffix=-aetuned" % (
              args.source_version, args.code_dim, args.swap_noise,
              best.params["hidden"], best.params["dropout"], best.params["lr"],
              best.params["batch_size"], args.val_frac))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
