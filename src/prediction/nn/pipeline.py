"""
Orchestration for the LSTM pipeline: the analogue of functions.forecast().

Same division of labour as the tabular side -- run_nn_forecast.py parses arguments
and writes files, this module loads data and runs the two phases:

    nn_forecast()
      load_tournament_data()      the same loader the tabular pipeline uses
      attach reference / proxy    the same helpers, in the same order
      select_seeds()              Phase 1
      walk_forward()              Phase 2
      evaluate() / per_period_table()   the REAL numerai_tools scoring

The data-preparation sequence deliberately mirrors `functions.forecast()` step for
step, calling the same public helpers from data_mani.utils rather than
reimplementing any of them. It is a separate function rather than a shared one
because forecast() is tangled up with the sklearn search -- wrappers, scorers,
train_target_col -- none of which exists here, and threading a "which pipeline am I"
flag through it would make the tabular path harder to read for no gain.
"""

from src.data_mani.utils import (
    META_PROXY_COL,
    attach_meta_model,
    attach_reference,
    build_meta_proxy,
    default_embargo,
    fill_missing_features,
    get_spec,
    load_meta_model,
    load_reference,
    load_tournament_data,
)
from src.prediction.functions import evaluate, filter_features
from src.prediction.metrics import per_period_table, summarize
from src.prediction.nn.losses import build_loss, needs_reference
from src.prediction.nn.model import infer_bin_values
from src.prediction.nn.train import resolve_device, select_seeds, walk_forward


def prepare_panel(tournament, target=None, data_version=None, feature_set=None,
                  era_step=1, val_frac=0.3, embargo=None, bmc_reference="meta_proxy",
                  proxy_alpha=1.0, feature_include=None, feature_exclude=None,
                  require_reference=False, verbose=True):
    """
    Load the panel and attach everything the losses and the scoring need.

    :param tournament: "crypto" or "numerai"
    :type tournament: str
    :param target: target column, or None for the tournament default
    :type target: str or None
    :param data_version: dataset version, or None for the default
    :type data_version: str or None
    :param feature_set: Classic feature set name, ignored on crypto
    :type feature_set: str or None
    :param era_step: keep every Nth period
    :type era_step: int
    :param val_frac: fraction of periods held out as validation
    :type val_frac: float
    :param embargo: periods between train and validation, or None to derive it
    :type embargo: int or None
    :param bmc_reference: "meta_proxy" or "benchmark"
    :type bmc_reference: str
    :param proxy_alpha: Ridge penalty for the meta proxy
    :type proxy_alpha: float
    :param feature_include: regex of features to keep
    :type feature_include: str or None
    :param feature_exclude: regex of features to drop, applied after include
    :type feature_exclude: str or None
    :param require_reference: whether the training loss needs a reference. When
                             False, an unbuildable meta proxy degrades to the
                             benchmark reference with a warning instead of raising --
                             a CORR run must not die over a series it never reads.
    :type require_reference: bool
    :param verbose: print progress
    :type verbose: bool
    :return: everything downstream needs, keyed as documented inline
    :rtype: dict
    """
    spec = get_spec(tournament)
    target_col = target or spec["target"]
    period_col = spec["period_col"]
    id_col = spec["id_col"]

    if embargo is None:
        embargo = default_embargo(tournament, target_col)

    df, features = load_tournament_data(
        tournament=tournament, version=data_version, feature_set=feature_set,
        era_step=era_step, val_frac=val_frac, embargo=embargo, verbose=verbose,
        target=target_col,
    )

    features = filter_features(features, feature_include, feature_exclude,
                               verbose=verbose)

    # An LSTM cannot take NaN, and dropping the rows would bias the panel toward
    # long-lived symbols -- exactly the survivorship problem the windower already
    # has to fight. Same neutral-bin fill the tabular pipeline uses.
    df = fill_missing_features(df, features, spec["neutral_value"], verbose)

    reference_col = None
    coverage = 0.0
    try:
        reference, reference_col = load_reference(tournament, data_version, verbose)
        df, coverage = attach_reference(df, reference, tournament, reference_col)
        if verbose:
            print("reference '{}' covers {:.1%} of rows".format(reference_col, coverage))
    except Exception as error:
        if verbose:
            print("could not load reference ({}); BMC will be skipped".format(error))
        reference_col = None

    meta_model_col = None
    meta_coverage = 0.0
    try:
        meta_frame, meta_model_col = load_meta_model(tournament, data_version, verbose)
        df, meta_coverage = attach_meta_model(df, meta_frame, tournament)
        if verbose:
            print("meta model '{}' covers {:.1%} of rows".format(
                meta_model_col, meta_coverage))
    except Exception as error:
        if verbose:
            print("could not load the meta model ({}); MMC will be skipped".format(error))
        meta_model_col = None

    proxy_periods = 0
    proxy_error = "no meta model available"
    if meta_model_col is not None:
        try:
            df, proxy_periods = build_meta_proxy(
                df, features=features, period_col=period_col,
                train_mask=(df["split"] == "train"), meta_col=meta_model_col,
                alpha=proxy_alpha, verbose=verbose,
            )
            proxy_error = None
        except Exception as error:
            proxy_error = str(error)
            if verbose:
                print("could not build the meta proxy ({}); "
                      "--bmc-reference meta_proxy is unavailable".format(error))

    # Which series a BMC *training loss* sees. The default differs from the tabular
    # pipeline's on purpose.
    #
    # Crypto's real reference is the published meta model, which covers 566 of 1,700
    # dates and does not begin until 2024-05. Scoring BMC on that subset is fine --
    # it just narrows the scored window. TRAINING on it is not: most batches would
    # carry no reference at all, so the BMC term would contribute nothing to the
    # gradient for most of history and the model would effectively be trained on
    # whatever fraction happened to be covered.
    #
    # The proxy is a Ridge fit on the meta model's per-period rank using training
    # rows only, predicted everywhere -- full coverage, no leakage. The honest
    # caveat is that it is a Ridge on the same 22 features, so "contribution over the
    # proxy" partly means "differs from a linear model on these features", which is
    # related to but not the same as contribution over the real meta model. Both are
    # reported at the end so the gap is visible.
    if bmc_reference not in ("benchmark", "meta_proxy"):
        raise ValueError("bmc_reference must be 'benchmark' or 'meta_proxy'")

    if bmc_reference == "meta_proxy":
        if META_PROXY_COL in df.columns:
            reference_col = META_PROXY_COL
            coverage = float(df[META_PROXY_COL].notna().mean())
            if verbose:
                print("BMC reference set to the meta proxy ({:.1%} coverage)".format(
                    coverage))
        elif require_reference:
            raise ValueError(
                "--bmc-reference meta_proxy is unavailable for '{}': {}. Use "
                "--bmc-reference benchmark instead, accepting that the BMC term "
                "only has gradient where the real reference exists.".format(
                    tournament, proxy_error)
            )
        else:
            # The proxy is only the DEFAULT because a BMC training loss needs
            # gradient everywhere. A CORR run never reads a reference, so failing it
            # over an unbuildable proxy would be refusing to do work that does not
            # depend on the thing that broke. Fall back to the benchmark, which is
            # still enough to REPORT BMC on whatever it covers, and say so.
            bmc_reference = "benchmark"
            print("note: the meta proxy could not be built ({}); BMC will be "
                  "reported against the benchmark instead. The training loss does "
                  "not use a reference, so this does not affect it.".format(
                      proxy_error))

    return {
        "df": df,
        "features": features,
        "target_col": target_col,
        "period_col": period_col,
        "id_col": id_col,
        "embargo": embargo,
        "reference_col": reference_col,
        "reference_coverage": coverage,
        # Returned because it may have been downgraded from meta_proxy above, and the
        # results file has to say which series was actually used, not which was asked
        # for.
        "bmc_reference": bmc_reference,
        "meta_model_col": meta_model_col,
        "meta_coverage": meta_coverage,
        "proxy_periods": proxy_periods,
        "spec": spec,
    }


def nn_forecast(tournament, config, verbose=True):
    """
    Run both phases end to end and score the result.

    :param tournament: "crypto" or "numerai"
    :type tournament: str
    :param config: the full resolved run config, as built by run_nn_forecast.py
    :type config: dict
    :param verbose: print progress
    :type verbose: bool
    :return: (predictions frame, per-period table, results dict)
    :rtype: (pd.DataFrame, pd.DataFrame, dict)
    """
    # Whether a missing reference is fatal depends entirely on the loss: BMC and the
    # mixture cannot be computed without one, CORR never looks at it.
    reference_required = needs_reference(config["loss"], config["bmc_weight"])

    panel = prepare_panel(
        tournament=tournament, target=config.get("target"),
        data_version=config.get("data_version"),
        feature_set=config.get("feature_set"),
        era_step=config["era_step"], val_frac=config["val_frac"],
        embargo=config.get("embargo"), bmc_reference=config["bmc_reference"],
        proxy_alpha=config["proxy_alpha"],
        feature_include=config.get("feature_include"),
        feature_exclude=config.get("feature_exclude"),
        require_reference=reference_required,
        verbose=verbose,
    )

    df = panel["df"]
    features = panel["features"]
    target_col = panel["target_col"]
    period_col = panel["period_col"]
    id_col = panel["id_col"]

    if reference_required and not panel["reference_col"]:
        raise ValueError(
            "--loss {} needs a reference series, which failed to load for "
            "tournament '{}'".format(config["loss"], tournament)
        )

    # The embargo is resolved inside prepare_panel from the target actually being
    # modelled, so put the resolved value back where the trainer will read it. Same
    # for bmc_reference, which prepare_panel may have downgraded.
    config = dict(config)
    config["embargo"] = panel["embargo"]
    config["bmc_reference"] = panel["bmc_reference"]
    config["reference_col"] = panel["reference_col"]
    config["meta_model_col"] = panel["meta_model_col"]

    # One class per bin, read off the TRAINING target only. Reading it off the whole
    # frame would be a (very mild) leak, and it would also mask a validation block
    # that contains a bin training never saw.
    bin_values = infer_bin_values(df.loc[df["split"] == "train", target_col])
    if verbose:
        print("multiclass head: {} bins {}".format(len(bin_values), bin_values))

    if config["era_step"] > 1:
        # Harmless for the row-independent tabular models, structural here: every
        # window's "consecutive" timesteps would be era_step periods apart.
        print("WARNING: --era-step {} breaks sequence contiguity -- {} 'consecutive' "
              "timesteps would span {} periods. Use it for smoke runs only.".format(
                  config["era_step"], config["seq_length"],
                  config["seq_length"] * config["era_step"]))

    device = resolve_device(config.get("device"))
    loss_fn = build_loss(
        config["loss"], bmc_weight=config["bmc_weight"],
        corr_weight=config["corr_weight"],
    )
    if verbose:
        print("training on {} with loss '{}'".format(device, config["loss"]))

    train_frame = df[df["split"] == "train"]
    selection = select_seeds(
        train_frame, features, target_col, period_col, id_col, bin_values,
        config, loss_fn, device,
        checkpoint_path=config.get("checkpoint_path"), verbose=verbose,
    )

    predictions, refits, skipped = walk_forward(
        df, features, target_col, period_col, id_col, bin_values,
        seeds=selection["selected"], config=config, loss_fn=loss_fn,
        device=device, verbose=verbose,
    )

    # Scored with the REAL numerai_tools functions, never the differentiable
    # surrogates the network trained on. The gap between the two is the point.
    results = evaluate(
        predictions, target_col=target_col, period_col=period_col,
        reference_col=panel["reference_col"], meta_model_col=panel["meta_model_col"],
    )
    periods_table = per_period_table(
        predictions, pred_col="prediction", target_col=target_col,
        period_col=period_col, reference_col=panel["reference_col"],
        meta_model_col=panel["meta_model_col"],
    )

    results["pnl"] = summarize(periods_table["pnl"])
    results["reference"] = {
        "column": panel["reference_col"],
        "row_coverage": panel["reference_coverage"],
        "kind": config["bmc_reference"],
        "proxy_fit_periods": (
            panel["proxy_periods"] if config["bmc_reference"] == "meta_proxy" else None
        ),
    }
    results["meta_model"] = {
        "column": panel["meta_model_col"], "row_coverage": panel["meta_coverage"],
    }
    results["refits"] = refits
    results["skipped_blocks"] = skipped
    results["seed_selection"] = {
        "n_trials": selection["n_trials"],
        "top_frac": config["top_frac"],
        "selected": selection["selected"],
        "val_scores": selection["val_scores"],
        "train_coverage": selection["coverage"],
    }
    results["config"] = {
        "tournament": tournament, "model": "lstm", "target": target_col,
        "data_version": config.get("data_version") or panel["spec"]["version"],
        "n_features": len(features),
        "feature_include": config.get("feature_include"),
        "feature_exclude": config.get("feature_exclude"),
        "era_step": config["era_step"], "val_frac": config["val_frac"],
        "refit_every": config["refit_every"], "embargo": panel["embargo"],
        "cv_scheme": config["cv_scheme"], "train_window": config.get("train_window"),
        "min_train": config.get("min_train", 0),
        # Network
        "hidden": config["hidden"], "num_layers": config["num_layers"],
        "embed_dim": config["embed_dim"], "dropout": config["dropout"],
        "dropout_type": config["dropout_type"], "layer_norm": config["layer_norm"],
        "n_classes": len(bin_values), "bin_values": bin_values,
        # Sequence
        "seq_length": config["seq_length"], "overlapping": config["overlapping"],
        "batch_size": config["batch_size"], "train_buffer": config["train_buffer"],
        # Optimisation
        "lr": config["lr"], "epochs": config["epochs"],
        "patience": config["patience"], "max_grad_norm": config["max_grad_norm"],
        "warm_start": config.get("warm_start", False),
        # Loss. The weights are independent multipliers, NOT build_scorer's convex
        # metric_weight -- recorded under their own names so the two conventions can
        # never be confused when reading a results file.
        "loss": config["loss"], "bmc_weight": config["bmc_weight"],
        "corr_weight": config["corr_weight"],
        "bmc_reference": config["bmc_reference"], "proxy_alpha": config["proxy_alpha"],
        # Seeds
        "seed": config["seed"], "max_trials": config["max_trials"],
        "top_frac": config["top_frac"], "device": str(device),
    }

    return predictions, periods_table, results
