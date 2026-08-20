"""
Fit symbol embeddings once, on the autoencoder's training block, and export them.

    python3 -m src.data_mani.fit_symbol_embedding --epochs 100 --embed-dim 16

This is the missing third of a chain whose other two parts already exist:

    prediction/nn/embeddings.py        pulls a matrix out of a trained network
    (this)                             trains the networks that produce them
    data_mani/build_symbol_embedding.py folds a matrix into a dataset

------------------------------------------------------------------------------
Why this exists next to run_nn_forecast's --export-embeddings
------------------------------------------------------------------------------

`run_nn_forecast --export-embeddings` writes one matrix PER WALK-FORWARD BLOCK,
because a forecaster refits per block and each refit has its own `train_end`. That
is the right shape for a model that is itself walking forward.

A FEATURE handed to ridge or lgb wants the other shape: one matrix, one boundary,
fitted once before the validation block -- exactly what `build_autoencoder.py` does
for its codes. One boundary is far easier to reason about, and it makes the
embedding directly comparable to the autoencoder arm, which is the point of the
comparison.

So this module fits on the SAME rows the autoencoder fits on: the pipeline's train
block, i.e. `make_validation_split`'s arithmetic at the same --val-frac and
--embargo. `--verify-boundary` reads an encoder.json and refuses to run unless the
two boundaries agree, rather than trusting that they do.

------------------------------------------------------------------------------
Three ways this deliberately differs from the autoencoder
------------------------------------------------------------------------------

1. **The ranking slice carries an embargo; the autoencoder's does not.** The
   encoder's loss is reconstruction -- features only, no target -- so its stop
   slice can start the period after its fit slice ends and nothing crosses. This
   network's loss is a correlation WITH THE TARGET, and a 20-day forward target
   from the last fit period reaches straight into the ranking slice. So an embargo
   sits between them, exactly as it does everywhere else a target is involved.

2. **No early stopping.** `--epochs` runs to completion and the FINAL weights are
   exported. The held-out slice is still scored, but only to rank seeds against
   each other -- it never chooses an epoch. That is what "100 epochs, no early
   stopping" means here, and it is why the export carries `val_loss` while nothing
   in the loop reads it.

3. **Every seed is exported, never averaged.** See the long note in
   prediction/nn/embeddings.py: two seeds' embedding matrices differ by an
   arbitrary rotation of a space whose axes carry no fixed meaning, so averaging
   them destroys the structure. The consumer picks one, by default the best
   `val_loss` -- which is the only reason a ranking slice exists at all.

torch is imported lazily, inside main(). src/live/features.py imports data_mani
builders at module scope, so a top-level torch import here would take the live
submission path down on a box where the wheel failed to install.
"""

import argparse
import json
import os

import numpy as np


def parse_args(argv=None):
    """
    Parse the command line.

    :param argv: argument list, or None to read sys.argv
    :type argv: [str] or None
    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m src.data_mani.fit_symbol_embedding",
        description="Fit LSTM symbol embeddings on the autoencoder's train block.",
    )

    # ---- the block to fit on, matching build_autoencoder.py -----------------
    parser.add_argument("--tournament", default="crypto")
    parser.add_argument("--source-version", default="crypto/v2.0",
                        help="dataset the network reads")
    parser.add_argument("--target", default=None,
                        help="target column; default is the tournament's own")
    parser.add_argument("--val-frac", type=float, default=0.3,
                        help="fraction the RUNS will hold out; with --embargo this "
                             "fixes the fit boundary. Must match the autoencoder's")
    parser.add_argument("--embargo", default="auto",
                        help="periods between blocks; 'auto' from the target horizon")
    parser.add_argument("--rank-frac", type=float, default=0.15,
                        help="chronological tail of the fit block held out to RANK "
                             "seeds. Never used to stop training. Matches the "
                             "autoencoder's --stop-frac")
    parser.add_argument("--verify-boundary", default=None, metavar="ENCODER_JSON",
                        help="refuse to run unless this encoder's fit boundary "
                             "matches the one computed here")

    # ---- sequences ----------------------------------------------------------
    parser.add_argument("--seq-length", type=int, default=21)
    parser.add_argument("--overlapping", action="store_true")
    parser.add_argument("--train-buffer", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)

    # ---- network ------------------------------------------------------------
    parser.add_argument("--embed-dim", type=int, default=16,
                        help="embedding width; this many dense columns reach the "
                             "downstream dataset")
    parser.add_argument("--hidden", type=int, default=60)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dropout-type", default="regular",
                        choices=("regular", "variational"))
    parser.add_argument("--no-layer-norm", action="store_true")

    # ---- objective ----------------------------------------------------------
    parser.add_argument("--loss", default="mixture",
                        choices=("corr", "bmc", "mixture"))
    parser.add_argument("--bmc-weight", type=float, default=1.0)
    parser.add_argument("--corr-weight", type=float, default=0.1)
    parser.add_argument("--bmc-reference", default="meta_proxy",
                        choices=("meta_proxy", "benchmark"))
    parser.add_argument("--proxy-alpha", type=float, default=1.0)

    # ---- optimisation. No --patience: early stopping is off by construction --
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # ---- seeds and output ---------------------------------------------------
    parser.add_argument("--seeds", type=int, default=5,
                        help="networks to fit; every one is exported separately")
    parser.add_argument("--seed", type=int, default=42,
                        help="meta-seed for drawing the seed set")
    parser.add_argument("--out", default=None,
                        help="destination parquet; default "
                             "models/embeddings/<name>.parquet")
    parser.add_argument("--name", default="symemb",
                        help="export stem, used for the parquet and its json sidecar")
    parser.add_argument("--device", default=None)
    parser.add_argument("-q", "--quiet", action="store_true")

    return parser.parse_args(argv)


def train_block_periods(df, period_col):
    """
    The periods the autoencoder is allowed to fit on.

    Read off the `split` column rather than recomputed. `load_tournament_data` has
    already applied `make_validation_split`, which is the SAME arithmetic
    `build_autoencoder.fit_boundary` uses -- so the train split IS the encoder's fit
    window, and there is nothing left to derive.

    Recomputing it here was a real bug: the loaded frame has already had the
    embargo periods REMOVED (1,701 periods in the parquet, 1,681 after loading), so
    subtracting the embargo a second time moved the boundary 20 periods early --
    2024-06-06 against the encoder's 2024-06-26. --verify-boundary caught it. The
    lesson is the general one: derive a boundary once, in one place, and check it
    against the artifact that already recorded it.

    :param df: the loaded panel, carrying a "split" column
    :type df: pd.DataFrame
    :param period_col: name of the period column
    :type period_col: str
    :return: the fit periods, ascending
    :rtype: np.ndarray
    :raises ValueError: if the train split is empty
    """
    periods = np.sort(df.loc[df["split"] == "train", period_col].unique())
    if len(periods) < 2:
        raise ValueError(
            "the train split has {} periods; --val-frac is too high".format(
                len(periods)))
    return periods


def rank_split(fit_periods, rank_frac, embargo):
    """
    Split the fit block into the rows trained on and a tail used only for ranking.

    The embargo between them is the one real difference from the autoencoder's
    equivalent split -- see point 1 of the module docstring.

    :param fit_periods: the fit block
    :type fit_periods: np.ndarray
    :param rank_frac: fraction of it held out for ranking
    :type rank_frac: float
    :param embargo: gap between the two, in periods
    :type embargo: int
    :return: (inner periods trained on, ranking periods)
    :rtype: (np.ndarray, np.ndarray)
    """
    n_rank = int(round(len(fit_periods) * rank_frac))
    if n_rank < 1:
        return fit_periods, fit_periods[:0]

    inner_end = max(0, len(fit_periods) - n_rank - embargo)
    if inner_end < 2:
        raise ValueError(
            "rank_frac={} leaves no periods to train on".format(rank_frac))
    return fit_periods[:inner_end], fit_periods[len(fit_periods) - n_rank:]


def check_boundary(encoder_json, boundary_period):
    """
    Refuse to proceed unless a reference encoder agrees on the fit boundary.

    Verifying beats asserting: --val-frac is a run-time flag, so an embedding built
    at one value and an autoencoder built at another would sit on different
    boundaries while every artifact still claimed to be "the train block".

    :param encoder_json: path to an encoder.json from build_autoencoder
    :type encoder_json: str
    :param boundary_period: last period this fit may see
    :type boundary_period: str
    :raises SystemExit: on a mismatch, naming both dates
    """
    with open(encoder_json) as handle:
        meta = json.load(handle)

    theirs = str(meta.get("fit_boundary_period"))
    if theirs != str(boundary_period):
        raise SystemExit(
            "boundary mismatch: {} fits through {}, this run through {}. They are "
            "not the same train sample -- check --val-frac and --embargo against "
            "the encoder's ({} / {}).".format(
                encoder_json, theirs, boundary_period,
                meta.get("val_frac"), meta.get("embargo"))
        )
    print("ok    boundary matches {}: {}".format(encoder_json, theirs))


def main(argv=None):
    """
    Fit every seed and write the export.

    :param argv: argument list, or None to read sys.argv
    :type argv: [str] or None
    :return: process exit code
    :rtype: int
    """
    args = parse_args(argv)
    verbose = not args.quiet

    # Lazy, for the reason in the module docstring.
    import pandas as pd
    import torch

    from src.data_mani.utils import default_embargo, get_spec
    from src.prediction.nn.dataset import (
        SymbolSequenceDataset, build_period_positions, build_symbol_index,
    )
    from src.prediction.nn.embeddings import extract_embedding, save_embeddings
    from src.prediction.nn.losses import build_loss
    from src.prediction.nn.model import infer_bin_values
    from src.prediction.nn.pipeline import prepare_panel
    from src.prediction.nn.train import (
        build_network, make_loader, resolve_device, run_epoch, set_seed, slice_loss,
    )

    if args.embed_dim < 1:
        raise SystemExit("--embed-dim must be >= 1; there is nothing to export at 0")

    spec = get_spec(args.tournament)
    target_col = args.target or spec["target"]
    embargo = (default_embargo(args.tournament, target_col)
               if args.embargo == "auto" else int(args.embargo))

    panel = prepare_panel(
        tournament=args.tournament, target=args.target,
        data_version=args.source_version, era_step=1, val_frac=args.val_frac,
        embargo=embargo, bmc_reference=args.bmc_reference,
        proxy_alpha=args.proxy_alpha, require_reference=(args.loss != "corr"),
        verbose=verbose,
    )
    df = panel["df"]
    period_col, id_col = panel["period_col"], panel["id_col"]

    all_periods = np.sort(df[period_col].unique())
    fit_periods = train_block_periods(df, period_col)
    boundary_period = pd.Timestamp(fit_periods[-1]).date()

    if args.verify_boundary:
        check_boundary(args.verify_boundary, boundary_period)

    inner_periods, rank_periods = rank_split(fit_periods, args.rank_frac, embargo)
    inner = df[df[period_col].isin(set(inner_periods))]
    ranking = df[df[period_col].isin(set(rank_periods))]

    if verbose:
        print("fit block   : {} periods through {} ({:,} rows)".format(
            len(fit_periods), boundary_period, int(df[period_col].isin(set(fit_periods)).sum())))
        print("  trained on: {} periods ({:,} rows)".format(len(inner_periods), len(inner)))
        print("  ranking   : {} periods ({:,} rows), embargo {} between them".format(
            len(rank_periods), len(ranking), embargo))

    positions = build_period_positions(all_periods)
    symbol_index, padding_id = build_symbol_index(inner[id_col].unique())
    bin_values = infer_bin_values(inner[target_col])

    shared = dict(
        features=panel["features"], target_col=target_col, period_col=period_col,
        id_col=id_col, period_positions=positions, symbol_index=symbol_index,
        padding_id=padding_id, seq_length=args.seq_length,
        reference_col=panel["reference_col"], overlapping=args.overlapping,
        train_buffer=args.train_buffer,
    )
    train_dataset = SymbolSequenceDataset(frame=inner, mode="train", **shared)
    rank_dataset = (SymbolSequenceDataset(frame=ranking, mode="train", **shared)
                    if len(ranking) else None)

    if not len(train_dataset):
        raise SystemExit(
            "no training windows at seq_length={} -- see the coverage report and "
            "try a shorter --seq-length".format(args.seq_length))
    if verbose:
        print("windows     : {} train / {} ranking | {} symbols get a vector".format(
            len(train_dataset), len(rank_dataset) if rank_dataset else 0,
            len(symbol_index)))
        print("coverage    : {}".format(train_dataset.coverage()))

    device = resolve_device(args.device)
    loss_fn = build_loss(args.loss, bmc_weight=args.bmc_weight,
                         corr_weight=args.corr_weight)

    config = {
        "hidden": args.hidden, "num_layers": args.num_layers,
        "embed_dim": args.embed_dim, "dropout": args.dropout,
        "dropout_type": args.dropout_type, "layer_norm": not args.no_layer_norm,
        "lr": args.lr, "max_grad_norm": args.max_grad_norm,
    }

    set_seed(args.seed)
    seeds = np.random.choice(np.arange(args.seeds * 100), size=args.seeds,
                             replace=False).tolist()

    out_path = args.out or os.path.join(
        "models", "embeddings", "{}.parquet".format(args.name))
    if os.path.exists(out_path):
        # save_embeddings APPENDS, so a stale file would silently mix two fits.
        os.remove(out_path)
        if verbose:
            print("removed stale export at {}".format(out_path))

    frames, records = [], []
    for position, trial_seed in enumerate(seeds, start=1):
        set_seed(trial_seed)
        model = build_network(len(panel["features"]), bin_values, config,
                              num_symbols=len(symbol_index)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        generator = torch.Generator()
        generator.manual_seed(int(trial_seed))
        loader = make_loader(train_dataset, args.batch_size, True, generator)

        # No early stopping, by construction: every epoch runs and the FINAL
        # weights are the ones exported.
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, loader, loss_fn, optimizer, device,
                                   args.max_grad_norm)
            if verbose and (epoch % 20 == 0 or epoch == 1):
                print("  seed {:>5} epoch {:>3}/{}  train {:+.4f}".format(
                    trial_seed, epoch, args.epochs, train_loss))

        rank_loss = (slice_loss(model, make_loader(rank_dataset, args.batch_size, False),
                                loss_fn, device)
                     if rank_dataset is not None and len(rank_dataset) else float("nan"))

        frames.append(extract_embedding(
            model, symbol_index, block=0, seed=trial_seed,
            train_end=boundary_period, val_loss=rank_loss,
        ))
        records.append({"seed": int(trial_seed), "rank_loss": float(rank_loss),
                        "final_train_loss": float(train_loss)})
        if verbose:
            print("seed {:>5} ({}/{}) done -- rank loss {:+.4f}".format(
                trial_seed, position, len(seeds), rank_loss))

    save_embeddings(out_path, frames)

    # A sidecar in encoder.json's spirit: everything a downstream leakage gate or a
    # future reader needs, without reopening the parquet.
    sidecar = os.path.splitext(out_path)[0] + ".json"
    with open(sidecar, "w") as handle:
        json.dump({
            "source_version": args.source_version, "target": target_col,
            "fit_boundary_period": str(boundary_period),
            "fit_periods": int(len(fit_periods)),
            "trained_periods": int(len(inner_periods)),
            "rank_periods": int(len(rank_periods)),
            "val_frac": args.val_frac, "embargo": embargo,
            "rank_frac": args.rank_frac, "early_stopping": False,
            "epochs": args.epochs, "seq_length": args.seq_length,
            "overlapping": args.overlapping, "train_buffer": args.train_buffer,
            "embed_dim": args.embed_dim, "hidden": args.hidden,
            "num_layers": args.num_layers, "dropout": args.dropout,
            "loss": args.loss, "bmc_weight": args.bmc_weight,
            "corr_weight": args.corr_weight, "bmc_reference": panel["bmc_reference"],
            "n_symbols": len(symbol_index), "n_features": len(panel["features"]),
            "seeds": records, "device": str(device),
        }, handle, indent=2, sort_keys=True)

    if verbose:
        best = min(records, key=lambda r: r["rank_loss"]
                   if np.isfinite(r["rank_loss"]) else float("inf"))
        print("\nwrote {} ({} symbols x {} dims x {} seeds)".format(
            out_path, len(symbol_index), args.embed_dim, len(seeds)))
        print("      {}".format(sidecar))
        print("best seed by rank loss: {} ({:+.4f})".format(
            best["seed"], best["rank_loss"]))
        print("\nnext:  python3 -m src.data_mani.build_symbol_embedding "
              "--export {} --source-version {} --out-suffix=-symemb".format(
                  out_path, args.source_version))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
