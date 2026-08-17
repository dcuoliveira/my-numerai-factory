"""
Training loop, seed selection and walk-forward for the LSTM pipeline.

Structure, mirroring iwtsf's two-phase driver:

    select_seeds()          Phase 1 -- train `max_trials` seeds on a fixed split,
                                       keep the best `top_frac` of them
    walk_forward()          Phase 2 -- roll forward in blocks, retraining each
                                       selected seed per block and averaging their
                                       predictions
      train_until_early_stop()   one seed, one block: epochs until patience runs out
        run_epoch()                one pass, with clipping
        slice_loss()               the statistic over a whole slice at once

------------------------------------------------------------------------------
Four things iwtsf got wrong, fixed here
------------------------------------------------------------------------------

Three are outright bugs in the source this is ported from, all verified against it:

1. **The "best" state was really the last state.** iwtsf's Phase 2 captured
   `best_model_state = model.state_dict()`, which holds *references* to the live
   parameter tensors that `optimizer.step()` keeps mutating in place. By the time
   `load_state_dict` ran, the "best" weights had been overwritten by whatever the
   final epoch produced -- so early stopping selected an epoch and then threw it
   away. Confirmed empirically: an aliased snapshot compares equal to the weights
   *now*, not to the weights at capture. Every snapshot here is cloned and detached
   to CPU (`clone_state`), which iwtsf's Phase 1 did correctly and its Phase 2 did
   not.

2. **Gradient clipping was applied in seed selection but not in the walk-forward.**
   Seeds were therefore chosen under a different optimisation regime than the one
   that produced the reported numbers. Here both phases go through `run_epoch`, so
   they cannot drift apart again.

3. **The model was built once per seed, outside the walk-forward loop**, so every
   block warm-started from the previous one and the whole backtest was one
   continuous training run. Each block gets a fresh model and optimizer unless
   `warm_start=True` is asked for explicitly.

The fourth is a design choice rather than a bug:

4. **Validation loss was the mean of per-batch statistics** (`val_loss / len(loader)`).
   A correlation is not linear in its sample, so the mean of per-batch correlations
   is not the correlation of the slice -- and it drifts with `batch_size`, which
   makes early stopping depend on a dial that should not affect it. `slice_loss`
   accumulates the whole slice and computes the statistic once.
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.prediction.nn.dataset import (
    SymbolSequenceDataset,
    build_period_positions,
    build_symbol_index,
    collate_windows,
)
from src.prediction.nn.model import LSTMForecastNetwork


def save_checkpoint(path, val_scores):
    """
    Persist seed-selection progress.

    JSON rather than a pickle: it is a handful of floats, and being able to read it
    with `cat` while a 100-seed run is in flight is worth more than the convenience
    of dumping arbitrary objects. Written to a temporary file and moved into place so
    an interrupt mid-write cannot leave a truncated checkpoint that then fails to
    load on resume.

    :param path: destination file
    :type path: str
    :param val_scores: {seed: best validation loss}
    :type val_scores: dict
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary = "{}.tmp".format(path)
    with open(temporary, "w") as handle:
        json.dump(
            {"val_scores": {str(k): float(v) for k, v in val_scores.items()}},
            handle, indent=2,
        )
    os.replace(temporary, path)


def load_checkpoint(path):
    """
    Read seed-selection progress back, with integer seed keys restored.

    :param path: checkpoint file
    :type path: str
    :return: {seed: best validation loss}
    :rtype: dict
    """
    with open(path) as handle:
        payload = json.load(handle)
    return {int(k): float(v) for k, v in payload.get("val_scores", {}).items()}


def clone_state(model):
    """
    A snapshot of the weights that will not change when training continues.

    `model.state_dict()` returns references, not copies. Cloning is the entire
    difference between early stopping working and early stopping being a very
    convincing no-op -- see bug 1 in the module docstring.

    :param model: the network to snapshot
    :type model: torch.nn.Module
    :return: detached CPU copies of every parameter and buffer
    :rtype: dict
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def set_seed(seed):
    """
    Seed every generator the training path touches.

    Deliberately lighter than iwtsf's version, which also set
    `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`. Those
    make cuDNN's LSTM refuse to run (it has no deterministic implementation), and
    iwtsf worked around that by disabling cuDNN entirely -- paying a large slowdown
    on the one layer that benefits most from it. Seeding the generators gives
    run-to-run reproducibility on a fixed device without that cost; exact
    cross-device reproducibility is not something this pipeline promises.

    :param seed: the seed
    :type seed: int
    """
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested=None):
    """
    Pick the device to train on.

    :param requested: an explicit torch device string, or None to auto-detect
    :type requested: str or None
    :return: the device
    :rtype: torch.device
    """
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_network(n_features, bin_values, config, num_symbols):
    """
    Construct the network from a flat config dict.

    :param n_features: feature count
    :type n_features: int
    :param bin_values: the target's bins, from model.infer_bin_values()
    :type bin_values: [float]
    :param config: keys hidden, num_layers, embed_dim, dropout, dropout_type,
                   layer_norm
    :type config: dict
    :param num_symbols: distinct training symbols
    :type num_symbols: int
    :return: the network, on the CPU
    :rtype: LSTMForecastNetwork
    """
    return LSTMForecastNetwork(
        input_dim=n_features,
        bin_values=bin_values,
        hidden_dim=config["hidden"],
        num_layers=config["num_layers"],
        num_symbols=num_symbols,
        embed_dim=config["embed_dim"],
        dropout=config["dropout"],
        dropout_type=config["dropout_type"],
        use_layer_norm=config["layer_norm"],
    )


def make_loader(dataset, batch_size, shuffle, generator=None):
    """
    Wrap a dataset in a DataLoader with the project's collate.

    num_workers is 0 on purpose. The dataset materialises every window up front, so
    `__getitem__` is a list index -- worker processes would add startup cost and
    memory copies to buy nothing. iwtsf used workers because its dataset did heavy
    per-item work.

    :param dataset: a SymbolSequenceDataset
    :type dataset: torch.utils.data.Dataset
    :param batch_size: rows per batch
    :type batch_size: int
    :param shuffle: whether to shuffle
    :type shuffle: bool
    :param generator: torch generator, so shuffling is seeded
    :type generator: torch.Generator or None
    :return: the loader
    :rtype: DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=shuffle,
        collate_fn=collate_windows,
        num_workers=0,
        generator=generator,
    )


def _forward(model, batch, device):
    """
    Move a batch to the device and run the network.

    :return: (signal, target, reference, mask), all (B, L) on the device
    :rtype: tuple of torch.Tensor
    """
    features = batch["features"].to(device)
    symbol_ids = batch["symbol_id"].to(device)
    signal = model(features, symbol_ids if model.use_symbol_embedding else None)
    return (
        signal,
        batch["target"].to(device),
        batch["reference"].to(device),
        batch["target_mask"].to(device),
    )


def run_epoch(model, loader, loss_fn, optimizer, device, max_grad_norm):
    """
    One training pass, returning the mean per-batch loss.

    The mean is for *reporting* only -- early stopping uses `slice_loss`, for the
    reason in the module docstring. Gradient clipping happens here so every caller
    gets it, which is fix 2.

    :param model: the network, in train mode
    :type model: torch.nn.Module
    :param loader: training batches
    :type loader: DataLoader
    :param loss_fn: one of prediction.nn.losses
    :type loss_fn: torch.nn.Module
    :param optimizer: the optimizer
    :type optimizer: torch.optim.Optimizer
    :param device: device to train on
    :type device: torch.device
    :param max_grad_norm: gradient-norm cap; 0 or None disables clipping
    :type max_grad_norm: float or None
    :return: mean batch loss
    :rtype: float
    """
    model.train()
    total = 0.0
    batches = 0

    for batch in loader:
        optimizer.zero_grad()
        signal, target, reference, mask = _forward(model, batch, device)
        loss = loss_fn(signal, target, reference=reference, mask=mask)

        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total += float(loss.item())
        batches += 1

    return total / batches if batches else float("nan")


@torch.no_grad()
def slice_loss(model, loader, loss_fn, device):
    """
    The loss over an entire slice, computed once on the concatenated predictions.

    Not the mean of per-batch losses. A correlation is not linear in its sample, so
    averaging per-batch correlations answers a different question and makes the
    answer depend on batch_size. This is fix 4.

    :param model: the network
    :type model: torch.nn.Module
    :param loader: batches to evaluate
    :type loader: DataLoader
    :param loss_fn: one of prediction.nn.losses
    :type loss_fn: torch.nn.Module
    :param device: device to evaluate on
    :type device: torch.device
    :return: the loss over the whole slice
    :rtype: float
    """
    model.eval()
    signals, targets, references, masks = [], [], [], []

    for batch in loader:
        signal, target, reference, mask = _forward(model, batch, device)
        signals.append(signal.reshape(-1))
        targets.append(target.reshape(-1))
        references.append(reference.reshape(-1))
        masks.append(mask.reshape(-1))

    if not signals:
        return float("nan")

    return float(
        loss_fn(
            torch.cat(signals),
            torch.cat(targets),
            reference=torch.cat(references),
            mask=torch.cat(masks),
        ).item()
    )


@torch.no_grad()
def predict_rows(model, loader, device, n_rows):
    """
    Predict a frame, averaging wherever windows overlap.

    Predict-mode windows overlap at the tail of every run (so no row is left
    uncovered), so a row can be predicted more than once. Averaging is the same
    reduction already used across seeds.

    :param model: the network
    :type model: torch.nn.Module
    :param loader: predict-mode batches
    :type loader: DataLoader
    :param device: device to run on
    :type device: torch.device
    :param n_rows: rows in the frame the dataset was built from
    :type n_rows: int
    :return: (mean prediction per row, times each row was predicted); rows never
             predicted hold NaN and 0
    :rtype: (np.ndarray, np.ndarray)
    """
    model.eval()
    totals = np.zeros(n_rows, dtype="float64")
    counts = np.zeros(n_rows, dtype="int64")

    for batch in loader:
        features = batch["features"].to(device)
        symbol_ids = batch["symbol_id"].to(device)
        signal = model(
            features, symbol_ids if model.use_symbol_embedding else None
        )

        flat_signal = signal.detach().cpu().numpy().reshape(-1)
        flat_rows = batch["row_index"].numpy().reshape(-1)
        flat_valid = batch["valid"].numpy().reshape(-1)

        # row_index is -1 on padding; without this filter a padded prediction lands
        # on row 0 of the frame.
        keep = flat_valid & (flat_rows >= 0)
        np.add.at(totals, flat_rows[keep], flat_signal[keep])
        np.add.at(counts, flat_rows[keep], 1)

    predictions = np.full(n_rows, np.nan, dtype="float64")
    covered = counts > 0
    predictions[covered] = totals[covered] / counts[covered]
    return predictions, counts


def train_until_early_stop(model, train_loader, val_loader, loss_fn, config,
                           device, verbose=True, tag=""):
    """
    Train one model until the validation loss stops improving.

    :param model: a freshly built network
    :type model: torch.nn.Module
    :param train_loader: training batches
    :type train_loader: DataLoader
    :param val_loader: validation batches, or None to skip early stopping
    :type val_loader: DataLoader or None
    :param loss_fn: one of prediction.nn.losses
    :type loss_fn: torch.nn.Module
    :param config: keys lr, epochs, patience, max_grad_norm
    :type config: dict
    :param device: device to train on
    :type device: torch.device
    :param verbose: print per-epoch progress
    :type verbose: bool
    :param tag: label for the progress lines
    :type tag: str
    :return: {"best_val_loss", "best_state", "epochs_run", "best_epoch",
              "train_history", "val_history"}
    :rtype: dict
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss = float("inf")
    best_state = clone_state(model)     # so an un-improved run still returns weights
    best_epoch = 0
    patience_left = config["patience"]
    train_history, val_history = [], []

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = run_epoch(
            model, train_loader, loss_fn, optimizer, device,
            config.get("max_grad_norm"),
        )
        train_history.append(train_loss)

        if val_loader is None:
            best_state = clone_state(model)
            best_epoch = epoch
            continue

        val_loss = slice_loss(model, val_loader, loss_fn, device)
        val_history.append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = clone_state(model)          # cloned; see fix 1
            patience_left = config["patience"]
        else:
            patience_left -= 1

        if verbose:
            print(
                "  {}epoch {:03d} | train {:+.4f} | val {:+.4f} | "
                "best {:+.4f} @{:03d} | patience {}".format(
                    "{} ".format(tag) if tag else "", epoch, train_loss, val_loss,
                    best_val_loss, best_epoch, patience_left,
                )
            )

        if patience_left <= 0:
            if verbose:
                print("  {}early stop at epoch {}".format(
                    "{} ".format(tag) if tag else "", epoch))
            break

    return {
        "best_val_loss": best_val_loss,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "epochs_run": len(train_history),
        "train_history": train_history,
        "val_history": val_history,
    }


# --------------------------------------------------------------------------- #
# Phase 1 -- seed selection
# --------------------------------------------------------------------------- #

def chronological_split(periods, val_frac, embargo):
    """
    Split periods into a train head and a validation tail, gapped by the embargo.

    The same shape as `make_validation_split`, applied one level down: this carves
    the early-stopping slice out of a training block. The embargoed periods belong
    to neither side, because a training row's forward target would otherwise reach
    into the validation slice and early stopping would be reading leaked signal.

    :param periods: ordered periods
    :type periods: [Any]
    :param val_frac: fraction of periods to hold out at the end
    :type val_frac: float
    :param embargo: periods to drop between the two
    :type embargo: int
    :return: (train_periods, val_periods)
    :rtype: ([Any], [Any])
    """
    n_val = int(round(len(periods) * val_frac))
    if n_val < 1:
        return list(periods), []

    train_end = max(0, len(periods) - n_val - embargo)
    return list(periods[:train_end]), list(periods[len(periods) - n_val:])


def select_seeds(frame, features, target_col, period_col, id_col, bin_values,
                 config, loss_fn, device, checkpoint_path=None, verbose=True):
    """
    Phase 1: train many seeds on one fixed split and keep the best few.

    Why seeds rather than hyperparameters: a correlation objective on a noisy target
    has a lot of run-to-run variance, so which initialisation you happened to draw
    matters as much as most architectural dials. iwtsf spends its search budget here
    -- 100 trainings, keep the top 10% -- and averages the survivors, which is
    variance reduction rather than model selection.

    Progress is checkpointed after every seed. At 100 trainings an interrupted run is
    a near-certainty, and without this it would restart from zero.

    :param frame: the training portion of the panel
    :type frame: pd.DataFrame
    :param features: feature column names
    :type features: [str]
    :param target_col: target column name
    :type target_col: str
    :param period_col: period column name
    :type period_col: str
    :param id_col: symbol column name
    :type id_col: str
    :param bin_values: the target's bins
    :type bin_values: [float]
    :param config: the run config (network, sequence, optimisation and seed keys)
    :type config: dict
    :param loss_fn: one of prediction.nn.losses
    :type loss_fn: torch.nn.Module
    :param device: device to train on
    :type device: torch.device
    :param checkpoint_path: where to save partial results, or None to disable
    :type checkpoint_path: str or None
    :param verbose: print progress
    :type verbose: bool
    :return: {"selected", "val_scores", "n_trials", "coverage"}
    :rtype: dict
    """
    periods = np.sort(frame[period_col].unique())
    train_periods, val_periods = chronological_split(
        periods, config["val_frac"], config["embargo"]
    )
    if not train_periods or not val_periods:
        raise ValueError(
            "seed selection needs both a train and a validation slice; got {} and "
            "{} periods. Lower --val-frac or --embargo.".format(
                len(train_periods), len(val_periods)
            )
        )

    train_frame = frame[frame[period_col].isin(set(train_periods))]
    val_frame = frame[frame[period_col].isin(set(val_periods))]

    # Built over the WHOLE run's periods, so the embargo hole between the two slices
    # is visible as a gap rather than closing up.
    positions = build_period_positions(periods)
    symbol_index, padding_id = build_symbol_index(train_frame[id_col].unique())

    shared = dict(
        features=features, target_col=target_col, period_col=period_col,
        id_col=id_col, period_positions=positions, symbol_index=symbol_index,
        padding_id=padding_id, seq_length=config["seq_length"],
        reference_col=config.get("reference_col"),
        overlapping=config["overlapping"], train_buffer=config["train_buffer"],
    )
    train_dataset = SymbolSequenceDataset(frame=train_frame, mode="train", **shared)
    val_dataset = SymbolSequenceDataset(frame=val_frame, mode="train", **shared)

    if not len(train_dataset) or not len(val_dataset):
        raise ValueError(
            "seed selection produced {} training and {} validation windows. "
            "seq_length={} is probably too long for this panel -- see the coverage "
            "report and try a smaller --seq-length.".format(
                len(train_dataset), len(val_dataset), config["seq_length"]
            )
        )

    if verbose:
        print("seed selection: train {} windows / {} periods, val {} windows / "
              "{} periods".format(
                  len(train_dataset), len(train_periods),
                  len(val_dataset), len(val_periods)))
        print("  train coverage: {}".format(train_dataset.coverage()))

    # Seeds drawn from a fixed meta-seed so the trial set itself is reproducible.
    set_seed(config["seed"])
    n_trials = int(config["max_trials"])
    trial_seeds = np.random.choice(
        np.arange(n_trials * 100), size=n_trials, replace=False
    ).tolist()

    val_scores = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        val_scores = load_checkpoint(checkpoint_path)
        if verbose:
            print("  resuming: {} of {} seeds already done".format(
                len(val_scores), n_trials))

    for position, trial_seed in enumerate(trial_seeds, start=1):
        if trial_seed in val_scores:
            continue

        set_seed(trial_seed)
        model = build_network(
            len(features), bin_values, config, num_symbols=len(symbol_index)
        )
        generator = torch.Generator()
        generator.manual_seed(int(trial_seed))

        started = time.time()
        result = train_until_early_stop(
            model,
            make_loader(train_dataset, config["batch_size"], config["shuffle"], generator),
            make_loader(val_dataset, config["batch_size"], False),
            loss_fn, config, device,
            verbose=False, tag="seed {}".format(trial_seed),
        )
        val_scores[trial_seed] = result["best_val_loss"]

        if verbose:
            print("  seed {:>5} ({:>3}/{}) val {:+.4f} | {} epochs | {:.0f}s".format(
                trial_seed, position, n_trials, result["best_val_loss"],
                result["epochs_run"], time.time() - started))

        if checkpoint_path:
            save_checkpoint(checkpoint_path, val_scores)

    # Lower loss is better, and the loss is already negated, so a plain sort works.
    keep = max(1, math.ceil(len(val_scores) * config["top_frac"]))
    ranked = sorted(val_scores.items(), key=lambda item: item[1])
    selected = [int(seed) for seed, _ in ranked[:keep]]

    if verbose:
        print("selected {} of {} seeds: {}".format(len(selected), len(val_scores), selected))

    return {
        "selected": selected,
        "val_scores": {int(k): float(v) for k, v in val_scores.items()},
        "n_trials": len(val_scores),
        "coverage": train_dataset.coverage(),
    }


# --------------------------------------------------------------------------- #
# Phase 2 -- walk-forward
# --------------------------------------------------------------------------- #

def walk_forward(frame, features, target_col, period_col, id_col, bin_values,
                 seeds, config, loss_fn, device, verbose=True):
    """
    Phase 2: roll forward in blocks, retraining per block, averaging over seeds.

    One block:

        [------------- training -------------][ embargo ][ predict this block ]
                              ^ last val_frac of it is the early-stopping slice

    Training may draw on earlier *validation* periods once the walk has moved past
    them -- that is what makes it a walk-forward rather than a single split.

    **The leakage rule, stated precisely.** A prediction window reaches back through
    the embargo gap into training periods, and that is legitimate: a feature at date
    `t` is observable at `t`, so feeding a symbol's own feature history to predict `t`
    uses nothing unknown at `t`. What must not cross the gap is *targets* -- a 20-day
    forward target from inside the embargo overlaps the block being predicted. So the
    embargo restricts which periods contribute LOSS TERMS, not which periods may be
    read as input. Predict-mode datasets are never passed to a loss function, and
    `test_no_target_leaks_across_embargo` pins the training side.

    :param frame: the full panel, carrying a "split" column
    :type frame: pd.DataFrame
    :param features: feature column names
    :type features: [str]
    :param target_col: target column name
    :type target_col: str
    :param period_col: period column name
    :type period_col: str
    :param id_col: symbol column name
    :type id_col: str
    :param bin_values: the target's bins
    :type bin_values: [float]
    :param seeds: the seeds selected by Phase 1
    :type seeds: [int]
    :param config: the run config
    :type config: dict
    :param loss_fn: one of prediction.nn.losses
    :type loss_fn: torch.nn.Module
    :param device: device to train on
    :type device: torch.device
    :param verbose: print progress
    :type verbose: bool
    :return: (predictions frame, per-block records, skipped blocks)
    :rtype: (pd.DataFrame, [dict], [dict])
    """
    all_periods = np.sort(frame[period_col].unique())
    position_of = {period: index for index, period in enumerate(all_periods)}
    positions = build_period_positions(all_periods)

    val_periods = np.sort(
        frame.loc[frame["split"] == "validation", period_col].unique()
    )
    if not len(val_periods):
        raise ValueError("no validation periods to predict")

    refit_every = int(config["refit_every"])
    embargo = int(config["embargo"])

    predictions = []
    records = []
    skipped = []

    for block_index, start in enumerate(range(0, len(val_periods), refit_every)):
        block = val_periods[start:start + refit_every]
        train_end = max(0, position_of[block[0]] - embargo)

        if config["cv_scheme"] == "expanding":
            train_start = 0
        else:
            train_start = max(0, train_end - int(config["train_window"]))
        train_periods = all_periods[train_start:train_end]

        if len(train_periods) < max(1, int(config.get("min_train", 0))):
            skipped.append({
                "block": block_index,
                "train_periods": int(len(train_periods)),
                "predict_start": str(block[0]),
                "predict_end": str(block[-1]),
            })
            if verbose:
                print("block {}: only {} training periods (< min_train={}), "
                      "not predicted".format(
                          block_index, len(train_periods), config.get("min_train", 0)))
            continue

        # The early-stopping slice is carved off the END of the training block, with
        # its own embargo, so it is the closest thing to the block being predicted.
        fit_periods, stop_periods = chronological_split(
            train_periods, config["val_frac"], embargo
        )
        if not fit_periods or not stop_periods:
            fit_periods, stop_periods = list(train_periods), []

        fit_frame = frame[frame[period_col].isin(set(fit_periods))]
        stop_frame = (
            frame[frame[period_col].isin(set(stop_periods))]
            if stop_periods else None
        )

        # The prediction frame is the block plus enough preceding periods to give the
        # first rows a lookback. Those extra periods are inputs only.
        lookback_start = max(0, position_of[block[0]] - int(config["seq_length"]))
        predict_periods = set(all_periods[lookback_start:position_of[block[-1]] + 1])
        predict_frame = frame[frame[period_col].isin(predict_periods)].reset_index(drop=True)

        symbol_index, padding_id = build_symbol_index(fit_frame[id_col].unique())
        shared = dict(
            features=features, target_col=target_col, period_col=period_col,
            id_col=id_col, period_positions=positions, symbol_index=symbol_index,
            padding_id=padding_id, seq_length=config["seq_length"],
            reference_col=config.get("reference_col"),
            overlapping=config["overlapping"], train_buffer=config["train_buffer"],
        )

        fit_dataset = SymbolSequenceDataset(frame=fit_frame, mode="train", **shared)
        stop_dataset = (
            SymbolSequenceDataset(frame=stop_frame, mode="train", **shared)
            if stop_frame is not None else None
        )
        predict_dataset = SymbolSequenceDataset(
            frame=predict_frame, mode="predict", **shared
        )

        if not len(fit_dataset) or not len(predict_dataset):
            skipped.append({
                "block": block_index,
                "reason": "no windows (fit={}, predict={})".format(
                    len(fit_dataset), len(predict_dataset)),
                "predict_start": str(block[0]),
                "predict_end": str(block[-1]),
            })
            if verbose:
                print("block {}: no windows to train or predict on, skipped".format(
                    block_index))
            continue

        if verbose:
            print("block {}: train {} windows / {} periods -> predict {} periods "
                  "({} windows)".format(
                      block_index, len(fit_dataset), len(fit_periods), len(block),
                      len(predict_dataset)))

        block_predictions = []
        seed_records = {}

        for seed in seeds:
            set_seed(seed)
            # Fresh model and optimizer per block -- fix 3. Warm-starting is
            # available but off, because it silently turns the walk-forward into one
            # continuous training run.
            model = build_network(
                len(features), bin_values, config, num_symbols=len(symbol_index)
            )
            if config.get("warm_start") and seed in getattr(walk_forward, "_carry", {}):
                model.load_state_dict(walk_forward._carry[seed])

            generator = torch.Generator()
            generator.manual_seed(int(seed))

            result = train_until_early_stop(
                model,
                make_loader(fit_dataset, config["batch_size"], config["shuffle"], generator),
                make_loader(stop_dataset, config["batch_size"], False)
                if stop_dataset is not None and len(stop_dataset) else None,
                loss_fn, config, device,
                verbose=False, tag="seed {}".format(seed),
            )

            # Restore the epoch early stopping actually chose -- possible only
            # because the snapshot was cloned.
            model.load_state_dict(result["best_state"])
            model = model.to(device)

            if config.get("warm_start"):
                if not hasattr(walk_forward, "_carry"):
                    walk_forward._carry = {}
                walk_forward._carry[seed] = result["best_state"]

            row_predictions, _ = predict_rows(
                model,
                make_loader(predict_dataset, config["batch_size"], False),
                device, len(predict_frame),
            )
            block_predictions.append(row_predictions)
            seed_records[int(seed)] = {
                "best_val_loss": float(result["best_val_loss"]),
                "best_epoch": int(result["best_epoch"]),
                "epochs_run": int(result["epochs_run"]),
            }

            if verbose:
                print("  seed {:>5} val {:+.4f} | {} epochs (best @{})".format(
                    seed, result["best_val_loss"], result["epochs_run"],
                    result["best_epoch"]))

        # The seed ensemble: mean across seeds, ignoring rows a seed could not cover.
        stacked = np.vstack(block_predictions)
        with np.errstate(invalid="ignore"):
            ensemble = np.nanmean(stacked, axis=0)

        out = predict_frame[[id_col, period_col, target_col]].copy()
        out["prediction"] = ensemble
        out["refit"] = block_index
        for column in (config.get("reference_col"), config.get("meta_model_col")):
            if column and column in predict_frame.columns:
                out[column] = predict_frame[column].to_numpy()

        # Drop the lookback rows: they were inputs, not predictions. Keeping them
        # would score periods that belong to an earlier block or to training.
        out = out[out[period_col].isin(set(block))]
        out = out[out["prediction"].notna()]
        predictions.append(out)

        records.append({
            "refit": block_index,
            "train_start": str(fit_periods[0]),
            "train_end": str(fit_periods[-1]),
            "predict_start": str(block[0]),
            "predict_end": str(block[-1]),
            "train_periods": int(len(fit_periods)),
            "train_rows": int(len(fit_frame)),
            "train_windows": int(len(fit_dataset)),
            "stop_periods": int(len(stop_periods)),
            "predicted_periods": int(len(block)),
            "predicted_rows": int(len(out)),
            "seeds": list(map(int, seeds)),
            "seed_results": seed_records,
            "best_val_loss": min(
                (record["best_val_loss"] for record in seed_records.values()),
                default=float("nan"),
            ),
        })

    if not predictions:
        raise ValueError(
            "the walk-forward produced no predictions; check --refit-every and "
            "--embargo against the number of validation periods"
        )

    return pd.concat(predictions, ignore_index=True), records, skipped
