"""
A bottleneck autoencoder over one row's feature vector, used to build features.

Not a forecaster. Nothing here predicts the target; the object is a k-dimensional
code that reconstructs the 22 raw indicators, and those k activations become new
`feature_*` columns in a dataset that every existing runner picks up unchanged.
The CLI that drives it is `src/data_mani/build_autoencoder.py` -- this module is
only the network and its training loop, because torch is optional and
`src/live/features.py` imports `data_mani` builders at module scope.

------------------------------------------------------------------------------
Rows, not sequences
------------------------------------------------------------------------------

`dataset.py` exists because symbol histories are full of holes -- 78% of symbols
have a gap inside their own span, and contiguous runs have a median length of 4
dates -- so a windowed model covers most rows and not all of them. That is
survivable for a forecaster, which is scored on the rows it covered. It is not
survivable for a DATASET builder: a row with no code would be filled with the
neutral 0.5 by `fill_missing_features`, which means "no view", and the rows left
uncovered are exactly the young, sparse tokens whose 60-day indicators are
already missing. That reintroduces through the back door the survivorship
problem `fill_missing_features` exists to avoid.

So this reads one row at a time and emits a code for every one of them. The
thing worth compressing here is the CROSS-SECTION of 22 correlated indicators at
one instant anyway; the time dimension is already inside the features, which are
20d/60d constructions, and is available orthogonally through
`build_rolling_features`.

------------------------------------------------------------------------------
Why the bottleneck is linear and the decoder is not
------------------------------------------------------------------------------

The code is rank-normalised within each date before it is written to the parquet
(see the builder's scale section), so bounding it inside the network would buy a
range we do not need at the price of saturating gradients. Worse, under a
sigmoid bottleneck "0.5 neutral" would mean "pre-activation zero" -- wherever
the bias happened to land -- rather than "absence of information", which is the
entire load-bearing claim of the repo's [0, 1] convention.

The DECODER output is sigmoid, because its target genuinely lives in [0, 1]: the
inputs are rank bins on a five-point grid. An unbounded head can reconstruct to
1.4 and pay MSE for a value the input can never take.

Hidden activations are ReLU. There is no precedent either way in this repo and
this was not tuned.

------------------------------------------------------------------------------
Swap noise
------------------------------------------------------------------------------

The denoising objective is the standard tabular trick: corrupt a fraction of
cells by replacing each with another row's value FROM THE SAME COLUMN, then ask
the network to reconstruct the clean input. It stops the bottleneck from
learning an identity-like projection through a wide first layer.

One repo-specific note. Every feature is rank-binned WITHIN EACH DATE, so its
marginal distribution is identical on every date by construction. That makes a
donor drawn from anywhere in the fit window distributionally equivalent to one
drawn from the same date -- which is why the simple, global implementation is
correct here. It would not be in general, and a reader coming from another panel
should not copy this without checking that claim first.

------------------------------------------------------------------------------
Why the training loop is not train.train_until_early_stop
------------------------------------------------------------------------------

That function is wired to `SymbolSequenceDataset`'s five-key batch and to the
loss signature `(signal, target, reference=None, mask=None)`. Reusing it would
mean fabricating a `reference` and a `target_mask` that mean nothing for a
reconstruction, giving this network a `.use_symbol_embedding` attribute and a
second positional argument it ignores, and writing a loss that discards half of
its own contract.

And `slice_loss`, the piece with the most value over there, buys nothing here.
It exists because a correlation is not linear in its sample, so the mean of
per-batch correlations is not the correlation of the slice (fix 4 in train.py).
An MSE *is* linear in its sample. The one function whose reuse would be
principled is the one that is redundant.

`clone_state`, `set_seed` and `resolve_device` ARE imported from train.py.
`clone_state` especially: it is the fix for `state_dict()` returning live
references, and re-deriving it here would re-commit that bug.
"""

import hashlib
import io
import math

import numpy as np
import torch
import torch.nn as nn

from src.prediction.nn.train import clone_state, resolve_device, set_seed

#: Encoder widths between the input and the code. Two layers, and wide ones:
#: (64, 32) was the initial guess and a 20-trial search beat it decisively --
#: held-out R^2 0.954 against 0.904, with every one of the eleven trials at this
#: width above every trial at any other. The worry that 40k parameters would
#: overfit 22 inputs did not materialise; the tuned codes are in fact LESS
#: redundant (max code-code |corr| 0.41 against 0.58) and less of a copy of any
#: single feature (0.72 against 0.81). See data_mani/tune_autoencoder.py.
DEFAULT_HIDDEN = (256, 128)

#: A code whose raw activation varies less than this is constant, and ranking a
#: constant produces an arbitrary ordering of floating-point noise.
DEAD_CODE_STD = 1e-3


class FeatureAutoencoder(nn.Module):
    """
    A row-wise bottleneck autoencoder over one date's feature vector.

    :param input_dim: number of feature columns
    :type input_dim: int
    :param code_dim: bottleneck width, the number of columns emitted
    :type code_dim: int
    :param hidden: encoder widths; the decoder mirrors them
    :type hidden: (int, ...)
    :param dropout: dropout between hidden layers, 0 to disable
    :type dropout: float
    :raises ValueError: if code_dim is not strictly smaller than input_dim
    """

    def __init__(self, input_dim, code_dim, hidden=DEFAULT_HIDDEN, dropout=0.1):
        super().__init__()

        input_dim = int(input_dim)
        code_dim = int(code_dim)

        if code_dim < 1:
            raise ValueError("code_dim must be at least 1, got {}".format(code_dim))
        if code_dim >= input_dim:
            raise ValueError(
                "code_dim={} with {} features is not a bottleneck; pass a "
                "code_dim below {}".format(code_dim, input_dim, input_dim)
            )

        self.input_dim = input_dim
        self.code_dim = code_dim
        self.hidden = tuple(int(h) for h in hidden)
        self.dropout_rate = float(dropout)

        self.encoder = self._stack(
            [input_dim] + list(self.hidden) + [code_dim], final_activation=None
        )
        # The decoder mirrors the encoder, and ends in a sigmoid because the
        # reconstruction target is a rank bin in [0, 1].
        self.decoder = self._stack(
            [code_dim] + list(reversed(self.hidden)) + [input_dim],
            final_activation=nn.Sigmoid(),
        )

    # ----------------------------------------------------------------------- #

    def _stack(self, widths, final_activation):
        """
        A ReLU MLP over the given widths, with dropout between hidden layers.

        :param widths: layer sizes, input first and output last
        :type widths: [int]
        :param final_activation: module applied after the last Linear, or None
        :type final_activation: torch.nn.Module or None
        :return: the stack
        :rtype: torch.nn.Sequential
        """
        layers = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            last = index == len(widths) - 2
            if last:
                if final_activation is not None:
                    layers.append(final_activation)
            else:
                layers.append(nn.ReLU())
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
        return nn.Sequential(*layers)

    # ----------------------------------------------------------------------- #

    def encode(self, x):
        """
        The bottleneck activations.

        Linear -- no squashing. See the module docstring: the code is rank
        normalised downstream, so a bound here would cost gradients and would
        break what 0.5 means.

        :param x: (n, input_dim) feature rows
        :type x: torch.Tensor
        :return: (n, code_dim) activations
        :rtype: torch.Tensor
        """
        return self.encoder(x)

    def decode(self, code):
        """
        Reconstruct the input from a code.

        :param code: (n, code_dim) activations
        :type code: torch.Tensor
        :return: (n, input_dim) reconstruction in [0, 1]
        :rtype: torch.Tensor
        """
        return self.decoder(code)

    def forward(self, x):
        """
        :param x: (n, input_dim) feature rows
        :type x: torch.Tensor
        :return: (reconstruction, code)
        :rtype: (torch.Tensor, torch.Tensor)
        """
        code = self.encode(x)
        return self.decode(code), code


# --------------------------------------------------------------------------- #
# Corruption
# --------------------------------------------------------------------------- #


def apply_swap_noise(values, probability, generator):
    """
    Corrupt a batch by swapping cells with other rows' values in the same column.

    Donors are drawn WITHIN each column, never across them, so no cell ever
    receives a value its own feature could not take. That is the invariant
    `test_swap_noise_draws_donors_from_the_same_column_only` pins.

    :param values: (n, f) clean batch
    :type values: torch.Tensor
    :param probability: per-cell corruption probability, 0 to disable
    :type probability: float
    :param generator: seeded RNG, so a run reproduces
    :type generator: torch.Generator
    :return: the corrupted batch (the input is not modified)
    :rtype: torch.Tensor
    """
    if probability <= 0:
        return values

    n_rows = values.shape[0]
    if n_rows < 2:
        # With a single row every donor is the row itself, so corruption is a
        # no-op; returning early keeps randint from being called with high=1.
        return values

    donor_rows = torch.randint(
        0, n_rows, values.shape, generator=generator, device=values.device
    )
    donors = torch.gather(values, 0, donor_rows)

    mask = (
        torch.rand(values.shape, generator=generator, device=values.device)
        < probability
    )
    return torch.where(mask, donors, values)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def to_tensor(values, device):
    """
    The whole slice as one resident tensor.

    float32 explicitly: a float64 slice would otherwise reach a Linear and raise
    a dtype error several frames from the caller that produced it. Moved to the
    device once rather than per batch -- the entire crypto panel is 504k x 22
    floats, about 44 MB, so there is nothing to stream.

    :param values: (n, f) array
    :type values: np.ndarray
    :return: the tensor, on the device
    :rtype: torch.Tensor
    """
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.float32)
    return torch.from_numpy(
        np.ascontiguousarray(values, dtype="float32")).to(device)


def iter_batches(tensor, batch_size, shuffle, generator=None):
    """
    Batches of independent rows, by slicing rather than by DataLoader.

    Deliberately not the LSTM path's DataLoader. That one earns its keep on
    ragged per-symbol windows that need a custom collate; here every row is the
    same shape, and TensorDataset's per-ROW __getitem__ plus default_collate
    turned a trivial slice into thousands of Python calls per epoch. Measured on
    the real panel it dominated the run: the arithmetic for a 22-64-32-8 MLP is
    nothing next to the indexing around it.

    :param tensor: (n, f) resident rows
    :type tensor: torch.Tensor
    :param batch_size: rows per step; clamped to the slice size
    :type batch_size: int
    :param shuffle: permute the row order each pass
    :type shuffle: bool
    :param generator: seeded RNG for the permutation
    :type generator: torch.Generator or None
    :return: an iterator of (b, f) views
    :rtype: iterator
    """
    n_rows = tensor.shape[0]
    size = max(1, min(int(batch_size), n_rows))

    if shuffle:
        order = torch.randperm(n_rows, generator=generator)
    else:
        order = torch.arange(n_rows)

    for start in range(0, n_rows, size):
        yield tensor[order[start:start + size].to(tensor.device)]


def steps_per_epoch(n_rows, batch_size):
    """
    How many optimizer steps one pass costs.

    Wall-clock here is steps, not rows, which is why a budget stated in epochs
    quietly means different things at different batch sizes.

    :rtype: int
    """
    size = max(1, min(int(batch_size), int(n_rows)))
    return int(math.ceil(n_rows / size))


@torch.no_grad()
def reconstruction_mse(model, values, device, batch_size=4096):
    """
    Mean squared reconstruction error over every cell of a slice.

    Computed batchwise and accumulated as a sum, not averaged per batch. That
    happens to be exact for an MSE -- it is linear in its sample -- but writing
    it as a sum means a ragged final batch cannot quietly reweight the result.

    :param model: the network
    :type model: FeatureAutoencoder
    :param values: (n, f) rows
    :type values: np.ndarray or torch.Tensor
    :return: mean squared error over all cells
    :rtype: float
    """
    model.eval()
    tensor = to_tensor(values, device)
    total = 0.0
    cells = 0

    for batch in iter_batches(tensor, batch_size, shuffle=False):
        reconstruction, _ = model(batch)
        total += float(((reconstruction - batch) ** 2).sum().item())
        cells += batch.numel()

    return total / cells if cells else float("nan")


def run_epoch(model, tensor, batch_size, optimizer, shuffle_generator,
              swap_probability, noise_generator, max_grad_norm):
    """
    One pass over the fit rows, corrupting the input and scoring against the clean.

    :param tensor: (n, f) resident training rows
    :type tensor: torch.Tensor
    :param shuffle_generator: RNG for the row permutation
    :type shuffle_generator: torch.Generator
    :param swap_probability: per-cell swap-noise probability
    :type swap_probability: float
    :param noise_generator: RNG for the corruption
    :type noise_generator: torch.Generator
    :param max_grad_norm: clip threshold, falsy to skip
    :type max_grad_norm: float or None
    :return: mean training loss over the epoch's batches
    :rtype: float
    """
    model.train()
    total = 0.0
    batches = 0

    for clean in iter_batches(tensor, batch_size, True, shuffle_generator):
        corrupted = apply_swap_noise(clean, swap_probability, noise_generator)

        optimizer.zero_grad()
        reconstruction, _ = model(corrupted)
        # Scored against the CLEAN input: the task is denoising, not copying.
        loss = torch.nn.functional.mse_loss(reconstruction, clean)

        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total += float(loss.item())
        batches += 1

    return total / batches if batches else float("nan")


def train_autoencoder(fit_values, stop_values, config, device, verbose=True):
    """
    Fit one autoencoder until the held-out reconstruction stops improving.

    A parallel loop rather than `train.train_until_early_stop` -- see the module
    docstring for why reusing that one would mean fabricating a reference and a
    mask that mean nothing here.

    The held-out slice is a chronological tail of the fit window carved by the
    caller with NO embargo, and that is deliberate: no target is involved, and a
    feature is observable at its own date, so nothing crosses the boundary. The
    embargo rule is a target rule (see build_target_encoding), and conflating
    the two is how people talk themselves into leaks in both directions.

    :param fit_values: (n, f) float32 rows to train on
    :type fit_values: np.ndarray
    :param stop_values: (m, f) held-out rows, or None to train a fixed number of
                        epochs with no early stopping
    :type stop_values: np.ndarray or None
    :param config: keys code_dim, hidden, dropout, lr, epochs, patience,
                   batch_size, swap_noise, max_grad_norm, seed
    :type config: dict
    :param device: where to train
    :type device: torch.device
    :return: {"model", "best_state", "best_val_mse", "best_epoch", "epochs_run",
              "train_history", "val_history", "early_stopped"}
    :rtype: dict
    """
    set_seed(config["seed"])

    model = FeatureAutoencoder(
        input_dim=fit_values.shape[1],
        code_dim=config["code_dim"],
        hidden=config.get("hidden", DEFAULT_HIDDEN),
        dropout=config.get("dropout", 0.1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Two generators, both seeded: one drives the shuffle, one the corruption.
    # Keeping them apart means changing --swap-noise does not also reshuffle the
    # batches, so the two effects can be read separately across arms.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int(config["seed"]))
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(int(config["seed"]) + 1)

    fit_tensor = to_tensor(fit_values, device)
    stop_tensor = (to_tensor(stop_values, device)
                   if stop_values is not None and len(stop_values) else None)

    best_val_mse = float("inf")
    best_state = clone_state(model)   # so an un-improved run still returns weights
    best_epoch = 0
    patience_left = int(config["patience"])
    train_history, val_history = [], []
    validated = False

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = run_epoch(
            model, fit_tensor, config["batch_size"], optimizer,
            shuffle_generator, config.get("swap_noise", 0.0), noise_generator,
            config.get("max_grad_norm"),
        )
        train_history.append(train_loss)

        if stop_tensor is None:
            best_state = clone_state(model)
            best_epoch = epoch
            continue

        val_mse = reconstruction_mse(
            model, stop_tensor, device, config["batch_size"]
        )
        val_history.append(val_mse)
        validated = True

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = clone_state(model)     # cloned; see train.py bug 1
            patience_left = int(config["patience"])
        else:
            patience_left -= 1

        if verbose:
            print("  epoch {:03d} | train {:.5f} | held-out {:.5f} | "
                  "best {:.5f} @{:03d} | patience {}".format(
                      epoch, train_loss, val_mse, best_val_mse, best_epoch,
                      patience_left))

        if patience_left <= 0:
            if verbose:
                print("  early stop at epoch {}".format(epoch))
            break

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_state": best_state,
        # NaN, not the initial +inf, when no held-out slice ever ran.
        "best_val_mse": best_val_mse if validated else float("nan"),
        "best_epoch": best_epoch,
        "epochs_run": len(train_history),
        "train_history": train_history,
        "val_history": val_history,
        "early_stopped": validated,
        # Ran out of epochs rather than out of patience: the reported numbers
        # are a TRUNCATED run, not a converged one, and comparing them with a
        # control that did converge understates the network. The caller has to
        # say so out loud -- a quietly under-trained arm is exactly the kind of
        # plausible wrong number this repo refuses.
        "hit_epoch_cap": bool(validated and patience_left > 0),
    }


@torch.no_grad()
def encode_values(model, values, device, batch_size=4096):
    """
    The raw bottleneck activations for every row, in the row order given.

    :param model: a trained network
    :type model: FeatureAutoencoder
    :param values: (n, f) float32 array, already aligned to the fitted column order
    :type values: np.ndarray
    :return: (n, code_dim) activations
    :rtype: np.ndarray
    """
    model.eval()
    codes = []

    for batch in iter_batches(to_tensor(values, device), batch_size, shuffle=False):
        codes.append(model.encode(batch).cpu().numpy())

    if not codes:
        return np.zeros((0, model.code_dim), dtype="float64")
    return np.concatenate(codes, axis=0).astype("float64")


@torch.no_grad()
def reconstruct_values(model, values, device, batch_size=4096):
    """
    The reconstruction of every row, in the row order given.

    Separate from encode_values because the builder needs both and reporting an
    R^2 is not worth an inline decode in the CLI.

    :param values: (n, f) float32 array, already aligned to the fitted order
    :type values: np.ndarray
    :return: (n, f) reconstruction in [0, 1]
    :rtype: np.ndarray
    """
    model.eval()
    out = []

    for batch in iter_batches(to_tensor(values, device), batch_size, shuffle=False):
        reconstruction, _ = model(batch)
        out.append(reconstruction.cpu().numpy())

    if not out:
        return np.zeros((0, model.input_dim), dtype="float64")
    return np.concatenate(out, axis=0).astype("float64")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def state_sha1(state):
    """
    A hash of the weights, for detecting a silently rebuilt encoder.

    `registry.feature_signature` hashes column names and order, which catches a
    vocabulary change but cannot see that the WEIGHTS changed. A rebuilt encoder
    with the same column names emits different codes under an identical feature
    signature, so a cached estimator would be reused against features that had
    quietly changed meaning. This is the number that closes that hole.

    :param state: a state dict
    :type state: dict
    :return: hex digest
    :rtype: str
    """
    buffer = io.BytesIO()
    torch.save({key: state[key] for key in sorted(state)}, buffer)
    return hashlib.sha1(buffer.getvalue()).hexdigest()


def save_encoder(path, model):
    """
    Write the weights and the shape needed to rebuild the module around them.

    :param path: destination .pt file
    :type path: str
    :param model: the trained network
    :type model: FeatureAutoencoder
    :return: sha1 of the saved state
    :rtype: str
    """
    state = clone_state(model)
    torch.save(
        {
            "state_dict": state,
            "input_dim": model.input_dim,
            "code_dim": model.code_dim,
            "hidden": list(model.hidden),
            "dropout": model.dropout_rate,
        },
        path,
    )
    return state_sha1(state)


def load_encoder(path, device=None):
    """
    Rebuild a saved encoder, on the CPU unless a device is asked for.

    :param path: a .pt file written by save_encoder
    :type path: str
    :return: the network in eval mode
    :rtype: FeatureAutoencoder
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)

    model = FeatureAutoencoder(
        input_dim=payload["input_dim"],
        code_dim=payload["code_dim"],
        hidden=tuple(payload["hidden"]),
        dropout=payload["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model.to(resolve_device(device))
