"""
Tests for the autoencoder hyperparameter search.

A tuner has a distinctive failure mode: it produces a confident answer whatever
you do to it. So these pin the things that would let it answer the wrong
question convincingly --

    * a ranking slice that overlaps what the trials trained or early-stopped on,
      which turns the search into a memorisation contest
    * a search over a knob that is monotone in the objective, which reads the
      monotonicity back out and calls it a tuned value
    * a budget stated in epochs while batch size is searched, which hands small
      batches several times the compute and then reports that they win

Everything runs offline in milliseconds; no trial is actually trained here. Both
torch and optuna are optional, and this module skips cleanly without either.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="torch is an optional dependency")
optuna = pytest.importorskip("optuna", reason="optuna is an optional dependency")

from src.data_mani import tune_autoencoder as tuner  # noqa: E402
from src.prediction.nn.autoencoder import steps_per_epoch  # noqa: E402


def _args(**overrides):
    """The parsed defaults, with any override applied."""
    args = tuner.parse_args([])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _trial(**params):
    """A trial that answers every suggest_* with a value you chose."""
    return optuna.trial.FixedTrial(params)


# --------------------------------------------------------------------------- #
# Where the trials are allowed to look
# --------------------------------------------------------------------------- #

def test_the_three_tuning_slices_are_disjoint_and_cover_the_fit_window():
    periods = np.arange(1000)

    inner, stop, tune = tuner.three_way_split(periods, 0.15, 0.15)

    assert len(inner) + len(stop) + len(tune) == 1000
    assert not set(inner) & set(stop)
    assert not set(inner) & set(tune)
    assert not set(stop) & set(tune)


def test_the_slices_are_chronological_with_the_ranking_slice_last():
    """
    A trial is ranked on the most recent data available to it -- while still
    sitting inside the pipeline's train block, which is the property that keeps
    selecting on these numbers from inflating a backtest.
    """
    inner, stop, tune = tuner.three_way_split(np.arange(1000), 0.15, 0.15)

    assert inner.max() < stop.min()
    assert stop.max() < tune.min()
    assert tune.max() == 999


def test_a_split_that_leaves_nothing_to_train_on_is_refused():
    with pytest.raises(ValueError, match="nothing left to train on"):
        tuner.three_way_split(np.arange(100), 0.5, 0.5)

    with pytest.raises(ValueError, match="at least one period"):
        tuner.three_way_split(np.arange(100), 0.0, 0.15)


def test_the_fit_window_the_tuner_uses_is_the_builders_own():
    """
    Same function, so the two cannot drift. If the tuner computed its own
    boundary, a change to one would silently leave the other tuning on rows the
    build treats as validation.
    """
    from src.data_mani.build_autoencoder import fit_boundary

    assert tuner.fit_boundary is fit_boundary


# --------------------------------------------------------------------------- #
# What is searched, and what is deliberately not
# --------------------------------------------------------------------------- #

def test_code_dim_and_swap_noise_are_taken_from_the_args_not_the_trial():
    """
    Both are monotone in held-out reconstruction -- a wider bottleneck always
    reconstructs better, and denoising always reconstructs worse -- so searching
    them would return the extremes and call it tuning.
    """
    trial = _trial(hidden="64,32", batch_size=4096, dropout=0.1, lr=1e-3)

    config = tuner.suggest_config(trial, _args(code_dim=5, swap_noise=0.25),
                                  n_fit_rows=100000)

    assert config["code_dim"] == 5
    assert config["swap_noise"] == 0.25
    assert "code_dim" not in trial.params
    assert "swap_noise" not in trial.params


def test_the_searched_space_is_exactly_the_four_knobs_with_an_interior_optimum():
    trial = _trial(hidden="64,32", batch_size=4096, dropout=0.1, lr=1e-3)
    tuner.suggest_config(trial, _args(), n_fit_rows=100000)

    assert set(trial.params) == {"hidden", "dropout", "lr", "batch_size"}


# --------------------------------------------------------------------------- #
# The budget
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("batch_size", tuner.BATCH_CHOICES)
def test_every_batch_size_gets_the_same_number_of_gradient_steps(batch_size):
    """
    The reason the budget is steps and not epochs. At a fixed epoch count, batch
    1024 takes four times the steps of batch 4096 -- so an epoch budget would
    hand small batches four times the compute and then report that small batches
    win.
    """
    rows = 239596
    config = tuner.suggest_config(
        _trial(hidden="64,32", batch_size=batch_size, dropout=0.1, lr=1e-3),
        _args(tune_steps=6000), n_fit_rows=rows)

    steps = config["epochs"] * steps_per_epoch(rows, batch_size)
    assert abs(steps - 6000) < steps_per_epoch(rows, batch_size)


def test_the_epoch_count_is_derived_and_never_suggested():
    trial = _trial(hidden="64,32", batch_size=4096, dropout=0.1, lr=1e-3)
    config = tuner.suggest_config(trial, _args(tune_steps=6000),
                                  n_fit_rows=239596)

    assert "epochs" not in trial.params
    assert config["epochs"] == round(6000 / steps_per_epoch(239596, 4096))


def test_patience_scales_with_the_trials_own_epoch_budget():
    """
    Otherwise a fixed patience is generous at one batch size and unreachable at
    another, and early stopping becomes a function of a knob it should be
    independent of.
    """
    rows = 239596
    patiences = {}
    for batch_size in tuner.BATCH_CHOICES:
        config = tuner.suggest_config(
            _trial(hidden="64,32", batch_size=batch_size, dropout=0.1, lr=1e-3),
            _args(tune_steps=6000, patience_frac=0.15), n_fit_rows=rows)
        patiences[batch_size] = config["patience"] / config["epochs"]

    # Every trial is equally forgiving, relative to its own budget.
    assert max(patiences.values()) - min(patiences.values()) < 0.05


def test_a_trial_always_gets_at_least_one_epoch_and_some_patience():
    config = tuner.suggest_config(
        _trial(hidden="64,32", batch_size=8192, dropout=0.1, lr=1e-3),
        _args(tune_steps=1), n_fit_rows=100)

    assert config["epochs"] >= 1
    assert config["patience"] >= 5


def test_the_hidden_string_becomes_the_tuple_the_network_wants():
    config = tuner.suggest_config(
        _trial(hidden="128,64,32", batch_size=4096, dropout=0.1, lr=1e-3),
        _args(), n_fit_rows=100000)

    assert config["hidden"] == (128, 64, 32)
