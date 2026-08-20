"""
The repo's torch code: an LSTM forecaster, and a feature autoencoder.

A deliberately separate world from the sklearn wrappers in prediction/models.py.
Those are row-independent estimators driven by RandomizedSearchCV; this is a
sequence model with its own training loop and its own walk-forward, ported from
github.com/dcuoliveira/iwtsf. The two share this repo's data loading, scoring and
output conventions and nothing else.

    dataset.py   panel -> per-symbol windows, never spanning a gap in history
    model.py     the network: symbol embedding -> LSTM -> multiclass head
    losses.py    differentiable CORR / BMC / mixture, computed ex-post on a batch
    train.py     epoch loop, seed selection, walk-forward

The second thing here does not forecast anything:

    autoencoder.py   a row-wise bottleneck whose activations become FEATURES,
                     driven by data_mani/build_autoencoder.py and written into a
                     dataset that the ordinary tabular runners read unchanged

It lives in this package rather than beside its builder for one reason: torch is
an OPTIONAL dependency (see setup.sh), so nothing here is imported by
prediction/functions.py or run_forecast.py, and importing this package requires
torch. data_mani is the opposite -- src/live/features.py imports its builders at
module scope, so a top-level torch import there would take the live submission
path down on a box where the wheel failed to install.
"""
