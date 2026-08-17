"""
The torch LSTM forecasting pipeline.

A deliberately separate world from the sklearn wrappers in prediction/models.py.
Those are row-independent estimators driven by RandomizedSearchCV; this is a
sequence model with its own training loop and its own walk-forward, ported from
github.com/dcuoliveira/iwtsf. The two share this repo's data loading, scoring and
output conventions and nothing else.

    dataset.py   panel -> per-symbol windows, never spanning a gap in history
    model.py     the network: symbol embedding -> LSTM -> multiclass head
    losses.py    differentiable CORR / BMC / mixture, computed ex-post on a batch
    train.py     epoch loop, seed selection, walk-forward

torch is an OPTIONAL dependency (see setup.sh), so nothing here is imported by
prediction/functions.py or run_forecast.py. Importing this package requires torch.
"""
