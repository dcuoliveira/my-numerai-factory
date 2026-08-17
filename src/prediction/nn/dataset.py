"""
Turning the Numerai panel into LSTM windows.

The panel is (symbol, period) rows; an LSTM needs `(B, L, F)` windows of
*consecutive* periods for one symbol. This module does that conversion, and the
whole reason it is its own module with its own tests is the word "consecutive".

------------------------------------------------------------------------------
Why a row-position slicer is wrong here
------------------------------------------------------------------------------

iwtsf, which this is ported from, slices each ticker's rows by position:
`rows[start : start + seq_length]`. That is correct on its futures panel, where
every ticker has a price every trading day. It is quietly wrong on Numerai Crypto:

  * Only ~300 of 1,309 symbols are active on any given date. Symbols list,
    delist and relist.
  * **78% of symbols have at least one hole** inside their own first-to-last span;
    165,121 (symbol, date) cells are missing within spans, a median of 64 dates
    for a symbol that has any.
  * Contiguous stretches are short: 7,349 of them, **median length 4 dates**. The
    long ones belong to a minority of established coins.
  * `make_validation_split` *removes* the embargo periods outright, punching one
    more hole into every symbol at the train/validation boundary.

Slice by position through that and the LSTM is handed 84 "consecutive" timesteps
that jump across weeks or years, with no indication anything is wrong. Measured on
the real panel at seq_length=84, **13% of position-sliced windows span a gap**.

So windows are cut inside *runs* of consecutive period positions instead. Each
row's period is mapped to its index in the GLOBAL sorted period list -- global, not
per-frame, so that filtering the frame to a training block cannot make two periods
look adjacent when a month separates them. A window is emitted only when
`position[end] - position[start] == length - 1`.

------------------------------------------------------------------------------
Train mode and predict mode want different things
------------------------------------------------------------------------------

**Train** wants clean, uniform windows and does not care about covering every row:
exactly `seq_length` steps, stepping by `seq_length` (or 1 with `overlapping`), and
runs shorter than `seq_length + train_buffer` are skipped. That is iwtsf's rule.

**Predict** must cover every row of the block -- a row with no window gets no
prediction and silently vanishes from the scored set. So it additionally emits:

  * a final window *ending* at each run's last period, overlapping the previous
    one, so a run's remainder is never dropped;
  * for a run shorter than `seq_length`, one shorter window over the whole run.

Overlap is harmless because predictions are averaged per (symbol, period) anyway,
which is how the seed ensemble is already combined.

This is a deliberate departure from iwtsf, which instead shrinks `seq_length`
globally to fit the *shortest* group in the frame (its `test_flag`). On this panel
one 5-day symbol would drag every window down to 4 steps, throwing away the
history of every established coin. Padding one short run is much cheaper than
truncating all of them.

Short runs are right-padded and carry a `valid` mask. Right-padding is safe for a
unidirectional LSTM: the output at a real timestep cannot depend on padding that
comes after it.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

#: iwtsf's training-only slack on top of seq_length. Its original purpose (leaving
#: room for macro-context sampling) is gone with the attention modules, but it also
#: keeps the training set away from runs that only just reach one window, so it is
#: kept and left configurable.
DEFAULT_TRAIN_BUFFER = 20


def build_period_positions(periods):
    """
    Map every period to its index in the global chronological order.

    This is the backbone of gap detection, and it must be built ONCE over the whole
    run's periods and then reused for every train/validation/predict frame. Build it
    per-frame instead and two periods either side of a filtered-out block look
    adjacent, which is exactly the bug this module exists to prevent.

    :param periods: every period in the run, in any order
    :type periods: np.ndarray or pd.Series
    :return: {period: position}
    :rtype: dict
    """
    ordered = np.sort(np.unique(np.asarray(periods)))
    return {period: index for index, period in enumerate(ordered)}


def build_symbol_index(symbols):
    """
    Assign each training symbol an embedding row, reserving one for the rest.

    The reserved row is at index `len(symbols)`, which the network passes to
    `nn.Embedding(padding_idx=...)`. A symbol first seen at predict time therefore
    contributes a zero embedding rather than either crashing on an out-of-range
    index or borrowing some other symbol's learned identity.

    :param symbols: symbols present in the training block
    :type symbols: iterable of str
    :return: ({symbol: id}, padding_id)
    :rtype: (dict, int)
    """
    ordered = sorted(set(symbols))
    index = {symbol: position for position, symbol in enumerate(ordered)}
    return index, len(ordered)


def contiguous_runs(positions):
    """
    Split ascending period positions into maximal consecutive runs.

    :param positions: ascending, strictly increasing global period positions
    :type positions: np.ndarray
    :return: (start, stop) index pairs into `positions`, stop exclusive
    :rtype: [(int, int)]
    """
    if not len(positions):
        return []

    # A break wherever the step is not exactly 1.
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    bounds = np.concatenate(([0], breaks, [len(positions)]))
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]


def window_starts(run_length, seq_length, overlapping, mode):
    """
    Where windows begin inside one run, and how long each is.

    :param run_length: number of consecutive periods in the run
    :type run_length: int
    :param seq_length: desired window length
    :type seq_length: int
    :param overlapping: step by 1 instead of by seq_length
    :type overlapping: bool
    :param mode: "train" or "predict"
    :type mode: str
    :return: (start, length) pairs, offsets relative to the run
    :rtype: [(int, int)]
    """
    step = 1 if overlapping else seq_length

    if run_length >= seq_length:
        spans = [
            (start, seq_length)
            for start in range(0, run_length - seq_length + 1, step)
        ]
    else:
        # Too short for a full window. Training skips it; prediction takes what
        # there is, because a row with no window gets no prediction.
        spans = [] if mode == "train" else [(0, run_length)]

    if mode == "predict" and run_length >= seq_length:
        # Make sure the run's tail is covered. Anchoring the last window at the
        # run's END rather than appending a short remainder gives those rows the
        # full seq_length of context; the overlap it creates is averaged out.
        tail_start = run_length - seq_length
        if not spans or spans[-1][0] != tail_start:
            spans.append((tail_start, seq_length))

    return spans


class SymbolSequenceDataset(Dataset):
    """
    Per-symbol windows of consecutive periods, ready for a DataLoader.

    Each item is a dict of tensors, all of length L (the window's length):

        features     (L, F) float32   the model's input
        target       (L,)   float32   0 where absent; always read with target_mask
        symbol_id    (L,)   int64     embedding row; padding_id where padded
        reference    (L,)   float32   0 where absent; for BMC
        period_pos   (L,)   int64     global period position, for leakage checks
        date         (L,)   int64     epoch seconds, for reporting
        row_index    (L,)   int64     positional index into `frame`; -1 for padding
        valid        (L,)   bool      False for right-padding
        target_mask  (L,)   bool      valid AND target finite (AND reference finite,
                                      when a reference column was supplied)

    Predictions are joined back to the panel through `row_index`, not through
    (symbol, date). Every symbol absent from the training block shares one embedding
    row, so symbol_id is deliberately not a unique key at predict time.

    :param frame: rows to build from; must already be restricted to the periods
                  this dataset is allowed to see
    :type frame: pd.DataFrame
    :param features: feature column names
    :type features: [str]
    :param target_col: target column name
    :type target_col: str
    :param period_col: period column name ("date" on Crypto, "era" on Classic)
    :type period_col: str
    :param id_col: symbol column name
    :type id_col: str
    :param period_positions: from build_period_positions(), over the WHOLE run
    :type period_positions: dict
    :param symbol_index: from build_symbol_index(), over the TRAINING symbols
    :type symbol_index: dict
    :param padding_id: embedding row for unknown symbols
    :type padding_id: int
    :param seq_length: window length
    :type seq_length: int
    :param reference_col: reference column for BMC, or None
    :type reference_col: str or None
    :param overlapping: step windows by 1 instead of by seq_length
    :type overlapping: bool
    :param mode: "train" (uniform windows, rows may be dropped) or "predict"
                 (every row covered)
    :type mode: str
    :param train_buffer: extra periods a run needs beyond seq_length, train only
    :type train_buffer: int
    """

    def __init__(self, frame, features, target_col, period_col, id_col,
                 period_positions, symbol_index, padding_id, seq_length,
                 reference_col=None, overlapping=False, mode="train",
                 train_buffer=DEFAULT_TRAIN_BUFFER):
        if mode not in ("train", "predict"):
            raise ValueError("mode must be 'train' or 'predict', got {!r}".format(mode))
        if seq_length < 1:
            raise ValueError("seq_length must be at least 1, got {}".format(seq_length))

        self.seq_length = int(seq_length)
        self.mode = mode
        self.overlapping = bool(overlapping)
        self.reference_col = reference_col
        self.n_features = len(features)
        self.padding_id = int(padding_id)

        min_run = self.seq_length + (train_buffer if mode == "train" else 0)

        # Counters for the coverage report. Silent truncation is the failure mode
        # this module is guarding against, so what got dropped is reported.
        self.n_runs = 0
        self.n_runs_too_short = 0
        self.n_symbols_seen = 0
        self.n_symbols_contributing = 0
        self.n_rows_covered = 0

        self.samples = []

        working = frame[[id_col, period_col, target_col] + list(features)].copy()
        if reference_col is not None and reference_col in frame.columns:
            working[reference_col] = frame[reference_col].to_numpy()

        # Positional index into the frame as passed in, captured BEFORE the sort so
        # predictions can be joined straight back with .iloc.
        working["_row"] = np.arange(len(working), dtype="int64")

        positions = working[period_col].map(period_positions)
        if positions.isna().any():
            raise ValueError(
                "some periods are missing from period_positions; it must be built "
                "over every period in the run, not just this frame's"
            )
        working["_pos"] = positions.astype("int64")
        working["_epoch"] = (
            working[period_col].astype("int64") // 10 ** 9
            if np.issubdtype(working[period_col].dtype, np.datetime64)
            else working["_pos"]
        )

        working = working.sort_values([id_col, "_pos"])

        for symbol, group in working.groupby(id_col, sort=False):
            self.n_symbols_seen += 1

            pos = group["_pos"].to_numpy()
            feature_values = group[list(features)].to_numpy(dtype="float32")
            target_values = group[target_col].to_numpy(dtype="float32")
            epochs = group["_epoch"].to_numpy(dtype="int64")
            row_indices = group["_row"].to_numpy(dtype="int64")
            if reference_col is not None and reference_col in group.columns:
                reference_values = group[reference_col].to_numpy(dtype="float32")
            else:
                reference_values = None

            symbol_id = symbol_index.get(symbol, self.padding_id)
            contributed = False

            for run_start, run_stop in contiguous_runs(pos):
                self.n_runs += 1
                run_length = run_stop - run_start

                if run_length < min_run:
                    self.n_runs_too_short += 1
                    # A predict-mode run still gets a (shorter) window as long as it
                    # has any rows at all; only train mode really discards it.
                    if self.mode == "train":
                        continue

                for offset, length in window_starts(
                    run_length, self.seq_length, self.overlapping, self.mode
                ):
                    lo = run_start + offset
                    hi = lo + length

                    self.samples.append(
                        self._make_sample(
                            feature_values[lo:hi],
                            target_values[lo:hi],
                            pos[lo:hi],
                            epochs[lo:hi],
                            row_indices[lo:hi],
                            None if reference_values is None else reference_values[lo:hi],
                            symbol_id,
                        )
                    )
                    self.n_rows_covered += length
                    contributed = True

            if contributed:
                self.n_symbols_contributing += 1

    def _make_sample(self, features, target, positions, epochs, row_indices,
                     reference, symbol_id):
        """
        Build one item, right-padded to seq_length when the run was shorter.

        Right-padding only. A unidirectional LSTM's output at timestep t depends on
        timesteps <= t, so padding appended at the end cannot contaminate any real
        prediction. Padding at the FRONT would.

        :return: the sample dict
        :rtype: dict
        """
        length = len(target)
        pad = max(0, self.seq_length - length) if self.mode == "predict" else 0

        valid = np.zeros(length + pad, dtype=bool)
        valid[:length] = True

        def _pad(values, fill):
            if not pad:
                return values
            padding = np.full((pad,) + values.shape[1:], fill, dtype=values.dtype)
            return np.concatenate([values, padding])

        target_mask = valid.copy()
        target_mask[:length] &= np.isfinite(target)
        if reference is not None:
            target_mask[:length] &= np.isfinite(reference)

        sample = {
            # nan_to_num, not dropna: the value is never read where the mask is
            # False, but a NaN left in the tensor would still propagate through a
            # matmul and take the whole batch with it.
            "features": torch.from_numpy(
                np.nan_to_num(_pad(features, 0.0), nan=0.0)
            ),
            "target": torch.from_numpy(
                np.nan_to_num(_pad(target, 0.0), nan=0.0)
            ),
            "symbol_id": torch.from_numpy(
                _pad(np.full(length, symbol_id, dtype="int64"), self.padding_id)
            ),
            "period_pos": torch.from_numpy(_pad(positions.astype("int64"), -1)),
            "date": torch.from_numpy(_pad(epochs, 0)),
            # -1 marks padding, so a caller can filter with row_index >= 0 and never
            # accidentally write a padded prediction onto row 0 of the frame.
            "row_index": torch.from_numpy(_pad(row_indices, -1)),
            "valid": torch.from_numpy(valid),
            "target_mask": torch.from_numpy(target_mask),
        }

        if reference is None:
            sample["reference"] = torch.zeros(length + pad, dtype=torch.float32)
        else:
            sample["reference"] = torch.from_numpy(
                np.nan_to_num(_pad(reference, 0.0), nan=0.0)
            )

        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def coverage(self):
        """
        What the windowing kept and what it threw away.

        Printed at startup because every number here is a silent-truncation risk:
        a seq_length that quietly excludes most of the panel produces a model that
        trains fine and generalises to nothing.

        :return: counts describing the windowing
        :rtype: dict
        """
        return {
            "mode": self.mode,
            "seq_length": self.seq_length,
            "n_windows": len(self.samples),
            "n_runs": self.n_runs,
            "n_runs_too_short": self.n_runs_too_short,
            "n_symbols_seen": self.n_symbols_seen,
            "n_symbols_contributing": self.n_symbols_contributing,
            "n_rows_covered": self.n_rows_covered,
        }


def collate_windows(batch):
    """
    Stack windows into a batch, right-padding to the longest one present.

    Train-mode windows are all `seq_length` long so this is a plain stack; only
    predict mode produces ragged lengths.

    :param batch: samples from SymbolSequenceDataset
    :type batch: [dict]
    :return: batched tensors, same keys as a sample
    :rtype: dict
    """
    longest = max(sample["target"].shape[0] for sample in batch)

    out = {}
    for key in batch[0]:
        column = []
        for sample in batch:
            tensor = sample[key]
            shortfall = longest - tensor.shape[0]
            if shortfall:
                pad_shape = (shortfall,) + tuple(tensor.shape[1:])
                # symbol_id pads with the padding row; masks pad False; everything
                # else pads with zero. Only `valid`/`target_mask` being False
                # actually matters -- it is what keeps padding out of the loss.
                if key == "symbol_id":
                    fill = int(sample["symbol_id"][-1].item())
                    filler = torch.full(pad_shape, fill, dtype=tensor.dtype)
                elif key in ("period_pos", "row_index"):
                    filler = torch.full(pad_shape, -1, dtype=tensor.dtype)
                else:
                    filler = torch.zeros(pad_shape, dtype=tensor.dtype)
                tensor = torch.cat([tensor, filler])
            column.append(tensor)
        out[key] = torch.stack(column)

    return out
