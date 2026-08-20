"""
Exporting the learned symbol embeddings so other models can use them.

The network learns an `nn.Embedding` row per symbol -- a dense vector standing for
"what this symbol tends to do", learned end to end against the objective. This
module gets those vectors out of a trained network and onto disk, so
`data_mani/build_symbol_embedding.py` can fold them into a dataset that ridge,
lgb or anything else can read.

It is the learned cousin of two things this repo already builds:

    build_onehot.py           740 symbol dummies -- the model estimates the effect
    build_target_encoding.py  one causal mean-target column -- handed to it directly
    (this)                    an embed_dim-wide vector, learned by the LSTM

------------------------------------------------------------------------------
Why the export is PER SEED and never averaged across them
------------------------------------------------------------------------------

The obvious move -- average the 10 selected seeds' embedding matrices into one,
the way their predictions are averaged -- is wrong, and quietly so.

Predictions from different seeds are comparable because they all live in the same
output space: a number whose ordering means the same thing whichever seed produced
it. Embeddings are not. Each seed initialises its embedding table randomly and the
network is free to use any basis for it, so seed A's dimension 3 and seed B's
dimension 3 have no relationship at all. The two matrices differ by an arbitrary
rotation (and permutation, and sign flip) of a space whose axes carry no fixed
meaning. Averaging across that cancels the structure and converges toward zero as
the seed count grows -- the more seeds you average, the *less* signal you keep.

So every seed's matrix is stored separately, tagged with the seed and block, and
the consumer picks one. `build_symbol_embedding.py` defaults to the seed with the
best validation loss.

------------------------------------------------------------------------------
The leakage rule this file is responsible for
------------------------------------------------------------------------------

Each exported matrix is tagged with `train_end`: the last period whose data the
network was allowed to see. A downstream feature for (symbol s, date t) is only
honest if it comes from a matrix with

    train_end <= t - lag

where `lag` is the target's resolution horizon, exactly as
`build_target_encoding.causal_target_encoding` requires. The walk-forward already
enforces `train_end = block_start - embargo`, so a block's own rows satisfy it by
construction -- but the tag is written out anyway, because the builder applies the
rule itself rather than trusting where the numbers came from.
"""

import os

import pandas as pd


#: Column prefix for the exported vectors. The crypto loader discovers features by
#: the "feature_" prefix, so the built dataset needs no loader changes -- the same
#: trick build_onehot and build_target_encoding rely on.
FEATURE_PREFIX = "feature_sym_emb"


def extract_embedding(model, symbol_index, block, seed, train_end, val_loss):
    """
    Pull one network's symbol embedding matrix into a tidy frame.

    The padding row (unknown symbols) is deliberately NOT exported. It is a
    structural zero the network never trains -- `nn.Embedding(padding_idx=...)`
    holds its gradient at zero -- so writing it out would put a row of zeros in the
    table that looks like a learned vector for a symbol called "unknown".

    :param model: a trained LSTMForecastNetwork with embeddings enabled
    :type model: LSTMForecastNetwork
    :param symbol_index: {symbol: embedding row}, as passed to the dataset
    :type symbol_index: dict
    :param block: walk-forward block index this network was trained for
    :type block: int
    :param seed: the seed that produced it
    :type seed: int
    :param train_end: last period the network was allowed to see
    :type train_end: str
    :param val_loss: its best validation loss, for picking between seeds later
    :type val_loss: float
    :return: one row per symbol, columns block/seed/symbol/train_end/val_loss/emb_*
    :rtype: pd.DataFrame
    :raises ValueError: if the network has no embedding table
    """
    if not model.use_symbol_embedding or model.symbol_embedding is None:
        raise ValueError(
            "this network was built with embed_dim=0, so it has no symbol "
            "embedding to export"
        )

    weights = model.symbol_embedding.weight.detach().cpu().numpy()
    symbols = sorted(symbol_index, key=lambda name: symbol_index[name])
    rows = weights[[symbol_index[name] for name in symbols]]

    frame = pd.DataFrame(
        rows, columns=["emb_{}".format(i) for i in range(rows.shape[1])]
    )
    frame.insert(0, "symbol", symbols)
    frame.insert(0, "train_end", str(train_end))
    frame.insert(0, "seed", int(seed))
    frame.insert(0, "block", int(block))
    frame["val_loss"] = float(val_loss)
    return frame


def save_embeddings(path, frames):
    """
    Append embedding frames to the export file.

    Written incrementally, one block at a time, for the same reason the seed
    checkpoint is: a 13-hour run that dies in the last block should not take every
    earlier block's embeddings with it.

    :param path: destination parquet file
    :type path: str
    :param frames: frames from extract_embedding()
    :type frames: [pd.DataFrame]
    """
    if not frames:
        return

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    combined = pd.concat(frames, ignore_index=True)
    if os.path.exists(path):
        combined = pd.concat([pd.read_parquet(path), combined], ignore_index=True)

    # Re-written whole rather than appended in place: parquet has no cheap append,
    # and the table is a few thousand rows, so simplicity wins over cleverness.
    combined.to_parquet(path, index=False)


def load_embeddings(path):
    """
    Read an export back, with its columns validated.

    :param path: the parquet file written by save_embeddings()
    :type path: str
    :return: the export
    :rtype: pd.DataFrame
    :raises ValueError: if it is missing the columns the builder needs
    """
    frame = pd.read_parquet(path)

    required = {"block", "seed", "symbol", "train_end", "val_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "{} is not a symbol-embedding export: missing {}".format(
                path, sorted(missing))
        )
    if not embedding_columns(frame):
        raise ValueError("{} has no emb_* columns".format(path))

    frame["train_end"] = pd.to_datetime(frame["train_end"])
    return frame


def embedding_columns(frame):
    """
    The emb_* columns, in dimension order.

    Sorted numerically, not lexically: plain sorted() puts emb_10 before emb_2 and
    silently permutes the dimensions, which would still "work" and quietly scramble
    the feature space.

    :param frame: an embedding export
    :type frame: pd.DataFrame
    :return: column names, dimension 0 first
    :rtype: [str]
    """
    columns = [c for c in frame.columns if c.startswith("emb_")]
    return sorted(columns, key=lambda name: int(name.split("_")[1]))


def select_best_seed(frame):
    """
    Keep one seed per block: the one with the lowest validation loss.

    Averaging across seeds is not an option here -- see the module docstring for
    why the embedding spaces are not comparable.

    :param frame: an embedding export covering one or more seeds per block
    :type frame: pd.DataFrame
    :return: the export with a single seed per block
    :rtype: pd.DataFrame
    """
    # NaN val_loss means the block had no early-stopping slice, so nothing
    # distinguishes its seeds; idxmin skips NaN and falls back to the first.
    best = (frame.groupby("block")["val_loss"]
            .apply(lambda s: s.idxmin() if s.notna().any() else s.index[0]))
    keep = frame.loc[best.values, ["block", "seed"]]
    return frame.merge(keep, on=["block", "seed"], how="inner").reset_index(drop=True)
