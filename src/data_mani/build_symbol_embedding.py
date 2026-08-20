"""
Build a Crypto dataset with the LSTM's learned symbol embeddings as features.

    python3 -m src.run_nn_forecast crypto --export-embeddings --run-name my_lstm ...
    python3 -m src.data_mani.build_symbol_embedding --export models/embeddings/my_lstm.parquet

Writes data/crypto/<version>-emb/train.parquet, used by passing

    --data-version crypto/v2.0-emb

to src.run_forecast. Nothing in the modelling code changes: the crypto loader picks
features up by the "feature_" prefix, so the new columns are found the same way the
original 22 are -- the same trick build_onehot and build_target_encoding use.

------------------------------------------------------------------------------
What this is, and what the repo already says about the idea
------------------------------------------------------------------------------

This is the third and most expensive way to hand a model "which symbol is this":

    build_onehot.py           740 dummies; the model estimates the effect itself
    build_target_encoding.py  ONE causal mean-target column, handed over directly
    (this)                    an embed_dim-wide vector, learned end-to-end by the LSTM

The per-symbol effect is real -- build_onehot measured per-symbol mean targets at
5.85x the dispersion noise would produce, persisting at 0.309 between halves. But
on this data both existing encodings have so far *cost* ridge accuracy rather than
added it: best CORR is 0.173 on raw v2.0, 0.156 with -tenc, 0.142 with -onehot.
More symbol dimensions has meant worse, and an embedding sits at the expensive end
of that range. Treat a run of this as a hypothesis to falsify cheaply, not as an
improvement to bank.

------------------------------------------------------------------------------
Avoiding leakage
------------------------------------------------------------------------------

An embedding is trained AGAINST THE TARGET, so it leaks far more readily than a
one-hot -- a symbol's vector is a compressed summary of how that symbol's target
behaved. Two constraints, matching build_target_encoding exactly:

1. STRICTLY PAST DATA. The vector attached to (symbol s, date t) comes from a
   network whose training data ended before t. A row never contributes to its own
   feature.

2. THE TARGET RESOLUTION LAG. A 20-day forward target is not observable until 20
   periods later, so a network trained through date d has really only seen targets
   resolved by d. The vector is therefore only used from `d + lag` onward.

Combined:  embedding(s, t) = vectors from the latest export with train_end <= t - lag

The LSTM walk-forward already enforces `train_end = block_start - embargo`, so a
block's own rows satisfy this by construction. The rule is applied here anyway
rather than trusting where the file came from -- an export produced with a
different embargo, or hand-edited, would otherwise pass silently.

------------------------------------------------------------------------------
Coverage, and why early rows get zeros
------------------------------------------------------------------------------

Rows before the first usable export have no honest vector and get zeros -- the same
value the network itself uses for an unknown symbol, which is the truthful encoding
of "nothing is known about this symbol yet". That is a real limitation, not a
rounding detail: with the default --refit-every 260 the LSTM only produces an export
once a year, so the early history of the panel is all zeros and ridge will fit its
coefficients mostly on the tail.

If that bites, re-run the LSTM with a smaller --refit-every to get more export
points. The number of usable vectors is printed below so the trade is visible.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.utils import (
    default_embargo,
    download_if_missing,
    get_api,
    get_spec,
    local_path,
)
from src.prediction.nn.embeddings import (
    FEATURE_PREFIX,
    embedding_columns,
    load_embeddings,
    select_best_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add LSTM symbol-embedding features to the Crypto dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--export", required=True,
                        help="embedding parquet written by "
                             "run_nn_forecast --export-embeddings")
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default="-emb")
    parser.add_argument("--seed", type=int, default=None,
                        help="use this seed's embeddings; default is the "
                             "best-validation seed of each block")
    parser.add_argument("--dims", type=int, default=0,
                        help="keep only the first N embedding dimensions; 0 keeps "
                             "all. Fewer dimensions is the obvious lever if the "
                             "full width overfits, as -onehot did")
    return parser.parse_args()


def attach_embeddings(df, export, symbol_col, period_col, lag, dims=0):
    """
    Attach each row the newest embedding that was legal to know at that date.

    Implemented as a merge_asof on `train_end` rather than a loop: for every row it
    finds the most recent export at or before `date - lag`, which is exactly the
    leakage rule in the module docstring.

    :param df: the source panel
    :type df: pd.DataFrame
    :param export: one seed per block, from load_embeddings()/select_best_seed()
    :type export: pd.DataFrame
    :param symbol_col: symbol column name
    :type symbol_col: str
    :param period_col: period column name
    :type period_col: str
    :param lag: the target's resolution horizon, in periods
    :type lag: int
    :param dims: keep only the first N dimensions; 0 keeps all
    :type dims: int
    :return: (frame with feature columns added, number of rows that got a vector)
    :rtype: (pd.DataFrame, int)
    """
    columns = embedding_columns(export)
    if dims:
        columns = columns[:dims]

    periods = np.sort(df[period_col].unique())
    position = pd.Series(np.arange(len(periods)), index=periods)

    # The cutoff each row is allowed to see: `lag` periods before its own date.
    # Done in period POSITIONS, not calendar days, because the panel's dates are
    # not evenly spaced and the embargo is defined in periods.
    row_position = df[period_col].map(position).to_numpy()
    cutoff_position = row_position - lag
    # Positions below zero have no legal cutoff at all; NaT keeps them out of the
    # merge instead of silently clamping them to the first period.
    cutoff = pd.Series(
        np.where(cutoff_position >= 0,
                 periods[np.clip(cutoff_position, 0, len(periods) - 1)],
                 np.datetime64("NaT")),
        index=df.index,
    )

    left = df[[symbol_col]].copy()
    left["_cutoff"] = cutoff
    left["_row"] = np.arange(len(left))
    left = left.dropna(subset=["_cutoff"]).sort_values("_cutoff")

    right = export[[symbol_col, "train_end"] + columns].sort_values("train_end")

    merged = pd.merge_asof(
        left, right,
        left_on="_cutoff", right_on="train_end",
        by=symbol_col, direction="backward",
    )

    out = df.copy()
    values = np.zeros((len(df), len(columns)), dtype="float32")
    got = merged[columns].notna().all(axis=1).to_numpy()
    values[merged.loc[got, "_row"].to_numpy()] = (
        merged.loc[got, columns].to_numpy(dtype="float32")
    )

    for index, column in enumerate(columns):
        out["{}_{}".format(FEATURE_PREFIX, index)] = values[:, index]

    return out, int(got.sum())


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")

    export = load_embeddings(args.export)
    if args.seed is not None:
        export = export[export["seed"] == args.seed]
        if export.empty:
            raise SystemExit(
                "seed {} is not in {}; it has {}".format(
                    args.seed, args.export, sorted(export["seed"].unique()))
            )
    else:
        export = select_best_seed(export)

    blocks = export[["block", "seed", "train_end"]].drop_duplicates()
    print("export: %d blocks, %d symbols, %d dimensions" % (
        len(blocks), export["symbol"].nunique(), len(embedding_columns(export))))
    for _, row in blocks.sort_values("block").iterrows():
        print("  block %d: seed %d, trained through %s" % (
            row["block"], row["seed"], pd.Timestamp(row["train_end"]).date()))

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[spec["target"]]).reset_index(drop=True)

    lag = default_embargo("crypto", spec["target"])
    print("\nsource: %d rows, %d symbols, %d dates" % (
        len(df), df["symbol"].nunique(), df["date"].nunique()))
    print("resolution lag: %d periods (a %s target is not observable until then)" % (
        lag, spec["target"]))

    df, covered = attach_embeddings(
        df, export, "symbol", "date", lag=lag, dims=args.dims,
    )

    print("rows with a usable embedding: %.1f%% (the rest get zeros -- "
          "'nothing known about this symbol yet')" % (100 * covered / len(df)))
    if covered == 0:
        raise SystemExit(
            "no row could be given an embedding: every export's train_end is later "
            "than the rows it would apply to. Re-run the LSTM with a smaller "
            "--refit-every so it exports earlier in the panel."
        )

    n_feat = sum(c.startswith("feature_") for c in df.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.set_index("symbol").to_parquet(dest)
    print("features: %d total -> %s (%.0f MB)" % (
        n_feat, dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("compare against:  --data-version %s   (raw, currently the stronger "
          "baseline at 0.173 CORR)" % args.source_version)


if __name__ == "__main__":
    main()
