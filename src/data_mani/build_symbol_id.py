"""
Build a Crypto dataset carrying the ticker as ONE integer code.

    python3 -m src.data_mani.build_symbol_id

Writes data/crypto/<version>-symid/train.parquet, used by passing

    --data-version crypto/v2.0-symid

The column is `feature_symbol_id`. It is a code, not a quantity: nothing about
BTC being 87 and ETH being 141 says one is larger than the other, and any model
that treats the column as continuous is reading noise. It exists so an estimator
with an EMBEDDING TABLE can look the ticker up -- see prediction/torch_mlp.py.

------------------------------------------------------------------------------
Why this, when build_onehot and build_symbol_embedding already exist
------------------------------------------------------------------------------

Four ways to hand a model "which symbol is this", and they differ in who
estimates the per-ticker effect and from how much data:

    build_target_encoding   ONE causal mean-target column, handed over directly
    build_onehot            740 dummies; the model estimates each effect itself
    build_symbol_embedding  an embed_dim-wide vector, learned by the LSTM in a
                            SEPARATE run and frozen into the dataset as features
    (this)                  the raw code, so the model can learn the vector
                            ITSELF, jointly with the rest of its weights

The third and fourth are the same object arrived at differently. Freezing an
embedding trained elsewhere means it was fitted for another objective on another
architecture; learning it jointly means the vectors are whatever helps THIS
model's loss. That is the whole point of the comparison this column enables.

Against the one-hot, the difference is parameter sharing. A one-hot into a dense
layer IS an embedding lookup -- the layer's weight column for ticker j is its
vector -- but sklearn cannot regularise that block separately, updates all 740
columns on every step whether or not the batch contains them, and has no row for
a ticker it never saw. An nn.Embedding has all three.

------------------------------------------------------------------------------
The codes are assigned over the WHOLE panel, and stay fixed
------------------------------------------------------------------------------

Sorted by symbol name, so the mapping is deterministic and reproducible from the
data alone -- rerunning the builder cannot silently renumber anything, which
would invalidate every fitted embedding table against it.

Assigning codes over the whole panel is NOT lookahead. The code is an identity,
not a statistic: it encodes no information about the target, no ordering, and
nothing that varies over time. Contrast build_feature_encoding, whose
whole-sample variant IS lookahead precisely because its value is a target
summary. A model still only sees the codes present in its own training block,
and what it does with a code it has never seen is the estimator's problem --
prediction/torch_mlp.py gives it a dedicated padding row.
"""

import argparse
import os
import shutil

import pandas as pd

from src.data_mani.utils import (
    download_if_missing,
    get_api,
    get_spec,
    local_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add an integer ticker code to the Crypto dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-version", type=str, default="crypto/v2.0",
                        help="dataset to read")
    parser.add_argument("--out-suffix", type=str, default="-symid",
                        help="appended to the source version for the output")
    parser.add_argument("--min-dates", type=int, default=0,
                        help="symbols seen on fewer dates than this share the "
                             "single code 0, rather than each getting their own. "
                             "0 keeps every symbol distinct. Raise it if a long "
                             "tail of one-day tickers is spending embedding rows "
                             "on vectors that can never be estimated")
    return parser.parse_args()


def assign_codes(symbols, min_dates_by_symbol=None, min_dates=0):
    """
    Map each symbol to an integer, sorted by name for determinism.

    :param symbols: the symbol column
    :type symbols: pd.Series
    :param min_dates_by_symbol: date counts per symbol, needed only when
                                min_dates > 0
    :type min_dates_by_symbol: pd.Series or None
    :param min_dates: symbols below this many dates collapse to code 0
    :type min_dates: int
    :return: (codes aligned to `symbols`, the mapping)
    :rtype: (pd.Series, dict)
    """
    distinct = sorted(symbols.unique())
    if min_dates > 0:
        if min_dates_by_symbol is None:
            raise ValueError("min_dates > 0 needs min_dates_by_symbol")
        rare = {s for s in distinct if min_dates_by_symbol.get(s, 0) < min_dates}
    else:
        rare = set()

    # Code 0 is reserved for the collapsed tail, so the useful codes start at 1
    # whether or not any symbol is actually collapsed. A mapping whose meaning
    # depends on --min-dates would make two builds incomparable.
    mapping = {}
    nxt = 1
    for symbol in distinct:
        if symbol in rare:
            mapping[symbol] = 0
        else:
            mapping[symbol] = nxt
            nxt += 1
    return symbols.map(mapping).astype("int32"), mapping


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[spec["target"]]).reset_index(drop=True)

    print("source: %d rows, %d symbols, %d dates" % (
        len(df), df["symbol"].nunique(), df["date"].nunique()))

    dates_by_symbol = df.groupby("symbol")["date"].nunique()
    codes, mapping = assign_codes(df["symbol"], dates_by_symbol, args.min_dates)
    df["feature_symbol_id"] = codes

    collapsed = sum(1 for v in mapping.values() if v == 0)
    print("codes: %d distinct (%d symbols collapsed into code 0 at --min-dates %d)"
          % (df["feature_symbol_id"].nunique(), collapsed, args.min_dates))
    print("date counts per symbol: min %d, median %d, max %d"
          % (dates_by_symbol.min(), dates_by_symbol.median(), dates_by_symbol.max()))
    # The number that decides whether an embedding can be estimated at all: a
    # ticker present on 3 dates of a 130-period training block contributes 3 rows
    # to its own vector.
    thin = (dates_by_symbol < 30).sum()
    print("symbols on fewer than 30 dates: %d of %d (%.0f%%) -- their vectors are "
          "estimated from very little" % (thin, len(dates_by_symbol),
                                          100.0 * thin / len(dates_by_symbol)))

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
    print("NOTE feature_symbol_id is a CODE. Any model that reads it as a")
    print("continuous feature is reading noise -- exclude it with")
    print("--feature-exclude 'feature_symbol_id' unless the estimator embeds it.")


if __name__ == "__main__":
    main()
