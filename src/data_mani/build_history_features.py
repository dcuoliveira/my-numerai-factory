"""
Build crypto/v2.0-histfe: the shipped features plus 352 own-history features, from
src/data_mani/history_features.py (22 features x 5 ranks, 5 MA differences, 5
days-since-max and 1 days-since-change).

    python3 -m src.data_mani.build_history_features

Run it with:  --data-version crypto/v2.0-histfe
Isolate the block with:  --feature-include '^feature_hfe_'

Every row of the source stays, including tokens that later left the universe: the
block needs no external data, so no coverage snapshot can decide which rows exist.
Live, the same add_history_features runs on train + live history + today's rows
(feature set "histfe" in src/live/features.py).
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_LABELS, BIN_MASS
from src.data_mani.history_features import add_history_features, history_columns
from src.data_mani.utils import download_if_missing, get_api, get_spec, local_path


def verify_shipped_encoding(base, feats, period):
    """Share of (date, feature) cross-sections whose bin counts are exactly what
    bin_ties_preserved's (rank-1)/(n-1) cuts give, the convention the block uses."""
    hits, total = 0, 0
    for f in feats:
        counts = (base.groupby(period)[f].value_counts().unstack()
                  .reindex(columns=BIN_LABELS.astype("float32")).fillna(0).to_numpy().astype(int))
        n = counts.sum(axis=1)
        cuts = [(n - 1) // 20 + 1, (n - 1) // 4 + 1, 3 * (n - 1) // 4 + 1, 19 * (n - 1) // 20 + 1]
        want = np.column_stack([cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1],
                                cuts[3] - cuts[2], n - cuts[3]])
        hits += int((counts == want).all(axis=1).sum())
        total += len(n)
    return hits, total


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-histfe")
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    period = spec["period_col"]
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base[period] = pd.to_datetime(base[period])
    feats = [c for c in base.columns if c.startswith("feature_")]
    print("source: %d rows, %d tokens (ucid), %d features, %s to %s (%s)"
          % (len(base), base["ucid"].nunique(), len(feats), base[period].min().date(),
             base[period].max().date(), args.source_version))

    hits, total = verify_shipped_encoding(base, feats, period)
    print("shipped encoding: %d of %d date-feature cross-sections (%.2f%%) match "
          "the (rank-1)/(n-1) cuts exactly" % (hits, total, 100 * hits / total))

    new = add_history_features(base, feats, "ucid", period)
    groups = history_columns(feats)
    print("added %d features: %s" % (new.shape[1], {k: len(v) for k, v in groups.items()}))
    for name, cols in groups.items():
        values = new[cols].to_numpy().ravel()
        values = values[~np.isnan(values)]
        mass = np.array([(values == b).mean() for b in BIN_LABELS])
        print("  %-6s present %5.1f%%  mass %s  (target %s)"
              % (name, 100 * new[cols].notna().to_numpy().mean(),
                 np.round(mass, 3), BIN_MASS))

    out = pd.concat([base, new], axis=1)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index("symbol").to_parquet(dest)
    print("wrote %d rows, %d features -> %s (%.0f MB)"
          % (len(out), len(feats) + new.shape[1], dest, os.path.getsize(dest) / 1e6))
    for filename in ("historical_meta_models.parquet",):
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(download_if_missing(api, "crypto", args.source_version, filename), target)
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
