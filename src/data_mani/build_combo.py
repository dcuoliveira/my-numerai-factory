"""
Assemble a COMBINED dataset by unioning feature blocks that already exist in other
built variants, then apply the fillmask logic (missing -> 0.5, present kept).

    python3 -m src.data_mani.build_combo \
        --spine crypto/v2.0-vol \
        --add crypto/v2.0-infl:^feature_macro_bei_ \
        --add crypto/v2.0-momswap-macro-carver:^feature_tf_ \
        --out-version crypto/v2.0-vol-infl-carver-fillmask

Why a union is valid here (no re-computation). Every block we pull from was already
built the right way: computed backward-looking per symbol, merged onto the panel
POINT-IN-TIME (as-of <= D-1), and rank-binned per date onto Numerai's
{0,.25,.5,.75,1} scale (see build_momentum / build_macro / build_trend / build_vol /
build_inflation). So a column carries the same meaning wherever it lives; taking it
by name and left-joining it onto a spine on (symbol, date) reproduces exactly the
value that block's own dataset holds for that row. This module does NOT invent any
feature -- it only selects, joins, and fills.

The spine provides the base features + target + the (symbol, date) universe; each
--add contributes the columns of one block, matched by a regex on its source
dataset. A spine row absent from a block's source is left NaN and then, like any
other missing value, set to the neutral 0.5 -- which is the whole point of the
fillmask logic: "any data we already have missing, and the signals of symbols we do
not have all signals for, become neutral 0.5; everything present is kept."

The reference / meta-model file travels with the output, same as every builder.
"""

import argparse
import os
import re
import shutil

import pandas as pd

from src.data_mani.build_mommask import apply_fillmask
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

NEUTRAL = 0.5


def parse_args():
    p = argparse.ArgumentParser(
        description="Union feature blocks from existing datasets, then fillmask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--spine", default="crypto/v2.0-vol",
                   help="dataset that provides the base features, target and the "
                        "(symbol,date) universe; its feature columns are all kept")
    p.add_argument("--add", action="append", default=[], metavar="VERSION:REGEX",
                   help="add the columns of a block: a source dataset version and a "
                        "regex selecting its feature columns, joined by ':'. Repeatable")
    p.add_argument("--out-version", required=True,
                   help="output dataset version, e.g. crypto/v2.0-vol-infl-carver-fillmask")
    p.add_argument("--mode", choices=["fill", "none"], default="fill",
                   help="'fill' applies the fillmask (missing feature cells -> "
                        "fill-value); 'none' leaves NaN for the loader to fill")
    p.add_argument("--fill-value", type=float, default=NEUTRAL)
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    id_col, period_col = spec["id_col"], spec["period_col"]

    # ---- spine: base + target + universe ---------------------------------- #
    src = download_if_missing(api, "crypto", args.spine, "train.parquet")
    combo = pd.read_parquet(src).reset_index()
    combo[period_col] = pd.to_datetime(combo[period_col])
    n_rows = len(combo)
    spine_feats = [c for c in combo.columns if c.startswith("feature_")]
    print("spine: %d rows, %d symbols, %d features (%s)"
          % (n_rows, combo[id_col].nunique(), len(spine_feats), args.spine))

    have = set(combo.columns)
    added_cols = []
    # ---- each block: select by regex, left-join on (symbol,date) ----------- #
    for spec_str in args.add:
        if ":" not in spec_str:
            raise SystemExit("--add must be VERSION:REGEX, got %r" % spec_str)
        version, rx = spec_str.split(":", 1)
        rx_c = re.compile(rx)
        bsrc = download_if_missing(api, "crypto", version, "train.parquet")
        bdf = pd.read_parquet(bsrc).reset_index()
        bdf[period_col] = pd.to_datetime(bdf[period_col])
        cols = [c for c in bdf.columns if c.startswith("feature_") and rx_c.match(c)]
        if not cols:
            raise SystemExit("regex %r matched no feature columns in %s" % (rx, version))
        clash = [c for c in cols if c in have]
        if clash:
            raise SystemExit("block %s cols already present in the spine/earlier "
                             "block: %s" % (version, clash))
        block = bdf[[id_col, period_col] + cols].drop_duplicates([id_col, period_col])
        combo = combo.merge(block, on=[id_col, period_col], how="left", validate="m:1")
        if len(combo) != n_rows:
            raise SystemExit("join with %s changed row count %d -> %d"
                             % (version, n_rows, len(combo)))
        cov = combo[cols].notna().all(axis=1).mean()
        print("  + %-42s %3d cols  (regex %s)  coverage %5.1f%% of spine rows"
              % (version, len(cols), rx, 100 * cov))
        have.update(cols)
        added_cols += cols

    feats = [c for c in combo.columns if c.startswith("feature_")]
    # ---- fillmask: missing feature cells -> neutral, present kept ---------- #
    if args.mode == "fill":
        n_missing = int(combo[feats].isna().sum().sum())
        rows_aff = int(combo[feats].isna().any(axis=1).sum())
        combo = apply_fillmask(combo, feats, fill_value=args.fill_value)
        print("mode=fill: filled %d missing feature values on %d/%d rows (%.1f%%) "
              "with %s; present values kept"
              % (n_missing, rows_aff, n_rows, 100 * rows_aff / n_rows, args.fill_value))

    # ---- write + carry the meta-model file -------------------------------- #
    dest = local_path("crypto", args.out_version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    combo.set_index(id_col).to_parquet(dest)
    print("features: %d total (%d spine + %d added) -> %s (%.0f MB)"
          % (len(feats), len(spine_feats), len(added_cols), dest,
             os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.spine, filename)
        target = local_path("crypto", args.out_version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % args.out_version)


if __name__ == "__main__":
    main()
