"""
E109's builder: the foundation-model representation block merged onto a base dataset.

    python -m src.data_mani.build_fmrep --base all

Reads what foundation_repr (TabPFN, rows 1-7 of experiments/features-foundation.md)
and foundation_repr_ts (TimesFM / Chronos-2, rows 8-9) cached under
data/crypto/fmrep_cache/, turns it into 41 `feature_fm_*` columns, merges them onto
each base, rank-bins them per date over that base's own cross-section, and writes

    raw         crypto/v2.0-fmrep                                       22 + 41
    fenccausal  crypto/v2.0-fenccausal-fmrep                            44 + 41
    allfeat_pp  crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmrep  812 + 41

The block, by row of the note:

    1  feature_fm_tp_{mean,median,mode}        TabPFN point predictions
    2  feature_fm_tp_{std,iqr,skew}            its predictive distribution
    3  feature_fm_tp_p{0..4}                   P(target bin), integrated from it
    4  feature_fm_tp_disagree                  std of the 4 estimators' means
    5  feature_fm_tp_emb_pc{00..07}            test-token embedding, 8 PCs
    6  feature_fm_head_{tenc,interact,size,price,raw}   TabPFN on one block alone
    7  feature_fm_ctx{030,390}                 TabPFN under a 30 / 390-date context
    8  feature_fm_{tsfm,chr}_{q50,spread,skew} 20-day return quantiles, TimesFM / Chronos-2
    9  feature_fm_chr_emb_pc{00..07}           Chronos-2 encoder embedding, 8 PCs

Every arm compares a base against itself: run the dataset whole for "with", and
with --feature-exclude '^feature_fm_' for "without". Same rows, same dates, the
block the only difference.

------------------------------------------------------------------------------
Why each merge is point in time
------------------------------------------------------------------------------
- TabPFN rows (1-7) are a SAME-DATE join on (symbol, date). Their inputs are
  Numerai's own features at D, which live at D's cutoff, and their context ends
  20 dates before the 20-date refresh block holding D (foundation_repr), so every
  label they read had resolved by D.
- TimesFM / Chronos-2 rows (8-9) were built from closes dated <= D-1, newest
  within 4 days (foundation_repr_ts), so they join on (ucid, date) as they are.
- Both embedding bases are fitted once on a burn-in window that ends before the
  first date any walk-forward trains on; main() checks it.

------------------------------------------------------------------------------
One spine for all three
------------------------------------------------------------------------------
The fenccausal and allfeat_pp datasets end 2026-07-08; v2.0 has since grown to
2026-09-17. The TabPFN inputs exist only to the earlier date, so the raw variant
is cut there too, and all three score the same 510 periods. Its "without" arm is
therefore raw v2.0 on 1,701 dates, not e72_*_base on 1,752.

Not live-wired: src/live/features.py runs neither TabPFN nor a TS model.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import bin_cross_sectional, bin_mass
from src.data_mani.foundation_repr import AFPP, CACHE_DIR, EMBARGO, FENC, PASSES, WINDOW
from src.data_mani.foundation_repr_ts import BURNIN, burnin_pca
from src.data_mani.utils import download_if_missing, get_api, local_path

BASES = {"raw": "crypto/v2.0", "fenccausal": FENC, "allfeat_pp": AFPP}
PREFIX = "feature_fm_"
N_PCS = 8
VAL_FRAC = 0.3


def tabpfn_block():
    """Rows 1-7 on (symbol, date), raw (unbinned) values; and the burn-in end."""
    main = pd.read_parquet(os.path.join(CACHE_DIR, "main.parquet"))
    dates = np.sort(main["date"].unique())
    burnin = dates[:BURNIN]

    out = main[["symbol", "date"]].copy()
    for c in ("mean", "median", "mode", "std"):
        out[PREFIX + "tp_" + c] = main["tp_" + c]
    out[PREFIX + "tp_iqr"] = main["tp_q75"] - main["tp_q25"]
    width = (main["tp_q90"] - main["tp_q10"]).replace(0, np.nan)
    out[PREFIX + "tp_skew"] = (main["tp_q90"] + main["tp_q10"] - 2 * main["tp_q50"]) / width
    for k in range(5):
        out[PREFIX + "tp_p%d" % k] = main["tp_p%d" % k]
    est = [c for c in main.columns if c.startswith("tp_est")]
    out[PREFIX + "tp_disagree"] = main[est].std(axis=1, ddof=0)
    emb = main[[c for c in main.columns if c.startswith("emb_")]].to_numpy(dtype="float64")
    pcs = burnin_pca(emb, main["date"].to_numpy(), burnin, N_PCS)
    for j in range(N_PCS):
        out[PREFIX + "tp_emb_pc%02d" % j] = pcs[:, j]

    for name in PASSES:
        if name == "main":
            continue
        f = pd.read_parquet(os.path.join(CACHE_DIR, name + ".parquet"), columns=["symbol", "date", name])
        out = out.merge(f.rename(columns={name: PREFIX + name}), on=["symbol", "date"], how="left")
    return out, burnin[-1]


def ts_block():
    """Rows 8-9 on (ucid, date), raw values."""
    out = None
    for model, tag in (("timesfm", "tsfm"), ("chronos2", "chr")):
        f = pd.read_parquet(os.path.join(CACHE_DIR, "ts_%s.parquet" % model))
        g = f[["ucid", "date"]].copy()
        g[PREFIX + tag + "_q50"] = f[tag + "_q50"]
        g[PREFIX + tag + "_spread"] = f[tag + "_q90"] - f[tag + "_q10"]
        g[PREFIX + tag + "_skew"] = ((f[tag + "_q90"] + f[tag + "_q10"] - 2 * f[tag + "_q50"])
                                     / (f[tag + "_q90"] - f[tag + "_q10"]).replace(0, np.nan))
        for j in range(N_PCS):
            column = "%s_emb_pc%02d" % (tag, j)
            if column in f.columns:
                g[PREFIX + column] = f[column]
        out = g if out is None else out.merge(g, on=["ucid", "date"], how="outer")
    return out


def untie(df, cols, key, threshold=0.05):
    """
    Replace each tie-heavy column by its within-date order on (column, key).

    rank_to_bins breaks ties by row order, which is noise on a column that takes
    a handful of values per date -- TabPFN's mode lands on ~4 bucket centres, and
    a head on 4 five-valued inputs repeats itself. Ordering a tie by the TabPFN
    mean instead is deterministic and never reorders two distinct values, so the
    column's own ranking survives and the bin masses stay exact.

    :return: (df with those columns replaced by ordinal ranks, {column: tie share})
    """
    shares = {}
    for c in cols:
        present = df[c].notna()
        if not present.any():
            continue
        share = 1 - df.loc[present].groupby("date")[c].nunique().sum() / present.sum()
        if share <= threshold:
            continue
        shares[c] = share
        sub = df.loc[present, ["date", c, key]].assign(_k=lambda f: f[key].fillna(f[c]))
        order = sub.sort_values(["date", c, "_k"], kind="mergesort")
        df.loc[order.index, c] = order.groupby("date").cumcount().astype("float64")
    return df, shares


def first_training_date(periods):
    """The earliest date any shared-protocol walk-forward trains on."""
    n_val = int(len(periods) * VAL_FRAC)
    return periods[len(periods) - n_val - EMBARGO - WINDOW]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="all", help="raw, fenccausal, allfeat_pp or all")
    args = parser.parse_args()
    bases = list(BASES) if args.base == "all" else args.base.split(",")

    tp, burnin_end = tabpfn_block()
    ts = ts_block()
    api = get_api("crypto")
    end = pd.read_parquet(local_path("crypto", FENC, "train.parquet"), columns=["date"])["date"].max()

    for base in bases:
        source = BASES[base]
        df = pd.read_parquet(download_if_missing(api, "crypto", source, "train.parquet")).reset_index()
        df = df[df["date"] <= end].reset_index(drop=True)
        periods = np.sort(df["date"].unique())
        first_train = first_training_date(periods)
        if not burnin_end < first_train:
            raise SystemExit("burn-in ends %s, not before the first training date %s"
                             % (burnin_end, first_train))

        merged = df.merge(tp, on=["symbol", "date"], how="left")
        merged = merged.merge(ts, on=["ucid", "date"], how="left")
        cols = [c for c in merged.columns if c.startswith(PREFIX)]
        merged, ties = untie(merged, cols, PREFIX + "tp_mean")
        merged = bin_cross_sectional(merged, cols, "date")
        # The "without" arms of fenccausal and allfeat_pp run on the base dataset
        # itself; that is only the same arm if the variant minus the block IS it.
        pd.testing.assert_frame_equal(merged[df.columns], df)

        live = merged["date"] >= first_train
        print("%s: %s, %d dates to %s; block %d columns; first training date %s, burn-in ends %s"
              % (base, source, len(periods), pd.Timestamp(end).date(), len(cols),
                 pd.Timestamp(first_train).date(), pd.Timestamp(burnin_end).date()))
        for c in cols:
            mass = bin_mass(merged[c])
            drift = np.abs(mass - [.05, .20, .50, .20, .05]).max()
            print("  %-30s coverage %5.1f%% (from first training date)  mass drift %.4f%s"
                  % (c, 100 * merged.loc[live, c].notna().mean(), drift,
                     "  untied (%.0f%% ties, ordered by tp_mean)" % (100 * ties[c]) if c in ties else ""))

        version = source + "-fmrep"
        dest = local_path("crypto", version, "train.parquet")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        merged.set_index("symbol").to_parquet(dest)
        n_feat = sum(c.startswith("feature_") for c in merged.columns)
        print("  -> %s: %d features (%d base + %d block), %d rows, %.0f MB"
              % (dest, n_feat, n_feat - len(cols), len(cols), len(merged), os.path.getsize(dest) / 1e6))
        for filename in ("historical_meta_models.parquet",):
            target = local_path("crypto", version, filename)
            if not os.path.exists(target):
                shutil.copy(download_if_missing(api, "crypto", source, filename), target)
        print("  run with:  --data-version %s   (without: --feature-exclude '^%s')" % (version, PREFIX))


if __name__ == "__main__":
    main()
