"""
crypto/v2.0-c2syn: v2.0 + Chronos-2-Synth pre-head representations, reduced and rank-binned.

    python -m src.data_mani.build_c2syn              # after chronos2_syn_repr has written every part
    python -m src.data_mani.build_c2syn --audit      # point-in-time check against --audit-cut runs

Reads the raw 768-d states chronos2_syn_repr wrote to /scratch -- one per panel row and mode
(cov: label + 22 features as covariates, ar: label alone) -- and turns each mode into N_PCS
features:

  1. a PCA basis fitted ONCE on the rows of the first BURNIN dates that carry a state, and applied
     to every row. The burn-in rows are then blanked: their basis saw dates up to BURNIN later, so
     only rows after it are strictly past-only.
  2. each PC rank-binned within date with Numerai's exact rule (history_features.bin_ties_preserved,
     the one that reproduces all 38,720 shipped cross-sections), onto {0, .25, .5, .75, 1}.

Columns feature_c2cov_pc01..16 and feature_c2ar_pc01..16, NaN where the row had no state (too few
resolved labels, or burn-in); the loader fills those with 0.5. Every v2.0 row is kept, the 22
shipped features and the targets are untouched.
"""

import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_mass
from src.data_mani.history_features import bin_ties_preserved

RAW = "/scratch/danielco/c2syn"
SOURCE = os.path.join("data", "crypto", "v2.0")
DEST = os.path.join("data", "crypto", "v2.0-c2syn")
MODES = ("cov", "ar")
N_PCS = 16
BURNIN = 60
DIM = 768


def states(raw=None):
    """(row keys, {mode: (n_rows, 768) float32 with NaN rows where no state})."""
    raw = raw or RAW
    keys = pd.read_parquet(os.path.join(raw, "rows.parquet")).reset_index(drop=True)
    out = {m: np.full((len(keys), DIM), np.nan, "float32") for m in MODES}
    seen = {m: np.zeros(len(keys), bool) for m in MODES}
    parts = sorted(glob.glob(os.path.join(raw, "part_*.npz")))
    for p in parts:
        z = np.load(p)
        for m in MODES:
            rows = z["%s_rows" % m]
            assert not seen[m][rows].any(), "row written twice: %s" % p
            seen[m][rows] = True
            out[m][rows] = z["%s_reps" % m].astype("float32")
    print("%d part files, %d rows; states: %s" % (len(parts), len(keys),
          ", ".join("%s %d" % (m, seen[m].sum()) for m in MODES)))
    return keys, out


def burnin_basis(x, dates):
    """Centre and top-N_PCS axes of the rows on the first BURNIN dates with a state; those dates."""
    have = ~np.isnan(x[:, 0])
    first = np.sort(np.unique(dates[have]))[:BURNIN]
    fit = have & np.isin(dates, first)
    centre = x[fit].mean(axis=0)
    _, sv, vt = np.linalg.svd(x[fit] - centre, full_matrices=False)
    share = (sv[:N_PCS] ** 2).sum() / (sv ** 2).sum()
    return centre, vt[:N_PCS], first, share


def reduce(keys, out):
    """{column: float32 binned values} for both modes."""
    dates = keys["date"].to_numpy()
    cols = {}
    for m in MODES:
        x = out[m]
        centre, axes, first, share = burnin_basis(x, dates)
        pcs = (x - centre) @ axes.T
        pcs[np.isin(dates, first)] = np.nan
        frame = pd.DataFrame(pcs, columns=["feature_c2%s_pc%02d" % (m, k + 1) for k in range(N_PCS)])
        binned = bin_ties_preserved(frame, dates)
        print("%s: burn-in %s..%s (%d dates, %d rows), %d PCs hold %.1f%% of burn-in variance, %.1f%% of rows observed"
              % (m, str(first[0])[:10], str(first[-1])[:10], len(first), int(np.isin(dates, first).sum()),
                 N_PCS, 100 * share, 100 * binned.iloc[:, 0].notna().mean()))
        cols.update({c: binned[c].to_numpy("float32") for c in binned})
    return cols


def build():
    keys, out = states()
    keys["date"] = pd.to_datetime(keys["date"])
    cols = reduce(keys, out)
    block = keys[["ucid", "date"]].assign(**cols)
    base = pd.read_parquet(os.path.join(SOURCE, "train.parquet")).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    merged = base.merge(block, on=["ucid", "date"], how="left", validate="1:1")
    missing = len(base) - base.merge(keys[["ucid", "date"]], on=["ucid", "date"]).shape[0]
    print("v2.0: %d rows, %d not in the extraction's panel (kept, block NaN)" % (len(base), missing))
    new = [c for c in merged if c.startswith("feature_c2")]
    drift = max(float(np.nanmax(np.abs(bin_mass(merged[c]) - BIN_MASS))) for c in new)
    print("%d block columns, max bin-mass drift %.4f" % (len(new), drift))
    os.makedirs(DEST, exist_ok=True)
    dest = os.path.join(DEST, "train.parquet")
    merged.set_index("symbol").to_parquet(dest)
    shutil.copy(os.path.join(SOURCE, "historical_meta_models.parquet"), os.path.join(DEST, "historical_meta_models.parquet"))
    n_src = sum(c.startswith("feature_") for c in base)
    print("wrote %d rows, %d features (%d shipped + %d c2syn) -> %s (%.0f MB)"
          % (len(merged), n_src + len(new), n_src, len(new), dest, os.path.getsize(dest) / 1e6))


def audit():
    """
    Compare each --audit-cut date's states, computed from a panel cut at that date with every label
    after the wall blanked, against the full run's states for the same rows: they must agree to
    numeric noise if the full run read nothing past the wall.
    """
    keys, out = states()
    keys["date"] = pd.to_datetime(keys["date"])
    keys["i"] = np.arange(len(keys))
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "audit", "cov_*.parquet"))):
        cut = os.path.basename(p)[4:-8]
        full = keys.loc[keys["date"] == pd.Timestamp(cut), "i"].to_numpy()
        for m in MODES:
            k = pd.read_parquet(os.path.join(RAW, "audit", "%s_%s.parquet" % (m, cut)))
            k["date"] = pd.to_datetime(k["date"])
            a = np.load(os.path.join(RAW, "audit", "%s_%s.npy" % (m, cut))).astype("float32")
            b = out[m][k.merge(keys, on=["ucid", "date"], how="left")["i"].to_numpy()]
            cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
            rows.append(dict(date=cut, mode=m, rows_on_date=len(full), states_cut=len(a),
                             states_full=int(np.isfinite(out[m][full, 0]).sum()),
                             cut_rows_without_full_state=int(np.isnan(b[:, 0]).sum()),
                             cells=int(a.size), differ=int((a != b).sum()),
                             cosine_min=float(np.nanmin(cos)), max_abs_diff=float(np.nanmax(np.abs(a - b)))))
            print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    audit() if a.audit else build()


if __name__ == "__main__":
    main()
