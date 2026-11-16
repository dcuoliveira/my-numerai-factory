"""
crypto/v2.0-c2ft: v2.0 + walk-forward LoRA fine-tuned Chronos-2-Synth representations.

    python -m src.data_mani.build_c2ft              # after chronos2_ft_repr has written every block
    python -m src.data_mani.build_c2ft --audit      # point-in-time check against its --audit-cut runs

chronos2_ft_repr fine-tunes one model pair per walk-forward block b (cut at b's first train date)
and represents block b's train and test rows with it. The 768-d states of different blocks come
from different models, so their axes are unrelated, and a row in block b's test window is also in
block b+1's train window with a DIFFERENT state. Hence two files:

  blocks.parquet   (block, role, symbol, ucid, date, 32 features): what the evaluation fits on. For
                   each block and mode, a PCA basis fitted on that block's TRAIN rows only, applied
                   to its train and test rows, 16 PCs, rank-binned per date with Numerai's rule.
  train.parquet    v2.0 + each row's TEST-role features (the block that predicts it): the
                   deployable view, one value per row, and what routing reads for coverage. Rows
                   outside every modelled test window (before 2021-08-27, blocks 0-1, after 2026-08-20) are NaN.

Blocks 0 and 1 have no model (block 0: no label before its train window; block 1: about six months
of labels, short of the fine-tune's 134-step minimum), so they have no rows here.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_mass
from src.data_mani.history_features import bin_ties_preserved

RAW = "/scratch/danielco/c2ft"
SOURCE = os.path.join("data", "crypto", "v2.0")
DEST = os.path.join("data", "crypto", "v2.0-c2ft")
MODES = ("cov", "ar")
N_PCS = 16
TRAIN, PURGE, BLOCK, FIRST = 130, 22, 130, 172
TEST_START, LAST = pd.Timestamp("2026-02-20"), pd.Timestamp("2026-08-20")


def blocks(grid):
    """The E133 windows; identical to chronos2_ft_repr.blocks (that module needs the chronos venv's imports only lazily)."""
    past = grid[grid < np.datetime64(TEST_START)]
    n = len(past)
    starts = list(range(FIRST, n - BLOCK + 1, BLOCK))
    assert len(starts) == 11 and starts[-1] + BLOCK == n, (starts, n)
    out = [(i, past[b - PURGE - TRAIN:b - PURGE], past[b:b + BLOCK]) for i, b in enumerate(starts)]
    test = grid[(grid >= np.datetime64(TEST_START)) & (grid <= np.datetime64(LAST))]
    return out + [(11, past[-(PURGE + TRAIN):-PURGE], test)]


def keys(raw=None):
    k = pd.read_parquet(os.path.join(raw or RAW, "rows.parquet")).reset_index(drop=True)
    k["date"] = pd.to_datetime(k["date"])
    return k


def block_states(b, mode, raw=None):
    d = os.path.join(raw or RAW, "block_%02d" % b)
    return np.load(os.path.join(d, "%s_rows.npy" % mode)), np.load(os.path.join(d, "%s_reps.npy" % mode)).astype("float32")


def reduce_block(k, b, tr, te, raw=None):
    """One block's binned 32 PCs for its train and test rows."""
    dates = k["date"].to_numpy()
    rows = np.flatnonzero(np.isin(dates, np.concatenate([tr, te])))
    out = k.loc[rows, ["symbol", "ucid", "date"]].copy()
    out["block"], out["role"] = b, np.where(np.isin(out["date"].to_numpy(), te), "test", "train")
    info = {}
    for m in MODES:
        ids, x = block_states(b, m, raw)
        assert len(np.unique(ids)) == len(ids), "row twice in block %d %s" % (b, m)
        full = np.full((len(k), x.shape[1]), np.nan, "float32")
        full[ids] = x
        x = full[rows]
        fit = np.isfinite(x[:, 0]) & (out["role"].to_numpy() == "train")
        centre = x[fit].mean(0)
        _, sv, vt = np.linalg.svd(x[fit] - centre, full_matrices=False)
        pcs = (x - centre) @ vt[:N_PCS].T
        frame = pd.DataFrame(pcs, columns=["feature_c2ft%s_pc%02d" % (m, j + 1) for j in range(N_PCS)], index=out.index)
        binned = bin_ties_preserved(frame, out["date"])
        for c in binned:
            out[c] = binned[c].to_numpy("float32")
        info[m] = dict(fit_rows=int(fit.sum()), share=round(float((sv[:N_PCS] ** 2).sum() / (sv ** 2).sum()), 4),
                       observed_train=round(float(np.isfinite(x[out["role"].to_numpy() == "train", 0]).mean()), 4),
                       observed_test=round(float(np.isfinite(x[out["role"].to_numpy() == "test", 0]).mean()), 4))
    return out, info


def build(raw=None, dest=DEST):
    k = keys(raw)
    grid = np.sort(k["date"].unique())
    parts, report = [], {}
    for b, tr, te in blocks(grid):
        if not os.path.exists(os.path.join(raw or RAW, "block_%02d" % b, "ar_reps.npy")):
            print("block %d: no states%s" % (b, "" if b <= 1 else "  <-- MISSING"))
            report[b] = None
            continue
        out, info = reduce_block(k, b, tr, te, raw)
        fj = os.path.join(raw or RAW, "block_%02d" % b, "fit.json")
        if os.path.exists(fj):
            info["fit"] = {m: {x: v for x, v in z.items() if x in ("cut", "newest_label", "series_train", "eval")} for m, z in json.load(open(fj)).items()}
        report[b] = info
        parts.append(out)
        print("block %d: train %s..%s test %s..%s, %s" % (b, str(tr[0])[:10], str(tr[-1])[:10], str(te[0])[:10], str(te[-1])[:10],
              ", ".join("%s %d%% var, observed %.0f%%/%.0f%%" % (m, 100 * info[m]["share"], 100 * info[m]["observed_train"], 100 * info[m]["observed_test"]) for m in MODES)))
    long = pd.concat(parts, ignore_index=True)
    new = [c for c in long if c.startswith("feature_c2ft")]
    os.makedirs(dest, exist_ok=True)
    long.to_parquet(os.path.join(dest, "blocks.parquet"), index=False)
    test = long[long["role"] == "test"].drop(columns=["block", "role", "symbol"])
    base = pd.read_parquet(os.path.join(SOURCE, "train.parquet")).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    merged = base.merge(test, on=["ucid", "date"], how="left", validate="1:1")
    drift = max(float(np.nanmax(np.abs(bin_mass(merged[c]) - BIN_MASS))) for c in new)
    merged.set_index("symbol").to_parquet(os.path.join(dest, "train.parquet"))
    shutil.copy(os.path.join(SOURCE, "historical_meta_models.parquet"), os.path.join(dest, "historical_meta_models.parquet"))
    json.dump(report, open(os.path.join(dest, "build.json"), "w"), indent=1, default=str)
    print("blocks.parquet %d rows (%d blocks); train.parquet %d rows, %d features, %.1f%% of rows with a test-role state, max bin-mass drift %.4f"
          % (len(long), long["block"].nunique(), len(merged), sum(c.startswith("feature_") for c in merged),
             100 * merged[new[0]].notna().mean(), drift))


def audit(raw=None):
    """
    Each --audit-cut date recomputed from a panel cut at it with its block's saved (fp16) models.
    differ/max_abs_diff: against the same model on the uncut panel -- must be 0 if nothing past the wall
    leaks. stored_*: against the dataset's states, which the in-memory fp32 merge produced, so they
    differ by the fp16 weight rounding only.
    """
    k = keys(raw)
    k["i"] = np.arange(len(k))
    rows = []
    for p in sorted(glob.glob(os.path.join(raw or RAW, "audit", "cov_*.parquet"))):
        cut = os.path.basename(p)[4:-8]
        for m in MODES:
            c = pd.read_parquet(os.path.join(raw or RAW, "audit", "%s_%s.parquet" % (m, cut)))
            c["date"] = pd.to_datetime(c["date"])
            b = int(c["block"].iloc[0])
            a = np.load(os.path.join(raw or RAW, "audit", "%s_%s.npy" % (m, cut))).astype("float32")
            ids, x = block_states(b, m, raw)
            full = np.full((len(k), x.shape[1]), np.nan, "float32")
            full[ids] = x
            day = k.loc[k["date"] == pd.Timestamp(cut), "i"].to_numpy()
            s = full[c.merge(k, on=["ucid", "date"], how="left")["i"].to_numpy()]
            # same reloaded (fp16-stored) model on the uncut panel: the leakage test proper
            fp = os.path.join(raw or RAW, "audit", "%s_%s_full.npy" % (m, cut))
            f = np.load(fp).astype("float32") if os.path.exists(fp) else np.full_like(a, np.nan)
            cos = (a * s).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(s, axis=1)
            rows.append(dict(date=cut, mode=m, block=b, rows_on_date=len(day), states_cut=len(a),
                             states_full=int(np.isfinite(full[day, 0]).sum()), cut_rows_without_full_state=int(np.isnan(s[:, 0]).sum()),
                             cells=int(a.size), differ=int((a != f).sum()), max_abs_diff=float(np.nanmax(np.abs(a - f))),
                             stored_differ=int((a != s).sum()), stored_max_abs_diff=float(np.nanmax(np.abs(a - s))),
                             stored_cosine_min=float(np.nanmin(cos)), stored_scale=float(np.nanmean(np.abs(s)))))
            print(rows[-1], flush=True)
    fits = []
    for fj in sorted(glob.glob(os.path.join(raw or RAW, "block_*", "fit.json"))):
        for m, z in json.load(open(fj)).items():
            lag = (pd.Timestamp(z["cut"]) - pd.Timestamp(z["newest_label"])).days
            fits.append(dict(block=int(fj.split("_")[-1][:2]), mode=m, cut=z["cut"], newest_label=z["newest_label"], lag_days=lag))
    assert all(f["lag_days"] >= 35 for f in fits), fits
    return dict(truncation=rows, finetune_cuts=fits)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    print(audit()) if a.audit else build()


if __name__ == "__main__":
    main()
