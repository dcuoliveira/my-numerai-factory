"""
Pre-run EDA of foundation-model outputs as candidate ridge features.

    python -m src.analysis.foundation_features_eda [--repr e19_fenccausal]

For every candidate column (experiments/features-foundation.md, section 2) it asks
the questions that decide whether the column is worth a ridge study, one score
per date and Newey-West (lag 20) across dates:

    ic        numerai_corr(column, target) -- signed; a spread or a tail
              probability may carry its signal with either sign
    rho_tp    mean within-date Spearman with the TabPFN mean (E19), which the
              blends already hold
    rho_rdg   the same against the fenccausal ridge (ridge/e10_ridge_fenc), the
              model the column would be added to
    r2_44     mean within-date R^2 of the column on the 44 fenccausal inputs --
              how much of it a linear model of the ridge's own inputs reproduces
    pic_44    IC of the column's residual on the 44 inputs, per date
    pic_rdg   IC of its residual on the ridge's prediction
    pic_r+tp  IC of its residual on the ridge AND the TabPFN mean -- what is new
              beyond both

Every column is gaussian-rank transformed within each date before any of this, so
a heavy tail cannot dominate a regression. The residualisations never see the
target, so a partial IC is not fitted to it. Candidates come from the forecast
CSVs on disk (the means) and, with --repr, from the parquet that
src.analysis.tabpfn_representations writes (everything else).
"""

import argparse
import os

import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr
from scipy.stats import norm

from src.analysis.cross_target import newey_west
from src.data_mani.utils import load_tournament_data

TARGET = "target_binned_return_20"
PERIOD, ID = "date", "symbol"
TABPFN_MEAN = "tabpfn/e19_fenccausal"
RIDGE = "ridge/e10_ridge_fenc"

#: On-disk point forecasts: rows 1, 6 and 7 of the note, plus the null TS models.
ON_DISK = [
    TABPFN_MEAN, RIDGE,
    "tabpfn/e24_reg_full_fenccausal", "tabpfn/e24_temp050_fenccausal",
    "tabpfn/e24_temp075_fenccausal", "tabpfn/e24_temp125_fenccausal",
    "tabpfn_clf/e24_clf_fenccausal",
    "tabpfn/e19_raw", "tabpfn/e84_zs_tenc", "tabpfn/e84_zs_interact",
    "tabpfn/e84_zs_size", "tabpfn/e84_zs_price", "tabpfn/e84_zs_onehot",
    "timesfm/e38_target", "chronos2/e56_fenccausal",
]
N_PCS = 16


def gauss_rank(frame, columns):
    """Within-date gaussianised ranks, ties averaged; NaN stays NaN."""
    grouped = frame.groupby(PERIOD)[columns]
    ranked = grouped.rank(method="average").to_numpy()
    counts = grouped.transform("count").to_numpy()      # non-missing, per column
    return pd.DataFrame(norm.ppf((ranked - 0.5) / counts),
                        index=frame.index, columns=columns)


def derived_columns(rep):
    """Summaries of the distributional block that a ridge could take directly."""
    est = [c for c in rep.columns if c.startswith("tp_est")]
    p = rep[["tp_p%d" % k for k in range(5)]].to_numpy(dtype="float64")
    out = pd.DataFrame(index=rep.index)
    out["tp_spread90"] = rep["tp_q90"] - rep["tp_q10"]
    out["tp_spread60"] = rep["tp_q80"] - rep["tp_q20"]
    out["tp_skew"] = ((rep["tp_q90"] + rep["tp_q10"] - 2 * rep["tp_q50"])
                      / (rep["tp_q90"] - rep["tp_q10"]).replace(0, np.nan))
    out["tp_ptail"] = rep["tp_p4"] - rep["tp_p0"]
    out["tp_ptails"] = rep["tp_p4"] + rep["tp_p0"]
    out["tp_entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
    out["tp_ebin"] = p @ np.array([0, .25, .5, .75, 1.0])
    out["tp_disagree"] = rep[est].std(axis=1, ddof=0)
    return out


def embedding_pcs(rep):
    """Top principal components of the embedding, fitted within each refit block.

    The PCA sees only that block's test-token embeddings -- features, no labels --
    which is the in-window fit the note's width constraint asks for.
    """
    emb = [c for c in rep.columns if c.startswith("emb_")]
    out = np.full((len(rep), N_PCS), np.nan)
    for _, idx in rep.groupby("refit").indices.items():
        x = rep[emb].to_numpy(dtype="float64")[idx]
        x = x - x.mean(axis=0)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        out[idx] = x @ vt[:N_PCS].T
    return pd.DataFrame(out, index=rep.index, columns=["emb_pc%02d" % k for k in range(N_PCS)])


def per_date_scores(frame, cand, inputs, controls):
    """One row per date: ic, rho_tp, rho_rdg, r2_44, pic_44, pic_rdg, pic_r+tp."""
    rows = {}
    for date, g in frame.groupby(PERIOD):
        g = g.dropna(subset=[cand, TARGET])
        if len(g) < 60:
            continue
        x = g[cand].to_numpy()
        target = g[TARGET]
        X44 = np.column_stack([np.ones(len(g)), g[inputs].to_numpy()])
        fit44 = X44 @ np.linalg.lstsq(X44, x, rcond=None)[0]
        resid = {"pic_44": x - fit44}
        for name, cols in controls.items():
            Xc = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in cols])
            resid[name] = x - Xc @ np.linalg.lstsq(Xc, x, rcond=None)[0]
        row = {
            "ic": numerai_corr(g[[cand]], target).iloc[0],
            "rho_tp": np.corrcoef(x, g[TABPFN_MEAN])[0, 1],
            "rho_rdg": np.corrcoef(x, g[RIDGE])[0, 1],
            "r2_44": 1 - ((x - fit44) ** 2).sum() / ((x - x.mean()) ** 2).sum(),
        }
        for name, r in resid.items():
            row[name] = (numerai_corr(pd.DataFrame({"r": r}, index=g.index), target).iloc[0]
                         if r.std() > 1e-9 else np.nan)
        rows[date] = row
    return pd.DataFrame(rows).T


def summarise(scores):
    out = {"n": len(scores)}
    for col in ["ic", "pic_44", "pic_rdg", "pic_r+tp"]:
        mean, t = newey_west(scores[col].dropna())
        out[col], out["t_" + col] = mean, t
    for col in ["rho_tp", "rho_rdg", "r2_44"]:
        out[col] = scores[col].mean()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repr", default=None,
                        help="run name under results/cache/tabpfn_repr/ to add rows 1-5")
    parser.add_argument("--out", default=os.path.join("results", "cache", "tabpfn_repr", "eda.csv"))
    args = parser.parse_args()

    df, features = load_tournament_data(
        tournament="crypto", version="crypto/v2.0-fenccausal", val_frac=0.3,
        embargo=20, verbose=False, target=TARGET)
    frame = df[[ID, "ucid", PERIOD, TARGET] + features]

    for run in ON_DISK:
        path = os.path.join("results", "forecast", "crypto", run + ".csv")
        f = pd.read_csv(path, usecols=[ID, PERIOD, "prediction"], parse_dates=[PERIOD])
        # The time-series runs (E38, E56) wrote the CMC id into their symbol column.
        # The panel stores ucid as a string ('1' = BTC).
        key = "ucid" if pd.api.types.is_integer_dtype(f[ID]) else ID
        f = f.rename(columns={ID: key, "prediction": run})
        f[key] = f[key].astype(str)
        frame = frame.merge(f, on=[key, PERIOD], how="left")
    frame = frame[frame[TABPFN_MEAN].notna() & frame[RIDGE].notna()].reset_index(drop=True)
    candidates = [r for r in ON_DISK if r != RIDGE]

    if args.repr:
        rep = pd.read_parquet(os.path.join("results", "cache", "tabpfn_repr", args.repr + ".parquet"))
        rep[PERIOD] = pd.to_datetime(rep[PERIOD])
        keep = ["tp_median", "tp_mode", "tp_std"] + \
               ["tp_q%02d" % q for q in (10, 50, 90)] + ["tp_p%d" % k for k in range(5)] + \
               [c for c in rep.columns if c.startswith("tp_est")]
        extra = pd.concat([rep[[ID, PERIOD] + keep], derived_columns(rep), embedding_pcs(rep)], axis=1)
        frame = frame.merge(extra, on=[ID, PERIOD], how="left")
        candidates += [c for c in extra.columns if c not in (ID, PERIOD)]

    # Gaussian ranks for everything that enters a regression; the target and
    # the 44 binned inputs are left as they are.
    ranked_cols = [c for c in candidates + [RIDGE] if frame[c].notna().any()]
    frame[ranked_cols] = gauss_rank(frame, ranked_cols)
    controls = {"pic_rdg": [RIDGE], "pic_r+tp": [RIDGE, TABPFN_MEAN]}

    table = []
    for cand in candidates:
        if frame[cand].notna().sum() == 0:
            continue
        s = summarise(per_date_scores(frame, cand, features, controls))
        s["column"] = cand
        table.append(s)
        print("%-34s ic %+.4f (t %+5.2f)  pic44 %+.4f (t %+5.2f)  pic_rdg %+.4f (t %+5.2f)  "
              "pic_r+tp %+.4f (t %+5.2f)  rho_tp %+.2f  rho_rdg %+.2f  r2_44 %.2f  n %d"
              % (cand, s["ic"], s["t_ic"], s["pic_44"], s["t_pic_44"], s["pic_rdg"],
                 s["t_pic_rdg"], s["pic_r+tp"], s["t_pic_r+tp"], s["rho_tp"], s["rho_rdg"],
                 s["r2_44"], s["n"]), flush=True)
    table = pd.DataFrame(table).set_index("column")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_csv(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
