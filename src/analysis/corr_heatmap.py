"""
Correlation heatmaps across the feature-engineering arms.

    python3 -m src.analysis.corr_heatmap

Two matrices, because they answer different questions and routinely disagree:

  PREDICTION correlation -- mean over periods of the WITHIN-PERIOD rank
      correlation between two arms' predictions. This is the right measure
      here: Numerai scores cross-sectionally inside each period, so a
      pooled correlation across the whole panel would be dominated by
      between-period level differences the scorer never sees.

      Reads: "are these two models ranking the same coins the same way?"
      1.000 means the arms are the same model wearing different names.

  SCORE correlation -- ordinary correlation of the two 510-long per-period
      CORR series.

      Reads: "do these two models have good and bad periods together?"
      This is the number that caps ensembling. Averaging N arms with mean
      pairwise score correlation rho scales the variance of the mean by
      1/N + (N-1)*rho/N, so at rho = 0.92 you get a ~3% reduction in
      standard deviation, not the ~50% independence would buy.

Prediction correlation is almost always the higher of the two -- two models
can rank near-identically and still diverge on which periods happen to pay.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FORECAST_DIR = "results/forecast/crypto/ridge_glmnet"
PER_PERIOD_DIR = "results/per_period/crypto/ridge_glmnet"
OUT = "results/plots/crypto/correlations_%s.png"

# Named collections. Each entry is (model_dir, run_name, label); the model
# directory is part of the key because the two ridge variants write into
# separate trees and a run name alone is ambiguous.
COLLECTIONS = {
    # The original feature-engineering sweep.
    "features": [
        ("ridge_glmnet", "w6r6_glmnet_reg", "baseline (22)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_tenc", "tenc (23)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_spread", "spread (33)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_interact", "interact (36)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_fenccausal", "fenccausal (44)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_fencglobal", "fencglobal (44)*"),
        ("ridge_glmnet", "w6r6_glmnet_reg_onehot", "onehot (762)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_tenc_onehot", "tenc+1hot (763)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_onehot_rollsel", "1hot+roll (786)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_tenc_onehot_rollsel", "tenc+1hot+roll (787)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_allfeat", "allfeat (788)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_allfeatpp", "allfeat++ (812)"),
    ],
    # Every block and raw group on its own, against the two references. This is
    # the matrix that shows WHAT each piece is measuring rather than how well:
    # two blocks built from different columns can still be the same signal.
    "alone": [
        ("ridge_glmnet", "w6r6_glmnet_reg_allfeatpp", "ALL 812"),
        ("ridge_glmnet", "w6r6_glmnet_reg", "raw 22"),
        ("ridge_glmnet", "w6r6_alone_rolling", "rolling (24)"),
        ("ridge_glmnet", "w6r6_alone_onehot", "onehot (740)"),
        ("ridge_glmnet", "w6r6_alone_tenc", "tenc (1)"),
        ("ridge_glmnet", "w6r6_alone_interact", "interact (14)"),
        ("ridge_glmnet", "w6r6_alone_spread", "spread (11)"),
        ("ridge_glmnet", "w6r6_grp_volatility_glmnet", "grp volatility (2)"),
        ("ridge", "w6r6_grp_size_fixed", "grp size (4)"),
        ("ridge_glmnet", "w6r6_grp_price_glmnet", "grp price (4)"),
        ("ridge", "w6r6_grp_oscillator_fixed", "grp oscillator (4)"),
        ("ridge_glmnet", "w6r6_grp_liquidity_glmnet", "grp liquidity (4)"),
        ("ridge_glmnet", "w6r6_grp_momentum_glmnet", "grp momentum (2)"),
        ("ridge_glmnet", "w6r6_grp_bollinger_glmnet", "grp bollinger (2)"),
    ],
    # The blend/concat study: every group combination against the incumbents.
    # Blends are rank-averages of separately fitted members, concats are one
    # model on the union of the same columns -- so a high correlation between a
    # pair says the two construction methods produce the same signal, which is
    # the question that experiment was built to answer.
    "blends": [
        ("ridge_glmnet", "w6r6_glmnet_reg_allfeatpp", "allfeat++ (812)"),
        ("ridge", "w6r6_fixed_reg", "fixed raw (22)"),
        ("ridge_glmnet", "w6r6_glmnet_reg", "glmnet raw (22)"),
        ("ridge_glmnet", "w6r6_blend_quintet2", "blend quintet2"),
        ("ridge_glmnet", "w6r6_concat_quintet2", "concat quintet2"),
        ("ridge_glmnet", "w6r6_blend_quintet", "blend quintet"),
        ("ridge_glmnet", "w6r6_concat_quintet", "concat quintet"),
        ("ridge_glmnet", "w6r6_blend_trio", "blend trio"),
        ("ridge_glmnet", "w6r6_concat_trio", "concat trio"),
    ],
    # The deployed book plus the group combinations -- the inputs to an
    # ensemble of ensembles. CURUPIRA_02 and _04 are the same runs as
    # "fixed raw" and "allfeat++" in the blends collection; they carry their
    # slot names here because this is the view used to decide what to stake.
    "book": [
        ("ridge_glmnet", "w6r6_glmnet_reg_tenc", "CURUPIRA_01 (23)"),
        ("ridge", "w6r6_fixed_reg", "CURUPIRA_02 (22)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_onehot_rollsel", "CURUPIRA_03 (786)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_allfeatpp", "CURUPIRA_04 (812)"),
        ("ridge_glmnet", "w6r6_blend_quintet2", "blend quintet2"),
        ("ridge_glmnet", "w6r6_blend_quintet", "blend quintet"),
        ("ridge_glmnet", "w6r6_blend_trio", "blend trio"),
        ("ridge_glmnet", "w6r6_concat_quintet2", "concat quintet2"),
        ("ridge_glmnet", "w6r6_concat_quintet", "concat quintet"),
        ("ridge_glmnet", "w6r6_concat_trio", "concat trio"),
    ],
    # What is actually deployed.
    "live": [
        ("ridge_glmnet", "w6r6_glmnet_reg_tenc", "CURUPIRA_01 (23)"),
        ("ridge", "w6r6_fixed_reg", "CURUPIRA_02 (22)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_onehot_rollsel", "CURUPIRA_03 (786)"),
        ("ridge_glmnet", "w6r6_glmnet_reg_allfeatpp", "CURUPIRA_04 (812)"),
    ],
}


def load_predictions(arms):
    """Wide frame of predictions, one column per arm, aligned on (symbol, date)."""
    frames = {}
    keys = None
    for model, run, label in arms:
        path = os.path.join("results/forecast/crypto", model, run + ".csv")
        if not os.path.exists(path):
            print("   skipping %s -- not run yet" % label)
            continue
        df = pd.read_csv(path, usecols=["symbol", "date", "prediction"])
        df = df.set_index(["symbol", "date"]).sort_index()
        # Every arm scores exactly the same rows, so a mismatch means one run
        # used a different dataset or a different walk-forward -- refuse rather
        # than silently comparing different panels.
        if keys is None:
            keys = df.index
        elif not df.index.equals(keys):
            raise SystemExit("row mismatch: %s does not align with the first arm" % run)
        frames[label] = df["prediction"]
    return pd.DataFrame(frames)


def within_period_rank_corr(predictions):
    """
    Mean over periods of the within-period rank correlation matrix.

    Ranking then z-scoring inside each period turns the per-period Spearman
    matrix into Z_d' Z_d / n_d, so the whole thing is one pass over the panel
    rather than a pairwise loop.
    """
    dates = predictions.index.get_level_values("date")

    ranked = predictions.groupby(dates).rank(pct=True)
    centred = ranked - ranked.groupby(dates).transform("mean")
    scale = centred.groupby(dates).transform("std")          # ddof=1
    z = (centred / scale.replace(0.0, np.nan)).fillna(0.0)

    total = np.zeros((z.shape[1], z.shape[1]))
    n_periods = 0
    for _, block in z.groupby(dates):
        values = block.to_numpy()
        # ddof=1 in the std above, so divide by n-1 to get a correlation.
        total += values.T @ values / max(len(values) - 1, 1)
        n_periods += 1
    return pd.DataFrame(total / n_periods, index=z.columns, columns=z.columns)


def score_series_corr(arms):
    """Correlation of the per-period CORR series -- the ensembling ceiling."""
    series = {}
    for model, run, label in arms:
        path = os.path.join("results/per_period/crypto", model, run + ".csv")
        if not os.path.exists(path):
            continue
        series[label] = pd.read_csv(path).set_index("date")["corr"]
    return pd.DataFrame(series).corr()


def draw(ax, matrix, title, vmin):
    image = ax.imshow(matrix.to_numpy(), cmap="RdYlBu_r", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(matrix)))
    ax.set_yticks(range(len(matrix)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title, fontsize=11, pad=10)

    span = 1.0 - vmin
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix.iat[i, j]
            # White text on the dark ends of the colormap, black in the middle.
            shade = (value - vmin) / span
            ax.text(j, i, "%.3f" % value, ha="center", va="center", fontsize=6.5,
                    color="white" if shade > 0.75 or shade < 0.15 else "black")
    return image


def main():
    parser = argparse.ArgumentParser(description="Prediction correlation heatmaps.")
    parser.add_argument("--set", default="alone", choices=sorted(COLLECTIONS),
                        help="which collection of arms to compare")
    args = parser.parse_args()

    arms = COLLECTIONS[args.set]
    predictions = load_predictions(arms)
    print("loaded %d arms x %d rows" % (predictions.shape[1], len(predictions)))

    pred_corr = within_period_rank_corr(predictions)
    score_corr = score_series_corr(arms)

    pd.set_option("display.width", 250)
    for name, matrix in (("PREDICTION (within-period rank)", pred_corr),
                         ("SCORE (per-period corr series)", score_corr)):
        off = matrix.to_numpy()[np.triu_indices(len(matrix), k=1)]
        print("\n=== %s ===" % name)
        print(matrix.round(3).to_string())
        print("off-diagonal: mean %.4f  min %.4f  max %.4f" % (off.mean(), off.min(), off.max()))

    figure, axes = plt.subplots(1, 2, figsize=(19, 8))
    draw(axes[0], pred_corr, "Prediction correlation\n(mean within-period rank corr)",
         vmin=float(pred_corr.to_numpy().min()))
    image = draw(axes[1], score_corr, "Score correlation\n(per-period CORR series, n=510)",
                 vmin=float(score_corr.to_numpy().min()))
    figure.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    figure.suptitle("collection '%s'  --  rolling 6m window, 6m refit, 510 periods" % args.set, fontsize=12)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = OUT % args.set
    figure.savefig(out, dpi=140, bbox_inches="tight")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
