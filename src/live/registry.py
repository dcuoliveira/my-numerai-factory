"""
What each live model name actually is.

    python3 -m src.live.registry            # print the mapping
    python3 -m src.live.registry --write    # also write models/live/registry.json

The Numerai UI shows a model slot name and nothing else. Six months from now
"CURUPIRA_02" tells you nothing about what is running, and the whole point of the
pseudo-out-of-sample study was to be able to say WHY a given configuration is
live. This module is the single source of truth for that mapping: the runner,
the refit routine and the notification email all read their configuration from
here, so the metadata cannot drift from what actually runs.

------------------------------------------------------------------------------
The three slots and why these three
------------------------------------------------------------------------------

Chosen from an eleven-arm feature study and a separate model/CV sweep, all
scored over the same 510 crypto dates. Two findings drove the selection:

  1. No arm beats the baseline by a statistically meaningful margin. The largest
     |t| on a paired payout difference is 1.30, and it belongs to a NEGATIVE
     result. With a 20-day forward target on daily dates the effective sample is
     ~26 independent windows, so mean payout cannot separate these models.

  2. What does separate them is risk and correlation. Scored on the payable
     series (0.1*CORR + 1*MMC) rather than the CORR column, the ordering by
     sharpe is tenc > baseline > spread > ... > fixed > 1hot+roll.

So the slots are not "the three highest means". They are the best risk-adjusted
model, plus the two configurations least correlated with it -- because three
slots that agree are one slot with three times the fees.

Pairwise per-period score correlation between the deployed three:

              CURUPIRA_01   CURUPIRA_02   CURUPIRA_03
  01 (glmnet)       1.000         0.833         0.808
  02 (fixed)        0.833         1.000         0.952
  03 (1hot+roll)    0.808         0.952         1.000

------------------------------------------------------------------------------
Fidelity to the backtest
------------------------------------------------------------------------------

Every field under "fit" is copied from the winning run's config block, so a live
refit reproduces the pseudo-OOS protocol exactly: same rolling 130-period
window, same purged/embargoed inner CV, same 50-draw randomised search, same
seed. If you change one here you have invalidated the backtest that justified
the slot -- so change the registry, re-run the arm, and update `backtest`.
"""

import argparse
import json
import os

# Where fitted models, their metadata and this registry's JSON copy live.
MODEL_DIR = "models/live"
REGISTRY_JSON = os.path.join(MODEL_DIR, "registry.json")

# Shared fit protocol. These are the exact values behind every w6r6 arm in the
# study; see results/metrics/crypto/*/w6r6_*.json -> "config".
FIT = {
    "tournament": "crypto",
    "task": "regression",
    "metric": "corr",           # CV objective
    "cv_scheme": "rolling",
    "train_window": 130,        # periods (~6 months of daily dates)
    "refit_every": 130,         # periods between refits
    "embargo": 20,              # = the 20-day forward target horizon
    "n_splits": 5,
    "n_iter": 50,
    "seed": 42,
    "target": "target_binned_return_20",
}

# feature_set values are resolved by src.live.features.build_features.
MODELS = {
    "SACI_01": {
        "wrapper": "lgb",
        "feature_set": "interact",
        "n_features": 36,
        "backtest_run": "fe_lgb_reg_interact",
        "why": (
            "The only feature block in the programme to beat raw at |t| > 2 on "
            "a paired payout test (+0.005710, t = +2.99, Newey-West lag 20 over "
            "510 periods). It is also the first non-ridge slot: LightGBM ranks "
            "0.705-0.751 against the ridge book where the ridge variants sit at "
            "0.883-0.926 against each other, so this adds diversification no "
            "fourth ridge arm can. 22 shipped features plus 14 products and "
            "acceleration gaps -- pure same-row transforms, no history, nothing "
            "fitted, so there is no live/train skew surface at all.\n\n"
            "READ THE CAVEAT BEFORE STAKING. The +2.99 is one result among 38 "
            "arms, and it does NOT reproduce at min_child_samples 50 (t = +0.36) "
            "-- so it is conditional on the leaf size, not a property of the "
            "columns. A Bonferroni-style adjustment over 38 tests puts the "
            "threshold near |t| = 2.9, which it only just clears. Treat this as "
            "a candidate under observation, not an established edge, and give it "
            "a small stake until it replicates on another seed and on the "
            "60-day target."
        ),
        "backtest": {
            "corr": 0.156145, "corr_sharpe": 1.8590, "corr_max_dd": 0.5005,
            "mmc": 0.073368, "mmc_sharpe": 1.2702,
            "payout": 0.088983, "payout_sharpe": 1.3877, "payout_max_dd": 1.0856,
            "n_periods": 510,
        },
        # The estimator is PINNED, not searched: this is the configuration
        # w6r6_lgb_reg's 50-draw search converged on (it won 3 of 4 refits), and
        # the cadence study showed pinning matched searching at 6m/6m (payout
        # 0.083272 against 0.082061) for 1/90th of the compute. refit.py passes
        # fixed_params straight through, so the live fit is the backtested fit.
        "fit_override": {
            "fixed_params": {
                "n_estimators": 4327, "learning_rate": 0.002572,
                "max_depth": 8, "num_leaves": 255,
                "colsample_bytree": 0.070213, "subsample": 0.816153,
                "subsample_freq": 1, "min_child_samples": 313,
                "reg_lambda": 1.757511,
            },
        },
        # Losing the interactions leaves the plain 22, which is a materially
        # different model here -- unlike CURUPIRA_01, whose tenc fallback is
        # near-identical. The interactions are same-row arithmetic on columns
        # that are always present, so this should never fire.
        "fallback_feature_set": "raw",
    },
    "SACI_02": {
        "wrapper": "lgb",
        "feature_set": "allfeat",
        "n_features": 788,
        "backtest_run": "ohc_lgb_reg_allfeat_m50",
        "why": (
            "The widest feature set the study has, run at the leaf size that "
            "lets it actually use the block it carries. At min_child_samples 313 "
            "a ticker's 130 rows in a 6-month window cannot satisfy the split "
            "minimum, so LightGBM discards all 740 one-hot columns before "
            "training -- allfeat at 313 is allfeat without its identity block. "
            "At 50 the block is live, and the same dataset scores 0.088451 "
            "against 0.087600.\n\n"
            "Its real argument is risk, not level: max drawdown 0.2443 against "
            "0.6806 for raw at the same leaf, at comparable sharpe. That is the "
            "most consistent pattern in the whole leaf study -- wide feature "
            "sets at leaf 50 roughly halve drawdown.\n\n"
            "CAVEAT. The payout edge over raw is NOT significant: +0.008183, "
            "t = +1.34 paired Newey-West over 510 periods. Deploy it for the "
            "drawdown profile and the decorrelation, not for the mean."
        ),
        "backtest": {
            "corr": 0.145425, "corr_sharpe": 1.7275, "corr_max_dd": 0.2443,
            "mmc": 0.073909, "mmc_sharpe": 1.0868,
            "payout": 0.088451, "payout_sharpe": 1.1813, "payout_max_dd": 0.5102,
            "n_periods": 510,
        },
        "fit_override": {
            "fixed_params": {
                "n_estimators": 4327, "learning_rate": 0.002572,
                "max_depth": 8, "num_leaves": 255,
                "colsample_bytree": 0.070213, "subsample": 0.816153,
                "subsample_freq": 1,
                # 50, NOT 313. This single number decides whether the 740
                # one-hot columns this slot exists to use are trained on at all.
                "min_child_samples": 50,
                "reg_lambda": 1.757511,
            },
        },
        "fallback_feature_set": "raw",
    },
    "CURUPIRA_SACI_01": {
        "wrapper": None,
        # A blend of SLOTS: it fits nothing of its own and rank-averages three
        # slots that are already fitted for their own sake, so it costs seconds.
        "feature_set": "allfeat_pp",   # superset of all three members' columns
        "n_features": 812,
        "backtest_run": "saci01_c03_c0560",
        "why": (
            "Highest payout of anything in the project at 0.097730, and it beats "
            "every one of its own members: +0.007769 over SACI_01, +0.008880 "
            "over CURUPIRA_03, +0.005529 over CURUPIRA_05_60. That is unusual "
            "here -- most blends dilute the mean -- and it happens because the "
            "members genuinely disagree. On the payout series SACI_01 and "
            "CURUPIRA_03 correlate 0.756, while CURUPIRA_05_60 correlates "
            "NEGATIVELY with both (-0.166 and -0.371). A 60-day label and a "
            "LightGBM on 36 columns are different bets from a 786-column ridge.\n\n"
            "It is also the first slot combining two model classes: LightGBM "
            "ranks 0.705-0.751 against the ridge book where ridge variants sit "
            "at 0.883-0.926 against each other."
        ),
        "backtest": {
            "corr": 0.157364, "corr_sharpe": 1.8586, "corr_max_dd": 0.8373,
            "mmc": 0.081993, "mmc_sharpe": 1.3208,
            "payout": 0.097730, "payout_sharpe": 1.4183, "payout_max_dd": 1.3094,
            "n_periods": 470,
        },
        "slot_members": [
            {"name": "SACI_01", "weight": 0.3333},
            {"name": "CURUPIRA_03", "weight": 0.3333},
            {"name": "CURUPIRA_05_60", "weight": 0.3334},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "NONE OF THE THREE MARGINS IS SIGNIFICANT. Paired Newey-West at lag "
            "20 over 470 periods gives t = +1.24, +0.72 and +0.23 against the "
            "members. On ~24 effective independent windows this sample cannot "
            "separate the blend from simply holding SACI_01.",
            "ITS DRAWDOWN IS WORSE THAN SACI_01 ALONE: 1.3094 against 1.0856. "
            "An earlier screen suggested this combination would CUT drawdown to "
            "0.23 -- that screen averaged the members' payout SERIES, which is "
            "not what run_ensemble does. The real per-period rank average has "
            "higher payout and higher drawdown. CURUPIRA_05_60 carries a 10.99 "
            "payout drawdown of its own and drags it up.",
            "Scored on 470 periods, not 510: CURUPIRA_05_60's 60-day label is "
            "NaN over the last 40 dates, so the blend cannot be compared "
            "directly to the 510-period arms.",
            "CURUPIRA_05_60 is in NOT_SUBMITTED -- fitted and cached but never "
            "uploaded on its own. This slot is one of the two ways its signal "
            "reaches the book (CURUPIRA_07 is the other).",
        ],
    },
    "CURUPIRA_01": {
        "wrapper": "ridge_glmnet",
        "feature_set": "tenc",
        "n_features": 23,
        "backtest_run": "w6r6_glmnet_reg_tenc",
        "why": (
            "Best risk-adjusted model in the study. Highest sharpe on the "
            "payable series (1.2960) and highest MMC sharpe (1.1794), with the "
            "lowest CORR drawdown of the leaders. 22 shipped features plus one "
            "leakage-safe mean-target encoding per symbol."
        ),
        "backtest": {
            "corr": 0.149275, "corr_sharpe": 1.8189, "corr_max_dd": 0.7762,
            "mmc": 0.069205, "mmc_sharpe": 1.1794,
            "payout": 0.084132, "payout_sharpe": 1.2960, "payout_max_dd": 1.3732,
            "n_periods": 510,
        },
        # If the tenc column cannot be built, this model degrades to the plain
        # 22 features. That is not a guess: tenc correlates 1.000 with the raw
        # baseline on both predictions and per-period scores, so the fallback is
        # very nearly the same model. Losing the column costs ~0.0003 payout.
        "fallback_feature_set": "raw",
    },
    "CURUPIRA_02": {
        "wrapper": "ridge",
        "feature_set": "raw",
        "n_features": 22,
        "backtest_run": "w6r6_fixed_reg",
        "why": (
            "Highest payout of any model needing no feature engineering "
            "(0.085763). Fixed alpha grid capped at 100 against glmnet's "
            "data-scaled path reaching ~1.2M -- a genuinely different shrinkage "
            "regime, correlating 0.833 with CURUPIRA_01 on per-period scores. "
            "Reads live.parquet directly: no build step, no cold start."
        ),
        "backtest": {
            "corr": 0.153169, "corr_sharpe": 1.5424, "corr_max_dd": 1.2192,
            "mmc": 0.070446, "mmc_sharpe": 0.9944,
            "payout": 0.085763, "payout_sharpe": 1.0818, "payout_max_dd": 1.6557,
            "n_periods": 510,
        },
        "fallback_feature_set": None,   # already the simplest set
    },
    "CURUPIRA_03": {
        "wrapper": "ridge_glmnet",
        "feature_set": "onehot_roll",
        "n_features": 786,
        "backtest_run": "w6r6_glmnet_reg_onehot_rollsel",
        "why": (
            "Highest payout mean of any deployable arm (0.086721) and the least "
            "correlated with CURUPIRA_01 (0.808) -- it is the diversification "
            "in the book. Pays for it with a payout drawdown of 1.879 against "
            "1.373, so it is sized as one slot of three, not as the core."
        ),
        "backtest": {
            "corr": 0.157008, "corr_sharpe": 1.4837, "corr_max_dd": 1.4652,
            "mmc": 0.071021, "mmc_sharpe": 0.9712,
            "payout": 0.086721, "payout_sharpe": 1.0554, "payout_max_dd": 1.8786,
            "n_periods": 510,
        },
        # The riskiest build of the three: 740 identity columns plus rolling
        # means that need contiguous history. If either cannot be assembled the
        # model runs on the raw 22 -- a real model, just not this one.
        "fallback_feature_set": "raw",
        "caveats": [
            "21-26% of symbols per refit never appear in that refit's 6-month "
            "training window, so their one-hot block is all zeros and only the "
            "raw features drive their prediction.",
            "Rolling features need contiguous history. train.parquet lags live "
            "by ~35 days because 20-day targets have to resolve first, so each "
            "round's live rows are appended to a local history store to close "
            "the gap over time (see src/live/features.py).",
        ],
    },
    "CURUPIRA_05": {
        # A BLEND, not a single estimator: five models, each on its own block of
        # columns with its own penalty, rank-averaged at equal weight. See
        # src/live/blend.py for why ranking comes before averaging.
        "wrapper": None,
        # Blends have no wrapper, so the results tree cannot be derived from one.
        # This is the model directory their backtest_run was scored under.
        "scored_under": "ridge_glmnet",
        "feature_set": "allfeat_pp",     # the base build every member carves from
        "n_features": 763,               # union of the members' columns
        "backtest_run": "w6r6_blend_quintet2",
        "why": (
            "The only arm in the study that is genuinely different from the "
            "rest of the book: it correlates 0.53-0.63 with CURUPIRA_01-04, "
            "which sit at 0.88-0.998 with each other. Highest MMC recorded "
            "(0.073687) at a third of their drawdown, and MMC carries weight "
            "1.0 in the crypto payout while CORR carries 0.1."
        ),
        "backtest": {
            "corr": 0.126305, "corr_sharpe": 1.7003, "corr_max_dd": 0.5587,
            "mmc": 0.073687, "mmc_sharpe": 1.2565,
            "payout": 0.086318, "payout_sharpe": 1.3307, "payout_max_dd": 0.6182,
            "n_periods": 510,
        },
        "members": [
            # include patterns are matched against the 812-column allfeat_pp
            # build. The two RAW groups are ANCHORED on purpose: they were fitted
            # on the 22-column dataset where "^feature_market_cap_(avg|ewa)_"
            # matched exactly 4 columns, but on the 812-column build the same
            # pattern also catches feature_market_cap_avg_spread. Without the
            # anchor the live model would fit 6 columns where the backtest fit 4.
            {"name": "onehot", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"^feature_sym_(?!tenc$)", "n_features": 740},
            {"name": "tenc", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"^feature_sym_tenc$", "n_features": 1},
            {"name": "interact", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"_x_|_accel_", "n_features": 14},
            {"name": "size", "wrapper": "ridge", "weight": 0.2,
             "include": r"^feature_market_cap_(avg|ewa)_(20d|60d)$", "n_features": 4},
            {"name": "price", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"^feature_close_(avg|ewa)_(20d|60d)$", "n_features": 4},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "Its members and weights were chosen by searching every 5-subset of "
            "twelve blocks over a weight simplex on the SAME 510 periods it is "
            "scored on -- roughly 26 effective independent windows. Some of the "
            "1.33 payout sharpe is fitted. Treat it as the most promising "
            "hypothesis in the book, not as an established result.",
            "Five fits per refit instead of one, so a refit round costs ~4 "
            "minutes rather than ~1. Still trivial against the ~66 minute "
            "submission window.",
            "Carries the one-hot cold start (~24% of live symbols) through its "
            "onehot member, but that member holds only 0.2 of the weight.",
        ],
    },
    "CURUPIRA_06": {
        # A BLEND OF SLOTS, not of feature blocks. Its members are CURUPIRA_01
        # and CURUPIRA_05 themselves, so it fits NOTHING of its own -- it reuses
        # their cached estimators and rank-averages the two outputs. A round
        # costs it about a second.
        "wrapper": None,
        "scored_under": "ridge_glmnet",
        "feature_set": "allfeat_pp",   # superset of both members' columns
        "n_features": 812,
        "backtest_run": "eoe_c01_bq2",
        "why": (
            "The only result in the study whose 95% confidence interval "
            "excludes zero. Against CURUPIRA_04: sharpe +0.3694 (CI +0.052 to "
            "+0.738, P=0.989) and max drawdown -0.6652 (CI -2.072 to -0.128, "
            "P=0.999) on a paired block bootstrap. Its two members correlate "
            "0.53 on the payout series where CURUPIRA_01-04 sit at 0.88-0.998, "
            "so the variance reduction is arithmetic rather than a fitted "
            "pattern -- and the 50/50 weight is chosen, not searched."
        ),
        "backtest": {
            "corr": 0.148223, "corr_sharpe": 1.6297, "corr_max_dd": 0.5838,
            "mmc": 0.075691, "mmc_sharpe": 1.3392,
            "payout": 0.090514, "payout_sharpe": 1.4476, "payout_max_dd": 1.0594,
            "n_periods": 510,
        },
        "slot_members": [
            {"name": "CURUPIRA_01", "weight": 0.5},
            {"name": "CURUPIRA_05", "weight": 0.5},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "Its PAYOUT edge is not significant: +0.003318 against CURUPIRA_04 "
            "at t = 0.38. Only the risk result clears a confidence interval. "
            "Deploy it for stability, not for expected return.",
            "It inherits CURUPIRA_05's in-sample selection -- those five members "
            "and their weights were searched over the same 510 periods. "
            "Averaging does not undo that, so the bootstrap probabilities are "
            "conditional on CURUPIRA_05 being a fair draw, which it is not "
            "entirely. The equal weight here is the part that is not fitted.",
            "It fits nothing itself, so it is only as fresh as CURUPIRA_01 and "
            "CURUPIRA_05. If either has no cached fit this slot cannot be built "
            "and falls through to the stand-in ladder.",
        ],
    },
    "CURUPIRA_05_60": {
        # NOT A SUBMITTED SLOT. Absent from ORDER on purpose: it exists only so
        # CURUPIRA_07 has a 60-day component to average. It is fitted and cached
        # like any blend, but nothing uploads it.
        #
        # Standalone it is the worst arm in the study -- payout 0.074914 on the
        # payable 20-day target, payout drawdown 7.88. Its entire value is that
        # it correlates 0.394-0.430 with the 20-day book, the lowest figure
        # anywhere in this project, against 0.61-0.998 among CURUPIRA_01-06.
        "wrapper": None,
        "scored_under": "ridge_glmnet",
        "feature_set": "allfeat_pp",
        "n_features": 763,
        "backtest_run": "t60_c05_blend",
        "fit_override": {
            # Both must move together. The 60-day label resolves 60 periods out
            # and is NaN over the panel's last 40 dates; fitting it under the
            # 20-day embargo would leak three times over and feed NaN targets in.
            "target": "target_binned_return_60",
            "embargo": 60,
        },
        "why": (
            "The only model that decorrelates the 20-day book. Exists solely as "
            "a member of CURUPIRA_07; never submitted on its own."
        ),
        "backtest": {
            "corr": 0.098256, "corr_sharpe": 0.7027, "corr_max_dd": 7.8815,
            "mmc": 0.065089, "mmc_sharpe": 0.7027,
            "payout": 0.074914, "payout_sharpe": 0.7027, "payout_max_dd": 7.8815,
            "n_periods": 470,
        },
        "members": [
            {"name": "onehot", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"^feature_sym_(?!tenc$)", "n_features": 740},
            {"name": "tenc", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"^feature_sym_tenc$", "n_features": 1},
            {"name": "interact", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"_x_|_accel_", "n_features": 14},
            {"name": "size", "wrapper": "ridge", "weight": 0.2,
             "include": r"^feature_market_cap_(avg|ewa)_(20d|60d)$", "n_features": 4},
            {"name": "price", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"^feature_close_(avg|ewa)_(20d|60d)$", "n_features": 4},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "Its feature_sym_tenc member still encodes the 20-DAY target at a "
            "20-period lag, because the datasets are shared with the 20-day "
            "slots. Causally sound -- it reads only resolved 20-day outcomes "
            "from at least 20 periods back -- but it encodes a related rather "
            "than the same quantity as the label being fitted.",
        ],
    },
    "CURUPIRA_07": {
        # Four slots rank-averaged at 1/4: three 20-day models plus the 60-day
        # blend. Fits nothing of its own.
        "wrapper": None,
        # No scored_under: its backtest_run is a screen, not a run, so there is
        # no per-period series for it anywhere. Nothing can weight it.
        "feature_set": "allfeat_pp",
        "n_features": 812,
        "backtest_run": "(screened, not run as a single arm)",
        "why": (
            "Best payout sharpe of any combination screened across both target "
            "blocks: 1.5095 against 1.4931 for the best pair and 1.3641 for the "
            "best single. Every one of the twelve best 4-model combinations "
            "contains exactly one 60-day model, and the top six all use "
            "CURUPIRA_05_60 -- the screen picks the decorrelator unprompted."
        ),
        "backtest": {
            "corr": None, "corr_sharpe": None, "corr_max_dd": None,
            "mmc": None, "mmc_sharpe": None,
            "payout": 0.084782, "payout_sharpe": 1.5095, "payout_max_dd": 0.9317,
            "n_periods": 470,
        },
        "slot_members": [
            {"name": "CURUPIRA_01", "weight": 0.25},
            {"name": "CURUPIRA_04", "weight": 0.25},
            {"name": "CURUPIRA_05", "weight": 0.25},
            {"name": "CURUPIRA_05_60", "weight": 0.25},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "ITS PAYOUT IS LOWER THAN CURUPIRA_05 ALONE: 0.084782 against "
            "0.089785, and zero of the 210 four-model combinations screened "
            "kept payout above that single arm. Deploy it for sharpe and "
            "drawdown, not for expected return.",
            "The sharpe advantage is not significant. Paired against "
            "CURUPIRA_05 alone the payout difference is -0.005003 at t = -0.76, "
            "and the whole k=2..5 sharpe range spans just 1.477-1.510.",
            "Screened in-sample on 470 correlated periods, ~23 effective "
            "independent windows, on a payout-series proxy rather than a "
            "prediction-level blend. Treat it as the best available hypothesis, "
            "not a measured result.",
        ],
    },
    "CURUPIRA_04": {
        "wrapper": "ridge_glmnet",
        "feature_set": "allfeat_pp",
        "n_features": 812,
        "backtest_run": "w6r6_glmnet_reg_allfeatpp",
        "why": (
            "Highest backtested payout of any deployable arm (0.087196) and the "
            "only one to beat CURUPIRA_03 on payout AND drawdown at once "
            "(1.725 against 1.879). Every feature block at once: the union of "
            "the other three slots plus 11 horizon spreads and 14 products."
        ),
        "backtest": {
            "corr": 0.158011, "corr_sharpe": 1.5136, "corr_max_dd": 1.2642,
            "mmc": 0.071394, "mmc_sharpe": 0.9916,
            "payout": 0.087196, "payout_sharpe": 1.0782, "payout_max_dd": 1.7246,
            "n_periods": 510,
        },
        "fallback_feature_set": "raw",
        "caveats": [
            "IT IS NEARLY THE SAME MODEL AS CURUPIRA_03: payout-series "
            "correlation 0.998, and the paired difference is +0.000474 at "
            "t = 0.72. Running both is close to running one slot twice.",
            "Because of that, ADDING it as a fourth slot makes the book worse "
            "on risk -- ERC over four gives payout +0.000384 but sharpe -0.0256 "
            "and drawdown +0.0247 against the three-slot book. SWAPPING it for "
            "CURUPIRA_03 is the better trade (payout +0.000148, sharpe +0.0068, "
            "drawdown -0.0069). To swap, drop CURUPIRA_03 from ORDER below.",
            "Carries the same one-hot cold start as CURUPIRA_03 (~24% of live "
            "symbols) and the same rolling-history gap. The 25 derived columns "
            "are same-row transforms and need no history.",
        ],
    },
    "CURUPIRA_08": {
        # C04 + C05 + the 60-day C05_60, rank-averaged at 1/3 each. Fits nothing
        # of its own; assembled from the three members' caches.
        "wrapper": None,
        "scored_under": "ridge_glmnet",
        "feature_set": "allfeat_pp",
        "n_features": 812,
        "backtest_run": "eoe_c04_bq2_t60",
        "why": (
            "Highest backtested payout in the book -- 0.099815 against 0.090514 "
            "for CURUPIRA_06, the best existing slot -- and the best CORR sharpe "
            "of any slot at 1.8870 with the shallowest corr drawdown at 0.6940. "
            "It exists because of one measured fact: the 60-day block is the "
            "only thing in the study whose per-period score series is NEGATIVELY "
            "correlated with the 20-day models (-0.24 against CURUPIRA_04), so "
            "CURUPIRA_05_60 is the single genuine decorrelator available. This "
            "slot is the smallest combination that uses it alongside the two "
            "best-scoring 20-day arms."
        ),
        "backtest": {
            "corr": 0.151458, "corr_sharpe": 1.8870, "corr_max_dd": 0.6940,
            "mmc": 0.084669, "mmc_sharpe": 1.3837,
            "payout": 0.099815, "payout_sharpe": 1.4713, "payout_max_dd": 1.0007,
            "n_periods": 470,
        },
        "slot_members": [
            {"name": "CURUPIRA_04", "weight": 1.0 / 3},
            {"name": "CURUPIRA_05", "weight": 1.0 / 3},
            {"name": "CURUPIRA_05_60", "weight": 1.0 / 3},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "IT IS NEARLY CURUPIRA_07: prediction correlation 0.978, because "
            "the two differ only by CURUPIRA_01's quarter. The paired payout "
            "difference is +0.001222 at t = +0.79, and C08's payout SHARPE is "
            "very slightly WORSE (1.4713 against 1.4775). Running both is close "
            "to running one slot twice; choosing between them is inside the "
            "noise, and if the book is trimmed these two are the pair to look "
            "at first.",
            "Its payout advantage over the rest of the book is not significant: "
            "+0.010026 over CURUPIRA_05 at t = +1.37, +0.010520 over "
            "CURUPIRA_04 at t = +0.90, +0.006488 over CURUPIRA_06 at t = +1.32. "
            "470 correlated periods is ~23 effective independent windows, and "
            "nothing here clears |t| > 2.",
            "SCORED ON 470 PERIODS, NOT 510, because CURUPIRA_05_60's label is "
            "NaN over its last 40 dates. Its headline payout is therefore not "
            "directly comparable to the 510-period slots' -- the paired deltas "
            "above are the comparable numbers, not the levels.",
            "It inherits CURUPIRA_05_60, which is in NOT_SUBMITTED for a "
            "reason: on its own label it is the worst-scoring arm in the study. "
            "It is carried here as a decorrelator, not as a source of return, "
            "and this slot's payout depends on that member continuing to be "
            "refit even though it is never uploaded.",
            "The CURUPIRA_07 comparison above uses a RECONSTRUCTION of that "
            "slot (payout 0.098594), not its registry-recorded 0.084782 -- "
            "CURUPIRA_07's backtest_run is a screen rather than a run, so there "
            "is nothing to score it against directly.",
        ],
    },
}

# Order matters: this is the order slots are refit and submitted in, so the
# cheapest and most robust models get their submissions in first when the
# submission window (~66 minutes) is tight. CURUPIRA_04 is last because it is
# the most expensive to build and the most redundant -- if the deadline bites,
# it should bite here.
#
# To SWAP CURUPIRA_03 for CURUPIRA_04 rather than run both, drop "CURUPIRA_03"
# from this list. The backtest prefers the swap (sharpe +0.0068, drawdown
# -0.0069) over the addition (sharpe -0.0256, drawdown +0.0247), because the two
# correlate 0.998 and a fourth near-duplicate slot dilutes the diversification
# the book is built on. Both differences are well inside the noise.
# CURUPIRA_06 fits nothing, so it is cheapest -- but it must run AFTER the
# slots it reuses, because it loads their caches. Placed last for that
# reason; the fit-cost ordering test allows it as a zero-cost tail.
ORDER = ["CURUPIRA_02", "CURUPIRA_01", "SACI_01", "CURUPIRA_03",
         "CURUPIRA_04", "SACI_02",
         "CURUPIRA_05", "CURUPIRA_05_60", "CURUPIRA_06", "CURUPIRA_07",
         "CURUPIRA_08", "CURUPIRA_SACI_01"]

# Slots that are fitted and cached but never uploaded. CURUPIRA_05_60 exists
# only as a component of CURUPIRA_07; submitting it would stake real money on
# the worst-scoring arm in the study.
NOT_SUBMITTED = {"CURUPIRA_05_60"}


def is_blend(spec):
    """True when a slot rank-averages feature-block members it fits itself."""
    return bool(spec.get("members"))


def is_slot_blend(spec):
    """
    True when a slot rank-averages OTHER SLOTS and fits nothing of its own.

    Kept distinct from is_blend because the two need opposite handling: a blend
    refits its members on the usual cadence, a slot blend must never refit at
    all -- it would duplicate work already done and, worse, could produce a fit
    that disagrees with the slot it is supposed to be reusing.
    """
    return bool(spec.get("slot_members"))


PER_PERIOD_DIR = "results/per_period/crypto"


def per_period_path(name):
    """
    Where this slot's backtested per-period score series lives, or None.

    The results tree is keyed by MODEL directory, which for a single estimator
    is its wrapper -- but a blend has no wrapper, so it carries `scored_under`
    instead. Returning None rather than a broken path matters: CURUPIRA_07's
    backtest_run is a screen and no such file was ever written, so anything that
    weights slots by their history has to be able to see that and skip it.
    """
    spec = get(name)
    model = spec.get("scored_under") or spec["wrapper"]
    if not model:
        return None
    return os.path.join(PER_PERIOD_DIR, model, "%s.csv" % spec["backtest_run"])


def base_exposure(weights):
    """
    Expand a stake split over SLOTS into exposure to the underlying models.

    CURUPIRA_06 and _08 are rank averages of other slots, so staking them is not
    a fresh bet -- it is a pre-mixed weight vector over CURUPIRA_01/04/05/05_60.
    Staking 01, 06 and 08 equally looks like three positions and is really 50%
    CURUPIRA_01. Recursing through slot_members is the only way to see that.
    """
    exposure = {}
    for name, weight in weights.items():
        spec = get(name)
        if is_slot_blend(spec):
            inner = {m["name"]: float(weight) * m["weight"]
                     for m in spec["slot_members"]}
            for base, value in base_exposure(inner).items():
                exposure[base] = exposure.get(base, 0.0) + value
        else:
            exposure[name] = exposure.get(name, 0.0) + float(weight)
    return exposure


def member_path(name, member):
    """Where one blend member's fitted estimator is cached."""
    return os.path.join(MODEL_DIR, "%s__%s.joblib" % (name, member))


def get(name):
    """Full spec for one slot: its own fields plus the shared fit protocol."""
    if name not in MODELS:
        raise KeyError("unknown model '%s', expected one of %s" % (
            name, sorted(MODELS)))
    spec = dict(MODELS[name])
    spec["name"] = name
    # A slot may override individual fit fields -- CURUPIRA_05_60 fits the
    # 60-day label, which changes both the target and the embargo. Overriding
    # rather than redefining keeps every other field pinned to the shared
    # protocol, so a slot cannot quietly drift on window or seed.
    fit = dict(FIT)
    fit.update(spec.get("fit_override") or {})
    spec["fit"] = fit
    return spec


def all_specs():
    """Every slot, in submission order."""
    return [get(name) for name in ORDER]


def model_path(name):
    """Where the fitted estimator for this slot is cached."""
    return os.path.join(MODEL_DIR, "%s.joblib" % name)


def meta_path(name):
    """Where that estimator's fit metadata (dates, params, rows) is cached."""
    return os.path.join(MODEL_DIR, "%s.json" % name)


def write_json(path=REGISTRY_JSON):
    """Dump the mapping so it can be read without importing Python."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "note": ("Maps live Numerai model slot names to the configuration they "
                 "actually run. Generated by src/live/registry.py -- edit that "
                 "file, not this one."),
        "shared_fit_protocol": FIT,
        "submission_order": ORDER,
        "models": {name: MODELS[name] for name in ORDER},
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    return path


def describe():
    """Human-readable mapping, used by the notification email."""
    lines = []
    for spec in all_specs():
        b = spec["backtest"]
        lines.append("%-12s %-13s %3d feat  %-28s" % (
            spec["name"], spec["wrapper"], spec["n_features"], spec["backtest_run"]))
        lines.append("             payout %.6f  p.sharpe %.4f  p.maxdd %.4f  (n=%d)" % (
            b["payout"], b["payout_sharpe"], b["payout_max_dd"], b["n_periods"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Show the live model registry.")
    parser.add_argument("--write", action="store_true",
                        help="also write %s" % REGISTRY_JSON)
    args = parser.parse_args()

    print(describe())
    for spec in all_specs():
        print("\n%s -- %s" % (spec["name"], spec["why"]))
        for caveat in spec.get("caveats", []):
            print("    caveat: %s" % caveat)
    if args.write:
        print("\nwrote %s" % write_json())


if __name__ == "__main__":
    main()
