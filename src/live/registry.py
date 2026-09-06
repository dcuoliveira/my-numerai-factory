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
        # Blends have no wrapper, so the results tree cannot be derived from
        # one. This is the model directory their backtest_run was scored under.
        "scored_under": "lgb",
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
    "SACI_03": {
        "wrapper": "lgb",
        "feature_set": "tenc_onehot",
        "n_features": 763,
        "backtest_run": "e01_tenc_onehot_m50",
        "why": (
            "A 60-DAY model, and the reason it exists is decorrelation, not its "
            "own score. From experiment E01: on the 60-day label the wide "
            "leaf-50 arms are the only ones that beat raw, and this one ranks "
            "0.63 against CURUPIRA_05_60 and 0.49-0.72 against the 20-day book "
            "-- a different label and a different model class from anything "
            "deployed.\n\n"
            "DO NOT STAKE IT ALONE. Its own payout sharpe is 0.7170 with a "
            "10.9473 payout drawdown, the worst risk profile of any slot here. "
            "It exists as a member of CURUPIRA_SACI_02, where blending tames it: "
            "that blend lands at sharpe 1.3941 and drawdown 1.0523."
        ),
        "backtest": {
            # SCORED ON THE 60-DAY LABEL, FITTED LIVE ON THE 20-DAY ONE.
            # Every number below comes from a target_binned_return_60 study, so
            # it does NOT describe what this slot now submits. There is no
            # re-scored figure for it either: results/cache/cross_target_payout.csv
            # covers the ridge CURUPIRA arms and the group blends, not this one.
            # The one comparable case, CURUPIRA_05_60, lost ~19% crossing labels
            # (0.092201 -> 0.074914). Produce the re-score before staking this.
            "scored_on": "target_binned_return_60",
            "corr": 0.142699, "corr_sharpe": 1.0926, "corr_max_dd": 7.7020,
            "mmc": 0.078748, "mmc_sharpe": 0.6709,
            "payout": 0.093018, "payout_sharpe": 0.7170, "payout_max_dd": 10.9473,
            "n_periods": 470,
        },
        "fit_override": {
            # SEARCHED on the 60-day label, FITTED on the payable 20-day one.
            #
            # These fixed_params are the ones the 60-day study selected, and they
            # are kept exactly: they say which shape of booster suits this feature
            # set, which does not depend on the horizon. But Numerai scores live
            # submissions on target_binned_return_20, so a slot that is uploaded
            # in its own right must be fitted to predict THAT. Fitting on the
            # 60-day label and being scored on the 20-day one is predicting the
            # wrong quantity, however good the 60-day backtest looked.
            #
            # Target and embargo move together -- the 20-day label resolves 20
            # periods out, so 20 is the matching embargo. A happy side effect:
            # the 20-day label resolves ~40 periods sooner than the 60-day one,
            # which pulls this slot's training edge from 2026-05-13 up to
            # 2026-07-08, the same edge every other slot sits on.
            #
            # CURUPIRA_05_60 deliberately does NOT do this -- see its entry.
            "target": "target_binned_return_20",
            "embargo": 20,
            "fixed_params": {
                "n_estimators": 4327, "learning_rate": 0.002572,
                "max_depth": 8, "num_leaves": 255,
                "colsample_bytree": 0.070213, "subsample": 0.816153,
                "subsample_freq": 1,
                # 50, so the 740 one-hot columns this arm carries are usable.
                "min_child_samples": 50,
                "reg_lambda": 1.757511,
            },
        },
        "fallback_feature_set": "tenc",
    },
    "CURUPIRA_SACI_02": {
        "wrapper": None,
        "feature_set": "allfeat_pp",   # superset of all three members' columns
        "n_features": 812,
        # Blends have no wrapper, so the results tree cannot be derived from
        # one. This is the model directory their backtest_run was scored under.
        "scored_under": "lgb",
        "backtest_run": "x3_int_e60tenc_c05",
        "why": (
            "Highest payout of anything in the project at 0.099528, and the "
            "first slot spanning BOTH labels and two model classes: a 20-day "
            "LightGBM on 36 columns, a 60-day LightGBM on 763, and a five-member "
            "ridge blend.\n\n"
            "The members are chosen for disagreement. On predictions "
            "SACI_01 ranks 0.60 against SACI_03 and 0.46 against CURUPIRA_05, "
            "and SACI_03 ranks 0.49 against CURUPIRA_05 -- against 0.91 for "
            "CURUPIRA_01 vs _03, which is what a redundant pair looks like."
        ),
        "backtest": {
            # DIVERGES FROM LIVE. This blend was measured with SACI_03 fitted on
            # the 60-day label. SACI_03 is now a submitted slot in its own right
            # and fits the payable 20-day label, and it carries ONE cached fit,
            # so this blend inherits the retargeted member. The numbers below
            # describe the arm as backtested, not as it now runs. Deliberate:
            # a per-role cache does not exist, and a submitted slot predicting
            # the wrong horizon was the worse of the two problems.
            "scored_on": "target_binned_return_20",
            "corr": 0.153321, "corr_sharpe": 1.9278, "corr_max_dd": 0.5894,
            "mmc": 0.084196, "mmc_sharpe": 1.2950,
            "payout": 0.099528, "payout_sharpe": 1.3941, "payout_max_dd": 1.0523,
            "n_periods": 470,
        },
        "slot_members": [
            {"name": "SACI_01", "weight": 0.3333},
            {"name": "SACI_03", "weight": 0.3333},
            {"name": "CURUPIRA_05", "weight": 0.3334},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "THE MARGIN IS NOT SIGNIFICANT: +0.006510 over its best member at "
            "t = +0.31, paired Newey-West lag 20 over 470 periods (~24 effective "
            "independent windows). A sibling blend swapping SACI_03 for the "
            "60-day RAW arm scored 0.097764 at t = +1.31 -- lower payout, but "
            "four times the t and a better sharpe (1.4754 vs 1.3941). This slot "
            "was chosen on payout; that one is the stronger evidence.",
            "Its payout advantage rests on SACI_03, whose standalone payout "
            "drawdown is 10.9473. The blend's own drawdown is 1.0523, so the "
            "combination is doing the work -- but a member that volatile is a "
            "reason to size this slot conservatively.",
            "Scored on 470 periods, not 510: the 60-day label is NaN over the "
            "panel's last 40 dates, so this is not directly comparable to the "
            "510-period slots.",
            "SACI_03 is NOT submitted on its own -- it exists only as a member "
            "here. Staking it directly would put money on the worst risk profile "
            "in the registry.",
        ],
    },
    "SACI_04": {
        "wrapper": "lgb",
        "feature_set": "raw",
        "n_features": 22,
        "backtest_run": "fe_lgb_reg_raw",
        "why": (
            "LightGBM on the 22 shipped features, nothing added. It exists as "
            "the non-ridge anchor of CURUPIRA_SACI_03: a booster on the raw set "
            "ranks 0.705-0.751 against the ridge book, where the ridge variants "
            "sit at 0.883-0.926 against each other, so it brings a model class "
            "no fourth ridge arm can.\n\n"
            "Pinned, not searched. The cadence study showed the pinned config "
            "matching a 50-draw search at this cadence (payout 0.083272 against "
            "0.082061) for 1/90th of the compute -- and that matters here, not "
            "just as economy: a searched refit takes ~159 min against a ~66 min "
            "submission window, so the searched version is not deployable at all."
        ),
        "backtest": {
            "corr": 0.149754, "corr_sharpe": 1.8080, "corr_max_dd": 0.6806,
            "mmc": 0.068297, "mmc_sharpe": 1.1741,
            "payout": 0.083272, "payout_sharpe": 1.2918, "payout_max_dd": 1.1964,
            "n_periods": 510,
        },
        "fit_override": {
            "fixed_params": {
                "n_estimators": 4327, "learning_rate": 0.002572,
                "max_depth": 8, "num_leaves": 255,
                "colsample_bytree": 0.070213, "subsample": 0.816153,
                "subsample_freq": 1, "min_child_samples": 313,
                "reg_lambda": 1.757511,
            },
        },
        "fallback_feature_set": None,   # already the simplest set
    },
    "CURUPIRA_SACI_03": {
        "wrapper": None,
        "feature_set": "allfeat_pp",
        "n_features": 812,
        # Blends have no wrapper, so the results tree cannot be derived from
        # one. This is the model directory their backtest_run was scored under.
        "scored_under": "lgb",
        "backtest_run": "saci04_c05_c08",
        "why": (
            "The highest payout in the project at 0.100904, and the best "
            "evidenced blend here: it beats TWO of its three members at "
            "|t| > 2 -- +0.017135 over SACI_04 at t = +4.35 and +0.011115 over "
            "CURUPIRA_05 at t = +2.26. Almost nothing in this study clears that "
            "bar, and no other blend beats more than one member.\n\n"
            "It works because the members are strong AND unalike: a LightGBM on "
            "22 raw features, a five-member ridge feature-block blend, and a "
            "slot blend carrying the 60-day label. Every previous attempt to "
            "blend WEAK decorrelated fragments -- the feature groups, at a "
            "median rho of 0.129 -- landed at or below a single 22-feature "
            "model. Strong and unalike beats orthogonal and weak."
        ),
        "backtest": {
            "corr": 0.157742, "corr_sharpe": 1.9401, "corr_max_dd": 0.6420,
            "mmc": 0.085130, "mmc_sharpe": 1.4074,
            "payout": 0.100904, "payout_sharpe": 1.5053, "payout_max_dd": 1.0301,
            "n_periods": 470,
        },
        "slot_members": [
            {"name": "SACI_04", "weight": 0.3333},
            {"name": "CURUPIRA_05", "weight": 0.3333},
            {"name": "CURUPIRA_08", "weight": 0.3334},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "IT IS NOT DISTINGUISHABLE FROM HOLDING CURUPIRA_08 ALONE: "
            "+0.001089 at t = +0.28. C08 is the binding member -- the blend's "
            "edge over the other two does not translate into an edge over the "
            "best one. Deploy it for the diversification and the sharpe "
            "(1.5053 against C08's 1.4713), not for expected return.",
            "CURUPIRA_05 and CURUPIRA_08 already carry 2.18 and 2.46 NMR and "
            "correlate 0.885 with each other. This slot adds a third position "
            "on top of that pair, so it concentrates rather than spreads unless "
            "the stake comes out of their allocation.",
            "Scored on 470 periods, not 510: CURUPIRA_08 carries the 60-day "
            "label, which is NaN over the panel's last 40 dates.",
        ],
    },
    "SACI_05": {
        "wrapper": "lgb",
        "feature_set": "cumlag3",
        "n_features": 88,
        "backtest_run": "e05_cumlag3",
        "why": (
            "The 22 raw features plus every lag 1..3 of them, from experiment "
            "E05. It is here for its RISK PROFILE, not its return: payout "
            "drawdown 0.5610 against 1.1964 for the same model on the raw 22 -- "
            "less than half -- which made it the best calmar (payout / drawdown) "
            "of any lag depth tried, 0.1367 against 0.0700.\n\n"
            "DO NOT STAKE IT ALONE. Its payout is the lowest of any deployed "
            "single model at 0.076671, below raw's 0.083272 (t = -1.52). It "
            "exists as the low-drawdown member of CURUPIRA_SACI_04, where "
            "pairing it with CURUPIRA_05 raises payout to 0.093888 while keeping "
            "the drawdown at 0.5182."
        ),
        "backtest": {
            "corr": 0.134524, "corr_sharpe": 1.7269, "corr_max_dd": 0.1652,
            "mmc": 0.063218, "mmc_sharpe": 1.0580,
            "payout": 0.076671, "payout_sharpe": 1.1672, "payout_max_dd": 0.5610,
            "n_periods": 510,
        },
        "fit_override": {
            "fixed_params": {
                "n_estimators": 4327, "learning_rate": 0.002572,
                "max_depth": 8, "num_leaves": 255,
                "colsample_bytree": 0.070213, "subsample": 0.816153,
                "subsample_freq": 1, "min_child_samples": 313,
                "reg_lambda": 1.757511,
            },
        },
        # Losing the lags leaves the plain 22 -- a materially different model,
        # but a working one. The lags are a within-symbol backward shift of
        # columns that are always present, so this should never fire.
        "fallback_feature_set": "raw",
    },
    "CURUPIRA_SACI_04": {
        "wrapper": None,
        # allfeat_pp does NOT contain SACI_05's 66 lag columns, so the single
        # frame load_slot_blend builds was missing them and this slot fell to
        # Numerai's example predictions. allfeat_pp_lag3 is allfeat_pp + cumlag3.
        "feature_set": "allfeat_pp_lag3",
        "n_features": 878,
        # Blends have no wrapper, so the results tree cannot be derived from
        # one. This is the model directory their backtest_run was scored under.
        "scored_under": "lgb",
        "backtest_run": "x_lag3_c05",
        "why": (
            "The best risk-adjusted slot in the book: calmar 0.1812 (payout per "
            "unit of worst-case drawdown), ahead of every single model and every "
            "other blend. It is also the rare blend that improves BOTH sides -- "
            "payout 0.093888 against CURUPIRA_05's 0.086318 while cutting the "
            "payout drawdown from 0.6182 to 0.5182. Most blends here trade one "
            "for the other.\n\n"
            "Two members that fail differently: a lagged LightGBM with the "
            "lowest drawdown and the lowest return in the book, and a five-member "
            "ridge feature-block blend with the return. Their prediction rank "
            "correlation is 0.46."
        ),
        "backtest": {
            "corr": 0.151251, "corr_sharpe": 1.9590, "corr_max_dd": 0.2006,
            "mmc": 0.078763, "mmc_sharpe": 1.3065,
            "payout": 0.093888, "payout_sharpe": 1.4182, "payout_max_dd": 0.5182,
            "n_periods": 510,
        },
        "slot_members": [
            {"name": "SACI_05", "weight": 0.5},
            {"name": "CURUPIRA_05", "weight": 0.5},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "The payout margin over CURUPIRA_05 is +0.007570 at t = +1.81 -- the "
            "strongest of any blend here, but still short of |t| > 2 on ~26 "
            "effective independent windows. Read it as 'no worse than C05 and "
            "better diversified', not as an established edge.",
            "Its case rests on calmar, which is a RATIO OF A PER-PERIOD MEAN TO "
            "A CUMULATIVE DRAWDOWN. That scales with 1/n_periods, so it is "
            "comparable only against arms measured on the same window. Do not "
            "rank it against the 470-period slots on calmar.",
            "SACI_05's lags are a within-symbol backward shift, so a token's "
            "first three rows in the live panel have no history and fall back to "
            "the neutral 0.5. Verified that no LIVE row is affected -- live rows "
            "are the newest periods and the best covered -- but the training "
            "block's first 3 rows per symbol differ slightly from the offline "
            "dataset unless the panel loads 3 extra periods of history.",
        ],
    },
    "SACI_06": {
        "wrapper": "lgb_ktz",
        "feature_set": "tenc_onehot",
        "n_features": 763,
        "backtest_run": "e29_tenc_onehot",
        "why": (
            "The E29 lgb_ktz tune on tenc_onehot, and the first slot whose live "
            "fit IS a from-scratch search rather than a pinned config: lgb_ktz's "
            "grid encodes the E27 (Kalfa-Timmermann-van der Zwan) priors -- dense "
            "low learning_rate, low colsample, small num_leaves, L=lr*n_est<=40 -- "
            "so every refit re-selects hyperparameters inside E27's good corner "
            "by purged/embargoed CV. In E29 the search landed on num_leaves=2, "
            "lr~=0.0035, colsample~=0.20 on this dataset, and payout 0.089991 at "
            "corr-sharpe 1.8387.\n\n"
            "Unlike SACI_03 (same wrapper family and dataset, but a 60-day model "
            "fitted on the 20-day label), this is a 20-DAY model scored on the "
            "20-day label -- backtest and live predict the same quantity, so the "
            "numbers below describe what it submits.\n\n"
            "Paired against the deployed single-model slots on 510 shared dates "
            "it is indistinguishable from the staked trio (|t|<=1.05) and beats "
            "only SACI_05 significantly; it loses to every blend. Its case is "
            "diversification -- a LightGBM on 763 columns against a ridge-heavy "
            "staked book -- not standalone payout. Size it as a blend member, "
            "not a large solo stake."
        ),
        "backtest": {
            # Scored AND fitted on target_binned_return_20 -- no cross-label gap.
            "scored_on": "target_binned_return_20",
            "corr": 0.157183, "corr_sharpe": 1.8387, "corr_max_dd": 0.6194,
            "mmc": 0.074273, "mmc_sharpe": 1.1989,
            "payout": 0.089991, "payout_sharpe": 1.3119, "payout_max_dd": 1.0529,
            "n_periods": 510,
        },
        "fit_override": {
            # Reproduce the E29 backtest exactly: it searched 60 draws per refit
            # (the shared FIT default is 50). Everything else -- 20-day target,
            # embargo 20, 6-month window/refit, 5 folds, seed 42 -- already
            # matches FIT, and lgb_ktz pins min_child_samples inside its own grid,
            # so no fixed_params block is needed here. The live refit runs the
            # same purged/embargoed walk-forward search that produced e29.
            "n_iter": 60,
        },
        # If the one-hot vocabulary cannot be built live, degrade to tenc (23),
        # exactly as SACI_03 does. The lgb_ktz grid still applies to the narrower
        # frame -- num_leaves and the learning schedule do not depend on width.
        "fallback_feature_set": "tenc",
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
            # n_features_varies: this block is one column per symbol carrying
            # ONEHOT_MIN_DATES of history, so its width tracks the live universe
            # rather than the model. 740 is what the backtest fitted and is kept
            # for the record, but pinning it made the blend unrefittable the
            # moment a symbol crossed the threshold -- refreshing train.parquet
            # on 2026-08-23 took it to 742 and left both blends with no usable
            # cache at all. The other four members have genuinely fixed widths
            # and stay pinned.
            {"name": "onehot", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"^feature_sym_(?!tenc$)", "n_features": 740,
             "n_features_varies": True},
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
            "corr": 0.148223, "corr_sharpe": 1.9393, "corr_max_dd": 0.5838,
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
            # DELIBERATELY NOT RETARGETED, unlike SACI_03. This slot is never
            # uploaded, so nothing it produces is ever scored on the payable
            # label directly -- it exists to decorrelate CURUPIRA_07, _08,
            # CURUPIRA_SACI_01 and _03, and it shares allfeat_pp with
            # CURUPIRA_05. The 60-day horizon IS its diversification: refit it on
            # the 20-day label and it collapses toward CURUPIRA_05, those four
            # blends stop reproducing their backtests, and the mean-variance
            # weights in stake.py -- solved on the current payout covariance and
            # carrying 4.9162 NMR through CURUPIRA_08 -- stop describing the book.
            "scored_on": "target_binned_return_60",
            "corr": 0.098256, # corr_sharpe and corr_max_dd were copies of the payout row -- identical
            # to payout_sharpe and payout_max_dd to four decimals, which is not
            # something two different series do. The payout figures below are
            # right (results/cache/cross_target_payout.csv reproduces all three
            # exactly), but the 20-day re-score has no CORR series cached, so
            # these cannot be recovered. None beats a wrong number.
            "corr_sharpe": None, "corr_max_dd": None,
            "mmc": 0.065089, "mmc_sharpe": 0.7027,
            "payout": 0.074914, "payout_sharpe": 0.7027, "payout_max_dd": 7.8815,
            "n_periods": 470,
        },
        "members": [
            # n_features_varies: this block is one column per symbol carrying
            # ONEHOT_MIN_DATES of history, so its width tracks the live universe
            # rather than the model. 740 is what the backtest fitted and is kept
            # for the record, but pinning it made the blend unrefittable the
            # moment a symbol crossed the threshold -- refreshing train.parquet
            # on 2026-08-23 took it to 742 and left both blends with no usable
            # cache at all. The other four members have genuinely fixed widths
            # and stay pinned.
            {"name": "onehot", "wrapper": "ridge_glmnet", "weight": 0.2,
             "include": r"^feature_sym_(?!tenc$)", "n_features": 740,
             "n_features_varies": True},
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

    # ---------------------------------------------------------------- #
    # The neural family
    # ---------------------------------------------------------------- #

    "IA_01": {
        "wrapper": "mlp",
        "feature_set": "fenccausal",
        "n_features": 44,
        "backtest_run": "a2_mlp_fenccausal",
        "why":
            "The first neural slot in the book, and the only one that can be built "
            "live. It is a plain feedforward MLP on the 22 raw features plus their "
            "22 CAUSAL per-symbol means (build_feature_encoding --mode causal): an "
            "expanding mean of each symbol's own past, so a row never enters its own "
            "encoding and no target is touched.\n\n"
            "It is here because it beats the torch LSTM outright on identical data -- "
            "CORR 0.15812 against 0.10495 on the raw 22, +51% -- which is why the "
            "sequence model is NOT in this registry. The LSTM's own study found the "
            "recurrence contributes nothing (seq_length 1 scores 0.10543 against "
            "seq_length 21's 0.10445) and that it is largely a per-symbol lookup "
            "table: a 405-number table of each symbol's mean target reproduces ~70% "
            "of its out-of-sample signal.\n\n"
            "NOT ON THE ACCOUNT, so it is in NOT_SUBMITTED -- it exists to be the "
            "second leg of CU_SA_IA_BO_01. Same reason CURUPIRA_05_60 is a component.",
        "backtest": {"corr": 0.16172, "corr_sharpe": 1.8094, "corr_max_dd": 0.6998, "mmc": 0.082492, "mmc_sharpe": 1.2392, "payout": 0.098664, "n_periods": 510, "payout_sharpe": 1.3391, "payout_max_dd": 1.1786},
        "fallback_feature_set": "raw",
        "caveats": [
            "The mlp family has FOUR runs in a book of 500+. Its 0.16172 CORR is one "
            "configuration, never replicated across seeds. The same single-run "
            "evidence on the LSTM side (cumlag3, +0.0088 CORR at t +2.47) reversed "
            "sign when redrawn at a second seed, so treat this as promising rather "
            "than established until it is replicated.",
            "fenccausal became live-buildable only recently; every earlier fenccausal "
            "number came from the offline dataset. src/live/features.py:add_fenccausal "
            "reproduces it bit-exactly (max diff 0.000e+00 over 504,196 rows), and the "
            "live path computes the shrinkage prior from non-live rows only so today's "
            "features cannot move a historical encoding.",
        ],
    },

    "CU_SA_IA_BO_01": {
        "wrapper": None,
        "scored_under": "lgb",
        "feature_set": "allfeat_pp_fenc",
        "n_features": 834,
        "backtest_run": "boto01_slot_mlpfenc",
        "why":
            "Curupira + Saci + the neural family, 50/50 over CURUPIRA_SACI_03 and "
            "IA_01. It is the first slot in the book to combine the tabular ensemble "
            "with a neural model.\n\n"
            "Against CURUPIRA_SACI_03 alone, paired Newey-West at lag 20 over 470 "
            "periods: CORR +0.010429 (t +2.83), MMC +0.005178 (t +1.58), payout "
            "+0.006221 (t +1.72). The CORR gain clears the house |t| >= 2 bar; the "
            "PAYOUT gain does not. Deploy it for the correlation improvement and the "
            "MMC, not on the expectation that the payout edge is established.\n\n"
            "The member choice is measured, not assumed: the same blend built with the "
            "RAW-feature MLP instead scores payout 0.09975, i.e. WORSE than "
            "CURUPIRA_SACI_03 alone (t -0.39). The causal feature encoding is what "
            "makes this slot work.",
        "backtest": {"corr": 0.168171, "corr_sharpe": 1.9179, "corr_max_dd": 0.6778, "mmc": 0.090308, "mmc_sharpe": 1.4182, "payout": 0.107125, "n_periods": 470, "payout_sharpe": 1.5146, "payout_max_dd": 1.1568},
        "slot_members": [
            {"name": "CURUPIRA_SACI_03", "weight": 0.5},
            {"name": "IA_01", "weight": 0.5},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "Payout +0.006221 at t = +1.72 is under the |t| >= 2 bar this repo holds "
            "itself to. On ~26 effective independent windows that is the expected "
            "outcome and is itself the finding -- it is not evidence of no effect, but "
            "it is not evidence of one either.",
            "Drawdown is slightly WORSE than the member it improves on: dd(pay) 1.16 "
            "against CURUPIRA_SACI_03's 1.03, and calmar 0.093 against 0.098. This "
            "slot buys level, not stability.",
            "It inherits CURUPIRA_SACI_03, which has been falling through to Numerai's "
            "EXAMPLE predictions in recent rounds because its members could not all be "
            "resolved. If that is still true this slot inherits the same fallback -- "
            "check round_report.json for source 'example' before trusting a live score.",
            "The torch LSTM is deliberately absent. Adding it at 20% scored payout "
            "0.10955 in backtest, but src/live/ has no torch path at all, so that "
            "blend cannot be produced for a live round.",
        ],
    },
    "CURUPIRA_BOITATA_01": {
        # CURUPIRA_08 + CU_SA_IA_BO_01, rank-averaged 50/50. Fits nothing of its
        # own; both members are already-deployed slots, so this reuses their
        # caches exactly as CURUPIRA_06/07 reuse theirs -- a zero-fit tail slot.
        "wrapper": None,
        # allfeat_pp_fenc (834) is the richer of the two members' feature sets
        # (CURUPIRA_08's allfeat_pp is 812; CU_SA_IA_BO_01's allfeat_pp_fenc adds
        # the 22 causal means), so one frame built from it is a superset of both.
        "scored_under": "ridge_glmnet",
        "feature_set": "allfeat_pp_fenc",
        "n_features": 834,
        "backtest_run": "ens_scr_c08_bo",
        "why": (
            "The risk-adjusted frontier point of the whole book. Against every "
            "benchmark family and grid ensemble screened over the CURUPIRA/ktz "
            "panel, this 50/50 of CURUPIRA_08 and CU_SA_IA_BO_01 has the highest "
            "payout SHARPE measured -- 1.5758, above CU_SA_IA_BO_01's 1.5146 and "
            "the staked portfolio's 1.566 -- at a payout (0.106399) statistically "
            "identical to CU_SA_IA_BO_01's best-member 0.107125 (paired "
            "Newey-West t = -0.21). It also cuts the drawdown: dd(pay) 1.0799 "
            "against CU_SA_IA_BO_01's 1.1568. It buys stability at unchanged "
            "level, which is the only thing blending on this panel ever buys."
        ),
        "backtest": {
            "corr": 0.164715, "corr_sharpe": 1.9828, "corr_max_dd": 0.6603,
            "mmc": 0.089928, "mmc_sharpe": 1.4782,
            "payout": 0.106399, "payout_sharpe": 1.5758, "payout_max_dd": 1.0799,
            "n_periods": 470,
        },
        "slot_members": [
            {"name": "CURUPIRA_08", "weight": 0.5},
            {"name": "CU_SA_IA_BO_01", "weight": 0.5},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "It is largely DOUBLING DOWN on CURUPIRA_08, not diversifying. "
            "CU_SA_IA_BO_01 is CURUPIRA_SACI_03 + IA_01, and CURUPIRA_SACI_03 "
            "already contains CURUPIRA_08 -- so this concentrates CURUPIRA_08 "
            "exposure. The sharpe gain is real but it is a lever into the "
            "book's best signal, not a new one.",
            "The payout gain over CU_SA_IA_BO_01 is NOT established: +0.00 at "
            "t = -0.21 means the two are indistinguishable on level. Deploy it "
            "for the drawdown and the sharpe, not for more payout.",
            "It inherits both members' fallbacks. CU_SA_IA_BO_01 in turn inherits "
            "CURUPIRA_SACI_03, which has fallen through to Numerai's EXAMPLE "
            "predictions in some rounds; if either member falls back, this blend "
            "does too. Check round_report.json for source 'example' before "
            "trusting a live score.",
            "SCORED ON 470 PERIODS, NOT 510, because it carries CURUPIRA_08's "
            "60-day member CURUPIRA_05_60, whose label is NaN over its last 40 "
            "dates.",
        ],
    },
    "IAUARETE_01": {
        # The CatBoost leg of CURUPIRA_CAIPORA_01. A NOT_SUBMITTED component, the
        # booster sibling of SACI_06's lgb_ktz search and IA_01's live MLP.
        "wrapper": "catboost_ktz",
        "feature_set": "fenccausal",
        "n_features": 44,
        "backtest_run": "e32_sel_fenccausal",
        "why": (
            "The CatBoost leg of CURUPIRA_CAIPORA_01. It fits catboost_ktz on the "
            "22 raw features plus their 22 CAUSAL per-symbol means (fenccausal -- "
            "the same 44 columns IA_01 uses, which src/live/features.py already "
            "builds live and bit-exactly). Its live fit SWEEPS the 12-cell E27/KTZ "
            "grid exactly as the e32_sel_fenccausal backtest did: "
            "build_wrapper('catboost_ktz') is iterations=2000 x lr{.005,.01,.02} x "
            "depth{2,4} x rsm{.1,.2}, and fit_override n_iter=12 sweeps all twelve "
            "and CV-picks one -- so the live refit reproduces the backtest's "
            "protocol, not just an approximation of it.\n\n"
            "NOT ON THE ACCOUNT, so it is in NOT_SUBMITTED. It exists to be the "
            "third leg of CURUPIRA_CAIPORA_01, the same reason IA_01 and "
            "CURUPIRA_05_60 are components."
        ),
        "backtest": {
            "scored_on": "target_binned_return_20",
            "corr": 0.156464, "corr_sharpe": 1.8311, "corr_max_dd": 0.6184,
            "mmc": 0.077381, "mmc_sharpe": 1.2380,
            "payout": 0.093027, "payout_sharpe": 1.3467, "payout_max_dd": 1.0665,
            "n_periods": 510,
        },
        "fit_override": {
            # Sweep the 12-cell catboost_ktz grid exhaustively, exactly as e32
            # did. The grid pins iterations=2000 internally, so no model_options
            # or fixed_params are needed -- every other field matches shared FIT.
            "n_iter": 12,
        },
        "fallback_feature_set": "raw",
        "caveats": [
            "Single-run evidence, like every ktz benchmark: e32_sel_fenccausal is "
            "one config on one dataset, never replicated across seeds. It is a "
            "blend member here, not a solo bet.",
            "Its CV picks one of the 12 cells each refit, so a LIVE fit can select "
            "a different cell than e32 did on a given window -- the numbers above "
            "are indicative of the protocol, not a promise of the exact estimator.",
        ],
    },
    "CURUPIRA_CAIPORA_01": {
        # CURUPIRA_05 + CURUPIRA_08 + the CatBoost leg IAUARETE_01, 1/3 each. A
        # zero-fit slot blend; the three members are fitted for their own sake.
        "wrapper": None,
        # allfeat_pp_fenc (834) is the superset covering CURUPIRA_05/08's
        # allfeat_pp columns AND IAUARETE_01's fenccausal means -- the same frame
        # CU_SA_IA_BO_01 builds.
        "scored_under": "ridge_glmnet",
        "feature_set": "allfeat_pp_fenc",
        "n_features": 834,
        "backtest_run": "ens_scr_c05_c08_cat",
        "why": (
            "CURUPIRA_05 + CURUPIRA_08 + the CatBoost leg IAUARETE_01, rank-averaged "
            "1/3 each. In the ktz-benchmark ensemble screen it was the ONLY priced "
            "blend with a positive payout delta over its own best member (+0.0037 "
            "over CURUPIRA_08, though t = +0.69, not established), and it holds the "
            "lowest drawdown of the top group -- dd(pay) 0.9648, calmar 0.107, burn "
            "33 of 470 days -- at payout 0.103468 and sharpe 1.5504. Its case is "
            "diversification: a booster on the causal encoding added to a "
            "ridge-heavy pair, a signal no fourth ridge arm brings."
        ),
        "backtest": {
            "corr": 0.158744, "corr_sharpe": 1.9600, "corr_max_dd": 0.6104,
            "mmc": 0.087593, "mmc_sharpe": 1.4554,
            "payout": 0.103468, "payout_sharpe": 1.5504, "payout_max_dd": 0.9648,
            "n_periods": 470,
        },
        "slot_members": [
            {"name": "CURUPIRA_05", "weight": 1.0 / 3},
            {"name": "CURUPIRA_08", "weight": 1.0 / 3},
            {"name": "IAUARETE_01", "weight": 1.0 / 3},
        ],
        "fallback_feature_set": "raw",
        "caveats": [
            "The payout edge over CURUPIRA_08 is NOT established: +0.0037 at "
            "t = +0.69 on ~23 effective independent windows. Deploy it for the "
            "drawdown and the diversification, not for established level.",
            "It carries CURUPIRA_08's 60-day member CURUPIRA_05_60, so it is scored "
            "on 470 periods, not 510.",
            "IAUARETE_01 is a fresh CatBoost search each refit and single-run "
            "evidence; if it cannot be fit or the fenccausal build fails, the blend "
            "degrades through the members' raw fallback.",
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
ORDER = ["CURUPIRA_02", "CURUPIRA_01", "SACI_04", "SACI_01",
         "CURUPIRA_03", "SACI_05",
         "CURUPIRA_04", "SACI_03", "IA_01", "SACI_02",
         # IAUARETE_01 is a catboost_ktz search: ~300 s, between SACI_02 (120)
         # and CURUPIRA_05 (440), so the fitting slots stay cheapest-first.
         "IAUARETE_01",
         "CURUPIRA_05", "CURUPIRA_05_60", "SACI_06", "CURUPIRA_06", "CURUPIRA_07",
         "CURUPIRA_08", "CURUPIRA_SACI_01", "CURUPIRA_SACI_02",
         "CURUPIRA_SACI_03", "CURUPIRA_SACI_04", "CU_SA_IA_BO_01",
         # Zero-fit slot blends: each rank-averages already-fitted slots, so both
         # must run AFTER their members -- last in the order, like CURUPIRA_06/07.
         # CURUPIRA_BOITATA_01 = C08 + CU_SA_IA_BO_01; CURUPIRA_CAIPORA_01 =
         # CURUPIRA_05 + CURUPIRA_08 + IAUARETE_01.
         "CURUPIRA_BOITATA_01", "CURUPIRA_CAIPORA_01"]

# Slots that are fitted and cached but never uploaded. CURUPIRA_05_60 exists
# only as a component of CURUPIRA_07; submitting it would stake real money on
# the worst-scoring arm in the study. IA_01 and IAUARETE_01 are the neural and
# CatBoost legs of CU_SA_IA_BO_01 and CURUPIRA_CAIPORA_01 respectively -- fitted
# so a blend can reuse them, never uploaded alone.
NOT_SUBMITTED = {"CURUPIRA_05_60", "IA_01", "IAUARETE_01"}


def is_blend(spec):
    """True when a slot rank-averages feature-block members it fits itself."""
    return bool(spec.get("members"))


def backtest_target(name):
    """
    The label a slot's backtest was SCORED on, which is not always what it fits.

    They were the same thing until SACI_03 was promoted to a submitted slot:
    its hyperparameters were selected on the 60-day label, but Numerai pays on
    the 20-day one, so it now FITS target_binned_return_20 while its backtest
    block still describes a 60-day study. Anything reasoning about the backtest
    (period counts, cross-slot comparisons) must ask this rather than reading
    fit["target"], or it will silently compare 470-period and 510-period arms.
    """
    spec = get(name)
    return spec["backtest"].get("scored_on") or spec["fit"]["target"]


def reused_slots():
    """
    Slots that another slot rank-averages, so a failure there does not stay put.

    SACI_03/04/05 are submitted models AND members of a CURUPIRA_SACI blend, so
    one bad fit shows up as several unrelated broken slots unless the report says
    which one to look at first.
    """
    out = set()
    for spec in MODELS.values():
        for member in (spec.get("slot_members") or []):
            out.add(member["name"])
    return out


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
