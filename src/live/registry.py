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
ORDER = ["CURUPIRA_02", "CURUPIRA_01", "CURUPIRA_03", "CURUPIRA_04",
         "CURUPIRA_05", "CURUPIRA_06"]


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
    spec["fit"] = dict(FIT)
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
