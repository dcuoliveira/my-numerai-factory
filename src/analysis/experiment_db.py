"""
One queryable index of every numbered study: what it asked, and where it got to.

    python3 -m src.analysis.experiment_db              # refresh, then print
    python3 -m src.analysis.experiment_db --sql "..."  # ad-hoc query
    python3 -m src.analysis.experiment_db --arms E09   # per-arm detail

Writes results/experiments.db, a SQLite file. That path is under a gitignored
directory ON PURPOSE -- see "what is stored where" below.

------------------------------------------------------------------------------
Why a database when experiments/EXPERIMENTS.md already exists
------------------------------------------------------------------------------

They answer different questions and neither replaces the other.

EXPERIMENTS.md is the durable record: it is tracked, diffable, and holds the
findings a study produced, written by hand once the study is over. It is what
you read in two years.

This is the operational view: which of nine studies is running RIGHT NOW, which
arm of twelve it is on, which arms already have artifacts, what they scored, and
how long each took. Those change by the minute and none of them belongs in a
markdown table that a human has to remember to update.

------------------------------------------------------------------------------
What is stored where, and why status is never stored
------------------------------------------------------------------------------

    STUDIES (this file)   the INTENT: id, title, the idea in a sentence, the
                          runner, which prefixes and model trees it writes.
                          Tracked in git, edited by hand, changes rarely.

    results/experiments.db   everything else, DERIVED on every run from
                          results/metrics/ and the process table. Gitignored,
                          disposable, rebuilt in under a second.

The split is the whole design. A tracker whose status is hand-maintained is
wrong the first time someone forgets to update it, and then it is worse than
nothing because it looks authoritative. So `refresh()` recomputes every status
field from the filesystem before any query runs, and the default command
refreshes first. There is deliberately no "mark this done" verb.

This mirrors what src/live/registry.py already does for the deployed slots: the
.py is the source of truth, the generated json is a cache that may go stale, and
the fix for staleness is to regenerate rather than to edit.
"""

import argparse
import json
import os
import sqlite3
import time

DB_PATH = os.path.join("results", "experiments.db")
METRICS_ROOT = os.path.join("results", "metrics", "crypto")

# --------------------------------------------------------------------------- #
# The intent. One entry per numbered study, added BEFORE it launches.
# --------------------------------------------------------------------------- #
#
# `prefix` is the reference handle from experiments/EXPERIMENTS.md: every
# artifact a study writes starts with it, which is what makes a study
# retrievable in one glob years later. `scout_prefix` is the downsampled
# wiring-check twin -- kept separate because a scout arm moves the split
# boundary and must never be counted as a result.
#
STUDIES = {
    "E01": {
        "title": "LightGBM feature blocks on the 60-day target",
        "idea": "Does any feature block beat raw when the label is the 60-day "
                "return instead of the 20-day one? Same 23 arms, same pinned "
                "booster, same cadence -- the label is the only thing varying.",
        "runner": "experiments/e01_blocks60_lgb_reg_crypto.sh",
        "prefix": "e01_", "scout_prefix": None,
        "models": ["lgb"], "expected_arms": 23,
    },
    "E02": {
        "title": "Group blend vs group concat, on LightGBM",
        "idea": "The ridge group-blend study re-asked of a model class whose "
                "answer should differ: fit each feature group as its own model "
                "and rank-average, or fit one model on the union?",
        "runner": "experiments/e02_group_blends_lgb_reg_crypto.sh",
        "prefix": "e02_", "scout_prefix": None,
        "models": ["lgb"], "expected_arms": 12,
    },
    "E03": {
        "title": "Group blend vs concat at leaf 50, with the one-hot block",
        "idea": "E02 again at a leaf size small enough that LightGBM can "
                "actually split on a one-hot column -- at leaf 313 it discards "
                "all 740 before training.",
        "runner": "experiments/e03_group_blends_onehot_lgb_reg_crypto.sh",
        "prefix": "e03_", "scout_prefix": None,
        "models": ["lgb"], "expected_arms": 12,
    },
    "E04": {
        "title": "Autoencoder codes as extra features, on LightGBM",
        "idea": "Do 8 bottleneck autoencoder codes add anything to the raw 22? "
                "One model, one thing varying -- the extra columns.",
        "runner": "experiments/e04_autoencoder_lgb_reg_crypto.sh",
        "prefix": "e04_", "scout_prefix": None,
        "models": ["lgb"], "expected_arms": 3,
    },
    "E05": {
        "title": "Cumulative feature lags, all 22 features",
        "idea": "Raw 22 plus every lag 1..p of all of them. A tree can split on "
                "both columns directly, so if lags help a booster it is for more "
                "history to condition on, not the representational fix ridge got.",
        "runner": "experiments/e05_cumlags_lgb_reg_crypto.sh",
        "prefix": "e05_", "scout_prefix": None,
        "models": ["lgb"], "expected_arms": 11,
    },
    "E06": {
        "title": "Cumulative lags, volatility group only",
        "idea": "E05 narrowed to the volatility group and its lags, to separate "
                "'lags help' from 'more columns help'. Same runner, SCOPE=volatility.",
        "runner": "experiments/e05_cumlags_lgb_reg_crypto.sh",
        "prefix": "e06_", "scout_prefix": None,
        "models": ["lgb"], "expected_arms": 11,
    },
    "E07": {
        "title": "A feedforward net against the deployed ridge, both tasks",
        "idea": "First non-linear, non-tree function class on the board. sklearn "
                "MLP on raw 22 at the CURUPIRA cadence, as regression against "
                "CURUPIRA_02's backtest run and as classification against its "
                "clf twin. The estimator is the only thing varying.",
        "runner": "experiments/e07_mlp_crypto.sh",
        "prefix": "e07_", "scout_prefix": "e07scout_",
        "models": ["mlp"], "expected_arms": 2,
    },
    "E08": {
        "title": "The MLP on the four one-hot datasets the book runs on",
        "idea": "Can an MLP use the 740 one-hot ticker dummies? Ridge cannot "
                "form a per-ticker interaction from them and LightGBM discards "
                "them outright -- fe_lgb_reg_onehot came back BYTE-IDENTICAL to "
                "raw -- but a dense first layer over a one-hot IS an embedding "
                "lookup. Scoped down from 12 datasets to 4 because E15 covered "
                "the rest; these are the last of the 9 the submitted book trains "
                "on. Was blocked on a scaler bug that sent rare dummies to "
                "z = 197; ScaleContinuousOnly fixed it (197.48 -> 2.29).",
        "runner": "experiments/e08_mlp_onehot_crypto.sh",
        "prefix": "e08_", "scout_prefix": "e08scout_",
        "models": ["mlp", "torch_mlp"], "expected_arms": 8,
    },
    "E09": {
        "title": "Does a torch MLP reproduce the sklearn one?",
        "idea": "Same architecture, same grid, same folds, different framework. "
                "A NULL is the success condition -- the two already agree at unit "
                "level, so this asks whether that survives the walk-forward.",
        "runner": "experiments/e09_torch_mlp_crypto.sh",
        "prefix": "e09_", "scout_prefix": "e09scout_",
        "models": ["mlp", "torch_mlp"], "expected_arms": 2,
    },
    "E10": {
        "title": "Walk-forward stacking: a fitted meta-learner over member predictions",
        "idea": "Every blend on the board is a FIXED rank average. This asks "
                "whether a meta-learner fitted on the members' out-of-sample "
                "predictions does better -- ridge for a signed linear combiner, "
                "MLP for a non-linear one, over single-class and multi-class "
                "member sets, against 1/N and constrained NNLS on the same "
                "members and the same dates. Each meta is tuned under BOTH inner "
                "CVs -- the book's purged chronological split and Wolpert's "
                "original shuffled k-fold (`_kf` arms) -- to price the purge.",
        "runner": "experiments/e10_stack_crypto.sh",
        "prefix": "e10_", "scout_prefix": "e10scout_",
        # 4 base + 3 sets x (2 metas x 2 inner-CVs + 2 controls) = 4 + 18. Only
        # the within-class sets (stackR, stackM) are run so far; stackX and the
        # kfold arms on stackX remain unrun, so the derived count sits below this.
        "models": ["ridge", "mlp"], "expected_arms": 22,
    },
    "E23": {
        "title": "Cross-class 1/N blends: does diversity buy what fitting could not?",
        "idea": "E10 showed a FITTED meta beats nothing when members are 0.86 "
                "correlated. The tabular book is one block (ridge/mlp/lgb/glmnet "
                "0.72-0.92); the torch LSTM is the only orthogonal thing on the "
                "board (0.35-0.41). So this asks the cheap question first: with "
                "NO fitting at all, how much does a plain 1/N rank average gain "
                "from adding a decorrelated member? Controls WITHOUT the LSTM "
                "isolate diversity from mere member count.",
        "runner": "experiments/e23_diverse_blend_crypto.sh",
        "prefix": "e23_", "scout_prefix": None,
        "models": ["ridge_glmnet"], "expected_arms": 4,
    },
    "E25": {
        "title": "Weighted and selection combiners over a diverse pool",
        "idea": "E10: fitting beats nothing when members are homogeneous. E23: a "
                "fixed 1/N cannot exploit a weak-but-orthogonal member because it "
                "cannot downweight anything. This is the untested cell -- eight "
                "walk-forward weighting/selection schemes over the SAME pool E23 "
                "blended, each against E23's 1/N on the identical members and "
                "dates. A tabular-only pool repeats every scheme without the "
                "orthogonal member, so a gain can be attributed to handling it.",
        "runner": "experiments/e25_weighted_combiners_crypto.sh",
        "prefix": "e25_", "scout_prefix": None,
        "models": ["ridge_glmnet"], "expected_arms": 16,
    },
    "E11": {
        "title": "Does a learned ticker embedding earn its place in the MLP?",
        "idea": "Two torch MLP arms on raw 22 differing only in whether the "
                "network sees which symbol a row is. The embedding is built the "
                "way prediction/nn/model.py builds the LSTM's -- one row per "
                "training symbol plus a zeroed padding row for unseen codes -- "
                "so the two models cannot disagree about what a symbol embedding "
                "is. Priors are against: 36% of a block's 460 symbols have under "
                "60 rows, and at embed_dim 32 the table is 4x the dense stack.",
        "runner": "experiments/e11_symbol_embedding_crypto.sh",
        "prefix": "e11_", "scout_prefix": "e11scout_",
        "models": ["torch_mlp", "torch_mlp_emb"], "expected_arms": 2,
    },
    "E12": {
        "title": "How wide should the ticker embedding be?",
        "idea": "Six torch MLP arms on raw 22 differing only in embedding width "
                "-- none, 4, 8, 16, 32, 64 -- each PINNED with --model-options "
                "while alpha, architecture, learning rate and batch size are "
                "still searched. E11 put the width in the grid and let one "
                "search pick; this asks it as six models, which gives a curve, "
                "five paired deltas and an answer to whether it degrades "
                "gracefully. The control has no embedding module at all, not a "
                "zero-width one.",
        "runner": "experiments/e12_embed_dim_crypto.sh",
        "prefix": "e12_", "scout_prefix": "e12scout_",
        "models": ["torch_mlp", "torch_mlp_emb"], "expected_arms": 6,
    },
    "E13": {
        "title": "Do autoregressive and distributed lags help the MLP?",
        "idea": "Four arms on one PINNED MLP -- the config e07_mlpreg_raw "
                "actually chose -- differing only in columns: contemporaneous "
                "only, + lags of the TARGET (AR), + lags of every FEATURE (DL), "
                "+ both (ADL). AR and DL are split so a move in ADL says which "
                "half caused it. A target lag under 20 periods is lookahead "
                "because the label is a 20-day forward return; build_adl refuses "
                "one. DL was already tried on lgb and did not help (E05 best "
                "depth ZERO, E06 t = -2.18..-2.30); the AR half is what is new.",
        "runner": "experiments/e13_adl_mlp_crypto.sh",
        "prefix": "e13_", "scout_prefix": "e13scout_",
        "models": ["mlp"], "expected_arms": 4,
    },
    "E14": {
        "title": "E13's ADL question again, each arm tuned for its own width",
        "idea": "E13 pinned a config chosen on 22 columns and applied it to 91, "
                "so its negative ADL result partly measured 'wrong shape' rather "
                "than 'lags do not help'. Here raw and adl each get a full "
                "50-draw search. Read WITH E13: pinning has no search noise but "
                "handicaps the wide arm, searching removes the handicap but adds "
                "noise E12 measured at 0.024 payout spread. Agreement between "
                "the two designs is worth more than either alone.",
        "runner": "experiments/e14_adl_searched_crypto.sh",
        "prefix": "e14_", "scout_prefix": "e14scout_",
        "models": ["mlp"], "expected_arms": 2,
    },
    "E15": {
        "title": "The MLP feature table -- 7 raw groups + 6 derived blocks, twice",
        "idea": "14 feature sets, each fitted by BOTH sklearn and torch, every "
                "arm fully searched: the 7 groups that partition the raw 22, "
                "plus tenc, spread, interact, fenccausal, rollsel and cumlag3. "
                "Two backends because E09 showed they agree on raw, so a block "
                "moving ONE and not the other is search noise -- a cheaper guard "
                "than any t-statistic against the failure E12 and E14 both hit, "
                "where a |t| > 2 traced to one bad draw. First study run wholly "
                "on the reverted 1e1 alpha grid.",
        "runner": "experiments/e15_mlp_feature_groups_crypto.sh",
        "prefix": "e15_", "scout_prefix": "e15scout_",
        "models": ["mlp", "torch_mlp"], "expected_arms": 28,
    },
    "E16": {
        "title": "Cumulative per-symbol differences on a lag ladder to one year",
        "idea": "Does handing the MLP DIFFERENCES rather than levels help? Six "
                "rungs -- 1 day, 5 week, 22 month, 66 quarter, 132 half-year, "
                "264 year -- each cumulative over those below it, in two "
                "variants: differences alone, and raw 22 plus differences. A "
                "ladder rather than every integer because 264 orders would add "
                "5,808 columns. The target difference is anchored at the "
                "20-period horizon: unanchored it correlates +0.2805 with the "
                "label, anchored -0.0114. Control is e15_mlp_raw, reused.",
        "runner": "experiments/e16_cumdiff_mlp_crypto.sh",
        "prefix": "e16_", "scout_prefix": "e16scout_",
        "models": ["mlp"], "expected_arms": 12,
    },
    # Was E17 until 2026-09-05: the id collided with the MLP hyperparameter-
    # sensitivity study below. Both were keyed "E17" in this dict and both write
    # the e17_ prefix, so the later (mlp) key silently overwrote this one -- this
    # study was absent from the tracker entirely, the exact failure the E30->E36
    # renumber fixed the same day. Renumbered to E37: E34/E35 are reserved for
    # E33's follow-ups and E36 is the renamed FARM grid. This study keeps the
    # e17_ prefix (e17_raw, e17_diff*, e17_lag*, e17_featdiff*, e17_rawdiff*,
    # e17_tgtdiff*); the mlp study narrows to e17_a32_ so study_of() routes
    # e17_a32_* there (longest prefix wins) and every other e17_* here. scan_arms
    # already separated them -- they write different model trees (ridge_glmnet vs
    # mlp) -- but study_of() could not, and now can.
    "E37": {
        "title": "Cumulative per-symbol differences to a year, on ridge_glmnet",
        "idea": "E16's question asked of the linear model, and asked as a BASIS "
                "question rather than a span one. Ladder 1,2,3,5,10,21,63,130,"
                "261 (261 dates = one calendar year here, measured from the "
                "panel at 261.0/yr), cumulative, as its own feature set and "
                "beside the raw 22. The span is already known null -- E05, E06, "
                "E13, E14 -- but span{f_t, f_t - f_t-k} == span{f_t, f_t-k}, so "
                "an unpenalised fit could not tell a difference set from a lag "
                "set and ridge can: ||b||^2 does not commute with a change of "
                "basis, and lag-1 autocorrelation of 0.82-0.99 means the "
                "lagged-level basis hands glmnet near-copies while the "
                "difference basis hands it near-orthogonal residuals. Levels 3, "
                "6 and 9 carry a featdiff/lag pair at identical span and feature "
                "count to isolate exactly that. Coverage is the caveat: an "
                "order-261 difference is real for only 61% of rows. The control "
                "is refitted on the CURRENT snapshot because w6r6_glmnet_reg is "
                "on 1,701 dates against today's 1,714.",
        "runner": "experiments/e17_cumdiff_glmnet_crypto.sh",
        # scout_prefix dropped (was e17scout_): a ridge study needs no scout, and
        # keeping it would re-collide with the mlp study, which owns e17scout_.
        "prefix": "e17_", "scout_prefix": None,
        "models": ["ridge_glmnet"], "expected_arms": 26,
    },
    "E18": {
        "title": "TabNet at library defaults, on every submitted slot's dataset",
        "idea": "First deep-tabular architecture on the board. dreamquark-ai's "
                "pytorch-tabnet, added to prediction/models.py as `tabnet` -- an "
                "attention-based net whose fitted model survives joblib, so it "
                "is deployable through the same refit path as every other MODELS "
                "entry, unlike the LSTM. One arm per DISTINCT training dataset a "
                "submitted slot fits on: raw, tenc, interact, fenccausal, "
                "cumlag3, tenc_onehot, allfeat, onehot_roll, allfeat_pp (9). "
                "Parameters are PINNED to the library defaults (n_d=n_a=8, "
                "n_steps=3, gamma=1.3, lambda_sparse=1e-3, lr=0.02, 100 epochs), "
                "so nothing is searched -- this is a viability probe, not a tuned "
                "model, on a panel that is small and noisy for a net this "
                "expressive. n_splits=2 not 5 because with fixed params the "
                "inner CV only produces an unused score; the per-period result "
                "is byte-identical either way (verified), so 2 halves the "
                "compute for free. No eval_set, to keep the fit leak-free.",
        "runner": "experiments/e18_tabnet_crypto.sh",
        "prefix": "e18_", "scout_prefix": "e18scout_",
        "models": ["tabnet"], "expected_arms": 9,
    },
    "E19": {
        "title": "TabPFN (tabular foundation model) at defaults, per submitted dataset",
        "idea": "E18's question asked of a tabular FOUNDATION model instead of a "
                "trained net. `tabpfn` (Prior-Labs TabPFN v2 regressor, added to "
                "prediction/models.py) solves each dataset by IN-CONTEXT learning "
                "-- no per-dataset training, no search, one forward pass -- so it "
                "is the cheapest arm to add and, on CPU, the most expensive to "
                "run: cost is O(context x query), so it wants a GPU. Each 39k-row "
                "block is subsampled to a few thousand context rows (a real "
                "fidelity/cost tradeoff, recorded in config), and the weights are "
                "the ungated v2 checkpoint vendored at models/tabpfn/ because the "
                "package's default download is gated + this box has no egress. "
                "Same 9 datasets as E18; the 5 narrow are the default set, the 4 "
                "wide (763-812 feat) exceed TabPFN's ~500-feature range and need "
                "feature reduction. Compare each arm to the deployed slot on the "
                "same dataset, paired Newey-West, as E18 does.",
        "runner": "experiments/e19_tabpfn_crypto.sh",
        "prefix": "e19_", "scout_prefix": "e19scout_",
        "models": ["tabpfn"], "expected_arms": 9,
    },
    "E20": {
        "title": "Exponential recency weighting on every submitted model's dataset",
        "idea": "Every fit in this project weights its training rows equally "
                "inside a rolling 130-date window -- a rectangular kernel with a "
                "hard edge. If the feature/target relationship drifts, recent "
                "dates should count for more, and GeneralisedRidge is the only "
                "estimator here that can express it, because W is one of its "
                "three free objects. Two arms per dataset -- uniform against "
                "w_scheme=exp_decay with half_life searched in DATES over "
                "32,45,65,90,130 -- structure pinned to identity and the same "
                "60-point alpha grid both sides, so W is the only difference. "
                "Nine datasets, being exactly the datasets behind the ten "
                "single-model slots; the other ten slots are blends of these, so "
                "a weighting that helps propagates for free. Floored at 32 dates "
                "because a 130-date block at a 21-date half-life keeps 59 "
                "effective dates, about 2.9 effective independent windows against "
                "6.5 uniform. Read a null as evidence about WEIGHTING, not "
                "staleness: refit-every is held at 130, and weighting the end of "
                "a block sharpens the fit for the START of the next one, which "
                "then goes stale for 130 more days. Numbered E20 because a "
                "concurrent session holds E18 (TabNet) and E19 (TabPFN).",
        "runner": "experiments/e20_expdecay_genridge_crypto.sh",
        "prefix": "e20_", "scout_prefix": "e20scout_",
        "models": ["gen_ridge"], "expected_arms": 18,
    },
    "E21": {
        "title": "Cluster regularisation on allfeat_pp -- shrink some blocks harder than others",
        "idea": "Every linear slot shrinks all 812 allfeat_pp columns with ONE "
                "alpha, and the repo already blames that: cumlagged_features "
                "states the mechanism as 'one alpha across the whole design "
                "matrix, so each added block raises the shrinkage every existing "
                "column receives', and CURUPIRA_04 on these 812 scores below the "
                "raw 22. GeneralisedRidge can give Delta's diagonal a per-block "
                "multiplier, so this tests three DIFFERENT mechanisms, not three "
                "doses of one. (a) level up: penalise a derived block harder than "
                "raw -- interactions 10x/100x, the 740 ticker dummies 10x/100x, "
                "all non-raw 10x. (b) level DOWN, which changes the alpha -> inf "
                "LIMIT: with raw nearly free the penalised blocks vanish and the "
                "model converges to the fit on raw alone rather than to a "
                "constant -- measured on a real block, flat ridge's prediction sd "
                "falls 1.03e-2 -> 1.64e-6 over alpha 1e2..1e6 while raw:p1e-6 "
                "plateaus at 3.3e-2. (c) dispersion not level: fuse within "
                "clusters, which is separately identified because rho never "
                "touches the group-mean direction. The blocks partition exactly: "
                "raw 22, spread 11, interact 14, rollsel 24, tenc 1, onehot 740.",
        "runner": "experiments/e21_cluster_reg_crypto.sh",
        "prefix": "e21_", "scout_prefix": "e21scout_",
        "models": ["gen_ridge"], "expected_arms": 11,
    },
    "E22": {
        "title": "The one-hot block: the penalty, the key (symbol vs ucid), the granularity",
        "idea": "E21 found penalising the 740 ticker dummies 10x to be a no-op "
                "(+0.000184, t +0.31, rho 0.998 with the control). This asks the "
                "three questions that leaves open, one change at a time. (1) THE "
                "PENALTY: 10x, one shared alpha, or NOT PENALISED -- an "
                "unpenalised dummy block is a SYMBOL FIXED EFFECTS "
                "specification, which nothing in this project has ever fitted. "
                "(2) THE KEY: the ticker is not a stable identity -- 30 tickers "
                "are reused by different coins (ACT covers three on disjoint "
                "date ranges) and 3 coins were renamed (GRAM->TON), so 12,606 "
                "rows (2.48%) sit on a ticker whose meaning changed. `ucid` is "
                "the stable coin id, shipped in the original dataset, and "
                "NOTHING in src/ had ever referenced it; keying on it gives 750 "
                "columns against 746. (3) THE GRANULARITY: three low/mid/large "
                "market-cap buckets instead of 746 identities, asking whether "
                "the persistence credited to the dummies is really a size "
                "effect. Plus arms where the one-hot REPLACES the raw 22 rather "
                "than joining it. Unpenalised is spelled p1e-6 because a literal "
                "p0 makes the solve singular.",
        "runner": "experiments/e22_onehot_reg_crypto.sh",
        "prefix": "e22_", "scout_prefix": "e22scout_",
        "models": ["gen_ridge"], "expected_arms": 13,
    },
    "E24": {
        "title": "Can TabPFN be pushed past its E19 default? Three Tier-1 levers",
        "idea": "E19 ran TabPFN handicapped (context 4096 ~= 10% of each block, "
                "n_estimators 4). This asks whether three cheap levers close the "
                "gap to the deployed book, each an arm differing from the E19 "
                "regressor baseline in ONE thing: (1) FULL STRENGTH -- context "
                "8192, n_estimators 'auto', memory_saving_mode auto to fit 8 GB; "
                "(2) CLASSIFIER HEAD -- TabPFNClassifier on the 5 target bins "
                "returning E[y|x], same context/estimators so only the head "
                "differs (added as model `tabpfn_clf`); (3) TEMPERATURE -- "
                "softmax_temperature swept {0.5,0.75,1.25} on fenccausal, "
                "full-strength. GPU only (deepfive); runs serialise on one card. "
                "Each arm compared to its E19 baseline and to the deployed slot "
                "on the same dataset, paired NW lag 20. The prize is Win B: a "
                "full-strength TabPFN that wins the neural-member swap "
                "SACI_03+TabPFN > SACI_03+IA_01 and so beats CU_SA_IA_BO_01.",
        "runner": "experiments/e24_tabpfn_tuning_crypto.sh",
        "prefix": "e24_", "scout_prefix": None,
        "models": ["tabpfn", "tabpfn_clf"], "expected_arms": 11,
    },
    "E26": {
        "title": "Sign-constrained GRR on allfeat_pp -- all betas positive, all betas negative",
        "idea": "A sign constraint is the one GRR knob that does not change the "
                "penalty: it leaves the objective alone and shrinks the FEASIBLE "
                "SET, turning the closed form into a box-constrained QP solved "
                "by coordinate descent with optimality judged on the KKT "
                "conditions. Three arms on allfeat_pp -- free, all 812 betas "
                ">= 0, all 812 <= 0 -- each fully searched. Both constraints "
                "BIND, measured before launch: the unconstrained fit is roughly "
                "half each sign at every alpha (206/188 at 0.01, 176/218 at 1, "
                "180/214 at 24.5) and 139-217 of 394 live coordinates sit on a "
                "bound, so neither arm is the no-op that sank an earlier design. "
                "The all-negative arm is NOT the mirror of the positive one: the "
                "objective is not symmetric in b and the intercept is free, so "
                "it keeps the naturally negative loadings and scored +0.3614 "
                "in-sample against +0.3364. Alpha is floored at 1e-2 in EVERY "
                "arm because a constrained solve at the 1e-4 default does not "
                "reach optimality in 2000 sweeps, which would truncate the grid "
                "for the treatments only and confound the comparison. Numbered "
                "E26 because a concurrent session took E23, E24 and E25.",
        "runner": "experiments/e26_sign_constraint_crypto.sh",
        "prefix": "e26_", "scout_prefix": "e26scout_",
        "models": ["gen_ridge"], "expected_arms": 3,
    },
    "E27": {
        "title": "FARM forecast combination vs the simple average",
        "idea": "Replicates Lee & Lee (2025), 'Solving the Forecast Combination "
                "Puzzle', on the book. FARM treats the simple average f_bar as a "
                "common factor and regresses its own error on the members' "
                "idiosyncratic deviations d_i = f_i - f_bar, adding back what is "
                "predictable (FARM1); FARM2/FARM3 first strip r <= 2 PCA / "
                "supervised-PCA factors. It is the one combiner family that does "
                "not merely reweight -- it can move a rank relative to the "
                "average -- so it earns a clean paired test even though E10, E23 "
                "and E25 all found fitted combiners lose to 1/N here. Two pools "
                "of the five named forecasters (most-decorrelated glmnet/lgb/mlp "
                "+ the E26 sign-constrained GRRs; and each family's best-CORR run "
                "+ the same GRRs), 12 arms each: equal (the 1/N baseline every arm "
                "is paired against), farm1/2/3 x {ridge, l2boost, postalasso}, and "
                "the paper's GR and DS reference combiners. The prior is that FARM "
                "cuts the paper's squared-error loss but not numerai_corr, because "
                "minimising MSE compresses the cross-sectional spread the ranking "
                "is paid for -- so it is run as a MEASURED null. See "
                "src/prediction/farm.py; highN (the paper's high-dimensional "
                "regime, ~134 members) is an opt-in POOLS extension.",
        "runner": "experiments/e27_farm_combination_crypto.sh",
        "prefix": "e27_", "scout_prefix": None,
        "models": ["ridge_glmnet"], "expected_arms": 24,
    },
    "E17": {
        "title": "MLP hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)",
        "idea": "The 'Overhyped?' framework applied to the sklearn MLP: one "
                "pinned walk-forward per cell of a 5x2x4x4 factorial "
                "(architecture x activation x learning rate x alpha, 240 cells), "
                "then the paper's four readings -- marginal distributions, "
                "functional-ANOVA importance (which HPs are worth tuning), good "
                "and bad corner cases (values that fix performance regardless of "
                "the rest), and an lr/alpha prior. Maps the whole HP surface "
                "once so the grid can be set from evidence, not one pinned draw.",
        "runner": "experiments/e17_hyperparam_sensitivity_crypto.sh",
        # Prefix narrowed e17_ -> e17_a32_ so study_of() splits this study from
        # the cumdiff E37 study, which keeps e17_ (see the E37 note above). All
        # 160 arms are e17_a32_<arch>_<act>_lr*_al*; none of E37's runs match it.
        # scout_prefix dropped (was e17scout_): the sensitivity scout left no
        # metrics, and e17scout_ is not under the narrowed e17_a32_ handle.
        "prefix": "e17_a32_", "scout_prefix": None,
        "models": ["mlp"], "expected_arms": 160,  # 5 archs x 2 acts x 4 lr x 4 alpha (tanh dropped in review)
    },
    "E29": {
        "title": "Per-subset from-scratch tuning + FARM over the full pool",
        "idea": "The widest dataset partitions into 7 feature SUBSETS (raw, tenc, "
                "onehot, rollsel, spread, interact, fenccausal), each shipped as "
                "raw + one block. Tune every family (ridge_glmnet, mlp, lgb) "
                "FROM SCRATCH in a pseudo-OOS manner (rolling walk-forward, "
                "train=refit=130, embargo 20, inner 5-fold purged CV, 50-draw "
                "random search, seed 42) on each subset -- but only the 3 cells a "
                "strict config scan found untuned: ridge_glmnet x rollsel, "
                "mlp x onehot, lgb x onehot (the rest already have a search). Add "
                "3 GRR base arms on crypto/v2.0-onehot: all-betas>=0, all<=0, and "
                "onehot-block-unpenalised (p1e-6, symbol fixed effects). Then run "
                "FARM (equal, farm1/2/3 x {ridge,l2boost,postalasso}, gr, ds) over "
                "the best searched run per (family, subset) plus the 3 GRRs = 24 "
                "base models -- a genuine high-dimensional pool, the regime the "
                "Lee-Lee paper targets, against the E27 finding that 1/N wins.",
        "runner": "experiments/e28_subset_tuning_farm_crypto.sh",
        # Runs carry the e28_ prefix and the runner keeps its e28 filename: this
        # study was launched as "E28", but a concurrent session independently took
        # the E28 id for an unrelated mlp_ktz study, so it is filed here as E29 to
        # break the dict-key collision. The counter filters by model tree
        # (this study's ridge_glmnet/mlp/lgb/gen_ridge vs their mlp_ktz), so the
        # shared e28_ prefix does not cross-count. Renaming the live e28_* fits
        # mid-flight would waste the ~12h mlp/onehot search.
        "prefix": "e28_", "scout_prefix": None,
        "models": ["ridge_glmnet", "mlp", "lgb", "gen_ridge"], "expected_arms": 18,
    },
    "E28": {
        "title": "One MLP tune per book dataset, with the E17 priors (mlp_ktz)",
        "idea": "For each of the 9 datasets the deployed book trains on, ONE "
                "from-scratch hyperparameter search of mlp_ktz -- the E17 "
                "sensitivity priors compiled into a grid: a DENSE learning-rate "
                "sweep (the one knob worth tuning), architecture as a coarse "
                "shallow/deep gate (4-5 layers collapse), alpha near-pinned (the "
                "least important axis, whose CV optimum diverges from OOS). Tests "
                "whether an evidence-primed, lr-focused tune matches or beats the "
                "earlier broad-grid MLP with less budget and no alpha wandering.",
        "runner": "experiments/e28_mlp_ktz_tune_crypto.sh",
        "prefix": "e28_", "scout_prefix": "e28scout_",
        "models": ["mlp_ktz"], "expected_arms": 9,
    },
    "E30": {
        "title": "lgb_ktz grid ENSEMBLE per book dataset (fit all, rank-average)",
        "idea": "For each of the 9 datasets the book trains on, fit EVERY cell of "
                "the lgb_ktz good-corner grid (54 = nl{2,4,8} x lr{.001,.003,.01} "
                "x cs{.05,.10,.20} x ne{2k,4k}, L=lr*n_est<=40) as pinned "
                "walk-forwards and rank-average their OOS predictions instead of "
                "letting the search SELECT one -- the lgb member of the E30(lgb) / "
                "E31(mlp) / E32(catboost) grid-ensemble family, paired against the "
                "E29 search-and-pick per dataset. The 54 cells/dataset "
                "(e30_<ds>_nl*) are INTERMEDIATES in the lgb tree: their forecasts "
                "are reclaimed the moment the ensemble is built, and their cell "
                "metrics json after the report has read per_period for the oracle "
                "-- so the 9 counted arms are the deployable rank-average "
                "ensembles e30_<ds>_ens, not the 495 raw cells. Finding: "
                "indistinguishable from E29 -- no dataset clears |t|>=2; ENS edges "
                "the pick on level on the 5 richer feature sets, loses on the 4 "
                "leaner, always inside noise and always below the oracle best "
                "cell. fenccausal has the steadiest curve (top sharpe, lowest dd).",
        "runner": "experiments/e30_ktz_gridensemble_lgb_reg_crypto.sh",
        "prefix": "e30_", "scout_prefix": None,
        "models": ["lgb"], "expected_arms": 9,
    },
    # Was E30 until 2026-09-05: the id collided with the lgb grid-ensemble above
    # (documented E30 in EXPERIMENTS.md and cross-referenced as "E30 (lgb)" by
    # E31/E32). Renumbered to E36 -- NOT E34/E35, which E33's writeup reserves for
    # the coming mlp tune/ensemble. Prefix narrowed e30_ -> e30_farm_ so study_of()
    # routes e30_farm_* here (longest prefix wins) and every other e30_* to E30.
    "E36": {
        "title": "FARM over the full (model x feature-group) grid",
        "idea": "The (model x feature-group) singles grid is already ~complete: "
                "every one of the 7 semantic raw-groups and the derived blocks "
                "(tenc/spread/interact/fenccausal/rollsel) has a from-scratch "
                "search for all three families (grp_* studies, E14 lgb, E15 mlp, "
                "E29 glmnet-rollsel). What was missing is a FARM combination over "
                "that FULL grid -- E29's FARM used only the 7 TOP-LEVEL blocks, "
                "never the 7 raw-groups. This runs FARM (equal, farm1/2/3 x "
                "{ridge,l2boost,postalasso}, gr, ds) over the best searched run "
                "per (family x 12 non-onehot feature-groups) = 36 base models "
                "plus the 3 E29 GRR arms = 39, reusing the existing tuned runs. "
                "Onehot is excluded (its expensive pairs were scoped out; its "
                "singles live in E29). Tests whether a 39-member high-dimensional "
                "pool -- the paper's regime -- lets any fitted combiner beat 1/N.",
        "runner": "experiments/e30_gridfarm_crypto.sh",
        "prefix": "e30_farm_", "scout_prefix": None,
        "models": ["ridge_glmnet"], "expected_arms": 12,
    },
    "E33": {
        "title": "MLP sensitivity v2 -- width + alpha opened, on 3 datasets",
        "idea": "E17's Kalfa-Timmermann-van der Zwan sweep re-run with the two "
                "axes IA_01 proved matter now OPENED: the first hidden layer is "
                "unfrozen (width 32/64/128/256) and alpha reaches 100 (E17 stopped "
                "at 0.1). IA_01 -- the best MLP slot, an sklearn mlp on fenccausal "
                "-- selects width 64-128 and alpha 8-10, both OUTSIDE E17's grid, "
                "so E17 could not have found it. 12 arch x 2 act x 4 lr x 6 alpha "
                "= 576 cells on each of raw(22)/fenccausal(44)/tenc_onehot(763) = "
                "1728 pinned walk-forwards, analysed PER DATASET (different scales) "
                "for new priors that feed the E34 tune and E35 ensemble.",
        "runner": "experiments/e33_hyperparam_sensitivity_v2_crypto.sh",
        "prefix": "e33_", "scout_prefix": "e33scout_",
        # Findings written; the 1,728 cells were always intermediates for the
        # E34 priors and the E35 ensemble, never an ensembling pool of their
        # own (E46 says so explicitly). per_period + metrics + hyperparams keep
        # the study fully reportable at 79 MB.
        "forecasts": "reclaim",
        "models": ["mlp"], "expected_arms": 1728,
    },
    "E31": {
        "title": "mlp_ktz grid ENSEMBLE per book dataset (fit all, rank-average)",
        "idea": "For each of the 9 datasets the book trains on, fit EVERY cell "
                "of the mlp_ktz grid (120 = the E17 priors: 20 lr x 3 arch x 2 "
                "alpha x identity) and rank-average their OOS predictions instead "
                "of letting inner CV SELECT one. E28 showed a single searched "
                "mlp_ktz loses to the book, whose strength is blending; this "
                "asks whether ensembling a model's own hyperparameter grid -- the "
                "same rank-average operator the deployed CURUPIRA_SACI/CU_SA "
                "blends use -- recovers the gap. Paired against the E28 pick per "
                "dataset (ensemble vs select, same data/cadence/seed). Runs at "
                "--n-iter 1 --n-splits 2: selection is removed, not searched.",
        "runner": "experiments/e31_mlp_ktz_ensemble_crypto.sh",
        "prefix": "e31_", "scout_prefix": "e31scout_",
        "models": ["mlp_ktz_ens"], "expected_arms": 9,
    },
    "E32": {
        "title": "catboost_ktz grid ENSEMBLE vs SEARCH per book dataset",
        "idea": "The CatBoost member of the E30 (lgb) / E31 (mlp) family: "
                "ensemble the ktz grid instead of searching it. Two arms per "
                "dataset over ONE 12-cell CatBoost KTZ grid (depth{2,4} x "
                "lr{.005,.01,.02} x rsm{.1,.2}; iterations 2000, l2_leaf_reg 10, "
                "min_data_in_leaf 100 pinned -- the E27 booster priors on "
                "CatBoost's names, compact so the ensemble can fit ALL of it): "
                "catboost_ktz_ens fits all 12 cells and rank-averages them (one "
                "deployable CatBoostGridEnsemble, --n-splits 2 so the unused inner "
                "CV costs one fold), against catboost_ktz which SEARCHES the same "
                "12 and keeps the CV-best (--n-iter 12, exhaustive). Both wrappers "
                "read the grid from CatBoostGridEnsemble, so ensemble-of-12 vs "
                "pick-1-of-12 is exact. Headline is the paired ENS - SEL delta "
                "(NW lag 20); each ENS arm is also ranked against its deployed "
                "slot. Prior: averaging buys RISK (sharpe, drawdown) over LEVEL.",
        "runner": "experiments/e32_catboost_ktz_ensemble_crypto.sh",
        "prefix": "e32_", "scout_prefix": "e32scout_",
        # Two model trees, both unique to this study, so the derived arm counter
        # cannot cross-count with any other study regardless of id collisions.
        "models": ["catboost_ktz_ens", "catboost_ktz"], "expected_arms": 18,
    },
    "E34": {
        "title": "One MLP tune per book dataset with the E33 priors (mlp_ktz2)",
        "idea": "E28 redone on the E33 re-sweep priors instead of E17's. The one "
                "change that matters: alpha {1,10} (~100x E17's 0.1 ceiling, the "
                "bug that kept IA_01's region out of the box), activation un-pinned "
                "to {identity,relu} (relu holds the peak), lr low peak 1e-3, width "
                "32-128 (weak axis), depth 1-2. From-scratch search of mlp_ktz2 on "
                "each of the 9 book datasets. Paired vs the E28 (old-prior) pick "
                "and against the book -- does the corrected box reach IA_01?",
        "runner": "experiments/e34_mlp_ktz2_tune_crypto.sh",
        "prefix": "e34_", "scout_prefix": "e34scout_",
        "models": ["mlp_ktz2"], "expected_arms": 9,
    },
    "E35": {
        "title": "Ensemble the mlp_ktz2 (E33-reprior) grid per book dataset",
        "idea": "E31 redone on the E33 priors: fit all 120 cells of the mlp_ktz2 "
                "grid (alpha {1,10}, {identity,relu}, lr 1e-4..1e-2, width 32-128, "
                "depth 1-2) and rank-average, vs the E34 search-and-pick over the "
                "same grid. The ensemble-vs-pick question (E28/E31) re-asked on the "
                "corrected box that now contains IA_01's region. Runs at --n-iter 1 "
                "--n-splits 2; paired vs E34 per dataset and against the book.",
        "runner": "experiments/e35_mlp_ktz2_ensemble_crypto.sh",
        "prefix": "e35_", "scout_prefix": "e35scout_",
        "models": ["mlp_ktz2_ens"], "expected_arms": 9,
    },
    "E38": {
        "title": "TimesFM zero-shot univariate forecast of each symbol's target",
        "idea": "First TIME-SERIES foundation model on the board, and the first "
                "univariate/per-symbol one. The ORIGINAL TimesFM 1.0 (200M, Das "
                "et al. 2023), loaded from the google/timesfm-1.0-200m-pytorch "
                "checkpoint via the v1.2.6 torch backend (no JAX), zero-shot: no "
                "training at all. One symbol at a time, it forecasts that coin's "
                "own target_binned_return_20 history and the forecast is the "
                "signal. Leak wall at the 20-period embargo -- context is only "
                "the symbol's targets resolved by t (dates <= t-20 on the global "
                "grid); forecast horizon 20, signal = mean of the horizon. A "
                "foundation-model cousin of the E13 AR study (mostly null), so a "
                "WEAK signal is the expected outcome. Bespoke inference (TimesFM "
                "is a sequence forecaster, outside the row-wise MODELS pipeline): "
                "src/analysis/timesfm_zeroshot.py on the GPU, scored by "
                "timesfm_score.py. Scored on tabpfn/e19_fenccausal's 510 dates.",
        "runner": "src/analysis/timesfm_zeroshot.py",
        "prefix": "e38_", "scout_prefix": None,
        "models": ["timesfm"], "expected_arms": 1,
    },
    "E39": {
        "title": "The tabular reframing of E38: TabPFN zero-shot on ALL features",
        "idea": "E38's direct counterpart. Same prediction problem, same window, "
                "but reframed from a univariate time series to a TABULAR one and "
                "given to TabPFN instead of TimesFM, zero-shot: each row is its "
                "full feature vector, and the foundation model predicts the "
                "target in-context from past resolved rows. Uses ALL the "
                "features -- allfeat_pp, the 812-column union of every engineered "
                "block (raw+tenc+onehot+rollsel+spread+interact) -- at context "
                "2048 (better than E19's memory-constrained 1024). Isolates the "
                "FRAMING: TimesFM on the target's own history (E38) got CORR "
                "0.012, statistically zero; this asks what the SAME kind of "
                "zero-shot foundation model extracts once the price history is "
                "presented as tabular features instead. e19_allfeat_pp already "
                "showed ~0.124 at context 1024; this is the deliberate, "
                "better-contexted arm framed against E38.",
        "runner": "run_forecast (tabpfn on crypto/v2.0-tenc-onehot-rollsel-spread-interact)",
        "prefix": "e39_", "scout_prefix": None,
        "models": ["tabpfn"], "expected_arms": 1,
    },
    "E40": {
        "title": "TEMPO zero-shot univariate forecast of each symbol's target",
        "idea": "The controlled model-swap sibling of E38: same per-symbol, "
                "univariate, leak-free zero-shot design, TEMPO in place of "
                "TimesFM, so E38-vs-E40 isolates the MODEL. TEMPO (Cao et al., "
                "ICLR 2024, arXiv:2310.04948) is a GPT-2-backbone (~80M) "
                "generative time-series foundation model that STL-decomposes each "
                "series into trend/seasonal/residual, RevIN-normalises, "
                "patch-tokenises and reads a soft prompt pool -- all inside "
                "predict(). Pretrained zero-shot weights Melady/TEMPO "
                "TEMPO-80M_v1.pth (HuggingFace), loaded via the v-latest pip "
                "package `timeagi` in an isolated /scratch venv reusing base "
                "torch 2.7 (--no-deps to dodge its torch==1.13 pin). One symbol "
                "at a time, context = the coin's own target_binned_return_20 up "
                "to the 20-period leak wall (dates <= t-20), seq_len 336, "
                "forecast horizon 20, signal = mean of the horizon. Bespoke "
                "inference (src/analysis/tempo_zeroshot.py), scored by "
                "timesfm_score.py --model tempo on tabpfn/e19_fenccausal's 510 "
                "dates. Like E38, a WEAK signal is the expected outcome -- the "
                "edge here is cross-sectional (E39), not autoregressive.",
        "runner": "src/analysis/tempo_zeroshot.py",
        "prefix": "e40_", "scout_prefix": None,
        "models": ["tempo"], "expected_arms": 1,
    },
    "E41": {
        "title": "Fine-tuned TEMPO -- does adapting the weights lift E40's null?",
        "idea": "E40 (TEMPO zero-shot) scored CORR -0.017, null. E41 fine-tunes "
                "TEMPO on the crypto panel and re-evaluates on the SAME 510 scored "
                "dates, so E40-vs-E41 isolates fine-tuning. Parameter-efficient "
                "(GPT-2 frozen; only LoRA + heads + RevIN affine train), warm-started "
                "from Melady/TEMPO and fine-tuned in-memory (never reconstructed -- "
                "the .pth loads strict=False). Training windows are (336->20) slices "
                "over each symbol's resolved target series with horizon-end <= the "
                "leak cutoff 2024-06-27 (min scored date - 20 embargo); loss is MSE "
                "on the first 20 output steps. A 2x2 matrix: scheme {single global, "
                "walk-forward refit-every 130} x selection {validation MSE, validation "
                "cross-sectional CORR}. Eval half reuses E40's build_tasks/"
                "forecast_signal verbatim; scored by timesfm_score.py --model tempo. "
                "Prior: fine-tuning cannot reach the cross-sectional edge (E39), so a "
                "null across arms is the likely, still-informative outcome.",
        "runner": "src/analysis/tempo_finetune.py",
        "prefix": "e41_", "scout_prefix": None,
        "models": ["tempo"], "expected_arms": 4,
    },
    "E42": {
        "title": "torch MLP training-loss study, config PINNED (learning dynamics)",
        "idea": "Does a different training OBJECTIVE change what the torch MLP "
                "learns and how stably? 6 losses -- mse(control), corr, bmc vs the "
                "sparse real meta model (benchmark) and vs the full-coverage "
                "meta_proxy, sharpe, calmar -- on raw + fenccausal, ALL at one "
                "pinned config so only the loss varies. Evaluated on the board AND "
                "on learning dynamics (per-epoch curve, epochs-to-converge, "
                "pred_std collapse, train/val gap) via loss_dynamics.py. bmc-bench "
                "vs bmc-proxy IS the 'optimise on MMC-only vs estimate with BMC' "
                "contrast (on crypto the benchmark==meta model, so the proxy is "
                "the only full-coverage reference).",
        "runner": "experiments/e42_torch_mlp_loss_dynamics_crypto.sh",
        "prefix": "e42_", "scout_prefix": "e42scout_",
        "models": ["torch_mlp_loss"], "expected_arms": 12,
    },
    "E43": {
        "title": "torch MLP training-loss study, config SEARCHED (best payout)",
        "idea": "E42's companion: the same 6 loss arms x raw/fenccausal, but each "
                "loss runs a full search over the torch_mlp grid and finds its own "
                "config -- the best-achievable-per-loss read where E42 holds the "
                "config fixed to isolate the loss's effect on learning. Paired NW "
                "vs the mse control, ranked against the book.",
        "runner": "experiments/e43_torch_mlp_loss_search_crypto.sh",
        "prefix": "e43_", "scout_prefix": "e43scout_",
        "models": ["torch_mlp_loss"], "expected_arms": 12,
    },
    "E44": {
        "title": "Fine-tuned TabPFN on fenccausal -- can adapting the weights beat E19?",
        "idea": "The tabular counterpart of E41, on the framing that WORKS. E19 ran "
                "TabPFN zero-shot (in-context, frozen weights) on crypto/"
                "v2.0-fenccausal (44 feat) and scored CORR 0.145 -- the strongest "
                "foundation-model arm on the board. E44 FINE-TUNES the TabPFN v2 "
                "transformer (tabpfn 8.5.0 finetuning API: get_preprocessed_datasets "
                "+ fit_from_preprocessed) on the 345,496 fenccausal rows on/before "
                "the leak cutoff 2024-06-27 (min scored date - 20 embargo), saves the "
                "adapted checkpoint, then re-runs the EXACT E19 walk-forward (rolling "
                "130/130, embargo 20, seed 42, n_estimators 4, ctx 4096) with "
                "model_path pointed at the fine-tuned .ckpt. So E19-vs-E44 isolates "
                "fine-tuning, holding data/cadence/context fixed. Unlike E41 "
                "(univariate, where fine-tuning could not reach the cross-sectional "
                "edge), here the framing already works, so a genuine lift over 0.145 "
                "is plausible. Scored on tabpfn/e19_fenccausal's 510 dates; paired NW "
                "vs E19.",
        "runner": "src/analysis/tabpfn_finetune.py",
        "prefix": "e44_", "scout_prefix": None,
        "models": ["tabpfn"], "expected_arms": 1,
    },
    "E45": {
        "title": "Fine-tune TabPFN with a METRIC-ALIGNED objective (CORR, MMC)",
        "idea": "E44 fine-tuned TabPFN with the library CRPS+MSE loss and it HURT "
                "(CORR 0.145 -> 0.118, t -3.02) -- part of the damage was objective "
                "mismatch: MSE compresses the cross-sectional spread the ranking "
                "pays for. E45 tests whether fine-tuning toward the PAYABLE metric "
                "instead recovers or beats E19. tabpfn 8.5.0's public API only ships "
                "CRPS/MSE, so this uses a CUSTOM differentiable fine-tune loop "
                "(fit_from_preprocessed context -> differentiable bar-distribution "
                "mean on the query -> loss) with two arms: (corr) minimise the "
                "negative per-date cross-sectional correlation of the query preds "
                "vs target; (mmc) minimise negative MMC = contribution over the "
                "meta model, computed ONLY on query rows where the meta model is "
                "available (~33% coverage), meta merged from historical_meta_models "
                "on (ucid,date). Same leak-safe pre-2024-06-27 pool as E44, same "
                "save->model_path round-trip, same E19 walk-forward eval. Paired NW "
                "vs E19 (zero-shot) and vs E44 (MSE fine-tune). If a correct "
                "objective still loses, prior-corruption (not just loss) is the "
                "cause; if it wins, the E44 loss was the problem.",
        "runner": "src/analysis/tabpfn_finetune_metric.py",
        "prefix": "e45_", "scout_prefix": None,
        "models": ["tabpfn"], "expected_arms": 2,
    },
    "E46": {
        "title": "FARM over the from-scratch ktz SEARCH-PICK models "
                 "(lgb E29 + catboost E32-sel + mlp E34), per-family and pooled, vs 1/N",
        "idea": "Combine the 27 best-tuned (model x dataset) forecasters -- the "
                "config each family's from-scratch KTZ-prior search actually PICKED "
                "per dataset -- with the Lee-Lee FARM combiner, and ask whether a "
                "fitted factor-adjusted combination beats the plain 1/N rank-average "
                "of the same 27. (An earlier plan FARMed the raw HP grid CELLS, but "
                "E30/E32 reclaim those forecasts and the surviving mlp E33 sweep is "
                "junk-heavy; the search-PICK models are the clean pool -- one tuned "
                "model per (family,dataset), all already on disk.) POOL: lgb "
                "lgb_ktz/e29_<ds> (60-draw search), catboost catboost_ktz/e32_sel_<ds> "
                "(exhaustive 12-cell search), mlp mlp_ktz2/e34_<ds> (80-draw search on "
                "the E33-reprior WIDER grid), over the 9 book datasets = 27 base "
                "forecasters. FARM -- equal (the 1/N baseline every arm is measured "
                "against), farm1/2/3 x {ridge,l2boost,postalasso}, gr, ds -- over four "
                "pools: per-family (lgb 9, cat 9, mlp 9) and the pooled union (all "
                "27, the cross-model/cross-dataset regime the book blends over). "
                "Paired NW lag 20 vs each pool's own 1/N; each pool's best arm also "
                "ranked against the deployed book. Prior from E27/E29/E36: 1/N wins "
                "on this panel, so a fitted combiner clearing |t|>=2 would be the "
                "surprise. Raw scores, ratios not NMR.",
        "runner": "experiments/e46_ktz_searchpick_farm_crypto.sh",
        "prefix": "e46_", "scout_prefix": None,
        "models": ["ktz_farm"], "expected_arms": 48,
    },
    "E47": {
        "title": "Deploy the E33 winner: a live pinned MLP (IA_02) + two blends",
        "idea": "Not a sweep -- a DEPLOYMENT. E33's top fenccausal cell (128,64 / "
                "relu / lr 1e-3 / ALPHA 10, the regularisation E17's 0.1 ceiling "
                "could not reach) is wired live as the pinned MLP component IA_02 "
                "(src/live/registry.py, fit_override.fixed_params, NOT_SUBMITTED), "
                "and two slot-blends are built on it and priced as REAL per-period "
                "rank-averages of their members' finished forecasts (the deployable "
                "number, not the screened payout-series proxy): CU_SA_IA_BO_02 = "
                "IA_02 + CURUPIRA_05_60 + CU_SA_IA_BO_01 (a version bump of "
                "CU_SA_IA_BO_01 -- same Curupira+Saci+Iara footprint, a 2nd neural "
                "leg), and CU_CA_IA_01 = IA_02 + IAUARETE_01 + CURUPIRA_05_60 "
                "(Curupira + Caipora/CatBoost + Iara). Priced over 470 periods vs "
                "each blend's best member: CU_SA_IA_BO_02 matches CU_SA_IA_BO_01 on "
                "payout (0.10635 vs 0.10712) but improves every risk metric "
                "(c.sharpe 2.03 vs 1.92, dd(pay) 1.02 vs 1.16, burn 30 vs 38); "
                "CU_CA_IA_01 OUT-LEVELS all three members (0.10422 vs ~0.095) at "
                "c.sharpe 2.05 -- its mlp/catboost/ridge legs are genuinely diverse. "
                "The two e47_ runs are the slots' backtest_run; the account slots "
                "cu_sa_ia_bo_02 and cu_ca_ia_01 exist, so the next open round "
                "submits them.",
        "runner": "experiments/e47_deploy_mlp_blends_crypto.sh",
        "prefix": "e47_", "scout_prefix": None,
        "models": ["lgb", "ridge_glmnet"], "expected_arms": 2,
    },
    "E48": {
        "title": "Random-forest hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)",
        "idea": "The 'Overhyped?' framework E17 ran on the MLP and E27/E29 on "
                "LightGBM, now on a BAGGED FOREST -- a model class the book has "
                "never tried. One pinned walk-forward per cell of a 3x4x4x4 "
                "factorial (n_estimators {100,300,1000} x max_depth {3,6,12,None} "
                "x max_features {0.1,0.3,0.6,1.0} x min_samples_leaf "
                "{1,10,100,1000} = 192 cells) on the raw 22 of crypto/v2.0, then "
                "the paper's four readings -- marginal distributions, "
                "functional-ANOVA importance, good/bad corners, priors. A forest "
                "is bagging not boosting, so there is NO learning-rate axis: "
                "max_features carries the Numerai colsample idiom (the most "
                "distinctive setting) and min_samples_leaf the heavy-leaf idiom, "
                "while n_estimators + max_depth stand in for the boosters' lr + "
                "num_leaves. The two-way claim tested is max_features x "
                "min_samples_leaf (do the two idioms interact?), the forest "
                "analogue of the paper's lr x alpha. bootstrap and max_samples "
                "are pinned at the forest default (row subsampling is the "
                "least-evidenced idiom); n_splits 2 since the pinned cells only "
                "record an unused cv_score. Maps the forest surface once so a "
                "future rf_ktz grid can be set from evidence, not sklearn "
                "defaults. Read with src.analysis.hyperparam_sensitivity_rf.",
        "runner": "experiments/e48_rf_hyperparam_sensitivity_crypto.sh",
        "prefix": "e48_", "scout_prefix": None,
        "models": ["rf"], "expected_arms": 192,
    },
    "E49": {
        "title": "LightGBM hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)",
        "idea": "The 'Overhyped?' framework on the BOOSTER -- the lgb sibling of "
                "E17 (mlp) and E48 (rf). A 4-axis num_leaves x learning_rate x "
                "n_estimators x colsample factorial, 315 cells plus the "
                "ktz_bridge_canonical arm the runner writes last, then the "
                "paper's readings via src.analysis.hyper_sensitivity. "
                "REGISTERED RETROSPECTIVELY: the sweep ran under the id E27, "
                "which a concurrent session had already given to the FARM "
                "combination study, so its 316 arms belonged to no entry here "
                "and were invisible to this index -- the largest un-indexed "
                "artifact set in the repo, and one a disk cleanup would have "
                "read as orphaned. Its artifacts keep the ktz_ handle rather "
                "than an e49_ one: the handle predates the numbering, "
                "hyper_sensitivity is invoked as --prefix ktz_, and "
                "results/reports/crypto/ktz_top10_vs_stacked.html is already "
                "published under it. Findings were never written; until they "
                "are, this study's forecasts are not reclaimable.",
        "runner": "experiments/e27_hyperparam_sensitivity_crypto.sh",
        "prefix": "ktz_", "scout_prefix": None,
        "models": ["lgb"], "expected_arms": 316,
    },
    "E50": {
        "title": "One LightGBM tune per book dataset, with the E49 priors (lgb_ktz)",
        "idea": "The lgb twin of E28: one from-scratch tune per submitted-model "
                "dataset, with the E49 sensitivity priors compiled into the "
                "search grid. REGISTERED RETROSPECTIVELY -- the runner declares "
                "EXPERIMENT=E29, but STUDIES' E29 is the subset-tuning/FARM "
                "study whose runner is a different file, so these 9 arms "
                "resolved to no entry. That mattered more than the count "
                "suggests: e29_tenc_onehot is SACI_06's backtest_run in "
                "src/live/registry.py, so an un-indexed run was backing a "
                "deployed slot. Prefix e29_ in the lgb_ktz tree, which is what "
                "separates it from E29's e28_* runs elsewhere.",
        "runner": "experiments/e29_lgb_ktz_tune_crypto.sh",
        "prefix": "e29_", "scout_prefix": None,
        "models": ["lgb_ktz"], "expected_arms": 9,
    },
}

def study_of(run_name, model=None):
    """
    Which numbered study wrote this run, from its prefix and model tree.

    The prefix IS the study handle -- experiments/EXPERIMENTS.md's naming rule
    exists so that `e13_*` retrieves one study years later, and this is the
    lookup that rule buys. Longest prefix first, so `e13scout_` is never read as
    `e13_`.

    A prefix alone is not always enough. E28 and E29 both write `e28_*`: E29
    launched as "E28" and a concurrent session had taken the id. Their ARTIFACTS
    never collided, because scan_arms separates studies by model tree as well --
    E28 owns mlp_ktz, E29 owns ridge_glmnet/mlp/lgb/gen_ridge. This function was
    the one place that looked at the prefix alone, so it answered E29 for both
    and test_every_declared_study_resolves_its_own_prefix failed for E28.
    Passing `model` applies the rule scan_arms already used. Renaming one
    study's runs would have been the alternative and is not available: E29's
    sibling e29_tenc_onehot is SACI_06's backtest_run in src/live/registry.py,
    and a rename breaks a live pointer.

    :param run_name: e.g. "e14_adl", "w6r6_fixed_reg"
    :type run_name: str
    :param model: the model tree the run lives in, e.g. "mlp_ktz". Optional, but
        REQUIRED to disambiguate a shared prefix -- with model=None a shared
        prefix resolves to whichever study declares it first, which is a coin
        flip. Callers that have the tree should pass it.
    :type model: str or None
    :return: ("E14", False) for a full arm, ("E14", True) for a scout, or
             (None, False) for a run that belongs to no numbered study --
             every deployed slot's backtest, and everything predating E01
    :rtype: (str or None, bool)
    """
    candidates = []
    for study_id, spec in STUDIES.items():
        if model is not None and model not in spec["models"]:
            continue
        candidates.append((spec["prefix"], study_id, False))
        if spec.get("scout_prefix"):
            candidates.append((spec["scout_prefix"], study_id, True))
    for prefix, study_id, is_scout in sorted(candidates, key=lambda c: -len(c[0])):
        if run_name.startswith(prefix):
            return study_id, is_scout
    return None, False


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    idea          TEXT NOT NULL,
    runner        TEXT NOT NULL,
    prefix        TEXT NOT NULL,
    models        TEXT NOT NULL,
    expected_arms INTEGER
);
CREATE TABLE IF NOT EXISTS arms (
    experiment_id TEXT NOT NULL,
    run_name      TEXT NOT NULL,
    model         TEXT NOT NULL,
    kind          TEXT NOT NULL,      -- 'full' or 'scout'
    target        TEXT,
    n_features    INTEGER,
    n_periods     INTEGER,
    corr          REAL,
    mmc           REAL,
    payout        REAL,
    corr_sharpe   REAL,
    corr_max_dd   REAL,
    runtime_min   REAL,
    finished_at   TEXT,
    PRIMARY KEY (experiment_id, model, run_name)
);
CREATE TABLE IF NOT EXISTS refreshed (
    at            TEXT NOT NULL,
    running       TEXT               -- ids with a live runner process, comma separated
);
"""


def running_studies():
    """
    Which studies have a live runner process, read from /proc.

    Derived rather than recorded: a status field saying "running" outlives the
    process that justified it, and the first time that happens the whole table
    stops being trustworthy. pgrep is not used so this has no shell dependency.

    :return: study ids whose runner is currently executing
    :rtype: set of str
    """
    live = set()
    # Our own process tree is skipped. Without this the detector finds itself:
    # the shell that runs this module can carry the text of this file -- and so
    # every runner name in STUDIES -- in its own command line, and every study
    # reports as running. That is not hypothetical, it is what the first version
    # did.
    mine = {os.getpid()}
    parent = os.getppid()
    while parent > 1:
        mine.add(parent)
        try:
            with open("/proc/%d/stat" % parent) as handle:
                parent = int(handle.read().split(") ", 1)[1].split()[1])
        except (OSError, IOError, IndexError, ValueError):
            break

    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in mine:
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as handle:
                raw = handle.read()
        except (OSError, IOError):
            continue                       # the process exited between listdir and open
        # cmdline is NUL-separated, so match whole ARGUMENTS rather than doing a
        # substring test on the joined string. `bash experiments/e09_*.sh` has the
        # path as its own argument; a shell whose single -c argument merely quotes
        # the name does not, and is correctly ignored.
        arguments = raw.decode("utf-8", "replace").split("\x00")
        for study_id, spec in STUDIES.items():
            name = os.path.basename(spec["runner"])
            if any(argument.endswith(name) for argument in arguments):
                live.add(study_id)
    return live


def scan_arms(study_id, spec):
    """
    Every artifact this study has written, read off results/metrics/.

    :param study_id: e.g. "E09"
    :type study_id: str
    :param spec: the STUDIES entry
    :type spec: dict
    :return: rows ready for the arms table
    :rtype: list of tuple
    """
    rows = []
    prefixes = [(spec["prefix"], "full")]
    if spec.get("scout_prefix"):
        prefixes.append((spec["scout_prefix"], "scout"))

    for model in spec["models"]:
        directory = os.path.join(METRICS_ROOT, model)
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".json") or filename.endswith(".seeds.json"):
                continue
            run = filename[:-5]
            # Longest prefix first: e09scout_ must not be read as e09_.
            kind = None
            for prefix, label in sorted(prefixes, key=lambda p: -len(p[0])):
                if run.startswith(prefix):
                    kind = label
                    break
            if kind is None:
                continue
            path = os.path.join(directory, filename)
            try:
                data = json.load(open(path))
            except (ValueError, IOError):
                continue
            config = data.get("config", {})
            corr = data.get("corr", {}).get("mean")
            mmc = data.get("mmc", {}).get("mean")
            payout = None
            if corr is not None and mmc is not None:
                # Rebuilt from the two means only for the INDEX. A real report
                # forms the payout series per period and summarises once -- see
                # CLAUDE.md. This column is for sorting a list, not for a claim.
                payout = 0.1 * corr + mmc
            rows.append((
                study_id, run, model, kind,
                config.get("target"), config.get("n_features"),
                data.get("n_periods"), corr, mmc, payout,
                data.get("corr", {}).get("sharpe"),
                data.get("corr", {}).get("max_drawdown"),
                data.get("runtime_minutes"),
                time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))),
            ))
    return rows


def refresh(db_path=DB_PATH):
    """
    Rebuild the whole database from STUDIES plus the filesystem.

    Destructive by design: the arms table is emptied first, so a run deleted from
    results/ disappears from here too. An index that keeps rows for artifacts
    that no longer exist is the same failure as a stale status field.

    :param db_path: where to write
    :type db_path: str
    :return: (n_experiments, n_arms)
    :rtype: (int, int)
    """
    directory = os.path.dirname(db_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(SCHEMA)
        connection.execute("DELETE FROM experiments")
        connection.execute("DELETE FROM arms")
        connection.execute("DELETE FROM refreshed")

        arm_rows = []
        for study_id, spec in STUDIES.items():
            connection.execute(
                "INSERT INTO experiments VALUES (?,?,?,?,?,?,?)",
                (study_id, spec["title"], spec["idea"], spec["runner"],
                 spec["prefix"], ",".join(spec["models"]), spec["expected_arms"]),
            )
            arm_rows.extend(scan_arms(study_id, spec))

        connection.executemany(
            "INSERT OR REPLACE INTO arms VALUES (%s)" % ",".join("?" * 14), arm_rows)
        connection.execute(
            "INSERT INTO refreshed VALUES (?,?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), ",".join(sorted(running_studies()))))
        connection.commit()
        return len(STUDIES), len(arm_rows)
    finally:
        connection.close()


def state_of(study_id, done, expected, live):
    """
    One word for where a study stands, computed from what is on disk.

    :return: "running", "done", "partial", "scouted" or "not started"
    :rtype: str
    """
    if study_id in live:
        return "running"
    if done == 0:
        return "not started"
    if expected and done >= expected:
        return "done"
    return "partial"


def print_summary(db_path=DB_PATH):
    """Print one line per study, newest id last."""
    connection = sqlite3.connect(db_path)
    try:
        live = set(
            filter(None, (connection.execute("SELECT running FROM refreshed")
                          .fetchone() or [""])[0].split(",")))
        at = connection.execute("SELECT at FROM refreshed").fetchone()[0]
        print("experiment index -- refreshed %s -- %s" % (at, db_path))
        print()
        print("%-5s %-11s %7s %7s  %-52s" % (
            "id", "state", "arms", "scout", "title"))
        print("-" * 92)
        for study_id, title, expected in connection.execute(
                "SELECT id, title, expected_arms FROM experiments ORDER BY id"):
            done = connection.execute(
                "SELECT COUNT(*) FROM arms WHERE experiment_id=? AND kind='full'",
                (study_id,)).fetchone()[0]
            scouted = connection.execute(
                "SELECT COUNT(*) FROM arms WHERE experiment_id=? AND kind='scout'",
                (study_id,)).fetchone()[0]
            print("%-5s %-11s %7s %7s  %-52s" % (
                study_id, state_of(study_id, done, expected, live),
                "%d/%s" % (done, expected if expected else "?"),
                scouted or "-", title[:52]))
        print("-" * 92)
        if live:
            print("running now: %s" % ", ".join(sorted(live)))
        print()
        print("ideas:")
        for study_id, idea in connection.execute(
                "SELECT id, idea FROM experiments ORDER BY id"):
            print("  %-5s %s" % (study_id, idea))
    finally:
        connection.close()


def print_arms(study_id, db_path=DB_PATH):
    """Print every arm of one study, best payout first."""
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT run_name, model, kind, n_features, n_periods, corr, mmc, "
            "payout, corr_sharpe, corr_max_dd, runtime_min, finished_at "
            "FROM arms WHERE experiment_id=? "
            "ORDER BY kind DESC, payout DESC", (study_id.upper(),)).fetchall()
        if not rows:
            print("no arms recorded for %s" % study_id.upper())
            return
        print("%-30s %-10s %-6s %5s %6s %9s %9s %9s %7s %7s %7s  %s" % (
            "run", "model", "kind", "feat", "n", "CORR", "MMC", "payout",
            "sharpe", "maxdd", "min", "finished"))
        print("-" * 140)
        for row in rows:
            values = ["" if v is None else v for v in row]
            print("%-30s %-10s %-6s %5s %6s %9s %9s %9s %7s %7s %7s  %s" % (
                values[0], values[1], values[2], values[3], values[4],
                "%.6f" % row[5] if row[5] is not None else "-",
                "%.6f" % row[6] if row[6] is not None else "-",
                "%.6f" % row[7] if row[7] is not None else "-",
                "%.3f" % row[8] if row[8] is not None else "-",
                "%.3f" % row[9] if row[9] is not None else "-",
                "%.1f" % row[10] if row[10] is not None else "-",
                values[11]))
        print("-" * 140)
        print("payout here is 0.1*corr + mmc rebuilt from the two MEANS, which is")
        print("fine for ranking a list and wrong for a claim -- a report forms the")
        print("series per period and summarises once. See CLAUDE.md.")
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Refresh and query the experiment index.")
    parser.add_argument("--db", default=DB_PATH, help="database path")
    parser.add_argument("--arms", metavar="ID",
                        help="print every arm of one study, e.g. E09")
    parser.add_argument("--sql", help="run an arbitrary read-only query")
    parser.add_argument("--no-refresh", action="store_true",
                        help="query without rescanning; only for a machine where "
                             "results/ is unavailable, since every status field "
                             "then describes the last refresh, not now")
    args = parser.parse_args(argv)

    if not args.no_refresh:
        studies, arms = refresh(args.db)
        if args.sql or args.arms:
            print("refreshed: %d studies, %d arms\n" % (studies, arms))

    if args.sql:
        connection = sqlite3.connect(args.db)
        try:
            cursor = connection.execute(args.sql)
            names = [d[0] for d in cursor.description or []]
            if names:
                print(" | ".join(names))
            for row in cursor.fetchall():
                print(" | ".join("" if v is None else str(v) for v in row))
        finally:
            connection.close()
    elif args.arms:
        print_arms(args.arms, args.db)
    else:
        print_summary(args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
