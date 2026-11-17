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
import math
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
    "E181": {
        "title": "Sklearn-initialised Torch MLP hidden states + fixed-effect glmnet ridge",
        "idea": "Seven architectures and seven state-processing/readout variants on allfeat_pp minus tenc plus x29; exact unpenalised asset dummies, fresh nested fits, three seeds, development only.",
        "runner": "experiments/e181_mlp_hidden_ridge.py",
        "prefix": "e181_", "scout_prefix": None,
        "models": ["ridge_glmnet"], "expected_arms": 49,
    },
    "E173": {
        "title": "Matched spot volume surprise, breadth and liquidity ablation",
        "idea": "Fixed raw22 baseline, A0-A5, nine/four venues common coverage and separate Yahoo liquidity; chronological paired Numerai CORR and Spearman.",
        "runner": "experiments/e173_spot_volume_comparison.py",
        "prefix": "e173_", "scout_prefix": None,
        "models": ["ridge", "lgb_ktz", "mlp_ktz"], "expected_arms": 15,
    },
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
    "E51": {
        "title": "RF tune from scratch with the E48 priors (rf_ktz)",
        "idea": "The rf sibling of E28 (mlp_ktz) / E50 (lgb_ktz): a from-scratch "
                "RandomizedSearchCV of rf_ktz -- the E48 sensitivity priors "
                "compiled into a search grid -- on the raw 22 (crypto/v2.0) and "
                "on fenccausal (44), the feature set the book's best neural slot "
                "uses. The grid encodes E48's four findings: n_estimators PINNED "
                "at 100 (fANOVA red herring, share 0.0002), max_depth a shallow "
                "gate {3,4,6} (shallow won monotonically), max_features a "
                "dense-low sweep {0.05..0.30} (the colsample idiom; 0.6/1.0 "
                "dropped, 0/48 good-cell support at 1.0), and min_samples_leaf "
                "heavy {300,1000,2000,4000} with the ceiling OPENED past E48's "
                "binding max of 1000 (its marginal was monotone to the grid edge "
                "-- the E07/E17 alpha-ceiling trap, so open it). 60-cell grid, "
                "every cell in E48's good region; searched at --n-iter 40, the "
                "CURUPIRA cadence (rolling 130/130, embargo 20, n_splits 5, seed "
                "42). Produces a deployable RF; paired vs the book is the next "
                "step. Best E48 cell reached CORR 0.1561 ~ the top MLP slot.",
        "runner": "experiments/e51_rf_ktz_tune_crypto.sh",
        "prefix": "e51_", "scout_prefix": "e51scout_",
        "models": ["rf_ktz"], "expected_arms": 2,
    },
    "E52": {
        "title": "One rf_ktz tune per book dataset, with the E48 priors",
        "idea": "The rf sibling of E28 (mlp_ktz) and E50 (lgb_ktz), and the "
                "multi-dataset scale-up of E51 (raw+fenc only): a from-scratch "
                "RandomizedSearchCV of rf_ktz -- the E48 sensitivity priors as a "
                "60-cell grid -- on ALL NINE datasets the deployed book trains on "
                "(raw22, tenc23, interact36, fenccausal44, cumlag3-88, "
                "tenc_onehot763, onehot_rollsel786, allfeat788, allfeatpp812), "
                "so every book slot gets an RF counterpart tuned on its own "
                "feature set. Grid: n_estimators pinned 100 (E48 red herring), "
                "max_depth {3,4,6}, max_features {0.05..0.30}, min_samples_leaf "
                "{300,1000,2000,4000} (E48 ceiling opened). --n-iter 40, 5-fold "
                "CV, CURUPIRA cadence (rolling 130/130, embargo 20, seed 42). "
                "SERIAL search x n_jobs (one joblib level -- E51's fenc arm hung "
                "on the nested search-jobs>1 x n_jobs>1 loky path), parallelised "
                "at the DATASET level. The pool a paired-vs-book comparison and "
                "any RF blend draws from. E51's raw arm scored CORR 0.154 / "
                "payout 0.080; the search picks leaf 300 (lighter than E48's OOS "
                "marginal of 1000 -- a CV/OOS nuance).",
        "runner": "experiments/e52_rf_ktz_tune_crypto.sh",
        "prefix": "e52_", "scout_prefix": "e52scout_",
        "models": ["rf_ktz"], "expected_arms": 9,
    },
    "E53": {
        "title": "Gaussian Process viability test -- full book, reg + clf",
        "idea": "First Gaussian Process on the board, and the gate before the "
                "expensive E54 KTZ sweep. One gp arm per (dataset x task) across "
                "ALL NINE book datasets (raw22, tenc23, interact36, fenccausal44, "
                "cumlag3-88, tenc_onehot763, onehot_rollsel786, allfeat788, "
                "allfeatpp812) x {regression (posterior mean on the 5-binned "
                "target), classification (GPC positive-class probability, target "
                "binarised at 0.5)} = 18 arms. A subsampled EXACT GP "
                "(src/prediction/gp.py): the training rows are capped (reg 6000, "
                "clf 1500) via --model-options because a GP is O(n^3) and a "
                "130-date block is ~40k rows; the kernel is frozen and the outer "
                "search tunes family + length-scale (+ alpha for reg). Light "
                "--n-iter 12 at the deployable cadence (rolling 130/130, embargo "
                "20, n_splits 5, seed 42); SERIAL search x arm-level pool (E52's "
                "anti-loky-hang pattern). Answers, before any sweep budget is "
                "spent: does the wrapper run end to end on both tasks, and is reg "
                "or clf the task worth carrying into E54/E55. A viability probe, "
                "like E18 (TabNet) -- not a tuned model.",
        "runner": "experiments/e53_gp_test_crypto.sh",
        "prefix": "e53_", "scout_prefix": "e53scout_",
        "models": ["gp"], "expected_arms": 18,
    },
    "E54": {
        "title": "Gaussian Process hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)",
        "idea": "The 'Overhyped?' framework E17 ran on the MLP, E27/E49 on the "
                "boosters and E48 on the random forest, now on a GAUSSIAN PROCESS "
                "-- the first kernel model on the board. One pinned walk-forward "
                "per cell of a 4x2x3x4 factorial: kernel {Matern-1/2,-3/2,-5/2, "
                "RBF} x {isotropic, ARD} x length_scale {0.25,1.0,4.0} (features "
                "are [0,1]-scaled) x alpha {1e-3,1e-2,1e-1,1.0} = 96 cells on the "
                "raw 22 of crypto/v2.0, then marginals, fANOVA, corners and priors "
                "via src.analysis.hyperparam_sensitivity_gp. The kernel is FROZEN "
                "(optimizer=None) so a pinned cell measures the pinned value, not "
                "the marginal-likelihood basin. subsample 6000 and normalize_y "
                "True are pinned (a subsample fidelity ladder is a separate probe); "
                "n_splits 2 since the pinned inner CV only records an unused score; "
                "regression only (GPC has no alpha). Two-way claim: length_scale x "
                "alpha -- function smoothness vs observation noise, the GP analogue "
                "of the paper's lr x alpha and of E48's max_features x "
                "min_samples_leaf. Maps the surface once so a gp_ktz grid can be "
                "set from evidence; the length-scale and alpha priors ALSO become "
                "the GP's own kernel prior (E55).",
        "runner": "experiments/e54_gp_hyperparam_sensitivity_crypto.sh",
        "prefix": "e54_", "scout_prefix": "e54scout_",
        "models": ["gp"], "expected_arms": 96,
    },
    "E55": {
        "title": "Factor-momentum stake timing across the submitted book",
        "idea": "Gupta-Kelly 'Factor Momentum Everywhere' applied to STAKE SIZING: "
                "each of the 24 submitted models is a factor, its per-period payout "
                "is its return, and the bet is to tilt stake toward models that did "
                "well on recently RESOLVED rounds, walk-forward. Two practical "
                "constraints make it a real test not a paper one: the 20-day "
                "DECISION LAG (signal .shift(20), decide on resolved payouts only) "
                "AND the WITHDRAW-DELAY LADDER (a stake change glides in over ~20 "
                "rounds because decreases bleed out as rungs resolve; realised "
                "weight = trailing-20 mean of the target, not the target) -- the "
                "friction dynamic_stake.py ignored. Plus the 0.1-NMR/round floor "
                "(the_arkivist 2026-09-08) via long-only + optional weight floor. "
                "Equal-capital (weights sum to 1), z-tilt allocation, sweep "
                "rolling lookbacks {20,40,60,90,120,180,250} and EWMA halflives "
                "{10,20,40,80,120}. Baselines: static equal weight, and full-stack-"
                "on-the-causal-best (topk=1, ladder-timed). Select-on-train / "
                "report-on-held-out, paired NW lag 20, turnover measured. Panel is "
                "the book's per-period payout (pseudo-OOS, ratios not NMR, as E42). "
                "FINDING: on the held-out third momentum BEATS equal weight across "
                "every lookback (t +2.2..+4.1) and best-single is strongest "
                "(+0.0092, t+3.95), lookback-insensitive, EWMA~=rolling, turnover "
                "tiny (0.004-0.04, ladder-smoothed) -- but the edge is regime-"
                "concentrated (full-window t only 0.5-1.8), so promising on the "
                "recent regime, not established across history.",
        "runner": "src/analysis/factor_momentum_stake.py",
        "prefix": "e55_", "scout_prefix": None,
        "models": [], "expected_arms": 0,
    },
    "E56": {
        "title": "Chronos-2 with the feature cross-section as covariates",
        "idea": "The missing cell of the foundation-model arc. Univariate "
                "time-series FMs got ~0 (TimesFM E38 +0.012, TEMPO E40 -0.017) "
                "because they saw only the target's own history; the tabular FM on "
                "the feature cross-section worked (TabPFN E19 CORR 0.145). Chronos-2 "
                "(amazon/chronos-2, ~120M, arXiv:2510.15821) is the first "
                "time-series FM that ingests EXOGENOUS COVARIATES (its 'group "
                "attention' does in-context learning across a target + its "
                "covariates AND across related series). E56 feeds the 44 fenccausal "
                "features as covariates so a time-series FM can finally use the "
                "cross-section. Per scored era T: one Chronos2Pipeline.predict_df "
                "panel call over ALL symbols (group attention shares cross-series "
                "info) -- context = every symbol's (target+features) up to the "
                "20-period leak wall T-20; features for eras T-19..T are passed as "
                "KNOWN-FUTURE covariates (features at date t are observed at t; only "
                "the 20-day-forward TARGET is embargoed), so the current "
                "cross-section conditions the forecast; prediction_length 20, the "
                "era-T median is the signal. Zero-shot, /scratch venv reusing base "
                "cu118 torch. Scored on tabpfn/e19_fenccausal's 510 dates; paired NW "
                "vs E38/E40 (does covariate support beat univariate?) and vs E19 "
                "(does a TS-FM with covariates reach the tabular edge?). Caveat: "
                "Chronos-2 covariates are TIME-ALIGNED per-series features, a weaker "
                "match to the purely cross-sectional edge -- the measured risk of "
                "the idea. Numbered E56 because E54's text earmarks E55 for a GP "
                "kernel-prior study.",
        "runner": "src/analysis/chronos2_covariates.py",
        "prefix": "e56_", "scout_prefix": None,
        "models": ["chronos2"], "expected_arms": 1,
    },
    "E57": {
        "title": "rf_ktz 60-day tune of E52's top-5-MMC datasets",
        "idea": "The 60-day twin of E52's best arms. E52 tuned rf_ktz from scratch "
                "on all 9 book datasets on target_binned_return_20 and ranked them "
                "by MMC (what a slot pays over the meta model); E57 re-tunes the "
                "FIVE highest-MMC datasets -- allfeat(788), fenccausal(44), "
                "tenc_onehot(763), allfeatpp(812), interact(36) -- on the 60-day "
                "label instead. The ONLY change from E52 is the target: same "
                "rf_ktz grid (E48 priors), same 40-draw from-scratch search, same "
                "rolling 130/130 / n_splits 5 / seed 42, run on the SAME sklearn "
                "0.24 (.e48venv) so only the label differs. --embargo auto derives "
                "60 from the horizon; 60-day runs score ~470 periods. Asks whether "
                "RF's 20-day MMC strength carries to the 60-day label and reaches "
                "the 60-day book (SACI_03 MMC 0.0788 / payout 0.0930, "
                "CURUPIRA_05_60 MMC 0.0804 / payout 0.0922).",
        "runner": "experiments/e57_rf_ktz_60d_tune_crypto.sh",
        "prefix": "e57_", "scout_prefix": "e57scout_",
        "models": ["rf_ktz"], "expected_arms": 5,
    },
    "E58": {
        "title": "lgb_ktz 60-day tune of E29's top-5-MMC datasets",
        "idea": "The LightGBM sibling of E57 and the 60-day twin of E29's best "
                "arms. E29 tuned lgb_ktz from scratch on all 9 book datasets on "
                "target_binned_return_20 and ranked them by MMC (what a slot pays "
                "over the meta model); E58 re-tunes the FIVE highest-MMC datasets "
                "-- tenc_onehot(763), allfeat(788), interact(36), allfeatpp(812), "
                "fenccausal(44), the SAME five the rf tune (E57) picked -- on the "
                "60-day label instead. The ONLY change from E29 is the target: "
                "same lgb_ktz grid (E27 priors, L=lr*n_est<=40), same 60-draw "
                "from-scratch RandomizedSearchCV, same rolling 6month/6month / "
                "n_splits 5 / seed 42 / default python3, so only the label "
                "differs. --embargo auto derives 60 from the horizon; 60-day runs "
                "score ~470 periods. Excludes the leaky non-causal global encoding "
                "(fencglobal, ~0.14 MMC an artefact) and all blend/ensemble arms "
                "(no hyperparameters to tune). Asks whether LightGBM's 20-day MMC "
                "strength carries to the 60-day label and reaches the 60-day book "
                "(SACI_03 MMC 0.0788 / payout 0.0930, CURUPIRA_05_60 MMC 0.0804 / "
                "payout 0.0922); secondary paired read vs E01's 60d m50 pins.",
        "runner": "experiments/e58_lgb_ktz_60d_tune_crypto.sh",
        "prefix": "e58_", "scout_prefix": "e58scout_",
        "models": ["lgb_ktz"], "expected_arms": 5,
    },
    "E59": {
        "title": "catboost_ktz 60-day tune of E32's top-5-MMC datasets",
        "idea": "The CatBoost sibling of E57 (rf) and E58 (lgb): the 60-day twin "
                "of E32's best catboost picks. E32 tuned catboost_ktz from scratch "
                "on all 9 book datasets on target_binned_return_20 and ranked them "
                "by MMC (what a slot pays over the meta model); E59 re-tunes the "
                "FIVE highest-MMC by CATBOOST's own ranking -- fenccausal(44), "
                "allfeatpp(812), tenc(23), allfeat(788), tenc_onehot(763) -- on "
                "the 60-day label instead. (The five differ from the rf/lgb tunes' "
                "set because each family's per-dataset MMC ordering differs; "
                "catboost keeps tenc where they take interact.) The ONLY intended "
                "change from E32's sel arms is the target: same 12-cell KTZ grid "
                "(depth{2,4} x lr{.005,.01,.02} x rsm{.1,.2}; iterations 2000, "
                "l2_leaf_reg 10, min_data_in_leaf 100 pinned), same exhaustive "
                "--n-iter 12 search, same rolling 130/130 / n_splits 5 / seed 42. "
                "--embargo auto derives 60 from the horizon (forced by the label, "
                "not chosen), which thins the inner CV to ~2 usable folds; 60-day "
                "runs score ~470 periods. CORR is NOT comparable to the 20-day E32 "
                "arms -- read against the 60-day book (SACI_03 MMC 0.0788 / payout "
                "0.0930, CURUPIRA_05_60 MMC 0.0804 / payout 0.0922). Asks whether "
                "catboost's 20-day MMC strength carries to the 60-day label.",
        "runner": "experiments/e59_catboost_ktz_tune_t60_crypto.sh",
        "prefix": "e59_", "scout_prefix": "e59scout_",
        "models": ["catboost_ktz"], "expected_arms": 5,
    },
    "E60": {
        "title": "MLP fit-60 / score-20: the best-MMC MLP setups re-tuned on 60d",
        "idea": "The MLP sibling of E57(rf)/E58(lgb)/E59(catboost), with one "
                "deliberate difference: those FIT AND SCORE on the 60-day label; "
                "E60 FITS on 60d but SCORES on the payable 20-day one (the inverse "
                "of SACI_03, which fits 20 and scores 60). The five MLP setups that "
                "led MMC per feature set are re-tuned FROM SCRATCH -- a fresh search, "
                "not the pinned 20-day config: mlp/fenccausal(44), "
                "mlp/tenc-onehot(763), torch_mlp/tenc(23), mlp/raw(22), and "
                "mlp_ktz2/fenccausal (the E33-reprior wider grid). Each fits "
                "target_binned_return_60 at embargo 60 (forced by the horizon, ~470 "
                "periods, ~8 effective windows -> NW lag 60), then its predictions "
                "are RE-SCORED against target_binned_return_20 via "
                "src/analysis/cross_target.py -- the same meta model every deployed "
                "slot is graded on. Deliverable is the 20-day MMC/payout vs each "
                "dataset's fit-20 MLP already on disk (fenccausal 0.0825, "
                "tenc-onehot 0.0781, tenc 0.0775, raw 0.0746): does training on the "
                "smoother 60-day signal pay better on the 20-day label? The runner "
                "also reports native 60d metrics, comparable to E57-59's 60d board.",
        "runner": "experiments/e60_mlp_fit60_score20_crypto.sh",
        "prefix": "e60_", "scout_prefix": "e60scout_",
        "models": ["mlp", "torch_mlp", "mlp_ktz2"], "expected_arms": 5,
    },
    "E61": {
        "title": "Multi-symbol multivariate (SUR-style) ridge vs the stacked ridge",
        "idea": "The deployed linear slots fit ONE shared beta on the STACKED "
                "panel -- every (symbol,date) row an independent observation, "
                "errors i.i.d. That is ridge_glmnet, pooled panel ridge. `sur_ridge` "
                "(src/prediction/sur_ridge.py) keeps the shared beta but partials "
                "out k common CROSS-SYMBOL factors from the residuals -- the "
                "feasible-GLS adjustment under a factor covariance, the "
                "Seemingly-Unrelated-Regressions idea -- so it exploits that coins' "
                "errors co-move within a date. n_factors=0 is an EXACT stacked-ridge "
                "control, so the whole value is the searched k>0. The symbol of each "
                "row is routed to fit() (WANTS_SYMBOL) alongside the date; neither "
                "is needed at predict, which stays x'beta. Two models x two tasks x "
                "9 book datasets = 36 from-scratch tunes on ONE protocol (rolling "
                "130/130, embargo 20, 5 folds, 50 draws, seed 42): sur_ridge and "
                "ridge_glmnet, regression and classification, on raw/tenc/interact/"
                "fenccausal/cumlag3/tenc_onehot/onehot_rollsel/allfeat/allfeatpp, so "
                "every sur_ridge arm pairs against its matched stacked ridge and reg "
                "pairs against clf. A result is sur_ridge beating ridge_glmnet on "
                "numerai_corr at paired |t|>=2; the prior (E27/E30) is a null -- "
                "modelling the error covariance sharpens beta in MSE but need not "
                "move rank corr. The n_factors the search KEEPS is the diagnostic.",
        "runner": "experiments/e61_sur_ridge_crypto.sh",
        "prefix": "e61_",
        "models": ["sur_ridge", "ridge_glmnet"], "expected_arms": 36,
    },
    "E62": {
        "title": "SUR cross-symbol factor-GLS on trees (lgb/rf/catboost) vs stacked",
        "idea": "E61 (sur_ridge) generalised to the tree families. A tree has no "
                "coefficients to partial, so SurAugment (src/prediction/sur_ridge.py) "
                "adds the k common CROSS-SYMBOL factor scores as EXTRA FEATURES at "
                "fit (the tree explains away common shocks in-sample) and predicts "
                "with them ZEROED -- future factor scores are unknowable, as sur_ridge "
                "predicts x'beta without F. n_factors=0 is an EXACT plain-tree control; "
                "factors come from a cheap stage-1 ridge's residuals so every family "
                "shares one model-agnostic common-shock. 108 arms = 3 families x 2 "
                "tasks x 9 book datasets x {sur, stacked}, matched protocol (rolling "
                "130/130, embargo 20, 5 folds, 50 draws, seed 42): sur_lgb vs lgb_ktz, "
                "sur_rf vs rf_ktz, sur_catboost vs catboost_ktz(reg)/catboost(clf) "
                "(catboost_ktz is reg-only). A result is sur beating stacked on "
                "numerai_corr at paired |t|>=2; prior (E61 null, E27/E30) is that the "
                "factor is picked in CV but dies out-of-sample. n_factors kept is the "
                "diagnostic. Symbol+date routed to fit() via WANTS_SYMBOL/WANTS_PERIOD.",
        "runner": "experiments/e62_sur_trees_crypto.sh",
        "prefix": "e62_",
        "models": ["sur_lgb", "sur_rf", "sur_catboost", "lgb_ktz", "rf_ktz",
                   "catboost_ktz", "catboost"], "expected_arms": 108,
    },
    "E63": {
        "title": "Literature crypto momentum features vs the raw 22",
        "idea": "The 22 shipped features are rank-binned and hide the price path. "
                "build_momentum rebuilds 17 momentum features from spot prices -- "
                "from the crypto momentum literature (Liu&Tsyvinski 2021, Grobys&"
                "Sapkota 2019, Tzouvanas 2019, Li 2021, Liu&Tsyvinski&Wu 2022, "
                "Begusic 2019) plus Tzotchev 2018 robust trend-following (bounded "
                "vol-adjusted trend t-stat 2N(sqrt(T)mean/std)-1 at six crypto "
                "horizons 7/14/30/60/120/240d + their composite sum) -- puts them "
                "on Numerai's exact {0,.25,.5,.75,1} "
                "scale (src/data_mani/binning.py) and merges them POINT-IN-TIME "
                "(row D uses spot closes <= D-1, the last completed bar known at "
                "submission; attach_signals_asof), producing data/crypto/v2.0-litmom. "
                "6 arms, matched protocol (rolling 130/130, embargo 20, 5 folds, "
                "50 draws, seed 42): {lgb, ridge_glmnet} x {base v2.0, mom "
                "v2.0-litmom, momonly ^feature_mom_}. A result is base+mom beating "
                "matched base on numerai_corr at paired |t|>=2 -- the momentum "
                "block adds ranking info the binned features lack. momonly is the "
                "standalone diagnostic, not expected to beat base.",
        "runner": "experiments/e63_momentum_features_crypto.sh",
        "prefix": "e63_", "scout_prefix": None,
        "models": ["lgb", "ridge_glmnet"], "expected_arms": 6,
    },
    "E64": {
        "title": "KTZ-prior model bake-off: litmom vs base across 5 model classes",
        "idea": "E63 asked 'does litmom help?' for two models; E64 widens it to "
                "five function classes, each TUNED FROM SCRATCH under its KTZ "
                "(Kalfa-Timmermann-van der Zwan) priors: a linear model (ridge -- "
                "no ktz variant, its fixed log-alpha grid IS the evidenced search, "
                "the linear control), two boosters (lgb_ktz E27, catboost_ktz "
                "E29/E32), a bagged forest (rf_ktz E48) and a feedforward net "
                "(mlp_ktz E17). 10 arms = 5 models x {base crypto/v2.0 (22f), "
                "litmom crypto/v2.0-litmom (22+17 momentum)}, matched cadence "
                "(rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3, seed "
                "42, target_binned_return_20, metric corr). n_iter per model at "
                "each grid's evidenced budget {ridge 50, lgb_ktz 50, catboost_ktz "
                "12, rf_ktz 40, mlp_ktz 80}, identical across a model's two arms. "
                "A result is litmom beating matched base on numerai_corr at paired "
                "Newey-West |t|>=2 (lag 20); the clean finding is CONSISTENCY "
                "across function classes. A null across all five, with intervals, "
                "is equally a finding (E63 EDA put the incremental IC at ~0.015).",
        "runner": "experiments/e64_ktz_litmom_bakeoff_crypto.sh",
        "prefix": "e64_", "scout_prefix": "e64scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E65": {
        "title": "E64 re-run on the updated litmom (22+17) at uniform n_iter=50",
        "idea": "E64 re-run, two things changed. (1) litmom grew: E64 ran on "
                "crypto/v2.0-litmom = 22 base + 10 momentum (32 cols); it was "
                "rebuilt to 22 + 17 (39 cols) -- the original 10 plus a 7-feature "
                "trend block (feature_mom_trend_7d/14d/30d/60d/120d/240d and "
                "feature_mom_trend_robust, the Tzotchev robust trend). (2) the "
                "search budget is now UNIFORM n_iter=50 for every model (E64 used "
                "per-model 50/50/12/40/80); catboost_ktz's 12-cell grid clamps 50 "
                "-> 12 (a full sweep). 10 arms = 5 models x {base crypto/v2.0 "
                "(22f), litmom crypto/v2.0-litmom (39f)}, same cadence as E64 "
                "(rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3, seed "
                "42, target_binned_return_20, metric corr). base arms re-run at 50 "
                "so the within-model pairing is exact. Result = litmom beats "
                "matched base on numerai_corr at paired |t|>=2; also read E65 vs "
                "E64 -- did the trend block change any model's verdict? E64 (22+10) "
                "was a null across all 5 at |t|<2.",
        "runner": "experiments/e65_ktz_litmom_v2_bakeoff_crypto.sh",
        "prefix": "e65_", "scout_prefix": "e65scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E66": {
        "title": "Swap the base momentum proxies for the literature momentum block",
        "idea": "E64/E65 found ADDING litmom to the full 22 is null-to-negative "
                "on numerai_corr because the new momentum features are redundant "
                "with the base's OWN momentum proxies -- feature_momentum, RSI, "
                "Sharpe, Bollinger (each 20d/60d; the new features load 0.5-0.81 "
                "on them). E66 asks the SUBSTITUTION question: drop those 8 base "
                "momentum proxies and REPLACE them with the 17 literature momentum "
                "features. 10 arms = 5 models x {base crypto/v2.0 (raw 22), swap "
                "(v2.0-litmom minus the 8 proxies via --feature-exclude = 14 "
                "non-momentum base + 17 mom = 31f)}, uniform n_iter=50, same "
                "cadence (rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3, "
                "seed 42, target_binned_return_20, metric corr). A result = swap "
                "beats the raw-22 base at paired Newey-West |t|>=2 (lag 20) -- the "
                "literature block does the momentum job better than the crude "
                "proxies. Confounds remove-native with add-literature (inherent to "
                "a swap); prior is null-or-worse (E63 EDA, E65 null-to-negative).",
        "runner": "experiments/e66_momswap_litmom_crypto.sh",
        "prefix": "e66_", "scout_prefix": "e66scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E67": {
        "title": "Old momentum proxies vs new literature momentum, momentum-only, from scratch",
        "idea": "The clean head-to-head E64-E66 kept confounded by the rest of the "
                "panel. E67 strips everything else away and races the two momentum "
                "blocks ALONE, each tuned from scratch: old = the 8 base momentum "
                "proxies (feature_momentum, RSI, Sharpe, Bollinger; each 20d/60d); "
                "new = the 17 literature momentum features (feature_mom_*). Both "
                "selected from crypto/v2.0-litmom via --feature-include (a regex), "
                "so the two arms share rows/dates/cadence/seed and differ ONLY in "
                "the momentum representation -- NO other features. 10 arms = 5 "
                "models x {old (8f), new (17f)}, uniform n_iter=50, same cadence "
                "(rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3, seed 42, "
                "target_binned_return_20, metric corr). Result = new beats old at "
                "paired Newey-West |t|>=2 (lag 20). Levels are toy (8-17 feat) and "
                "NOT comparable to the deployed book -- read the DELTA only.",
        "runner": "experiments/e67_old_vs_new_momentum_crypto.sh",
        "prefix": "e67_", "scout_prefix": "e67scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E68": {
        "title": "Cross-sectional momentum-regime block on v2.0-momswap",
        "idea": "Does a market-REGIME descriptor of momentum help on top of the "
                "momswap set (14 base + 17 momentum, 31f)? build_xsmom collapses "
                "the raw (un-binned) momentum cross-section into 3 day-level "
                "numbers per date -- the cross-sectional MEAN, VARIANCE and SKEW of "
                "the momentum distribution across coins (per-signal moment of each "
                "of the 17 raw signals -> causal percentile -> average -> causal "
                "percentile), broadcast POINT-IN-TIME (D-1) as feature_xsmom_"
                "{mean,var,skew} -> crypto/v2.0-momswap-xsmom (34f). Because they "
                "are IDENTICAL across the cross-section on a date they carry ~zero "
                "standalone within-era numerai_corr and are INERT for a linear "
                "model (ridge is the negative control -- a day-constant cannot move "
                "a within-date rank); only trees/MLP can use them via regime "
                "gating/interactions. 10 arms = 5 models x {base v2.0-momswap, "
                "xsmom v2.0-momswap-xsmom}, tuned from scratch, uniform n_iter=50, "
                "same cadence (rolling 130/130, embargo auto=20, 5 folds, seed 42, "
                "target_binned_return_20). A result = xsmom beats base at paired "
                "|t|>=2 on a tree/MLP; prior is a null (regime features are hard to "
                "make pay on ~26 effective windows).",
        "runner": "experiments/e68_xsmom_momswap_crypto.sh",
        "prefix": "e68_", "scout_prefix": "e68scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E69": {
        "title": "Macro / cross-asset information set on v2.0-momswap",
        "idea": "The E68 sibling with a different information set: E68 added a "
                "cross-sectional momentum REGIME block, E69 adds EXTERNAL MACRO "
                "state. On top of v2.0-momswap (31f), add the 18 shared macro / "
                "cross-asset features (feature_macro_*: gold & silver returns "
                "20d/60d, S&P returns 5d/20d/60d, DXY dollar-index momentum "
                "5d/20d/60d, US Treasury yields y1/y2/y5/y10/y30, curve slopes "
                "1s10s/2s10s/5s30s; build_macro.py -> "
                "crypto/v2.0-momswap-macro, 49f), causal-percentile normalised "
                "across time and broadcast point-in-time (D-1). Like xsmom, macro "
                "is IDENTICAL across the cross-section on a date -> ~zero standalone "
                "within-era numerai_corr and INERT for ridge (the negative "
                "control); only trees/MLP can use it via regime gating / "
                "interactions. 10 arms = 5 models x {base v2.0-momswap, macro "
                "v2.0-momswap-macro}, tuned from scratch, uniform n_iter=50, same "
                "cadence (rolling 130/130, embargo auto=20, 5 folds, seed 42, "
                "target_binned_return_20). A result = macro beats base at paired "
                "|t|>=2 on a tree/MLP; prior is a null (the macro block's own note "
                "reports ~zero standalone IC, help only via interactions).",
        "runner": "experiments/e69_macro_momswap_crypto.sh",
        "prefix": "e69_", "scout_prefix": "e69scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E70": {
        "title": "Engineered allfeat-style block on v2.0-momswap-macro",
        "idea": "Does the allfeat_pp recipe help on top of momswap-macro? "
                "build_momswap_eng.py adds, retargeted to the swapped set (which "
                "dropped RSI/base-momentum): tenc (1, causal lag-20 symbol "
                "target-encoding), onehot (752 symbol dummies), rollsel (87, "
                "rolling means {5,20,60} of the 17 momentum + 10 macro-return + 2 "
                "volatility; RSI absent so dropped), spread (12, term-structure "
                "differences within each momentum/macro family), interact (24 = "
                "accel + adapted products with feature_mom_ret_28d standing in for "
                "the removed feature_momentum, PLUS 12 momentum x macro-momentum "
                "products = crypto trend gated by the macro regime) -> "
                "crypto/v2.0-momswap-macro-eng (925f). 10 arms = 5 models x {base "
                "v2.0-momswap-macro (49f), eng (925f)}, from scratch, uniform "
                "n_iter=50, same cadence. Onehot/tenc give symbol identity, "
                "momXmacro gives regime gating -- both work via INTERACTIONS "
                "(trees/MLP), less a linear model. A result = eng beats base at "
                "paired |t|>=2; prior is a null (11 feature blocks tried on this "
                "panel, none clears the raw set at |t|>=2).",
        "runner": "experiments/e70_eng_momswap_macro_crypto.sh",
        "prefix": "e70_", "scout_prefix": "e70scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E71": {
        "title": "Interactions-only block on v2.0-momswap-macro (the E70 follow-up)",
        "idea": "E70 dumped the full 925f allfeat block and it was null-to-negative "
                "-- the 752 onehot dummies crushed ridge/MLP on CORR (-0.053/-0.040 "
                "at |t|>2), the boosters showed only a slight non-significant +MMC "
                "(~0.006, t<1). E71 strips it to JUST the 24-column interact block "
                "(build_momswap_eng.py --blocks interact -> "
                "crypto/v2.0-momswap-macro-xinteractions, 73f): 6 accel + 6 adapted "
                "products (feature_mom_ret_28d stands in for the removed "
                "feature_momentum) + 12 momentum x macro-momentum products (crypto "
                "momentum GATED by the macro regime, a per-symbol cross term that "
                "CAN move a within-date rank, so ridge is no longer a pure negative "
                "control). 10 arms = 5 models x {base v2.0-momswap-macro (49f), xint "
                "(73f)}, from scratch, uniform n_iter=50, same cadence. A result = "
                "xint beats base at paired |t|>=2; tests whether the boosters' E70 "
                "hint firms up once the dummy noise is removed. Prior: small effect "
                "at most, likely a tighter null.",
        "runner": "experiments/e71_xinteractions_momswap_macro_crypto.sh",
        "prefix": "e71_", "scout_prefix": "e71scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E72": {
        "title": "Carver trend-following stack vs the raw 22",
        "idea": "Bottom-line test of the whole engineering stack against the "
                "original raw 22. crypto/v2.0-momswap-macro-carver (64f) = the "
                "momentum swap (14 non-momentum base + 17 literature momentum) + 18 "
                "macro/cross-asset (incl. DXY momentum) + 15 Robert Carver-style "
                "TREND-FOLLOWING features (feature_tf_*: EWMAC crossovers "
                "2_8/4_16/8_32/16_64/32_128, breakouts 20/40/80/160, Bollinger-z "
                "20/40/80, Clenow 60/90, new-high-20-in-5). E72 pairs it against "
                "crypto/v2.0 (raw 22, the base). 10 arms = 5 models x {base v2.0, "
                "carver}, from scratch, uniform n_iter=50, same cadence (rolling "
                "130/130, embargo auto=20, 5 folds, seed 42, target_binned_return_20). "
                "Whole-stack comparison (many blocks differ at once, incl. the raw "
                "momentum proxies being replaced) -- the deployment question 'is the "
                "engineered 64 better than the raw 22', not a single-block ablation. "
                "A result = carver beats raw at paired |t|>=2; prior is a null "
                "(no engineered block has beaten the raw 22 on this panel at |t|>=2).",
        "runner": "experiments/e72_carver_vs_raw_crypto.sh",
        "prefix": "e72_", "scout_prefix": "e72scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E73": {
        "title": "litmom with base masked on momentum-missing rows (mommask), 3-way",
        "idea": "crypto/v2.0-litmom-mommask = litmom (39f) with the 22 BASE features "
                "set to neutral 0.5 in ~17% of cells, CONCENTRATED on rows where the "
                "momentum block is missing (P(base masked | momentum missing)=0.996 "
                "vs 0.263 when present; ~11 of 22 base features masked per masked "
                "row); the 17 momentum columns are identical to litmom. So on the "
                "~22% no-spot rows that already lean entirely on the base block, "
                "about half the base is also hidden. Masking only removes info (0.5 "
                "neutral) so it is leakage-safe. E73 runs 5 models on mommask, from "
                "scratch, uniform n_iter=50 at EXACTLY E65's protocol (rolling "
                "130/130, embargo auto=20, 5 folds, seed 42, target_binned_return_20), "
                "then compares THREE-WAY against v2.0 (e65_*_base) and litmom "
                "(e65_*_litmom), reused from E65 -- so only the 5 mommask arms run. "
                "Prior: masking base info should not help numerai_corr; a null-or-"
                "worse (esp. vs litmom) is expected, and a significant drop quantifies "
                "how much the base block carries the no-spot names.",
        "runner": "experiments/e73_litmom_mommask_crypto.sh",
        "prefix": "e73_", "scout_prefix": "e73scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E74": {
        "title": "Realised-volatility estimator block vs the raw 22",
        "idea": "Does a proper volatility block help on top of the base's two crude "
                "volatility features? crypto/v2.0-vol (44f) = the raw 22 + 22 "
                "realised-volatility estimators (feature_vol_*) computed from spot "
                "OHLCV over a hyperparameter grid: four estimators x their lookback -- "
                "squared-returns realised variance (feature_vol_sqret_{5,10,20,40,60,"
                "120}d), EWMA/RiskMetrics variance (feature_vol_ewma_lam{90,94,97,99}), "
                "Parkinson high/low range (feature_vol_park_{5,10,20,40,60,120}d), and "
                "Yang-Zhang full-OHLC (feature_vol_yz_{5,10,20,40,60,120}d, the only "
                "classical estimator robust to drift and opening gaps). All per-symbol, "
                "computed raw from spot, attached as-of <= D-1, then rank-binned per "
                "date (build_vol.py -> volatility_features.py, the first feature module "
                "that reads OHLC not close alone). Coverage ~82-89% of panel rows (the "
                "no-spot memecoins are neutral-filled). E74 pairs it against crypto/v2.0 "
                "(raw 22, base). 10 arms = 5 models x {base v2.0 (22f), vol v2.0-vol "
                "(44f)}, from scratch, uniform n_iter=50, same cadence (rolling 130/130, "
                "embargo auto=20, 5 folds, seed 42, target_binned_return_20). Isolate "
                "the block alone with --feature-include '^feature_vol_'. A result = vol "
                "beats raw at paired |t|>=2; prior is a null (no engineered block has "
                "beaten the raw 22 on this panel at |t|>=2), and the base already "
                "carries feature_volatility_{20,60}d, so the block must add over those.",
        "runner": "experiments/e74_vol_vs_raw_crypto.sh",
        "prefix": "e74_", "scout_prefix": "e74scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E75": {
        "title": "momswap-macro mommask, 3-way vs momswap-macro + v2.0",
        "idea": "The momswap-macro analog of E73's mommask (E74 was already taken by "
                "the vol study, so this is E75). crypto/v2.0-momswap-macro-mommask "
                "(49f) = momswap-macro with features set to neutral 0.5 concentrated "
                "on momentum-missing rows (P(base masked|mom missing)=0.961) -- unlike "
                "E73 it masks ALL groups: base ~17%, momentum ~10%, macro ~34% of "
                "cells (real->0.5, info removed; 5.27M extra neutral cells), leakage-"
                "safe. 5 models on mommask, from scratch, n_iter=50 at the shared "
                "protocol (rolling 130/130, embargo auto=20, 5 folds, seed 42, "
                "target_20), compared 3-way vs momswap-macro (e70_*_base) and v2.0 "
                "(e72_*_base), reused. Prior: masking info should not help; null-or-"
                "worse. Shared mommask_core.sh via the e75 wrapper.",
        "runner": "experiments/e75_momswap_macro_mommask_crypto.sh",
        "prefix": "e75_", "scout_prefix": "e75scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E76": {
        "title": "xinteractions mommask, 3-way vs xinteractions + v2.0",
        "idea": "The xinteractions analog. crypto/v2.0-momswap-macro-xinteractions-"
                "mommask (219f) = the 219-feature xinteractions set (momswap-macro + "
                "170 macro-mom x crypto-mom products) with features masked to neutral "
                "0.5 concentrated on momentum-missing rows. 5 models from scratch, "
                "n_iter=50 at the shared protocol, compared 3-way vs xinteractions "
                "(e71_*_xint) and v2.0 (e72_*_base), reused. e71_ridge_xint was "
                "missing from E71 and is run separately to complete the ridge cell. "
                "Prior: null-or-worse. Shared mommask_core.sh via e76.",
        "runner": "experiments/e76_xinteractions_mommask_crypto.sh",
        "prefix": "e76_", "scout_prefix": "e76scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E77": {
        "title": "carver mommask, 3-way vs carver + v2.0",
        "idea": "The carver analog. crypto/v2.0-momswap-macro-carver-mommask (64f) = "
                "the carver set (momswap-macro + 15 Carver/Clenow trend-following "
                "features) with features masked to neutral 0.5 concentrated on "
                "momentum-missing rows. 5 models from scratch, n_iter=50 at the "
                "shared protocol, compared 3-way vs carver (e72_*_carver) and v2.0 "
                "(e72_*_base), reused. Prior: null-or-worse. Shared mommask_core.sh "
                "via e77.",
        "runner": "experiments/e77_carver_mommask_crypto.sh",
        "prefix": "e77_", "scout_prefix": "e77scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E78": {
        "title": "vol mommask, 3-way vs vol + v2.0",
        "idea": "The volatility analog of the mommask sweep (E73/E75/E76/E77), keyed "
                "on the vol block instead of momentum. crypto/v2.0-vol-mommask (44f) = "
                "the vol set (raw 22 + 22 feature_vol_ realised-volatility estimators) "
                "with the WHOLE row set to neutral 0.5 wherever any of the 22 vol "
                "features is missing -- build_mommask --gate created --require all "
                "--neutralize all, the same recipe as the others but gating on "
                "feature_vol_. 90,164/514,696 rows gated (17.5%): the no-spot names "
                "plus rows with <120d history (the longest vol window; that is why it "
                "gates less than litmom-mommask's 34.4%, whose feature_mom_ret_12_1 "
                "needs 360d). Masking only removes information (real->0.5), so it is "
                "leakage-safe. 5 models from scratch, n_iter=50 at the shared protocol "
                "(rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3, seed 42, "
                "target_binned_return_20), compared 3-way vs vol (e74_*_vol) and v2.0 "
                "(e72_*_base), reused -- so only the 5 mommask arms run (E74 must be on "
                "disk for the vol cells). Shared mommask_core.sh via e78. Prior: "
                "masking information should not help numerai_corr; null-or-worse, and a "
                "drop quantifies how much the vol block carries the no-spot/young names.",
        "runner": "experiments/e78_vol_mommask_crypto.sh",
        "prefix": "e78_", "scout_prefix": "e78scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E79": {
        "title": "xinteractions un-gated (fillmask==raw 219f), 3-way vs mommask + v2.0",
        "idea": "The one non-redundant fillmask. build_mommask --mode fill replaces "
                "ONLY the missing (NaN) cells with 0.5 and keeps present values -- "
                "which is EXACTLY what the loader (fill_missing_features, 0.5) already "
                "does, so a fillmask dataset is byte-identical AS LOADED to its raw "
                "parent (verified for all 5). Four fillmask datasets therefore "
                "reproduce E65/E70/E72/E74 exactly and are NOT re-run. xinteractions "
                "is the exception: crypto/v2.0-momswap-macro-xinteractions was rebuilt "
                "73f->219f AFTER E71 ran at 73f, so no un-gated 219f run exists (E76 "
                "ran the 219f GATED block but reused the 73f e71_*_xint as 'parent', a "
                "feature-count mismatch). E79 runs the 5 models from scratch on "
                "crypto/v2.0-momswap-macro-xinteractions-fillmask (219f, == raw 219f "
                "as loaded), n_iter=50 at the shared protocol, and reports a clean "
                "3-way: v2.0 (e72_*_base) / mommask (e76_*_mommask, gated) / fillmask "
                "(this, un-gated). Answers whether cell-level fill beats whole-row "
                "gating on the 219f interactions block, and supplies the proper "
                "un-gated 219f baseline. Shared mommask_core.sh via e79 (VARIANT=fillmask).",
        "runner": "experiments/e79_xinteractions_fillmask_crypto.sh",
        "prefix": "e79_", "scout_prefix": "e79scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E80": {
        "title": "inflation-expectations (breakeven) macro block vs v2.0 + momswap-macro",
        "idea": "The direct sibling of E69 (macro), keyed on inflation. "
                "crypto/v2.0-momswap-macro-infl (61f) = momswap-macro (49f) + 12 "
                "breakeven-inflation (BEI) features: market-implied inflation "
                "expectations at 5y/10y/30y/5y5y-forward plus their 20d and 60d "
                "changes. feature_macro_ prefix, DAY-LEVEL broadcast (median 1 "
                "distinct value/date, 0 NaN, causal trailing percentile + as-of "
                "attach) -- so like E69's macro it is INERT for ridge on a within-date "
                "rank metric (negative control) and can only help via tree/MLP "
                "interactions. 5 models from scratch, n_iter=50 at the shared protocol "
                "(rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3, seed 42, "
                "target_binned_return_20), reported 3-way: vs raw v2.0 (e72_*_base, "
                "the requested baseline) and vs momswap-macro (e70_*_base, to isolate "
                "the 12 BEI features' marginal value), both reused. Shared "
                "mommask_core.sh via e80 (VARIANT=infl). Prior: ~zero standalone "
                "within-era IC; null-or-marginal like every macro/engineered block.",
        "runner": "experiments/e80_infl_momswap_macro_crypto.sh",
        "prefix": "e80_", "scout_prefix": "e80scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E81": {
        "title": "inflation block on raw baseline: raw+12 BEI (34f) vs raw v2.0",
        "idea": "crypto/v2.0-infl (rebuilt 2026-09-29) = raw 22 + 12 breakeven-"
                "inflation features (feature_macro_bei_* levels + 20d/60d changes) = "
                "34f, a clean SUPERSET of raw. The clean isolation of inflation's "
                "marginal value on the DEPLOYED baseline -- the better-controlled "
                "sibling of E80, which buried the same 12 BEI feats in the 61f "
                "momswap-macro stack. The 12 BEI columns are DAY-LEVEL (inert for "
                "ridge on the within-date rank metric) but the raw 22 restore the "
                "cross-section (289 distinct feature-vectors/date, NOT degenerate); so "
                "ridge ~ raw and only trees/MLP can use inflation as a regime tilt. "
                "(An earlier build of this dataset was inflation-ONLY/12f and "
                "within-date degenerate; it was rebuilt to append onto raw before this "
                "ran.) 5 models from scratch, n_iter=50 at the shared protocol, "
                "reported 3-way vs raw v2.0 (e72_*_base, requested baseline) and "
                "momswap-macro-infl (e80_*_infl), both reused. Shared mommask_core.sh "
                "via e81 (VARIANT=inflraw). Prior: null, like every macro/inflation "
                "block on this within-date metric.",
        "runner": "experiments/e81_inflonly_vs_raw_crypto.sh",
        "prefix": "e81_", "scout_prefix": "e81scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E82": {
        "title": "kitchen-sink combo (raw+vol+infl+carver+ix, 753f) vs raw v2.0",
        "idea": "Does stacking every individually-null engineered block finally beat "
                "raw? crypto/v2.0-vol-infl-carver-ix-fillmask (753f), a clean superset "
                "of raw: raw 22 + feature_vol_ 22 (E74) + feature_macro_bei_ 12 "
                "(inflation, E80/E81) + feature_tf_ 15 (Carver, E72) + feature_ix_ 682 "
                "(interactions -- dominates the count). fillmask = missing cells "
                "pre-filled to 0.5 (== the loader's own fill; no non-fillmask twin, so "
                "this is the canonical form as loaded). NOT within-date degenerate (300 "
                "distinct feature-vectors/date). 5 models from scratch, n_iter=50 at the "
                "shared protocol, reported 3-way vs raw v2.0 (e72_*_base, the requested "
                "baseline) and momswap-macro (e70_*_base, a mid-size engineered stack "
                "for reference -- not nested), both reused. Shared mommask_core.sh via "
                "e82 (VARIANT=combo). Prior (E70 925f eng, E71/E79 219f xint): a huge "
                "interaction count DROWNS ridge (t<-2.7) and hurts MLP while trees stay "
                "~flat; unlikely to beat raw.",
        "runner": "experiments/e82_vol_infl_carver_ix_combo_crypto.sh",
        "prefix": "e82_", "scout_prefix": "e82scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E83": {
        "title": "Frozen TabPFN backbone + a trained 5-class head (E[y|x] readout)",
        "idea": "The GENTLE adaptation E44/E45 lacked. Full fine-tuning corrupted "
                "TabPFN's meta-learned prior and HURT (E44 CRPS+MSE −0.027 t−3.02; "
                "E45 CORR/MMC also worse). Here the backbone is FROZEN (vendored "
                "checkpoint untouched) and only a NEW head trains: a Linear over the "
                "softmax'd bar-distribution (~5000 bins) → 5 classes for the target "
                "bins {0,.25,.5,.75,1} → softmax → E[y|x]. The signal is the "
                "CONTINUOUS bounded expectation, NOT a hard label (numerai_corr is "
                "rank-based; a hard 5-level output would tie ~245 coins/era and "
                "destroy rank resolution — 'discrete' lives in the head + CE loss, "
                "not the emitted number). Head trained with cross-entropy on the "
                "leak-safe pre-2024-06-27 pool. Eval = per-date walk-forward "
                "(context = 130 dates before T−20, subsample 4096); ONE forward per "
                "date yields BOTH e83_head (head E[y|x]) and e83_head_base (raw "
                "bar-dist mean = matched n_est=1 zero-shot, validated ≈ E19). "
                "Pairing head vs base isolates the HEAD exactly; also vs E19. "
                "src/analysis/tabpfn_head.py. Numbered E83 because a concurrent "
                "session took E82 (combo).",
        "runner": "src/analysis/tabpfn_head.py",
        "prefix": "e83_", "scout_prefix": None,
        "models": ["tabpfn_head"], "expected_arms": 2,
    },
    "E84": {
        "title": "TabPFN zero-shot AND frozen+head on each of CU_SA_IA_BO_01's 8 feature-sets",
        "idea": "E19/E83 established TabPFN zero-shot beats fine-tuning and beats a "
                "trained head -- but only ON fenccausal. This sweeps the 8 "
                "feature-sets that compose the deployed slot CU_SA_IA_BO_01 (raw, "
                "fenccausal, allfeat_pp, and the 5 ridge-quintet BLOCKS onehot / "
                "tenc / interact / size / price, the last five as feature-include "
                "regexes) and, for each, runs (a) TabPFN zero-shot at E19's config "
                "(n_est 4, ctx 4096, rolling 130/130, embargo 20, seed 42) and (b) "
                "the E83 frozen-backbone + trained 5-class head (which also emits "
                "its matched n_est=1 zero-shot base). Answers: which feature-set is "
                "best for TabPFN zero-shot, and does the head EVER beat its base on "
                "any block (E83 found it neutral on fenccausal). raw/fenccausal/"
                "allfeat_pp zero-shot reuse E19's e19_raw/e19_fenccausal/"
                "e19_allfeat_pp; 5 new zero-shot + 8 head runs. e84_zs_* (run_forecast) "
                "+ e84_head_* (tabpfn_head).",
        "runner": "/home/danielco/e84_chain.sh",
        "prefix": "e84_", "scout_prefix": None,
        "models": ["tabpfn", "tabpfn_head"], "expected_arms": 21,
    },
    "E85": {
        "title": "combo + causal per-symbol means (1506f) vs combo (753f) + v2.0",
        "idea": "Does a CAUSAL per-symbol expanding-mean twin of every kitchen-sink "
                "feature help over the kitchen sink? crypto/v2.0-vol-infl-carver-ix-"
                "fillmask-fenccausal (1506f), a clean superset of the E82 combo (753f): "
                "it appends a _symmean column for EVERY combo feature (raw 22->44, "
                "feature_vol_ 22->44, feature_macro_bei_ 12->24, feature_tf_ 15->30, "
                "feature_ix_ 682->1364). Each _symmean is the mean of that symbol's OWN "
                "PAST values (a row never enters its own encoding), leak-safe by "
                "construction -- the fenccausal idea behind IA_01 (best MLP slot), here "
                "on the whole 753f stack instead of the raw 22. Biggest feature set run "
                "to date (1506f). 5 models from scratch, n_iter=50 at the shared "
                "protocol, reported 3-way vs combo (e82_*_combo, the requested baseline) "
                "and v2.0 (e72_*_base), both reused. Shared mommask_core.sh via e85 "
                "(VARIANT=fenc). Numbered E85 -- a concurrent session took E83/E84 "
                "(TabPFN) mid-work (id-collision hazard, cf. E74). Prior: the combo "
                "already trailed raw (interactions drown signal), so 1506f is unlikely "
                "to beat the combo; watch the mlp arm, since fenccausal is the one block "
                "that has genuinely helped an MLP.",
        "runner": "experiments/e85_combo_fenccausal_crypto.sh",
        "prefix": "e85_", "scout_prefix": "e85scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E86": {
        "title": "diverse-primitives bake-off: 7 raw groups x 5 models from scratch, rank + ensembles",
        "idea": "Tune every model class FROM SCRATCH on each primitive feature GROUP "
                "of the raw 22, rank the (model x group) cells, and screen ensembles. "
                "The user's g1_diverse_primitives.json never reached disk, so the "
                "groups are the raw-22 primitive families that partition the 22 "
                "exactly: momentum(2), volatility(4=volatility+bollinger), rsi(2), "
                "sharpe(2), volume(4), size(4=market_cap), price_ma(4=close_). Each is "
                "a --feature-include regex on crypto/v2.0 (one dataset -> one clean "
                "common window). 7 groups x 5 models (ridge, lgb_ktz, catboost_ktz, "
                "rf_ktz, mlp_ktz) = 35 from-scratch runs at the shared protocol "
                "(rolling 130/130, embargo auto=20, 5-fold, val_frac 0.3, seed 42, "
                "n_iter 50, target_binned_return_20). Each cell paired (NW lag 20) vs "
                "the FULL raw 22 (e72_*_base, reused) to show how much of raw's signal "
                "the group carries; ensemble screen over the 35 e86_* arms + book via "
                "report_html afterwards. Groups run concurrently (~2.5h). runner "
                "e86_group_bakeoff_crypto.sh. Prior: no single group should match full "
                "raw; the open question is which model best exploits which primitive "
                "and whether a cross-group/model blend beats the single best.",
        "runner": "experiments/e86_group_bakeoff_crypto.sh",
        "prefix": "e86_", "scout_prefix": "e86scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 35,
    },
    "E87": {
        "title": "SPONGE-cluster bake-off: 10 signed-spectral clusters x 5 models, rank + ensembles",
        "idea": "The E86 idea on the 753f combo, with DATA-DRIVEN feature groups. "
                "Tune every model class from scratch on each of the 10 SPONGE "
                "signed-spectral clusters (Cucuringu et al. AISTATS 2019; signed "
                "Pearson affinity, k=10 by silhouette) of crypto/v2.0-vol-infl-carver-"
                "ix-fillmask (753f), from the user's "
                "data/.../sponge_feature_groups.json. Clusters partition all 753 feats "
                "(sizes 330/176/132/45/24/22/12/4/4/4); each selected via a "
                "--feature-include anchored alternation of its exact feature names. 10 "
                "clusters x 5 models (ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz) = "
                "50 from-scratch runs at the shared protocol (rolling 130/130, embargo "
                "auto=20, 5-fold, seed 42, n_iter 50, target_20). Each cell paired (NW "
                "lag 20) vs the FULL combo (e82_*_combo); ensemble screen over the 50 "
                "e87_* cells + book via report_html afterwards. Clusters run "
                "concurrently (CLUSTER_JOBS=5), ~5-6h. Prior: no single cluster beats "
                "the full combo (which itself trails raw); the questions are which "
                "model best exploits which signed-cluster and whether a cross-cluster "
                "blend diversifies. runner experiments/e87_sponge_bakeoff.py.",
        "runner": "experiments/e87_sponge_bakeoff.py",
        "prefix": "e87_", "scout_prefix": "e87scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 50,
    },
    "E88": {
        "title": "SPONGE-cluster bake-off on the 1506f causal-combo: 25 clusters x 5 models, rank + ensembles",
        "idea": "The E87 idea on the BIGGER causal-combo. Tune every model class "
                "from scratch on each of the 25 SPONGE signed-spectral clusters "
                "(Cucuringu et al. AISTATS 2019; signed Pearson affinity, k=25 by "
                "silhouette) of crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal "
                "(1506f), from the user's data/.../sponge_feature_groups.json. This "
                "is the 1506f twin of E87 (753f, 10 clusters, WITHOUT the causal "
                "per-symbol means): the feature universe here adds a leak-safe "
                "causal expanding per-symbol mean (_symmean) for every combo "
                "feature -- the fenccausal idea behind IA_01, the best MLP slot -- "
                "so each cluster mixes a primitive with its own symbol-level twin. "
                "The 25 clusters partition all 1506 feats EXACTLY (0 missing, 0 "
                "overlap; sizes 330/308/198/165/119/66/48/46/45/44/23/22/22/22/16/"
                "8/8/4/3/3/2/1/1/1/1), each selected via a --feature-include "
                "anchored alternation of its exact feature names. 7 of the 25 -- "
                "the pure day-level break-even-inflation clusters (sponge_02/03/06/"
                "14/19/21/24, 12 raw feature_macro_bei_* feats) -- are EXCLUDED "
                "from the fitted roster because a BEI series is one macro value per "
                "date broadcast to every coin, so those clusters are CONSTANT "
                "within each date and the within-date numerai_corr is undefined "
                "(ridge/lgb/catboost/rf cannot score a candidate; mlp fits only "
                "init noise ~0). The runner detects this data-drivenly (within-date "
                "nunique on a ~30-date sample) and NAMES the exclusion in the "
                "report; it reproduces the scout failures exactly and matches the "
                "E80/E81 BEI-inert prior. Leaves 18 active clusters x 5 models "
                "(ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz) = 90 from-scratch "
                "runs at the shared protocol (rolling 130/130, embargo auto=20, "
                "5-fold, val_frac 0.3, seed 42, n_iter 50, target_20) -- identical "
                "to E85/E86/E87 so every cell is comparable to the book. Each cell "
                "paired (NW lag 20) vs the FULL 1506f combo (e85_*_fenc, reused); "
                "ensemble screen over the 90 e88_* cells + book via report_html "
                "afterwards. Clusters run concurrently (CLUSTER_JOBS=5). runner "
                "experiments/e88_sponge_fenccausal_bakeoff_crypto.py. Prior: no "
                "single cluster beats the full combo (which itself trails raw); the "
                "questions are which model best exploits which signed cluster and "
                "whether a cross-cluster/model blend diversifies.",
        "runner": "experiments/e88_sponge_fenccausal_bakeoff_crypto.py",
        "prefix": "e88_", "scout_prefix": "e88scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 90,
    },
    "E89": {
        "title": "SPONGE-cluster bake-off on the 1526f fenccausal-pairs combo: 25 clusters x 5 models",
        "idea": "The E87 bake-off on the biggest dataset: crypto/v2.0-vol-infl-carver-"
                "ix-fillmask-fenccausal-pairs (1526f = the 1506f fenccausal combo + 20 "
                "pair features), with 25 SPONGE signed-spectral clusters from its own "
                "sponge_feature_groups.json (k=25, partitioning all 1526; sizes "
                "406/220/216/182/154/.../1). Sibling of E87 (753f, 10 clusters) and the "
                "concurrent E88 (1506f fenccausal, no pairs, 18 clusters). 25 clusters "
                "x 5 models = 125 from-scratch runs at the shared protocol (rolling "
                "130/130, embargo auto=20, 5-fold, seed 42, n_iter 50, target_20), each "
                "selected via a --feature-include anchored alternation of its exact "
                "feature names. Each cell paired (NW lag 20) vs raw v2.0 (e72_*_base, "
                "the deployed baseline -- no full-1526f run exists, so raw is the "
                "anchor); ensemble screen over the 125 e89_* cells + book afterwards. "
                "Reuses the env-driven core e87_sponge_bakeoff.py via the e89 wrapper; "
                "CLUSTER_JOBS=4 to coexist with E87. Prior: no single cluster beats raw "
                "(the full stack already trails raw per E82/E85).",
        "runner": "experiments/e89_sponge_bakeoff_fenccausal_pairs.sh",
        "prefix": "e89_", "scout_prefix": "e89scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 125,
    },
    "E90": {
        "title": "reversal block vs raw v2.0: crypto/v2.0-rev (36f) vs v2.0 (22f)",
        "idea": "Classic block-vs-raw test on a reversal/mean-reversion block. "
                "crypto/v2.0-rev = raw 22 + 14 feature_rev_ features: short-term "
                "reversal (str_1/7/14d), MAX/lottery effect (max1/max5_28d), "
                "idiosyncratic vol (ivol_28d), return skew (skew_28d), Amihud "
                "illiquidity (amihud_28d), and mean-reversion oscillators (rsi_14d, "
                "bbz_20d, stoch_14d, cci_20d, dist_ma_20/50d). Clean superset of raw, "
                "all per-symbol (cross-sectional, not day-level -> scoreable). 5 models "
                "from scratch, n_iter=50 at the shared protocol, reported 3-way vs raw "
                "v2.0 (e72_*_base, the requested comparison) and vs the vol block "
                "(e74_*_vol, raw+22 vol estimators) as a free secondary -- does "
                "reversal beat the leading prior raw add-on (vol was +0.0053 CORR on "
                "catboost, ns)? Reuses the generic 3-way runner mommask_core.sh with "
                "VARIANT=rev. Prior: reversal signals overlap the raw rsi/bollinger, so "
                "null-or-marginal like every block tested; the question is whether "
                "short-horizon reversal adds anything the raw 22 lack.",
        "runner": "experiments/e90_rev_vs_raw_crypto.sh",
        "prefix": "e90_", "scout_prefix": "e90scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E91": {
        "title": "cointegration block vs raw v2.0: crypto/v2.0-coint-fillmask (26f) vs v2.0 (22f)",
        "idea": "Block-vs-raw test on a cointegration / error-correction (stat-arb) "
                "block. crypto/v2.0-coint-fillmask = raw 22 + 4 feature_coint_ "
                "features: residual z-score at 60d/120d (distance from the "
                "cointegrating relationship), mean-reversion half-life (120d), and the "
                "ADF stationarity stat (120d). fillmask = missing cells pre-filled to "
                "0.5 (== the loader's own fill, so as-loaded == a raw-coint dataset). "
                "Clean superset of raw, per-symbol/binned (~5 values/date -> scoreable, "
                "not day-level degenerate). 5 models from scratch, n_iter=50 at the "
                "shared protocol, reported 3-way vs raw v2.0 (e72_*_base, the requested "
                "comparison) and vs the vol block (e74_*_vol, the leading prior raw "
                "add-on -- same secondary as E90, so reversal and cointegration are "
                "comparable). Reuses mommask_core.sh with VARIANT=coint. Prior: "
                "null-or-marginal like every block; the question is whether a mean-"
                "reversion/stat-arb signal adds what momentum/vol/reversal did not.",
        "runner": "experiments/e91_coint_vs_raw_crypto.sh",
        "prefix": "e91_", "scout_prefix": "e91scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E92": {
        "title": "BTC-beta block vs raw: crypto/v2.0-beta (raw 22 + 3 BTC-beta) vs v2.0, 5 models from scratch",
        "idea": "Does a coin's rolling market beta to BTC help the raw panel? "
                "crypto/v2.0-beta is a STRICT SUPERSET of crypto/v2.0 -- the raw 22 "
                "plus exactly 3 new features, feature_beta_btc_30d/_60d/_90d (each "
                "coin's rolling beta to Bitcoin over 30/60/90 days). So this is a "
                "clean leave-the-rest-fixed ablation of the BTC-beta block: the only "
                "difference between an e92_<m>_beta arm and its reused raw baseline "
                "e72_<m>_base is those 3 columns. 5 models (ridge, lgb_ktz, "
                "catboost_ktz, rf_ktz, mlp_ktz) tuned FROM SCRATCH on v2.0-beta at "
                "the shared protocol (rolling 130/130, embargo auto=20, 5-fold, "
                "val_frac 0.3, seed 42, n_iter 50, target_binned_return_20, era 1) "
                "-- identical to e72_*_base, so each cell pairs validly (NW lag 20) "
                "against its raw twin and is comparable to the book. The raw side is "
                "REUSED (e72_*_base), not re-run; only the 5 beta arms are fit (~1-3h, "
                "25-feature sets are cheap). runner "
                "experiments/e92_beta_vs_raw_crypto.sh. Prior: BTC-beta is a "
                "market/regime signal with ~zero standalone within-date IC (like the "
                "macro/inflation blocks E69/E80/E81), so expect null-to-marginal on "
                "ridge and a possible small tree/MLP interaction lift at best.",
        "runner": "experiments/e92_beta_vs_raw_crypto.sh",
        "prefix": "e92_", "scout_prefix": "e92scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E93": {
        "title": "betting-against-beta (short-constraint) block vs raw: crypto/v2.0-bab (31f) vs v2.0",
        "idea": "Operationalizes the Miller/Hong-Sraer 'speculative betas' anomaly for "
                "a short-constrained crypto market. crypto/v2.0-bab = raw 22 + 9 BAB "
                "features (3 windows 30/60/90d x {beta_btc, beta x Amihud-illiquidity, "
                "beta x realised-vol}). The interactions are the short-constraint / "
                "disagreement conditioning: high-beta coins that are ALSO hard-to-short "
                "(illiquid) and high-disagreement (high vol) are overpriced by optimists "
                "-> expected LOW forward return. Products formed from per-date centered "
                "percentiles (robust to Amihud skew), then binned; per-symbol, "
                "point-in-time (lag 1), clean superset of raw. Distinct from the "
                "concurrent E92 (crypto/v2.0-beta, 25f, beta-only) -- E93 carries the "
                "short-constraint INTERACTIONS, the novel content. 5 models from "
                "scratch, n_iter=50 at the shared protocol, reported 3-way vs raw v2.0 "
                "(e72_*_base, requested) and vs the vol block (e74_*_vol, same secondary "
                "as E90/E91). Reuses mommask_core.sh with VARIANT=bab. Prior: beta and "
                "its interactions are collinear with the raw volatility/volume/"
                "market_cap, so the honest risk is redundancy; the test is whether the "
                "short-constraint interaction carries INCREMENTAL signal, and whether "
                "plain-beta IC is negative as the theory predicts. Builders "
                "bab_features.py + build_bab.py.",
        "runner": "experiments/e93_bab_vs_raw_crypto.sh",
        "prefix": "e93_", "scout_prefix": "e93scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E94": {
        "title": "futures-basis & funding block vs raw: crypto/v2.0-basis-fillmask (32f) vs v2.0",
        "idea": "The FIRST genuinely new data class in this program -- perp funding + "
                "spot-futures basis, not a re-transform of the 22. "
                "crypto/v2.0-basis-fillmask = raw 22 + 10 features: feature_basis_pct, "
                "basis_ma_20d, basis_z_20d, basis_z_60d, basis_chg_5d/20d, "
                "feature_funding_rate, funding_ma_7d, funding_ma_30d, funding_z_30d. "
                "This is the DIRECT short-constraint / carry measure the E93 (BAB) "
                "post-mortem flagged as the real unlock: persistently positive funding "
                "= crowded longs / costly shorts, where high-beta overpricing should "
                "concentrate; a coin with no listed perp has no funding and is "
                "neutral-filled, and that absence is itself the hard-to-short signal. "
                "Per-symbol, binned (~5/date, not degenerate), point-in-time "
                "(fillmask = loader fill). 5 models from scratch, n_iter=50 at the "
                "shared protocol, reported 3-way vs raw v2.0 (e72_*_base, requested) "
                "and vs the vol block (e74_*_vol, same secondary as E90/E91/E93). "
                "Reuses mommask_core.sh with VARIANT=basis. Built on a user request to "
                "run whenever crypto/v2.0-basis-fillmask finished building. This is the "
                "experiment most likely to add over raw, since funding/basis is "
                "information the price-derived 22 do not contain.",
        "runner": "experiments/e94_basis_vs_raw_crypto.sh",
        "prefix": "e94_", "scout_prefix": "e94scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E95": {
        "title": "coint+rev+basis stack vs raw: crypto/v2.0-coint-rev-basis-fillmask (50f) vs v2.0",
        "idea": "Does the COMBINED stat-arb / microstructure stack beat raw where "
                "each block alone did not? crypto/v2.0-coint-rev-basis-fillmask is a "
                "STRICT SUPERSET of crypto/v2.0 -- the raw 22 plus 28 features in "
                "four families: coint (4: ADF-120d, half-life-120d, residual-z at "
                "60d/120d -- cointegration / error-correction stat-arb), rev (14: "
                "short-term reversal -- rsi/stoch/cci/amihud/ivol/max1/max5/skew/"
                "str_{1,7,14}d/dist_ma_{20,50}d/bbz_20d), basis (6: futures-spot "
                "basis pct/ma/chg/z) and funding (4: perp funding rate/ma/z). "
                "fillmask = missing cells pre-filled to 0.5 (== the loader's own "
                "fill), so as loaded it is a clean raw+28 superset. The only "
                "difference between an e95_<m>_crb arm and its reused raw baseline "
                "e72_<m>_base is those 28 columns. This is the COMBINATION of the "
                "individually-tested blocks E91 (coint, +4), E94 (basis/funding) and "
                "E90 (reversal). 5 models (ridge, lgb_ktz, catboost_ktz, rf_ktz, "
                "mlp_ktz) tuned FROM SCRATCH at the shared protocol (rolling 130/130, "
                "embargo auto=20, 5-fold, val_frac 0.3, seed 42, n_iter 50, "
                "target_binned_return_20, era 1) -- identical to e72_*_base, so each "
                "cell pairs validly (NW lag 20) vs its raw twin and is comparable to "
                "the book. Raw side REUSED (e72_*_base), not re-run; 5 arms to fit. "
                "runner experiments/e95_coint_rev_basis_vs_raw_crypto.sh. Prior: each "
                "constituent block was null-or-marginal vs raw (E90/E91/E94); the "
                "open question is whether stacking the reversal + stat-arb + futures "
                "microstructure signals together clears the bar -- or, like the big "
                "engineered combos (E82/E85), a wide add-on just dilutes the raw "
                "panel (cf. the BTC-beta block E92, which HURT at |t|>2).",
        "runner": "experiments/e95_coint_rev_basis_vs_raw_crypto.sh",
        "prefix": "e95_", "scout_prefix": "e95scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E96": {
        "title": "coint+rev+basis+mom+carver stack vs raw: crypto/v2.0-coint-rev-basis-mom-carver-fillmask (82f) vs v2.0",
        "idea": "The widest stat-arb + trend add-on: does E95's coint+rev+basis "
                "(50f) PLUS the momentum and Carver trend blocks beat raw? "
                "crypto/v2.0-coint-rev-basis-mom-carver-fillmask is a STRICT SUPERSET "
                "of crypto/v2.0 -- raw 22 + 60 features in six families: coint (4), "
                "rev (14, short-term reversal), basis (6, futures-spot), funding (4, "
                "perp), mom (17, literature momentum incl. Tzotchev robust trend; cf. "
                "E63/E66/E67) and tf (15, Carver/Clenow trend-following; cf. E72). "
                "fillmask == the loader's own 0.5 fill, so as loaded it is a clean "
                "raw+60 superset. The only difference between an e96_<m>_crbmc arm "
                "and its reused raw baseline e72_<m>_base is those 60 columns. 5 "
                "models (ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz) tuned FROM "
                "SCRATCH at the shared protocol (rolling 130/130, embargo auto=20, "
                "5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20, "
                "era 1) -- identical to e72_*_base, so each cell pairs validly (NW "
                "lag 20) vs its raw twin and is comparable to the book. Raw side "
                "REUSED (e72_*_base), not re-run; 5 arms to fit. runner "
                "experiments/e96_coint_rev_basis_mom_carver_vs_raw_crypto.sh. Prior: "
                "the individual blocks (coint E91, basis E94, reversal E90, momentum "
                "E66/E67, carver E72) were each null-or-marginal vs raw; a 60-column "
                "add-on is squarely in the width where ridge gets diluted (E82/E85) "
                "and E92/E95 found null-to-negative -- watch whether the momentum + "
                "trend blocks rescue it on the tree/MLP side or sink it further.",
        "runner": "experiments/e96_coint_rev_basis_mom_carver_vs_raw_crypto.sh",
        "prefix": "e96_", "scout_prefix": "e96scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E97": {
        "title": "crbmc + CURATED interactions vs raw (and vs xfull E98): v2.0-crbmc-xcurated-fillmask (98f)",
        "idea": "Curated-vs-full interaction expansion on the crbmc combined-signal "
                "base. crypto/v2.0-crbmc-xcurated-fillmask (98f) = crbmc base (raw 22 + "
                "coint 4 + rev 14 + basis 6 + funding 4 + mom 17 + carver/tf 15 = 82) + "
                "16 CURATED pairwise interactions (feature_x2_). 5 models from scratch, "
                "n_iter=50 at the shared protocol, reported 3-way vs raw v2.0 "
                "(e72_*_base, requested) and vs crbmc-xfull (e98_*_xfull, 2856f) -- so "
                "curated (E97) and full (E98) each pair vs raw AND vs each other. "
                "Reuses mommask_core.sh VARIANT=xcur. The curated-vs-full cell may be "
                "blank at report time if E98 is unfinished; regenerate once both land. "
                "Prior: a curated interaction set should hold up where the full "
                "expansion (E98) drowns the linear/NN models.",
        "runner": "experiments/e97_crbmc_xcurated_crypto.sh",
        "prefix": "e97_", "scout_prefix": "e97scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E98": {
        "title": "crbmc + FULL interactions vs raw (and vs xcurated E97): v2.0-crbmc-xfull-fillmask (2856f)",
        "idea": "The full-interaction twin of E97. crypto/v2.0-crbmc-xfull-fillmask "
                "(2856f) = crbmc base (82) + 2774 FULL pairwise interactions "
                "(feature_x2_). Biggest feature set run to date. 5 models from scratch, "
                "n_iter=50 at the shared protocol, reported 3-way vs raw v2.0 "
                "(e72_*_base) and vs crbmc-xcurated (e97_*_xcur, 98f). Reuses "
                "mommask_core.sh VARIANT=xfull; 5 arms sequential, budget ~8-10h. "
                "Prior: 2774 interaction columns should DROWN ridge/mlp (cf. E70 925f "
                "t<-2.7, E85 1506f) while trees stay ~flat; the E97<->E98 contrast "
                "measures whether curation beats brute-force expansion.",
        "runner": "experiments/e98_crbmc_xfull_crypto.sh",
        "prefix": "e98_", "scout_prefix": "e98scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E99": {
        "title": "coint+rev+basis+macro stack vs raw: v2.0-coint-rev-basis-macro-fillmask (68f) vs v2.0",
        "idea": "Pools the per-symbol stat-arb/carry signals with the day-level macro "
                "regime block. crypto/v2.0-coint-rev-basis-macro-fillmask (68f, clean "
                "superset of raw) = raw 22 + coint 4 (error-correction) + rev 14 "
                "(reversal) + basis 6 + funding 4 (perp carry / short-constraint) + "
                "macro 18 (gold/silver/S&P/DXY/US-yield-curve, day-level). Ridge is "
                "inert on the 18 macro cols (constant within date) and uses only the "
                "per-symbol 28; trees/MLP can regime-gate on macro. 5 models from "
                "scratch, n_iter=50 at the shared protocol, reported 3-way vs raw v2.0 "
                "(e72_*_base, requested) and vs the vol block (e74_*_vol, same secondary "
                "as E90/E91/E93/E94). Reuses mommask_core.sh with VARIANT=crbm. Sibling "
                "of the concurrent E95 (coint+rev+basis) and E96 (coint+rev+basis+mom+"
                "carver) -- this swaps in macro+funding. Prior: the per-symbol blocks "
                "were individually null-to-negative (E90/E91/E94) and macro is "
                "regime-only, so expect null-to-marginal vs raw.",
        "runner": "experiments/e99_coint_rev_basis_macro_vs_raw_crypto.sh",
        "prefix": "e99_", "scout_prefix": "e99scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E100": {
        "title": "crbmc interactions from RAW products vs BINNED products (E100 xraw vs E101 xbin) + raw",
        "idea": "A/B on interaction CONSTRUCTION. crypto/v2.0-crbmc-xraw-fillmask "
                "(1469f = crbmc base 82 + 1387 feature_x2_) forms each interaction as "
                "the product of the RAW (continuous, pre-bin) parent values, THEN "
                "rank-bins. Same 1469 columns as xbin (E101) but different values -- "
                "raw-product vs binned-product of a pair are only weakly/negatively "
                "correlated (-0.16..-0.49), since binning parents first discards the "
                "magnitude the product needs. 5 models from scratch, n_iter=50 at the "
                "shared protocol, reported 3-way vs crbmc-xbin (e101_*_xbin, the "
                "headline A/B) and vs raw v2.0 (e72_*_base). Reuses mommask_core.sh "
                "VARIANT=xraw; 1469f -> ~5h. The xraw<->xbin cell may be blank at report "
                "time if the twin is unfinished; regenerate once both land. Tests "
                "whether forming interactions before or after binning matters.",
        "runner": "experiments/e100_crbmc_xraw_crypto.sh",
        "prefix": "e100_", "scout_prefix": "e100scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E101": {
        "title": "crbmc interactions from BINNED products vs RAW products (E101 xbin vs E100 xraw) + raw",
        "idea": "A/B twin of E100. crypto/v2.0-crbmc-xbin-fillmask (1469f) forms each "
                "interaction as the product of the already-BINNED {0,.25,.5,.75,1} "
                "parent values (a product of cross-sectional RANKS, magnitude "
                "discarded), THEN rank-bins. Same 1469 columns as xraw, different "
                "values (corr to xraw -0.16..-0.49). 5 models from scratch, n_iter=50 "
                "at the shared protocol, reported 3-way vs crbmc-xraw (e100_*_xraw) and "
                "vs raw v2.0 (e72_*_base). Reuses mommask_core.sh VARIANT=xbin; ~5h. "
                "Prior: both are large-interaction sets that should drown ridge/mlp "
                "(cf. E85/E98); the question is whether raw-product or rank-product "
                "interactions carry more usable signal for the trees.",
        "runner": "experiments/e101_crbmc_xbin_crypto.sh",
        "prefix": "e101_", "scout_prefix": "e101scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E102": {
        "title": "LoRA fine-tune of TabPFN with a per-date cross-sectional CORR loss",
        "idea": "The one fine-tune cell E44/E45/E83 never hit on all three axes at "
                "once -- the rank-aware, cross-sectional, prior-preserving scheme. "
                "(a) RANK-ALIGNED: differentiable cross-sectional CORR loss (1 - "
                "pearson, the numerai_corr stand-in). (b) CROSS-SECTIONAL/listwise: "
                "each training episode is ONE date's full cross-section of coins "
                "(context = that coin-set's fenccausal history up to the T-20 leak "
                "wall, query = the date-T cross-section), so the loss is over the "
                "date's ranking, not pointwise rows. (c) PRIOR-PRESERVING: only LoRA "
                "adapters (r=8) on the attention q/k/v/out projections train -- "
                "294,912 params of 11.4M, backbone frozen -- vs E44/E45 full FT that "
                "corrupted the prior (-0.027 t-3.02) and E83's pointwise head "
                "(neutral). Readout = bar-dist mean (what eval uses); LoRA MERGED "
                "into the backbone and saved, so eval is the ORDINARY E19 "
                "walk-forward via model_path. src/analysis/tabpfn_lora.py, leak-safe "
                "(<=2024-06-27). Paired NW vs E19. Prior: likely neutral (zero-shot "
                "already reads the cross-section via pre-binned features; E83 head "
                "neutral; Chronos-2 group attention null) -- but the definitive test "
                "of objective-alignment. Numbered E102 (concurrent sessions reached "
                "E101).",
        "runner": "/home/danielco/e102_chain.sh",
        "prefix": "e102_", "scout_prefix": None,
        "models": ["tabpfn"], "expected_arms": 1,
    },
    "E103": {
        "title": "E102 LoRA fine-tune per Tier-1 (crypto-derived-structural) feature set",
        "idea": "E102's LoRA scheme (per-date cross-sectional CORR loss, r=8 adapters "
                "on q/k/v/out, backbone frozen, merged then evaluated as the ordinary "
                "E19 walk-forward) applied to each Tier-1 feature set -- the ones built "
                "by STRUCTURAL processing of the raw 22, no external data. Seven arms: "
                "raw v2.0 (22f), lag3 (44f), cumdiff3 (69f), ar (25f), interact (36f), "
                "tenc (23f), aetuned (30f); fenccausal (44f) is e102_lora itself, "
                "reused as the 8th screen member. Dropped: allfeat 812f (TabPFN "
                "stall/OOM hazard on 8GB, froze E84), e10stack 2f (degenerate), and the "
                "cumulative-lags family (cumlag2/3 OOM the 8GB cards at 66-88f; its "
                "theme is covered by lag3 + cumdiff3). Trained on "
                "deepfive (2x idle GPU, shared /home). THEN: (2) screen ensembles of "
                "the 8 LoRA arms for one that beats the top single arm; (3) screen "
                "ensembles of the winners with the submitted book + e102. Hypothesis: "
                "each single arm underperforms E19 zero-shot (as e102 did), but the "
                "LoRA objective decorrelates them, so a blend -- and a blend with the "
                "book -- may add a risk (drawdown/calmar) edge as the e102x book 4-way "
                "did, not a payout-level one. Leak-safe (<=2024-06-27), paired NW.",
        "runner": "/home/danielco/e103_chain.sh",
        "prefix": "e103_", "scout_prefix": None,
        "models": ["tabpfn"], "expected_arms": 7,
    },
    "E104": {
        "title": "Day-indexed minibatches vs row-shuffled minibatches, on the sklearn-init torch MLP",
        "idea": "Does it matter HOW a training window is cut into minibatches? One "
                "pinned model -- the from-scratch torch MLP with sklearn's exact "
                "initial weights that tracks mlp/e33_fenc_a64_32_16_identity_lre3_al100 "
                "(identity (64,32,16), alpha 100, lr 1e-3, fenccausal 44f, rolling "
                "130/130, embargo 20) -- and only the batch scheme varies. row "
                "(baseline) shuffles the ROWS and slices 1024; day3 shuffles the DAYS "
                "and a batch is every stock of 3 shuffled days (~900 rows, "
                "size-matched); day1 is one day per batch. A batch of whole days has "
                "no fixed row count (the panel holds 241..300 stocks a day), so two "
                "things are decided, not left to the arithmetic: the loss gives each "
                "DAY 1/k (a thin day is not out-voted; equals the row mean at equal "
                "counts), and the L2 term -- sklearn divides it by rows-in-batch, so "
                "alpha's strength would move with breadth -- is pinned to alpha/1024. "
                "Controls: rowfix (rows, penalty pinned) and day3lit/day1lit (days, "
                "penalty / actual rows) separate 'batches are days' from 'the penalty "
                "moved'; row900 and row300 (rows, at the day arms' batch size and so "
                "their step count, added once day1 came back constant on every seed) "
                "separate 'batches are days' from 'batches are smaller'. "
                "8 arms x 10 seeds (42..51) = 80 runs, because this arm is "
                "numerically degenerate (seeds range from a constant predictor to "
                "CORR 0.13) and one seed says nothing; the test is paired NW lag 20 "
                "on the SEED-MEAN per-period delta. Gate: row s42 reproduces "
                "torch_scratch/e33repl_scratch_SKINIT_* bit for bit. Prior: a null -- "
                "the seed spread should swamp the batch scheme at alpha 100. "
                "Second pass (E104_TAG=s100): row, row900 and day3 extended to 100 "
                "seeds (42..141, the same seeds in every arm) for boxplots of CORR, "
                "MMC and dd(pay) -- 270 more runs, so 80 + 270 = 350 on disk.",
        "runner": "experiments/e104_daybatch_torch_mlp_crypto.py",
        "prefix": "e104_", "scout_prefix": None,
        "models": ["torch_scratch"], "expected_arms": 350,
    },
    "E105": {
        "title": "Learning-rate schedulers on the day3 batch scheme (torch MLP, sklearn init)",
        "idea": "E104's day3 arm unchanged -- torch from scratch, sklearn-exact init, "
                "identity (64,32,16), alpha 100, a batch = every stock of 3 shuffled "
                "days, L2 pinned to alpha/1024 -- with ONLY the learning rate over "
                "training varied. const (1e-3 throughout) is the baseline and is gated "
                "bit for bit against e104_day3 on every seed. Seven of torch's own "
                "schedulers, each peaking at 1e-3 over a 60-epoch horizon (constant-lr "
                "training stops at ~59, so a schedule laid over the 200-epoch cap would "
                "barely have moved): StepLR x0.5/15, ExponentialLR 0.955, "
                "CosineAnnealingLR, 5-epoch warm-up + cosine, cosine warm restarts "
                "T_0 15, OneCycleLR, ReduceLROnPlateau x0.5 patience 3. Every schedule "
                "also lowers the AVERAGE rate, so two controls hold it constant at "
                "5e-4 and 2.5e-4 -- they separate the schedule's shape from 'less "
                "learning rate', as row900 separated batch composition from batch size "
                "in E104. 10 arms x 100 seeds (42..141, shared by every arm) = 1000 "
                "runs; the arm is numerically degenerate, so the distribution across "
                "seeds is the result. Paired by seed against const. Prior: a lower "
                "rate late in training slows the decay toward the origin, so more "
                "signal survives -- and the constant-lr controls should reproduce "
                "most of whatever a schedule does.",
        "runner": "experiments/e105_lr_schedulers_daybatch_crypto.py",
        "prefix": "e105_", "scout_prefix": None,
        "models": ["torch_scratch"], "expected_arms": 1000,
    },
    "E106": {
        "title": "Residualise the -CORR network on the meta model (no training)",
        "idea": "Take the from-scratch -CORR network's stitched held-out predictions "
                "(identity 64-32-16, alpha 0, epoch 500, 12 full-layout refits, seeds "
                "42/43) and, per day on the 547 meta days, move them off the meta model "
                "in the space MMC is computed in: r = gauss(rank(p)) - s * beta * "
                "gauss(rank(m)), beta = p.m/m.m, submitted as rank(r). s sweeps 0..1 in "
                "21 steps. The same-day meta model is published after the round closes, "
                "so residualising on it is an ORACLE; four deployable arms use the "
                "latest meta model at least 1/5/20/30 days old. Every arm is scored "
                "against the same-day meta model, paired against s = 0 of its seed (NW "
                "lag 20). Gate: s = 0 reproduces the source per-period CORR/MMC to "
                "1e-12. A result is a LAGGED arm whose payout gain clears |t| >= 2 on "
                "both seeds. 2 seeds x (1 + 5 variants x 20 strengths) = 202 runs.",
        "runner": "experiments/e106_meta_residualise_crypto.py",
        "prefix": "e106_", "scout_prefix": None,
        "models": ["torch_scratch"], "expected_arms": 202,
    },
    "E107": {
        "title": "Residualise every model on a 5-day-old meta model, s = 0.5 (no training)",
        "idea": "E106's deployable transform applied to every model at once: the walk-forward "
                "ridge on fenccausal, the from-scratch MSE and -CORR networks (seeds 42 and 43), "
                "and every slot in the registry scored on the 20-day label (SACI_03 and "
                "CURUPIRA_05_60 are 60-day and left out). Per day, rank(gauss(rank(p)) - 0.5 * "
                "beta * gauss(rank(m))), m = the latest meta model at least 5 days old, scored "
                "against the same-day meta model on the 470 days every model shares "
                "(2024-07-25..2026-05-13). Gate: s = 0 reproduces every source's stored "
                "per-period CORR and MMC. The question: do all models converge to the same "
                "residualised payout? Converge = the cross-model spread collapses and the share "
                "of model pairs whose paired payout difference clears |t| >= 2 falls to the "
                "level of two seeds of one network. If so, model choice is settled and the work "
                "moves to the residualisation / portfolio layer.",
        "runner": "experiments/e107_residualise_everything_crypto.py",
        "prefix": "e107_", "scout_prefix": None,
        "models": ["meta_resid"], "expected_arms": 38,
    },
    "E108": {
        "title": "all momentum/trend families in one dataset vs raw: v2.0-allmom-fillmask (57f) vs v2.0",
        "idea": "First dataset to hold EVERY momentum/trend family at once. "
                "crypto/v2.0-allmom-fillmask (57f), assembled with build_combo (union "
                "of already-built, already-binned blocks onto the raw v2.0 spine -- no "
                "recomputation, leak-safe): raw 22 (incl the 2 base feature_momentum_) "
                "+ feature_mom_ 17 (litmom: cumret horizons, 12-1, MAX, vol-scaled, "
                "Tzotchev robust trend) + feature_xsmom_ 3 (cross-sectional momentum "
                "mean/var/skew, day-level) + feature_tf_ 15 (Carver/Clenow trend: "
                "EWMAC, breakout, Bollinger-z, Clenow). Coverage: mom 64.9%, xsmom "
                "94.6%, tf 78.7% (rest neutral-filled). 5 models from scratch, "
                "n_iter=50 at the shared protocol, reported 3-way (CORR/MMC/payout/"
                "c.sharpe/calmar/n) vs raw v2.0 (e72_*_base, requested) and vs the vol "
                "block (e74_*_vol). Reuses mommask_core.sh VARIANT=allmom. Prior: "
                "momentum was null-to-negative vs raw (E64-E67); pooling every variant "
                "lets a tree pick among them, but the raw 22 already absorb most of it.",
        "runner": "experiments/e108_allmom_vs_raw_crypto.sh",
        "prefix": "e108_", "scout_prefix": "e108scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 5,
    },
    "E109": {
        "title": "foundation-model representations as features: allfeat_pp / fenccausal / raw, with vs without",
        "idea": "The 9 representations of experiments/features-foundation.md built CAUSALLY and merged as "
                "one 41-column feature_fm_ block onto three bases (crypto/v2.0-fmrep raw 22+41, "
                "crypto/v2.0-fenccausal-fmrep 44+41, crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmrep "
                "allfeat_pp 812+41). TabPFN v2 on fenccausal: point preds, std/IQR/skew, 5 bin "
                "probabilities, estimator disagreement, 8 embedding PCs; TabPFN per block (tenc, interact, "
                "size, price, raw); 30/390-date context variants; TimesFM + Chronos-2 on each coin's log "
                "close <= D-1 (20-day q50/spread/skew) and 8 Chronos-2 embedding PCs. TabPFN context ends "
                "20 dates before each 20-date refresh block; embedding bases fitted on a burn-in that ends "
                "before the first training date; every column rank-binned per date. 3 bases x 5 models x "
                "{with, without}; without = same dataset with --feature-exclude '^feature_fm_'. 1,701-date "
                "spine to 2026-07-08, 510 scored periods. Not live-wired.",
        "runner": "experiments/e109_fmrep_crypto.sh",
        "prefix": "e109_", "scout_prefix": "e109scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 30,
    },
    "E110": {
        "title": "all-momentum dataset vs raw on the 60-day target (allmom60 vs raw60)",
        "idea": "The 60-day twin of E108. Two lanes, 5 models each tuned from scratch "
                "on target_binned_return_60 (embargo auto -> 60): raw60 on crypto/v2.0 "
                "(22f) -- the matched 60-day baseline, which did not exist for 4 of the "
                "5 model classes -- and allmom60 on crypto/v2.0-allmom-fillmask (57f = "
                "raw + feature_mom_ 17 + feature_xsmom_ 3 + feature_tf_ 15). Shared "
                "protocol otherwise (rolling 130/130, 5-fold, seed 42, n_iter 50). "
                "Report per model: raw60/allmom60 rows (CORR MMC payout c.sharpe "
                "calmar n, payout+calmar per period) and allmom60 vs raw60 paired (NW "
                "lag 20). 60-day numbers are NOT comparable to 20-day tables. Feeds "
                "E111 (meta residualisation + ensembles), which also rescores these "
                "forecasts on the payable 20-day label. Prior: momentum was null vs "
                "raw at 20d (E64-E67); slower signals like 12-1 / Carver trend might "
                "fit a 60-day horizon better.",
        "runner": "experiments/e110_allmom_t60_crypto.sh",
        "prefix": "e110_", "scout_prefix": "e110scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 10,
    },
    "E111": {
        "title": "meta residualisation + ensembles of the all-momentum models (20d and 60d)",
        "idea": "No training. E107's deployable meta residualisation (s=0.5, meta >= "
                "5 days old, imported from e107 so the arithmetic is identical; gate: "
                "s=0 reproduces the repo scorer to 1e-9) applied to the held-out "
                "forecasts of allmom20 (E108), raw20 (E72 base), allmom60 + raw60 "
                "(E110) and the book CU_SA_IA_BO_02; 60-day forecasts scored in-target "
                "AND on the payable 20-day label (joined from crypto/v2.0). Part B: "
                "every 2/3/4-member equal-weight rank-average of the 10 residualised "
                "allmom models (5 @20d + 5 @60d-on-20d), paired (NW lag 20) vs best "
                "member and vs the residualised book. Part C: residualised book + best "
                "ensemble at 10/20/30%. Report carries calmar. 20d-only validation "
                "(2026-10-21): residualisation lifts all (book +0.0082 t+4.7) but "
                "allmom trails raw after it too; no allmom ensemble beats its best "
                "member or the book.",
        "runner": "experiments/e111_allmom_resid_ensembles_crypto.py",
        "prefix": "e111_", "scout_prefix": None,
        "models": ["meta_resid"], "expected_arms": 21,
    },
    "E112": {
        "title": "symbol encodings on top of all-momentum: fenccausal / tenc / rolling means / all vs allmom",
        "idea": "Which of the three per-symbol encodings adds to "
                "crypto/v2.0-allmom-fillmask (57f, E108) when stacked on it? Four "
                "supersets, each differing from the base ONLY by the encoding: "
                "-fenccausal (114f, +57 feature_<f>_symmean = causal expanding mean "
                "of each feature over the symbol's strictly earlier dates, shrunk "
                "m=50), -tenc (58f, +feature_sym_tenc = mean of the symbol's own "
                "20d target over resolved dates <= t-20, shrunk m=50), -roll "
                "(228f, +171 feature_<f>_rm{5,20,60} = per-symbol trailing means "
                "incl today), -all (286f, all three). 4 datasets x 5 models "
                "(ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz) from scratch at the "
                "shared protocol (rolling 130/130, embargo auto=20, 5-fold, val_frac "
                "0.3, seed 42, n_iter 50, target_binned_return_20) = 20 arms; base "
                "REUSED (e108_<m>_allmom), so each arm pairs (NW lag 20) vs its own "
                "model's allmom twin. Report carries calmar. Four lanes run "
                "concurrently via mommask_core.sh. Prior: fenccausal is what makes "
                "IA_01 work on the raw 22 (MLP); tenc and rolling means rarely moved "
                "trees on raw; the 286f stack risks ridge dilution (E82/E85).",
        "runner": "experiments/e112_allmom_encodings_crypto.sh",
        "prefix": "e112_", "scout_prefix": "e112scout_",
        "models": ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"],
        "expected_arms": 20,
    },
    "E113": {
        "title": "row ablation of E109's feature_fm_ block on the fenccausal ridge",
        "idea": "Which part of E109's 41-column foundation-model block makes the ridge worse "
                "(full block on fenccausal: payout -0.0170, t -2.35)? Ridge on "
                "crypto/v2.0-fenccausal-fmrep, the 44 base features always kept, ONE sub-block "
                "added per arm via --feature-exclude '^feature_fm_(?!<sub-block>)': row 1 "
                "TabPFN point predictions (3), rows 1-4 point + distribution + bin "
                "probabilities + disagreement (12), rows 2-4 alone (9, features-foundation.md's "
                "proposed first study), row 5 TabPFN embedding PCs (8), rows 6-7 per-block "
                "heads + context variants (7), rows 8-9 TimesFM/Chronos-2 on spot (14). 6 arms "
                "at the shared protocol, each paired (NW lag 20, 510 periods) vs e109_ridge_fenc "
                "(no block) and vs e109_ridge_fenc_fm (whole block), both REUSED.",
        "runner": "experiments/e113_fmrep_ablation_crypto.sh",
        "prefix": "e113_", "scout_prefix": "e113scout_",
        "models": ["ridge"],
        "expected_arms": 6,
    },    "E114": {
        "title": "60-day twins of the E112 picks + 3/4-member ensembles with the book",
        "idea": "Fits the three E112 picks on target_binned_return_60 (embargo auto -> 60) on "
                "the SAME datasets and protocol: lgb_ktz and catboost_ktz on "
                "crypto/v2.0-allmom-fillmask-tenc (58f), mlp_ktz on "
                "crypto/v2.0-allmom-fillmask-roll (228f) -> e114_<m>_{tenc,roll}60. Paired "
                "in-target vs E110's allmom60 twins. Then, PER MODEL, every 3- and 4-member "
                "equal-weight rank average of six items -- the model plain@20d, RESID@20d, "
                "plain@60d, RESID@60d (E107 transform, s = 0.5, meta >= 5 days old), "
                "CU_SA_IA_BO_02, CU_SA_IA_BO_MA_R_01 -- 35 combos x 3 models, scored on the "
                "payable 20-day label over one common window, paired (NW lag 20) vs each slot "
                "alone and vs the best member, with calmar.",
        "runner": "experiments/e114_selected_t60_ensembles_crypto.sh",
        "prefix": "e114_", "scout_prefix": None,
        "models": ["lgb_ktz", "catboost_ktz", "mlp_ktz"],
        "expected_arms": 3,
    },
    "E115": {
        "title": "ridge lambda chosen by prequential loss vs by time-series CV (fenccausal, monthly refit)",
        "idea": "Ridge on crypto/v2.0-fenccausal (44f), rolling 130, refit every 22 periods (one "
                "month), embargo 20, 510 scored periods. One FIXED alpha grid (51 log-spaced, "
                "1e-3..1e7, sklearn scale, bracketing the ~0.02..2e4 X'X spectrum); at every refit the ridge is fitted for every alpha "
                "(one SVD, src.prediction.lambda_selection) and two rules pick the column: "
                "PREQUENTIAL = argmin of each alpha's mean per-period loss over all its own "
                "out-of-sample forecasts resolved by that refit (position < train_end; 46 "
                "warm-up blocks from 2020-08 give it history), TSCV = argmin of mean held-out "
                "loss over 5 purged folds inside the training window (the pipeline's own "
                "rule; the tscv/corr arm reproduces hyper_params_search on the same grid, "
                "tested). Losses: corr (-numerai_corr), mse, and payout (-(0.1 corr + mmc) "
                "vs the REAL meta model, which starts 2024-05-07, so early payout picks "
                "stand on few periods; a block with none takes the corr pick) -> 6 arms "
                "from one pass, sharing every fit. Paired preq-tscv per loss and loss-vs-"
                "loss per rule (NW lag 20) on CORR/MMC/payout with calmar; ex-post "
                "fixed-alpha CORR and payout curves say whether lambda can matter.",
        "runner": "experiments/e115_ridge_prequential_crypto.py",
        "prefix": "e115_", "scout_prefix": "e115scout_",
        "models": ["ridge"],
        "expected_arms": 6,
    },
    "E116": {
        "title": "ridge refit cadence x lambda rule: every period / 1 / 3 / 6 months (fenccausal, live lag)",
        "idea": "E115 refit monthly and trailed the 6-monthly house ridge, but cadence, grid and "
                "rule all differed. Here grid (51 alphas 1e-3..1e7) and rule are fixed and ONLY "
                "the refit cadence moves: r1 (every period = every live round, the fastest a "
                "live round can refit since train.parquet gains one resolved date per round), "
                "r22, r65, r130 (registry.FIT's live cadence), for three E115 rules: tscv_corr, "
                "preq_payout, tscv_mse -> 12 arms. Embargo 21 (not 20) so training ends 22 "
                "periods before the predicted date, the live train/live gap; scored dates "
                "unchanged (510). Paired per arm vs r130, rules vs tscv_corr per cadence, "
                "embargo check r22 vs E115 (emb 20); NW lag 20, with calmar. Prior: the early "
                "raw-ridge cadence sweep's inverted U (monthly 0.127 < 6-monthly 0.153).",
        "runner": "experiments/e116_ridge_lambda_cadence_crypto.py",
        "prefix": "e116_", "scout_prefix": "e116scout_",
        "models": ["ridge"],
        "expected_arms": 12,
    },
    "E117": {
        "title": "Training window of the best single ridge, 1 week to 6 months",
        "idea": "How much history should each refit use? The payout-best single "
                "ridge of 360 clean runs -- ridge_glmnet on allfeat_pp minus the "
                "tenc column (w6r6_lobo_tenc, 811 features) -- with everything "
                "held except the rolling training window: 5 / 10 / 15 / 20 / 22 "
                "/ 65 / 130 periods (1, 2, 3, 4 weeks, 1, 3, 6 months; the panel "
                "is weekday-only at 5.01 periods/week). Lambda is SEARCHED FROM "
                "SCRATCH at every refit. The purged inner CV cannot fit inside a "
                "window of 22 periods or fewer, so those windows are tuned with "
                "the new --tune-window 130: the search runs on the 130 periods "
                "ending where the window ends, with rolling folds that each "
                "train on the window's own length, and the winner is refitted on "
                "the window alone. Every arm thus selects lambda on the same four "
                "validation slices. The glmnet path is built from the training "
                "window (alphas are lambda x N). 130 reproduces w6r6_lobo_tenc. "
                "Refit cadence stays 130: this is how-much-history, NOT "
                "freshness (E116's axis). A pinned-lambda first version was "
                "superseded before reporting.",
        "runner": "experiments/e117_train_window_crypto.sh",
        "prefix": "e117_", "scout_prefix": "e117scout_",
        "models": ["ridge_glmnet"], "expected_arms": 7,
    },
    "E119": {
        "title": "Training window of the best standalone MLP (IA_01), 1 week to 6 months",
        "idea": "E117's question asked of IA_01 -- mlp/a2_mlp_fenccausal, the payout-best of the "
                "2,515 20-day standalone MLP runs on the 510-day window (sklearn MLPRegressor, "
                "relu, early stopping off, crypto/v2.0-fenccausal, 44 features). Everything held "
                "-- data, the 50-draw random search with 5 inner splits, refit 130, embargo 20, "
                "seed 42 -- except the rolling training window: 5 / 10 / 15 / 20 / 22 / 65 / 130 "
                "periods. Hyperparameters SEARCHED FROM SCRATCH at every refit; windows below "
                "130 use --tune-window 130 (search on the 130 periods ending where the window "
                "ends, folds that train on the window's length, winner refitted on the window). "
                "The MLP grid is fixed, so nothing is rescaled to the window. 130 must reproduce "
                "a2_mlp_fenccausal. Cadence stays 130: how much history, not freshness.",
        "runner": "experiments/e119_ia01_train_window_crypto.sh",
        "prefix": "e119_", "scout_prefix": "e119scout_",
        "models": ["mlp"], "expected_arms": 7,
    },
    "E118": {
        "title": "34 causal model-representation blocks, one at a time, on the best ridge / LightGBM / CatBoost",
        "idea": "Each of 34 representation blocks drawn from the book's own models is added ALONE to the "
                "payout-best clean standalone run of three families, all on the 1,701-date spine, 510 "
                "scored periods: ridge_glmnet/w6r6_lobo_tenc (allfeat_pp minus tenc, 811f, 0.08779), "
                "lgb_ktz/e29_tenc_onehot (tenc_onehot, 763f, 0.08999), catboost_ktz/e109_catboost_ktz_fenc "
                "(fenccausal, 44f, 0.09303); baselines REUSED. Blocks (feature_xNN_): TabPFN on fenccausal -- "
                "x01 point, x02 distribution + bin probs, x03 estimator disagreement, x04 embedding PCs, x05 "
                "per-block heads + 30/390 contexts, x07 attention familiarity over context rows, x08 rank "
                "uncertainty, x09 forecast drift, x10 embedding neighbours, x11 residual learner (label = y - "
                "OOS ridge), x12 crowd (label = meta model), x13 60-day term structure, x14 predicted "
                "extremeness, x15 feature nowcasts, x16 self-context, x17 BTC-regime context, x18 pairwise "
                "ranking, x19 local drivers, x20 evidence age; x06 TimesFM/Chronos-2 on spot; and the book's "
                "models refit walk-forward from 2023-05 -- x22-25 random forest (MA_FENC: prediction, tree "
                "std, leaf PCs, leaf support), x26-27 CatBoost (IAUARETE_01: prediction, virtual-ensemble "
                "std), x28-30 LightGBM (SACI_04: prediction, stage profile, SHAP summary), x31-33 MLP (IA_01: "
                "prediction, hidden PCs, seed std), x34 CURUPIRA_05's 5 block ridges, x35 member spread. "
                "Every block causal and live-computable: TabPFN context = labelled rows ending 20 (60 for "
                "x13) dates before each 20-date refresh block; source models searched + refit every 130 "
                "inside their own 130-date window, embargo 20; spot as-of D-1; reductions fitted on past "
                "rows only; each column rank-binned per date. 3 x 34 = 102 block arms + 3 x00 arms (the base "
                "re-run on its -fmx dataset, the in-study 'without'); each block paired (NW lag 20) vs x00.",
        "runner": "experiments/e118_fmx_crypto.sh",
        "prefix": "e118_", "scout_prefix": "e118scout_",
        "models": ["ridge_glmnet", "lgb_ktz", "catboost_ktz"], "expected_arms": 105,
    },
    "E120": {
        "title": "Training window of the best standalone LightGBM (concat_trio), 1 week to 6 months",
        "idea": "The LightGBM twin of E117 (ridge) and the E119 MLP/CatBoost "
                "training-window studies. How much history should each refit "
                "use? The payout-best standalone LightGBM -- e03_concat_trio, "
                "the canonical pinned booster (num_leaves 255, 4327 trees, lr "
                "0.00257, colsample 0.07) on the group-CONCAT subset of "
                "crypto/v2.0-onehot (market-cap + momentum + volatility + "
                "per-symbol one-hot, 748 feat), 20-day payout 0.09074 -- with "
                "everything held except the rolling training window: 5 / 10 / 15 "
                "/ 20 / 22 / 65 / 130 periods (1,2,3,4 weeks, 1,3,6 months; the "
                "panel is weekday-only at 5.01 periods/week). The model is PINNED "
                "(--fixed-params), so unlike E117 there is NO lambda search and "
                "no --tune-window: every arm is the same booster refitted on that "
                "much history, cleanly paired to the w130 control (= "
                "e03_concat_trio). Refit cadence and tree count stay fixed -- this "
                "is how-much-history, NOT freshness (E116's axis) and NOT model "
                "capacity. Each window paired (NW lag 20) vs w130. Expect short "
                "windows to overfit 4327 trees on ~1 effective observation and "
                "lose, as E117 expected for ridge.",
        "runner": "experiments/e120_train_window_lgb_crypto.sh",
        "prefix": "e120_", "scout_prefix": "e120scout_",
        "models": ["lgb"], "expected_arms": 7,
    },
    "E121": {
        "title": "Training window of the best standalone CatBoost (IAUARETE_01), 1 week to 6 months",
        "idea": "The CatBoost twin of E117 (ridge), E119 (MLP IA_01) and E120 "
                "(LightGBM concat_trio): how much history should each refit use? "
                "The model is the payout- and MMC-best single catboost of E32, "
                "catboost_ktz on fenccausal (crypto/v2.0-fenccausal, 44 features "
                "= raw 22 + 22 causal per-symbol means), deployed as the component "
                "IAUARETE_01 (backtest run catboost_ktz/e32_sel_fenccausal). "
                "Everything is held -- data, features, 130-period refit cadence, "
                "embargo 20, seed, the 12-cell KTZ grid -- and ONLY the rolling "
                "training window moves: 5/10/15/20/22/65/130 periods (1/2/3/4 "
                "weeks, 1/3/6 months; weekday-only panel ~5.01 periods/week). The "
                "12-cell KTZ grid is SEARCHED exhaustively (--n-iter 12) at every "
                "refit. The purged inner CV cannot fit a window of 22 periods or "
                "fewer under embargo 20, so those arms use --tune-window 130 (the "
                "E117 mechanism): the search runs on the 130 periods ending where "
                "the window ends, with rolling folds that each train on the "
                "window's own length, and the winner is refitted on the window "
                "alone. Unlike ridge_glmnet there is no lambda-path-scaling issue "
                "-- the KTZ grid is static, so grid_from is ignored. Refit cadence "
                "stays 130: this is how-much-history, NOT freshness (E116's axis). "
                "w130 reproduces e32_sel_fenccausal / IAUARETE_01. Each window "
                "paired (NW lag 20) vs w130; expect short windows to overfit a "
                "2000-tree booster on ~1 effective observation and lose.",
        "runner": "experiments/e121_catboost_train_window_crypto.sh",
        "prefix": "e121_", "scout_prefix": "e121scout_",
        "models": ["catboost_ktz"], "expected_arms": 7,
    },
    "E122": {
        "title": "Training window of the best standalone RF (e57_tenc_onehot), 1 week to 6 months",
        "idea": "The rf sibling of E117 (ridge), E119 (MLP), E120 (lgb) and E121 "
                "(catboost): how much history should each refit use? The "
                "payout-best standalone random forest -- e57_tenc_onehot, rf_ktz "
                "(E48 priors) on crypto/v2.0-tenc-onehot (763f), the 60-DAY label "
                "(payout 0.0988 / MMC 0.0826 at the 130 window) -- with everything "
                "held (data, the 40-draw search, 5 inner folds, 130-period refit "
                "cadence, embargo auto=60, seed 42) and ONLY the rolling training "
                "window moving: 5/10/15/20/22/65/130 periods. --tune-window 130 "
                "for every window < 130 (a <70-period block cannot host its own "
                "purged folds at embargo 60; the search runs on 130 periods with "
                "folds that train on the window's length, then refits on the "
                "window alone), exactly as E117; w130 reproduces e57_tenc_onehot. "
                "All arms score the same ~470 periods, so each window is paired "
                "(NW lag 20) vs w130. Expect short windows to lose: 5 periods at a "
                "60-day target is under one effective observation, and the rf_ktz "
                "leaf floor {300..4000} exceeds a 5-period block's rows, pushing "
                "the search to its lightest leaves.",
        "runner": "experiments/e122_rf_window_crypto.sh",
        "prefix": "e122_", "scout_prefix": "e122scout_",
        "models": ["rf_ktz"], "expected_arms": 7,
    },
    "E123": {
        "title": "Training window of a FROM-SCRATCH-SEARCHED LightGBM (concat_trio feature set), 1 week to 6 months",
        "idea": "The from-scratch-search twin of E120, and the true LightGBM "
                "analogue of E117 (which searched ridge's lambda per window; E120 "
                "was the pinned simplification). Same base as E120 -- model class "
                "lgb, the group-CONCAT subset of crypto/v2.0-onehot (748 feat), "
                "130-period refit cadence, embargo 20, seed 42, windows 5/10/15/20/"
                "22/65/130 -- but the booster is SEARCHED afresh at every window "
                "instead of pinned to e03_concat_trio. The question E120 could not "
                "answer: when the model may get SIMPLER as its history shrinks, "
                "does a short window still lose, or does a re-tuned booster recover "
                "the payout that pinning the 4327-tree booster threw away? A "
                "--n-iter 60 --n-splits 5 random search of the plain lgb grid, "
                "which spans concat_trio on every axis (n_est 2000..30000, lr "
                "0.001..0.01, colsample 0.05..0.30, num_leaves 31..1023, and "
                "min_child_samples, whose range fit_grid rescales from the block "
                "every refit -- 20..780 at 130 periods, so concat_trio's 50 is "
                "inside). A 60-draw search need not land on the pin; whether it "
                "finds something as good is the question. Windows < 130 use --tune-window 130 "
                "(the E117 mechanism): the search runs on the 130 periods ending "
                "where the window ends, inner folds each train on the window's "
                "length, winner refitted on the window alone; w130 searches its own "
                "window. Each window paired (NW lag 20) vs the searched w130, and "
                "cross-paired against the pinned E120 arm at the same window "
                "(does searching beat pinning, window for window?) and against the "
                "deployed pinned concat_trio (e120_w130 = 0.09074).",
        "runner": "experiments/e123_train_window_lgb_search_crypto.sh",
        "prefix": "e123_", "scout_prefix": "e123scout_",
        "models": ["lgb"], "expected_arms": 7,
    },
    "E124": {
        "title": "E117's training-window ladder, fitted on the 60-day label, scored on the 20-day one",
        "idea": "Same model, data and windows as E117 -- ridge_glmnet on "
                "allfeat_pp minus tenc (811 features), rolling windows of 5 / 10 "
                "/ 15 / 20 / 22 / 65 / 130 periods, refit every 130, lambda "
                "searched from scratch at every refit, windows under 130 tuned on "
                "the 130 periods ending where the window ends -- but FITTED and "
                "TUNED on target_binned_return_60, with the embargo of 60 that "
                "label forces. The deliverable is the 20-day RE-SCORE: each "
                "arm's predictions graded on target_binned_return_20 by "
                "cross_target.score_on_payable_target against the same meta "
                "model and reference as every deployed slot, and consolidated "
                "with E117's seven fit-20 arms in one table on one common window. "
                "Costs of the label: two inner folds instead of four (exactly "
                "production's own 60-day search), the last ~40 dates lost, and "
                "paired t reported at Newey-West lag 20 and lag 60.",
        "runner": "experiments/e124_train_window_t60_crypto.sh",
        "prefix": "e124_", "scout_prefix": "e124scout_",
        "models": ["ridge_glmnet"], "expected_arms": 7,
    },
    "E125": {
        "title": "E119's training-window ladder for IA_01, fitted on the 60-day label, scored on the 20-day one",
        "idea": "The MLP twin of E124. Same model, data and windows as E119 -- mlp on "
                "crypto/v2.0-fenccausal (IA_01's setup, 44 features), rolling windows of "
                "5 / 10 / 15 / 20 / 22 / 65 / 130 periods, refit every 130, the 50-draw "
                "search run from scratch at every refit, windows under 130 tuned on the "
                "130 periods ending where the window ends -- but FITTED and TUNED on "
                "target_binned_return_60 with embargo 60. Deliverable: every arm "
                "RE-SCORED on target_binned_return_20 by "
                "cross_target.score_on_payable_target and consolidated with E119's seven "
                "fit-20 arms on the 470 dates they share, paired at Newey-West lag 20 and "
                "60. The arms run through E119's runner (E119_TARGET=60). e125_w130 must "
                "reproduce mlp/e60_mlp_fenccausal.",
        "runner": "experiments/e125_ia01_train_window_t60_crypto.sh",
        "prefix": "e125_", "scout_prefix": "e125scout_",
        "models": ["mlp"], "expected_arms": 7,
    },
    "E126": {
        "title": "E123's training-window ladder for the searched LightGBM, fitted on the 60-day label, scored on the 20-day one",
        "idea": "The LightGBM twin of E124 (ridge) and E125 (MLP), and the 60-day "
                "sibling of E123. Same model, data and windows as E123 -- lgb on the "
                "group-CONCAT subset of crypto/v2.0-onehot (748 feat), rolling windows "
                "of 5/10/15/20/22/65/130 periods, refit 130, a --n-iter 60 --n-splits 5 "
                "from-scratch search run at every refit, windows under 130 tuned on the "
                "130 periods ending where the window ends (--tune-window 130) -- but "
                "FITTED and TUNED on target_binned_return_60 with embargo 60 (--embargo "
                "auto). Crypto pays on the 20-day label, so the deliverable is every arm "
                "RE-SCORED on target_binned_return_20 by "
                "cross_target.score_on_payable_target and consolidated with E123's seven "
                "fit-20 arms in ONE table (src.analysis.e126_report: 7 windows x fit "
                "20/60 x raw / meta-residualised at the live strength 0.5 = 28 rows, "
                "plus the 378-pair prediction/payout correlation histogram), each window "
                "paired at Newey-West lag 20 on the dates the pair shares and against the "
                "deployed pinned concat_trio (0.09074) re-scored the same way. The "
                "question: does training the booster on the smoother 60-day horizon and "
                "deploying on the 20-day payable label beat training directly on 20, and "
                "does it depend on the window? A 60-day model's own metrics file scores "
                "it on the 60-day label and is never used.",
        "runner": "experiments/e126_train_window_lgb_search_t60_crypto.sh",
        "prefix": "e126_", "scout_prefix": "e126scout_",
        "models": ["lgb"], "expected_arms": 7,
    },
    "E127": {
        "title": "E121's CatBoost training-window ladder, fitted on the 60-day label, scored on the 20-day one",
        "idea": "The catboost twin of E124 (ridge), E125 (MLP) and E126 (lgb). "
                "Same model, data and windows as E121 -- catboost_ktz on "
                "crypto/v2.0-fenccausal (IAUARETE_01's setup, 44 features), rolling "
                "windows of 5/10/15/20/22/65/130 periods, refit every 130, the "
                "exhaustive 12-cell KTZ grid searched from scratch at every refit, "
                "windows under 130 tuned on the 130 periods ending where the window "
                "ends (--tune-window 130) -- but FITTED and TUNED on "
                "target_binned_return_60 with the embargo it forces (60, not 20). "
                "At embargo 60 a 130-period tuning block keeps only the LAST TWO of "
                "its five inner folds; production's own 60-day search "
                "(CURUPIRA_05_60: window 130, embargo 60) does the same, so it is "
                "kept for fidelity. Crypto pays on the 20-day label, so the "
                "deliverable is every arm RE-SCORED on target_binned_return_20 by "
                "cross_target.score_on_payable_target and consolidated with E121's "
                "seven fit-20 arms in ONE table (src.analysis.e127_report), each "
                "window paired at Newey-West lag 20 and 60 and against the deployed "
                "e121_w130 (IAUARETE_01) re-scored the same way. The arms run "
                "through E121's runner (TW_TARGET=60); the longest window relates "
                "to catboost_ktz/e59_fenccausal (catboost on 60d). A 60-day model's "
                "own metrics file scores it on the 60-day label and is never used.",
        "runner": "experiments/e127_catboost_train_window_t60_crypto.sh",
        "prefix": "e127_", "scout_prefix": "e127scout_",
        "models": ["catboost_ktz"], "expected_arms": 7,
    },
    "E128": {
        "title": "E122's RF training-window ladder fitted on the 20-day label; with E122, raw and meta-residualised, all scored on 20",
        "idea": "The fit-20 half E122 lacks. Same model, data and windows as E122 -- "
                "rf_ktz (E48 priors) on crypto/v2.0-tenc-onehot (763f), rolling "
                "windows of 5/10/15/20/22/65/130 periods, refit 130, the 40-draw "
                "search run from scratch at every refit, windows under 130 tuned on "
                "the 130 periods ending where the window ends (--tune-window 130) -- "
                "but FITTED and TUNED on target_binned_return_20 with embargo 20. "
                "The arms run through E122's runner (TW_TARGET=20). Deliverable: "
                "src.analysis.e128_report, the RF mirror of e124/e127_report -- 28 "
                "rows, 7 windows x {fit 20 (E128), fit 60 (E122)} x {raw, "
                "meta-residualised at the live strength 0.5 / 5-day lag, applied to "
                "the finished forecasts}, every row scored on "
                "target_binned_return_20 by cross_target.score_on_payable_target on "
                "the dates all share, paired at Newey-West lag 20 (and 60 for fit 60 "
                "vs fit 20), plus the 378 pairwise prediction and payout correlations "
                "tagged across-label / same-label window / raw-vs-resid twin.",
        "runner": "experiments/e128_rf_window_t20_crypto.sh",
        "prefix": "e128_", "scout_prefix": "e128scout_",
        "models": ["rf_ktz"], "expected_arms": 7,
    },
    "E129": {
        "title": "Causal leaf-value features of the 6-month RF (E128 w130) added to the 8 models inside CU_SA_IA_BO_MA_R_01 and CU_SA_IA_BO_02",
        "idea": "Does the structure of the 6-month random forest help the book's best slots? "
                "src/data_mani/build_rfleaf.py re-runs E128's w130 forest (rf_ktz, tenc-onehot, fit 20, "
                "rolling 130 / refit 130 / embargo 20, 40-draw purged search, seed 42) through the "
                "pipeline's own forecast() with the validation start moved back 8 blocks, and for every row "
                "summarises the 100 trees' leaf VALUES -- mean, std, q10/25/50/75/90 -- plus the forest's "
                "prediction residualised on the 5-day-lagged meta model: 8 feature_rfleaf_* columns, binned "
                "to {0,.25,.5,.75,1}, merged onto each member's dataset (crypto/v2.0[-...]-rfleaf). Causal: "
                "a row's leaves come from its own walk-forward block's forest, trained only on rows that had "
                "resolved; leaf IDS are not used because they mean something different in every refit. "
                "The 8 base models (IA_01, IA_02, SACI_04, MA_FENC, MA_ALLFEAT, CURUPIRA_04 each re-run on "
                "their exact recorded protocol as control and +rfleaf; CURUPIRA_05 and _05_60 gain a sixth, "
                "leaf-only ridge block) are reassembled into both slots by src.analysis.e129_report, whose "
                "assembler first reproduces the stored slot series. Paired Newey-West lag 20 on the 20-day "
                "payout; a result is |t| >= 2 for a slot or member.",
        "runner": "experiments/e129_rfleaf_members_crypto.sh",
        "prefix": "e129_", "scout_prefix": "e129scout_",
        "models": ["mlp", "lgb", "rf_ktz", "ridge_glmnet"], "expected_arms": 14,
    },
    "E130": {
        "title": "Causal leaf-value features of CatBoost (fenccausal + x29) and RF (allfeat) added to the best ridge and MLP",
        "idea": "Do two of the book's best tree models carry structure the best linear and neural models miss? "
                "src/data_mani/build_treeleaf.py re-runs catboost_ktz on fenccausal + E118's x29 block "
                "(e118_catboost_x29) and rf_ktz on allfeat (MA_ALLFEAT, e52_allfeat) through forecast() -- "
                "fit 20, rolling 130 / refit 130 / embargo 20, searched from scratch at every refit, seed 42 -- "
                "with the validation start moved back 2 blocks, and gives every row the distribution of its "
                "trees' leaf VALUES (mean, std, q10/25/50/75/90): 7 feature_lfcat_* + 7 feature_lfrf_* "
                "columns, rank-binned within date to {0,.25,.5,.75,1}, merged as crypto/v2.0-fenccausal-treeleaf "
                "and crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmx-treeleaf. Causal as E129: a row's "
                "leaves come from its own walk-forward block's model, trained only on resolved rows. Arms, "
                "every one a 50-draw purged search from scratch: ridge_glmnet on allfeat_pp minus tenc + x29 "
                "(e118_ridge_x29's protocol) and mlp on fenccausal (IA_01's), each as control / +cat / +rf / "
                "+both, plus the mlp control and +both again at seed 43 for a noise floor. Paired Newey-West "
                "lag 20 against the same-seed in-study control; a result is |t| >= 2 on payout.",
        "runner": "experiments/e130_treeleaf_crypto.sh",
        "prefix": "e130_", "scout_prefix": "e130scout_",
        "models": ["ridge_glmnet", "mlp"], "expected_arms": 10,
    },
    "E131": {
        "title": "litmom-fillind vs v2.0: fixed-config model x feature-group factorial (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Does crypto/v2.0-litmom-fillind (60f = 22 base + 17 feature_mom_ + 21 feature_isna_ "
                "indicators: 17 momentum, 4 base) add out-of-sample value over crypto/v2.0 (22f) on "
                "target_binned_return_20? Controlled factorial: each of the three families (ridge, "
                "lgb_ktz as the gbm, mlp_ktz) at ONE fixed previously-searched configuration, refit from "
                "scratch per chronological fold, across feature variants B0 (v2.0), B0C (candidate's base "
                "columns), B1 (+4 base indicators), M1 (+17 mom), M2 (+17 mom +17 mom indicators), FULL "
                "(60), ADDED_ONLY, plus leave-one-family-out and an alternative-fill control; identical "
                "aligned rows; walk-forward with a 22-date purge and an untouched final test block; "
                "5 seeds for the stochastic families; scored with numerai_tools numerai_corr; paired "
                "deltas with Newey-West/block-bootstrap CIs; custom contribution relative to B0; blends "
                "B0+ADDED_ONLY with CV-chosen weight evaluated once on the final block. Audits first: "
                "schema/alignment, base integrity, indicator correctness, point-in-time, missingness, "
                "config provenance. Report results/reports/crypto/e131_litmom_fillind/.",
        "runner": "experiments/e131_litmom_fillind_factorial.py",
        "prefix": "e131_", "scout_prefix": "e131scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E132": {
        "title": "feed-fillind datasets vs v2.0 with a neutral 0.5 for no-feature coins (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Rerun of the E131 protocol on crypto/v2.0-litmom-feed-fillind (58f), "
                "-momswap-feed-fillind (48f) and -momswap-macro-carver-feed-fillind (96f): spot rebuilt on "
                "the extended file, coins with no price dropped. Every key stays scored: a dropped row "
                "takes a neutral prediction (per-date rank, then 0.5). B0F (raw v2.0 on the feed rows) "
                "separates the row handling from the features; P_DROP (look-ahead no-feed flag) sizes the "
                "survivorship channel; P_HIST controls tenure; equal-budget retunes and CV-chosen blends "
                "as in E131. Report results/reports/crypto/e132_feed_fillind/.",
        "runner": "experiments/e132_feed_fillind_factorial.py",
        "prefix": "e132_", "scout_prefix": "e132scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E133": {
        "title": "own-history features crypto/v2.0-histfe vs v2.0 (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Numerai-style history features on the 22 shipped bins, per ucid and on calendar windows: "
                "own-history rank, MA differences, days since max, days since change (352 columns). "
                "Factorial B0 / SHORT / LONG / COMBO / TSC / FULL / ADDED_ONLY / families / P_HIST at the "
                "E131 fixed configs, 5 seeds, plus an equal-budget retune. Dev only: dates from 2026-02-20 "
                "are never read, so no verdict can pass the test-block bar. "
                "Report results/reports/crypto/e133_histfe/.",
        "runner": "experiments/e133_histfe_factorial.py",
        "prefix": "e133_", "scout_prefix": "e133scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E134": {
        "title": "feed-fillind datasets as live would use them: route each coin by what it has (ridge / lgb_ktz / mlp_ktz)",
        "idea": "No refit: E132's saved predictions, scored under the live rule. raw v2.0 ranks every coin; "
                "the richer model only reorders the coins that have its features, inside the rank slots "
                "those coins already hold, so a no-feed coin keeps its v2.0 rank. Tests L/S/C, each with "
                "and without missing-data indicators, against B0 on all keys, plus the same retuned. "
                "Sensitivity: pooled merge, and routing on observed share. Replaces E132's neutral 0.5. "
                "Report results/reports/crypto/e134_route/.",
        "runner": "experiments/e134_route_feature_sets.py",
        "prefix": "e134_", "scout_prefix": "e134scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E135": {
        "title": "own-history features crypto/v2.0-histfe as live would use them: route each coin by what it has (ridge / lgb_ktz / mlp_ktz)",
        "idea": "E134's rule and bar on E133's dataset. Covered now means the arm's own history features are "
                "observed that date, since histfe has no feed mask, only warm-up, which is known at the time. "
                "E133's dev predictions plus a new fit of the final block, the one E133 held back. "
                "Primary arm FULL, against B0 on all keys, plus horizon and family arms, the tenure control "
                "and the equal-budget retune. Report results/reports/crypto/e135_route_histfe/.",
        "runner": "experiments/e135_route_histfe.py",
        "prefix": "e135_", "scout_prefix": "e135scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E136": {
        "title": "crypto/v2.0-histfe: complete the feature-set evaluation procedure (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Runs only what E133 and E135 left open and reuses their predictions: final block for ADDED_ONLY, "
                "dev-weighted blends scored on test, leave-one-family-out ablations, block + tenure against "
                "tenure, a causal warm-up placebo, and a point-in-time audit on the real panel. Consolidates "
                "dev and test into the procedure's HTML. Report results/reports/crypto/e136_histfe_evaluation/.",
        "runner": "experiments/e136_histfe_evaluation.py",
        "prefix": "e136_", "scout_prefix": "e136scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E137": {
        "title": "crypto/v2.0-fenccausal-binned vs v2.0: full feature-set evaluation procedure (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Does the 22 causal per-symbol feature means, rank-binned per date onto Numerai's scale, add value "
                "over the 22 shipped features, and does binning cost anything against the continuous means the live "
                "IA slots use? 12 walk-forward blocks, 5 seeds, factorial of feature groups, ablations, tenure, "
                "a global-mean look-ahead placebo, retune, blends, routing, audit. Reuses E133/E135's B0. "
                "Run on lovelace. Report results/reports/crypto/e137_fenccausal_binned/.",
        "runner": "experiments/e137_fenccausal_binned_evaluation.py",
        "prefix": "e137_", "scout_prefix": "e137scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E138": {
        "title": "crypto/v2.0-carver-macro vs v2.0: full feature-set evaluation procedure (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Do the 15 Carver/Clenow trend signals (recomputed from the current spot file, D-1, binned per date) and the "
                "18 shared macro features add value over ALL 22 shipped features, under the routed missing-value "
                "standard (no row dropped, no indicators, uncovered coins keep v2.0's rank)? 12 walk-forward blocks, "
                "5 seeds, block/family ablations, tenure, warm-up and no-feed look-ahead placebos, capped-grid retune, "
                "blends, R_SLOT/R_HALF routing, truncation audit. Reuses E133/E135's B0. "
                "Report results/reports/crypto/e138_carver_macro/.",
        "runner": "experiments/e138_carver_macro_evaluation.py",
        "prefix": "e138_", "scout_prefix": "e138scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E139": {
        "title": "crypto/v2.0-carver-macro-infl vs v2.0 and vs carver-macro: full feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": 'Do 12 shared break-even inflation features (4 levels, 8 changes) add value on top of crypto/v2.0-carver-macro (E138), and is the 67f set better than v2.0? E138 protocol, routing and bar; reuses E138 FULL as parent CM and its P_WARM/P_NM. Run on dostoievski + tolstoi. Report results/reports/crypto/e139_carver_macro_infl/.',
        "runner": "experiments/e139_carver_macro_infl_evaluation.py",
        "prefix": "e139_", "scout_prefix": "e139scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E140": {
        "title": "crypto/v2.0-carver-macro-infl-xinteractions vs v2.0 and vs carver-macro-infl: full feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": 'Do 270 binned raw-product interactions of the 15 trend signals with 18 signed macro momentum signals (10 returns, 8 break-even changes) add value on top of crypto/v2.0-carver-macro-infl (E139)? E138 protocol, routing and bar; reuses E139 FULL as parent CMI. Run on deepfour + deepsix. Report results/reports/crypto/e140_carver_macro_infl_xinteractions/.',
        "runner": "experiments/e140_carver_macro_infl_xinteractions_evaluation.py",
        "prefix": "e140_", "scout_prefix": "e140scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E141": {
        "title": "crypto/v2.0-macrobeta vs v2.0: rolling coin betas to macro/inflation factors, full feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Do 32 causal rolling betas (60/120 macro business days) of each coin's log return on one-day changes of 16 macro/inflation factors (gold, silver, S&P, DXY, y1-y30, 3 slopes, 4 break-evens), rank-binned per date, add value over the 22 shipped features? Cross-sectional replacement for the date-constant macro/infl blocks that were inert or harmful in E138/E139. E138 protocol, routing on beta coverage, own P_WARM/P_NM. Report results/reports/crypto/e141_macrobeta/.",
        "runner": "experiments/e141_macrobeta_evaluation.py",
        "prefix": "e141_", "scout_prefix": "e141scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E142": {
        "title": "crypto/v2.0-future-trends vs v2.0: look-ahead positive control for the feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Sanity check of the procedure: one column, each coin's spot return over the NEXT 20 calendar days, rank-binned "
                "per date. It knows the future, so FULL and ADDED_ONLY must beat B0 by far more than any honest block and be "
                "read as leakage. E138 protocol, routing on the column's coverage, no retune. "
                "Report results/reports/crypto/e142_future_trends/.",
        "runner": "experiments/e142_future_trends_evaluation.py",
        "prefix": "e142_", "scout_prefix": "e142scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E143": {
        "title": "crypto/v2.0-c2syn vs v2.0: Chronos-2-Synth pre-head representations of each coin's panel history, full feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": "The v2.0 panel as one series per coin (label + 22 features on the panel-date grid), run zero-shot through "
                "autogluon/chronos-2-synth in two modes -- cov: label with the 22 features as covariates (group attention), "
                "ar: label alone -- keeping the encoder state the forecasting head reads for date D's step. Labels only from "
                "dates <= D-35 days; features after the wall enter as known-future covariates. 768-d states reduced to 16 PCs "
                "per mode on a past-only burn-in basis, rank-binned per date with Numerai's rule. E138 protocol, routing on "
                "the block's coverage, own P_WARM/P_NM. Report results/reports/crypto/e143_c2syn/.",
        "runner": "experiments/e143_c2syn_evaluation.py",
        "prefix": "e143_", "scout_prefix": "e143scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E144": {
        "title": "crypto/v2.0-c2ft vs v2.0: walk-forward LoRA fine-tuned Chronos-2-Synth pre-head representations, full feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": "E143 with the model fine-tuned instead of zero-shot. One LoRA fine-tune pair (cov: label + 22 features as "
                "covariates; ar: label alone) per walk-forward block, on the panel cut at the block's first train date "
                "(labels >= 35 days older than any row it represents), early-stopped inside the cut; the block's train and "
                "test rows represented by its own models, PCA basis on its train rows, 16 PCs per mode, rank-binned. "
                "Features differ by block (blocks.parquet); blocks 0-1 have no model. Report results/reports/crypto/e144_c2ft/.",
        "runner": "experiments/e144_c2ft_evaluation.py",
        "prefix": "e144_", "scout_prefix": "e144scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E145": {
        "title": "crypto/v2.0-c2head vs v2.0: Chronos-2-Synth FORECAST-head features (zero-shot + walk-forward LoRA, cov/ar, deltas), full feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Chronos-2-Synth's own 20-step quantile forecasts of a coin's label, not its hidden state. Per row: 256-step context, "
                "labels after D-35 blanked, 22 features through D as PAST-ONLY covariates; zero-shot and one LoRA fine-tune per "
                "block 2-11 (training windows built like inference rows). Five summaries per branch (median mean/terminal/slope, "
                "q90-q10, asymmetry) + COV-minus-AR deltas, within-date percentile rank (5-bin robustness copy). Rules frozen "
                "in experiments/e145_registration.md before any fit. Report results/reports/crypto/e145_c2head/.",
        "runner": "experiments/e145_c2head_evaluation.py",
        "prefix": "e145_", "scout_prefix": "e145scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E146": {
        "title": "crypto/v2.0-c2syn-c2ft-dims vs v2.0: all 3,072 binned hidden dimensions of the zero-shot and fine-tuned Chronos-2-Synth states, full feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": "E143/E144 kept 16 PCs per state. This keeps every one of the 768 dimensions of all four encoder states "
                "(zero-shot cov / ar, per-block LoRA fine-tuned cov / ar), each rank-binned (zero-shot within date, fine-tuned "
                "within block x date), 22 + 3,072 = 3,094 features. Arms: FULL, ADDED_ONLY, one state at a time, zero-shot pair, "
                "fine-tuned pair, tenure / warm-up / look-ahead controls, v2.0 minus each shipped group, and the E143/E144 PC arms "
                "reused so dims vs PCs is paired. Fine-tuned dims differ by block (blocks.parquet). Report results/reports/crypto/e146_c2dims/.",
        "runner": "experiments/e146_c2dims_evaluation.py",
        "prefix": "e146_", "scout_prefix": "e146scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E147": {
        "title": "crypto/v2.0-c2syn-c2ft group-only importance: ridge and MLP tuned from scratch on ONE feature group at a time (6 shipped groups, 4 Chronos-2 PC groups, 4 Chronos-2 768-dim groups)",
        "idea": "Standalone importance, not marginal value: each of 14 groups (momentum, reversion, volume, volatility, price level, size; "
                "c2cov/c2ar/c2ftcov/c2ftar 16 PCs; the same four as 768 raw dims) is the ONLY input to a ridge and an MLP whose "
                "hyperparameters are picked per block on an inner split (MLP grid includes relu), plus ALL22 as the reference. "
                "Scored on all 468k keys and on the common rows where every state is observed. Report results/reports/crypto/e147_group_importance/.",
        "runner": "experiments/e147_group_importance.py",
        "prefix": "e147_", "scout_prefix": "e147scout_",
        "models": ["ridge", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E148": {
        "title": "crypto/v2.0-c2resid: zero-shot Chronos-2 forecasts of the BTC+ETH beta-residual return path, with BTC / ETH / leave-one-out size-peer covariates, against a ridge on the same inputs, C2 on the raw and residual paths alone, and C2 with grouped size-bucket context",
        "idea": "Decompose any gain into residualising the input (res vs raw), adding peer/market information (cov vs res, grp vs res) and encoding with C2 (cov/grp vs a causal rolling ridge on the same windows). "
                "Six signals (ridge, residual momentum, C2 raw/res/cov/grp), each the forecast of the residual return over the target's span, scored standalone and added to v2.0 "
                "under the routed feature-set protocol. Report results/reports/crypto/e148_c2resid/.",
        "runner": "experiments/e148_c2resid_evaluation.py",
        "prefix": "e148_", "scout_prefix": "e148scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E149": {
        "title": "crypto/v2.0-vol-causal vs v2.0: the proposed realised-volatility block (22 levels + 132 own-history ranks, feature_vol_*), full feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
        "idea": "Candidate = feature_vol_* only; the shipped feature_volatility_20d/60d stay in the 22-feature base. Arms: FULL, ADDED_ONLY, levels vs ranks, one estimator family "
                "(sqret / ewma / park / yz), short vs long windows, F_SWAP (FULL minus the shipped volatility), tenure / warm-up / look-ahead controls, equal-budget retune. "
                "Routed R_SLOT on each arm's coverage. Report results/reports/crypto/e149_vol_causal/.",
        "runner": "experiments/e149_vol_causal_evaluation.py",
        "prefix": "e149_", "scout_prefix": "e149scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },
    "E150": {
        "title": "crypto/v2.0-vol-causal-macrovol: does coin volatility interact with macro volatility regimes? Matched covered-sample models, routed (ridge / lgb_ktz / mlp_ktz)",
        "idea": "578 feature_mvx_ columns: per source (gold, S&P 500, DXY, 10y yield, VIX level, VIX vol-of-vol) zc*zm, |zc-zm|, zc in high / low "
                "macro-vol regimes, plus coin-level macro cov/corr/beta/resvol/r2. B_COV, VC_COV and every candidate trained on ONE common covered "
                "mask, routed R_SLOT through B_ALL, compared routed against routed. Per-source, ALL, LOO, exposure splits, dev-selected PARSIM, "
                "scalar control, circular and future (look-ahead) placebos, regime split, equal-budget retune. "
                "Corrected observed-only development rerun completed: REJECT all-macro on ridge / LightGBM / identity MLP. "
                "Searched matched routed CORR deltas -0.01649 / -0.00881 / -0.03112 (NW t -4.09 / -2.46 / -4.87); "
                "no source qualifies for PARSIM. Separate hi/lo arms, no neutral-imputed candidate training. "
                "Retrospective GLD proxy and unverified vintages; no untouched-test claim. "
                "Report results/reports/crypto/e150_macrovol_observed_dev/E150_does_macrovol_help_REJECT_DEVELOPMENT.pdf.",
        "runner": "experiments/e150_macrovol_evaluation.py",
        "prefix": "e150_", "scout_prefix": "e150scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },


    "E151": {
        "title": "crypto/v2.0-volumealpha-causal: does point-in-time expected trading volume / liquidity add marginal value? (Trading Volume Alpha analogue; ridge / lgb_ktz / mlp_ktz, routed)",
        "idea": "Venue dollar volume (Binance archive incl. delisted + 8 CEX + Yahoo aggregator, separate source families), the paper's 8-column "
                "technical block (lagged MA 1/5/22/252 of return and log dollar volume), extensions, crypto fundamental analogues (Coin Metrics / "
                "DefiLlama by cmcId, revised), Binance OI/funding leverage, Dimson beta, age, point-in-time Snapshot votes + listings + expiry "
                "interactions, cross-asset volume x sensitivity, out-of-fold volume forecasts (ridge/LGB/MLP/32-16-8) and trading-cost proxies. "
                "Matched covered mask, R_SLOT routed through B_ALL, source ablations, P_HIST/P_WARM controls, dev-selected PARSIM. "
                "Volume-forecast skill is reported as an execution result unless it also moves CORR/MMC. Rules: experiments/e151_registration.md. "
                "Report results/reports/crypto/e151_volumealpha/.",
        "runner": "experiments/e151_volumealpha_evaluation.py",
        "prefix": "e151_", "scout_prefix": "e151scout_",
        "models": ["ridge", "lgb_ktz", "mlp_ktz"],
        "expected_arms": 0,
    },

}


STUDIES["E152"] = {
    "title": "Volume-alpha: strict-vintage admission plus user-authorized inferred-source completion, OOF volume models and routed evaluation",
    "idea": "Complementary to E151's retrospective-source experiment: admit only evidenced historical vintages, "
            "dated chain/contract mappings and contemporaneous USD conversion. Preserve every v2.0 column and row; "
            "separate paper technical predictors, extensions, fundamentals, announced events, cross-asset exposures, "
            "OOF ridge/GBM/MLP volume forecasts and cost scenarios. Matched covered-only return fits with B_ALL rank-slot "
            "routing, equal searches and five seeds. Forward test begins 2026-11-13, after dev-only selection freezes. "
            "The strict audit still distinguishes genuine vintages. On 2026-11-13 the user authorized alternative sources and inference: "
            "build_volumealpha_inferred uses Coinbase USD quotes, measured legacy FX, permanent CMC/contract candidates with a causal gate, "
            "D-2 observations and expanding OOF ridge/GBM/MLP. Reconstructed-source status is explicit; no return-alpha result claimed. "
            "Rules: experiments/e152_registration.md. Report results/reports/crypto/e152_volumealpha_causal/.",
    "runner": "experiments/e152_volumealpha_causal_evaluation.py",
    "prefix": "e152_", "scout_prefix": "e152scout_",
    "models": ["ridge", "lgb_ktz", "mlp_ktz"], "expected_arms": 0,
}


STUDIES["E153"] = {
    "title": "TabNet with a wide KTZ-style search (tabnet_ktz) on raw and fenccausal",
    "idea": "E18 pinned TabNet at library defaults and it lost to the deployed slot on every dataset; its "
            "write-up named tuning as the only lever. tabnet_ktz opens every pytorch-tabnet axis (width 8-64 "
            "with n_a tied to n_d, n_steps 2-7, gamma, GLU depth, lambda_sparse 1e-6..1e-1, BN momentum, "
            "sparsemax/entmax, lr 3e-4..5e-2, batch/ghost batch, epochs 30-150). Random search n_iter 60, "
            "3 folds, rolling 130/130, CPU. Read against e18_* and the deployed slot on the same dataset.",
    "runner": "experiments/e153_tabnet_ktz_crypto.sh",
    "prefix": "e153_", "scout_prefix": "e153scout_",
    "models": ["tabnet_ktz"], "expected_arms": 2,
}


# E154/E155: two independent zero-shot arms on the frozen Chronos-2-Synth checkpoint
# (autogluon/chronos-2-synth @ 3607918a, the E143-E148 revision). Same rounds (keys.parquet
# blocks 9-10 = validation, block 11 = test), same universe (the 300 panel symbols of each round),
# same scoring (e134.one numerai_corr, e131 contrib MMC, NW lag 20, block-20 bootstrap).
STUDIES["E154"] = {
    "title": "numerai_crypto_c2synth_zero_shot_return_rank -- Chronos-2-Synth zero-shot forecast of each token's daily log returns, summed over the target interval, ranked within round",
    "idea": "PREDECLARED: primary input = daily spot log returns (clipped to [-1,1]) through the round cutoff C-1. "
            "Forecast steps C..C+30; raw score = sum of the q50 (the head's median; no mean exists) over days C+3..C+30, "
            "the official interval close(C+2)->close(C+30) whose start is not yet known at C-1. Per-token groups, frozen fp32. "
            "Context in {64,128,256} days chosen on validation CORR only. Eligible = >=32 finite returns in the last 64 days. "
            "Baseline: 28-day historical momentum (sum of the last 28 log returns through C-1). Compared paired against B0 "
            "(lgb_ktz on v2.0, 5-seed rank average) and the staked slot CURUPIRA_05 (test rounds to 2026-07-08). "
            "Runner src/data_mani/c2synth_zeroshot.py --arm return_rank; output results/reports/crypto/e154_c2synth_return_rank/.",
    "runner": "src/data_mani/c2synth_zeroshot.py",
    "prefix": "e154_", "scout_prefix": "e154scout_",
    "models": ["chronos2"], "expected_arms": 0,
}

STUDIES["E155"] = {
    "title": "numerai_crypto_c2synth_zero_shot_bin_forecast -- Chronos-2-Synth zero-shot forecast of each token's own target_binned_return_20 label series, ranked within round",
    "idea": "PREDECLARED: series = official target_binned_return_20 labels indexed by their original date on a daily calendar "
            "(missing days NaN). A label for date d is available at round C only if d <= C - 32 days (close(d+30) realised by C-2 "
            "and Numerai's ~31-32 day serving lag). Forecast from the last available label date L through C, take the step at C "
            "(h = C - L days, skip if h > 64), keep the continuous q50. Context in {64,128,256} calendar days ending at L, chosen on "
            "validation only. Eligible = >=20 labels in the 64 days ending at L. Baselines (same availability): last available label, "
            "mean of the last five. Same comparators as E154. Blend rule (predeclared): if on validation the mean within-round rank "
            "correlation with E154 is < 0.5 AND the 50/50 rank average beats the better arm's validation CORR, score the blend on test. "
            "Runner src/data_mani/c2synth_zeroshot.py --arm bin_forecast; output results/reports/crypto/e155_c2synth_bin_forecast/.",
    "runner": "src/data_mani/c2synth_zeroshot.py",
    "prefix": "e155_", "scout_prefix": "e155scout_",
    "models": ["chronos2"], "expected_arms": 0,
}

STUDIES["E156"] = {
    "title": "Simulated addition of the two shipped raw volatility columns: v2.0-exrawvolfeat (20 features) vs full v2.0 (22)",
    "idea": "Predeclared full-universe add-back: remove feature_volatility_20d / feature_volatility_60d, "
            "then add each and both back. Same labels, rows, 130/22/130 development folds; ridge, LightGBM and "
            "identity MLP, matched seeds, fixed configs plus equal six-candidate searches on every arm and seed. "
            "Five stochastic seeds for both searched and fixed fits; no macro or reconstructed feed coverage masks. "
            "Positive FULL minus EXRAWVOL paired CORR/MMC means adding the shipped columns helps. "
            "Other volatility proxies such as Sharpe and Bollinger remain; conditional column value only. "
            "COMPLETED: adding both shipped volatility columns raises searched development CORR by "
            "+0.04768 / +0.02817 / +0.05487 (ridge / LightGBM / MLP; NW t +5.07 / +5.02 / +6.52). "
            "Positive mean CORR gains in all 11 blocks and every matched seed; MMC gains positive but t < 2. "
            "PROMISING on development, not untouched-test confirmed. Previously scored final block excluded. "
            "Report results/reports/crypto/e156_rawvol_addback/E156_does_adding_raw_volatility_help_PROMISING_DEVELOPMENT.pdf.",
    "runner": "experiments/e156_rawvol_addback.py",
    "prefix": "e156_", "scout_prefix": "e156scout_",
    "models": ["ridge", "lgb_ktz", "mlp_ktz"], "expected_arms": 0,
}




STUDIES["E158"] = {
    "title": "tabnet_ktz hyperparameter importance map (raw + fenccausal, 2 seeds)",
    "idea": "E153's grid and walk-forward, but every candidate's inner-CV CORR and MMC is kept "
            "(--save-cv-results -> results/tuning/crypto/tabnet_ktz/<run>_cv.csv) with a fresh draw per refit "
            "(--reseed-per-refit), n_iter 120, seeds 42/43. src.analysis.hparam_importance gives main-effect "
            "importance vs a permutation null, stability across refits, interactions, sensitivity curves and "
            "a suggested search space for later walk-forward studies. "
            "Report results/reports/crypto/e158_tabnet_ktz_hparams/.",
    "runner": "experiments/e158_tabnet_ktz_hparams_crypto.sh",
    "prefix": "e158_", "scout_prefix": "e158scout_",
    "models": ["tabnet_ktz"], "expected_arms": 4,
}

STUDIES["E159"] = {
    "title": "crypto/v2.0-volumealpha-causal vs v2.0: the 70-column volume-alpha block (E152 inferred build), routed feature-set evaluation (ridge / lgb_ktz / mlp_ktz)",
    "idea": "E149's routed protocol on E152's dataset: FULL, ADDED_ONLY, one arm per family (technical, liquidity, fundamentals, "
            "cross-asset, OOF volume forecasts, costs), raw vs model-derived, tenure / warm-up / look-ahead controls, equal-budget retune. "
            "Test = block 11 as in every feature-set study, not E152's forward test. Report results/reports/crypto/e159_volumealpha/.",
    "runner": "experiments/e159_volumealpha_evaluation.py",
    "prefix": "e159_", "scout_prefix": "e159scout_",
    "models": ["ridge", "lgb_ktz", "mlp_ktz"], "expected_arms": 0,
}































STUDIES["E183"] = {
    "title": "TabNet with epochs tied to model size (Timmermann-Vulicevic), raw + fenccausal, 2 seeds",
    "idea": "tabnet_ktz_tv: no max_epochs axis; each candidate trains E = floor(0.2*P^0.8)+10 epochs from the "
            "trainable parameter count P of the network its sampled settings build on the dataset's column count "
            "(src/prediction/tabnet_tv.py). Space capped at width 8-16, steps 2-5, GLU depth 1-2 (TV 136-906 "
            "epochs). E158's walk-forward and per-candidate capture (--save-cv-results, --reseed-per-refit), "
            "n_iter 40, seeds 42/43. Compares TV-length training with E153/E158's 30-150 epochs and feeds "
            "src.analysis.hparam_importance. Launched with a one-off src.cluster --extra-cores 10.",
    "runner": "experiments/e183_tabnet_tv_crypto.sh",
    "prefix": "e183_", "scout_prefix": "e183scout_",
    "models": ["tabnet_ktz_tv"], "expected_arms": 4,
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
                if spec.get("proxy_kind"):
                    if "--kind" not in arguments:
                        continue
                    position = arguments.index("--kind")+1
                    if position >= len(arguments) or arguments[position] != spec["proxy_kind"]:
                        continue
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
    if spec.get("artifact_root"):
        return scan_checkpoint_arms(study_id, spec)
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


def scan_checkpoint_arms(study_id, spec):
    """Index custom-study task checkpoints; scores appear only when summarized.

    No manually maintained status and no means inferred from incomplete folds.
    Feature count varies with training-active IDs, so leave it unspecified.
    """
    import csv
    import glob
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), spec["artifact_root"])
    summary = {}
    try:
        with open(os.path.join(directory, "all_model_window_rows.csv")) as handle:
            summary = {row["tag"]: row for row in csv.DictReader(handle)}
    except (OSError, ValueError, KeyError):
        pass
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "fits", "*", "COMPLETE.json"))):
        try:
            with open(path) as handle:
                marker = json.load(handle)
            task = marker["task"]
            tag = task["tag"]
            model = task["model"]
            if model not in spec["models"]:
                continue
        except (OSError, ValueError, KeyError):
            continue
        level = summary.get(tag, {})
        def number(key):
            try:
                value = float(level[key])
                return value if math.isfinite(value) else None
            except (ValueError, TypeError, KeyError):
                return None
        rows.append((study_id, spec["prefix"]+tag, model, "full",
                     "target_binned_return_20", None, number("n_dates"),
                     number("corr"), number("mmc"), number("payout"),
                     number("corr_sharpe"), number("corr_maxdd"), None,
                     time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))))
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


STUDIES["E182"] = {
    "title": "Volatility replacement: ~1M-parameter MLP at 1x and 10x epoch schedule",
    "idea": "Add-on to the volatility-replacement study (experiments/volatility_proxies.md): ReLU MLP widths 984x984 "
            "(~1M params, overparameterised) trained on CUDA at the size-dependent recipe schedule x1 and x10 "
            "(overcomputation), on original full / original matched / replacement-volatility matched v2.0 inputs, "
            "11 development folds to 2026-02-19. Ridge/LightGBM reused from results/volatility_replacement_mlp_10x. "
            "Writes OUTSIDE results/metrics/: results/volatility_replacement_mlp_1m_{1x,10x}/, checkpointed every "
            "300 epochs; queue results/volatility_replacement_mlp_1m_1x/queue.sh. Report = the 10x report "
            "(results/volatility_replacement_mlp_10x/volatility_replacement_report.pdf) re-rendered with each run as --addon, "
            "keeping the prior 10x MLP. Launched 2026-11-16 on westeros GPU; registered after launch on 2026-11-17.",
    "runner": "src/analysis/volatility_replacement.py",
    "prefix": "e182_", "scout_prefix": None,
    "models": ["mlp"], "expected_arms": 0,
}


STUDIES["E184"] = {
    "title": "TabPFN + a new output layer, LoRA fine-tuned on an MMC objective for the TV epoch count",
    "idea": "src/analysis/tabpfn_lora_head_mmc.py: LoRA r=8 on every attention q/k/v/out projection (E102's set, "
            "backbone frozen) plus a NEW zero-initialised Linear(192->1) on each query row's final hidden state (the "
            "input of TabPFN's output_projection), added as a residual to the bar-distribution mean, so step 0 is "
            "zero-shot. Loss = 1 - BMC against the real meta model (E45's mmc_loss), one date's cross-section per "
            "step, context = 130 dates before the T-20 wall (4096 rows). Leak-safe meta-covered training dates only: "
            "38 dates 2024-05-07..2024-06-27, 11,400 rows. Epochs = Timmermann-Vulicevic floor(0.2*P^0.8)+10 with "
            "P = 295,105 trainable (LoRA 294,912 + layer 193) -> 4,763 epochs, 180,994 steps, no early stopping; "
            "measured 375 s/epoch on one RTX 2070 -> ~21 days. Walk-forward eval on E19's 510 dates at epochs "
            "1,3,10,30,100,300,1000,3000 (descriptive) and 4763 (pre-specified result): e184_ep<N> (LoRA+layer), "
            "e184_lora_ep<N> (layer dropped), e184_zs (matched n_est=1 zero-shot). Writes forecasts/metrics under "
            "the tabpfn_head model tree (results/*/crypto/tabpfn_head/), resumable state models/tabpfn/e184/state.pt. "
            "deepfive GPU via src.cluster (experiments/e184_tabpfn_lora_head_mmc.jobs).",
    "runner": "src/analysis/tabpfn_lora_head_mmc.py",
    "prefix": "e184_", "scout_prefix": None,
    "models": ["tabpfn_head"], "expected_arms": 19,
}

STUDIES["E185"] = {
    "title": "IA_01..IA_04 sklearn MLPs at 1x and 10x Timmermann-Vulicevic epochs",
    "idea": "Paired retrain of the four live MLP legs (sklearn MLPRegressor, pinned registry configs, IA_01 from its "
            "November cache) on 4 rolling holdouts (130 train / 22 gap / 30 val / 130 test dates, 520 scored): current "
            "recipe vs fixed 1x TV reference epochs vs 10x final vs 10x validation-selected checkpoint, same weights path. "
            "Writes OUTSIDE results/metrics/: results/ia_mlp_10x/ (run.py, build_report.py, mlp_10x_report.pdf). "
            "Ran 2026-11-16 on westeros; registered after the fact on 2026-11-17.",
    "runner": "results/ia_mlp_10x/run.py",
    "prefix": "e185_", "scout_prefix": None,
    "models": ["mlp"], "expected_arms": 0,
}

STUDIES["E186"] = {
    "title": "IA_01..IA_04 as torch MLPs (sklearn init) trained on an MMC loss at TV reference epochs",
    "idea": "Same data, folds and pinned configs as E185, but the network is TorchMLPRegressor._build (sklearn "
            "Glorot init, float64, Adam, sklearn L2) trained for exactly the TV reference epochs "
            "max(floor(0.2 P^0.8),1)+10 with no early stopping, on whole-date batches. Arms: per-date negative BMC "
            "against the real meta model (torch_mmc) vs half-MSE in the same loop (torch_mse), 5 seeds each; paired "
            "against E185's sklearn 'before' and 'reference' arms on the same 520 dates. Fold 0 trains 66% of rows on "
            "dates before the meta model exists, where the loss reduces to per-date CORR. Writes OUTSIDE "
            "results/metrics/: results/e186_ia_torch_mmc/.",
    "runner": "experiments/e186_ia_torch_mmc.py",
    "prefix": "e186_", "scout_prefix": None,
    "models": ["torch_mlp"], "expected_arms": 0,
}


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

