#!/usr/bin/env bash
#
# E127 -- E121 (the training window of IAUARETE_01, catboost_ktz on fenccausal)
#         FITTED on the 60-day label and SCORED on the 20-day one. The catboost
#         twin of E124 (ridge) and E125 (MLP).
#
# Same seven windows, same exhaustive 12-cell KTZ search and --tune-window 130
# as E121; only the label (target_binned_return_60) and the embargo it forces
# (60, not 20) change. The arms live in E121's runner under TW_TARGET=60 so the
# two halves cannot drift apart; this file only names the study. Runs: e127_w<N>.
#
# At embargo 60 a 130-period tuning block keeps the LAST TWO of its five inner
# folds (validation only fits past period ~78) -- exactly what production's own
# 60-day search does (CURUPIRA_05_60: window 130, embargo 60). The 12 KTZ cells
# are static, so the search still ranks all twelve; it just does so on two slices.
#
# The deliverable is the consolidated table in src.analysis.e127_report: all 14
# arms (E121's fit-20, E127's fit-60) RE-SCORED on target_binned_return_20 by
# src.analysis.cross_target.score_on_payable_target, on the dates they share,
# paired at Newey-West lag 20 (the scored label's overlap) and 60 (conservative).
#
#   bash experiments/e127_catboost_train_window_t60_crypto.sh          # all 7, then point to the report
#   ONLY=e127_w5 bash experiments/e127_catboost_train_window_t60_crypto.sh   # one arm
#   E121_SCOUT=1 / DRY_RUN=1 / NO_MAIL=1 as in E121's runner
#
exec env TW_TARGET=60 bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/e121_catboost_train_window_crypto.sh" "$@"
