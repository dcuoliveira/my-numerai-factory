#!/usr/bin/env bash
#
# E125 -- E119 (the training window of IA_01, mlp on fenccausal) FITTED on the
#         60-day label and scored on the 20-day one. The E124 of the MLP.
#
# Same seven windows, same from-scratch 50-draw search and --tune-window 130 as
# E119; only the label (target_binned_return_60) and the embargo it forces (60)
# change. The arms live in E119's runner under E119_TARGET=60 so the two halves
# cannot drift apart; this file only names the study. Runs: e125_w<N>.
#
# The deliverable is the consolidated table in src.analysis.e119_report: all 14
# arms (E119's fit-20, E125's fit-60) RE-SCORED on target_binned_return_20 by
# src.analysis.cross_target.score_on_payable_target, on the 470 dates they share,
# paired at Newey-West lag 20 and 60. e125_w130 must reproduce mlp/e60_mlp_fenccausal.
#
#   bash experiments/e125_ia01_train_window_t60_crypto.sh          # all 7, then the report
#   ONLY=e119_w5 bash experiments/e125_ia01_train_window_t60_crypto.sh   # one arm (cluster)
#   E119_SCOUT=1 / DRY_RUN=1 / NO_MAIL=1 / REPORT_ONLY=1 as in E119's runner
#
exec env E119_TARGET=60 bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/e119_ia01_train_window_crypto.sh" "$@"
