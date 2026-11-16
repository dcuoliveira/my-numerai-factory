#!/usr/bin/env bash
#
# E128 -- E122 (the training window of the best standalone RF, rf_ktz on
#         crypto/v2.0-tenc-onehot) FITTED on the 20-day label. The fit-20 half
#         that E122 (fit 60) lacks, so RF gets the same fit-20 x fit-60 x
#         raw/residualised table as ridge (E117/E124), the MLP (E119/E125) and
#         catboost (E121/E127).
#
# Same seven windows, same 40-draw rf_ktz search, 5 inner folds, refit 130 and
# --tune-window 130 as E122; only the label (target_binned_return_20) and the
# embargo it forces (20, not 60) change. The arms live in E122's runner under
# TW_TARGET=20 so the two halves cannot drift apart; this file only names the
# study. Runs: e128_w<N>.
#
# The deliverable is src.analysis.e128_report: all 14 arms (E128 fit-20, E122
# fit-60), raw AND meta-residualised (the live "resid": 0.5 transform, applied to
# the finished forecasts, nothing refitted), all scored on target_binned_return_20
# on the dates they share, plus the pairwise prediction / payout correlations.
#
#   bash experiments/e128_rf_window_t20_crypto.sh            # all 7, then the native report
#   ONLY_W="5 130" / DRY_RUN=1 / NO_MAIL=1 / RFK_POOL as in E122's runner
#
exec env TW_TARGET=20 bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/e122_rf_window_crypto.sh" "$@"
