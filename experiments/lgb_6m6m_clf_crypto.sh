#!/usr/bin/env bash
#
# LightGBM, CLASSIFICATION, rolling 6-month window / 6-month refit, crypto.
#
# Nine arms trained from scratch pseudo-OOS: raw 22 -> allfeat++ 812 -> the
# seven raw feature groups, each fitted alone. Same protocol as every ridge arm
# (50 draws, 5 purged folds, seed 42), so the results are directly comparable.
#
# Classification binarises the binned target at 0.5 and uses the positive-class
# probability as the signal. That throws away the ordering inside each half,
# which cost ridge a little CORR -- the open question is whether it costs a
# booster the same, since a tree does not model the target's scale anyway.
#
# Budget ~8 h on a free 32-core machine; see experiments/lgb_6m6m_crypto.sh for
# the timing measurements and for every knob (ONLY, SKIP, SHARD/OF, SMOKE,
# DRY_RUN), which pass through.
#
#   bash experiments/lgb_6m6m_clf_crypto.sh
#   SMOKE=1 bash experiments/lgb_6m6m_clf_crypto.sh
#
# Under screen:
#   screen -L -Logfile lgb-w6r6-clf.txt bash experiments/lgb_6m6m_clf_crypto.sh
#
set -uo pipefail
exec env LGB_TASK=classification \
    bash "$(dirname "${BASH_SOURCE[0]}")/lgb_6m6m_crypto.sh" "$@"
