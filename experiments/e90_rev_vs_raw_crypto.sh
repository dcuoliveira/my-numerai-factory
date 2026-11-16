#!/usr/bin/env bash
#
# E90 -- reversal block vs raw v2.0: crypto/v2.0-rev (36f) vs crypto/v2.0 (22f).
#
# crypto/v2.0-rev = raw 22 + 14 reversal/mean-reversion features (feature_rev_*):
# short-term reversal (str_1/7/14d), MAX/lottery effect (max1/max5_28d), idiosyncratic
# vol (ivol_28d), return skew (skew_28d), Amihud illiquidity (amihud_28d), and
# mean-reversion oscillators (rsi_14d, bbz_20d, stoch_14d, cci_20d, dist_ma_20/50d).
# Clean superset of raw, all per-symbol (cross-sectional, not day-level -> scoreable).
#
# 5 models from scratch at the shared protocol (rolling 130/130, embargo auto=20,
# 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20); reported 3-way:
#   - vs raw v2.0 (e72_*_base)   -- the requested comparison
#   - vs the vol block (e74_*_vol, raw+22 vol estimators) -- free secondary: does
#     reversal beat the leading prior raw add-on (vol was +0.0053 CORR on catboost)?
# Reuses the generic 3-way runner mommask_core.sh with VARIANT=rev.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E90 PREFIX=e90 VARIANT=rev
export MM_DATA=crypto/v2.0-rev MM_FEATS=36
export PARENT_PREFIX=e74 PARENT_SUFFIX=vol PARENT_LABEL=vol
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
