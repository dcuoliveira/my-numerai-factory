#!/usr/bin/env bash
#
# E99 -- coint+rev+basis+macro stack vs raw v2.0:
#        crypto/v2.0-coint-rev-basis-macro-fillmask (68f) vs crypto/v2.0 (22f).
#
# Dataset (68f, clean superset of raw) = raw 22 + coint 4 (cointegration/error-
# correction) + rev 14 (reversal/mean-reversion) + basis 6 + funding 4 (perp carry /
# short-constraint) + macro 18 (gold/silver/S&P/DXY/US-yield-curve, day-level). It
# pools the per-symbol stat-arb/carry signals (coint/rev/basis/funding) with the
# day-level macro regime block -- so ridge is inert on the 18 macro cols (constant
# within date) and can only use the per-symbol 28; trees/MLP can regime-gate on macro.
#
# 5 models from scratch at the shared protocol (rolling 130/130, embargo auto=20,
# 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20); reported 3-way:
#   - vs raw v2.0 (e72_*_base)   -- the requested comparison
#   - vs the vol block (e74_*_vol) -- same secondary as E90/E91/E93/E94.
# Reuses the generic 3-way runner mommask_core.sh with VARIANT=crbm.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E99 PREFIX=e99 VARIANT=crbm
export MM_DATA=crypto/v2.0-coint-rev-basis-macro-fillmask MM_FEATS=68
export PARENT_PREFIX=e74 PARENT_SUFFIX=vol PARENT_LABEL=vol
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
