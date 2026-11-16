#!/usr/bin/env bash
#
# E108 -- ALL momentum/trend families in one dataset vs raw v2.0:
#         crypto/v2.0-allmom-fillmask (57f) vs crypto/v2.0 (22f).
#
# Assembled with build_combo (union of already-built, already-binned blocks onto the
# raw v2.0 spine -- no recomputation, leak-safe): raw 22 (incl the 2 base
# feature_momentum_) + feature_mom_ 17 (litmom literature momentum: cumret horizons,
# 12-1, MAX, vol-scaled, Tzotchev robust trend) + feature_xsmom_ 3 (cross-sectional
# momentum mean/var/skew, day-level) + feature_tf_ 15 (Carver/Clenow trend: EWMAC,
# breakout, Bollinger-z, Clenow). First dataset to hold every momentum family at once.
#
# 5 models from scratch at the shared protocol (rolling 130/130, embargo auto=20,
# 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20); reported 3-way
# (CORR/MMC/payout/c.sharpe/calmar/n per arm):
#   - vs raw v2.0 (e72_*_base)   -- the requested comparison
#   - vs the vol block (e74_*_vol) -- standard secondary.
# Reuses the generic 3-way runner mommask_core.sh with VARIANT=allmom.
# Prior: momentum has been null-to-negative vs raw (E64-E67); pooling every variant
# lets a tree pick among them, but the raw 22 already absorb most of the signal.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E108 PREFIX=e108 VARIANT=allmom
export MM_DATA=crypto/v2.0-allmom-fillmask MM_FEATS=57
export PARENT_PREFIX=e74 PARENT_SUFFIX=vol PARENT_LABEL=vol
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
