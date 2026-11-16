#!/usr/bin/env bash
#
# E91 -- cointegration block vs raw v2.0: crypto/v2.0-coint-fillmask (26f) vs v2.0 (22f).
#
# crypto/v2.0-coint-fillmask = raw 22 + 4 cointegration/error-correction features
# (feature_coint_*): residual z-score at 60d/120d (distance from the cointegrating
# relationship), mean-reversion half-life (120d), and the ADF stationarity stat (120d).
# A stat-arb / mean-reversion signal. fillmask = missing cells pre-filled to 0.5
# (== the loader's own fill; so as-loaded == a raw-coint dataset). Clean superset of
# raw, per-symbol (binned, ~5 values/date -> scoreable, not day-level degenerate).
#
# 5 models from scratch at the shared protocol (rolling 130/130, embargo auto=20,
# 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20); reported 3-way:
#   - vs raw v2.0 (e72_*_base)   -- the requested comparison
#   - vs the vol block (e74_*_vol) -- free secondary (the leading prior raw add-on),
#     same reference as E90 so the reversal and cointegration blocks are comparable.
# Reuses the generic 3-way runner mommask_core.sh with VARIANT=coint.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E91 PREFIX=e91 VARIANT=coint
export MM_DATA=crypto/v2.0-coint-fillmask MM_FEATS=26
export PARENT_PREFIX=e74 PARENT_SUFFIX=vol PARENT_LABEL=vol
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
