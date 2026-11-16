#!/usr/bin/env bash
#
# E94 -- futures-basis & funding block vs raw v2.0: crypto/v2.0-basis-fillmask (32f) vs v2.0.
#
# The FIRST genuinely new data class in this program -- perp funding + spot-futures
# basis, not a re-transform of the 22. crypto/v2.0-basis-fillmask = raw 22 + 10:
#   feature_basis_pct, basis_ma_20d, basis_z_20d, basis_z_60d, basis_chg_5d/20d
#   feature_funding_rate, funding_ma_7d, funding_ma_30d, funding_z_30d
# This is the DIRECT short-constraint / carry measure the E93 (BAB) post-mortem
# flagged as the real unlock: persistently positive funding = crowded longs /
# costly shorts = exactly where high-beta overpricing should concentrate; a coin
# with no listed perp (no funding) is neutral-filled, and that absence is itself
# the hard-to-short signal. Per-symbol, binned, point-in-time (fillmask = loader fill).
#
# 5 models from scratch at the shared protocol (rolling 130/130, embargo auto=20,
# 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20); reported 3-way:
#   - vs raw v2.0 (e72_*_base)   -- the requested comparison
#   - vs the vol block (e74_*_vol) -- same secondary as E90/E91/E93.
# Reuses the generic 3-way runner mommask_core.sh with VARIANT=basis.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E94 PREFIX=e94 VARIANT=basis
export MM_DATA=crypto/v2.0-basis-fillmask MM_FEATS=32
export PARENT_PREFIX=e74 PARENT_SUFFIX=vol PARENT_LABEL=vol
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
