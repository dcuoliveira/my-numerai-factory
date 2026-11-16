#!/usr/bin/env bash
#
# E82 -- kitchen-sink combo: raw + vol + inflation + carver + interactions (753f) vs raw v2.0.
#
# crypto/v2.0-vol-infl-carver-ix-fillmask (753f), a clean SUPERSET of raw:
#   raw            22   (the deployed base)
#   feature_vol_   22   (realised-volatility estimators, E74)
#   feature_macro_bei_ 12 (breakeven-inflation, E80/E81)
#   feature_tf_    15   (Carver/Clenow trend, E72)
#   feature_ix_   682   (interactions -- dominates the count)
# fillmask = missing cells pre-filled to 0.5 (== the loader's own fill; no non-fillmask
# twin exists, so this is the canonical form). NOT within-date degenerate (300 distinct
# feature-vectors/date). The "does stacking every individually-null block finally beat
# raw" test. Prior (E70 925f, E71/E79 219f xint): a huge interaction count DROWNS ridge
# and hurts MLP while trees stay ~flat; unlikely to beat raw.
#
# 5 models from scratch at the shared protocol; reported 3-way: vs raw v2.0 (e72_*_base,
# the requested baseline) and vs momswap-macro (e70_*_base, a mid-size engineered stack
# for reference -- not nested), both reused. Shared mommask_core.sh, VARIANT=combo.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E82 PREFIX=e82 VARIANT=combo
export MM_DATA=crypto/v2.0-vol-infl-carver-ix-fillmask MM_FEATS=753
export PARENT_PREFIX=e70 PARENT_SUFFIX=base PARENT_LABEL=momswap-macro
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
