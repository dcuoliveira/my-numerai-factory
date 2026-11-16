#!/usr/bin/env bash
#
# E80 -- inflation-expectations (breakeven) macro block vs raw v2.0 (and momswap-macro).
#
# crypto/v2.0-momswap-macro-infl (61f) = momswap-macro (49f) + 12 breakeven-inflation
# (BEI) features: market-implied inflation expectations at 5y / 10y / 30y / 5y5y-forward
# plus their 20d and 60d changes. feature_macro_ prefix, DAY-LEVEL broadcast (median 1
# distinct value per date, 0 NaN) -- the direct sibling of E69 (macro): inert for ridge
# on a within-date rank metric (negative control), can only help via tree/MLP interactions.
#
# 5 models from scratch at the shared protocol (rolling 130/130, embargo auto=20, 5 folds,
# val_frac 0.3, seed 42, n_iter 50, target_binned_return_20), reported 3-way:
#   - vs raw v2.0 (e72_*_base)          -- the requested "vs crypto/v2.0" baseline
#   - vs momswap-macro (e70_*_base)     -- isolates the marginal value of the 12 BEI feats
# Reuses the shared 3-way runner mommask_core.sh with VARIANT=infl.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E80 PREFIX=e80 VARIANT=infl
export MM_DATA=crypto/v2.0-momswap-macro-infl MM_FEATS=61
export PARENT_PREFIX=e70 PARENT_SUFFIX=base PARENT_LABEL=momswap-macro
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
