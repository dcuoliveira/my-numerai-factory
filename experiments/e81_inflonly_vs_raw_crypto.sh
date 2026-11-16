#!/usr/bin/env bash
#
# E81 -- inflation block on the RAW baseline: raw 22 + 12 breakeven feats (34f) vs raw v2.0.
#
# crypto/v2.0-infl (rebuilt 2026-09-29 16:18) = raw 22 crypto features + 12 breakeven-
# inflation features (feature_macro_bei_* : 5y/10y/30y/5y5y levels + 20d/60d changes),
# a clean SUPERSET of raw (nothing dropped). The 12 BEI features are DAY-LEVEL, but the
# raw 22 restore the cross-section (289 distinct feature-vectors/date), so this is NOT
# degenerate. It is the clean isolation of inflation's marginal value on the DEPLOYED
# baseline -- the better-controlled sibling of E80 (which buried inflation in the 61f
# momswap-macro stack). Ridge stays inert on the 12 day-level columns (they add nothing
# to within-date ranking) so ridge ~ raw; trees/MLP can use inflation as a regime tilt.
#
# 5 models from scratch at the shared protocol; reported 3-way: vs raw v2.0 (e72_*_base,
# the requested baseline) and vs momswap-macro-infl (e80_*_infl, E80 -- same 12 BEI feats
# on the full 61f stack), both reused. Shared mommask_core.sh, VARIANT=inflraw.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E81 PREFIX=e81 VARIANT=inflraw
export MM_DATA=crypto/v2.0-infl MM_FEATS=34
export PARENT_PREFIX=e80 PARENT_SUFFIX=infl PARENT_LABEL=mm-macro-infl
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
