#!/usr/bin/env bash
#
# E79 -- xinteractions (219f) UN-GATED, via the fillmask (cell-level 0.5 fill).
#
# fillmask (build_mommask --mode fill) replaces ONLY the missing (NaN) cells with
# 0.5 and keeps every present value -- which is EXACTLY what the pipeline loader
# (fill_missing_features, neutral_value 0.5) already does at load time. Verified:
# crypto/v2.0-momswap-macro-xinteractions-fillmask loaded == raw
# crypto/v2.0-momswap-macro-xinteractions loaded, cell for cell (219f, 513,796 rows).
#
# So of the five fillmask datasets, four are byte-identical AS LOADED to runs we
# already have and are intentionally NOT re-run:
#   litmom-fillmask (39f)        == E65 e65_*_litmom
#   momswap-macro-fillmask (49f) == E70 e70_*_base
#   carver-fillmask (64f)        == E72 e72_*_carver
#   vol-fillmask (44f)           == E74 e74_*_vol
# xinteractions is the ONE exception: the dataset was rebuilt 73f -> 219f AFTER E71
# ran at 73f, so no un-gated 219f run exists. E76 ran the 219f GATED (mommask) block
# but reused the 73f e71_*_xint as "parent" -- a feature-count mismatch. E79 supplies
# the missing un-gated 219f arm and a clean 3-way: v2.0 (raw) / mommask (E76, gated)
# / fillmask (this, un-gated == raw 219f). Reuses mommask_core.sh with VARIANT=fillmask.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E79 PREFIX=e79 VARIANT=fillmask
export MM_DATA=crypto/v2.0-momswap-macro-xinteractions-fillmask MM_FEATS=219
export PARENT_PREFIX=e76 PARENT_SUFFIX=mommask PARENT_LABEL=mommask
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
