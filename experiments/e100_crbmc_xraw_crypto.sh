#!/usr/bin/env bash
#
# E100 -- crbmc interactions from RAW products vs from BINNED products (vs E101 xbin),
#         and vs raw v2.0. crypto/v2.0-crbmc-xraw-fillmask (1469f).
#
# A/B on interaction CONSTRUCTION: xraw = feature_x2_ = product of the RAW (continuous,
# pre-bin) parent values, THEN rank-binned. Same 1469 columns as xbin (crbmc base 82 +
# 1387 feature_x2_), but different VALUES -- the raw-product and binned-product versions
# of a pair are only weakly / negatively correlated (-0.16..-0.49), because binning the
# parents first discards the magnitude the product needs. 3-way:
#   - vs crbmc-xbin (e101_*_xbin, product-of-binned) -- the headline A/B
#   - vs raw v2.0 (e72_*_base)                        -- the baseline anchor
# 5 models from scratch, shared protocol, mommask_core.sh VARIANT=xraw. 1469f -> ~5h.
# (E101's arms may be unfinished when this report writes -> that cell blank; regenerate
#  the xraw-vs-xbin paired delta once both land.)
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E100 PREFIX=e100 VARIANT=xraw
export MM_DATA=crypto/v2.0-crbmc-xraw-fillmask MM_FEATS=1469
export PARENT_PREFIX=e101 PARENT_SUFFIX=xbin PARENT_LABEL=crbmc-xbin
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
