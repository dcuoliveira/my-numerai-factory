#!/usr/bin/env bash
#
# E101 -- crbmc interactions from BINNED products vs from RAW products (vs E100 xraw),
#         and vs raw v2.0. crypto/v2.0-crbmc-xbin-fillmask (1469f).
#
# A/B twin of E100. xbin = feature_x2_ = product of the already-BINNED {0,.25,.5,.75,1}
# parent values, THEN rank-binned -- a product of cross-sectional RANKS (magnitude
# discarded). Same 1469 columns as xraw, different values (corr to xraw -0.16..-0.49).
# 3-way: vs crbmc-xraw (e100_*_xraw) and vs raw v2.0 (e72_*_base). 5 models from
# scratch, shared protocol, mommask_core.sh VARIANT=xbin. 1469f -> ~5h.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E101 PREFIX=e101 VARIANT=xbin
export MM_DATA=crypto/v2.0-crbmc-xbin-fillmask MM_FEATS=1469
export PARENT_PREFIX=e100 PARENT_SUFFIX=xraw PARENT_LABEL=crbmc-xraw
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
