#!/usr/bin/env bash
#
# E97 -- crbmc + CURATED interactions vs raw (and vs the full-interaction variant E98).
#
# crypto/v2.0-crbmc-xcurated-fillmask (98f) = the crbmc combined-signal base
# (raw 22 + coint 4 + rev 14 + basis 6 + funding 4 + mom 17 + carver/tf 15 = 82)
# + 16 CURATED pairwise interactions (feature_x2_). Compared 3-way:
#   - vs raw v2.0 (e72_*_base)                 -- the requested baseline
#   - vs crbmc-xFULL (e98_*_xfull, 2856f)      -- curated vs full interaction expansion
# so E97 (curated) and E98 (full) each pair against raw AND against each other.
# 5 models from scratch, shared protocol, mommask_core.sh VARIANT=xcur.
# (E98's arms may not exist yet when E97's report writes -> that cell is blank;
#  regenerate the curated-vs-full paired delta once both finish.)
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E97 PREFIX=e97 VARIANT=xcur
export MM_DATA=crypto/v2.0-crbmc-xcurated-fillmask MM_FEATS=98
export PARENT_PREFIX=e98 PARENT_SUFFIX=xfull PARENT_LABEL=crbmc-xfull
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
