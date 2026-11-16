#!/usr/bin/env bash
#
# E98 -- crbmc + FULL interactions vs raw (and vs the curated variant E97).
#
# crypto/v2.0-crbmc-xfull-fillmask (2856f) = the crbmc combined-signal base
# (raw 22 + coint 4 + rev 14 + basis 6 + funding 4 + mom 17 + carver/tf 15 = 82)
# + 2774 FULL pairwise interactions (feature_x2_). Compared 3-way:
#   - vs raw v2.0 (e72_*_base)                 -- the requested baseline
#   - vs crbmc-xCURATED (e97_*_xcur, 98f)      -- full vs curated interaction expansion
# so E98 (full) and E97 (curated) each pair against raw AND against each other.
# 5 models from scratch, shared protocol, mommask_core.sh VARIANT=xfull.
# BIGGEST feature set run to date (2856f); 5 arms sequential -> budget ~8-10h.
# Prior: the 2774-interaction expansion should DROWN ridge/mlp (cf. E70 925f, E85
# 1506f); the question is whether curated (E97) holds up where full collapses.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E98 PREFIX=e98 VARIANT=xfull
export MM_DATA=crypto/v2.0-crbmc-xfull-fillmask MM_FEATS=2856
export PARENT_PREFIX=e97 PARENT_SUFFIX=xcur PARENT_LABEL=crbmc-xcurated
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
