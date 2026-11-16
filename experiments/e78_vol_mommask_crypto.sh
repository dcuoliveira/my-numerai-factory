#!/usr/bin/env bash
# E78: v2.0-vol-mommask (44f) vs vol (e74_*_vol) vs v2.0 (e72_*_base).
# mommask gates on the 22 feature_vol_ estimators (require all present) and sets the
# WHOLE row (all 44 features) to neutral 0.5 on the 17.5% of rows missing any of them
# -- the no-spot names plus rows with <120d history. Same recipe (build_mommask
# --gate created --require all --neutralize all) as E73/E75/E76/E77, keyed on the vol
# block. NOTE the parent cells need E74's vol arms on disk; run E74 first for the
# 3-way (the mommask arms themselves run standalone).
export EXPERIMENT=E78 PREFIX=e78 MM_DATA=crypto/v2.0-vol-mommask MM_FEATS=44 \
       PARENT_PREFIX=e74 PARENT_SUFFIX=vol PARENT_LABEL=vol V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(dirname "${BASH_SOURCE[0]}")/mommask_core.sh"
