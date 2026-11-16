#!/usr/bin/env bash
# E77: v2.0-momswap-macro-carver-mommask (64f) vs carver (e72_*_carver) vs v2.0 (e72_*_base).
export EXPERIMENT=E77 PREFIX=e77 MM_DATA=crypto/v2.0-momswap-macro-carver-mommask MM_FEATS=64 \
       PARENT_PREFIX=e72 PARENT_SUFFIX=carver PARENT_LABEL=carver V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(dirname "${BASH_SOURCE[0]}")/mommask_core.sh"
