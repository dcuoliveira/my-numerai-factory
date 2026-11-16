#!/usr/bin/env bash
# E75: v2.0-momswap-macro-mommask (49f) vs momswap-macro (e70_*_base) vs v2.0 (e72_*_base).
# mommask masks base(17%)+momentum(10%)+macro(34%) to 0.5, concentrated on momentum-missing rows.
export EXPERIMENT=E75 PREFIX=e75 MM_DATA=crypto/v2.0-momswap-macro-mommask MM_FEATS=49 \
       PARENT_PREFIX=e70 PARENT_SUFFIX=base PARENT_LABEL=momswap-macro V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(dirname "${BASH_SOURCE[0]}")/mommask_core.sh"
