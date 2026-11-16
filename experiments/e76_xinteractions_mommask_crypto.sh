#!/usr/bin/env bash
# E76: v2.0-momswap-macro-xinteractions-mommask (219f) vs xinteractions (e71_*_xint) vs v2.0 (e72_*_base).
export EXPERIMENT=E76 PREFIX=e76 MM_DATA=crypto/v2.0-momswap-macro-xinteractions-mommask MM_FEATS=219 \
       PARENT_PREFIX=e71 PARENT_SUFFIX=xint PARENT_LABEL=xinteractions V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(dirname "${BASH_SOURCE[0]}")/mommask_core.sh"
