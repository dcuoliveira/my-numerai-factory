#!/usr/bin/env bash
#
# E93 -- betting-against-beta (short-constraint) block vs raw v2.0:
#        crypto/v2.0-bab (31f) vs crypto/v2.0 (22f).
#
# crypto/v2.0-bab = raw 22 + 9 BAB features (3 windows x {beta, beta x Amihud-
# illiquidity, beta x realised-vol}). The interactions operationalize Miller /
# Hong-Sraer "speculative betas": high-beta coins that are also hard-to-short
# (illiquid) and high-disagreement (high vol) are overpriced -> expected LOW
# forward return. Per-symbol, binned, point-in-time (lag 1) -- clean superset of raw.
#
# Distinct from the concurrent E92 (crypto/v2.0-beta, 25f, beta-only) -- E93 carries
# the short-constraint INTERACTIONS, which is the novel content.
#
# 5 models from scratch at the shared protocol (rolling 130/130, embargo auto=20,
# 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20); reported 3-way:
#   - vs raw v2.0 (e72_*_base)   -- the requested comparison
#   - vs the vol block (e74_*_vol) -- same secondary as E90/E91 (the leading prior
#     raw add-on), so reversal/cointegration/BAB are comparable.
# Reuses the generic 3-way runner mommask_core.sh with VARIANT=bab.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E93 PREFIX=e93 VARIANT=bab
export MM_DATA=crypto/v2.0-bab MM_FEATS=31
export PARENT_PREFIX=e74 PARENT_SUFFIX=vol PARENT_LABEL=vol
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
