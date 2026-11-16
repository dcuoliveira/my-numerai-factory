#!/usr/bin/env bash
#
# E89 -- SPONGE-cluster bake-off on the 1526f fenccausal-PAIRS combo (25 clusters).
#
# Sibling of E87 (753f combo, 10 clusters) and the concurrent E88 (1506f fenccausal,
# no pairs, 18 clusters). Dataset crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal-
# pairs (1526f) with 25 SPONGE signed-spectral clusters from its own
# sponge_feature_groups.json (k=25 by silhouette), partitioning all 1526 feats.
# 25 clusters x 5 models = 125 from-scratch runs; each cell paired (NW lag 20) vs raw
# v2.0 (e72_*_base, the deployed baseline -- no full-1526f run exists, so raw is the
# meaningful anchor). Reuses the env-driven core e87_sponge_bakeoff.py. CLUSTER_JOBS=4
# to coexist with E87; RAM is ample (235G free). ~10-15h background.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export SPONGE_DATA="crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal-pairs"
export SPONGE_SPEC="data/${SPONGE_DATA}/sponge_feature_groups.json"
export SPONGE_EXP=E89 SPONGE_PREFIX=e89
export SPONGE_BASE_PREFIX=e72 SPONGE_BASE_SUFFIX=base SPONGE_BASE_LABEL=raw-v2.0
export CLUSTER_JOBS="${CLUSTER_JOBS:-4}"
exec "$PYTHON" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/e87_sponge_bakeoff.py"
