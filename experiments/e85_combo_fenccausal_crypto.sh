#!/usr/bin/env bash
#
# E85 -- causal per-symbol means on the kitchen sink: combo+fenccausal (1506f) vs combo (753f).
# (Numbered E85 not E83: a concurrent session took E83/E84 for TabPFN studies -- id-collision
#  hazard, same class as the E74 clash. Always check the next free id when another session runs.)
#
# crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal (1506f), a clean SUPERSET of the E82
# combo: it appends a CAUSAL per-symbol expanding-mean twin (_symmean) of EVERY one of the
# 753 combo features, doubling the set --
#   raw 22->44, feature_vol_ 22->44, feature_macro_bei_ 12->24, feature_tf_ 15->30,
#   feature_ix_ 682->1364.
# Each _symmean is the mean of that symbol's OWN PAST values of the feature (a row never
# enters its own encoding), so it is leak-safe by construction -- the same fenccausal idea
# behind IA_01 (the best MLP slot), here applied to the whole 753f stack instead of raw 22.
#
# Biggest feature set run to date (1506f); budget ~2x E82's per-arm cost. 5 models from
# scratch at the shared protocol; reported 3-way: vs combo (e82_*_combo, the requested
# baseline) and vs raw v2.0 (e72_*_base, free context), both reused. Shared mommask_core.sh,
# VARIANT=fenc. Prior: the combo already trailed raw (interactions drown signal); doubling
# to 1506f is unlikely to beat the combo, though the raw-feature _symmeans are the one block
# (fenccausal/IA_01) that has genuinely helped an MLP before, so watch the mlp arm.
set -euo pipefail
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
export EXPERIMENT=E85 PREFIX=e85 VARIANT=fenc
export MM_DATA=crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal MM_FEATS=1506
export PARENT_PREFIX=e82 PARENT_SUFFIX=combo PARENT_LABEL=combo
export V0_PREFIX=e72 V0_SUFFIX=base
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mommask_core.sh"
