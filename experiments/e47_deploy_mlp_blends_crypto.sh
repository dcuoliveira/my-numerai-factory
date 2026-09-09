#!/usr/bin/env bash
#
# Experiment E47: deploy the E33 winner as a live MLP and two MLP-containing
# blends. E47 is the reference handle; every priced artefact is e47_*.
#
# WHAT THIS IS
#   Not a sweep -- a DEPLOYMENT. E33's top fenccausal cell (128,64 / relu /
#   lr 1e-3 / ALPHA 10, the regularisation E17's 0.1 ceiling could not reach) is
#   wired live as the pinned MLP slot IA_02 (src/live/registry.py), and two
#   slot-blends are built on it:
#     CU_SA_IA_BO_02 = IA_02 + CURUPIRA_05_60 + CU_SA_IA_BO_01   (a version bump
#         of CU_SA_IA_BO_01: same Curupira+Saci+Iara footprint, a 2nd neural leg)
#     CU_CA_IA_01    = IA_02 + IAUARETE_01 + CURUPIRA_05_60      (Curupira +
#         Caipora/CatBoost + Iara -- CURUPIRA_CAIPORA's footprint plus a neural leg)
#
# WHAT THIS RUNNER DOES
#   Reproduces the two blends' DEPLOYABLE numbers -- the real per-period
#   rank-average of the members' finished forecasts, the same run_ensemble the
#   report screens, NOT the screened payout-series proxy (which read dd(pay) 0.28
#   where the real number is ~1.0). These two runs are the slots' backtest_run,
#   and the source experiment_db counts as E47's arms.
#
# THE FINDING (priced, 470 periods, vs each blend's best member):
#   CU_SA_IA_BO_02  payout 0.10635 ~= CU_SA_IA_BO_01 0.10712 (-0.0008, noise) but
#     c.sharpe 2.03 vs 1.92, dd(pay) 1.02 vs 1.16, calmar 0.104 vs 0.093,
#     burn 30 vs 38 -- a risk upgrade at unchanged level.
#   CU_CA_IA_01     payout 0.10422 OUT-LEVELS all three members (~0.095 each) AND
#     c.sharpe 2.05 -- its mlp/catboost/ridge legs are genuinely diverse.
#
# Usage:
#   bash experiments/e47_deploy_mlp_blends_crypto.sh
#   DRY_RUN=1 bash experiments/e47_deploy_mlp_blends_crypto.sh   # print, don't run
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-.}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

TGT="target_binned_return_20"           # the payable label; members carry 20/60-day
NEURAL="mlp/e33_fenc_a128_64_relu_lre3_al10"   # IA_02's backtest cell

# Each arm: OUTPUT TREE (= the slot's scored_under) then its members.
run_blend() {
    local out_tree="$1" name="$2"; shift 2
    echo "[E47] ${name}  (${out_tree})  <- $*"
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    python3 -m src.run_ensemble crypto "${out_tree}" "$@" --target "${TGT}" -n "${name}" -q
}

# CU_SA_IA_BO_02, scored_under lgb: IA_02 + CURUPIRA_05_60 + CU_SA_IA_BO_01
run_blend lgb e47_cusaiabo02 \
    "${NEURAL}" ridge_glmnet/t60_c05_blend boto01_slot_mlpfenc

# CU_CA_IA_01, scored_under ridge_glmnet: IA_02 + IAUARETE_01 + CURUPIRA_05_60
run_blend ridge_glmnet e47_cucaia01 \
    "${NEURAL}" catboost_ktz/e32_sel_fenccausal t60_c05_blend

[ "${DRY_RUN:-0}" = "1" ] && exit 0
echo
echo "[E47] priced. Board + the live wiring:"
echo "  python3 -m src.analysis.results_table --model lgb --runs e47_cusaiabo02 ridge_glmnet/e47_cucaia01"
echo "  python3 -c 'from src.analysis import slots; print([s.name for s in slots.slot_rows(\"all\")[0] if s.name in (\"CU_SA_IA_BO_02\",\"CU_CA_IA_01\")])'"
