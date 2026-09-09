#!/usr/bin/env bash
# E28 -- per-subset from-scratch tuning + FARM over the full pool.
#
# THE QUESTION. The widest dataset partitions into 7 feature SUBSETS, each shipped
# as raw + one block: raw(22) tenc(23) onehot(762) rollsel(46) spread(33)
# interact(36) fenccausal(44). We want every model family tuned FROM SCRATCH in a
# pseudo-OOS manner on each subset, then a FARM combination over the whole pool.
#
# WHAT IS ALREADY DONE, AND WHAT IS NOT. A strict scan of results/metrics/ (config
# markers: fixed_params null, model_options empty, n_iter>=10 => a genuine search;
# ridge_glmnet cannot be pinned at all, models.py:347) found that 18 of the 21
# (family x subset) cells already carry a from-scratch search -- ridge_glmnet via
# the w6r6_glmnet_reg_* studies, mlp via E15, lgb via E13/E14. Exactly THREE gaps
# remain, and only these are re-tuned here:
#     ridge_glmnet x rollsel   (no glmnet run ever on the pure 46-col block)
#     mlp          x onehot    (E15 has no onehot arm)
#     lgb          x onehot    (only PINNED runs exist; fixed_params bypasses the
#                               search, so no from-scratch tune ever ran)
#
# THREE GRR ARMS on crypto/v2.0-onehot (raw 22 + 740 symbol dummies), added as
# extra base models:
#     grr_pos    all betas >= 0        (sign constraint, both blocks)
#     grr_neg    all betas <= 0
#     grr_symfe  onehot block ~unpenalised (p1e-6, symbol fixed effects); raw
#                keeps the searched alpha. p0 is forbidden -- the 740 dummies sum
#                to 1 and a literal-zero multiplier is singular (generalised_ridge
#                :331). alpha searched on the 1e-2..1e2 floor (E21/E26 convention),
#                structures pinned to identity.
#
# THE PSEUDO-OOS PROTOCOL is identical for every arm and matches E14/E15/E17:
#   rolling walk-forward, train-window 130 = refit-every 130 (~4 non-overlapping
#   refit blocks over the 510-period panel), embargo 20; inner 5-fold PURGED
#   chronological CV; 50-draw random search over the family's grid; metric
#   numerai_corr; seed 42; target_binned_return_20. One common window, so the
#   three new cells are directly comparable to each other and to the 18 existing.
#
# THEN FARM. run_farm_ensemble combines the best searched run per (family, subset)
# -- 21 members -- plus the 3 GRR arms = 24 base models, by equal (1/N),
# FARM1/2/3 x {ridge, l2boost, postalasso}, GR and DS. This is a genuine
# high-dimensional pool (24 members vs E27's 5), the regime the paper targets.
#
# COST. ridge_glmnet/rollsel ~30 min; lgb/onehot ~3-4 h; mlp/onehot ~11-12 h (762
# features, the long pole); 3 GRR ~30-60 min each. The mlp arm runs in the
# background while the rest run sequentially, so wall-time ~= the mlp arm.
#
# Usage:
#   bash experiments/e28_subset_tuning_farm_crypto.sh              # tune, then FARM
#   STAGE=tune bash experiments/e28_subset_tuning_farm_crypto.sh   # tuning only
#   STAGE=farm bash experiments/e28_subset_tuning_farm_crypto.sh   # FARM only (tunes must exist)
#   DRY_RUN=1 ...                                                  # print, do not run
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"

STAGE="${STAGE:-all}"
PREFIX="e28"
TREE="ridge_glmnet"            # FARM output tree (cosmetic for a blend)
COMMON="--task regression --metric corr --bmc-reference benchmark --cv-scheme rolling --train-window 130 --refit-every 130 --embargo 20 --n-splits 5 --n-iter 50 --seed 42 --val-frac 0.3"

tune_one() {   # name  "extra run_forecast args..."
  local name="$1"; shift
  if [ -f "results/metrics/crypto/$2/${name}.json" ]; then
    echo "  [skip] ${name} already on disk"; return 0
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "  would tune  ${name}  ($*)"; return 0; fi
  echo; echo "--- tune ${name}  $(date '+%H:%M:%S')"
  python3 -u -m src.run_forecast "$@" --run-name "${name}" 2>&1 | tee "logs/${name}.log"
}

# ---------------------------------------------------------------- TUNE ------ #
if [ "${STAGE}" = "tune" ] || [ "${STAGE}" = "all" ]; then
  echo "=============================================================="
  echo "E28 TUNE: 3 gap searches + 3 GRR arms  |  started $(date '+%F %T')"
  echo "=============================================================="

  # long pole first, in the background
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "  would tune  e28_mlp_onehot  (mlp crypto/v2.0-onehot, ~12h, backgrounded)"
  elif [ ! -f results/metrics/crypto/mlp/e28_mlp_onehot.json ]; then
    ( python3 -u -m src.run_forecast crypto mlp ${COMMON} --era-step 1 --search-jobs 1 \
        --data-version crypto/v2.0-onehot --run-name e28_mlp_onehot > logs/e28_mlp_onehot.log 2>&1 ) &
    MLP_PID=$!
    echo "  e28_mlp_onehot launched in background (pid ${MLP_PID})"
  else
    echo "  [skip] e28_mlp_onehot already on disk"; MLP_PID=""
  fi

  # the rest, sequentially (they finish well before the mlp arm)
  tune_one e28_glmnet_rollsel  crypto ridge_glmnet ${COMMON} --data-version crypto/v2.0-rollsel
  tune_one e28_grr_pos   crypto gen_ridge ${COMMON} --n-jobs 1 --search-jobs 4 --data-version crypto/v2.0-onehot \
      --model-options '{"structures":["identity"],"alpha_lo":-2,"alpha_hi":2,"group_specs":["raw:s+;onehot:s+"]}'
  tune_one e28_grr_neg   crypto gen_ridge ${COMMON} --n-jobs 1 --search-jobs 4 --data-version crypto/v2.0-onehot \
      --model-options '{"structures":["identity"],"alpha_lo":-2,"alpha_hi":2,"group_specs":["raw:s-;onehot:s-"]}'
  tune_one e28_grr_symfe crypto gen_ridge ${COMMON} --n-jobs 1 --search-jobs 4 --data-version crypto/v2.0-onehot \
      --model-options '{"structures":["identity"],"alpha_lo":-2,"alpha_hi":2,"group_specs":["onehot:p1e-6"]}'
  tune_one e28_lgb_onehot crypto lgb ${COMMON} --n-jobs 8 --search-jobs 4 --era-step 1 --data-version crypto/v2.0-onehot

  if [ "${DRY_RUN:-0}" != "1" ] && [ -n "${MLP_PID:-}" ]; then
    echo; echo "  waiting on e28_mlp_onehot (pid ${MLP_PID}) ..."; wait "${MLP_PID}" || echo "  >>> e28_mlp_onehot exited non-zero"
  fi
  echo "E28 TUNE done $(date '+%F %T')"
fi

# ---------------------------------------------------------------- FARM ------ #
if [ "${STAGE}" = "farm" ] || [ "${STAGE}" = "all" ]; then
  echo; echo "=============================================================="
  echo "E28 FARM over the tuned pool  |  started $(date '+%F %T')"
  echo "=============================================================="

  build_pool() {   # best SEARCHED run per (family, subset) on the 7 subsets + 3 GRR
    python3 - <<'PY'
import glob, json, os
import pandas as pd
from src.analysis.results_table import target_of
SUBSETS = {"crypto/v2.0","crypto/v2.0-tenc","crypto/v2.0-onehot","crypto/v2.0-rollsel",
           "crypto/v2.0-spread","crypto/v2.0-interact","crypto/v2.0-fenccausal"}
FAM = ("ridge_glmnet","lgb","mlp")
def is_search(model, cfg):
    if cfg.get("fixed_params"): return model == "ridge_glmnet"   # glmnet cannot be pinned
    if cfg.get("model_options"): return False
    return int(cfg.get("n_iter") or cfg.get("bayes_init_points") or 0) >= 10 or model == "ridge_glmnet"
best = {}
for path in sorted(glob.glob("results/metrics/crypto/*/*.json")):
    model = path.split(os.sep)[-2]; run = os.path.basename(path)[:-5]
    if model not in FAM or "clf" in run: continue
    cfg = json.load(open(path)).get("config", {})
    if cfg.get("members"): continue
    ds = str(cfg.get("feature_set") or cfg.get("data_version"))
    if ds not in SUBSETS: continue
    if not is_search(model, cfg): continue
    if target_of("crypto", model, run) != "target_binned_return_20": continue
    pp = "results/per_period/crypto/%s/%s.csv" % (model, run)
    if not os.path.exists(pp): continue
    f = pd.read_csv(pp)
    if "date" in f.columns: f = f.set_index("date")
    if "corr" not in f.columns or len(f) < 470: continue
    if any(k in run.lower() or k in ds.lower() for k in ("leak", "fencglobal")): continue
    c = float(f["corr"].mean()); key = (model, ds)
    if key not in best or c > best[key][1]: best[key] = ("%s/%s" % (model, run), c)
members = [v[0] for v in best.values()]
for grr in ("e28_grr_pos", "e28_grr_neg", "e28_grr_symfe"):
    if os.path.exists("results/metrics/crypto/gen_ridge/%s.json" % grr):
        members.append("gen_ridge/%s" % grr)
print(" ".join(members))
PY
  }

  MEMBERS="$(build_pool)"
  NMEM=$(echo ${MEMBERS} | wc -w)
  echo "pool: ${NMEM} base models"
  for m in ${MEMBERS}; do echo "   ${m}"; done
  if [ "${NMEM}" -lt 6 ]; then echo ">>> pool too small (${NMEM}); tunes not on disk yet?"; exit 1; fi

  ARMS="equal farm1:ridge farm1:l2boost farm1:postalasso farm2:ridge farm2:l2boost farm2:postalasso farm3:ridge farm3:l2boost farm3:postalasso gr ds"
  for arm in ${ARMS}; do
    method="${arm%%:*}"; beta="${arm#*:}"; [ "${beta}" = "${arm}" ] && beta="ridge"
    run_name="${PREFIX}_farm_${arm/:/_}"
    if [ "${DRY_RUN:-0}" = "1" ]; then printf '  would run FARM  %-26s %s\n' "${run_name}" "${method}/${beta}"; continue; fi
    echo; echo "--- ${run_name}"
    python3 -u -m src.run_farm_ensemble crypto "${TREE}" ${MEMBERS} \
        --method "${method}" --beta-method "${beta}" --run-name "${run_name}" 2>&1 | tee "logs/${run_name}.log"
  done
  [ "${DRY_RUN:-0}" = "1" ] && exit 0

  # paired-vs-equal report
  REPORT="logs/${PREFIX}_report.txt"
  { echo "E28 -- FARM over the 24-model per-subset + GRR pool"; echo
    python3 - "${TREE}" "${PREFIX}" <<'PY'
import os, sys
import numpy as np, pandas as pd
from src.analysis.results_table import payout_series
tree, prefix = sys.argv[1], sys.argv[2]
def pp(run):
    p = "results/per_period/crypto/%s/%s.csv" % (tree, run)
    return pd.read_csv(p).set_index("date") if os.path.exists(p) else None
def nw(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m; v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1): v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n); return m, (m / se if se > 0 else float("nan"))
base = pp("%s_farm_equal" % prefix)
if base is None: print("no equal baseline"); sys.exit()
print("paired vs equal (1/N), Newey-West lag 20")
print("%-22s %8s %8s %10s %10s" % ("arm", "CORR", "payout", "dPay(t)", "dCORR(t)"))
rows = []
import glob
for path in sorted(glob.glob("results/metrics/crypto/%s/%s_farm_*.json" % (tree, prefix))):
    run = os.path.basename(path)[:-5]; arm = run[len("%s_farm_" % prefix):]
    f = pp(run)
    if f is None: continue
    c = f.index.intersection(base.index)
    dpay = payout_series(f).reindex(c) - payout_series(base).reindex(c)
    dcor = f["corr"].reindex(c) - base["corr"].reindex(c)
    mp, tp = nw(dpay.dropna()); mc, tc = nw(dcor.dropna())
    rows.append((arm, f["corr"].mean(), float(payout_series(f).mean()), mp, tp, mc, tc, arm == "equal"))
for arm, corr, pay, mp, tp, mc, tc, eq in sorted(rows, key=lambda r: -r[2]):
    print("%-22s %8.4f %8.4f %+8.4f(%+.2f) %+.4f(%+.2f)%s"
          % (arm, corr, pay, mp, tp, mc, tc, "  <- baseline" if eq else ""))
PY
  } | tee "${REPORT}"
  echo "wrote ${REPORT}"
fi
