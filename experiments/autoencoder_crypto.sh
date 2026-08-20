#!/usr/bin/env bash
#
# Experiment: bottleneck-autoencoder codes as extra features
#
# Every other derived block in this repo is a hand-specified transform -- the
# one-hot is identity, tenc is past target performance, the rolling means are
# recent standing, the spreads are horizon differences, the interactions are
# named products. Each encodes a hypothesis somebody wrote down. None of them
# ask the data what structure is already sitting in the 22 raw indicators.
#
# This arm family does. An autoencoder squeezes all 22 through a k-dimensional
# bottleneck and back; the k activations, rank-normalised within each date, are
# appended as feature_ae_00.. and everything downstream runs unchanged.
#
# Same fixed model as every other arm, so the feature set is the only thing
# varying:  ridge_glmnet, regression, rolling 6-month window, 6-month refit.
# That makes these directly comparable with the existing w6r6_glmnet_reg
# (raw 22) baseline already sitting in results/metrics/.
#
# ---------------------------------------------------------------------------
# What to expect, and what would count as a result
# ---------------------------------------------------------------------------
#
# The honest prior is FLAT. There are only 22 inputs, so there is very little to
# compress, and every "more features" arm in this study so far has landed flat
# or worse -- glmnet applies one alpha across all columns, so adding collinear
# ones pushes alpha up and shrinks the blocks that were working.
#
# Which is exactly why pca_plus_raw is not optional. It builds the same dataset
# through a linear projection with identical column names. Three outcomes:
#
#   ae ~ raw  and pca ~ raw    there is nothing in the 22 beyond what the model
#                              already extracts. Closes the question.
#   ae > raw  and pca ~ ae     the compression helped and the NONLINEARITY did
#                              not. Ship PCA and drop the torch dependency.
#   ae > raw  and ae > pca     the only outcome that justifies an autoencoder.
#
# One thing the search already settled, before any of these arms run. At the
# first-guess architecture (64,32) the autoencoder reconstructed WORSE than the
# linear PCA control, 0.904 against 0.932, which reads as "this panel is linear,
# drop torch". Twenty trials gave 0.954 at 256,128 -- above the linear ceiling
# of 0.927. So the linearity reading was a statement about the network, not
# about the data.
#
# That does NOT settle the arms below. The search maximises RECONSTRUCTION, and
# reconstruction is not what these are scored on. A tuned autoencoder that
# reconstructs far better and still does not beat pca_plus_raw is a clean
# answer, and ae_small against ae_plus_raw is what measures whether the
# reconstruction gain travels at all.
#
# Do NOT settle this on the reconstruction R^2 the builder prints. With swap
# noise on, the autoencoder is trained on corrupted rows and scored on clean
# ones while PCA is fitted on the clean rows it is scored on -- so PCA wins that
# number by construction (0.932 vs 0.901 at k=8 on the real panel). Fidelity is
# not usefulness. These arms are what settles it, and ae_nonoise is what says
# whether the denoising earned its keep downstream rather than in the R^2.
#
# ae_lookahead is the upper bound, exactly as build_feature_encoding --mode
# global is: it fits the encoder on the whole panel, which you could never have
# live. If it matches ae_plus_raw the causal encoding carries no hidden
# advantage and can be trusted. If it is far higher, the gap is the part that is
# not available at prediction time.
#
# Read the comparison as a PAIRED PER-PERIOD DELTA (src/analysis/ablation_report.py),
# not as side-by-side means. Two arms on the same 510 periods share almost all
# of their variance, and the unpaired standard error is several times too wide.
#
# Usage:
#   bash experiments/autoencoder_crypto.sh
#   DRY_RUN=1 bash experiments/autoencoder_crypto.sh
#   ONLY=pca_plus_raw bash experiments/autoencoder_crypto.sh
#   SKIP_BUILD=1 bash experiments/autoencoder_crypto.sh   # datasets already built
#   SHARD=1 OF=2 bash experiments/autoencoder_crypto.sh
#
# Under screen:
#   screen -L -Logfile autoencoder.txt bash experiments/autoencoder_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# AE_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 under screen and every training block comes out empty.
AE_MODEL="${AE_MODEL:-ridge_glmnet}"
AE_TASK="${AE_TASK:-regression}"
AE_SCHEME="${AE_SCHEME:-rolling}"
AE_TRAIN_WINDOW="${AE_TRAIN_WINDOW:-6month}"
AE_REFIT="${AE_REFIT:-6month}"
AE_N_ITER="${AE_N_ITER:-50}"
AE_N_SPLITS="${AE_N_SPLITS:-5}"
AE_N_JOBS="${AE_N_JOBS:-8}"
AE_SEED="${AE_SEED:-42}"
AE_VAL_FRAC="${AE_VAL_FRAC:-0.3}"
AE_CODE_DIM="${AE_CODE_DIM:-8}"

case "${AE_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "AE_TRAIN_WINDOW=${AE_TRAIN_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

SOURCE="crypto/v2.0"

# Arms are FIVE PARALLEL ARRAYS, not packed "name|version|..." strings, for the
# same reason ablation_crypto.sh uses them: the include/exclude regexes contain
# "|" as alternation, and packing them into a pipe-delimited record and
# splitting with IFS='|' silently tears them apart.
ARM_NAME=(); ARM_SUFFIX=(); ARM_BUILD=(); ARM_NFEAT=(); ARM_INC=()
add_arm() { ARM_NAME+=("$1"); ARM_SUFFIX+=("$2"); ARM_BUILD+=("$3")
            ARM_NFEAT+=("$4"); ARM_INC+=("$5"); }

# Both autoencoder arms state their architecture EXPLICITLY rather than taking
# the builder's default. The default moved once already, when the search beat the
# first guess, and an arm that silently follows it is an arm whose meaning
# changes under you between runs.
AE_SMALL="--hidden 64,32 --lr 0.001 --batch-size 4096"      # the first guess
AE_BEST="--hidden 256,128 --lr 0.00279 --batch-size 8192"   # 20-trial winner

#        name           suffix       extra build flags ("-" reuses an earlier build)   n_feat  include
add_arm  ae_plus_raw    "-aetuned"   "--code-dim ${AE_CODE_DIM} ${AE_BEST} --out-suffix=-aetuned"  $((22 + AE_CODE_DIM))  "-"
add_arm  ae_alone       "-aetuned"   "-"                                                $((22 + AE_CODE_DIM))  '^feature_ae_'
add_arm  pca_plus_raw   "-aepca"     "--method pca --code-dim ${AE_CODE_DIM}"           $((22 + AE_CODE_DIM))  "-"
add_arm  ae_lookahead   "-aeglobal"  "--fit-scope all --code-dim ${AE_CODE_DIM} ${AE_BEST}"  $((22 + AE_CODE_DIM))  "-"
add_arm  ae_nonoise     "-aeclean"   "--swap-noise 0 ${AE_BEST} --out-suffix=-aeclean"   $((22 + AE_CODE_DIM))  "-"
# The pre-tuning architecture, kept as an arm. The search moved held-out
# reconstruction from 0.904 to 0.954; this is what asks whether any of that
# translates into a forecast, which is a different question and may well answer
# no.
add_arm  ae_small       "-ae"        "--code-dim ${AE_CODE_DIM} ${AE_SMALL}"            $((22 + AE_CODE_DIM))  "-"
add_arm  ae_k4          "-ae4"       "--code-dim 4 ${AE_BEST} --out-suffix=-ae4"                   26                     "-"
add_arm  ae_k16         "-ae16"      "--code-dim 16 ${AE_BEST} --out-suffix=-ae16"                 38                     "-"

total="${#ARM_NAME[@]}"
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

echo "=============================================================="
echo "arms=${total}  shard ${SHARD}/${OF}  source=${SOURCE}  k=${AE_CODE_DIM}"
echo "=============================================================="

for i in "${!ARM_NAME[@]}"; do
    arm="${ARM_NAME[$i]}"; suffix="${ARM_SUFFIX[$i]}"; flags="${ARM_BUILD[$i]}"
    want="${ARM_NFEAT[$i]}"; inc="${ARM_INC[$i]}"
    version="${SOURCE}${suffix}"
    index=$((index + 1))

    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${arm}" ] && continue
    [ -n "${SKIP:-}" ] && [ "${SKIP}" = "${arm}" ] && continue
    # Round-robin sharding so each machine gets a mix of cheap and dear arms.
    [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

    build_cmd=(python3 -m src.data_mani.build_autoencoder
               --source-version "${SOURCE}" --val-frac "${AE_VAL_FRAC}")
    [ "${flags}" != "-" ] && build_cmd+=(${flags})

    run_cmd=(python3 -u -m src.run_forecast crypto "${AE_MODEL}"
             --task "${AE_TASK}"
             --cv-scheme "${AE_SCHEME}" --train-window "${AE_TRAIN_WINDOW}"
             --refit-every "${AE_REFIT}" --data-version "${version}"
             --val-frac "${AE_VAL_FRAC}"
             --n-iter "${AE_N_ITER}" --n-splits "${AE_N_SPLITS}"
             --n-jobs "${AE_N_JOBS}" --seed "${AE_SEED}"
             --run-name "w6r6_glmnet_reg_${arm}")
    [ "${inc}" != "-" ] && run_cmd+=(--feature-include "${inc}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-14s\n' "$index" "$total" "$arm"
        [ "${flags}" != "-" ] && printf '        build: %s\n' "${build_cmd[*]}"
        printf '        run  : %s\n' "${run_cmd[*]}"
        continue
    fi

    ran=$((ran + 1))
    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${arm}   $(date '+%H:%M:%S')"
    echo "=============================================================="

    if [ "${flags}" != "-" ] && [ "${SKIP_BUILD:-0}" != "1" ] \
       && [ ! -f "data/${version}/train.parquet" ]; then
        "${build_cmd[@]}" 2>&1 | tee "logs/build_${arm}.log"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            failed=$((failed + 1))
            echo ">>> ${arm} BUILD FAILED -- continuing"
            continue
        fi
    fi

    # An arm that reuses an earlier arm's dataset ("-" build) has nothing to
    # build itself, so say so plainly rather than letting pyarrow raise.
    if [ ! -f "data/${version}/train.parquet" ]; then
        failed=$((failed + 1))
        echo ">>> data/${version}/train.parquet is missing -- run the arm that" \
             "builds it first, or drop SKIP_BUILD=1. Skipping ${arm}." >&2
        continue
    fi

    # Refuse to run on the wrong dataset. A silent feature-count drift is the
    # one error that produces a plausible number attached to the wrong arm.
    n_feat=$(python3 - "${version}" <<'PY'
import sys, pyarrow.parquet as pq
path = "data/%s/train.parquet" % sys.argv[1]
print(sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names))
PY
)
    if [ "${n_feat}" != "${want}" ]; then
        failed=$((failed + 1))
        echo ">>> expected ${want} features in ${version}, found ${n_feat} -- skipping ${arm}" >&2
        continue
    fi

    # And refuse to run an encoder whose fit window reaches into THIS study's
    # validation block. The boundary is set at build time from --val-frac, which
    # is a run-time flag everywhere else, so the two can drift apart silently:
    # build at 0.3, run at 0.5, and the validation block starts before the
    # encoder's boundary. Nothing else in the pipeline would notice.
    python3 - "${version}" "${AE_VAL_FRAC}" <<'PY' || { failed=$((failed + 1)); continue; }
import json, sys
meta = json.load(open("data/%s/encoder.json" % sys.argv[1]))
wanted = float(sys.argv[2])
if meta["fit_scope"] == "all":
    print("  note: %s is the LOOKAHEAD upper bound, not a deployable arm"
          % sys.argv[1])
elif meta["val_frac"] < wanted - 1e-9:
    sys.exit(
        "  encoder built for val_frac=%.2f but this study runs %.2f -- its fit "
        "window reaches into the validation block. Rebuild with --val-frac %.2f."
        % (meta["val_frac"], wanted, wanted))
else:
    print("  encoder saw nothing after %s (val_frac=%.2f)"
          % (meta["fit_boundary_period"], meta["val_frac"]))
PY

    started=$(date +%s)
    # A failed arm must not cost you the others.
    "${run_cmd[@]}" 2>&1 | tee "logs/ae_${arm}.log"
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))

    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        echo ">>> ${arm} FAILED (exit ${status}) after ${elapsed}s -- continuing"
    else
        echo ">>> ${arm} done in ${elapsed}s"
    fi
done

echo
echo "=============================================================="
echo "ran ${ran} arms in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
if [ "${DRY_RUN:-0}" != "1" ]; then
    echo
    echo "rank them, together with the raw-22 baseline they have to beat:"
    echo "  python3 -c \"import json,glob;[print('%-40s payout %.6f  corr %.6f  dd %.4f' % (p.split('/')[-1][:-5], 0.1*json.load(open(p))['corr']['mean']+json.load(open(p))['mmc']['mean'], json.load(open(p))['corr']['mean'], json.load(open(p))['corr']['max_drawdown'])) for p in sorted(glob.glob('results/metrics/crypto/ridge_glmnet/w6r6_glmnet_reg*.json'))]\""
    echo
    echo "then take the PAIRED per-period delta against the raw-22 baseline."
    echo "The unpaired spread between two arms on the same 510 periods is"
    echo "several times too wide to read anything from -- ablation_report.py"
    echo "measured t=0.75 unpaired against t=3.14 paired on the same data."
    echo "Newey-West at lag 20, because a 20-day target on daily dates makes"
    echo "consecutive periods ~95% overlapping:"
    echo "  python3 -c \"import os,pandas as pd;"\
         "from src.analysis.ablation_report import newey_west;"\
         "d='results/per_period/crypto/ridge_glmnet/';"\
         "b=pd.read_csv(d+'w6r6_glmnet_reg.csv',index_col=0);"\
         "[print('%-16s corr delta %+.6f  t %+.2f' % ((n,)+newey_west("\
         "pd.read_csv(d+'w6r6_glmnet_reg_'+n+'.csv',index_col=0)['corr']"\
         ".sub(b['corr'],axis=0).dropna())[::2])) for n in"\
         "['ae_plus_raw','ae_alone','pca_plus_raw','ae_lookahead','ae_nonoise',"\
         "'ae_small','ae_k4','ae_k16']"\
         "if os.path.exists(d+'w6r6_glmnet_reg_'+n+'.csv')]\""
fi
echo "=============================================================="
