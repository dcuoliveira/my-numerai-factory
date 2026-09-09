#!/usr/bin/env bash
#
# E22 -- what should the one-hot block be, and should it be penalised at all?
#
# E21 found e21_onehot_x10 (the 740 ticker dummies penalised 10x) to be a
# no-op: +0.000184 at t +0.31, and 0.998 correlated with the control on
# within-period ranks. This asks the three questions that result leaves open,
# changing ONE thing at a time.
#
#   1. THE PENALTY.  10x harder, one shared alpha, or NOT PENALISED AT ALL.
#      An unpenalised dummy block is a SYMBOL FIXED EFFECTS specification --
#      each symbol gets its own free intercept -- which is the standard panel
#      treatment and something no model in this project has ever fitted. A
#      penalised block is the random-effects analogue, shrinking each symbol
#      toward the common intercept.
#
#      NOTE the parser refuses a literal p0: a zero multiplier removes the
#      penalty entirely and makes the system singular on a collinear block.
#      "Unpenalised" is spelled p1e-6, which is the same thing numerically and
#      cannot produce a singular solve.
#
#   2. THE KEY.  symbol or ucid. The TICKER IS NOT A STABLE IDENTITY in this
#      panel: 30 tickers are reused by different coins (ACT covers three
#      projects on disjoint date ranges) and 3 coins were renamed (GRAM->TON),
#      so 12,606 rows -- 2.48% -- sit on a ticker whose meaning changed.
#      `ucid` is the stable coin id and is shipped in the original dataset;
#      NOTHING in src/ has ever used it. Keying on it splits the reused tickers
#      apart and rejoins the renamed ones: 750 columns against 746.
#
#      The block's whole justification is "a symbol's persistent tendency to
#      rank high or low" (build_onehot's docstring), so an identifier that is
#      not persistent is the wrong key, and this measures what that costs.
#
#   3. THE GRANULARITY.  Identity, or just SIZE. A three-column low/mid/large
#      market-cap bucket asks whether the persistence credited to 746 ticker
#      dummies is really a cap effect that three columns could carry. The
#      bucket is a within-date rank, so a coin moves between buckets over time.
#
#   4. REPLACE THE RAW COLUMNS. The one-hot alone, no raw 22, so the dummies
#      are the whole model rather than a passenger.
#
# In every arm that names the one-hot block, the block is left UNPENALISED
# unless the arm exists specifically to vary that.
#
# ---------------------------------------------------------------------------
# Reading it
# ---------------------------------------------------------------------------
#
# e22_raw is the universal reference -- the raw 22 with nothing added. Each
# family's own `_flat` arm (one alpha across everything) is the local control
# that isolates the penalty scheme from the feature set. So:
#
#     e22_sym_flat  vs e22_raw        does the block help at all
#     e22_sym_p0    vs e22_sym_flat   does NOT penalising it help
#     e22_ucid_p0   vs e22_sym_p0     does the stable id help
#     e22_cap_p0    vs e22_sym_p0     do three columns do the job of 746
#     e22_only_*    vs their raw twin what the block carries alone
#
# Priors are poor. The one-hot block is in three deployed slots and CURUPIRA_03
# (786 columns, mostly dummies) pays 0.08885 against CURUPIRA_02's 0.08723 on
# the raw 22 -- a 0.0016 edge for 764 extra columns.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash experiments/e22_onehot_reg_crypto.sh
#   DRY_RUN=1 bash experiments/e22_onehot_reg_crypto.sh
#   ONLY=e22_ucid_p0 bash experiments/e22_onehot_reg_crypto.sh
#   NO_MAIL=1 bash experiments/e22_onehot_reg_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

OH_MODEL="${OH_MODEL:-gen_ridge}"
OH_SCHEME="${OH_SCHEME:-rolling}"
OH_TRAIN_WINDOW="${OH_TRAIN_WINDOW:-130}"
OH_REFIT="${OH_REFIT:-130}"
OH_EMBARGO="${OH_EMBARGO:-20}"
OH_N_ITER="${OH_N_ITER:-50}"
OH_N_SPLITS="${OH_N_SPLITS:-5}"
OH_SEED="${OH_SEED:-42}"
OH_VAL_FRAC="${OH_VAL_FRAC:-0.3}"
OH_SEARCH_JOBS="${OH_SEARCH_JOBS:-4}"
OH_N_JOBS="${OH_N_JOBS:-1}"

case "${OH_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "OH_TRAIN_WINDOW=${OH_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 130" >&2; exit 1 ;;
esac

NAMES=(); VERSIONS=(); OPTS=(); EXPECT=(); WHY=()
add() { NAMES+=("$1"); VERSIONS+=("$2"); OPTS+=("$3"); EXPECT+=("$4"); WHY+=("$5"); }

FLAT='{"structures":["identity"]}'

# the universal reference
add e22_raw            "crypto/v2.0"                "$FLAT"                                                    22  "raw 22, nothing added"
# 1. THE PENALTY, on the ticker key
add e22_sym_flat       "crypto/v2.0-e22ohsymbol"    "$FLAT"                                                   768  "raw + ticker dummies, one alpha"
add e22_sym_p10        "crypto/v2.0-e22ohsymbol"    '{"structures":["identity"],"group_specs":["onehot:p10"]}'   768  "ticker dummies 10x  (E21's scheme)"
add e22_sym_p0         "crypto/v2.0-e22ohsymbol"    '{"structures":["identity"],"group_specs":["onehot:p1e-6"]}' 768  "ticker dummies UNPENALISED"
# 2. THE KEY: same three, on the stable coin id
add e22_ucid_flat      "crypto/v2.0-e22ohucid"      "$FLAT"                                                   772  "raw + ucid dummies, one alpha"
add e22_ucid_p10       "crypto/v2.0-e22ohucid"      '{"structures":["identity"],"group_specs":["onehot:p10"]}'   772  "ucid dummies 10x"
add e22_ucid_p0        "crypto/v2.0-e22ohucid"      '{"structures":["identity"],"group_specs":["onehot:p1e-6"]}' 772  "ucid dummies UNPENALISED"
# 3. THE GRANULARITY: three cap buckets instead of identity
add e22_cap_flat       "crypto/v2.0-e22ohcap"       "$FLAT"                                                    25  "raw + low/mid/large, one alpha"
add e22_cap_p0         "crypto/v2.0-e22ohcap"       '{"structures":["identity"],"group_specs":["cap:p1e-6"]}'    25  "cap buckets UNPENALISED"
# 4. REPLACE the raw columns
add e22_only_sym_p0    "crypto/v2.0-e22onlysymbol"  '{"structures":["identity"],"group_specs":["onehot:p1e-6"]}' 746  "ticker dummies ALONE, unpenalised"
add e22_only_sym_flat  "crypto/v2.0-e22onlysymbol"  "$FLAT"                                                   746  "ticker dummies ALONE, one alpha"
add e22_only_ucid_p0   "crypto/v2.0-e22onlyucid"    '{"structures":["identity"],"group_specs":["onehot:p1e-6"]}' 750  "ucid dummies ALONE, unpenalised"
add e22_only_cap_p0    "crypto/v2.0-e22onlycap"     '{"structures":["identity"],"group_specs":["cap:p1e-6"]}'     3  "cap buckets ALONE, unpenalised"

REPORT="logs/e22_report.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }
notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "E22 one-hot regularisation -- ABORTED (exit ${code})"
      fi' EXIT

# Gate on the feature count AND the block partition: an arm naming a block the
# data does not have would raise at the first refit, hours in.
python3 - "${VERSIONS[@]}" --- "${EXPECT[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
from src.prediction.generalised_ridge import block_index
args = sys.argv[1:]; cut = args.index("---")
versions, expected = args[:cut], [int(x) for x in args[cut + 1:]]
for version, want in zip(versions, expected):
    path = "data/%s/train.parquet" % version
    try:
        names = [c for c in pq.ParquetFile(path).schema.names
                 if c.startswith("feature_")]
    except Exception as error:
        sys.exit("cannot read %s: %s" % (path, error))
    if len(names) != want:
        sys.exit("%s has %d features, expected %d" % (version, len(names), want))
    blocks = {k: len(v) for k, v in block_index(names).items()}
    print("  %-32s %4d feat  %s" % (version.replace("crypto/", ""), len(names), blocks))
print("gate OK: %d datasets" % len(set(versions)))
PY

total="${#NAMES[@]}"; index=0; failed=0; ran=0
started_all=$(date +%s)

say "=============================================================="
say "E22 one-hot: the penalty, the key, the granularity | ${total} arms"
say "unpenalised is spelled p1e-6 -- a literal p0 makes the solve singular"
say "ucid is the STABLE coin id: 30 tickers are reused, 3 coins renamed"
say "model=${OH_MODEL} scheme=${OH_SCHEME} window=${OH_TRAIN_WINDOW} refit=${OH_REFIT}"
say "n-iter=${OH_N_ITER} n-splits=${OH_N_SPLITS} seed=${OH_SEED} search-jobs=${OH_SEARCH_JOBS}"
say "started $(date -Is)"
say "=============================================================="

for i in "${!NAMES[@]}"; do
    index=$((index + 1))
    name="${NAMES[$i]}"; version="${VERSIONS[$i]}"; opts="${OPTS[$i]}"; why="${WHY[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue

    cmd=(python3 -u -m src.run_forecast crypto "${OH_MODEL}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme "${OH_SCHEME}" --train-window "${OH_TRAIN_WINDOW}"
         --refit-every "${OH_REFIT}" --embargo "${OH_EMBARGO}"
         --n-splits "${OH_N_SPLITS}" --n-iter "${OH_N_ITER}"
         --seed "${OH_SEED}" --val-frac "${OH_VAL_FRAC}"
         --n-jobs "${OH_N_JOBS}" --search-jobs "${OH_SEARCH_JOBS}"
         --model-options "${opts}"
         --data-version "${version}" --run-name "${name}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-20s %-36s %s\n' "$index" "$total" "$name" "$why" "$opts"
        continue
    fi

    ran=$((ran + 1)); started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${name}.log" >/dev/null
    status="${PIPESTATUS[0]}"; elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        say "$(printf '[%2d/%2d] %-20s FAILED (exit %d) after %5ds  -- %s' \
             "$index" "$total" "$name" "$status" "$elapsed" "$why")"
    else
        say "$(printf '[%2d/%2d] %-20s done in %5ds  -- %s' "$index" "$total" "$name" "$elapsed" "$why")"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }

say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""
python3 -u -m src.analysis.e22_report >>"${REPORT}" 2>&1 || \
    say "(report generation failed; the runs are in results/)"

trap - EXIT
if [ "${failed}" -eq 0 ]; then notify "E22 one-hot -- OK (${ran}/${total})"
else notify "E22 one-hot -- ${failed} arm(s) failed"; fi
echo
echo "report: ${REPORT}"
