#!/usr/bin/env bash
#
# Launcher/monitor for E130 (experiments/e130_treeleaf_crypto.sh) on a contended
# cluster. Each tick it places, through `src.cluster run` and the 30% share rule,
# whatever stage is ready and neither finished nor running:
#   1. the two leaf fits (leaf_cat, leaf_rf), in parallel;
#   2. the merge into the two -treeleaf variants, inline on this host, once both
#      leaf caches exist;
#   3. the 10 arms, once both variants exist;
#   4. the report and the finish email, once all 10 metrics files exist.
# A job that dies (no unit, no output) is simply placed again on a later tick, under
# a fresh unit name, so a failed unit's leftover name never blocks it.
#
# Run it detached, NOT as a systemd --user unit: linger is off on every host, so a
# user unit dies with the last login session, while a setsid process survives
# (KillUserProcesses=no):
#
#   setsid nohup bash experiments/e130_orchestrate.sh > /dev/null 2>&1 &
#
# E130_FORCE_CPUS=0-15 skips the cluster and the share rule, and runs every stage
# on THIS host pinned to those cpus: leaf fits in parallel, then a ridge lane and an
# MLP lane in parallel. Use it only when the queue is not an option.
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PATH="/home/danielco/miniconda3/bin:${PATH}"
PY=/home/danielco/miniconda3/bin/python3
RUNNER=experiments/e130_treeleaf_crypto.sh
LOG=logs/cluster/e130_orchestrate.log
JOBS=logs/cluster/e130_pending.jobs
mkdir -p logs/cluster
TICK="${E130_TICK:-600}"
DEADLINE=$(( $(date +%s) + ${E130_MAX_HOURS:-48}*3600 ))
ARMS="r_ctl r_cat r_rf r_both m_ctl m_cat m_rf m_both m_ctl_s43 m_both_s43"
CACHE=data/crypto/treeleaf_cache
VARIANTS="data/crypto/v2.0-fenccausal-treeleaf/train.parquet data/crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmx-treeleaf/train.parquet"
ONE='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1'

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
model_of() { case "$1" in r_*) echo ridge_glmnet ;; *) echo mlp ;; esac; }
arm_done() { [ -f "results/metrics/crypto/$(model_of "$1")/e130_$1.json" ]; }
leaf_done() { [ -f "$CACHE/$1.json" ]; }
variants_done() { for v in $VARIANTS; do [ -f "$v" ] || return 1; done; }
all_arms_done() { for a in $ARMS; do arm_done "$a" || return 1; done; }
mail() { "$PY" -c "import sys; from src.live.notify import send_text; send_text(sys.argv[1], sys.argv[2])" "$1" "$2" \
         || say "note: email failed"; }

finish() {
    say "all 10 arms landed -> report"
    STAGE=report NO_MAIL=1 bash "$RUNNER" >> "$LOG" 2>&1
    mail "[crypto] E130 finished" "$(cat logs/e130_report.txt 2>/dev/null)"
    say "done."
}

if [ -n "${E130_FORCE_CPUS:-}" ]; then
    C="${E130_FORCE_CPUS}"; N=$(taskset -c "$C" nproc)
    say "FORCE mode on $(hostname), cpus ${C} (${N}); the 30% share rule is NOT applied"
    half=$(( N / 2 > 1 ? N / 2 : 1 ))
    leaf_done cat || taskset -c "$C" env $ONE STAGE=leaf_cat NO_MAIL=1 E130_LEAF_SJ=$(( half / 2 > 1 ? half / 2 : 1 )) E130_LEAF_NJ=2 bash "$RUNNER" >> "$LOG" 2>&1 &
    leaf_done rf  || taskset -c "$C" env $ONE STAGE=leaf_rf  NO_MAIL=1 E130_LEAF_SJ=$(( half / 2 > 1 ? half / 2 : 1 )) E130_LEAF_NJ=2 bash "$RUNNER" >> "$LOG" 2>&1 &
    wait
    { leaf_done cat && leaf_done rf; } || { say "a leaf fit failed -- see logs/e130_leaf_*.log"; exit 1; }
    variants_done || STAGE=merge NO_MAIL=1 bash "$RUNNER" >> "$LOG" 2>&1 || { say "merge failed"; exit 1; }
    todo_r=""; todo_m=""
    for a in $ARMS; do arm_done "$a" && continue; case "$a" in r_*) todo_r="$todo_r $a" ;; *) todo_m="$todo_m $a" ;; esac; done
    [ -n "$todo_r" ] && taskset -c "$C" env $ONE STAGE=arms NO_MAIL=1 ONLY="$todo_r" E130_SJ=$half E130_NJ=1 bash "$RUNNER" >> "$LOG" 2>&1 &
    [ -n "$todo_m" ] && taskset -c "$C" env $ONE STAGE=arms NO_MAIL=1 ONLY="$todo_m" E130_SJ=$half E130_NJ=1 bash "$RUNNER" >> "$LOG" 2>&1 &
    wait
    all_arms_done && finish || say "some arms failed -- see logs/e130_*.log"
    exit 0
fi

say "orchestrator up on $(hostname) (pid $$); tick ${TICK}s, deadline $(date -d @${DEADLINE} '+%F %T')"
while :; do
    # Full unit names from the API: `cluster status` prints them cut to 22 chars.
    # Only live units count; a failed one is placed again under a new name.
    running=$("$PY" -c "
from src.cluster.manager import status
for host, rows in status().items():
    for r in rows:
        u = r['unit'].removesuffix('.service')
        if u.startswith('nmr-e130_') and r['state'].split('/')[0] in ('active', 'activating', 'reloading'):
            print(u[4:])" 2>/dev/null | sort -u)
    is_running() { printf '%s\n' "$running" | grep -qE "^e130_$1_t[0-9]+$"; }
    stamp=$(date +%s)
    : > "$JOBS"

    for s in cat rf; do
        leaf_done "$s" || is_running "leaf_$s" || \
            echo "cores=8 ram=32 name=e130_leaf_${s}_t${stamp} -- env $ONE STAGE=leaf_$s NO_MAIL=1 E130_LEAF_SJ=4 E130_LEAF_NJ=2 bash $RUNNER" >> "$JOBS"
    done

    if leaf_done cat && leaf_done rf && ! variants_done; then
        say "both leaf caches present -> merging"
        STAGE=merge NO_MAIL=1 bash "$RUNNER" >> "$LOG" 2>&1 || say "merge FAILED -- see logs/e130_merge.log"
    fi

    if variants_done; then
        for a in $ARMS; do
            arm_done "$a" || is_running "$a" || \
                echo "cores=8 ram=16 name=e130_${a}_t${stamp} -- env $ONE STAGE=arms ONLY=$a NO_MAIL=1 E130_SJ=8 E130_NJ=1 bash $RUNNER" >> "$JOBS"
        done
    fi

    if [ -s "$JOBS" ]; then
        say "placing up to $(wc -l < "$JOBS") pending job(s)"
        "$PY" -m src.cluster run --jobs "$JOBS" 2>&1 | tail -8 | tee -a "$LOG"
    fi

    if all_arms_done; then finish; break; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        miss=""; for a in $ARMS; do arm_done "$a" || miss="$miss $a"; done
        say "deadline reached; still missing:${miss}"
        mail "[crypto] E130 orchestrator deadline" "still missing:${miss}"
        break
    fi
    sleep "$TICK"
done
