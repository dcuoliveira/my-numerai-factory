#!/usr/bin/env bash
#
# Launcher/monitor for the E123 (20-day) + E126 (60-day) LightGBM training-window
# pair on a CONTENDED cluster. The first `cluster run` placed 5 of E123's 7 arms;
# this loop tops up the rest and launches all of E126 as slots free, then fires
# both reports. It is bounded (12 h) and session-independent -- launch it with:
#
#   systemd-run --user --unit=nmr-e126_orchestrate --collect \
#     bash experiments/e126_orchestrate.sh
#
# Safe to run repeatedly: each tick it launches only arms that are neither
# already running (cluster run refuses a name clash and aborts, so we exclude
# them) nor already finished (a finished arm leaves no unit but writes a metrics
# file, so we exclude those too). It touches ONLY e123_/e126_ names -- never the
# concurrent sessions' e118_/e124_/e125_ units.
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/danielco/miniconda3/bin/python3
MET=results/metrics/crypto/lgb
LOG=logs/cluster/e126_orchestrate.log
mkdir -p logs/cluster
TICK="${E126_TICK:-600}"
DEADLINE=$(( $(date +%s) + ${E126_MAX_HOURS:-12}*3600 ))
JOBS="$REPO/logs/cluster/e126_pending.jobs"

# Arms this loop is responsible for LAUNCHING (any E123 arm not running and not finished -- w5/w65 never placed, w15/w20 died in the full-quota window; all
# E126). Already-running and finished arms are filtered out every tick.
E123_LAUNCH="e123_w5 e123_w10 e123_w15 e123_w20 e123_w22 e123_w65 e123_w130"
E126_LAUNCH="e126_w5 e126_w10 e126_w15 e126_w20 e126_w22 e126_w65 e126_w130"
# All arms whose metrics gate each report.
E123_ALL="e123_w5 e123_w10 e123_w15 e123_w20 e123_w22 e123_w65 e123_w130"
E126_ALL="e126_w5 e126_w10 e126_w15 e126_w20 e126_w22 e126_w65 e126_w130"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
done_arm() { [ -f "$MET/$1.json" ]; }
all_done() { for a in $1; do done_arm "$a" || return 1; done; return 0; }

e123_reported=0
say "orchestrator up on $(hostname); tick ${TICK}s, deadline $(date -d @${DEADLINE} '+%F %T')"

while :; do
    running=$("$PY" -m src.cluster status 2>/dev/null \
              | grep -oE 'nmr-e12[36]_w[0-9]+' | sed 's/nmr-//' | sort -u)
    : > "$JOBS"
    for a in $E123_LAUNCH; do
        done_arm "$a" && continue
        printf '%s\n' "$running" | grep -qx "$a" && continue
        echo "cores=8 ram=16 name=$a -- env ONLY=$a NO_MAIL=1 E123_SEARCH_JOBS=4 E123_N_JOBS=2 bash experiments/e123_train_window_lgb_search_crypto.sh" >> "$JOBS"
    done
    for a in $E126_LAUNCH; do
        done_arm "$a" && continue
        printf '%s\n' "$running" | grep -qx "$a" && continue
        echo "cores=8 ram=16 name=$a -- env ONLY=$a NO_MAIL=1 E126_SEARCH_JOBS=4 E126_N_JOBS=2 bash experiments/e126_train_window_lgb_search_t60_crypto.sh" >> "$JOBS"
    done
    if [ -s "$JOBS" ]; then
        n=$(wc -l < "$JOBS")
        say "launching up to ${n} pending arm(s)"
        "$PY" -m src.cluster run --jobs "$JOBS" 2>&1 | tail -6 | tee -a "$LOG"
    else
        say "nothing to launch this tick (all pending arms running or done)"
    fi

    # E123's own report (searched vs pinned E120) once its seven have landed.
    if [ "$e123_reported" -eq 0 ] && all_done "$E123_ALL"; then
        say "E123 complete -> rendering its report"
        REPORT_ONLY=1 bash experiments/e123_train_window_lgb_search_crypto.sh >>"$LOG" 2>&1 \
            && e123_reported=1
    fi

    # The consolidation (t20 vs t60->20) once ALL fourteen have landed.
    if all_done "$E123_ALL" && all_done "$E126_ALL"; then
        say "all 14 arms complete -> consolidating E123 + E126"
        REPORT_ONLY=1 bash experiments/e126_train_window_lgb_search_t60_crypto.sh >>"$LOG" 2>&1
        say "done."
        break
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        miss=""; for a in $E123_ALL $E126_ALL; do done_arm "$a" || miss="$miss $a"; done
        say "deadline reached; still missing:${miss:- none}"
        break
    fi
    sleep "$TICK"
done
