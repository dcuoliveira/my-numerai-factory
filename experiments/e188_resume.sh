#!/bin/bash
# E188 auto-resume after a reboot of westeros (run by cron @reboot).
# Clears claims left by jobs that were mid-run when the box went down (none can
# be live right after boot), then starts the 8 GPU workers if any grid job is
# still missing. When the grid is complete it does nothing.
# westeros has no systemd linger, so `systemd-run --user` is unavailable at
# boot; workers are started with setsid/nohup instead.
cd /home/danielco/Downloads/my-numerai-factory || exit 1
exec 9>/tmp/e188_resume.lock; flock -n 9 || exit 0
LOG=logs/e188/resume.log; mkdir -p logs/e188
R=results/e188_iara_scaling
PY=/home/danielco/miniconda3/bin/python3
sleep 60   # let the GPU driver come up
echo "$(date '+%F %T') boot resume" >> $LOG
if pgrep -u "$USER" -f "e188_iara_scaling.py worker" >/dev/null; then
  echo "workers already running, exit" >> $LOG; exit 0
fi
n=$(ls -d $R/runs/[AB]_*/summary.json 2>/dev/null | wc -l)
if [ "$n" -ge 1400 ]; then echo "grid complete ($n), exit" >> $LOG; exit 0; fi
for c in $R/claims/*; do
  j=$(basename "$c"); case $j in *.failed|*.lock) continue;; esac
  [ -f "$R/runs/$j/summary.json" ] || { echo "stale $j" >> $LOG; rm -f "$c"; }
done
for i in 1 2 3 4 5 6 7 8; do
  setsid nohup $PY experiments/e188_iara_scaling.py worker --device cuda \
    --min-params 30001 >> logs/e188/gpu$i.log 2>&1 < /dev/null &
done
echo "started 8 workers ($n done)" >> $LOG
