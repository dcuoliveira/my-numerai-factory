#!/bin/bash
# E235: src.cluster has no queue; retry placing each E235 job until it has started.
cd /home/danielco/Downloads/my-numerai-agent || exit 1
D=/scratch/danielco/e235_mom3_featstats_hs/launch; mkdir -p "$D"
PY=/home/danielco/miniconda3/bin/python3
while IFS= read -r line; do
  name=$(sed -n 's/.*name=\([^ ]*\) --.*/\1/p' <<<"$line"); printf '%s\n' "$line" > "$D/$name.jobs"
done < experiments/e235_mom3_featstats_hs.jobs
while :; do
  left=0
  for f in "$D"/*.jobs; do
    name=$(basename "$f" .jobs); [ -f "$D/$name.started" ] && continue
    left=1
    if $PY -m src.cluster run --jobs "$f" --gpu-slots 1 2>&1 | tee -a "$D/launcher.log" | grep -q '1 started'; then
      date > "$D/$name.started"; echo "$(date) started $name" >> "$D/launcher.log"
    fi
  done
  [ $left = 0 ] && exit 0
  sleep 600
done
