#!/usr/bin/env bash
# E212 only: expose copied runtime at the frozen extractor's canonical paths.
set -euo pipefail
horizon=${1:?20 or 60}
case "$horizon" in 20) wall=35;; 60) wall=91;; *) exit 2;; esac
repo=/home/danielco/Downloads/my-numerai-factory
runtime=/home/danielco/e212_runtime
root=$repo/results/reports/crypto/e212_volatility99_tsfm
out=$root/cache/t$horizon/embeddings/timestral
mkdir -p "$out"
while [ ! -f "$runtime/READY.json" ]; do sleep 15; done
exec 9>"$runtime/gpu0.lock"
flock 9
# Another user's work may have arrived since setup. Wait rather than compete.
while ! nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if ($1>=9000 && $2<=10) ok=1} END {exit !ok}'; do sleep 30; done
cd "$repo"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TSFM_EMBED_ROOT=$root TSFM_EMBED_SCR=$root/cache/t$horizon
export TSFM_EMBED_TARGET=target_binned_return_$horizon TSFM_EMBED_WALL=$wall E206_BATCH=4
sandbox=(bwrap --tmpfs / --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /bin /bin --ro-bind /etc /etc --bind /home /home --bind /tmp /tmp --dev-bind /dev /dev --proc /proc --bind "$runtime/scratch" /scratch/danielco --ro-bind "$runtime/python312" /home/danielco/miniconda3)
command=(/scratch/danielco/tsfm2/venv_tt/bin/python -u -m src.data_mani.e212_signal_embeddings --model timestral --device cuda:0)
"${sandbox[@]}" -- "${command[@]}" --smoke
# The bridge checks fingerprint, package versions and host audit before releasing.
while [ ! -f "$out/WESTEROS_APPROVED.json" ]; do sleep 10; done
"${sandbox[@]}" -- "${command[@]}"
touch "$out/WESTEROS_COMPLETE"
