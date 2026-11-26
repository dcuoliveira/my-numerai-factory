#!/usr/bin/env bash
set -euo pipefail
model=${1:?model}; horizon=${2:?horizon}
repo=/home/danielco/Downloads/my-numerai-factory
canonical=/home/danielco/Downloads/my-numerai-agent
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Mount the separate checkout at the frozen paths only inside this process.
sandbox=(bwrap --tmpfs / --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /bin /bin --ro-bind /etc /etc --bind /home /home --bind /tmp /tmp --dev-bind /dev /dev --proc /proc --bind "$repo" "$canonical" --chdir "$canonical")
"${sandbox[@]}" -- /home/danielco/miniconda3/bin/python3 -u experiments/e212_volatility99_tsfm.py fit --model "$model" --target "$horizon"
cd "$repo"
python3 - "$model" "$horizon" <<'PY'
import pathlib,hashlib,json,sys,platform,importlib.metadata
p=pathlib.Path('results/reports/crypto/e212_volatility99_tsfm/fits')/f'{sys.argv[1]}_t{sys.argv[2]}'
(p/'predictions.sha256.json').write_text(json.dumps(dict(sha256=hashlib.sha256((p/'predictions.parquet').read_bytes()).hexdigest())))
(p/'execution.json').write_text(json.dumps(dict(host='westeros',python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn']},cpus='0-3',ram_gb=32)))
(p/'WESTEROS_RETURN_READY.json').write_text('{}')
PY
