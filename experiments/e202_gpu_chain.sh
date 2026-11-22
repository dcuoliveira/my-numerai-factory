#!/bin/bash
# E202 GPU extraction chain, one card: <first model> then Toto then TabPFN-TS, all in --claim mode so the
# two cards (and the CPU workers, for TabPFN-TS) split each model's parts. Usage: e202_gpu_chain.sh <first>
cd /home/danielco/Downloads/my-numerai-agent
export HF_HOME=/scratch/danielco/hf PYTHONUNBUFFERED=1
run() { echo "== $1 start $(date)"; env "${@:3}" "$2" -m src.data_mani.tsfm_states --model "$1" --target 60 --claim; echo "== $1 exit $? $(date)"; }
run "$1" /scratch/danielco/tsfm2/venv_tt/bin/python
run toto /scratch/danielco/tsfm2/venv_toto/bin/python TSFM_AMP=fp32 TSFM_CHUNK=16
run tabpfnts /scratch/danielco/tsfm/envs/tabpfnts/bin/python
