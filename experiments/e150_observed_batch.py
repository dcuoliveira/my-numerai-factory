"""One managed batch of the corrected, development-only E150 comparison."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SOURCES = ("GOLD", "SP500", "DXY", "Y10", "VIXLVL", "VIXVOV")
GROUPS = {
    "main": ["B_COV", "VC_COV", "ALL", "NO_EXPO", "EXPO_ONLY"] + ["LOO_" + k for k in SOURCES],
    "sources": ["F_" + k for k in SOURCES],
    "gates": ["GATE_%s_%s" % (k, g) for k in SOURCES for g in ("HI", "LO")],
    "controls": ["CTL_SCALAR", "PLC_CIRC", "PLC_FUTURE"],
}
ROOT = Path("results/reports/crypto/e150_macrovol_observed_dev")


def write_status(path, data):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ridge", "lgb_ktz", "mlp_ktz"], required=True)
    ap.add_argument("--group", choices=GROUPS, required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    base = [sys.executable, "experiments/e150_macrovol_evaluation.py", "--fit", "--model", args.model,
            "--n-jobs", str(args.n_jobs), "--research-retrospective", "--dataset-dir",
            "data/crypto/v2.0-vol-causal-macrovol-retrospective", "--arms"] + GROUPS[args.group]
    started = time.time()
    status = ROOT / "batch_status" / (args.model + "_" + args.group + ".json")
    status.parent.mkdir(parents=True, exist_ok=True)
    required = [Path("data/crypto/v2.0-vol-causal-macrovol-retrospective/manifest.json")]
    if args.group == "controls":
        required += [Path("data/crypto/v2.0-vol-causal-macrovol-" + name + "-retrospective/manifest.json")
                     for name in ("control-scalar", "placebo-circular", "placebo-future-LOOKAHEAD")]
    while not all(path.exists() for path in required):
        if time.time() - started > 3600:
            write_status(status, dict(state="failed", error="Dataset preparation did not finish in one hour"))
            raise RuntimeError("Dataset preparation timed out")
        write_status(status, dict(state="waiting_for_datasets", started=started))
        time.sleep(10)
    write_status(status, dict(state="running", started=started))
    try:
        for extra in ([], ["--tune"]):
            subprocess.run(base + extra, check=True)
    except Exception as error:
        write_status(status, dict(state="failed", started=started, error=str(error)))
        raise
    write_status(status, dict(state="complete", started=started, elapsed_seconds=time.time() - started))


if __name__ == "__main__":
    main()
