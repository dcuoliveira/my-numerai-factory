"""Rebuild exposure alignment into separate retrospective datasets."""
import subprocess
import sys
for variant in ("main", "scalar", "circular", "future"):
    subprocess.run([sys.executable, "-m", "src.data_mani.build_vol_macrovol", "--variant", variant,
                    "--research-retrospective"], check=True)
