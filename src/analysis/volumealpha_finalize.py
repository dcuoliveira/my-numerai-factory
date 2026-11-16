"""Finalize and audit the user-authorized reconstructed E152 artifacts."""
import json
from pathlib import Path

from src.analysis.stabilize_volumealpha_oof import main as stabilize
from src.analysis.volumealpha_build_report import main as report


if __name__ == '__main__':
    p = Path('data/crypto/v2.0-volumealpha-causal/manifest.json')
    if 'auxiliary_stability_rule' not in json.loads(p.read_text()):
        stabilize()
    report()
