"""E152 strict-vintage volume-alpha evaluation. See e152_registration.md.

    python3 -m experiments.e152_volumealpha_causal_evaluation --phase dev
    python3 -m experiments.e152_volumealpha_causal_evaluation --select
    python3 -m experiments.e152_volumealpha_causal_evaluation --phase test

Existing retrospective E151 inputs are not silently promoted to verified inputs.
"""
from src.analysis.volumealpha_evaluation import main


if __name__ == "__main__":
    main()
