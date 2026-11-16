#!/usr/bin/env python3
"""E132: per-date blend CORR by w, which e132_analysis.py computes but does not keep. The per-dataset
PDFs need it for their blend curves.

    python3 experiments/e132_blend_series.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e132_analysis as A  # noqa: E402

jobs = [(m, s, k, os.path.join(A.PRED, m, "B0_s%d.parquet" % s), os.path.join(A.PRED, m, "%s_s%d.parquet" % (a, s)))
        for m in A.MODELS for s in ([0] if m == "ridge" else range(5)) for k, a in A.BLEND.items()]
with ProcessPoolExecutor(max_workers=int(os.environ.get("E132_WORKERS", "8"))) as ex:
    BL = pd.concat(list(ex.map(A.blend_job, jobs)), ignore_index=True)
BA = BL.groupby(["model", "dataset", "date", "kind", "block", "w"], as_index=False)["corr"].mean()
BA.to_parquet(os.path.join(A.ROOT, "blend_by_date.parquet"), index=False)
print(BA.shape)
