"""E216: assemble CU_SA_IA_BO_MA_JA_01 exactly as run_round would, without uploading.

Uses run_round's own load_panel / add_lagged_meta / prepare_slot and
submit.predict on the live panel currently on disk (the last round's live
date), and compares the result with the CU leg's prediction on the same rows.
It writes nothing under results/live/ -- a plain `run_round --dry-run` on a
closed round would overwrite that round's report.

    python3 -m src.analysis.e216_slot_dryrun
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.live import run_round as RR
from src.live import submit

OUT = Path("results/reports/crypto/e216_jaci_live")
SLOT, LEG = "CU_SA_IA_BO_MA_JA_01", "CU_SA_IA_BO_MA_R_01"


def main():
    panel, live_date, gap = RR.load_panel(verbose=False)
    panel, meta_info = RR.add_lagged_meta(panel, live_date)
    budget, warnings = RR.Budget(30, 2), []
    out = {}
    for name in (LEG, SLOT):
        est, cols, built, meta, info = RR.prepare_slot(name, panel, live_date, budget, warnings,
                                                       cache_only=True)
        if est is None:
            raise SystemExit("%s could not be prepared: %s" % (name, info))
        frame = submit.predict(est, built, cols, live_date, "regression")
        out[name] = (frame, est)
    leg, slot = out[LEG][0], out[SLOT][0]
    route = out[SLOT][1].route_log()
    resid = out[SLOT][1].resid_log()
    merged = leg.merge(slot, on="symbol", suffixes=("_leg", "_slot"))
    jaci = out[SLOT][1].members[1][1].last
    moved = (merged.prediction_leg != merged.prediction_slot)
    summary = dict(live_date=str(live_date.date()), rows=len(slot), unique=int(slot.symbol.nunique()),
                   finite=bool(np.isfinite(slot.prediction).all()),
                   in_unit_interval=bool(slot.prediction.between(0, 1).all()),
                   route=route, jaci=jaci, resid=resid, meta=meta_info,
                   moved_rows=int(moved.sum()),
                   spearman_slot_vs_leg=float(merged.prediction_leg.corr(merged.prediction_slot,
                                                                         method="spearman")),
                   warnings=warnings)
    OUT.mkdir(parents=True, exist_ok=True)
    slot.to_csv(OUT / ("dryrun_%s_%s.csv" % (SLOT, live_date.date())), index=False)
    (OUT / "dryrun.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
