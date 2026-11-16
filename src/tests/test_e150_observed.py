"""Ensure the actual experiment fits candidates and comparators on identical rows."""
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_mani import build_vol_causal as V


def module():
    name = "e150_observed_test_runner"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "experiments/e150_macrovol_evaluation.py"
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


def test_gate_arms_never_combine_opposite_gates():
    m = module()
    cols = ["feature_base_" + str(i) for i in range(22)] + V.COLS + V.TSR_COLS + list(m.MV.columns())
    arms = m.arms(cols, {})
    assert not any("_hi_" in c or "_lo_" in c for c in arms["ALL"])
    for source in m.SOURCES:
        assert not any("_lo_" in c for c in arms["GATE_" + source.upper() + "_HI"])
        assert not any("_hi_" in c for c in arms["GATE_" + source.upper() + "_LO"])


def test_candidate_and_comparators_use_same_observed_rows():
    m = module()
    dates = pd.date_range("2024-01-01", periods=12)
    frame = pd.DataFrame(dict(symbol="A", date=dates, covered=True, feature_base=1.,
                              feature_vol=2., feature_macro=np.arange(12, dtype=float), target=np.arange(12)))
    frame.loc[[2, 4, 9], "feature_macro"] = np.nan
    arms = {"B_COV": ["feature_base"], "VC_COV": ["feature_base", "feature_vol"],
            "ALL": ["feature_base", "feature_vol", "feature_macro"]}
    seen = []

    def predictor(model, params, seed, n_jobs, x, y, xt):
        assert np.isfinite(x).all() and np.isfinite(xt).all()
        seen.append(y.copy())
        return np.arange(len(xt), dtype=float)

    args = SimpleNamespace(arms=["ALL"], tune=False, model="ridge", n_jobs=1)
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(m, "PRED", directory), patch.object(m, "MIN_TRAIN", 2), \
            patch.object(m, "load", return_value=(frame, {})), patch.object(m, "arms", return_value=arms), \
            patch.object(m.H, "blocks", return_value=[(1, dates[:8], dates[8:])]), \
            patch.object(m.H, "T20", "target"), patch.object(m.H, "fit_predict", side_effect=predictor):
        m.fit(args)
        assert len(seen) == 3
        for rows in seen:
            np.testing.assert_array_equal(rows, [0, 1, 3, 5, 6, 7])
        output = pd.read_parquet(Path(directory) / "ridge/ALL_s0.parquet")
        assert output.date.tolist() == [dates[8], dates[10], dates[11]]
