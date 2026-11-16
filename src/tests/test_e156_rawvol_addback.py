"""Guard the counterfactual feature definition and paired comparison direction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import e156_rawvol_addback as E


def test_removes_exactly_two_shipped_columns_and_adds_them_back():
    full = ["feature_x" + str(i) for i in range(20)] + E.RAW
    arms = E.feature_arms(["date", "target_binned_return_20"] + full)
    assert len(arms["EXRAWVOL"]) == 20
    assert len(arms["ADD20"]) == len(arms["ADD60"]) == 21
    assert arms["FULL"] == full
    assert set(arms["FULL"]) - set(arms["EXRAWVOL"]) == set(E.RAW)
    assert set(arms["ADD20"]) - set(arms["EXRAWVOL"]) == {E.RAW[0]}
    assert set(arms["ADD60"]) - set(arms["EXRAWVOL"]) == {E.RAW[1]}
    with pytest.raises(ValueError):
        E.feature_arms(full + ["feature_extra"])


def test_positive_delta_means_full_model_beats_counterfactual_baseline():
    rng = np.random.default_rng(156)
    dates = pd.date_range("2024-01-01", periods=80)
    baseline = pd.DataFrame(dict(date=dates, block=np.repeat([0, 1], 40), corr=rng.normal(size=80)))
    full = baseline.copy()
    full["corr"] += np.linspace(.001, .003, 80)
    result = E.describe_delta(full, baseline, "corr")
    assert result["delta"] == pytest.approx(.002)
    assert result["t"] > 0
    assert result["win"] == 1
    assert result["n"] == 80
