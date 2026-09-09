"""
Tests for the level-space FARM validation harness (src/prediction/farm_paper.py).

The harness exists to prove the E27 estimators are correct by reproducing the
paper's positive result. These tests pin the pieces that the FRED-MD run itself
cannot check in isolation:

  * the McCracken-Ng transform codes are applied correctly (a wrong code silently
    de-stationarises a series and every downstream number drifts);
  * the individual forecasts are real-time -- a forecast of y_{t+h} never sees
    data past its origin t;
  * the level `equal` combiner IS the arithmetic mean of the forecasts;
  * a SYNTHETIC POSITIVE CONTROL: on data built so FARM1 must win, it does. This
    is the offline half of the validation -- it shows the estimators can beat the
    simple average when the signal is there, which the crypto null (E27) cannot,
    without waiting on the network fetch.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.prediction import farm_paper as fp


# --------------------------------------------------------------------------
# transform codes
# --------------------------------------------------------------------------

def test_transform_codes_match_mccracken_ng():
    # code 2: first difference
    assert np.allclose(fp._apply_tcode([1.0, 3.0, 6.0, 10.0], 2),
                       [np.nan, 2, 3, 4], equal_nan=True)
    # code 5: first difference of logs
    x = np.exp([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(fp._apply_tcode(x, 5), [np.nan, 1, 1, 1], equal_nan=True)
    # code 6: second difference of logs
    assert np.allclose(fp._apply_tcode(x, 6), [np.nan, np.nan, 0, 0], equal_nan=True)
    # code 1: identity
    assert np.allclose(fp._apply_tcode([2.0, 5.0], 1), [2, 5])


def test_unknown_transform_code_is_rejected():
    with pytest.raises(ValueError):
        fp._apply_tcode([1.0, 2.0], 99)


# --------------------------------------------------------------------------
# individual forecasts -- real-time / leak-free
# --------------------------------------------------------------------------

def _toy_panel(n=260, k=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1980-01-01", periods=n, freq="MS")
    return pd.DataFrame(rng.normal(size=(n, k)),
                        columns=["TGT"] + ["x%d" % i for i in range(k - 1)], index=idx)


def test_individual_forecasts_are_real_time():
    """Corrupting the target after some origin must not change any forecast whose
    origin (plus horizon) precedes the corruption."""
    panel = _toy_panel()
    h = 2
    F_clean, y_clean, names, origins = fp.individual_forecasts(panel, "TGT", h)

    dirty = panel.copy()
    cut = 180
    rng = np.random.default_rng(1)
    dirty.iloc[cut:, dirty.columns.get_loc("TGT")] += rng.normal(0, 50, len(dirty) - cut)
    F_dirty, y_dirty, _, origins_d = fp.individual_forecasts(dirty, "TGT", h)

    # forecasts whose target (origin+h) resolved before the corruption are built
    # only from pre-cut data and must be identical.
    safe = np.array([panel.index.get_loc(o) + h < cut for o in origins])
    # allclose, not array_equal: copying the frame perturbs BLAS summation order
    # by ~1e-16 on identical inputs. A real leak moves a forecast by O(1), since
    # the corruption has sd 50 -- 1e-9 cleanly separates the two.
    assert np.allclose(F_clean[safe], F_dirty[safe], atol=1e-9), (
        "an individual forecast changed when only the future target moved")


def test_individual_forecasts_shapes():
    panel = _toy_panel()
    F, y, names, origins = fp.individual_forecasts(panel, "TGT", 1, t1=60)
    assert F.shape[0] == len(y) == len(origins)
    assert F.shape[1] == len(names)
    assert np.isfinite(F).all()


# --------------------------------------------------------------------------
# the level combiner and the positive control
# --------------------------------------------------------------------------

def test_equal_level_combiner_is_the_mean_of_forecasts():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(300, 7))
    y = rng.normal(size=300)
    pred, y_oos = fp.combine(F, y, "equal", "equal", None, t2=150, t3=80)
    assert np.allclose(pred, F.mean(axis=1)[-80:], atol=0)


def test_positive_control_farm1_beats_the_average_when_signal_exists():
    """
    Build forecasts whose average error u_bar = y - f_bar is a genuine sparse
    linear function of the idiosyncratic deviations. FARM1 must recover it and
    crush the simple average; if it does not, the estimators are broken.
    """
    rng = np.random.default_rng(3)
    T, N = 500, 8
    F = rng.normal(size=(T, N))
    fbar = F.mean(axis=1)
    d = F - fbar[:, None]
    beta_star = np.zeros(N)
    beta_star[[1, 4]] = [0.9, -0.7]                # only two members carry signal
    y = fbar + d @ beta_star + rng.normal(0, 0.05, T)

    base, base_y = fp.combine(F, y, "equal", "equal", None, t2=250, t3=150)
    for bm in ("ridge", "l2boost", "postalasso"):
        pred, y_oos = fp.combine(F, y, "farm1.%s" % bm, "farm1", bm, t2=250, t3=150)
        rel = ((y_oos - pred) ** 2).mean() / ((base_y - base) ** 2).mean()
        assert rel < 0.5, "FARM1.%s failed the positive control (relMSFE %.3f)" % (bm, rel)


def test_gr_reduces_to_average_direction_on_a_pure_factor():
    """When forecasts are a common factor plus tiny noise and y equals it, both
    equal and GR should track it closely (sanity that GR is estimable)."""
    rng = np.random.default_rng(4)
    T, N = 400, 5
    factor = rng.normal(size=T)
    F = factor[:, None] + rng.normal(0, 0.01, (T, N))
    y = factor + rng.normal(0, 0.01, T)
    pred, y_oos = fp.combine(F, y, "gr", "gr", None, t2=200, t3=100)
    assert np.corrcoef(pred, y_oos)[0, 1] > 0.99


# --------------------------------------------------------------------------
# the vendored data loads (offline)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists("data/fredmd/fred-md.csv"),
                    reason="vendored FRED-MD not present")
def test_vendored_fredmd_loads_with_the_five_targets():
    panel = fp.load_fredmd("data/fredmd/fred-md.csv")
    assert panel.shape[0] > 600                    # ~ 716 monthly rows from 1960
    for target in fp.TARGETS:
        assert target in panel.columns, "target %s missing after transform" % target
    assert panel.notna().all().all(), "load should drop columns with any NaN"
    assert panel.index.is_monotonic_increasing
