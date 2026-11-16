"""
The E107 transform as live runs it must be the transform the backtest priced.

Three things are pinned: src.prediction.meta_resid agrees with the copy inside
experiments/e107_residualise_everything_crypto.py; the live wrapper
(blend.MetaResidual inside a RankBlend) reproduces a run_ensemble-style
per-period rank average of residualised forecasts; and the lagged-meta attach
obeys the lag and the staleness guard.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from src.live import registry
from src.live.blend import META_COL, MetaResidual, RankBlend
from src.live.features import add_lagged_meta
from src.prediction import meta_resid

E107 = os.path.join("experiments", "e107_residualise_everything_crypto.py")


def _e107():
    spec = importlib.util.spec_from_file_location("e107", E107)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Const(object):
    """An 'estimator' that returns a fixed vector, whatever it is handed."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype="float64")

    def predict(self, X):
        return self.values[:len(X)]


def test_gauss_rank_matches_the_e107_copy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    x[[3, 17, 50]] = np.nan
    x[[5, 6]] = x[7]            # ties
    assert np.array_equal(meta_resid.gauss_rank(x), _e107().gauss_rank(x))


def test_residualise_matches_the_e107_formula():
    rng = np.random.default_rng(1)
    p, m = rng.normal(size=300), rng.normal(size=300)
    m[:40] = np.nan
    e107 = _e107()
    gp, gm = e107.gauss_rank(p), e107.gauss_rank(m)
    beta = (gp @ gm) / (gm @ gm)
    r, b = meta_resid.residualise(p, m, 0.5)
    assert b == pytest.approx(beta, abs=1e-15)
    assert np.allclose(r, gp - e107.STRENGTH * beta * gm, atol=1e-15, rtol=0)
    # And it removes what it says: half the projection onto the meta model.
    assert (r @ gm) / (gm @ gm) == pytest.approx(0.5 * beta, abs=1e-12)


def test_no_snapshot_passes_through():
    p = np.arange(10.0)
    for meta in (None, np.full(10, np.nan)):
        r, beta = meta_resid.residualise(p, meta)
        assert np.isnan(beta)
        assert np.array_equal(r, meta_resid.gauss_rank(p))


def test_lagged_meta_matches_the_e107_lookup(tmp_path):
    meta = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-09"] * 3 + ["2026-10-14"] * 3 + ["2026-10-16"] * 2),
        "symbol": ["A", "B", "C", "A", "B", "C", "A", "B"],
        "prediction": [.1, .2, .3, .4, .5, .6, .7, .8]})
    # live 2026-10-19 minus 5 days is 10-14: the 10-16 snapshot is too new.
    values, snapshot = meta_resid.lagged_meta(meta, ["C", "A", "Z"], pd.Timestamp("2026-10-19"))
    assert snapshot == pd.Timestamp("2026-10-14")
    assert np.allclose(values[:2], [.6, .4]) and np.isnan(values[2])

    e107 = _e107()
    path = tmp_path / "meta.parquet"
    meta.to_parquet(path)
    e107.META_FILE = str(path)
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-10-19"] * 3), "symbol": ["C", "A", "Z"]})
    theirs, src = e107.lagged_meta(frame, 5)
    assert np.allclose(values, theirs, equal_nan=True)
    assert pd.Timestamp(src[0]) == snapshot


def test_live_blend_reproduces_a_rank_average_of_residualised_forecasts():
    """
    What the backtest priced: per period, rank each e107_*_s050 forecast (itself
    rank(r)), average the ranks. What live does: RankBlend over MetaResidual
    members. Same live date, same rows -> same ordering, and the same values up
    to RankBlend's pct-rank convention.
    """
    rng = np.random.default_rng(2)
    n = 250
    p1, p2, p3 = rng.normal(size=(3, n))
    m = rng.normal(size=n) + 0.6 * p1
    m[rng.choice(n, 30, replace=False)] = np.nan
    X = pd.DataFrame({"x": np.zeros(n), META_COL: m})

    blend = RankBlend([
        ("A", MetaResidual("A", Const(p1), ["x"], 0.5), ["x", META_COL], 0.25),
        ("B", Const(p2), ["x"], 0.25),
        ("C", MetaResidual("C", Const(p3), ["x"], 0.5), ["x", META_COL], 0.5)])
    live = blend.predict(X)

    def rank01(v):
        v = pd.Series(v)
        return ((v.rank(method="average") - 0.5) / v.count()).to_numpy()
    backtest = (0.25 * pd.Series(rank01(meta_resid.residualise(p1, m)[0])).rank(pct=True)
                + 0.25 * pd.Series(p2).rank(pct=True)
                + 0.5 * pd.Series(rank01(meta_resid.residualise(p3, m)[0])).rank(pct=True))
    assert np.allclose(live, backtest.to_numpy(), atol=1e-12)
    log = blend.resid_log()
    assert [entry["member"] for entry in log] == ["A", "C"]
    assert all(entry["residualised"] and entry["coverage"] == n - 30 for entry in log)


def test_missing_meta_column_is_a_loud_error_not_a_silent_pass():
    wrapped = MetaResidual("A", Const(np.arange(5.0)), ["x"])
    with pytest.raises(ValueError, match=META_COL):
        wrapped.predict(pd.DataFrame({"x": np.zeros(5)}))


def _panel(live_date):
    return pd.DataFrame({
        "symbol": ["A", "B", "A", "B"],
        "date": pd.to_datetime(["2026-09-01", "2026-09-01", live_date, live_date]),
        "is_live": [False, False, True, True]})


def test_add_lagged_meta_fills_only_the_live_date(tmp_path):
    path = tmp_path / "meta.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-10-14", "2026-10-14"]),
                  "symbol": ["A", "B"], "prediction": [.3, .7]}).to_parquet(path)
    panel, info = add_lagged_meta(_panel("2026-10-19"), pd.Timestamp("2026-10-19"), str(path))
    assert info["status"] == "ok" and info["stale_days"] == 5 and info["coverage"] == 2
    assert panel[META_COL].isna().tolist() == [True, True, False, False]


def test_add_lagged_meta_refuses_a_stale_or_absent_file(tmp_path):
    path = tmp_path / "meta.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-08-08"]), "symbol": ["A"],
                  "prediction": [.3]}).to_parquet(path)
    panel, info = add_lagged_meta(_panel("2026-10-19"), pd.Timestamp("2026-10-19"), str(path))
    assert info["status"].startswith("stale") and panel[META_COL].isna().all()
    panel, info = add_lagged_meta(_panel("2026-10-19"), pd.Timestamp("2026-10-19"),
                                  str(tmp_path / "absent.parquet"))
    assert info["status"] == "missing" and panel[META_COL].isna().all()


def test_residualised_slots_are_wired_as_backtested():
    a = registry.get("CU_SA_IA_BO_MA_R_01")
    assert [(m["name"], m["weight"], m.get("resid")) for m in a["slot_members"]] == [
        ("CU_SA_IA_BO_01", 0.5, 0.5), ("CU_SA_IA_BO_MA_01", 0.5, 0.5)]
    b = registry.get("CU_R_IA_MA_60_01")
    assert [(m["name"], m.get("resid")) for m in b["slot_members"]] == [
        ("CURUPIRA_05", 0.5), ("IA_01", None), ("MA_FENC", None), ("CURUPIRA_05_60", None)]
    for name in ("CU_SA_IA_BO_MA_R_01", "CU_R_IA_MA_60_01"):
        assert name in registry.ORDER and name not in registry.NOT_SUBMITTED
        position = registry.ORDER.index(name)
        for member in registry.get(name)["slot_members"]:
            assert registry.ORDER.index(member["name"]) < position, member["name"]
    # Retired from the account, still fitted: the sibling rung and C_SACI_01 need them.
    for name in ("CURUPIRA_02", "CURUPIRA_03"):
        assert name in registry.ORDER and name in registry.NOT_SUBMITTED
