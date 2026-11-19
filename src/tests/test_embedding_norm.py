"""E189 transformation library: formulas, reference axes, causality, ties, constants, small groups."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.data_mani.history_features import bin_ties_preserved
from src.prediction import embedding_norm as N

RNG = np.random.default_rng(189)


def naive(tf, R, X):
    """Element-by-element definition, independent of the vectorised code."""
    out = np.empty(X.shape)
    n = R.shape[0]
    for j in range(X.shape[1]):
        r = R[:, j]
        for i, x in enumerate(X[:, j]):
            L, E = (r < x).sum(), (r == x).sum()
            if r.max() - r.min() < N.EPS or n == 1:
                out[i, j] = N.NEUTRAL[tf]
            elif tf in ("Z", "CZ"):
                sd = r.std()
                v = N.NEUTRAL[tf] if sd < N.EPS else (x - r.mean()) / sd
                out[i, j] = np.clip(v, -3, 3) if tf == "CZ" else v
            elif tf == "ROB":
                q = np.quantile(r, [0.25, 0.5, 0.75])
                out[i, j] = 0.0 if q[2] - q[0] < N.EPS else (x - q[1]) / (q[2] - q[0])
            elif tf == "MM":
                out[i, j] = min(max((x - r.min()) / (r.max() - r.min()), 0), 1)
            elif tf == "PCT":
                out[i, j] = (L + 0.5 * E) / n
            elif tf == "RNK":
                out[i, j] = norm.ppf(min(max((L + 0.5 * E) / n, 0.5 / n), 1 - 0.5 / n))
            else:
                q = min(max((L + 0.5 * E - 0.5) / (n - 1), 0), 1)
                out[i, j] = N.BIN_LABELS[int(q > .05) + int(q > .25) + int(q > .75) + int(q > .95)]
    return out


def tied(n, p):
    """Values with many ties, like float16 states."""
    return np.round(RNG.normal(size=(n, p)), 1)


@pytest.mark.parametrize("tf", N.TRANSFORMS)
def test_reference_matches_definition_out_of_sample(tf):
    R, X = tied(37, 5), tied(23, 5) * 1.7          # X reaches outside R's range
    assert np.allclose(N.Reference(tf, R).apply(X), naive(tf, R, X), atol=1e-12)


@pytest.mark.parametrize("tf", N.TRANSFORMS)
def test_in_sample_equals_reference(tf):
    X = tied(41, 6)
    assert np.allclose(N.in_sample(tf, X), N.Reference(tf, X).apply(X), atol=1e-12)
    assert np.allclose(N.in_sample(tf, X), naive(tf, X, X), atol=1e-12)


def test_bin_reproduces_repo_rule_with_ties():
    for n in (2, 3, 7, 20, 244, 300):
        X = np.round(RNG.normal(size=(n, 8)), 1)
        got = N.in_sample("BIN", X)
        ref = bin_ties_preserved(pd.DataFrame(X), np.zeros(n)).to_numpy()
        assert np.array_equal(got, ref), n


def test_rank_transforms_are_bounded_and_ordered():
    R, X = tied(50, 3), np.array([[-99.0] * 3, [99.0] * 3])
    p = N.Reference("PCT", R).apply(X)
    assert (p[0] == 0).all() and (p[1] == 1).all()
    z = N.Reference("RNK", R).apply(X)
    assert np.allclose(z[1], norm.ppf(1 - 0.5 / 50)) and np.allclose(z[0], -z[1])
    b = N.Reference("BIN", R).apply(X)
    assert (b[0] == 0).all() and (b[1] == 1).all()


@pytest.mark.parametrize("tf", N.TRANSFORMS)
def test_constant_and_single_reference_give_neutral(tf):
    const = np.ones((10, 2))
    X = np.array([[0.0, 5.0], [1.0, 1.0]])
    assert (N.Reference(tf, const).apply(X) == N.NEUTRAL[tf]).all()
    assert (N.Reference(tf, X[:1]).apply(X) == N.NEUTRAL[tf]).all()
    assert (N.in_sample(tf, const) == N.NEUTRAL[tf]).all()


def test_nan_is_refused_not_imputed():
    X = tied(5, 2)
    X[0, 0] = np.nan
    with pytest.raises(AssertionError):
        N.Reference("Z", X)


def panel(nd=6, na=30, p=4):
    dates = np.repeat(np.arange(nd), na)
    assets = np.tile(np.arange(na), nd)
    return tied(nd * na, p), dates, assets


@pytest.mark.parametrize("tf", N.TRANSFORMS)
def test_cs_uses_only_its_date(tf):
    X, d, _ = panel()
    out, _ = N.snapshot(tf, "CS", X, d)
    Y = X.copy()
    Y[d != 2] = RNG.normal(size=Y[d != 2].shape) * 50    # change every other date
    out2, _ = N.snapshot(tf, "CS", Y, d)
    assert np.array_equal(out[d == 2], out2[d == 2])
    assert np.allclose(out[d == 2], N.in_sample(tf, X[d == 2]))


@pytest.mark.parametrize("tf", N.TRANSFORMS)
def test_dim_uses_only_its_row(tf):
    X, d, _ = panel()
    out, _ = N.snapshot(tf, "DIM", X, d)
    for i in (0, 17, 99):
        assert np.allclose(out[i], N.in_sample(tf, X[i:i + 1].T).T[0])


def test_grp_uses_its_group_and_falls_back_to_cs():
    X, d, _ = panel(nd=2, na=60)
    g = np.tile(np.r_[np.zeros(25), np.ones(25), np.full(10, 2)], 2)   # group 2 has 10 < MIN_GROUP
    out, fb = N.snapshot("Z", "GRP", X, d, g)
    assert fb == 20
    m = (d == 0) & (g == 0)
    assert np.allclose(out[m], N.in_sample("Z", X[m]))
    small = (d == 1) & (g == 2)
    assert np.allclose(out[small], N.in_sample("Z", X[d == 1])[g[d == 1] == 2])


@pytest.mark.parametrize("tf", N.TRANSFORMS)
@pytest.mark.parametrize("axis", N.TRAIN_AXES)
def test_train_reference_ignores_non_fit_rows(tf, axis):
    """Validation/test rows (and anything outside the fit rows) cannot move the transform."""
    X, d, a = panel(nd=8, na=12)
    fit, app = d < 5, np.ones(len(d), bool)
    out, _ = N.train_referenced(tf, axis, X, a, fit, app)
    Y = X.copy()
    Y[~fit] = RNG.normal(size=Y[~fit].shape) * 50
    out2, _ = N.train_referenced(tf, axis, Y, a, fit, app)
    assert np.array_equal(out[fit], out2[fit])            # training rows: unchanged
    # a test row maps through the frozen training reference only
    ref = N.Reference(tf, X[fit]) if axis == "POOL" else None
    if axis == "POOL":
        assert np.allclose(out2[~fit], ref.apply(Y[~fit]))


def test_ts_reference_is_the_assets_own_training_rows_with_pool_fallback():
    X, d, a = panel(nd=40, na=5)
    a = a.copy()
    a[(a == 4) & (d < 35)] = 99          # asset 4 has only 5 dates; asset 99 never in the apply set
    fit, app = d < 35, d >= 0
    app[a == 99] = False
    out, info = N.train_referenced("Z", "TS", X, a, fit, app)
    m = (a == 1)
    assert np.allclose(out[m], N.Reference("Z", X[m & fit]).apply(X[m]))
    assert info["fallback_assets"] == 1 and info["fallback_rows"] == 5
    assert np.allclose(out[a == 4], N.Reference("Z", X[fit]).apply(X[a == 4]))
    assert np.isnan(out[a == 99]).all()


def test_deterministic():
    X, d, a = panel()
    for tf in N.TRANSFORMS:
        for ax in N.SNAPSHOT_AXES:
            g = a % 2
            assert np.array_equal(N.snapshot(tf, ax, X, d, g)[0], N.snapshot(tf, ax, X, d, g)[0])
        for ax in N.TRAIN_AXES:
            f = d < 3
            assert np.array_equal(N.train_referenced(tf, ax, X, a, f, ~f)[0][~f],
                                  N.train_referenced(tf, ax, X, a, f, ~f)[0][~f])
