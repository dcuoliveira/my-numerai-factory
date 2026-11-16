import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge

from src.prediction.fe_ridge import FixedEffectRidge


def panel(seed=0, n=600, p=5, k=7, exhaustive=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    g = rng.integers(0, k, size=n)
    D = np.eye(k)[g]
    if not exhaustive:
        D[g == 0] = 0.0
    y = X @ rng.normal(size=p) + 3 * rng.normal(size=k)[g] + rng.normal(size=n)
    cols = [f"x{i}" for i in range(p)] + [f"d{j}" for j in range(k)]
    return pd.DataFrame(np.c_[X, D], columns=cols), y, cols[p:]


class FixedEffectRidgeTest(unittest.TestCase):

    def test_matches_ridge_with_dummies_scaled_to_no_penalty(self):
        for exhaustive in (True, False):
            X, y, dummies = panel(exhaustive=exhaustive)
            alpha = 25.0
            fe = FixedEffectRidge(alpha=alpha, unpenalised=dummies).fit(X, y)
            # Penalty on c*d is alpha*(g/c)^2 -> 0 as c grows.
            scaled = X.copy()
            scaled[dummies] *= 1e4
            ref = Ridge(alpha=alpha, solver="cholesky").fit(scaled, y)
            np.testing.assert_allclose(fe.coef_, ref.coef_[:5], atol=1e-6)
            np.testing.assert_allclose(fe.predict(X), ref.predict(scaled), atol=1e-5)

    def test_reduces_to_sklearn_ridge_without_dummies(self):
        X, y, dummies = panel()
        Xp = X.drop(columns=dummies)
        fe = FixedEffectRidge(alpha=3.0, unpenalised=dummies).fit(Xp, y)
        ref = Ridge(alpha=3.0).fit(Xp, y)
        np.testing.assert_allclose(fe.predict(Xp), ref.predict(Xp), atol=1e-9)

    def test_unseen_group_gets_pooled_intercept_and_is_counted(self):
        X, y, dummies = panel()
        train = X["d3"] == 0
        fe = FixedEffectRidge(alpha=1.0, unpenalised=dummies).fit(X[train], y[train])
        pred = fe.predict(X[~train])
        self.assertEqual(fe.unseen_rows_, int((~train).sum()))
        xp = X[~train].drop(columns=dummies).to_numpy()
        np.testing.assert_allclose(pred, xp @ fe.coef_ + fe.pooled_effect_)

    def test_clone_and_array_input(self):
        X, y, dummies = panel()
        a = clone(FixedEffectRidge(alpha=2.0, unpenalised=dummies)).fit(X, y).predict(X)
        pos = [X.columns.get_loc(c) for c in dummies]
        b = FixedEffectRidge(alpha=2.0, unpenalised=pos).fit(X.to_numpy(), y).predict(X.to_numpy())
        np.testing.assert_allclose(a, b)

    def test_rejects_multi_hot(self):
        X, y, dummies = panel()
        X.loc[0, dummies[:2]] = 1.0
        with self.assertRaises(ValueError):
            FixedEffectRidge(unpenalised=dummies).fit(X, y)


if __name__ == "__main__":
    unittest.main()
