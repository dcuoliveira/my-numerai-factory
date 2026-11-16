"""
Ridge with an UNPENALISED one-hot block (fixed effects), sklearn semantics otherwise.

Objective, identical to sklearn.linear_model.Ridge(fit_intercept=True) except
that the coefficients of the one-hot columns carry no penalty:

    min_{b, g}  || y - X b - D g ||^2  +  alpha * ||b||^2

X are the penalised columns, D the one-hot block. Because D is unpenalised
the problem separates (Frisch-Waugh-Lovell): demean X and y within each
one-hot group over the TRAINING rows, solve plain ridge on the demeaned data,
then each group's effect is  mean_g(y) - mean_g(X) b.  That is exact, and it
never forms the near-singular [X | D] system -- a full one-hot block plus an
intercept is collinear, so "a very large scale on D" is only approximately the
same thing and is how the test checks it.

Rows are assigned to groups by WHICH one-hot column is 1. A row with no 1 in
the block (a non-exhaustive block) forms its own "none" group, so a block that
does not sum to one is handled rather than mis-assigned. A row with more than
one 1 is rejected: that is not a one-hot block.

A group with no training rows has no identifiable effect. It is predicted at
the row-weighted average effect, ybar - xbar b, i.e. the pooled intercept; the
number of such rows is recorded in `unseen_rows_` on every predict.

With no one-hot columns at all (`unpenalised` empty, or no column matches) this
is exactly sklearn Ridge(fit_intercept=True).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class FixedEffectRidge(BaseEstimator, RegressorMixin):
    """
    :param alpha: penalty on the non-one-hot coefficients, sklearn Ridge units
    :param unpenalised: column NAMES of the one-hot block (DataFrame input), or
        integer positions (array input)
    """

    def __init__(self, alpha=1.0, unpenalised=()):
        self.alpha = alpha
        self.unpenalised = unpenalised

    def _split(self, X):
        if hasattr(X, "columns"):
            names = list(X.columns)
            wanted = set(self.unpenalised)
            dummy = np.array([c in wanted for c in names])
            X = X.to_numpy(dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)
            dummy = np.zeros(X.shape[1], bool)
            dummy[[int(i) for i in self.unpenalised]] = True
        return X[:, ~dummy], X[:, dummy]

    @staticmethod
    def _groups(D):
        if D.shape[1] == 0:
            return np.zeros(len(D), np.int64)
        hits = (D > 0.5)
        count = hits.sum(axis=1)
        if (count > 1).any():
            raise ValueError("FixedEffectRidge: %d rows have more than one active "
                             "one-hot column" % int((count > 1).sum()))
        # 0 = "no active column"; column j -> j + 1.
        return np.where(count == 1, hits.argmax(axis=1) + 1, 0)

    def fit(self, X, y):
        Xp, D = self._split(X)
        y = np.asarray(y, dtype=np.float64)
        g = self._groups(D)
        n_groups = D.shape[1] + 1
        counts = np.bincount(g, minlength=n_groups).astype(np.float64)
        seen = counts > 0
        safe = np.where(seen, counts, 1.0)
        mx = np.zeros((n_groups, Xp.shape[1]))
        np.add.at(mx, g, Xp)
        mx /= safe[:, None]
        my = np.bincount(g, weights=y, minlength=n_groups) / safe
        Xc = Xp - mx[g]
        yc = y - my[g]
        gram = Xc.T @ Xc
        gram[np.diag_indices_from(gram)] += float(self.alpha)
        self.coef_ = np.linalg.solve(gram, Xc.T @ yc) if Xp.shape[1] else np.zeros(0)
        effect = my - mx @ self.coef_
        pooled = y.mean() - Xp.mean(axis=0) @ self.coef_
        self.group_effect_ = np.where(seen, effect, pooled)
        self.group_seen_ = seen
        self.pooled_effect_ = pooled
        self.n_unpenalised_ = D.shape[1]
        return self

    def predict(self, X):
        Xp, D = self._split(X)
        if D.shape[1] != self.n_unpenalised_:
            raise ValueError("one-hot width changed between fit and predict")
        g = self._groups(D)
        self.unseen_rows_ = int((~self.group_seen_[g]).sum())
        return Xp @ self.coef_ + self.group_effect_[g]


class FixedEffectRidgeWrapper:
    """The E170 ridge search space (logspace 1e-4..1e2, 100 alphas), FE estimator."""

    def __init__(self, unpenalised, seed=None):
        del seed
        self.model_name = "fe_ridge"
        self.search_type = "random"
        self.param_grid = {"alpha": np.logspace(-4, 2, 100)}
        self.ModelClass = FixedEffectRidge(unpenalised=tuple(unpenalised))
