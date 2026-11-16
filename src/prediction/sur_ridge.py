"""
Multi-symbol multivariate ridge -- feasible GLS across the cross-section.

The deployed linear slots fit ONE shared coefficient vector on the STACKED panel:
every (symbol, date) row is an independent observation and the errors are treated
as i.i.d. That is pooled OLS/ridge -- `ridge_glmnet`. It ignores the fact that on
any given date the residuals of different coins move together (a market-wide shock
lifts or drops most of them at once), which is exactly the cross-equation error
correlation a Seemingly-Unrelated-Regressions (SUR) / multivariate model exists to
exploit.

`SurRidge` keeps a single shared coefficient vector -- per-symbol coefficients are
hopeless here (460 coins, many under 60 rows, up to 812 features) and would need
the symbol routed to predict() as well -- but replaces the i.i.d. error assumption
with a FACTOR model of the cross-symbol covariance and does feasible GLS:

    e_{i,t} = lambda_i' g_t + u_{i,t}          (k common factors g_t, idiosyncratic u)

The k common factors capture the correlated part of the errors; controlling for
them while fitting beta is algebraically the GLS adjustment under a factor
covariance (it whitens the direction the errors share). This is the panel analogue
of factor-adjusted regression: fit beta having partialled out the common
cross-symbol shocks, then predict with x'beta -- no symbol needed at predict time,
and no future factor scores assumed.

Nesting is exact and is the control: n_factors=0 makes SurRidge identical to a
plain stacked ridge on the same standardised design, so the whole value of the
method is the paired delta between k>0 and k=0. The symbol of each row is routed to
fit() the same way the DATE already is (WANTS_SYMBOL, positional, aligned to X).

Classification (the pipeline's binary up/down reframing) fits a logistic head on
the same factor-augmented design and predicts from the feature block alone.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SurRidge(RegressorMixin, BaseEstimator):
    """
    Ridge with a cross-symbol factor-GLS adjustment.

    :param alpha: L2 penalty on the feature coefficients, on the standardised
                  design (columns z-scored, so alpha is scale-free)
    :param n_factors: k, the number of common cross-symbol factors partialled out
                      of the fit. 0 reduces EXACTLY to a stacked ridge.
    :param min_symbol_rows: symbols with fewer rows than this do not contribute
                            to the residual covariance (too short to estimate a
                            loading); they still get predictions from the shared
                            beta like everything else
    :param task: "regression" or "classification" (binary, from the pipeline)
    :param fit_intercept: unpenalised intercept by centring
    :param random_state: stored, never read -- the fit is deterministic
    """

    def __init__(self, alpha=1.0, n_factors=1, min_symbol_rows=60,
                 task="regression", fit_intercept=True, random_state=None):
        self.alpha = alpha
        self.n_factors = n_factors
        self.min_symbol_rows = min_symbol_rows
        self.task = task
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _standardise(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0
        return (X - mean) / std, mean, std

    def _common_factors(self, resid, symbol, period):
        """
        k common cross-symbol factor scores, one row per observation.

        The correlated part of the residuals is estimated as a symbol-by-symbol
        covariance over shared dates; its top-k eigenvectors are the symbol
        loadings, and a row's factor score is the loading of its symbol times the
        date's mean-removed residual. Returns an (n, k) array aligned to `resid`,
        or an (n, 0) array when there is nothing to extract.
        """
        n = len(resid)
        symbols, sym_idx = np.unique(symbol, return_inverse=True)
        periods, per_idx = np.unique(period, return_inverse=True)
        S, T = len(symbols), len(periods)
        if self.n_factors <= 0 or S < 2 or T < 3:
            return np.zeros((n, 0))

        # Residual panel: dates x symbols, NaN where a coin is absent that date.
        panel = np.full((T, S), np.nan)
        panel[per_idx, sym_idx] = resid
        counts = np.sum(~np.isnan(panel), axis=0)
        live = counts >= max(self.min_symbol_rows, 3)
        if live.sum() < 2:
            return np.zeros((n, 0))

        # De-mean each date across the coins present (a factor is cross-sectional
        # co-movement, not the common level the ranking never sees), then fill
        # gaps with 0 so absent coins contribute nothing to the covariance.
        row_mean = np.nanmean(panel, axis=1, keepdims=True)
        centred = panel - row_mean
        filled = np.nan_to_num(centred, nan=0.0)
        filled = filled[:, live]

        # Top-k symbol loadings = the top-k right singular vectors of the
        # residual panel, identical (up to sign, which cancels in loading*score)
        # to the leading eigenvectors of the symbol covariance filled'filled --
        # but O(T*S*k) via randomized SVD instead of a full O(S^3) eigh. On a
        # 460-symbol block that is ~0.01s a fit against ~1.1s, the difference
        # between a runnable search and a multi-day one.
        from sklearn.utils.extmath import randomized_svd
        k = int(min(self.n_factors, filled.shape[1], filled.shape[0]))
        if k < 1:
            return np.zeros((n, 0))
        _, _, vt = randomized_svd(filled, n_components=k, n_iter=7,
                                  random_state=0)
        loadings_live = vt.T                                 # S_live x k
        loadings = np.zeros((S, k))
        loadings[live] = loadings_live

        # Per-date factor scores g_t = mean over present coins of loading*residual,
        # then map back to each row. Uses centred residuals so the level is gone.
        scores = np.zeros((T, k))
        present = ~np.isnan(centred)
        cen0 = np.nan_to_num(centred, nan=0.0)
        for j in range(k):
            w = loadings[:, j]
            num = cen0 @ w
            den = present @ (w ** 2)
            den[den < 1e-12] = 1.0
            scores[:, j] = num / den
        # Row factor design: score of the row's date times the symbol's loading.
        F = scores[per_idx] * loadings[sym_idx]
        # Drop any degenerate (all-zero) factor columns.
        keep = F.std(axis=0) > 1e-12
        return F[:, keep] if keep.any() else np.zeros((n, 0))

    # ------------------------------------------------------------------ #
    def fit(self, X, y, period=None, symbol=None):
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64").ravel()
        n, p = X.shape

        Xs, self.x_mean_, self.x_std_ = self._standardise(X)
        self.y_mean_ = y.mean() if self.fit_intercept else 0.0
        yc = y - self.y_mean_

        # Stage 1: plain stacked ridge, to get residuals for the factor model.
        gram = Xs.T @ Xs
        ridge = gram + self.alpha * np.eye(p)
        beta0 = np.linalg.solve(ridge, Xs.T @ yc)

        F = np.zeros((n, 0))
        if self.n_factors > 0 and symbol is not None and period is not None:
            resid = yc - Xs @ beta0
            F = self._common_factors(resid, np.asarray(symbol), np.asarray(period))
        self.n_factors_used_ = F.shape[1]

        if F.shape[1] == 0:
            # Exact stacked-ridge control (n_factors=0 or nothing extractable).
            self.coef_ = beta0
        else:
            # Feasible GLS: refit beta having partialled out the common factors.
            # Penalise only the feature block; the factor controls are free.
            Z = np.hstack([Xs, F])
            pen = np.concatenate([np.full(p, self.alpha), np.zeros(F.shape[1])])
            zg = Z.T @ Z + np.diag(pen)
            coef_full = np.linalg.solve(zg, Z.T @ yc)
            self.coef_ = coef_full[:p]        # predict from the feature block only

        if self.task == "classification":
            # A logistic head on the same factor-augmented, standardised design;
            # predict_proba uses the feature block alone (no future factors).
            from sklearn.linear_model import LogisticRegression
            Z = np.hstack([Xs, F]) if F.shape[1] else Xs
            clf = LogisticRegression(C=1.0 / max(self.alpha, 1e-8),
                                     solver="lbfgs", max_iter=1000)
            try:
                clf.fit(Z, (y > np.median(y)).astype(int))
                self.clf_coef_ = clf.coef_.ravel()[:p]
                self.clf_intercept_ = float(clf.intercept_[0])
            except Exception:
                self.clf_coef_ = self.coef_
                self.clf_intercept_ = 0.0
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype="float64")
        Xs = (X - self.x_mean_) / self.x_std_
        if self.task == "classification":
            return Xs @ self.clf_coef_ + self.clf_intercept_
        return Xs @ self.coef_ + self.y_mean_

    def predict(self, X):
        return self.decision_function(X)

    def predict_proba(self, X):
        z = self.decision_function(X)
        pos = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - pos, pos])


# --------------------------------------------------------------------------- #
# Generic factor-augmented SUR, for ANY base learner (trees included)
# --------------------------------------------------------------------------- #
def cross_symbol_factors(resid, symbol, period, n_factors, min_symbol_rows):
    """
    k common cross-symbol factor scores from a residual vector, one row per obs.

    Module-level twin of SurRidge._common_factors, so a tree wrapper and the
    linear ridge extract factors the SAME way. Top-k symbol loadings are the
    leading right singular vectors of the (date x symbol) residual panel; a
    row's score is its date's factor value times its symbol's loading. Returns
    (n, k) aligned to `resid`, or (n, 0) when nothing is extractable.
    """
    resid = np.asarray(resid, dtype="float64")
    n = len(resid)
    if n_factors <= 0:
        return np.zeros((n, 0))
    symbols, sym_idx = np.unique(np.asarray(symbol), return_inverse=True)
    periods, per_idx = np.unique(np.asarray(period), return_inverse=True)
    S, T = len(symbols), len(periods)
    if S < 2 or T < 3:
        return np.zeros((n, 0))
    panel = np.full((T, S), np.nan)
    panel[per_idx, sym_idx] = resid
    live = np.sum(~np.isnan(panel), axis=0) >= max(min_symbol_rows, 3)
    if live.sum() < 2:
        return np.zeros((n, 0))
    centred = panel - np.nanmean(panel, axis=1, keepdims=True)
    filled = np.nan_to_num(centred, nan=0.0)[:, live]
    from sklearn.utils.extmath import randomized_svd
    k = int(min(n_factors, filled.shape[1], filled.shape[0]))
    if k < 1:
        return np.zeros((n, 0))
    _, _, vt = randomized_svd(filled, n_components=k, n_iter=7, random_state=0)
    loadings = np.zeros((S, k))
    loadings[live] = vt.T
    scores = np.zeros((T, k))
    present = ~np.isnan(centred)
    cen0 = np.nan_to_num(centred, nan=0.0)
    for j in range(k):
        w = loadings[:, j]
        den = present @ (w ** 2)
        den[den < 1e-12] = 1.0
        scores[:, j] = (cen0 @ w) / den
    F = scores[per_idx] * loadings[sym_idx]
    keep = F.std(axis=0) > 1e-12
    return F[:, keep] if keep.any() else np.zeros((n, 0))


class SurAugment(RegressorMixin, BaseEstimator):
    """
    Wrap ANY base estimator with the cross-symbol factor-GLS adjustment.

    The linear SurRidge partials the k common factors out of a penalised solve;
    a tree has no coefficients to partial, so the uniform generalisation is:
    add the k factor scores as EXTRA FEATURES while fitting (the learner explains
    away the common cross-symbol shocks in-sample), then PREDICT with those
    columns ZEROED -- future factor scores are unknowable, exactly as SurRidge
    predicts x'beta without F. n_factors=0 fits the base on X alone, an EXACT
    stacked-model control. With a Ridge base this reproduces SurRidge (predicting
    [X|0] on a fit of [X|F] with the F block unpenalised is x'beta).

    Factors are extracted from a cheap stage-1 ridge's residuals so every family
    shares one, model-agnostic, notion of "common cross-symbol shock". `estimator`
    is a base sklearn estimator; get_params exposes its params as estimator__*, so
    the search ranges over base hyperparameters x n_factors.

    :param estimator: base sklearn estimator (cloned per fit)
    :param n_factors: k common cross-symbol factors to add as features
    :param min_symbol_rows: coins shorter than this do not seed a loading
    :param ridge_alpha: penalty of the stage-1 ridge used only for factor residuals
    :param task: "regression" or "classification" (the pipeline's binary reframing)
    """

    def __init__(self, estimator=None, n_factors=1, min_symbol_rows=60,
                 ridge_alpha=100.0, task="regression", random_state=None):
        self.estimator = estimator
        self.n_factors = n_factors
        self.min_symbol_rows = min_symbol_rows
        self.ridge_alpha = ridge_alpha
        self.task = task
        self.random_state = random_state

    def _factors_for_fit(self, X, y, period, symbol):
        n, p = X.shape
        if self.n_factors <= 0 or symbol is None or period is None:
            return np.zeros((n, 0))
        mean = X.mean(axis=0); std = X.std(axis=0); std[std < 1e-12] = 1.0
        Xs = (X - mean) / std
        yf = np.asarray(y, dtype="float64"); yc = yf - yf.mean()
        try:
            beta = np.linalg.solve(Xs.T @ Xs + self.ridge_alpha * np.eye(p), Xs.T @ yc)
            resid = yc - Xs @ beta
        except np.linalg.LinAlgError:
            resid = yc
        return cross_symbol_factors(resid, symbol, period,
                                    self.n_factors, self.min_symbol_rows)

    def fit(self, X, y, period=None, symbol=None):
        from sklearn.base import clone
        X = np.asarray(X, dtype="float64"); y = np.asarray(y).ravel()
        F = self._factors_for_fit(X, y, period, symbol)
        self.n_factors_used_ = F.shape[1]
        Z = np.hstack([X, F]) if F.shape[1] else X
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Z, y)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def _augment(self, X):
        X = np.asarray(X, dtype="float64")
        k = getattr(self, "n_factors_used_", 0)
        return np.hstack([X, np.zeros((len(X), k))]) if k else X

    def predict(self, X):
        return self.estimator_.predict(self._augment(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._augment(X))
