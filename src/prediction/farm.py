"""
Factor-Adjusted Regularized Model combination (Lee & Lee 2025), the layer that
turns a set of finished member forecasts into ONE combined forecast by treating
the simple average as a factor rather than as a weighting scheme.

    from src.prediction.farm import walk_forward_combination

Like src/prediction/weighting.py this sits ABOVE the members -- they are already
fitted -- but it is a different object. weighting.py chooses a convex weight
vector over the members' SCORES; this regresses the simple average's own error on
the members' IDIOSYNCRATIC deviations and adds back whatever is predictable. The
combined forecast can leave the simplex, which is the whole point: FARM is allowed
to move a rank up or down relative to the average, not merely to reweight.

------------------------------------------------------------------------------
The decomposition (paper Section 3)
------------------------------------------------------------------------------

Members enter as per-period percentile ranks r_i, so f_bar = mean_i r_i is the
simple-average combined forecast and d_i = r_i - f_bar is member i's idiosyncratic
component. Writing u_bar = rank(y) - f_bar for the simple average's own forecast
error (rank space, so u_bar and d_i share a scale), the paper's Equation (6) is

    u_bar = sum_i beta_i d_i + u_c ,     combined = f_bar + sum_i beta_i d_i.

FARM1 treats f_bar as the sole common factor and estimates beta by regressing
u_bar on the d_i. FARM2/FARM3 first strip r <= 2 extra common factors g from the
d_i (PCA / supervised-PCA), project (y - f_bar) on g, and estimate beta on the
factor-removed residual -- Equations (7)-(9). GR is the unregularised
Granger-Ramanathan OLS with weights summing to one; DS is Diebold-Shin's
Egalitarian Lasso, which -- and this distinction is easy to lose -- regresses
u_bar on the FORECASTS f_i, not the deviations d_i, because span{f_i} strictly
contains span{d_i} (sum_i d_i = 0 kills the f_bar direction).

------------------------------------------------------------------------------
Why any of it can beat 1/N, and why it usually does not here
------------------------------------------------------------------------------

The simple average is optimal only under equal error variances. Deviating from it
buys something exactly when a member's d_i carries signal for u_bar that the
average misses. But the estimator minimises SQUARED error, and this book is paid
on cross-sectional rank correlation: lowering MSE pulls the combined forecast
toward the conditional mean, which compresses the very spread the ranking needs.
So relMSFE can fall while numerai_corr does not rise -- the forecast combination
puzzle, and the reason E27 exists as a measured null rather than an assumption.

------------------------------------------------------------------------------
The lag, and why it is not the embargo
------------------------------------------------------------------------------

Every estimated object -- beta, the factor loadings, the ridge alpha and the
factor count r -- is fit only on rows whose 20-day target has RESOLVED: period
position < start - lag. This is the same arithmetic as run_weighted_ensemble's
weight lag and a different mechanism from the CV embargo: it separates the OUTCOME
a fit reads from the decision that uses it, not training rows from validation rows
within one fit. f_bar and d_i are pure transforms of the member predictions and
carry no target, so they are available at prediction time; only the fitted beta
waits for resolved history. Before min_history periods exist the combiner returns
f_bar, exactly as the weighting cold start returns 1/N.
"""

import numpy as np


LAG = 20
UPDATE_EVERY = 22
MIN_HISTORY = 60
ROLL_PERIODS = 252      # rolling estimation window (the paper estimates on T2, rolling)
MAX_FIT_ROWS = 15000    # cap fitting rows; subsample deterministically above this
BOOST_NU = 0.001        # L2-Boosting step size (paper's value)
BOOST_M = 3000          # L2-Boosting iterations (paper's value)
RIDGE_ALPHAS = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
MAX_R = 2               # maximum extra common factors for FARM2/FARM3 (paper's cap)

METHODS = ("equal", "farm1", "farm2", "farm3", "gr", "ds")
BETA_METHODS = ("ridge", "l2boost", "postalasso")


# --------------------------------------------------------------------------- #
# beta estimators for Equation (6): each returns a length-K coefficient vector,
# all reduced to the members' K x K Gram matrix so 3000 boosting steps are free
# --------------------------------------------------------------------------- #
def fit_ridge(design, target, gram, cross):
    """Ridge regression of `target` on `design`, alpha by 3-fold CV.

    The closed form uses the pre-built Gram = design.T @ design and
    cross = design.T @ target; the CV refits on shuffled row folds so a fold is
    not a contiguous block of one symbol's rows.
    """
    n = len(target)
    idx = np.arange(n)
    np.random.RandomState(0).shuffle(idx)
    folds = np.array_split(idx, 3)
    best_alpha, best_mse = RIDGE_ALPHAS[0], np.inf
    eye = np.eye(design.shape[1])
    for alpha in RIDGE_ALPHAS:
        mse = 0.0
        for f in range(3):
            test = folds[f]
            train = np.concatenate([folds[g] for g in range(3) if g != f])
            beta = np.linalg.solve(design[train].T @ design[train]
                                   + alpha * n * eye, design[train].T @ target[train])
            err = target[test] - design[test] @ beta
            mse += err @ err
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    return np.linalg.solve(gram + best_alpha * n * eye, cross)


def fit_l2boost(design, target, gram, cross, nu=BOOST_NU, iterations=BOOST_M):
    """Componentwise L2-Boosting (Buhlmann 2006), started from beta = 0 (i.e. from
    the simple average). Each step selects the single component most correlated
    with the current residual and takes a small step along its univariate slope.

    resid_proj = d_j^T (u - D beta) = cross - gram @ beta needs only the Gram
    matrix, so the loop is K x K per step rather than a pass over the rows.
    """
    k = design.shape[1]
    beta = np.zeros(k)
    gnorm = np.diag(gram).copy()
    gnorm[gnorm <= 0] = 1e-12
    for _ in range(iterations):
        resid_proj = cross - gram @ beta
        component = int(np.argmax(np.abs(resid_proj) / np.sqrt(gnorm)))
        beta[component] += nu * resid_proj[component] / gnorm[component]
    return beta


def fit_postalasso(design, target, gram, cross):
    """Adaptive Lasso selection (Zou 2006) then post-selection OLS (Belloni-
    Chernozhukov 2013). Adaptive weights 1/|b0| from an initial ridge fit rescale
    the columns so weakly-loaded members are effectively unpenalised; the
    selection alpha is picked by BIC along a single LARS path (one pass, no CV
    loop), and the surviving columns are refit by plain OLS.
    """
    from sklearn.linear_model import LassoLarsIC, LinearRegression
    n, k = design.shape
    b0 = np.linalg.solve(gram + 1e-3 * n * np.eye(k), cross)
    weights = 1.0 / (np.abs(b0) + 1e-4)
    scaled = design / weights
    try:
        selected = np.abs(LassoLarsIC(criterion="bic", fit_intercept=False)
                          .fit(scaled, target).coef_) > 1e-10
    except Exception:
        selected = np.ones(k, bool)
    beta = np.zeros(k)
    if selected.any():
        beta[selected] = LinearRegression(fit_intercept=False).fit(
            design[:, selected], target).coef_
    return beta


BETA_FITTERS = {"ridge": fit_ridge, "l2boost": fit_l2boost,
                "postalasso": fit_postalasso}


# --------------------------------------------------------------------------- #
# extra common factors for FARM2 / FARM3
# --------------------------------------------------------------------------- #
def pca_loadings(design, r):
    """Top-r principal directions of the idiosyncratic components (Bai-Ng)."""
    cov = np.cov(design, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    return vectors[:, np.argsort(values)[::-1][:r]]


def spca_loadings(design, target, r):
    """Supervised PCA (Giglio-Xiu-Zhang 2025): one factor per iteration, built by
    selecting the components most correlated with the residual, taking their first
    principal direction, then deflating both the residual and the design on the
    extracted factor. Handles factors that are weak after the simple average is
    removed, which standard PCA cannot.
    """
    k = design.shape[1]
    loadings = []
    residual = target.copy()
    work = design.copy()
    for _ in range(r):
        corr = np.array([np.corrcoef(work[:, j], residual)[0, 1]
                         if work[:, j].std() > 0 else 0.0 for j in range(k)])
        corr = np.nan_to_num(corr)
        selected = np.abs(corr) >= np.median(np.abs(corr))
        if not selected.any():
            selected = np.ones(k, bool)
        cov = np.cov(work[:, selected], rowvar=False)
        vals, vecs = np.linalg.eigh(np.atleast_2d(cov))
        vector = np.zeros(k)
        vector[selected] = vecs[:, np.argmax(vals)]
        loadings.append(vector)
        score = work @ vector
        denom = score @ score + 1e-12
        residual = residual - score * (score @ residual) / denom
        work = work - np.outer(score, score @ work) / denom
    return np.column_stack(loadings) if loadings else np.zeros((k, 0))


def factor_combination(design_train, target_train, design_current,
                       kind, beta_method, r):
    """The added FARM2/FARM3 signal alpha'g + beta'd on the current rows.

    The r factors g = design @ V are projected out of BOTH the target and the
    design (Equation 7) before beta is fit, so beta on the residualised design
    cannot reload the factor subspace that alpha already carries.
    """
    k = design_train.shape[1]
    if r == 0:
        residual = target_train
        design_res_train, design_res_current, alpha, loadings = (
            design_train, design_current, None, np.zeros((k, 0)))
    else:
        loadings = (pca_loadings(design_train, r) if kind == "pca"
                    else spca_loadings(design_train, target_train, r))
        factors = design_train @ loadings
        alpha, *_ = np.linalg.lstsq(factors, target_train, rcond=None)
        residual = target_train - factors @ alpha
        projection, *_ = np.linalg.lstsq(factors, design_train, rcond=None)
        design_res_train = design_train - factors @ projection
        design_res_current = design_current - (design_current @ loadings) @ projection
    gram = design_res_train.T @ design_res_train
    cross = design_res_train.T @ residual
    beta = BETA_FITTERS[beta_method](design_res_train, residual, gram, cross)
    added = design_res_current @ beta
    if r > 0:
        added = added + (design_current @ loadings) @ alpha
    return added


def pick_r(design, target, kind):
    """Choose the factor count r in {0..MAX_R} by 3-fold CV minimising MSE.

    r selection uses a ridge beta-fit regardless of the estimator that will run
    afterward -- r is a structural choice about how many factors the deviations
    carry, and folding the estimator into this loop would call the Lasso path
    thousands of times for a decision it barely moves.
    """
    n = len(target)
    idx = np.arange(n)
    np.random.RandomState(0).shuffle(idx)
    folds = np.array_split(idx, 3)
    best_r, best_mse = 0, np.inf
    for r in range(0, MAX_R + 1):
        mse = 0.0
        for f in range(3):
            test = folds[f]
            train = np.concatenate([folds[g] for g in range(3) if g != f])
            try:
                pred = factor_combination(design[train], target[train],
                                          design[test], kind, "ridge", r)
            except Exception:
                pred = np.zeros(len(test))
            err = target[test] - pred
            mse += err @ err
        if mse < best_mse - 1e-12:
            best_mse, best_r = mse, r
    return best_r


# --------------------------------------------------------------------------- #
# the paper's two reference combiners
# --------------------------------------------------------------------------- #
def fit_gr(forecasts, target):
    """Granger-Ramanathan: OLS of the target on the forecasts with weights
    constrained to sum to one (no intercept), via the KKT closed form. A tiny
    ridge keeps the K x K solve non-singular when members are near-collinear.
    """
    k = forecasts.shape[1]
    ones = np.ones(k)
    try:
        both = np.linalg.solve(forecasts.T @ forecasts + 1e-8 * np.eye(k),
                               np.column_stack([forecasts.T @ target, ones]))
        unconstrained, adjust = both[:, 0], both[:, 1]
        lam = (1 - ones @ unconstrained) / (ones @ adjust)
        return unconstrained + lam * adjust
    except Exception:
        return ones / k


def fit_ds(target, forecasts):
    """Diebold-Shin Egalitarian Lasso: regress u_bar on the FORECASTS f_i to get
    delta (Equation 3); the combined forecast is f_bar + F @ delta (Equation 2).
    Fitting on f_i rather than d_i is what makes this DISTINCT from FARM1 -- the
    f_bar direction lives only in span{f_i}, since sum_i d_i = 0.
    """
    from sklearn.linear_model import LassoLarsIC
    k = forecasts.shape[1]
    try:
        return LassoLarsIC(criterion="bic", fit_intercept=False).fit(
            forecasts, target).coef_
    except Exception:
        return np.zeros(k)


# --------------------------------------------------------------------------- #
# walk-forward driver
# --------------------------------------------------------------------------- #
def walk_forward_combination(ranks, target, periods, method, beta_method="ridge",
                             lag=LAG, update_every=UPDATE_EVERY,
                             min_history=MIN_HISTORY, roll=ROLL_PERIODS):
    """One combined prediction per row, every coefficient re-estimated on the
    schedule and never reading beyond the resolution lag.

    :param ranks: (n_rows, K) member percentile ranks, one clean per-period
        permutation per member -- rank on the COMMON panel before calling, so
        f_bar and the d_i are exactly per-period centred and the no-intercept
        Equation (6) fit is unbiased.
    :type ranks: np.ndarray
    :param target: (n_rows,) the target, in the same order as `ranks`
    :type target: np.ndarray
    :param periods: (n_rows,) the period label of each row
    :type periods: np.ndarray
    :param method: one of METHODS
    :type method: str
    :param beta_method: for farm1/2/3, one of BETA_METHODS
    :type beta_method: str
    :return: (prediction of length n_rows, list of per-update records)
    :rtype: (np.ndarray, list[dict])
    """
    if method not in METHODS:
        raise ValueError("unknown method %r; expected one of %s" % (method, METHODS))

    order = np.array(sorted(set(periods)))
    position = {p: i for i, p in enumerate(order)}
    row_pos = np.array([position[p] for p in periods])

    fbar = ranks.mean(axis=1)
    deviations = ranks - fbar[:, None]

    # per-period rank of the target, used ONLY to fit on resolved history
    target_rank = np.empty(len(target))
    for p in order:
        mask = periods == p
        target_rank[mask] = _rank_pct(target[mask])
    ubar = target_rank - fbar

    prediction = fbar.copy()                 # default and cold start = simple average
    records = []
    if method == "equal":
        return prediction, records

    for start in range(0, len(order), update_every):
        usable = max(0, start - lag)
        low = max(0, usable - roll)
        history = (row_pos >= low) & (row_pos < usable)
        current = (row_pos >= start) & (row_pos < min(start + update_every, len(order)))
        if not current.any():
            continue
        if usable < min_history or history.sum() < 50:
            records.append({"from_period": str(order[start]), "scheme": "equal (cold start)",
                            "history_periods": int(usable - low)})
            continue

        hidx = np.where(history)[0]
        if len(hidx) > MAX_FIT_ROWS:
            hidx = hidx[np.linspace(0, len(hidx) - 1, MAX_FIT_ROWS).astype(int)]
        design_h, ubar_h = deviations[hidx], ubar[hidx]
        record = {"from_period": str(order[start]), "scheme": method,
                  "history_periods": int(usable - low)}

        if method == "farm1":
            gram = design_h.T @ design_h
            cross = design_h.T @ ubar_h
            beta = BETA_FITTERS[beta_method](design_h, ubar_h, gram, cross)
            prediction[current] = fbar[current] + deviations[current] @ beta
        elif method in ("farm2", "farm3"):
            kind = "pca" if method == "farm2" else "spca"
            r = pick_r(design_h, ubar_h, kind)
            record["r"] = int(r)
            prediction[current] = fbar[current] + factor_combination(
                design_h, ubar_h, deviations[current], kind, beta_method, r)
        elif method == "gr":
            weights = fit_gr(ranks[hidx], target_rank[hidx])
            prediction[current] = ranks[current] @ weights
        elif method == "ds":
            delta = fit_ds(ubar_h, ranks[hidx])
            prediction[current] = fbar[current] + ranks[current] @ delta
        records.append(record)

    return prediction, records


def _rank_pct(values):
    """Percentile rank of a 1-D array in (0, 1], average ties -- matches
    pandas Series.rank(pct=True), which is how members enter."""
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    # average ties
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return ranks / len(values)
