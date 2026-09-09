"""
The paper side of the FARM validation: reproduce Lee & Lee (2025) Table 1 on
FRED-MD, driving the SAME estimators E27 uses (src/prediction/farm.py) in the
paper's LEVEL space instead of the repo's rank space.

Why this exists. E27 found FARM does not beat the simple average on the crypto
book (numerai_corr). A null is only worth trusting if the implementation is
correct, and crypto cannot supply a positive control -- it has no signal FARM can
exploit. FRED-MD does: this module runs the identical fit_ridge / fit_l2boost /
fit_postalasso / pca / spca / factor_combination / fit_gr / fit_ds functions on
the paper's macro data and checks they reproduce Table 1's wins. If they do, the
crypto null is a property of the data and the metric, not a bug.

Only the DRIVER differs from farm.py. There, u_bar = rank(y) - f_bar and the
walk-forward carries a 20-day resolution lag; here u_bar = y - f_bar in levels and
the split is the paper's T1/T2/T3. The nine estimator functions are imported and
called unchanged.

Pipeline (paper Section 4):
  1. FRED-MD, monthly. Apply each series' McCracken-Ng transform code to make it
     stationary. The five targets' codes match the paper exactly: RPI/INDPRO = 5
     (Dlog), CPIAUCSL/PCEPI = 6 (D2log), UNRATE = 2 (Dlevel).
  2. Individual forecasts, one predictor at a time: y_{t+h} = a_i + b_i x_{i,t},
     a_i,b_i by rolling OLS on the T1 most recent RESOLVED pairs. N ~= 119
     predictors (columns with full history over the window).
  3. Combine the N forecasts by each method, weights re-estimated on a rolling T2
     window over the T3 out-of-sample period.
  4. Score: MSFE relative to the simple average, with a Diebold-Mariano test
     (Newey-West on the squared-loss differential).
"""

import numpy as np
import pandas as pd

from src.prediction.farm import (BETA_FITTERS, fit_gr, fit_ds, pca_loadings,
                                 spca_loadings, pick_r)

# paper's split, Section 4
T1 = 120     # rolling window for the individual forecasts
T2 = 480     # estimation window for the combining weights
T3 = 120     # out-of-sample evaluation period
HORIZONS = (1, 2, 3)
TARGETS = ("RPI", "CPIAUCSL", "PCEPI", "INDPRO", "UNRATE")

METHODS = []                              # (label, method, beta_method)
METHODS.append(("equal", "equal", None))
for m in ("farm1", "farm2", "farm3"):
    for bm in ("postalasso", "l2boost", "ridge"):
        METHODS.append(("%s.%s" % (m, bm), m, bm))
METHODS.append(("gr", "gr", None))
METHODS.append(("ds", "ds", None))


# --------------------------------------------------------------------------- #
# FRED-MD load + McCracken-Ng stationarity transforms
# --------------------------------------------------------------------------- #
def _apply_tcode(x, code):
    """McCracken-Ng (2016) transform codes. Returns a series the same length as x
    with leading NaNs where differencing consumes observations."""
    x = np.asarray(x, dtype=float)
    if code == 1:                                   # level
        return x
    if code == 2:                                   # first difference
        return np.append(np.nan, np.diff(x))
    if code == 3:                                   # second difference
        return np.append([np.nan, np.nan], np.diff(x, 2))
    if code == 4:                                   # log
        return np.log(x)
    if code == 5:                                   # first difference of log
        return np.append(np.nan, np.diff(np.log(x)))
    if code == 6:                                   # second difference of log
        return np.append([np.nan, np.nan], np.diff(np.log(x), 2))
    if code == 7:                                   # first difference of pct change
        pct = np.append(np.nan, x[1:] / x[:-1] - 1.0)
        return np.append(np.nan, np.diff(pct))
    raise ValueError("unknown transform code %r" % code)


def load_fredmd(path, start="1960-01-01"):
    """Load the vendored FRED-MD CSV and return the stationary panel.

    :return: DataFrame indexed by month, columns = series, all transformed and
        restricted to `start` onward; columns with any missing value in the
        window are dropped (this is how ~128 series becomes the paper's ~119).
    :rtype: pd.DataFrame
    """
    raw = pd.read_csv(path)
    codes = raw.iloc[0, 1:].astype(float).astype(int)
    body = raw.iloc[1:].copy()
    body = body[body.iloc[:, 0].notna()]
    dates = pd.to_datetime(body.iloc[:, 0])
    values = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    transformed = {name: _apply_tcode(values[name].to_numpy(), int(codes[name]))
                   for name in values.columns}
    panel = pd.DataFrame(transformed, index=dates.to_numpy())
    panel = panel.loc[panel.index >= pd.Timestamp(start)]
    return panel.dropna(axis=1, how="any")


# --------------------------------------------------------------------------- #
# individual forecasts: one predictor at a time, rolling OLS (leak-free)
# --------------------------------------------------------------------------- #
def individual_forecasts(panel, target, h, t1=T1):
    """Build the N one-predictor forecasts of y_{t+h} for every origin where a
    full T1-pair window has resolved.

    At origin t the model y_{s+h} = a_i + b_i x_{i,s} is fit on the T1 pairs
    ending at s = t - h (the last pair whose target is known by time t), then the
    forecast is a_i + b_i x_{i,t}. Enforcing s <= t - h is what keeps the
    individual forecasts real-time and leak-free.

    :return: (forecasts F [n_origins x N], realized y [n_origins], N predictor
        names, origin index)
    :rtype: (np.ndarray, np.ndarray, list, pd.Index)
    """
    y = panel[target].to_numpy()
    X = panel.to_numpy()
    n, k = X.shape
    names = list(panel.columns)

    rows, realized, origins = [], [], []
    # origin t predicts y[t+h]; needs T1 resolved pairs (x[s], y[s+h]) with
    # s+h <= t, i.e. s <= t-h, and the window is s in [t-h-T1+1, t-h].
    for t in range(t1 + h - 1, n - h):
        s_hi = t - h
        s_lo = s_hi - t1 + 1
        if s_lo < 0:
            continue
        xs = X[s_lo:s_hi + 1]                       # (T1, k) predictors at s
        ys = y[s_lo + h:s_hi + 1 + h]               # (T1,) targets at s+h
        # vectorised simple OLS slope/intercept per predictor
        xm = xs.mean(axis=0)
        ym = ys.mean()
        xc = xs - xm
        var = (xc * xc).sum(axis=0)
        var[var == 0] = np.nan
        b = (xc * (ys - ym)[:, None]).sum(axis=0) / var
        a = ym - b * xm
        f = a + b * X[t]                            # forecast of y[t+h]
        rows.append(f)
        realized.append(y[t + h])
        origins.append(panel.index[t])
    F = np.array(rows)
    good = ~np.isnan(F).any(axis=0)                 # drop degenerate predictors
    return F[:, good], np.array(realized), [names[i] for i in np.where(good)[0]], pd.Index(origins)


# --------------------------------------------------------------------------- #
# level-space combination, rolling weights over the OOS tail
# --------------------------------------------------------------------------- #
def combine(F, y, label, method, beta_method, t2=T2, t3=T3, update_every=1):
    """One combined out-of-sample forecast per OOS origin, weights re-estimated
    on the rolling T2 window of resolved (forecast, realized) pairs.

    Reuses the farm.py estimators unchanged: design = d_i = f_i - f_bar,
    target = u_bar = y - f_bar, both in LEVELS (no ranking).

    `update_every` re-estimates the weights every K origins and holds them
    between updates (K=1 is the paper's fully-rolling scheme). The PCA/SPCA
    factor methods carry a 3-fold CV per fit, so the runner throttles them to
    keep the full 5x3x11 sweep tractable; the deviation regression itself is
    unchanged, only how often it is refreshed.

    :return: (combined OOS forecasts [T3], realized y over the same OOS [T3])
    :rtype: (np.ndarray, np.ndarray)
    """
    n = len(y)
    oos_start = n - t3
    fbar = F.mean(axis=1)
    deviations = F - fbar[:, None]
    ubar = y - fbar

    combined = np.empty(t3)
    predictor = None
    for j, i in enumerate(range(oos_start, n)):
        if predictor is None or j % update_every == 0:
            lo = max(0, i - t2)
            win = slice(lo, i)                       # resolved estimation window
            d_win, u_win, F_win, y_win = (deviations[win], ubar[win], F[win], y[win])
            predictor = _fit_predictor(method, beta_method, d_win, u_win, F_win, y_win)
        combined[j] = predictor(i, fbar, deviations, F)
    return combined, y[oos_start:]


def _fit_predictor(method, beta_method, d_win, u_win, F_win, y_win):
    """Fit one method on the estimation window and return a closure that maps an
    OOS origin i to the combined forecast. Isolates the fit from the apply so a
    held weight vector can be reused across `update_every` origins."""
    if method == "equal":
        return lambda i, fbar, dev, F: fbar[i]
    if method == "farm1":
        beta = BETA_FITTERS[beta_method](d_win, u_win, d_win.T @ d_win, d_win.T @ u_win)
        return lambda i, fbar, dev, F: fbar[i] + dev[i] @ beta
    if method in ("farm2", "farm3"):
        # Fit the factor loadings, alpha and beta ONCE on the window, then apply
        # cheaply per origin (mirrors farm.factor_combination, split fit/apply so
        # update_every actually throttles the expensive extraction).
        kind = "pca" if method == "farm2" else "spca"
        r = pick_r(d_win, u_win, kind)
        if r == 0:
            beta = BETA_FITTERS[beta_method](d_win, u_win, d_win.T @ d_win, d_win.T @ u_win)
            return lambda i, fbar, dev, F: fbar[i] + dev[i] @ beta
        loadings = (pca_loadings(d_win, r) if kind == "pca"
                    else spca_loadings(d_win, u_win, r))
        factors = d_win @ loadings
        alpha, *_ = np.linalg.lstsq(factors, u_win, rcond=None)
        resid = u_win - factors @ alpha
        proj, *_ = np.linalg.lstsq(factors, d_win, rcond=None)
        d_res = d_win - factors @ proj
        beta = BETA_FITTERS[beta_method](d_res, resid, d_res.T @ d_res, d_res.T @ resid)

        def apply_factor(i, fbar, dev, F):
            row = dev[i:i + 1]
            row_res = row - (row @ loadings) @ proj
            return fbar[i] + (row_res @ beta)[0] + ((row @ loadings) @ alpha)[0]
        return apply_factor
    if method == "gr":
        w = fit_gr(F_win, y_win)
        return lambda i, fbar, dev, F: F[i] @ w
    if method == "ds":
        delta = fit_ds(u_win, F_win)
        return lambda i, fbar, dev, F: fbar[i] + F[i] @ delta
    raise ValueError("unknown method %r" % method)


# --------------------------------------------------------------------------- #
# evaluation: relative MSFE + Diebold-Mariano
# --------------------------------------------------------------------------- #
def diebold_mariano(loss_base, loss_method, lag):
    """DM test on the squared-loss differential d_t = loss_base - loss_method
    (positive => method better). Uses the repo's canonical Newey-West HAC. Returns
    (relative MSFE, DM t, one-sided p that method improves)."""
    from src.analysis.ablation_report import newey_west
    from scipy.stats import norm
    rel = loss_method.mean() / loss_base.mean()
    mean, se, t = newey_west(pd.Series(loss_base - loss_method), lag=lag)
    p_improve = float(norm.cdf(-t)) if np.isfinite(t) else np.nan   # H1: method better
    return rel, (t if np.isfinite(t) else np.nan), p_improve
