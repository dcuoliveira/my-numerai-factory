"""
Ways to weight the members of an ensemble, and what each is really assuming.

Every function here takes only information available at estimation time and
returns a weight vector summing to 1. None of them fit a model -- the members
are already fitted; this is the layer above.

------------------------------------------------------------------------------
The constraint that shapes all of it
------------------------------------------------------------------------------

A 20-day forward target on daily dates means 510 scored periods carry roughly
25 INDEPENDENT observations. Five members leave 4 free weights and 15 covariance
parameters. That is ~6 observations per weight, and it shows: re-estimating
tangency weights on three non-overlapping thirds of the crypto sample gives

    window   onehot   tenc  interact   size  price
    T1         0.09  -0.01      0.16   0.53   0.21
    T2         0.44  -0.07      0.07   0.22   0.35
    T3        -0.21   0.50      0.43  -0.02   0.31

Signs flip. This is the forecast combination puzzle (Timmermann 2006; Smith &
Wallis 2009): estimated optimal weights routinely lose to 1/N out of sample
because estimation error swamps the theoretical gain. On this data optimising
the weights buys +0.045 sharpe in sample while changing MEMBERSHIP swings it by
0.17 -- roughly 4x more.

So the schemes below are ordered by how much they read from the covariance,
which is the same as ordering them by how much trouble they can cause:

    equal              reads nothing
    inverse_variance   reads the diagonal only          (Bates-Granger 1969)
    complete_subset    reads means, averages over subsets (Elliott+ 2013)
    caruana / bagged   reads the objective, coarsely     (Caruana+ 2004)
    shrunk_tangency    reads the full covariance, shrunk (Diebold-Pauly 1990)
    granger_ramanathan reads the target directly         (GR 1984)
    stacking_nnls      same, ridge-penalised

------------------------------------------------------------------------------
On Granger-Ramanathan versus "stacking"
------------------------------------------------------------------------------

In this setting they are the same estimator wearing different clothes: both
regress the realised target on the members' out-of-sample predictions. The
econometric version constrains the weights to sum to one and drops the
intercept; the machine-learning version adds a ridge penalty and works on
per-period-centred ranks. Both are implemented because the difference in
regularisation is exactly what decides whether the estimator survives 25
observations, and that is worth measuring rather than assuming.
"""

import itertools

import numpy as np


# Every scheme in this module, in the order the docstring introduces them.
SCHEMES = ("equal", "inverse_variance", "complete_subset", "caruana",
           "bagged_caruana", "shrunk_tangency", "granger_ramanathan",
           "stacking_nnls")

# Schemes that need row-level predictions and the target rather than the
# members' per-period score series.
PREDICTION_BASED = ("granger_ramanathan", "stacking_nnls")


def _normalise(weights, floor=0.0):
    """Clip to `floor`, renormalise to sum 1, fall back to 1/N if degenerate."""
    weights = np.asarray(weights, dtype="float64")
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if floor is not None:
        weights = np.clip(weights, floor, None)
    total = weights.sum()
    if not np.isfinite(total) or total <= 1e-12:
        return np.ones(len(weights)) / len(weights)
    return weights / total


def equal(scores, **kwargs):
    """
    1/N. The benchmark every other scheme has to beat.

    Not a strawman: on the crypto blend it sits at the 97th percentile of 20,000
    random long-only weightings, and the coarse simplex search that selected
    CURUPIRA_05's members returned exactly equal weights as its optimum.
    """
    return np.ones(scores.shape[1]) / scores.shape[1]


def inverse_variance(scores, **kwargs):
    """
    Bates-Granger (1969): w proportional to 1/variance, covariance ignored.

    Throwing away the off-diagonal is the point. The covariance needs 15
    parameters here against ~25 observations; the diagonal needs 5. This is the
    cheapest possible step away from 1/N and the usual first thing to try when
    the sample is short.
    """
    variance = scores.var(axis=0, ddof=1)
    return _normalise(np.where(variance > 0, 1.0 / np.maximum(variance, 1e-12), 0.0))


def complete_subset(scores, subset_size=3, objective="sharpe", **kwargs):
    """
    Elliott, Gargano & Timmermann (2013): average over all k-member subsets.

    Every subset is equally weighted internally, and the final weight of a
    member is how often it appears -- so with all subsets of a given size and no
    selection this reduces to 1/N. It only departs from equal weighting when
    subsets are SCORED and the good ones kept, which is what `keep_fraction`
    does here.

    This addresses the exposure that actually matters on this data -- member
    selection -- while never estimating a continuous weight.
    """
    keep_fraction = kwargs.get("keep_fraction", 0.5)
    n = scores.shape[1]
    subset_size = max(1, min(int(subset_size), n))

    ranked = []
    for subset in itertools.combinations(range(n), subset_size):
        combined = scores[:, list(subset)].mean(axis=1)
        ranked.append((_objective(combined, objective), subset))
    ranked.sort(reverse=True)

    keep = max(1, int(round(len(ranked) * keep_fraction)))
    counts = np.zeros(n)
    for _, subset in ranked[:keep]:
        counts[list(subset)] += 1.0
    return _normalise(counts)


def _objective(series, name):
    """Scalar to maximise. `sharpe` is the default because payout is noisier."""
    series = np.asarray(series, dtype="float64")
    if len(series) < 2:
        return -np.inf
    mean = series.mean()
    if name == "mean":
        return mean
    deviation = series.std(ddof=0)
    return mean / deviation if deviation > 0 else -np.inf


def caruana(scores, iterations=50, objective="sharpe", initial=None, **kwargs):
    """
    Caruana et al. (2004) greedy ensemble selection, forward, WITH replacement.

    Start from nothing, then repeatedly add whichever member most improves the
    objective of the running average -- allowing the same member to be picked
    again, which is what lets the method express non-uniform weights without
    ever solving for one. Final weights are selection counts over iterations.

    Two properties make it robust where least squares is not: it only ever reads
    the objective (never inverts a covariance), and its weights are quantised to
    multiples of 1/iterations, so it cannot express the knife-edge solutions that
    make tangency weights flip sign between subsamples.
    """
    n = scores.shape[1]
    counts = np.zeros(n) if initial is None else np.array(initial, dtype="float64")
    running = scores @ counts if counts.sum() else np.zeros(len(scores))
    picked = counts.sum()

    for _ in range(int(iterations)):
        best, best_index = -np.inf, None
        for j in range(n):
            candidate = (running + scores[:, j]) / (picked + 1.0)
            value = _objective(candidate, objective)
            if value > best:
                best, best_index = value, j
        if best_index is None:
            break
        counts[best_index] += 1.0
        running = running + scores[:, best_index]
        picked += 1.0

    return _normalise(counts)


def bagged_caruana(scores, iterations=25, bags=20, bag_fraction=0.6,
                   objective="sharpe", seed=0, **kwargs):
    """
    Caruana's own fix for greedy selection's variance: bag the model library.

    Each bag hill-climbs over a random subset of the MEMBERS (not the rows), and
    the final weights average the bags. A member that only wins on some draws
    gets a proportionally smaller weight instead of all or nothing, which is the
    cheapest available protection against selecting on noise.
    """
    rng = np.random.default_rng(seed)
    n = scores.shape[1]
    size = max(2, int(round(n * bag_fraction)))

    total = np.zeros(n)
    for _ in range(int(bags)):
        members = rng.choice(n, size=size, replace=False)
        weights = caruana(scores[:, members], iterations=iterations,
                          objective=objective)
        total[members] += weights
    return _normalise(total)


def shrunk_tangency(scores, shrinkage=0.2, **kwargs):
    """
    Diebold & Pauly (1990): shrink the mean-variance optimal weights toward 1/N.

        w = shrinkage * w_tangency + (1 - shrinkage) * (1 / N)

    `shrinkage` is how much you trust the covariance. At 0 this is equal
    weighting; at 1 it is the raw tangency portfolio that flipped signs between
    thirds of this sample. The default of 0.2 reflects ~6 observations per
    parameter, and the covariance is ridged before inversion because a 5x5
    estimated from 25 effective points is close to singular.
    """
    mean = scores.mean(axis=0)
    covariance = np.cov(scores, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    ridge = 1e-8 + 1e-3 * np.trace(covariance) / len(covariance)
    try:
        raw = np.linalg.solve(covariance + ridge * np.eye(len(covariance)), mean)
    except np.linalg.LinAlgError:
        return equal(scores)
    if not np.isfinite(raw).all() or abs(raw.sum()) < 1e-12:
        return equal(scores)
    raw = raw / raw.sum()

    n = scores.shape[1]
    blended = shrinkage * raw + (1.0 - shrinkage) * (np.ones(n) / n)
    # Long-only: a Numerai slot cannot hold a negative position in a member, and
    # the negative weights are exactly the unstable ones.
    return _normalise(blended, floor=0.0)


def granger_ramanathan(predictions, target, constrained=True, **kwargs):
    """
    Granger & Ramanathan (1984): regress the realised target on the forecasts.

    Three variants exist in the literature -- with intercept, without, and with
    weights constrained to sum to one. The constrained no-intercept version is
    implemented because it is the only one whose output is a weighting rather
    than an affine rescaling, and because the unconstrained version is what
    produced the sign-flipping table in this module's docstring.

    Solved as non-negative least squares on the simplex. Non-negativity does
    most of the regularising here: it alone removes every negative weight the
    unconstrained fit wanted.
    """
    from scipy.optimize import nnls

    A = np.asarray(predictions, dtype="float64")
    b = np.asarray(target, dtype="float64")
    if len(A) < A.shape[1] + 2:
        return np.ones(A.shape[1]) / A.shape[1]

    if not constrained:
        weights, *_ = np.linalg.lstsq(A, b, rcond=None)
        return _normalise(weights)

    # Sum-to-one enforced by an augmented row; the large multiplier makes it a
    # hard constraint in practice without needing a QP solver.
    penalty = 1e3 * np.abs(A).mean()
    A_aug = np.vstack([A, penalty * np.ones((1, A.shape[1]))])
    b_aug = np.concatenate([b, [penalty]])
    try:
        weights, _ = nnls(A_aug, b_aug)
    except Exception:
        return np.ones(A.shape[1]) / A.shape[1]
    return _normalise(weights)


def stacking_nnls(predictions, target, alpha=1.0, **kwargs):
    """
    Level-2 stacking: ridge-penalised non-negative least squares on centred ranks.

    The machine-learning cousin of granger_ramanathan. Two differences, both
    deliberate:

      * the ridge penalty is a second regulariser on top of non-negativity,
        appended as sqrt(alpha) * I rows rather than solved in closed form so
        the non-negativity constraint survives;
      * inputs are centred per column, because the members' rank outputs all sit
        near 0.5 and the constant part carries no information about the target.

    The member predictions are genuinely out of sample -- they come from the
    walk-forward -- so this is stacking on OOF predictions in the usual sense,
    without needing a second CV layer.
    """
    from scipy.optimize import nnls

    A = np.asarray(predictions, dtype="float64")
    b = np.asarray(target, dtype="float64")
    if len(A) < A.shape[1] + 2:
        return np.ones(A.shape[1]) / A.shape[1]

    A = A - A.mean(axis=0)
    b = b - b.mean()

    n_members = A.shape[1]
    scale = np.sqrt(max(alpha, 0.0)) * (np.abs(A).mean() + 1e-12)
    A_aug = np.vstack([A, scale * np.eye(n_members)])
    b_aug = np.concatenate([b, np.zeros(n_members)])
    try:
        weights, _ = nnls(A_aug, b_aug)
    except Exception:
        return np.ones(n_members) / n_members
    return _normalise(weights)


DISPATCH = {
    "equal": equal,
    "inverse_variance": inverse_variance,
    "complete_subset": complete_subset,
    "caruana": caruana,
    "bagged_caruana": bagged_caruana,
    "shrunk_tangency": shrunk_tangency,
    "granger_ramanathan": granger_ramanathan,
    "stacking_nnls": stacking_nnls,
}


def estimate(scheme, scores=None, predictions=None, target=None, **kwargs):
    """
    Dispatch to a scheme, passing whichever inputs it needs.

    :param scheme: one of SCHEMES
    :param scores: (periods x members) per-period payout series, for the
                   score-based schemes
    :param predictions: (rows x members) member predictions, for the
                        prediction-based schemes
    :param target: (rows,) realised target aligned with `predictions`
    :return: weights summing to 1
    """
    if scheme not in DISPATCH:
        raise ValueError("unknown weighting scheme '%s', expected one of %s"
                         % (scheme, list(SCHEMES)))
    if scheme in PREDICTION_BASED:
        if predictions is None or target is None:
            raise ValueError("scheme '%s' needs predictions and target" % scheme)
        return DISPATCH[scheme](predictions, target, **kwargs)
    if scores is None:
        raise ValueError("scheme '%s' needs a score matrix" % scheme)
    return DISPATCH[scheme](scores, **kwargs)
