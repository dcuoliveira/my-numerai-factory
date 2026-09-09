"""
Generalised Ridge Regression, as an sklearn estimator on this pipeline's contract.

Vanilla ridge minimises

    ||Y - X b||^2 + lambda ||b||^2

which fixes three things by fiat: every row counts the same, every coefficient is
shrunk toward ZERO, and every coefficient is shrunk by the SAME amount. The
generalised form frees all three:

    b_hat = argmin  (Y - X b)' W (Y - X b)  +  (b - b0)' Delta (b - b0)

    W      diagonal weighting matrix   -- which ROWS matter          (ridge: I)
    b0     the coefficient prior       -- what b is shrunk TOWARD    (ridge: 0)
    Delta  a PSD penalty matrix        -- the SHAPE of the shrinkage (ridge: lambda I)

Setting W = I, b0 = 0 and Delta = lambda*I recovers vanilla ridge exactly, and
`test_generalised_ridge.py` asserts that against sklearn's own `Ridge` rather
than trusting the reduction.

------------------------------------------------------------------------------
Why a generalised Delta is the interesting one here
------------------------------------------------------------------------------

This project has a documented, repeated failure mode: adding a block of columns
makes the model WORSE, and the stated mechanism is that one scalar penalty is
shared across the whole design matrix, so every new block raises the shrinkage
that every existing column receives (experiments/cumlagged_features_crypto.sh).
A non-identity Delta is the direct answer to that -- it lets the penalty have a
shape, so related columns can be tied to each other instead of each being pulled
independently toward zero.

The shape offered here is GROUP FUSION. The 22 raw features are 8 quantities
measured at two horizons and, for three of them, under two smoothings:

    feature_close_avg_20d  feature_close_avg_60d
    feature_close_ewa_20d  feature_close_ewa_60d      -> one group, "feature_close"

Fusing a group shrinks its members toward their COMMON value rather than toward
zero, which is the natural prior for four measurements of one underlying thing.
At rho = 0 it is exactly ridge; at large rho the group collapses to a single
effective coefficient. It is the ridge analogue of the fused-lasso idea, and it
costs one hyperparameter.

------------------------------------------------------------------------------
alpha is a RATIO, not an absolute penalty
------------------------------------------------------------------------------

`alpha_scale="gram"` (the default) multiplies alpha by the mean diagonal of the
weighted centred Gram over live columns. That makes alpha dimensionless: the
same value means the same amount of shrinkage at 22 columns and at 812, on 39k
rows and on 500k.

This is what `ridge_glmnet` uses a data-derived lambda path for, achieved with a
static grid instead -- which matters for three reasons this pipeline cares about.
`--fixed-params` BYPASSES fit_grid by design (functions.py), so a pinned slot on
a path-based model silently gets a penalty from a different regime; a pinned
absolute alpha is only valid at the row count it was selected on; and a static
grid can be checked before a study launches rather than after.

`alpha_scale="absolute"` turns that off and reproduces sklearn's `alpha` exactly.
It exists so the equivalence test can be written at all.

------------------------------------------------------------------------------
Per-group specifications are STRINGS
------------------------------------------------------------------------------

Three of the knobs are per-group: the penalty multiplier, the coefficient prior,
and the sign constraint. Each is expressed as one string:

    "feature_volatility:0.1,feature_close:2.0"      penalty multipliers
    "feature_momentum:0"                            priors
    "feature_volatility:-,feature_close:+"          sign constraints

A string rather than a dict, and the reason is mechanical rather than
stylistic. A dict cannot go in a param_grid -- to_dimension() cannot hash it --
so a dict-valued knob could never be searched, never appear in best_params_,
never be pinned through --fixed-params for a live slot, and would land in the
hyperparams CSV as an unparseable cell. A string does all four, and a LIST of
such strings is a legal Categorical dimension under both RandomizedSearchCV and
BayesSearchCV.

A group named in a spec that does not exist in the data RAISES, listing the
groups that were found. Silently ignoring it would run the arm, score it, record
the spec, and report a comparison that never happened.

------------------------------------------------------------------------------
What this deliberately does NOT do
------------------------------------------------------------------------------

* No classification head. There is no closed form for a penalised logistic fit,
  and the closed form is the entire claim. Declared TASKS = ("regression",) on
  the wrapper so `supported_tasks` can answer without constructing one.
* No smoothness/Almon Delta along a lag axis. The repo carries three
  incompatible lag spellings and the difference ladders are log-spaced, so
  "penalise consecutive terms" is the wrong metric and one rho cannot serve
  both a 2-member and a 40-member ladder.
* No time-decay W derived from ROW POSITION. The period is ROUTED to fit() as a
  separate argument instead, because data_mani/utils.py's loader ends in a bare
  reset_index(drop=True) with no sort -- row order is chronological on today's
  parquets and nothing enforces it. Adding a period FEATURE was also refused: it
  would be picked up by filter_features for every OTHER model on that build,
  silently handing ridge and lgb a trend regressor.
* No matrix-valued hyperparameter. `search_spaces.to_dimension` cannot hash an
  ndarray, and it would write hundreds of thousands of floats per refit into one
  hyperparams CSV cell.
"""

import re

import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

#: Trailing tokens that name a position on an ORDERED axis rather than a
#: different quantity: horizons (20d/60d), lags and differences. Stripped
#: repeatedly, because a column can carry more than one.
_AXIS = re.compile(r"_(?:(?:20|60)d|(?:d|ar)?lag\d+|diff\d+|e16d\d+|rm\d+)$")

#: A trailing SMOOTHING token. Stripped once, after the axis tokens, so that
#: close_avg_20d and close_ewa_60d land in one group.
_AGG = re.compile(r"_(?:avg|ewa)$")

#: The tournament's neutral bin. Crypto features and targets are rank-binned to
#: {0, .25, .5, .75, 1}, so 0.5 is literally the "no information" value. The
#: tail weighting anchors here rather than on the sample mean: a sample mean
#: lands within 1e-4 of 0.5 and the resulting weights then delete half the
#: training rows over a rounding difference.
NEUTRAL = 0.5


def stem_of(name):
    """
    The quantity a feature column measures, with its axis position removed.

    feature_close_avg_20d -> feature_close ; feature_volatility_60d_diff5 ->
    feature_volatility. Columns sharing a stem are the same thing measured
    differently, which is what makes fusing them a sensible prior.

    :param name: a feature column name
    :type name: str
    :return: the stem
    :rtype: str
    """
    previous = None
    while previous != name:
        previous, name = name, _AXIS.sub("", name)
    return _AGG.sub("", name)


def feature_groups(names):
    """
    Partition column names into groups of two or more sharing a stem.

    Singletons are omitted: a group of one has no within-group variation to
    penalise, so including it would add a zero block and a misleading entry.

    Keyed on NAMES rather than positions on purpose. Backtest column order is
    parquet order and live order is build_features' explicit build order; they
    agree only by construction, and a positional rule would go wrong silently.

    :param names: column names, in order
    :type names: [str]
    :return: stem -> the indices of its members
    :rtype: {str: [int]}
    """
    return {stem: idx for stem, idx in stem_index(names).items() if len(idx) > 1}


#: Named feature BLOCKS, by construction rather than by stem. Order matters --
#: first match wins -- and `raw` is the fallback, so the list is a partition of
#: any dataset in this repo.
#:
#: A block is not a stem. `stem_of` groups columns that measure the SAME
#: QUANTITY differently (feature_close at four horizon/smoothing combinations),
#: which is the right unit for fusing. A block groups columns that came from the
#: SAME CONSTRUCTION STEP, which is the right unit for deciding how hard to
#: shrink: every ticker one-hot is its own stem, so the 740-column dummy block
#: is unnameable in stem vocabulary and unmissable in block vocabulary.
#:
#: The regexes are the ones CURUPIRA_05's members already use in
#: src/live/registry.py, so a spec written here means the same thing the
#: deployed blend means.
FEATURE_BLOCKS = (
    ("tenc", r"^feature_sym_tenc$"),
    ("cap", r"^feature_cap_"),
    ("onehot", r"^feature_sym_(?!tenc$)"),
    ("interact", r"_x_|_accel_"),
    ("spread", r"_spread$"),
    ("rollsel", r"_rm\d+$|_roll"),
    ("lag", r"_(?:d|ar)?lag\d+$"),
    ("diff", r"_diff\d+$"),
    ("code", r"^feature_(?:ae|emb)_"),
    ("symmean", r"_symmean$"),
)

_BLOCK_PATTERNS = tuple((name, re.compile(pattern))
                        for name, pattern in FEATURE_BLOCKS)

#: Every block name, including the fallback. Used to tell a block from a stem.
BLOCK_NAMES = tuple(name for name, _ in FEATURE_BLOCKS) + ("raw",)


def block_of(name):
    """
    Which construction block a column came from; "raw" if none claims it.

    :param name: a feature column name
    :type name: str
    :return: the block name
    :rtype: str
    """
    for block, pattern in _BLOCK_PATTERNS:
        if pattern.search(name):
            return block
    return "raw"


def block_index(names):
    """
    Block name -> column indices. A partition: every column lands in exactly one.

    :param names: column names, in order
    :type names: [str]
    :return: block -> the indices of its members
    :rtype: {str: [int]}
    """
    blocks = {}
    for index, name in enumerate(names):
        blocks.setdefault(block_of(name), []).append(index)
    return blocks


def stem_index(names):
    """
    Every stem and its column indices, INCLUDING singletons.

    feature_groups() is the two-or-more filter over this. The distinction
    matters: fusing a singleton is meaningless, so fusion uses feature_groups,
    but shrinking or sign-constraining one is perfectly sensible, so a group_spec
    addresses stem_index.

    :param names: column names, in order
    :type names: [str]
    :return: stem -> the indices of its members
    :rtype: {str: [int]}
    """
    stems = {}
    for index, name in enumerate(names):
        stems.setdefault(stem_of(name), []).append(index)
    return stems


def parse_group_spec(text):
    """
    Parse a per-group specification into {stem: {field: value}}.

        "feature_volatility:p0.1"                        shrink it 10x less
        "feature_market_cap:k0;feature_volume:k0"        zero prior on two groups
        "feature_momentum:s+,p0.25;feature_volatility:s-"  sign and penalty

    ";" separates CLAUSES (one per group), "," separates FIELDS within a clause:

        field := "p" <float>    penalty multiplier for that group, > 0
               | "k" <float>    prior multiplier for that group, >= 0
               | "s" ("+"|"-")  sign constraint on every member of that group

    None, "" and "none" all mean no spec, which is exactly the unextended
    estimator. One string rather than three parameters or a dict, and the reason
    is mechanical: a string survives clone, joblib, both JSON writers and one
    hyperparams CSV cell, and a LIST of strings is a legal Categorical dimension
    under both RandomizedSearchCV and BayesSearchCV. A dict is none of those --
    to_dimension cannot hash it, and it would fail at the first refit rather than
    at argument parsing.

    :param text: the spec
    :type text: str or None
    :return: stem -> {"p": float, "k": float, "s": +1 or -1}
    :rtype: {str: dict}
    """
    if text is None:
        return {}
    if not isinstance(text, str):
        raise ValueError(
            "group_spec must be a STRING like 'feature_volatility:p0.1', got %r "
            "of type %s -- a dict cannot be searched, pinned or recorded; see "
            "the module docstring" % (text, type(text).__name__))
    stripped = text.strip()
    if not stripped or stripped == "none":
        return {}

    parsed = {}
    for clause in stripped.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        if ":" not in clause:
            raise ValueError(
                "bad group_spec clause %r: expected '<stem>:<field>[,<field>]', "
                "and ';' between groups rather than ','" % clause)
        stem, _, body = clause.partition(":")
        stem = stem.strip()
        if stem in parsed:
            raise ValueError("group_spec names %r twice" % stem)
        fields = {}
        for field in body.split(","):
            field = field.strip()
            if not field:
                continue
            letter, value = field[:1], field[1:].strip()
            if letter not in ("p", "k", "s"):
                raise ValueError(
                    "unknown field %r for %r: expected p (penalty multiplier), "
                    "k (prior multiplier) or s (sign)" % (field, stem))
            if letter in fields:
                raise ValueError("field %r given twice for %r" % (letter, stem))
            if letter == "s":
                if value not in ("+", "-"):
                    raise ValueError(
                        "sign field for %r must be s+ or s-, got %r" % (stem, field))
                fields["s"] = 1 if value == "+" else -1
            else:
                try:
                    number = float(value)
                except ValueError:
                    raise ValueError(
                        "field %r for %r needs a number, got %r"
                        % (letter, stem, value))
                if letter == "p" and number <= 0.0:
                    # A zero multiplier means "no penalty at all on this group",
                    # which makes the system singular whenever that group is
                    # collinear -- and it is, that is why the groups exist.
                    raise ValueError(
                        "penalty multiplier for %r must be > 0, got %r; spell "
                        "'barely shrink this group' as p1e-6, not p0" % (stem, value))
                if letter == "k" and number < 0.0:
                    raise ValueError(
                        "prior multiplier for %r must be >= 0, got %r" % (stem, value))
                fields[letter] = number
        if not fields:
            raise ValueError("group_spec clause %r has no fields" % clause)
        parsed[stem] = fields
    return parsed


def validate_group_spec(parsed, stems):
    """
    Refuse a spec naming a stem the data does not have.

    Silently ignoring it would run the arm, score it, record the spec in the
    hyperparams CSV, and report a comparison that never happened.

    The message names the vocabulary trap on purpose: this repo already carries a
    SECOND, incompatible group vocabulary in experiments/feature_groups_crypto.sh
    -- seven semantic groups (price, size, liquidity, oscillator, bollinger,
    momentum, volatility), where `oscillator` is rsi plus sharpe_ratio, which
    stem_of splits in two. Runs under those names already exist in results/.
    This API commits to STEMS.

    :param parsed: output of parse_group_spec
    :param stems: output of stem_index
    """
    unknown = sorted(stem for stem in parsed if stem not in stems)
    if unknown:
        raise ValueError(
            "group_spec names %s, which %s neither a construction BLOCK (%s) "
            "nor a feature stem in this data.\n"
            % (unknown, "is" if len(unknown) == 1 else "are",
               ", ".join(BLOCK_NAMES)) +
            "  %s\n"
            "  note the seven semantic groups in "
            "experiments/feature_groups_crypto.sh are a THIRD vocabulary -- "
            "'oscillator' there is feature_relative_strength_index plus "
            "feature_sharpe_ratio here."
            % ("names present (%d): %s" % (len(stems), sorted(stems))))


def _kkt_violation(system, rhs, coefficients, lower, upper):
    """
    How far a point is from satisfying the box-QP optimality conditions.

    The gradient must vanish on free coordinates, be >= 0 at a lower bound and
    <= 0 at an upper one. Returned unscaled; the caller compares it to the
    problem's own scale.
    """
    gradient = system @ coefficients - rhs
    at_lower = coefficients <= lower + 1e-12
    at_upper = coefficients >= upper - 1e-12
    free = ~(at_lower | at_upper)
    return max(
        float(np.abs(gradient[free]).max()) if free.any() else 0.0,
        float(np.maximum(-gradient[at_lower], 0.0).max()) if at_lower.any() else 0.0,
        float(np.maximum(gradient[at_upper], 0.0).max()) if at_upper.any() else 0.0)


def solve_box_constrained(system, rhs, lower, upper, tol=1e-8, max_sweeps=2000,
                          check_every=25):
    """
    Minimise (1/2) b' M b - r' b subject to lower <= b <= upper.

    Cyclic coordinate descent, warm-started from the UNCONSTRAINED solution. M is
    positive definite with a ridge-inflated diagonal, so each coordinate's
    subproblem is a one-dimensional quadratic solved in closed form and clipped;
    M @ b is carried incrementally so a sweep costs one matrix-vector product.

    Chosen over scipy.optimize.lsq_linear on measurement, not preference. Against
    its Cholesky-plus-bounded-least-squares route on real-shaped Grams:

        p =  22     2.2 ms  against     5.6 ms
        p = 229    34   ms  against  1550   ms
        p = 812   397   ms  against 19774   ms

    and at p = 812 this reaches a LOWER objective than lsq_linear does.

    CONVERGENCE IS JUDGED ON THE KKT CONDITIONS, not on the step size. A step-size
    rule has to be scaled by something, and scaling it by max(1, |b|.max()) makes
    it effectively ABSOLUTE whenever the coefficients are small -- which they are
    here, so an all-negative constraint on 812 columns stalled at a largest step
    of 5.5e-11 and was reported as a failure to converge while sitting on the
    optimum. The step size is kept only as a cheap short-circuit; optimality is
    what decides, and it is measured against the problem's own scale.

    :param system: M, the positive definite Hessian
    :param rhs: r
    :param lower: per-coordinate lower bounds, -inf where free
    :param upper: per-coordinate upper bounds, +inf where free
    :return: (solution, sweeps used)
    :rtype: (np.ndarray, int)
    """
    coefficients = np.clip(
        scipy.linalg.solve(system, rhs, assume_a="pos"), lower, upper)
    product = system @ coefficients
    diagonal = np.diag(system)
    width = len(rhs)
    limit = 1e-6 * max(1.0, float(np.abs(rhs).max()))

    used = 0
    for sweep in range(max_sweeps):
        largest = 0.0
        for j in range(width):
            step = np.clip(
                coefficients[j] - (product[j] - rhs[j]) / diagonal[j],
                lower[j], upper[j]) - coefficients[j]
            if step != 0.0:
                coefficients[j] += step
                product += system[:, j] * step
                largest = max(largest, abs(step))
        used = sweep + 1
        if largest == 0.0:
            break
        if largest <= tol * max(1e-12, float(np.abs(coefficients).max())):
            break
        if used % check_every == 0 and _kkt_violation(
                system, rhs, coefficients, lower, upper) <= limit:
            break

    violation = _kkt_violation(system, rhs, coefficients, lower, upper)
    if violation > limit:
        # Raise rather than return an unconverged iterate: it would score like a
        # real fit and be indistinguishable from one.
        raise ValueError(
            "the sign-constrained solve did not reach optimality in %d sweeps; "
            "the KKT violation is %.3e against a tolerance of %.3e"
            % (max_sweeps, violation, limit))
    return coefficients, used


class GeneralisedRidge(RegressorMixin, BaseEstimator):
    """
    Ridge with a weighting matrix, a coefficient prior and a shaped penalty.

    b_hat = (X'WX + Delta)^-1 (X'WY + Delta b0)

    :param alpha: penalty strength. Under alpha_scale="gram" this is a
                  dimensionless RATIO to the Gram's mean live diagonal
    :param alpha_scale: "gram" (default) or "absolute"
    :param structure: Delta family, "identity" or "group"
    :param group_penalty: rho, how hard a group is fused toward its own mean
    :param w_scheme: W family, "uniform" or "tail"
    :param tail_weight: tau in [0, 1]; 0 gives W = I exactly
    :param half_life: for w_scheme="exp_decay", the decay half-life in DATES
                      (not rows, not calendar days)
    :param b0_scheme: prior family, "zero", "marginal" or "group_mean"
    :param prior_shrinkage: kappa; 0 gives b0 = 0 exactly under every scheme
    :param group_spec: per-group penalty multipliers, prior multipliers and sign
                       constraints, as one string -- see parse_group_spec
    :param fit_intercept: fit an UNPENALISED intercept by weighted centring
    :param scale_floor: relative threshold below which a column counts as dead
    :param random_state: stored, never read -- the fit is deterministic. Present
                         because build_wrapper passes a seed to every estimator
    """

    #: The families this estimator implements, exposed so the wrapper and the
    #: tests agree on one list rather than each carrying a copy.
    STRUCTURES = ("identity", "group")
    W_SCHEMES = ("uniform", "tail", "exp_decay")
    B0_SCHEMES = ("zero", "marginal", "group_mean")

    def __init__(self, alpha=1.0, alpha_scale="gram", structure="identity",
                 group_penalty=0.0, w_scheme="uniform", tail_weight=0.0,
                 half_life=None, b0_scheme="zero", prior_shrinkage=0.0,
                 group_spec=None, fit_intercept=True,
                 scale_floor=1e-10, random_state=None):
        # Every argument is stored verbatim under its own name and NOTHING is
        # derived here. clone() rebuilds from get_params() for every candidate,
        # so a value computed in __init__ would be silently dropped.
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.structure = structure
        self.group_penalty = group_penalty
        self.w_scheme = w_scheme
        self.tail_weight = tail_weight
        # Stored VERBATIM, like everything else here. clone() compares
        # get_params() before and after construction by IDENTITY, so a
        # float(half_life) or a str(group_spec or "") here makes clone raise.
        # Normalisation happens in fit(), into trailing-underscore attributes.
        self.half_life = half_life
        self.b0_scheme = b0_scheme
        self.prior_shrinkage = prior_shrinkage
        self.group_spec = group_spec
        self.fit_intercept = fit_intercept
        self.scale_floor = scale_floor
        self.random_state = random_state

    # ----------------------------------------------------------------- #
    # the three generalisations
    # ----------------------------------------------------------------- #

    def _row_weights(self, y, period=None):
        """
        W's diagonal, or None for the uniform case.

        None rather than np.ones(n) so every weighted branch can be SKIPPED
        outright. That is what makes the reduction to sklearn's Ridge bit-exact
        instead of merely close: multiplying by a vector of ones still reorders
        the floating-point operations.

        "tail" weights rows by how far the label sits from the neutral bin, so
        the fit leans on the tokens the tournament actually pays for. mean(w) is
        1 by construction, which keeps the Gram on the same scale and stops
        alpha's meaning from drifting with tau.
        """
        if self.w_scheme == "uniform":
            return None
        if self.w_scheme not in self.W_SCHEMES:
            raise ValueError(
                "unknown w_scheme %r, expected one of %s"
                % (self.w_scheme, list(self.W_SCHEMES)))

        if self.w_scheme == "exp_decay":
            # The weight is a function of the DATE, so every row of one date
            # gets the same weight by construction. Deriving it from row
            # position instead would give the ~300 symbols of a single date 300
            # different weights -- a cross-sectional reweighting nobody asked
            # for -- and would depend on a row order that data_mani/utils.py
            # does not enforce (its loader ends in a bare reset_index(drop=True)
            # with no sort).
            if period is None:
                raise ValueError(
                    "w_scheme='exp_decay' needs the period of each row, and "
                    "fit() was called without one. The walk-forward routes it "
                    "for wrappers declaring WANTS_PERIOD; a caller fitting this "
                    "estimator directly must pass period=<array of dates>")
            if self.half_life is None:
                raise ValueError("w_scheme='exp_decay' needs half_life (in dates)")
            half_life = float(self.half_life)
            if half_life <= 0.0:
                raise ValueError("half_life must be > 0, got %r" % (self.half_life,))
            period = np.asarray(period)
            if period.shape[0] != y.shape[0]:
                raise ValueError(
                    "period has %d entries but there are %d rows; it must be "
                    "positional, aligned to X's ROWS -- sklearn slices fit "
                    "params by position, not by pandas index"
                    % (period.shape[0], y.shape[0]))
            # Rank the distinct dates: age is measured in DATES, so a gap in the
            # calendar is one period, matching how the rest of the pipeline
            # counts embargoes and windows.
            _, codes = np.unique(period, return_inverse=True)
            age = codes.max() - codes
            weights = 0.5 ** (age / half_life)
            # Normalised to mean 1 so the Gram keeps its scale. Under
            # alpha_scale="gram" a uniform rescale is absorbed anyway (measured
            # at 2e-15), which is what makes per-fold slicing safe -- but the
            # normalisation keeps alpha comparable under "absolute" too.
            return weights / weights.mean()

        tau = float(self.tail_weight)
        if tau == 0.0:
            return None                      # exactly the reduction case
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tail_weight must be in [0, 1], got %r" % (tau,))
        distance = np.abs(y - NEUTRAL)
        mean_distance = distance.mean()
        if not np.isfinite(mean_distance) or mean_distance <= 0.0:
            return None                      # degenerate y: no tail to lean on
        return (1.0 - tau) + tau * (distance / mean_distance)

    def _penalty_multipliers(self, spec, stems, width):
        """
        The per-group diagonal of Delta, in units of alpha_eff.

        Delta = alpha_eff * (diag(m) + rho * S L S). For one group of size k the
        block is  m_g * I_k + rho * s^2 * (I_k - J_k/k), whose eigenvalues are
        m_g along the group-MEAN direction and m_g + rho*s^2 along the k-1
        directions orthogonal to it. So m_g governs how hard the group's common
        level is shrunk and rho governs its dispersion: two eigenvalues, two
        knobs, separately identified rather than a reparameterisation of each
        other. Verified numerically.

        One trap this encodes: a multiplier applied to EVERY group is exactly
        alpha rescaled (measured difference 0.0), and alpha is already searched,
        so such an arm would be a relabelling. Unnamed groups therefore stay at
        1.0 and a spec is only ever RELATIVE.

        :return: (multipliers, the levels actually applied)
        :rtype: (np.ndarray, dict)
        """
        multipliers = np.ones(width)
        applied = {}
        for stem, fields in spec.items():
            if "p" not in fields:
                continue
            multipliers[stems[stem]] = fields["p"]
            applied[stem] = fields["p"]
        return multipliers, applied

    def _sign_bounds(self, spec, stems, width):
        """
        Box bounds from the sign fields, -inf/+inf where unconstrained.

        A sign constraint turns the closed form into a box-constrained quadratic
        program; see solve_box_constrained. The intercept never appears here --
        it is profiled out by the weighted centring, so the bounds vector has
        length p and not p + 1.

        :return: (lower, upper, the constraints actually applied)
        :rtype: (np.ndarray, np.ndarray, dict)
        """
        lower = np.full(width, -np.inf)
        upper = np.full(width, np.inf)
        applied = {}
        for stem, fields in spec.items():
            if "s" not in fields:
                continue
            index = stems[stem]
            if fields["s"] > 0:
                lower[index] = 0.0
            else:
                upper[index] = 0.0
            applied[stem] = "+" if fields["s"] > 0 else "-"
        return lower, upper, applied

    def _prior(self, gram_diagonal, cross, live, y_norm, groups):
        """
        b0, the vector the coefficients are shrunk toward.

        Built from this block's own weighted moments, which are already in hand,
        so it costs nothing and cannot see a validation row.

        "marginal"   each coefficient shrunk toward its own univariate slope.
        "group_mean" each coefficient shrunk toward the group's average
                     marginal CORRELATION, rescaled back to its own units --
                     the "these four measure one thing" prior.
        """
        p = gram_diagonal.shape[0]
        if self.b0_scheme == "zero":
            return np.zeros(p)
        if self.b0_scheme not in self.B0_SCHEMES:
            raise ValueError(
                "unknown b0_scheme %r, expected one of %s"
                % (self.b0_scheme, list(self.B0_SCHEMES)))
        kappa = float(self.prior_shrinkage)
        prior = np.zeros(p)
        if kappa == 0.0:
            return prior                     # exactly the reduction case

        if self.b0_scheme == "marginal":
            prior[live] = kappa * cross[live] / gram_diagonal[live]
            return prior

        # group_mean. t_j is the weighted marginal correlation of column j; the
        # group's mean t is mapped back into each member's own scale.
        if y_norm <= 0.0:
            return prior
        t = np.zeros(p)
        t[live] = cross[live] / (np.sqrt(gram_diagonal[live]) * y_norm)
        for indices in groups.values():
            members = [i for i in indices if live[i]]
            if len(members) < 2:
                continue
            members = np.asarray(members)
            prior[members] = (kappa * t[members].mean() * y_norm
                              / np.sqrt(gram_diagonal[members]))
        return prior

    def _apply_group_penalty(self, matrix, rhs, gram_diagonal, gram_scale,
                             live, prior, alpha_eff, groups):
        """
        Add Delta's group-fusion term to the system, in place.

        Delta = alpha_eff * (I + rho * S L S), where L is block-diagonal over the
        stem partition with each block the centring projector I_m - J_m/m, so

            b' S L S b = sum over groups of sum_j (s_j b_j - mean_k(s_k b_k))^2

        which is zero exactly when every scaled member of a group agrees. S
        carries the Gram's own column scale, so one rho can serve a design matrix
        mixing raw features (sd 0.22) with one-hots (sd 0.05).

        L is PSD (a centring projector) and S is diagonal, so S L S is PSD and
        Delta stays PSD for every rho >= 0 -- which is the condition the whole
        method rests on.

        :return: the group sizes actually penalised
        :rtype: {str: int}
        """
        rho = float(self.group_penalty)
        scale = np.zeros(gram_diagonal.shape[0])
        scale[live] = np.sqrt(gram_diagonal[live] / gram_scale)

        penalised = {}
        for stem, indices in groups.items():
            # A dead column is DROPPED from its group, not zero-scaled. Leaving
            # it in drags the group mean toward a coefficient that is not being
            # estimated, which roughly halves the surviving members.
            members = [i for i in indices if live[i]]
            if len(members) < 2:
                continue
            penalised[stem] = len(members)
            if rho == 0.0:
                continue
            members = np.asarray(members)
            s = scale[members]
            block = np.diag(s * s) - np.outer(s, s) / len(members)
            grid = np.ix_(members, members)
            matrix[grid] += alpha_eff * rho * block
            rhs[members] += alpha_eff * rho * (block @ prior[members])
        return penalised

    # ----------------------------------------------------------------- #
    # sklearn surface
    # ----------------------------------------------------------------- #

    def fit(self, X, y, period=None):
        """
        Solve (X'WX + Delta) b = X'WY + Delta b0, subject to any sign bounds.

        :param X: features; a DataFrame is required for structure="group" and
                  for any group_spec, because both are keyed on column NAMES
        :param y: target
        :param period: the DATE of each row, positional and aligned to X's rows.
                       Required by w_scheme="exp_decay" and ignored otherwise.
                       Routed by the walk-forward, never derived from row order
        :return: self
        """
        if self.alpha_scale not in ("gram", "absolute"):
            raise ValueError(
                "unknown alpha_scale %r, expected 'gram' or 'absolute'"
                % (self.alpha_scale,))
        if self.structure not in self.STRUCTURES:
            raise ValueError(
                "unknown structure %r, expected one of %s"
                % (self.structure, list(self.STRUCTURES)))

        names = list(getattr(X, "columns", []))     # BEFORE asarray drops them
        design = np.asarray(X, dtype="float64")
        target = np.asarray(y, dtype="float64").ravel()
        if design.ndim != 2:
            raise ValueError("X must be 2-d, got shape %s" % (design.shape,))
        if design.shape[0] != target.shape[0]:
            raise ValueError("X has %d rows, y has %d"
                             % (design.shape[0], target.shape[0]))
        n_rows, n_features = design.shape

        weights = self._row_weights(target, period)

        # Weighted centring, so the intercept absorbs the level and is NEVER
        # penalised. Unweighted centring under non-uniform weights biases every
        # coefficient -- measured at 5.4e-04 against sklearn on recency-like
        # weights, which is far above the 1e-15 the weighted form achieves.
        if self.fit_intercept:
            if weights is None:
                x_mean = design.mean(axis=0)
                y_mean = target.mean()
            else:
                total = weights.sum()
                x_mean = (weights @ design) / total
                y_mean = (weights @ target) / total
            centred = design - x_mean
            target_centred = target - y_mean
        else:
            x_mean = np.zeros(n_features)
            y_mean = 0.0
            centred = design
            target_centred = target

        # Spelled as a scaled design rather than X'(w[:, None] * X): the latter
        # materialises a second n-by-p array and measured 5x slower at p = 462.
        if weights is None:
            gram = centred.T @ centred
            cross = centred.T @ target_centred
            y_norm = float(np.sqrt(target_centred @ target_centred))
        else:
            scaled = centred * np.sqrt(weights)[:, None]
            gram = scaled.T @ scaled
            cross = centred.T @ (weights * target_centred)
            y_norm = float(np.sqrt(target_centred @ (weights * target_centred)))

        gram_diagonal = np.diag(gram).copy()
        # A constant column contributes nothing and would make the penalty scale
        # meaningless. On allfeat_pp roughly half the 812 columns are exactly
        # constant on a 130-date block (delisted tickers' one-hots).
        live = gram_diagonal > self.scale_floor * max(gram_diagonal.max(), 1.0)
        if not live.any():
            raise ValueError(
                "every one of the %d feature columns is constant on this "
                "training block; there is nothing to fit" % n_features)
        gram_scale = float(gram_diagonal[live].mean())

        alpha_eff = (float(self.alpha) * gram_scale
                     if self.alpha_scale == "gram" else float(self.alpha))
        if alpha_eff < 0.0:
            raise ValueError("alpha must be >= 0, got %r" % (self.alpha,))

        spec = parse_group_spec(self.group_spec)
        self.group_spec_ = spec

        groups = {}
        stems = {}
        needs_names = (self.structure == "group" or self.b0_scheme == "group_mean"
                       or bool(spec))
        if needs_names:
            if not names:
                raise ValueError(
                    "structure=%r / b0_scheme=%r / group_spec=%r needs column "
                    "NAMES, but X was passed without them; give fit() a DataFrame"
                    % (self.structure, self.b0_scheme, self.group_spec))
            stems = stem_index(names)
            groups = {stem: idx for stem, idx in stems.items() if len(idx) > 1}
            # A spec may name a construction BLOCK or a stem. Blocks are merged
            # in first and stems cannot shadow them: every stem starts with
            # "feature_" and no block name does, so the two namespaces are
            # disjoint by construction rather than by convention.
            stems = dict(stems)
            stems.update(block_index(names))
            validate_group_spec(spec, stems)

        prior = self._prior(gram_diagonal, cross, live, y_norm, groups)

        # Hypothesis 2: a per-group prior MULTIPLIER. k0 pins a group's prior to
        # zero while the rest keep the data-driven one, which is what "zero
        # priors for these groups" means when b0_scheme is not "zero".
        self.prior_levels_ = {}
        for stem, fields in spec.items():
            if "k" not in fields:
                continue
            prior[stems[stem]] *= fields["k"]
            self.prior_levels_[stem] = fields["k"]

        # Hypothesis 4: per-group penalty multipliers on Delta's diagonal.
        multipliers, self.group_alpha_levels_ = self._penalty_multipliers(
            spec, stems, n_features)

        system = gram.copy()
        system.flat[::n_features + 1] += alpha_eff * multipliers
        rhs = cross + alpha_eff * multipliers * prior

        self.groups_ = {}
        if self.structure == "group":
            self.groups_ = self._apply_group_penalty(
                system, rhs, gram_diagonal, gram_scale, live, prior,
                alpha_eff, groups)
            if not self.groups_:
                # Raise rather than quietly behaving as plain ridge. A silent
                # degradation still runs, still scores, and records
                # structure=group in the hyperparams CSV -- so the study would
                # report a comparison it never made.
                raise ValueError(
                    "structure='group' found no multi-column feature group "
                    "among %d columns, so the penalty would be plain ridge "
                    "while reporting itself as fused" % n_features)

        # Hypothesis 3: sign constraints. The branch is on whether ANY bound is
        # finite -- an unconstrained fit must take the closed form untouched, or
        # turning the feature on would silently perturb every existing arm.
        lower, upper, self.sign_levels_ = self._sign_bounds(spec, stems, n_features)
        constrained = np.isfinite(lower).any() or np.isfinite(upper).any()

        if not constrained:
            # assume_a="pos" takes LAPACK's Cholesky route -- the same one
            # sklearn's Ridge uses, which is why the reduction is exact -- and
            # RAISES on a non-positive-definite system where a general solve
            # would hand back a huge finite vector. A raise scores the candidate
            # UNSCORABLE, which is the honest outcome; no lstsq fallback.
            self.coef_ = scipy.linalg.solve(system, rhs, assume_a="pos")
            self.solver_ = "cholesky"
            self.solver_sweeps_ = 0
        else:
            # Solve on the LIVE subspace only. A dead column has an all-zero
            # Gram row after centring, so it decouples; including it would put a
            # column of the identity into the box problem and let a constraint
            # "bind" on a coefficient nobody is estimating.
            index = np.flatnonzero(live)
            block = system[np.ix_(index, index)]
            solution, sweeps = solve_box_constrained(
                block, rhs[index], lower[index], upper[index])
            self.coef_ = np.zeros(n_features)
            self.coef_[index] = solution
            # Dead coordinates keep their closed form, clipped into the box.
            dead = np.flatnonzero(~live)
            if dead.size:
                self.coef_[dead] = np.clip(
                    rhs[dead] / (alpha_eff * multipliers[dead]),
                    lower[dead], upper[dead])
            self.solver_ = "coordinate_descent"
            self.solver_sweeps_ = sweeps

        # How many LIVE coordinates actually sit on a bound. Counting dead ones
        # here would inflate the number and make a no-op arm look like a working
        # one: a constrained arm whose n_binding_ is 0 is bit-identical to its
        # unconstrained twin, and that has to be visible rather than inferred
        # from a null t-statistic.
        on_bound = np.zeros(n_features, dtype=bool)
        if constrained:
            on_bound = ((np.isfinite(lower) & (self.coef_ <= lower + 1e-12))
                        | (np.isfinite(upper) & (self.coef_ >= upper - 1e-12)))
        self.n_binding_ = int((on_bound & live).sum())
        self.n_constrained_ = int((np.isfinite(lower) | np.isfinite(upper)).sum())

        self.intercept_ = float(y_mean - x_mean @ self.coef_)

        self.n_features_in_ = n_features
        if names:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        self.alpha_eff_ = alpha_eff
        self.n_live_features_ = int(live.sum())
        self.weight_levels_ = (None if weights is None
                               else np.unique(np.round(weights, 12)))
        self.half_life_ = (float(self.half_life)
                           if self.w_scheme == "exp_decay" else None)
        return self

    def predict(self, X):
        """
        :param X: features, in the SAME columns and order as fit saw
        :return: predictions
        :rtype: np.ndarray
        """
        check_is_fitted(self, "coef_")
        if hasattr(X, "columns") and hasattr(self, "feature_names_in_"):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError(
                    "predict() got different feature names or a different "
                    "order than fit(); a name-keyed penalty makes a silent "
                    "reorder a wrong answer rather than an error")
        design = np.asarray(X, dtype="float64")
        if design.ndim != 2 or design.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has %s columns, this estimator was fitted on %d"
                % (design.shape[1] if design.ndim == 2 else "?",
                   self.n_features_in_))
        return design @ self.coef_ + self.intercept_


def _audit(data_version):
    """Print the stem partition of a dataset, from its parquet schema alone."""
    import pyarrow.parquet as pq

    path = "data/crypto/%s/train.parquet" % data_version.replace("crypto/", "")
    names = [c for c in pq.ParquetFile(path).schema.names
             if c.startswith("feature_")]
    groups = feature_groups(names)
    cliqued = sum(len(v) for v in groups.values())
    print("%s: %d features, %d groups, %d cliqued, %d singletons"
          % (data_version, len(names), len(groups), cliqued, len(names) - cliqued))
    for stem, indices in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print("  %-46s %2d  %s" % (stem, len(indices),
                                   ", ".join(names[i].replace(stem, "~")
                                             for i in indices[:4])))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit the feature-group partition of a dataset.")
    parser.add_argument("--data-version", default="v2.0")
    _audit(parser.parse_args().data_version)
