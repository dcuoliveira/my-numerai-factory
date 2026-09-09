"""
Bayesian hyperparameter search: translating this pipeline's grids for skopt.

hyper_params_search() can drive either RandomizedSearchCV or scikit-optimize's
BayesSearchCV. This module is what makes the second one possible without
writing a single new grid.

------------------------------------------------------------------------------
Why translate rather than write a second grid per wrapper
------------------------------------------------------------------------------

Every wrapper in prediction.models declares its grid in one dialect: scipy
frozen distributions (sp_randint, sp_uniform) plus explicit numpy arrays. Two of
them build that grid FROM THE TRAINING DATA at refit time -- RidgeGlmnetWrapper's
lambda path, which depends on y, and _BoosterWrapper's leaf range, which depends
on the block size. A hand-written second grid would have to duplicate both, and
would drift from them silently the first time either was tuned.

Translating instead means the Bayesian option inherits every grid, including the
two data-derived ones, and a change to a wrapper's ranges is picked up by both
search types at once.

------------------------------------------------------------------------------
What the translation changes, and why that matters when comparing runs
------------------------------------------------------------------------------

One thing, and it is worth stating plainly: an explicit array becomes a
CONTINUOUS interval.

    np.logspace(-3, -2, 40)  ->  Real(0.001, 0.01, prior="log-uniform")

The random search could only ever draw one of those 40 points. The Bayesian
search can propose anything in between. That is a superset, and it is the whole
point -- a surrogate fitted over a continuous interval is what makes the search
Bayesian rather than a lookup over a fixed list.

But it does mean the two search types are not sampling the same space, so
`--search-type bayes` vs `--search-type random` compares two SEARCH STRATEGIES,
not one strategy against another over one grid. Do not read a difference between
them as "Bayesian found a better optimum" without checking the chosen values in
results/hyperparams/: a value strictly between two grid points is evidence the
continuity mattered, and a value the random grid also contained is not.

The log-uniform prior is inferred, not assumed. An array is treated as
log-spaced when its values are all positive and evenly spaced in log10, which
catches every scale parameter here (alpha, C, learning_rate, reg_lambda are all
built with np.logspace, and the glmnet path is a logspace too). A linearly
spaced array falls back to a uniform prior. Getting this wrong is not cosmetic:
a uniform prior over [1e-4, 1e2] puts 99% of its mass above 1, which is exactly
the failure the RidgeWrapper docstring warns about for linspace.

------------------------------------------------------------------------------
The budget problem, which is the reason this module has a warning in it
------------------------------------------------------------------------------

skopt fits ONE INDEPENDENT SURROGATE PER SUB-GRID. The boosters ship six
sub-grids -- one per max_depth, so num_leaves stays coherent with it, see
models.depth_leaf_grids -- so a nominal --n-iter 50 is about 8 draws per
surrogate, not 50.

skopt's own default is n_initial_points=10 random draws before a surrogate is
fitted at all. At 8 draws per sub-grid a Bayesian search would therefore never
leave its random-initialisation phase: it would be random search wearing a
different name, slower, and with an extra dependency. That is a silent failure
-- the run completes and reports plausible numbers -- which is exactly the class
of bug this pipeline is built to refuse.

So initial_points() scales the warm-up to the budget rather than taking skopt's
default, and describe_budget() returns notes that say plainly when the budget is
too thin for the surrogate to do anything. The caller prints them once per run.

To actually get Bayesian behaviour out of a booster, either raise --n-iter to
150+ (25+ draws per sub-grid, of which the first few are warm-up), or use a
model whose grid is a single dict -- ridge and ridge_glmnet both are, so they
get the full budget on one surrogate.
"""

import numpy as np


INSTALL_HINT = (
    "scikit-optimize is required for --search-type bayes. Install it with:\n"
    "    pip install scikit-optimize"
)

# Below this many draws per sub-grid, the surrogate gets so little to fit that
# the search is effectively random. Not a hard limit -- the run proceeds and
# says so -- because "random search with more machinery" is a legitimate thing
# to measure once, and refusing to run would make that impossible.
THIN_BUDGET = 12


def _skopt_space():
    """Import skopt's dimension types, with an actionable error if it is absent."""
    try:
        from skopt.space import Categorical, Integer, Real
    except ImportError as exc:                      # pragma: no cover - env dependent
        raise ImportError(INSTALL_HINT) from exc
    return Real, Integer, Categorical


def is_log_spaced(values, rtol=1e-6):
    """
    Decide whether an explicit grid was laid out with np.logspace.

    The test is on the SPACING, not on the range: constant steps in log10 over
    strictly positive values. That is what np.logspace produces and what a
    linspace over the same interval does not, so a wide linear grid is correctly
    left on a uniform prior.

    :param values: the explicit grid
    :type values: array-like
    :param rtol: tolerance on step variation, relative to the mean step
    :type rtol: float
    :return: True if a log-uniform prior is the right reading of this grid
    :rtype: bool
    """
    unique = np.unique(np.asarray(values, dtype="float64"))
    if unique.size < 3 or (unique <= 0.0).any():
        return False
    steps = np.diff(np.log10(unique))
    mean_step = steps.mean()
    if mean_step <= 0.0:
        return False
    return bool(np.ptp(steps) <= rtol * mean_step)


def to_dimension(name, value):
    """
    Translate one grid entry into an skopt dimension.

    :param name: parameter name, used only in error messages
    :type name: str
    :param value: a frozen scipy distribution, or an explicit list/array
    :return: an skopt Real, Integer or Categorical
    """
    Real, Integer, Categorical = _skopt_space()

    if hasattr(value, "rvs"):
        # A frozen scipy distribution. support() returns INCLUSIVE bounds for
        # the discrete ones -- sp_randint(2000, 30001).support() is
        # (2000, 30000) -- so no off-by-one correction belongs here.
        low, high = value.support()
        if value.dist.name == "randint":
            return Integer(int(low), int(high))
        if not np.isfinite([low, high]).all():
            raise ValueError(
                "cannot translate unbounded distribution for '{}' into a "
                "Bayesian search space; give it explicit bounds".format(name)
            )
        if float(low) >= float(high):
            return Categorical([float(low)])
        return Real(float(low), float(high))

    # A grid whose entries are THEMSELVES sequences stays categorical. An MLP's
    # hidden_layer_sizes is one tuple per architecture, and the ravel below would
    # flatten [(128, 64), (256, 128)] into [128, 64, 256, 128] and hand back an
    # Integer over 64..256 -- so the search would propose a plain int where the
    # estimator needs a tuple. That does not fail here: it fails inside every
    # candidate fit, each one scores UNSCORABLE, and the run completes having
    # selected nothing. Checked against `value` rather than the raveled list,
    # because raveling is what destroys the evidence.
    if isinstance(value, (list, tuple)) and any(
        isinstance(entry, (list, tuple, np.ndarray)) for entry in value
    ):
        # Normalised to tuples so the categories are hashable whatever the grid
        # was written with -- a list entry or an ndarray entry is not.
        return Categorical([
            tuple(entry) if isinstance(entry, (list, tuple, np.ndarray)) else entry
            for entry in value
        ])

    raw = np.asarray(value).ravel().tolist()
    if not raw:
        raise ValueError("empty grid for '{}'".format(name))

    try:
        numeric = np.asarray(raw, dtype="float64")
    except (TypeError, ValueError):
        return Categorical(raw)                     # strings, booleans, None

    unique = np.unique(numeric)
    # One value is a pin (subsample_freq, and the max_depth/num_leaves pairing).
    # Two is a switch. Neither has an interior to interpolate over, so keep them
    # categorical rather than inventing a continuum between them.
    if unique.size <= 2:
        return Categorical(raw)

    prior = "log-uniform" if is_log_spaced(numeric) else "uniform"
    if np.issubdtype(np.asarray(raw).dtype, np.integer):
        return Integer(int(unique.min()), int(unique.max()), prior=prior)
    return Real(float(unique.min()), float(unique.max()), prior=prior)


def to_space(grid):
    """
    Translate one sub-grid (a dict) into a dict of skopt dimensions.

    :param grid: parameter name -> distribution or explicit list
    :type grid: dict
    :rtype: dict
    """
    return {name: to_dimension(name, value) for name, value in grid.items()}


def split_budget(n_grids, n_iter):
    """
    Divide the draw budget across sub-grids, one surrogate each.

    Every sub-grid gets at least one draw. If n_iter is smaller than the number
    of sub-grids the total therefore EXCEEDS n_iter -- deliberately, because the
    alternative is dropping whole sub-grids, and for the boosters that means
    silently never trying some tree depths. The caller reports the real total.

    :param n_grids: number of sub-grids
    :type n_grids: int
    :param n_iter: requested total draws
    :type n_iter: int
    :return: draws per sub-grid
    :rtype: [int]
    """
    if n_grids <= 0:
        raise ValueError("n_grids must be positive, got {}".format(n_grids))
    if n_iter <= 0:
        raise ValueError("n_iter must be positive, got {}".format(n_iter))

    base = max(1, n_iter // n_grids)
    budgets = [base] * n_grids
    for i in range(max(0, n_iter - base * n_grids)):
        budgets[i % n_grids] += 1
    return budgets


def initial_points(per_grid, requested=0):
    """
    How many random draws precede the first surrogate fit, for one sub-grid.

    skopt defaults to 10 regardless of budget, which silently turns a small
    per-grid budget into pure random search. Scale it instead: a third of the
    budget, capped at skopt's 10 and floored at 3, and never more than the
    budget itself.

    :param per_grid: draws available to this sub-grid
    :type per_grid: int
    :param requested: explicit override, 0 for auto
    :type requested: int
    :rtype: int
    """
    if requested > 0:
        return int(max(1, min(requested, per_grid)))
    return int(max(1, min(10, max(3, per_grid // 3), per_grid)))


def describe_budget(budgets, n_init):
    """
    Explain what the surrogate will actually get, and warn when that is nothing.

    :param budgets: draws per sub-grid, from split_budget()
    :type budgets: [int]
    :param n_init: random warm-up draws per sub-grid
    :type n_init: int
    :return: lines for the caller to print
    :rtype: [str]
    """
    n_grids = len(budgets)
    total = sum(budgets)
    smallest = min(budgets)
    modelled = smallest - n_init

    notes = [
        "bayes: {} sub-grid{} x ~{} draws = {} fits per refit, "
        "{} random warm-up draws each, ~{} surrogate-guided".format(
            n_grids, "" if n_grids == 1 else "s", smallest, total,
            n_init, max(0, modelled)),
    ]

    if modelled <= 0:
        notes.append(
            "bayes WARNING: every draw is warm-up -- no surrogate is ever fitted, "
            "so this is random search with extra overhead. Raise --n-iter to at "
            "least {} (about {} per sub-grid) or use a single-grid model "
            "(ridge, ridge_glmnet).".format(THIN_BUDGET * n_grids, THIN_BUDGET)
        )
    elif smallest < THIN_BUDGET:
        notes.append(
            "bayes NOTE: only ~{} surrogate-guided draw(s) per sub-grid. The "
            "search is mostly random at this budget; --n-iter {} would give it "
            "something to work with.".format(modelled, THIN_BUDGET * n_grids)
        )
    return notes


def space_bounds(spaces):
    """
    Report what each parameter was searched over, matching the random-search form.

    Same (min, max, size) triple hyper_params_search attaches for
    RandomizedSearchCV, so hyperparams_table() needs no special case. `size` is
    the number of REACHABLE values, and 0 means a continuous interval -- which
    is the honest answer for a Real dimension and the one visible difference
    between the two search types in the hyperparameter history.

    :param spaces: list of (space dict, n_iter) tuples
    :return: name -> (min, max, size)
    :rtype: dict
    """
    merged = {}
    for space, _ in spaces:
        for name, dim in space.items():
            kind = type(dim).__name__
            if kind == "Categorical":
                try:
                    values = np.asarray(dim.categories, dtype="float64")
                except (TypeError, ValueError):
                    continue                        # non-numeric: nothing to bound
                # A Categorical over SEQUENCES needs the ndim check as well as
                # the except. An MLP's hidden_layer_sizes is one tuple per
                # architecture: ragged entries raise above and are caught, but
                # equal-length ones coerce to a 2-D array instead, and its min
                # and max would be over layer widths pooled across architectures
                # -- a bound on nothing. The scalar update below would then fail
                # on a whole row rather than a value.
                if values.ndim != 1 or not values.size:
                    continue
                lo, hi = float(values.min()), float(values.max())
            else:
                lo, hi = float(dim.low), float(dim.high)

            state = merged.setdefault(
                name, {"lo": lo, "hi": hi, "kinds": set(), "values": set()})
            state["lo"] = min(state["lo"], lo)
            state["hi"] = max(state["hi"], hi)
            state["kinds"].add(kind)
            if kind == "Categorical":
                state["values"].update(float(v) for v in values)

    # Counting is per KIND, not a sum over sub-grids. Every booster sub-grid
    # carries the same n_estimators range, so adding the six counts would report
    # 168,006 reachable values where there are 28,001 -- and that column is what
    # a report reads to decide whether a chosen value sat against an edge.
    bounds = {}
    for name, state in merged.items():
        if "Real" in state["kinds"]:
            size = 0                                # continuous anywhere wins
        elif "Integer" in state["kinds"]:
            size = int(state["hi"] - state["lo"]) + 1
        else:
            size = len(state["values"])             # distinct pins, e.g. 6 depths
        bounds[name] = (state["lo"], state["hi"], size)
    return bounds


def sequence_valued_params(param_grid):
    """
    Names in a grid whose VALUES are sequences, which skopt cannot search.

    An MLP's hidden_layer_sizes is one tuple per architecture. to_dimension()
    translates that correctly, into a Categorical whose categories are tuples --
    but skopt cannot round-trip such a dimension. Even a single-category one
    fails inside BayesSearchCV.fit() with "can only convert an array of size 1
    to a Python scalar", because its transformer assumes a category is a scalar.

    That is a limitation of the optimiser, not of the translation, so it is
    reported rather than worked around: encoding architectures behind an integer
    index would need a custom estimator, to buy a search strategy that the
    random one already covers.

    :param param_grid: a dict, or a list of sub-grids, in the wrapper dialect
    :return: the offending parameter names, sorted
    :rtype: [str]
    """
    names = set()
    for grid in (param_grid if isinstance(param_grid, list) else [param_grid]):
        for name, value in grid.items():
            if hasattr(value, "rvs"):
                continue
            if isinstance(value, (list, tuple)) and any(
                isinstance(entry, (list, tuple, np.ndarray)) for entry in value
            ):
                names.add(name)
    return sorted(names)


def build_bayes_search(estimator, param_grid, n_iter, cv, scoring, refit,
                       search_jobs, seed, error_score, init_points=0,
                       return_train_score=True):
    """
    Configure a BayesSearchCV equivalent to the RandomizedSearchCV it replaces.

    Everything the pipeline reads off the returned object -- best_estimator_,
    best_params_, best_score_, best_index_, cv_results_ (including the
    ride-along corr and bmc scorers) -- is present and carries the same meaning,
    so walk_forward_fit_and_predict() does not branch on search type.

    Two knobs differ from the random search:

    * n_points batches proposals so search_jobs stays useful. Bayesian
      optimisation is sequential by nature -- it fits a surrogate, then asks it
      where to look next -- so with n_points=1 the only parallelism left is
      across CV folds. Batching costs a little sample efficiency (the second
      point in a batch is chosen without seeing the first one's result) and buys
      back the candidate-level parallelism the random search had for free.

    * n_initial_points is derived from the budget rather than left at skopt's
      default of 10. See this module's docstring for why that default is a trap
      here.

    :param param_grid: a dict, or a list of sub-grids, in the wrapper dialect
    :param init_points: explicit warm-up draws per sub-grid, 0 for auto
    :return: an unfitted BayesSearchCV, with searched_bounds_ and budget_notes_
    """
    try:
        from skopt import BayesSearchCV
    except ImportError as exc:                      # pragma: no cover - env dependent
        raise ImportError(INSTALL_HINT) from exc

    unsupported = sequence_valued_params(param_grid)
    if unsupported:
        raise ValueError(
            "--search-type bayes cannot search {}: skopt has no representation "
            "for a parameter whose values are sequences (an MLP's "
            "hidden_layer_sizes is one tuple per architecture), and it fails "
            "inside the fit rather than here. Use --search-type random, which "
            "is the default and searches this grid exactly as written.".format(
                ", ".join(unsupported))
        )

    grids = param_grid if isinstance(param_grid, list) else [param_grid]
    budgets = split_budget(len(grids), n_iter)
    n_init = initial_points(min(budgets), requested=init_points)

    spaces = [(to_space(grid), budget) for grid, budget in zip(grids, budgets)]

    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=spaces,
        cv=cv,
        scoring=scoring,
        refit=refit,
        return_train_score=return_train_score,
        error_score=error_score,      # a crashed candidate loses, it does not abort
        random_state=seed,
        n_jobs=search_jobs,
        # Never more batched proposals than the smallest sub-grid can afford,
        # or a one-draw sub-grid would ask the optimizer for points it will
        # then throw away.
        n_points=max(1, min(search_jobs, min(budgets))),
        optimizer_kwargs={"n_initial_points": n_init},
    )
    search.searched_bounds_ = space_bounds(spaces)
    search.budget_notes_ = describe_budget(budgets, n_init)
    return search
