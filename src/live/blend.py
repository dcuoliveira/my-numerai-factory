"""
A slot whose prediction is a rank-average of several fitted models.

Every other CURUPIRA slot is one estimator on one feature set. CURUPIRA_05 is
five estimators, each fitted on its own block of columns with its own penalty,
whose predictions are ranked and averaged. That is a different object and it
needs its own fit/predict path.

------------------------------------------------------------------------------
Why rank, then average
------------------------------------------------------------------------------

The members are not on a common scale. A glmnet-path ridge on 740 one-hot
columns and a fixed-alpha ridge on 4 market-cap columns produce outputs with
quite different spreads, so averaging the raw numbers would silently weight them
by variance rather than by the weights asked for. Ranking first makes the
weights mean what they say.

It is also free: Numerai scores by rank within each period, so ranking a single
member's output changes nothing about how that member would have scored alone.

------------------------------------------------------------------------------
Ranking is within-period, and live that is automatic
------------------------------------------------------------------------------

In the backtest, run_ensemble ranks inside each period before averaging, because
the panel spans 510 dates. Live there is exactly one date in the block, so
ranking across the block IS ranking within the period. predict() therefore takes
the live rows for a single date and nothing about the period has to be threaded
through -- but that assumption is asserted rather than trusted, because feeding
it a multi-date frame would produce a plausible number computed the wrong way.
"""

import numpy as np
import pandas as pd

from src.prediction import meta_resid

# The column run_round attaches to the live panel: each live symbol's meta model
# as it stood LAG_DAYS before the live date, NaN where it had none. It starts
# with an underscore so raw_features never mistakes it for a model input.
META_COL = "_meta_lagged"


class MetaResidual(object):
    """
    One estimator's output residualised on the lagged meta model (E107).

    Wraps a fitted estimator -- a single model or a whole RankBlend -- and
    returns gp - strength * beta * gm instead of its raw prediction, so that a
    RankBlend member marked `resid` contributes rank(r) exactly as the backtest's
    per-period rank average of the e107_*_s050 forecasts did.

    A live date with no usable snapshot passes through unchanged (r = gp), the
    E107 rule; `last` records which of the two happened, and why, so the round
    report can say so rather than leaving it to be inferred.
    """

    def __init__(self, name, estimator, columns, strength=meta_resid.STRENGTH):
        self.name = name
        self.estimator = estimator
        self.inner_columns = list(columns)
        self.strength = float(strength)
        self.last = None

    @property
    def columns(self):
        return self.inner_columns + [META_COL]

    def predict(self, X):
        if META_COL not in X.columns:
            raise ValueError("'%s' needs %s on the live panel; run_round attaches "
                             "it with features.add_lagged_meta" % (self.name, META_COL))
        raw = np.asarray(self.estimator.predict(X[self.inner_columns]), dtype="float64")
        if not np.isfinite(raw).all():
            raise ValueError("'%s' produced non-finite output before residualising"
                             % self.name)
        meta = X[META_COL].to_numpy(dtype="float64")
        r, beta = meta_resid.residualise(raw, meta, self.strength)
        self.last = {"member": self.name, "beta": beta,
                     "coverage": int(np.isfinite(meta).sum()), "rows": len(meta),
                     "residualised": bool(np.isfinite(beta))}
        return r


class RankBlend(object):
    """
    Weighted rank-average of several fitted estimators.

    :param members: [(name, estimator, columns, weight)], weights are
                    renormalised to sum to 1
    """

    def __init__(self, members):
        if not members:
            raise ValueError("a blend needs at least one member")
        total = float(sum(m[3] for m in members))
        if total <= 0:
            raise ValueError("blend weights must sum to something positive")
        self.members = [(name, est, cols, w / total) for name, est, cols, w in members]

    @property
    def columns(self):
        """Every column any member needs, so the caller can build one frame."""
        seen = []
        for _, _, cols, _ in self.members:
            for c in cols:
                if c not in seen:
                    seen.append(c)
        return seen

    def predict(self, X):
        """
        Rank-average the members' predictions over a single-period block.

        :param X: live rows for one date, carrying every member's columns
        :type X: pd.DataFrame
        """
        blended = np.zeros(len(X), dtype="float64")
        for name, estimator, columns, weight in self.members:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                raise ValueError("blend member '%s' is missing %d columns (%s)"
                                 % (name, len(missing), missing[:3]))
            raw = estimator.predict(X[columns])
            if not np.isfinite(raw).all():
                raise ValueError("blend member '%s' produced non-finite output" % name)
            # pct ranks land in (0, 1]; the caller rank-normalises again before
            # submitting, so the exact endpoint convention here does not matter.
            blended += weight * pd.Series(raw).rank(pct=True).to_numpy()
        return blended

    def resid_log(self):
        """What each residualised member did on the last predict, outermost first."""
        out = []
        for _, estimator, _, _ in self.members:
            if isinstance(estimator, MetaResidual) and estimator.last is not None:
                out.append(estimator.last)
            inner = getattr(estimator, "estimator", estimator)
            if isinstance(inner, RankBlend):
                out.extend(inner.resid_log())
        return out
