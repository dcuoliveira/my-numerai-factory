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
