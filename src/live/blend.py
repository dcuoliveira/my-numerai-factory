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


# --------------------------------------------------------------------------- E216
#: Where the JACI pre-stage (src.live.jaci) leaves its forecast for run_round.
JACI_FORECASTS = "models/live/jaci/forecasts.parquet"
#: Fewer JACI-covered live coins than this and the routed slot passes its full
#: members through unchanged rather than re-ranking a handful of coins.
MIN_ROUTED_ROWS = 20


class JaciForecast(object):
    """
    JACI_01 live: the pre-stage's equal-rank 4-stream forecast for today's
    coins, residualised at 0.75 on the lagged meta model (E214 FIXED_s0.75).

    It fits nothing. src.live.jaci writes jaci_raw per (date, symbol) before
    the round; this reads the live date's rows and applies the same
    meta_resid.residualise E214 used, on the meta snapshot run_round attached.
    A coin JACI does not cover -- too little price history, no head, no
    source -- comes back NaN, and RoutedBlend keeps the other members' rank
    for it. Nothing here substitutes another model's numbers for JACI's.
    """

    def __init__(self, name, path=JACI_FORECASTS, strength=0.75):
        from src.live.features import PERIOD, SYMBOL
        self.name = name
        self.path = path
        self.strength = float(strength)
        self.symbol_col, self.period_col = SYMBOL, PERIOD
        self.last = None

    @property
    def columns(self):
        return [self.symbol_col, self.period_col, META_COL]

    def predict(self, X):
        dates = pd.to_datetime(X[self.period_col]).unique()
        if len(dates) != 1:
            raise ValueError("JaciForecast predicts one live date, got %d" % len(dates))
        date = pd.Timestamp(dates[0])
        out = np.full(len(X), np.nan)
        self.last = {"member": self.name, "date": str(date.date()), "covered": 0,
                     "rows": len(X), "beta": np.nan, "residualised": False,
                     "note": None}
        try:
            frame = pd.read_parquet(self.path, columns=["date", "symbol", "jaci_raw"])
        except Exception as error:
            self.last["note"] = "no JACI forecast file (%s)" % error
            return out
        frame = frame[(pd.to_datetime(frame["date"]) == date) & frame["jaci_raw"].notna()]
        if frame.empty:
            self.last["note"] = "JACI forecast has no rows for %s" % date.date()
            return out
        raw = X[self.symbol_col].map(
            frame.drop_duplicates("symbol").set_index("symbol")["jaci_raw"]).to_numpy(float)
        covered = np.isfinite(raw)
        if covered.any():
            meta = X[META_COL].to_numpy(dtype="float64")[covered] if META_COL in X else None
            out[covered], beta = meta_resid.residualise(raw[covered], meta, self.strength)
            self.last.update(covered=int(covered.sum()), beta=beta,
                             residualised=bool(np.isfinite(beta)),
                             coverage=int(np.isfinite(meta).sum()) if meta is not None else 0)
        return out


class RoutedBlend(object):
    """
    A slot blend with members that cover only part of the universe (E216).

    `full` members must predict every live coin; `partial` ones (JACI) may
    return NaN. On the coins every member covers, the members' ranks WITHIN
    that subset are averaged with the declared weights -- E215's equal-rank
    blend on the shared rows -- and that ordering is routed into the full
    members' own rank slots there (asset_momentum.route_slots, the repo's
    deployment rule since E134). Every other coin keeps the full members' rank.

    If fewer than MIN_ROUTED_ROWS coins are covered, the full members go out
    unchanged and `route` says so; run_round turns that into a warning.
    """

    def __init__(self, members, partial):
        self.inner = RankBlend(members)
        self.members = self.inner.members
        self.partial = set(partial)
        if not self.partial or self.partial >= {m[0] for m in self.members}:
            raise ValueError("a routed blend needs both full and partial members")
        self.route = None

    @property
    def columns(self):
        return self.inner.columns

    def predict(self, X):
        from src.analysis.asset_momentum import route_slots
        from src.live.submit import rank_normalise
        values, weights = {}, {}
        for name, estimator, columns, weight in self.members:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                raise ValueError("blend member '%s' is missing %d columns (%s)"
                                 % (name, len(missing), missing[:3]))
            raw = np.asarray(estimator.predict(X[columns]), dtype="float64")
            if name not in self.partial and not np.isfinite(raw).all():
                raise ValueError("blend member '%s' produced non-finite output" % name)
            values[name], weights[name] = raw, weight
        full = [n for n in values if n not in self.partial]
        total = sum(weights[n] for n in full)
        base = sum(weights[n] / total * pd.Series(values[n]).rank(pct=True).to_numpy()
                   for n in full)
        covered = np.all([np.isfinite(values[n]) for n in self.partial], axis=0)
        notes = [getattr(est, "last", None) or {} for n, est, _, _ in self.members
                 if n in self.partial]
        self.route = {"covered": int(covered.sum()), "rows": len(X),
                      "pass_through": bool(covered.sum() < MIN_ROUTED_ROWS),
                      "partial": sorted(self.partial),
                      "why": "; ".join(filter(None, (n.get("note") for n in notes))) or None}
        if self.route["pass_through"]:
            return base
        candidate = np.full(len(X), np.nan)
        candidate[covered] = sum(weights[n] * rank_normalise(values[n][covered]) for n in values)
        return route_slots(rank_normalise(base), candidate)

    def route_log(self):
        return self.route

    def resid_log(self):
        out = []
        for name, estimator, _, _ in self.members:
            if isinstance(estimator, (MetaResidual, JaciForecast)) and estimator.last is not None:
                if isinstance(estimator, JaciForecast) and not estimator.last["covered"]:
                    continue    # nothing residualised; route_log carries the reason
                out.append(estimator.last)
            inner = getattr(estimator, "estimator", estimator)
            if isinstance(inner, RankBlend):
                out.extend(inner.resid_log())
        return out
