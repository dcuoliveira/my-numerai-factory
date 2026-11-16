"""Walk-forward auxiliary volume forecasts and slot-preserving return routing.

Only historical predictions generated before their labels resolved may become
features. In-sample fitted values and fitted residuals never enter the panel.
"""
from collections import deque

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_mani.volumealpha_features import PREFIX

FORECAST_NAMES = ["hat_eta_1d", "hat_logdv_1d", "hat_logdv_5d", "hat_logdv_challenge",
                  "prob_low", "prob_high", "uncertainty", "disagreement", "previous_error",
                  "lower_logdv_1d", "downside_logdv_1d"]
FORECAST_COLUMNS = [PREFIX + x for x in FORECAST_NAMES]
COST_NAMES = ["inverse_dv", "lambda", "low_liquidity_risk", "capacity", "expected_amihud",
              "lower_dv", "downside_dv"]
COST_COLUMNS = [PREFIX + "cost_" + x for x in COST_NAMES]


def _prior_release_max(release, window=252):
    """Exact int64 timestamp rolling max, O(n), without float rounding."""
    values = release.astype('int64').to_numpy()
    nat = np.iinfo('int64').min
    result = np.full(len(values), nat, dtype='int64')
    queue = deque()
    for i in range(len(values)):
        while queue and queue[0] < i - window:
            queue.popleft()
        j = i - 1
        if j >= 0 and values[j] != nat:
            while queue and values[queue[-1]] <= values[j]:
                queue.pop()
            queue.append(j)
        if queue:
            result[i] = values[queue[0]]
    return result


def volume_targets(daily, challenge=20):
    """Direct future SUM-dollar-volume targets, not sums of overlapping forecasts.

    daily must carry the release time of the selected volume vintage. The target
    at t is unresolved until every daily observation of its horizon is available.
    """
    pieces = []
    for asset, g in daily.groupby("asset_id"):
        g = g.sort_values("date").set_index("date")
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        v = g["dollar_volume"]
        logv = np.log1p(v)
        p5 = logv.shift(1).rolling(5, min_periods=5).mean()
        p10 = logv.shift(1).rolling(252, min_periods=60).quantile(0.1)
        p90 = logv.shift(1).rolling(252, min_periods=60).quantile(0.9)
        out = pd.DataFrame({"date": g.index, "asset_id": asset, "eta": (logv - p5).to_numpy(),
                            "logdv_1d": logv.to_numpy(), "baseline": p5.to_numpy(),
                            "low": (logv < p10).where(logv.notna() & p10.notna()).to_numpy(),
                            "high": (logv > p90).where(logv.notna() & p90.notna()).to_numpy()})
        # The baseline and tail thresholds also need to have been published.
        release = pd.to_datetime(g["available_at"], utc=True)
        # Keep timestamp arithmetic exact: rolling a float64 representation of
        # epoch nanoseconds can round a just-after-cutoff release backward.
        prior_release_ns = _prior_release_max(release)
        for horizon, name in ((1, "1d"), (5, "5d"), (challenge, "challenge")):
            future = pd.concat([v.shift(-k) for k in range(horizon)], axis=1)
            out["logdv_" + name] = np.log1p(future.sum(axis=1, min_count=horizon)).to_numpy()
            releases = np.column_stack([release.shift(-k).astype('int64').to_numpy() for k in range(horizon)])
            nat = np.iinfo('int64').min
            missing = (releases == nat).any(axis=1)
            ready_ns = np.maximum(releases.max(axis=1), prior_release_ns)
            ready_ns[missing] = nat
            out["ready_" + name] = pd.to_datetime(ready_ns, utc=True)
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def auxiliary_models(seed=42, iterations=100):
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "gbm": HistGradientBoostingRegressor(max_iter=iterations, max_leaf_nodes=15,
                                             early_stopping=False, random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16, 8),
                      alpha=1.0, max_iter=iterations, early_stopping=False, random_state=seed)),
    }


def bound_to_training_range(predictions, training_target):
    """Causal stability guard: no extrapolation beyond observed training labels."""
    values = np.asarray(training_target, float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('no finite training targets for forecast bounds')
    return np.clip(predictions, values.min(), values.max())


def walk_forward_volume(raw, targets, feature_columns, refit_every=22, min_dates=130,
                        min_rows=500, seed=42, iterations=100, challenge=20,
                        max_train_rows=None, verbose=False):
    """Expanding OOF fits; label-specific publication purges at EVERY refit.

    raw.date is the prediction date; raw features must be frozen as-of that date.
    Features, labels and model metadata are returned separately for audit.
    """
    if refit_every < 1 or min_dates < 1 or min_rows < 2:
        raise ValueError("invalid auxiliary training window")
    if raw.duplicated(["asset_id", "date"]).any():
        raise ValueError("duplicate auxiliary feature rows")
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    t = targets.copy()
    t["date"] = pd.to_datetime(t["date"], utc=True)
    for c in ("ready_1d", "ready_5d", "ready_challenge"):
        t[c] = pd.to_datetime(t[c], utc=True)
    d = d.merge(t, on=["asset_id", "date"], how="left", validate="1:1")
    d = d.sort_values(["date", "asset_id"]).reset_index(drop=True)
    out = d[["asset_id", "date"]].copy()
    for c in FORECAST_COLUMNS:
        out[c] = np.nan
    per_model, logs = [], []
    complete = np.isfinite(d[feature_columns].to_numpy(float)).all(axis=1)
    dates = sorted(d["date"].unique())
    models = auxiliary_models(seed, iterations)
    def cap(mask, salt):
        if max_train_rows is None or mask.sum() <= max_train_rows:
            return mask
        selected = np.random.default_rng(seed + salt).choice(np.flatnonzero(mask), max_train_rows, replace=False)
        bounded = pd.Series(False, index=d.index)
        bounded.iloc[selected] = True
        return bounded
    for start in range(min_dates, len(dates), refit_every):
        predicted = dates[start:start + refit_every]
        cutoff = pd.Timestamp(predicted[0])
        test = d["date"].isin(predicted) & complete
        if not test.any():
            continue
        if verbose:
            print(f"volume refit {cutoff.date()}: {int(test.sum()):,} future rows", flush=True)
        for target, suffix, ready in (("eta", "eta", "ready_1d"), ("logdv_1d", "1d", "ready_1d"), ("logdv_5d", "5d", "ready_5d"),
                                       ("logdv_challenge", "challenge", "ready_challenge")):
            train = (d["date"] < cutoff) & (d[ready] <= cutoff) & complete & d[target].notna()
            if train.sum() < min_rows or d.loc[train, "date"].nunique() < min_dates:
                continue
            train = cap(train, start)
            if not (d.loc[train, "date"].max() < cutoff and d.loc[train, ready].max() <= cutoff):
                raise AssertionError("auxiliary labels cross prediction cutoff")
            estimates = []
            for name, estimator in models.items():
                fitted = clone(estimator).fit(d.loc[train, feature_columns], d.loc[train, target].astype(float))
                pred = bound_to_training_range(fitted.predict(d.loc[test, feature_columns]), d.loc[train, target])
                estimates.append(pred)
                q = d.loc[test, ["date", "asset_id"]].copy()
                q["model"], q["target"], q["prediction"] = name, target, pred
                q["fit_at"] = cutoff
                q["max_train_date"] = d.loc[train, "date"].max()
                q["max_label_available_at"] = d.loc[train, ready].max()
                per_model.append(q)
            mean = np.mean(estimates, axis=0)
            col = "hat_eta_1d" if suffix == "eta" else "hat_logdv_" + suffix
            out.loc[test, PREFIX + col] = mean
            if suffix == "1d":
                out.loc[test, PREFIX + "disagreement"] = np.std(estimates, axis=0)
            logs.append({"fit_at": str(cutoff), "target": target, "rows": int(train.sum()),
                         "max_train_date": str(d.loc[train, "date"].max()),
                         "max_label_available_at": str(d.loc[train, ready].max())})
            if verbose:
                print(f"  {target}: {int(train.sum()):,} past training rows; ridge/GBM/MLP complete", flush=True)

        # A lower-quantile model explicitly prices overestimation asymmetrically.
        train = (d["date"] < cutoff) & (d["ready_1d"] <= cutoff) & complete & d["logdv_1d"].notna()
        if train.sum() >= min_rows and d.loc[train, "date"].nunique() >= min_dates:
            train = cap(train, start)
            q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, max_iter=iterations,
                    max_leaf_nodes=15, early_stopping=False, random_state=seed)
            q10.fit(d.loc[train, feature_columns], d.loc[train, "logdv_1d"])
            lower = q10.predict(d.loc[test, feature_columns])
            out.loc[test, PREFIX + "lower_logdv_1d"] = lower
            mean = out.loc[test, PREFIX + "hat_logdv_1d"].to_numpy()
            out.loc[test, PREFIX + "downside_logdv_1d"] = np.minimum(mean, lower)
            for target, name in (("low", "prob_low"), ("high", "prob_high")):
                tr = train & d[target].notna()
                if tr.sum() < min_rows:
                    continue
                labels = d.loc[tr, target].astype(int)
                if labels.nunique() == 1:
                    prob = float(labels.iloc[0])
                else:
                    classifier = HistGradientBoostingClassifier(max_iter=iterations, max_leaf_nodes=15,
                                          early_stopping=False, random_state=seed)
                    classifier.fit(d.loc[tr, feature_columns], labels)
                    prob = classifier.predict_proba(d.loc[test, feature_columns])[:, 1]
                out.loc[test, PREFIX + name] = prob

    # Only previously REALIZED OOF errors can inform a later feature. In
    # particular a 5-day or challenge forecast is never treated as a 1-day label.
    for asset, ix in d.groupby("asset_id").groups.items():
        ix = np.asarray(ix)
        date_ns = d.loc[ix, "date"].astype("int64").to_numpy()
        release_ns = d.loc[ix, "ready_1d"].astype("int64").to_numpy()
        errors = (d.loc[ix, "logdv_1d"].to_numpy(float) -
                  out.loc[ix, PREFIX + "hat_logdv_1d"].to_numpy(float))
        valid = np.isfinite(errors) & d.loc[ix, "ready_1d"].notna().to_numpy()
        previous, uncertainty = np.full(len(ix), np.nan), np.full(len(ix), np.nan)
        for j, day in enumerate(date_ns):
            resolved = np.flatnonzero(valid[:j] & (release_ns[:j] <= day))
            if len(resolved):
                previous[j] = errors[resolved[-1]]
            if len(resolved) >= 20:
                uncertainty[j] = np.std(errors[resolved[-120:]])
        out.loc[ix, PREFIX + "previous_error"] = previous
        out.loc[ix, PREFIX + "uncertainty"] = uncertainty
    return out, pd.concat(per_model, ignore_index=True) if per_model else pd.DataFrame(), logs


def trading_costs(forecasts, raw, trade_sizes=(1000, 10000, 100000)):
    """Scenario sizes are assumptions, not market observations or executed costs."""
    out = forecasts[["asset_id", "date"]].copy()
    predicted = np.expm1(forecasts[PREFIX + "hat_logdv_1d"].clip(upper=50)).where(lambda x: x > 0)
    inv = 1.0 / predicted
    out[PREFIX + "cost_inverse_dv"] = inv
    out[PREFIX + "cost_lambda"] = 0.2 * inv
    out[PREFIX + "cost_low_liquidity_risk"] = forecasts[PREFIX + "prob_low"]
    x = raw.set_index(["asset_id", "date"]).reindex(pd.MultiIndex.from_frame(out[["asset_id", "date"]]))
    out[PREFIX + "cost_capacity"] = predicted.to_numpy() / x[PREFIX + "market_cap"].replace(0, np.nan).to_numpy()
    out[PREFIX + "cost_expected_amihud"] = x[PREFIX + "return_ma1"].abs().to_numpy() * inv.to_numpy()
    for kind in ("lower", "downside"):
        out[PREFIX + f"cost_{kind}_dv"] = np.expm1(forecasts[PREFIX + f"{kind}_logdv_1d"].clip(upper=50)).where(lambda v: v >= 0)
    for size in trade_sizes:
        if not np.isfinite(size) or size <= 0:
            raise ValueError("trade sizes must be positive finite scenario parameters")
        out[PREFIX + f"scenario_{size:g}_participation"] = size * inv
    return out.replace([np.inf, -np.inf], np.nan)


def auxiliary_metrics(forecasts, targets, raw, challenge=20, model_predictions=None):
    """OOF volume skill, lower-tail calibration and known-at-cutoff group splits."""
    d = forecasts.merge(targets, on=["asset_id", "date"], how="left", validate="1:1")
    x = raw[["asset_id", "date", PREFIX + "market_cap", PREFIX + "return_ma1"]]
    d = d.merge(x, on=["asset_id", "date"], how="left", validate="1:1")
    def group_rank(col):
        r = d.groupby("date")[col].rank(pct=True, method="average")
        return np.ceil(3 * r).clip(1, 3)
    d["liquidity_group"] = group_rank("baseline")
    d["market_cap_group"] = group_rank(PREFIX + "market_cap")
    # Predeclared stress dates are diagnostic labels, never forecast inputs.
    stress = {"2020-03", "2021-05", "2022-05", "2022-11", "2024-08", "2025-10"}
    d["stress"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m").isin(stress)
    groups = [("all", "all", d)]
    for column in ("liquidity_group", "market_cap_group", "stress"):
        groups += [(column, str(key), g) for key, g in d.groupby(column)]
    rows = []
    for grouping, key, g in groups:
        for suffix, target, horizon in (("1d", "logdv_1d", 1), ("5d", "logdv_5d", 5),
                                         ("challenge", "logdv_challenge", challenge)):
            prediction = PREFIX + "hat_logdv_" + suffix
            q = g.dropna(subset=[prediction, target, "baseline"])
            if q.empty:
                continue
            # The baseline for a direct cumulative-volume horizon is the
            # trailing-5-day mean log level converted to dollars, times h.
            base = np.log1p(horizon * np.expm1(q["baseline"].clip(upper=50)))
            err = q[prediction] - q[target]
            mse0 = np.square(base - q[target]).mean()
            row = {"grouping": grouping, "group": key, "horizon": horizon, "n": len(q),
                   "oof_r2_vs_trailing5": 1 - np.square(err).mean() / mse0 if mse0 > 0 else np.nan,
                   "mae_log_volume": float(err.abs().mean())}
            if suffix == "1d":
                low = q.dropna(subset=[PREFIX + "lower_logdv_1d"])
                row["q10_realized_fraction_below"] = float((low[target] < low[PREFIX + "lower_logdv_1d"]).mean()) if len(low) else np.nan
                row["overestimate_weighted_loss"] = float((9 * err.clip(lower=0) + (-err).clip(lower=0)).mean())
                for name, label in (("prob_low", "low"), ("prob_high", "high")):
                    z = q.dropna(subset=[PREFIX + name, label])
                    row[name + "_brier"] = float(np.square(z[PREFIX + name] - z[label].astype(float)).mean()) if len(z) else np.nan
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["model"] = "ensemble"
    per_model = []
    if model_predictions is not None and not model_predictions.empty:
        truth = targets.set_index(["asset_id", "date"])
        for (model, target), p in model_predictions.groupby(["model", "target"]):
            y = truth.reindex(pd.MultiIndex.from_frame(p[["asset_id", "date"]]))
            horizon = {"eta": 1, "logdv_1d": 1, "logdv_5d": 5, "logdv_challenge": challenge}[target]
            actual = y[target].to_numpy(float)
            pred = p["prediction"].to_numpy(float)
            baseline = np.zeros(len(p)) if target == "eta" else np.log1p(horizon * np.expm1(y["baseline"].clip(upper=50).to_numpy(float)))
            valid = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(baseline)
            if not valid.any():
                continue
            e, e0 = pred[valid] - actual[valid], baseline[valid] - actual[valid]
            mse0 = np.square(e0).mean()
            per_model.append({"model": model, "target": target, "grouping": "all", "group": "all", "horizon": horizon,
                              "n": int(valid.sum()), "oof_r2_vs_trailing5": 1 - np.square(e).mean() / mse0 if mse0 > 0 else np.nan,
                              "mae_log_volume": float(np.abs(e).mean())})
    return pd.concat([result, pd.DataFrame(per_model)], ignore_index=True)


def route_slots(baseline, candidate, covered):
    """Uncovered coins keep their exact B0 ranks, including tied rank values.

    For deterministic routing of candidate ties use stable baseline order. The
    returned B0 vector is the explicit baseline prediction for every fallback.
    """
    baseline, candidate = np.asarray(baseline, float), np.asarray(candidate, float)
    covered = np.asarray(covered, bool) & np.isfinite(candidate)
    if not np.isfinite(baseline).all() or len(baseline) != len(candidate) or len(covered) != len(baseline):
        raise ValueError("routing requires aligned full-universe finite baseline predictions")
    ranks = pd.Series(baseline).rank(method="average").to_numpy() / len(baseline)
    out = ranks.copy()
    ix = np.flatnonzero(covered)
    order = ix[np.argsort(candidate[ix], kind="stable")]
    out[order] = np.sort(ranks[ix])
    if not np.array_equal(out[~covered], ranks[~covered]):
        raise AssertionError("uncovered ranks changed")
    return out, ranks
