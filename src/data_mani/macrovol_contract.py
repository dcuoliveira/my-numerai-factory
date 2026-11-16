"""Point-in-time source validation and publication-aware macro attachment."""

import numpy as np
import pandas as pd

REQUIRED = {"date", "series", "value", "source", "source_id", "available_at",
            "retrieved_at", "series_version", "availability_basis", "raw_sha256",
            "series_definition", "source_timestamp", "source_timestamp_precision"}
DEFINITIONS = {"gold": {"gold_spot", "gold_futures"}, "dxy": {"ice_dxy"},
               "sp500": {"sp500_price_index"}, "vix": {"vix_close"},
               "y10": {"10y_constant_maturity_percent", "10y_par_yield_percent"}}


def validate_store(store):
    """Accept frozen observation versions with explicit availability evidence.

    The metadata is an auditable input contract, not independent certification
    of a vendor. Revision selection must happen upstream; silently keeping the
    last value in today's download would violate the contract.
    """
    missing = REQUIRED - set(store)
    if missing:
        raise ValueError("Macro store lacks historical availability evidence: " + ", ".join(sorted(missing)))
    f = store.copy()
    if f.empty or set(f["series"]) != set(DEFINITIONS):
        raise ValueError("Supply all five exact macro series, including gold spot/futures and ICE DXY")
    for c in REQUIRED - {"source_timestamp", "value"}:
        if f[c].isna().any() or f[c].astype(str).str.strip().eq("").any():
            raise ValueError("Missing macro provenance: " + c)
    f["date"] = pd.to_datetime(f["date"])
    if f["date"].dt.tz is not None or not f["date"].eq(f["date"].dt.normalize()).all():
        raise ValueError("Observation dates must be timezone-free calendar dates")
    for c in ("available_at", "retrieved_at"):
        if any(pd.Timestamp(v).tzinfo is None for v in f[c]):
            raise ValueError(c + " must contain explicit timezones")
        f[c] = pd.to_datetime(f[c], utc=True)
    exact = f["availability_basis"].ne("alfred_vintage_day_upper_bound")
    if (f.loc[exact, "retrieved_at"] < f.loc[exact, "available_at"]).any():
        raise ValueError("Retrieval predates the stated availability bound")
    if (f["available_at"] < f["date"].dt.tz_localize("UTC")).any():
        raise ValueError("Macro observation cannot be available before its observation date")
    if not f["raw_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Every observation needs a SHA-256 source hash")
    allowed = {"alfred_vintage_day_upper_bound", "archived_publication_timestamp", "archived_market_close"}
    if not f["availability_basis"].isin(allowed).all():
        raise ValueError("Current downloads or inferred close times do not prove historical availability")
    timestamps = pd.to_datetime(f["source_timestamp"], utc=True)
    if timestamps[exact].isna().any() or (timestamps[exact] > f.loc[exact, "available_at"]).any():
        raise ValueError("Archived sources require source timestamps no later than availability")
    if f.duplicated(["series", "date"]).any():
        raise ValueError("Select a frozen version per observation; duplicate revisions are not permitted")
    if not np.isfinite(f["value"].to_numpy(float)).all():
        raise ValueError("Nonfinite macro source values")
    for name, definitions in DEFINITIONS.items():
        part = f[f["series"] == name]
        if not part["series_definition"].isin(definitions).all():
            raise ValueError("Wrong series definition for " + name)
        if part["source_id"].nunique() != 1 or part["series_definition"].nunique() != 1:
            raise ValueError("Do not splice underlying series within " + name)
        if name != "y10" and (part["value"] <= 0).any():
            raise ValueError("Log-price sources must be strictly positive: " + name)
    return f.sort_values(["series", "date"]).reset_index(drop=True)


def attach_available(panel, signals, cols, observations, lag_days=1, stale_days=4):
    """Backward join using availability, with freshness measured from observation.

    Each signal may depend on its entire earlier history (EWMA/normalization),
    so the cumulative availability maximum is a conservative bound for all
    its dependencies. This deliberately sacrifices coverage for late releases.
    Crypto closes are complete only at the end of their UTC observation day.
    """
    if lag_days < 1 or stale_days < lag_days:
        raise ValueError("Invalid cutoff/staleness")
    metadata = observations.sort_values("date")[["date", "available_at"]].copy()
    metadata["available_at"] = pd.to_datetime(metadata["available_at"], utc=True).cummax()
    right = signals.merge(metadata, on="date", how="left", validate="m:1")
    if right["available_at"].isna().any():
        raise ValueError("Signal date has no source availability evidence")
    end_crypto_day = pd.to_datetime(right["date"], utc=True) + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    right["available_at"] = right["available_at"].where(right["available_at"] >= end_crypto_day, end_crypto_day)
    right = right.rename(columns={"date": "_observation_date"})
    right = right.sort_values(["available_at", "_observation_date"])
    left = panel[["symbol", "date"]].copy().assign(_row=np.arange(len(panel)))
    left["_cutoff"] = (pd.to_datetime(left["date"], utc=True) - pd.Timedelta(days=lag_days - 1)
                       - pd.Timedelta(1, "ns"))
    merged = pd.merge_asof(left.sort_values("_cutoff"), right[["symbol", "_observation_date", "available_at"] + cols],
                           left_on="_cutoff", right_on="available_at", by="symbol", direction="backward")
    age = (pd.to_datetime(merged["date"]) - pd.to_datetime(merged["_observation_date"])).dt.days
    merged.loc[age.gt(stale_days) | age.lt(lag_days) | age.isna(), cols] = np.nan
    return merged.sort_values("_row")[cols].set_axis(panel.index)


def covered_rows(frame, features):
    """The same complete-observation mask for candidate training and inference.

    A combined high AND low gate has no coverage by construction. Evaluate
    each gate separately; never neutral-impute an inactive gate into training.
    """
    if not features or any(not c.startswith("feature_") for c in features):
        raise ValueError("Model inputs must be an explicit nonempty feature list")
    return pd.Series(np.isfinite(frame[features].to_numpy(float)).all(axis=1), index=frame.index)
