"""
Data loading for the Numerai Classic ("numerai") and Crypto ("crypto") tournaments.

Everything that differs between the two tournaments lives in the TOURNAMENTS dict
below. The rest of the pipeline reads from that dict and never branches on the
tournament name again -- so adding a third tournament means adding a dict entry
and a loader branch, nothing more.

The main thing to understand here is the `split` column. Every frame returned by
load_tournament_data() carries a "split" column holding "train" or "validation":

  * Classic ships this natively -- train.parquet is train, and the rows of
    validation.parquet marked data_type == "validation" are validation.
  * Crypto ships no validation file at all, so we build the equivalent with a
    chronological cut plus an embargo gap. See make_validation_split().

The walk-forward warms up over the train block and only scores the validation
block, so this column is what keeps the reported metrics honest.
"""

import json
import math
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Tournament constants
# --------------------------------------------------------------------------- #
#
# period_days is how many market days one period covers. It is what makes the
# embargo portable: an era is a week (5 market days), a crypto date is 1 day, so
# the *same* 20-day target needs a 4-period embargo in Classic and a 20-period
# embargo in Crypto. See default_embargo().
#
TOURNAMENTS = {
    "numerai": {
        "id_col": "id",
        "period_col": "era",
        "period_days": 5,                       # one era == one week
        "target": "target",                     # alias for target_ender_20
        "version": "v5.3",
        "reference_col": "v53_lgbm_ender20",    # benchmark model -> BMC
        "feature_set": "small",                 # small | medium | quantum | all
        "neutral_value": 2.0,                   # middle of the 0-4 integer bins
        "meta_model_file": "meta_model.parquet",
        "meta_model_col": "numerai_meta_model",  # covers eras 1133-1218 only
    },
    "crypto": {
        "id_col": "symbol",
        "period_col": "date",
        "period_days": 1,                       # one date == one day
        "target": "target_binned_return_20",
        "version": "crypto/v2.0",
        "reference_col": "prediction",          # meta model -> true MMC
        "feature_set": None,                    # crypto has one fixed feature set
        "neutral_value": 0.5,                   # middle of the {0,.25,.5,.75,1} bins
        "meta_model_file": "historical_meta_models.parquet",
        "meta_model_col": "prediction",         # covers 566 of 1,700 dates
    },
}

# Where downloaded parquet files are cached.
DATA_DIR = "data"

# The three reference series a run can score against, and what each is for:
#
#   REFERENCE_COL   the benchmark BMC is computed against by default.
#                   Classic: the official v53_lgbm_ender20 benchmark model.
#                   Crypto:  the published meta model (there is no separate
#                            benchmark model), so it covers only ~33% of dates.
#
#   META_MODEL_COL  the real, published meta model. Used for MMC, which is only
#                   meaningful where it actually exists -- 86 eras on Classic,
#                   566 of 1,700 dates on Crypto.
#
#   META_PROXY_COL  a Ridge stand-in for the meta model, fit on features and
#                   predicted everywhere. This is what lets BMC be scored over
#                   the full history rather than only the slice the real meta
#                   model covers. See build_meta_proxy().
#
# Canonical name the reference series is renamed to once loaded.
#
# This is not cosmetic. Crypto's meta model column is called "prediction" in the
# shipped parquet -- exactly the name we give our own model's output. Joining it
# in under its original name silently overwrote our predictions with the meta
# model's, which scored a suspiciously good CORR and a BMC of precisely zero
# (a series contributes nothing over itself). Renaming on load makes that class
# of collision impossible for any tournament.
REFERENCE_COL = "reference_prediction"
META_MODEL_COL = "meta_model"
META_PROXY_COL = "meta_proxy"

# Below this many training periods, the meta proxy is warned about as weakly
# grounded. Not a hard failure -- it still fits -- but BMC against a reference
# built from a handful of periods is not saying much.
MIN_PROXY_PERIODS = 20


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def get_spec(tournament):
    """
    Look up the constants for a tournament.

    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :return: the tournament's constants
    :rtype: dict
    """
    if tournament not in TOURNAMENTS:
        raise ValueError(
            "unknown tournament '{}', expected one of {}".format(
                tournament, sorted(TOURNAMENTS)
            )
        )
    return TOURNAMENTS[tournament]


def get_api(tournament):
    """
    Return the numerapi client for a tournament.

    Classic and Crypto are served by different client classes, and the Crypto
    datasets do not appear in NumerAPI's dataset listing at all.

    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :return: a numerapi client
    """
    if tournament == "crypto":
        from numerapi import CryptoAPI
        return CryptoAPI()

    from numerapi import NumerAPI
    return NumerAPI()


def local_path(tournament, version, filename):
    """
    Local cache path for a dataset file, e.g. data/crypto/v2.0/train.parquet.

    The Crypto version string already contains a slash ("crypto/v2.0"), so we
    keep only its last component to avoid nesting a redundant directory.

    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :param version: dataset version string
    :type version: str
    :param filename: file name within the dataset
    :type filename: str
    :return: path on disk
    :rtype: str
    """
    short_version = version.split("/")[-1]
    return os.path.join(DATA_DIR, tournament, short_version, filename)


def download_if_missing(api, tournament, version, filename, verbose=True):
    """
    Download a dataset file unless it is already cached locally.

    :param api: numerapi client from get_api()
    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :param version: dataset version string
    :type version: str
    :param filename: file name within the dataset
    :type filename: str
    :param verbose: print a line when downloading
    :type verbose: bool
    :return: path to the local file
    :rtype: str
    """
    dest = local_path(tournament, version, filename)

    # makedirs, not mkdir: the parent chain may not exist, and exist_ok keeps
    # parallel runs from racing each other.
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if not os.path.exists(dest):
        if verbose:
            print("downloading {}/{} ...".format(version, filename))
        api.download_dataset("{}/{}".format(version, filename), dest_path=dest)

    return dest


def target_horizon_days(target):
    """
    Read the forward horizon, in market days, off the end of a target name.

    Numerai's naming convention -- target_<name>_<horizon> -- holds across
    Classic ("target_ender_20"), Crypto ("target_binned_return_20") and Signals
    ("target_chili_60"), so this one parser covers all of them. The bare Classic
    alias "target" is target_ender_20, hence the default of 20.

    :param target: target column name
    :type target: str
    :return: horizon in market days
    :rtype: int
    """
    tail = target.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return 20


def default_embargo(tournament, target):
    """
    Number of periods to embargo, derived from the target horizon.

    This is the fix for the single most portable-looking, least portable number
    in the pipeline. A 20-day target overlaps the next 3 Classic eras (weekly),
    but the next 19 Crypto dates (daily) -- so a hard-coded embargo silently
    leaks when moved between tournaments.

    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :param target: target column name
    :type target: str
    :return: embargo length in periods
    :rtype: int
    """
    spec = get_spec(tournament)
    return int(math.ceil(target_horizon_days(target) / spec["period_days"]))


# Market days in an average calendar month. Used to express a refit cadence in
# months rather than in periods, whose length differs per tournament.
MARKET_DAYS_PER_MONTH = 21.7


def default_refit_every(tournament, months=1):
    """
    Refit cadence in periods, given a cadence in months.

    "Refit monthly" is the natural way to describe the schedule, but the runner
    counts periods, and a period is not the same length in the two tournaments.
    Same conversion as default_embargo(), using the tournament's period_days:

        crypto   21.7 / 1 = 21.7  -> 22 dates   (matches the measured median of
                                                 22 dates per calendar month)
        numerai  21.7 / 5 = 4.3   -> 4 eras     (eras are weekly)

    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :param months: cadence in months
    :type months: float
    :return: refit cadence in periods, at least 1
    :rtype: int
    """
    spec = get_spec(tournament)
    periods = round(months * MARKET_DAYS_PER_MONTH / spec["period_days"])
    return max(1, int(periods))


def binarize_target(y):
    """
    Turn the [0, 1] binned target into an up/down label for --task classification.

    The Numerai target is binned to {0, 0.25, 0.5, 0.75, 1.0} with the middle bin
    holding half the mass, so the split at 0.5 puts the flat middle bin on the
    "down" side. That is a coarse framing of a graded target and is exactly why
    regression is the default; this exists so the two framings can be compared.

    :param y: target values
    :type y: pd.Series
    :return: 0/1 labels
    :rtype: pd.Series
    """
    return (y > 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_tournament_data(tournament, version=None, feature_set=None,
                         era_step=1, val_frac=0.3, embargo=None, verbose=True,
                         target=None):
    """
    Load one tournament's data, already carrying a train/validation `split` column.

    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :param version: dataset version, or None for the tournament default
    :type version: str or None
    :param feature_set: Classic feature set name, or None for the default
    :type feature_set: str or None
    :param era_step: keep every Nth period -- the cheap-scout knob for Classic
    :type era_step: int
    :param val_frac: Crypto only, fraction of periods held out for validation
    :type val_frac: float
    :param embargo: periods to drop between train and validation (Crypto only)
    :type embargo: int or None
    :param verbose: print progress
    :type verbose: bool
    :return: (frame, feature column names)
    :rtype: (pd.DataFrame, [str])
    """
    spec = get_spec(tournament)
    version = version or spec["version"]

    if tournament == "crypto":
        # Candidate-only datasets must not enter the generic neutral-fill path:
        # it would let unavailable all-neutral blocks contribute to fitting.
        manifest_path = local_path(tournament, version, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("missing_value_policy") == "coverage_required":
                raise ValueError(
                    "Dataset %s requires coverage-aware training using coverage.parquet; "
                    "load its train.parquet with NaNs preserved, fit candidates only on "
                    "eligible rows, and use matched controls and full-universe rank-slot routing. "
                    "The generic neutral-fill loader is disabled for this dataset." % version)
        df, features = _load_crypto(tournament, version, verbose)
        if embargo is None:
            embargo = default_embargo(tournament, spec["target"])
        # Crypto ships no validation file, so we build one.
        df = make_validation_split(df, spec["period_col"], val_frac, embargo)
    else:
        df, features = _load_classic(
            tournament, version, feature_set or spec["feature_set"], verbose
        )

    # Downsample periods for fast iteration. Applied after the split so the
    # train/validation boundary does not move when the step changes.
    if era_step and era_step > 1:
        df = _downsample_periods(df, spec["period_col"], era_step)

    # Rows without a target cannot be trained on or scored. This must drop on
    # the target BEING MODELLED, not the tournament default: crypto's 60-day
    # target is NaN over its last 40 dates (12,000 rows) where the window has
    # not resolved, while the 20-day target has none. Dropping on the default
    # would leave those NaNs in and hand them straight to the estimator.
    target_column = target or spec["target"]
    if target_column not in df.columns:
        raise ValueError("target %r is not in the dataset; available: %s"
                         % (target_column,
                            sorted(c for c in df.columns if c.startswith("target"))))
    before = len(df)
    df = df.dropna(subset=[target_column])
    if verbose and len(df) < before:
        print("dropped {:,} rows with no {} ({:.1%} of the panel)".format(
            before - len(df), target_column, (before - len(df)) / before))

    # Missing features get the neutral middle bin rather than being dropped.
    df = fill_missing_features(df, features, spec["neutral_value"], verbose)

    if verbose:
        n_val = int((df["split"] == "validation").sum())
        print("loaded {}: {:,} rows, {:,} features, {:,} periods "
              "({:,} validation rows)".format(
                  tournament, len(df), len(features),
                  df[spec["period_col"]].nunique(), n_val))

    return df.reset_index(drop=True), features


def _load_classic(tournament, version, feature_set, verbose):
    """
    Load Classic. The train/validation split is native to the dataset.

    train.parquet is the training block; validation.parquet holds both resolved
    validation rows and unresolved live-ish rows, so we keep only the rows
    labelled data_type == "validation".
    """
    api = get_api(tournament)
    spec = get_spec(tournament)

    # features.json lists the named feature sets (small / medium / quantum / all).
    features_path = download_if_missing(
        api, tournament, version, "features.json", verbose
    )
    with open(features_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    available = metadata["feature_sets"]
    if feature_set not in available:
        raise ValueError(
            "unknown feature_set '{}', expected one of {}".format(
                feature_set, sorted(available)
            )
        )
    features = available[feature_set]

    # Only read the columns we need. The full Classic parquet is ~2400 feature
    # columns wide, so restricting here is the difference between fitting in
    # memory and not.
    columns = [spec["period_col"], spec["target"]] + features

    train_path = download_if_missing(api, tournament, version, "train.parquet", verbose)
    train = pd.read_parquet(train_path, columns=columns)
    train["split"] = "train"

    valid_path = download_if_missing(
        api, tournament, version, "validation.parquet", verbose
    )
    valid = pd.read_parquet(valid_path, columns=columns + ["data_type"])
    valid = valid[valid["data_type"] == "validation"].drop(columns=["data_type"])
    valid["split"] = "validation"

    full = pd.concat([train, valid])

    # The id lives in the parquet index; promote it to a real column so the
    # reference join has something to match on.
    full.index.name = full.index.name or spec["id_col"]
    full = full.reset_index()

    return full, features


def _load_crypto(tournament, version, verbose):
    """
    Load Crypto. One file, and the features are found by prefix.

    Crypto has no features.json and no feature-set names -- there are 22 features
    and you use all of them.
    """
    api = get_api(tournament)
    spec = get_spec(tournament)

    train_path = download_if_missing(api, tournament, version, "train.parquet", verbose)
    full = pd.read_parquet(train_path)

    # symbol is the index in the shipped file; make it a column. A custom
    # dataset may already carry it as one, in which case there is nothing to
    # promote and reset_index would collide.
    if spec["id_col"] not in full.columns:
        full.index.name = full.index.name or spec["id_col"]
        full = full.reset_index()
    else:
        full = full.reset_index(drop=True)

    features = [c for c in full.columns if c.startswith("feature_")]
    full[spec["period_col"]] = pd.to_datetime(full[spec["period_col"]])

    return full, features


def make_validation_split(df, period_col, val_frac, embargo):
    """
    Build a Crypto validation block: a chronological tail, gapped by the embargo.

    Classic ships a validation split; Crypto does not, so we make the equivalent:

        [-------- train --------][ embargo ][---- validation ----]

    The embargo gap matters. Without it the last training rows carry a 20-day
    forward target whose window reaches into the validation block, which would
    leak the very thing we are trying to measure. Those periods are dropped
    outright rather than assigned to either side.

    The cut is strictly chronological and never shuffled.

    :param df: frame to split
    :type df: pd.DataFrame
    :param period_col: name of the period column
    :type period_col: str
    :param val_frac: fraction of periods to hold out
    :type val_frac: float
    :param embargo: periods to discard between the blocks
    :type embargo: int
    :return: frame with a "split" column, embargoed periods removed
    :rtype: pd.DataFrame
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be strictly between 0 and 1")

    periods = np.sort(df[period_col].unique())
    n_val = int(round(len(periods) * val_frac))
    if n_val < 1:
        raise ValueError("val_frac too small: no validation periods would remain")

    val_periods = set(periods[-n_val:])
    # Everything before the validation block, minus the embargo gap.
    train_periods = set(periods[: max(0, len(periods) - n_val - embargo)])

    if not train_periods:
        raise ValueError(
            "no training periods left after val_frac={} and embargo={}".format(
                val_frac, embargo
            )
        )

    out = df.copy()
    out["split"] = np.where(
        out[period_col].isin(val_periods), "validation",
        np.where(out[period_col].isin(train_periods), "train", "embargo"),
    )
    # Drop the embargoed periods entirely -- they belong to neither block.
    return out[out["split"] != "embargo"]


def fill_missing_features(df, features, neutral_value, verbose=True):
    """
    Fill missing feature values with the neutral middle bin.

    Numerai features are rank-binned, so the middle bin is literally the "no
    information" value -- half the mass sits there by construction. Classic
    encodes it as 2 on a 0-4 integer scale, Crypto as 0.5 on {0, .25, .5, .75, 1}.
    This is the same convention Numerai itself uses for missing Classic features.

    Why fill rather than drop: on Crypto about 3.3% of rows carry a NaN, and it
    is overwhelmingly concentrated in the 60-day indicators (momentum_60d,
    bollinger_60d) for tokens that simply have not existed for 60 days yet.
    Dropping those rows would quietly remove young tokens from the universe --
    a survivorship bias in exactly the segment where a model might find an edge.
    Filling says "this feature has nothing to tell us about this row", which is
    both true and harmless.

    It also matters mechanically: the boosters tolerate NaN natively but Ridge
    does not, so without this the linear models cannot run at all.

    :param df: frame containing the feature columns
    :type df: pd.DataFrame
    :param features: feature column names
    :type features: [str]
    :param neutral_value: the middle-bin value for this tournament
    :type neutral_value: float
    :param verbose: report how much was filled
    :type verbose: bool
    :return: frame with no missing feature values
    :rtype: pd.DataFrame
    """
    missing = df[features].isna()
    n_missing = int(missing.to_numpy().sum())

    if n_missing == 0:
        return df

    if verbose:
        affected_rows = float(missing.any(axis=1).mean())
        print("filling {:,} missing feature values ({:.2%} of rows affected) "
              "with the neutral value {}".format(
                  n_missing, affected_rows, neutral_value))

    out = df.copy()
    out[features] = out[features].fillna(neutral_value)
    return out


def _downsample_periods(df, period_col, era_step):
    """Keep every Nth period, preserving chronological order."""
    periods = np.sort(df[period_col].unique())
    keep = set(periods[::era_step])
    return df[df[period_col].isin(keep)]


# --------------------------------------------------------------------------- #
# Reference predictions (for BMC / MMC)
# --------------------------------------------------------------------------- #

def load_reference(tournament, version=None, verbose=True):
    """
    Load the reference prediction series that BMC is computed against.

    The two tournaments use structurally different references:

      Classic -- the official benchmark model `v53_lgbm_ender20`. Numerai's own
                 meta model only covers ~86 recent eras, far too few to score a
                 full history against, which is exactly why BMC exists as a
                 stand-in for MMC.

      Crypto  -- the published meta model, so this is *true* MMC rather than a
                 proxy. It only covers the most recent third of history though,
                 so check the coverage that attach_reference() reports.

    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :param version: dataset version, or None for the tournament default
    :type version: str or None
    :param verbose: print progress
    :type verbose: bool
    The column is always renamed to REFERENCE_COL on the way out, so the caller
    never has to care what the source file called it -- and so a source name can
    never collide with our own "prediction" column. See REFERENCE_COL.

    :return: (frame with id + reference column, REFERENCE_COL)
    :rtype: (pd.DataFrame, str)
    """
    spec = get_spec(tournament)
    version = version or spec["version"]
    api = get_api(tournament)
    id_col = spec["id_col"]
    source_col = spec["reference_col"]

    if tournament == "crypto":
        path = download_if_missing(
            api, tournament, version, "historical_meta_models.parquet", verbose
        )
        reference = pd.read_parquet(path)
        reference = reference[[id_col, "date", source_col]].copy()
        reference["date"] = pd.to_datetime(reference["date"])
        # Crypto keys on (symbol, date) -- a symbol appears once per date.
        reference = reference.rename(columns={source_col: REFERENCE_COL})
        return reference, REFERENCE_COL

    # Classic benchmark models arrive split the same way the data does.
    frames = []
    for filename in ("train_benchmark_models.parquet",
                     "validation_benchmark_models.parquet"):
        path = download_if_missing(api, tournament, version, filename, verbose)
        part = pd.read_parquet(path, columns=[source_col])
        frames.append(part)

    reference = pd.concat(frames)
    reference.index.name = reference.index.name or id_col
    reference = reference.reset_index()[[id_col, source_col]]
    reference = reference.rename(columns={source_col: REFERENCE_COL})
    return reference, REFERENCE_COL


def load_meta_model(tournament, version=None, verbose=True):
    """
    Load the real, published meta model. This is what MMC is scored against.

    Distinct from load_reference(): the meta model is the stake-weighted
    aggregate of everyone's live submissions, so contributing over it is what
    Numerai actually pays for. The catch is coverage -- it only exists for
    rounds that have resolved, which is 86 eras on Classic (1133-1218 of 1198
    available) and 566 of 1,700 dates on Crypto. That thin coverage is precisely
    why BMC exists as a stand-in, and why build_meta_proxy() is useful.

    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :param version: dataset version, or None for the tournament default
    :type version: str or None
    :param verbose: print progress
    :type verbose: bool
    :return: (frame keyed for joining, META_MODEL_COL)
    :rtype: (pd.DataFrame, str)
    """
    spec = get_spec(tournament)
    version = version or spec["version"]
    api = get_api(tournament)
    id_col = spec["id_col"]
    source_col = spec["meta_model_col"]

    path = download_if_missing(
        api, tournament, version, spec["meta_model_file"], verbose
    )

    if tournament == "crypto":
        meta = pd.read_parquet(path)[[id_col, "date", source_col]].copy()
        meta["date"] = pd.to_datetime(meta["date"])
    else:
        meta = pd.read_parquet(path, columns=[source_col])
        meta.index.name = meta.index.name or id_col
        meta = meta.reset_index()[[id_col, source_col]]

    return meta.rename(columns={source_col: META_MODEL_COL}), META_MODEL_COL


def build_meta_proxy(df, features, period_col, train_mask,
                     meta_col=META_MODEL_COL, alpha=1.0, verbose=True):
    """
    Fit a Ridge stand-in for the meta model, so BMC can be scored everywhere.

    The problem this solves: the real meta model covers only a slice of history,
    so BMC against it is restricted to that slice. A model trained on the
    features to reproduce the meta model's *ranking* can then be predicted over
    every period, giving a full-coverage reference.

    The target is the meta model's per-period percentile rank, not its raw
    value. That is the right choice for two reasons: Numerai scores by rank
    anyway, and ranking within a period removes any period-level drift in the
    meta model's scale, so the proxy learns cross-sectional shape rather than
    level.

    Leakage: the proxy is fit on `train_mask` rows only. It never sees a
    validation period, so it cannot smuggle future information into the
    reference. Note that a reference which saw the future would make BMC harder
    to earn, not easier -- but fitting on train keeps the whole pipeline honest
    under one rule rather than two.

    :param df: full frame, with features and the meta model column attached
    :type df: pd.DataFrame
    :param features: feature column names
    :type features: [str]
    :param period_col: name of the period column
    :type period_col: str
    :param train_mask: rows the proxy is allowed to be fit on
    :type train_mask: pd.Series of bool
    :param meta_col: name of the real meta model column
    :type meta_col: str
    :param alpha: Ridge penalty
    :type alpha: float
    :param verbose: report what it was fit on
    :type verbose: bool
    :return: (frame with META_PROXY_COL added, number of periods fit on)
    :rtype: (pd.DataFrame, int)
    """
    from sklearn.linear_model import Ridge

    # Only train rows that actually have a meta model value to learn from.
    fit_mask = train_mask & df[meta_col].notna()
    n_fit_periods = int(df.loc[fit_mask, period_col].nunique())

    if n_fit_periods == 0:
        raise ValueError(
            "cannot build a meta-model proxy: no training rows have a meta "
            "model value. On Classic the meta model only covers recent eras, "
            "which may all fall inside the validation block."
        )

    fit_rows = df.loc[fit_mask]
    # Per-period percentile rank of the meta model -- the ranking, not the level.
    y = fit_rows.groupby(period_col)[meta_col].rank(pct=True)

    proxy = Ridge(alpha=alpha).fit(fit_rows[features], y)

    out = df.copy()
    out[META_PROXY_COL] = proxy.predict(df[features])

    if verbose:
        print("meta proxy: Ridge(alpha={}) fit on {:,} rows across {} training "
              "periods, predicted over all {} periods".format(
                  alpha, int(fit_mask.sum()), n_fit_periods,
                  df[period_col].nunique()))

        # The proxy is only as good as what it was fit on, and there is a
        # structural squeeze here: the meta model tends to start late in the
        # sample, while the validation block is the tail of it, so the overlap
        # of "in training" and "has a meta model" can be very small. On Crypto
        # the meta model begins 2024-05 and a 30% validation split starts only
        # months later. Downsampling with --era-step shrinks it further.
        if n_fit_periods < MIN_PROXY_PERIODS:
            print("WARNING: the proxy saw only {} training periods (< {}). "
                  "BMC against it is weakly grounded -- lower --era-step or "
                  "--val-frac to give it more history, or use "
                  "--bmc-reference benchmark instead.".format(
                      n_fit_periods, MIN_PROXY_PERIODS))

    return out, n_fit_periods


def attach_meta_model(df, meta, tournament):
    """
    Join the real meta model on, returning its row coverage.

    :param df: tournament data
    :type df: pd.DataFrame
    :param meta: frame from load_meta_model()
    :type meta: pd.DataFrame
    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :return: (joined frame, fraction of rows with a meta model value)
    :rtype: (pd.DataFrame, float)
    """
    spec = get_spec(tournament)
    id_col = spec["id_col"]

    if tournament == "crypto":
        merged = df.merge(meta, on=[id_col, "date"], how="left")
    else:
        merged = df.merge(meta, on=id_col, how="left")

    coverage = float(merged[META_MODEL_COL].notna().mean()) if len(merged) else 0.0
    return merged, coverage


def attach_reference(df, reference, tournament, reference_col):
    """
    Left-join the reference onto the data and report how much of it is covered.

    Coverage is not a detail. Neither reference spans its tournament's full
    history -- Classic's benchmark models start at era 0158 of 1198 (~87%), and
    Crypto's meta model covers 566 of 1700 dates (~33%). Rows without a reference
    get NaN and are dropped from BMC but still counted for CORR, so two runs can
    otherwise look comparable while being scored over very different windows.
    The caller writes this number into the metrics file.

    :param df: tournament data
    :type df: pd.DataFrame
    :param reference: frame from load_reference()
    :type reference: pd.DataFrame
    :param tournament: "numerai" or "crypto"
    :type tournament: str
    :param reference_col: name of the reference column
    :type reference_col: str
    :return: (joined frame, fraction of rows with a reference)
    :rtype: (pd.DataFrame, float)
    """
    spec = get_spec(tournament)
    id_col = spec["id_col"]

    if tournament == "crypto":
        # A crypto symbol is only unique within a date, so join on both.
        merged = df.merge(reference, on=[id_col, "date"], how="left")
    else:
        merged = df.merge(reference, on=id_col, how="left")

    coverage = float(merged[reference_col].notna().mean()) if len(merged) else 0.0
    return merged, coverage
