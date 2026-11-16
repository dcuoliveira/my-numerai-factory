"""
Build a variant that NEUTRALISES features on any row where a feature WE created
from scratch is missing -- so the model abstains (predicts ~neutral) on
assets/timesteps for which we do not have the full engineered information.

    python3 -m src.data_mani.build_mommask   # v2.0-litmom -> v2.0-litmom-mommask
    python3 -m src.data_mani.build_mommask --source-version crypto/v2.0-momswap-macro-carver

Run it with:  --data-version crypto/<source>-mommask

------------------------------------------------------------------------------
Two masking modes
------------------------------------------------------------------------------

--mode gate (default, the -mommask family): ROW-level. If a row is missing ANY
gate feature, the WHOLE row is set to 0.5 -- present values included -- so the row
carries no signal and the model abstains on it.

--mode fill (the -fillmask variant): CELL-level, and much gentler. We ONLY replace
the values that are actually MISSING with 0.5, and KEEP every present value. What
counts as missing is any NaN already in the panel -- which covers both a value we
never had (a long window a young token cannot fill yet) AND every signal of a
symbol we do not have all signals for (a no-spot coin whose engineered block is
entirely NaN). Present signals on a partially-covered row are left untouched, so
the model still uses what we do know for that asset/timestep. NOTE this is what
Numerai's loader (fill_missing_features) already does at load time; baking it into
the dataset makes the neutral fill explicit and independent of the loader (e.g. for
the live path), and gives a clean contrast to the aggressive --mode gate.

------------------------------------------------------------------------------
The rule (gates on ALL our engineered groups, not just momentum)
------------------------------------------------------------------------------

"Created by us from scratch" = any feature we engineered, identified by prefix:
feature_mom_ (litmom momentum), feature_macro_ (macro), feature_ix_ (interactions),
feature_tf_ (Carver/Clenow trend), feature_vol_ (realised-volatility estimators),
feature_coint_ (cointegration / stat-arb residuals). The prefixes deliberately
EXCLUDE Numerai's shipped features (feature_momentum_, feature_bollinger_,
feature_volatility_, feature_volume_, ...).

For every row (symbol, date), if any of our created features is missing, we apply
the mask. Knobs:

  --gate created (default) : gate on ALL our-created features (all groups).
                             --gate momentum restores the old momentum-only gate.
  --require all  (default) : a row is gated unless ALL gate features are present
                             (strict "complete information"). --require none gates
                             only when they are ALL absent.
  --neutralize all (default): on a gated row set EVERY feature to 0.5 -- a fully
                             neutral row, the cleanest push to a neutral prediction.
                             --neutralize shipped touches only the Numerai base;
                             --neutralize nongate touches everything except the
                             gate group.
  --fill-value 0.5 (default): the neutral bin. 0.5 pushes toward a NEUTRAL (no-bet)
                             prediction; a literal 0.0 is the bottom bin, which
                             under a positive-weight model tilts toward a SHORT.

------------------------------------------------------------------------------
What this does and does not guarantee
------------------------------------------------------------------------------

A fully-neutralised row carries no signal, so its prediction converges to the
model's neutral response; identical neutral rows tie and cluster at the middle
rank (abstain). For this to hold live as well as in backtest, the SAME masking
must be applied to the live features. It is a soft push, not a hard guarantee: a
hard "prediction == neutral" is cleaner as a POST-prediction override on the same
gated rows. A gated row keeps its real forward-return target, so training still
sees the row -- the mask removes its features, not the row.
"""

import argparse
import os
import shutil

import pandas as pd

from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

NEUTRAL = 0.5

# Prefixes of the features WE engineered from scratch. Distinctive enough to
# exclude Numerai's shipped ones: "feature_momentum_" does not start with
# "feature_mom_" (index 11 is 'e' vs '_'), "feature_market_cap_" does not start with
# "feature_macro_", and "feature_volatility_"/"feature_volume_" do not start with
# "feature_vol_" (the char after "feature_vol" is 'a'/'u', not '_').
CREATED_PREFIXES = ("feature_mom_", "feature_macro_", "feature_ix_", "feature_tf_",
                    "feature_vol_", "feature_coint_", "feature_basis_",
                    "feature_funding_")


def is_created(col):
    return any(col.startswith(p) for p in CREATED_PREFIXES)


def apply_mommask(df, gate_cols, neutralize_cols, require="all", fill_value=NEUTRAL):
    """
    Neutralise ``neutralize_cols`` on the gated rows and return (masked, gated).

    :param df: the panel
    :param gate_cols: the feature columns whose presence is the gate
    :param neutralize_cols: the columns to set to ``fill_value`` on gated rows
    :param require: "all" gates a row unless every gate feature is present;
        "none" gates only when they are all absent
    :param fill_value: the value written on gated rows (0.5 neutral)
    :return: (a copy of df with the mask applied, the boolean gated mask)
    :rtype: (pd.DataFrame, pd.Series)
    """
    present = df[gate_cols].notna()
    if require == "all":
        gated = ~present.all(axis=1)
    elif require == "none":
        gated = ~present.any(axis=1)
    else:
        raise ValueError("require must be 'all' or 'none', got %r" % require)
    out = df.copy()
    out.loc[gated, neutralize_cols] = fill_value
    return out, gated


def apply_fillmask(df, cols, fill_value=NEUTRAL):
    """
    Cell-level mask: replace ONLY the missing (NaN) values in ``cols`` with
    ``fill_value``, keeping every present value. Returns a copy.

    Unlike ``apply_mommask`` this never touches a value we actually have -- a row
    that is only partially covered keeps its present signals and gets the neutral
    value only in the holes. A symbol we lack all signals for is all-NaN in the
    engineered block and so is filled to neutral there, while its shipped features
    (which we always have) stay as they are.

    :param df: the panel
    :param cols: the feature columns to fill
    :param fill_value: the value written into the missing cells (0.5 neutral)
    :return: a copy of df with the missing cells in ``cols`` filled
    :rtype: pd.DataFrame
    """
    out = df.copy()
    out[cols] = out[cols].fillna(fill_value)
    return out


def indicator_name(col):
    """feature_mom_ret_7d -> feature_isna_mom_ret_7d. A distinct prefix, so the
    indicators never match a block regex (^feature_mom_) or CREATED_PREFIXES."""
    return "feature_isna_" + col[len("feature_"):]


def apply_fillind(df, cols, fill_value=NEUTRAL):
    """
    ``apply_fillmask`` plus a 0/1 missingness indicator next to every column in
    ``cols`` that has at least one missing value (1.0 = the value was missing and
    has been filled). A column with no NaN gets no indicator -- it would be a
    constant. The filled value alone cannot tell a real middle-bin 0.5 from a
    filled hole; the indicator can.

    :param df: the panel
    :param cols: the feature columns to fill
    :param fill_value: the value written into the missing cells (0.5 neutral)
    :return: (a copy of df, filled and with the indicators appended, the list of
        indicator column names)
    :rtype: (pd.DataFrame, list)
    """
    miss = df[cols].isna()
    holed = [c for c in cols if miss[c].any()]
    ind = miss[holed].astype("float32")
    ind.columns = [indicator_name(c) for c in holed]
    out = pd.concat([apply_fillmask(df, cols, fill_value=fill_value), ind], axis=1)
    return out, list(ind.columns)


def parse_args():
    p = argparse.ArgumentParser(
        description="Neutralise features where an engineered feature is missing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0-litmom")
    p.add_argument("--out-suffix", default=None,
                   help="output dir suffix; defaults to -mommask (gate mode) or "
                        "-fillmask (fill mode)")
    p.add_argument("--mode", choices=["gate", "fill", "fillind"], default="gate",
                   help="'gate' (row-level): neutralise the WHOLE row where a gate "
                        "feature is missing. 'fill' (cell-level): replace ONLY the "
                        "missing values with the fill value, keeping present ones. "
                        "'fillind': fill, plus a 0/1 feature_isna_<name> indicator "
                        "for every feature that has any missing value")
    p.add_argument("--gate", choices=["created", "momentum"], default="created",
                   help="'created' gates on all our engineered features "
                        "(feature_{mom,macro,ix,tf}_); 'momentum' only feature_mom_")
    p.add_argument("--require", choices=["all", "none"], default="all",
                   help="'all' = gate unless every gate feature is present; "
                        "'none' = gate only when all are absent")
    p.add_argument("--neutralize", choices=["all", "shipped", "nongate"],
                   default="all",
                   help="which features to neutralise on gated rows: 'all' every "
                        "feature; 'shipped' only Numerai base; 'nongate' all but "
                        "the gate group")
    p.add_argument("--fill-value", type=float, default=NEUTRAL,
                   help="value written into the neutralised features on gated rows")
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    feats = [c for c in base.columns if c.startswith("feature_")]
    created = [c for c in feats if is_created(c)]
    shipped = [c for c in feats if not is_created(c)]
    if not created:
        raise SystemExit("no engineered (feature_{mom,macro,ix,tf}_) columns in %s"
                         % args.source_version)

    # per-group breakdown for transparency
    groups = {"mom": sum(c.startswith("feature_mom_") for c in created),
              "macro": sum(c.startswith("feature_macro_") for c in created),
              "ix": sum(c.startswith("feature_ix_") for c in created),
              "tf": sum(c.startswith("feature_tf_") for c in created),
              "vol": sum(c.startswith("feature_vol_") for c in created),
              "coint": sum(c.startswith("feature_coint_") for c in created),
              "basis": sum(c.startswith("feature_basis_") for c in created),
              "funding": sum(c.startswith("feature_funding_") for c in created)}
    print("source: %d rows, %d features (created=%d %s, shipped=%d) (%s)"
          % (len(base), len(feats), len(created),
             {k: v for k, v in groups.items() if v}, len(shipped), args.source_version))

    if args.mode in ("fill", "fillind"):
        # CELL-level: replace only the missing values (any NaN we already have --
        # young-token holes AND the all-NaN engineered block of no-spot symbols)
        # with the neutral value; every present value is kept.
        n_missing = int(base[feats].isna().sum().sum())
        rows_aff = int(base[feats].isna().any(axis=1).sum())
        sym_aff = int(base.loc[base[feats].isna().any(axis=1), "symbol"].nunique())
        if args.mode == "fill":
            masked = apply_fillmask(base, feats, fill_value=args.fill_value)
            out_suffix = args.out_suffix or "-fillmask"
        else:
            masked, inds = apply_fillind(base, feats, fill_value=args.fill_value)
            out_suffix = args.out_suffix or "-fillind"
            print("mode=fillind: %d missingness indicators (feature_isna_*), one per "
                  "feature with any missing value" % len(inds))
            feats = feats + inds
        print("mode=fill: filled %d missing feature values (%.2f%% of cells) on "
              "%d/%d rows (%.1f%%), %d symbols, with %s; present values kept"
              % (n_missing, 100 * n_missing / (len(base) * len(feats)),
                 rows_aff, len(base), 100 * rows_aff / len(base), sym_aff,
                 args.fill_value))
    else:
        gate_cols = created if args.gate == "created" else [c for c in feats if c.startswith("feature_mom_")]
        if args.neutralize == "all":
            neutralize_cols = feats
        elif args.neutralize == "shipped":
            neutralize_cols = shipped
        else:  # nongate
            neutralize_cols = [c for c in feats if c not in set(gate_cols)]

        masked, gated = apply_mommask(base, gate_cols, neutralize_cols,
                                      require=args.require, fill_value=args.fill_value)
        out_suffix = args.out_suffix or "-mommask"
        n_gated = int(gated.sum())
        n_sym_gated = int(masked.loc[gated, "symbol"].nunique())
        print("gate=%s(%d feats) require=%s neutralize=%s(%d feats) -> %d/%d rows gated "
              "(%.1f%%), %d symbols; set to %s on those rows"
              % (args.gate, len(gate_cols), args.require, args.neutralize,
                 len(neutralize_cols), n_gated, len(masked), 100 * n_gated / len(masked),
                 n_sym_gated, args.fill_value))

    version = args.source_version + out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    masked.set_index("symbol").to_parquet(dest)
    print("wrote %d features -> %s (%.0f MB)"
          % (len(feats), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
