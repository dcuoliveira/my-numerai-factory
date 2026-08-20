"""
Build a Crypto dataset with extra features learned by a bottleneck autoencoder.

    python3 -m src.data_mani.build_autoencoder --code-dim 8
    python3 -m src.data_mani.build_autoencoder --method pca --code-dim 8

Writes data/crypto/<version>-ae/train.parquet, used by passing

    --data-version crypto/v2.0-ae

Every other derived block in this repo is a HAND-SPECIFIED transform: the one-hot
is identity, tenc is past target performance, the rolling means are recent
standing, the spreads are horizon differences, the interactions are named
products. Each of them encodes a hypothesis somebody wrote down. None of them ask
the data what structure is already sitting in the 22 raw indicators.

That is the gap this fills. close_avg_20d and close_ewa_20d correlate about 0.98,
and the seven raw groups are plainly not 22 independent things. An autoencoder
squeezes all 22 through a k-dimensional bottleneck and back; the k bottleneck
activations become new feature_* columns -- a learned, nonlinear summary rather
than a specified one.

This is a DATASET, not a pipeline. Nothing in src/prediction/ or src/run_*.py
needs to know these columns came from a neural network, which is the whole point
of the startswith("feature_") convention.

------------------------------------------------------------------------------
The PCA control is not optional
------------------------------------------------------------------------------

With only 22 inputs there is not much to compress, and this repo's record is that
most added blocks do not beat the raw 22 by a meaningful margin. So --method pca
builds the same dataset through a linear projection, with identical column names,
and the autoencoder build ALWAYS reports PCA's held-out reconstruction at the
same k alongside its own. Without that number a positive result is
uninterpretable.

Read the R^2 comparison carefully, though, because with --swap-noise on it is not
a like-for-like fight: the autoencoder is TRAINED on corrupted rows and SCORED on
clean ones, while PCA is fitted directly on the clean rows it is then scored on.
Denoising trades reconstruction for a code that does not lean on any single
input. Rerun with --swap-noise 0 for the honest reconstruction comparison.

Measured on the real 22-feature panel at k=8, and worth reading as a caution
about defaults rather than about autoencoders:

    PCA control                       0.932   (linear ceiling, scree: 0.927)
    autoencoder at the first guess    0.904   64,32   -- loses to PCA
    autoencoder after 20 trials       0.954   256,128 -- beats it by 0.022

The first number stood for one afternoon and said "the nonlinearity did no work,
ship PCA and drop torch". It was wrong, and it was wrong because the network was
four times too narrow, not because the panel is linear. An untuned baseline is
evidence about the baseline.

And note what the R^2 does NOT answer. Reconstruction fidelity is not usefulness.
Whether the CODES help is settled downstream, by ae_plus_raw against
pca_plus_raw in experiments/autoencoder_crypto.sh, on a paired per-period delta.

------------------------------------------------------------------------------
Avoiding lookahead
------------------------------------------------------------------------------

The encoder is fitted ONCE, on the rows before the train/validation boundary,
and then applied to every row. This is exactly build_meta_proxy's regime: a model
fitted on train rows only, predicted everywhere so that scoring gets full
coverage. The boundary sits at or before the first walk-forward block's start
minus the embargo, so relative to every PREDICTION date the encoder is strictly
causal.

No target is involved, so no resolution lag applies and the fit window may run
right up to the boundary. That is the feature rule; the target rule is different
and stricter (see build_target_encoding). Conflating them is how people talk
themselves into leaks in both directions.

--fit-scope all fits on the whole panel. That is LOOKAHEAD -- not target leakage,
no target is touched, but at prediction time you would not know the future
feature values a full-sample encoder was shaped by, and 62% of tokens do not
survive three years, so a full-sample basis is partly a statement about which
tokens survived. Build it as an UPPER BOUND, not a candidate, exactly as
build_feature_encoding --mode global is. If causal and global score the same, the
causal encoding carries no hidden advantage and can be trusted.

THE ONE TRAP. The boundary depends on --val-frac, which is a RUN-time flag.
Building with --val-frac 0.3 and then running with --val-frac 0.5 puts the run's
validation block before the encoder's fit boundary: silent lookahead, no error.
The defaults here match the pipeline's, encoder.json records the boundary as an
absolute date, and experiments/autoencoder_crypto.sh refuses to run on a
mismatch. Do not remove that gate.

------------------------------------------------------------------------------
Scale convention
------------------------------------------------------------------------------

Bottleneck activations are unbounded, and every derived column in this repo has
to land on [0, 1] with 0.5 neutral, because the pipeline fills missing features
with that value and 0.5 must mean "no view" (see build_derived).

Each code is therefore rank-normalised WITHIN ITS DATE, using the same form as
src/live/submit.py:rank_normalise -- (rank - 0.5) / n -- which is exactly centred
on 0.5 and never emits a hard 0 or 1.

Two alternatives were rejected. A sigmoid bottleneck bounds the code natively but
buys a modelling constraint for a formatting reason, and would redefine 0.5 as
"pre-activation zero", i.e. wherever the bias landed, rather than "absence of
information". A train-fitted min-max or quantile transform is the only option
that preserves cross-date LEVEL, but it adds a second persisted, fittable,
staleable artifact whose live values drift outside the fitted range and get
clipped -- and the largest source of live/backtest skew is having two things that
can disagree.

Losing cross-date level costs nothing here: all 22 inputs are already within-date
rank bins and the target is rank-binned within date too, so a code built from
within-date ranks has no cross-date level to preserve. And the transform is
stateless in time -- it needs one date's ~300 rows and nothing else -- which puts
these codes in add_derived's class rather than add_rolling's when the live path
eventually wants them.

------------------------------------------------------------------------------
A note for whoever puts these into the allfeat chain
------------------------------------------------------------------------------

The column names feature_ae_00.. deliberately match NONE of the standardised
block regexes in experiments/ablation_crypto.sh -- in particular the prefix
carries no bare "x", which _x_|_accel_ would catch. But RE_DERIVED is the union
of exactly those five patterns, and alone_raw / lobo_raw are defined against it.
So the day an AE block enters the allfeat chain, ablation_crypto.sh needs
RE_AE='^feature_ae_[0-9]+$' added to RE_DERIVED and to the lobo_raw lookahead, or
the "raw 22" block silently becomes 22+k.
"""

import argparse
import datetime
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

from src.data_mani.utils import (
    default_embargo,
    download_if_missing,
    fill_missing_features,
    get_api,
    get_spec,
    local_path,
)

#: Below this a code is constant, and ranking a constant orders floating-point
#: noise. Mirrors autoencoder.DEAD_CODE_STD, duplicated so the PCA path -- which
#: must run without torch -- can check it too.
DEAD_CODE_STD = 1e-3

#: Two codes above this correlation are one code and a wasted column.
DUPLICATE_CODE_CORR = 0.99

#: How many components the scree report covers.
SCREE_MAX = 12


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Add bottleneck-autoencoder codes as extra features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default=None,
                        help="defaults to -ae, plus 'pca'/'global' for the "
                             "control and the lookahead upper bound")
    parser.add_argument("--method", default="ae", choices=["ae", "pca"],
                        help="ae needs torch; pca is the linear control and "
                             "needs only scikit-learn")
    parser.add_argument("--encoder-from", default=None, metavar="VERSION",
                        help="apply the encoder already built for VERSION to "
                             "--source-version instead of fitting a new one. "
                             "The codes then mean the same thing across every "
                             "dataset they are attached to, which is what makes "
                             "arms on different feature sets comparable")
    parser.add_argument("--code-dim", type=int, default=8,
                        help="bottleneck width, i.e. how many columns are added")

    # ---- what the encoder is allowed to see ----
    parser.add_argument("--fit-scope", default="train", choices=["train", "all"],
                        help="train fits before the validation boundary; all is "
                             "LOOKAHEAD, built as an upper bound only")
    parser.add_argument("--val-frac", type=float, default=0.3,
                        help="must match the --val-frac the runs will use")
    parser.add_argument("--embargo", default="auto",
                        help="periods between train and validation, or auto")
    parser.add_argument("--stop-frac", type=float, default=0.15,
                        help="chronological tail of the fit window held out for "
                             "early stopping and for the reported R^2")

    # ---- network and optimisation (ignored by --method pca) ----
    parser.add_argument("--hidden", default="256,128",
                        help="comma-separated encoder widths; the decoder "
                             "mirrors them. Searched, not guessed -- see "
                             "tune_autoencoder")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="between the hidden layers, never on the code")
    parser.add_argument("--swap-noise", type=float, default=0.15,
                        help="per-cell probability of swapping in another row's "
                             "value from the same column")
    parser.add_argument("--lr", type=float, default=2.8e-3,
                        help="Adam learning rate; there is no scheduler. Chosen "
                             "jointly with --hidden and --batch-size, so moving "
                             "one of the three alone leaves the tuned point")
    # 200 was not enough on the real 22-feature panel: the held-out MSE was
    # still falling at the cap, patience never ran out, and the truncated
    # network lost to its own PCA control (R^2 0.901 vs 0.932). Early stopping
    # is what should end a run, not the ceiling.
    parser.add_argument("--epochs", type=int, default=600,
                        help="ceiling only -- early stopping should end the "
                             "run, and the builder warns if this cap is what "
                             "did")
    parser.add_argument("--patience", type=int, default=15,
                        help="epochs without a held-out improvement before "
                             "stopping")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="much larger than the LSTM's, because rows here "
                             "are independent and cheap")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="gradient clipping threshold; 0 disables it")
    parser.add_argument("--device", default=None,
                        help="torch device, or auto-detect")

    # ---- run ----
    parser.add_argument("--column-prefix", default="feature_ae",
                        help="emitted columns are <prefix>_00 .. zero-padded; "
                             "must not match any ablation block regex")
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds the weights, the shuffle and the swap noise")
    parser.add_argument("--allow-dead-codes", action="store_true",
                        help="write the dataset even if a code is constant or "
                             "duplicated; for diagnosis, not for an arm")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="hide the per-epoch training lines")
    return parser.parse_args(argv)


def load_backend(method):
    """
    Import the encoder backend, failing with the fix rather than a traceback.

    Deferred deliberately. torch is optional (setup.sh, OPTIONAL set) and
    src/live/features.py imports data_mani builders at module scope, so a
    top-level torch import here would take the live submission path down on a box
    where the wheel failed to install.

    :param method: "ae" or "pca"
    :type method: str
    :return: the autoencoder module, or None for the PCA control
    :rtype: module or None
    """
    if method == "pca":
        return None
    try:
        from src.prediction.nn import autoencoder
    except ImportError as error:
        raise SystemExit(
            "--method ae needs torch, which is an optional dependency (%s).\n"
            "Install it with:  bash setup.sh   (torch is in the OPTIONAL set)\n"
            "Or build the linear control with --method pca, which needs only "
            "scikit-learn." % error)
    return autoencoder


def default_suffix(method, fit_scope):
    """Version suffix implied by the arms, so two arms cannot overwrite each other."""
    return "-ae%s%s" % ("pca" if method == "pca" else "",
                        "global" if fit_scope == "all" else "")


def code_columns(prefix, code_dim):
    """
    Names for the emitted columns, zero-padded.

    Padding is not cosmetic: live/features.py:raw_features sorts its columns
    lexically, so an unpadded feature_ae_10 would sort before feature_ae_2 and a
    re-derived column order would not match the one the encoder was fitted on --
    which, for a dense first layer, is silently wrong rather than loud.

    :param prefix: column prefix, e.g. "feature_ae"
    :type prefix: str
    :param code_dim: number of columns
    :type code_dim: int
    :return: names, in code order
    :rtype: [str]
    """
    return ["%s_%02d" % (prefix, index) for index in range(int(code_dim))]


def align_features(df, feature_columns):
    """
    Take the feature matrix in the exact order the encoder was fitted on.

    This is the sharpest live/backtest trap in the design and the reason it gets
    its own function. The encoder's first layer is a dense matmul: permute its
    input and every code is wrong, with no NaN, no exception and a perfectly
    plausible distribution. And the two sides of this repo ALREADY disagree --
    build_derived takes feature columns in parquet order, while
    live/features.py:raw_features returns them sorted. Every other builder is
    order-free, so it has never mattered before. Here it is fatal.

    :param df: frame holding the features
    :type df: pd.DataFrame
    :param feature_columns: the fitted order
    :type feature_columns: [str]
    :return: (n, f) float32 matrix in that order
    :rtype: np.ndarray
    :raises ValueError: naming any column that is missing
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "the frame is missing {} of the {} columns this encoder was fitted "
            "on: {}".format(len(missing), len(feature_columns),
                            ", ".join(missing[:8])))

    return df[list(feature_columns)].to_numpy(dtype="float32")


def fit_boundary(periods, val_frac, embargo, fit_scope):
    """
    Index of the first period the encoder may NOT see.

    Same arithmetic as make_validation_split, so the encoder's fit window is the
    pipeline's train block and nothing else.

    :param periods: sorted unique periods
    :type periods: np.ndarray
    :param val_frac: fraction the runs will hold out
    :type val_frac: float
    :param embargo: gap between the blocks
    :type embargo: int
    :param fit_scope: "train" or "all"
    :type fit_scope: str
    :return: number of leading periods the encoder may use
    :rtype: int
    :raises ValueError: if no fit periods would remain
    """
    if fit_scope == "all":
        return len(periods)

    n_val = int(round(len(periods) * val_frac))
    boundary = max(0, len(periods) - n_val - embargo)

    if boundary < 2:
        raise ValueError(
            "no fit periods left after val_frac={} and embargo={}; lower "
            "--val-frac".format(val_frac, embargo))
    return boundary


def rank_within_period(values, periods):
    """
    Map each code onto (0, 1) by its rank among that date's rows.

    Same form as src/live/submit.py:rank_normalise -- (rank - 0.5) / n -- which
    is exactly centred on 0.5 and never emits a hard 0 or 1, unlike
    rank(pct=True), whose top row is exactly 1.0 and whose mean is (n + 1) / 2n.

    Ties break by AVERAGE, not by position, and that differs from
    rank_normalise on purpose. A submission wants evenly spaced points, so
    method="first" is right there. A FEATURE wants two identical inputs to
    produce two identical values -- and here ties are common rather than
    pathological: the raw features live on a five-point grid, so on a typical
    date around 8 of 241 rows share their entire feature vector with another
    row and therefore get bit-identical activations. Under method="first" those
    rows are separated by whatever order the parquet happened to store them in,
    which made the same encoder emit different codes for the same row depending
    on which dataset it was attached to. Averaging removes the dependence on row
    order entirely, and the mean of the column stays exactly 0.5 either way.

    :param values: (n, k) raw activations
    :type values: np.ndarray
    :param periods: (n,) the period each row belongs to
    :type periods: pd.Series
    :return: (n, k) values strictly inside (0, 1)
    :rtype: np.ndarray
    """
    frame = pd.DataFrame(np.asarray(values, dtype="float64"))
    grouped = frame.groupby(periods.to_numpy(), sort=False)

    ranks = grouped.rank(method="average")
    sizes = grouped[0].transform("size")
    return (ranks.sub(0.5)).div(sizes, axis=0).to_numpy()


def column_mean_r2(truth, reconstruction, baseline_means):
    """
    Reconstruction R^2 against the "predict each column's mean" baseline.

    A raw MSE on rank bins is uninterpretable. This says whether anything was
    learned at all: 0 means the encoder is no better than a constant per column,
    and negative means it is worse. The baseline means come from the FIT rows,
    not from the slice being scored, so the baseline gets no held-out advantage
    the encoder was denied.

    :param truth: (n, f) clean values
    :type truth: np.ndarray
    :param reconstruction: (n, f) reconstruction
    :type reconstruction: np.ndarray
    :param baseline_means: (f,) per-column means from the fit rows
    :type baseline_means: np.ndarray
    :return: R^2
    :rtype: float
    """
    residual = float(((truth - reconstruction) ** 2).sum())
    baseline = float(((truth - baseline_means) ** 2).sum())
    if baseline <= 0:
        return float("nan")
    return 1.0 - residual / baseline


def pca_scree(fit_values, max_components):
    """
    Cumulative explained variance for k = 1..max_components, on the fit rows.

    Printed so --code-dim is an informed choice rather than an assertion.

    :return: cumulative ratios, one per k
    :rtype: [float]
    """
    from sklearn.decomposition import PCA

    n = min(int(max_components), fit_values.shape[1], fit_values.shape[0])
    model = PCA(n_components=n).fit(fit_values)
    return [float(v) for v in np.cumsum(model.explained_variance_ratio_)]


def fit_pca(fit_values, code_dim, seed):
    """
    The linear control, and the yardstick the autoencoder has to beat.

    :return: (model, encode_fn, reconstruct_fn)
    :rtype: (Any, callable, callable)
    """
    from sklearn.decomposition import PCA

    model = PCA(n_components=int(code_dim), random_state=int(seed))
    model.fit(fit_values)
    return (model,
            lambda values: model.transform(values),
            lambda values: model.inverse_transform(model.transform(values)))


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def code_diagnostics(raw_codes, ranked_codes, feature_values, feature_columns,
                     had_missing, fit_mask, target=None):
    """
    Everything worth knowing about the codes before they are written to disk.

    Note that per-code std is reported on the RAW activations, not the ranked
    ones: after ranking, every column has std 1/sqrt(12) ~= 0.2887 by
    construction and tells you nothing at all.

    :param raw_codes: (n, k) activations
    :type raw_codes: np.ndarray
    :param ranked_codes: (n, k) the columns as written
    :type ranked_codes: np.ndarray
    :param feature_values: (n, f) the inputs, after the neutral fill
    :type feature_values: np.ndarray
    :param had_missing: (n,) whether the row carried a NaN before the fill
    :type had_missing: np.ndarray
    :param fit_mask: (n,) rows the encoder was allowed to see
    :type fit_mask: np.ndarray
    :param target: (n,) the target, or None if the source has none
    :type target: np.ndarray or None
    :return: a JSON-serialisable dict
    :rtype: dict
    """
    n_codes = raw_codes.shape[1]
    out = {}

    out["code_std_raw"] = [float(np.std(raw_codes[:, j])) for j in range(n_codes)]

    # Code against code: two units at 0.99 are one unit and a wasted column.
    code_corr = np.corrcoef(np.nan_to_num(raw_codes, nan=0.0), rowvar=False)
    code_corr = np.atleast_2d(code_corr)
    np.fill_diagonal(code_corr, 0.0)
    worst = int(np.nanargmax(np.abs(code_corr))) if n_codes > 1 else 0
    out["max_code_code_corr"] = (
        float(np.nanmax(np.abs(code_corr))) if n_codes > 1 else 0.0)
    out["max_code_code_pair"] = (
        [int(worst // n_codes), int(worst % n_codes)] if n_codes > 1 else [0, 0])

    # Code against each raw input: a code at 0.99 with one feature is a copy,
    # not a compression. Common as k approaches the intrinsic dimension.
    against_inputs = []
    for j in range(n_codes):
        correlations = [
            abs(float(np.corrcoef(raw_codes[:, j], feature_values[:, f])[0, 1]))
            for f in range(feature_values.shape[1])
        ]
        best = int(np.nanargmax(correlations))
        against_inputs.append(
            {"code": j, "feature": feature_columns[best],
             "abs_corr": float(correlations[best])})
    out["closest_input"] = against_inputs

    # Reported, never used: is there anything here at all. Fit rows only, so a
    # number that looks encouraging cannot have come from the validation block.
    if target is not None:
        fit_target = pd.Series(target[fit_mask]).rank()
        out["code_target_spearman"] = [
            float(pd.Series(raw_codes[fit_mask, j]).rank().corr(fit_target))
            for j in range(n_codes)
        ]

    # Where does a row we know nothing about land? Not necessarily the 50th
    # percentile, and worth knowing before trusting 0.5 to mean "no view".
    out["neutral_row_percentile"] = None

    # 3.29% of rows carried a NaN, skewed toward young tokens. A code that
    # separates them is encoding survivorship, not structure.
    if had_missing.any() and not had_missing.all():
        out["code_mean_with_missing"] = [
            float(np.mean(ranked_codes[had_missing, j])) for j in range(n_codes)]
        out["code_mean_without_missing"] = [
            float(np.mean(ranked_codes[~had_missing, j])) for j in range(n_codes)]

    return out


def check_codes(diagnostics, allow_dead):
    """
    Refuse to write a dataset whose columns are constant or duplicated.

    :param diagnostics: output of code_diagnostics
    :type diagnostics: dict
    :param allow_dead: downgrade the refusals to warnings
    :type allow_dead: bool
    :raises SystemExit: with the fix in the message
    """
    problems = []

    for index, std in enumerate(diagnostics["code_std_raw"]):
        if std < DEAD_CODE_STD:
            problems.append(
                "code %d is dead (raw std %.2e). Lower --code-dim or raise "
                "--epochs." % (index, std))

    if diagnostics["max_code_code_corr"] > DUPLICATE_CODE_CORR:
        left, right = diagnostics["max_code_code_pair"]
        problems.append(
            "codes %d and %d correlate %.3f -- they are one column. Lower "
            "--code-dim." % (left, right, diagnostics["max_code_code_corr"]))

    if not problems:
        return

    if allow_dead:
        for problem in problems:
            print("WARN  %s" % problem)
        print("WARN  writing anyway because --allow-dead-codes was passed; this "
              "is for diagnosis, not for an arm")
        return

    raise SystemExit(
        "refusing to write the dataset:\n  %s\n"
        "Pass --allow-dead-codes to write it anyway." % "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Reading a built encoder back
# --------------------------------------------------------------------------- #


def encoder_meta_path(version):
    """Where a built dataset's encoder metadata lives."""
    return local_path("crypto", version, "encoder.json")


def read_encoder(version, device=None):
    """
    Load a built encoder and return something that codes a frame.

    This is the entry point a live path would import, which is why it lives here
    with the builder rather than in the torch module: the PCA control has to be
    loadable on a box with no torch, and the caller should not have to know which
    backend built the dataset.

    The returned function reindexes its input to the fitted column order (see
    align_features) and returns RAW activations; rank normalisation is the
    caller's, because it has to happen within a date and the caller owns the
    dates.

    :param version: dataset version, e.g. "crypto/v2.0-ae"
    :type version: str
    :return: (metadata, encode_fn) where encode_fn(df) -> (n, k) activations
    :rtype: (dict, callable)
    """
    with open(encoder_meta_path(version)) as handle:
        meta = json.load(handle)

    columns = meta["feature_columns"]

    if meta["method"] == "pca":
        import joblib

        model = joblib.load(local_path("crypto", version, "encoder.joblib"))
        return meta, lambda df: model.transform(align_features(df, columns))

    autoencoder = load_backend("ae")
    model = autoencoder.load_encoder(
        local_path("crypto", version, "encoder.pt"), device)
    resolved = autoencoder.resolve_device(device)

    def encode(df):
        return autoencoder.encode_values(
            model, align_features(df, columns), resolved,
            meta.get("batch_size", 4096))

    return meta, encode


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.quiet
    spec = get_spec("crypto")
    api = get_api("crypto")

    backend = load_backend(args.method)
    suffix = args.out_suffix or default_suffix(args.method, args.fit_scope)
    version = args.source_version + suffix

    # ---- source ----------------------------------------------------------- #
    # No dropna on any target, unlike build_target_encoding: this builder needs
    # no target at all, and dropping on target_binned_return_60 would silently
    # remove the 2.4% of rows that CURUPIRA_05_60 is fitted on.
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    # Sorted, so that the fit-window mask and the row order agree even if a
    # future source parquet arrives unsorted. Stable, so the within-date order
    # the source chose is preserved -- rank ties break by position.
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    # Parquet order, captured once and persisted. Everything downstream aligns
    # to this list rather than re-deriving it.
    feature_columns = [c for c in df.columns if c.startswith("feature_")]
    if not feature_columns:
        raise SystemExit(
            "no feature_ columns in %s; is --source-version right?" % src)

    print("source: %d rows, %d features, %d symbols, %d dates" % (
        len(df), len(feature_columns), df["symbol"].nunique(),
        df["date"].nunique()))

    had_missing = df[feature_columns].isna().any(axis=1).to_numpy()
    df = fill_missing_features(df, feature_columns, spec["neutral_value"], verbose)
    values = align_features(df, feature_columns)

    # ---- the fit window --------------------------------------------------- #
    embargo = (default_embargo("crypto", spec["target"])
               if args.embargo == "auto" else int(args.embargo))
    periods = np.sort(df["date"].unique())
    boundary = fit_boundary(periods, args.val_frac, embargo, args.fit_scope)
    boundary_period = pd.Timestamp(periods[boundary - 1])

    fit_mask = (df["date"] <= boundary_period).to_numpy()
    fit_periods = periods[:boundary]

    if args.encoder_from:
        pass          # the inherited banner below says where the encoder came from
    elif args.fit_scope == "all":
        print("fit scope: the whole panel, all %d periods -- this is LOOKAHEAD, "
              "an upper bound only" % len(fit_periods))
    else:
        print("fit scope: %d of %d periods, ending %s (val_frac=%.2f, "
              "embargo=%d)" % (len(fit_periods), len(periods),
                               boundary_period.date(), args.val_frac, embargo))
        print("causal for runs with --val-frac <= %.2f; the encoder saw nothing "
              "after %s" % (args.val_frac, boundary_period.date()))

    # The early-stopping slice is a chronological tail of the fit window, carved
    # with NO embargo: no target is involved, and a feature is observable at its
    # own date, so nothing crosses. The embargo is a target rule.
    n_stop = int(round(len(fit_periods) * args.stop_frac))
    stop_periods = fit_periods[len(fit_periods) - n_stop:] if n_stop else []
    stop_mask = fit_mask & df["date"].isin(stop_periods).to_numpy()
    inner_mask = fit_mask & ~stop_mask

    fit_values = values[inner_mask]
    stop_values = values[stop_mask]
    if not args.encoder_from:
        print("encoder trains on %d rows, holds out the last %d periods (%d rows)"
              % (len(fit_values), len(stop_periods), len(stop_values)))

    # ---- inherit an encoder instead of fitting one ------------------------ #
    #
    # The point of this path is COMPARABILITY. Fitting a separate autoencoder per
    # dataset would give every arm its own basis -- feature_ae_03 would mean a
    # different thing in each -- so a table of arms would not be a table of one
    # treatment. One encoder, applied everywhere, keeps the eight added columns
    # identical in meaning across every slot that gets them.
    #
    # It is also the only honest option here: the codes are a function of the raw
    # 22, which every derived dataset still carries, so re-fitting per dataset
    # would be re-deriving the same thing from the same inputs and calling the
    # differences signal.
    if args.encoder_from:
        source_meta, encode = read_encoder(args.encoder_from, args.device)

        # Inherited, not re-derived: these describe the ENCODER, and restating
        # them from this run's flags is how the two quietly come apart.
        args.code_dim = source_meta["code_dim"]
        embargo = source_meta["embargo"]
        boundary_period = pd.Timestamp(source_meta["fit_boundary_period"])
        fit_mask = (df["date"] <= boundary_period).to_numpy()

        if source_meta["fit_scope"] != "train":
            raise SystemExit(
                "the encoder for %s was built with --fit-scope %s, which is "
                "LOOKAHEAD; refusing to attach it to another dataset."
                % (args.encoder_from, source_meta["fit_scope"]))

        # align_features raises, naming any column, if this dataset does not
        # carry the exact 22 the encoder was fitted on.
        raw_codes = encode(df)

        print("inheriting the encoder built for %s" % args.encoder_from)
        print("  %d codes, fitted on periods <= %s (val_frac %.2f, embargo %d)"
              % (args.code_dim, source_meta["fit_boundary_period"],
                 source_meta["val_frac"], embargo))
        print("  its own held-out R^2 was %.4f against a PCA control at %.4f"
              % (source_meta["recon_r2_holdout"], source_meta["pca_r2_holdout"]))
        print("  %d of this dataset's %d rows are at or before that boundary"
              % (int(fit_mask.sum()), len(df)))

        scree, pca_r2, recon_r2, training, device = [], float("nan"), \
            float("nan"), {}, None
        names = source_meta["code_columns"]
        ranked = rank_within_period(raw_codes, df["date"])
    else:
        names = code_columns(args.column_prefix, args.code_dim)

        # ---- what a linear projection can do ------------------------------- #
        scree = pca_scree(fit_values, SCREE_MAX)
        print("PCA cumulative explained variance on the fit rows:")
        print("    " + "  ".join(
            "k=%d %.3f" % (k + 1, v) for k, v in enumerate(scree)
            if k + 1 in (1, 2, 4, 6, 8, 10, 12)))

        baseline_means = fit_values.mean(axis=0)
        pca_model, pca_encode, pca_reconstruct = fit_pca(
            fit_values, args.code_dim, args.seed)
        # With --stop-frac 0 there is no held-out slice, so every R^2 below is
        # in-sample. Say so rather than printing an in-sample number under a
        # held-out label.
        holdout = stop_values if len(stop_values) else fit_values
        holdout_label = "held-out" if len(stop_values) else "IN-SAMPLE (--stop-frac 0)"
        pca_r2 = column_mean_r2(holdout, pca_reconstruct(holdout), baseline_means)

        # ---- fit ---------------------------------------------------------------#
        training = {}
        device = None if args.method == "pca" else backend.resolve_device(args.device)
        if args.method == "pca":
            raw_codes = pca_encode(values)
            recon_r2 = pca_r2
            print("PCA control at k=%d: %s R^2 %.4f" % (
                args.code_dim, holdout_label, recon_r2))
        else:
            config = {
                "code_dim": args.code_dim,
                "hidden": tuple(int(h) for h in args.hidden.split(",") if h.strip()),
                "dropout": args.dropout,
                "lr": args.lr,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "swap_noise": args.swap_noise,
                "max_grad_norm": args.max_grad_norm,
                "seed": args.seed,
            }
            print("training the autoencoder on %s: %d -> %s -> %d" % (
                device, len(feature_columns), config["hidden"], args.code_dim))

            training = backend.train_autoencoder(
                fit_values, stop_values if len(stop_values) else None,
                config, device, verbose=verbose)
            model = training["model"]

            raw_codes = backend.encode_values(model, values, device, args.batch_size)
            recon_r2 = column_mean_r2(
                holdout,
                backend.reconstruct_values(model, holdout, device, args.batch_size),
                baseline_means)

            print("autoencoder: best epoch %d of %d, %s MSE %.5f, R^2 %.4f" % (
                training["best_epoch"], training["epochs_run"], holdout_label,
                training["best_val_mse"], recon_r2))
            if recon_r2 > pca_r2:
                verdict = "the nonlinearity bought %+.4f" % (recon_r2 - pca_r2)
            elif args.swap_noise > 0:
                # Not a fair fight, and saying "PCA wins" here would be wrong. The
                # autoencoder is TRAINED on corrupted rows and SCORED on clean ones;
                # PCA is fitted directly on the clean rows it is then scored on.
                # Denoising deliberately trades reconstruction for a code that does
                # not depend on any single input, so losing this number is the
                # expected price, not a verdict. Rerun with --swap-noise 0 for the
                # like-for-like reconstruction comparison; the question of whether
                # the CODES are more useful is settled downstream by ae_plus_raw
                # against pca_plus_raw, not here.
                verdict = ("PCA reconstructs %+.4f better, but it was fitted on the "
                           "clean rows it is scored on while this was trained on "
                           "rows corrupted at --swap-noise %.2f -- rerun with "
                           "--swap-noise 0 to compare like for like"
                           % (pca_r2 - recon_r2, args.swap_noise))
            else:
                verdict = ("THE LINEAR PROJECTION IS AS GOOD -- use --method pca "
                           "and drop torch")

            print("PCA control at the same k=%d: R^2 %.4f  (%s)" % (
                args.code_dim, pca_r2, verdict))

            if training["hit_epoch_cap"]:
                print("WARN  training stopped at the --epochs cap with patience "
                      "still left, so this network is TRUNCATED rather than "
                      "converged. Its R^2 and everything downstream of it "
                      "understate what the architecture can do -- raise --epochs "
                      "(currently %d) before comparing it with the PCA control."
                      % args.epochs)

        if recon_r2 <= 0:
            raise SystemExit(
                "refusing to write the dataset: held-out R^2 is %.4f, so the encoder "
                "reconstructs worse than each column's own mean. Raise --epochs or "
                "--code-dim." % recon_r2)

        ranked = rank_within_period(raw_codes, df["date"])

    # ---- to the repo's scale ---------------------------------------------- #

    diagnostics = code_diagnostics(
        raw_codes, ranked, values, feature_columns, had_missing, fit_mask,
        target=(df[spec["target"]].to_numpy()
                if spec["target"] in df.columns else None))

    # Where an all-neutral row lands, computed by pushing one through. Worth
    # knowing before trusting 0.5 to mean "no view": it is not necessarily the
    # 50th percentile.
    neutral_row = np.full((1, len(feature_columns)), spec["neutral_value"],
                          dtype="float32")
    if args.encoder_from:
        # The inherited encoder takes a frame, and reindexes it to its own
        # fitted column order -- which is the whole point of going through it
        # rather than reaching for the raw array.
        neutral_code = encode(pd.DataFrame(neutral_row, columns=feature_columns))
    elif args.method == "pca":
        neutral_code = pca_encode(neutral_row)
    else:
        neutral_code = backend.encode_values(
            training["model"], neutral_row, device, args.batch_size)
    diagnostics["neutral_row_percentile"] = [
        float((raw_codes[:, j] < neutral_code[0, j]).mean())
        for j in range(args.code_dim)]

    check_codes(diagnostics, args.allow_dead_codes)

    print("codes: raw std %s" % "  ".join(
        "%.3f" % s for s in diagnostics["code_std_raw"]))
    print("       max code-code |corr| %.3f (codes %d and %d)" % (
        diagnostics["max_code_code_corr"], *diagnostics["max_code_code_pair"]))
    closest = max(diagnostics["closest_input"], key=lambda d: d["abs_corr"])
    print("       closest single input: code %d vs %s at |corr| %.3f" % (
        closest["code"], closest["feature"], closest["abs_corr"]))
    print("       an all-neutral row lands at percentiles %s" % "  ".join(
        "%.2f" % p for p in diagnostics["neutral_row_percentile"]))
    if "code_target_spearman" in diagnostics:
        print("       spearman vs the target (fit rows, reported not used): %s" %
              "  ".join("%+.3f" % s for s in diagnostics["code_target_spearman"]))

    for index, name in enumerate(names):
        df[name] = ranked[:, index].astype("float32")

    # ---- assertions and write --------------------------------------------- #
    source_rows = len(pd.read_parquet(src, columns=["date"]))
    if len(df) != source_rows:
        raise SystemExit(
            "row count changed: %d in, %d out. This builder must not drop rows."
            % (source_rows, len(df)))
    if df[names].isna().to_numpy().any():
        raise SystemExit("some codes are NaN, which should be impossible here")

    n_feat = sum(c.startswith("feature_") for c in df.columns)
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d raw + %d codes) -> %s (%.0f MB)" % (
        n_feat, len(feature_columns), args.code_dim, dest,
        os.path.getsize(dest) / 1e6))

    # ---- the encoder itself ------------------------------------------------#
    meta = {
        "source_version": args.source_version,
        "out_version": version,
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "builder_argv": sys.argv[1:] if argv is None else list(argv),
        "method": args.method,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "code_dim": args.code_dim,
        "code_columns": names,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "swap_noise": args.swap_noise,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "stop_frac": args.stop_frac,
        "seed": args.seed,
        # Both describe the ENCODER's fit window, so on the inherited path they
        # must come from the encoder, not from this invocation's flags. A
        # downstream leakage gate reads exactly these two fields.
        "fit_scope": (source_meta["fit_scope"] if args.encoder_from
                      else args.fit_scope),
        "val_frac": (source_meta["val_frac"] if args.encoder_from
                     else args.val_frac),
        "embargo": embargo,
        "fit_boundary_period": str(boundary_period.date()),
        "fit_rows": (int(fit_mask.sum()) if args.encoder_from
                     else int(inner_mask.sum())),
        "fit_periods": (source_meta["fit_periods"] if args.encoder_from
                        else int(len(fit_periods))),
        "neutral_fill": float(spec["neutral_value"]),
        "scale": "rank_within_period",
        "pca_scree": scree,
        "recon_r2_holdout": recon_r2,
        "pca_r2_holdout": pca_r2,
        "r2_is_out_of_sample": (source_meta["r2_is_out_of_sample"]
                                if args.encoder_from else bool(len(stop_values))),
        "diagnostics": diagnostics,
    }

    if args.encoder_from:
        # Copy the encoder in rather than leaving a pointer. The rule from the
        # persistence note holds here too: an encoder that lives beside the
        # parquet it produced makes "which encoder built this?" unanswerable by
        # mistake, whereas a reference can be followed to a directory somebody
        # has since rebuilt. Everything about the fit is inherited verbatim, so
        # the two copies cannot disagree about anything except where they sit.
        meta["encoder_from"] = args.encoder_from
        for key in ("weights_sha1", "torch_version", "sklearn_version",
                    "best_epoch", "epochs_run", "best_val_mse", "hit_epoch_cap",
                    "hidden", "dropout", "swap_noise", "lr", "batch_size",
                    "seed", "stop_frac"):
            if key in source_meta:
                meta[key] = source_meta[key]

        for filename in ("encoder.pt", "encoder.joblib"):
            origin = local_path("crypto", args.encoder_from, filename)
            if os.path.exists(origin):
                shutil.copy(origin, local_path("crypto", version, filename))
    elif args.method == "pca":
        import joblib
        import sklearn

        joblib.dump(pca_model, local_path("crypto", version, "encoder.joblib"))
        meta["sklearn_version"] = sklearn.__version__
    else:
        import torch

        meta["weights_sha1"] = backend.save_encoder(
            local_path("crypto", version, "encoder.pt"), training["model"])
        meta["torch_version"] = torch.__version__
        meta["best_epoch"] = training["best_epoch"]
        meta["epochs_run"] = training["epochs_run"]
        meta["best_val_mse"] = training["best_val_mse"]
        meta["hit_epoch_cap"] = training["hit_epoch_cap"]

    with open(encoder_meta_path(version), "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True, default=str)

    # Copied so BMC and MMC stay identical across arms, as every builder does.
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
