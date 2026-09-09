"""
Build a Crypto dataset whose FEATURES are other runs' out-of-sample predictions.

    python3 -m src.data_mani.build_stack_dataset \
        --members ridge/e10_base_ridge_raw mlp/e10_base_mlp_raw \
        --out-version crypto/v2.0-stackX

This is the level-0 half of stacked generalization (Wolpert). It reads finished
forecast CSVs, ranks each within period, and writes them out as `feature_stk_*`
columns of an ordinary dataset version. The level-1 half is then just a normal
run against that version:

    python3 -m src.run_forecast crypto ridge --data-version crypto/v2.0-stackX
    python3 -m src.run_forecast crypto mlp   --data-version crypto/v2.0-stackX

and it costs no pipeline changes at all, because Crypto discovers its features
by the `feature_` prefix (_load_crypto) rather than from a fixed list. The
meta-learner therefore inherits the walk-forward, the purged inner CV, the
hyperparameter search, the scoring and the four-file output tree unmodified,
and a stack is directly comparable with every other arm in results/.

Which members you list decides whether you are stacking ONE model class or
SEVERAL; which model you pass to run_forecast decides the meta-learner. Neither
is a property of this file.

------------------------------------------------------------------------------
Why this is not lookahead
------------------------------------------------------------------------------

Three separate guarantees, and it is worth being clear that they are separate:

1. Every member column is already out-of-sample. walk_forward_fit_and_predict
   only predicts periods marked "validation", and each block is fitted on
   strictly earlier periods minus the embargo. A member forecast CSV therefore
   contains no in-sample rows to leak.

2. Members are inner-joined, so the panel spans the INTERSECTION of their
   validation windows -- never earlier than a member was scored, never wider.

3. The stack run splits that panel again, chronologically and with its own
   embargo (make_validation_split), so the meta-learner trains on the earlier
   part and is scored only on the tail.

The cost of (2) and (3) is period count, and it is not small: members validated
on ~510 dates at val_frac=0.3 leave a stack scored on the ~150-date tail. That
is the honest window. Compare a stack only against baselines measured on the
same dates -- the reporting layer inner-joins per-period series, so it already
does this, but a hand-written comparison will not.

------------------------------------------------------------------------------
Two things this refuses to do
------------------------------------------------------------------------------

Mixed targets. A member scored on target_binned_return_60 and one scored on
_20 do not describe the same problem, and a stack over both would be learning
the label as much as the models. Every member's forecast header is checked
against --target and a mismatch is fatal.

Classic. Crypto finds features by prefix; Classic reads a named feature set out
of features.json and selects those columns explicitly, so a synthesised version
would load with zero features and still run. Rather than half-support that, this
is crypto-only.
"""

import argparse
import os
import shutil

import pandas as pd

from src.analysis.results_table import target_of
from src.data_mani.utils import (
    META_MODEL_COL,
    REFERENCE_COL,
    download_if_missing,
    get_api,
    get_spec,
    local_path,
)

#: Column prefix for the member predictions. Must start with "feature_" or the
#: Crypto loader will not see them and the stack will fit on nothing.
DEFAULT_PREFIX = "feature_stk_"

#: Never written into the dataset. The split is derived at load time from
#: --val-frac/--embargo, and the reference/meta series are joined on by
#: forecast() out of historical_meta_models.parquet. Emitting our own copies
#: would shadow both, which is the sort of thing that scores fine and means
#: nothing.
RESERVED = ("split", REFERENCE_COL, META_MODEL_COL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble member predictions into a stacking dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--members", nargs="+", required=True,
                        help="member runs, as 'model/run' or bare 'run' (which "
                             "then needs --model)")
    parser.add_argument("--model", default=None,
                        help="default model directory for bare member names")
    parser.add_argument("--tournament", default="crypto", choices=["crypto"],
                        help="crypto only -- see the module docstring")
    parser.add_argument("--source-version", default="crypto/v2.0",
                        help="version the targets and the meta model come from")
    parser.add_argument("--out-version", default=None,
                        help="version to write, e.g. crypto/v2.0-stackX")
    parser.add_argument("--out-suffix", default=None,
                        help="alternative to --out-version: appended to the "
                             "source version. Defaults to -stack<member count>")
    parser.add_argument("--target", default=None,
                        help="label the members must all be scored on. Defaults "
                             "to the tournament's")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help="feature-column prefix for member predictions")
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def split_member(name, default_model):
    """
    Resolve a member name to (model directory, run name).

    Members addressed "model/run" is the same convention run_ensemble and
    paired_delta use, and it is the interesting case: stacking is most useful
    across wrappers, which write into separate trees.

    :param name: "model/run" or a bare run name
    :type name: str
    :param default_model: model directory for bare names, or None
    :type default_model: str or None
    :return: (model, run)
    :rtype: (str, str)
    """
    if "/" in name:
        model, run = name.split("/", 1)
        return model, run
    if not default_model:
        raise SystemExit(
            "member %r has no model directory: pass it as 'model/run' or set "
            "--model" % name)
    return default_model, name


def member_column(model, run, prefix=DEFAULT_PREFIX):
    """
    The feature-column name a member's predictions are written under.

    The slash becomes a double underscore so two runs sharing a name in
    different model trees cannot collide into one column.

    :param model: model directory
    :type model: str
    :param run: run name
    :type run: str
    :param prefix: feature-column prefix
    :type prefix: str
    :return: column name
    :rtype: str
    """
    return "%s%s__%s" % (prefix, model, run)


def rank_within_period(frame, period_col):
    """
    Rank a member's predictions to (0, 1] within each period.

    Members are not on a common scale -- a regression on the binned target and a
    classifier's predict_proba occupy different ranges, and even two ridges with
    different penalties have different spreads. Numerai scores by rank anyway,
    so ranking first is the correct normalisation and changes nothing about how
    a member would have scored alone. It also puts the meta-learner's inputs on
    the scale its neutral fill value (0.5) assumes.

    :param frame: member frame indexed by (id, period) with a prediction column
    :type frame: pd.DataFrame
    :param period_col: name of the period index level
    :type period_col: str
    :return: per-period percentile ranks
    :rtype: pd.Series
    """
    return frame.groupby(level=period_col)["prediction"].rank(pct=True)


def load_member(tournament, model, run, id_col, period_col):
    """
    Read one member's forecast CSV, indexed on (id, period).

    :param tournament: e.g. "crypto"
    :type tournament: str
    :param model: model directory under results/forecast/<tournament>/
    :type model: str
    :param run: run name
    :type run: str
    :param id_col: id column name
    :type id_col: str
    :param period_col: period column name
    :type period_col: str
    :return: member frame indexed by (id, period)
    :rtype: pd.DataFrame
    """
    path = os.path.join("results", "forecast", tournament, model, "%s.csv" % run)
    if not os.path.exists(path):
        raise SystemExit("member run not found: %s" % path)

    frame = pd.read_csv(
        path, parse_dates=[period_col] if period_col == "date" else None)
    return frame.set_index([id_col, period_col])


def assemble(ranked, id_col, period_col, prefix=DEFAULT_PREFIX):
    """
    Inner-join the ranked members into one panel.

    The join is what bounds the panel to the members' shared window -- see the
    module docstring. Duplicate column names are refused rather than silently
    concatenated into a wider frame than the caller asked for.

    :param ranked: ordered [(column name, per-period-ranked Series)]
    :type ranked: [(str, pd.Series)]
    :param id_col: id index level name
    :type id_col: str
    :param period_col: period index level name
    :type period_col: str
    :param prefix: feature-column prefix, checked against the loader's contract
    :type prefix: str
    :return: panel indexed by (id, period), one column per member
    :rtype: pd.DataFrame
    """
    if not prefix.startswith("feature_"):
        raise SystemExit(
            "--prefix must start with 'feature_' (got %r): the Crypto loader "
            "finds features by that prefix, so anything else loads as a dataset "
            "with no features at all" % prefix)

    names = [name for name, _ in ranked]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SystemExit("duplicate member columns: %s" % duplicates)

    joined = pd.concat([series.rename(name) for name, series in ranked],
                       axis=1, join="inner")
    joined.index.names = [id_col, period_col]
    return joined


def attach_targets(panel, source, verbose=True):
    """
    Join the real target columns onto the member panel.

    Read off the source dataset rather than off a member's forecast, for two
    reasons: it carries every target (so a 60-day stack run stays possible even
    when the members were scored on the 20-day label), and it cannot inherit a
    member's own column naming.

    Rows absent from the source have no target at all and are dropped here.
    Rows present but with a NaN target are LEFT ALONE -- the 60-day label is
    genuinely unresolved over the last 40 dates, and load_tournament_data drops
    on whichever target is actually being modelled.

    :param panel: member panel indexed by (id, period)
    :type panel: pd.DataFrame
    :param source: source frame indexed by (id, period), target columns only
    :type source: pd.DataFrame
    :param verbose: print what was dropped
    :type verbose: bool
    :return: panel with the target columns appended
    :rtype: pd.DataFrame
    """
    known = panel.index.isin(source.index)
    if not known.all() and verbose:
        print("dropped %d rows absent from the source dataset (%.2f%%)" % (
            (~known).sum(), 100 * (~known).mean()))
    panel = panel[known]

    return panel.join(source.reindex(panel.index))


def main():
    args = parse_args()
    verbose = not args.quiet

    spec = get_spec(args.tournament)
    id_col, period_col = spec["id_col"], spec["period_col"]
    target_col = args.target or spec["target"]

    # ------------------------------------------------------------------ #
    # Members
    # ------------------------------------------------------------------ #
    resolved = [(name,) + split_member(name, args.model) for name in args.members]

    # Check every label before reading a single row. The label a run was SCORED
    # on lives in the forecast header, not in the metrics config -- run_ensemble
    # writes no target key, so a config lookup reports every 60-day blend as
    # 20-day. Reading the header costs milliseconds; loading four 10 MB CSVs to
    # then refuse them does not.
    scored_on = {name: target_of(args.tournament, model, run)
                 for name, model, run in resolved}

    # A missing forecast and a wrong label are different mistakes, and reporting
    # the first as the second sends the reader looking for a target problem that
    # is not there.
    absent = [name for name, target in scored_on.items() if target is None]
    if absent:
        raise SystemExit("member run not found: %s" % ", ".join(
            os.path.join("results", "forecast", args.tournament, model, "%s.csv" % run)
            for name, model, run in resolved if name in absent))

    mismatched = [(name, t) for name, t in scored_on.items() if t != target_col]
    if mismatched:
        raise SystemExit(
            "every member must be scored on %s, but: %s\nStacking across labels "
            "would learn the label as much as the models." % (
                target_col,
                ", ".join("%s is %s" % (n, t) for n, t in mismatched)))

    ranked = []
    for name, model, run in resolved:
        frame = load_member(args.tournament, model, run, id_col, period_col)
        column = member_column(model, run, args.prefix)
        ranked.append((column, rank_within_period(frame, period_col)))

        if verbose:
            print("  %-46s %7d rows, %4d periods" % (
                name, len(frame), frame.index.get_level_values(period_col).nunique()))

    panel = assemble(ranked, id_col, period_col, args.prefix)
    if verbose:
        periods = panel.index.get_level_values(period_col)
        print("aligned on %d rows, %d periods (%s .. %s)" % (
            len(panel), periods.nunique(),
            periods.min().date() if period_col == "date" else periods.min(),
            periods.max().date() if period_col == "date" else periods.max()))

    # ------------------------------------------------------------------ #
    # Targets, off the source dataset
    # ------------------------------------------------------------------ #
    api = get_api(args.tournament)
    source_path = download_if_missing(
        api, args.tournament, args.source_version, "train.parquet", verbose)
    source = pd.read_parquet(source_path)
    # Same promotion _load_crypto does: the shipped file keys on the index, but a
    # derived version may already carry it as a column, and reset_index would
    # then invent an "index" column and lose the id.
    if id_col not in source.columns:
        source.index.name = source.index.name or id_col
        source = source.reset_index()
    else:
        source = source.reset_index(drop=True)
    source[period_col] = pd.to_datetime(source[period_col])
    target_cols = [c for c in source.columns if c.startswith("target")]
    source = source.set_index([id_col, period_col])[target_cols]

    if target_col not in target_cols:
        raise SystemExit("source version carries no %s (has: %s)" % (
            target_col, sorted(target_cols)))

    panel = attach_targets(panel, source, verbose)
    panel = panel.astype(
        {c: "float32" for c in panel.columns if c.startswith(args.prefix)})

    # ------------------------------------------------------------------ #
    # Write it as an ordinary dataset version
    # ------------------------------------------------------------------ #
    suffix = args.out_suffix or ("-stack%d" % len(args.members))
    version = args.out_version or (args.source_version + suffix)

    frame = panel.reset_index()
    leaked = [c for c in RESERVED if c in frame.columns]
    if leaked:
        raise SystemExit("refusing to write reserved columns: %s" % leaked)

    dest = local_path(args.tournament, version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    frame.set_index(id_col).to_parquet(dest)

    # forecast() joins the reference and meta-model series out of this file. If
    # it is missing the run still succeeds and simply reports no BMC/MMC, which
    # would quietly make the stack non-comparable with every other arm.
    for filename in (spec["meta_model_file"],):
        origin = download_if_missing(
            api, args.tournament, args.source_version, filename, verbose)
        target = local_path(args.tournament, version, filename)
        if not os.path.exists(target):
            shutil.copy(origin, target)

    if verbose:
        features = [c for c in frame.columns if c.startswith("feature_")]
        print("%d stack features -> %s (%.1f MB)" % (
            len(features), dest, os.path.getsize(dest) / 1e6))
        for name in features:
            print("    %-52s mean %.4f  sd %.4f" % (
                name, frame[name].mean(), frame[name].std()))
        print("\nrun a meta-learner with:")
        for meta in ("ridge", "mlp"):
            print("    python3 -m src.run_forecast %s %s --data-version %s "
                  "--target %s" % (args.tournament, meta, version, target_col))
        print()


if __name__ == "__main__":
    main()
