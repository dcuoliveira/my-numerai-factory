"""
One results table, with level, risk and risk-adjusted level side by side.

    python3 -m src.analysis.results_table --model lstm --runs a b c

    run                   tgt     n      CORR      MMC | c.sharpe dd(corr) |   payout p.sharpe  dd(pay) | calmar

`tgt` and `n` are there because the other columns are only readable next to them:
CORR on the 60-day label is structurally higher than on the 20-day one, so a mixed
table ranks the LABEL as much as the model, and calmar's denominator grows with
series length, so it is comparable only across rows with the same n.

Why the payout columns are rebuilt per period rather than read off the metrics
json: the json records summaries of the CORR series and the MMC series separately,
and payout is `0.1 * corr + mmc`. A sum's Sharpe is NOT any function of its parts'
Sharpes -- corr and mmc are correlated, so combining their summaries would give a
number that looks plausible and is wrong. The payout SERIES is formed first, one
value per period, and summarised once.

Everything is summarised through prediction.metrics.summarize, so the conventions
match every other number this repo reports: std with ddof=0, and max_drawdown
returned POSITIVE (bigger is worse).

Calmar is payout mean / payout max drawdown -- level per unit of worst peak-to-
trough. It answers a different question from the Sharpe: Sharpe punishes ordinary
period-to-period noise, Calmar punishes the one bad stretch you would actually have
had to sit through. An arm can win on one and lose on the other, which is exactly
when it is worth looking at both.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.prediction.metrics import summarize

#: Payout weighting, as Numerai pays it.
CORR_WEIGHT = 0.1


def degenerate_periods(periods_table):
    """
    Periods where the model produced no signal but WAS scoreable.

    `_safe_corr` returns NaN when a period's correlation is undefined, and its
    docstring tells the caller to drop the period. That is right when the period
    could not be scored at all, and wrong when the model simply predicted one
    constant value for every symbol: dropping charges nothing for the day, so a
    model that died is scored only on the days it happened to work.

    The two cases are separable without opening the forecast. `mmc` is computed
    from the same rows against the reference, so an mmc that IS defined proves
    the period had rows and a reference -- leaving a degenerate prediction as
    the only remaining explanation for the missing corr. Both missing means the
    period was genuinely unscoreable, and those stay dropped.

    This is not hypothetical. E10's `mlp/e10_meta_mlp_stackR` collapsed to a
    constant in 2 of its 7 refit blocks and, scored the old way, ranked FIRST in
    its group; it also shrank every other arm's common window from 111 dates to
    67, because Block intersects the non-null payout indices. Of 815 per-period
    files it is the only one affected, so this costs nothing elsewhere.

    :param periods_table: a per_period frame with corr and mmc columns
    :type periods_table: pd.DataFrame
    :return: boolean mask over the frame's index
    :rtype: pd.Series
    """
    corr = periods_table["corr"]
    if "mmc" not in periods_table:
        return pd.Series(False, index=periods_table.index)
    return corr.isna() & periods_table["mmc"].notna()


def charged_corr(periods_table):
    """
    The corr series with a degenerate period charged the 0 it earned.

    :param periods_table: a per_period frame with corr and mmc columns
    :type periods_table: pd.DataFrame
    :return: corr, with degenerate periods set to 0.0
    :rtype: pd.Series
    """
    return periods_table["corr"].mask(degenerate_periods(periods_table), 0.0)


def payout_series(periods_table):
    """
    The per-period payout, `0.1 * corr + mmc`.

    A degenerate period contributes `0.1 * 0 + mmc` rather than dropping out --
    see degenerate_periods() for why that is the honest treatment and not a
    convenience.

    :param periods_table: a per_period frame with corr and mmc columns
    :type periods_table: pd.DataFrame
    :return: one payout per period
    :rtype: pd.Series
    """
    mmc = periods_table["mmc"] if "mmc" in periods_table else 0.0
    return CORR_WEIGHT * charged_corr(periods_table) + mmc


def burn_days(periods_table):
    """
    Periods whose payout came out negative -- how OFTEN the model lost, not how
    deep.

    A downside-FREQUENCY companion to dd(pay), which is downside DEPTH. The two
    part ways: a model can hold a shallow max drawdown while bleeding a little
    most days, or take one brutal stretch and stay green the rest -- same
    dd(pay), opposite lived experience.

    A raw COUNT of periods, so -- like calmar and n -- it is comparable only
    across rows on the same window; read it beside n, and prefer the rate
    burn(d)/n when windows differ. One honest caveat to state when reporting it:
    the target is a 20-day forward return, so consecutive periods overlap ~95%
    and a red run is not a red run of INDEPENDENT days -- the count tracks time
    spent under water, not that many separate losing bets.

    :param periods_table: a per_period frame with corr and mmc columns
    :type periods_table: pd.DataFrame
    :return: number of negative-payout periods, or None if the series is empty
    :rtype: int or None
    """
    payout = payout_series(periods_table).dropna()
    if payout.empty:
        return None
    return int((payout < 0).sum())


def short_target(target):
    """
    The compact form of a target column name, for a table column.

    :param target: e.g. "target_binned_return_60"
    :type target: str or None
    :return: e.g. "t60", or "-" when the target could not be determined
    :rtype: str
    """
    if not target:
        return "-"
    return "t" + target.rsplit("_", 1)[-1]


def target_of(tournament, model, run):
    """
    The label a run was actually scored against.

    Read from the forecast header, NOT from the metrics config. run_ensemble
    writes no `target` key into a blend's config, so a config lookup falls
    through to the 20-day default and silently mislabels every 60-day blend --
    t60_c05_blend is that case in this repo. The forecast file's target column
    is the label the run was scored on, and reading the header alone costs
    milliseconds on a 10 MB file.

    :param tournament: e.g. "crypto"
    :type tournament: str
    :param model: model tree under results/forecast/<tournament>/
    :type model: str
    :param run: run name
    :type run: str
    :return: the target column name, or None if the forecast is missing
    :rtype: str or None
    """
    path = os.path.join("results", "forecast", tournament, model,
                        "{}.csv".format(run))
    if not os.path.exists(path):
        return None
    for column in pd.read_csv(path, nrows=0).columns:
        if column.startswith("target_"):
            return column
    return None


def row_from_frame(label, periods, target=None, cv=None):
    """
    Assemble one row from an already-sliced per-period frame.

    Split out from row_for so a caller can summarise a run on a window other
    than its own. A table mixing 510-period and 470-period runs can only
    compare calmar on their intersection, because calmar's denominator grows
    with series length -- so the report slices every frame to the common dates
    first and comes through here.

    Every column is computed from the frame, none from the metrics json. That
    is the same arithmetic the json already stores: summarising the csv's corr
    and mmc columns reproduces the json's summaries exactly, and computing them
    here means a windowed row and a full-window row are built by one code path
    rather than two that can drift.

    :param label: how the row should be named
    :type label: str
    :param periods: per-period frame with corr and mmc columns
    :type periods: pd.DataFrame
    :param target: the scored target column name, if known
    :type target: str or None
    :param cv: the cv description, if known
    :type cv: str or None
    :return: the row
    :rtype: dict
    """
    # Charged, not dropped: a period the model spent predicting one constant
    # value earned nothing and is scored as nothing. See degenerate_periods().
    dead = int(degenerate_periods(periods).sum()) if "corr" in periods else 0
    corr = summarize(charged_corr(periods).dropna())
    mmc = (summarize(periods["mmc"].dropna()) if "mmc" in periods
           else {"mean": None, "sharpe": None, "max_drawdown": None})
    payout = payout_series(periods).dropna()
    summary = summarize(payout)

    # Calmar is undefined against a zero drawdown -- a series that never gave
    # anything back. Report it as infinite rather than dividing by zero.
    drawdown = summary["max_drawdown"]
    calmar = (summary["mean"] / drawdown) if drawdown and drawdown > 0 else float("inf")

    return {
        "run": label,
        # Carried so the caller can refuse to put two targets in one table: CORR on
        # the 60-day label is structurally higher, so mixing them ranks the label
        # rather than the model. That mistake has already been made once here.
        "target": target,
        "cv": cv,
        # Printed whenever rows of different lengths share a table, because
        # calmar is only comparable at equal n.
        "n": int(len(payout)),
        # Periods the arm spent degenerate. A non-zero value here is a model
        # FAILURE, not a weak score, and must be read before the payout column.
        "dead": dead,
        "CORR": corr["mean"],
        "MMC": mmc["mean"],
        "c.sharpe": corr["sharpe"],
        "dd(corr)": corr["max_drawdown"],
        "payout": summary["mean"],
        "p.sharpe": summary["sharpe"],
        "dd(pay)": summary["max_drawdown"],
        "calmar": calmar,
        # How OFTEN it lost, to sit beside dd(pay)'s how DEEP. A count of losing
        # periods, so comparable only at equal window -- read with n.
        "burn(d)": burn_days(periods),
    }


def row_for(tournament, model, run):
    """
    Assemble one run's row.

    `run` may be addressed as "model/run" to reach another model tree, the same
    convention paired_delta.py uses -- deployed slots are spread across ridge,
    ridge_glmnet and lgb, so a table of them cannot live in one tree.

    :param tournament: e.g. "crypto"
    :type tournament: str
    :param model: model tree to use when `run` does not name one
    :type model: str
    :param run: run name, optionally "model/run"
    :type run: str
    :return: the row, or None if the run's artifacts are missing
    :rtype: dict or None
    """
    label = run
    if "/" in run:
        model, run = run.split("/", 1)

    metrics_path = os.path.join("results", "metrics", tournament, model,
                                "{}.json".format(run))
    periods_path = os.path.join("results", "per_period", tournament, model,
                                "{}.csv".format(run))
    if not (os.path.exists(metrics_path) and os.path.exists(periods_path)):
        return None

    config = (json.load(open(metrics_path)).get("config") or {})
    periods = pd.read_csv(periods_path)

    return row_from_frame(
        label, periods,
        target=target_of(tournament, model, run) or config.get("target"),
        cv="{}/{}/{}".format(config.get("cv_scheme"), config.get("train_window"),
                             config.get("refit_every")))


def render(rows, sort_by="payout"):
    """
    Format the rows as the fixed-width table.

    :param rows: rows from row_for()
    :type rows: [dict]
    :param sort_by: column to sort descending by, or None to keep the given order
    :type sort_by: str or None
    :return: the rendered table
    :rtype: str
    """
    if sort_by:
        rows = sorted(
            rows,
            key=lambda r: (r.get(sort_by) if r.get(sort_by) is not None
                           and np.isfinite(r[sort_by]) else -np.inf),
            reverse=True,
        )

    width = max([len(r["run"]) for r in rows] + [20])
    header = ("{:<%d} {:>4} {:>5} {:>9} {:>9} | {:>8} {:>8} | {:>9} {:>8} {:>8} | {:>7} {:>7}"
              % width).format(
        "run", "tgt", "n", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout",
        "p.sharpe", "dd(pay)", "calmar", "burn(d)")
    lines = [header, "-" * len(header)]

    def cell(value, spec):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        if isinstance(value, float) and np.isinf(value):
            return "inf"
        return format(value, spec)

    def mixed_targets(rows):
        """The distinct targets in the table, when there is more than one."""
        seen = {r.get("target") for r in rows if r.get("target")}
        return sorted(seen) if len(seen) > 1 else []

    for r in rows:
        lines.append(
            ("{:<%d} {:>4} {:>5} {:>9} {:>9} | {:>8} {:>8} | {:>9} {:>8} {:>8} | {:>7} {:>7}"
             % width).format(
                r["run"],
                short_target(r.get("target")),
                cell(r.get("n"), "d"),
                cell(r["CORR"], ".5f"), cell(r["MMC"], ".5f"),
                cell(r["c.sharpe"], ".2f"), cell(r["dd(corr)"], ".2f"),
                cell(r["payout"], ".5f"), cell(r["p.sharpe"], ".2f"),
                cell(r["dd(pay)"], ".2f"), cell(r["calmar"], ".3f"),
                cell(r.get("burn(d)"), "d"),
            )
        )
    return "\n".join(lines)


def parse_args(argv=None):
    """
    Parse the command line.

    :param argv: argument list, or None to read sys.argv
    :type argv: [str] or None
    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m src.analysis.results_table",
        description="Level, risk and risk-adjusted level for a set of runs.")
    parser.add_argument("--tournament", default="crypto")
    parser.add_argument("--model", default="lstm",
                        help="model tree under results/metrics/<tournament>/")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--sort-by", default="payout",
                        choices=("CORR", "MMC", "payout", "p.sharpe", "calmar", "none"))
    return parser.parse_args(argv)


def main(argv=None):
    """
    Print the table.

    :param argv: argument list, or None to read sys.argv
    :type argv: [str] or None
    :return: process exit code
    :rtype: int
    """
    args = parse_args(argv)

    rows, missing = [], []
    for run in args.runs:
        row = row_for(args.tournament, args.model, run)
        (rows.append(row) if row else missing.append(run))

    if not rows:
        print("no runs found for model '{}': {}".format(args.model, ", ".join(args.runs)))
        return 1

    print(render(rows, None if args.sort_by == "none" else args.sort_by))

    targets = sorted({r["target"] for r in rows if r.get("target")})
    if len(targets) > 1:
        # CORR on the 60-day label is structurally higher, so a mixed table ranks
        # the label as much as the model. Say so rather than letting the reader
        # take one ordering for two questions.
        print("\nWARNING: two targets in one table ({}). CORR is not comparable "
              "across them -- read the tgt column, or split the table."
              .format(", ".join(targets)))

    lengths = sorted({r["n"] for r in rows if r.get("n")})
    if len(lengths) > 1:
        print("\nNOTE: rows span {} period counts ({}). calmar's denominator grows "
              "with series length, so it is not comparable across them."
              .format(len(lengths), ", ".join(str(n) for n in lengths)))

    if missing:
        # Named, never dropped silently: a table that quietly omits an arm reads as
        # "this is everything" when it is not.
        print("\nmissing (no artifacts): {}".format(", ".join(missing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
