"""
Everything a results report computes, with nothing about how it is drawn.

Two renderers read this module -- `report_pdf` for the archival document and
`report_html` for the interactive one -- and they must never disagree about a
number. Keeping the arithmetic here is what makes that structural rather than
a matter of remembering to update both.

Four conventions are enforced here rather than left to the reader, because each
one has already produced a wrong conclusion in this repo:

  ONE TARGET PER TABLE. CORR on the 60-day label is structurally higher than on
  the 20-day one, so a mixed table ranks the LABEL as much as the model. Runs
  are split into blocks by the target their forecast was actually scored on.

  ONE WINDOW PER BLOCK. Runs here are 510, 470, 253, 128 and 64 periods long,
  and calmar's denominator grows with series length while its numerator does
  not. Every number in a block is computed on the intersection of its rows'
  dates, so calmar is comparable by construction rather than by hope. Each row
  still carries its own length in n(own), and --window own reproduces
  results_table and the registry exactly.

  A DRAWDOWN BELONGS TO THE SERIES NAMED BESIDE IT. dd(corr) and dd(pay) are
  labelled separately and the payout chart carries its own underwater panel,
  because the payout drawdown runs 1.8-3.4x the CORR drawdown and adjacent
  columns read as one story.

  SCREENING IS NOT PRICING. The ensemble section averages payout SERIES;
  run_ensemble rank-averages PREDICTIONS. Its column says "screened payout" and
  the report carries the commands to price the shortlist for real.
"""

import argparse
import collections
import fnmatch
import glob
import os

import numpy as np
import pandas as pd

from src.analysis import slots as slotlib
from src.analysis.experiment_db import study_of
from src.analysis.ablation_report import newey_west
from src.analysis.corr_heatmap import within_period_rank_corr
from src.analysis.ensemble_screen import beat_their_best, rank, screen
from src.analysis.results_table import (degenerate_periods, payout_series,
                                        row_from_frame, scored_target,
                                        short_target, target_of)
from src.live import registry

#: Newey-West lag by target. A 20-day label on daily dates overlaps ~95% between
#: consecutive periods, a 60-day label ~98%, so 510 periods are roughly 26
#: effective independent windows and 470 sixty-day periods roughly 8.
LAG_BY_TARGET = {"target_binned_return_20": 20, "target_binned_return_60": 60}

#: What the correlation numbers mean on this panel, measured across the studies.
CALIBRATION = [
    "ridge variants against each other      0.883 - 0.926   effectively one model",
    "lgb against the ridge book             0.705 - 0.751",
    "lgb classification against ridge       0.253 - 0.293",
    "the torch LSTM against everything      0.061 - 0.183   genuinely orthogonal",
]

#: Printed at the top of every ensemble section, in both renderers.
SCREEN_WARNING = [
    "SCREEN, NOT A PRICE. These rows average the members' payout SERIES.",
    "run_ensemble rank-averages their PREDICTIONS. On the one combination",
    "carried through, the two disagreed badly: screened 0.0903 payout /",
    "0.23 drawdown, actual 0.0977 / 1.31. Rank with this, then price the",
    "shortlist with the commands below before quoting a number.",
]

#: The summary table's columns, in order, with the format each cell takes.
COLUMNS = [
    ("run", "run", None), ("exp", "exp", None), ("kind", "kind", None),
    ("tgt", "target", None),
    ("n", "n", "d"), ("n(own)", "n(own)", "d"),
    ("CORR", "CORR", ".5f"), ("MMC", "MMC", ".5f"),
    ("c.sharpe", "c.sharpe", ".2f"), ("dd(corr)", "dd(corr)", ".2f"),
    ("payout", "payout", ".5f"), ("p.sharpe", "p.sharpe", ".2f"),
    ("dd(pay)", "dd(pay)", ".2f"), ("calmar", "calmar", ".3f"),
    ("dpay", "dpay", "+.6f"), ("t", "t", "+.2f"), ("live", "live", None),
]

Arm = collections.namedtuple("Arm", [
    "label", "model", "run", "kind", "target", "frame", "payout",
    "staked", "live_status", "note",
])

def short_label(label):
    """
    A slot name short enough for a heatmap tick.

    Full names are kept everywhere else: the tables have room for them and a
    reader should not have to learn an abbreviation to read the ranking.

    :param label: arm label
    :type label: str
    :return: the compact form
    :rtype: str
    """
    if label.startswith("CURUPIRA_SACI_"):
        return "CS_" + label[len("CURUPIRA_SACI_"):]
    if label.startswith("CURUPIRA_"):
        return "C" + label[len("CURUPIRA_"):]
    if label.startswith("SACI_"):
        return "S" + label[len("SACI_"):]
    return label


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def read_per_period(tournament, model, run):
    """
    A run's per-period frame, indexed by date when it has one.

    Fifty-five files in results/per_period were written before the index label
    was set and carry no date column. They load, but they cannot be aligned with
    anything, so callers that need a common window drop them by name.

    :param tournament: e.g. "crypto"
    :type tournament: str
    :param model: model tree
    :type model: str
    :param run: run name
    :type run: str
    :return: the frame, or None if the file is missing
    :rtype: pd.DataFrame or None
    """
    path = os.path.join("results", "per_period", tournament, model, "%s.csv" % run)
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    return frame.set_index("date") if "date" in frame.columns else frame


def expand_runs(tournament, default_model, patterns, exclude=()):
    """
    Resolve run patterns to (model, run) pairs.

    Globs are expanded against the metrics tree. A pattern that matches nothing
    is returned as missing rather than dropped: a table that quietly omits an arm
    reads as "this is everything" when it is not.

    :param tournament: e.g. "crypto"
    :type tournament: str
    :param default_model: tree to use for bare run names
    :type default_model: str
    :param patterns: run names, "model/run" addresses, or globs
    :type patterns: [str]
    :param exclude: patterns to drop after expansion
    :type exclude: [str]
    :return: (pairs, unmatched patterns)
    :rtype: ([(str, str)], [str])
    """
    pairs, missing = [], []
    for pattern in patterns:
        model, name = (pattern.split("/", 1) if "/" in pattern
                       else (default_model, pattern))
        base = os.path.join("results", "metrics", tournament, model)
        hits = sorted(os.path.basename(p)[:-5]
                      for p in glob.glob(os.path.join(base, name + ".json")))
        if not hits:
            missing.append(pattern)
            continue
        pairs.extend((model, hit) for hit in hits)

    dropped = set()
    for pattern in exclude:
        name = pattern.split("/", 1)[-1]
        dropped.update(run for _, run in pairs
                       if fnmatch.fnmatch(run, name))
    pairs = [p for p in pairs if p[1] not in dropped]

    seen, unique = set(), []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique, missing


def proposed_arms(tournament, pairs):
    """
    The experiment runs, as arms.

    :param tournament: e.g. "crypto"
    :type tournament: str
    :param pairs: (model, run) pairs
    :type pairs: [(str, str)]
    :return: (arms, [(label, why it was dropped)])
    :rtype: ([Arm], [(str, str)])
    """
    arms, dropped = [], []
    for model, run in pairs:
        frame = read_per_period(tournament, model, run)
        if frame is None:
            dropped.append(("%s/%s" % (model, run), "no per-period file"))
            continue
        arms.append(Arm(label=run, model=model, run=run, kind="prop",
                        target=scored_target(tournament, model, run),
                        frame=frame, payout=payout_series(frame).dropna(),
                        staked=False, live_status="-", note=None))
    return arms, dropped


def slot_arms(which, tournament):
    """
    The deployed slots, as arms.

    :param which: "all", "on-account", "staked" or "none"
    :type which: str
    :param tournament: e.g. "crypto"
    :type tournament: str
    :return: (arms, [(name, why it was dropped)])
    :rtype: ([Arm], [(str, str)])
    """
    resolved, dropped = slotlib.slot_rows(which, tournament)
    arms = []
    for slot in resolved:
        frame = read_per_period(tournament, slot.model, slot.run)
        arms.append(Arm(label=slot.name, model=slot.model, run=slot.run,
                        kind=slot.kind, target=slot.target, frame=frame,
                        payout=slot.payout, staked=slot.staked,
                        live_status=slot.live_status, note=slot.note))
    return arms, dropped


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #

class Block(object):
    """One target's arms, all measured on one window."""

    def __init__(self, target, arms, window, lag=None):
        self.target = target
        self.arms = arms
        self.window = window
        self.lag = lag or LAG_BY_TARGET.get(target, 20)
        self.undated = [a for a in arms if not self._dated(a)]
        dated = [a for a in arms if self._dated(a)]

        if window == "common" and dated:
            index = dated[0].payout.index
            for arm in dated[1:]:
                index = index.intersection(arm.payout.index)
            self.index = index.sort_values()
            self.arms = dated
        else:
            self.index = None

        self.baseline = None
        self.rejected = []

    @staticmethod
    def _dated(arm):
        return arm.payout.index.dtype == object or isinstance(
            arm.payout.index, pd.DatetimeIndex)

    def slice(self, arm):
        """The arm's frame restricted to this block's window."""
        if self.index is None:
            return arm.frame
        return arm.frame.loc[arm.frame.index.isin(self.index)]

    def payout(self, arm):
        """The arm's payout series on this block's window."""
        if self.index is None:
            return arm.payout
        return arm.payout.reindex(self.index).dropna()

    def dead_arms(self):
        """
        Arms that spent part of this window predicting one constant value.

        Surfaced separately because it is a model FAILURE wearing the costume of
        a weak score: the arm still produces a row, and before these periods
        were charged rather than dropped it could out-rank the very controls it
        lost to. See results_table.degenerate_periods().

        :return: [(label, dead periods, periods in the window)]
        :rtype: [(str, int, int)]
        """
        out = []
        for arm in self.arms:
            frame = self.slice(arm)
            if frame is None or "corr" not in frame:
                continue
            dead = int(degenerate_periods(frame).sum())
            if dead:
                out.append((arm.label, dead, len(frame)))
        return out

    def rows(self):
        """One row per arm, with the block's extras attached."""
        base = self.payout(self.baseline) if self.baseline else None
        out = []
        for arm in self.arms:
            row = row_from_frame(arm.label, self.slice(arm), target=self.target)
            row["kind"] = arm.kind + ("$" if arm.staked else "")
            # Which numbered study wrote this run, from its prefix. A deployed
            # slot's backtest belongs to no study and shows "-"; a scout is
            # marked so it can never be read as a full-resolution arm.
            study, is_scout = study_of(arm.run)
            row["exp"] = "-" if study is None else (
                study + "s" if is_scout else study)
            row["n(own)"] = len(arm.payout)
            row["live"] = arm.live_status
            row["dpay"], row["t"] = None, None
            if base is not None and arm is not self.baseline:
                paired = (self.payout(arm) - base).dropna()
                if len(paired) > 2:
                    mean, _, t = newey_west(paired, lag=self.lag)
                    row["dpay"], row["t"] = float(mean), float(t)
            out.append(row)
        return out


def build_blocks(arms, target_filter, window, lag):
    """
    Split arms into one block per scored target.

    :param arms: every arm in the report
    :type arms: [Arm]
    :param target_filter: "auto", "all", "20" or "60"
    :type target_filter: str
    :param window: "common" or "own"
    :type window: str
    :param lag: Newey-West lag override, or None to follow the target
    :type lag: int or None
    :return: blocks, longest first
    :rtype: [Block]
    """
    by_target = collections.OrderedDict()
    for arm in arms:
        by_target.setdefault(arm.target or "unknown", []).append(arm)

    if target_filter in ("20", "60"):
        want = "target_binned_return_%s" % target_filter
        by_target = collections.OrderedDict(
            (k, v) for k, v in by_target.items() if k == want)

    blocks = [Block(target, group, window, lag)
              for target, group in by_target.items()]
    return sorted(blocks, key=lambda b: -len(b.arms))


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #

def pick_baseline(block, policy, override=None):
    """
    Choose the arm every other row is measured against, and say why.

    The naive reading of "best single model among the submitted" is a trap: on
    this data ranking single-estimator slots by payout puts SACI_03 first, and
    SACI_03 is not on the Numerai account and is scored on the 60-day label. So
    eligibility is checked explicitly and the runners-up are reported with the
    reason each was passed over, rather than a winner being asserted.

    :param block: the block to choose within
    :type block: Block
    :param policy: "best-single-submitted", "best-staked", "best-any-submitted"
                   or "none"
    :type policy: str
    :param override: an arm label to use instead
    :type override: str or None
    :return: (baseline arm or None, [(label, reason)])
    :rtype: (Arm or None, [(str, str)])
    """
    if policy == "none":
        return None, []

    if override:
        for arm in block.arms:
            if arm.label == override:
                return arm, [("--baseline", "overridden by hand: %s" % override)]
        return None, [("--baseline", "%s is not in this block" % override)]

    _, on_account = slotlib.live_state()
    candidates, rejected = [], []
    for arm in block.arms:
        if arm.kind == "prop":
            continue
        why = None
        if policy == "best-single-submitted" and not arm.kind.startswith("sub"):
            why = "a blend of other slots, not a single model"
        elif policy == "best-staked" and not arm.staked:
            why = "carries no stake"
        if why is None and arm.label in registry.NOT_SUBMITTED:
            why = "component by design, never uploaded"
        if why is None and on_account is not None and arm.label not in on_account:
            why = "not a slot on the Numerai account"
        (rejected.append((arm.label, why)) if why else candidates.append(arm))

    if not candidates:
        # Rather than silently borrowing a baseline from the other target block,
        # fall back to the best submitted arm in THIS block and say that is what
        # happened. A 20-day baseline under a 60-day table would be worse than
        # no baseline at all.
        fallback = [a for a in block.arms if a.kind != "prop"]
        if not fallback:
            return None, rejected
        best = max(fallback, key=lambda a: block.payout(a).mean())
        rejected = [(label, why) for label, why in rejected if label != best.label]
        rejected.insert(0, (best.label, "USED ANYWAY: nothing in this block meets "
                                        "the %s policy" % policy))
        return best, rejected

    best = max(candidates, key=lambda a: block.payout(a).mean())
    rejected = sorted(rejected, key=lambda r: -block.payout(
        next(a for a in block.arms if a.label == r[0])).mean())
    return best, rejected


# --------------------------------------------------------------------------- #
# correlation
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# correlation
# --------------------------------------------------------------------------- #

def load_predictions_common(tournament, arms, min_shared_dates):
    """
    Predictions for a set of arms, on the panel they actually share.

    corr_heatmap.load_predictions raises SystemExit on an index mismatch, which
    is right there -- every arm in its collections is one panel -- and fatal
    here. The xgb, catboost and mlp runs live on a 128-date, 744-symbol panel
    that is a strict SUBSET of the main one, so an intersection with them
    succeeds quietly and silently cuts the matrix from 470 dates to 128. Drop
    the shortest arm until the shared panel is big enough, and name every drop.

    :param tournament: e.g. "crypto"
    :type tournament: str
    :param arms: arms to load
    :type arms: [Arm]
    :param min_shared_dates: refuse to shrink the panel below this
    :type min_shared_dates: int
    :return: (wide frame, n dates, [(label, why)])
    :rtype: (pd.DataFrame, int, [(str, str)])
    """
    series, excluded = {}, []
    for arm in arms:
        path = os.path.join("results", "forecast", tournament, arm.model,
                            "%s.csv" % arm.run)
        if not os.path.exists(path):
            excluded.append((arm.label, "no forecast file"))
            continue
        frame = pd.read_csv(path, usecols=["symbol", "date", "prediction"])
        frame = frame.set_index(["symbol", "date"]).sort_index()
        series[arm.label] = frame["prediction"].astype("float32")

    while len(series) >= 2:
        shared = None
        for values in series.values():
            shared = values.index if shared is None else shared.intersection(values.index)
        n_dates = len(set(shared.get_level_values("date")))
        if n_dates >= min_shared_dates or len(series) == 2:
            wide = pd.DataFrame({k: v.reindex(shared) for k, v in series.items()})
            return wide, n_dates, excluded
        shortest = min(series, key=lambda k: len(
            set(series[k].index.get_level_values("date"))))
        excluded.append((shortest, "covers only %d dates; keeping it would cut the "
                                   "shared panel to %d"
                         % (len(set(series[shortest].index.get_level_values("date"))),
                            n_dates)))
        del series[shortest]

    return pd.DataFrame(), 0, excluded


def score_series_matrix(block):
    """
    Correlation of the per-period CORR series -- the ensembling ceiling.

    Built here rather than with corr_heatmap.score_series_corr for two reasons:
    that function does set_index("date"), which raises on the date-less files,
    and its pairwise-complete .corr() lets different cells use different n.

    :param block: the block whose arms to correlate
    :type block: Block
    :return: (matrix, n periods)
    :rtype: (pd.DataFrame, int)
    """
    columns = {arm.label: block.slice(arm)["corr"] for arm in block.arms}
    frame = pd.DataFrame(columns).dropna()
    return frame.corr(), len(frame)


# --------------------------------------------------------------------------- #
# rendering: text
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# the canonical text table -- printed to stdout by every renderer
# --------------------------------------------------------------------------- #

HEAD = ("{:<%d} {:>5} {:<11} {:>4} {:>5} {:>6} | {:>9} {:>9} | {:>8} {:>8} | "
        "{:>9} {:>8} {:>8} | {:>7} | {:>9} {:>6} | {:<8}")

def cell(value, spec):
    """One table cell, with the repo's missing/infinite conventions."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    return format(value, spec)


def render_block(block, sort_by="payout"):
    """
    The block's table.

    :param block: the block
    :type block: Block
    :param sort_by: column to sort descending by, or None
    :type sort_by: str or None
    :return: header lines and body lines
    :rtype: ([str], [str])
    """
    rows = block.rows()
    if sort_by:
        rows = sorted(rows, key=lambda r: (r.get(sort_by) if r.get(sort_by) is not None
                                           and np.isfinite(r[sort_by]) else -np.inf),
                      reverse=True)

    width = max([len(r["run"]) for r in rows] + [22])
    fmt = HEAD % width
    header = fmt.format("run", "exp", "kind", "tgt", "n", "n(own)", "CORR", "MMC",
                        "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)",
                        "calmar", "dpay", "t", "live")

    body = []
    for r in rows:
        body.append(fmt.format(
            r["run"], r.get("exp", "-"), r["kind"], short_target(r.get("target")),
            cell(r.get("n"), "d"), cell(r.get("n(own)"), "d"),
            cell(r["CORR"], ".5f"), cell(r["MMC"], ".5f"),
            cell(r["c.sharpe"], ".2f"), cell(r["dd(corr)"], ".2f"),
            cell(r["payout"], ".5f"), cell(r["p.sharpe"], ".2f"),
            cell(r["dd(pay)"], ".2f"), cell(r["calmar"], ".3f"),
            cell(r.get("dpay"), "+.6f"), cell(r.get("t"), "+.2f"),
            r.get("live", "-")))
    return [header, "-" * len(header)], body


def block_caption(block):
    """The one-paragraph statement of what the block's numbers mean."""
    lines = []
    if block.index is not None and len(block.index):
        lines.append("window: %d common dates, %s to %s. Every number is computed "
                     "on this window; n(own) is each row's native length."
                     % (len(block.index), block.index[0], block.index[-1]))
        lines.append("Common-window figures do not equal the registry's, which are "
                     "on each slot's own window. Re-run with --window own to "
                     "reproduce those exactly.")
    else:
        lines.append("window: each row on its own periods. calmar is NOT comparable "
                     "across rows of different n.")
    lines.append("dpay/t: paired Newey-West delta of payout against %s, lag %d, "
                 "inner join on periods."
                 % (block.baseline.label if block.baseline else "(no baseline)",
                    block.lag))
    dead = block.dead_arms()
    if dead:
        lines.append("DEGENERATE: the arm(s) below predicted ONE constant value for "
                     "every symbol on part of this window. Those periods are scored "
                     "as the 0 they earned, not dropped -- dropping them would rank "
                     "the arm only on the days it worked. Read this before payout:")
        for label, n, total in dead:
            lines.append("    %-20s %d of %d periods" % (label, n, total))
    if block.rejected:
        lines.append("baseline: %s. Passed over, best first:" % (
            block.baseline.label if block.baseline else "NONE ELIGIBLE"))
        for label, why in block.rejected[:6]:
            lines.append("    %-20s %s" % (label, why))
    lines.append("kind: prop=proposed run  sub=submitted single  blend=feature "
                 "blend  slotblend=blend of slots  $=staked")
    lines.append("dd(corr) is the drawdown of the CORR series; dd(pay) is the "
                 "drawdown of the payout series. They are different series and "
                 "the payout one runs 1.8-3.4x larger.")
    return lines


# --------------------------------------------------------------------------- #
# rendering: pdf pages
# --------------------------------------------------------------------------- #


def price_commands(block, rows, price_top):
    """
    The run_ensemble commands that would price the top screened combinations.

    run_ensemble takes the target column from its FIRST member, which is why
    eoe_c04_bq2_t60 is 20-day-scored despite holding a 60-day member. The flag
    is emitted explicitly whenever the block is not on the 20-day label.

    :param block: the block
    :type block: Block
    :param rows: rows from ensemble_screen.screen
    :type rows: [dict]
    :param price_top: how many commands to emit
    :type price_top: int
    :return: the commands, as lines
    :rtype: [str]
    """
    by_label = {arm.label: arm for arm in block.arms}
    if not rows:
        return []

    lines = ["Price the shortlist -- these are real blends, not screens:", ""]
    for index, row in enumerate(rank(rows, "sharpe")[:price_top], start=1):
        arms = [by_label[m] for m in row["members"]]
        out_tree = arms[0].model
        addressed = [a.run if a.model == out_tree else "%s/%s" % (a.model, a.run)
                     for a in arms]
        command = ("python3 -m src.run_ensemble crypto %s \\\n        %s \\\n"
                   "        --run-name rep_%s_e%d"
                   % (out_tree, " ".join(addressed),
                      short_target(block.target), index))
        if block.target and block.target != "target_binned_return_20":
            command += " \\\n        --target %s" % block.target
        lines += ["  # " + " + ".join(row["members"]), "  " + command, ""]
    return lines


def matrix_arms(block, max_arms):
    """
    Which arms go into a correlation matrix, and what that cut left out.

    The proposed runs are what the report is about, so they are never the ones
    cut: they and the baseline are kept first, and the cap is spent on the
    highest-paying slots after that. Sorting by payout alone would drop every
    proposal, since the deployed blends outrank them all.

    :param block: the block
    :type block: Block
    :param max_arms: legibility cap
    :type max_arms: int
    :return: (arms to use, [(label, why)])
    :rtype: ([Arm], [(str, str)])
    """
    ordered = sorted(block.arms, key=lambda a: -block.payout(a).mean())
    proposals = [a for a in ordered if a.kind == "prop"]
    others = [a for a in ordered if a.kind != "prop"]

    keep = []
    if block.baseline is not None and block.baseline in ordered:
        keep.append(block.baseline)
    # Reserve half the room for the deployed book. Taking proposals first and
    # filling afterwards looks right until a study has twenty arms, at which
    # point it crowds out the very slots the study is being compared against
    # and the matrix says nothing about deployment.
    reserved = min(len(others), max_arms // 2)
    for arm in proposals:
        if len(keep) >= max_arms - reserved:
            break
        if arm not in keep:
            keep.append(arm)
    for group in (others, ordered):
        for arm in group:
            if len(keep) >= max_arms:
                break
            if arm not in keep:
                keep.append(arm)
    excluded = [(a.label, "beyond the %d-arm cap" % max_arms)
                for a in ordered if a not in keep]
    return [a for a in ordered if a in keep], excluded


def correlation_matrices(tournament, block, max_arms, min_shared_dates):
    """
    Both correlation matrices for a block.

    They answer different questions and routinely disagree: prediction
    correlation reads "are these two models ranking the same coins the same
    way", score correlation reads "do they have good and bad periods together",
    and only the second one caps ensembling.

    :param tournament: e.g. "crypto"
    :type tournament: str
    :param block: the block
    :type block: Block
    :param max_arms: legibility cap
    :type max_arms: int
    :param min_shared_dates: refuse to shrink the panel below this
    :type min_shared_dates: int
    :return: (predictions, scores, n dates, n periods, [(label, why)])
    :rtype: (pd.DataFrame or None, pd.DataFrame or None, int, int, [(str, str)])
    """
    arms, excluded = matrix_arms(block, max_arms)
    if len(arms) < 2:
        return None, None, 0, 0, excluded

    wide, n_dates, dropped = load_predictions_common(tournament, arms, min_shared_dates)
    excluded.extend(dropped)
    if wide.empty or wide.shape[1] < 2:
        return None, None, 0, 0, excluded

    kept = [a for a in arms if a.label in wide.columns]
    predictions = within_period_rank_corr(wide)
    scores, n_periods = score_series_matrix(
        Block(block.target, kept, block.window, block.lag))
    return predictions, scores, n_dates, n_periods, excluded


def chart_frame(block, min_common_dates):
    """
    The per-period payout of every chartable arm, on one shared date index.

    Per period, NOT cumulative: a reader who narrows the date range wants the
    curve to restart from the window they chose, and a cumulative series cannot
    be re-based after the fact. Both renderers sum it themselves.

    The inner join happens BEFORE any summing: cumulative curves computed on
    different denominators are not comparable, and the divergence looks exactly
    like alpha.

    :param block: the block
    :type block: Block
    :param min_common_dates: drop an arm sharing fewer dates than this
    :type min_common_dates: int
    :return: (frame indexed by date, [(label, why)])
    :rtype: (pd.DataFrame, [(str, str)])
    """
    if block.baseline is None:
        return pd.DataFrame(), [("(chart)", "no eligible baseline in this block")]

    base = block.payout(block.baseline)
    keep, excluded = [], []
    for arm in block.arms:
        shared = block.payout(arm).index.intersection(base.index)
        if len(shared) < min_common_dates:
            excluded.append((arm.label, "shares only %d dates with the baseline"
                             % len(shared)))
            continue
        keep.append(arm)
    return pd.DataFrame({a.label: block.payout(a) for a in keep}).dropna(), excluded


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def common_parser(prog, description):
    """
    The arguments every renderer shares.

    :param prog: the `python3 -m ...` invocation
    :type prog: str
    :param description: one-line help
    :type description: str
    :return: the parser, for the caller to add output flags to
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--tournament", default="crypto")
    parser.add_argument("--model", default="lgb",
                        help="model tree for run names that do not carry one")
    parser.add_argument("--runs", nargs="+", default=[],
                        help="run names, 'model/run' addresses, or globs")
    parser.add_argument("--exclude", nargs="+", default=[])
    parser.add_argument("--submitted", default="all",
                        choices=("all", "on-account", "staked", "none"))
    parser.add_argument("--baseline-policy", default="best-single-submitted",
                        choices=("best-single-submitted", "best-staked",
                                 "best-any-submitted", "none"))
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--target", default="auto", choices=("auto", "all", "20", "60"))
    parser.add_argument("--window", default="common", choices=("common", "own"))
    parser.add_argument("--lag", type=int, default=None,
                        help="override the target's Newey-West lag")
    parser.add_argument("--sort-by", default="payout",
                        choices=("CORR", "MMC", "payout", "p.sharpe", "calmar", "none"))
    parser.add_argument("--ens-size", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--ens-top", type=int, default=8)
    parser.add_argument("--price-top", type=int, default=3)
    parser.add_argument("--corr-max-arms", type=int, default=12)
    parser.add_argument("--ens-max-arms", type=int, default=12,
                        help="arms entering the ensemble screen; combinations "
                             "grow as the cube of this")
    parser.add_argument("--min-common-dates", type=int, default=400)
    parser.add_argument("--title", default="Results board")
    parser.add_argument("--no-corr", action="store_true")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--no-chart", action="store_true")
    return parser


def build(args):
    """
    Assemble every block, without rendering anything.

    :param args: parsed arguments
    :type args: argparse.Namespace
    :return: (blocks, missing patterns, [(label, why)])
    :rtype: ([Block], [str], [(str, str)])
    """
    pairs, missing = expand_runs(args.tournament, args.model, args.runs, args.exclude)
    proposed, dropped = proposed_arms(args.tournament, pairs)
    submitted, slot_dropped = slot_arms(args.submitted, args.tournament)
    dropped += slot_dropped

    # A slot and a proposed run can be the same file under two names -- SACI_01
    # IS lgb/fe_lgb_reg_interact. Keep the slot's name, so the row says what is
    # deployed rather than what it was called during the study.
    by_run = {(a.model, a.run): a.label for a in submitted}
    absorbed = [(a.label, "same series as slot %s; listed under the slot name"
                 % by_run[(a.model, a.run)])
                for a in proposed if (a.model, a.run) in by_run]
    proposed = [a for a in proposed if (a.model, a.run) not in by_run]
    dropped += absorbed

    blocks = build_blocks(proposed + submitted, args.target, args.window, args.lag)
    for block in blocks:
        block.baseline, block.rejected = pick_baseline(
            block, args.baseline_policy, args.baseline)
        dropped += [(a.label, "no date column in its per-period file; cannot be "
                              "aligned to a common window") for a in block.undated]
    return blocks, missing, dropped
