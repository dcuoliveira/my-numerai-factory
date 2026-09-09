"""
The archival PDF of a results report.

    python3 -m src.analysis.report_pdf --model lgb --runs e05_cumlag3

Writes results/reports/<tournament>/report_<stamp>.pdf and prints the summary
table to stdout from the same rows, so the document and the chat message cannot
disagree. Sections: the ranked table, both correlation matrices, screened
ensembles, and cumulative backtested payout against the best single submitted
model.

Every number comes from `report_data`, which `report_html` also reads -- this
module only draws. For a version you can sort, filter and zoom, use
`report_html`; for one you can attach to a mail and read in five years, use
this.

Rendered with matplotlib's PdfPages rather than an HTML-to-PDF route because two
of the four sections are already matplotlib figures: corr_heatmap.draw takes an
axes, so it lands on a PDF page unchanged. Every other renderer would need those
two pictures rasterised to PNG and re-embedded, which is a second code path for
the same image and a resolution decision nobody wants to make twice.
"""

import datetime
import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.analysis import slots as slotlib
from src.analysis.corr_heatmap import draw
from src.analysis.ensemble_screen import beat_their_best, rank, screen
from src.analysis.report_data import (CALIBRATION, Block, SCREEN_WARNING, build,
                                      block_caption, chart_frame, common_parser,
                                      correlation_matrices, price_commands,
                                      render_block, short_label)
from src.analysis.results_table import short_target

#: A4 landscape. Tables are paginated onto it rather than shrunk to fit -- a
#: 40-row table on one page renders at 4pt and still looks fine in a thumbnail.
PAGE = (11.69, 8.27)
ROWS_PER_PAGE = 30
MONO = {"family": "monospace", "fontsize": 7.0}


class Pages(object):
    """
    The open document: every figure goes through here.

    Exists so --also-png cannot drift from the PDF. A second savefig call
    scattered through the section builders would eventually miss one, and a
    PNG set that is quietly missing a page is worse than no PNG set.
    """

    def __init__(self, pdf, png_stem=None, dpi=140):
        self.pdf = pdf
        self.png_stem = png_stem
        self.dpi = dpi
        self.count = 0

    def save(self, figure):
        """Write one figure to the PDF, and to a PNG when asked."""
        self.count += 1
        self.pdf.savefig(figure)
        if self.png_stem:
            figure.savefig("%s_p%02d.png" % (self.png_stem, self.count),
                           dpi=self.dpi, bbox_inches="tight")
        plt.close(figure)


def text_page(pdf, title, lines, subtitle=None, mono=True, fontsize=None):
    """
    One page of text, monospaced by default.

    :param pdf: the open PdfPages
    :type pdf: PdfPages
    :param title: page heading
    :type title: str
    :param lines: body lines
    :type lines: [str]
    :param subtitle: optional line under the heading
    :type subtitle: str or None
    :param mono: whether to use the monospace family
    :type mono: bool
    :param fontsize: override the default size
    :type fontsize: float or None
    """
    figure = plt.figure(figsize=PAGE)
    figure.text(0.04, 0.95, title, fontsize=13, weight="bold", va="top")
    if subtitle:
        figure.text(0.04, 0.915, subtitle, fontsize=8.5, va="top", color="#444444")

    style = dict(MONO) if mono else {"family": "sans-serif", "fontsize": 9}
    if fontsize:
        style["fontsize"] = fontsize
    figure.text(0.04, 0.885 if subtitle else 0.90, "\n".join(lines),
                va="top", linespacing=1.45, **style)
    pdf.save(figure)

def table_pages(pdf, title, header, body, caption, rows_per_page=ROWS_PER_PAGE):
    """
    A table, paginated rather than shrunk.

    :param pdf: the open PdfPages
    :type pdf: PdfPages
    :param title: page heading
    :type title: str
    :param header: header lines, repeated on every page
    :type header: [str]
    :param body: the rows
    :type body: [str]
    :param caption: lines printed under the last page's table
    :type caption: [str]
    :param rows_per_page: rows before spilling to a new page
    :type rows_per_page: int
    """
    pages = [body[i:i + rows_per_page] for i in range(0, len(body), rows_per_page)] or [[]]
    for number, chunk in enumerate(pages, start=1):
        lines = list(header) + chunk
        if number == len(pages):
            lines += [""] + caption
        text_page(pdf, title, lines,
                  subtitle=("page %d of %d" % (number, len(pages))
                            if len(pages) > 1 else None))

def correlation_page(pdf, tournament, block, max_arms, min_shared_dates):
    """
    Both correlation matrices, side by side.

    :return: [(label, why)] for anything left out
    :rtype: [(str, str)]
    """
    predictions, scores, n_dates, n_periods, excluded = correlation_matrices(
        tournament, block, max_arms, min_shared_dates)
    if predictions is None:
        return excluded

    predictions = predictions.rename(index=short_label, columns=short_label)
    scores = scores.rename(index=short_label, columns=short_label)

    figure, axes = plt.subplots(1, 2, figsize=PAGE)
    figure.subplots_adjust(left=0.10, right=0.90, top=0.86, bottom=0.26,
                           wspace=0.32)
    draw(axes[0], predictions,
         "Prediction correlation\n(mean within-period rank corr, %d dates)" % n_dates,
         vmin=float(predictions.to_numpy().min()))
    image = draw(axes[1], scores,
                 "Score correlation\n(per-period CORR series, n=%d)" % n_periods,
                 vmin=float(scores.to_numpy().min()))
    figure.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    figure.suptitle("Correlation -- %s" % short_target(block.target), fontsize=13)
    figure.text(0.04, 0.08, "What these numbers mean on this panel:\n  "
                + "\n  ".join(CALIBRATION)
                + "\nAn arm at 0.9 with a better mean is a worse slot than an arm "
                  "at 0.2 with a similar one.",
                fontsize=6.5, family="monospace", va="bottom", color="#333333")
    pdf.save(figure)
    return excluded


def ensemble_section(block, sizes, top, price_top):
    """
    The screened ensembles, as text.

    :param block: the block to screen within
    :type block: Block
    :param sizes: combination sizes
    :type sizes: (int,)
    :param top: rows per ranking
    :type top: int
    :param price_top: how many run_ensemble commands to emit
    :type price_top: int
    :return: (the section's lines, the screened rows)
    :rtype: ([str], [dict])
    """
    members = {arm.label: block.payout(arm) for arm in block.arms}
    rows, solo, n_periods = screen(members, tuple(sizes), block.lag)
    if not rows:
        return ["fewer than two members share periods here -- nothing to screen."], []

    lines = list(SCREEN_WARNING) + [
        "",
        "%d members over %d shared periods, sizes %s"
        % (len(solo), n_periods, ", ".join(str(s) for s in sizes)),
        "",
    ]
    head = ("%-54s %14s %8s %8s %11s %6s %5s"
            % ("ensemble", "screened payout", "sharpe", "maxdd", "d vs best", "t", "rho"))
    for title, key in (("by sharpe", "sharpe"), ("by max drawdown", "maxdd")):
        lines += ["=== %s ===" % title, head]
        for row in rank(rows, key)[:top]:
            lines.append("%-54s %14.6f %8.4f %8.4f %+11.6f %6.2f %5.2f"
                         % (" + ".join(row["members"]), row["payout"], row["sharpe"],
                            row["maxdd"], row["delta"], row["t"], row["rho"]))
        lines.append("")

    beat = beat_their_best(rows)
    lines.append("%d of %d combinations beat their own best member on payout at "
                 "|t| > 2." % (len(beat), len(rows)))
    if not beat:
        lines += ["  None. Equal-weight averaging dilutes the mean by construction,",
                  "  so read these as a RISK decision (sharpe, drawdown, correlation),",
                  "  not a level one. That null is the finding, not a missing result."]
    return lines, rows

def chart_page(pdf, block, min_common_dates, title):
    """
    Cumulative payout and its underwater curve, on one shared window.

    :return: [(label, why)] for anything left out
    :rtype: [(str, str)]
    """
    frame, excluded = chart_frame(block, min_common_dates)
    if frame.empty:
        return excluded

    keep = [a for a in block.arms if a.label in frame.columns]
    cumulative = frame.cumsum()
    underwater = cumulative.cummax() - cumulative
    dates = pd.to_datetime(frame.index)

    figure, axes = plt.subplots(2, 1, figsize=PAGE, sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1]})
    palette = plt.get_cmap("viridis")
    proposals = [a for a in keep if a.kind == "prop"]

    for arm in keep:
        if arm is block.baseline or arm.kind == "prop":
            continue
        axes[0].plot(dates, cumulative[arm.label], color="#cccccc", linewidth=0.8,
                     zorder=1)
    for index, arm in enumerate(proposals):
        colour = palette(0.15 + 0.7 * index / max(len(proposals) - 1, 1))
        axes[0].plot(dates, cumulative[arm.label], color=colour, linewidth=1.4,
                     zorder=3, label="%s  %.5f / %.2f / %.2f"
                     % (arm.label, frame[arm.label].mean(),
                        frame[arm.label].mean() / frame[arm.label].std(ddof=0),
                        underwater[arm.label].max()))
        axes[1].plot(dates, underwater[arm.label], color=colour, linewidth=1.0)

    label = block.baseline.label
    axes[0].plot(dates, cumulative[label], color="black", linewidth=2.4, zorder=4,
                 label="%s (baseline)  %.5f / %.2f / %.2f"
                 % (label, frame[label].mean(),
                    frame[label].mean() / frame[label].std(ddof=0),
                    underwater[label].max()))
    axes[1].plot(dates, underwater[label], color="black", linewidth=1.6)

    axes[0].set_title("%s\n%d common dates, %s to %s   (grey: other submitted slots)"
                      % (title, len(frame), frame.index[0], frame.index[-1]),
                      fontsize=11)
    axes[0].set_ylabel("cumulative payout")
    axes[0].legend(fontsize=6.5, loc="upper left", framealpha=0.9,
                   title="legend: mean / p.sharpe / dd(pay)", title_fontsize=6.5)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("underwater")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    pdf.save(figure)
    return excluded


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    """Parse the command line."""
    parser = common_parser("python3 -m src.analysis.report_pdf",
                           "The archival PDF of a results report.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--also-png", action="store_true",
                        help="write each page alongside the PDF as a PNG too")
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Build the report.

    :param argv: argument list, or None to read sys.argv
    :type argv: [str] or None
    :return: process exit code
    :rtype: int
    """
    args = parse_args(argv)
    blocks, missing, dropped = build(args)
    if not blocks:
        print("nothing to report: no run matched and no slot resolved.")
        if missing:
            print("unmatched: %s" % ", ".join(missing))
        return 1

    sort_by = None if args.sort_by == "none" else args.sort_by
    excluded = list(dropped)

    for block in blocks:
        header, body = render_block(block, sort_by)
        print("\n=== %s -- %d arms ===" % (short_target(block.target), len(block.arms)))
        print("\n".join(header + body))
        for line in block_caption(block):
            print("  " + line)

    if missing:
        print("\nmissing (no artifacts): %s" % ", ".join(missing))
    for label, why in excluded:
        print("  excluded: %s -- %s" % (label, why))

    if args.no_pdf:
        return 0

    out = args.out or os.path.join(
        "results", "reports", args.tournament, "report_%s.pdf"
        % datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    with PdfPages(out) as handle:
        pdf = Pages(handle, os.path.splitext(out)[0] if args.also_png else None)
        round_number, _ = slotlib.latest_round_report()
        cover = [
            "Generated %s UTC" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "Live state from round %s." % (round_number or "(none on disk)"),
            "",
            "Proposed runs : %s" % (", ".join(args.runs) or "(none)"),
            "Submitted book: %s" % args.submitted,
            "Baseline      : %s" % args.baseline_policy,
            "Window        : %s" % args.window,
            "",
            "Blocks in this report:",
        ]
        for block in blocks:
            cover.append("  %-6s %3d arms   lag %d   baseline %s"
                         % (short_target(block.target), len(block.arms), block.lag,
                            block.baseline.label if block.baseline else "(none)"))
        cover += [
            "",
            "Every payout figure is rebuilt per period as 0.1*corr + mmc and",
            "summarised once, never combined from the stored corr and mmc",
            "summaries: a sum's Sharpe is not a function of its parts' Sharpes.",
            "",
            "Backtest artifacts are out-of-sample predictions for historical",
            "dates. Nothing here is a realised live result -- no live score is",
            "stored locally, and only a handful of rounds have been submitted.",
        ]
        text_page(pdf, args.title, cover,
                  subtitle="experiment runs against the deployed book")

        for block in blocks:
            header, body = render_block(block, sort_by)
            table_pages(pdf, "Results -- %s" % short_target(block.target),
                        header, body, block_caption(block))

        for block in blocks:
            if not args.no_corr:
                excluded += correlation_page(pdf, args.tournament, block,
                                             args.corr_max_arms, args.min_common_dates)
            if not args.no_ensemble:
                lines, screened = ensemble_section(block, args.ens_size,
                                                   args.ens_top, args.price_top)
                lines += [""] + price_commands(block, screened, args.price_top)
                text_page(pdf, "Ensembles -- %s" % short_target(block.target), lines,
                          subtitle="screened on the payout series, ranked not priced")
            if not args.no_chart:
                excluded += chart_page(
                    pdf, block, args.min_common_dates,
                    "Cumulative backtested payout -- %s" % short_target(block.target))

        notes = ["Anything named here was left out of the tables above. A blank row",
                 "reads as 'tested and null' when it means 'never run', so nothing is",
                 "dropped silently.", ""]
        notes.append("Unmatched run patterns: %s"
                     % (", ".join(missing) if missing else "none"))
        notes.append("")
        notes.append("Excluded arms:")
        notes += (["  %-28s %s" % (label, why) for label, why in excluded]
                  or ["  none"])
        notes += ["", "Slots whose recorded figure differs from their series:"]
        for slot in slotlib.slot_rows("all", args.tournament)[0]:
            if not slot.matches_registry:
                notes += ["  %s" % slot.name] + [
                    "    " + slot.note[i:i + 110]
                    for i in range(0, len(slot.note or ""), 110)]
        notes += ["", "Live state, newest round on disk:"]
        for slot in slotlib.slot_rows("all", args.tournament)[0]:
            if slot.live_status not in ("ok", "-"):
                notes.append("  %-20s %s" % (slot.name, slot.live_status))
        text_page(pdf, "Notes", notes, subtitle="what is missing, and why")

    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
