# -*- coding: utf-8 -*-
"""
Add a model to the momentum book and say what it does to the top 3.

    python3 -m src.analysis.book_update CU_SA_IA_BO_MA_JA_01            # report only
    python3 -m src.analysis.book_update CU_SA_IA_BO_MA_JA_01 --write    # also add it to BOOK

The allocation routine this feeds is E55's momentum watch
(`python3 -m src.analysis.factor_momentum_stake --watch`). It ranks every model
in `factor_momentum_stake.BOOK` by its trailing-250 mean payout (0.1*corr + mmc)
over RESOLVED rounds only, i.e. after a 20-period shift. The top model is where
the stake goes ("top1 roll250", the E55 out-of-sample pick). The watch stores its
last ranking in `results/live/momentum_top1.json` and e-mails when top-1 changes.

This module answers one question when a new model joins: does the top 3 change?
It puts three rankings side by side:

  previous (watch)   the ranking the watch last stored. This is what the book
                     looked like before anyone touched it. The as-of date can be
                     older than today's panel.
  previous (same)    the book WITHOUT the candidate, ranked on the panel that
                     includes it. This is the clean counterfactual: same dates,
                     same live splice, so any difference from "recent" comes
                     from the candidate and nothing else.
  recent             the book WITH the candidate.

A difference between "previous (watch)" and "previous (same)" is drift since the
last watch (new resolved rounds or a shorter common window), not the candidate.
The report names which of the two caused each change.

Two things the module refuses or flags rather than quietly absorbing:

  * A candidate whose backtest is scored on the 60-day label is refused. Its
    payout is on another scale, so it would poison a cross-sectional ranking
    (that is why SACI_03 is not in BOOK).
  * The book is joined on COMMON dates (stake.per_period intersects them). A
    candidate with a shorter history shrinks the panel for every model, so the
    report prints the backtest date count before and after the join.

`--write` appends the candidate to the BOOK list in factor_momentum_stake.py, so
the next scheduled watch tracks it. It does not touch the watch's state file:
if the candidate changes top-1, the next watch run sends the "move the stake"
e-mail, which is the point of the watch.
"""

import argparse
import datetime
import html
import json
import os
import re

import numpy as np
import pandas as pd

from src.analysis import factor_momentum_stake as fm
from src.analysis.ablation_report import newey_west

STATE_PATH = "results/live/momentum_top1.json"
OUT_DIR = "results/reports/crypto/book_update"
BOOK_FILE = fm.__file__
PAYABLE_TARGET = "target_binned_return_20"


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def check_candidate(name):
    """Raise SystemExit unless `name` is a registry slot with a 20-day scored series."""
    from src.live import registry, stake
    if name not in registry.MODELS:
        raise SystemExit("%s is not in src/live/registry.MODELS -- define the slot "
                         "(spec + backtest_run) before adding it to the book" % name)
    target = registry.backtest_target(name)
    if target != PAYABLE_TARGET:
        raise SystemExit("%s is backtested on %s, not %s. Its payout is on another "
                         "scale and would distort the ranking, so it stays out of "
                         "the book" % (name, target, PAYABLE_TARGET))
    frames, index = stake.per_period([name])     # raises if no scored run
    return len(index)


# --------------------------------------------------------------------------- #
# rankings
# --------------------------------------------------------------------------- #

def rank_frame(panel, names, lookback=250, lag=fm.LAG):
    """
    One row per model: the watch's signal at the last date, its rank, and the
    trailing window's prop calmar and drawdown, all on RESOLVED rounds (the
    same shifted window the signal averages).
    """
    sub = panel[names]
    sig = fm.signal(sub, lookback=lookback, lag=lag).iloc[-1]
    window = sub.shift(lag).iloc[-lookback:]
    rows = {}
    for m in names:
        x = window[m].dropna()
        cum = x.cumsum()
        rows[m] = {"signal": float(sig.get(m, np.nan)),
                   "calmar": fm.prop_calmar(x),
                   "dd": float((cum.cummax() - cum).max()) if len(x) else np.nan,
                   "n": int(len(x))}
    df = pd.DataFrame(rows).T.sort_values("signal", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def paired_vs(panel, candidate, others, lookback=250, lag=fm.LAG):
    """Newey-West (lag 20) on candidate-minus-other payout over the resolved window."""
    window = panel.shift(lag).iloc[-lookback:]
    out = {}
    for o in others:
        if o == candidate:
            continue
        d = (window[candidate] - window[o]).dropna()
        if len(d) < 30:
            out[o] = (np.nan, np.nan, len(d))
            continue
        mean, _, t = newey_west(d.values, lag=fm.LAG)
        out[o] = (float(mean), float(t), len(d))
    return out


def stored_ranking(path=STATE_PATH):
    """The watch's last stored ranking (ordered dict name -> signal) and its as-of date."""
    if not os.path.exists(path):
        return None, None
    try:
        state = json.load(open(path))
    except (OSError, ValueError):
        return None, None
    return state.get("ranking") or {}, state.get("asof")


def coverage(candidates, book_names):
    """Backtest date coverage of the candidates against the book (no live rows)."""
    from src.live import stake
    _, book_idx = stake.per_period(book_names)
    out = {}
    for c in candidates:
        _, idx = stake.per_period([c])
        common = idx.intersection(book_idx)
        out[c] = {"first": str(idx.min().date()), "last": str(idx.max().date()),
                  "n": int(len(idx)), "n_common": int(len(common)),
                  "rows_missing_at_tail": int((book_idx > idx.max()).sum())}
    return {"book_first": str(book_idx.min().date()), "book_last": str(book_idx.max().date()),
            "book_n": int(len(book_idx)), "candidates": out}


def writable(cov, lookback=250, lag=fm.LAG):
    """(ok, reasons): may the candidates join BOOK without degrading the live watch?

    BOOK is joined on common dates, so a candidate whose backtest stops early moves
    every model's window back in time, and one that is too short leaves fewer rows
    than the signal needs."""
    reasons = []
    for c, v in cov["candidates"].items():
        if v["rows_missing_at_tail"]:
            reasons.append("%s's backtest ends %s, %d dates before the book's %s; joining it "
                           "would cut those dates from every model in the watch"
                           % (c, v["last"], v["rows_missing_at_tail"], cov["book_last"]))
        if v["n_common"] < lookback + lag:
            reasons.append("%s shares only %d dates with the book; the signal needs %d"
                           % (c, v["n_common"], lookback + lag))
    return not reasons, reasons


def compare(candidates, lookback=250, top=3, live=True, state_path=STATE_PATH):
    """Build the rankings and the verdict. Returns a dict (also the JSON record).

    previous (watch)   stored by the last watch run
    previous (now)     the book without the candidate on its own full panel today
    previous / recent  without / with the candidate on the candidate's rows only
    """
    candidates = list(candidates)
    previous_names = [m for m in fm.BOOK if m not in candidates]
    recent_names = previous_names + candidates
    load = fm.combined_panel if live else fm.load_panel
    native = load(previous_names)
    joined = load(recent_names)
    panel = joined.loc[joined[candidates].notna().all(axis=1)]
    cov = coverage(candidates, previous_names)
    ok, reasons = writable(cov, lookback)

    prev_now = rank_frame(native, previous_names, lookback)
    prev_same = rank_frame(panel, previous_names, lookback)
    recent = rank_frame(panel, recent_names, lookback)
    watch_rank, watch_asof = stored_ranking(state_path)

    top_now = list(prev_now.index[:top])
    top_prev = list(prev_same.index[:top])
    top_recent = list(recent.index[:top])
    top_watch = list(watch_rank)[:top] if watch_rank else None
    entered = [m for m in top_recent if m not in top_prev]
    left = [m for m in top_prev if m not in top_recent]

    lines = []
    if top_recent == top_prev:
        lines.append("Top %d unchanged by %s on the candidate's window."
                     % (top, ", ".join(candidates)))
    elif entered:
        lines.append("%s enter%s the top %d on the candidate's window, displacing %s."
                     % (", ".join(entered), "s" if len(entered) == 1 else "", top,
                        ", ".join(left) or "nobody"))
    else:
        lines.append("Same top %d members on the candidate's window, new order: %s."
                     % (top, " > ".join(top_recent)))
    for c in candidates:
        lines.append("%s ranks #%d of %d (signal %.5f; #%d is %.5f)."
                     % (c, recent.loc[c, "rank"], len(recent), recent.loc[c, "signal"],
                        top, recent["signal"].iloc[top - 1]))
    if panel.index.max() < native.index.max() or len(panel) < len(native):
        lines.append("That window ends %s (%d rows); today's book panel ends %s (%d rows). "
                     "The comparison describes the candidate's period, not today's ranking."
                     % (panel.index.max().date(), len(panel), native.index.max().date(),
                        len(native)))
    if top_watch is not None and top_watch != top_now:
        lines.append("Without any candidate, the book has drifted since the last watch "
                     "(%s): %s -> %s." % (watch_asof, " > ".join(top_watch), " > ".join(top_now)))
    if ok:
        top1_before = top_watch[0] if top_watch else top_now[0]
        if top_recent[0] != top1_before:
            lines.append("Top-1 would change from %s to %s: once added, the next watch run "
                         "e-mails the stake move." % (top1_before, top_recent[0]))
        else:
            lines.append("Top-1 stays %s, so the stake does not move." % top_recent[0])
    else:
        lines.append("NOT safe to add to BOOK yet: " + "; ".join(reasons) +
                     ". Extend the candidate's backtest through %s first." % cov["book_last"])

    pairs = {c: paired_vs(panel, c, top_prev, lookback) for c in candidates}
    return {
        "candidates": candidates, "lookback": lookback, "top": top,
        "asof_now": str(native.index.max().date()), "asof_window": str(panel.index.max().date()),
        "n_now": int(len(native)), "n_window": int(len(panel)),
        "coverage": cov, "writable": ok, "write_blockers": reasons,
        "watch_asof": watch_asof, "top_watch": top_watch, "top_now": top_now,
        "top_previous": top_prev, "top_recent": top_recent,
        "entered": entered, "left": left, "verdict": lines,
        "previous_now": prev_now, "previous": prev_same, "recent": recent,
        "watch": watch_rank, "pairs": pairs,
    }


# --------------------------------------------------------------------------- #
# adding to the book
# --------------------------------------------------------------------------- #

_BOOK_RE = re.compile(r"(^BOOK = \[\n)(.*?)(^\]\n)", re.S | re.M)


def add_to_book(names, path=BOOK_FILE):
    """Append `names` to the BOOK list in factor_momentum_stake.py. Returns the
    names actually added (ones already present are skipped). Writes to a temp file
    and renames, so a failure never leaves the module truncated."""
    src = open(path).read()
    match = _BOOK_RE.search(src)
    if not match:
        raise SystemExit("could not find the BOOK = [...] block in %s" % path)
    body = match.group(2)
    present = set(re.findall(r'"([^"]+)"', body))
    new = [n for n in names if n not in present]
    if not new:
        return []
    body = body.rstrip("\n") + "\n" + "".join('    "%s",\n' % n for n in new)
    out = src[:match.start(2)] + body + src[match.end(2):]
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, path)
    return new


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

def _top_columns(res):
    """(header, rows) of the four top-k columns, shared by both renderers."""
    head = ["#", "previous (watch %s)" % (res["watch_asof"] or "-"),
            "previous (now, %s)" % res["asof_now"],
            "previous (window %s)" % res["asof_window"],
            "recent (window, with candidate)"]
    rows = []
    for i in range(res["top"]):
        w = res["top_watch"][i] if res["top_watch"] and i < len(res["top_watch"]) else "-"
        r = res["top_recent"][i]
        rows.append([str(i + 1), w, res["top_now"][i], res["top_previous"][i],
                     r + (" *" if r in res["candidates"] else "")])
    return head, rows


def _shown(res):
    rec = res["recent"]
    shown = list(rec.index[:max(res["top"] + 3, 6)])
    return shown + [c for c in res["candidates"] if c not in shown]


def text_report(res):
    cov = res["coverage"]
    out = ["book update: %s  (roll%d, lag %d)" % (", ".join(res["candidates"]),
                                                 res["lookback"], fm.LAG),
           "book backtest %s..%s (%d dates)" % (cov["book_first"], cov["book_last"], cov["book_n"])]
    for c, v in cov["candidates"].items():
        out.append("%s backtest %s..%s (%d dates, %d shared, %d missing at the tail)"
                   % (c, v["first"], v["last"], v["n"], v["n_common"], v["rows_missing_at_tail"]))
    out.append("")
    head, rows = _top_columns(res)
    fmt = "%-3s %-28s %-28s %-28s %-30s"
    out.append(fmt % tuple(head))
    out.extend(fmt % tuple(r) for r in rows)
    out.append("")
    rec, prev = res["recent"], res["previous"]
    out.append("ranking on the candidate's window (%d rows to %s):" % (res["n_window"], res["asof_window"]))
    out.append("%-4s %-4s %-24s %9s %8s %7s %4s" % ("rank", "was", "model", "signal", "calmar", "dd", "n"))
    for m in _shown(res):
        r = rec.loc[m]
        was = prev["rank"].get(m)
        out.append("%-4d %-4s %-24s %9.5f %8.4f %7.3f %4d%s"
                   % (r["rank"], "-" if pd.isna(was) else int(was), m, r["signal"], r["calmar"],
                      r["dd"], r["n"], "  <- candidate" if m in res["candidates"] else ""))
    for c, pairs in res["pairs"].items():
        out.append("")
        out.append("%s minus each previous top-%d member, resolved window, NW lag %d:"
                   % (c, res["top"], fm.LAG))
        for o, (mean, t, n) in pairs.items():
            out.append("   vs %-24s %+.5f  t %+.2f  (n %d)" % (o, mean, t, n))
    out.append("")
    out.extend(res["verdict"])
    return "\n".join(out)


def html_report(res, text):
    e = html.escape
    head, top_rows = _top_columns(res)
    top_html = "<tr>%s</tr>" % "".join("<th>%s</th>" % e(h) for h in head) + "".join(
        "<tr>%s</tr>" % "".join("<td>%s</td>" % e(c) for c in r) for r in top_rows)
    rec, prev = res["recent"], res["previous"]
    rows = []
    for m, r in rec.iterrows():
        cls = ' class="cand"' if m in res["candidates"] else (
            ' class="top"' if r["rank"] <= res["top"] else "")
        was = prev["rank"].get(m)
        rows.append("<tr%s><td>%d</td><td>%s</td><td>%s</td><td>%.5f</td><td>%.4f</td>"
                    "<td>%.3f</td><td>%d</td></tr>"
                    % (cls, r["rank"], "-" if pd.isna(was) else int(was), e(m),
                       r["signal"], r["calmar"], r["dd"], r["n"]))
    pair_rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%+.5f</td><td>%+.2f</td><td>%d</td></tr>"
        % (e(c), e(o), mean, t, n)
        for c, pairs in res["pairs"].items() for o, (mean, t, n) in pairs.items())
    cov = res["coverage"]
    cov_rows = "<tr><td>book</td><td>%s</td><td>%s</td><td>%d</td><td>-</td><td>-</td></tr>" % (
        cov["book_first"], cov["book_last"], cov["book_n"]) + "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%d</td></tr>"
        % (e(c), v["first"], v["last"], v["n"], v["n_common"], v["rows_missing_at_tail"])
        for c, v in cov["candidates"].items())
    return """<!doctype html><html><head><meta charset="utf-8">
<title>Book update</title>
<style>
body{font:13px/1.45 system-ui,sans-serif;color:#1d232b;background:#fff;max-width:1040px;margin:24px auto;padding:0 16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 6px}
.meta{color:#5b6570}.verdict{background:#f3f6f9;border-left:4px solid #1a4f7a;padding:8px 12px}
table{border-collapse:collapse;margin:6px 0}td,th{padding:3px 10px;border-bottom:1px solid #e3e7eb;text-align:left}
tr.top td{font-weight:600}tr.cand td{background:#fff4d6}
pre{font-size:11px;background:#f7f7f7;padding:8px;white-space:pre-wrap}
</style></head><body>
<h1>Book update: %s</h1>
<p class="meta">Momentum watch ranking: trailing %d mean payout (0.1&middot;corr + mmc) over resolved rounds only (lag %d).</p>
<div class="verdict">%s</div>
<h2>Top %d</h2>
<table>%s</table>
<p class="meta"><b>previous (now)</b> is the book without the candidate on its own full panel today.
<b>previous (window)</b> and <b>recent</b> are without and with the candidate on the rows where the candidate has data,
so the difference between those two columns is the candidate's doing and nothing else.</p>
<h2>Coverage</h2>
<table><tr><th>series</th><th>first</th><th>last</th><th>dates</th><th>shared with book</th><th>missing at tail</th></tr>%s</table>
<p class="meta">BOOK is joined on common dates, so a candidate that stops early would move every model's window back in time.
Adding it to the book is %s.</p>
<h2>Ranking with the candidate (candidate's window)</h2>
<table><tr><th>rank</th><th>was</th><th>model</th><th>signal</th><th>calmar</th><th>dd(pay)</th><th>n</th></tr>%s</table>
<p class="meta">calmar and dd(pay) are on the same resolved trailing window as the signal. Raw payout: no payout factor or clipping, so read ratios, not NMR.</p>
<h2>Candidate against the previous top %d</h2>
<table><tr><th>candidate</th><th>vs</th><th>mean diff</th><th>t (NW 20)</th><th>n</th></tr>%s</table>
<p class="meta">The watch ranks on point estimates. A gap at |t| &lt; 2 is not an established difference, but the watch would still move the stake on it; that is how E55's top1 rule was backtested.</p>
<h2>Console output</h2><pre>%s</pre>
</body></html>""" % (
        e(", ".join(res["candidates"])), res["lookback"], fm.LAG,
        "<br>".join(e(v) for v in res["verdict"]), res["top"], top_html, cov_rows,
        "allowed" if res["writable"] else "<b>blocked</b>: " + e("; ".join(res["write_blockers"])),
        "".join(rows), res["top"], pair_rows, e(text))


def record(res):
    keep = {k: v for k, v in res.items()
            if k not in ("previous", "previous_now", "recent", "pairs")}
    keep["recent_ranking"] = {m: round(float(v), 6) for m, v in res["recent"]["signal"].items()}
    keep["pairs"] = {c: {o: {"mean": v[0], "t": v[1], "n": v[2]} for o, v in p.items()}
                     for c, p in res["pairs"].items()}
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Add a model to the momentum book and compare top 3.")
    ap.add_argument("names", nargs="+", help="registry slot name(s) to add")
    ap.add_argument("--write", action="store_true",
                    help="append the candidate(s) to factor_momentum_stake.BOOK")
    ap.add_argument("--force", action="store_true",
                    help="add to BOOK even when coverage would truncate the watch panel")
    ap.add_argument("--lookback", type=int, default=250)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--no-live", action="store_true", help="backtest panel only, no API")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--host", default="lovelace", help="box with Chrome for the PDF")
    args = ap.parse_args(argv)

    for n in args.names:
        print("%s: %d scored backtest dates" % (n, check_candidate(n)))
    res = compare(args.names, lookback=args.lookback, top=args.top, live=not args.no_live)
    text = text_report(res)
    print(text)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, "%s_%s" % (datetime.date.today().isoformat(),
                                            "_".join(n.lower() for n in args.names)))
    with open(stem + ".json", "w") as fh:
        json.dump(record(res), fh, indent=2, default=str)
    with open(stem + ".html", "w") as fh:
        fh.write(html_report(res, text))
    print("\nwrote %s.html and .json" % stem)
    if not args.no_pdf:
        from src.analysis.html_pdf import to_pdf
        try:
            to_pdf(stem + ".html", args.host)
        except Exception as error:
            # A local Chromium that exists but cannot run (deepfour) does not trigger
            # html_pdf's ssh path, so retry the whole print on --host.
            import subprocess
            print("local PDF print failed (%s); retrying on %s" % (type(error).__name__, args.host))
            cmd = "cd %s && /home/danielco/miniconda3/bin/python3 -m src.analysis.html_pdf %s.html" % (
                os.getcwd(), os.path.abspath(stem))
            done = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, cmd],
                                  capture_output=True, text=True, timeout=300)
            print(done.stdout.strip() or "PDF failed on %s: %s" % (args.host, done.stderr.strip()[-300:]))

    if args.write and not res["writable"] and not args.force:
        print("BOOK: NOT added -- %s. Pass --force to add anyway."
              % "; ".join(res["write_blockers"]))
    elif args.write:
        added = add_to_book(args.names)
        print("BOOK: added %s" % (", ".join(added) if added else "nothing (already present)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
