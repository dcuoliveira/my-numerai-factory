# -*- coding: utf-8 -*-
"""
The Numerai Crypto stake report: what the staked book is worth right now.

    python3 -m src.analysis.crypto_stake_report            # build, write, print subject
    python3 -m src.analysis.crypto_stake_report --email    # also mail it
    python3 -m src.analysis.crypto_stake_report --no-live  # backtest sections only

Writes a single self-contained HTML file -- inline styles, inline SVG, no
scripts and no external assets -- so the same bytes serve as an email body and
as a file on disk.

------------------------------------------------------------------------------
Where every number comes from, and why not from somewhere easier
------------------------------------------------------------------------------

STAKE AND PAYOUT come from `v2RoundModelPerformances(tournament: 12)`, one query
per model. Nothing else on the API knows: crypto stakes are V3 "Fusion" stakes
held on-chain in the allocation-strategy contract, so `account.availableNmr` and
`heldForFusionStakes` describe the CLASSIC book and are wrong here -- during
development they read 130.05 NMR staked while the crypto strategy held 118.04,
because they are two different books sharing one wallet. Per-model V3 stake is
not exposed at all; the standing amount is recovered as the sum of `atRisk` over
a model's open rounds, which reproduces the dashboard's TOTAL STAKE column.

NOTHING HAS RESOLVED. Every payout here is `payoutPending`: Numerai recomputes it
each scoring day off a longer slice of the 20-day forward return. The report says
so at the top rather than in a footnote, because a mark-to-market number that
looks like a settled one is the single most misleading thing this file could
print.

REALISED CORRELATION uses live per-round `corr`, pairwise on each pair's own
overlap -- the models were staked on different days, so a single common window
would throw away most of the data. Every cell carries its n, and cells under
MIN_LIVE_ROUNDS are left blank rather than drawn: three observations make a
correlation the eye will read as a finding.

THE META MODEL STRIP IS A DIFFERENT STATISTIC from the rest of that matrix, and
is fenced off for it. Model-to-model cells correlate per-round CORR -- outcome
correlation, which is what decides whether holding several of these diversifies
anything. The Meta Model row and column are mean `cwmm`, a correlation between
PREDICTIONS. Numerai does not publish the Meta Model's own per-round CORR, so a
like-for-like cell is not computable live. Section 5 has both, computed the same
way, on the backtest.

BACKTESTED CORRELATION reads the slots' stored per-period series through
src.analysis.slots, so it inherits that module's two resolutions: CURUPIRA_07 is
a screen rather than a run and resolves to its reconstruction, and CURUPIRA_05_60
is scored on the 60-day label. The Meta Model is a genuine row here, scored
against the same target on the dates it exists.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

TOURNAMENT_ID = 12
OUT_DIR = "results/reports/crypto"
OUT_FILE = "stake_report.html"

#: Model slug -> the registry slot name. Order is the report's row order.
STAKED_SLOTS = [
    ("curupira_08", "CURUPIRA_08"),
    ("curupira_05", "CURUPIRA_05"),
    ("cu_sa_ia_bo_01", "CU_SA_IA_BO_01"),
    ("curupira_01", "CURUPIRA_01"),
    ("cu_sa_ia_bo_02", "CU_SA_IA_BO_02"),
    ("cu_sa_ia_bo_ma_r_01", "CU_SA_IA_BO_MA_R_01"),
    ("cu_sa_ia_bo_ma_01", "CU_SA_IA_BO_MA_01"),
]
COLOUR = {"CURUPIRA_08": "#c0392b", "CURUPIRA_05": "#2c6fbb",
          "CU_SA_IA_BO_01": "#e08a1e", "CURUPIRA_01": "#3f8f4f",
          "CU_SA_IA_BO_02": "#8e44ad", "CU_SA_IA_BO_MA_R_01": "#00897b",
          "CU_SA_IA_BO_MA_01": "#6d4c41"}

#: Section 3 colours a round's diamond by SETTLEMENT STATUS, not by model, so the
#: scheme reads the same way in every panel: red while the figure is still a
#: mark-to-market estimate (the round is open), blue once the round has resolved
#: and the payout is final. The per-model COLOUR above stays for section 2, where
#: several models share one chart and the colour has to tell them apart.
PENDING_COLOUR = "#c5221f"    # latest / current pending (open round)
RESOLVED_COLOUR = "#1a73e8"   # resolved / settled (final payout)

#: Below this many shared rounds a realised correlation is not drawn. Two points
#: always give |r| = 1 and three are barely better; the cell stays blank and the
#: n is still printed, which is the honest rendering of "not yet knowable".
MIN_LIVE_ROUNDS = 5

META = "META MODEL"


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #

# Crypto pays 0.1*CORR + 1*MMC per NMR at risk (the repo convention), scaled by
# the round's payout factor. Applying it to each of intraRound's DAILY CORR/MMC
# reconstructs the daily pending-payout path -- and applying it to the last day
# reproduces the API's own `payout` to the digit, which is how the multipliers
# were checked (r1337: 0.1*0.1231 + 0.0569 = 0.06922, x1.2594 = 0.08718 = payout).
CORR_MULT, MMC_MULT = 0.1, 1.0

#: A crypto round pays on a ~20-day forward return, so a year holds 365/20 of
#: them for annualising a per-round Sharpe.
ANNUAL_ROUNDS = 365.0 / 20.0

PERF_QUERY = """
query($m: ID, $t: Int, $n: Int) {
  v2RoundModelPerformances(modelId: $m, tournament: $t, lastNRounds: $n) {
    roundNumber roundResolved atRisk payout roundPayoutFactor roundOpenTime
    submissionScores { date day displayName value }
    intraRoundSubmissionScores { date day displayName value }
  }
}"""


def daily_payout_path(intra, at_risk, factor):
    """
    The round's pending payout as of each scoring day, oldest first.

    intraRound serves the round's cumulative CORR and MMC as of every day it has
    been scored -- so this is the actual history the pending figure has walked,
    not a reconstruction from our own snapshots. Each day maps to a payout
    through the same formula the latest day reproduces the API's `payout` with.

    :param intra: intraRoundSubmissionScores rows
    :param at_risk: NMR staked on the round
    :param factor: the round's payout factor
    :return: [payout_day1, payout_day2, ...]
    :rtype: [float]
    """
    by_day = {}
    for score in intra or []:
        if score.get("displayName") in ("corr", "mmc") and score.get("value") is not None:
            by_day.setdefault(score["day"], {})[score["displayName"]] = score["value"]
    path = []
    for day in sorted(by_day):
        cell = by_day[day]
        if "corr" in cell and "mmc" in cell:
            path.append(at_risk * factor * (CORR_MULT * cell["corr"] + MMC_MULT * cell["mmc"]))
    return path


def fetch_live(slugs, rounds=80):
    """
    Per-round stake, pending payout, scores and daily path for each model.

    :param slugs: model slugs as they appear on the account
    :type slugs: [str]
    :param rounds: how far back to ask
    :type rounds: int
    :return: {slug: {"rounds": [...], "scores": {round: {name: value}}}}
    :rtype: dict
    """
    from src.live.env import get_api, load_env

    load_env()
    api = get_api()
    models = api.get_models() or {}
    out = {}
    for slug in slugs:
        if slug not in models:
            continue
        payload = api.raw_query(
            PERF_QUERY, {"m": models[slug], "t": TOURNAMENT_ID, "n": rounds},
            authorization=True)["data"]["v2RoundModelPerformances"] or []
        rows, scores = [], {}
        for item in sorted(payload, key=lambda x: x["roundNumber"]):
            values = {s["displayName"]: s["value"]
                      for s in (item.get("submissionScores") or [])
                      if s.get("value") is not None}
            if values:
                scores[item["roundNumber"]] = values
            if item.get("atRisk") and float(item["atRisk"]) > 0:
                at_risk = float(item["atRisk"])
                factor = float(item.get("roundPayoutFactor") or 1.0)
                rows.append({
                    "round": item["roundNumber"],
                    "at_risk": at_risk,
                    "payout": None if item["payout"] is None else float(item["payout"]),
                    "resolved": bool(item["roundResolved"]),
                    "opened": item.get("roundOpenTime"),
                    "path": daily_payout_path(
                        item.get("intraRoundSubmissionScores"), at_risk, factor),
                })
        if rows:
            out[slug] = {"rounds": rows, "scores": scores}
    return out


def nmr_rates():
    """
    NMR/EUR and NMR/BRL, live, falling back to None rather than to a stale guess.

    A fiat column built on a hardcoded rate is worse than no fiat column: it
    looks current and nobody re-reads the constant.

    :return: (eur, brl, source note)
    :rtype: (float or None, float or None, str)
    """
    try:
        import urllib.request
        url = ("https://api.coingecko.com/api/v3/simple/price"
               "?ids=numeraire&vs_currencies=eur,brl")
        with urllib.request.urlopen(url, timeout=20) as handle:
            data = json.load(handle)["numeraire"]
        return float(data["eur"]), float(data["brl"]), "CoinGecko, live at build time"
    except Exception as error:                       # pragma: no cover - network
        print("FX lookup failed (%s); fiat columns omitted" % str(error)[:120])
        return None, None, "unavailable"


# --------------------------------------------------------------------------- #
# backtest side
# --------------------------------------------------------------------------- #

def backtest_frames(slot_names):
    """
    Per-period CORR for each slot, plus the meta model's own CORR series.

    :param slot_names: registry slot names
    :type slot_names: [str]
    :return: (corr frame indexed by date, meta model corr series)
    :rtype: (pd.DataFrame, pd.Series)
    """
    from src.analysis import slots

    columns = {}
    for name in slot_names:
        resolved = slots.slot_series(name)
        if resolved is None:
            continue
        frame = pd.read_csv(
            "results/per_period/crypto/%s/%s.csv" % (resolved.model, resolved.run),
            parse_dates=["date"]).set_index("date")
        columns[name] = frame["corr"]
    return pd.DataFrame(columns).dropna(), meta_model_corr()


def meta_model_corr(version="v2.0", target="target_binned_return_20"):
    """
    The published meta model's own per-period CORR, or an empty series.

    Spearman against the same target the slots are scored on, so the row is
    comparable with theirs rather than being a different statistic wearing the
    same label.

    :return: per-date correlation
    :rtype: pd.Series
    """
    meta_path = "data/crypto/%s/historical_meta_models.parquet" % version
    train_path = "data/crypto/%s/train.parquet" % version
    if not (os.path.exists(meta_path) and os.path.exists(train_path)):
        return pd.Series(dtype=float)
    meta = pd.read_parquet(meta_path)
    truth = pd.read_parquet(train_path, columns=["symbol", "date", target])
    joined = meta.merge(truth, on=["symbol", "date"], how="inner").dropna()
    joined["date"] = pd.to_datetime(joined["date"])
    return joined.groupby("date")[["prediction", target]].apply(
        lambda g: g["prediction"].corr(g[target], method="spearman")).dropna()


def prediction_matrix(slot_names, version="v2.0"):
    """
    Mean within-period rank correlation between the slots' predictions, with the
    meta model as an extra column.

    :return: (matrix, n dates) or (None, 0)
    :rtype: (pd.DataFrame or None, int)
    """
    from src.analysis import slots
    from src.analysis.corr_heatmap import within_period_rank_corr
    from src.analysis.report_data import Arm, load_predictions_common

    arms = []
    for name in slot_names:
        resolved = slots.slot_series(name)
        if resolved is None:
            continue
        arms.append(Arm(label=name, model=resolved.model, run=resolved.run,
                        kind=resolved.kind, target=resolved.target, frame=None,
                        payout=0.0, staked=False, live_status="", note=None))
    if len(arms) < 2:
        return None, 0
    wide, _, _ = load_predictions_common("crypto", arms, 100)
    if wide.empty:
        return None, 0

    meta_path = "data/crypto/%s/historical_meta_models.parquet" % version
    if os.path.exists(meta_path):
        meta = pd.read_parquet(meta_path)
        meta["date"] = pd.to_datetime(meta["date"]).dt.strftime("%Y-%m-%d")
        series = meta.set_index(["symbol", "date"])["prediction"].astype("float32")
        wide[META] = series.reindex(wide.index)
        wide = wide.dropna()
    return within_period_rank_corr(wide), wide.index.get_level_values("date").nunique()


def realised_matrix(live, slot_of):
    """
    Pearson of live per-round CORR, pairwise on each pair's own overlap.

    Pairwise rather than on one common window because the slots were staked on
    different days: CURUPIRA_01 has been scored on 11 rounds and CU_SA_IA_BO_01
    on 2, and intersecting all four would throw away nine tenths of what exists.
    The price is that cells rest on different samples, so every one carries its
    n and thin cells are not drawn at all.

    :return: (r matrix, n matrix, mean cwmm by slot)
    :rtype: (pd.DataFrame, pd.DataFrame, dict)
    """
    labels = [slot_of[s] for s in live]
    series = {}
    for slug, payload in live.items():
        got = {r: v["corr"] for r, v in payload["scores"].items() if "corr" in v}
        series[slot_of[slug]] = pd.Series(got, dtype=float).sort_index()
    cwmm = {}
    for slug, payload in live.items():
        values = [v["cwmm"] for v in payload["scores"].values() if "cwmm" in v]
        cwmm[slot_of[slug]] = float(np.mean(values)) if values else None

    r = pd.DataFrame(index=labels, columns=labels, dtype=float)
    n = pd.DataFrame(index=labels, columns=labels, dtype=int)
    for a, b in itertools.product(labels, labels):
        joined = pd.concat([series[a], series[b]], axis=1, join="inner").dropna()
        n.loc[a, b] = len(joined)
        if a == b:
            r.loc[a, b] = 1.0
        elif len(joined) >= MIN_LIVE_ROUNDS:
            r.loc[a, b] = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    return r, n, cwmm


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def num(x, d=2):
    return format(x, ",.%df" % d)


def signed(x, d=4):
    return format(x, "+,.%df" % d)


def klass(x):
    return "pos" if x > 0 else ("neg" if x < 0 else "flat")


def heat(r):
    """RdBu cell colour for a correlation, matching the report's scale."""
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "#f5f5f5", "#9aa0a6"
    r = max(-1.0, min(1.0, float(r)))
    if r >= 0:
        stops = ((255, 255, 255), (178, 24, 43))
    else:
        stops = ((255, 255, 255), (33, 102, 172))
    t = abs(r)
    rgb = tuple(int(round(a + (b - a) * t)) for a, b in zip(*stops))
    ink = "#ffffff" if t > 0.55 else "#202124"
    return "#%02x%02x%02x" % rgb, ink


def heatmap(matrix, note_of=None, fenced=None, width_first=170):
    """
    A correlation matrix as an email-safe HTML table.

    An HTML table rather than an image or SVG because this has to survive every
    mail client, and rather than a canvas because the numbers must stay
    selectable text.

    :param matrix: square frame of correlations
    :param note_of: optional {(row, col): "title text"}
    :param fenced: label whose row and column are drawn as a separate statistic
    """
    labels = list(matrix.columns)
    out = ['<table class="heat"><tr><th class="hl"></th>']
    for c in labels:
        out.append('<th class="hc%s">%s</th>' % (" fence" if c == fenced else "", c))
    out.append("</tr>")
    for row in labels:
        out.append('<tr><th class="hl%s" width="%d">%s</th>'
                   % (" fence" if row == fenced else "", width_first, row))
        for col in labels:
            value = matrix.loc[row, col]
            bg, ink = heat(value)
            title = (note_of or {}).get((row, col), "")
            text = "&mdash;" if value is None or pd.isna(value) else "%.2f" % value
            out.append('<td class="hcell%s" style="background:%s;color:%s" title="%s">%s</td>'
                       % (" fence" if fenced in (row, col) else "", bg, ink, title, text))
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def line_chart(series, width=760, height=300, pad=(52, 18, 42, 62)):
    """Inline SVG line chart. series: [(label, colour, [(x, y)])]."""
    pt, pr, pb, pl = pad
    xs = sorted({x for _, _, pts in series for x, _ in pts})
    ys = [y for _, _, pts in series for _, y in pts] or [0.0]
    lo, hi = min(ys + [0.0]), max(ys + [0.0])
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.12, hi + span * 0.12
    w, h = width - pl - pr, height - pt - pb
    fx = lambda v: pl + (w if len(xs) == 1 else w * xs.index(v) / (len(xs) - 1))
    fy = lambda v: pt + h - h * (v - lo) / (hi - lo)
    out = ['<svg width="100%%" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'style="max-width:%dpx">' % (width, height, width),
           '<rect width="%d" height="%d" fill="#fff"/>' % (width, height)]
    for i in range(6):
        v = lo + (hi - lo) * i / 5.0
        y = fy(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
                   % (pl, y, width - pr, y, "#9aa0a6" if abs(v) < 1e-12 else "#eceff1"))
        out.append('<text x="%d" y="%.1f" font-size="10" fill="#5f6368" '
                   'text-anchor="end">%.4f</text>' % (pl - 6, y + 3, v))
    for x in xs:
        out.append('<text x="%.1f" y="%d" font-size="9" fill="#5f6368" text-anchor="middle" '
                   'transform="rotate(-90 %.1f %d)">%s</text>'
                   % (fx(x), height - pb + 18, fx(x), height - pb + 18, x))
    for label, colour, pts in series:
        if not pts:
            continue
        path = " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", fx(x), fy(y))
                        for i, (x, y) in enumerate(pts))
        out.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>' % (path, colour))
        for x, y in pts:
            out.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"/>' % (fx(x), fy(y), colour))
    out.append("</svg>")
    return "".join(out)


def bar_chart(points, colour, width=760, height=190, pad=(16, 18, 42, 62)):
    """Inline SVG column chart; None values are drawn as 'not yet scored'."""
    pt, pr, pb, pl = pad
    values = [v for _, v in points if v is not None] or [0.0]
    lo, hi = min(values + [0.0]), max(values + [0.0])
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.18, hi + span * 0.18
    w, h = width - pl - pr, height - pt - pb
    step = w / max(len(points), 1)
    fy = lambda v: pt + h - h * (v - lo) / (hi - lo)
    zero = fy(0.0)
    out = ['<svg width="100%%" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'style="max-width:%dpx">' % (width, height, width),
           '<rect width="%d" height="%d" fill="#fff"/>' % (width, height)]
    for i in range(4):
        v = lo + (hi - lo) * i / 3.0
        y = fy(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
                   % (pl, y, width - pr, y, "#9aa0a6" if abs(v) < 1e-12 else "#eceff1"))
        out.append('<text x="%d" y="%.1f" font-size="10" fill="#5f6368" '
                   'text-anchor="end">%s</text>'
                   % (pl - 6, y + 3, ("%+.1f%%" % (100 * v)) if pct else "%+.3f" % v))
    for i, (x, v) in enumerate(points):
        cx = pl + step * (i + 0.5)
        if v is None:
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#f1f3f4" '
                       'stroke="#dadce0" stroke-dasharray="2 2"/>'
                       % (cx - step * 0.3, pt, step * 0.6, h))
        else:
            y = fy(v)
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" '
                       'opacity="0.85"/>' % (cx - step * 0.3, min(y, zero), step * 0.6,
                                             max(abs(zero - y), 1.0), colour))
        out.append('<text x="%.1f" y="%d" font-size="9" fill="#5f6368" text-anchor="middle" '
                   'transform="rotate(-90 %.1f %d)">%s</text>'
                   % (cx, height - pb + 18, cx, height - pb + 18, x))
    out.append("</svg>")
    return "".join(out)


def load_history(order):
    """
    Every past snapshot's per-round pending payout, grouped by model then round.

    The box plots and the confidence band show how each round's pending figure
    has moved over time, and Numerai does not serve that -- its API returns only
    the latest cumulative value per round. So it has to be reconstructed from the
    daily snapshots this report writes itself (save_snapshot). Today there are a
    handful and the boxes are nearly points; the mechanism is built so they fill
    in as the daily cron accumulates history, which is the only way this series
    can ever exist.

    :param order: model labels to collect, in report order
    :return: {label: {round: [payout, ...]}}
    :rtype: dict
    """
    directory = os.path.join(OUT_DIR, "history")
    history = {label: {} for label in order}
    if not os.path.isdir(directory):
        return history
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            snapshot = json.load(open(os.path.join(directory, name)))
        except (ValueError, IOError):
            continue
        for label in order:
            for row in snapshot.get(label, []):
                if row.get("payout") is None:
                    continue
                history[label].setdefault(row["round"], []).append(float(row["payout"]))
    return history


def box_stats(values):
    """Five-number summary of a list, for one box. None if empty."""
    if not values:
        return None
    arr = sorted(values)
    q = lambda p: float(np.percentile(arr, p))
    return {"lo": arr[0], "q1": q(25), "med": q(50), "q3": q(75), "hi": arr[-1],
            "n": len(arr)}


def box_plot(rounds, boxes, latest, colour, paths=None, colours=None,
             width=760, height=210, pad=(16, 18, 42, 62), pct=False):
    """
    Per-round box of the pending payout's visited values, latest marked.

    A box rather than a line because the story is dispersion: how much a round's
    estimate has swung before it settles. Each daily observation is a faint dot,
    the box its quartiles, the diamond where it stands now. Every column carries
    a transparent hover zone (class hz) whose data-tip the shared tooltip reads,
    so the chart is interactive without a plotting library.
    """
    pt, pr, pb, pl = pad
    # pct: values are returns (payout / stake); axis and tips read in percent.
    fmt = (lambda v: "%+.2f%%" % (100 * v)) if pct else signed
    unit = "" if pct else " NMR"
    flat = [v for b in boxes if b for v in (b["lo"], b["hi"])] + list(
        x for x in latest if x is not None) + [0.0]
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.18, hi + span * 0.18
    w, h = width - pl - pr, height - pt - pb
    step = w / max(len(rounds), 1)
    fy = lambda v: pt + h - h * (v - lo) / (hi - lo)
    out = ['<svg width="100%%" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'style="max-width:%dpx">' % (width, height, width),
           '<rect width="%d" height="%d" fill="#fff"/>' % (width, height)]
    for i in range(4):
        v = lo + (hi - lo) * i / 3.0
        y = fy(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
                   % (pl, y, width - pr, y, "#9aa0a6" if abs(v) < 1e-12 else "#eceff1"))
        out.append('<text x="%d" y="%.1f" font-size="10" fill="#5f6368" '
                   'text-anchor="end">%+.3f</text>' % (pl - 6, y + 3, v))
    zones = []
    for i, (rnd, box, now) in enumerate(zip(rounds, boxes, latest)):
        cx = pl + step * (i + 0.5)
        bw = min(step * 0.5, 22)
        if box:
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#9aa0a6"/>'
                       % (cx, fy(box["hi"]), cx, fy(box["q3"])))
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#9aa0a6"/>'
                       % (cx, fy(box["lo"]), cx, fy(box["q1"])))
            top, bot = fy(box["q3"]), fy(box["q1"])
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#e8eaed" '
                       'stroke="#9aa0a6"/>' % (cx - bw / 2, min(top, bot), bw,
                                               max(abs(bot - top), 1.0)))
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#5f6368" '
                       'stroke-width="1.5"/>' % (cx - bw / 2, fy(box["med"]),
                                                 cx + bw / 2, fy(box["med"])))
        path = (paths or [None] * len(rounds))[i] or []
        for j, val in enumerate(path):
            jitter = ((j - (len(path) - 1) / 2.0) / max(len(path), 1)) * bw * 0.7
            out.append('<circle cx="%.1f" cy="%.1f" r="1.7" fill="#9aa0a6" opacity="0.55"/>'
                       % (cx + jitter, fy(val)))
        if now is not None:
            y = fy(now)
            # One colour per diamond when a list is given (settlement status),
            # otherwise the single series colour. The box and whiskers stay
            # neutral grey, so the diamond is the only mark the colour speaks
            # through.
            dc = colours[i] if colours else colour
            out.append('<path d="M%.1f %.1f L%.1f %.1f L%.1f %.1f L%.1f %.1f Z" fill="%s"/>'
                       % (cx, y - 4, cx + 4, y, cx, y + 4, cx - 4, y, dc))
        out.append('<text x="%.1f" y="%d" font-size="9" fill="#5f6368" text-anchor="middle" '
                   'transform="rotate(-90 %.1f %d)">%s</text>'
                   % (cx, height - pb + 18, cx, height - pb + 18, rnd))
        tip = "round %s<br>latest %s%s<br>%d scoring day%s" % (
            rnd, fmt(now) if now is not None else "&ndash;", unit,
            len(path), "" if len(path) == 1 else "s")
        if box and box["n"] > 1:
            tip += "<br>range %s &hellip; %s" % (fmt(box["lo"]), fmt(box["hi"]))
        zones.append('<rect class="hz" x="%.1f" y="%d" width="%.1f" height="%d" '
                     'fill="transparent" data-tip="%s"/>'
                     % (cx - step / 2, pt, step, h, tip))
    out.extend(zones)
    out.append("</svg>")
    return "".join(out)


def band_chart(cum_line, band, colour, width=760, height=230, pad=(52, 18, 42, 62)):
    """
    Cumulative pending P&L with the spread it has been quoted at.

    cum_line: [(round, value)] -- cum of each round's latest pending.
    band: [(round, lo, q1, q3, hi)] -- cum spread across snapshots, or [] if
    there is not yet more than one snapshot to spread over.
    """
    pt, pr, pb, pl = pad
    ys = [v for _, v in cum_line] + [v for row in band for v in row[1:]] + [0.0]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.12, hi + span * 0.12
    rounds = [r for r, _ in cum_line]
    w, h = width - pl - pr, height - pt - pb
    fx = lambda r: pl + (w if len(rounds) <= 1 else w * rounds.index(r) / (len(rounds) - 1))
    fy = lambda v: pt + h - h * (v - lo) / (hi - lo)
    out = ['<svg width="100%%" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'style="max-width:%dpx">' % (width, height, width),
           '<rect width="%d" height="%d" fill="#fff"/>' % (width, height)]
    for i in range(6):
        v = lo + (hi - lo) * i / 5.0
        y = fy(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
                   % (pl, y, width - pr, y, "#9aa0a6" if abs(v) < 1e-12 else "#eceff1"))
        out.append('<text x="%d" y="%.1f" font-size="10" fill="#5f6368" '
                   'text-anchor="end">%.3f</text>' % (pl - 6, y + 3, v))

    def area(points_hi, points_lo, fill, opacity):
        top = " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", fx(r), fy(v))
                       for i, (r, v) in enumerate(points_hi))
        bot = " ".join("L%.1f,%.1f" % (fx(r), fy(v))
                       for r, v in reversed(points_lo))
        out.append('<path d="%s %s Z" fill="%s" opacity="%.2f"/>' % (top, bot, fill, opacity))

    if band and any(row[1] != row[4] for row in band):
        area([(r, hi_) for r, _, _, _, hi_ in band], [(r, lo_) for r, lo_, _, _, _ in band],
             colour, 0.12)
        area([(r, q3) for r, _, _, q3, _ in band], [(r, q1) for r, q1, _, _, _ in band],
             colour, 0.25)
    path = " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", fx(r), fy(v))
                    for i, (r, v) in enumerate(cum_line))
    out.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>' % (path, colour))
    for r, v in cum_line:
        out.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"/>' % (fx(r), fy(v), colour))
    for r in rounds:
        out.append('<text x="%.1f" y="%d" font-size="9" fill="#5f6368" text-anchor="middle" '
                   'transform="rotate(-90 %.1f %d)">%s</text>'
                   % (fx(r), height - pb + 18, fx(r), height - pb + 18, r))
    band_by_round = {row[0]: row for row in band}
    step = w / max(len(rounds), 1)
    for r, v in cum_line:
        tip = "round %s<br>cum %s NMR" % (r, signed(v))
        row = band_by_round.get(r)
        if row and row[1] != row[4]:
            tip += ("<br>&plusmn;1 std %s &hellip; %s<br>bound %s &hellip; %s"
                    % (signed(row[2]), signed(row[3]), signed(row[1]), signed(row[4])))
        out.append('<rect class="hz" x="%.1f" y="%d" width="%.1f" height="%d" '
                   'fill="transparent" data-tip="%s"/>'
                   % (fx(r) - step / 2, pt, step, h, tip))
    out.append("</svg>")
    return "".join(out)


CSS = """
body{margin:0;padding:0;background:#f6f7f9}
.wrap{max-width:900px;margin:0 auto;padding:24px 18px 48px;background:#fff;color:#202124;
 font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;line-height:1.5}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.3px}
h2{font-size:19px;margin:34px 0 10px;border-bottom:1px solid #e0e0e0;padding-bottom:8px}
h3{font-size:14px;margin:22px 0 4px}
.built{color:#5f6368;font-size:13px;margin:0 0 20px}
.pre{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:6px;padding:14px 16px;margin:0 0 26px;
 font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;white-space:pre-wrap}
.warn{border-left:4px solid #c5221f;background:#fce8e6;padding:12px 16px;margin:0 0 22px;font-size:13.5px}
.note{border-left:4px solid #f9ab00;background:#fef7e0;padding:12px 16px;margin:16px 0;font-size:13px}
.cards{width:100%;border-collapse:separate;border-spacing:8px 0;margin:0 0 8px}
.card{border:1px solid #dadce0;border-radius:8px;padding:12px 14px;vertical-align:top}
.card .k{font-size:10.5px;letter-spacing:.9px;color:#5f6368;text-transform:uppercase}
.card .v{font-size:21px;font-weight:700;margin:5px 0 3px;letter-spacing:-.3px}
.card .s{font-size:11.5px;color:#5f6368}
table.data{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0}
table.data th{text-align:right;padding:8px;border-bottom:2px solid #202124;font-size:10.5px;
 letter-spacing:.7px;text-transform:uppercase;color:#5f6368;white-space:nowrap}
table.data td{text-align:right;padding:8px;border-bottom:1px solid #ececec;
 font-variant-numeric:tabular-nums;white-space:nowrap}
table.data th.l,table.data td.l{text-align:left}
table.data tbody tr:hover td{background:#f1f3f4}
tr.total td{border-top:2px solid #202124;border-bottom:none;font-weight:700}
tr.total:hover td{background:transparent}
.chart-controls{margin:6px 0 4px;font-size:12.5px;color:#5f6368}
.chart-controls select{font:inherit;padding:3px 6px;border:1px solid #dadce0;border-radius:5px;
 background:#fff;color:#202124}
.js-legend{font-size:12px;margin:6px 0 2px;line-height:1.9}
.js-legend .item{cursor:pointer;user-select:none;margin-right:14px;white-space:nowrap;
 display:inline-block}
.js-legend .item.off{opacity:.35;text-decoration:line-through}
.js-legend .dot{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:5px}
.js-only{display:none}
.chart-host{position:relative}
.tip{position:absolute;pointer-events:none;background:#202124;color:#fff;font-size:11.5px;
 border-radius:6px;padding:7px 9px;line-height:1.5;box-shadow:0 2px 8px rgba(0,0,0,.25);
 white-space:nowrap;opacity:0;transition:opacity .05s;z-index:5}
.tip b{color:#fff}
details.panel{padding:0}
details.panel summary{padding:11px 14px;font-weight:700;font-size:14px;background:#fafafa;
 border-radius:8px 8px 0 0;cursor:pointer;list-style:none}
details.panel summary::-webkit-details-marker{display:none}
details.panel summary::before{content:"\\25B6";font-size:10px;color:#5f6368;margin-right:9px}
details.panel[open] summary::before{content:"\\25BC"}
details.panel[open] summary{border-bottom:1px solid #ececec}
details.panel summary:hover{background:#f1f3f4}
.pos{color:#137333}.neg{color:#c5221f}.flat{color:#5f6368}
.cap{font-size:12px;color:#5f6368;margin:6px 0 0}
.panel{border:1px solid #dadce0;border-radius:8px;margin:14px 0;background:#fdfdfd}
.phead{padding:11px 14px;font-weight:700;font-size:14px;border-bottom:1px solid #ececec;
 background:#fafafa;border-radius:8px 8px 0 0}
.pbody{padding:10px 12px 14px}
.ctitle{font-size:13.5px;color:#1a4f7a;margin:8px 0 2px;font-weight:600}
.bandkey{font-size:11.5px;color:#5f6368;margin:5px 0 2px}
.bandkey .ln{display:inline-block;width:20px;height:3px;vertical-align:middle;margin:0 5px 0 12px}
.bandkey .bx{display:inline-block;width:20px;height:11px;vertical-align:middle;margin:0 5px 0 16px;
 border-radius:2px}
.bandkey .dia{margin:0 4px 0 12px;font-size:12px}
.hz{pointer-events:all;cursor:crosshair}
.gtip{position:fixed;pointer-events:none;background:#202124;color:#fff;font-size:11.5px;
 border-radius:6px;padding:7px 9px;line-height:1.5;box-shadow:0 2px 8px rgba(0,0,0,.25);
 white-space:nowrap;opacity:0;transition:opacity .05s;z-index:20}
.legend{font-size:11.5px;color:#5f6368;margin:2px 0 8px}
.sw{display:inline-block;width:22px;height:3px;vertical-align:middle;margin:0 5px 0 12px}
table.heat{border-collapse:collapse;font-size:12px;margin:8px 0 4px;table-layout:fixed;width:100%}
table.heat th{font-size:11px;color:#1a4f7a;font-weight:600;padding:5px 6px}
table.heat th.hl{text-align:right;white-space:nowrap}
table.heat th.hc{text-align:center}
table.heat td.hcell{text-align:center;padding:11px 4px;font-variant-numeric:tabular-nums;
 border:1px solid #fff}
.fence{border-left:3px solid #202124!important;border-top:3px solid #202124!important}
.scale{font-size:11px;color:#5f6368;margin:4px 0 0}
.scale span{display:inline-block;width:26px;height:11px;vertical-align:middle;margin:0 1px}
"""


def render(ctx):
    """Assemble the whole document from a prepared context dict."""
    h = ["<style>%s</style>" % CSS, '<div class="wrap">']
    h.append('<div class="pre">%s</div>' % ctx["top_text"])
    h.append("<h1>Numerai Crypto stake report</h1>")
    h.append('<p class="built">Built %s &middot; %d models with a stake &middot; %d staked rounds '
             '&middot; <strong>%d resolved</strong></p>'
             % (ctx["built"], len(ctx["models"]), ctx["n_rounds"], ctx["n_resolved"]))
    h.append('<div class="warn">%s</div>' % ctx["banner"])

    h.append('<table class="cards"><tr>')
    cw = 100 // max(len(ctx["cards"]), 1)
    for key, value, sub in ctx["cards"]:
        h.append('<td class="card" width="%d%%"><div class="k">%s</div><div class="v">%s</div>'
                 '<div class="s">%s</div></td>' % (cw, key, value, sub))
    h.append("</tr></table>")

    # 1 -------------------------------------------------------------------- #
    h.append("<h2>1 &middot; The book</h2>")
    h.append(ctx["book_table"])
    h.append('<p class="cap">%s</p>' % ctx["book_note"])
    h.append(ctx["model_table"])
    h.append('<div class="ctitle">Returns by window</div>')
    h.append(ctx["returns_table"])
    h.append('<p class="cap">%s</p>' % ctx["returns_note"])

    # 2 -------------------------------------------------------------------- #
    h.append("<h2>2 &middot; All staked models together</h2>")
    h.append('<p class="cap">Cumulative provisional P&amp;L, year to date by default. Use the '
             'dropdowns to switch the range (year to date / full history), the currency and the '
             'stake basis; click a legend entry to hide a model. Only scored rounds appear.</p>')
    h.append('<div class="ctitle">All staked models &mdash; cumulative provisional P&amp;L</div>')
    options = ['<option value="pernmr">NMR &middot; per 1 NMR staked</option>',
               '<option value="stake">NMR &middot; actual stake</option>']
    if ctx["chart_data"]["eur"] is not None:
        options += ['<option value="eurstake">EUR &middot; actual stake</option>',
                    '<option value="eurpernmr">EUR &middot; per 1 NMR staked</option>']
    if ctx["chart_data"]["brl"] is not None:
        options += ['<option value="brlstake">BRL &middot; actual stake</option>',
                    '<option value="brlpernmr">BRL &middot; per 1 NMR staked</option>']
    h.append('<div class="chart-controls js-only">range <select id="rangesel">'
             '<option value="ytd">year to date</option>'
             '<option value="full">full history</option></select>'
             ' &nbsp; units <select id="unitsel">%s</select></div>' % "".join(options))
    h.append('<div class="js-legend js-only" id="leg"></div>')
    h.append('<div class="chart-host" id="chost">')
    h.append('<div id="staticchart"><div class="legend">%s</div>%s</div>'
             % (ctx["legend"], ctx["combined_chart"]))
    h.append('<div id="jschart" class="js-only"></div><div class="tip" id="tip"></div>')
    h.append("</div>")
    if ctx["unpriced_note"]:
        h.append('<p class="cap">%s</p>' % ctx["unpriced_note"])

    # 3 -------------------------------------------------------------------- #
    h.append("<h2>3 &middot; Model by model</h2>")
    h.append('<p class="cap">Click a model to expand it. Each box is one round: the spread of that '
             'round\'s pending payout across the scoring days it has been open &mdash; how much the '
             'figure has swung before settling &mdash; and the diamond is the latest day. A round '
             'open one day is a single point; the box grows toward its final width as the round '
             'ages through its 20-day window. Rounds shown are year to date.</p>')
    h.extend(ctx["panels"])

    # 3b ------------------------------------------------------------------- #
    h.append("<h2>3b &middot; Model by model &mdash; round stake returns</h2>")
    h.append('<p class="cap">The same boxes as section 3, but every daily value is divided by '
             'that round\'s own stake, so each box is the round\'s return (payout / NMR at risk) '
             'and rounds of very different stake size share one scale. Click a model to expand it; '
             'the diamond is the latest day, blue once the round has resolved. Year to date.</p>')
    h.extend(ctx.get("ret_panels", []))

    # 4 -------------------------------------------------------------------- #
    h.append("<h2>4 &middot; Realised correlation (live)</h2>")
    if ctx["realised"] is None:
        h.append('<p class="cap">Not enough scored rounds yet.</p>')
    else:
        h.append('<div class="ctitle">Realised correlation &mdash; live per-round CORR across '
                 'models</div>')
        h.append('<p class="cap">models: Pearson of per-round CORR. Meta Model strip (fenced): mean '
                 '<code>cwmm</code>, a PREDICTION correlation &mdash; a different statistic, shown '
                 'here because Numerai does not publish the Meta Model\'s own per-round CORR. Each '
                 'cell rests on that pair\'s own overlap; hover for n. Cells under %d shared rounds '
                 'are left blank.</p>' % MIN_LIVE_ROUNDS)
        h.append(ctx["realised"])
        h.append(ctx["scale"])
        h.append('<div class="note"><strong>The Meta Model strip is a different statistic from the '
                 'rest of this matrix.</strong> Model-to-model cells are the correlation of per-round '
                 'CORR &mdash; outcome correlation, which is what determines whether holding several '
                 'of these at once diversifies anything. The fenced Meta Model row and column are the '
                 'mean of <code>cwmm</code>, a correlation between <em>predictions</em>. Numerai does '
                 'not publish the Meta Model\'s own per-round CORR, so a like-for-like cell is not '
                 'computable live. Panel 5 has both, computed the same way, on the backtest.</div>')

    # 5 -------------------------------------------------------------------- #
    h.append("<h2>5 &middot; Backtested correlation</h2>")
    h.append('<div class="ctitle">Backtested correlation &mdash; per-period CORR across models</div>')
    h.append('<p class="cap">%s</p>' % ctx["bt_note"])
    h.append(ctx["bt_corr"])
    h.append('<div class="ctitle">Backtested correlation &mdash; per-period prediction Spearman</div>')
    h.append('<p class="cap">Mean per-period Spearman between the raw predictions. This is the same '
             'statistic Numerai reports live as <code>cwmm</code>, so its Meta Model column is '
             'directly comparable with the fenced strip in panel 4.</p>')
    h.append(ctx["bt_pred"])
    h.append(ctx["scale"])

    h.append("<h3>Realised vs backtested, pair by pair</h3>")
    h.append(ctx["pair_table"])
    h.append(ctx["verdict"])
    h.append('<p class="cap">%s</p>' % ctx["bt_footer"])
    h.append('<hr style="border:none;border-top:1px solid #e0e0e0;margin:26px 0 12px">')
    h.append('<p class="cap">%s</p>' % ctx["footer"])
    h.append(chart_script(ctx["chart_data"]))
    h.append(hover_script())
    h.append("</div>")
    return "".join(h)


def hover_script():
    """
    One shared tooltip for every hover zone in the section-3 charts.

    The box plots and band charts are server-rendered SVG -- static is the right
    default so they survive a client with no JS -- but each column carries a
    transparent `.hz` rect with a data-tip. This wires them all with a single
    delegated listener rather than a handler per chart, so the interactivity
    costs one small script no matter how many models the book grows to.
    """
    body = r"""
(function(){
  var tip=document.getElementById('gtip'); if(!tip) return;
  document.addEventListener('mousemove',function(e){
    var z=e.target && e.target.closest ? e.target.closest('.hz') : null;
    if(!z){ tip.style.opacity=0; return; }
    tip.innerHTML=z.getAttribute('data-tip')||'';
    tip.style.opacity=1;
    var x=e.clientX+14, y=e.clientY+12;
    if(x+tip.offsetWidth>window.innerWidth-8) x=e.clientX-tip.offsetWidth-14;
    if(y+tip.offsetHeight>window.innerHeight-8) y=e.clientY-tip.offsetHeight-12;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  });
  window.addEventListener('blur',function(){ tip.style.opacity=0; });
})();
"""
    return '<div class="gtip" id="gtip"></div><script>%s</script>' % body


def chart_script(data):
    """
    The section-2 interactive chart, as a self-contained vanilla-JS <script>.

    Vanilla rather than a plotting library so the file stays small and works
    offline from an email attachment -- a 3.5 MB Plotly bundle inline, or a CDN
    load that fails with no network, would both be worse than hand-drawn SVG. It
    reveals the .js-only controls, hides the server-rendered fallback, and
    redraws under whatever currency/basis the dropdown holds; with JS off the
    fallback chart and its static legend remain.
    """
    body = r"""
(function(){
  var DATA = __DATA__;
  var host=document.getElementById('jschart'), stat=document.getElementById('staticchart'),
      tip=document.getElementById('tip'), leg=document.getElementById('leg'),
      sel=document.getElementById('unitsel'),
      rangeSel=document.getElementById('rangesel');
  if(!host||!DATA||!DATA.models.some(function(m){return m.rows.length;})) return;
  stat.style.display='none';
  Array.prototype.forEach.call(document.querySelectorAll('.js-only'),
      function(e){ e.style.display='block'; });
  var RATE={NMR:1,EUR:DATA.eur,BRL:DATA.brl};
  var rset={}; DATA.models.forEach(function(m){m.rows.forEach(function(r){rset[r[0]]=1;});});
  var allRounds=Object.keys(rset).map(Number).sort(function(a,b){return a-b;});
  var rounds=allRounds;
  function applyRange(){
    var ytd = rangeSel && rangeSel.value==='ytd' && DATA.ytd_min!=null;
    rounds = ytd ? allRounds.filter(function(rn){return rn>=DATA.ytd_min;}) : allRounds;
  }
  applyRange();
  var hidden={}, W=760,H=300,P=[52,18,42,62], NS='http://www.w3.org/2000/svg';
  function unit(){var v=sel.value;
    return {rate:RATE[v.indexOf('eur')===0?'EUR':(v.indexOf('brl')===0?'BRL':'NMR')],
            cur:v.indexOf('eur')===0?'EUR':(v.indexOf('brl')===0?'BRL':'NMR'),
            basis:v.indexOf('pernmr')>=0?'pernmr':'stake'};}
  function build(){var u=unit(),series=[];
    DATA.models.forEach(function(m){var byr={}; m.rows.forEach(function(r){byr[r[0]]=r;});
      var run=0,pts=[]; rounds.forEach(function(rn){var r=byr[rn];
        if(r) run+=(u.basis==='pernmr'?r[1]/r[2]:r[1])*u.rate;
        pts.push([rn, r?run:null]);});
      series.push({label:m.label,colour:DATA.colours[m.label]||'#5f6368',pts:pts});});
    var pr=0,pp=[]; rounds.forEach(function(rn){var num=0,den=0,any=false;
      DATA.models.forEach(function(m){var r=m.rows.filter(function(x){return x[0]===rn;})[0];
        if(r){any=true;num+=r[1];den+=r[2];}});
      if(any) pr+=(u.basis==='pernmr'?num/den:num)*u.rate; pp.push([rn,any?pr:null]);});
    series.push({label:'portfolio',colour:'#202124',pts:pp}); return series;}
  function el(t,a){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);return e;}
  function draw(){var series=build(),vis=series.filter(function(s){return !hidden[s.label];});
    var ys=[0]; vis.forEach(function(s){s.pts.forEach(function(p){if(p[1]!=null)ys.push(p[1]);});});
    var lo=Math.min.apply(null,ys),hi=Math.max.apply(null,ys),sp=(hi-lo)||1; lo-=sp*0.12; hi+=sp*0.12;
    var w=W-P[3]-P[1],h=H-P[0]-P[2];
    function fx(rn){var i=rounds.indexOf(rn);return P[3]+(rounds.length<=1?w:w*i/(rounds.length-1));}
    function fy(v){return P[0]+h-h*(v-lo)/(hi-lo);}
    var svg=el('svg',{viewBox:'0 0 '+W+' '+H,width:'100%'}); svg.style.maxWidth=W+'px'; svg.style.display='block';
    svg.appendChild(el('rect',{width:W,height:H,fill:'#fff'}));
    for(var i=0;i<=5;i++){var val=lo+(hi-lo)*i/5,y=fy(val);
      svg.appendChild(el('line',{x1:P[3],y1:y,x2:W-P[1],y2:y,stroke:Math.abs(val)<1e-12?'#9aa0a6':'#eceff1'}));
      var tx=el('text',{x:P[3]-6,y:y+3,'font-size':10,fill:'#5f6368','text-anchor':'end'});tx.textContent=val.toFixed(4);svg.appendChild(tx);}
    rounds.forEach(function(rn){var t=el('text',{x:fx(rn),y:H-P[2]+18,'font-size':9,fill:'#5f6368','text-anchor':'middle',transform:'rotate(-90 '+fx(rn)+' '+(H-P[2]+18)+')'});t.textContent=rn;svg.appendChild(t);});
    vis.forEach(function(s){var d='',on=false;
      s.pts.forEach(function(p){if(p[1]==null)return;d+=(on?'L':'M')+fx(p[0]).toFixed(1)+','+fy(p[1]).toFixed(1)+' ';on=true;});
      if(d)svg.appendChild(el('path',{d:d,fill:'none',stroke:s.colour,'stroke-width':s.label==='portfolio'?2.5:2,'stroke-dasharray':s.label==='portfolio'?'2 3':''}));
      s.pts.forEach(function(p){if(p[1]!=null)svg.appendChild(el('circle',{cx:fx(p[0]),cy:fy(p[1]),r:2.4,fill:s.colour}));});});
    var guide=el('line',{x1:0,y1:P[0],x2:0,y2:H-P[2],stroke:'#9aa0a6','stroke-dasharray':'3 3',opacity:0});
    svg.appendChild(guide); host.innerHTML=''; host.appendChild(svg);
    var u=unit(),sym=u.cur==='EUR'?'€':(u.cur==='BRL'?'R$':'');
    svg.addEventListener('mousemove',function(ev){var b=svg.getBoundingClientRect(),px=(ev.clientX-b.left)*(W/b.width);
      var best=rounds[0],bd=1e9; rounds.forEach(function(rn){var d=Math.abs(fx(rn)-px);if(d<bd){bd=d;best=rn;}});
      guide.setAttribute('x1',fx(best));guide.setAttribute('x2',fx(best));guide.setAttribute('opacity',1);
      var rows='<b>round '+best+'</b>';
      vis.forEach(function(s){var p=s.pts.filter(function(x){return x[0]===best;})[0];
        if(p&&p[1]!=null)rows+='<br><span style="color:'+s.colour+'">&#9632;</span> '+s.label+' '+sym+p[1].toFixed(u.cur==='NMR'?4:2);});
      tip.innerHTML=rows;tip.style.opacity=1;
      var hb=document.getElementById('chost').getBoundingClientRect();
      var lx=ev.clientX-hb.left+14; if(lx>hb.width-190)lx-=200; tip.style.left=lx+'px'; tip.style.top=(ev.clientY-hb.top+8)+'px';});
    svg.addEventListener('mouseleave',function(){tip.style.opacity=0;guide.setAttribute('opacity',0);});}
  function drawLegend(){leg.innerHTML=''; build().forEach(function(s){
    var it=document.createElement('span'); it.className='item'+(hidden[s.label]?' off':'');
    it.innerHTML='<span class="dot" style="background:'+s.colour+'"></span>'+s.label;
    it.addEventListener('click',function(){hidden[s.label]=!hidden[s.label];it.className='item'+(hidden[s.label]?' off':'');draw();});
    leg.appendChild(it);});}
  sel.addEventListener('change',draw);
  if(rangeSel) rangeSel.addEventListener('change',function(){applyRange();draw();});
  drawLegend(); draw();
})();
"""
    return "<script>%s</script>" % body.replace("__DATA__", json.dumps(data))


def scale_bar():
    """The colour key that goes under a heatmap."""
    cells = []
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        bg, _ = heat(value)
        cells.append('<span style="background:%s;border:1px solid #dadce0"></span>' % bg)
    return ('<p class="scale">r &nbsp;-1 %s +1</p>' % "".join(cells))


def save_snapshot(rows, built):
    """
    Append today's pending figure per round to a history file.

    The within-round box plots need the daily path of the pending payout, and
    Numerai serves only the current value -- so it has to be recorded as it goes.
    One JSON per build, never overwritten, cheap to accumulate and trivially
    droppable if the format changes.
    """
    directory = os.path.join(OUT_DIR, "history")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "%s.json" % built.replace(":", "").replace(" ", "T"))
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=1)
    return path


def build(live, rates, built, slot_of):
    """Turn fetched data into the context `render` consumes."""
    eur, brl, fx_note = rates
    labels = [slot_of[s] for s in live]
    per = {}
    for slug, payload in live.items():
        rows = payload["rounds"]
        label = slot_of[slug]
        # Stake is what sits in OPEN rounds. A settled round's NMR has already
        # come back -- to the wallet, or rolled into a newer round's atRisk
        # (round 1379 carried exactly 1355's 37 NMR plus its 0.9776 payout) --
        # so summing every round ever staked counts the same NMR again each
        # time it is re-deployed: 322.67 NMR against a real ~97 on 2026-10-27.
        open_rows = [r for r in rows if not r["resolved"]]
        per[label] = {
            "risk": sum(r["at_risk"] for r in open_rows),
            "open_pl": sum(r["payout"] for r in open_rows if r["payout"] is not None),
            "pl": sum(r["payout"] for r in rows if r["payout"] is not None),
            "n": len(rows),
            "priced": sum(1 for r in rows if r["payout"] is not None),
            "resolved": sum(1 for r in rows if r["resolved"]),
            "rows": rows,
        }
    order = [slot_of[s] for s, _ in STAKED_SLOTS if slot_of[s] in per]
    risk = sum(per[m]["risk"] for m in order)
    pl = sum(per[m]["open_pl"] for m in order)        # provisional: open rounds only
    total_pl = sum(per[m]["pl"] for m in order)       # every round, open and settled
    book = risk + pl
    n_rounds = sum(per[m]["n"] for m in order)
    n_priced = sum(per[m]["priced"] for m in order)
    n_resolved = sum(per[m]["resolved"] for m in order)
    pct = 100 * pl / risk if risk else 0.0

    # Three P&L windows, all expressed against the SAME denominator -- the
    # current stake at risk -- so they never invite the "1D vs cumulative"
    # confusion of the dashboard, where each figure has its own hidden basis.
    #   1-day     : how much the open book moved on the last scoring day, summed
    #               as (latest day - previous day) over every round with >= 2
    #               days. This is the daily delta the dashboard's "1D" measures;
    #               it lands near but not on Numerai's %, whose exact basis is
    #               undocumented, so it is labelled as our own reconstruction.
    #   YTD       : payout over rounds opened this calendar year.
    #   inception : payout over every round ever staked. Uses the SETTLED payout
    #               for resolved rounds and the pending one for open rounds, so
    #               it keeps counting rounds after they drop out of the open book
    #               -- today, with nothing resolved, it equals the provisional
    #               P&L above, and it will diverge the first time a round settles.
    year = int(built[:4])
    settled = lambda r: r["payout"] if r["payout"] is not None else 0.0
    all_rows = [r for m in order for r in per[m]["rows"]]
    day_pl = sum(r["path"][-1] - r["path"][-2] for r in all_rows if len(r["path"]) >= 2)
    ytd_pl = sum(settled(r) for r in all_rows
                 if r.get("opened") and int(r["opened"][:4]) == year)
    incep_pl = sum(settled(r) for r in all_rows)
    rpct = lambda x: 100 * x / risk if risk else 0.0

    # Money that is actually BANKED, as opposed to the mark-to-market book above.
    # A crypto round resolves about 20 days after it opens -- the horizon of the
    # target it pays on -- so the first staked round here (1337, opened
    # 2026-08-20) only comes due around 2026-09-09. Until rounds start settling
    # this is 0 by construction and the provisional figure is the whole story;
    # after that the two diverge and this is the half that cannot be taken back.
    # It is DISJOINT from the provisional P&L above (open rounds only) and is not
    # part of book value: settled money has already left the rounds it was in.
    resolved_pl = sum(settled(r) for r in all_rows if r["resolved"])

    def settle_line(dash):
        """The settlement sentence, in HTML or plain-text punctuation."""
        if n_resolved:
            return ("%d of %d staked rounds have resolved; the other %d are still "
                    "mark-to-market."
                    % (n_resolved, n_rounds, n_rounds - n_resolved))
        return ("%d of %d staked rounds have resolved %s everything below is "
                "mark-to-market." % (n_resolved, n_rounds, dash))

    # How many rounds sit behind each window, counted two ways: distinct
    # tournament rounds, and staked positions (model x round) -- the second is
    # the unit the settlement counts elsewhere in the report use.
    day_rows = [r for r in all_rows if len(r["path"]) >= 2]
    ytd_rows_all = [r for r in all_rows if r.get("opened") and int(r["opened"][:4]) == year]
    count = lambda rows: (len({r["round"] for r in rows}), len(rows))

    # Realised Sharpe, annualised. The series is the BOOK's return per round:
    # summed payout over summed stake across every model in that round, over
    # priced rounds only (an unscored round has no return yet, not a zero one).
    # Each round is a ~20-day holding-period return, so the annualising factor is
    # sqrt(365 / 20), not sqrt(365) -- the daily-round cadence does not make a
    # round a daily return. Rounds open daily and overlap, so this is a
    # level-and-noise summary, not an independent-sample statistic.
    def realised_sharpe(rows):
        by_round = {}
        for r in rows:
            if r["payout"] is None or not r["at_risk"]:
                continue
            slot = by_round.setdefault(r["round"], [0.0, 0.0])
            slot[0] += r["payout"]
            slot[1] += r["at_risk"]
        rets = np.array([pay / stk for pay, stk in by_round.values() if stk])
        if len(rets) < 3 or rets.std() == 0:
            return None
        return float(rets.mean() / rets.std() * np.sqrt(ANNUAL_ROUNDS))

    # Calendar days from the first stake in the window (the earliest round
    # opened) to the report date. Unlike Rounds, this counts the gaps: a pause
    # in staking makes Days exceed Rounds.
    report_day = pd.Timestamp(built[:10])

    def days_since_first(rows):
        opened = [r["opened"][:10] for r in rows if r.get("opened")]
        if not opened:
            return None
        return (min(opened), int((report_day - pd.Timestamp(min(opened))).days))

    ctx_returns = [
        ("1-day change", "last scoring day, reconstructed from the daily paths",
         day_pl, rpct(day_pl)) + count(day_rows) + (None, None),
        ("Return YTD", "rounds opened in %d" % year, ytd_pl, rpct(ytd_pl))
        + count(ytd_rows_all) + (days_since_first(ytd_rows_all), realised_sharpe(ytd_rows_all)),
        ("Return since inception", "every round ever staked", incep_pl, rpct(incep_pl))
        + count(all_rows) + (days_since_first(all_rows), realised_sharpe(all_rows)),
    ]

    def fiat(value, rate, digits=2, sign=False):
        if rate is None:
            return "&mdash;"
        return (signed if sign else num)(value * rate, digits)

    ctx = {"built": built, "models": order, "n_rounds": n_rounds,
           "n_priced": n_priced, "n_resolved": n_resolved, "scale": scale_bar()}

    # The headline banner. It used to assert "Nothing here has settled"
    # unconditionally; once rounds began resolving that sat directly above a
    # non-zero resolved figure and contradicted it. Driven off n_resolved so it
    # flips on its own the first time a round settles.
    n_unscored = n_rounds - n_priced
    unscored_tail = (" A further %d have never been scored and contribute 0." % n_unscored
                     if n_unscored else "")
    if n_resolved:
        ctx["banner"] = (
            "<strong>%d of %d staked rounds have settled.</strong> Their payout is final and is "
            "shown as resolved P&amp;L; the other %d are still open and carry provisional P&amp;L, a "
            "mark-to-market estimate Numerai recomputes each scoring day off a longer slice of the "
            "20-day forward return and which moves in both directions.%s"
            % (n_resolved, n_rounds, n_rounds - n_resolved, unscored_tail))
    else:
        ctx["banner"] = (
            "<strong>Nothing here has settled.</strong> 0 of %d staked rounds have resolved, so "
            "every figure below is a mark-to-market estimate: what would be paid if scoring stopped "
            "today. Numerai recomputes it each scoring day off a longer slice of the 20-day forward "
            "return, and it moves in both directions.%s" % (n_rounds, unscored_tail))

    ctx["top_text"] = (
        "Numerai Crypto stake report &mdash; %s\n\n"
        "TOTAL BOOK VALUE  %s NMR  |  EUR %s  |  BRL %s\n"
        " stake at risk   %s NMR  |  EUR %s  |  BRL %s\n"
        " provisional P&amp;L %s NMR  |  EUR %s  |  BRL %s  (%+.2f%% of stake)\n"
        " resolved P&amp;L    %s NMR  |  EUR %s  |  BRL %s  (%+.2f%% of stake)\n\n"
        "Rates: %s.\n"
        "%s\n\n"
        "Staked positions: %s.\n\n"
        "The interactive report is attached; open it in a browser."
        % (built, num(book, 4), fiat(book, eur), fiat(book, brl),
           num(risk, 4), fiat(risk, eur), fiat(risk, brl),
           signed(pl), fiat(pl, eur, sign=True), fiat(pl, brl, sign=True), pct,
           signed(resolved_pl), fiat(resolved_pl, eur, sign=True),
           fiat(resolved_pl, brl, sign=True), rpct(resolved_pl),
           fx_note, settle_line("&mdash;"), ", ".join(order)))

    # The email BODY is this summary as plain text -- the interactive HTML rides
    # as an attachment, not as the body, so the same entities that make the <pre>
    # block render in a browser must not leak into a text/plain part. Build the
    # body once here from real characters rather than un-escaping top_text.
    def plain(value, rate, digits=2, sign=False):
        return "n/a" if rate is None else (signed if sign else num)(value * rate, digits)

    ctx["plain_text"] = (
        "Numerai Crypto stake report — %s\n\n"
        "TOTAL BOOK VALUE  %s NMR  |  EUR %s  |  BRL %s\n"
        " stake at risk   %s NMR  |  EUR %s  |  BRL %s\n"
        " provisional P&L %s NMR  |  EUR %s  |  BRL %s  (%+.2f%% of stake)\n"
        " resolved P&L    %s NMR  |  EUR %s  |  BRL %s  (%+.2f%% of stake)\n\n"
        "Rates: %s.\n"
        "%s\n\n"
        "Staked positions: %s.\n\n"
        "The full interactive report is attached as %s; open it in a browser."
        % (built, num(book, 4), plain(book, eur), plain(book, brl),
           num(risk, 4), plain(risk, eur), plain(risk, brl),
           signed(pl), plain(pl, eur, sign=True), plain(pl, brl, sign=True), pct,
           signed(resolved_pl), plain(resolved_pl, eur, sign=True),
           plain(resolved_pl, brl, sign=True), rpct(resolved_pl),
           fx_note, settle_line("—"), ", ".join(order), OUT_FILE))

    # Every card carries NMR in the headline and EUR + BRL beneath it, so the
    # three currencies read the same way across the whole row. The P&L cards add
    # their own context: provisional its % of stake, resolved the word "settled".
    ctx["cards"] = [
        ("Total book value", "%s NMR" % num(book, 4),
         "&euro;%s &middot; R$%s" % (fiat(book, eur), fiat(book, brl))),
        ("Stake at risk", "%s NMR" % num(risk, 2),
         "&euro;%s &middot; R$%s" % (fiat(risk, eur), fiat(risk, brl))),
        ("Provisional P&amp;L", "%s NMR" % signed(pl),
         "%+.2f%% &middot; &euro;%s &middot; R$%s" % (pct, fiat(pl, eur), fiat(pl, brl))),
        ("Resolved P&amp;L", "%s NMR" % signed(resolved_pl),
         "settled &middot; &euro;%s &middot; R$%s"
         % (fiat(resolved_pl, eur), fiat(resolved_pl, brl))),
        ("Rounds priced", "%d of %d" % (n_priced, n_rounds),
         "the other %d opened too recently to carry a score" % (n_rounds - n_priced)),
    ]

    book_rows = [
        ("stake at risk", num(risk, 4), fiat(risk, eur), fiat(risk, brl), "flat"),
        ("provisional P&amp;L (open rounds)", signed(pl), fiat(pl, eur, sign=True),
         fiat(pl, brl, sign=True), klass(pl)),
        ("resolved P&amp;L (settled)", signed(resolved_pl), fiat(resolved_pl, eur, sign=True),
         fiat(resolved_pl, brl, sign=True), klass(resolved_pl)),
    ]
    table = ['<table class="data"><tr><th class="l"></th><th>NMR</th><th>EUR</th><th>BRL</th></tr>']
    for name, a, b, c, cl in book_rows:
        table.append('<tr><td class="l"><strong>%s</strong></td><td class="%s">%s</td>'
                     '<td class="%s">%s</td><td class="%s">%s</td></tr>'
                     % (name, cl, a, cl, b, cl, c))
    table.append('<tr class="total"><td class="l">total book value</td><td>%s</td><td>%s</td>'
                 '<td>%s</td></tr></table>' % (num(book, 4), fiat(book, eur), fiat(book, brl)))
    ctx["book_table"] = "".join(table)
    ctx["book_note"] = (
        "Book value = stake at risk + provisional P&amp;L, both over OPEN rounds only, i.e. what the "
        "position would be worth if every open round settled at today's estimate. Resolved P&amp;L "
        "is already realised &mdash; settled and paid, back in the wallet or rolled into a newer "
        "round's stake &mdash; so it is listed separately and not added into book value. Stake per "
        "model is the sum of <code>atRisk</code> over its open rounds (a settled round's stake is "
        "not counted: that NMR has moved on), which is what the dashboard's TOTAL STAKE column "
        "shows; the account's <code>heldForFusionStakes</code> describes the CLASSIC book and is not "
        "this one. Rates: %s. Fiat is a valuation &mdash; the tournament pays NMR." % fx_note)

    table = ['<table class="data"><tr><th class="l">Model</th><th>Stake (NMR)</th><th>Rounds</th>'
             '<th>Priced</th><th>P&amp;L NMR</th><th>EUR</th><th>BRL</th><th>% of stake</th></tr>']
    for m in order:
        p, r = per[m]["pl"], per[m]["risk"]
        table.append('<tr><td class="l"><strong>%s</strong></td><td>%s</td><td>%d</td><td>%d</td>'
                     '<td class="%s">%s</td><td class="%s">%s</td><td class="%s">%s</td>'
                     '<td class="%s">%+.2f</td></tr>'
                     % (m, num(r, 4), per[m]["n"], per[m]["priced"], klass(p), signed(p),
                        klass(p), fiat(p, eur, sign=True), klass(p), fiat(p, brl, sign=True),
                        klass(p), 100 * p / r if r else 0.0))
    table.append('<tr class="total"><td class="l">portfolio</td><td>%s</td><td>%d</td><td>%d</td>'
                 '<td class="%s">%s</td><td class="%s">%s</td><td class="%s">%s</td>'
                 '<td class="%s">%+.2f</td></tr></table>'
                 % (num(risk, 2), n_rounds, n_priced, klass(total_pl), signed(total_pl),
                    klass(total_pl), fiat(total_pl, eur, sign=True), klass(total_pl),
                    fiat(total_pl, brl, sign=True), klass(total_pl),
                    100 * total_pl / risk if risk else 0.0))
    ctx["model_table"] = "".join(table)

    # Every window here counts open and settled rounds alike, so it says so
    # rather than leaving the reader to reconcile it with the header's resolved
    # line. The old wording asserted "no round resolved yet" unconditionally,
    # which stops being true the first time one settles.
    if n_resolved:
        drift = ("%d of %d rounds have now settled: the header's resolved P&amp;L is that "
                 "banked subset, while every window in this table still counts open and "
                 "settled rounds together." % (n_resolved, n_rounds))
    else:
        drift = ("With no round resolved yet, YTD and since-inception both equal the "
                 "provisional P&amp;L above, and the header's resolved P&amp;L is 0; they "
                 "diverge as rounds settle or a year turns.")

    rt = ['<table class="data"><tr><th class="l">Window</th><th>Rounds</th>'
          '<th>Stakes</th><th>Days since 1st stake</th><th>NMR</th><th>% of stake</th>'
          '<th>Sharpe (ann.)</th><th class="l">what it counts</th></tr>']
    for name, note, value, percent, n_rnd, n_pos, first, sharpe in ctx_returns:
        days_cell = ("&mdash;" if first is None else
                     '<span title="first stake opened %s">%d</span>' % (first[0], first[1]))
        rt.append('<tr><td class="l"><strong>%s</strong></td><td>%d</td><td>%d</td><td>%s</td>'
                  '<td class="%s">%s</td><td class="%s">%+.2f</td><td class="%s">%s</td>'
                  '<td class="l" style="color:#5f6368;font-weight:400">%s</td></tr>'
                  % (name, n_rnd, n_pos, days_cell, klass(value), signed(value), klass(percent),
                     percent, klass(sharpe or 0.0),
                     "&mdash;" if sharpe is None else "%+.2f" % sharpe,
                     note if first is None else "%s; first stake %s" % (note, first[0])))
    rt.append("</table>")
    ctx["returns_table"] = "".join(rt)
    ctx["returns_note"] = (
        "All three are P&amp;L as a %% of the current stake at risk (%s NMR) &mdash; one "
        "denominator, differing only in window, so they never carry the hidden-basis ambiguity the "
        "dashboard's row does. The 1-day figure is reconstructed from the daily paths, not Numerai's "
        "dashboard \"1D\": expect the same sign and rough size, not the same number, since its basis "
        "is undocumented. <em>Rounds</em> counts distinct tournament rounds; <em>Stakes</em> "
        "counts staked positions, one per model per round. <em>Days since 1st stake</em> is "
        "calendar days from the earliest round opened in the window to the report date; it "
        "counts gaps, so it exceeds Rounds if staking ever paused. <em>Sharpe (ann.)</em> is the realised "
        "Sharpe of the book's per-round return (summed payout over summed stake, priced rounds "
        "only, std ddof=0), annualised by &radic;(365/20) because each round is a ~20-day holding "
        "period; rounds overlap, so read it as a summary, not a significance test. A single "
        "scoring day has no Sharpe. %s"
        % (num(risk, 2), drift))

    returns_body = "".join(
        "\n %-22s %s NMR (%+.2f%% of stake, %d rounds / %d stakes%s%s)"
        % (name, signed(value), percent, n_rnd, n_pos,
           "" if first is None else ", %d days since first stake %s" % (first[1], first[0]),
           "" if sharpe is None else ", Sharpe %+.2f ann." % sharpe)
        for name, _, value, percent, n_rnd, n_pos, first, sharpe in ctx_returns)
    ctx["top_text"] = ctx["top_text"].replace("\n\nRates:", returns_body + "\n\nRates:", 1)
    ctx["plain_text"] = ctx["plain_text"].replace("\n\nRates:", returns_body + "\n\nRates:", 1)

    # Year-to-date is the rounds opened in the current calendar year -- the same
    # cut the "Return YTD" line uses. Section 3 is always scoped to it, and
    # section 2 defaults to it with a switch to the full history.
    def _ytd(row):
        return not row.get("opened") or int(row["opened"][:4]) == year
    ytd_min = min((r["round"] for mm in order for r in per[mm]["rows"]
                   if r.get("opened") and int(r["opened"][:4]) == year), default=None)

    # section 2 -- the static (no-JS) fallback is the YTD view, which is the
    # default; the interactive chart adds the full-history option.
    series, totals = [], {}
    for m in order:
        run, pts = 0.0, []
        for row in per[m]["rows"]:
            if row["payout"] is None or not _ytd(row):
                continue
            run += row["payout"] / row["at_risk"]
            pts.append((row["round"], run))
            slot = totals.setdefault(row["round"], [0.0, 0.0])
            slot[0] += row["payout"]
            slot[1] += row["at_risk"]
        series.append((m, COLOUR.get(m, "#5f6368"), pts))
    run, pts = 0.0, []
    for rnd in sorted(totals):
        run += totals[rnd][0] / totals[rnd][1]
        pts.append((rnd, run))
    series.append(("portfolio", "#202124", pts))
    ctx["legend"] = "".join('<span class="sw" style="background:%s"></span>%s' % (c, l)
                            for l, c, _ in series)
    ctx["combined_chart"] = line_chart(series)
    silent = [m for m in order if per[m]["priced"] == 0]
    ctx["unpriced_note"] = ("%s ha%s no line: every one of its rounds opened after the last scoring "
                            "day, so it has never been priced and its stake is marked at zero."
                            % (", ".join(silent), "ve" if len(silent) > 1 else "s")) if silent else ""

    # Data the section-2 chart script recomputes in the browser under whatever
    # currency and basis the reader picks. Only payout and stake go over; the
    # cumulation happens client-side so the dropdown is instant.
    ctx["chart_data"] = {
        "eur": eur, "brl": brl,
        "colours": {m: COLOUR.get(m, "#5f6368") for m in order},
        "ytd_min": ytd_min,
        "models": [{
            "label": m,
            "stake": round(per[m]["risk"], 6),
            "rows": [[r["round"], r["payout"], r["at_risk"]]
                     for r in per[m]["rows"] if r["payout"] is not None],
        } for m in order],
    }

    # section 3 -- collapsible panel per model. The box for a round is the spread
    # of that round's pending payout across its scoring days (its daily path from
    # intraRound); the diamond is the latest day. No snapshot history needed --
    # the whole path is served per round.
    panels, ret_panels = [], []
    for position, m in enumerate(order):
        # Section 3 is always year to date: only rounds opened this year, and the
        # header summarises those rather than the all-time book above.
        ytd_rows = [r for r in per[m]["rows"] if _ytd(r)]
        n_ytd = len(ytd_rows)
        priced_ytd = sum(1 for r in ytd_rows if r["payout"] is not None)
        pl_ytd = sum(r["payout"] for r in ytd_rows if r["payout"] is not None)
        rows = [r for r in ytd_rows if r["path"]]
        open_attr = " open" if position == 0 else ""
        panels.append('<details class="panel"%s><summary>%s &mdash; %s NMR staked &middot; '
                      '%d YTD rounds &middot; %d priced &middot; book <span class="%s">%s NMR</span>'
                      '</summary><div class="pbody">'
                      % (open_attr, m, num(per[m]["risk"], 2), n_ytd, priced_ytd,
                         klass(pl_ytd), signed(pl_ytd)))
        if rows:
            scored = [r["round"] for r in rows]
            boxes = [box_stats(r["path"]) for r in rows]
            latest = [r["path"][-1] for r in rows]
            # Colour the diamonds by settlement, the same way in every panel:
            # red while the round is still a mark-to-market estimate, blue once
            # it has resolved and the payout is final. A reader comparing two
            # models then reads colour as status, not as which model it is.
            diamond_colours = [RESOLVED_COLOUR if r["resolved"] else PENDING_COLOUR
                               for r in rows]
            panels.append('<div class="ctitle">%s &mdash; spread of the provisional payout across '
                          'each round\'s scoring days (NMR)</div>' % m)
            panels.append('<div class="bandkey">'
                          '<span class="dia" style="color:%s">&#9670;</span>latest (current pending)'
                          '<span class="dia" style="color:%s">&#9670;</span>resolved (settled)</div>'
                          % (PENDING_COLOUR, RESOLVED_COLOUR))
            panels.append(box_plot(scored, boxes, latest, PENDING_COLOUR,
                                   paths=[r["path"] for r in rows],
                                   colours=diamond_colours))
            # The cumulative line is cum of each round's latest pending. Around it,
            # two clouds mirror the Classic report: the inner is cum of the daily
            # path's mean +/- one standard deviation, the outer is cum of its min
            # and max. Both cumulate per-round like a bound, not a path -- the
            # extremes fall on different scoring days -- which is the honest way
            # to draw an envelope over spreads that are not aligned in time.
            cum_line, band = [], []
            run = cmin = cmax = cmean = cstd = 0.0
            for r in rows:
                path = np.asarray(r["path"], float)
                run += path[-1]
                cum_line.append((r["round"], run))
                cmin += path.min(); cmax += path.max()
                cmean += path.mean(); cstd += path.std()
                band.append((r["round"], cmin, cmean - cstd, cmean + cstd, cmax))
            panels.append('<div class="ctitle">%s &mdash; cumulative provisional P&amp;L (NMR), '
                          'with the &plusmn;1 std and min&ndash;max clouds of the daily path</div>' % m)
            panels.append('<div class="bandkey"><span class="ln" style="background:%s"></span>'
                          'cum of each round\'s latest pending'
                          '<span class="bx" style="background:%s;opacity:.25"></span>'
                          'cum &plusmn;1 std of daily path'
                          '<span class="bx" style="background:%s;opacity:.12"></span>'
                          'bound: cum low &hellip; high</div>'
                          % (PENDING_COLOUR, PENDING_COLOUR, PENDING_COLOUR))
            panels.append(band_chart(cum_line, band, PENDING_COLOUR))
        # Same rounds, same boxes, but each daily value divided by that round's
        # own stake: the round's return. Stake size drops out, so rounds staked
        # at 3 NMR and at 45 NMR sit on one scale.
        rrows = [r for r in rows if r["at_risk"]]
        if rrows:
            rpaths = [[v / r["at_risk"] for v in r["path"]] for r in rrows]
            rets = [p[-1] for p in rpaths]
            mean_ret = float(np.mean(rets))
            ret_panels.append('<details class="panel"%s><summary>%s &mdash; %d YTD rounds '
                              '&middot; mean round return <span class="%s">%+.2f%%</span>'
                              '</summary><div class="pbody">'
                              % (open_attr, m, len(rrows), klass(mean_ret), 100 * mean_ret))
            ret_panels.append('<div class="ctitle">%s &mdash; spread of the round stake return '
                              '(payout / stake) across each round\'s scoring days</div>' % m)
            ret_panels.append('<div class="bandkey">'
                              '<span class="dia" style="color:%s">&#9670;</span>latest (current pending)'
                              '<span class="dia" style="color:%s">&#9670;</span>resolved (settled)</div>'
                              % (PENDING_COLOUR, RESOLVED_COLOUR))
            ret_panels.append(box_plot([r["round"] for r in rrows],
                                       [box_stats(p) for p in rpaths], rets, PENDING_COLOUR,
                                       paths=rpaths, pct=True,
                                       colours=[RESOLVED_COLOUR if r["resolved"] else PENDING_COLOUR
                                                for r in rrows]))
            ret_panels.append("</div></details>")
        else:
            panels.append('<p class="cap">No scored rounds yet &mdash; nothing to plot.</p>')
        panels.append("</div></details>")
    ctx["panels"] = panels
    ctx["ret_panels"] = ret_panels
    ctx["per"] = per
    ctx["order"] = order
    return ctx


def add_correlation(ctx, live, slot_of):
    """Attach sections 4 and 5 to a context built by `build`."""
    order = ctx["order"]

    # ---- 4 realised ------------------------------------------------------ #
    r, n, cwmm = realised_matrix(live, slot_of)
    rows = [m for m in order if m in r.index]
    matrix = r.loc[rows, rows].copy()
    counts = {(a, b): "n = %d shared rounds" % n.loc[a, b]
              for a in rows for b in rows}
    matrix[META] = [cwmm.get(m) for m in rows]
    meta_row = pd.Series({m: cwmm.get(m) for m in rows}, dtype=float)
    meta_row[META] = 1.0
    matrix.loc[META] = meta_row
    for m in rows:
        counts[(m, META)] = counts[(META, m)] = "mean cwmm over live rounds"
    ctx["realised"] = heatmap(matrix, counts, fenced=META)
    ctx["realised_r"] = r
    ctx["realised_n"] = n

    # ---- 5 backtested ---------------------------------------------------- #
    bt, mm = backtest_frames(order)
    bt = bt.rename(columns={c: c for c in bt.columns})
    corr = bt.corr()
    if len(mm):
        joined = pd.concat([bt, mm.rename(META)], axis=1, join="inner").dropna()
        corr = joined.corr()
        n_meta = len(joined)
    else:
        n_meta = 0
    ctx["bt_corr"] = heatmap(corr, fenced=None)
    ctx["bt_corr_frame"] = corr
    ctx["bt_note"] = (
        "Pearson of per-period CORR on the %d dates every slot shares. The Meta Model is a genuine "
        "row here: its own per-period Spearman against <code>target_binned_return_20</code>, on the "
        "%d dates where it exists. Hover any cell for n."
        % (len(bt), n_meta))

    pred, pred_dates = prediction_matrix(order)
    if pred is not None:
        keep = [c for c in order + [META] if c in pred.columns]
        ctx["bt_pred"] = heatmap(pred.loc[keep, keep])
    else:
        ctx["bt_pred"] = '<p class="cap">No prediction panel available.</p>'

    # ---- pair by pair ---------------------------------------------------- #
    pairs = []
    for a, b in itertools.combinations(order, 2):
        if a not in r.index or b not in r.index:
            continue
        live_r = r.loc[a, b]
        back_r = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
        pairs.append((a, b, live_r, back_r, int(n.loc[a, b])))
    scored = [p for p in pairs if not pd.isna(p[2]) and not pd.isna(p[3])]
    scored.sort(key=lambda p: -(p[2] - p[3]))

    table = ['<table class="data"><tr><th class="l">Pair</th><th>Realised r</th>'
             '<th>Backtested r</th><th>Difference</th><th>Rounds live</th></tr>']
    for a, b, lr, br, rounds in scored:
        diff = lr - br
        table.append('<tr><td class="l"><strong>%s &times; %s</strong></td><td>%.2f</td>'
                     '<td>%.2f</td><td class="%s">%+.2f</td><td>%d</td></tr>'
                     % (a, b, lr, br, klass(diff), diff, rounds))
    for a, b, lr, br, rounds in pairs:
        if pd.isna(lr):
            table.append('<tr><td class="l">%s &times; %s</td><td class="flat">&mdash;</td>'
                         '<td>%.2f</td><td class="flat">&mdash;</td><td>%d</td></tr>'
                         % (a, b, br, rounds))
    if scored:
        mean_live = float(np.mean([p[2] for p in scored]))
        mean_back = float(np.mean([p[3] for p in scored]))
        table.append('<tr class="total"><td class="l">mean over %d comparable pairs</td>'
                     '<td>%.2f</td><td>%.2f</td><td class="%s">%+.2f</td><td>&mdash;</td></tr>'
                     % (len(scored), mean_live, mean_back, klass(mean_live - mean_back),
                        mean_live - mean_back))
    table.append("</table>")
    ctx["pair_table"] = "".join(table)

    if scored:
        gap = mean_live - mean_back
        thin = min(p[4] for p in scored)
        direction = "more" if gap > 0 else "less"
        ctx["verdict"] = (
            '<div class="warn"><strong>The live scores are %s correlated than the backtest says '
            '&mdash; on a sample far too short to act on.</strong> Mean pairwise correlation is %.2f '
            'live against %.2f in the backtest, a gap of %+.2f, but the thinnest of these pairs rests '
            'on %d shared rounds against the backtest\'s %d dates. At that length a correlation is '
            'mostly noise, and the rounds themselves overlap ~95%% because the target is a 20-day '
            'forward return, so the effective sample is smaller again. Read this panel as a tripwire '
            'to re-check in a month, not as evidence that the covariance behind the stake weights is '
            'wrong. If the gap survives to 30 rounds it is real, and sizing these as diversified '
            'positions on the backtested matrix would be overstating the diversification obtained.'
            '</div>' % (direction, mean_live, mean_back, gap, thin, len(bt)))
    else:
        ctx["verdict"] = ('<div class="note">No pair has %d shared scored rounds yet, so there is '
                          'nothing to compare. This panel fills in as the ladder ages.</div>'
                          % MIN_LIVE_ROUNDS)

    ctx["bt_footer"] = (
        "Built from the slots' stored per-period series through <code>src.analysis.slots</code>, so "
        "CURUPIRA_07 resolves to its prediction-level reconstruction and CURUPIRA_05_60 keeps its "
        "60-day label. Prediction panel: %d dates."
        % (pred_dates if pred is not None else 0))
    return ctx


def subject(ctx):
    """The email subject line, carrying the verdict so it triages unopened."""
    per, order = ctx["per"], ctx["order"]
    risk = sum(per[m]["risk"] for m in order)
    pl = sum(per[m]["open_pl"] for m in order)
    book = risk + pl
    return ("[numerai-crypto] Stake report -- book %s NMR - provisional %s NMR (%+.2f%%)"
            % (num(book, 2), signed(pl), 100 * pl / risk if risk else 0.0))


def send_email(subject_line, body_text, attach_path):
    """
    Mail the plain-text summary as the body, the full report as an attachment.

    The body is text/plain -- the at-a-glance figures readable in the message
    itself -- and the interactive HTML rides as an attachment the reader opens in
    a browser, rather than as the body. Two reasons the HTML does not belong
    inline: mail clients strip its inline SVG (the charts vanish) and mangle its
    layout, so an inline render is a worse version of the file; and a short text
    body triages from the phone lock screen without loading anything.

    Same rule as src.live.notify: a mail failure must never fail the job. The
    file is already on disk by the time this runs, so the report is not lost.
    """
    import mimetypes
    import smtplib
    from email.message import EmailMessage
    from src.live.env import load_env

    load_env(required=())
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to = os.environ.get("NOTIFY_TO")
    if not (host and user and password and to):
        print("SMTP not configured; not mailing")
        return False

    message = EmailMessage()
    message["Subject"] = subject_line
    message["From"] = user
    message["To"] = to
    message.set_content(body_text)                      # the body IS the summary
    try:
        with open(attach_path, "rb") as handle:
            data = handle.read()
        subtype = (mimetypes.guess_type(attach_path)[0] or "text/html").split("/", 1)[1]
        message.add_attachment(data, maintype="text", subtype=subtype,
                               filename=os.path.basename(attach_path))
    except OSError as error:
        # No attachment is better than no mail: the body still carries the book.
        print("could not attach %s (%s); sending body only" % (attach_path, error))

    try:
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587)), timeout=60) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(message)
        print("mailed to %s" % to)
        return True
    except Exception as error:
        print("mail FAILED: %s" % str(error)[:200])
        return False


def main():
    parser = argparse.ArgumentParser(description="Build the Numerai Crypto stake report.")
    parser.add_argument("--email", action="store_true", help="also mail the report")
    parser.add_argument("--out", default=os.path.join(OUT_DIR, OUT_FILE))
    parser.add_argument("--no-fx", action="store_true", help="skip the FX lookup")
    args = parser.parse_args()

    built = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    slot_of = dict(STAKED_SLOTS)
    live = fetch_live([s for s, _ in STAKED_SLOTS])
    if not live:
        raise SystemExit("no staked crypto models found -- nothing to report")

    rates = (None, None, "not fetched") if args.no_fx else nmr_rates()
    ctx = build(live, rates, built, slot_of)
    ctx = add_correlation(ctx, live, slot_of)
    ctx["footer"] = (
        "Generated by <code>src/analysis/crypto_stake_report.py</code>. Stakes and scores from "
        "<code>v2RoundModelPerformances(tournament: %d)</code>; NMR prices from CoinGecko. Payouts "
        "are Numerai's own <code>payoutPending</code>, not reconstructed &mdash; the crypto payout "
        "is 0.1&middot;CORR + 1&middot;MMC on the round's <code>atRisk</code>." % TOURNAMENT_ID)

    html = render(ctx)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write(html)
    snapshot = save_snapshot(
        {slot_of[s]: v["rounds"] for s, v in live.items()}, built)

    line = subject(ctx)
    print("wrote %s (%d bytes)" % (args.out, os.path.getsize(args.out)))
    print("snapshot %s" % snapshot)
    print("SUBJECT: %s" % line)
    if args.email:
        send_email(line, ctx["plain_text"], args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
