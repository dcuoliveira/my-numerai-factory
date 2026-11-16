"""
P&L in BRL, day by day, for every crypto model that has ever carried stake --
settled rounds and still-open (provisional) rounds, each row tagged.

    python3 -m src.analysis.crypto_realised_pnl            # CSV + HTML + PDF
    python3 -m src.analysis.crypto_realised_pnl --no-pdf

Every round is placed on the day it resolves (`roundResolveTime`). A SETTLED
round carries its final payout, converted at that day's NMR/BRL price (CoinGecko
daily, UTC). A PROVISIONAL round is still open: its payout is Numerai's current
mark, revised daily until it settles, and its resolve date is in the future, so
it is converted at the latest price available. Open rounds not yet scored have
no payout and are left out (counted in the header). The bottom rows give the
settled-only and the full (provisional included) totals and Sharpes.

Every model on the account is scanned, not just `crypto_stake_report.STAKED_SLOTS`,
so a model that was staked and later unstaked still has its history here. A
model deleted from the account cannot be seen through the API and is absent.

Rows are resolution days (Numerai resolves crypto rounds on weekdays, so
weekends have no rows); the last column, TOTAL, sums every model on that day,
and the last row sums every day.
"""

import argparse
import json
import os
import urllib.request

import pandas as pd

TOURNAMENT_ID = 12
OUT_DIR = "results/reports/crypto"
OUT_STEM = "realised_pnl_brl"

QUERY = """
query($m: ID!, $t: Int!) {
  v2RoundModelPerformances(modelId: $m, tournament: $t, lastNRounds: 2000) {
    roundNumber atRisk payout roundResolved roundResolveTime
  }
}"""


COLUMNS = ["model", "round", "at_risk", "payout_nmr", "date", "settled"]


def fetch_settlements():
    """
    Every staked round of every model on the account, settled or provisional.

    :return: (one row per priced (model, round) with COLUMNS, number of staked
              rounds not yet scored and therefore left out)
    :rtype: (pd.DataFrame, int)
    """
    from src.live.env import get_api, load_env

    load_env()
    api = get_api()
    rows, unpriced = [], 0
    for slug, model_id in sorted((api.get_models() or {}).items()):
        payload = api.raw_query(QUERY, {"m": model_id, "t": TOURNAMENT_ID},
                                authorization=True)["data"]["v2RoundModelPerformances"] or []
        for item in payload:
            at_risk = float(item.get("atRisk") or 0)
            if at_risk <= 0 or not item.get("roundResolveTime"):
                continue
            if item.get("payout") is None:
                unpriced += 1
                continue
            rows.append({"model": slug, "round": int(item["roundNumber"]), "at_risk": at_risk,
                         "payout_nmr": float(item["payout"]),
                         "date": item["roundResolveTime"][:10],
                         "settled": bool(item.get("roundResolved"))})
    return pd.DataFrame(rows, columns=COLUMNS), unpriced


def brl_history(days=365):
    """
    NMR/BRL by UTC date from CoinGecko's daily series.

    :return: {"YYYY-MM-DD": brl_per_nmr}
    :rtype: dict
    """
    url = ("https://api.coingecko.com/api/v3/coins/numeraire/market_chart"
           "?vs_currency=brl&days=%d&interval=daily" % days)
    with urllib.request.urlopen(url, timeout=30) as handle:
        prices = json.load(handle)["prices"]
    out = {}
    for stamp, price in prices:          # later points of the same date overwrite earlier
        out[pd.Timestamp(stamp, unit="ms", tz="UTC").strftime("%Y-%m-%d")] = float(price)
    return out


def build_table(settled, rates):
    """
    Day x model P&L in BRL, plus a TOTAL column.

    A day with no price of its own takes the most recent earlier price
    (CoinGecko occasionally skips a day, and a provisional round's resolve date
    is in the future, so it takes today's); one with no earlier price at all is
    an error rather than a silent zero.

    :param settled: fetch_settlements() frame (settled and provisional rows)
    :param rates: {"YYYY-MM-DD": brl_per_nmr}
    :return: (brl, per-day rate, models in column order, stake, return,
              per-day status "settled" / "provisional" / "mixed")
    """
    rate = pd.Series(rates, dtype=float)
    rate.index = pd.to_datetime(rate.index)
    rate = rate.sort_index()
    dates = pd.to_datetime(sorted(settled["date"].unique()))
    day_rate = rate.reindex(rate.index.union(dates)).ffill().reindex(dates)
    if day_rate.isna().any():
        missing = [d.strftime("%Y-%m-%d") for d in day_rate[day_rate.isna()].index]
        raise SystemExit("no NMR/BRL price on or before %s -- widen --fx-days" % missing[0])

    nmr = settled.pivot_table(index="date", columns="model", values="payout_nmr",
                              aggfunc="sum", fill_value=0.0)
    nmr.index = pd.to_datetime(nmr.index)
    # columns: first-staked model first, so the table reads left to right in time
    order = settled.groupby("model")["date"].min().sort_values().index.tolist()
    brl = nmr[order].mul(day_rate, axis=0)
    brl["TOTAL"] = brl.sum(axis=1)
    # NMR at risk on the rounds settling that day, same shape as brl: the stake
    # is what makes one day's R$ large and another's small at the same return
    stake = settled.pivot_table(index="date", columns="model", values="at_risk",
                                aggfunc="sum", fill_value=0.0)
    stake.index = pd.to_datetime(stake.index)
    stake = stake[order]
    stake["TOTAL"] = stake.sum(axis=1)
    # return on stake per settlement day: payout NMR / stake NMR. FX cancels, so
    # this is the same in NMR and BRL; TOTAL is the stake-weighted book return.
    nmr = nmr[order]
    nmr["TOTAL"] = nmr.sum(axis=1)
    ret = (nmr / stake.where(stake > 0)).astype(float)
    flags = settled.groupby("date")["settled"].agg(["all", "any"])
    flags.index = pd.to_datetime(flags.index)
    status = pd.Series(["settled" if a else "mixed" if y else "provisional"
                        for a, y in zip(flags["all"], flags["any"])], index=flags.index)
    return brl, day_rate, order, stake, ret, status.reindex(brl.index)


#: Each settlement is one round's ~20-day holding-period return, and a new round
#: starts daily, so consecutive days overlap ~95%. Annualise per ROUND (365/20 a
#: year), never by sqrt(252): that treats overlapping returns as independent.
ANNUAL_ROUNDS = 365.0 / 20.0


def annual_sharpe(returns):
    """mean/std (ddof=0) of the per-settlement returns x sqrt(365/20); None under 3."""
    r = returns.dropna()
    if len(r) < 3 or r.std(ddof=0) == 0:
        return None
    return float(r.mean() / r.std(ddof=0) * ANNUAL_ROUNDS ** 0.5)


def render_html(brl, day_rate, settled, built, stake, ret, status, unpriced):
    def cell(x, cls=""):
        if abs(x) < 5e-5:
            return '<td class="z%s">&middot;</td>' % cls
        return '<td class="%s%s">%s</td>' % ("pos" if x > 0 else "neg", cls, "{:+,.2f}".format(x))

    def ret_cell(x):
        if pd.isna(x):
            return '<td class="z">&middot;</td>'
        return '<td class="%s">%+.2f%%</td>' % ("pos" if x > 0 else "neg" if x < 0 else "z", 100 * x)

    def stake_cell(x, cls):
        if abs(x) < 5e-9:
            return '<td class="z st%s">&middot;</td>' % cls
        return '<td class="st%s">%s</td>' % (cls, "{:,.2f}".format(x))

    def sharpe_cell(x):
        if x is None:
            return '<td class="z">&mdash;</td>'
        return '<td class="%s">%+.2f</td>' % ("pos" if x > 0 else "neg", x)

    cols = list(brl.columns)
    group = lambda c: " tg" if c == "TOTAL" else " g"
    head = ('<tr><th class="l" rowspan="2">Date</th><th class="l" rowspan="2">Status</th>'
            '<th rowspan="2">NMR/BRL</th>%s</tr><tr>%s</tr>'
            % ("".join('<th colspan="3" class="c%s">%s</th>' % (group(c), c) for c in cols),
               "".join('<th class="st%s">stake NMR</th><th>return</th><th>R$</th>' % group(c)
                       for c in cols)))
    body = []
    for day, row in brl.iterrows():
        tag = status[day]
        body.append('<tr class="%s"><td class="l">%s</td><td class="l"><span class="tag %s">%s</span></td>'
                    '<td>%.2f</td>%s</tr>'
                    % ("prov" if tag != "settled" else "", day.strftime("%Y-%m-%d"), tag, tag,
                       day_rate[day],
                       "".join(stake_cell(stake.at[day, c], group(c)) + ret_cell(ret.at[day, c])
                               + cell(row[c]) for c in cols)))

    is_settled = (status == "settled").values
    windows = [("settled", is_settled), ("full (provisional incl.)", [True] * len(brl))]
    staked_brl = stake.mul(day_rate, axis=0)                   # R$ value of each day's stake
    figures = {}
    for label, mask in windows:
        totals = brl[mask].sum(axis=0)
        base = staked_brl[mask].sum(axis=0)
        pooled = {c: totals[c] / base[c] if base[c] else float("nan") for c in cols}
        sharpe = {c: annual_sharpe(ret.loc[mask, c]) for c in cols}
        figures[label] = (totals, sharpe)
        body.append('<tr class="tot"><td class="l" colspan="2">TOTAL %s</td><td></td>%s</tr>'
                    % (label, "".join('<td class="%s"></td>' % group(c).strip() + ret_cell(pooled[c])
                                      + cell(totals[c]) for c in cols)))
    for k, (label, _) in enumerate(windows):
        body.append('<tr class="%s"><td class="l" colspan="2">Sharpe (ann.) %s</td><td></td>%s</tr>'
                    % ("tot" if k == 0 else "tot2", label,
                       "".join('<td class="%s"></td>' % group(c).strip() + sharpe_cell(figures[label][1][c])
                               + "<td></td>" for c in cols)))

    set_tot, full_tot = figures["settled"][0], figures["full (provisional incl.)"][0]
    n_set, n_prov = int(settled["settled"].sum()), int((~settled["settled"]).sum())
    summary = ("%d models with stake, %d settled and %d provisional rounds over %d days "
               "(%s &rarr; %s); %d staked rounds not yet scored are left out. Settled <b>R$ %s</b>, "
               "provisional R$ %s, full <b>R$ %s</b>. Sharpe (ann.) settled %s, full %s."
               % (len(cols) - 1, n_set, n_prov, len(brl), brl.index[0].strftime("%Y-%m-%d"),
                  brl.index[-1].strftime("%Y-%m-%d"), unpriced,
                  "{:+,.2f}".format(set_tot["TOTAL"]),
                  "{:+,.2f}".format(full_tot["TOTAL"] - set_tot["TOTAL"]),
                  "{:+,.2f}".format(full_tot["TOTAL"]),
                  *["&mdash;" if figures[w][1]["TOTAL"] is None else "%+.2f" % figures[w][1]["TOTAL"]
                    for w, _ in windows]))
    per_model = []
    for m in cols[:-1]:
        mine = settled[settled["model"] == m]
        per_model.append(
            '<tr><td class="l">%s</td><td>%d</td><td>%d</td><td>%s</td><td>%s</td>%s%s%s%s%s</tr>'
            % (m, mine["settled"].sum(), (~mine["settled"]).sum(), mine["date"].min(), mine["date"].max(),
               cell(set_tot[m]), cell(full_tot[m] - set_tot[m]), cell(full_tot[m]),
               sharpe_cell(figures["settled"][1][m]),
               sharpe_cell(figures["full (provisional incl.)"][1][m])))
    return """<!doctype html><html><head><meta charset="utf-8">
<title>P&amp;L in BRL</title>
<style>
body{font:13px/1.4 -apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#202124;background:#fff;margin:24px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 6px}
p.n{color:#5f6368;margin:4px 0 12px;max-width:980px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:3px 8px;border-bottom:1px solid #e8eaed;text-align:right;white-space:nowrap}
th{background:#f1f3f4;font-weight:600;position:sticky;top:0}
td.l,th.l{text-align:left}.pos{color:#137333}.neg{color:#c5221f}.z{color:#bdc1c6}
tr.tot td{font-weight:700;border-top:2px solid #5f6368}tr.tot2 td{font-weight:700}
tr.prov td{background:#fef7e0}
.tag{display:inline-block;padding:0 6px;border-radius:8px;font-size:11px;font-weight:600}
.tag.settled{background:#e8f0fe;color:#1a73e8}.tag.provisional,.tag.mixed{background:#fce8e6;color:#c5221f}
th.c{text-align:center}.g{border-left:1px solid #dadce0}.tg{border-left:2px solid #5f6368}
td.st{color:#5f6368}td:last-child{font-weight:700}
</style></head><body>
<h1>P&amp;L in BRL by day and model: settled and provisional</h1>
<p class="n">Built %s &middot; %s</p>
<p class="n">Each round sits on the day it <b>resolves</b>. <span class="tag settled">settled</span>
rows carry the final payout at that day's NMR/BRL price (CoinGecko daily, UTC).
<span class="tag provisional">provisional</span> rows (shaded) are open rounds: the payout is
Numerai's current mark, revised daily until the round settles, and the date is the scheduled
resolution, so it is converted at the latest NMR/BRL price. Each model's <i>stake NMR</i> is the
NMR at risk on that day's round(s); <i>return</i> is payout NMR / stake NMR (FX cancels, so it is
the same in NMR and BRL), TOTAL's being the stake-weighted book return; the TOTAL rows' return is
R$ over R$ staked. <i>Sharpe (ann.)</i> = mean / std (ddof=0) of the daily returns &times;
&radic;(365/20): each day is one round's ~20-day return and rounds overlap, so &radic;252 would
overstate it ~4.5&times;; needs 3+ days. <i>Settled</i> uses settled rows only, <i>full</i> adds
the provisional ones. Every model on the account is scanned; a model deleted from the account is
not visible to the API.</p>
<h2>Per model</h2>
<table><tr><th class="l">Model</th><th>Settled</th><th>Provisional</th><th>First</th><th>Last</th>
<th>Settled R$</th><th>Provisional R$</th><th>Full R$</th><th>Sharpe settled</th><th>Sharpe full</th></tr>%s</table>
<h2>Day by day</h2>
<table>%s%s</table>
</body></html>""" % (built, summary, "".join(per_model), head, "".join(body))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--fx-days", type=int, default=365)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--pdf-host", default="lovelace")
    args = parser.parse_args()

    built = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    settled, unpriced = fetch_settlements()
    if settled.empty:
        raise SystemExit("no priced staked rounds on the account")
    brl, day_rate, _, stake, ret, status = build_table(settled, brl_history(args.fx_days))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, OUT_STEM)
    out = pd.concat({"stake_nmr": stake, "return": ret, "pnl_brl": brl}, axis=1).swaplevel(axis=1)
    out = out[[(c, k) for c in brl.columns for k in ("stake_nmr", "return", "pnl_brl")]]
    out.columns = ["%s_%s" % pair for pair in out.columns]
    out.insert(0, "nmr_brl", day_rate)
    out.insert(0, "status", status)
    done = (status == "settled").values
    for label, mask in (("SETTLED", done), ("FULL", [True] * len(brl))):
        out.loc["TOTAL_" + label] = [brl.loc[mask, c[:-8]].sum() if c.endswith("_pnl_brl") else None
                                     for c in out.columns]
    for label, mask in (("SETTLED", done), ("FULL", [True] * len(brl))):
        out.loc["SHARPE_ANN_" + label] = [annual_sharpe(ret.loc[mask, c[:-7]]) if c.endswith("_return")
                                          else None for c in out.columns]
    out.index = [d if isinstance(d, str) else d.strftime("%Y-%m-%d") for d in out.index]
    out.to_csv(stem + ".csv", index_label="date", float_format="%.4f")
    with open(stem + ".html", "w") as handle:
        handle.write(render_html(brl, day_rate, settled, built, stake, ret, status, unpriced))
    print("wrote %s.csv and %s.html" % (stem, stem))
    if not args.no_pdf:
        from src.analysis import html_pdf
        try:
            html_pdf.to_pdf(stem + ".html", args.pdf_host)
        except Exception as error:
            # deepfive's chromium exits 1 under headless print; home is shared
            # NFS, so print the same file on a box whose google-chrome works
            print("local PDF print failed (%s); printing on %s" % (str(error)[:80], args.pdf_host))
            import subprocess
            subprocess.run(["ssh", "-o", "BatchMode=yes", args.pdf_host,
                            "cd %s && /home/danielco/miniconda3/bin/python3 -m src.analysis.html_pdf %s"
                            % (os.getcwd(), os.path.abspath(stem + ".html"))],
                           check=True, timeout=600)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(brl.round(2).assign(status=status).to_string())
        print("TOTAL settled R$ %.2f  full R$ %.2f  Sharpe settled %s  full %s"
              % (brl.loc[done, "TOTAL"].sum(), brl["TOTAL"].sum(),
                 annual_sharpe(ret.loc[done, "TOTAL"]), annual_sharpe(ret["TOTAL"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
