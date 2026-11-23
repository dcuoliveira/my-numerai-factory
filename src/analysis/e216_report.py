"""E216 report: JACI_01 live-path gates, forward check and slot dry-run (HTML -> PDF).

    python3 -m src.analysis.e216_report
"""
import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/reports/crypto/e216_jaci_live")


def load(name):
    path = ROOT / name
    return json.loads(path.read_text()) if path.exists() else None


def cumulative_png(daily):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.2))
    d = daily.assign(date=pd.to_datetime(daily.date)).set_index("date")
    ax.plot((.1 * d["corr"] + d["mmc"]).cumsum(), label="payout (0.1 corr + mmc)", color="#1f5f8b")
    ax.plot(d["corr"].cumsum(), label="corr", color="#999999", lw=1)
    ax.axhline(0, color="black", lw=.5)
    ax.set_title("JACI_01 live path on resolved dates after the frozen test (cumulative)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    feats, heads, blend = load("parity_features.json"), load("parity_heads.json"), load("parity_blend.json")
    states, fwd, dry = load("parity_states.json"), load("forward_check.json"), load("dryrun.json")
    rows = []
    if feats:
        rows.append(("Features + coverage (E205 functions, rebuilt from %s)" % feats["from_date"],
                     "%d frozen rows: %d coverage, %d value mismatches" % (
                         feats["rows"], feats["coverage_mismatches"], feats["value_mismatches"]),
                     feats["passed"]))
    for h in heads or []:
        n = h["normalization"]
        rows.append(("Normalization %s (w%d)" % (h["stream"], n["window"]),
                     "%d rows, max |diff| %.1e" % (n["rows"], n["max_abs_diff"]), n["passed"]))
        rows.append(("Kernel + frozen heads %s vs E206 test predictions" % h["stream"],
                     "%d rows, max |diff| %.1e" % (h["rows"], h["max_abs_diff"]), h["passed"]))
    if blend:
        rows.append(("Equal-rank blend + 0.75 resid vs E214 FIXED_s0.75",
                     "%d rows, max |diff| %.1e" % (blend["rows"], blend["max_abs_diff"]), blend["passed"]))
    for s in states or []:
        rows.append(("Re-extracted frozen states, %s" % s["model"],
                     "%s probes, max rel diff %s" % (s.get("probes"), "%.1e" % s["max_rel_diff"]
                                                     if s.get("max_rel_diff") is not None else "-"),
                     s["passed"]))
    table = "".join("<tr><td>%s</td><td>%s</td><td class='%s'>%s</td></tr>" % (
        a, b, "ok" if c else "bad", "PASS" if c else "FAIL") for a, b, c in rows)
    fwd_html = ""
    if fwd:
        daily = pd.read_csv(ROOT / "forward_daily.csv")
        fwd_html = ("<h2>Forward check (not used for any choice)</h2><p>The live path scored on the %d "
                    "resolved dates %s to %s, after the frozen E206 test period that selected JACI "
                    "(mean %.0f coins/date, JACI-covered rows only, residualised 0.75 on the meta "
                    "lagged 5 days; MMC against the true meta model).</p>"
                    "<table><tr><th>CORR</th><th>MMC</th><th>payout</th><th>p.sharpe</th></tr>"
                    "<tr><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.3f</td></tr></table>"
                    "<p>E214 FIXED_s0.75 on its 297 test dates: CORR 0.1002, MMC 0.0710, payout 0.0810, "
                    "p.sharpe 1.028. Different dates and a wider coin set, so this is a health "
                    "check, not a paired comparison.</p><img src='data:image/png;base64,%s'/>"
                    % (fwd["dates"], fwd["first"], fwd["last"], fwd["mean_assets"], fwd["corr"],
                       fwd["mmc"], fwd["payout"], fwd["payout_sharpe"], cumulative_png(daily)))
    dry_html = ""
    if dry:
        r = dry["route"]
        dry_html = ("<h2>Slot dry-run (live date %s)</h2><p>CU_SA_IA_BO_MA_JA_01 assembled with "
                    "run_round's own prepare_slot and submit.predict, cache only, nothing uploaded: "
                    "%d rows, %d unique symbols, all finite in (0,1): %s. JACI covered %d of %d coins "
                    "(pass-through: %s); its residualisation beta %.3f on meta snapshot %s. %d coins "
                    "moved relative to the CU leg; Spearman(slot, CU leg) = %.4f. Warnings: %s.</p>"
                    % (dry["live_date"], dry["rows"], dry["unique"], dry["finite"] and dry["in_unit_interval"],
                       r["covered"], r["rows"], r["pass_through"], dry["jaci"]["beta"],
                       dry["meta"]["snapshot"], dry["moved_rows"], dry["spearman_slot_vs_leg"],
                       "; ".join(dry["warnings"]) or "none"))
    html = """<html><head><meta charset='utf-8'><style>
body{font-family:Helvetica,Arial,sans-serif;font-size:11px;margin:28px;color:#222}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 7px;text-align:left}
.ok{color:#176b2c;font-weight:bold}.bad{color:#a11;font-weight:bold}img{width:100%%}</style></head><body>
<h1>E216 &mdash; JACI_01 live path for CU_SA_IA_BO_MA_JA_01</h1>
<p>JACI_01 = E214 FIXED_s0.75 (equal-rank Chronos-2 rolling, t0 cross_section, t0 rolling, Timestral
cross_section, all design C; 0.75 residualisation on the meta lagged &ge;5 days). The slot routes the E215
50/50 rank blend with CU_SA_IA_BO_MA_R_01 into the CU leg's rank slots on JACI-covered coins. Heads are
E206's frozen final fits (training rows &le; 2024-11-27); nothing was refitted or re-selected.</p>
<h2>Parity gates</h2><table><tr><th>gate</th><th>evidence</th><th></th></tr>%s</table>
%s%s
<h2>Limits</h2><ul><li>Prices come from free vendors (yahoo + exchanges), one source per coin, refreshed
daily; Numerai's feed is undisclosed, so coverage and values can drift from what the backtest saw.</li>
<li>The cross-sectional streams normalise over the coins covered on the date; the pre-stage uses the
previous round's universe as a proxy for today's, so a changed universe shifts them slightly.</li>
<li>JACI covers only coins with &ge;130 clean daily bars and a fitted head; the rest get the CU leg's rank.</li>
<li>E215's 0.1207 payout is a same-history selection on 297 common dates, not an out-of-sample estimate.</li></ul>
</body></html>""" % (table, fwd_html, dry_html)
    path = ROOT / "E216_jaci_live_path.html"
    path.write_text(html)
    subprocess.run([sys.executable, "-m", "src.analysis.html_pdf", str(path)], check=True)
    print(path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
