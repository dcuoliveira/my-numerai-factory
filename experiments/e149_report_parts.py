"""E149-only sections for e138_report.render: the builder's audit, and the contrasts inside the proposed volatility block."""
import json
import os

import pandas as pd

DIR = "data/crypto/v2.0-vol-causal"
ROOT = "results/reports/crypto/e149_vol_causal"


def assemble():
    """audit.json for Study: the builder's audit and gate statistics, plus coverage on the scored keys."""
    a = json.load(open(os.path.join(DIR, "audit.json")))
    m = json.load(open(os.path.join(DIR, "manifest.json")))
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    a["gate"], a["attached"], a["gated_out"] = m["gate"], m["attached"], m["gated_out"]
    a["keys_observed"] = float((o["obs_tf"] > 0).mean())
    a["keys_no_feed"] = float(o["no_feed"].mean())
    json.dump(a, open(os.path.join(ROOT, "audit.json"), "w"), indent=1, default=str)
    return a


def audit(s, W):
    au = s.AU
    tr, g, v = au["truncation"], au["gate"], au["vs_old_vol"]
    H = ['<h2 id="audit">Point-in-time audit</h2><ul>',
         '<li><b>No later data is used:</b> the builder rebuilt the block from the spot store cut at %d dates (%s) and compared row D with the dataset: %s of %s cells differ.</li>'
         % (len(tr), ", ".join(t["date"] for t in tr), "{:,}".format(sum(t["differ"] for t in tr)), "{:,}".format(sum(t["cells"] for t in tr))),
         '<li><b>Which coins count:</b> a series enters only behind a causal identity gate (expanding mean gap between our rebuilt 20/60-day volatility percentile and the shipped one, '
         'frozen threshold %.2f). It passes %.1f%% of right-coin rows and %.1f%% of wrong-coin rows under a deranged null; %.1f%% of attached rows were gated out. '
         'The shipped 22 features are copied unchanged (%s).</li>'
         % (g["threshold"], 100 * g["right_coin_pass_rate"], 100 * g["wrong_coin_pass_rate"], 100 * au["gated_out"], "verified" if au["shipped_identical"] else "<b>NOT identical</b>"),
         '<li><b>The candidate is <code>feature_vol_*</code> only.</b> The shipped <code>feature_volatility_20d</code> and <code>_60d</code> are base features: they are in B0 and in every arm. '
         'Where both exist, the rebuilt 20/60-day close-to-close bin equals the shipped one on %.1f%% of rows. F_SWAP drops the shipped two to ask whether the block can stand in for them.</li>' % (100 * v["both_equal"]),
         '<li><b>Coverage on the scored keys:</b> %.1f%% of keys have the block; %.1f%% belong to a coin that never passes the gate anywhere in the file (P_NM&#39;s flag). %d of %d panel coins are ever covered.</li>'
         % (100 * au["keys_observed"], 100 * au["keys_no_feed"], au["coins_ever_covered"]["n"], au["coins_ever_covered"]["of"]),
         '<li><b>Scale and redundancy:</b> level bins carry mass %s; own-history ranks %s (ties at the per-date binning). Mean |Spearman| among the 132 ranks is %.2f, %d pairs above 0.8. '
         'A rank moves away from its level as the window shortens (sqret 20d: %.2f at 365 days, %.2f at 7).</li>'
         % (" / ".join("%.3f" % x for x in au["mass"].values()), " / ".join("%.3f" % x for x in au["tsr_mass"].values()), au["tsr_redundancy"]["mean_abs"],
            au["tsr_redundancy"]["pairs_over_08"], au["tsr_vs_level_spearman"]["feature_vol_sqret_20d"]["365"], au["tsr_vs_level_spearman"]["feature_vol_sqret_20d"]["7"]),
         '<li><b>Not point-in-time:</b> the spot store was fetched in 2026, so which coins have prices is a retrospective reconstruction. Routing keeps an uncovered coin on v2.0&#39;s rank, but the block is judged on coins that survived into the store.</li></ul>']
    y = pd.DataFrame(au["coverage_by_year"])
    H += [W.table(["year", "short levels observed", "all 22 levels observed"], [[int(r.date), "%.1f%%" % (100 * r.cov_short), "%.1f%%" % (100 * r.cov_all)] for r in y.itertuples()])]
    return H


def extra(s, W, HDR):
    return ['<h2 id="inside">Inside the block</h2>',
            '<h3>Levels against own-history ranks</h3>', W.table(HDR, s.rows([("FULL", "F_LVL"), ("FULL", "F_TSR"), ("F_TSR", "F_LVL")]), "l2"),
            '<h3>Short against long windows</h3>', W.table(HDR, s.rows([("F_LONG", "F_SHORT")]), "l2"),
            '<h3>Can the block replace the shipped volatility?</h3><p>F_SWAP is FULL without <code>feature_volatility_20d</code> / <code>_60d</code>. Against FULL it prices the two shipped columns given the block; against v2.0 it asks whether the block alone stands in for them.</p>',
            W.table(HDR, s.rows([("F_SWAP", "FULL"), ("F_SWAP", "B0")]), "l2")]
