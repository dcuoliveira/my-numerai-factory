"""E146-only sections for e138_report.render: the audit (raw-state truncation + re-binning identity) and the
comparisons the dims study adds (dims vs PCs, fine-tuned vs zero-shot, covariates, shipped groups)."""
import pandas as pd

FAM = {"c2cov": "zero-shot cov", "c2ar": "zero-shot ar", "c2ftcov": "fine-tuned cov", "c2ftar": "fine-tuned ar"}


def audit(s, W):
    au = s.AU
    zs, ft, idn = au["truncation_zero_shot"], au["truncation_fine_tuned"], au["identity"]
    cv = au["coverage"]
    H = ['<h2 id="audit">Point-in-time audit</h2><ul>',
         '<li><b>The states read nothing past their wall.</b> The extractor was rerun on the panel cut at %d dates with every label newer than 35 days blanked, and the cut states compared with the full run, all 768 values per state: zero-shot %s of %s values differ (lowest cosine %.5f); fine-tuned, the same reloaded block model on cut and uncut panels, %s of %s differ. These are the raw states every dimension here is binned from, the same audits E143 and E144 ran.</li>'
         % (len(dict.fromkeys(t["date"] for t in zs)), "{:,}".format(sum(t["differ"] for t in zs)), "{:,}".format(sum(t["cells"] for t in zs)),
            min(t["cosine_min"] for t in zs), "{:,}".format(sum(t["differ"] for t in ft)), "{:,}".format(sum(t["cells"] for t in ft))),
         '<li><b>The dataset is those states, binned.</b> Re-binning the raw states with Numerai&#39;s rule on %d sampled (family, date) cross-sections reproduces the dataset: %s of %s cells differ. Zero-shot dims are binned within date; fine-tuned dims within (block, date), from <code>blocks.parquet</code>.</li>'
         % (len(idn), "{:,}".format(sum(t["differ"] for t in idn)), "{:,}".format(sum(t["cells"] for t in idn))),
         '<li><b>Coverage on the scored keys:</b> zero-shot states on %.1f%%, fine-tuned on %.1f%% (none in blocks 0 and 1). %.1f%% of keys belong to a coin with no zero-shot state anywhere in the file (P_NM&#39;s flag).</li>' % (100 * cv["zero_shot"], 100 * cv["fine_tuned"], 100 * cv["no_feed"])]
    rows = []
    for f, lab in FAM.items():
        m, r, ic = au["mass"][f], au["redundancy"][f], au["date_ic"][f]
        rows.append([lab, " / ".join("%.3f" % v for v in m.values()), "%.2f" % r["mean_abs_rho_within"], str(r["pairs_over_08"]), "%.2f" % r["mean_max_rho_with_base"],
                     "%.2f" % r["max_rho_with_base"], "%.3f" % ic["mean_abs_ic"], "%.3f" % ic["max_abs_ic"], "%d of %d" % (ic["dims_t_over_2"], au["sample_dims_per_family"])])
    H += ['<li><b>What the dims look like</b>, on %d randomly sampled dims per state (seed 146): bin mass, redundancy within the state and with the 22 shipped features, and raw per-date IC with the label (|t| at lag-20 scaling). Standalone IC is not marginal value: E143&#39;s PCs had strong IC and no marginal value.</li></ul>' % au["sample_dims_per_family"],
          W.table(["state", "bin mass", "mean |ρ| within", "pairs > 0.8", "mean max |ρ| with shipped", "max |ρ| with shipped", "mean |IC|", "max |IC|", "|t| ≥ 2"], rows)]
    y = pd.DataFrame(au["coverage_by_year"])
    H += [W.table(["year", "scope", "zero-shot observed", "fine-tuned observed"], [[int(r.date), r.kind, "%.1f%%" % (100 * r.zero_shot), "%.1f%%" % (100 * r.fine_tuned)] for r in y.itertuples()]),
          '<p><small>Coverage over time on the scored keys.</small></p>']
    return H


def extra(s, W, HDR):
    return ['<h2 id="dims">What the four states add, and how</h2>',
            '<h3>All 768 dims against the 16 PCs of the same state</h3><p>Both legs unrouted (the PC arms are E143/E144&#39;s predictions, routed there on a slightly different coverage, since the PCs blank 60 burn-in dates). A positive cell means keeping every dimension beats the 16-PC summary.</p>',
            W.table(HDR, s.rows([("F_ZS", "PC_ZS"), ("F_FT", "PC_FT"), ("F_C2COV", "PC_C2COV"), ("F_C2AR", "PC_C2AR"), ("F_C2FTCOV", "PC_C2FTCOV"), ("F_C2FTAR", "PC_C2FTAR")], mode="plain"), "l2"),
            '<h3>Fine-tuned against zero-shot, and each given the other</h3>',
            W.table(HDR, s.rows([("F_FT", "F_ZS"), ("FULL", "F_ZS"), ("FULL", "F_FT")]), "l2"),
            '<h3>With the 22 features as covariates against the label alone</h3>',
            W.table(HDR, s.rows([("F_C2COV", "F_C2AR"), ("F_C2FTCOV", "F_C2FTAR")]), "l2"),
            '<h3>Scale: v2.0 with one shipped group removed</h3><p>What dropping one shipped group from v2.0 costs, unrouted: a yardstick for the size of a real feature family on these models and dates. A negative cell means the group carries signal.</p>',
            W.table(HDR[:-1], s.rows([("NO_" + g, "B0") for g in ("MOM", "REV", "VOLU", "VOLA", "PRICE", "SIZE")], False, "plain"), "l2")]
