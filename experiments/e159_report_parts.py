"""E159-only sections for e138_report.render: E152's builder validation, coverage, and the family contrasts."""
import json
import os

import pandas as pd

DIR = "data/crypto/v2.0-volumealpha-causal"
VAL = "results/reports/crypto/e152_volumealpha_causal_build/validation.json"
ROOT = "results/reports/crypto/e159_volumealpha"
ARMS = ["F_TECH", "F_LIQ", "F_FUND", "F_XASSET", "F_FCST", "F_COST"]


def assemble():
    v = json.load(open(VAL))
    m = json.load(open(os.path.join(DIR, "manifest.json")))
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    tr = v.get("technical_real_source_truncation_audit", [])
    out = dict(status=v.get("status"), oof_rows=v.get("oof_prediction_rows_checked"), oof_viol=v.get("oof_publication_boundary_violations"),
               trunc_cells=int(sum(t["compared_observed_cells"] for t in tr)), trunc_mism=int(sum(t["mismatches"] for t in tr)), trunc_n=len(tr),
               base_exact=len(v.get("base_columns_preserved_exactly", [])), lag=m["observation_lag_days"], pit=m["point_in_time_verified"],
               keys_observed=float((o["obs_tf"] > 0).mean()), keys_no_feed=float(o["no_feed"].mean()),
               arm_cov={a: float((o["obs_" + a] > 0).mean()) for a in ARMS},
               coverage_by_year=o.assign(year=pd.to_datetime(o["date"]).dt.year).groupby("year")[["obs_F_TECH", "obs_F_FCST", "obs_F_LIQ"]]
               .apply(lambda g: (g > 0).mean()).reset_index().to_dict("records"))
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1, default=str)
    return out


def audit(s, W):
    au = s.AU
    H = ['<h2 id="audit">Point-in-time audit</h2><ul>',
         '<li><b>The builder&#39;s own validation (E152, %s):</b> the technical block was recomputed from the source cut at %d dates: %s cells compared, %d mismatches. '
         '%s out-of-fold volume-forecast rows were checked against publication and train-window limits: %d violations. %d base columns are byte-identical to v2.0.</li>'
         % (au["status"], au["trunc_n"], "{:,}".format(au["trunc_cells"]), au["trunc_mism"], "{:,}".format(au["oof_rows"]), au["oof_viol"], au["base_exact"]),
         '<li><b>Not verified vintages:</b> the dataset is an inferred reconstruction (point_in_time_verified = %s). Each source day enters with a %d-day lag. '
         'Ticker-to-coin identities, venue availability and fundamentals are reconstructed today, and vendors may have revised history since.</li>' % (au["pit"], au["lag"]),
         '<li><b>Coverage on the scored keys:</b> %.1f%% of keys have some of the block; %.1f%% belong to a coin never observed anywhere in the file (P_NM&#39;s flag). By family: %s.</li>'
         % (100 * au["keys_observed"], 100 * au["keys_no_feed"], "; ".join("%s %.1f%%" % (a[2:], 100 * x) for a, x in au["arm_cov"].items())),
         '<li><b>Survivorship:</b> exchange REST coverage reflects pairs listed today, so which coins are covered is partly a today-snapshot. Routing keeps an uncovered coin on v2.0&#39;s rank.</li></ul>']
    y = pd.DataFrame(au["coverage_by_year"])
    H += [W.table(["year", "technical observed", "forecasts observed", "liquidity observed"],
                  [[int(r.year), "%.1f%%" % (100 * r.obs_F_TECH), "%.1f%%" % (100 * r.obs_F_FCST), "%.1f%%" % (100 * r.obs_F_LIQ)] for r in y.itertuples()])]
    return H


def extra(s, W, HDR):
    return ['<h2 id="inside">Inside the block</h2>',
            '<h3>Raw inputs against model-derived forecasts and costs</h3>', W.table(HDR, s.rows([("FULL", "F_RAW"), ("FULL", "F_MODEL"), ("F_MODEL", "F_RAW")]), "l2"),
            '<h3>Volume forecasts against the paper&#39;s technical inputs</h3>', W.table(HDR, s.rows([("F_FCST", "F_TECH")]), "l2")]
