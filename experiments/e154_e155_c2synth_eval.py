"""E154 / E155 evaluation: zero-shot Chronos-2-Synth return-rank (Arm A) and bin-forecast (Arm B) arms.

    python3 experiments/e154_e155_c2synth_eval.py        # after both GPU runs have written predictions.parquet

Same rounds for every series: keys.parquet blocks 9-10 (validation, 260 rounds) and 11 (test, 130 rounds).
Scoring is the house code: numerai_corr (e131.corr_one) against target_binned_return_20 and MMC as
correlation_contribution against the historical meta model (e131.contrib_one); Newey-West lag 20 and a
moving-block bootstrap (block 20 rounds, 2000 draws) for every paired per-round difference -- block 20 because
consecutive rounds share ~95% of their 20-trading-day target window.

  1. context per arm   argmax of validation mean CORR over {64, 128, 256}; test never read for this.
  2. standalone        each arm on its own eligible universe, with its predeclared baselines on the same rows.
  3. paired            arms, the existing model and each other on the per-round COMMON token subset
                       (eligible in A and in B and scored by the existing model), every series re-ranked there.
     existing model    primary: B0, lgb_ktz on the 22 v2.0 features, walk-forward 130/22/130, 5-seed rank average
                       (the comparator every feature study uses; covers every round). secondary: the staked slot
                       CURUPIRA_05's backtest, test rounds up to 2026-07-08 where it ends.
  4. blend             predeclared: complementary iff validation mean within-round rank correlation(A, B) < 0.5
                       AND the 50/50 rank average's validation CORR beats the better arm's. Only then scored on test.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import e131_analysis as E  # noqa: E402
from src.data_mani import c2synth_zeroshot as Z  # noqa: E402

T20 = Z.TARGET
CTXS = Z.CTXS
ROOT = "results/reports/crypto/e154_e155_c2synth_comparison"
B0_DIRS = ["results/reports/crypto/e133_histfe/preds/lgb_ktz", "results/reports/crypto/e135_route_histfe/preds/lgb_ktz"]
SLOT = "CURUPIRA_05"
RULE_RHO, BOOT_BLOCK = 0.5, 20


def keys():
    k = pd.read_parquet(Z.KEYS, columns=["symbol", "date", T20, "meta_model", "block"])
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["block"].isin(Z.VAL_BLOCKS + Z.TEST_BLOCKS)].copy()
    k["split"] = np.where(k["block"].isin(Z.TEST_BLOCKS), "test", "validation")
    return k


def b0():
    parts = []
    for s in range(5):
        f = pd.concat([pd.read_parquet(os.path.join(d, "B0_s%d.parquet" % s), columns=["symbol", "date", "prediction"]) for d in B0_DIRS])
        f["date"] = pd.to_datetime(f["date"])
        f["r"] = f.groupby("date")["prediction"].rank(pct=True)
        parts.append(f.set_index(["symbol", "date"])["r"].rename(s))
    return pd.concat(parts, axis=1).mean(axis=1).rename("b0").reset_index()


def slot():
    from src.analysis import slots
    m, r, _ = slots.slot_run(SLOT)
    f = pd.read_csv("results/forecast/crypto/%s/%s.csv" % (m, r), usecols=["symbol", "date", "prediction"])
    f["date"] = pd.to_datetime(f["date"])
    return f.rename(columns={"prediction": "slot"}), "%s/%s" % (m, r)


def rnk(s):
    return (s.rank(method="average") - 0.5) / s.notna().sum()


def per_round(df, col, rows_mask=None):
    """[date, split, corr, mmc, n] scoring df[col] on the rows where it (and rows_mask) is present."""
    out = []
    for d, g in df.groupby("date", sort=True):
        g = g[g[col].notna()] if rows_mask is None else g[g[col].notna() & g[rows_mask]]
        if len(g) < 10 or g[col].nunique() < 2:
            continue
        h = g.assign(prediction=rnk(g[col])).set_index("symbol")
        out.append(dict(date=d, split=g["split"].iloc[0], corr=E.corr_one(h, "prediction"),
                        mmc=E.contrib_one(h, "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan, n=len(g)))
    return pd.DataFrame(out)


def summ(s, label, split, extra=None):
    x = s[s["split"] == split]
    if not len(x):
        return None
    c = E.describe(x["corr"])
    m = x["mmc"].dropna()
    r = dict(series=label, split=split, rounds=len(x), mean_corr=c["mean"], std_corr=c["std"], pos_frac=c["pos_rate"],
             corr_nw_t=c["nw_t"], corr_boot_lo=c["boot_lo"], corr_boot_hi=c["boot_hi"],
             mean_mmc=m.mean() if len(m) else np.nan, mmc_rounds=len(m), mean_tokens=x["n"].mean())
    r.update(extra or {})
    return r


def paired(a, b, label, split):
    z = a.merge(b, on=["date", "split"], suffixes=("_a", "_b"))
    z = z[z["split"] == split]
    out = {}
    for met in ("corr", "mmc"):
        d = (z[met + "_a"] - z[met + "_b"]).dropna()
        if len(d) < 3:
            continue
        m, se, t = E.newey_west(d.values, lag=20)
        lo, hi = E.block_bootstrap_ci(d.values, block=BOOT_BLOCK)
        out[met] = dict(delta=m, nw_t=t, boot_lo=lo, boot_hi=hi, rounds=len(d))
    return dict(pair=label, split=split, **{"%s_%s" % (met, k): v for met, r in out.items() for k, v in r.items()})


def main():
    os.makedirs(ROOT, exist_ok=True)
    K = keys()
    A = pd.read_parquet(os.path.join(Z.OUT["return_rank"], "predictions.parquet"))
    B = pd.read_parquet(os.path.join(Z.OUT["bin_forecast"], "predictions.parquet"))
    cA = ["raw_ctx%d" % c for c in CTXS] + ["mom28", "eligible"]
    cB = ["raw_ctx%d" % c for c in CTXS] + ["last_label", "mean5_label", "eligible"]
    F = (K.merge(A[["symbol", "date"] + cA].rename(columns={c: "A_" + c for c in cA}), on=["symbol", "date"], how="left", validate="1:1")
          .merge(B[["symbol", "date"] + cB].rename(columns={c: "B_" + c for c in cB}), on=["symbol", "date"], how="left", validate="1:1")
          .merge(b0(), on=["symbol", "date"], how="left", validate="1:1"))
    S, slot_run = slot()
    F = F.merge(S, on=["symbol", "date"], how="left", validate="1:1")
    for arm in "AB":
        F[arm + "_eligible"] = F[arm + "_eligible"].fillna(False).astype(bool)
    assert F["b0"].notna().all(), "B0 missing rows"
    rows, decisions = [], {}

    # 1. context selection, validation only (own eligible universe)
    sel = {}
    for arm, name in (("A", "return_rank"), ("B", "bin_forecast")):
        cv = {}
        for c in CTXS:
            s = per_round(F, "%s_raw_ctx%d" % (arm, c), arm + "_eligible")
            for sp in ("validation", "test"):
                r = summ(s, "%s ctx%d" % (name, c), sp, dict(arm=name, role="context grid"))
                if r:
                    rows.append(r)
            cv[c] = s.loc[s["split"] == "validation", "corr"].mean()
        sel[arm] = max(cv, key=cv.get)
        decisions["%s_context" % name] = dict(selected=sel[arm], validation_corr_by_ctx={str(k): v for k, v in cv.items()})
    F["A_sig"], F["B_sig"] = F["A_raw_ctx%d" % sel["A"]], F["B_raw_ctx%d" % sel["B"]]

    # 2. standalone, own universe, with baselines on the same rows
    own = {}
    for lab, col, mask, arm in [("A return_rank (selected ctx)", "A_sig", "A_eligible", "return_rank"),
                                ("A baseline mom28", "A_mom28", "A_eligible", "return_rank"),
                                ("B0 on A-eligible rows", "b0", "A_eligible", "return_rank"),
                                ("B bin_forecast (selected ctx)", "B_sig", "B_eligible", "bin_forecast"),
                                ("B baseline last label", "B_last_label", "B_eligible", "bin_forecast"),
                                ("B baseline mean of last 5", "B_mean5_label", "B_eligible", "bin_forecast"),
                                ("B0 on B-eligible rows", "b0", "B_eligible", "bin_forecast")]:
        s = per_round(F, col, mask)
        own[lab] = s
        for sp in ("validation", "test"):
            r = summ(s, lab, sp, dict(arm=arm, role="standalone, own universe"))
            if r:
                rows.append(r)
    full_b0 = per_round(F, "b0")
    for sp in ("validation", "test"):
        rows.append(summ(full_b0, "B0 existing model, full universe", sp, dict(arm="existing", role="reference")))

    # coverage
    cov = (F.assign(both=F["A_eligible"] & F["B_eligible"]).groupby(["split", "date"])
             .agg(universe=("symbol", "size"), A=("A_eligible", "sum"), B=("B_eligible", "sum"), both=("both", "sum")).reset_index())
    cov.to_csv(os.path.join(ROOT, "coverage_by_round.csv"), index=False)
    covsum = cov.groupby("split")[["universe", "A", "B", "both"]].mean()
    covsum["A_share"], covsum["B_share"], covsum["both_share"] = covsum["A"] / covsum["universe"], covsum["B"] / covsum["universe"], covsum["both"] / covsum["universe"]

    # 3. paired on the common subset
    F["common"] = F["A_eligible"] & F["B_eligible"] & F["b0"].notna()
    C = F[F["common"]].copy()
    for c in ("A_sig", "B_sig", "b0", "A_mom28", "B_last_label", "B_mean5_label"):
        C[c] = C.groupby("date")[c].transform(rnk)
    com = {lab: per_round(C, col) for lab, col in [("A", "A_sig"), ("B", "B_sig"), ("B0", "b0"), ("MOM28", "A_mom28"),
                                                    ("LAST", "B_last_label"), ("MEAN5", "B_mean5_label")]}
    for lab, s in com.items():
        for sp in ("validation", "test"):
            rows.append(summ(s, lab + " (common subset)", sp, dict(arm="common", role="common subset")))
    pairs = []
    for a, b in [("A", "B0"), ("B", "B0"), ("A", "B"), ("A", "MOM28"), ("B", "LAST"), ("B", "MEAN5")]:
        for sp in ("validation", "test"):
            pairs.append(paired(com[a], com[b], "%s - %s (common subset)" % (a, b), sp))
    for lab, base in [("A return_rank (selected ctx)", "A baseline mom28"), ("A return_rank (selected ctx)", "B0 on A-eligible rows"),
                      ("B bin_forecast (selected ctx)", "B baseline last label"), ("B bin_forecast (selected ctx)", "B baseline mean of last 5"),
                      ("B bin_forecast (selected ctx)", "B0 on B-eligible rows")]:
        for sp in ("validation", "test"):
            pairs.append(paired(own[lab], own[base], "%s - %s (own universe)" % (lab.split(" (")[0], base), sp))

    # secondary existing model: the staked slot, on the test rounds its backtest covers
    D = F[F["common"] & F["slot"].notna()].copy()
    for c in ("A_sig", "B_sig", "slot", "b0"):
        D[c] = D.groupby("date")[c].transform(rnk)
    sc = {lab: per_round(D, col) for lab, col in [("A", "A_sig"), ("B", "B_sig"), (SLOT, "slot"), ("B0", "b0")]}
    for lab, s in sc.items():
        r = summ(s, "%s (common subset with %s)" % (lab, SLOT), "test", dict(arm="slot", role="secondary existing model"))
        if r:
            rows.append(r)
    for a in ("A", "B"):
        pairs.append(paired(sc[a], sc[SLOT], "%s - %s (common subset, test to %s)" % (a, SLOT, D["date"].max().date()), "test"))

    # 4. complementarity on validation, predeclared blend rule
    V = C[C["split"] == "validation"]
    rho = V.groupby("date").apply(lambda g: g["A_sig"].corr(g["B_sig"], method="spearman"), include_groups=False)
    C["blend"] = (C["A_sig"] + C["B_sig"]) / 2                 # both already re-ranked within the common subset
    sb = per_round(C, "blend")
    vb = sb.loc[sb["split"] == "validation", "corr"].mean()
    va, vB = com["A"].query("split=='validation'")["corr"].mean(), com["B"].query("split=='validation'")["corr"].mean()
    comp = bool(rho.mean() < RULE_RHO and vb > max(va, vB))
    decisions["blend"] = dict(rule="mean within-round spearman(A,B) < %.2f AND validation CORR(blend) > max(A, B), common subset" % RULE_RHO,
                              mean_rank_corr=float(rho.mean()), validation_corr_blend=vb, validation_corr_A=va, validation_corr_B=vB,
                              complementary=comp, evaluated_on_test=comp)
    rows.append(summ(sb, "50/50 blend (common subset)", "validation", dict(arm="blend", role="blend")))
    if comp:
        rows.append(summ(sb, "50/50 blend (common subset)", "test", dict(arm="blend", role="blend")))
        for b in ("B0", "A", "B"):
            pairs.append(paired(sb, com[b], "BLEND - %s (common subset)" % b, "test"))

    L = pd.DataFrame([r for r in rows if r])
    P = pd.DataFrame([p for p in pairs if p])
    L.to_csv(os.path.join(ROOT, "levels.csv"), index=False)
    P.to_csv(os.path.join(ROOT, "paired.csv"), index=False)
    covsum.to_csv(os.path.join(ROOT, "coverage_summary.csv"))
    decisions.update(existing_model_primary="B0: lgb_ktz on v2.0 (22 features), walk-forward 130/22/130, 5-seed rank average (%s)" % ", ".join(B0_DIRS),
                     existing_model_secondary="%s backtest %s, test rounds to %s" % (SLOT, slot_run, D["date"].max().date()),
                     common_subset_tokens_per_round=float(C.groupby("date").size().mean()))
    json.dump(decisions, open(os.path.join(ROOT, "decisions.json"), "w"), indent=1, default=float)
    for name, c in (("return_rank", "A"), ("bin_forecast", "B")):
        src = A if c == "A" else B
        fin = src.copy()
        fin["selected_ctx"] = sel[c]
        fin["raw_score"], fin["signal"] = fin["raw_ctx%d" % sel[c]], fin["signal_ctx%d" % sel[c]]
        keep = ["date", "symbol", "ucid", "split", "raw_score", "signal", "coverage_flag", "skip_reason", "selected_ctx"]
        keep += (["last_label_date", "gap_steps", "last_label", "mean5_label"] if c == "B" else ["mom28"])
        fin[keep].rename(columns={"date": "round"}).to_parquet(os.path.join(Z.OUT[name], "prediction_table.parquet"), index=False)
        L[L["arm"].isin([name])].to_csv(os.path.join(Z.OUT[name], "levels.csv"), index=False)
    pd.set_option("display.width", 220)
    print(json.dumps(decisions, indent=1, default=float))
    print(L[["series", "split", "rounds", "mean_corr", "std_corr", "pos_frac", "corr_nw_t", "mean_mmc", "mean_tokens"]].round(4).to_string())
    print(P.round(4).to_string())
    print(covsum.round(3))


if __name__ == "__main__":
    main()
