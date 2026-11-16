"""E154-E163 evaluation: every zero-shot TSFM in both arms of the E154/E155 design, plus a cross-model board.

    python3 experiments/tsfm_zeroshot_eval.py [--models chronos2synth tempopfn ...]

Per model (identical to experiments/e154_e155_c2synth_eval.py, whose helpers it imports):
  context    argmax of validation mean CORR over the model's supported grid; test never read for this
  standalone own covered universe, with the predeclared baselines (mom28; last label, mean of last 5) on the same rows
  paired     A, B, B0 and baselines on the per-round common subset (covered in A and B), re-ranked there;
             NW lag 20 + block-20 bootstrap; secondary existing model CURUPIRA_05 (test rounds to 2026-07-08)
  blend      predeclared: validation rank corr(A, B) < 0.5 AND 50/50 blend beats the better arm on validation
Cross-model, per arm: the tokens covered by EVERY model in that arm, each model at its selected context, paired
against Chronos-2-Synth (the reference model) and against B0.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import e154_e155_c2synth_eval as V  # noqa: E402
from src.data_mani import c2synth_zeroshot as Z  # noqa: E402

ROOT = "results/reports/crypto/tsfm_zeroshot_comparison"
MODELS = json.load(open(os.path.join(REPO, "experiments", "tsfm_zeroshot_models.json")))
REF = "chronos2synth"


def load(model, arm):
    P = pd.read_parquet(os.path.join(MODELS[model]["out"][arm], "predictions.parquet"))
    if "coverage_flag" not in P:
        P["coverage_flag"] = P["eligible"].astype(int)
    P["covered"] = P["coverage_flag"].astype(bool)
    return P


def model_eval(model, K, B0, S):
    A, B = load(model, "return_rank"), load(model, "bin_forecast")
    ctxA = sorted(int(c[7:]) for c in A if c.startswith("raw_ctx"))
    ctxB = sorted(int(c[7:]) for c in B if c.startswith("raw_ctx"))
    cA = ["raw_ctx%d" % c for c in ctxA] + ["mom28", "covered"]
    cB = ["raw_ctx%d" % c for c in ctxB] + ["last_label", "mean5_label", "covered"]
    F = (K.merge(A[["symbol", "date"] + cA].rename(columns={c: "A_" + c for c in cA}), on=["symbol", "date"], how="left", validate="1:1")
          .merge(B[["symbol", "date"] + cB].rename(columns={c: "B_" + c for c in cB}), on=["symbol", "date"], how="left", validate="1:1")
          .merge(B0, on=["symbol", "date"], how="left", validate="1:1").merge(S, on=["symbol", "date"], how="left", validate="1:1"))
    for a in "AB":
        F[a + "_covered"] = F[a + "_covered"].fillna(False).astype(bool)
    rows, pairs, dec = [], [], {}
    sel = {}
    for arm, name, ctxs in (("A", "return_rank", ctxA), ("B", "bin_forecast", ctxB)):
        cv = {}
        for c in ctxs:
            s = V.per_round(F, "%s_raw_ctx%d" % (arm, c), arm + "_covered")
            for sp in ("validation", "test"):
                r = V.summ(s, "%s ctx%d" % (name, c), sp, dict(model=model, arm=name, role="context grid", ctx=c))
                if r:
                    rows.append(r)
            cv[c] = s.loc[s["split"] == "validation", "corr"].mean()
        sel[arm] = max(cv, key=lambda k: -np.inf if pd.isna(cv[k]) else cv[k])
        dec[name + "_context"] = dict(selected=sel[arm], grid=ctxs, validation_corr_by_ctx={str(k): v for k, v in cv.items()})
    F["A_sig"], F["B_sig"] = F["A_raw_ctx%d" % sel["A"]], F["B_raw_ctx%d" % sel["B"]]
    own = {}
    for lab, col, mask, arm in [("A", "A_sig", "A_covered", "return_rank"), ("A mom28", "A_mom28", "A_covered", "return_rank"),
                                ("B0 on A rows", "b0", "A_covered", "return_rank"), ("B", "B_sig", "B_covered", "bin_forecast"),
                                ("B last label", "B_last_label", "B_covered", "bin_forecast"),
                                ("B mean of last 5", "B_mean5_label", "B_covered", "bin_forecast"), ("B0 on B rows", "b0", "B_covered", "bin_forecast")]:
        own[lab] = V.per_round(F, col, mask)
        for sp in ("validation", "test"):
            r = V.summ(own[lab], lab, sp, dict(model=model, arm=arm, role="own universe"))
            if r:
                rows.append(r)
    for a, b in [("A", "A mom28"), ("A", "B0 on A rows"), ("B", "B last label"), ("B", "B mean of last 5"), ("B", "B0 on B rows")]:
        for sp in ("validation", "test"):
            pairs.append(dict(model=model, **V.paired(own[a], own[b], "%s - %s (own universe)" % (a, b), sp)))
    F["common"] = F["A_covered"] & F["B_covered"]
    C = F[F["common"]].copy()
    for c in ("A_sig", "B_sig", "b0", "A_mom28", "B_last_label", "B_mean5_label"):
        C[c] = C.groupby("date")[c].transform(V.rnk)
    com = {lab: V.per_round(C, col) for lab, col in [("A", "A_sig"), ("B", "B_sig"), ("B0", "b0")]}
    for lab, s in com.items():
        for sp in ("validation", "test"):
            rows.append(V.summ(s, lab + " (common)", sp, dict(model=model, arm="common", role="common subset")))
    for a, b in [("A", "B0"), ("B", "B0"), ("A", "B")]:
        for sp in ("validation", "test"):
            pairs.append(dict(model=model, **V.paired(com[a], com[b], "%s - %s (common)" % (a, b), sp)))
    D = F[F["common"] & F["slot"].notna()].copy()
    for c in ("A_sig", "B_sig", "slot"):
        D[c] = D.groupby("date")[c].transform(V.rnk)
    sc = {lab: V.per_round(D, col) for lab, col in [("A", "A_sig"), ("B", "B_sig"), ("slot", "slot")]}
    for a in ("A", "B"):
        pairs.append(dict(model=model, **V.paired(sc[a], sc["slot"], "%s - %s (common, test to %s)" % (a, V.SLOT, D["date"].max().date()), "test")))
    Vd = C[C["split"] == "validation"]
    rho = Vd.groupby("date").apply(lambda g: g["A_sig"].corr(g["B_sig"], method="spearman"), include_groups=False).mean()
    C["blend"] = (C["A_sig"] + C["B_sig"]) / 2
    sb = V.per_round(C, "blend")
    vb = sb.loc[sb.split == "validation", "corr"].mean()
    va, vB = com["A"].query("split=='validation'")["corr"].mean(), com["B"].query("split=='validation'")["corr"].mean()
    comp = bool(rho < V.RULE_RHO and vb > max(va, vB))
    dec["blend"] = dict(mean_rank_corr=rho, validation_corr_blend=vb, validation_corr_A=va, validation_corr_B=vB, complementary=comp)
    rows.append(V.summ(sb, "50/50 blend (common)", "validation", dict(model=model, arm="blend", role="blend")))
    if comp:
        rows.append(V.summ(sb, "50/50 blend (common)", "test", dict(model=model, arm="blend", role="blend")))
        for b in ("B0", "A", "B"):
            pairs.append(dict(model=model, **V.paired(sb, com[b], "BLEND - %s (common)" % b, "test")))
    cov = {}
    for arm, P in (("return_rank", A), ("bin_forecast", B)):
        x = P.merge(K[["symbol", "date"]], on=["symbol", "date"])
        cov[arm] = {sp: float(g["covered"].mean()) for sp, g in x.groupby("split")}
        cov[arm]["model_nan_rows"] = int((P["skip_reason"] == "model_nan").sum())
    dec["coverage"] = cov
    dec["common_tokens_per_round"] = float(C.groupby("date").size().mean())
    sigs = {"return_rank": F[["symbol", "date", "A_sig", "A_covered"]], "bin_forecast": F[["symbol", "date", "B_sig", "B_covered"]]}
    return [r for r in rows if r], [p for p in pairs if "corr_delta" in p], dec, sigs


def cross_board(sigs, K, B0):
    rows, pairs = [], []
    for arm, key in (("return_rank", "A"), ("bin_forecast", "B")):
        F = K.merge(B0, on=["symbol", "date"])
        ms = [m for m in sigs]
        for m in ms:
            s = sigs[m][arm].rename(columns={key + "_sig": m, key + "_covered": m + "_cov"})
            F = F.merge(s, on=["symbol", "date"], how="left")
        mask = np.logical_and.reduce([F[m + "_cov"].fillna(False).astype(bool) for m in ms])
        C = F[mask].copy()
        for c in ms + ["b0"]:
            C[c] = C.groupby("date")[c].transform(V.rnk)
        sc = {m: V.per_round(C, m) for m in ms + ["b0"]}
        for m in ms + ["b0"]:
            for sp in ("validation", "test"):
                r = V.summ(sc[m], m, sp, dict(arm=arm, tokens_per_round=float(C.groupby("date").size().mean())))
                if r:
                    rows.append(r)
        for m in ms:
            for b in ([REF] if m != REF and REF in ms else []) + ["b0"]:
                for sp in ("validation", "test"):
                    pairs.append(dict(arm=arm, **V.paired(sc[m], sc[b], "%s - %s" % (m, b), sp)))
    return pd.DataFrame(rows), pd.DataFrame([p for p in pairs if "corr_delta" in p])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    K = V.keys()
    B0 = V.b0()
    S, _ = V.slot()
    L, P, D, SIG = [], [], {}, {}
    for m in a.models:
        r, p, d, s = model_eval(m, K, B0, S)
        L += r
        P += p
        D[m] = d
        SIG[m] = s
        print("done", m, json.dumps({k: v for k, v in d.items() if k.endswith("context")}, default=float), flush=True)
    L, P = pd.DataFrame(L), pd.DataFrame(P)
    XL, XP = cross_board(SIG, K, B0)
    L.to_csv(os.path.join(ROOT, "levels.csv"), index=False)
    P.to_csv(os.path.join(ROOT, "paired.csv"), index=False)
    XL.to_csv(os.path.join(ROOT, "cross_levels.csv"), index=False)
    XP.to_csv(os.path.join(ROOT, "cross_paired.csv"), index=False)
    json.dump(D, open(os.path.join(ROOT, "decisions.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    cols = ["series", "split", "rounds", "mean_corr", "std_corr", "pos_frac", "corr_nw_t", "mean_mmc"]
    print(XL[["arm"] + cols + ["tokens_per_round"]].round(4).to_string(index=False))
    print(XP.round(4).to_string(index=False))
    print(L[L.role == "own universe"][["model", "arm"] + cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
