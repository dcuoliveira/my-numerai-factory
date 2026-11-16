"""
E129 report: does the 6-month RF's leaf-value block help the models inside
CU_SA_IA_BO_MA_R_01 and CU_SA_IA_BO_02?

    python3 -m src.analysis.e129_report            # tables + rebuilt slots written as runs (tree e129)
    python3 -m src.analysis.e129_report --no-write

Three things, in order:

1. The ASSEMBLER. Both slots are nested per-period rank averages (run_ensemble) of
   eight base models, with E107's meta residualisation on two legs of R_01. The tree
   is read from each stored run's metrics config (members / weights / source_run), so
   nothing about it is retyped here. Rebuilt from the STORED leaf runs, it must
   reproduce the stored slot series (lgb/e47_cusaiabo02, meta_resid/e107px_cu_sa_cu_sa)
   before any comparison is trusted.
2. MEMBERS: each e129_lf_<m> against its e129_ctl_<m> re-run, paired Newey-West lag
   20 on the 20-day payout. Both are re-run now so the pair differs ONLY by the
   feature_rfleaf_ block; the control is also checked against the stored original.
3. SLOTS: the same assembly from the controls and from the +rfleaf arms (the two
   ridge block blends gaining their sixth, leaf-only member), scored on the slots'
   common window, paired.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.analysis.ablation_report import newey_west  # noqa: E402
from src.analysis.cross_target import TARGET_20, score_on_payable_target  # noqa: E402
from src.analysis.meta_resid_rescore import Residualiser  # noqa: E402
from src.analysis.results_table import burn_days, payout_series  # noqa: E402
from src.prediction.metrics import summarize  # noqa: E402

SLOTS = {"CU_SA_IA_BO_MA_R_01": "meta_resid/e107px_cu_sa_cu_sa", "CU_SA_IA_BO_02": "lgb/e47_cusaiabo02"}
MEMBERS = {  # stored base run -> E129 arm suffix
    "mlp/a2_mlp_fenccausal": ("mlp", "ia01", "IA_01"),
    "mlp/e33_fenc_a128_64_relu_lre3_al10": ("mlp", "ia02", "IA_02"),
    "lgb/fe_lgb_reg_raw": ("lgb", "saci04", "SACI_04"),
    "rf_ktz/e52_fenccausal": ("rf_ktz", "mafenc", "MA_FENC"),
    "rf_ktz/e52_allfeat": ("rf_ktz", "maallfeat", "MA_ALLFEAT"),
    "ridge_glmnet/w6r6_glmnet_reg_allfeatpp": ("ridge_glmnet", "c04", "CURUPIRA_04"),
}
BLOCKS = {  # stored block blend -> its leaf-only sixth member
    "ridge_glmnet/w6r6_blend_quintet2": ("ridge_glmnet/e129_lf_c05blk", "CURUPIRA_05"),
    "ridge_glmnet/t60_c05_blend": ("ridge_glmnet/e129_lf_c0560blk", "CURUPIRA_05_60"),
}


def config(run):
    path = os.path.join("results", "metrics", "crypto", run + ".json")
    return json.load(open(path)).get("config", {})


def forecast(run):
    model, name = run.split("/")
    f = pd.read_csv(os.path.join("results", "forecast", "crypto", model, name + ".csv"),
                    usecols=["symbol", "date", "prediction"], parse_dates=["date"])
    return f.set_index(["symbol", "date"])["prediction"]


class Assembler(object):
    """Rebuild a stored blend tree with some leaves swapped and some blends extended."""

    def __init__(self, residualiser, swap=None, extend=None):
        self.res, self.swap, self.extend, self.cache = residualiser, swap or {}, extend or {}, {}

    def __call__(self, run):
        if run in self.cache:
            return self.cache[run]
        cfg = config(run)
        if cfg.get("members"):
            members = [m if "/" in m else "%s/%s" % (cfg["model"], m) for m in cfg["members"]]
            weights = list(cfg.get("weights") or [1.0] * len(members))
            if run in self.extend:
                # a sixth block member at the blend's own weighting scheme: equal
                members, weights = members + [self.extend[run]], [1.0] * (len(members) + 1)
            total = float(sum(weights))
            ranked = [self(m).groupby(level="date").rank(pct=True).rename(m) for m in members]
            joined = pd.concat(ranked, axis=1, join="inner")
            value = sum(joined[m] * (w / total) for m, w in zip(members, weights))
        elif cfg.get("source_run") and cfg.get("strength"):
            child = self(cfg["source_run"])
            value = self.res.residualised(child.to_frame("prediction"))
        else:
            value = forecast(self.swap.get(run, run))
        self.cache[run] = value
        return value


def score(prediction, extras, window):
    table, _ = score_on_payable_target(prediction, extras)
    table.index = pd.to_datetime(table.index)
    return table.reindex(window)


def stats(table):
    pay = payout_series(table).dropna()
    s, c = summarize(pay), summarize(table["corr"].dropna())
    return {"CORR": c["mean"], "MMC": table["mmc"].mean(), "payout": s["mean"], "p.sharpe": s["sharpe"],
            "dd(pay)": s["max_drawdown"], "calmar": s["mean"] / s["max_drawdown"], "burn": burn_days(table),
            "n": len(pay)}


def paired(a, b, lag=20):
    d = (payout_series(a) - payout_series(b)).dropna()
    m, se, t = newey_west(d.to_numpy(), lag=lag)
    dc = (a["corr"] - b["corr"]).dropna()
    mc, _, tc = newey_west(dc.to_numpy(), lag=lag)
    return {"d_payout": m, "t_payout": t, "d_corr": mc, "t_corr": tc}


def main():
    ap = argparse.ArgumentParser(description="E129 report")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    res = Residualiser()
    extras = res.extras

    missing = [m for m in list(MEMBERS) for kind in ("ctl", "lf")
               if not os.path.exists(os.path.join("results", "metrics", "crypto", MEMBERS[m][0], "e129_%s_%s.json" % (kind, MEMBERS[m][1])))]
    missing += [b[0] for b in BLOCKS.values() if not os.path.exists(os.path.join("results", "metrics", "crypto", b[0] + ".json"))]
    if missing:
        print("NOT READY -- missing arms:", missing)

    # ---- 1. the assembler must reproduce the stored slots
    stored = Assembler(res)
    window = None
    for slot, run in SLOTS.items():
        ref = pd.read_csv(os.path.join("results", "per_period", "crypto", run + ".csv"), index_col=0, parse_dates=True)
        mine = score(stored(run), extras, ref.index)
        window = ref.index if window is None else window.intersection(ref.index)
        p_mine, p_ref = payout_series(mine).reindex(ref.index), payout_series(ref)
        diff, corr = abs(p_mine.mean() - p_ref.mean()), p_mine.corr(p_ref)
        print("assembler check %-20s stored payout %.6f | rebuilt %.6f | |d mean| %.1e | per-period corr %.6f"
              % (slot, p_ref.mean(), p_mine.mean(), diff, corr))
        # Not bit-exact: the stored blends were built in August from member files
        # that have been rewritten since, and a sum of rank averages that ties in
        # one float order breaks the tie in another -- every node differs by 1-2
        # rank positions of 300. A structural mismatch (wrong member, weight or
        # transform) moves the mean by >1e-3; this bounds it well below that.
        assert diff < 2e-4 and corr > 0.999, "the assembler does not reproduce %s" % slot
    if missing:
        return

    # ---- 2. members: lf vs ctl (and ctl vs the stored original)
    rows = []
    for base, (model, tag, slot_name) in MEMBERS.items():
        member_window = pd.read_csv(os.path.join("results", "per_period", "crypto", "%s/e129_ctl_%s.csv" % (model, tag)),
                                    index_col=0, parse_dates=True).index
        t_ctl = score(forecast("%s/e129_ctl_%s" % (model, tag)), extras, member_window)
        t_lf = score(forecast("%s/e129_lf_%s" % (model, tag)), extras, member_window)
        t_old = score(forecast(base), extras, member_window)
        repro = (payout_series(t_ctl) - payout_series(t_old)).abs().max()
        rows.append(dict(member=slot_name, arm="control", **stats(t_ctl), ctl_vs_stored_maxdiff=repro))
        rows.append(dict(member=slot_name, arm="+rfleaf", **stats(t_lf), **paired(t_lf, t_ctl)))
    for blend, (extra, slot_name) in BLOCKS.items():
        asm_ctl, asm_lf = Assembler(res), Assembler(res, extend={blend: extra})
        bw = pd.read_csv(os.path.join("results", "per_period", "crypto", blend + ".csv"), index_col=0, parse_dates=True).index
        t_ctl, t_lf = score(asm_ctl(blend), extras, bw), score(asm_lf(blend), extras, bw)
        rows.append(dict(member=slot_name, arm="5 blocks (stored)", **stats(t_ctl)))
        rows.append(dict(member=slot_name, arm="+ rfleaf block", **stats(t_lf), **paired(t_lf, t_ctl, lag=20)))
    members = pd.DataFrame(rows)
    print("\n=== MEMBERS (20-day payout; each on its own window)")
    print(members.round(5).to_string(index=False))

    # ---- 3. slots: control assembly vs +rfleaf assembly, on the slots' common window
    swap_ctl = {b: "%s/e129_ctl_%s" % (m, t) for b, (m, t, _) in MEMBERS.items()}
    swap_lf = {b: "%s/e129_lf_%s" % (m, t) for b, (m, t, _) in MEMBERS.items()}
    extend = {b: e for b, (e, _) in BLOCKS.items()}
    asm = {"stored": Assembler(res), "control": Assembler(res, swap_ctl), "+rfleaf": Assembler(res, swap_lf, extend)}
    out, tables = [], {}
    for slot, run in SLOTS.items():
        for kind, a in asm.items():
            tables[(slot, kind)] = score(a(run), extras, window)
            row = dict(slot=slot, assembly=kind, **stats(tables[(slot, kind)]))
            if kind == "+rfleaf":
                row.update(paired(tables[(slot, kind)], tables[(slot, "control")]))
            out.append(row)
            if kind != "stored" and not args.no_write:
                write("e129_%s_%s" % (slot.lower(), "ctl" if kind == "control" else "rfleaf"),
                      a(run), extras, window, {"slot": slot, "assembly": kind, "source": run})
    slots = pd.DataFrame(out)
    print("\n=== SLOTS (common window, %d periods)" % len(window))
    print(slots.round(5).to_string(index=False))
    members.to_csv("results/reports/crypto/e129_members.csv", index=False)
    slots.to_csv("results/reports/crypto/e129_slots.csv", index=False)


def write(name, prediction, extras, window, config_extra):
    prediction = prediction[prediction.index.get_level_values("date").isin(window)]
    table, _ = score_on_payable_target(prediction, extras)
    results = {"n_periods": int(table["corr"].notna().sum())}
    for family in ("corr", "bmc", "mmc"):
        results[family] = summarize(table[family].dropna())
    results["config"] = dict(config_extra, model="e129", tournament="crypto", task="regression", target=TARGET_20,
                             study="E129", assembled_by="src/analysis/e129_report.py")
    for kind in ("forecast", "per_period", "metrics"):
        os.makedirs(os.path.join("results", kind, "crypto", "e129"), exist_ok=True)
    base = os.path.join("results", "%s", "crypto", "e129", name)
    rows = extras.join(prediction.rename("prediction"), how="inner").dropna(subset=["prediction"])
    rows.reset_index()[["symbol", "date", TARGET_20, "prediction", "reference_prediction", "meta_model"]] \
        .to_csv(base % "forecast" + ".csv", index=False)
    table.to_csv(base % "per_period" + ".csv")
    json.dump(results, open(base % "metrics" + ".json", "w"), indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
