"""
Consolidated feature-block results for LightGBM regression on crypto.

    python3 -m src.analysis.lgb_blocks_report
    python3 -m src.analysis.lgb_blocks_report --email

Gathers every pinned-config block arm across the three studies that produced
them and puts them in one table, GROUPED BY LEAF SIZE:

    fe_lgb_reg_*        the feature-block study        min_child_samples 313
    ohl_lgb_reg_*_m<L>  the leaf-size sweep            20 / 50 / 80 / 99
    ohc_lgb_reg_*_m<L>  the one-hot combination study  50

Why grouping by leaf size is not cosmetic
-----------------------------------------
min_child_samples decides whether the 740 ticker one-hot columns exist at all.
A ticker holds at most 130 rows in a 6-month block, so at 313 no split can leave
313 rows on both sides, LightGBM's feature_pre_filter discards the whole block
at Dataset construction, and the arm comes back BYTE-IDENTICAL to its
non-one-hot twin. Ranking arms from different leaf sizes in one list therefore
compares runs where a 740-column block was silently present or absent.

Every delta below is against the RAW ARM AT THAT LEAF SIZE for the same reason:
crossing leaf sizes confounds the columns with the regularisation.

Newey-West at lag 20 throughout -- a 20-day forward target on daily dates leaves
consecutive periods ~95% overlapping, so 510 periods are ~26 effective
independent windows and treating them as independent inflates every t by ~sqrt(20).
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

METRICS = "results/metrics/crypto/lgb"
PER_PERIOD = "results/per_period/crypto/lgb"
NEWEY_WEST_LAG = 20

# Arms whose score is not a fair measurement, and why. Printed as footnotes
# rather than dropped, because a silently missing row reads as "never tried".
CAVEATS = {
    "fencglobal": "LOOKAHEAD -- whole-sample feature mean; upper bound, not deployable",
    "allfeatppp": "contains fenccausal, the worst standalone block",
}


def payout(frame):
    """Crypto pays 0.1 * CORR + 1 * MMC."""
    return 0.1 * frame["corr"] + frame["mmc"]


def newey_west(series, lag=NEWEY_WEST_LAG):
    """(mean, se, t) for the mean of an autocorrelated series."""
    x = np.asarray(series, dtype="float64")
    n = len(x)
    mean = x.mean()
    e = x - mean
    variance = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(variance, 0.0) / n)
    return mean, se, (mean / se if se > 0 else np.nan)


def per_period(run):
    path = os.path.join(PER_PERIOD, run + ".csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    return frame.set_index("date") if "date" in frame.columns else frame


def collect():
    """Every pinned block arm, with its leaf size resolved from the config."""
    rows = []
    for path in sorted(glob.glob(os.path.join(METRICS, "*.json"))):
        run = os.path.basename(path)[:-5]
        if not re.match(r"^(fe|ohl|ohc)_lgb_reg_", run):
            continue
        d = json.load(open(path))
        config = d.get("config", {})
        fixed = config.get("fixed_params") or {}
        # The leaf size is authoritative from the config, not the run name.
        leaf = fixed.get("min_child_samples")
        if leaf is None:
            continue                      # a searched arm: different study
        arm = re.sub(r"^(fe|ohl|ohc)_lgb_reg_", "", run)
        arm = re.sub(r"_m\d+$", "", arm)
        rows.append(dict(
            run=run, arm=arm, leaf=int(leaf), feat=config["n_features"],
            corr=d["corr"]["mean"], mmc=d["mmc"]["mean"],
            sharpe=d["corr"]["sharpe"], maxdd=d["corr"]["max_drawdown"],
            payout=0.1 * d["corr"]["mean"] + d["mmc"]["mean"],
            minutes=d.get("runtime_minutes", 0.0),
        ))
    return pd.DataFrame(rows)


def render(table):
    out = []
    w = out.append
    w("LightGBM regression -- feature blocks, pinned config, rolling 6m/6m")
    w("payout = 0.1*CORR + MMC   |   delta and t are vs RAW AT THE SAME LEAF SIZE")
    w("Newey-West lag 20; 510 periods ~ 26 effective independent windows, so")
    w("|t| < 2 is the expected outcome and a null is the finding.")
    w("")

    for leaf in sorted(table["leaf"].unique()):
        block = table[table["leaf"] == leaf].copy()
        base = block[block["arm"] == "raw"]
        base_run = base["run"].iloc[0] if len(base) else None
        base_series = per_period(base_run) if base_run else None

        w("=" * 96)
        w("min_child_samples = %d%s" % (
            leaf, "   (one-hot block INERT at this size -- see module docstring)"
            if leaf >= 313 else "   (one-hot block live)"))
        w("=" * 96)
        w("%-22s %5s %10s %10s %8s %8s %10s %11s %7s" % (
            "arm", "feat", "CORR", "MMC", "sharpe", "maxdd", "payout",
            "d vs raw", "t"))
        w("-" * 96)

        for _, r in block.sort_values("payout", ascending=False).iterrows():
            delta = tstat = None
            if base_series is not None and r["arm"] != "raw":
                arm_series = per_period(r["run"])
                if arm_series is not None:
                    joined = pd.concat(
                        [payout(base_series), payout(arm_series)],
                        axis=1, join="inner", keys=["base", "arm"])
                    delta, _, tstat = newey_west(joined["arm"] - joined["base"])
            mark = " *" if r["arm"] in CAVEATS else ""
            w("%-22s %5d %10.6f %10.6f %8.4f %8.4f %10.6f %11s %7s%s" % (
                r["arm"], r["feat"], r["corr"], r["mmc"], r["sharpe"],
                r["maxdd"], r["payout"],
                "%+.6f" % delta if delta is not None else "  baseline",
                "%+.2f" % tstat if tstat is not None and np.isfinite(tstat) else "-",
                mark))
        w("")

    hits = table["arm"].isin(CAVEATS)
    if hits.any():
        w("footnotes:")
        for arm in sorted(set(table.loc[hits, "arm"])):
            w("  * %-12s %s" % (arm, CAVEATS[arm]))
        w("")

    sig = []
    for leaf in sorted(table["leaf"].unique()):
        block = table[table["leaf"] == leaf]
        base = block[block["arm"] == "raw"]
        if not len(base):
            continue
        bs = per_period(base["run"].iloc[0])
        for _, r in block.iterrows():
            if r["arm"] == "raw":
                continue
            a = per_period(r["run"])
            if a is None or bs is None:
                continue
            j = pd.concat([payout(bs), payout(a)], axis=1, join="inner",
                          keys=["b", "a"])
            _, _, t = newey_west(j["a"] - j["b"])
            if np.isfinite(t) and abs(t) > 2:
                sig.append((leaf, r["arm"], t))
    w("arms clearing |t| > 2 against their own leaf-size baseline: %d" % len(sig))
    for leaf, arm, t in sig:
        w("  leaf %-4d %-22s t = %+.2f%s" % (
            leaf, arm, t, "   (%s)" % CAVEATS[arm] if arm in CAVEATS else ""))
    if not sig:
        w("  none -- on this panel that is the expected result, not a failed run")
    return "\n".join(out)


def _short(run):
    """Study prefixes stripped, so the same arm reads the same however it was run."""
    if run == "w6r6_lgb_reg":
        return "raw"
    for prefix in ("fe_lgb_reg_", "ohl_lgb_reg_", "ohc_lgb_reg_",
                   "cad_lgb_reg_", "w6r6_lgb_reg_"):
        if run.startswith(prefix):
            return run[len(prefix):]
    m = re.match(r"^w6r6_lgb_grp_(.+)_reg$", run)
    return "grp_" + m.group(1) if m else run


def render_flat():
    """
    Every lgb regression arm in one table, ranked by payout.

    The leaf column is not decoration. min_child_samples decides whether the
    740 one-hot columns exist at all (inert at 313, live below ~130), and the
    same arm name appears at up to five leaf sizes, so a flat ranking without it
    silently compares runs whose feature sets differed by 740 columns. `search`
    distinguishes the pinned arms from the ones that ran a 50-draw search --
    they are not the same experiment either.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(METRICS, "*.json"))):
        run = os.path.basename(path)[:-5]
        d = json.load(open(path))
        config = d.get("config", {})
        if config.get("task") != "regression":
            continue
        if config.get("search_type") == "bayes":
            continue
        # Early scout runs predate the MMC column; they are not comparable to
        # anything here and a missing metric must not be silently zero-filled.
        if "mmc" not in d or "corr" not in d:
            continue
        fixed = config.get("fixed_params") or {}
        leaf = fixed.get("min_child_samples")
        rows.append(dict(
            arm=_short(run),
            leaf="%d" % leaf if leaf else "search",
            search="pinned" if fixed else "50-draw",
            feat=config["n_features"],
            window=config.get("train_window"), refit=config.get("refit_every"),
            corr=d["corr"]["mean"], mmc=d["mmc"]["mean"],
            sharpe=d["corr"]["sharpe"], maxdd=d["corr"]["max_drawdown"],
            payout=0.1 * d["corr"]["mean"] + d["mmc"]["mean"]))
    t = pd.DataFrame(rows)
    # The same configuration was run by two studies in a couple of cases
    # (onehot and raw at leaf 50). Identical config -> identical numbers, so
    # collapse them rather than double-count the row in a ranking.
    before = len(t)
    t = t.drop_duplicates(["arm", "leaf", "search", "feat", "window", "refit"])
    collapsed = before - len(t)
    t = t.sort_values("payout", ascending=False)

    out = ["LightGBM regression -- every arm, ranked by payout = 0.1*CORR + MMC",
           "leaf = min_child_samples (one-hot block is INERT at 313, live below ~130)",
           "window/refit in periods; 22 periods = 1 month",
           ""]
    out.append("%-24s %6s %8s %5s %6s %6s %10s %10s %8s %8s %10s" % (
        "arm", "leaf", "search", "feat", "window", "refit",
        "CORR", "MMC", "sharpe", "maxdd", "payout"))
    out.append("-" * 116)
    for _, r in t.iterrows():
        mark = " *" if r["arm"] in CAVEATS else ""
        out.append("%-24s %6s %8s %5d %6s %6s %10.6f %10.6f %8.4f %8.4f %10.6f%s" % (
            r["arm"], r["leaf"], r["search"], r["feat"], r["window"], r["refit"],
            r["corr"], r["mmc"], r["sharpe"], r["maxdd"], r["payout"], mark))
    out.append("-" * 116)
    out.append("%d arms%s" % (
        len(t), "  (%d duplicate config(s) collapsed)" % collapsed if collapsed else ""))
    hits = t["arm"].isin(CAVEATS)
    if hits.any():
        out.append("")
        out.append("footnotes:")
        for arm in sorted(set(t.loc[hits, "arm"])):
            out.append("  * %-12s %s" % (arm, CAVEATS[arm]))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat", action="store_true",
                        help="one flat table of every lgb regression arm, "
                             "ranked by payout, instead of the per-leaf view")
    parser.add_argument("--email", action="store_true",
                        help="mail the table via src/live/notify.py")
    args = parser.parse_args()

    table = collect()
    if table.empty:
        print("no pinned block arms found")
        return

    # The same (arm, leaf) can exist in two studies -- onehot and raw at leaf 50
    # were run by both the sweep and the combination study. Identical config, so
    # they must agree to the last digit; that is a free reproduction check, and
    # keeping both would double-count the row in the table.
    dupes = table[table.duplicated(["arm", "leaf"], keep=False)]
    notes = []
    for (arm, leaf), grp in dupes.groupby(["arm", "leaf"]):
        spread = grp["payout"].max() - grp["payout"].min()
        notes.append("  %-16s leaf %-4d %d runs, payout spread %.2e -> %s"
                     % (arm, leaf, len(grp), spread,
                        "REPRODUCES" if spread < 1e-12 else "*** MISMATCH ***"))
    table = table.sort_values("run").drop_duplicates(["arm", "leaf"], keep="first")

    report = render_flat() if args.flat else render(table)
    if notes and not args.flat:
        report += "\n\nduplicate (arm, leaf) pairs run by two studies:\n" + "\n".join(notes)
    print(report)

    if args.email:
        # Never let SMTP break the caller: report the failure and carry on.
        try:
            from src.live.notify import send_text
            send_text("[crypto] LightGBM feature-block table (%d arms)" % len(table),
                      report)
            print("\nemailed the table")
        except Exception as exc:                       # noqa: BLE001
            print("\nemail FAILED (%s) -- the table above is unaffected" % exc)


if __name__ == "__main__":
    main()
