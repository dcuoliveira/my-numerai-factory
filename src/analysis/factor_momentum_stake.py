# -*- coding: utf-8 -*-
"""
E55 -- Factor-momentum stake timing across the whole submitted book.

    python3 -m src.analysis.factor_momentum_stake                 # the sweep
    python3 -m src.analysis.factor_momentum_stake --plot out.png

The idea, from Gupta & Kelly "Factor Momentum Everywhere" (2019): a factor that
has done well recently keeps doing well for a while. Here each MODEL is a factor
and its payout series is its return, so the bet is -- tilt stake toward the
models that have performed well on recently RESOLVED rounds, away from the ones
that have not, and rebalance walk-forward. Does timing the book by its own
momentum beat holding it flat?

Three constraints make this a PRACTICAL test, not a paper one:

  * THE 20-DAY DECISION LAG. The target is a 20-day forward return, so a round's
    payout is not known until ~20 rounds later. Every momentum signal is
    `.shift(LAG)` before it can size a stake -- the decision at round t sees only
    payouts that had resolved by t.

  * THE WITHDRAW-DELAY LADDER. You cannot rebalance a crypto stake instantly. A
    stake DECREASE does not settle immediately -- the capital already committed
    to the ~20 open rounds bleeds out as they resolve, one rung per day; an
    INCREASE only applies to the next round. So the REALISED allocation is the
    trailing-LADDER (=20) mean of the TARGET allocation, a ~20-round glide, not
    the target itself. simulate() applies that glide; skipping it would report a
    responsiveness the mechanism does not have (see dynamic_stake.py, which did
    NOT model it and so overstated how fast a throttle can de-risk).

  * THE 0.1-NMR/ROUND FLOOR. Atomic Blockchain Staking (the_arkivist, 2026-09-08)
    rejects per-round stakes < 0.1 NMR -- a model gets a real stake or none, no
    dust. Long-only weights plus the optional `floor` (zero any weight below a
    cut, renormalise) mimic that lumpiness.

EQUAL CAPITAL. Every scheme's weights sum to 1, so all deploy the same total
at-risk -- the comparison is WHICH models hold the stake, not extra leverage.
Baselines: `equal` (static 1/N) and `best` (full stack on the causal best model
as of each date, ladder-timed -- the concentrated-momentum extreme). Panel is
the submitted book's per-period payout (0.1*corr + mmc), pseudo-OOS: ratios, not
NMR, same caveat as src/live/stake.py and E42.
"""

import argparse

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

#: The 20-day target's resolution lag AND the stake ladder depth, in periods.
LAG = 20
LADDER = 20

#: TRAIN is the first this-fraction of dates; the rest is held out for reporting.
TRAIN_FRAC = 2.0 / 3.0

#: The submitted book (the 24 t20 models of the comparison table). SACI_03 is on
#: the 60-day label and is left out -- a different payout scale would poison the
#: cross-sectional signal.
BOOK = [
    "CURUPIRA_01", "CURUPIRA_02", "CURUPIRA_03", "CURUPIRA_04", "CURUPIRA_05",
    "CURUPIRA_06", "CURUPIRA_07", "CURUPIRA_08", "SACI_01", "SACI_02",
    "CURUPIRA_SACI_01", "CURUPIRA_SACI_02", "CURUPIRA_SACI_03", "CURUPIRA_SACI_04",
    "SACI_04", "SACI_05", "CU_SA_IA_BO_01", "IU_IA_01", "SACI_06",
    "CURUPIRA_BOITATA_01", "CURUPIRA_CAIPORA_01", "CU_SA_IA_BO_MA_01",
    "CU_SA_IA_BO_02", "CU_CA_IA_01", "TU_CU_SA_IA_BO_CA_01",
]


def load_panel(names=None):
    """The book's per-period payout on the models' common dates (via stake)."""
    from src.live import stake
    return stake.payout_series(names or BOOK)


# --------------------------------------------------------------------------- #
# momentum signal -- trailing performance, causal (only resolved payouts)
# --------------------------------------------------------------------------- #

def signal(payout, lookback=None, halflife=None, lag=LAG):
    """
    Each model's trailing mean payout over RESOLVED rounds, one row per date.

    Two estimators: a flat rolling window (`lookback`) or an EWMA decay
    (`halflife`). `.shift(lag)` first, so round t's signal averages only payouts
    that had resolved by t -- no unresolved round can move a stake.
    """
    resolved = payout.shift(lag)
    if halflife is not None:
        return resolved.ewm(halflife=halflife,
                            min_periods=max(5, int(round(halflife)))).mean()
    return resolved.rolling(lookback, min_periods=max(5, lookback // 3)).mean()


# --------------------------------------------------------------------------- #
# allocation -- cross-sectional weights from one date's signals, summing to 1
# --------------------------------------------------------------------------- #

def allocate(sig, rule="ztilt", kappa=1.0, topk=1):
    """
    Target weights (dates x models, each row sums to 1) from the signal frame.

    * equal  -- static 1/N, the flat baseline (ignores the signal).
    * ztilt  -- tilt equal weight by the cross-sectional z-score of the signal:
                w ~ clip(1 + kappa*z, 0). kappa=0 IS equal weight; a model one
                sigma above the book's mean gets 2x base weight at kappa=1, one
                sigma below gets zero. Long-only and diversified.
    * topk   -- equal weight on the top-`topk` models by signal, zero elsewhere.
                topk=1 is 'full stack on the current best' (the `best` baseline);
                larger topk is concentrated momentum. Respects the 0.1-NMR floor
                by construction (a handful of real stakes, no dust).
    * rankprop -- weight proportional to cross-sectional rank (1..N).
    """
    n = sig.shape[1]
    if rule == "equal":
        w = pd.DataFrame(1.0 / n, index=sig.index, columns=sig.columns)
        return w
    valid = sig.notna().any(axis=1)
    if rule == "ztilt":
        mu = sig.mean(axis=1); sd = sig.std(axis=1).replace(0.0, np.nan)
        z = sig.sub(mu, axis=0).div(sd, axis=0)
        w = (1.0 + kappa * z).clip(lower=0.0)
    elif rule == "rankprop":
        w = sig.rank(axis=1)
    elif rule == "topk":
        rank = sig.rank(axis=1, ascending=False)
        w = (rank <= topk).astype(float)
    else:
        raise SystemExit("unknown rule %r" % rule)
    total = w.sum(axis=1).replace(0.0, np.nan)
    w = w.div(total, axis=0)
    # Before any signal exists (warm-up) fall back to equal weight, so every
    # scheme covers the identical dates and only the momentum window differs.
    w.loc[~valid] = 1.0 / n
    return w.fillna(1.0 / n)


def ladder(target, depth=LADDER):
    """
    The REALISED allocation: a trailing-`depth` mean of the target weights.

    A stake change glides in over the open rounds -- you add to the next round
    and the rest of the ladder still holds the old size until each rung resolves
    -- so what is actually at risk at t is the average of the target set over the
    last `depth` rounds, not the target at t. Re-normalised so the row still sums
    to 1 (a mean of sum-1 vectors already does, up to warm-up).
    """
    rolled = target.rolling(depth, min_periods=1).mean()
    return rolled.div(rolled.sum(axis=1), axis=0)


def floor_weights(weights, floor):
    """Zero any weight below `floor` and renormalise -- the 0.1-NMR/round rule:
    a model gets a real stake or none. floor=0 leaves the weights untouched."""
    if not floor:
        return weights
    w = weights.where(weights >= floor, 0.0)
    total = w.sum(axis=1).replace(0.0, np.nan)
    return w.div(total, axis=0).fillna(1.0 / weights.shape[1])


# --------------------------------------------------------------------------- #
# simulate
# --------------------------------------------------------------------------- #

def simulate(payout, rule="ztilt", lookback=None, halflife=None, kappa=1.0,
             topk=1, floor=0.0, lag=LAG, depth=LADDER):
    """
    Walk-forward realised portfolio payout for one scheme.

    :return: (realised payout Series, realised weights DataFrame, target weights)
    """
    if rule == "equal":
        target = allocate(payout, "equal")
    else:
        sig = signal(payout, lookback, halflife, lag)
        target = allocate(sig, rule, kappa, topk)
    realised = floor_weights(ladder(target, depth), floor)
    path = (realised * payout).sum(axis=1)
    return path, realised, target


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def summarize_path(path):
    """mean / sharpe / max_dd / calmar / burn for a realised payout path."""
    x = path.dropna()
    mean, sd = x.mean(), x.std(ddof=0)
    cum = x.cumsum(); dd = float((np.maximum.accumulate(cum) - cum).max())
    return {"payout": mean, "sharpe": (mean / sd if sd > 0 else float("nan")),
            "max_dd": dd, "calmar": (mean / dd if dd > 0 else float("inf")),
            "burn": int((x < 0).sum()), "n": len(x)}


def turnover(weights):
    """One-way turnover: mean over dates of 0.5*sum_m |w_m(t) - w_m(t-1)|."""
    return float(0.5 * weights.diff().abs().sum(axis=1).mean())


def paired_t(path, base, lag=LAG):
    """Newey-West t of the per-round (scheme - baseline) payout difference."""
    diff = (path - base).dropna()
    if len(diff) < 3:
        return float("nan"), float("nan")
    mean, _, t = newey_west(diff.to_numpy(), lag=lag)
    return float(mean), float(t)


def _split(index, frac=TRAIN_FRAC):
    cut = int(len(index) * frac)
    return index[:cut], index[cut:]


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #

def rolling_grid():
    return [20, 40, 60, 90, 120, 180, 250]        # short / mid / long


def ewma_grid():
    return [10, 20, 40, 80, 120]                   # halflives


def run(names=None, rule="ztilt", kappa=1.0, floor=0.0, plot=None):
    payout = load_panel(names)
    train_idx, eval_idx = _split(payout.index)

    # baselines
    eq_path, eq_w, _ = simulate(payout, "equal")
    best_path, best_w, _ = simulate(payout, "topk", lookback=60, topk=1)

    print("factor-momentum stake timing (E55)  rule=%s kappa=%.1f floor=%.3f"
          % (rule, kappa, floor))
    print("book: %d models  periods: %d (%d train, %d held-out %s..%s)"
          % (payout.shape[1], len(payout), len(train_idx), len(eval_idx),
             eval_idx[0].date(), eval_idx[-1].date()))
    print("lag=%d  ladder=%d  (decision on resolved payouts, glide over %d rounds)\n"
          % (LAG, LADDER, LADDER))

    def line(tag, path, w, idx):
        s = summarize_path(path.loc[idx])
        d, t = paired_t(path.loc[idx], eq_path.loc[idx])
        return ("%-16s %8.5f %7.3f %7.3f %7.3f %5d %7.3f %+8.5f(%+.2f)"
                % (tag, s["payout"], s["sharpe"], s["max_dd"], s["calmar"],
                   s["burn"], turnover(w.loc[idx]), d, t))

    for window_name, idx in (("HELD-OUT", eval_idx), ("full", payout.index)):
        print("== %s ==" % window_name)
        print("%-16s %8s %7s %7s %7s %5s %7s %14s"
              % ("scheme", "payout", "sharpe", "dd", "calmar", "burn", "turn",
                 "dPay vs eq(t)"))
        print("-" * 92)
        print(line("equal (base)", eq_path, eq_w, idx))
        print(line("best-single", best_path, best_w, idx))
        rows = []
        for L in rolling_grid():
            p, w, _ = simulate(payout, rule, lookback=L, kappa=kappa, floor=floor)
            rows.append(("roll L=%d" % L, p, w))
        for H in ewma_grid():
            p, w, _ = simulate(payout, rule, halflife=H, kappa=kappa, floor=floor)
            rows.append(("ewma hl=%d" % H, p, w))
        for tag, p, w in rows:
            print(line(tag, p, w, idx))
        print()

    # OOS pick: best rolling window by TRAIN calmar, reported on held-out
    best = None
    for L in rolling_grid():
        p, _, _ = simulate(payout, rule, lookback=L, kappa=kappa, floor=floor)
        c = summarize_path(p.loc[train_idx])["calmar"]
        if best is None or c > best[0]:
            best = (c, L)
    print("OOS-selected rolling window (train calmar): L=%d\n" % best[1])

    if plot:
        _plot(payout, eq_path, best_path, rule, kappa, floor, eval_idx, plot)
    return payout


def _plot(payout, eq_path, best_path, rule, kappa, floor, eval_idx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the three headline schemes: equal, best-single, and mid + long momentum
    mid, _, _ = simulate(payout, rule, lookback=60, kappa=kappa, floor=floor)
    lon, _, _ = simulate(payout, rule, lookback=180, kappa=kappa, floor=floor)
    ew, _, _ = simulate(payout, rule, halflife=40, kappa=kappa, floor=floor)
    fig, ax = plt.subplots(figsize=(11, 6))
    for series, color, label in (
        (eq_path, "#5f6368", "equal weight (baseline)"),
        (best_path, "#8e44ad", "best-single (topk=1)"),
        (mid, "#c0392b", "momentum roll L=60"),
        (lon, "#e08e0b", "momentum roll L=180"),
        (ew, "#1a7a4f", "momentum ewma hl=40"),
    ):
        c = series.cumsum()
        ax.plot(c.index, c.values, color=color, lw=1.8, label=label)
    ax.axvline(eval_idx[0], color="#1a4f7a", ls="--", lw=1, label="train | held-out")
    ax.set_title("Factor-momentum stake timing — cumulative realised payout\n"
                 "rule=%s kappa=%.1f (ladder %d, lag %d)" % (rule, kappa, LADDER, LAG),
                 fontsize=11)
    ax.set_ylabel("cumulative payout (score units, raw)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight")
    print("wrote %s" % path)


def grid(floor=0.0):
    """Every arm: allocation rule x rule-param x signal window."""
    signals = ([("roll%d" % L, {"lookback": L}) for L in rolling_grid()]
               + [("ewma%d" % H, {"halflife": H}) for H in ewma_grid()])
    arms = [("equal", {"rule": "equal"})]
    for sname, sp in signals:
        for k in (0.5, 1.0, 2.0):
            arms.append(("ztilt k%.1f %s" % (k, sname),
                         dict(rule="ztilt", kappa=k, floor=floor, **sp)))
        for K in (1, 3, 6):
            arms.append(("top%d %s" % (K, sname),
                         dict(rule="topk", topk=K, floor=floor, **sp)))
        arms.append(("rankprop %s" % sname,
                     dict(rule="rankprop", floor=floor, **sp)))
    return arms


def grid_run(names=None, floor=0.0, objective="calmar"):
    """Run the whole arm grid; OOS-select on TRAIN, report on HELD-OUT."""
    payout = load_panel(names)
    train_idx, eval_idx = _split(payout.index)
    eq_path = simulate(payout, "equal")[0]

    rows = []
    for label, kw in grid(floor):
        path, w, _ = simulate(payout, **kw)
        tr = summarize_path(path.loc[train_idx])
        ev = summarize_path(path.loc[eval_idx])
        fu = summarize_path(path)
        d_ev, t_ev = paired_t(path.loc[eval_idx], eq_path.loc[eval_idx])
        d_fu, t_fu = paired_t(path, eq_path)
        rows.append(dict(label=label, kw=kw, tr=tr, ev=ev, fu=fu,
                         turn=turnover(w.loc[eval_idx]),
                         d_ev=d_ev, t_ev=t_ev, d_fu=d_fu, t_fu=t_fu))

    print("E55 factor-momentum stake timing -- FULL ARM GRID  (floor=%.3f)" % floor)
    print("book %d models  %d periods  (%d train, %d held-out %s..%s)  lag=%d ladder=%d"
          % (payout.shape[1], len(payout), len(train_idx), len(eval_idx),
             eval_idx[0].date(), eval_idx[-1].date(), LAG, LADDER))
    print("%d arms + equal baseline  (equal held-out payout %.5f)\n"
          % (len(rows) - 1, rows[0]["ev"]["payout"]))

    def show(r):
        return ("%-18s | tr %7.4f  ev %7.4f sh %5.2f dd %5.2f cal %5.3f brn %3d "
                "turn %5.3f | dPay(ev) %+8.5f(%+.2f)  dPay(full) %+.5f(%+.2f)"
                % (r["label"], r["tr"]["payout"], r["ev"]["payout"], r["ev"]["sharpe"],
                   r["ev"]["max_dd"], r["ev"]["calmar"], r["ev"]["burn"], r["turn"],
                   r["d_ev"], r["t_ev"], r["d_fu"], r["t_fu"]))

    arms = [r for r in rows if r["label"] != "equal"]

    # 1) OOS discipline: pick on TRAIN objective, report HELD-OUT
    pick = max(arms, key=lambda r: (r["tr"][objective]
                                    if np.isfinite(r["tr"][objective]) else -1e9))
    print("== OOS-SELECTED (max train %s, reported held-out) ==" % objective)
    print(show(pick)); print()

    # 2) top by held-out payout (in-sample-on-test -- context, not a claim)
    print("== top 12 by HELD-OUT payout (peeking -- robustness context) ==")
    for r in sorted(arms, key=lambda r: -r["ev"]["payout"])[:12]:
        print(show(r))
    print()

    # 3) champion of each rule family (best held-out payout within family)
    print("== best arm per rule family (held-out) ==")
    fam = {}
    for r in arms:
        f = r["kw"]["rule"] + (":k%.1f" % r["kw"]["kappa"] if r["kw"]["rule"] == "ztilt"
                               else (":K%d" % r["kw"]["topk"] if r["kw"]["rule"] == "topk" else ""))
        if f not in fam or r["ev"]["payout"] > fam[f]["ev"]["payout"]:
            fam[f] = r
    for f in sorted(fam, key=lambda f: -fam[f]["ev"]["payout"]):
        print("%-12s %s" % (f, show(fam[f])))
    print()

    # 4) robustness: how many arms beat equal on held-out, and at |t|>=2
    beat = sum(1 for r in arms if r["d_ev"] > 0)
    sig = sum(1 for r in arms if r["t_ev"] >= 2)
    sig_fu = sum(1 for r in arms if r["t_fu"] >= 2)
    print("robustness: %d/%d arms beat equal held-out; %d clear t>=2 held-out, "
          "%d clear t>=2 full-window" % (beat, len(arms), sig, sig_fu))
    return payout, rows, pick


# --------------------------------------------------------------------------- #
# live monitor: rerun top1 roll250 each round, alert when the rank changes
# --------------------------------------------------------------------------- #

def live_resolved_payout(names, rounds=320):
    """
    RESOLVED per-round payout (0.1*corr + mmc) per model from the live crypto API,
    indexed by date. Empty if nothing has resolved yet -- these models have very
    little resolved live history (the 20-day resolution plus a recent history
    reset), so this is often empty and the panel leans on the backtest.
    """
    from src.live.env import get_api, load_env
    from src.analysis.crypto_stake_report import PERF_QUERY, TOURNAMENT_ID
    load_env(); api = get_api(); models = api.get_models() or {}
    cols = {}
    for n in names:
        mid = models.get(n.lower())
        if not mid:
            continue
        perf = api.raw_query(PERF_QUERY, {"m": mid, "t": TOURNAMENT_ID, "n": rounds},
                             authorization=True)["data"]["v2RoundModelPerformances"] or []
        rec = {}
        for it in perf:
            if not it.get("roundResolved"):
                continue
            vals, date = {}, None
            for s in it.get("submissionScores") or []:
                if s.get("displayName") in ("corr", "mmc") and s.get("value") is not None:
                    vals[s["displayName"]] = s["value"]; date = date or s.get("date")
            if "corr" in vals and "mmc" in vals and date:
                ts = pd.Timestamp(date)
                if ts.tzinfo is not None:          # API dates are tz-aware; the
                    ts = ts.tz_localize(None)      # backtest panel is tz-naive
                rec[ts.normalize()] = CORR_W * vals["corr"] + vals["mmc"]
        if rec:
            cols[n] = pd.Series(rec)
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


#: payout weight on CORR (MMC weight is 1.0), the crypto multipliers.
CORR_W = 0.1


def combined_panel(names=None):
    """Backtest per-period payout with any RESOLVED live rounds spliced onto the
    tail -- so the signal updates as live rounds resolve. Best-effort: a live
    fetch failure or an empty resolve set leaves the backtest panel untouched."""
    names = names or BOOK
    bt = load_panel(names)
    try:
        live = live_resolved_payout(names)
    except Exception as error:            # read-only monitor never dies on the API
        print("live fetch failed (%s); backtest panel only" % error); live = None
    if live is not None and not live.empty:
        # Append the RESOLVED live rounds as new rows -- assigning to bt[m]
        # would reindex onto the old dates and silently drop them, so concat
        # new-date rows instead (NaN for any model without that date; the
        # rolling signal's min_periods handles the holes). A ~4-month gap
        # between the backtest end and the post-migration live rounds is real
        # and expected -- the signal is row-based, so it treats the recent live
        # rounds as the trailing edge of the window.
        new = live.index[live.index > bt.index.max()]
        if len(new):
            bt = pd.concat([bt, live.reindex(index=new, columns=bt.columns)])
    return bt.sort_index()


def current_ranking(names=None, lookback=250, lag=LAG):
    """The models ranked by the current roll<lookback> signal (latest date)."""
    panel = combined_panel(names)
    sig = signal(panel, lookback=lookback, lag=lag).iloc[-1].dropna().sort_values(ascending=False)
    return sig, panel.index.max()


def calmar_signal(payout, window=250, lag=LAG):
    """
    Each model's trailing-`window` prop calmar as of each date, causal.

    calmar = mean(payout over the window) / max-drawdown of its cumulative
    payout -- a drawdown-adjusted momentum. `.shift(lag)` first so round t sees
    only resolved rounds. Zero-drawdown windows get a large finite value so they
    rank at the top without poisoning the sort with inf.
    """
    resolved = payout.shift(lag)

    def _cal(x):
        x = x[~np.isnan(x)]
        if len(x) < max(5, window // 3):
            return np.nan
        cum = np.cumsum(x)
        dd = float((np.maximum.accumulate(cum) - cum).max())
        return x.mean() / (dd if dd > 1e-9 else 1e-9)

    return resolved.rolling(window, min_periods=max(5, window // 3)).apply(_cal, raw=True)


def weights_top1(sig):
    """Full stack on the top model per date by `sig` (ties share); equal on a
    row with no signal."""
    rank = sig.rank(axis=1, ascending=False)
    w = (rank <= 1).astype(float)
    tot = w.sum(axis=1).replace(0.0, np.nan)
    w = w.div(tot, axis=0)
    w.loc[~sig.notna().any(axis=1)] = 1.0 / sig.shape[1]
    return w.fillna(1.0 / sig.shape[1])


def weights_conditional(mom, cal, k):
    """Top-1 by MOMENTUM, but only among the top-`k` models by CALMAR each date
    -- momentum timing gated on drawdown-adjusted quality."""
    return weights_top1(mom.where(cal.rank(axis=1, ascending=False) <= k))


def run_calmar_study(names=None, window=250, plot=None):
    """Backtest: calmar-criterion vs momentum, and momentum gated on top-k calmar."""
    payout = load_panel(names)                       # backtest panel (no live gap)
    train_idx, eval_idx = _split(payout.index)
    mom = signal(payout, lookback=window)
    cal = calmar_signal(payout, window=window)

    strat = {
        "equal (base)":        allocate(payout, "equal"),
        "mom_r250 (top1)":     weights_top1(mom),
        "calmar_r250 (top1)":  weights_top1(cal),
        "mom | top5 calmar":   weights_conditional(mom, cal, 5),
        "mom | top10 calmar":  weights_conditional(mom, cal, 10),
    }
    realized = {n: floor_weights(ladder(w), 0.0) for n, w in strat.items()}
    paths = {n: (realized[n] * payout).sum(axis=1) for n in strat}
    eqp, momp = paths["equal (base)"], paths["mom_r250 (top1)"]

    print("E55 calmar-criterion & calmar-gated momentum  window=%d  lag=%d ladder=%d"
          % (window, LAG, LADDER))
    print("book %d models  %d periods (%d train, %d held-out %s..%s)\n"
          % (payout.shape[1], len(payout), len(train_idx), len(eval_idx),
             eval_idx[0].date(), eval_idx[-1].date()))

    for wname, idx in (("HELD-OUT", eval_idx), ("full", payout.index)):
        print("== %s ==" % wname)
        print("%-20s %8s %7s %7s %7s %5s %6s | %16s %16s"
              % ("strategy", "payout", "sharpe", "dd", "calmar", "burn", "turn",
                 "dPay vs equal(t)", "dPay vs mom(t)"))
        print("-" * 108)
        for n in strat:
            p = paths[n].loc[idx]; s = summarize_path(p)
            de, te = paired_t(p, eqp.loc[idx])
            dm, tm = paired_t(p, momp.loc[idx])
            de_s = "" if n.startswith("equal") else "%+.5f(%+.2f)" % (de, te)
            dm_s = "" if n.startswith("mom_r250") else "%+.5f(%+.2f)" % (dm, tm)
            print("%-20s %8.5f %7.3f %7.3f %7.3f %5d %6.3f | %16s %16s"
                  % (n, s["payout"], s["sharpe"], s["max_dd"], s["calmar"],
                     s["burn"], turnover(realized[n].loc[idx]), de_s, dm_s))
        print()

    if plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = {"equal (base)": "#5f6368", "mom_r250 (top1)": "#c0392b",
                  "calmar_r250 (top1)": "#1a7a4f", "mom | top5 calmar": "#8e44ad",
                  "mom | top10 calmar": "#e08e0b"}
        for n in strat:
            c = paths[n].cumsum()
            ax.plot(c.index, c.values, lw=1.8, color=colors[n], label=n)
        ax.axvline(eval_idx[0], color="#1a4f7a", ls="--", lw=1, label="train | held-out")
        ax.set_title("Calmar-criterion & calmar-gated momentum — cumulative realised payout",
                     fontsize=11)
        ax.set_ylabel("cumulative payout (raw)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
        plt.tight_layout(); plt.savefig(plot, dpi=140, bbox_inches="tight")
        print("wrote %s" % plot)
    return paths


def dd_signal(payout, window=250, lag=LAG):
    """Each model's trailing-`window` max-drawdown of cumulative payout, causal
    (.shift(lag)). The drawdown LEVEL alone -- not divided by mean like calmar."""
    resolved = payout.shift(lag)

    def _dd(x):
        x = x[~np.isnan(x)]
        if len(x) < max(5, window // 3):
            return np.nan
        cum = np.cumsum(x)
        return float((np.maximum.accumulate(cum) - cum).max())

    return resolved.rolling(window, min_periods=max(5, window // 3)).apply(_dd, raw=True)


def weights_minmax(mmc_mom, dd, n):
    """Full stack on the MAX MMC-momentum model among the `n` LOWEST-drawdown
    models each date -- minimise dd(pay), then maximise mom_mmc within it."""
    low_dd = dd.rank(axis=1, ascending=True) <= n          # 1 = lowest drawdown
    return weights_top1(mmc_mom.where(low_dd))


def run_minmax_study(names=None, window=250, plot=None):
    """Backtest: max mom_mmc<window> within the low-{2,5,10} dd(pay) models."""
    from src.live import stake
    names = names or BOOK
    payout = load_panel(names)
    mmc = stake.mmc_series(names).reindex(index=payout.index, columns=payout.columns)
    train_idx, eval_idx = _split(payout.index)
    mmc_mom = signal(mmc, lookback=window)                 # trailing-250 mean MMC
    dd = dd_signal(payout, window=window)                  # trailing-250 dd(pay)

    strat = {
        "equal (base)":       allocate(payout, "equal"),
        "mom_r250 (top1)":    weights_top1(signal(payout, lookback=window)),
        "minmax N=2":         weights_minmax(mmc_mom, dd, 2),
        "minmax N=5":         weights_minmax(mmc_mom, dd, 5),
        "minmax N=10":        weights_minmax(mmc_mom, dd, 10),
    }
    realized = {k: ladder(w) for k, w in strat.items()}
    paths = {k: (realized[k] * payout).sum(axis=1) for k in strat}
    eqp, momp = paths["equal (base)"], paths["mom_r250 (top1)"]

    print("E55 min-max: max mom_mmc%d within low-N dd(pay)   lag=%d ladder=%d"
          % (window, LAG, LADDER))
    print("book %d models  %d periods (%d train, %d held-out %s..%s)\n"
          % (payout.shape[1], len(payout), len(train_idx), len(eval_idx),
             eval_idx[0].date(), eval_idx[-1].date()))
    for wname, idx in (("HELD-OUT", eval_idx), ("full", payout.index)):
        print("== %s ==" % wname)
        print("%-18s %8s %7s %7s %7s %5s %6s | %16s %16s"
              % ("strategy", "payout", "sharpe", "dd", "calmar", "burn", "turn",
                 "dPay vs equal(t)", "dPay vs mom(t)"))
        print("-" * 106)
        for n in strat:
            p = paths[n].loc[idx]; s = summarize_path(p)
            de, te = paired_t(p, eqp.loc[idx]); dm, tm = paired_t(p, momp.loc[idx])
            de_s = "" if n.startswith("equal") else "%+.5f(%+.2f)" % (de, te)
            dm_s = "" if n.startswith("mom_r250") else "%+.5f(%+.2f)" % (dm, tm)
            print("%-18s %8.5f %7.3f %7.3f %7.3f %5d %6.3f | %16s %16s"
                  % (n, s["payout"], s["sharpe"], s["max_dd"], s["calmar"],
                     s["burn"], turnover(realized[n].loc[idx]), de_s, dm_s))
        print()

    if plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        col = {"equal (base)": "#5f6368", "mom_r250 (top1)": "#c0392b",
               "minmax N=2": "#8e44ad", "minmax N=5": "#1a7a4f", "minmax N=10": "#e08e0b"}
        for n in strat:
            c = paths[n].cumsum(); ax.plot(c.index, c.values, lw=1.8, color=col[n], label=n)
        ax.axvline(eval_idx[0], color="#1a4f7a", ls="--", lw=1, label="train | held-out")
        ax.set_title("min-max: max mom_mmc250 within low-N dd(pay) — cumulative payout",
                     fontsize=11)
        ax.set_ylabel("cumulative payout (raw)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
        plt.tight_layout(); plt.savefig(plot, dpi=140, bbox_inches="tight"); print("wrote %s" % plot)
    return paths


def prop_calmar(series):
    """
    Our prop calmar: per-period payout MEAN over the max peak-to-trough fall of
    the CUMULATIVE payout, in score units (CLAUDE.md -- not the textbook Calmar).
    Only comparable across series of EQUAL length, so the full-history column is
    read on the common 470-period backtest and the rolling column on a fixed
    trailing window.
    """
    x = series.dropna()
    if len(x) < 2:
        return float("nan")
    cum = x.cumsum()
    dd = float((cum.cummax() - cum).max())
    return float(x.mean() / dd) if dd > 0 else float("inf")


def ranking_table(names=None, lookback=250):
    """Models ranked by the roll<lookback> momentum signal, with prop calmar
    (full backtest, equal-n comparable) and a rolling-<lookback> prop calmar."""
    names = names or BOOK
    combined = combined_panel(names)
    bt = load_panel(names)                      # 470, equal-n -> comparable calmar
    sig = signal(combined, lookback=lookback).iloc[-1]
    rows = {}
    for m in names:
        rows[m] = {
            "signal": float(sig.get(m, float("nan"))),
            "calmar": prop_calmar(bt[m]) if m in bt.columns else float("nan"),
            "calmar_roll": (prop_calmar(combined[m].dropna().iloc[-lookback:])
                            if m in combined.columns else float("nan")),
        }
    df = pd.DataFrame(rows).T[["signal", "calmar", "calmar_roll"]]
    return df.sort_values("signal", ascending=False), combined.index.max()


def watch(state_path="results/live/momentum_top1.json", lookback=250, notify=True):
    """Rerun top1 roll<lookback>, compare to the stored pick, e-mail on a CHANGE.
    Returns (top1, previous, changed, ranking)."""
    import json
    import os
    sig, asof = current_ranking(lookback=lookback)
    top1 = sig.index[0]
    prev = None
    if os.path.exists(state_path):
        try:
            prev = json.load(open(state_path)).get("top1")
        except Exception:
            prev = None
    changed = prev is not None and prev != top1
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    json.dump({"top1": top1, "asof": str(asof.date()), "lookback": lookback,
               "ranking": {k: round(float(v), 5) for k, v in sig.head(6).items()}},
              open(state_path, "w"), indent=2)
    print("top1 roll%d = %s  (as of %s)  prev=%s  changed=%s"
          % (lookback, top1, asof.date(), prev, changed))
    for k, v in sig.head(6).items():
        print("   %-22s %.5f%s" % (k, v, "  <-- TOP1" if k == top1 else ""))
    if changed and notify:
        try:
            from src.live.notify import send_text
            body = ("Momentum top1 (roll%d) CHANGED: %s -> %s (as of %s)\n\n"
                    "Move the stake accordingly. Current top 6:\n%s"
                    % (lookback, prev, top1, asof.date(),
                       "\n".join("  %-22s %.5f" % (k, v) for k, v in sig.head(6).items())))
            send_text("[crypto] momentum top1 changed: %s -> %s" % (prev, top1), body)
            print("emailed the change")
        except Exception as error:
            print("email failed: %s" % error)
    return top1, prev, changed, sig


def main():
    ap = argparse.ArgumentParser(description="Factor-momentum stake timing (E55).")
    ap.add_argument("--watch", action="store_true",
                    help="rerun top1 roll250 and e-mail if the rank changed")
    ap.add_argument("--no-mail", action="store_true")
    ap.add_argument("--grid", action="store_true", help="run the full arm grid")
    ap.add_argument("--rule", default="ztilt",
                    choices=("ztilt", "topk", "rankprop"))
    ap.add_argument("--kappa", type=float, default=1.0, help="ztilt strength")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="zero weights below this (the 0.1-NMR/round rule)")
    ap.add_argument("--objective", default="calmar", choices=("calmar", "payout", "sharpe"))
    ap.add_argument("--plot", default=None)
    ap.add_argument("--calmar-study", action="store_true",
                    help="backtest calmar-criterion vs momentum + calmar-gated momentum")
    ap.add_argument("--minmax-study", action="store_true",
                    help="max mom_mmc250 within low-{2,5,10} dd(pay)")
    args = ap.parse_args()
    if args.watch:
        watch(notify=not args.no_mail)
    elif args.minmax_study:
        run_minmax_study(plot=args.plot)
    elif args.calmar_study:
        run_calmar_study(plot=args.plot)
    elif args.grid:
        grid_run(floor=args.floor, objective=args.objective)
    else:
        run(rule=args.rule, kappa=args.kappa, floor=args.floor, plot=args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
