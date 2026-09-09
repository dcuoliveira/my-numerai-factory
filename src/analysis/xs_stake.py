# -*- coding: utf-8 -*-
"""
Walk-forward backtest of a CROSS-SECTIONAL, rank-based dynamic stake allocator.

    python3 -m src.analysis.xs_stake --report
    python3 -m src.analysis.xs_stake --plot /tmp/xs.png

Motivation (Out of Sample Ep 5, NasdaqJockey's live allocator): run a large
stable of models, score each on several dimensions -- level, recent level,
Sharpe, a forward proxy -- RANK the models cross-sectionally on each dimension,
build the composite from the RANKS (not the raw scores), stake the top ranks and
zero the rest, re-ranked every round. This module ports that idea to crypto and
tests whether it beats the deployed static mean-variance book.

The repo already has the STATIC allocator (src/live/stake.py, one weight vector
over all history) and a PER-MODEL dynamic backtest (src/analysis/dynamic_stake.py,
a time-varying multiplier per model on a fixed book). NasdaqJockey's system is
neither: it is a time-varying CROSS-SECTIONAL allocation over the whole stable.
So this adds a rolling cross-sectional simulator and reuses everything else.

Discipline (inherited from dynamic_stake, do not weaken):
  * NO LOOKAHEAD, lag=20. The 20-day target means a round's payout is not known
    for ~20 rounds. Every allocator sees only `payout.shift(lag)` -- the
    information resolved by the decision date. simulate_xs builds one shifted
    "info" panel and never lets an allocator touch the unshifted payout.
  * EQUAL CAPITAL is automatic: every allocator returns weights that sum to 1, so
    all arms (incl. the 3-slot static book) deploy the same total each round --
    the comparison is allocation SHAPE, not leverage.
  * SELECT ON TRAIN, REPORT ON HELD-OUT. Rank arms have a small grid (K, window)
    chosen on the first ~2/3 of dates by train Sharpe-net-turnover; every
    headline number is the held-out third.

Payout is `0.1*corr + mmc` (stake.payout_series), raw -- no payout factor, no
per-round clipping (same caveat as stake.py/dynamic_stake.py): drawdown levels
are an upper bound and only the RATIO between arms is trustworthy. The panel is
SIMULATED pseudo-OOS (rolling 130-period purged CV), not realized live results.

Judge: the classic pipeline's dependency-light stats battery
(my-numerai-pipeline/src/experiment/stats.py), loaded by file path so its
equities package deps are never imported. Primary metric: held-out Sharpe of the
net-of-turnover portfolio path; ADOPT/REJECT via stats.decide.
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

from src.live import stake, registry
from src.analysis.dynamic_stake import LAG, TRAIN_FRAC, _split, summarize_path

EPS = 1e-9

#: Pre-declared re-lock/turnover cost per unit of round-to-round weight change,
#: in payout units. Payout mean is ~0.03, so 0.005 is a deliberately non-trivial
#: friction. The report sweeps a few values; the verdict uses PRIMARY_COST.
COST_SWEEP = (0.0, 0.005, 0.01)
PRIMARY_COST = 0.005

#: Minimum resolved observations before an allocator may deploy (else flat/zero).
MIN_OBS = 60
#: Trailing window for covariance-based allocators (mv, hybrid sizing).
COV_WINDOW = 250

#: The deployed static book (src/live/stake.WEIGHTS) -- the incumbent B0.
STATIC_BOOK = {k: float(v) for k, v in stake.WEIGHTS.items()}


# --------------------------------------------------------------------------- #
# classic stats battery, loaded by file path (no package import)
# --------------------------------------------------------------------------- #

def _load_stats():
    here = os.path.dirname(os.path.abspath(__file__))
    # my-numerai-agent/src/analysis -> up 3 -> Downloads -> my-numerai-pipeline
    downloads = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    path = os.path.join(downloads, "my-numerai-pipeline", "src", "experiment", "stats.py")
    if not os.path.exists(path):
        raise SystemExit("classic stats.py not found at %s -- the judge lives in "
                         "the sibling my-numerai-pipeline repo." % path)
    spec = importlib.util.spec_from_file_location("_classic_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_panels(names=None):
    """(payout, mmc) DataFrames over the full stable's common dates.

    Reuses the live loader so a backtest number and a live stake number come off
    the identical joined series. Defaults to the full 24-slot registry stable.
    """
    names = names or list(registry.ORDER)
    payout = stake.payout_series(names)
    mmc = stake.mmc_series(names)
    common = payout.dropna(how="any").index.intersection(mmc.dropna(how="any").index)
    return payout.loc[common], mmc.loc[common]


# --------------------------------------------------------------------------- #
# allocators -- allocator(hist_p, hist_m, **params) -> weights Series (sum 1).
# hist_p / hist_m are the LAG-SHIFTED payout / mmc info up to the decision date:
# every value in them had resolved by that date. An allocator that only reads
# these cannot look ahead, by construction.
# --------------------------------------------------------------------------- #

def _mv_weights(payout_win, mmc_win, mu="mmc"):
    """Tangency weights w ~ inv(cov)@mean, negatives clipped, summed to 1.

    Same math as stake.solve_mv, but on a passed-in trailing window instead of
    the full-history disk series, so it can be re-solved each round.
    """
    cols = list(payout_win.columns)
    cov = payout_win.cov().to_numpy()
    exp = (mmc_win[cols].mean() if mu == "mmc" else payout_win.mean()).to_numpy()
    try:
        w = np.linalg.solve(cov, exp)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(cov) @ exp
    w = np.clip(w, 0.0, None)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(cols))
    return pd.Series(w / w.sum(), index=cols)


def _composite_rank(hist_p, recent, with_future=False):
    """Average cross-sectional rank over {level, recent, sharpe[, future]}.

    Higher is better. Each dimension is ranked across slots (scale-free), then
    the ranks are averaged -- the "score from the ranks, not from the scores"
    step NasdaqJockey credits.
    """
    lvl = hist_p.mean()                                   # lifetime "performance"
    rec = hist_p.tail(recent).mean()                      # "recent"
    tail = hist_p.tail(recent)
    shp = tail.mean() / (tail.std(ddof=1) + EPS)          # "sharp"
    dims = [lvl.rank(), rec.rank(), shp.rank()]
    if with_future:
        # "future": a point-in-time momentum proxy -- recent window minus the
        # window before it. A learned forward-performance model is out of scope
        # (too data-hungry here, and lookahead-prone); this is the honest stand-in.
        prior = hist_p.tail(2 * recent).head(recent).mean()
        fut = rec - prior
        dims.append(fut.rank())
    return pd.concat(dims, axis=1).mean(axis=1)


def alloc_equal(hist_p, hist_m):
    cols = hist_p.columns
    return pd.Series(1.0 / len(cols), index=cols)


def alloc_single_best(hist_p, hist_m, recent=60):
    tail = hist_p.tail(recent)
    shp = tail.mean() / (tail.std(ddof=1) + EPS)
    w = pd.Series(0.0, index=hist_p.columns)
    w.loc[shp.idxmax()] = 1.0
    return w


def alloc_rolling_mv(hist_p, hist_m, cov_window=COV_WINDOW):
    pw, mw = hist_p.tail(cov_window), hist_m.tail(cov_window)
    return _mv_weights(pw, mw).reindex(hist_p.columns).fillna(0.0)


def alloc_rank(hist_p, hist_m, k=5, recent=60):
    """A1 -- the video's system: composite cross-sectional rank, top-K, weight by
    rank. Correlation-blind by design (this is the arm expected to over-crowd)."""
    comp = _composite_rank(hist_p, recent, with_future=False)
    top = comp.sort_values(ascending=False).head(k).index
    w = comp.loc[top] - comp.loc[top].min() + 1.0        # positive, monotone in rank
    w = w / w.sum()
    return w.reindex(hist_p.columns).fillna(0.0)


def alloc_rank_future(hist_p, hist_m, k=5, recent=60):
    """A2 -- A1 plus the momentum 'future' dimension."""
    comp = _composite_rank(hist_p, recent, with_future=True)
    top = comp.sort_values(ascending=False).head(k).index
    w = comp.loc[top] - comp.loc[top].min() + 1.0
    w = w / w.sum()
    return w.reindex(hist_p.columns).fillna(0.0)


def alloc_hybrid(hist_p, hist_m, k=5, recent=60, cov_window=COV_WINDOW):
    """A3 -- rank-SELECT the top-K from the full stable, then mean-variance SIZE
    within it (correlation-aware). The proposed fix for rank's crowding blindness."""
    comp = _composite_rank(hist_p, recent, with_future=False)
    top = list(comp.sort_values(ascending=False).head(k).index)
    pw, mw = hist_p[top].tail(cov_window), hist_m[top].tail(cov_window)
    return _mv_weights(pw, mw).reindex(hist_p.columns).fillna(0.0)


def alloc_static_book(hist_p, hist_m):
    """B0 -- the deployed 3-slot mv book, constant weights (incumbent)."""
    w = pd.Series(0.0, index=hist_p.columns)
    for name, wt in STATIC_BOOK.items():
        if name in w.index:
            w.loc[name] = wt
    return w / w.sum() if w.sum() > 0 else w


ARMS = {
    "B0_static_mv":   (alloc_static_book, {}),
    "B1_equal":       (alloc_equal, {}),
    "B2_single_best": (alloc_single_best, {}),
    "B3_rolling_mv":  (alloc_rolling_mv, {}),
    "A1_rank":        (alloc_rank, "grid"),
    "A2_rank_future": (alloc_rank_future, "grid"),
    "A3_hybrid":      (alloc_hybrid, "grid"),
}

GRID = [{"k": k, "recent": r} for k in (3, 5, 8) for r in (30, 60)]


# --------------------------------------------------------------------------- #
# rolling cross-sectional simulator
# --------------------------------------------------------------------------- #

def simulate_xs(payout, mmc, allocator, params=None, lag=LAG, min_obs=MIN_OBS):
    """Walk-forward cross-sectional allocation.

    At each date t the allocator sees only info resolved by t (payout/mmc shifted
    back `lag`), returns weights summing to 1, and earns the UNSHIFTED payout at
    t. Returns (portfolio payout Series, per-round turnover Series, weight frame).
    """
    params = params or {}
    info_p, info_m = payout.shift(lag), mmc.shift(lag)
    cols = payout.columns
    port, weights = [], []
    prev = pd.Series(0.0, index=cols)
    turn = []
    for t in payout.index:
        hp = info_p.loc[:t].dropna(how="any")
        if len(hp) < min_obs:
            w = pd.Series(0.0, index=cols)
        else:
            hm = info_m.loc[hp.index]
            w = allocator(hp, hm, **params).reindex(cols).fillna(0.0)
        port.append(float((w * payout.loc[t]).sum()))
        turn.append(float((w - prev).abs().sum()))
        prev = w
        weights.append(w)
    return (pd.Series(port, index=payout.index),
            pd.Series(turn, index=payout.index),
            pd.DataFrame(weights, index=payout.index))


def net_path(port, turn, cost):
    """Portfolio payout net of a per-unit turnover cost."""
    return port - cost * turn


def effective_n(weight_frame, idx):
    """Mean effective number of positions = 1 / sum(w^2) (inverse Herfindahl)."""
    w = weight_frame.loc[idx]
    hhi = (w ** 2).sum(axis=1)
    return float((1.0 / hhi.replace(0.0, np.nan)).mean())


# --------------------------------------------------------------------------- #
# select-on-train
# --------------------------------------------------------------------------- #

def select_params(payout, mmc, allocator, grid, train_idx, cost=PRIMARY_COST,
                  lag=LAG):
    """Pick grid params maximising TRAIN Sharpe-net-turnover (selection touches
    train only; the caller reports the winner on held-out)."""
    best = None
    for params in grid:
        port, turn, _ = simulate_xs(payout, mmc, allocator, params, lag)
        net = net_path(port, turn, cost).loc[train_idx]
        s = summarize_path(net)["sharpe"]
        s = s if np.isfinite(s) else -np.inf
        if best is None or s > best[0]:
            best = (s, params)
    return best[1]


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def run(names=None, lag=LAG, train_frac=TRAIN_FRAC, cost=PRIMARY_COST,
        n_boot=5000, block=20, plot=None):
    payout, mmc = load_panels(names)
    train_idx, eval_idx = _split(payout.index, train_frac)
    S = _load_stats()

    print("cross-sectional dynamic stake backtest")
    print("stable: %d slots | %d dates (%s..%s) | %d train / %d held-out"
          % (payout.shape[1], len(payout), payout.index.min().date(),
             payout.index.max().date(), len(train_idx), len(eval_idx)))
    ev = payout.corr().where(~np.eye(payout.shape[1], dtype=bool)).stack()
    print("stable pairwise corr: median %.2f  max %.2f  (crowded: effective "
          "independent directions are few)\n" % (ev.median(), ev.max()))

    # simulate every arm; rank arms select params on train
    paths, turns, effN, sel = {}, {}, {}, {}
    for name, (fn, params) in ARMS.items():
        p = (select_params(payout, mmc, fn, GRID, train_idx, cost, lag)
             if params == "grid" else params)
        sel[name] = p
        port, turn, W = simulate_xs(payout, mmc, fn, p, lag)
        paths[name], turns[name], effN[name] = port, turn, effective_n(W, eval_idx)

    base_net = net_path(paths["B0_static_mv"], turns["B0_static_mv"], cost)

    # ---- held-out table ----------------------------------------------------
    print("== HELD-OUT (cost=%.3f/unit turnover) ==" % cost)
    hdr = ("%-16s %8s %8s %8s %8s %7s %6s %-16s"
           % ("arm", "sharpe", "net_shp", "mean", "max_dd", "turn", "effN", "params"))
    print(hdr); print("-" * len(hdr))
    rows = {}
    for name in ARMS:
        gross = paths[name].loc[eval_idx]
        net = net_path(paths[name], turns[name], cost).loc[eval_idx]
        st = summarize_path(net)
        rows[name] = st
        print("%-16s %8.3f %8.3f %8.4f %8.3f %7.3f %6.1f %-16s"
              % (name, summarize_path(gross)["sharpe"], st["sharpe"], st["mean"],
                 st["max_dd"], turns[name].loc[eval_idx].mean(), effN[name],
                 sel[name] if sel[name] else "-"))

    # ---- paired vs incumbent B0 -------------------------------------------
    print("\n== paired vs B0 on held-out net path (block=%d, n_boot=%d) ==" % (block, n_boot))
    print("%-16s %9s %9s %9s %7s  %s" % ("arm", "dShp", "ci_lo", "ci_hi", "p>0", "verdict"))
    verdicts = {}
    for name in ARMS:
        if name == "B0_static_mv":
            continue
        a = net_path(paths[name], turns[name], cost).loc[eval_idx].to_numpy()
        b = base_net.loc[eval_idx].to_numpy()
        ds = S.delta_sharpe_test(a, b, block=block, n_boot=n_boot)
        A = pd.DataFrame({"payout": a}, index=range(len(a)))
        B = pd.DataFrame({"payout": b}, index=range(len(b)))
        batt = S.paired_battery(A, B, primary="payout", block=block, n_boot=n_boot)
        dec = S.decide(batt)
        verdicts[name] = dec
        print("%-16s %+9.3f %+9.3f %+9.3f %7.2f  %s"
              % (name, ds["est"], ds["lo"], ds["hi"], ds["p_gt0"], dec["verdict"]))

    # ---- power + omnibus ---------------------------------------------------
    mde = S.mde_mean((net_path(paths["A3_hybrid"], turns["A3_hybrid"], cost).loc[eval_idx]
                      - base_net.loc[eval_idx]).to_numpy(), lag=lag)
    print("\nMDE(mean payout delta, 80%% power) vs B0: %.5f  "
          "(payout mean ~%.4f -- read INCONCLUSIVE against this yardstick)"
          % (mde, base_net.loc[eval_idx].mean()))
    arms_paths = {n: net_path(paths[n], turns[n], cost).loc[eval_idx].to_numpy() for n in ARMS}
    try:
        fr = S.friedman(arms_paths)
        print("Friedman across arms: stat=%.2f p=%.4f (arms differ overall; "
              "does not say any beats B0)" % (fr["stat"], fr["p"]))
    except Exception as e:
        print("Friedman: n/a (%s)" % str(e)[:60])
    try:
        pw = S.pairwise_vs_control(arms_paths, control="B0_static_mv")
        print("\nPairwise vs B0 (Wilcoxon, Holm-adjusted):")
        print(pw.to_string(index=False,
                           formatters={"mean_delta": "{:+.4f}".format,
                                       "sharpe_delta": "{:+.3f}".format,
                                       "wilcoxon_p": "{:.3f}".format,
                                       "p_holm": "{:.3f}".format}))
    except Exception as e:
        print("pairwise_vs_control: n/a (%s)" % str(e)[:60])

    print("\nCAVEATS: simulated pseudo-OOS panel (no payout factor / no clipping) "
          "-> levels are not NMR, only arm-vs-arm ratios are trustworthy; "
          "20-day overlap -> few effective blocks -> low power (see MDE); "
          "stable is ~2 effective bets -> watch effN.")

    if plot:
        _plot(paths, turns, eval_idx, cost, plot)
    return paths, turns, effN, sel, verdicts


def _plot(paths, turns, eval_idx, cost, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in ARMS:
        net = net_path(paths[name], turns[name], cost).loc[eval_idx].cumsum()
        lw = 2.5 if name == "B0_static_mv" else 1.3
        ls = "-" if name.startswith(("B", "A3")) else "--"
        ax.plot(net.index, net.values, lw=lw, ls=ls, label=name)
    ax.set_title("Held-out cumulative payout, net of turnover (cost=%.3f)" % cost)
    ax.set_ylabel("cumulative payout (raw units)"); ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight")
    print("wrote %s" % path)


def main():
    ap = argparse.ArgumentParser(description="Cross-sectional dynamic stake backtest.")
    ap.add_argument("--report", action="store_true", help="print the full arm table + verdicts")
    ap.add_argument("--models", nargs="+", default=None, help="override the stable (default: all 24)")
    ap.add_argument("--cost", type=float, default=PRIMARY_COST, help="turnover cost per unit")
    ap.add_argument("--lag", type=int, default=LAG)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    run(names=a.models, lag=a.lag, cost=a.cost, n_boot=a.n_boot, block=a.block, plot=a.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
