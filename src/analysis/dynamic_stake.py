# -*- coding: utf-8 -*-
"""
Walk-forward backtest of a PER-MODEL, performance-adaptive stake rule.

    python3 -m src.analysis.dynamic_stake                # the 4 staked slots
    python3 -m src.analysis.dynamic_stake --plot out.png

The repo's live allocator (src/live/stake.py) is STATIC: it solves one weight
vector over the whole history and never varies stake over time. This module asks
the opposite question -- if you moved each model's stake UP after recent good
rounds and DOWN after bad ones, throttled by a drawdown ceiling, would the
realized book have been better on a drawdown-adjusted basis?

Three rules that decide whether the answer can be trusted rather than fitted:

  * NO LOOKAHEAD, and the lag is 20. The target is a 20-day forward return, so a
    round's payout is not known until ~20 rounds later. Every rule signal is
    therefore `.shift(lag)` before it can size a stake -- the decision at round t
    sees only payouts that had resolved by t. simulate() enforces it and
    test_dynamic_stake.py asserts a future payout cannot move an earlier stake.

  * EQUAL CAPITAL. The dynamic multiplier is normalised to average 1.0 over the
    TRAIN window, so dynamic and static deploy the same average NMR -- the
    comparison is sizing SHAPE, not extra leverage. Normalising on train (not the
    whole window) keeps the normalisation itself out-of-sample.

  * SELECT ON TRAIN, REPORT ON HELD-OUT. The rule has parameters (window, clip,
    drawdown cap). They are chosen on the first ~2/3 of dates and every headline
    number is the last ~1/3. In-sample-optimised sizing always looks good; the
    held-out split is the only honest read, exactly as the grid-subset work did.

Payout is `0.1*corr + mmc` (results_table.payout_series), raw -- no payout factor
or per-round clipping, same convention and caveat as src/live/stake.py: the
drawdowns here are an upper bound on the real NMR path, and the RATIO (dynamic vs
static) is the trustworthy part, not the level.
"""

import argparse

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west
from src.prediction.metrics import summarize

#: The deployed staked book. CU_SA_IA_BO_01 joins the three STAKED slots because
#: it carries live stake even though src/live/stake.STAKED (the solver's set) is
#: the three-slot subset it was last solved over.
STAKED = ["CURUPIRA_01", "CURUPIRA_05", "CURUPIRA_08", "CU_SA_IA_BO_01"]

#: The 20-day forward target's resolution lag, in periods. A rule may only use
#: payouts that had resolved by the decision date, i.e. shifted back this far.
LAG = 20

#: TRAIN is the first this-fraction of dates; the rest is held out for reporting.
TRAIN_FRAC = 2.0 / 3.0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_payouts(names=None):
    """
    Per-model payout series on the models' common dates.

    Reuses the live allocator's loader so a dynamic-stake number and a static
    stake number come off the identical series joined the identical way.

    :param names: slot names; defaults to the staked book
    :return: DataFrame indexed by date, one column per model
    :rtype: pd.DataFrame
    """
    from src.live import stake as stakelib

    return stakelib.payout_series(names or STAKED)


# --------------------------------------------------------------------------- #
# rules -- each maps ONE model's payout series to a RAW stake multiplier,
# already made causal (shifted by `lag`) so it never sees an unresolved round.
# --------------------------------------------------------------------------- #

def _causal(signal, lag):
    """Shift a signal back by the resolution lag; nothing else may size a stake."""
    return signal.shift(lag)


def rule_static(payout, lag=LAG):
    """The baseline: a flat multiplier of 1.0 every round."""
    return pd.Series(1.0, index=payout.index)


def rule_perf_adaptive(payout, window=60, fmin=0.25, fmax=2.0, lag=LAG,
                       halflife=None):
    """
    Size by how good the recent run has been against the model's own long run.

    multiplier = clip( recent_mean / expanding_mean , fmin, fmax )

    `recent_mean` is a trailing average of resolved payouts, estimated one of two
    ways: a flat `rolling(window)` mean (default), or -- when `halflife` is given
    -- an `ewm(halflife)` mean that weights nearer rounds more (decay estimator).
    Both are causal, and the whole ratio is `.shift(lag)` before it can size a
    stake, so round t compares "recent resolved payouts" to "everything resolved
    before t". Above 1 when recent > lifetime (size up), below when worse (size
    down). The expanding baseline -- not a fixed target -- keeps the rule
    scale-free across models with different mean payout.
    """
    if halflife is not None:
        recent = payout.ewm(halflife=halflife,
                            min_periods=max(5, int(round(halflife)))).mean()
    else:
        recent = payout.rolling(window, min_periods=max(5, window // 3)).mean()
    baseline = payout.expanding(min_periods=20).mean()
    ratio = recent / baseline.replace(0.0, np.nan)
    return _causal(ratio, lag).clip(fmin, fmax).fillna(1.0)


def rule_vol_target(payout, window=60, fmin=0.25, fmax=2.0, lag=LAG):
    """
    Context variant: scale inversely with recent volatility (risk targeting).

    multiplier = clip( median_vol / rolling_std(window) , fmin, fmax ), causal.
    Grows the stake when the model is calm, shrinks it when choppy.
    """
    vol = payout.rolling(window, min_periods=max(5, window // 3)).std()
    target = vol.expanding(min_periods=20).median()
    return _causal(target / vol.replace(0.0, np.nan), lag).clip(fmin, fmax).fillna(1.0)


def rule_dd_throttle(payout, dd_cap=1.0, lag=LAG):
    """
    A drawdown ceiling: cut stake once the model's own cumulative payout has
    fallen more than `dd_cap` from its peak, restore as it recovers.

    factor = clip( 1 - max(0, drawdown - dd_cap)/dd_cap , 0, 1 ), causal. This
    is the "drawdown-controlled" half of the deployed rule -- it never sizes up,
    only defends.
    """
    cumulative = payout.cumsum()
    drawdown = cumulative.cummax() - cumulative
    over = (drawdown - dd_cap).clip(lower=0.0)
    factor = (1.0 - over / (dd_cap if dd_cap > 0 else 1.0)).clip(lower=0.0)
    return _causal(factor, lag).clip(0.0, 1.0).fillna(1.0)


def rule_composed(payout, window=60, fmin=0.25, fmax=2.0, dd_cap=1.0, lag=LAG,
                  halflife=None):
    """
    The deployed rule: performance-adaptive sizing UNDER a drawdown ceiling.

    The two picks combined -- size up on a good run, but let the drawdown
    throttle veto that when the model is bleeding. The throttle can pull the
    multiplier below fmin (defence overrides the adaptive floor). `halflife`
    switches the adaptive numerator from a flat window to an EWMA decay.
    """
    adaptive = rule_perf_adaptive(payout, window, fmin, fmax, lag, halflife)
    throttle = rule_dd_throttle(payout, dd_cap, lag)
    return (adaptive * throttle).clip(0.0, fmax).fillna(1.0)


RULES = {
    "static": rule_static,
    "perf_adaptive": rule_perf_adaptive,
    "vol_target": rule_vol_target,
    "dd_throttle": rule_dd_throttle,
    "composed": rule_composed,
}


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #

def _split(index, train_frac=TRAIN_FRAC):
    cut = int(len(index) * train_frac)
    return index[:cut], index[cut:]


def simulate(payouts, rule_fn, params=None, lag=LAG, train_frac=TRAIN_FRAC):
    """
    Apply a rule per model and return the realized static and dynamic paths.

    For each model: raw multiplier from `rule_fn`, normalised to average 1.0 over
    TRAIN (equal capital), then realized dynamic payout = multiplier * payout.
    Static is the payout itself (flat unit stake).

    :return: (static DataFrame, dynamic DataFrame, stake DataFrame, train idx, eval idx)
    """
    params = params or {}
    train_idx, eval_idx = _split(payouts.index, train_frac)
    static, dynamic, stake = {}, {}, {}
    for col in payouts.columns:
        p = payouts[col].dropna()
        raw = rule_fn(p, lag=lag, **params)
        in_train = p.index.intersection(train_idx)
        mu = raw.loc[in_train].mean()
        g = raw / mu if mu and mu > 0 else raw
        stake[col] = g
        dynamic[col] = g * p
        static[col] = p
    return (pd.DataFrame(static), pd.DataFrame(dynamic),
            pd.DataFrame(stake), train_idx, eval_idx)


def portfolio(frame):
    """Equal-book aggregate: mean realized payout across the models each round."""
    return frame.mean(axis=1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def summarize_path(realized):
    """mean / sharpe / max_dd / calmar / burn(d) for one realized payout path."""
    realized = realized.dropna()
    stats = summarize(realized)
    dd = stats["max_drawdown"]
    calmar = (stats["mean"] / dd) if dd and dd > 0 else float("inf")
    return {"mean": stats["mean"], "sharpe": stats["sharpe"], "max_dd": dd,
            "calmar": calmar, "burn": int((realized < 0).sum()), "n": len(realized)}


def turnover(stake_col):
    """Mean absolute round-to-round change in stake -- the churn / settlement cost."""
    return float(stake_col.diff().abs().mean())


def paired_t(dynamic_path, static_path, lag=LAG):
    """Newey-West t of the per-round (dynamic - static) realized-payout difference."""
    diff = (dynamic_path - static_path).dropna()
    if len(diff) < 3:
        return float("nan"), float("nan")
    mean, _, t = newey_west(diff.to_numpy(), lag=lag)
    return float(mean), float(t)


# --------------------------------------------------------------------------- #
# out-of-sample parameter selection
# --------------------------------------------------------------------------- #

def default_grid():
    """The parameter grid searched on TRAIN for the composed rule."""
    grid = []
    for window in (30, 60, 90):
        for fmax in (1.5, 2.0):
            for dd_cap in (0.75, 1.0, 1.5):
                grid.append({"window": window, "fmin": 0.25,
                             "fmax": fmax, "dd_cap": dd_cap})
    return grid


def select_params(payouts, rule_fn, grid, lag=LAG, train_frac=TRAIN_FRAC,
                  objective="calmar"):
    """
    Pick the params that maximise the TRAIN-window portfolio objective.

    Selection touches TRAIN only; the caller then reports the winner on the
    held-out slice. calmar (return over drawdown) is the default because the
    stated goal is drawdown-controlled return.
    """
    best = None
    for params in grid:
        _, dyn, _, train_idx, _ = simulate(payouts, rule_fn, params, lag, train_frac)
        agg = portfolio(dyn).loc[train_idx]
        stats = summarize_path(agg)
        score = stats[objective] if np.isfinite(stats[objective]) else -np.inf
        if best is None or score > best[0]:
            best = (score, params)
    return best[1]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def run(names=None, rule="composed", lag=LAG, train_frac=TRAIN_FRAC,
        objective="calmar", plot=None):
    """Select on train, evaluate on held-out, print the static-vs-dynamic table."""
    payouts = load_payouts(names)
    rule_fn = RULES[rule]
    params = (select_params(payouts, rule_fn, default_grid(), lag, train_frac, objective)
              if rule == "composed" else {})
    static, dynamic, stake, train_idx, eval_idx = simulate(
        payouts, rule_fn, params, lag, train_frac)

    print("dynamic stake backtest -- rule=%s  lag=%d  train_frac=%.2f" % (rule, lag, train_frac))
    print("models: %s" % ", ".join(payouts.columns))
    print("selected params (on TRAIN): %s" % params)
    print("periods: %d total, %d train, %d held-out (%s .. %s)\n" % (
        len(payouts), len(train_idx), len(eval_idx),
        eval_idx[0].date(), eval_idx[-1].date()))

    header = ("%-16s %-9s %8s %8s %8s %8s %6s %8s %8s %6s"
              % ("model", "window", "s.pay", "d.pay", "s.dd", "d.dd",
                 "d.burn", "s.calm", "d.calm", "turn"))
    for window, idx in (("HELD-OUT", eval_idx), ("full", payouts.index)):
        print("== %s ==" % window)
        print(header); print("-" * len(header))
        for col in list(payouts.columns) + ["PORTFOLIO"]:
            if col == "PORTFOLIO":
                s = portfolio(static).loc[idx]; d = portfolio(dynamic).loc[idx]
                g = stake.mean(axis=1).loc[idx]
            else:
                s = static[col].loc[idx]; d = dynamic[col].loc[idx]; g = stake[col].loc[idx]
            ss, ds = summarize_path(s), summarize_path(d)
            print("%-16s %-9s %8.4f %8.4f %8.3f %8.3f %6d %8.3f %8.3f %6.3f" % (
                col, window, ss["mean"], ds["mean"], ss["max_dd"], ds["max_dd"],
                ds["burn"], ss["calmar"], ds["calmar"], turnover(g)))
        # paired significance on the portfolio path over this window
        d_mean, t = paired_t(portfolio(dynamic).loc[idx], portfolio(static).loc[idx], lag)
        print("portfolio paired (dyn-stat): mean %+.5f  NW(lag %d) t %+.2f\n" % (d_mean, lag, t))

    if plot:
        _plot(static, dynamic, eval_idx, plot, rule, params)
    return payouts, static, dynamic, stake, train_idx, eval_idx, params


def _plot(static, dynamic, eval_idx, path, rule, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    s = portfolio(static).cumsum(); d = portfolio(dynamic).cumsum()
    ax.plot(s.index, s.values, color="#5f6368", lw=2, label="static portfolio")
    ax.plot(d.index, d.values, color="#c0392b", lw=2, label="dynamic portfolio (%s)" % rule)
    ax.axvline(eval_idx[0], color="#1a4f7a", ls="--", lw=1, label="train | held-out")
    ax.set_title("Cumulative realized payout — static vs dynamic per-model stake\n%s" % params,
                 fontsize=11)
    ax.set_ylabel("cumulative payout (score units, raw)"); ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight")
    print("wrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward dynamic stake backtest.")
    parser.add_argument("--rule", default="composed", choices=sorted(RULES))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--lag", type=int, default=LAG)
    parser.add_argument("--objective", default="calmar", choices=("calmar", "mean", "sharpe"))
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()
    run(names=args.models, rule=args.rule, lag=args.lag,
        objective=args.objective, plot=args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
