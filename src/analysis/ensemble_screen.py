"""
Screen equal-weight ensembles of finished runs, on the payout series.

Costs seconds: run_ensemble rank-averages FINISHED forecasts, so nothing here is
refitted.

THIS IS A SCREEN, NOT A PRICE. It averages the members' payout SERIES, which is a
cheap proxy for what run_ensemble actually does -- a per-period rank average of
PREDICTIONS. The two can differ a lot: SACI_01 + CURUPIRA_03 + CURUPIRA_05_60
screened at payout 0.0903 / drawdown 0.23, and the real blend came out at
0.0977 / 1.31. Higher payout, and drawdown five times worse. Use this to RANK
candidates, then run the shortlist through run_ensemble before believing any
number or deploying anything. Two rules this encodes, both learned the hard way:

  * An ensemble is compared to its OWN BEST MEMBER, never to a global baseline.
    "Does combining beat holding the best of these" is the decision; equal-weight
    averaging dilutes the mean by construction, so a global comparison flatters
    every blend.
  * Rank by sharpe and drawdown, not payout. Across 3,276 combinations on this
    panel, zero beat their best member on payout at |t| > 2 and every point
    estimate was negative -- while the best pairing cut max drawdown from 1.00
    to 0.23. Blending here is a risk decision.

The arithmetic lives here rather than in the skill script so that the report and
the command line cannot drift apart, and so newey_west has one definition: the
one in ablation_report, which paired_delta.py already imports.
"""

import itertools

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

#: Numerai pays 20 days out, so consecutive daily periods overlap ~95%. Twenty
#: lags is the 20-day-target setting; a 60-day target needs 60.
LAG = 20


def stats(series):
    """
    Level and risk of one payout series.

    :param series: per-period payout
    :type series: pd.Series
    :return: payout, sharpe and max drawdown
    :rtype: dict
    """
    cumulative = series.cumsum()
    return dict(payout=float(series.mean()),
                sharpe=float(series.mean() / series.std(ddof=0)),
                maxdd=float((np.maximum.accumulate(cumulative) - cumulative).max()))


def _stats_array(values):
    """stats() on a bare numpy array, for the inner screening loop."""
    cumulative = values.cumsum()
    standard = values.std(ddof=0)
    return dict(payout=float(values.mean()),
                sharpe=float(values.mean() / standard) if standard > 0 else float("nan"),
                maxdd=float((np.maximum.accumulate(cumulative) - cumulative).max()))


def screen(members, sizes=(2, 3), lag=LAG):
    """
    Every equal-weight combination of the members, scored against its own best.

    :param members: name -> per-period payout series
    :type members: dict
    :param sizes: combination sizes to enumerate
    :type sizes: (int,)
    :param lag: Newey-West lag for the paired delta
    :type lag: int
    :return: (rows, solo stats by name, n shared periods)
    :rtype: ([dict], dict, int)
    """
    frame = pd.DataFrame(members).dropna()
    if frame.shape[1] < 2:
        return [], {}, len(frame)

    solo = {name: stats(frame[name]) for name in frame}
    names = list(frame.columns)

    # Everything below works on numpy and on ONE precomputed correlation matrix.
    # The obvious loop -- rebuilding a DataFrame slice and calling .corr() per
    # pair -- costs about 3ms a combination, which is 25s on a 37-arm book and
    # grows as the cube. The numbers are identical either way.
    values = frame.to_numpy(dtype="float64")
    correlations = np.corrcoef(values, rowvar=False)
    payouts = np.array([solo[n]["payout"] for n in names])

    rows = []
    for size in sizes:
        for combo in itertools.combinations(range(len(names)), size):
            picked = list(combo)
            blend = values[:, picked].mean(axis=1)
            row = _stats_array(blend)
            # The comparison is against the best of the members being combined,
            # not a global baseline: the decision is "does combining beat just
            # holding the best of these".
            best = picked[int(np.argmax(payouts[picked]))]
            difference = blend - values[:, best]
            _, _, t = newey_west(difference, lag=lag)
            rho = float(np.mean([correlations[a, b]
                                 for a, b in itertools.combinations(picked, 2)]))
            row.update(members=tuple(names[i] for i in picked), best=names[best],
                       delta=float(difference.mean()), t=float(t), rho=rho)
            rows.append(row)
    return rows, solo, len(frame)


def rank(rows, by):
    """
    Sort screened combinations.

    :param rows: rows from screen()
    :type rows: [dict]
    :param by: "sharpe" (descending) or "maxdd" (ascending)
    :type by: str
    :return: the sorted rows
    :rtype: [dict]
    """
    if by == "sharpe":
        return sorted(rows, key=lambda r: -r["sharpe"])
    if by == "maxdd":
        return sorted(rows, key=lambda r: r["maxdd"])
    if by == "payout":
        return sorted(rows, key=lambda r: -r["payout"])
    raise ValueError("unknown ranking: %s" % by)


def beat_their_best(rows):
    """
    The combinations that beat their own best member at |t| > 2.

    Reported even when it is empty, because that emptiness is the finding: on
    ~26 effective independent windows a null is the expected result.

    :param rows: rows from screen()
    :type rows: [dict]
    :return: the subset clearing the threshold
    :rtype: [dict]
    """
    return [r for r in rows if r["t"] > 2]
