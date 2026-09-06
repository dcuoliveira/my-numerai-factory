"""
Allocate a stake across the live slots by mean-variance optimisation.

    python3 -m src.live.stake                  # show the plan, change nothing
    python3 -m src.live.stake --execute        # actually stake

Dry by default. Staked NMR is at risk and stake decreases do not settle
immediately, so this never moves money without --execute.

------------------------------------------------------------------------------
Which slots are staked, and why not all of them
------------------------------------------------------------------------------

STAKED is deliberately smaller than the submitted book. Eight slots upload every
round; three of them are staked. Two measured facts force that:

  1. CURUPIRA_02/03/04 correlate 0.938-0.998 on the payout series. Staking all
     three is staking one model three times and paying three drawdowns for one
     bet. CURUPIRA_07 correlates 0.978 with CURUPIRA_08 and has no scored run at
     all, so nothing can even weight it.

  2. The only genuine decorrelator in the project is CURUPIRA_05_60, whose score
     series runs -0.28 to -0.37 against the 20-day models -- and it is in
     NOT_SUBMITTED, so it cannot be staked directly. CURUPIRA_08 is the only
     staked route to that exposure. That, not its headline mean, is why it is in.

Measured on the 470 dates all three share (CURUPIRA_05_60's label is NaN over
the last 40, so CURUPIRA_08 is scored on 470, not 510):

    book                          payout   p.sharpe   p.maxdd
    01+02+03+04 (the old book)   0.0874     1.2093    1.4462
    01+05+08                     0.0920     1.5277    0.9233

Paired block bootstrap (L=20, 5000 draws) against the old book: sharpe +0.30,
95% CI +0.02 to +0.63, P = 0.983. The PAYOUT gain is +0.0038 at P = 0.73 and is
NOT significant -- same as every other level result on this panel. This is a
risk trade, not a return trade. Say so when reporting it.

------------------------------------------------------------------------------
How the weights are set
------------------------------------------------------------------------------

SCHEME picks the solver; `mv` is deployed. All four are available:

  mv    the deployed scheme. Tangency weights, w proportional to inv(C) @ mmc,
        normalised to sum to 1 -- the classical mean-variance answer given both
        estimates. Mean MMC is the expected return because MMC carries weight
        1.0 in the crypto payout against CORR's 0.1; the covariance is of the
        full payout series, because that is what a stake's value moves with.
        `--mv-mu payout` maximises the payout mean instead.

  mmc   risk budgeting: each slot's share of portfolio VARIANCE is proportional
        to its mean MMC. The same two inputs, spent more conservatively -- it
        uses the covariance without inverting it against the means.

  erc   equal risk contribution: risk budgeting with a flat budget. No view on
        returns at all.

  equal one third each.

Measured on the 470 shared dates:

    scheme     w_01     w_05     w_08   payout  p.sharpe  p.maxdd  risk share
    equal    0.3333   0.3333   0.3333   0.0916    1.5261   0.9330  32/32/36
    erc      0.3448   0.3452   0.3099   0.0913    1.5253   0.9305  33/33/33
    mmc      0.3184   0.3432   0.3384   0.0917    1.5271   0.9221  30/33/36
    mv       0.2134   0.3702   0.4165   0.0930    1.5280   0.8690  19/36/45

Note how little separates them in the objective and how much in the allocation.
mv beats the risk budget by 0.0009 of sharpe and takes CURUPIRA_01 from 32% to
19% of portfolio variance to get it. That is the shape of the estimation problem
here, not a quirk of these three slots.

WHAT MEAN-VARIANCE COSTS, STATED PLAINLY. It spends its whole budget on the
least identifiable half of the inputs. Block-bootstrap the 470 dates in 20-day
blocks and re-solve (`--stability`):

    scheme   CURUPIRA_01           CURUPIRA_05           CURUPIRA_08
    mv       [-0.076, +0.474]      [+0.046, +0.734]      [+0.014, +0.809]
    mmc      [+0.287, +0.353]      [+0.308, +0.374]      [+0.320, +0.357]

mv wants CURUPIRA_01 SHORT in 10% of draws -- a weight that cannot be
implemented, since a stake cannot be negative; solve_mv clips it and says so.
Shrinking the means 50% toward their average, an arbitrary but entirely
defensible choice, moves the mv weight on CURUPIRA_08 from 0.416 to 0.165. An
allocation that swings that far on a modelling decision nobody can adjudicate is
reading the estimator as much as the data. Run `--stability` before re-solving,
and read the bands rather than the point estimate.

The plan still prints each slot's realised variance share and flags anything
over REVIEW_SHARE, which matters more under mv than under the risk budget: mv
has no mechanism that stops one slot taking the book.

------------------------------------------------------------------------------
Staking a blend is not a third position
------------------------------------------------------------------------------

CURUPIRA_06 and _08 are rank averages of OTHER SLOTS, so staking them is not a
fresh bet -- it is a pre-mixed weight vector over CURUPIRA_01/04/05/05_60. The
plan prints the expansion (registry.base_exposure) for that reason: 01+05+08 at
the deployed mv weights is really 51% CURUPIRA_05, and a book of 01+06+08 would
be 50% CURUPIRA_01 while looking like three positions.

Watch that number under mv in particular. CURUPIRA_05's own members and weights
were searched in-sample over these same periods, so half the book now sits on an
arm the registry itself calls the most promising hypothesis rather than an
established result -- the risk-budget scheme held it at 46%, the mv solution
pushes it to 51%.

------------------------------------------------------------------------------
What the drawdown column means in NMR
------------------------------------------------------------------------------

These are cumulative-payout drawdowns, in units of stake. A book drawdown of
0.92 means that over the backtested history the cumulative payout fell 0.92x the
staked amount from its running peak, and the whole of it arrived inside one
20-day window (worst 20-day sum -0.92). Stake is genuinely burnable: size the
position as risk capital you would accept losing in full, not as a deposit.

One honest caveat on the level: these series are raw scores. Nothing here
applies Numerai's payout factor or per-round clipping, so the drawdown above is
an upper bound on the real NMR path rather than a forecast of it. The SHARPE
ranking is unaffected -- a common scale factor cancels -- which is another
reason the allocation reads second moments and not means.
"""

import argparse
import os
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd

from src.live import registry
from src.live.env import get_api, load_env

# The staked book. Deliberately a subset of registry.ORDER -- see the docstring.
# Everything else still submits every round; it just carries no money.
STAKED = ["CURUPIRA_01", "CURUPIRA_05", "CURUPIRA_08"]

# The scheme the cached WEIGHTS below were solved with. Changing this does not
# change what is staked -- re-solve and update WEIGHTS too, or the test in
# test_live.py fails, which is the point.
SCHEME = "mv"

# Mean-variance (tangency) on mean MMC against the payout covariance, solved on
# the 470 dates the three slots share. Recompute with
# `--solve --scheme mv` if any member's backtest is re-run.
WEIGHTS = {"CURUPIRA_01": Decimal("0.2134"),
           "CURUPIRA_05": Decimal("0.3702"),
           "CURUPIRA_08": Decimal("0.4165")}

QUANTUM = Decimal("0.000001")

# An MMC tilt sizes on a mean, and means are the part of this panel the sample
# cannot support. Past this share of portfolio variance in one slot, the
# allocation has stopped being diversified and is a bet on that mean -- the plan
# says so rather than silently placing it.
REVIEW_SHARE = 0.5

# numerapi's stake_increase() sends v2ChangeStake, which Numerai rejects on
# Crypto with "V2 staking is disabled for this tournament". increaseStake() was
# believed to be the V3 entry point -- it is not. Measured 2026-08-19, it
# returns the SAME rejection for every crypto slot.
#
# Crypto stakes are V3 "Fusion" stakes and live on-chain: the staking contract
# is 0xC1f46Adf341145369eeE9fBe0D84fA1cb0c24706 on mainnet (v3StakeConfig),
# funded from the account's Privy wallet, and a stake change is an Ethereum
# transaction authorised by v3StakeAuth(maxAmount, staker, submissionId) -- not
# a GraphQL mutation. Placing one needs a signer for that wallet, which this
# process does not have, so --execute cannot currently complete on crypto and
# the plan is the deliverable. The mutation is kept because the schema still
# exposes it and the rejection is informative.
INCREASE_STAKE = """
mutation($amount: String!, $modelId: ID!) {
  increaseStake(amount: $amount, modelId: $modelId) {
    id type status requestedAmount dueDate drain
  }
}"""


def increase_stake(api, amount, model_id):
    """Place a V3 stake increase and return the request record."""
    response = api.raw_query(INCREASE_STAKE,
                             {"amount": str(amount), "modelId": model_id},
                             authorization=True)
    return response["data"]["increaseStake"]


def per_period(names=None):
    """
    Per-period backtest scores for the staked slots, on their COMMON dates.

    Joining on date rather than stacking columns by position is not pedantry:
    CURUPIRA_08 inherits CURUPIRA_05_60's 60-day label and is scored on 470
    dates where CURUPIRA_01 has 510. Lining those up by row silently pairs
    different days and produces a covariance of nothing in particular.
    """
    frames = {}
    for name in (names or STAKED):
        path = registry.per_period_path(name)
        if path is None:
            # CURUPIRA_07's registry backtest_run is a SCREEN, not a run, so the
            # registry cannot resolve it -- but it is a submitted slot and has
            # to be weightable. src.analysis.slots keeps the reconstruction that
            # answers "what does this slot actually submit". Imported lazily:
            # slots imports this module back, for STAKED.
            from src.analysis import slots as _slots
            model, run, why = _slots.slot_run(name)
            if not model:
                raise SystemExit("%s has no scored run (%s) -- it cannot be "
                                 "weighted" % (name, why))
            path = os.path.join(_slots.registry.PER_PERIOD_DIR,
                                model, "%s.csv" % run)
        frames[name] = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    index = None
    for frame in frames.values():
        index = frame.index if index is None else index.intersection(frame.index)
    return {name: frame.loc[index] for name, frame in frames.items()}, index


def payout_series(names=None):
    """The payable 0.1*CORR + 1*MMC series for each staked slot."""
    frames, _ = per_period(names)
    return pd.DataFrame({name: 0.1 * f["corr"] + f["mmc"]
                         for name, f in frames.items()})


def mmc_series(names=None):
    """The MMC column alone -- what the default scheme tilts on."""
    frames, _ = per_period(names)
    return pd.DataFrame({name: f["mmc"] for name, f in frames.items()})


def solve_risk_budget(covariance, budget=None, iterations=20000):
    """
    Weights whose RISK contributions match `budget`, by fixed-point iteration.

    Each step nudges weights toward w_i * (Cw)_i proportional to budget_i. With
    budget=None the target is equal, which is plain equal risk contribution; the
    MMC scheme passes mean MMC instead. The 0.1 exponent damps the update -- a
    full step oscillates on a covariance this correlated.

    Budgeting the RISK rather than the weight is what lets a tilt on MMC also
    read the correlations: a slot that diversifies the others spends less
    portfolio variance per unit of weight, so it earns more weight for the same
    budget. That is why CURUPIRA_05 ends up marginally ahead of CURUPIRA_08 here
    despite the lower mean MMC -- it correlates 0.569 with CURUPIRA_01 where
    CURUPIRA_08 correlates 0.768.
    """
    n = len(covariance)
    target = np.ones(n) / n if budget is None else np.asarray(budget, float)
    if (target <= 0).any():
        raise SystemExit("risk budgets must all be positive, got %s" % target)
    target = target / target.sum()
    weights = np.ones(n) / n
    for _ in range(iterations):
        contribution = weights * (covariance @ weights)
        weights *= (target / (contribution / contribution.sum())) ** 0.1
        weights /= weights.sum()
    return weights


def solve_erc(covariance, iterations=20000):
    """Equal risk contribution -- risk budgeting with a flat budget."""
    return solve_risk_budget(covariance, None, iterations)


def solve_mmc(names=None):
    """
    Risk budgets proportional to each slot's mean MMC.

    MMC carries weight 1.0 in the crypto payout against CORR's 0.1, so this
    sizes each slot by the part of the score that actually pays -- but it spends
    that budget in units of portfolio variance, so the correlations set what a
    given budget buys in weight. It is still a tilt on a MEAN, which is the least
    identifiable quantity on this panel, so the plan prints the realised variance
    share and REVIEW_SHARE flags an allocation that has become a bet on one slot.
    """
    payout = payout_series(names)
    means = mmc_series(names)[payout.columns].mean().to_numpy()
    if (means <= 0).any():
        raise SystemExit("a slot has non-positive mean MMC; refusing to tilt on it")
    return payout.columns.tolist(), solve_risk_budget(payout.cov().to_numpy(), means)


def solve_mv(names=None, mu="mmc"):
    """
    Tangency weights, w proportional to inv(C) @ mu, normalised to sum to 1.

    `mu` picks what is being maximised: "mmc" uses each slot's mean MMC (the
    payable component that carries weight 1.0), "payout" the full 0.1*CORR + MMC
    series. Either way the covariance is of the PAYOUT series, because that is
    what a stake's value actually moves with.

    Read the health block the plan prints before staking this. Mean-variance
    spends its whole budget on the least identifiable half of the inputs, and on
    470 correlated dates the block-bootstrapped 90% band on these weights is
    roughly +-0.35 wide against +-0.03 for the risk-budget scheme. A negative
    weight is not implementable -- you cannot short a stake -- so any slot the
    solver wants short is clipped to zero and the plan says so out loud.
    """
    payout = payout_series(names)
    columns = payout.columns.tolist()
    covariance = payout.cov().to_numpy()
    if mu == "mmc":
        expected = mmc_series(names)[columns].mean().to_numpy()
    elif mu == "payout":
        expected = payout.mean().to_numpy()
    else:
        raise SystemExit("unknown --mv-mu '%s'" % mu)
    weights = np.linalg.solve(covariance, expected)
    if (weights < 0).any():
        short = [n for n, w in zip(columns, weights) if w < 0]
        print("mean-variance wants a SHORT in %s -- clipped to zero. A stake "
              "cannot be negative, so this is no longer the optimum it solved "
              "for." % ", ".join(short))
        weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        raise SystemExit("mean-variance produced no positive weights")
    return columns, weights / weights.sum()


def bootstrap_weights(scheme, names=None, draws=500, block=20, seed=0):
    """
    Re-solve the scheme on block-resampled dates, to show how firm it is.

    20-day blocks because a 20-day forward target on daily dates leaves
    consecutive periods ~95% overlapping; resampling single days would treat one
    event as twenty and report a stability the panel does not have.
    """
    payout = payout_series(names)
    mmc = mmc_series(names)[payout.columns]
    scores, means = payout.to_numpy(), mmc.to_numpy()
    n = len(scores)
    generator = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        index = []
        while len(index) < n:
            start = generator.integers(0, n - block)
            index.extend(range(start, start + block))
        index = np.array(index[:n])
        covariance = np.cov(scores[index], rowvar=False)
        expected = means[index].mean(axis=0)
        if scheme == "mv":
            weights = np.linalg.solve(covariance, expected)
        elif scheme == "mmc":
            weights = solve_risk_budget(covariance, expected, 2000)
        else:
            weights = solve_erc(covariance, 2000)
        drawn.append(weights / weights.sum())
    return payout.columns.tolist(), np.array(drawn)


def solve(scheme, names=None, mv_mu="mmc"):
    """Weights for one scheme, as a name -> Decimal mapping rounded to 4dp."""
    names = names or STAKED
    if scheme == "mmc":
        columns, weights = solve_mmc(names)
    elif scheme == "mv":
        columns, weights = solve_mv(names, mv_mu)
    elif scheme == "erc":
        payout = payout_series(names)
        columns, weights = payout.columns.tolist(), solve_erc(payout.cov().to_numpy())
    elif scheme == "equal":
        columns, weights = list(names), np.ones(len(names)) / len(names)
    else:
        raise SystemExit("unknown scheme '%s'" % scheme)
    # Rounding to 4dp can leave the total off 1.0 by a few 1e-4. plan() rounds
    # every slot DOWN afterwards, so the drift lands in the residual and can
    # never overdraw -- but it must not be large enough to strand real NMR.
    rounded = {name: Decimal(str(round(float(w), 4)))
               for name, w in zip(columns, weights)}
    drift = abs(sum(rounded.values()) - Decimal("1"))
    if drift > Decimal("0.001"):
        raise SystemExit("weights sum to %s, not 1" % sum(rounded.values()))
    return rounded


def describe_weights(weights):
    """Level, risk and variance-share for a candidate allocation."""
    names = list(weights)
    payout = payout_series(names)[names]
    vector = np.array([float(weights[n]) for n in names])
    combined = payout.to_numpy() @ vector
    cumulative = np.cumsum(combined)
    contribution = vector * (payout.cov().to_numpy() @ vector)
    return {
        "n_periods": len(payout),
        "payout": combined.mean(),
        "mmc": (mmc_series(names)[names].to_numpy() @ vector).mean(),
        "sharpe": combined.mean() / combined.std(ddof=1),
        "max_dd": (np.maximum.accumulate(cumulative) - cumulative).max(),
        "risk_share": dict(zip(names, contribution / contribution.sum())),
    }


def balances(api):
    """
    What the account actually holds, split the way V3 staking splits it.

    Crypto stakes are NOT held in the custodial balance the Classic tournament
    uses: they sit on-chain in the Fusion staking contract, funded from the
    account's Privy wallet. So `availableNmr` is what is LEFT after those locks,
    and reading it alone hides the position entirely -- 0.27 free against 130
    already staked looks identical to an empty account.
    """
    account = api.raw_query(
        """query{ account{ availableNmr onChainWalletBalance heldForFusionStakes
                           privyWalletAddress } }""",
        authorization=True)["data"]["account"]
    return {
        "available": Decimal(account["availableNmr"]),
        "on_chain": Decimal(account["onChainWalletBalance"]),
        "staked": Decimal(account["heldForFusionStakes"]),
        "wallet": account["privyWalletAddress"],
    }


def plan(balance, weights=WEIGHTS):
    """
    Split `balance` by weight, rounding every slot DOWN.

    Rounding down throughout guarantees the total can never exceed the balance;
    the leftover dust stays in the wallet rather than failing the last call.

    Weights are renormalised first. They are stored to 4dp, so a solved vector
    can sum to 1.0001 and round-down alone is not enough to stay inside the
    balance -- the mean-variance weights do exactly that, and the guard below
    fired on it. Dividing through by the actual total costs nothing when they
    already sum to one.
    """
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("weights sum to %s" % total_weight)
    allocation = {name: (balance * weight / total_weight).quantize(
                      QUANTUM, rounding=ROUND_DOWN)
                  for name, weight in weights.items()}
    total = sum(allocation.values())
    if total > balance:
        raise ValueError("allocation %s exceeds balance %s" % (total, balance))
    return allocation, balance - total


def main():
    parser = argparse.ArgumentParser(
        description="Stake the wallet across the staked slots by MMC weight.")
    parser.add_argument("--execute", action="store_true",
                        help="actually place the stakes (default is a dry plan)")
    parser.add_argument("--amount", default=None,
                        help="NMR to allocate; default is the whole balance")
    parser.add_argument("--solve", action="store_true",
                        help="re-solve the weights from the backtested series")
    parser.add_argument("--scheme", default=SCHEME,
                        choices=("mmc", "mv", "erc", "equal"),
                        help="weighting scheme for --solve (default: mmc)")
    parser.add_argument("--mv-mu", default="mmc", choices=("mmc", "payout"),
                        help="what --scheme mv maximises (default: mmc)")
    parser.add_argument("--stability", action="store_true",
                        help="block-bootstrap the chosen scheme's weights")
    parser.add_argument("--offline", action="store_true",
                        help="print the weights and their backtest, touch no API")
    args = parser.parse_args()

    weights = WEIGHTS
    if args.solve:
        weights = solve(args.scheme, mv_mu=args.mv_mu)
        label = args.scheme + (" on %s" % args.mv_mu if args.scheme == "mv" else "")
        print("re-solved %s weights: %s\n"
              % (label, {k: str(v) for k, v in weights.items()}))

    if args.stability:
        columns, drawn = bootstrap_weights(args.scheme)
        print("block bootstrap of the %s weights (%d draws, 20-day blocks)"
              % (args.scheme, len(drawn)))
        for position, name in enumerate(columns):
            low, median, high = np.percentile(drawn[:, position], [5, 50, 95])
            print("  %-13s median %+0.3f  90%% band [%+0.3f, %+0.3f]  "
                  "P(short)=%.2f" % (name, median, low, high,
                                     (drawn[:, position] < 0).mean()))
        print()

    stats = describe_weights(weights)
    heavy = [name for name, share in stats["risk_share"].items()
             if share > REVIEW_SHARE]
    print("backtest of this allocation, %d common periods" % stats["n_periods"])
    print("  payout %.6f   mmc %.6f   p.sharpe %.4f   p.maxdd %.4f"
          % (stats["payout"], stats["mmc"], stats["sharpe"], stats["max_dd"]))
    print("  variance share : %s%s" % ("  ".join(
        "%s %.0f%%" % (name.replace("CURUPIRA_", "C"), 100 * share)
        for name, share in stats["risk_share"].items()),
        "   <-- REVIEW: %s" % ", ".join(heavy) if heavy else ""))
    # A slot blend is a pre-mixed weight vector over other slots, so three
    # positions can be two models. Print what is actually being bet on.
    print("  base exposure  : %s" % "  ".join(
        "%s %.0f%%" % (name.replace("CURUPIRA_", "C"), 100 * value)
        for name, value in sorted(registry.base_exposure(weights).items(),
                                  key=lambda item: -item[1])))

    if args.offline:
        return

    load_env()
    api = get_api()
    account = balances(api)
    balance = account["available"]
    # availableNmr alone reads as "you have 0.27 NMR" when 130 are already at
    # risk in V3 stakes. Crypto stakes are held on-chain in the Fusion contract,
    # so the free balance is onChainWalletBalance minus what is locked -- print
    # all three or the plan looks like a first deposit when it is a top-up.
    print("\non-chain wallet : %s NMR (%s)"
          % (account["on_chain"], account["wallet"]))
    print("held in stakes  : %s NMR" % account["staked"])
    print("free to stake   : %s NMR" % balance)
    amount = Decimal(args.amount) if args.amount else balance
    if amount > balance:
        raise SystemExit("asked to stake %s but the wallet holds %s" % (amount, balance))

    allocation, residual = plan(amount, weights)
    models = {name.upper(): mid for name, mid in (api.get_models() or {}).items()}

    print("allocating      : %s NMR\n" % amount)
    print("%-13s %8s %14s %10s  %s" % ("SLOT", "WEIGHT", "NMR", "~USD", "MODEL"))
    for name, value in allocation.items():
        spec = registry.get(name)
        print("%-13s %8s %14s %10s  %s (%d feat)" % (
            name, weights[name], value, "$%.2f" % (float(value) * 8.23),
            spec["wrapper"] or "blend", spec["n_features"]))
    print("%-13s %8s %14s" % ("residual", "-", residual))

    missing = [n for n in allocation if n.upper() not in models]
    if missing:
        raise SystemExit("\nthese slots are not on the account: %s" % missing)

    if not args.execute:
        print("\nDRY RUN -- nothing staked. Re-run with --execute to place these.")
        return

    print()
    for name, value in allocation.items():
        try:
            result = increase_stake(api, value, models[name.upper()])
            print("%-13s staked %s NMR -> %s" % (name, value, result))
        except Exception as error:
            # One failure must not abort the others: a partial stake is a valid
            # state, and the remaining slots are still worth placing.
            message = str(error)
            print("%-13s FAILED: %s" % (name, message[:250]))
            if "V2 staking is disabled" in message:
                print("              crypto stakes are on-chain V3 Fusion "
                      "stakes; place this split from the dashboard's V3 Stakes "
                      "tab, or sign the contract call from %s"
                      % account["wallet"])
                break


if __name__ == "__main__":
    main()
