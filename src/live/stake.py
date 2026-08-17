"""
Allocate a stake across the live slots by equal risk contribution.

    python3 -m src.live.stake                  # show the plan, change nothing
    python3 -m src.live.stake --execute        # actually stake

Dry by default. Staked NMR is at risk and stake decreases do not settle
immediately, so this never moves money without --execute.

------------------------------------------------------------------------------
Why equal risk contribution, and not the obvious alternatives
------------------------------------------------------------------------------

The eleven-arm study could not separate any model from the baseline on mean
payout: the largest paired |t| was 1.30 and it belonged to a negative result.
With a 20-day forward target on daily dates the effective sample is ~26
independent windows, which is nowhere near enough to rank three means that sit
within 0.003 of each other.

Second moments do not have that problem. Variances and correlations are
estimated far more precisely than means from the same sample -- that asymmetry
is the whole reason risk-based allocation exists. So the allocation uses the part
of the estimate that is trustworthy and ignores the part that is not:

  weight by mean payout     fits noise; the ranking is not significant
  minimum variance          on this covariance it puts 84% in one slot. It
                            "wins" on backtested sharpe (1.3128 vs 1.2413) by
                            inverting a matrix whose 02/03 block correlates
                            0.938 -- it is choosing between two near-identical
                            models on a distinction the data cannot support
  equal weight              ignores that CURUPIRA_03's payout volatility is 27%
                            higher than CURUPIRA_01's, so it is not actually an
                            equal bet
  equal risk contribution   every slot contributes an equal share of portfolio
                            variance. Uses the covariance without inverting it
                            aggressively.

Measured on the 510-period payout series (0.1*CORR + 1*MMC), four slots:

    scheme                w_01   w_02   w_03   w_04    sharpe   maxdd
    equal                0.250  0.250  0.250  0.250
    equal risk contrib   0.321  0.232  0.222  0.225    1.2157  1.4463

CURUPIRA_04 correlates 0.998 with CURUPIRA_03, so ERC cannot do much with it:
the pair splits one model's risk budget between two slots. The three-slot book
(01+02+03) scored sharpe 1.2413 / drawdown 1.4217, i.e. BETTER than the four.
Swapping 04 for 03 rather than adding it gives 1.2481 / 1.4148, the best of the
three configurations -- see the caveats on CURUPIRA_04 in registry.py. Weights
here follow whatever registry.ORDER holds; re-solve with --solve after changing
it.

------------------------------------------------------------------------------
What the drawdown column means in NMR
------------------------------------------------------------------------------

These are cumulative-payout drawdowns, in units of stake. A blended max drawdown
of 1.42 means that over the backtested history the cumulative payout fell 1.42x
the staked amount from its running peak. Stake is genuinely burnable. Size the
position as risk capital, not as a deposit.
"""

import argparse
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd

from src.live import registry
from src.live.env import get_api, load_env

# Equal-risk-contribution weights, solved on the 510-period payout series.
# Recompute with --solve if the backtest is ever re-run.
WEIGHTS = {"CURUPIRA_01": Decimal("0.3210"),
           "CURUPIRA_02": Decimal("0.2317"),
           "CURUPIRA_03": Decimal("0.2221"),
           "CURUPIRA_04": Decimal("0.2252")}

QUANTUM = Decimal("0.000001")
PER_PERIOD = "results/per_period/crypto"

# numerapi's stake_increase() sends the v2ChangeStake mutation, which Numerai
# rejects on Crypto with "V2 staking is disabled for this tournament" -- the
# dashboard's tab is labelled "V3 Stakes" for a reason. The V3 mutation is
# increaseStake(amount, modelId); it returns the same V2ChangeStakeRequest
# object, so only the entry point differs.
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


def payout_series():
    """The 0.1*CORR + 1*MMC series for each slot, from its backtest run."""
    columns = {}
    for spec in registry.all_specs():
        model = spec["wrapper"]
        frame = pd.read_csv("%s/%s/%s.csv" % (PER_PERIOD, model, spec["backtest_run"]))
        columns[spec["name"]] = (0.1 * frame["corr"] + frame["mmc"]).to_numpy()
    return pd.DataFrame(columns)


def solve_erc(covariance, iterations=20000):
    """
    Equal risk contribution by fixed-point iteration.

    Each step nudges weights toward equalising w_i * (Cw)_i. The 0.1 exponent
    damps the update; a full step oscillates on a covariance this correlated.
    """
    n = len(covariance)
    weights = np.ones(n) / n
    for _ in range(iterations):
        contribution = weights * (covariance @ weights)
        weights *= (contribution.mean() / contribution) ** 0.1
        weights /= weights.sum()
    return weights


def plan(balance, weights=WEIGHTS):
    """
    Split `balance` by weight, rounding every slot DOWN.

    Rounding down throughout guarantees the total can never exceed the balance;
    the leftover dust stays in the wallet rather than failing the last call.
    """
    allocation = {name: (balance * weight).quantize(QUANTUM, rounding=ROUND_DOWN)
                  for name, weight in weights.items()}
    total = sum(allocation.values())
    if total > balance:
        raise ValueError("allocation %s exceeds balance %s" % (total, balance))
    return allocation, balance - total


def main():
    parser = argparse.ArgumentParser(
        description="Stake the wallet across the live slots by equal risk contribution.")
    parser.add_argument("--execute", action="store_true",
                        help="actually place the stakes (default is a dry plan)")
    parser.add_argument("--amount", default=None,
                        help="NMR to allocate; default is the whole balance")
    parser.add_argument("--solve", action="store_true",
                        help="re-solve the ERC weights from the payout series")
    args = parser.parse_args()

    load_env()
    api = get_api()

    weights = WEIGHTS
    if args.solve:
        series = payout_series()
        solved = solve_erc(series.cov().to_numpy())
        weights = {name: Decimal(str(round(w, 4)))
                   for name, w in zip(series.columns, solved)}
        print("re-solved ERC weights: %s\n" % {k: str(v) for k, v in weights.items()})

    balance = Decimal(api.get_account()["availableNmr"])
    amount = Decimal(args.amount) if args.amount else balance
    if amount > balance:
        raise SystemExit("asked to stake %s but the wallet holds %s" % (amount, balance))

    allocation, residual = plan(amount, weights)
    models = {name.upper(): mid for name, mid in (api.get_models() or {}).items()}

    print("wallet balance : %s NMR" % balance)
    print("allocating     : %s NMR\n" % amount)
    print("%-13s %8s %14s %10s  %s" % ("SLOT", "WEIGHT", "NMR", "~USD", "MODEL"))
    for name, value in allocation.items():
        spec = registry.get(name)
        print("%-13s %8s %14s %10s  %s (%d feat)" % (
            name, weights[name], value, "$%.2f" % (float(value) * 8.23),
            spec["wrapper"], spec["n_features"]))
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
            print("%-13s FAILED: %s" % (name, str(error)[:250]))


if __name__ == "__main__":
    main()
