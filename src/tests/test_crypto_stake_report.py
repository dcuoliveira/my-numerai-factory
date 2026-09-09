"""
Tests for the stake report's headline summary.

    python3 -m pytest src/tests/test_crypto_stake_report.py -q

This is the money block -- the four lines a reader takes at face value before
opening anything -- so the tests concentrate on the distinction it exists to
draw: what is BANKED versus what is merely marked to market. Everything here is
synthetic; nothing touches the Numerai API.
"""

import pandas as pd

from src.analysis import crypto_stake_report as R

BUILT = "2026-09-09 18:00 UTC"
NO_FX = (None, None, "not fetched")


def one_round(number, at_risk, payout, resolved, opened="2026-08-20T00:00:00Z"):
    return {"round": number, "at_risk": at_risk, "payout": payout,
            "resolved": resolved, "opened": opened,
            "path": [0.0, payout if payout is not None else 0.0]}


def fixture(rounds):
    """One staked model carrying `rounds`, shaped as fetch_live returns it."""
    slug, _ = R.STAKED_SLOTS[0]
    return {slug: {"rounds": rounds, "scores": {}}}


def summary(rounds):
    live = fixture(rounds)
    ctx = R.build(live, NO_FX, BUILT, dict(R.STAKED_SLOTS))
    return ctx["plain_text"]


def test_resolved_pl_counts_only_settled_rounds():
    """
    The header's whole job is separating banked money from an open mark. A
    round that has scored but not resolved carries a payout the API will still
    revise, so it belongs in the provisional line and nowhere else.
    """
    text = summary([one_round(1337, 10.0, 0.50, resolved=True),
                    one_round(1338, 10.0, 0.25, resolved=False)])
    assert "provisional P&L +0.7500 NMR" in text
    assert "resolved P&L    +0.5000 NMR" in text


def test_nothing_settled_reads_zero_and_says_so():
    text = summary([one_round(1337, 10.0, 0.50, resolved=False),
                    one_round(1338, 10.0, None, resolved=False)])
    assert "resolved P&L    +0.0000 NMR" in text
    assert "0 of 2 staked rounds have resolved" in text
    assert "everything below is mark-to-market" in text


def test_the_settlement_sentence_stops_claiming_mark_to_market():
    """
    The line used to assert "everything below is mark-to-market" unconditionally,
    which stops being true the first time a round settles -- and settles right
    next to a resolved figure that contradicts it.
    """
    text = summary([one_round(1337, 10.0, 0.50, resolved=True),
                    one_round(1338, 10.0, 0.25, resolved=False),
                    one_round(1339, 10.0, 0.25, resolved=False)])
    assert "1 of 3 staked rounds have resolved; the other 2 are still" in text
    assert "everything below is mark-to-market" not in text


def test_a_settled_loss_is_carried_with_its_sign():
    text = summary([one_round(1337, 10.0, -0.40, resolved=True)])
    assert "resolved P&L    -0.4000 NMR" in text


def test_resolved_is_a_subset_of_provisional():
    """Never a second independent number -- it can never exceed the whole."""
    rounds = [one_round(1337, 10.0, 0.50, resolved=True),
              one_round(1338, 10.0, 0.25, resolved=True),
              one_round(1339, 10.0, 0.25, resolved=False)]
    live = fixture(rounds)
    ctx = R.build(live, NO_FX, BUILT, dict(R.STAKED_SLOTS))
    assert "provisional P&L +1.0000 NMR" in ctx["plain_text"]
    assert "resolved P&L    +0.7500 NMR" in ctx["plain_text"]


def test_html_and_plain_summaries_carry_the_same_figure():
    """
    Two renderings of one block, built separately so the HTML entities never
    leak into the text/plain mail body. They must not drift apart.
    """
    live = fixture([one_round(1337, 10.0, 0.50, resolved=True),
                    one_round(1338, 10.0, 0.25, resolved=False)])
    ctx = R.build(live, NO_FX, BUILT, dict(R.STAKED_SLOTS))
    assert "resolved P&amp;L    +0.5000 NMR" in ctx["top_text"]
    assert "resolved P&L    +0.5000 NMR" in ctx["plain_text"]
    assert "&amp;" not in ctx["plain_text"], "HTML entity leaked into the mail body"
