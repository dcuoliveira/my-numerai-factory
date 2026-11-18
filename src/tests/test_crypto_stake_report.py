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
    assert "provisional P&L +0.2500 NMR" in text
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


def test_resolved_and_provisional_are_disjoint():
    """Provisional covers open rounds, resolved covers settled ones -- never both."""
    rounds = [one_round(1337, 10.0, 0.50, resolved=True),
              one_round(1338, 10.0, 0.25, resolved=True),
              one_round(1339, 10.0, 0.25, resolved=False)]
    live = fixture(rounds)
    ctx = R.build(live, NO_FX, BUILT, dict(R.STAKED_SLOTS))
    assert "provisional P&L +0.2500 NMR" in ctx["plain_text"]
    assert "resolved P&L    +0.7500 NMR" in ctx["plain_text"]


def test_stake_and_book_count_only_open_rounds():
    """
    A settled round's NMR has left it -- back to the wallet or rolled into a
    newer round, which then carries it as its own atRisk. Round 1379 opened with
    exactly 1355's 37 NMR stake plus its 0.977596 payout. Summing every round
    ever staked counted that NMR twice, and the report read 322.67 NMR staked
    against a real ~97.
    """
    rounds = [one_round(1355, 37.0, 0.977596, resolved=True),
              one_round(1378, 3.1, 0.05, resolved=False),
              one_round(1379, 37.977596, 1.0, resolved=False)]
    ctx = R.build(fixture(rounds), NO_FX, BUILT, dict(R.STAKED_SLOTS))
    text = ctx["plain_text"]
    assert " stake at risk   41.0776 NMR" in text
    assert "TOTAL BOOK VALUE  42.1276 NMR" in text
    assert "book 42.13 NMR" in R.subject(ctx)


def test_banner_distinguishes_settled_from_mark_to_market():
    """Once a round resolves the banner must stop claiming nothing has."""
    ctx = R.build(fixture([one_round(1337, 10.0, 0.50, resolved=True),
                           one_round(1338, 10.0, 0.25, resolved=False)]),
                  NO_FX, BUILT, dict(R.STAKED_SLOTS))
    assert "1 of 2 staked rounds have settled" in ctx["banner"]
    assert "Nothing here has settled" not in ctx["banner"]


def test_banner_falls_back_when_nothing_settled():
    ctx = R.build(fixture([one_round(1337, 10.0, 0.50, resolved=False)]),
                  NO_FX, BUILT, dict(R.STAKED_SLOTS))
    assert "Nothing here has settled" in ctx["banner"]


def test_section3_diamonds_are_coloured_by_settlement_not_by_model():
    """
    One scheme across every panel: red while a round is still pending, blue once
    it has resolved. The per-model palette colour must not leak into section 3,
    or the colour would read as "which model" instead of "settled or not".
    """
    panels = "".join(R.build(
        fixture([one_round(1337, 10.0, 0.50, resolved=True),
                 one_round(1338, 10.0, 0.25, resolved=False)]),
        NO_FX, BUILT, dict(R.STAKED_SLOTS))["panels"])
    assert R.PENDING_COLOUR in panels and R.RESOLVED_COLOUR in panels
    assert R.COLOUR["CURUPIRA_08"] not in panels
    assert "latest (current pending)" in panels
    assert "resolved (settled)" in panels


def test_section3_is_year_to_date_only_and_section2_keeps_full():
    """
    Section 3 is always scoped to the current calendar year; section 2 keeps the
    full history available so its toggle can switch back to it.
    """
    live = fixture([one_round(1300, 10.0, 0.40, resolved=True, opened="2025-12-15T00:00:00Z"),
                    one_round(1337, 10.0, 0.50, resolved=True, opened="2026-08-20T00:00:00Z")])
    ctx = R.build(live, NO_FX, BUILT, dict(R.STAKED_SLOTS))  # BUILT is 2026
    panels = "".join(ctx["panels"])
    assert ">1337</text>" in panels          # this year's round is plotted
    assert ">1300</text>" not in panels       # last year's round is not in section 3
    assert "1 YTD rounds" in panels           # the header counts YTD rounds only
    # section 2 still ships every round, so the full-history toggle has data
    runs = [row[0] for m in ctx["chart_data"]["models"] for row in m["rows"]]
    assert 1300 in runs and 1337 in runs
    assert ctx["chart_data"]["ytd_min"] == 1337


def test_cards_and_book_carry_nmr_eur_brl_and_resolved():
    """Every card shows EUR and BRL; the book and cards split provisional vs resolved."""
    ctx = R.build(fixture([one_round(1337, 10.0, 0.50, resolved=True),
                           one_round(1338, 10.0, 0.25, resolved=False)]),
                  (12.04, 67.25, "test"), BUILT, dict(R.STAKED_SLOTS))
    titles = [c[0] for c in ctx["cards"]]
    assert "Provisional P&amp;L" in titles and "Resolved P&amp;L" in titles
    # the provisional card now carries EUR and BRL, not EUR alone
    prov = next(c for c in ctx["cards"] if c[0] == "Provisional P&amp;L")
    assert "&euro;" in prov[2] and "R$" in prov[2]
    assert "resolved P&amp;L (settled)" in ctx["book_table"]
    assert ctx["model_table"].count("<th>BRL</th>") == 1


def test_days_since_first_stake_counts_calendar_days_not_rounds():
    """Two rounds a month apart: Rounds says 2, Days says the gap to the report date."""
    live = fixture([one_round(1300, 10.0, 0.40, resolved=True, opened="2025-12-15T00:00:00Z"),
                    one_round(1337, 10.0, 0.50, resolved=True, opened="2026-08-20T00:00:00Z")])
    ctx = R.build(live, NO_FX, BUILT, dict(R.STAKED_SLOTS))  # report day 2026-09-09
    assert "Days since 1st stake" in ctx["returns_table"]
    assert ">20</span>" in ctx["returns_table"]       # YTD: 2026-08-20 -> 09-09
    assert ">268</span>" in ctx["returns_table"]      # inception: 2025-12-15 -> 2026-09-09
    assert "268 days since first stake 2025-12-15" in ctx["plain_text"]


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


def test_return_boxplots_divide_by_each_rounds_own_stake():
    """Section 3b: a 0.5 NMR payout on 10 NMR staked reads +5.00%, not +0.5."""
    ctx = R.build(fixture([one_round(1337, 10.0, 0.50, resolved=True)]),
                  NO_FX, BUILT, dict(R.STAKED_SLOTS))
    panels = "".join(ctx["ret_panels"])
    assert "latest +5.00%" in panels
    assert "mean round return" in panels and "+5.00%" in panels
