"""
Tests for the live submission pipeline.

    python3 -m pytest src/tests/test_live.py -q

This code moves real money on a deadline, so the tests concentrate on the three
failure modes that would cost a round rather than on happy-path coverage:

  * a submission file that Numerai would reject (NaN, duplicates, out of range)
  * a fallback ladder that does not actually catch a fall
  * a refit cadence or embargo that quietly stops matching the backtest

The heavy fixtures (a fitted estimator, the real panel) are avoided on purpose:
these run in about a second and can sit in a pre-submission hook.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.live import registry, submit
from src.live.features import (ONEHOT_MIN_DATES, ROLL_WINDOWS, TENC_LAG,
                               add_derived,
                               add_onehot, add_rolling, add_tenc, build_features)
from src.live.refit import feature_signature, refit_due, training_block

SYMBOL, PERIOD, TARGET = "symbol", "date", "target_binned_return_20"


def make_panel(n_symbols=6, n_periods=60, n_live=1, seed=0):
    """A small synthetic panel with the same shape as the real one."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_periods + n_live, freq="D")
    rows = []
    for s in range(n_symbols):
        for i, date in enumerate(dates):
            live = i >= n_periods
            row = {SYMBOL: "SYM%d" % s, PERIOD: date, "is_live": live,
                   TARGET: np.nan if live else rng.choice([0, .25, .5, .75, 1.0])}
            for feature in ("feature_sharpe_ratio", "feature_relative_strength_index",
                            "feature_volatility", "feature_momentum"):
                for horizon in ("20d", "60d"):
                    row["%s_%s" % (feature, horizon)] = rng.choice([0, .25, .5, .75, 1.0])
            rows.append(row)
    return pd.DataFrame(rows).sort_values([SYMBOL, PERIOD]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Submission file validity -- the guards that stop a bad file leaving the box
# --------------------------------------------------------------------------

def test_rank_normalise_is_strictly_inside_the_unit_interval():
    values = np.array([-5.0, 0.0, 0.0, 0.0, 12.0])
    out = submit.rank_normalise(values)
    assert (out > 0).all() and (out < 1).all()
    # method="first" keeps ties as a strict permutation rather than collapsing
    # them to one rank, which would throw away ordering the model did express.
    assert len(set(out)) == len(values)


def test_rank_normalise_preserves_order():
    values = np.array([3.0, 1.0, 2.0])
    out = submit.rank_normalise(values)
    assert list(np.argsort(out)) == list(np.argsort(values))


@pytest.mark.parametrize("frame,reason", [
    (pd.DataFrame({SYMBOL: ["A", "A"], "prediction": [0.1, 0.2]}), "duplicate"),
    (pd.DataFrame({SYMBOL: ["A"], "prediction": [np.nan]}), "NaN"),
    (pd.DataFrame({SYMBOL: ["A"], "prediction": [1.5]}), "outside"),
    (pd.DataFrame({SYMBOL: ["A"], "prediction": [0.0]}), "outside"),
])
def test_write_csv_rejects_invalid_submissions(frame, reason, tmp_path):
    original = submit.SUBMISSION_DIR
    submit.SUBMISSION_DIR = str(tmp_path)
    try:
        with pytest.raises(ValueError, match=reason):
            submit.write_csv(frame, 1, "T")
    finally:
        submit.SUBMISSION_DIR = original


# --------------------------------------------------------------------------
# Fallback ladder
# --------------------------------------------------------------------------

class ConstantEstimator(object):
    def predict(self, X):
        return np.arange(len(X), dtype="float64")


def test_fallback_ladder_descends_one_rung_at_a_time(tmp_path):
    original = submit.SUBMISSION_DIR
    submit.SUBMISSION_DIR = str(tmp_path)
    try:
        panel = make_panel()
        live_date = panel[PERIOD].max()
        columns = [c for c in panel.columns if c.startswith("feature_")]
        symbols = panel.loc[panel["is_live"], SYMBOL].tolist()

        def context(estimators):
            return {"estimators": estimators, "live_date": live_date,
                    "round_number": 10, "live_symbols": symbols}

        entry = (ConstantEstimator(), columns, panel, {})

        # 1. own model
        _, source, _ = submit.build_predictions("CURUPIRA_03", context(
            {"CURUPIRA_03": entry}))
        assert source == "model"

        # 2. own model absent -> the raw-feature sibling stands in
        _, source, note = submit.build_predictions("CURUPIRA_03", context(
            {"CURUPIRA_02": entry}))
        assert source == "sibling" and "CURUPIRA_02" in note

        # 3. nothing fitted -> last round's file for this same slot
        previous = os.path.join(str(tmp_path), "9")
        os.makedirs(previous, exist_ok=True)
        pd.DataFrame({SYMBOL: symbols,
                      "prediction": np.linspace(0.1, 0.9, len(symbols))}
                     ).to_csv(os.path.join(previous, "CURUPIRA_03.csv"), index=False)
        _, source, _ = submit.build_predictions("CURUPIRA_03", context({}))
        assert source == "previous_round"

        # 4. no history at all -> Numerai's example predictions
        shutil.rmtree(previous)
        example = pd.DataFrame({SYMBOL: symbols, "prediction": 0.5})
        example_path = os.path.join(str(tmp_path), "example.csv")
        example.to_csv(example_path, index=False)
        original_example = submit.EXAMPLE_PREDS
        submit.EXAMPLE_PREDS = example_path
        try:
            _, source, _ = submit.build_predictions("CURUPIRA_03", context({}))
            assert source == "example"
        finally:
            submit.EXAMPLE_PREDS = original_example
    finally:
        submit.SUBMISSION_DIR = original


def test_previous_submission_never_reaches_forward_in_time(tmp_path):
    original = submit.SUBMISSION_DIR
    submit.SUBMISSION_DIR = str(tmp_path)
    try:
        for round_number in (8, 12):
            directory = os.path.join(str(tmp_path), str(round_number))
            os.makedirs(directory)
            pd.DataFrame({SYMBOL: ["A"], "prediction": [0.5]}).to_csv(
                os.path.join(directory, "CURUPIRA_01.csv"), index=False)
        found = submit.previous_submission("CURUPIRA_01", 10)
        assert found.endswith(os.path.join("8", "CURUPIRA_01.csv"))
    finally:
        submit.SUBMISSION_DIR = original


def test_upload_retries_then_succeeds(monkeypatch, tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("symbol,prediction\nA,0.5\n")
    monkeypatch.setattr(submit.time, "sleep", lambda _: None)

    calls = {"n": 0}

    class Flaky(object):
        def upload_predictions(self, file_path, model_id):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("502 Bad Gateway")
            return "sub-id"

    submission_id, attempts = submit.upload(Flaky(), str(path), "mid", "T")
    assert submission_id == "sub-id" and attempts == 3


def test_upload_gives_up_after_the_last_attempt(monkeypatch, tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("symbol,prediction\nA,0.5\n")
    monkeypatch.setattr(submit.time, "sleep", lambda _: None)

    class Broken(object):
        def upload_predictions(self, file_path, model_id):
            raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="after %d attempts" % submit.MAX_UPLOAD_ATTEMPTS):
        submit.upload(Broken(), str(path), "mid", "T")


def test_dry_run_never_touches_the_api():
    class Exploding(object):
        def upload_predictions(self, *a, **k):
            raise AssertionError("dry run must not upload")

    assert submit.upload(Exploding(), "x", "mid", "T", dry_run=True) == ("DRY_RUN", 0)


# --------------------------------------------------------------------------
# Fidelity to the backtest protocol
# --------------------------------------------------------------------------

def test_registry_protocol_matches_the_backtest_config():
    fit = registry.FIT
    assert fit["train_window"] == 130 and fit["refit_every"] == 130
    assert fit["embargo"] == 20, "embargo must equal the 20-day target horizon"
    assert fit["cv_scheme"] == "rolling" and fit["n_splits"] == 5
    assert fit["n_iter"] == 50 and fit["seed"] == 42


def test_every_slot_declares_a_backtest_run_and_stats():
    for spec in registry.all_specs():
        assert spec["backtest_run"], spec["name"]
        assert spec["backtest"]["n_periods"] == 510
        assert spec["why"]


def test_submission_order_is_cheapest_first():
    """
    The submission window is ~66 minutes, so slots run in ascending FIT COST:
    if the deadline bites it must bite the most expensive slot, not the core.

    Cost, not feature count. Those tracked each other until CURUPIRA_05, a blend
    whose 763 union columns are cheaper to store but five times more expensive to
    fit than CURUPIRA_04's 812 -- five searches instead of one. Testing the proxy
    rather than the thing would have forced the wrong ordering.
    """
    from src.live.run_round import DEFAULT_FIT_SECONDS

    missing = [n for n in registry.ORDER if n not in DEFAULT_FIT_SECONDS]
    assert not missing, "no fit-cost estimate for %s" % missing

    # Slot blends are exempt from cheapest-first: they fit nothing, but they
    # load their members' caches, so a DEPENDENCY constraint overrides the cost
    # one. They are checked separately below.
    fitting = [n for n in registry.ORDER
               if not registry.is_slot_blend(registry.get(n))]
    costs = [DEFAULT_FIT_SECONDS[n] for n in fitting]
    assert costs == sorted(costs), (
        "fitting slots must run cheapest-first, got %s" % list(zip(fitting, costs)))
    assert registry.ORDER[0] == "CURUPIRA_02", "the no-build slot must go first"
    assert set(registry.ORDER) == set(registry.MODELS)


def test_slot_blends_run_after_every_slot_they_reuse():
    """
    A slot blend loads its members' fitted caches, so it must come later in
    ORDER than all of them -- otherwise it would assemble from a stale or
    missing fit on the very round its member was refreshed.
    """
    position = {name: i for i, name in enumerate(registry.ORDER)}
    for name in registry.ORDER:
        spec = registry.get(name)
        if not registry.is_slot_blend(spec):
            continue
        for member in spec["slot_members"]:
            assert member["name"] in position, (
                "%s reuses unknown slot %s" % (name, member["name"]))
            assert position[member["name"]] < position[name], (
                "%s must run after its member %s" % (name, member["name"]))
        assert abs(sum(m["weight"] for m in spec["slot_members"]) - 1.0) < 1e-9
        # It must not also declare its own fittable members -- the two
        # composition types need opposite refit handling.
        assert not registry.is_blend(spec)


def test_training_block_refuses_to_train_inside_the_embargo():
    panel = make_panel(n_periods=40)
    columns = [c for c in panel.columns if c.startswith("feature_")]
    live_date = panel[PERIOD].max()
    # Only 1 day between the last training period and the live date, against a
    # 20-day horizon: those training rows' targets overlap the prediction date.
    with pytest.raises(ValueError, match="overlap"):
        training_block(panel, columns, train_window=130, embargo=20,
                       live_date=live_date)


def test_training_block_takes_exactly_the_last_window_periods():
    panel = make_panel(n_periods=60)
    columns = [c for c in panel.columns if c.startswith("feature_")]
    live_date = panel[PERIOD].max() + pd.Timedelta(days=40)
    X, y, meta, periods, gap = training_block(panel, columns, 30, 20, live_date)
    assert len(periods) == 30
    assert meta[PERIOD].nunique() == 30
    assert X.index.equals(y.index), "the scorer aligns through this index"
    assert not panel.loc[X.index, "is_live"].any(), "live rows must never train"


def test_refit_cadence():
    live = pd.Timestamp("2026-08-12")
    assert refit_due(None, live, 130, "sig")[0] is True
    fresh = {"fitted_through": "2026-07-08", "feature_signature": "sig"}
    assert refit_due(fresh, live, 130, "sig")[0] is False
    stale = {"fitted_through": "2025-01-01", "feature_signature": "sig"}
    assert refit_due(stale, live, 130, "sig")[0] is True
    # A changed one-hot vocabulary invalidates the cache regardless of the clock.
    due, reason = refit_due(fresh, live, 130, "different")
    assert due is True and "feature set changed" in reason


def test_feature_signature_tracks_column_identity_and_order():
    assert feature_signature(["a", "b"]) == feature_signature(["a", "b"])
    assert feature_signature(["a", "b"]) != feature_signature(["b", "a"])
    assert feature_signature(["a", "b"]) != feature_signature(["a", "b", "c"])


# --------------------------------------------------------------------------
# Live feature construction
# --------------------------------------------------------------------------

def test_tenc_never_sees_a_live_row_target():
    panel = make_panel(n_periods=60, n_live=3)
    encoded, added = add_tenc(panel)
    assert added == ["feature_sym_tenc"]
    assert np.isfinite(encoded["feature_sym_tenc"]).all()
    # The lag means the live rows' support stops 20 periods short of the end.
    live_support = encoded.loc[encoded["is_live"], "_tenc_support"]
    assert (live_support <= 60 - TENC_LAG + 3).all()


def test_tenc_lag_equals_the_target_horizon():
    assert TENC_LAG == registry.FIT["embargo"] == 20


def test_onehot_vocabulary_comes_from_training_rows_only():
    panel = make_panel(n_symbols=4, n_periods=ONEHOT_MIN_DATES + 5, n_live=1)
    # A token that only ever appears live must not earn a column -- live, you
    # cannot have fit a coefficient for something you have never trained on.
    newcomer = panel[panel["is_live"]].head(1).copy()
    newcomer[SYMBOL] = "NEWCOIN"
    panel = pd.concat([panel, newcomer], ignore_index=True)

    encoded, columns, cold, vocabulary = add_onehot(panel, ~panel["is_live"])
    assert "feature_sym_NEWCOIN" not in columns
    assert vocabulary == 4 and cold == 1
    assert encoded.loc[encoded[SYMBOL] == "NEWCOIN", columns].to_numpy().sum() == 0


def test_onehot_respects_the_minimum_history_threshold():
    panel = make_panel(n_symbols=3, n_periods=ONEHOT_MIN_DATES - 1, n_live=1)
    _, columns, _, vocabulary = add_onehot(panel, ~panel["is_live"])
    assert vocabulary == 0 and columns == []


def test_rolling_adds_one_column_per_feature_and_window():
    panel = make_panel(n_periods=30)
    rolled, added = add_rolling(panel)
    assert len(added) == 8 * len(ROLL_WINDOWS)      # 4 indicators x 2 horizons
    assert not rolled[added].isna().any().any(), "min_periods=1 leaves no NaN"


def test_rolling_never_crosses_symbols():
    panel = make_panel(n_symbols=2, n_periods=10)
    rolled, added = add_rolling(panel)
    column = "feature_momentum_20d"
    for symbol in panel[SYMBOL].unique():
        block = rolled[rolled[SYMBOL] == symbol].sort_values(PERIOD)
        expected = block[column].rolling(5, min_periods=1).mean().to_numpy()
        np.testing.assert_allclose(block[column + "_rm5"].to_numpy(), expected, rtol=1e-5)


def test_build_features_leaves_no_missing_values():
    panel = make_panel()
    panel.loc[panel.index[:5], "feature_momentum_20d"] = np.nan
    built, columns, diagnostics = build_features(panel, "tenc", verbose=False)
    assert not built[columns].isna().any().any()
    assert diagnostics["n_features"] == len(columns)


def test_allfeat_pp_is_the_union_of_every_other_slot_plus_derived():
    """
    CURUPIRA_04's live feature set must be a strict superset of the others.

    This is the property that makes it "everything at once", and it is easy to
    break: the offline `allfeat` dataset (788) looked like a superset because it
    had MORE columns than CURUPIRA_03 (786), while actually lacking all 24
    rolling means. Counting columns does not establish containment.
    """
    panel = make_panel(n_symbols=3, n_periods=ONEHOT_MIN_DATES + 25, n_live=1)
    sets = {}
    for name in ("raw", "tenc", "onehot_roll", "allfeat_pp"):
        _, columns, _ = build_features(panel.copy(), name, verbose=False)
        sets[name] = set(columns)

    for name in ("raw", "tenc", "onehot_roll"):
        missing = sets[name] - sets["allfeat_pp"]
        assert not missing, "allfeat_pp lacks %d columns of %s" % (len(missing), name)

    # And it adds exactly the derived block on top of that union -- no more, no
    # less. The expectation is derived from add_derived on the same panel rather
    # than hardcoded, because the synthetic fixture carries 4 indicators where
    # the real data carries 11, so the absolute count is a property of the
    # fixture (8 here, 25 live) while the containment is the real invariant.
    union = sets["raw"] | sets["tenc"] | sets["onehot_roll"]
    _, derived = add_derived(panel.copy())
    assert sets["allfeat_pp"] - union == set(derived)
    assert all(c.endswith("_spread") or "_x_" in c or "_accel_" in c
               for c in derived)


def test_derived_columns_need_no_history():
    """
    Spreads and interactions must be same-row transforms.

    They are what lets CURUPIRA_04 avoid the rolling block's dependence on
    contiguous history, so a row's derived values must not change when its
    neighbours are removed.
    """
    panel = make_panel(n_symbols=4, n_periods=40, n_live=1)
    full, columns = add_derived(panel.copy())
    derived = [c for c in columns]

    # Recompute on a shuffled, decimated panel: same rows, no neighbours.
    subset = panel.iloc[::3].copy().sample(frac=1.0, random_state=0)
    partial, _ = add_derived(subset)

    common = full.loc[partial.index, derived]
    np.testing.assert_allclose(
        common.to_numpy(), partial[derived].to_numpy(), rtol=1e-6,
        err_msg="a derived column changed when neighbouring rows were dropped")


# --------------------------------------------------------------------------
# Ablation report sign convention
# --------------------------------------------------------------------------

def test_ablation_importance_sign_is_not_inverted(tmp_path, monkeypatch):
    """
    Removing a block that helps must report POSITIVE importance and a positive
    t, and be labelled as contributing.

    This is the single easiest thing in the analysis to get backwards, and it
    was: the table printed the importance's t while the verdict tested the
    delta's, so the one block that actually contributed was labelled "drop it".
    A sign error here does not crash or look wrong -- it just reverses every
    conclusion.
    """
    from src.analysis import ablation_report as ar

    dates = pd.date_range("2025-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, len(dates))

    def write(directory, name, corr):
        os.makedirs(directory, exist_ok=True)
        pd.DataFrame({"date": dates, "corr": corr, "bmc": 0.0,
                      "mmc": 0.0, "pnl": 0.0}).to_csv(
            os.path.join(directory, name + ".csv"), index=False)

    metrics_dir = str(tmp_path / "metrics")
    per_dir = str(tmp_path / "per_period")
    os.makedirs(metrics_dir, exist_ok=True)

    # Reference scores higher than the arm -> the removed block was helping.
    write(per_dir, "REF", 0.10 + noise)
    write(per_dir, "w6r6_lobo_useful", 0.05 + noise)
    for name in ("REF", "w6r6_lobo_useful"):
        with open(os.path.join(metrics_dir, name + ".json"), "w") as handle:
            handle.write('{"config": {"n_features": 10}, "n_periods": 200}')

    monkeypatch.setattr(ar, "METRICS", metrics_dir)
    monkeypatch.setattr(ar, "PER_PERIOD", per_dir)
    monkeypatch.setitem(ar.FAMILIES, "lobo",
                        {"prefix": "w6r6_lobo_", "reference": "REF"})

    rows = ar.report("lobo", verbose=False)
    row = rows[0]
    assert row["importance"] > 0, "removing a helpful block must cost, not gain"
    assert row["t"] > 0, "the importance t must share the importance's sign"
    assert row["t"] > 2, "a 0.05 gap on 200 periods should be unambiguous"


# --------------------------------------------------------------------------
# Blend slots
# --------------------------------------------------------------------------

def test_rank_blend_weights_are_renormalised_and_order_preserving():
    from src.live.blend import RankBlend

    class Const(object):
        def __init__(self, v): self.v = np.asarray(v, dtype="float64")
        def predict(self, X): return self.v[:len(X)]

    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]})
    # Weights that do not sum to 1 must be renormalised, not taken literally.
    blend = RankBlend([("m1", Const([1, 2, 3]), ["a"], 2.0),
                       ("m2", Const([1, 2, 3]), ["b"], 2.0)])
    out = blend.predict(X)
    assert np.all(np.diff(out) > 0), "two agreeing members must preserve order"
    assert out.max() <= 1.0 + 1e-9

    # A member pulling the other way must move the blend toward a tie.
    opposed = RankBlend([("m1", Const([1, 2, 3]), ["a"], 0.5),
                         ("m2", Const([3, 2, 1]), ["b"], 0.5)])
    assert np.allclose(opposed.predict(X), opposed.predict(X)[0])


def test_rank_blend_rejects_a_member_missing_its_columns():
    from src.live.blend import RankBlend

    class Const(object):
        def predict(self, X): return np.arange(len(X), dtype="float64")

    blend = RankBlend([("m", Const(), ["absent"], 1.0)])
    with pytest.raises(ValueError, match="missing"):
        blend.predict(pd.DataFrame({"present": [1.0, 2.0]}))


def test_blend_member_patterns_select_the_widths_the_backtest_fitted():
    """
    Each member's include pattern must pick exactly the column count its
    backtest run used.

    The two raw groups are the trap: they were fitted on the 22-column dataset,
    where '^feature_market_cap_(avg|ewa)_' matches 4 columns. Against the
    812-column live build the same pattern also catches
    feature_market_cap_avg_spread, giving 6. The registry anchors them for that
    reason, and this test is what stops the anchor being "tidied" away.
    """
    from src.prediction.functions import filter_features

    spec = registry.get("CURUPIRA_05")
    assert registry.is_blend(spec)

    # Stand-in for the real 812-column set: the raw pairs plus the derived
    # columns whose names would collide with an unanchored pattern.
    columns = ([f"feature_{n}_{h}" for n in ("market_cap_avg", "market_cap_ewa",
                                             "close_avg", "close_ewa")
                for h in ("20d", "60d")]
               + ["feature_market_cap_avg_spread", "feature_close_avg_spread",
                  "feature_sym_tenc", "feature_sym_BTC", "feature_sym_ETH",
                  "feature_momentum_x_volatility_20d", "feature_close_accel_20d"])

    widths = {m["name"]: len(filter_features(columns, m["include"], None, verbose=False))
              for m in spec["members"]}
    assert widths["size"] == 4, "size caught the spread column: %d" % widths["size"]
    assert widths["price"] == 4, "price caught the spread column: %d" % widths["price"]
    assert widths["tenc"] == 1
    assert widths["onehot"] == 2, "one-hot must exclude feature_sym_tenc"
    assert widths["interact"] == 2

    assert abs(sum(m["weight"] for m in spec["members"]) - 1.0) < 1e-9
