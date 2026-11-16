"""Adversarial tests for the strict volume-alpha information boundary."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data_mani.volumealpha_sources import (consolidate, identity_at, snapshot,
        validate_identities, validate_market, MARKET_KEYS)
from src.data_mani.volumealpha_features import (FAMILIES, PREFIX, causal_winsorize,
                                               features_at, dimson_beta)
from src.data_mani.build_volumealpha_causal import extend_panel, read_bundle
from src.prediction.volumealpha import (route_slots, volume_targets, walk_forward_volume,
                                       trading_costs)


def market_rows(n=300, assets=("bitcoin:native", "ethereum:native")):
    dates = pd.date_range("2020-01-01", periods=n, tz="UTC")
    rows = []
    for k, asset in enumerate(assets):
        for i, day in enumerate(dates):
            rows.append(dict(asset_id=asset, venue="binance", pair=f"{k}USDT", date=day,
                observed_at=day + pd.Timedelta(hours=23, minutes=59),
                available_at=day + pd.Timedelta(days=1), retrieved_at=pd.Timestamp("2024-01-01", tz="UTC"),
                evidence="archive:first-publication-receipt", vintage_verified=True,
                source_family="cex", market_type="spot", base_volume=10 + k + i % 7,
                price_quote=100 + k * 3 + i / 3, quote="USDT", quote_usd=0.98,
                quote_fx_at=day + pd.Timedelta(hours=23), quote_fx_available_at=day + pd.Timedelta(days=1),
                quote_fx_evidence="independent-timestamped-usdt-usd-feed"))
    return pd.DataFrame(rows)


def identities():
    return pd.DataFrame([dict(ucid=str(i + 1), asset_id=a, chain=a.split(":")[0],
        contract_address="native", valid_from="2019-01-01Z".replace("01Z", "01T00:00Z"),
        valid_to="2030-01-01T00:00Z", available_at="2019-01-01T00:00Z",
        retrieved_at="2024-01-01T00:00Z", evidence="dated-registry-proof", vintage_verified=True)
        for i, a in enumerate(("bitcoin:native", "ethereum:native"))])


def test_source_requires_provenance_and_identity_not_ticker():
    d = market_rows(3).drop(columns="available_at")
    with pytest.raises(ValueError, match="missing"):
        validate_market(d, ["binance"])
    d = identities()
    d.loc[0, "asset_id"] = "BTC"
    with pytest.raises(ValueError, match="chain:contract"):
        validate_identities(d)


def test_identity_known_then_and_valid_then():
    d = validate_identities(identities())
    d.loc[0, "available_at"] = pd.Timestamp("2021-01-01", tz="UTC")
    assert "1" not in set(identity_at(d, "2020-01-01")["ucid"])
    assert "1" in set(identity_at(d, "2022-01-01")["ucid"])


def test_duplicate_pair_rejected_and_multiple_pairs_count_one_venue():
    d = market_rows(2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_market(pd.concat([d, d.iloc[[0]]]), ["binance"])
    second = d.copy().assign(pair=lambda f: f["pair"].str.replace("USDT", "USDC"), quote="USDC")
    x = consolidate(validate_market(pd.concat([d, second]), ["binance"]), "cex")
    assert x["venue_count"].eq(1).all()
    assert x["venue_hhi"].eq(1).all()
    assert x.iloc[0]["dollar_volume"] == pytest.approx(2 * 10 * 100 * 0.98)


def test_untrusted_venue_perp_and_noncontemporaneous_fx_rejected():
    d = market_rows(2)
    with pytest.raises(ValueError, match="allowlist"):
        validate_market(d, ["coinbase"])
    with pytest.raises(ValueError, match="not spot"):
        validate_market(d.assign(market_type="perp"), ["binance"])
    with pytest.raises(ValueError, match="not contemporaneous"):
        validate_market(d.assign(quote_fx_at=pd.Timestamp("2018-01-01", tz="UTC")), ["binance"])


def test_fx_release_and_revisions_respected():
    d = validate_market(market_rows(3), ["binance"])
    revision = d.iloc[[0]].copy().assign(base_volume=999,
                         available_at=pd.Timestamp("2020-02-01", tz="UTC"))
    both = pd.concat([d, revision])
    early = snapshot(both, "2020-01-04", MARKET_KEYS)
    late = snapshot(both, "2020-02-02", MARKET_KEYS)
    assert early["base_volume"].max() < 999
    assert late["base_volume"].max() == 999
    d["quote_fx_available_at"] = pd.Timestamp("2022-01-01", tz="UTC")
    assert snapshot(d, "2020-01-04", MARKET_KEYS).empty


def test_retrospective_rows_never_admitted_by_economic_date():
    d = validate_market(market_rows(3), ["binance"]).assign(vintage_verified=False)
    assert snapshot(d, "2023-01-01", MARKET_KEYS).empty


def test_source_families_not_silently_mixed():
    d = validate_market(market_rows(2), ["binance"])
    dex = d.assign(venue="dex", source_family="dex", base_volume=1)
    assert consolidate(pd.concat([d, dex]), "cex")["dollar_volume"].sum() == consolidate(d)["dollar_volume"].sum()
    assert consolidate(pd.concat([d, dex]), "combined")["dollar_volume"].sum() > consolidate(d)["dollar_volume"].sum()


def test_technical_eight_calendar_windows_and_d_minus_one():
    d = consolidate(validate_market(market_rows(), ["binance"]))
    D = pd.Timestamp("2020-10-15", tz="UTC")
    assets = ["bitcoin:native", "ethereum:native"]
    x = features_at(d, D, assets)
    assert len(FAMILIES["technical"]) == 8
    past = d[(d["asset_id"] == assets[0]) & (d["date"] < D)]
    assert x.loc[assets[0], PREFIX + "logdv_ma5"] == pytest.approx(np.log1p(past["dollar_volume"].iloc[-5:]).mean())
    future = d["date"] >= D
    changed = d.copy()
    changed.loc[future, ["close", "dollar_volume"]] = 1e12
    assert_frame_equal(x, features_at(changed, D, assets))


def test_missing_calendar_day_is_not_compressed_or_backfilled():
    d = consolidate(validate_market(market_rows(300), ["binance"]))
    D = pd.Timestamp("2020-10-15", tz="UTC")
    d = d[d["date"] != D - pd.Timedelta(days=2)]
    x = features_at(d, D, ["bitcoin:native"])
    assert np.isnan(x.iloc[0][PREFIX + "logdv_ma5"])


def test_unannounced_future_schedule_and_late_revision_hidden():
    d = consolidate(validate_market(market_rows(300), ["binance"]))
    D = pd.Timestamp("2020-10-15", tz="UTC")
    e = pd.DataFrame([dict(asset_id="bitcoin:native", kind="unlock", scheduled_for=D,
                          announced_at=D + pd.Timedelta(days=1), available_at=D + pd.Timedelta(days=1), vintage_verified=True)])
    x = features_at(d, D, ["bitcoin:native"], events=e)
    assert pd.isna(x.iloc[0][PREFIX + "distance_unlock"])
    e[["announced_at", "available_at"]] = D - pd.Timedelta(days=2)
    x = features_at(d, D, ["bitcoin:native"], events=e)
    assert x.iloc[0][PREFIX + "distance_unlock"] == 0


def test_dimson_beta_uses_only_current_and_lagged_market():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(size=150))
    y = 2 * m + 3 * m.shift(1) - m.shift(2)
    assert dimson_beta(y, m) == pytest.approx(4)


def test_winsorization_uses_only_previous_dates_not_current_peers():
    f = pd.DataFrame({"date": np.repeat(pd.date_range("2020-01-01", periods=4), 3),
                      "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1e6, 1e7, 1e8]})
    x = causal_winsorize(f, ["x"], minimum=3)
    short = causal_winsorize(f.iloc[:9], ["x"], minimum=3)
    assert_frame_equal(x.iloc[:9], short)
    assert x["x"].iloc[-1] < 10
    changed = f.copy()
    changed.loc[4, "x"] = 1e9
    xx = causal_winsorize(changed, ["x"], minimum=3)
    assert xx.loc[3, "x"] == x.loc[3, "x"]


def test_base_frame_is_exact_and_ties_and_uncovered_rows_preserved():
    dates = pd.to_datetime(["2020-01-01"] * 3)
    base = pd.DataFrame({"symbol": ["A", "B", "C"], "ucid": [1, 2, 3], "date": dates,
                         "feature_base": np.array([0, 0.5, 1], dtype="float32"), "target": [0.2, np.nan, 0.9]}).set_index("symbol")
    keys = pd.DataFrame({"ucid": ["1", "2"], "date": pd.to_datetime(dates[:2], utc=True), "asset_id": ["a", "b"]})
    raw = keys[["asset_id", "date"]].assign(feature_va_x=[1.0, 1.0])
    out, cov, _ = extend_panel(base, raw, keys, {"x": ["feature_va_x"]})
    assert_frame_equal(out[base.columns], base)
    assert out.loc["A", "feature_va_x"] == out.loc["B", "feature_va_x"]
    assert pd.isna(out.loc["C", "feature_va_x"])
    assert not any(c.startswith("cov_") for c in out)
    assert cov["cov_x"].tolist() == [True, True, False]


def test_routing_preserves_uncovered_ranks_and_all_rank_slots():
    b = [1, 2, 2, 4, 5]
    c = [9, 8, np.nan, 7, 6]
    mask = np.array([True, True, False, False, True])
    routed, ranks = route_slots(b, c, mask)
    assert np.array_equal(routed[~mask], ranks[~mask])
    assert np.array_equal(np.sort(routed), np.sort(ranks))
    assert routed[0] > routed[1] > routed[4]
    # Even re-ranking the output preserves each uncovered baseline tie.
    assert np.array_equal(pd.Series(routed).rank().to_numpy()[~mask], pd.Series(b).rank().to_numpy()[~mask])


def test_volume_targets_are_direct_nonoverlapping_daily_sums():
    d = consolidate(validate_market(market_rows(30), ["binance"]))
    t = volume_targets(d)
    g = d[d["asset_id"] == "bitcoin:native"].sort_values("date")
    got = t[(t["asset_id"] == "bitcoin:native") & (t["date"] == g.iloc[10]["date"])].iloc[0]
    assert got["logdv_5d"] == pytest.approx(np.log1p(g["dollar_volume"].iloc[10:15].sum()))
    assert got["ready_5d"] >= g["available_at"].iloc[14]


def test_target_publication_preserves_nanoseconds_at_cutoff():
    d = consolidate(validate_market(market_rows(30), ["binance"]))
    cutoff = pd.Timestamp("2020-01-12", tz="UTC")
    d.loc[(d["asset_id"] == "bitcoin:native") & (d["date"] == pd.Timestamp("2020-01-10", tz="UTC")),
          "available_at"] = cutoff + pd.Timedelta(nanoseconds=1)
    target = volume_targets(d)
    row = target[(target["asset_id"] == "bitcoin:native") &
                 (target["date"] == pd.Timestamp("2020-01-11", tz="UTC"))].iloc[0]
    assert row["ready_1d"] == cutoff + pd.Timedelta(nanoseconds=1)


def auxiliary_fixture():
    d = consolidate(validate_market(market_rows(60), ["binance"]))
    targets = volume_targets(d)
    raw = targets[["asset_id", "date"]].copy()
    # A deliberately simple causal auxiliary predictor, not fitted values.
    raw[PREFIX + "logdv_ma5"] = targets["baseline"]
    return raw, targets


def test_auxiliary_oof_labels_resolve_before_fit_and_future_changes_do_not_propagate():
    raw, targets = auxiliary_fixture()
    kwargs = dict(feature_columns=[PREFIX + "logdv_ma5"], refit_every=10, min_dates=10,
                  min_rows=15, iterations=3)
    out, predictions, logs = walk_forward_volume(raw, targets, **kwargs)
    assert len(predictions) and len(logs)
    assert (pd.to_datetime(predictions["max_train_date"], utc=True) < predictions["fit_at"]).all()
    assert (pd.to_datetime(predictions["max_label_available_at"], utc=True) <= predictions["fit_at"]).all()
    cutoff = pd.Timestamp("2020-02-10", tz="UTC")
    modified = targets.copy()
    modified.loc[modified["date"] >= cutoff, ["eta", "logdv_1d", "logdv_5d", "logdv_challenge"]] = 1e9
    second, _, _ = walk_forward_volume(raw, modified, **kwargs)
    assert_frame_equal(out[out["date"] < cutoff], second[second["date"] < cutoff])
    assert out.loc[out["date"] < pd.Timestamp("2020-01-21", tz="UTC"), PREFIX + "hat_logdv_1d"].isna().all()


def test_cost_scenarios_are_separate_and_missing_mean_never_fabricated():
    raw, targets = auxiliary_fixture()
    forecasts, _, _ = walk_forward_volume(raw, targets, [PREFIX + "logdv_ma5"],
                         refit_every=10, min_dates=10, min_rows=15, iterations=2)
    raw[PREFIX + "market_cap"] = 1e7
    raw[PREFIX + "return_ma1"] = 0.02
    cost = trading_costs(forecasts, raw, (1000, 10000))
    a, b = cost[PREFIX + "scenario_1000_participation"], cost[PREFIX + "scenario_10000_participation"]
    np.testing.assert_allclose(b, 10 * a, equal_nan=True)
    assert cost.loc[forecasts[PREFIX + "hat_logdv_1d"].isna(), PREFIX + "cost_inverse_dv"].isna().all()


def test_future_availability_does_not_enter_candidate_panel():
    # The bar says 2020 but no authentic vintage is certified. Its publication
    # date must not be manufactured by shifting the economic date one day.
    d = validate_market(market_rows(3), ["binance"])
    d["available_at"] = pd.Timestamp("2023-01-01", tz="UTC")
    assert snapshot(d, "2020-01-05", MARKET_KEYS).empty


def test_builder_end_to_end_verified_fixture_preserves_parquet(tmp_path):
    import json
    from src.data_mani.build_volumealpha_causal import build
    bundle = tmp_path / "inputs"
    bundle.mkdir()
    (bundle / "sources.json").write_text(json.dumps({"trusted_venues": ["binance"], "fixture": True}))
    identities().to_parquet(bundle / "identities.parquet", index=False)
    market_rows(330).to_parquet(bundle / "market.parquet", index=False)
    source = tmp_path / "base"
    source.mkdir()
    days = pd.date_range("2020-10-01", periods=30)
    base = pd.DataFrame({"symbol": np.tile(["A", "B"], len(days)), "ucid": np.tile([1, 2], len(days)),
                         "date": np.repeat(days, 2), "feature_base": np.tile([0.25, 0.75], len(days)),
                         "target_binned_return_20": np.tile([0, 1.0], len(days))}).set_index("symbol")
    base.to_parquet(source / "train.parquet")
    pd.DataFrame({"symbol": [], "date": [], "prediction": []}).to_parquet(source / "historical_meta_models.parquet")
    args = SimpleNamespace(inputs=bundle, base=source / "train.parquet", output=tmp_path / "out",
            source_family="cex", challenge_horizon=20, volume_refit=10, volume_min_dates=10,
            volume_min_rows=10, seed=42, volume_iterations=2, trade_sizes=[1000, 10000])
    build(args)
    output = pd.read_parquet(args.output / "train.parquet")
    assert_frame_equal(output[base.columns], base)
    assert output[FAMILIES["technical"]].notna().all().all()
    assert any(output[PREFIX + "hat_logdv_1d"].notna())
    assert not any(c.startswith("cov_") for c in output)
    assert (args.output / "volume_oof_metrics.csv").exists()
    with pytest.raises(ValueError, match="no overwrite"):
        build(args)


def test_full_return_evaluation_routes_every_uncovered_coin(tmp_path):
    import json
    from src.analysis.volumealpha_evaluation import run, TEST_START
    rng = np.random.default_rng(6)
    days = pd.date_range("2020-01-01", periods=200)
    d = pd.DataFrame({"date": np.repeat(days, 6), "symbol": np.tile(list("ABCDEF"), len(days)),
                      "ucid": np.tile(np.arange(1, 7), len(days))})
    d["feature_base"] = rng.random(len(d))
    d["feature_market_cap_avg_20d"] = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], len(d))
    d["target_binned_return_20"] = d["feature_base"] + rng.normal(0, 0.2, len(d))
    candidate = PREFIX + "logdv_ma5"
    d[candidate] = rng.random(len(d))
    d.loc[d["symbol"] == "A", candidate] = np.nan
    source = tmp_path / "data"
    source.mkdir()
    d.set_index("symbol").to_parquet(source / "train.parquet")
    meta = d[["symbol", "date"]].assign(prediction=rng.random(len(d)))
    meta.to_parquet(source / "historical_meta_models.parquet", index=False)
    # Test fixture contract, explicitly not an actual-source performance result.
    families = {"technical": [candidate]}
    (source / "manifest.json").write_text(json.dumps({"status": "BUILT_VERIFIED_INPUTS", "families": families, "fixture": True}))
    args = SimpleNamespace(data=source, output=tmp_path / "report", models=["ridge_ktz"], seeds=[0],
            phase="dev", test_start=TEST_START, arms=["TECHNICAL"], n_iter=2, n_splits=3, n_jobs=1)
    run(args)
    f = pd.read_parquet(args.output / "dev_ridge_ktz_s0_TECHNICAL_predictions.parquet")
    assert len(f) == (200 - 152) * 6
    uncovered = ~f["covered"]
    assert uncovered.any()
    np.testing.assert_array_equal(f.loc[uncovered, "prediction"], f.loc[uncovered, "baseline_prediction"])
    assert (args.output / "dev_report.html").exists()
    args.phase = "test"
    with pytest.raises(ValueError, match="frozen"):
        run(args)
