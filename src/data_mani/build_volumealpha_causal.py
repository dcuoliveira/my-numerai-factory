"""Build crypto/v2.0-volumealpha-causal from verified vintage inputs.

    python3 -m src.data_mani.build_volumealpha_causal --audit-local
    python3 -m src.data_mani.build_volumealpha_causal --inputs /path/to/verified_bundle

Unlike the retrospective volalpha fetchers this entry point fails closed. It
never labels ticker matches, modern archive retrievals or revised fundamentals
as historically point-in-time. No production all-NaN placeholder is written.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import bin_cross_sectional
from src.data_mani.volumealpha_features import FAMILIES, PREFIX, RAW_COLUMNS, causal_winsorize, features_at
from src.data_mani.volumealpha_sources import (MARKET_KEYS, audit_local, consolidate, digest,
        identity_at, require, snapshot, utc, validate_identities, validate_market, validate_provenance)
from src.prediction.volumealpha import (COST_COLUMNS, FORECAST_COLUMNS, auxiliary_metrics, trading_costs,
                                       volume_targets, walk_forward_volume)

DEFAULT_OUTPUT = Path("data/crypto/v2.0-volumealpha-causal")


def read_bundle(root):
    root = Path(root)
    paths = {k: root / (k + ".parquet") for k in ("identities", "market", "metrics", "events")}
    config = json.loads((root / "sources.json").read_text())
    venues = config.get("trusted_venues", [])
    if not venues:
        raise ValueError("sources.json needs an explicit trusted_venues allowlist")
    identities = validate_identities(pd.read_parquet(paths["identities"]))
    market = validate_market(pd.read_parquet(paths["market"]), venues)
    optional = {}
    for key in ("metrics", "events"):
        if not paths[key].exists():
            optional[key] = None
            continue
        d = validate_provenance(pd.read_parquet(paths[key]), key)
        if key == "metrics":
            require(d, ["asset_id", "date", "metric", "value"], key)
            d["date"] = utc(d["date"])
            if d.duplicated(["asset_id", "date", "metric", "available_at"]).any():
                raise ValueError("metrics: duplicate vintage")
        else:
            require(d, ["event_id", "asset_id", "kind", "scheduled_for", "announced_at"], key)
            for c in ("scheduled_for", "announced_at"):
                d[c] = utc(d[c])
            if (d["available_at"] < d["announced_at"]).any():
                raise ValueError("events: availability precedes public announcement")
            if d.duplicated(["event_id", "available_at"]).any():
                raise ValueError("events: duplicate schedule vintage")
        optional[key] = d
    return identities, market, optional["metrics"], optional["events"], config, paths


def raw_panel(base, identities, market, metrics=None, events=None, source_family="cex"):
    """Slow reference engine: recompute every D from precisely its own vintage.

    This favors auditable semantics over incremental revision-cache complexity.
    Calendar days, including weekends, are emitted for auxiliary targets.
    """
    panel = base.reset_index() if "symbol" not in base.columns else base.copy()
    panel["date"] = utc(panel["date"])
    panel["ucid"] = panel["ucid"].astype(str)
    if panel.duplicated(["date", "ucid"]).any():
        raise ValueError("base has duplicate date/ucid rows")
    grid = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    # Only the most recently observed panel membership may be carried over a
    # non-panel day. No future membership is used to define earlier aggregates.
    universe, current = {}, []
    for D in grid:
        rows = panel[panel["date"] == D]
        ids = identity_at(identities, D)
        mapping = ids.set_index("ucid")["asset_id"]
        if len(rows):
            current = rows["ucid"].map(mapping).dropna().tolist()
        universe[D] = list(current)
    records, row_keys = [], []
    for i, D in enumerate(grid):
        ids = identity_at(identities, D)
        members = set(universe[D]) & set(ids["asset_id"])
        if not members:
            continue
        m = snapshot(market, D, MARKET_KEYS)
        m = m[m["asset_id"].isin(set(ids["asset_id"]))]
        daily = consolidate(m, source_family)
        met = snapshot(metrics, D, ["asset_id", "date", "metric"]) if metrics is not None else None
        ev = snapshot(events, D, ["event_id"], date_col=None) if events is not None else None
        mapping = ids.set_index("ucid")["asset_id"]
        sectors = ids.set_index("asset_id")["sector"].to_dict() if "sector" in ids else None
        x = features_at(daily, D, sorted(members), met, ev,
                        btc=mapping.get("1"), eth=mapping.get("1027"), sectors=sectors, universe=universe)
        records.append(x.reset_index().assign(date=D))
        rows = panel.loc[panel["date"] == D, ["date", "ucid"]].copy()
        rows["asset_id"] = rows["ucid"].map(mapping)
        row_keys.append(rows)
        if i % 100 == 0:
            print(f"as-of snapshots {i + 1}/{len(grid)}", flush=True)
    if not records:
        raise ValueError("no verified point-in-time identities overlap the Numerai panel")
    return pd.concat(records, ignore_index=True), pd.concat(row_keys, ignore_index=True)


def extend_panel(base, raw, row_keys, families=None):
    """Return a strict superset preserving base order, index, dtypes and values."""
    flat = base.reset_index() if "symbol" not in base.columns else base.copy()
    keys = pd.DataFrame({"date": utc(flat["date"]), "ucid": flat["ucid"].astype(str)})
    joined = keys.merge(row_keys, on=["date", "ucid"], how="left", validate="1:1", sort=False)
    joined = joined.merge(raw, on=["asset_id", "date"], how="left", validate="m:1", sort=False)
    columns = [c for c in raw if c.startswith(PREFIX)]
    if not columns:
        raise ValueError("no candidate features")
    continuous = joined[["date"] + columns].copy()
    continuous = causal_winsorize(continuous, columns)
    binned = bin_cross_sectional(continuous, columns, "date", ties="average")
    out = base.copy()
    for c in columns:
        if c in out:
            raise ValueError(f"candidate collides with base column {c}")
        out[c] = binned[c].to_numpy()
    pd.testing.assert_frame_equal(out[base.columns], base, check_exact=True)
    coverage = pd.DataFrame({"date": keys["date"], "ucid": keys["ucid"]})
    for name, cols in (families or FAMILIES).items():
        cols = [c for c in cols if c in columns]
        coverage["cov_" + name] = joined[cols].notna().all(axis=1) if cols else False
    return out, coverage, continuous


def build(args):
    target = Path(args.output)
    if target.exists() and any(target.iterdir()):
        raise ValueError("output directory already contains artifacts; choose a new --output (no overwrite)")
    root = Path(args.inputs)
    ids, market, metrics, events, config, paths = read_bundle(root)
    source = Path(args.base)
    meta = source.parent / "historical_meta_models.parquet"
    if not meta.exists():
        raise ValueError("base historical_meta_models.parquet missing; evaluation cannot score MMC")
    base = pd.read_parquet(source)
    raw, keys = raw_panel(base, ids, market, metrics, events, args.source_family)
    if not raw[RAW_COLUMNS].notna().any().any():
        raise ValueError("no admitted source data: refusing to write an empty candidate dataset")
    last_date = utc(base["date"]).max() + pd.Timedelta(days=1)
    # Labels use the latest vintage in the bundle, with its actual release time.
    # That vintage is conservatively unusable at any earlier auxiliary refit.
    labels_daily = consolidate(snapshot(market, last_date, MARKET_KEYS), args.source_family)
    targets = volume_targets(labels_daily, challenge=args.challenge_horizon)
    frozen = raw[["asset_id", "date", PREFIX + "logdv_ma5"]]
    targets = targets.merge(frozen, on=["asset_id", "date"], how="left", validate="1:1")
    # The surprise's trailing expectation is the one actually known on its
    # forecast date, not a subsequently revised version of that expectation.
    targets["baseline"] = targets[PREFIX + "logdv_ma5"]
    targets["eta"] = targets["logdv_1d"] - targets["baseline"]
    targets = targets.drop(columns=[PREFIX + "logdv_ma5"])
    aux_cols = FAMILIES["technical"]
    forecasts, model_predictions, fit_log = walk_forward_volume(raw, targets, aux_cols,
            refit_every=args.volume_refit, min_dates=args.volume_min_dates,
            min_rows=args.volume_min_rows, seed=args.seed, iterations=args.volume_iterations,
            challenge=args.challenge_horizon)
    aux_report = auxiliary_metrics(forecasts, targets, raw, args.challenge_horizon, model_predictions)
    costs = trading_costs(forecasts, raw, args.trade_sizes)
    raw = raw.merge(forecasts, on=["asset_id", "date"], how="left", validate="1:1")
    raw = raw.merge(costs, on=["asset_id", "date"], how="left", validate="1:1")
    families = dict(FAMILIES, forecasts=FORECAST_COLUMNS, costs=COST_COLUMNS)
    for size in args.trade_sizes:
        families[f"scenario_{size:g}"] = [PREFIX + f"scenario_{size:g}_participation"]
    out, coverage, continuous = extend_panel(base, raw, keys, families)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "crypto/" + target.name, "status": "BUILT_VERIFIED_INPUTS",
        "built_at": datetime.now(timezone.utc).isoformat(), "rows": len(out),
        "source_family": args.source_family, "source_config": config,
        "cutoff": "D 00:00 UTC, bars through D-1; conservative relative to D submission",
        "calendar_windows": True, "base_sha256": digest(source),
        "input_sha256": {k: digest(p) for k, p in paths.items() if p.exists()},
        "families": families, "coverage": coverage.filter(like="cov_").mean().to_dict(),
        "volume_training": {"scheme": "expanding", "refit_every": args.volume_refit,
                            "min_dates": args.volume_min_dates, "min_rows": args.volume_min_rows,
                            "seed": args.seed, "iterations": args.volume_iterations,
                            "challenge_horizon": args.challenge_horizon},
        "unavailable_families": [k for k, v in coverage.filter(like="cov_").any().items() if not v],
        "market_aggregate_scope": "then-known Numerai membership with observed source volume; not global market census",
        "trade_sizes_usd_scenarios": args.trade_sizes,
        "notes": ["All provenance assertions depend on evidence supplied in the admitted bundle.",
                  "Volume predictability is not evidence of Numerai return alpha.",
                  "Constant multiples of inverse volume have identical cross-sectional ranks; scenario arms are not independent evidence.",
                  "Expected Amihud uses the known last absolute return as a proxy, not a return forecast."],
    }
    # Every base column is written in precisely its original index convention.
    out.to_parquet(target / "train.parquet")
    coverage.to_parquet(target / "coverage.parquet", index=False)
    raw.to_parquet(target / "raw_signals.parquet", index=False)
    continuous.to_parquet(target / "continuous_panel.parquet", index=False)
    keys.to_parquet(target / "identity_keys.parquet", index=False)
    targets.to_parquet(target / "volume_targets.parquet", index=False)
    model_predictions.to_parquet(target / "volume_oof_predictions.parquet", index=False)
    aux_report.to_csv(target / "volume_oof_metrics.csv", index=False)
    (target / "volume_fits.json").write_text(json.dumps(fit_log, indent=2))
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(meta, target / meta.name)
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--base", default="data/crypto/v2.0/train.parquet")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--source-family", choices=["cex", "dex", "combined"], default="cex")
    ap.add_argument("--audit-local", action="store_true")
    ap.add_argument("--audit-output", default="results/reports/crypto/volumealpha_causal/source_audit.json")
    ap.add_argument("--volume-refit", type=int, default=22)
    ap.add_argument("--volume-min-dates", type=int, default=130)
    ap.add_argument("--volume-min-rows", type=int, default=500)
    ap.add_argument("--volume-iterations", type=int, default=100)
    ap.add_argument("--challenge-horizon", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trade-sizes", type=float, nargs="+", default=[1000, 10000, 100000])
    args = ap.parse_args()
    if args.audit_local:
        report = audit_local()
        p = Path(args.audit_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
        from html import escape
        html = ("<html><head><meta charset='utf-8'><title>Volume-alpha source audit</title>"
                "<style>body{font:16px system-ui;max-width:1000px;margin:40px}"
                "pre{white-space:pre-wrap;background:#f5f5f5;padding:20px}</style></head><body>"
                "<h1>Volume-alpha · strict point-in-time source audit</h1>"
                "<p><strong>Dataset materialization and real-data evaluation are blocked.</strong> "
                "The sampled local Binance archives and revised on-chain stores do not establish "
                "all the historical identity, original-vintage publication and quote/USD conversion evidence "
                "required by the requested study. An archive bar dated D−1 is not by itself proof "
                "that its downloaded vintage was available on D.</p>"
                "<p>The strict builder and evaluator are implemented and tested with synthetic fixtures. "
                "Those tests establish software behavior, not volume predictability or return alpha. "
                "No all-missing production dataset or real-data performance table has been fabricated. "
                "A verified input bundle is required for the historical study; the registered forward "
                "return test starts 2026-11-13 and requires mature labels.</p>"
                "<h2>Source samples and missing contract fields</h2><pre>" +
                escape(json.dumps(report, indent=2)) + "</pre>"
                "<p><a href='https://www.nber.org/system/files/working_papers/w33037/w33037.pdf'>"
                "Trading Volume Alpha</a> concerns volume forecasts for portfolio execution. "
                "Success at that task would not itself establish Numerai CORR/MMC improvement.</p></body></html>")
        p.with_suffix(".html").write_text(html)
        print(json.dumps(report, indent=2))
    elif args.inputs is None:
        ap.error("--inputs is required: existing Binance caches are retrospective, not verified vintage bundles. "
                 "Run --audit-local and see docs/volumealpha-causal.md.")
    else:
        build(args)


if __name__ == "__main__":
    main()
