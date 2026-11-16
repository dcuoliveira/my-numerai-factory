"""Replace the two shipped v2.0 volatility inputs with nine fixed causal columns."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.volatility_experiment import cross_section_bins, digest
from src.data_mani.volatility_proxies import bins, build_asset, clean_bars

REMOVED = ("feature_volatility_20d", "feature_volatility_60d")
# Modal prior inner-fold nominations. Ties prefer the standard estimator,
# then daily bars. The semivariance pair uses the most frequent joint variant.
SPECS = (
    ("feature_idio_vol_120_tsbin_csbin", "idiosyncratic", "daily", "btc_eth", 120, "residual", "bins"),
    ("feature_semivar_down_120_csbin", "semivariance", "weekday", "side_count_log", 120, "down", "raw"),
    ("feature_semivar_up_120_csbin", "semivariance", "weekday", "side_count_log", 120, "up", "raw"),
    ("feature_rogers_satchell_120_tsbin_csbin", "rogers_satchell", "daily", "gap_added", 120, "range", "bins"),
    ("feature_idio_vol_40_tspct_csbin", "idiosyncratic", "weekday", "btc", 40, "residual", "percentile"),
    ("feature_parkinson_120_tsbin_csbin", "parkinson", "daily", "standard", 120, "range", "bins"),
    ("feature_yang_zhang_120_tsbin_csbin", "yang_zhang", "daily", "standard", 120, "range", "bins"),
    ("feature_parkinson_120_csbin", "parkinson", "weekday", "standard", 120, "range", "raw"),
    ("feature_rogers_satchell_40_tspct_csbin", "rogers_satchell", "daily", "ewma", 40, "range", "percentile"),
)
NEW = tuple(row[0] for row in SPECS)


def estimate_asset(dates, source, benchmark):
    families = defaultdict(list)
    for j, (_, family, calendar, method, window, side, view) in enumerate(SPECS):
        families[family].append((j, (calendar, method, window, side), view))
    estimates = np.full((len(dates), len(NEW)), np.nan)
    for family, settings in families.items():
        values = build_asset(dates, source, benchmark if family == "idiosyncratic" else {},
                             family, [config for _, config, _ in settings])
        for j, (calendar, method, window, side), view in settings:
            raw, hist = values[calendar, method, window]
            col = 1 if side == "up" else 0
            estimates[:, j] = raw[:, col] if view == "raw" else hist[:, col]
            if view == "bins":
                estimates[:, j] = bins(estimates[:, j])
    return estimates


def build(panel_path, price_path, output):
    base = pd.read_parquet(panel_path)
    keys = base[["date", "ucid"]].reset_index(drop=True)
    work = keys.assign(ucid=keys.ucid.astype(int)).sort_values(["date", "ucid"])
    if work.duplicated().any():
        raise ValueError("Numerai asset-date keys must be unique")
    source = pd.read_parquet(price_path)
    if source.duplicated(["ucid", "bar_date"]).any():
        raise ValueError("Duplicate vendor bars")
    by_asset = {ucid: clean_bars(frame) for ucid, frame in source.groupby("ucid")}
    benchmark = {ucid: by_asset[ucid].close for ucid in (1, 1027)}
    candidates = np.full((len(keys), len(NEW)), np.nan)
    checked = []
    for ucid, group in work.groupby("ucid", sort=False):
        if ucid not in by_asset:
            continue
        bars = by_asset[ucid]
        dates = pd.DatetimeIndex(group.date)
        estimate = estimate_asset(dates, bars, benchmark)
        if ucid in benchmark:
            estimate[:, [0, 4]] = np.nan
        candidates[group.index] = estimate
        if len(checked) < 12 and len(dates) >= 380 and ucid not in benchmark:
            cut = dates[len(dates) * 3 // 4]
            keep = dates <= cut
            replay = estimate_asset(dates[keep], bars.loc[bars.index < cut],
                                    {i: s.loc[s.index < cut] for i, s in benchmark.items()})
            np.testing.assert_allclose(replay, estimate[keep], atol=1e-12, rtol=1e-9)
            checked.append({"ucid": int(ucid), "cutoff": str(cut.date())})
    order = work.index.to_numpy()
    covered = np.isfinite(candidates).all(axis=1)
    sorted_covered = order[covered[order]]
    block = np.full(candidates.shape, np.nan, dtype=np.float32)
    block[sorted_covered] = cross_section_bins(
        candidates[sorted_covered], keys.date.to_numpy()[sorted_covered])
    dataset = base.drop(columns=list(REMOVED)).assign(**dict(zip(NEW, block.T, strict=True)))
    if len(dataset.filter(regex=r"^feature_").columns) != 29:
        raise ValueError("Replacement schema must contain 20 retained plus nine candidate features")
    pd.testing.assert_frame_equal(dataset[["ucid", "date"]], base[["ucid", "date"]])
    dataset.to_parquet(output / "v2.0_volatility_replacement.parquet")
    pd.concat([keys, pd.DataFrame(candidates, columns=NEW)], axis=1).assign(
        covered=covered).to_parquet(output / "candidate_availability.parquet", index=False)
    keys.assign(covered=covered).groupby("date").covered.agg(
        panel_rows="size", covered_rows="sum", covered_share="mean"
    ).to_csv(output / "coverage_by_date.csv")
    ticker = base.reset_index() if "symbol" in base.index.names else base
    identities = ticker[["ucid", "symbol"]].drop_duplicates().assign(
        ucid=lambda x: x.ucid.astype(int))
    diagnostics = identities.merge(source[["ucid", "source_symbol"]].drop_duplicates(), on="ucid")
    mismatches = diagnostics[diagnostics.symbol.str.upper() != diagnostics.source_symbol.str.upper()]
    mismatches.to_csv(output / "ticker_diagnostics.csv", index=False)
    reuse = identities.groupby(identities.symbol.str.upper()).ucid.nunique()
    reuse[reuse.gt(1)].to_csv(output / "ticker_reuse.csv", header=["distinct_ucid"])
    report = {
        "panel_rows": len(keys), "covered_rows": int(covered.sum()), "cutoff_replays": checked,
        "ticker_mismatch_rows": len(mismatches), "reused_tickers": int(reuse.gt(1).sum()),
        "removed": REMOVED,
        "columns": [dict(zip(("column", "family", "calendar", "estimator", "window", "side", "view"),
                            spec, strict=True)) for spec in SPECS],
        "inputs": {str(path): digest(path) for path in (panel_path, price_path, price_path.with_name("price_requests.json"))},
    }
    (output / "dataset_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Dataset: {len(keys):,} rows, {covered.sum():,} covered, {len(checked)} cutoff replays", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/crypto/v2.0/train.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/volatility_replacement"))
    parser.add_argument("--prices", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build(args.panel, args.prices or args.output / "ohlc.parquet", args.output)


if __name__ == "__main__":
    main()
