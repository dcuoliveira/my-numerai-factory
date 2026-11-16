"""Materialized-data report for E152. Does not claim return alpha."""
from html import escape
import json
from pathlib import Path

import pandas as pd

from src.data_mani.prune_volumealpha_constants import prune_dataset
from src.data_mani.volumealpha_sources import digest
from src.analysis.validate_volumealpha_dataset import validate


def main():
    work = Path('/scratch/danielco/volumealpha_e152')
    root = Path('data/crypto/v2.0-volumealpha-causal')
    folder = Path('results/reports/crypto/e152_volumealpha_causal_build')
    folder.mkdir(parents=True, exist_ok=True)
    paths = [root] + [Path(str(root) + '-' + k) for k in ['binance', 'other-cex', 'coinbase-fx']]
    rows = []
    fingerprints = {p.name: digest(p) for p in [work / 'venue_observations.parquet', work / 'fx_rates.parquet',
                    work / 'identities_inferred.parquet', work / 'daily_combined.parquet',
                    Path('/scratch/danielco/volalpha/onchain/coinmetrics.parquet'),
                    Path('/scratch/danielco/volalpha/onchain/llama.parquet')]}
    for path in paths:
        m = prune_dataset(path)
        m['source_sha256'] = fingerprints
        m['dollar_volume_definition'] = 'sum of daily base volume times same-day close in measured USD; daily-bar notional proxy, not exact trade-level USD turnover'
        m['observation_age_definition'] = 'days since first observation in the downloaded archive; left-censored by archive start, not protocol launch age'
        m['auxiliary_optimization_note'] = 'Some MLP fits reached the predeclared 50-iteration budget without convergence; predictions are still OOF'
        m['auxiliary_features_in_this_arm'] = path == root
        (path / 'manifest.json').write_text(json.dumps(m, indent=2))
        coverage = m['coverage'].get('cov_technical', 0)
        rows.append({'dataset': 'crypto/' + path.name, 'rows': m['rows'], 'added_features': m['feature_count'],
                     'total_features': 22 + m['feature_count'], 'technical_complete_coverage': coverage})
    audit = validate()
    manifest = json.loads((root / 'manifest.json').read_text())
    definitions = []
    for family, cc in manifest['families'].items():
        definitions.append({'family': family, 'columns': len(cc), 'complete_coverage': manifest['coverage']['cov_' + family]})
    metrics = pd.read_csv(root / 'volume_oof_metrics.csv')
    metrics = metrics[metrics.grouping.eq('all')]
    metric_cols = [c for c in ['model', 'target', 'horizon', 'n', 'oof_r2_vs_trailing5', 'mae_log_volume', 'q10_realized_fraction_below'] if c in metrics]
    pd.DataFrame(rows).to_csv(folder / 'datasets.csv', index=False)
    pd.DataFrame(definitions).to_csv(folder / 'coverage.csv', index=False)
    text = '''<html><head><meta charset="utf-8"><title>E152 volume-alpha dataset complete</title>
<style>body{font:14px system-ui;margin:30px;color:#17212c;max-width:1300px}h1,h2{color:#173e5b}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:6px;border:1px solid #d0d9e0;text-align:left}th{background:#edf2f6}pre{white-space:pre-wrap;font-size:11px}p{line-height:1.5}</style></head><body>
<h1>E152 · volume-alpha dataset completed</h1>
<p><b>The requested dataset and three source ablations are materialized and validated.</b>
Every original Numerai v2.0 row, column, target, dtype and index is preserved exactly.
This is the user-authorized inferred-source workaround: timestamp-causal at D−2,
<b>not verified historical-vintage point-in-time data</b>. No return-alpha verdict is made.</p>
<h2>Datasets</h2>'''
    text += pd.DataFrame(rows).to_html(index=False, float_format=lambda x: f'{x:.2%}')
    text += '''<h2>What resolved the blocker</h2>
<p>Downloaded independent Coinbase USDT/USD and DAI/USD candles; use measured reconstructed Yahoo USD rates on uncovered dates and currencies. No missing stablecoin quote is replaced by $1. Binance spot archives include delisted pairs. Other CEX observations are a separate family because their REST history selects currently listed pairs. Daily archives have documented next-day publication; a two-day observation lag supplies an additional safety day. Exact original-vintage publication and historical mappings remain inferred. Stored HTTP metadata corroborates the sampled publication timing.</p>
<p>Permanent CMC IDs anchor the panel; native chains or current contract metadata are retained where available. Excluded 31 reused ticker histories before joins. Unique candidates also pass a causal expanding volatility cross-check against their own shipped features. This reduces identity mistakes but is not proof of historical identity. Removed 66 known wrong-token Gate/HYPE observations. Perpetual open-interest notionals are converted with measured quote/USD rates; funding rates are not summed over duplicate quote contracts.</p>
<h2>Feature families and complete causal coverage</h2>'''
    text += pd.DataFrame(definitions).to_html(index=False, float_format=lambda x: f'{x:.2%}')
    text += '''<p>The technical replication has eight features: lagged calendar-day return and log-dollar-volume moving averages over 1, 5, 22 and 252 days. Extensions, fundamental proxies, cross-asset exposures, OOF volume forecasts and cost scenarios are separable. Continuous signals use strictly earlier-date winsorization before tie-preserving five-bin ranks on the dated panel. Unavailable values stay missing. Coverage flags remain sidecars, never production inputs. Features constant within every development-date cross-section are omitted, including the inactive weekend interaction on this weekday-only panel.</p>
<h2>Out-of-sample volume diagnostics</h2>
<p>Expanding fits every 90 calendar days; up to 50,000 earlier rows, seed 152, 50 iterations.
Ridge, histogram GBM and a 32/16/8 MLP predict eta and direct 1-, 5- and 20-calendar-day volume targets. The latter is an explicit volume scenario, not a claim of exact alignment with Numerai’s payable return horizon. Direct level fits avoid using an unobserved D−1 baseline under the D−2 cutoff. Quantile/classification and realized-error features are also OOF. Some MLP fits reached the fixed iteration budget. eta R² uses eta=0 as baseline; level R² uses the trailing-five-day volume-level baseline available at the cutoff.</p>'''
    text += metrics[metric_cols].to_html(index=False, na_rep='—', float_format=lambda x: f'{x:.5g}')
    text += '<h2>Validation on actual artifacts</h2><pre>' + escape(json.dumps(audit, indent=2)) + '</pre>'
    text += '''<h2>Limits and next experiment</h2>
<p>Vendor revisions, reconstructed availability and current contract metadata remain limitations. Other-CEX survivorship must be checked against Binance-only and direct-USD-conversion source arms. Dollar volume is a daily-bar notional proxy. Observation age is left-censored by archive coverage. FDV, liquidations, reliable launch dates, dated coin-event catalogs and per-token DEX venue volume are named unavailable inputs; they were not fabricated. Revised fundamental families are separable.</p>
<p>Volume predictability and execution/capacity proxies do not establish Numerai return alpha. The return-feature comparison has not been run by this dataset build. Its evaluator requires --allow-inferred and uses matched covered-only fits with full-universe baseline rank-slot routing; source limitations must remain in that report. A forward test needs mature labels after the registered 2026-11-13 boundary.</p>
<h2>Sources and reproducibility</h2>
<p><a href="https://github.com/binance/binance-public-data/blob/master/README.md">Binance public-data documentation</a>; <a href="https://api.exchange.coinbase.com/products/USDT-USD/candles">Coinbase direct USD candles</a>; Coin Metrics community and DefiLlama caches with their recorded retrieval dates. Source tables, raw HTTP responses, identities and hashes are retained under /scratch/danielco/volumealpha_e152/. The root dataset manifest contains definitions and fingerprints.</p></body></html>'''
    (folder / 'dataset_report.html').write_text(text)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
