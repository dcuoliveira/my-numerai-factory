"""Complete E152 with explicitly inferred availability and reconstructed sources.

User authorized public-source alternatives and inference on 2026-11-13. The
strict vintage builder remains unchanged. This builder NEVER marks inferred
rows vintage_verified=True. It produces a real, fully retained Numerai panel,
source ablations, OOF volume predictions, and a reproducible source/audit report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import requests

from src.data_mani import volalpha_features as VF
from src.data_mani.binning import bin_cross_sectional
from src.data_mani.volumealpha_features import FAMILIES, PREFIX, causal_winsorize
from src.data_mani.volumealpha_sources import digest
from src.prediction.volumealpha import (FORECAST_COLUMNS, COST_COLUMNS, auxiliary_metrics,
                                       trading_costs, volume_targets, walk_forward_volume)

SOURCE = Path('/scratch/danielco/volalpha')
WORK = Path('/scratch/danielco/volumealpha_e152')
OUTPUT = Path('data/crypto/v2.0-volumealpha-causal')
NATIVE = {'1': 'bitcoin:native', '1027': 'ethereum:native', '2': 'litecoin:native',
          '52': 'xrpl:native', '74': 'dogecoin:native', '328': 'monero:native',
          '1839': 'bsc:native', '5426': 'solana:native', '2010': 'cardano:native',
          '1958': 'tron:native', '512': 'stellar:native', '1321': 'ethereum-classic:native'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(stamp(), message, flush=True)


def load_parts(kind, columns=None):
    files = sorted((SOURCE / kind).glob('*.parquet'))
    parts = []
    for p in files:
        d = pd.read_parquet(p)
        if d.empty:
            continue
        if columns:
            d = d[[c for c in columns if c in d]]
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def fetch_fx(first, last):
    """Independent USD quotes, with cached raw responses; no $1 peg assumption."""
    folder = WORK / 'coinbase_fx'
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(coin, a, min(a + pd.Timedelta(days=299), last))
            for coin in ('USDT', 'USDC', 'DAI') for a in pd.date_range(first, last, freq='299D')]
    def fetch(job):
        coin, a, b = job
        path = folder / f'{coin}_{a.date()}.json'
        if path.exists():
            saved = json.loads(path.read_text())
        else:
            url = f'https://api.exchange.coinbase.com/products/{coin}-USD/candles'
            params = {'granularity': 86400, 'start': str(a.date()), 'end': str(b.date())}
            for retry in range(3):
                try:
                    r = requests.get(url, params=params, timeout=25)
                    if r.status_code == 429:
                        time.sleep(1 + retry)
                        continue
                    saved = {'url': r.url, 'retrieved_at': stamp(), 'status': r.status_code,
                             'headers': dict(r.headers), 'rows': r.json()}
                    path.write_text(json.dumps(saved))
                    break
                except (requests.RequestException, ValueError):
                    time.sleep(1 + retry)
            else:
                return pd.DataFrame()
        if saved['status'] != 200 or not isinstance(saved['rows'], list):
            return pd.DataFrame()
        rows = saved['rows']
        if not rows:
            return pd.DataFrame()
        d = pd.DataFrame(rows, columns=['ts', 'low', 'high', 'open', 'usd', 'volume'])
        d['date'] = pd.to_datetime(d['ts'], unit='s').dt.normalize()
        return d[['date', 'usd']].assign(quote=coin, source='coinbase_direct_usd',
                   retrieved_at=saved['retrieved_at'], evidence=str(path))
    results = list(ThreadPoolExecutor(3).map(fetch, jobs))
    cb = pd.concat([r for r in results if len(r)], ignore_index=True)
    cb = cb.drop_duplicates(['quote', 'date'])
    legacy = pd.read_parquet(SOURCE / 'onchain/stables.parquet').rename(columns={'stable': 'quote'})
    legacy['date'] = pd.to_datetime(legacy['date']).dt.normalize()
    legacy['source'] = 'yahoo_reconstructed_usd'
    legacy['evidence'] = str(SOURCE / 'onchain/stables.parquet')
    # Direct USD market first; legacy conversion fills only dates it lacks.
    both = pd.concat([cb, legacy], ignore_index=True).drop_duplicates(['quote', 'date'])
    both = both[both.usd.gt(0) & np.isfinite(both.usd)]
    both['retrieved_at'] = pd.to_datetime(both.retrieved_at, utc=True)
    both['availability_assumption_days'] = 2
    both['vintage_verified'] = False
    both.to_parquet(WORK / 'fx_rates.parquet', index=False)
    return both


def identities(panel):
    p = panel.assign(ticker=panel.symbol.astype(str).str.upper().str.lstrip('$'), ucid=panel.ucid.astype(str))
    pairs = p[['ticker', 'ucid']].drop_duplicates()
    counts = pairs.groupby('ticker').ucid.nunique()
    ambiguous = set(counts[counts > 1].index)
    # Exclude ambiguous ticker histories entirely. Never resolve reuse with a
    # future price pattern or choose the currently surviving coin.
    pairs = pairs[~pairs.ticker.isin(ambiguous)].copy()
    counts_u = pairs.groupby('ucid').ticker.nunique()
    pairs = pairs[~pairs.ucid.isin(counts_u[counts_u > 1].index)]
    protos = json.loads((SOURCE / 'onchain/llama_protocols.json').read_text())
    address = {}
    for item in protos:
        cmc, addr = item.get('cmcId'), item.get('address')
        if cmc is not None and addr and isinstance(addr, str):
            address.setdefault(str(cmc), set()).add(addr)
    def choose(ucid):
        if ucid in NATIVE:
            return NATIVE[ucid], 'curated_native_chain'
        aa = address.get(ucid, set())
        if len(aa) == 1:
            a = next(iter(aa))
            return a if ':' in a else 'ethereum:' + a, 'current_contract_metadata_inferred_history'
        return 'cmc:' + ucid, 'permanent_cmc_id_with_causal_price_gate'
    values = pairs.ucid.map(choose)
    pairs['asset_id'] = values.map(lambda x: x[0])
    pairs['identity_method'] = values.map(lambda x: x[1])
    pairs['historical_mapping_verified'] = False
    duplicated = pairs.asset_id.duplicated(keep=False)
    pairs.loc[duplicated, 'asset_id'] = 'cmc:' + pairs.loc[duplicated, 'ucid']
    pairs.loc[duplicated, 'identity_method'] = 'permanent_cmc_id_ambiguous_contract_metadata'
    pairs.to_parquet(WORK / 'identities_inferred.parquet', index=False)
    return pairs, sorted(ambiguous)


def prepare_sources(panel):
    WORK.mkdir(parents=True, exist_ok=True)
    ids, ambiguous = identities(panel)
    mapping = ids.set_index('ticker').ucid
    if (WORK / 'venue_observations.parquet').exists():
        v = pd.read_parquet(WORK / 'venue_observations.parquet')
    else:
        log('reading Binance spot archives')
        sp = load_parts('binance/spot')
        meta = json.loads((SOURCE / 'binance/pairs.json').read_text())
        lookup = meta['spot']
        sp['ticker'] = sp.pair.map(lambda p: lookup[p][0])
        sp['quote'] = sp.pair.map(lambda p: lookup[p][2])
        sp['ucid'] = sp.ticker.map(mapping)
        sp = sp[sp.ucid.notna()].copy()
        sp['venue'], sp['source_family'] = 'binance', 'binance'
        fx = fetch_fx(pd.Timestamp('2018-11-01'), panel.date.max())
        rates = fx[['quote', 'date', 'usd', 'source']].copy()
        # Non-stable quotes converted through same-day measured USDT/USD.
        usdt = rates[rates.quote.eq('USDT')].set_index('date').usd
        extra = []
        for coin in ('BTC', 'ETH', 'BNB'):
            p = sp[sp.pair.eq(coin + 'USDT')].set_index('date')['close']
            z = p * usdt.reindex(p.index)
            extra.append(pd.DataFrame({'date': z.index, 'usd': z.values, 'quote': coin, 'source': 'binance_cross_via_measured_usdt'}))
        rates = pd.concat([rates] + extra, ignore_index=True)
        rates = pd.concat([rates, pd.DataFrame({'date': pd.date_range('2018-11-01', panel.date.max()),
                                 'quote': 'USD', 'usd': 1., 'source': 'usd_numeraire'})], ignore_index=True)
        sp = sp.rename(columns={'volume': 'base_volume'})
        cex = load_parts('cex')
        cex['ticker'] = cex.symbol.astype(str).str.upper().str.lstrip('$')
        cex['ucid'] = cex.ticker.map(mapping)
        cex = cex[cex.ucid.notna()].copy()
        cex['quote'] = cex.venue.map({'coinbase': 'USD', 'hyperliquid': 'USDC'}).fillna('USDT')
        cex = cex.rename(columns={'volume': 'base_volume'})
        cex['source_family'] = 'other_cex'
        cex['src_file'] = cex.venue + '/' + cex.pair
        cols = ['ucid', 'ticker', 'venue', 'pair', 'quote', 'date', 'open', 'high', 'low', 'close',
                'base_volume', 'source_family', 'src_file', 'retrieved_at']
        v = pd.concat([sp[cols], cex[cols]], ignore_index=True)
        v['date'] = pd.to_datetime(v.date).dt.normalize()
        key = ['venue', 'pair', 'date']
        if v.duplicated(key).any():
            dup = v[v.duplicated(key, keep=False)]
            if dup.groupby(key).ucid.nunique().gt(1).any():
                raise ValueError('one venue/pair/day assigned to multiple assets')
            v = v.drop_duplicates(key)
        v = v.merge(rates.rename(columns={'source': 'fx_source'}), on=['quote', 'date'], how='left', validate='m:1')
        v['dollar_volume'] = v.base_volume * v.close * v.usd
        v['close_usd'], v['high_usd'], v['low_usd'] = v.close * v.usd, v.high * v.usd, v.low * v.usd
        v = v[v.close_usd.gt(0) & v.base_volume.ge(0) & v.usd.gt(0)].copy()
        median = v.groupby(['ucid', 'date']).close_usd.transform('median')
        v['price_crosscheck_pass'] = (np.log(v.close_usd / median).abs() <= np.log(1.25))
        v['observed_at_inferred'] = v.date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        v['available_at_inferred'] = v.date + pd.Timedelta(days=2)
        v['vintage_verified'] = False
        v = v.merge(ids[['ucid', 'asset_id', 'identity_method']], on='ucid', how='left', validate='m:1')
        v.to_parquet(WORK / 'venue_observations.parquet', index=False)
    from src.data_mani.fetch_exchange_spot import BAD_BARS_BEFORE
    for (ticker, venue), earliest in BAD_BARS_BEFORE.items():
        v = v[~(v.ticker.eq(ticker) & v.venue.eq(venue) & v.date.lt(earliest))]
    v.to_parquet(WORK / 'venue_observations.parquet', index=False)
    def daily(q):
        q = q[q.price_crosscheck_pass]
        q = q.assign(wprice=q.close_usd * q.dollar_volume)
        g = q.groupby(['ucid', 'date'])
        d = g.agg(dollar_volume=('dollar_volume', 'sum'), weighted=('wprice', 'sum'),
                  close=('close_usd', 'median'), high=('high_usd', 'max'), low=('low_usd', 'min'))
        d['close'] = (d.weighted / d.dollar_volume.replace(0, np.nan)).fillna(d.close)
        ven = q.groupby(['ucid', 'date', 'venue']).dollar_volume.sum()
        shares = ven / ven.groupby(level=[0, 1]).transform('sum').replace(0, np.nan)
        d['venue_hhi'] = shares.pow(2).groupby(level=[0, 1]).sum(min_count=1)
        d['venue_count'] = ven.groupby(level=[0, 1]).size()
        d = d.drop(columns='weighted').reset_index()
        d['available_at'] = pd.to_datetime(d.date + pd.Timedelta(days=2), utc=True)
        return d
    fam = {'combined': daily(v), 'binance': daily(v[v.venue.eq('binance')]),
           'other_cex': daily(v[~v.venue.eq('binance')]),
           'coinbase_fx': daily(v[v.fx_source.isin(['coinbase_direct_usd', 'usd_numeraire'])])}
    for k, d in fam.items():
        d.to_parquet(WORK / f'daily_{k}.parquet', index=False)
        log(f'{k}: {len(d):,} asset-days, {d.ucid.nunique()} assets')
    (WORK / 'ambiguous_tickers.json').write_text(json.dumps(ambiguous))
    return ids, fam


def wide(daily, column, grid, assets):
    return daily.pivot(index='date', columns='ucid', values=column).reindex(index=grid, columns=assets)


def compute_raw(daily, panel, ids, lag=2, metrics=True, technical_only=False):
    """Vectorized calendar-day equivalent of the reference feature layer."""
    assets = sorted(daily.ucid.unique())
    grid = pd.date_range(daily.date.min(), panel.date.max())
    close, dv = (wide(daily, c, grid, assets) for c in ('close', 'dollar_volume'))
    ret, logdv = close.pct_change(fill_method=None), np.log1p(dv)
    raw = {}
    def put(name, v): raw[PREFIX + name] = v
    for w in (1, 5, 22, 252):
        put(f'return_ma{w}', ret.rolling(w, min_periods=w).mean())
        put(f'logdv_ma{w}', logdv.rolling(w, min_periods=w).mean())
    if not technical_only:
        shock = logdv - logdv.shift(1).rolling(5, min_periods=5).mean()
        for w in (5, 22): put(f'shock{w}', logdv - logdv.shift(1).rolling(w, min_periods=w).mean())
        for w in (1, 5, 22): put(f'growth{w}', logdv.diff(w))
        put('acceleration', logdv.diff().diff())
        put('dv_vol22', dv.rolling(22, min_periods=22).std())
        put('amihud', ret.abs() / dv.replace(0, np.nan))
        put('range_dv', (wide(daily, 'high', grid, assets) - wide(daily, 'low', grid, assets)) / close / dv.replace(0, np.nan))
        for a, c in [('venue_count_change', 'venue_count'), ('venue_hhi_change', 'venue_hhi')]:
            put(a, wide(daily, c, grid, assets).diff())
        put('volume_price_divergence', logdv.diff(5) - np.log(close).diff(5))
        # Membership is then-known, forward-carried over weekends only. Values are
        # NEVER forward-filled. A coin missing on a dated panel is not a member.
        membership = pd.crosstab(panel.date, panel.ucid).gt(0).reindex(columns=assets, fill_value=False)
        membership = membership.reindex(grid, method='ffill').fillna(False)
        mret, mdv = ret.where(membership), dv.where(membership)
        n = mret.notna().sum(axis=1)
        loo = mret.rsub(mret.sum(axis=1, min_count=1), axis=0).div(
            mret.notna().astype(int).rsub(n, axis=0).replace(0, np.nan))
        for name, market in [('btc', ret['1'] if '1' in ret else None), ('eth', ret['1027'] if '1027' in ret else None), ('market_loo', loo)]:
            if market is not None: put('beta_' + name, VF.dimson_beta(ret, market, win=120, minp=60))
        total = mdv.sum(axis=1, min_count=1)
        market_log = np.log1p(total)
        market_shock = market_log - market_log.shift(1).rolling(5, min_periods=5).mean()
        beta = raw[PREFIX + 'beta_market_loo']
        put('market_shock_exposure', beta.mul(market_shock, axis=0))
        put('market_share', dv.div(total.replace(0, np.nan), axis=0))
        put('share_change', raw[PREFIX + 'market_share'].diff())
        breadth = shock.where(membership).mean(axis=1)
        positive = ((shock > 0) & membership).sum(axis=1) / shock.where(membership).notna().sum(axis=1).replace(0, np.nan)
        put('breadth_exposure', beta.mul(breadth, axis=0))
        put('positive_fraction_exposure', beta.mul(positive, axis=0))
        for key, coin in [('btc', '1'), ('eth', '1027')]:
            if coin not in shock: continue
            put(key + '_shock_exposure', raw[PREFIX + 'beta_' + key].mul(shock[coin], axis=0))
            market = pd.DataFrame(np.tile(shock[coin].shift(1).to_numpy()[:, None], (1, len(assets))), index=grid, columns=assets)
            put(key + '_leadlag', shock.rolling(60, min_periods=40).corr(market))
        # Statistical sectors use only the previous 180 days; refit each quarter.
        sect = pd.DataFrame(np.nan, index=grid, columns=assets)
        for q in pd.date_range(grid.min().to_period('Q').start_time, grid.max(), freq='QS'):
            labels = VF.sectors(ret.where(membership), q)
            ix = (grid >= q) & (grid < q + pd.DateOffset(months=3))
            for a, k in labels.items(): sect.loc[ix, a] = k
        sector_share, sector_shock = sect * np.nan, sect * np.nan
        for k in range(VF.SECTOR_K):
            values = mdv.where(sect.eq(k))
            s = values.sum(axis=1, min_count=1)
            sl = np.log1p(s)
            ss = sl - sl.shift(1).rolling(5, min_periods=5).mean()
            sector_share = sector_share.where(~sect.eq(k), dv.div(dv.rsub(s, axis=0).replace(0, np.nan)))
            sector_shock = sector_shock.where(~sect.eq(k), beta.mul(ss, axis=0))
        put('sector_loo_share', sector_share)
        put('sector_shock_exposure', sector_shock)
        for name, x in [('price_age', close), ('volume_age', dv)]:
            first = x.apply(lambda s: s.first_valid_index())
            age = pd.DataFrame({a: (grid - first[a]).days.astype(float) if pd.notna(first[a]) else np.nan for a in assets}, index=grid)
            put(name, age.where(age >= 0))
        weekend = pd.Series(grid.dayofweek >= 5, index=grid)
        past_we = shock.where(weekend, axis=0).rolling(120, min_periods=15).mean()
        past_wd = shock.where(~weekend, axis=0).rolling(120, min_periods=40).mean()
        # The event is the future PREDICTION date's weekend, not the source bar day.
        prediction_weekend = pd.Series((grid + pd.Timedelta(days=lag)).dayofweek >= 5, index=grid)
        put('weekend_sensitivity', (past_we - past_wd).mul(prediction_weekend.astype(float), axis=0))
        metric_frames = {}
        if metrics:
            cm = pd.read_parquet(SOURCE / 'onchain/coinmetrics.parquet')
            cm['ticker'] = cm.asset.str.upper()
            mapping = ids.set_index('ticker').ucid
            cm['ucid'] = cm.ticker.map(mapping)
            cm = cm[cm.ucid.notna()].drop_duplicates(['ucid', 'date'])
            # Current Coin Metrics ticker data need an independent daily price
            # cross-check before their revised fundamentals count for a coin.
            cm = cm.merge(daily[['ucid', 'date', 'close']], on=['ucid', 'date'], how='left', validate='1:1')
            cm['price_check'] = (np.log(cm.PriceUSD / cm.close).abs() <= np.log(1.25))
            for key, col in {'market_cap': 'CapMrktCurUSD', 'transactions': 'TxCnt', 'active_addresses': 'AdrActCnt', 'fees': 'FeeTotUSD', 'realized_cap': 'CapRealUSD'}.items():
                if col in cm:
                    q = cm.loc[cm.price_check, ['ucid', 'date', col]].rename(columns={col: 'value'})
                    metric_frames[key] = wide(q, 'value', grid, assets)
            llama = pd.read_parquet(SOURCE / 'onchain/llama.parquet')
            llama['ucid'] = llama.cmcId.astype(str)
            # Multiple protocols for a token must not be summed into a fake token
            # fundamental. Prefer a deterministic, single provider record per day.
            if 'kind' in llama: llama = llama.sort_values('kind')
            llama = llama.drop_duplicates(['ucid', 'date'])
            for key in ['fees', 'revenue', 'dex_volume', 'tvl']:
                if key in llama:
                    q = wide(llama.rename(columns={key: 'value'}), 'value', grid, assets)
                    metric_frames[key] = q
            for key in ['market_cap', 'fdv']:
                if key in metric_frames: put(key, metric_frames[key])
            cap = metric_frames.get('market_cap')
            if cap is not None:
                put('turnover', dv / cap.replace(0, np.nan))
                for key in ['fees', 'revenue', 'tvl', 'realized_cap']:
                    if key not in metric_frames: continue
                    denominator = metric_frames[key]
                    if key in ['fees', 'revenue']: denominator = 365 * denominator.rolling(30, min_periods=30).mean()
                    put('mvrv' if key == 'realized_cap' else 'cap_' + key, cap / denominator.replace(0, np.nan))
            for key in ['fees', 'revenue', 'transactions', 'active_addresses', 'dex_volume']:
                if key not in metric_frames: continue
                x = metric_frames[key]
                past = x.shift(1).rolling(22, min_periods=22)
                put(key + '_surprise', (x - past.mean()) / past.std().replace(0, np.nan))
            perp = load_parts('binance/metrics')
            if len(perp) and 'sum_open_interest_value_last' in perp:
                metrics_lookup = json.loads((SOURCE / 'binance/pairs.json').read_text())['metrics']
                perp['ticker'] = perp.pair.map(lambda p: metrics_lookup.get(p, [None])[0])
                perp['ucid'] = perp.ticker.map(mapping)
                perp = perp[perp.ucid.notna()].copy()
                perp['quote'] = perp.pair.map(lambda p: metrics_lookup.get(p, [None, None, None])[2])
                fx = pd.read_parquet(WORK / 'fx_rates.parquet', columns=['quote', 'date', 'usd'])
                perp = perp.merge(fx, on=['quote', 'date'], how='left', validate='m:1')
                perp['sum_open_interest_value_last'] *= perp.usd
                oi = perp.groupby(['ucid', 'date']).sum_open_interest_value_last.sum(min_count=1).reset_index(name='value')
                oiw = wide(oi, 'value', grid, assets)
                put('oi_spot', oiw / dv.replace(0, np.nan))
                if cap is not None: put('oi_cap', oiw / cap.replace(0, np.nan))
            funding = load_parts('binance/funding')
            if len(funding):
                lookup = json.loads((SOURCE / 'binance/pairs.json').read_text())['um']
                funding['ucid'] = funding.pair.map(lambda p: mapping.get(lookup.get(p, [None])[0]))
                # One standard USDT-margined perpetual per base, not a sum of rates.
                funding = funding[funding.pair.str.endswith('USDT') & funding.ucid.notna()]
                funding = funding.drop_duplicates(['ucid', 'date'])
                if 'funding_sum' in funding:
                    put('funding', wide(funding.rename(columns={'funding_sum': 'value'}), 'value', grid, assets))
    # Missing families stay absent (and are named in the manifest), rather than
    # all-NaN columns forcing every broad arm's coverage to zero.
    data = VF.long(raw).rename(columns={'symbol': 'ucid'})
    data['date'] = pd.to_datetime(data.date, utc=True) + pd.Timedelta(days=lag)
    data = data[data.date.le(pd.to_datetime(panel.date.max(), utc=True))]
    data['asset_id'] = data.ucid.map(ids.set_index('ucid').asset_id)
    data = data.replace([np.inf, -np.inf], np.nan)
    # Causal identity gate on this asset's own rows: same time period, no target
    # or future price, 20/60-day volatility percentile gaps to shipped features.
    v20 = ret.rolling(20, min_periods=20).std()
    v60 = ret.rolling(60, min_periods=60).std()
    gate_raw = VF.long({'v20': v20, 'v60': v60}).rename(columns={'symbol': 'ucid'})
    gate_raw['date'] = pd.to_datetime(gate_raw.date, utc=True) + pd.Timedelta(days=lag)
    p = panel.copy()
    p['date'] = pd.to_datetime(p.date, utc=True)
    g = p[['ucid', 'date', 'feature_volatility_20d', 'feature_volatility_60d']].merge(gate_raw, on=['ucid', 'date'], how='left', validate='1:1')
    gaps = []
    for a, b in [('v20', 'feature_volatility_20d'), ('v60', 'feature_volatility_60d')]:
        gaps.append((g.groupby('date')[a].rank(pct=True) - g[b]).abs())
    g['gap'] = pd.concat(gaps, axis=1).mean(axis=1)
    g = g.sort_values(['ucid', 'date'])
    g['identity_stat'] = g.groupby('ucid').gap.transform(lambda s: s.expanding(min_periods=40).mean())
    gate = g[['ucid', 'date', 'identity_stat']]
    # For calendar days between panel dates, use the latest PAST gate result.
    chunks = []
    for a, x in data.groupby('ucid', sort=False):
        y = gate[gate.ucid.eq(a)][['date', 'identity_stat']].sort_values('date')
        if y.empty: continue
        z = pd.merge_asof(x.sort_values('date'), y, on='date', direction='backward')
        passed = z.identity_stat.le(0.18)
        z.loc[~passed, list(raw)] = np.nan
        chunks.append(z)
    return pd.concat(chunks, ignore_index=True), gate, grid


def write_dataset(base, raw, output, families, manifest):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'refusing to overwrite {output}')
    output.mkdir(parents=True)
    flat = base.reset_index() if 'symbol' not in base else base.copy()
    keys = pd.DataFrame({'ucid': flat.ucid.astype(str).to_numpy(), 'date': pd.to_datetime(flat.date, utc=True).to_numpy()})
    cols = [c for c in raw if c.startswith(PREFIX)]
    joined = keys.merge(raw[['ucid', 'date'] + cols], on=['ucid', 'date'], how='left', validate='1:1', sort=False)
    continuous = causal_winsorize(joined[['date'] + cols], cols)
    binned = bin_cross_sectional(continuous, cols, 'date', ties='average')
    out = base.copy()
    for col in cols: out[col] = binned[col].to_numpy()
    pd.testing.assert_frame_equal(out[base.columns], base, check_exact=True)
    coverage = keys.copy()
    for family, cc in families.items(): coverage['cov_' + family] = joined[cc].notna().all(axis=1)
    manifest = dict(manifest, rows=len(base), families=families, coverage=coverage.filter(like='cov_').mean().to_dict(),
                    feature_count=len(cols), version='crypto/' + output.name)
    out.to_parquet(output / 'train.parquet')
    coverage.to_parquet(output / 'coverage.parquet', index=False)
    continuous.to_parquet(output / 'continuous_panel.parquet', index=False)
    shutil.copyfile(WORK / 'base_historical_meta_models.parquet', output / 'historical_meta_models.parquet')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    from src.data_mani.prune_volumealpha_constants import prune_dataset
    manifest = prune_dataset(output)
    log(f'wrote {output}: {len(base):,} rows, {manifest["feature_count"]} added features')
    return manifest


def run(args):
    WORK.mkdir(parents=True, exist_ok=True)
    protocol = {'lag': args.lag, 'iterations': args.iterations, 'max_train': args.max_train}
    saved = WORK / 'build_protocol.json'
    if saved.exists() and json.loads(saved.read_text()) != protocol:
        raise ValueError('cached work directory uses a different build protocol; select a fresh --work directory')
    saved.write_text(json.dumps(protocol, indent=2))
    if not (WORK / 'base_train.parquet').exists():
        shutil.copyfile('data/crypto/v2.0/train.parquet', WORK / 'base_train.parquet')
        shutil.copyfile('data/crypto/v2.0/historical_meta_models.parquet', WORK / 'base_historical_meta_models.parquet')
    base = pd.read_parquet(WORK / 'base_train.parquet')
    panel = base.reset_index() if 'symbol' not in base else base.copy()
    panel['ucid'] = panel.ucid.astype(str)
    ids_path = WORK / 'identities_inferred.parquet'
    if ids_path.exists() and (WORK / 'daily_combined.parquet').exists():
        ids = pd.read_parquet(ids_path)
        fam = {k: pd.read_parquet(WORK / f'daily_{k}.parquet') for k in ['combined', 'binance', 'other_cex', 'coinbase_fx']}
    else:
        ids, fam = prepare_sources(panel)
    if (WORK / 'raw_main.parquet').exists():
        raw = pd.read_parquet(WORK / 'raw_main.parquet')
    else:
        log('computing main calendar-day features, with causal identity inference')
        raw, gate, grid = compute_raw(fam['combined'], panel, ids, args.lag)
        raw.to_parquet(WORK / 'raw_main.parquet', index=False)
        gate.to_parquet(WORK / 'identity_gate.parquet', index=False)
    families = {k: [c for c in cc if c in raw] for k, cc in FAMILIES.items()}
    missing = {k: [c for c in cc if c not in raw] for k, cc in FAMILIES.items()}
    families = {k: cc for k, cc in families.items() if cc}
    if (WORK / 'forecasts.parquet').exists():
        forecast = pd.read_parquet(WORK / 'forecasts.parquet')
    else:
        log('fitting expanding OOF ridge / GBM / 32-16-8 MLP volume models')
        daily = fam['combined'].copy()
        daily['asset_id'] = daily.ucid.map(ids.set_index('ucid').asset_id)
        daily['date'] = pd.to_datetime(daily.date, utc=True)
        targets = volume_targets(daily, challenge=20)
        targets = targets.merge(raw[['asset_id', 'date', PREFIX + 'logdv_ma5']], on=['asset_id', 'date'], how='left', validate='1:1')
        targets['baseline'] = targets[PREFIX + 'logdv_ma5']
        # Keep the paper's eta label (prior actual five days). Direct logdv
        # forecasts are fitted separately, so D-1 volume need not be smuggled
        # into a D-2 information set when reconstructing the level forecast.
        targets = targets.drop(columns=PREFIX + 'logdv_ma5')
        forecast, per_model, logs = walk_forward_volume(raw, targets, families['technical'], refit_every=90,
                    min_dates=130, min_rows=500, iterations=args.iterations, seed=152,
                    max_train_rows=args.max_train, verbose=True)
        forecast.to_parquet(WORK / 'forecasts.parquet', index=False)
        targets.to_parquet(WORK / 'volume_targets.parquet', index=False)
        per_model.to_parquet(WORK / 'volume_oof_predictions.parquet', index=False)
        (WORK / 'volume_fits.json').write_text(json.dumps(logs, indent=2))
        auxiliary_metrics(forecast, targets, raw, 20, per_model).to_csv(WORK / 'volume_oof_metrics.csv', index=False)
    cost = trading_costs(forecast, raw.assign(**{PREFIX + 'market_cap': raw.get(PREFIX + 'market_cap', np.nan)}))
    raw = raw.merge(forecast, on=['asset_id', 'date'], how='left', validate='1:1')
    raw = raw.merge(cost, on=['asset_id', 'date'], how='left', validate='1:1')
    # Suppress all-missing outputs from production feature lists; retain each
    # missing definition and the reason in the manifest instead.
    usable = [c for c in raw if c.startswith(PREFIX) and raw[c].notna().any()]
    families = {k: [c for c in cc if c in usable] for k, cc in FAMILIES.items()}
    families.update(forecasts=[c for c in FORECAST_COLUMNS if c in usable], costs=[c for c in COST_COLUMNS if c in usable])
    for size in [1000, 10000, 100000]:
        c = PREFIX + f'scenario_{size:g}_participation'
        if c in usable: families[f'scenario_{size}'] = [c]
    families = {k: cc for k, cc in families.items() if cc}
    raw = raw[['ucid', 'asset_id', 'date'] + usable]
    raw.to_parquet(WORK / 'raw_with_forecasts.parquet', index=False)
    manifest = {'status': 'BUILT_INFERRED_RECONSTRUCTION', 'built_at': stamp(), 'study': 'E152',
        'authorization': 'User authorized alternative sources and inferred identities/availability on 2026-11-13',
        'point_in_time_verified': False, 'timestamp_causal': True, 'observation_lag_days': args.lag,
        'base_sha256': digest(WORK / 'base_train.parquet'), 'identity_sha256': digest(ids_path),
        'raw_source_sidecars': str(WORK), 'identity_rule': 'unique ticker-to-CMC candidate + expanding causal volatility gate; chain/contract metadata where present',
        'publication_rule': 'Binance documented next-day publication + one full safety day; inferred, not vintage proof',
        'fx_rule': 'Coinbase measured direct USD first, reconstructed Yahoo USD fallback; never assume stablecoin=1',
        'source_limitations': ['Historical vendor revisions remain possible', 'Other CEX REST coverage selects currently listed pairs',
                              'Contract metadata reconstructed today does not prove historical mapping',
                              'Cross-asset market totals cover observed then-known Numerai members, not the entire market'],
        'unavailable_definitions': missing,
        'named_gaps': ['verified historical event catalogs and announcement revisions', 'FDV', 'token/protocol launch dates', 'liquidations',
                       'per-token DEX venue volume', 'fundamental vintages; available fundamental rows are explicitly revised/reconstructed'],
        'auxiliary': {'models': ['ridge', 'gbm', 'mlp_32_16_8'], 'scheme': 'expanding', 'refit_calendar_days': 90,
                      'max_train_rows': args.max_train, 'iterations': args.iterations, 'seed': 152,
                      'targets': 'one-day eta; direct log1p cumulative 5-day and 20-calendar-day volume; publication-time purged'},
        'auxiliary_stability_rule': 'Each model forecast clipped to the min/max target of its own past training sample',
        'warning': 'This is a timestamp-causal reconstructed research dataset, NOT verified historical point-in-time data. No return-alpha claim.'}
    out_manifest = write_dataset(base, raw, args.output, families, manifest)
    for name in ['volume_targets.parquet', 'volume_oof_predictions.parquet', 'volume_fits.json', 'volume_oof_metrics.csv']:
        shutil.copyfile(WORK / name, Path(args.output) / name)
    ids.to_parquet(Path(args.output) / 'identities_inferred.parquet', index=False)
    # Source arms remain raw-only so forecast differences do not confound them.
    for family in ['binance', 'other_cex', 'coinbase_fx']:
        target = Path(str(args.output) + '-' + family.replace('_', '-'))
        if target.exists(): continue
        log('source ablation ' + family)
        x, _, _ = compute_raw(fam[family], panel, ids, args.lag, metrics=False, technical_only=True)
        cc = [c for c in families['technical'] if c in x]
        write_dataset(base, x[['ucid', 'asset_id', 'date'] + cc], target, {'technical': cc}, dict(manifest, source_arm=family))
    report = Path('results/reports/crypto/e152_volumealpha_causal_build')
    report.mkdir(parents=True, exist_ok=True)
    coverage = pd.Series(out_manifest['coverage'], name='coverage').to_frame()
    coverage.to_csv(report / 'coverage.csv')
    html = '<html><head><meta charset="utf-8"><title>E152 dataset complete</title><style>body{font:16px system-ui;margin:40px;max-width:1200px}pre{white-space:pre-wrap}td,th{padding:7px}</style></head><body>'
    html += '<h1>E152 · volume-alpha dataset completed</h1><p>Real historical source data, full Numerai panel retained. User-authorized inferred provenance; not verified vintage point-in-time data. No feature-comparison results are claimed here.</p>'
    html += coverage.to_html() + '<h2>Manifest and limitations</h2><pre>' + escape(json.dumps(out_manifest, indent=2, default=str)) + '</pre></body></html>'
    (report / 'dataset_report.html').write_text(html)
    log('COMPLETE: dataset, OOF forecasts, three source ablations, coverage and provenance report')


def main():
    global WORK
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', default=str(WORK), help='frozen source/feature/model cache; use a fresh directory for a different protocol')
    ap.add_argument('--output', default=str(OUTPUT))
    ap.add_argument('--lag', type=int, default=2, choices=[2, 3])
    ap.add_argument('--iterations', type=int, default=50)
    ap.add_argument('--max-train', type=int, default=50000)
    args = ap.parse_args()
    WORK = Path(args.work)
    run(args)


if __name__ == '__main__':
    main()
