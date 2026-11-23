"""JACI_01 live pre-stage (E216): momentum3 TSFM design-C forecast for every live coin.

JACI_01 is E214's FIXED_s0.75 -- the equal-rank blend of four E206 design-C
streams (Chronos-2 rolling, t0 cross_section, t0 rolling, Timestral
cross_section), residualised at 0.75 on the meta model lagged >= 5 days. It is
not a sklearn estimator and cannot be fitted inside run_round: it needs fresh
spot prices, own-history momentum features, and a GPU pass through three frozen
foundation models. So it runs as a pre-stage, before the 09:05 round, and leaves
one file that run_round reads:

    models/live/jaci/forecasts.parquet   date, symbol, ucid, 4 streams, jaci_raw

The residualisation is NOT applied here. run_round attaches the lagged meta
model to the live panel every round, and blend.JaciForecast residualises
jaci_raw against it, so the meta snapshot is the one the round itself uses.

Stages (python3 -m src.live.jaci <stage>), `all` runs them in order:

  prices    refresh data/crypto/live/jaci_prices.parquet (a copy of the spot
            store) -- each symbol re-fetched from the ONE source/ticker it
            already has, so the E157 identity cohort is unchanged
  panel     frozen E206 panel + rows after 2026-02-19 (train, live history,
            live, and weekday "proxy" dates up to today on the last live
            universe), with the 33 momentum3 features and E205 coverage
            recomputed by the E205 functions (asset_momentum.signal,
            prior_bins, attach, price_context)
  extract   ssh to the GPU box and run src.live.jaci_extract per model
  forecast  states -> tsfm_momentum_study.normalize -> design-C query kernel
            -> the FROZEN E206 per-asset dual-ridge heads -> equal-rank blend

The heads are E206's final fits (training rows <= 2024-11-27). No refit: the
E214/E215 backtest that justified the slot predicted 2025-01..2026-02 with these
same heads, and live is that protocol carried forward.

Gates (`parity`): the recomputed features/coverage equal the frozen panel; the
recomputed normalization equals the stored final normalized arrays; the kernel
+ head path reproduces E206's stored test predictions; the blend + 0.75
residualisation reproduces E214's FIXED_s0.75; re-extracted states match the
stored float16 states. Every run of `forecast` re-checks the normalization gate
on the frozen rows, so a drifted input fails loudly instead of quietly.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis import asset_momentum as M          # noqa: E402
from src.analysis import tsfm_momentum_study as N     # noqa: E402

TARGET = "target_binned_return_20"
FROZEN_END = pd.Timestamp("2026-02-19")
E206_ROOT = REPO / "results/reports/crypto/e206_momentum3_tsfm"
E206_SCR = Path("/scratch/danielco/e206_momentum3_tsfm")
STATE_ROOT = Path(os.environ.get("JACI_STATE_ROOT", "/scratch/danielco/jaci_live"))
REPORT_ROOT = REPO / "results/reports/crypto/e216_jaci_live"
LIVE_DIR = REPO / "data/crypto/live"
SPOT = REPO / "data/crypto/spot/ohlcv.parquet"
PRICES = LIVE_DIR / "jaci_prices.parquet"
FORECASTS = REPO / "models/live/jaci/forecasts.parquet"
TRAIN = REPO / "data/crypto/v2.0/train.parquet"
META = REPO / "data/crypto/v2.0/historical_meta_models.parquet"
PYTHON = "/home/danielco/miniconda3/bin/python3"
GPU_HOST = os.environ.get("JACI_GPU_HOST", "deepfive")
VENVS = {"chronos2": "/scratch/danielco/chronos2/venv/bin/python",
         "t0": "/scratch/danielco/tsfm2/venv_tt/bin/python",
         "timestral": "/scratch/danielco/tsfm2/venv_tt/bin/python"}
STUDY = "e206_crypto-v2_0-e205-ts-momentum3_%s_%s_C_r01"
# (label, model, standardization) in E213's member order for e213_129.
STREAMS = [("CR", "chronos2", "rolling"), ("TC", "t0", "cross_section"),
           ("TR", "t0", "rolling"), ("SC", "timestral", "cross_section")]
STRENGTH = 0.75
REFRESH_OVERLAP_DAYS = 3


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".%s_%d.tmp" % (socket.gethostname(), os.getpid()))
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


# --------------------------------------------------------------------------- prices
def _yahoo(tickers, start, end):
    import yfinance as yf
    out = []
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        data = yf.download(chunk, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                           interval="1d", auto_adjust=False, progress=False, group_by="ticker",
                           threads=True)
        if data is None or data.empty:
            continue
        for t in chunk:
            try:
                frame = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            frame = frame.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
            frame = frame[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            out.append(frame.assign(provider_ticker=t))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _binance(ticker, start):
    import requests
    rows, t0 = [], int(start.timestamp()) * 1000
    while True:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": ticker, "interval": "1d", "startTime": t0, "limit": 1000},
                         timeout=20)
        if r.status_code != 200:
            return None
        part = r.json()
        rows += part
        if len(part) < 1000:
            break
        t0 = part[-1][0] + 86400000
    if not rows:
        return None
    frame = pd.DataFrame([(pd.Timestamp(x[0], unit="ms"), *map(float, x[1:6])) for x in rows],
                         columns=["date", "open", "high", "low", "close", "volume"])
    return frame


def _exchange(source, ticker, start):
    from src.data_mani import fetch_exchange_spot as X
    suffix = X.TICKER_FMT[source].format("")
    base = ticker[:-len(suffix)] if ticker.endswith(suffix) else ticker
    frame = X.VENUES[source](base, pd.Timestamp.utcnow().tz_localize(None))
    if frame is None:
        return None
    return frame[frame.date >= start]


def refresh_prices(universe_symbols, today=None):
    """Append complete daily bars to the JACI price store, per symbol from its own source."""
    today = pd.Timestamp(today or pd.Timestamp.utcnow().tz_localize(None).normalize())
    if not PRICES.exists():
        shutil.copyfile(SPOT, PRICES)
        log("initialised", PRICES, "from", SPOT)
    store = pd.read_parquet(PRICES)
    identity = store.groupby("symbol").agg(ucid=("ucid", "first"), source=("source", "first"),
                                           ticker=("provider_ticker", "first"), last=("date", "max"))
    identity = identity[identity.index.isin(set(universe_symbols))]
    identity = identity[identity.source != "binance_archive_busd"]
    fetched, failures = [], []
    yahoo = identity[identity.source == "yahoo"]
    if len(yahoo):
        start = yahoo["last"].min() - pd.Timedelta(days=REFRESH_OVERLAP_DAYS)
        got = _yahoo(sorted(yahoo.ticker.unique()), start, today + pd.Timedelta(days=1))
        if len(got):
            got = got.merge(yahoo.reset_index()[["symbol", "ticker", "last"]],
                            left_on="provider_ticker", right_on="ticker")
            got = got[got.date >= got["last"] - pd.Timedelta(days=REFRESH_OVERLAP_DAYS)]
            fetched.append(got.drop(columns=["ticker", "last"]).assign(source="yahoo"))
        missing = set(yahoo.index) - set(got.symbol if len(got) else [])
        failures += [(s, "yahoo") for s in sorted(missing)]
    for symbol, row in identity[identity.source != "yahoo"].iterrows():
        start = row["last"] - pd.Timedelta(days=REFRESH_OVERLAP_DAYS)
        try:
            frame = _binance(row.ticker, start) if row.source == "binance" \
                else _exchange(row.source, row.ticker, start)
        except Exception as error:          # one venue failing must not stop the rest
            frame = None
            log("  %s/%s: %s" % (symbol, row.source, error))
        if frame is None or not len(frame):
            failures.append((symbol, row.source))
            continue
        fetched.append(frame.assign(symbol=symbol, provider_ticker=row.ticker, source=row.source))
    if not fetched:
        raise RuntimeError("price refresh fetched nothing (%d failures)" % len(failures))
    new = pd.concat(fetched, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None).dt.normalize()
    new = new[new.date < today]     # only COMPLETE UTC daily bars
    new["ucid"] = new.symbol.map(identity.ucid)
    new = new.drop_duplicates(["symbol", "date"], keep="last")
    for c in ["open", "high", "low", "close", "volume"]:
        new[c] = pd.to_numeric(new[c], errors="coerce")
    first_new = new.groupby("symbol").date.min()
    drop = store.symbol.map(first_new)
    keep = store[~(drop.notna() & (store.date >= drop))]
    merged = pd.concat([keep, new[store.columns]], ignore_index=True)
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    assert not merged.duplicated(["symbol", "date"]).any()
    assert merged.groupby("symbol").source.nunique().max() == 1
    atomic_parquet(merged, PRICES)
    report = dict(refreshed=int(new.symbol.nunique()), bars=len(new), failures=failures,
                  through=str(new.date.max().date()), today=str(today.date()))
    save_json(report, STATE_ROOT / "prices_last_run.json")
    log("prices: %d symbols refreshed through %s, %d failures"
        % (report["refreshed"], report["through"], len(failures)))
    return report


# --------------------------------------------------------------------------- panel
@njit(cache=True)
def prior_bins_tail(values, start, history=252, minimum=126):
    """asset_momentum.prior_bins, evaluated only for rows >= start (same arithmetic)."""
    out = np.full(values.shape, np.nan)
    for c in range(values.shape[1]):
        for t in range(start, values.shape[0]):
            v = values[t, c]
            if not np.isfinite(v):
                continue
            n, below, equal = 0, 0, 0
            for j in range(max(0, t - history), t):
                other = values[j, c]
                if np.isfinite(other):
                    n += 1
                    below += other < v
                    equal += other == v
            if n >= minimum:
                p = (below + .5 * equal) / n
                out[t, c] = .25 * (int(p > .05) + int(p > .25) + int(p > .75) + int(p > .95))
    return out


def feature_name(family, window):
    return "feature_%s_ts_%dd" % (family, window)


def universe_rows(today):
    """Every (symbol, ucid, date) the forecast may be asked for, after the frozen panel."""
    train = pd.read_parquet(TRAIN, columns=["date", "ucid", TARGET]).reset_index()
    train = train[train.date > FROZEN_END]
    live = pd.read_parquet(LIVE_DIR / "live.parquet", columns=["date", "ucid"]).reset_index()
    history = pd.read_parquet(LIVE_DIR / "live_history.parquet").reset_index()[["symbol", "date"]]
    ucid = pd.concat([pd.read_parquet(TRAIN, columns=["date", "ucid"]).reset_index()
                      .sort_values("date")[["symbol", "ucid"]], live[["symbol", "ucid"]]])
    ucid = ucid.drop_duplicates("symbol", keep="last").set_index("symbol").ucid
    history = history[history.date > train.date.max()].assign(ucid=lambda d: d.symbol.map(ucid))
    live = live[live.date > max(train.date.max(), history.date.max())]
    known = max(train.date.max(), history.date.max(), live.date.max() if len(live) else train.date.min())
    proxy_dates = pd.bdate_range(known + pd.Timedelta(days=1), today)
    last_universe = live if len(live) else history[history.date == history.date.max()]
    proxy = pd.concat([last_universe[["symbol", "ucid"]].assign(date=d) for d in proxy_dates]) \
        if len(proxy_dates) else pd.DataFrame(columns=["symbol", "ucid", "date"])
    parts = [train.assign(source="train"), history.assign(source="live_history"),
             live.assign(source="live"), proxy.assign(source="proxy")]
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[pd.to_numeric(rows.ucid, errors="coerce").notna()]
    rows["ucid"] = rows.ucid.astype(int)
    rows = rows.drop_duplicates(["ucid", "date"], keep="first")
    if TARGET not in rows:
        rows[TARGET] = np.nan
    return rows[["symbol", "ucid", "date", TARGET, "source"]]


def momentum_features(rows, prices, first_date):
    """33 momentum3 features + E205 coverage for rows with date >= first_date.

    `rows` must be the FULL key panel: price_context's identity rule (a symbol
    must map to one ucid across the whole panel) is panel-wide, as in E205.
    """
    close, columns, cohort = M.price_context(rows, prices)
    need = (rows.date >= first_date).to_numpy()
    sub = rows.loc[need].reset_index(drop=True)
    sub_cols = columns[need]
    start = int(np.searchsorted(close.index.values, np.datetime64(first_date - pd.Timedelta(days=14))))
    covered = cohort[need].copy()
    out = {}
    for family in M.FAMILIES:
        for window in M.WINDOWS:
            raw = M.signal(close, family, window - 1)
            ts = prior_bins_tail(raw.to_numpy(), start)
            frame = pd.DataFrame(ts[start:], index=raw.index[start:], columns=raw.columns)
            values = M.attach(sub, frame, sub_cols)
            covered &= np.isfinite(values)
            if family in ("price_sma_ratio", "ewma_return", "price_ema_ratio"):
                out[feature_name(family, window)] = values.astype(np.float32)
    features = pd.DataFrame(out)
    features.loc[~covered, :] = np.nan
    order = [feature_name(f, w) for f in ("price_sma_ratio", "ewma_return", "price_ema_ratio")
             for w in M.WINDOWS]
    return pd.concat([sub, features[order]], axis=1).assign(covered=covered)


def load_frozen(columns=None):
    """The E206 panel with ucid as int (it is stored as str; the heads use int)."""
    frozen = pd.read_parquet(E206_ROOT / "panel.parquet", columns=columns)
    frozen["ucid"] = frozen.ucid.astype(int)
    return frozen


def build_panel(today=None, parity_from=None):
    today = pd.Timestamp(today or pd.Timestamp.utcnow().tz_localize(None).normalize())
    frozen = load_frozen()
    feats = [c for c in frozen if c.startswith("feature_")]
    new = universe_rows(today)
    keys = pd.concat([frozen[["symbol", "ucid", "date"]], new[["symbol", "ucid", "date"]]],
                     ignore_index=True)
    prices = pd.read_parquet(PRICES)
    first = pd.Timestamp(parity_from) if parity_from else FROZEN_END + pd.Timedelta(days=1)
    log("panel: %d new rows (%s), features from %s" % (len(new), new.source.value_counts().to_dict(),
                                                       first.date()))
    built = momentum_features(keys, prices, first)
    check = None
    if parity_from:
        old = built[built.date <= FROZEN_END].merge(
            frozen[["ucid", "date", "covered"] + feats], on=["ucid", "date"], suffixes=("", "_frozen"))
        diff_cov = int((old.covered != old.covered_frozen).sum())
        diff_val = int(sum((~np.isclose(old[f], old[f + "_frozen"], equal_nan=True)).sum() for f in feats))
        check = dict(rows=len(old), coverage_mismatches=diff_cov, value_mismatches=diff_val,
                     passed=diff_cov == 0 and diff_val == 0, from_date=str(first.date()))
        save_json(check, REPORT_ROOT / "parity_features.json")
        log("feature parity on %d frozen rows: coverage %d, values %d mismatches"
            % (len(old), diff_cov, diff_val))
    fresh = built[built.date > FROZEN_END].merge(new[["ucid", "date", TARGET, "source"]],
                                                 on=["ucid", "date"], how="left")
    frozen_part = frozen[["symbol", "ucid", "date", TARGET] + feats + ["covered", "row"]].rename(
        columns={"row": "frozen_row"}).assign(source="frozen")
    fresh = fresh.assign(frozen_row=-1)[frozen_part.columns]
    panel = pd.concat([frozen_part, fresh], ignore_index=True)
    panel["row"] = np.arange(len(panel))
    grid = np.unique(panel.date.to_numpy())
    panel["step"] = np.searchsorted(grid, panel.date.to_numpy())
    assert (panel.step[panel.frozen_row >= 0].to_numpy()
            == frozen.step.to_numpy()).all(), "frozen steps moved"
    assert not panel.duplicated(["ucid", "date"]).any()
    atomic_parquet(panel, STATE_ROOT / "panel.parquet")
    summary = dict(rows=len(panel), new_rows=int((panel.frozen_row < 0).sum()),
                   new_covered=int(((panel.frozen_row < 0) & panel.covered).sum()),
                   last_date=str(panel.date.max().date()),
                   covered_by_new_date=panel[panel.frozen_row < 0].groupby("date").covered.sum()
                   .tail(10).astype(int).rename(lambda d: str(d.date())).to_dict(), feature_parity=check)
    save_json(summary, STATE_ROOT / "panel_last_run.json")
    log("panel: %d rows, %d new covered rows through %s"
        % (len(panel), summary["new_covered"], summary["last_date"]))
    return panel


# --------------------------------------------------------------------------- extract
def extract(models=("chronos2", "t0", "timestral"), devices=("cuda:0", "cuda:1"), dates=None,
            verify=0):
    """Run the GPU worker on GPU_HOST; two models at a time, one per card."""
    pending = list(models)
    while pending:
        batch, pending = pending[:len(devices)], pending[len(devices):]
        procs = []
        for model, device in zip(batch, devices):
            cmd = ("cd %s && JACI_STATE_ROOT=%s %s -m src.live.jaci_extract --model %s --device %s%s%s"
                   % (REPO, STATE_ROOT, VENVS[model], model, device,
                      (" --dates " + " ".join(dates)) if dates else "",
                      (" --verify %d" % verify) if verify else ""))
            target = ["bash", "-lc", cmd] if socket.gethostname() == GPU_HOST else \
                ["ssh", "-o", "BatchMode=yes", GPU_HOST, cmd]
            log("extract:", model, "on", GPU_HOST, device)
            procs.append((model, subprocess.Popen(target, stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT, text=True)))
        for model, proc in procs:
            tail = []
            for line in proc.stdout:
                tail = (tail + [line.rstrip()])[-20:]
                if "JACI_EXTRACT_DONE" in line or "Error" in line:
                    log(model, line.rstrip())
            if proc.wait() != 0:
                raise RuntimeError("extraction failed for %s:\n%s" % (model, "\n".join(tail)))


# --------------------------------------------------------------------------- forecast
def assemble_states(model, panel):
    width = json.loads((E206_SCR / "embeddings" / model / "extractor.json").read_text())["width"]
    h = np.full((len(panel), width), np.nan, np.float16)
    frozen = panel.frozen_row.to_numpy() >= 0
    stored = np.load(E206_SCR / "embeddings" / model / "H_rows.npy", mmap_mode="r")
    h[frozen] = stored[panel.frozen_row.to_numpy()[frozen]]
    fresh = panel[~frozen]
    found = 0
    for ucid, g in fresh.groupby("ucid"):
        path = STATE_ROOT / "states" / model / ("asset_%d.npz" % ucid)
        if not path.exists():
            continue
        z = np.load(path)
        lookup = pd.Series(np.arange(len(z["dates"])),
                           index=pd.DatetimeIndex(z["dates"].astype("datetime64[ns]")))
        position = lookup.reindex(pd.DatetimeIndex(g.date)).to_numpy()
        ok = np.isfinite(position)
        h[g.index.to_numpy()[ok]] = z["states"][position[ok].astype(int)]
        found += int(ok.sum())
    valid = np.isfinite(h).all(axis=1) & panel.covered.to_numpy()
    return h, valid, found


def stream_predictions(label, model, norm, panel, h, valid, query, check_frozen=True):
    """Normalized states -> design-C query kernel -> frozen per-asset heads."""
    tag = STUDY % (model, norm)
    params = json.loads((E206_ROOT / "fits" / tag / "ridge_parameters.json").read_text())
    window = int(params["window"])
    vocab = np.array(params["design"]["vocabulary"], dtype=int)
    assets = panel.ucid.to_numpy()
    dates = panel.date.to_numpy()
    z, _ = N.normalize(h, assets, dates, valid, np.zeros(len(panel), bool), norm, window)
    frozen = panel.frozen_row.to_numpy() >= 0
    gate = None
    if check_frozen:
        stored = np.load(E206_SCR / "normalizations" / model / norm / ("final_%d" % window) / "H.npy",
                         mmap_mode="r")
        ref = stored[panel.frozen_row.to_numpy()[frozen]]
        mine = z[frozen]
        same_nan = bool((np.isnan(ref) == np.isnan(mine)).all())
        err = float(np.nanmax(np.abs(ref - mine)))
        gate = dict(stream=label, window=window, rows=int(frozen.sum()), nan_pattern_equal=same_nan,
                    max_abs_diff=err, passed=same_nan and err <= 1e-5)
        if not gate["passed"]:
            raise RuntimeError("normalization gate failed for %s: %s" % (label, gate))
    finite = np.isfinite(z).all(axis=1)
    steps = panel.step.to_numpy()
    n_frozen = int(steps[frozen].max()) + 1
    query = query & finite
    qsteps = np.unique(steps[query])
    qi = pd.Series(np.arange(len(qsteps)), index=qsteps)
    kernel = np.zeros((len(qsteps), n_frozen))
    by_asset = pd.Series(np.arange(len(panel))).groupby(assets).indices
    for b in vocab:
        idx = by_asset.get(b)
        if idx is None:
            continue
        idx = idx[finite[idx]]
        train_cols = idx[steps[idx] < n_frozen]
        q_rows = idx[query[idx]]
        if not len(train_cols) or not len(q_rows):
            continue
        block = np.asarray(z[q_rows], float) @ np.asarray(z[train_cols], float).T
        # One asset has at most one row per step, so these indices never repeat.
        kernel[np.ix_(qi[steps[q_rows]].to_numpy(), steps[train_cols])] += block
    head_dir = E206_SCR / "ridge_parameters" / tag / ("w%d_final" % window)
    frozen_step = panel.loc[frozen].set_index("frozen_row").step
    out = np.full(len(panel), np.nan)
    for record in params["assets"]:
        a = int(record["asset"])
        idx = by_asset.get(a)
        if idx is None:
            continue
        rows = idx[query[idx]]
        if not len(rows):
            continue
        p = np.load(head_dir / ("asset_%d.npz" % a))
        ft = frozen_step.reindex(p["training_rows"]).to_numpy()
        kqt = kernel[np.ix_(qi[steps[rows]].to_numpy(), ft)]
        centred = kqt - p["kernel_train_mean"][None, :] - kqt.mean(axis=1)[:, None] + float(p["kernel_grand_mean"])
        out[rows] = centred @ p["dual_weights"] + float(p["intercept"])
    return out, gate


def rank_blend(panel, streams):
    """E213 fixed blend: per date, mean of each stream's rank over rows where all exist."""
    from src.live.submit import rank_normalise
    values = np.column_stack(streams)
    common = np.isfinite(values).all(axis=1)
    out = np.full(len(panel), np.nan)
    for _, idx in panel.groupby("date").indices.items():
        rows = idx[common[idx]]
        if len(rows):
            out[rows] = np.mean([rank_normalise(values[rows, i]) for i in range(values.shape[1])],
                                axis=0)
    return out


def forecast(gates=True):
    panel = pd.read_parquet(STATE_ROOT / "panel.parquet")
    panel["ucid"] = panel.ucid.astype(int)
    fresh = (panel.frozen_row < 0).to_numpy()
    cols, report = {}, dict(streams=[])
    for model in sorted({m for _, m, _ in STREAMS}):
        h, valid, found = assemble_states(model, panel)
        log("states %s: %d fresh states, %d fresh valid rows" % (model, found, int((valid & fresh).sum())))
        for label, m, norm in STREAMS:
            if m != model:
                continue
            pred, gate = stream_predictions(label, m, norm, panel, h, valid, fresh, check_frozen=gates)
            cols[label] = pred
            report["streams"].append(dict(stream=label, predictions=int(np.isfinite(pred).sum()),
                                          normalization_gate=gate))
            log("stream %s: %d fresh predictions" % (label, int(np.isfinite(pred).sum())))
        del h
    streams = [cols[label] for label, _, _ in STREAMS]
    panel["jaci_raw"] = rank_blend(panel, streams)
    for label, _, _ in STREAMS:
        panel[label] = cols[label]
    out = panel.loc[fresh, ["date", "symbol", "ucid", "source", TARGET]
                    + [s[0] for s in STREAMS] + ["jaci_raw"]]
    out = out[np.isfinite(out[[s[0] for s in STREAMS]]).any(axis=1)]
    atomic_parquet(out, FORECASTS)
    by_date = out.groupby("date").jaci_raw.apply(lambda s: int(s.notna().sum()))
    report.update(rows=len(out), dates=int(by_date.size), last_date=str(out.date.max().date()),
                  covered_last_dates=by_date.tail(8).rename(lambda d: str(d.date())).to_dict(), written=str(FORECASTS),
                  finished=pd.Timestamp.now().isoformat(), host=socket.gethostname())
    save_json(report, STATE_ROOT / "forecast_last_run.json")
    log("forecast: %d rows over %d dates -> %s; last dates %s"
        % (len(out), by_date.size, FORECASTS, by_date.tail(3).to_dict()))
    return out


# --------------------------------------------------------------------------- parity
def parity_heads():
    """Kernel + head path on the FROZEN panel reproduces E206's stored test predictions."""
    frozen = load_frozen()
    panel = frozen[["symbol", "ucid", "date", "covered", "row", "step", "test"]].rename(
        columns={"row": "frozen_row"})
    panel = panel.assign(row=panel.frozen_row)
    results = []
    for model in sorted({m for _, m, _ in STREAMS}):
        h, valid, _ = assemble_states(model, panel.assign(covered=frozen.covered))
        for label, m, norm in STREAMS:
            if m != model:
                continue
            pred, gate = stream_predictions(label, m, norm, panel, h, valid, panel.test.to_numpy())
            ref = pd.read_parquet(E206_ROOT / "fits" / (STUDY % (m, norm)) / "predictions.parquet",
                                  columns=["row", "prediction"]).set_index("row").prediction
            mine = pd.Series(pred, index=panel.frozen_row).reindex(ref.index)
            same = bool((mine.isna() == ref.isna()).all())
            err = float(np.nanmax(np.abs(mine - ref)))
            results.append(dict(stream=label, rows=int(ref.notna().sum()), nan_pattern_equal=same,
                                max_abs_diff=err, passed=same and err < 1e-8, normalization=gate))
            log("head parity %s: %s" % (label, results[-1]))
    save_json(results, REPORT_ROOT / "parity_heads.json")
    return results


def parity_blend():
    """Blend + 0.75 lagged-meta residualisation reproduces E214's FIXED_s0.75."""
    from src.prediction import meta_resid as MR
    e214 = pd.read_parquet(REPO / "results/reports/crypto/e214_meta_variants/variant_predictions.parquet",
                           columns=["row", "symbol", "date", "FIXED_s0.75"])
    preds = []
    for label, m, norm in STREAMS:
        ref = pd.read_parquet(E206_ROOT / "fits" / (STUDY % (m, norm)) / "predictions.parquet",
                              columns=["row", "prediction"]).set_index("row").prediction
        preds.append(ref.reindex(e214.row).to_numpy())
    raw = rank_blend(e214, preds)
    meta = pd.read_parquet(META).reset_index()[["symbol", "date", "prediction"]]
    mine = np.full(len(e214), np.nan)
    for date, idx in e214.groupby("date").indices.items():
        lagged, _ = MR.lagged_meta(meta, e214.loc[idx, "symbol"], date)
        mine[idx], _ = MR.residualise(raw[idx], lagged, STRENGTH)
    err = float(np.nanmax(np.abs(mine - e214["FIXED_s0.75"].to_numpy())))
    result = dict(rows=len(e214), max_abs_diff=err, passed=err < 1e-9)
    save_json(result, REPORT_ROOT / "parity_blend.json")
    log("blend parity:", result)
    return result


def parity_states(models=("chronos2", "t0", "timestral")):
    """Compare re-extracted frozen origins (jaci_extract --verify) with E206 H_rows."""
    results = []
    for model in models:
        path = STATE_ROOT / "states" / model / "parity_probe.json"
        if not path.exists():
            results.append(dict(model=model, passed=False, note="no probe file"))
            continue
        probes = json.loads(path.read_text())
        stored = np.load(E206_SCR / "embeddings" / model / "H_rows.npy", mmap_mode="r")
        errs = [float(np.max(np.abs(np.asarray(p["state"], np.float32)
                                    - stored[p["row"]].astype(np.float32)))) for p in probes]
        scale = [float(np.max(np.abs(stored[p["row"]].astype(np.float32)))) for p in probes]
        rel = max(e / max(s, 1e-6) for e, s in zip(errs, scale)) if errs else np.nan
        results.append(dict(model=model, probes=len(probes), max_abs_diff=max(errs) if errs else None,
                            max_rel_diff=rel, passed=bool(errs) and rel < 1e-2))
        log("state parity:", results[-1])
    save_json(results, REPORT_ROOT / "parity_states.json")
    return results


def forward_check():
    """Score the live path on resolved dates after the frozen panel (not used for any choice)."""
    from src.prediction import meta_resid as MR
    sys.path.insert(0, str(REPO / "experiments"))
    from e150_finish_scoring import matrix_scores
    out = pd.read_parquet(FORECASTS)
    out = out[out[TARGET].notna() & out.jaci_raw.notna()]
    meta = pd.read_parquet(META).reset_index()[["symbol", "date", "prediction"]]
    true_meta = meta.set_index(["symbol", "date"]).prediction
    rows = []
    for date, g in out.groupby("date"):
        lagged, _ = MR.lagged_meta(meta, g.symbol, date)
        resid, _ = MR.residualise(g.jaci_raw.to_numpy(), lagged, STRENGTH)
        m = true_meta.reindex(pd.MultiIndex.from_arrays([g.symbol, g.date])).to_numpy()
        ok = np.isfinite(m) & np.isfinite(g[TARGET].to_numpy())
        if ok.sum() < 10:
            continue
        corr, mmc = matrix_scores(resid[ok, None], g[TARGET].to_numpy()[ok], m[ok])
        rows.append(dict(date=date, n=int(ok.sum()), corr=float(corr[0]), mmc=float(mmc[0])))
    daily = pd.DataFrame(rows)
    pay = .1 * daily["corr"] + daily["mmc"]
    summary = dict(dates=len(daily), first=str(daily.date.min().date()), last=str(daily.date.max().date()),
                   mean_assets=float(daily.n.mean()), corr=float(daily["corr"].mean()),
                   mmc=float(daily["mmc"].mean()), payout=float(pay.mean()),
                   payout_sharpe=float(pay.mean() / pay.std(ddof=0)))
    daily.to_csv(REPORT_ROOT / "forward_daily.csv", index=False)
    save_json(summary, REPORT_ROOT / "forward_check.json")
    log("forward check:", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=["prices", "panel", "extract", "forecast", "all",
                                     "parity", "forward"])
    p.add_argument("--today", default=None, help="UTC date to treat as today (default: now)")
    p.add_argument("--parity-from", default=None,
                   help="panel: also rebuild frozen rows from this date and compare")
    p.add_argument("--verify", type=int, default=0, help="extract: frozen origins to re-extract")
    p.add_argument("--no-gates", action="store_true", help="forecast: skip the frozen-row gate")
    a = p.parse_args()
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("prices", "all"):
        symbols = set(universe_rows(pd.Timestamp(a.today) if a.today else
                                    pd.Timestamp.utcnow().tz_localize(None).normalize()).symbol)
        refresh_prices(symbols, a.today)
    if a.stage in ("panel", "all"):
        build_panel(a.today, a.parity_from)
    if a.stage in ("extract", "all"):
        extract(verify=a.verify)
    if a.stage in ("forecast", "all"):
        forecast(gates=not a.no_gates)
    if a.stage == "parity":
        parity_blend()
        parity_heads()
        parity_states()
    if a.stage == "forward":
        forward_check()


if __name__ == "__main__":
    main()
