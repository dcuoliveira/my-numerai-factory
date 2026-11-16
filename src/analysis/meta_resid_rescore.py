"""
Residualise a finished run's forecasts on the lagged meta model, then score
them on the payable 20-day label.

    from src.analysis.meta_resid_rescore import Residualiser
    res = Residualiser()
    table = res.rescore("ridge_glmnet", "e117_w130")      # per-period corr, mmc

The transform is src.prediction.meta_resid -- the module live uses through
blend.MetaResidual -- at its deployed settings: STRENGTH 0.5, meta model at
least LAG_DAYS = 5 calendar days old. So a residualised row here means what a
live "resid": 0.5 slot means, and nothing about it can drift from live.

What this adds over calling meta_resid.lagged_meta per period is a cache:
that function scans the whole 176,400-row meta file for every date it is
asked about, which is ~7,000 scans for a 14-arm table. The cache does the
date-to-snapshot lookup once per date through meta_resid.snapshot_for, and
check_against_live() asserts it returns exactly what lagged_meta returns.
"""

import os

import numpy as np
import pandas as pd

from src.analysis.cross_target import load_scoring_frame, score_on_payable_target
from src.prediction import meta_resid

META_FILE = os.path.join("data", "crypto", "v2.0", "historical_meta_models.parquet")


class Residualiser(object):

    def __init__(self, strength=meta_resid.STRENGTH, lag=meta_resid.LAG_DAYS):
        self.strength, self.lag = strength, lag
        meta = pd.read_parquet(META_FILE).reset_index()[["date", "symbol", "prediction"]]
        meta["date"] = pd.to_datetime(meta["date"])
        self.meta = meta
        self.meta_dates = meta["date"].unique()
        self._days = {}            # snapshot date -> Series symbol -> prediction
        self._snapshot = {}        # period -> snapshot date or None
        self.extras = load_scoring_frame()
        self.passthrough = 0

    def _lagged(self, symbols, date):
        if date not in self._snapshot:
            self._snapshot[date] = meta_resid.snapshot_for(self.meta_dates, date, self.lag)
        snapshot = self._snapshot[date]
        if snapshot is None:
            return np.full(len(symbols), np.nan), None
        if snapshot not in self._days:
            day = self.meta[self.meta["date"] == snapshot].drop_duplicates("symbol", keep="last")
            self._days[snapshot] = day.set_index("symbol")["prediction"]
        return self._days[snapshot].reindex(list(symbols)).to_numpy(dtype="float64"), snapshot

    def check_against_live(self, dates, symbols):
        """The cache must return exactly what meta_resid.lagged_meta returns."""
        for date in dates:
            mine, snap_mine = self._lagged(symbols, date)
            live, snap_live = meta_resid.lagged_meta(self.meta, symbols, date, self.lag)
            assert snap_mine == snap_live, (date, snap_mine, snap_live)
            assert np.array_equal(np.isnan(mine), np.isnan(live)), date
            assert np.allclose(mine[~np.isnan(mine)], live[~np.isnan(live)]), date

    def residualised(self, forecast):
        """forecast: frame indexed (symbol, date) with `prediction`. Returns rank(r)."""
        out, self.passthrough = [], 0
        for date, block in forecast.groupby(level="date", sort=True):
            symbols = block.index.get_level_values("symbol")
            meta, _ = self._lagged(symbols, pd.Timestamp(date))
            r, beta = meta_resid.residualise(block["prediction"].to_numpy(), meta, self.strength)
            if np.isnan(beta):
                self.passthrough += 1
            out.append(pd.Series(pd.Series(r).rank(pct=True).to_numpy(), index=block.index))
        return pd.concat(out)

    def rescore(self, model, run, residualise=True):
        path = os.path.join("results", "forecast", "crypto", model, run + ".csv")
        forecast = pd.read_csv(path, usecols=["symbol", "date", "prediction"],
                               parse_dates=["date"]).set_index(["symbol", "date"])
        if residualise:
            prediction = self.residualised(forecast)
        else:
            prediction = forecast.groupby(level=1)["prediction"].rank(pct=True)
        table, _ = score_on_payable_target(prediction, self.extras)
        table = table[["corr", "mmc"]].copy()
        table.index = pd.to_datetime(table.index).strftime("%Y-%m-%d")
        return table
