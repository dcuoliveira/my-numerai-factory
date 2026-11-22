"""E203: retrain CU_SA_IA_BO_02, CU_SA_IA_BO_MA_01 and CU_SA_IA_BO_MA_R_01 from
scratch with the nine replacement volatility measures in place of the shipped
feature_volatility_20d / _60d.

Every leaf under the three slots (SACI_04, CURUPIRA_04, CURUPIRA_05 and its five
members, CURUPIRA_05_60 and its five members, IA_01, IA_02, MA_FENC, MA_ALLFEAT)
is refitted with the live protocol (src.live.refit.fit_one: 50-draw purged-CV
search or the slot's pinned params) on each of the eleven 130-date development
folds of the volatility-replacement study, under three arms that follow the
Numerai Crypto missing-value standard:

    B_all   deployed feature builders, full universe
    B_cov   deployed feature builders, rows where all nine new measures exist ("old vol")
    F_cov   replaced feature builders, the same rows                         ("new vol")

Feature sets are rebuilt from the panel, so every statistic of v2.0 that reads a
volatility column is recomputed on the new block: the causal per-symbol means
(fenccausal), the rolling means (allfeat_pp), the 20d-60d volatility spread
(replaced by the 40d-120d spreads of the two families measured at both windows)
and the three volatility products (whose volatility operand becomes the mean of
the nine new measures). The one-hot vocabulary is taken from each arm's own
historical rows.

Slots are rank-averaged inside each arm exactly as src.live.blend does, with the
E107 meta residualisation (lag 5 days, strength 0.5) on `resid` legs. Covered slot
predictions are then routed into the B_all slot's rank positions, so uncovered
coins carry identical ranks in both columns of the comparison.

    python3 experiments/e203_volrep_slots.py validate
    python3 experiments/e203_volrep_slots.py worker --units 0-32 --search-jobs 8
    python3 experiments/e203_volrep_slots.py assemble
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.analysis.volatility_experiment import digest, rank_columns, route, score_date, check_scoring  # noqa: E402
from src.data_mani.build_derived import build_interactions, build_spreads, difference  # noqa: E402
from src.data_mani.build_feature_encoding import causal_feature_means  # noqa: E402
from src.data_mani.build_volatility_replacement import NEW, REMOVED  # noqa: E402
from src.data_mani.utils import fill_missing_features  # noqa: E402
from src.live import features as F  # noqa: E402
from src.live import refit, registry  # noqa: E402
from src.prediction import meta_resid  # noqa: E402
from src.prediction.functions import filter_features  # noqa: E402

OUT = ROOT / "results" / "e203_volrep_slots"
PANEL = ROOT / "data/crypto/v2.0/train.parquet"
META = ROOT / "data/crypto/v2.0/historical_meta_models.parquet"
REPLACEMENT = ROOT / "results/volatility_replacement/v2.0_volatility_replacement.parquet"
FOLDS = ROOT / "results/volatility_proxies/ewma/folds.csv"
TARGET = F.TARGET
ARMS = ("B_all", "B_cov", "F_cov")
SLOTS = ("CU_SA_IA_BO_02", "CU_SA_IA_BO_MA_01", "CU_SA_IA_BO_MA_R_01")
LEAVES = ("SACI_04", "IA_02", "IA_01", "MA_FENC", "CURUPIRA_04", "MA_ALLFEAT",
          "CURUPIRA_05", "CURUPIRA_05_60")
GAP_60 = 40          # extra dates the 60-day leaf's window ends earlier (gap 63 >= embargo 60)
VOL_SPREADS = {"feature_idio_vol_spread": ("feature_idio_vol_40_tspct_csbin", "feature_idio_vol_120_tsbin_csbin"),
               "feature_rogers_satchell_spread": ("feature_rogers_satchell_40_tspct_csbin",
                                                  "feature_rogers_satchell_120_tsbin_csbin")}
COMPOSITE = "feature_volatility"


# --------------------------------------------------------------------------- data

def panels():
    """(old panel, new panel, covered mask), frozen to the replacement file's keys."""
    base = pd.read_parquet(PANEL)
    replacement = pd.read_parquet(REPLACEMENT)
    base = base.loc[base.date.le(replacement.date.max())]
    # The refreshed train file resolved some 60-day targets that were still NaN in
    # the frozen copy (the panel's last 40 dates). Accept a difference only there,
    # and keep the frozen values so both arms see exactly the frozen panel.
    late = "target_binned_return_60"
    differs = ~(replacement[late].to_numpy() == base[late].to_numpy()) & base[late].notna().to_numpy()
    assert replacement.loc[differs, late].isna().all()
    base = base.assign(**{late: replacement[late].to_numpy()})
    pd.testing.assert_frame_equal(replacement.drop(columns=list(NEW)), base.drop(columns=list(REMOVED)))
    old = base.reset_index()
    new = replacement.reset_index()
    values = new[list(NEW)].to_numpy(float)
    covered = np.isfinite(values).all(axis=1)
    assert not np.isfinite(values[~covered]).any()
    assert np.isin(values[covered], [0, .25, .5, .75, 1]).all()
    for frame in (old, new):
        frame["ucid"] = frame["ucid"].astype(int)
        frame["covered"] = covered
    meta = pd.read_parquet(META, columns=["id", "date", "prediction"]).rename(
        columns={"id": "ucid", "prediction": "meta"})
    meta["ucid"] = meta.ucid.astype(int)
    old = old.merge(meta, on=["date", "ucid"], how="left", validate="one_to_one")
    new = new.merge(meta, on=["date", "ucid"], how="left", validate="one_to_one")
    return old, new, meta


def folds():
    frame = pd.read_csv(FOLDS, parse_dates=list(range(1, 7)))
    return list(frame.itertuples(index=False))


def nan_aware_means(panel, columns, prior):
    """Causal per-symbol mean counting only observed values (new block only)."""
    panel = panel.sort_values([F.SYMBOL, F.PERIOD])
    grouped = panel.groupby(F.SYMBOL, sort=False)
    out = {}
    for column in columns:
        seen = panel[column].notna().astype(float)
        total = grouped[column].transform(lambda s: s.fillna(0).cumsum()) - panel[column].fillna(0)
        count = seen.groupby(panel[F.SYMBOL], sort=False).cumsum() - seen
        out[column + "_symmean"] = (total + F.FENC_SMOOTHING * prior[column]) / (count + F.FENC_SMOOTHING)
    return pd.DataFrame(out, index=panel.index).sort_index()


def build(panel, variant, history_end):
    """Every block the eight leaves read, before one-hot. Returns frame and column groups."""
    panel = panel.sort_values([F.SYMBOL, F.PERIOD]).reset_index(drop=True)
    panel["is_live"] = panel.date.gt(history_end)
    raw = F.raw_features(panel)
    panel, tenc = F.add_tenc(panel)
    if variant == "old":
        derived_panel = panel
    else:
        composite = panel[list(NEW)].mean(axis=1, skipna=False)
        derived_panel = panel.assign(**{f"{COMPOSITE}_20d": composite, f"{COMPOSITE}_60d": composite})
    derived = {}
    derived.update(build_spreads(derived_panel))
    derived.update(build_interactions(derived_panel))
    if variant == "new":
        derived.pop(f"{COMPOSITE}_spread")
        for name, (short, long) in VOL_SPREADS.items():
            derived[name] = difference(panel[short], panel[long])
    derived = pd.DataFrame(derived, index=panel.index).astype("float32")
    panel = pd.concat([panel, derived], axis=1)

    if variant == "old":
        panel, rolled = F.add_rolling(panel)
    else:
        selected = ["%s_%s" % (f, h) for f in F.ROLL_FEATURES for h in F.ROLL_HORIZONS
                    if f != "feature_volatility"] + list(NEW)
        grouped = panel.groupby(F.SYMBOL, sort=False)[selected]
        rolled = []
        for window in F.ROLL_WINDOWS:
            block = grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
            block.columns = ["%s_rm%d" % (c, window) for c in selected]
            panel = pd.concat([panel, block.astype("float32")], axis=1)
            rolled += list(block.columns)

    prior = panel.loc[~panel.is_live, raw].mean()
    shipped = [c for c in raw if c not in NEW]
    encoded, _ = causal_feature_means(panel, shipped, F.SYMBOL, F.PERIOD, F.FENC_SMOOTHING, prior=prior)
    if variant == "new":
        encoded = pd.concat([encoded, nan_aware_means(panel, list(NEW), prior)], axis=1)
    symmean = [c + "_symmean" for c in raw]
    panel = pd.concat([panel, encoded[symmean].astype("float32")], axis=1)
    groups = dict(raw=raw, tenc=tenc, derived=list(derived.columns), rolled=rolled, symmean=symmean)
    return panel, groups


def leaf_columns(feature_set, groups, onehot):
    g = groups
    if feature_set == "raw":
        return g["raw"]
    if feature_set == "fenccausal":
        return g["raw"] + g["symmean"]
    if feature_set == "allfeat":
        return g["raw"] + g["tenc"] + g["derived"] + onehot
    if feature_set == "allfeat_pp":
        return g["raw"] + g["tenc"] + g["derived"] + g["rolled"] + onehot
    raise ValueError(feature_set)


def unit_list():
    return [(fold, arm) for fold in range(11) for arm in ARMS]


# --------------------------------------------------------------------------- fit

def fit_unit(index, old, new, search_jobs):
    fold_no, arm = unit_list()[index]
    path = OUT / "leaves" / f"fold{fold_no:02d}_{arm}.parquet"
    if path.exists():
        print(f"unit {index} done already", flush=True)
        return
    refit.SEARCH_JOBS = search_jobs
    fold = folds()[fold_no]
    source = new if arm == "F_cov" else old
    panel = source.loc[source.date.le(fold.test_end)]
    built, groups = build(panel, "new" if arm == "F_cov" else "old", fold.train_end)
    rows = np.ones(len(built), bool) if arm == "B_all" else built.covered.to_numpy()
    vocab_rows = built.date.le(fold.train_end).to_numpy() & rows
    built, onehot, _, vocab = F.add_onehot(built, pd.Series(vocab_rows, index=built.index))
    every = sorted(set(sum(groups.values(), [])) | set(onehot), key=list(built.columns).index)
    built = fill_missing_features(built, every, F.NEUTRAL, verbose=False)
    dates = pd.DatetimeIndex(np.sort(built.date.unique()))
    test = built.date.between(fold.test_start, fold.test_end).to_numpy() & rows
    scored = built.loc[test].sort_values(["date", "ucid"])
    out = scored[["date", "ucid"]].reset_index(drop=True)
    log = []
    for leaf in LEAVES:
        spec = registry.get(leaf)
        fit = dict(spec["fit"])
        end = fold.train_end
        if fit["target"] != TARGET:
            end = dates[dates.get_loc(fold.train_end) - GAP_60]
        window = dates[dates <= end][-fit["train_window"]:]
        train = built.date.isin(window).to_numpy() & rows & built[fit["target"]].notna().to_numpy()
        columns = leaf_columns(spec["feature_set"], groups, onehot)
        members = spec.get("members") or [dict(name=leaf, wrapper=spec["wrapper"], include=None, weight=1.0)]
        blend = np.zeros(len(scored))
        for member in members:
            cols = columns if member["include"] is None else filter_features(columns, member["include"], None, verbose=False)
            started = time.time()
            X = built.loc[train, cols]
            search = refit.fit_one(member["wrapper"], X, built.loc[train, fit["target"]],
                                   built.loc[train, [F.PERIOD]], fit)
            raw = np.asarray(search.best_estimator_.predict(scored[cols]), float)
            assert np.isfinite(raw).all(), (leaf, member["name"])
            if spec.get("members"):
                out[f"{leaf}/{member['name']}"] = raw
                ranks = scored[["date"]].assign(p=raw).groupby("date").p.rank(pct=True).to_numpy()
                blend += member["weight"] * ranks
            else:
                out[leaf] = raw
            log.append(dict(fold=fold_no, arm=arm, leaf=leaf, member=member["name"], n_features=len(cols),
                            train_rows=int(train.sum()), train_start=str(window[0].date()),
                            train_end=str(window[-1].date()), vocabulary=vocab,
                            best_params=json.dumps({k: str(v) for k, v in search.best_params_.items()}),
                            cv=float(search.best_score_), seconds=round(time.time() - started, 1)))
            print(f"unit {index} fold {fold_no} {arm} {leaf}/{member['name']}: {len(cols)} cols "
                  f"{int(train.sum())} rows cv {search.best_score_:.5f} {log[-1]['seconds']}s", flush=True)
        if spec.get("members"):
            out[leaf] = blend
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(path.with_suffix(".log.csv"), index=False)
    out.to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)


# --------------------------------------------------------------------------- blend

def lagged(meta, frame):
    """Each row's meta prediction on the latest meta date <= date - 5 days."""
    out = np.full(len(frame), np.nan)
    meta_dates = np.sort(meta.date.unique())
    by_date = {d: g.set_index("ucid").meta for d, g in meta.groupby("date")}
    for date, ix in frame.groupby("date").indices.items():
        snap = meta_resid.snapshot_for(meta_dates, date)
        if snap is not None:
            out[ix] = by_date[snap].reindex(frame.ucid.to_numpy()[ix]).to_numpy()
    return out


def slot_prediction(name, frame, lag_meta, cache):
    if name in cache:
        return cache[name]
    spec = registry.get(name)
    if "slot_members" not in spec:
        cache[name] = frame[name].to_numpy()
        return cache[name]
    blended = np.zeros(len(frame))
    for member in spec["slot_members"]:
        values = slot_prediction(member["name"], frame, lag_meta, cache)
        if member.get("resid"):
            values = values.copy()
            for ix in frame.groupby("date").indices.values():
                values[ix], _ = meta_resid.residualise(values[ix], lag_meta[ix], float(member["resid"]))
        ranks = frame[["date"]].assign(p=values).groupby("date").p.rank(pct=True).to_numpy()
        blended += member["weight"] * ranks
    cache[name] = blended
    return blended


def nw_t(series, lag=20):
    x = np.asarray(series, float)
    centred = x - x.mean()
    n = len(x)
    var = centred @ centred / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (centred[k:] @ centred[:-k]) / n
    return x.mean() / np.sqrt(var / n)


def assemble(fold_numbers=range(11), tag=""):
    from src.prediction.metrics import summarize
    meta = pd.read_parquet(META, columns=["id", "date", "prediction"]).rename(
        columns={"id": "ucid", "prediction": "meta"})
    meta["ucid"] = meta.ucid.astype(int)
    keys = pd.read_parquet(OUT / "scored_keys_full.parquet").merge(
        meta, on=["date", "ucid"], how="left", validate="one_to_one")
    rows = []
    for fold_no in fold_numbers:
        frames = {arm: pd.read_parquet(OUT / "leaves" / f"fold{fold_no:02d}_{arm}.parquet") for arm in ARMS}
        full = frames["B_all"]
        full = full.merge(keys, on=["date", "ucid"], how="left", validate="one_to_one")
        assert full[TARGET].notna().all()
        lag_full = lagged(meta, full)
        preds = {}
        cache = {}
        for slot in SLOTS:
            preds[("B_all", slot)] = slot_prediction(slot, full, lag_full, cache)
        for arm in ("B_cov", "F_cov"):
            frame = frames[arm]
            lag_cov = lagged(meta, frame)
            cache = {}
            for slot in SLOTS:
                preds[(arm, slot)] = slot_prediction(slot, frame, lag_cov, cache)
        for date, ix in full.groupby("date").indices.items():
            covered = full.covered.to_numpy()[ix]
            target = full[TARGET].to_numpy()[ix]
            ref = full.meta.to_numpy()[ix]
            for slot in SLOTS:
                base = rank_columns(preds[("B_all", slot)][ix])
                columns = [base]
                for arm in ("B_cov", "F_cov"):
                    frame = frames[arm]
                    sel = np.flatnonzero(frame.date.to_numpy() == date)
                    local = preds[(arm, slot)][sel]
                    assert np.array_equal(frame.ucid.to_numpy()[sel], full.ucid.to_numpy()[ix][covered])
                    routed = route(base[:, None], covered, local[:, None])[:, 0]
                    assert np.array_equal(routed[~covered], base[~covered])
                    columns.append(routed)
                matrix = np.column_stack(columns)
                corr_all, corr, mmc = score_date(matrix, target, ref)
                for arm, a, c, m in zip(ARMS, corr_all, corr, mmc):
                    rows.append(dict(fold=fold_no, date=date, slot=slot, arm=arm, corr_all=a, corr=c, mmc=m,
                                     covered=int(covered.sum()), n=len(ix)))
        print(f"fold {fold_no} assembled", flush=True)
    scores = pd.DataFrame(rows)
    scores["payout"] = .1 * scores["corr"] + scores.mmc
    scores.to_parquet(OUT / f"scores{tag}.parquet", index=False)
    window = scores.dropna(subset=["mmc"])
    table, deltas = {}, []
    for slot in SLOTS:
        for arm, label in (("B_cov", "old vol"), ("F_cov", "new vol"), ("B_all", "B_all (context)")):
            s = window[(window.slot == slot) & (window.arm == arm)].sort_values("date")
            pay = summarize(s.payout.to_numpy())
            table[(slot, label)] = dict(corr=s["corr"].mean(), mmc=s.mmc.mean(), payout=s.payout.mean(),
                                        **{"sharpe.p": pay["sharpe"], "maxdd.p": pay["max_drawdown"]},
                                        corr_all=s.corr_all.mean(), n=len(s))
        old = window[(window.slot == slot) & (window.arm == "B_cov")].sort_values("date").reset_index(drop=True)
        new = window[(window.slot == slot) & (window.arm == "F_cov")].sort_values("date").reset_index(drop=True)
        assert old.date.equals(new.date)
        for metric in ("corr", "mmc", "payout"):
            d = new[metric] - old[metric]
            deltas.append(dict(slot=slot, metric=metric, delta=d.mean(), t_nw20=nw_t(d), n=len(d)))
    out = pd.DataFrame(table).T
    out.index.names = ["slot", "arm"]
    out.to_csv(OUT / f"summary_long{tag}.csv")
    pd.DataFrame(deltas).to_csv(OUT / f"paired_deltas{tag}.csv", index=False)
    wide = out.loc[(slice(None), ["old vol", "new vol"]), ["corr", "mmc", "payout", "sharpe.p", "maxdd.p"]]
    wide = wide.unstack("arm").swaplevel(0, 1, axis=1)
    wide = wide.reindex(columns=pd.MultiIndex.from_product(
        [["old vol", "new vol"], ["corr", "mmc", "payout", "sharpe.p", "maxdd.p"]]))
    wide.index = [s.lower() for s in wide.index]
    wide.to_csv(OUT / f"table{tag}.csv")
    print(wide.round(5).to_string())
    print(pd.DataFrame(deltas).round(5).to_string(index=False))


# --------------------------------------------------------------------------- checks

def validate():
    """The old-variant builder must reproduce the deployed build_features exactly."""
    check_scoring()
    old, new, _ = panels()
    fold = folds()[0]
    panel = old.loc[old.date.le(fold.test_end)]
    built, groups = build(panel, "old", fold.train_end)
    built, onehot, _, _ = F.add_onehot(built, ~built.is_live)
    reference = panel.sort_values([F.SYMBOL, F.PERIOD]).reset_index(drop=True)
    reference["is_live"] = reference.date.gt(fold.train_end)
    for feature_set in ("raw", "fenccausal", "allfeat", "allfeat_pp"):
        expect, cols, _ = F.build_features(reference.copy(), feature_set, verbose=False)
        mine = leaf_columns(feature_set, groups, onehot)
        assert sorted(mine) == sorted(cols), (feature_set, set(mine) ^ set(cols))
        got = fill_missing_features(built[cols].copy(), cols, F.NEUTRAL, verbose=False)
        e = expect.set_index(["symbol", "date"])[cols]
        g = got.set_index(pd.MultiIndex.from_frame(built[["symbol", "date"]]))
        pd.testing.assert_frame_equal(g.loc[e.index], e, check_dtype=False, atol=1e-6)
        print(f"{feature_set}: {len(cols)} columns reproduce build_features")
    nb, ng = build(new.loc[new.date.le(fold.test_end)], "new", fold.train_end)
    cov = nb.covered.to_numpy()
    for name in ("raw", "symmean", "rolled", "derived"):
        print(f"new {name}: {len(ng[name])} columns, covered NaN {int(nb.loc[cov, ng[name]].isna().sum().sum())}")
    keys = old[["date", "ucid", "symbol", "covered", TARGET]]
    OUT.mkdir(parents=True, exist_ok=True)
    keys[["date", "ucid", "covered", TARGET]].to_parquet(OUT / "scored_keys_full.parquet", index=False)
    pd.DataFrame(sorted(set(old.columns) ^ set(new.columns))).to_csv(OUT / "swapped_columns.csv", index=False)
    manifest = dict(inputs={str(p.relative_to(ROOT)): digest(p) for p in (REPLACEMENT, META, FOLDS)},
                    panel_rows=len(old), covered_rows=int(old.covered.sum()),
                    panel_end=str(old.date.max().date()), new=list(NEW), removed=list(REMOVED),
                    vol_spreads=VOL_SPREADS, composite="row mean of the nine new measures",
                    gap_60=GAP_60, arms=ARMS, leaves=LEAVES, slots=SLOTS)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("validation passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    w = sub.add_parser("worker")
    w.add_argument("--units", required=True, help="e.g. 0-10 or 3,7")
    w.add_argument("--search-jobs", type=int, default=8)
    a = sub.add_parser("assemble")
    a.add_argument("--folds", type=int, nargs="+", default=list(range(11)))
    a.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.cmd == "validate":
        return validate()
    if args.cmd == "assemble":
        return assemble(args.folds, args.tag)
    spec = args.units
    units = (list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1)) if "-" in spec
             else [int(u) for u in spec.split(",")])
    old, new, _ = panels()
    for index in units:
        claim = OUT / "claims" / f"{index}.claim"
        claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except FileExistsError:
            continue
        fit_unit(index, old, new, args.search_jobs)


if __name__ == "__main__":
    main()
