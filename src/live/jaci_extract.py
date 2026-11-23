"""JACI live pre-stage, GPU half: frozen TSFM states for the live momentum3 panel.

Runs inside a model venv on a GPU box (deepfive), launched by src.live.jaci:

    /scratch/danielco/chronos2/venv/bin/python -m src.live.jaci_extract --model chronos2
    /scratch/danielco/tsfm2/venv_tt/bin/python  -m src.live.jaci_extract --model t0
    /scratch/danielco/tsfm2/venv_tt/bin/python  -m src.live.jaci_extract --model timestral

It reuses the E206 extractor unchanged (pinned checkpoints, context 256, label
wall 35 days, float32 inference stored as float16) -- `adapter`, `inputs` and
`safe_states` are imported, not copied -- so a live state is the same function
of the same panel as a research state. Only the panel is new: the extended
panel written by `src.live.jaci panel` (frozen E206 rows plus rows after
2026-02-19, keyed by ucid and date).

Exactly as E206, every model call sees ONE asset: several origins of that asset
may be batched (same horizon only), never two assets, so the cross-series
attention some of these models offer cannot leak one coin into another.

States are cached per asset as dates + float16 states in
<STATE_ROOT>/states/<model>/asset_<ucid>.npz and only origins not already cached
are computed. A state depends on nothing after its origin date, so a cached
state never needs recomputing.
"""
import os
import sys
os.environ.setdefault("TSFM_EMBED_ROOT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "results/reports/crypto/e206_momentum3_tsfm"))
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data_mani import e206_tsfm_embeddings as E  # noqa: E402

STATE_ROOT = Path(os.environ.get("JACI_STATE_ROOT", "/scratch/danielco/jaci_live"))
TARGET = "target_binned_return_20"
FROZEN_END = np.datetime64("2026-02-19")


def asset_file(model, ucid):
    return STATE_ROOT / "states" / model / ("asset_%d.npz" % int(ucid))


def load_states(model, ucid):
    path = asset_file(model, ucid)
    if not path.exists():
        return np.array([], "datetime64[D]"), None
    z = np.load(path)
    return z["dates"].astype("datetime64[D]"), z["states"]


def run(model, device, limit_dates=None, verify=0, reverse=False):
    panel = pd.read_parquet(STATE_ROOT / "panel.parquet")
    features = [c for c in panel if c.startswith("feature_")]
    assert len(features) == 33, len(features)
    grid = np.unique(panel.date.to_numpy().astype("datetime64[D]"))
    ext, manifest = E.adapter(model, device)
    out_dir = STATE_ROOT / "states" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest, live_panel=str(STATE_ROOT / "panel.parquet"),
                    host=socket.gethostname())
    (out_dir / "extractor.json").write_text(json.dumps(manifest, indent=2, default=str))
    wanted = panel.covered & (panel.date > pd.Timestamp(FROZEN_END))
    if limit_dates is not None:
        wanted &= panel.date.isin(pd.to_datetime(limit_dates))
    todo_assets = np.sort(panel.loc[wanted, "ucid"].unique())
    if reverse:     # a second worker on another card walks from the other end
        todo_assets = todo_assets[::-1]
    started, computed = time.time(), 0
    checks = []
    for ucid in todo_assets:
        frame = panel[panel.ucid == ucid].sort_values("date")
        positions = np.searchsorted(grid, frame.date.to_numpy().astype("datetime64[D]"))
        y = np.full(len(grid), np.nan, np.float32)
        x = np.full((len(grid), len(features)), np.nan, np.float32)
        y[positions] = frame[TARGET].to_numpy(np.float32)
        x[positions] = frame[features].to_numpy(np.float32)
        have_dates, have_states = load_states(model, ucid)
        # Compare as int day numbers: set(datetime64.tolist()) holds datetime.date
        # objects, which never equal np.datetime64, so every cached state looked
        # missing and was recomputed and appended again.
        have = set(have_dates.astype("datetime64[D]").astype(np.int64).tolist())
        mask = wanted.loc[frame.index].to_numpy()
        pending = []
        for pos, date, ok in zip(positions, frame.date.to_numpy().astype("datetime64[D]"), mask):
            if not ok or int(date.astype(np.int64)) in have:
                continue
            args = E.inputs(y, x, grid, int(pos))
            if args is not None:
                pending.append((date, args))
        # Optional extraction-parity probe: recompute a few FROZEN origins of
        # this asset on the extended panel, for comparison with E206 H_rows.
        if verify and len(checks) < verify:
            frozen = frame[(frame.date <= pd.Timestamp(FROZEN_END)) & frame.covered]
            for _, row in frozen.tail(2).iterrows():
                pos = int(np.searchsorted(grid, np.datetime64(row.date, "D")))
                args = E.inputs(y, x, grid, pos)
                if args is not None:
                    state = E.safe_states(ext, [args])[0].astype(np.float16)
                    checks.append(dict(ucid=int(ucid), date=str(row.date.date()), row=int(row.row),
                                       state=state.astype(float).tolist()))
        if not pending:
            continue
        groups = {}
        for date, args in pending:
            groups.setdefault(args[3], []).append((date, args))
        dates, states = [], []
        batch = int(os.getenv("E206_BATCH", "4"))
        for group in groups.values():
            for i in range(0, len(group), batch):
                items = group[i:i + batch]
                state = E.safe_states(ext, [a for _, a in items])
                if state.shape != (len(items), ext.dim) or not np.isfinite(state).all():
                    raise ValueError("invalid pre-head representation for %s" % ucid)
                stored = state.astype(np.float16)
                if not np.isfinite(stored).all():
                    raise FloatingPointError("state overflows float16 for %s" % ucid)
                dates.extend(d for d, _ in items)
                states.extend(stored)
        all_dates = np.concatenate([have_dates, np.array(dates, "datetime64[D]")])
        all_states = np.concatenate([have_states, np.stack(states)]) if have_states is not None \
            else np.stack(states)
        all_dates, first = np.unique(all_dates, return_index=True)   # sorted, first copy kept
        all_states = all_states[first]
        order = np.arange(len(all_dates))
        path = asset_file(model, ucid)
        tmp = path.with_suffix(".%s_%d.tmp.npz" % (socket.gethostname(), os.getpid()))
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, dates=all_dates[order], states=all_states[order])
        tmp.replace(path)
        computed += len(dates)
        print("%s %s +%d states (%d cached) %.0fs" % (model, ucid, len(dates), len(have),
                                                     time.time() - started), flush=True)
    if verify:
        (out_dir / "parity_probe.json").write_text(json.dumps(checks))
    summary = dict(model=model, computed=computed, assets=len(todo_assets),
                   seconds=round(time.time() - started, 1), host=socket.gethostname(),
                   finished=pd.Timestamp.now().isoformat())
    (out_dir / "last_run.json").write_text(json.dumps(summary, indent=2))
    print("JACI_EXTRACT_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=["chronos2", "t0", "timestral"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dates", nargs="*", default=None, help="restrict to these origin dates")
    p.add_argument("--verify", type=int, default=0,
                   help="also recompute this many frozen origins for the parity probe")
    p.add_argument("--reverse", action="store_true", help="walk assets from the other end")
    a = p.parse_args()
    run(a.model, a.device, a.dates, a.verify, a.reverse)
