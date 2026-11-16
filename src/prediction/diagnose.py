"""
Train one walk-forward refit of a torch MLP and take the diagnostic panel.

    python -m src.prediction.diagnose --tag e105_const_s42_r0 --seed 42 --refit 0 \
        --activation identity --alpha 100 --batch days --l2 pinned --schedule const
    python -m src.analysis.plot_training_diagnostics e105_const_s42_r0

The recipe flags cover every torch MLP trained in this repo, and the loop below
evaluates the same expressions in the same order as the one it describes, so the
panel is of THAT run and not of a look-alike:

    torch_mlp (TorchMLPRegressor)   --batch rows --l2 batch  --init torch
    E104 row / rowfix / row900      --batch rows --l2 batch|pinned [--batch-size 900] --init sklearn
    E104 day3 / day1 (+lit)         --batch days --days 3|1 --l2 pinned|batch --init sklearn
    E105 arms                       --batch days --days 3 --l2 pinned --schedule <arm>

Checked, not assumed: with these flags the final held-out predictions equal the
e105_const / e105_onecycle runs' and TorchMLPRegressor.fit's for the same refit and
seed. The data are one refit of the shared protocol (rolling 130/130, embargo 20,
val_frac 0.3); the held-out block is the 130 days that refit predicts, which is
clean only because nothing here is selected on it.

Output: results/diag/crypto/<tag>/{log.json, preds.npz, config.json} and, unless
--no-plot, diagnostics.png. --save-preds adds every epoch's predictions on both
sets; --predict-next adds them on the next refit's block (pred_next.npy), which is
what cross-fold churn compares with the next refit's own pred_val.npy.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_ROOT = os.path.join("results", "diag", "crypto")
SCHEDULES = ("const", "step", "exp", "cosine", "warmcos", "restarts", "onecycle", "plateau")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.prediction.diagnose",
                                description=__doc__.split("\n")[1])
    p.add_argument("--tag", required=True, help="results/diag/crypto/<tag>/")
    p.add_argument("--data-version", default="crypto/v2.0-fenccausal")
    p.add_argument("--target", default="target_binned_return_20")
    p.add_argument("--refit", type=int, default=0, help="which walk-forward block (0-based)")
    p.add_argument("--scheme", choices=("standard", "full", "even"), default="standard",
                   help="standard: the 4 blocks of the shared protocol (the last 510 days); "
                        "full: extend them backwards in --refit-every steps over the whole "
                        "history, so the standard 4 are the last 4 of 12; even: cut every "
                        "predictable day (after the first train window + embargo) into "
                        "--n-refits equal consecutive blocks")
    p.add_argument("--n-refits", type=int, default=12, help="blocks for --scheme even")
    p.add_argument("--meta-only", action="store_true",
                   help="keep only the days the meta model covers (2024-05-07 on), before the "
                        "blocks are laid out -- what --loss mmc needs, since it trains on MMC")
    p.add_argument("--save-preds", action="store_true",
                   help="write every epoch's train and held-out predictions (pred_train.npy, "
                        "pred_val.npy, float64, one row per epoch) with their row index")
    p.add_argument("--no-train-preds", action="store_true",
                   help="with --save-preds, skip pred_train.npy: nothing downstream reads it "
                        "(walkforward_diagnostics and assemble_walkforward use pred_val/pred_next), "
                        "and it is a third of a run's disk -- 24 runs filled the /home quota once")
    p.add_argument("--predict-next", action="store_true",
                   help="also predict the NEXT walk-forward block (refit+1) at every epoch, "
                        "scaled by this refit's own scaler, into pred_next.npy + next_rows.parquet: "
                        "what this refit's model would have said where the next refit takes over "
                        "(cross-fold churn). No-op on the last block")
    p.add_argument("--train-window", type=int, default=130)
    p.add_argument("--refit-every", type=int, default=130)
    p.add_argument("--embargo", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--hidden", default="64,32,16")
    p.add_argument("--activation", choices=("identity", "relu"), default="identity")
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--schedule", choices=SCHEDULES, default="const")
    p.add_argument("--horizon", type=int, default=60, help="schedule length in epochs")
    p.add_argument("--batch", choices=("rows", "days"), default="rows")
    p.add_argument("--batch-size", type=int, default=1024, help="rows per batch (rows scheme)")
    p.add_argument("--days", default="3", help="days per batch (days scheme), or 'auto'")
    p.add_argument("--l2", choices=("batch", "pinned"), default="batch",
                   help="divide the penalty by the rows in the batch (sklearn) or by 1024")
    p.add_argument("--init", choices=("sklearn", "torch", "xavier"), default="sklearn")
    p.add_argument("--loss", choices=("mse", "corr", "mmc"), default="mse",
                   help="mse: 0.5*MSE (every recipe above). corr: minus the mean per-day "
                        "Pearson correlation of prediction and target over the days in the "
                        "batch -- the differentiable stand-in for CORR; with --batch days "
                        "each batch holds whole days. mmc: minus the mean per-day "
                        "contribution over the meta model (training_diagnostics."
                        "day_contribution); needs the meta model on every training row, "
                        "so use --meta-only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--no-stop", action="store_true",
                   help="train exactly --max-epochs, ignoring the stop rule, to see past it")
    p.add_argument("--metric-every", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu",
                   help="cpu (default; the bit-for-bit gates hold here) or cuda / cuda:N")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def load_refit(args):
    """One block of the shared walk-forward, scaled on its own training window."""
    return refit_parts(load_panel(args), args)


def load_panel(args):
    """The whole panel with reference and meta model attached -- loaded once, split per refit."""
    from src.data_mani.utils import (load_tournament_data, load_reference, attach_reference,
                                     load_meta_model, attach_meta_model)
    df, feats = load_tournament_data(tournament="crypto", version=args.data_version, era_step=1,
                                     val_frac=args.val_frac, embargo=args.embargo,
                                     verbose=False, target=args.target)
    ref, rcol = load_reference("crypto", args.data_version, verbose=False)
    df, _ = attach_reference(df, ref, "crypto", rcol)
    meta, _ = load_meta_model("crypto", args.data_version, verbose=False)
    df, _ = attach_meta_model(df, meta, "crypto")
    if getattr(args, "meta_only", False):
        from src.data_mani.utils import META_MODEL_COL
        df = df[df[META_MODEL_COL].notna()].reset_index(drop=True)
    return df, feats


def refit_parts(panel, args):
    """(train, held, meta, next) for args.refit, from a panel load_panel returned."""
    from src.data_mani.utils import META_MODEL_COL, REFERENCE_COL
    from src.prediction.models import ScaleContinuousOnly
    df, feats = panel
    allp = sorted(pd.unique(df["date"])); pos = {p: i for i, p in enumerate(allp)}
    valp = sorted(pd.unique(df.loc[df["split"] == "validation", "date"]))
    blocks = block_layout(allp, pos, valp, args)
    if not 0 <= args.refit < len(blocks):
        raise SystemExit("--refit must be in 0..%d for --scheme %s" % (len(blocks) - 1, args.scheme))
    block = blocks[args.refit]
    standard = [valp[s:s + args.refit_every] for s in range(0, len(valp), args.refit_every)]
    nxt_block = (blocks[args.refit + 1]
                 if args.predict_next and args.refit + 1 < len(blocks) else None)
    te = max(0, pos[block[0]] - args.embargo); tp = allp[max(0, te - args.train_window):te]
    if pos[block[0]] - pos[tp[-1]] <= args.embargo:
        raise SystemExit("training reaches inside the embargo")
    tm, bm = df["date"].isin(tp), df["date"].isin(block)
    nm = df["date"].isin(nxt_block) if nxt_block is not None else None
    Xtr = df.loc[tm, feats].to_numpy("float64")
    scaler = ScaleContinuousOnly().fit(Xtr)
    # y trains in float64; y_score keeps the parquet's own dtype (float32) because
    # that is what every per_period number in results/ was scored against, and
    # numerai_corr on a float64 copy of the same values differs by ~1e-9.
    part = lambda mask, X: {"X": X, "y": df.loc[mask, args.target].to_numpy("float64"),
                            "reference": df.loc[mask, REFERENCE_COL].to_numpy("float64"),
                            "y_score": df.loc[mask, args.target].to_numpy(),
                            "date": df.loc[mask, "date"].to_numpy(),
                            "symbol": df.loc[mask, "symbol"].to_numpy(),
                            "meta": df.loc[mask, META_MODEL_COL].to_numpy("float64")}
    nxt = (part(nm, scaler.transform(df.loc[nm, feats].to_numpy("float64")))
           if nm is not None else None)
    return (part(tm, scaler.transform(Xtr)),
            part(bm, scaler.transform(df.loc[bm, feats].to_numpy("float64"))),
            {"features": len(feats), "train_days": len(tp), "held_days": len(block),
             "scheme": args.scheme, "n_refits": len(blocks),
             "standard_refit": next((i for i, b in enumerate(standard) if list(b) == list(block)), None),
             "train_start": str(tp[0])[:10], "train_end": str(tp[-1])[:10],
             "held_start": str(block[0])[:10], "held_end": str(block[-1])[:10],
             "next_start": str(nxt_block[0])[:10] if nxt_block is not None else None,
             "next_end": str(nxt_block[-1])[:10] if nxt_block is not None else None},
            nxt)


def block_layout(allp, pos, valp, args):
    """
    The walk-forward's predicted blocks, as lists of periods.

    standard: the shared protocol -- the validation tail (val_frac) cut into
    refit_every-day blocks. full: the same blocks, plus as many earlier ones as the
    history allows, stepping back refit_every days at a time from the standard start
    (a block needs train_window days and the embargo before it), and a shorter first
    block to reach the earliest predictable day. The standard blocks are therefore
    the last ones of the full layout, unchanged.
    """
    if args.scheme == "standard":
        return [valp[s:s + args.refit_every] for s in range(0, len(valp), args.refit_every)]
    if args.scheme == "even":
        lo = args.train_window + args.embargo
        if len(allp) - lo < args.n_refits:
            raise SystemExit("%d predictable days cannot make %d blocks" % (len(allp) - lo, args.n_refits))
        return [list(b) for b in np.array_split(np.asarray(allp[lo:], dtype=object), args.n_refits)]
    first, lo = pos[valp[0]], args.train_window + args.embargo
    starts = list(range(first, len(allp), args.refit_every))
    s = first - args.refit_every
    while s >= lo:
        starts.insert(0, s); s -= args.refit_every
    if starts[0] > lo:
        starts.insert(0, lo)
    return [allp[a:b] for a, b in zip(starts, starts[1:] + [len(allp)])]


def prediction_store(out_dir, train, held, epochs, keep_train=True):
    """
    Every epoch's predictions on both sets, written straight to disk.

    Two float64 .npy arrays of shape (epochs + 1, rows) -- row e is the network after
    e epochs, NaN where a run stopped earlier -- memory-mapped so a 500-epoch run
    holds nothing in RAM and a stopped run keeps every epoch it finished. The rows
    are indexed by train_rows.parquet / val_rows.parquet (date, symbol, target as the
    pipeline scores it, reference prediction, meta model), in the same order.
    keep_train=False (--no-train-preds) writes the held-out array only.
    """
    stores = {}
    for name, part, rows_file in (("train", train, "train_rows.parquet"),
                                  ("held", held, "val_rows.parquet")):
        if name == "train" and not keep_train:
            continue
        pd.DataFrame({"date": part["date"], "symbol": part["symbol"], "target": part["y_score"],
                      "reference_prediction": part["reference"], "meta_model": part["meta"]}
                     ).to_parquet(os.path.join(out_dir, rows_file), index=False)
        array = np.lib.format.open_memmap(
            os.path.join(out_dir, "pred_train.npy" if name == "train" else "pred_val.npy"),
            mode="w+", dtype="float64", shape=(epochs + 1, len(part["y"])))
        array[:] = np.nan
        stores[name] = array

    def sink(epoch, preds):
        for name, array in stores.items():
            array[epoch] = preds[name]
        if epoch % 10 == 0:
            for array in stores.values():
                array.flush()
    sink.flush = lambda: [a.flush() for a in stores.values()]
    return sink


def next_block_store(out_dir, nxt, epochs):
    """
    The network's predictions on the NEXT refit's held-out block, every epoch.

    pred_next.npy has the same layout as pred_val.npy -- (epochs + 1, rows), float64,
    NaN where the run stopped -- indexed by next_rows.parquet. The next refit's own
    pred_val.npy covers the same rows, so the two together give the churn at the
    refit boundary. A forward pass under no_grad in eval mode: it steps nothing,
    writes no .grad and draws no random numbers, so the run is unchanged.
    """
    import torch
    pd.DataFrame({"date": nxt["date"], "symbol": nxt["symbol"], "target": nxt["y_score"],
                  "reference_prediction": nxt["reference"], "meta_model": nxt["meta"]}
                 ).to_parquet(os.path.join(out_dir, "next_rows.parquet"), index=False)
    array = np.lib.format.open_memmap(os.path.join(out_dir, "pred_next.npy"), mode="w+",
                                      dtype="float64", shape=(epochs + 1, len(nxt["y"])))
    array[:] = np.nan
    X = {}

    def record(epoch, net):
        from src.prediction.training_diagnostics import _device
        device = _device(net)
        if device not in X:
            X[device] = torch.as_tensor(np.asarray(nxt["X"], dtype="float64"), device=device)
        was = net.training; net.eval()
        try:
            with torch.no_grad():
                array[epoch] = net(X[device]).cpu().numpy().ravel()
        finally:
            net.train(was)
        if epoch % 10 == 0:
            array.flush()
    record.flush = array.flush
    return record


def rows_by_day(dates):
    codes, uniq = pd.factorize(dates, sort=True)
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes, minlength=len(uniq)))[:-1])


def day_batches(day_rows, day_order, k):
    """Every stock of k shuffled days per batch; each day weighs 1/k (E104's sampler)."""
    for start in range(0, len(day_order), k):
        days = day_order[start:start + k]
        sizes = np.array([len(day_rows[d]) for d in days])
        yield (np.concatenate([day_rows[d] for d in days]),
               np.repeat(1.0 / (len(days) * sizes), sizes), len(days))


def build_net(n_in, args):
    """The MLP, initialised the way the run being diagnosed was."""
    import torch
    from torch import nn
    dims = [n_in] + [int(h) for h in args.hidden.split(",")] + [1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 and args.activation == "relu":
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers).to(dtype=torch.float64)
    linears = [m for m in net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        if args.init == "sklearn":          # sklearn _init_coef: numpy draws, weight then bias
            rs = np.random.RandomState(int(args.seed))
            for lin in linears:
                b = math.sqrt(6.0 / (lin.in_features + lin.out_features))
                lin.weight.copy_(torch.tensor(rs.uniform(-b, b, (lin.in_features, lin.out_features)).T))
                lin.bias.copy_(torch.tensor(rs.uniform(-b, b, lin.out_features)))
        elif args.init == "torch":          # TorchMLPRegressor._build: one seeded generator
            g = torch.Generator(device="cpu").manual_seed(int(args.seed))
            for lin in linears:
                b = float(np.sqrt(6.0 / (lin.in_features + lin.out_features)))
                lin.weight.uniform_(-b, b, generator=g); lin.bias.uniform_(-b, b, generator=g)
        else:                               # torch's own idiom
            torch.manual_seed(int(args.seed))
            for lin in linears:
                nn.init.xavier_uniform_(lin.weight, gain=1.0); nn.init.zeros_(lin.bias)
    return net


def make_scheduler(name, opt, args, per_epoch):
    """E105's schedulers, same settings; (scheduler, when it steps, how often it may)."""
    import torch.optim.lr_scheduler as S
    floor, h = 1e-5, args.horizon
    if name == "const":
        return None, None, None
    if name == "step":
        return S.StepLR(opt, step_size=15, gamma=0.5), "epoch", None
    if name == "exp":
        return S.ExponentialLR(opt, gamma=0.955), "epoch", None
    if name == "cosine":
        return S.CosineAnnealingLR(opt, T_max=h, eta_min=floor), "epoch", h
    if name == "warmcos":
        return S.SequentialLR(opt, [S.LinearLR(opt, start_factor=0.1, total_iters=5),
                                    S.CosineAnnealingLR(opt, T_max=h - 5, eta_min=floor)],
                              milestones=[5]), "epoch", h
    if name == "restarts":
        return S.CosineAnnealingWarmRestarts(opt, T_0=15, eta_min=floor), "epoch", None
    if name == "onecycle":
        return S.OneCycleLR(opt, max_lr=args.lr, total_steps=h * per_epoch), "batch", h * per_epoch - 1
    return S.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, threshold=1e-3,
                               threshold_mode="abs", min_lr=floor), "plateau", None


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    if args.device.startswith("cuda"):
        # cuBLAS is only deterministic with a fixed workspace, and the variable has
        # to be set before CUDA initialises.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch import nn
    from src.prediction.training_diagnostics import Objective, Snapshotter, day_codes, meta_gaussian
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    try:
        # warn_only on a GPU: a CUDA op with no deterministic kernel should not end
        # the run. The result is then reproducible on that GPU, not bit-identical to
        # the CPU path the gates check -- float64 kernels sum in a different order.
        torch.use_deterministic_algorithms(True, warn_only=device.type == "cuda")
    except Exception:
        pass

    started = time.time()
    train, held, data_meta, nxt = load_refit(args)
    out_dir = os.path.join(OUT_ROOT, args.tag)
    day_rows = rows_by_day(train["date"])
    k = (max(1, int(round(args.batch_size / np.mean([len(r) for r in day_rows]))))
         if args.days == "auto" else int(args.days))
    denominator = args.batch_size if args.l2 == "pinned" else (
        args.batch_size if args.batch == "rows" else k * np.mean([len(r) for r in day_rows]))
    objective = Objective(args.alpha, denominator, day_balanced=args.batch == "days",
                          kind=args.loss)
    train_days = day_codes(train["date"])
    train_meta_g = None
    if args.loss == "mmc":
        try:
            train_meta_g = meta_gaussian(train["meta"], train["date"])
        except ValueError as error:
            raise SystemExit("--loss mmc: %s" % error)

    # The gradient probe: one fixed batch of the kind the run trains on, drawn
    # from a RandomState of its own so the training shuffle is not disturbed.
    probe_rs = np.random.RandomState(12345)
    if args.batch == "days":
        days = probe_rs.choice(len(day_rows), size=k, replace=False)
        rows, weights, _ = next(day_batches(day_rows, days, k))
        probe = {"rows": rows, "weights": weights}
    else:
        probe = {"rows": np.sort(probe_rs.choice(len(train["y"]), size=min(args.batch_size,
                                                                             len(train["y"])),
                                                 replace=False))}
    config = dict(vars(args), days_per_batch=k if args.batch == "days" else None,
                  penalty_denominator=float(denominator), **data_meta)
    if device.type == "cuda":
        config["gpu"] = "%s on %s" % (torch.cuda.get_device_name(device), os.uname().nodename)
    os.makedirs(out_dir, exist_ok=True)
    sink = (prediction_store(out_dir, train, held, args.max_epochs, keep_train=not args.no_train_preds)
            if args.save_preds else None)
    next_sink = next_block_store(out_dir, nxt, args.max_epochs) if nxt is not None else None
    snap = Snapshotter(out_dir, objective, train, held, probe, target_col=args.target,
                       metric_every=args.metric_every, meta=config, pred_sink=sink)

    net = build_net(train["X"].shape[1], args).to(device)      # initialised on the CPU, then moved
    Xt = torch.tensor(train["X"], dtype=torch.float64, device=device)
    yt = torch.tensor(train["y"], dtype=torch.float64, device=device).reshape(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    weights = [m.weight for m in net if isinstance(m, nn.Linear)]
    rng = np.random.RandomState(int(args.seed))
    n = len(train["y"]); bs = max(1, min(int(args.batch_size), n))
    per_epoch = int(math.ceil(len(day_rows) / float(k))) if args.batch == "days" else int(math.ceil(n / float(bs)))
    scheduler, when, cap = make_scheduler(args.schedule, opt, args, per_epoch)
    ticks, best, bad, stopped_at = 0, np.inf, 0, None
    snap.snapshot(0, net, lr=opt.param_groups[0]["lr"])
    if next_sink is not None:
        next_sink(0, net)

    for epoch in range(1, int(args.max_epochs) + 1):
        if args.batch == "rows":
            order = rng.permutation(n)
            batches = ((order[s:s + bs], None, None) for s in range(0, n, bs))
        else:
            batches = day_batches(day_rows, rng.permutation(len(day_rows)), k)
        tot, seen = 0.0, 0
        for rows, day_w, n_days in batches:
            idx = torch.as_tensor(rows, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True); out = net(Xt[idx])
            if args.loss in ("corr", "mmc"):
                days = torch.as_tensor(day_codes(train_days[rows]), device=device)
                mg = (torch.as_tensor(train_meta_g[rows], device=device)
                      if train_meta_g is not None else None)
                data = objective.data(out, yt[idx], days=days, meta=mg)
            elif day_w is None:
                data = 0.5 * ((yt[idx] - out) ** 2).mean()
            else:
                data = 0.5 * (torch.as_tensor(day_w, device=device).reshape(-1, 1)
                              * (yt[idx] - out) ** 2).sum()
            denom = len(idx) if args.l2 == "batch" else args.batch_size
            loss = data + (0.5 * args.alpha) * sum((w * w).sum() for w in weights) / denom
            loss.backward(); opt.step()
            unit = len(idx) if n_days is None else n_days
            tot += float(loss.detach()) * unit; seen += unit
            if when == "batch" and ticks < cap:
                scheduler.step(); ticks += 1
        ep = tot / max(seen, 1)
        if when == "epoch" and (cap is None or ticks < cap):
            scheduler.step(); ticks += 1
        elif when == "plateau":
            scheduler.step(ep)
        snap.snapshot(epoch, net, lr=opt.param_groups[0]["lr"], running_loss=ep)
        if next_sink is not None:
            next_sink(epoch, net)
        bad = bad + 1 if ep > best - 1e-4 else 0
        if ep < best:
            best = ep
        if bad > 10 and stopped_at is None:
            stopped_at = epoch                       # sklearn's rule: where the run WOULD stop
            if not args.no_stop:
                break

    if sink is not None:
        sink.flush()
    if next_sink is not None:
        next_sink.flush()
    snap.finish(net, extra={"stop_rule_epoch": stopped_at, "epochs_trained": epoch,
                            "minutes": round((time.time() - started) / 60.0, 2)})
    with open(os.path.join(out_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=1, default=str)
    last = snap.log["snapshots"][-1]
    m = last.get("metric") or {}
    print("%s: %d epochs (stop rule %s), held-out CORR %+.5f MMC %s payout %s, train CORR %+.5f"
          % (args.tag, epoch, stopped_at, m.get("held_corr", float("nan")),
             "%+.5f" % m["held_mmc"] if "held_mmc" in m else "n/a",
             "%+.5f" % m["held_payout"] if "held_payout" in m else "n/a",
             m.get("train_corr", float("nan"))))
    if not args.no_plot:
        from src.analysis.plot_training_diagnostics import plot_run
        print("wrote %s" % plot_run(args.tag))
    return snap.log


if __name__ == "__main__":
    main()
