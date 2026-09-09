"""
Fine-tune TEMPO on Numerai crypto, then evaluate it exactly like the zero-shot
E40 (study E41).

    # smoke test (tiny), on the GPU box in the tempo venv:
    HF_HOME=/scratch/danielco/hf CUDA_VISIBLE_DEVICES=1 \
        /scratch/danielco/tempo/venv/bin/python -m src.analysis.tempo_finetune \
        --run-name e41_smoke --epochs 1 --limit 500

    # a real arm:
    HF_HOME=/scratch/danielco/hf CUDA_VISIBLE_DEVICES=1 \
        /scratch/danielco/tempo/venv/bin/python -m src.analysis.tempo_finetune \
        --run-name e41_sg_mse --scheme single --early-stop-metric mse

E41 differs from E40 (src.analysis.tempo_zeroshot) in EXACTLY one thing: the model
weights. The eval half -- the leak boundary, the task set, the signal, the scoring
-- is E40's, reused verbatim by importing build_tasks / forecast_signal / the four
constants below. So E40 (zero-shot) vs E41 (fine-tuned) isolates fine-tuning.

Design (see the plan): load the pretrained TEMPO exactly as E40 does, fine-tune
that in-memory object (never reconstruct TEMPO(cfg) -- the shipped .pth loads with
strict=False and reconstruction silently drops weights), then evaluate the
fine-tuned model in the same process. Only the LoRA adapters + the fresh heads +
the RevIN affine params are trainable (GPT-2 body is frozen); we optimise whatever
came out requires_grad=True.

Training windows are (336 context -> 20 horizon) slices over each symbol's RESOLVED
target series -- gap-compressed, the same series E40 feeds at eval -- kept only when
the horizon-end date is on or before the leak cutoff (min scored date - 20-period
embargo = 2024-06-27). The loss is MSE on the first HORIZON=20 output steps, which
is exactly what forecast_signal averages. The forward() call MUST pass test=True:
plain forward(x) dereferences an unbound STL loss and raises NameError, and test=True
does not wrap in no_grad, so backprop still works. Its outputs are RevIN-denormalised
to the raw {0..1} target scale, so the MSE target is the raw horizon values.

Two schemes and two selection metrics (the E41 2x2):
  --scheme single : one fine-tune on all pre-cutoff windows -> predict all 510 dates.
  --scheme walk   : expanding refit every --refit-every scored dates (~4 blocks);
                    each block fine-tunes on resolved history up to block_start-embargo.
  --early-stop-metric {mse,corr} : validation MSE (native) or validation per-date
                    cross-sectional CORR (the payable metric). Training loss is MSE
                    either way; only checkpoint selection differs.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.analysis.tempo_zeroshot import (
    EMBARGO, CONTEXT_LEN, HORIZON, MIN_CONTEXT,   # noqa: F401 (import, don't redefine)
    load_panel, scored_dates, build_tasks, load_tempo, forecast_signal,
)

WIN = CONTEXT_LEN + HORIZON      # 356: a full (context -> horizon) training window


def subsample(items, k):
    """Up to k evenly-spaced elements of a sorted list (all if k<=0 or fewer)."""
    if k <= 0 or len(items) <= k:
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, k).round().astype(int))
    return [items[i] for i in idx]


def build_train_windows(df, target, max_date, stride, cap_per_symbol):
    """(context 336 -> horizon 20) windows over each symbol's resolved series.

    A window is indexed by its horizon-end position e in the symbol's sorted
    resolved series: context = ys[e-355 : e-19] (336 own-steps), target = the next
    ys[e-19 : e+1] (20 own-steps). Kept only when dts[e] <= max_date, so every
    supervised label is resolved and observable before the eval window.
    """
    xs, ys_out = [], []
    for _, g in df.groupby("ucid", sort=False):
        dts = g["date"].values
        ys = g[target].to_numpy(dtype=np.float32)
        n = len(ys)
        if n < WIN:
            continue
        last = int(np.searchsorted(dts, max_date, side="right")) - 1   # last dts<=max_date
        if last < WIN - 1:
            continue
        ends = list(range(WIN - 1, last + 1))[::stride]
        if cap_per_symbol and len(ends) > cap_per_symbol:
            ends = ends[-cap_per_symbol:]                              # most recent
        for e in ends:
            xs.append(ys[e - (WIN - 1):e - HORIZON + 1])              # 336
            ys_out.append(ys[e - HORIZON + 1:e + 1])                  # 20
    if not xs:
        return np.empty((0, CONTEXT_LEN), np.float32), np.empty((0, HORIZON), np.float32)
    return np.asarray(xs, np.float32), np.asarray(ys_out, np.float32)


def compute_val(model, val_ctxs, val_dates, val_tgts, metric):
    """Validation statistic on E40-style tasks; higher-better for corr, lower for mse."""
    if not val_ctxs:
        return float("nan")
    sig = forecast_signal(model, val_ctxs)            # per-series predict() -> eval mode
    d = pd.DataFrame({"date": val_dates, "sig": sig, "tgt": val_tgts})
    if metric == "corr":
        def cs(grp):
            if len(grp) < 3 or grp["sig"].std() == 0 or grp["tgt"].std() == 0:
                return np.nan
            return float(np.corrcoef(grp["sig"], grp["tgt"])[0, 1])
        return float(d.groupby("date").apply(cs).mean())
    return float(np.mean((d["sig"].to_numpy() - d["tgt"].to_numpy()) ** 2))


def finetune_once(train_x, train_y, val_ctxs, val_dates, val_tgts, args, tag=""):
    """Fresh pretrained TEMPO -> fine-tune with early stopping -> return best model."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda:0" if (args.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    model = load_tempo(args.device)                   # known-good pretrained object
    params = [p for p in model.parameters() if p.requires_grad]
    print("  %strainable tensors %d  params %d"
          % (tag, len(params), sum(p.numel() for p in params)), flush=True)

    opt = torch.optim.Adam(params, lr=args.lr)
    mse = torch.nn.MSELoss()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    gen = torch.Generator(); gen.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=gen)

    higher_better = args.early_stop_metric == "corr"
    best = -1e18 if higher_better else 1e18
    best_state, bad = None, 0

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        tot, nb = 0.0, 0
        for i, (xb, yb) in enumerate(loader):
            xb = xb.unsqueeze(-1).to(device)          # [B,336,1]
            yb = yb.unsqueeze(-1).to(device)          # [B,20,1]
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs, _ = model.forward(xb, itr=0, test=True)   # [B,96,1], raw scale
                loss = mse(outputs[:, :HORIZON, :], yb) / args.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item() * args.grad_accum; nb += 1
        if nb % args.grad_accum != 0:                 # flush a partial accumulation
            scaler.step(opt); scaler.update(); opt.zero_grad()

        val = compute_val(model, val_ctxs, val_dates, val_tgts, args.early_stop_metric)
        model.train()                                 # predict() left it in eval mode
        improved = (val > best) if higher_better else (val < best)
        print("  %sepoch %d/%d  train_mse %.6f  val_%s %.6f%s"
              % (tag, epoch + 1, args.epochs, tot / max(nb, 1),
                 args.early_stop_metric, val, "  *" if improved else ""), flush=True)
        if improved or best_state is None:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("  %searly stop" % tag, flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _val_window(global_dates, pos, upto_date, args):
    """Leak-safe inner validation: subsampled tail dates <= upto_date, and the
    train-window cutoff (val_embargo before the first val date)."""
    pre = [d for d in global_dates if d <= upto_date]
    n_val = max(1, int(len(pre) * args.val_frac))
    val_dates = subsample(pre[-n_val:], args.val_dates)
    train_max = global_dates[pos[min(val_dates)] - args.val_embargo]
    return val_dates, train_max


def run_single(df, target, scored, global_dates, pos, cutoff, args):
    val_dates, train_max = _val_window(global_dates, pos, cutoff, args)
    tx, ty = build_train_windows(df, target, train_max, args.stride, args.max_windows_per_symbol)
    if args.limit:
        tx, ty = tx[:args.limit], ty[:args.limit]
    vsym, vdate, vtgt, vctx = build_tasks(df, target, set(val_dates))
    print("[single] cutoff %s  train_max %s  train windows %d  val dates %d tasks %d"
          % (cutoff, train_max, len(tx), len(val_dates), len(vctx)), flush=True)
    assert train_max < min(val_dates), "train cutoff must precede validation"
    assert max(val_dates) <= cutoff, "validation labels must be resolved by the cutoff"
    assert min(scored) > cutoff, "eval window must sit after the leak cutoff"

    model = finetune_once(tx, ty, vctx, vdate, vtgt, args)
    esym, edate, etgt, ectx = build_tasks(df, target, scored)
    if args.limit:
        esym, edate, etgt, ectx = (esym[:args.limit], edate[:args.limit],
                                   etgt[:args.limit], ectx[:args.limit])
    preds = forecast_signal(model, ectx)
    out = pd.DataFrame({"symbol": esym, "date": edate, target: etgt, "prediction": preds})
    return out, model


def run_walk(df, target, scored, global_dates, pos, cutoff, args):
    import torch
    scored_sorted = sorted(scored)
    frames, last_model = [], None
    for b0 in range(0, len(scored_sorted), args.refit_every):
        block = scored_sorted[b0:b0 + args.refit_every]
        train_end = global_dates[pos[block[0]] - EMBARGO]             # wall before the block
        val_dates, train_max = _val_window(global_dates, pos, train_end, args)
        tx, ty = build_train_windows(df, target, train_max, args.stride,
                                     args.max_windows_per_symbol)
        vsym, vdate, vtgt, vctx = build_tasks(df, target, set(val_dates))
        if args.limit:                                    # smoke: cap train + val
            tx, ty = tx[:args.limit], ty[:args.limit]
            vdate, vtgt, vctx = vdate[:args.limit], vtgt[:args.limit], vctx[:args.limit]
        tag = "b%d " % (b0 // args.refit_every)
        print("[walk %sblock %s..%s]  train_end %s  windows %d  val %d"
              % (tag, block[0], block[-1], train_end, len(tx), len(vctx)), flush=True)
        assert train_max < min(val_dates) and max(val_dates) <= train_end
        model = finetune_once(tx, ty, vctx, vdate, vtgt, args, tag=tag)
        bsym, bdate, btgt, bctx = build_tasks(df, target, set(block))
        if args.limit:
            bsym, bdate, btgt, bctx = (bsym[:args.limit], bdate[:args.limit],
                                       btgt[:args.limit], bctx[:args.limit])
        preds = forecast_signal(model, bctx)
        frames.append(pd.DataFrame({"symbol": bsym, "date": bdate,
                                    target: btgt, "prediction": preds}))
        last_model = model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True), last_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="crypto/v2.0")
    ap.add_argument("--target", default="target_binned_return_20")
    ap.add_argument("--match-run", default="tabpfn/e19_fenccausal")
    ap.add_argument("--run-name", default="e41_target")
    ap.add_argument("--scheme", choices=["single", "walk"], default="single")
    ap.add_argument("--early-stop-metric", choices=["mse", "corr"], default="mse")
    ap.add_argument("--cutoff", default="auto")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-embargo", type=int, default=EMBARGO)
    ap.add_argument("--val-dates", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--refit-every", type=int, default=130)
    ap.add_argument("--max-windows-per-symbol", type=int, default=0)
    ap.add_argument("--stl-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.stl_weight:
        print("note: --stl-weight ignored; training uses pure horizon MSE (test=True path)")

    df = load_panel(args.data_version, args.target)
    scored = scored_dates(args.match_run)
    global_dates = np.sort(df["date"].unique())
    pos = {d: i for i, d in enumerate(global_dates)}
    scored = {d for d in scored if d in pos}
    min_scored = min(scored)
    cutoff = (global_dates[pos[min_scored] - EMBARGO] if args.cutoff == "auto"
              else np.datetime64(args.cutoff))
    print("scheme %s  select %s  CUTOFF %s  min_scored %s  scored %d dates"
          % (args.scheme, args.early_stop_metric, cutoff, min_scored, len(scored)), flush=True)

    runner = run_single if args.scheme == "single" else run_walk
    out, model = runner(df, args.target, scored, global_dates, pos, cutoff, args)

    dest = "results/forecast/crypto/tempo/%s.csv" % args.run_name
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False)
    if model is not None:
        import torch
        mp = "models/tempo/%s.pth" % args.run_name
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        torch.save(model.state_dict(), mp)
        print("saved %s" % mp, flush=True)
    print("wrote %s  (%d rows)" % (dest, len(out)), flush=True)
    print("score: python -m src.analysis.timesfm_score --model tempo --run-name %s"
          % args.run_name, flush=True)


if __name__ == "__main__":
    main()
