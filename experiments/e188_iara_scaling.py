"""E188: compute scaling laws for IARA-family MLPs on Numerai Crypto (fenccausal).

Stages (run from the repository root):

    python3 experiments/e188_iara_scaling.py measure         # section 0 numbers, writes data/ cache
    python3 experiments/e188_iara_scaling.py smoke           # tiny end-to-end run, no outputs kept
    python3 experiments/e188_iara_scaling.py worker --device cuda [--max-params N] [--min-params N]
    python3 experiments/e188_iara_scaling.py status

Workers pull jobs from a file-claim queue under results/e188_iara_scaling/claims/
(O_EXCL lock files on the shared disk), so any number can run on any device and a
crashed job is retried by deleting its claim. Jobs:

    sweep_a<alpha>_P<P>_f<fold>   loss A alpha sweep, 3 sizes x 4 alphas x 4 folds, seed 42
    A_P<P>_f<fold>_s<seed>        loss A grid, claimable once alpha.json exists
    B_P<P>_f<fold>_s<seed>        loss B grid

alpha.json is written by the first worker that finds every sweep job finished: the
alpha with the best mean retained-checkpoint VALIDATION R^2 over sizes and folds.
The test window is never used to choose anything.

Protocol (spec sections 2-5): standardisation on training moments -> 5 x [Linear(w)
-> ReLU -> LayerNorm], dropout 0.1 after the 3 interior blocks -> Linear(1); He-normal
weights, zero biases; Adam lr 0.01, global grad-norm clip 1.0; float32, deterministic
kernels; E(P) = max(floor(0.2 P^0.8), 1) + 10 epochs, all run; the retained weights are
the centre of the best 5-epoch median-smoothed validation window, restored at the end.
Batches are whole eras (4 eras = 1,200 rows -> 33 steps/epoch; fold-0 loss B, which
drops its 87 meta-less eras, uses 1 era/batch -> 43 steps).
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

OUT = ROOT / "results" / "e188_iara_scaling"
DATA = OUT / "data"
RUNS = OUT / "runs"
CLAIMS = OUT / "claims"
FEATURE_SET = "fenccausal"
ERA = 300
FOLDS = 4
SEEDS = (42, 43, 44, 45, 46)
LR, CLIP, DROPOUT = 0.01, 1.0, 0.10
TARGETS = [j * k for k in (100, 1000, 10000, 100000) for j in range(2, 11)]   # the 36 requested sizes
# Integer widths cannot hit every target: 700 and 800 both give w=8 (P=737), so the
# grid runs that network once -> 35 distinct sizes. measure_grid.csv lists all 36.
SIZES = [t for t in TARGETS if t != 800]
SWEEP_SIZES = (1000, 10000, 100000)
SWEEP_ALPHAS = (0.0, 1e-6, 1e-4, 1e-2)
ERAS_PER_BATCH = 4
MIN_STEPS = 32


# --------------------------------------------------------------------------- sizes

def n_params(w, d):
    return (d + 1) * w + 4 * (w + 1) * w + 5 * 2 * w + (w + 1)


def width_for(target, d):
    lo, hi = 1, 10000
    while lo < hi:                                   # smallest w with P(w) >= target
        mid = (lo + hi) // 2
        if n_params(mid, d) < target:
            lo = mid + 1
        else:
            hi = mid
    candidates = [w for w in (lo - 1, lo) if w >= 1]
    return min(candidates, key=lambda w: abs(n_params(w, d) - target))


def epochs_for(p):
    return max(math.floor(0.2 * p ** 0.8), 1) + 10


def pflop_days(p, rows, epochs):
    return 6.0 * p * rows * epochs / 8.64e19


def log_grid(limit):
    out, k = [], 0
    while True:
        for m in (1, 3):
            v = m * 10 ** k
            if v > limit:
                return out
            out.append(v)
        k += 1


# --------------------------------------------------------------------------- data

def prepare():
    """Build fenccausal folds with E186's splitter (22-date purge/embargo gaps) and cache them."""
    import e186_ia_torch_mmc as e186
    DATA.mkdir(parents=True, exist_ok=True)
    report = []
    for name, fold, cols, arrays in e186.tasks():
        if name != "IA_01":                    # IA_01 is the fenccausal model in E186's task list
            continue
        payload = {}
        for split in ("train", "val", "test"):
            a = arrays[split]
            payload.update({f"{split}_x": a["x"].astype("float32"), f"{split}_y": a["y"].astype("float32"),
                            f"{split}_meta": a["meta"].astype("float32"), f"{split}_era": a["date"],
                            f"{split}_dates": a["dates"].astype("datetime64[D]").astype(str)})
        np.savez(DATA / f"fold{fold}.npz", cols=np.array(cols), **payload)
        report.append(fold)
    assert sorted(report) == list(range(FOLDS))


def load_fold(fold):
    z = np.load(DATA / f"fold{fold}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def measure():
    if not (DATA / "fold3.npz").exists():
        prepare()
    rows = []
    for fold in range(FOLDS):
        f = load_fold(fold)
        rec = dict(fold=fold, n_features=f["train_x"].shape[1])
        for split in ("train", "val", "test"):
            counts = np.bincount(f[f"{split}_era"])
            meta_eras = np.array([np.isfinite(f[f"{split}_meta"][f[f"{split}_era"] == e]).any()
                                  for e in range(len(counts))])
            rec.update({f"n_rows_{split}": int(counts.sum()), f"n_eras_{split}": len(counts),
                        f"rows_per_era_{split}": f"{counts.min()}/{int(np.median(counts))}/{counts.max()}",
                        f"eras_without_meta_{split}": int((~meta_eras).sum()),
                        f"{split}_span": f"{f[f'{split}_dates'][0]}..{f[f'{split}_dates'][-1]}"})
        # Leakage check: no era in train is in its own val/test, and the gaps are >= 20 dates.
        tr, va, te = (set(f[f"{s}_dates"]) for s in ("train", "val", "test"))
        assert not (tr & va) and not (tr & te) and not (va & te)
        rows.append(rec)
    table = pd.DataFrame(rows)
    d = int(table.n_features.iloc[0])
    grid = pd.DataFrame([dict(target_P=t, w=width_for(t, d), P=n_params(width_for(t, d), d)) for t in TARGETS])
    grid["epochs"] = grid.P.map(epochs_for)
    grid["P_over_N"] = grid.P / 39000
    grid["pflop_days"] = [pflop_days(p, 39000, e) for p, e in zip(grid.P, grid.epochs)]
    grid["steps_per_epoch_A"] = math.ceil(130 / ERAS_PER_BATCH)
    pd.set_option("display.width", 220)
    print(table.T.to_string())
    print(grid.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "measure_splits.csv", index=False)
    grid.to_csv(OUT / "measure_grid.csv", index=False)
    return table, grid


# --------------------------------------------------------------------------- scoring

def _gauss_rank(a):
    n = a.shape[1]
    return ndtri((rankdata(a, axis=1) - .5) / n)


def era_scores(pred, y, meta):
    """Per-era CORR, MMC, payout and CWMM for (n_eras, 300) arrays with full meta.

    Vectorised copy of volatility_experiment.score_date (checked against it in
    check_scorer). CWMM is the Pearson correlation of the gaussianised ranks of
    the prediction and the meta model.
    """
    p = _gauss_rank(pred)
    x = np.sign(p) * np.abs(p) ** 1.5
    yc = y - y.mean(1, keepdims=True)
    t = np.sign(yc) * np.abs(yc) ** 1.5
    x = x - x.mean(1, keepdims=True)
    t = t - t.mean(1, keepdims=True)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(t, axis=1)
    corr = np.divide((x * t).sum(1), den, out=np.zeros(len(den)), where=den > 0)
    m = _gauss_rank(meta)
    energy = (m * m).sum(1, keepdims=True)
    neutral = p - m * ((m * p).sum(1, keepdims=True) / energy)
    sy = y * 4
    mmc = ((sy - sy.mean(1, keepdims=True)) * neutral).sum(1) / y.shape[1]
    cw = (p * m).sum(1) / np.sqrt((p * p).sum(1) * energy[:, 0])
    return corr, mmc, .1 * corr + mmc, cw


def check_scorer():
    from src.analysis.volatility_experiment import score_date
    rng = np.random.default_rng(1)
    pred = rng.normal(size=(5, ERA))
    y = rng.integers(0, 5, (5, ERA)) / 4
    meta = rng.random((5, ERA))
    corr, mmc, pay, _ = era_scores(pred, y, meta)
    for i in range(5):
        c, cm, mm = score_date(pred[i][:, None], y[i], meta[i])
        np.testing.assert_allclose([corr[i], mmc[i]], [c[0], mm[0]], atol=1e-10)


def summarise_eras(pay):
    pay = np.asarray(pay)
    sd = pay.std(ddof=0)
    return dict(mean=float(pay.mean()), std=float(sd), hit=float((pay > 0).mean()),
                sharpe=float(pay.mean() / sd) if sd > 0 else float("nan"))


# --------------------------------------------------------------------------- model

def build(torch, nn, d, w, seed):
    torch.manual_seed(seed)
    layers, width_in = [], d
    for k in range(5):
        lin = nn.Linear(width_in, w)
        nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU(), nn.LayerNorm(w)]
        if k in (1, 2, 3):
            layers.append(nn.Dropout(DROPOUT))
        width_in = w
    head = nn.Linear(w, 1)
    nn.init.kaiming_normal_(head.weight, nonlinearity="linear")
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)


def mmc_loss(out, y, ref, n_eras):
    """-(mean over eras of corr(pred neutralised on meta, target)). Whole eras of 300 rows."""
    o = out.view(n_eras, ERA)
    t = y.view(n_eras, ERA)
    r = ref.view(n_eras, ERA)
    o = o - o.mean(1, keepdim=True)
    r = r - r.mean(1, keepdim=True)
    t = t - t.mean(1, keepdim=True)
    resid = o - r * (o * r).sum(1, keepdim=True) / ((r * r).sum(1, keepdim=True) + 1e-12)
    corr = (resid * t).sum(1) / (resid.norm(dim=1) * t.norm(dim=1) + 1e-12)
    return -corr.mean(), corr.detach()


# --------------------------------------------------------------------------- one run

def run(job, device, keep=True, epochs_override=None):
    import torch
    from torch import nn

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    kind, p_target, fold, seed, alpha = job["kind"], job["P"], job["fold"], job["seed"], job["alpha"]
    loss = "A" if kind in ("A", "sweep") else "B"
    f = load_fold(fold)
    d = f["train_x"].shape[1]
    w = width_for(p_target, d)
    p = n_params(w, d)
    epochs = epochs_override or epochs_for(p)
    dev = torch.device(device)

    # ----- data. Target centred at 0.5 so "R^2 vs a zero benchmark" means vs predicting 0.5.
    x, y, meta, era = f["train_x"], f["train_y"] - .5, f["train_meta"], f["train_era"]
    dropped = []
    if loss == "B":
        keep_era = np.array([np.isfinite(meta[era == e]).all() for e in range(era.max() + 1)])
        dropped = [str(f["train_dates"][era == e][0]) for e in np.flatnonzero(~keep_era)]
        rows = keep_era[era]
        x, y, meta, era = x[rows], y[rows], meta[rows], pd.factorize(era[rows])[0]
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    std = lambda a: ((a - mu) / sd).astype("float32")
    n_eras = era.max() + 1
    assert np.all(np.bincount(era) == ERA)
    per_batch = ERAS_PER_BATCH
    while per_batch > 1 and math.ceil(n_eras / per_batch) < MIN_STEPS:
        per_batch -= 1
    steps = math.ceil(n_eras / per_batch)
    rows_train = len(y)

    X = torch.as_tensor(std(x), device=dev).view(n_eras, ERA, d)
    Y = torch.as_tensor(y, device=dev).view(n_eras, ERA, 1)
    R = torch.as_tensor(np.nan_to_num(meta, nan=0.5), device=dev).view(n_eras, ERA, 1)
    vx, vy, vm = std(f["val_x"]), f["val_y"] - .5, f["val_meta"]
    tx, ty, tm = std(f["test_x"]), f["test_y"] - .5, f["test_meta"]
    VX = torch.as_tensor(vx, device=dev)
    TX = torch.as_tensor(tx, device=dev)
    XA = torch.as_tensor(std(x), device=dev)
    nv, nt = len(vy) // ERA, len(ty) // ERA
    tr_meta_eras = np.array([np.isfinite(meta[era == e]).all() for e in range(n_eras)])

    net = build(torch, nn, d, w, seed).to(dev)
    assert sum(q.numel() for q in net.parameters()) == p
    linears = [m for m in net if isinstance(m, nn.Linear)]
    weights = [m.weight for m in linears]
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rng = np.random.RandomState(seed)

    def predict(t):
        net.eval()
        with torch.no_grad():
            out = net(t).view(-1).float().cpu().numpy()
        net.train()
        return out

    def scores(pred, yy, mm, n):
        pr, yr, mr = pred.reshape(n, ERA), (yy + .5).reshape(n, ERA), mm.reshape(n, ERA)
        return era_scores(pr, yr, mr)

    def train_scores(pred):
        pr, yr = pred.reshape(n_eras, ERA), (y + .5).reshape(n_eras, ERA)
        c, mmc, pay, cw = era_scores(pr[tr_meta_eras], yr[tr_meta_eras], meta.reshape(n_eras, ERA)[tr_meta_eras]) \
            if tr_meta_eras.any() else (np.array([np.nan]),) * 4
        return c, mmc, pay, cw

    diag_epochs = set(log_grid(epochs)) | set(range(max(1, epochs // 100), epochs + 1, max(1, epochs // 100)))
    diag_epochs |= set(range(max(1, int(epochs * .8)), epochs + 1, max(1, epochs // 50)))
    diag_epochs.add(epochs)
    snap_steps = set(log_grid(epochs * steps))
    snap_eras = [0, nv // 2, nv - 1]

    epoch_rows, diag_rows, snap_rows, estim_rows = [], [], [], []
    trail = []                      # (epoch, val_score, state) for the last 5 epochs
    best = (-np.inf, None, None)    # (smoothed score, epoch, state)
    prev_val = None
    step_count = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_eras)
        gnorms, losses, datas = [], [], []
        do_diag = epoch in diag_epochs
        for s in range(steps):
            idx = torch.as_tensor(order[s * per_batch:(s + 1) * per_batch], device=dev)
            k = len(idx)
            xb, yb = X[idx].reshape(-1, d), Y[idx].reshape(-1, 1)
            last = do_diag and s == steps - 1
            if last:
                before = [wt.detach().clone() for wt in weights]
            opt.zero_grad(set_to_none=True)
            out = net(xb)
            if loss == "A":
                data = ((out - yb) ** 2).mean()
                l2 = alpha * sum((wt * wt).sum() for wt in weights) if alpha else out.sum() * 0
                total = data + l2
            else:
                data, _ = mmc_loss(out, yb, R[idx].reshape(-1, 1), k)
                total = data
            if not torch.isfinite(total):
                raise FloatingPointError(f"{job['id']}: non-finite loss at epoch {epoch}")
            total.backward()
            gnorms.append(torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP))
            opt.step()
            losses.append(total.detach())
            datas.append(data.detach())
            step_count += 1
            if step_count in snap_steps:
                vp = predict(VX)
                for e in snap_eras:
                    sl = slice(e * ERA, (e + 1) * ERA)
                    snap_rows.append(pd.DataFrame(dict(step=step_count, epoch=epoch, era=e, pred=vp[sl],
                                                       target=vy[sl] + .5, meta=vm[sl])))
            if last:
                ratios = [float((wt.detach() - b).norm() / (b.norm() + 1e-12)) for wt, b in zip(weights, before)]
        g = torch.stack(gnorms).float().cpu().numpy()
        train_loss = float(torch.stack(losses).mean())
        train_data = float(torch.stack(datas).mean())

        vp = predict(VX)
        vc, vmmc, vpay, vcw = scores(vp, vy, vm, nv)
        v_mse = float(np.mean((vp - vy) ** 2))
        v_r2 = 1 - float(np.sum((vp - vy) ** 2) / np.sum(vy ** 2))
        val_score = v_r2 if loss == "A" else float(vmmc.mean())
        churn = float(np.corrcoef(rankdata(vp), rankdata(prev_val))[0, 1]) if prev_val is not None else np.nan
        prev_val = vp
        ev = summarise_eras(vpay)
        epoch_rows.append(dict(epoch=epoch, step=step_count, cum_pflop_days=pflop_days(p, rows_train, epoch),
                               train_loss=train_loss, train_data_loss=train_data,
                               grad_norm_mean=float(g.mean()), grad_norm_min=float(g.min()), grad_norm_max=float(g.max()),
                               val_mse=v_mse, val_rmse=math.sqrt(v_mse), val_r2=v_r2, val_corr=float(vc.mean()),
                               val_mmc=float(vmmc.mean()), val_cwmm=float(vcw.mean()), val_payout=ev["mean"],
                               val_era_std=ev["std"], val_hit=ev["hit"], val_sharpe=ev["sharpe"],
                               val_pred_std=float(vp.std()),
                               val_pred_era_std=float(vp.reshape(nv, ERA).std(1).mean()),
                               churn=churn, val_score=val_score))

        trail.append((epoch, val_score, {k_: v.detach().clone() for k_, v in net.state_dict().items()}))
        if len(trail) > 5:
            trail.pop(0)
        if len(trail) == 5:
            smooth = float(np.median([t[1] for t in trail]))
            if smooth > best[0]:
                best = (smooth, trail[2][0], trail[2][2])
        elif epochs < 5 and val_score > best[0]:
            best = (val_score, epoch, trail[-1][2])

        if do_diag:
            tp = predict(XA)
            tc, tmmc, tpay, tcw = train_scores(tp)
            row = dict(epoch=epoch, step=step_count, cum_pflop_days=pflop_days(p, rows_train, epoch),
                       train_mse=float(np.mean((tp - y) ** 2)), train_r2=1 - float(np.sum((tp - y) ** 2) / np.sum(y ** 2)),
                       train_corr=float(np.nanmean(tc)), train_mmc=float(np.nanmean(tmmc)),
                       train_cwmm=float(np.nanmean(tcw)), train_pearson=float(np.corrcoef(tp, y)[0, 1]),
                       train_pred_std=float(tp.std()))
            for i, (wt, r_) in enumerate(zip(weights, ratios)):
                row[f"wnorm_{i}"] = float(wt.detach().norm())
                row[f"update_ratio_{i}"] = r_
            diag_rows.append(row)
            if loss == "B" and (epoch in set(log_grid(epochs)) or epoch == epochs):
                net.eval()
                with torch.no_grad():
                    full, per_era = mmc_loss(net(X.view(-1, d)), Y.view(-1, 1), R.view(-1, 1), n_eras)
                batch_vals = [float(per_era[torch.as_tensor(order[s * per_batch:(s + 1) * per_batch])].mean())
                              for s in range(steps)]
                grads = []
                for half in (order[:per_batch], order[per_batch:2 * per_batch]):
                    idx = torch.as_tensor(half, device=dev)
                    net.zero_grad(set_to_none=True)
                    lo_, _ = mmc_loss(net(X[idx].reshape(-1, d)), Y[idx].reshape(-1, 1), R[idx].reshape(-1, 1), len(idx))
                    lo_.backward()
                    grads.append(torch.cat([q.grad.reshape(-1) for q in net.parameters()]))
                net.zero_grad(set_to_none=True)
                net.train()
                cos = float(torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0))
                estim_rows.append(pd.DataFrame(dict(epoch=epoch, full_epoch_mmc=float(-full), batch_mmc=batch_vals,
                                                    grad_cosine=cos)))

    seconds = time.perf_counter() - started
    final_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
    final_test = predict(TX)
    net.load_state_dict(best[2])
    vp = predict(VX)
    test = predict(TX)
    assert np.isfinite(test).all()
    tc, tmmc, tpay, tcw = scores(test, ty, tm, nt)
    fc, fmmc, fpay, _ = scores(final_test, ty, tm, nt)
    te = summarise_eras(tpay)
    test_mse = float(np.mean((test - ty) ** 2))
    summary = dict(id=job["id"], kind=kind, loss=loss, alpha=alpha, target_P=p_target, P=p, w=w, d=d, fold=fold,
                   seed=seed, epochs=epochs, steps_per_epoch=steps, eras_per_batch=per_batch, n_rows_train=rows_train,
                   n_eras_train=int(n_eras), dropped_eras=len(dropped), P_over_N=p / rows_train,
                   pflop_days=pflop_days(p, rows_train, epochs), retained_epoch=int(best[1]),
                   retained_val_score=float(best[0]), val_r2_retained=1 - float(np.sum((vp - vy) ** 2) / np.sum(vy ** 2)),
                   test_rmse=math.sqrt(test_mse), test_r2=1 - float(np.sum((test - ty) ** 2) / np.sum(ty ** 2)),
                   test_corr=float(tc.mean()), test_mmc=float(tmmc.mean()), test_cwmm=float(tcw.mean()),
                   test_payout=te["mean"], test_era_sharpe=te["sharpe"], test_hit=te["hit"],
                   final_test_payout=float(fpay.mean()), final_test_rmse=math.sqrt(float(np.mean((final_test - ty) ** 2))),
                   seconds=seconds, device=device, host=os.uname().nodename, torch=torch.__version__)
    if not keep:
        return summary, pd.DataFrame(epoch_rows), pd.DataFrame(diag_rows)
    out = RUNS / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    tag = dict(id=job["id"], loss=loss, P=p, fold=fold, seed=seed)
    pd.DataFrame(epoch_rows).assign(**tag).to_parquet(out / "epochs.parquet", index=False)
    pd.DataFrame(diag_rows).assign(**tag).to_parquet(out / "diag.parquet", index=False)
    if snap_rows:
        pd.concat(snap_rows).assign(**tag).to_parquet(out / "snapshots.parquet", index=False)
    if estim_rows:
        pd.concat(estim_rows).assign(**tag).to_parquet(out / "estimability.parquet", index=False)
    dates = f["test_dates"].reshape(nt, ERA)[:, 0]
    pd.DataFrame(dict(date=dates, corr=tc, mmc=tmmc, payout=tpay, cwmm=tcw,
                      rmse=np.sqrt(((test - ty) ** 2).reshape(nt, ERA).mean(1)))).assign(**tag).to_parquet(
        out / "test_eras.parquet", index=False)
    if dropped:
        (out / "dropped_eras.json").write_text(json.dumps(dropped))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))     # written last = done marker
    return summary, None, None


# --------------------------------------------------------------------------- queue

def jobs():
    out = []
    for p in SWEEP_SIZES:
        for a in SWEEP_ALPHAS:
            for fold in range(FOLDS):
                out.append(dict(id=f"sweep_a{a:g}_P{p}_f{fold}", kind="sweep", P=p, fold=fold, seed=42, alpha=a))
    for kind in ("B", "A"):
        for p in SIZES:
            for fold in range(FOLDS):
                for seed in SEEDS:
                    out.append(dict(id=f"{kind}_P{p}_f{fold}_s{seed}", kind=kind, P=p, fold=fold, seed=seed, alpha=None))
    return out


def done(job):
    return (RUNS / job["id"] / "summary.json").exists()


def chosen_alpha():
    path = OUT / "alpha.json"
    if path.exists():
        return json.loads(path.read_text())["alpha"]
    sweep = [j for j in jobs() if j["kind"] == "sweep"]
    if not all(done(j) for j in sweep):
        return None
    rows = pd.DataFrame([json.loads((RUNS / j["id"] / "summary.json").read_text()) for j in sweep])
    table = rows.groupby("alpha").val_r2_retained.mean()
    alpha = float(table.idxmax())
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as fh:
            json.dump(dict(alpha=alpha, mean_val_r2_by_alpha={f"{k:g}": v for k, v in table.items()},
                           rule="max mean retained-checkpoint validation R^2 over sizes 1k/10k/100k x 4 folds",
                           written=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=1)
    except FileExistsError:
        pass
    return json.loads(path.read_text())["alpha"]


def claim(job):
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(CLAIMS / job["id"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{os.uname().nodename} {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return True


def worker(device, min_params, max_params):
    d = load_fold(0)["train_x"].shape[1]
    while True:
        alpha = chosen_alpha()
        todo = [j for j in jobs() if not done(j) and not (CLAIMS / j["id"]).exists()
                and min_params <= n_params(width_for(j["P"], d), d) <= max_params]
        ready = [j for j in todo if j["kind"] != "A" or alpha is not None]
        if not ready:
            if not todo:
                print("no jobs left for this worker", flush=True)
                return
            time.sleep(60)                           # loss A waits for the sweep
            continue
        # Sweep first, then the largest (longest) jobs, so the tail is short.
        ready.sort(key=lambda j: (j["kind"] != "sweep", -j["P"], j["fold"], j["seed"]))
        for job in ready:
            if claim(job):
                break
        else:
            continue
        if job["kind"] == "A":
            job = dict(job, alpha=alpha)
        t = time.perf_counter()
        try:
            summary, _, _ = run(job, device)
            print(f"done {job['id']} {time.perf_counter() - t:.0f}s retained@{summary['retained_epoch']}/"
                  f"{summary['epochs']} test_payout {summary['test_payout']:.4f}", flush=True)
        except Exception:
            (CLAIMS / (job["id"] + ".failed")).write_text(traceback.format_exc())
            print(f"FAILED {job['id']}\n{traceback.format_exc()}", flush=True)


def status():
    all_jobs = jobs()
    d = load_fold(0)["train_x"].shape[1]
    rows = []
    for j in all_jobs:
        state = "done" if done(j) else ("failed" if (CLAIMS / (j["id"] + ".failed")).exists()
                                        else "running" if (CLAIMS / j["id"]).exists() else "queued")
        p = n_params(width_for(j["P"], d), d)
        rows.append(dict(kind=j["kind"], state=state, epochs=epochs_for(p)))
    t = pd.DataFrame(rows)
    print(pd.crosstab(t.kind, t.state, margins=True))
    w = t.groupby("state").epochs.sum()
    print(f"epoch-weighted progress: {w.get('done', 0) / w.sum():.1%}")
    path = OUT / "alpha.json"
    print("alpha.json:", path.read_text() if path.exists() else "not yet")


def smoke(device):
    check_scorer()
    for kind, a in (("A", 1e-4), ("B", None)):
        for fold in (0, 3):
            job = dict(id=f"smoke_{kind}", kind=kind, P=2000, fold=fold, seed=42, alpha=a)
            s, ep, dg = run(job, device, keep=False, epochs_override=8)
            print(kind, "fold", fold, {k: s[k] for k in ("P", "w", "steps_per_epoch", "eras_per_batch", "n_eras_train",
                                                          "dropped_eras", "retained_epoch", "test_payout", "seconds")})
            assert len(ep) == 8 and len(dg) >= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure", "smoke", "worker", "status"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-params", type=int, default=0)
    ap.add_argument("--max-params", type=int, default=10 ** 9)
    args = ap.parse_args()
    if args.stage == "measure":
        measure()
    elif args.stage == "smoke":
        smoke(args.device)
    elif args.stage == "worker":
        worker(args.device, args.min_params, args.max_params)
    else:
        status()


if __name__ == "__main__":
    main()
