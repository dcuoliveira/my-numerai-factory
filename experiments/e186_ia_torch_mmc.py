"""E186: torch MLP, sklearn initialisation, trained on an MMC loss, for IA_01..IA_04.

Run from the repository root:

    python3 experiments/e186_ia_torch_mmc.py [--device cuda] [--workers 4] [--seeds 42 43 44 45 46]
    python3 experiments/e186_ia_torch_mmc.py --smoke      # one tiny fit per model, no outputs

Design (registered in experiments/EXPERIMENTS.md, E186):

* Same data, features, folds and pinned hyperparameters as E185 (results/ia_mlp_10x):
  four rolling holdouts, 130 training dates, 22-date gaps, 30 validation dates,
  130 test dates each, 520 scored dates. IA_01 uses its cached best_params.
* Network: src.prediction.torch_mlp.TorchMLPRegressor._build, i.e. sklearn's
  Glorot-uniform initialisation (weights AND biases), float64, relu, Adam with
  sklearn's betas/eps, and sklearn's L2 term (0.5 * alpha * sum W^2 / batch rows).
* Epochs: the Timmermann-Vulicevic reference schedule E = max(floor(0.2 P^0.8), 1) + 10,
  run in full with no early stopping. The final epoch is the primary arm.
* Arms (the loss is the only thing that changes between them):
    torch_mmc  -- per-date negative BMC of the prediction against the real meta model
                  (src.prediction.nn.losses.BMCLoss, averaged over the dates in a batch)
    torch_mse  -- sklearn's half-MSE, same loop, same batches: the control
* Batches are WHOLE DATES (dates shuffled each epoch, ~batch_size rows per step), so
  each per-date term sees the full cross-section rather than ~8 rows of it.
* Fold 0's training window starts before the published meta model (2024-05-07). On a
  training date with no meta row the reference is set to a constant, the projection
  then removes nothing, and BMCLoss reduces to per-date CORR on that date. That is
  34% of fold 0's rows with meta and 100% for folds 1-3.
* Scoring is E185's: rank-normalised predictions, CORR on the full cross-section,
  proxy MMC and payout (MMC + 0.1 x matched CORR) on rows with a meta score. The
  E185 sklearn arms (before, reference) share the same dates, so deltas are paired.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import floor
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.volatility_experiment import digest, groups, score_date, check_scoring  # noqa: E402
from src.live import features, registry  # noqa: E402
from src.live.submit import rank_normalise  # noqa: E402
from src.prediction.models import build_wrapper  # noqa: E402

OUT = ROOT / "results" / "e186_ia_torch_mmc"
TRAIN = Path(features.TRAIN_PARQUET)
META = ROOT / "data/crypto/v2.0/historical_meta_models.parquet"
MODELS = ("IA_01", "IA_02", "IA_03", "IA_04")
LOSSES = ("mmc", "mse")
TARGET = features.TARGET
PERIODS, GAP, VALID, FOLDS = 130, 22, 30, 4
END = pd.Timestamp("2026-10-14")


def model_params(name):
    if name == "IA_01":
        params = json.loads((ROOT / "models/live/IA_01.json").read_text())["best_params"]
    else:
        params = registry.get(name)["fit"]["fixed_params"].copy()
    params["mlp__hidden_layer_sizes"] = tuple(params["mlp__hidden_layer_sizes"])
    return params


def tv_epochs(n_features, hidden):
    width = [n_features, *hidden, 1]
    p = sum((a + 1) * b for a, b in zip(width, width[1:]))
    return p, max(floor(.2 * p ** .8), 1) + 10


# --------------------------------------------------------------------------- data

def design(frame, feature_set, cutoff):
    """E185's feature build, including its fit-available tenc prior correction."""
    history = frame.copy()
    history["is_live"] = history.date.gt(cutoff)
    built, cols, _ = features.build_features(history, feature_set, verbose=False)
    if feature_set == "tenc":
        prior_delta = history.loc[~history.is_live, TARGET].mean() - history[TARGET].mean()
        support = built["_tenc_support"]
        built["feature_sym_tenc"] += prior_delta * features.TENC_SMOOTHING / (support + features.TENC_SMOOTHING)
    assert built[cols].notna().all().all()
    return built, cols


def tasks():
    raw = pd.read_parquet(TRAIN).reset_index()
    assert not raw.duplicated(["date", "ucid"]).any() and raw[TARGET].notna().all()
    meta = pd.read_parquet(META)[["date", "id", "prediction"]].rename(
        columns={"id": "ucid", "prediction": "meta"})
    raw = raw.merge(meta, how="left", on=["date", "ucid"], validate="one_to_one")
    # Pin the panel to E185's last date so both studies score the same 520 dates
    # (train.parquet gained 2026-10-15 after E185 ran).
    raw = raw.loc[raw.date.le(END)]
    raw = raw.sort_values(["symbol", "date"], ignore_index=True)
    dates = pd.DatetimeIndex(raw.date.unique()).sort_values()
    sets = {name: registry.get(name)["feature_set"] for name in MODELS}
    out = []
    for fold in range(FOLDS):
        start = len(dates) - PERIODS * (FOLDS - fold)
        val_end = start - GAP
        val_start = val_end - VALID
        train_end = val_start - GAP
        blocks = (dates[train_end - PERIODS:train_end], dates[val_start:val_end],
                  dates[start:start + PERIODS])
        assert [len(b) for b in blocks] == [PERIODS, VALID, PERIODS]
        complete = raw.loc[raw.date.le(blocks[2][-1])].reset_index(drop=True)
        test = complete.date.isin(blocks[2])
        # Every block date-then-symbol ordered, so a date is one contiguous run.
        complete = pd.concat([complete.loc[~test].sort_values(["date", "symbol"]),
                              complete.loc[test].sort_values(["date", "symbol"])]).reset_index(drop=True)
        for feature_set in dict.fromkeys(sets.values()):
            built, cols = design(complete, feature_set, blocks[0][-1])
            arrays = {}
            for label, block in zip(("train", "val", "test"), blocks):
                part = built.loc[built.date.isin(block)]
                arrays[label] = dict(
                    x=part[cols].to_numpy("float64"), y=part[TARGET].to_numpy("float64"),
                    meta=part["meta"].to_numpy("float64"),
                    date=pd.factorize(part["date"])[0], dates=part["date"].to_numpy(),
                    span=(str(part.date.min().date()), str(part.date.max().date())))
            for name in MODELS:
                if sets[name] == feature_set:
                    out.append((name, fold, cols, arrays))
    return out


def train_reference(meta, date_codes):
    """Meta as a BMC reference. A date with no meta gets a constant (-> per-date CORR)."""
    ref = meta.copy()
    for ix in groups(date_codes):
        present = np.isfinite(ref[ix])
        if not present.any():
            ref[ix] = 0.5
        elif not present.all():
            ref[ix[~present]] = ref[ix[present]].mean()
    return ref


# --------------------------------------------------------------------------- fit

def fit_one(name, fold, cols, arrays, loss, seed, device, epochs_override=None, log_every=1):
    import torch
    from torch import nn
    from src.prediction.torch_mlp import TorchMLPRegressor
    from src.prediction.torch_mlp_losses import PerPeriodLoss
    from src.prediction.nn.losses import BMCLoss

    torch.set_num_threads(1)
    params = model_params(name)
    hidden = params["mlp__hidden_layer_sizes"]
    alpha = float(params["mlp__alpha"])
    lr = float(params["mlp__learning_rate_init"])
    batch_rows = int(params["mlp__batch_size"])
    p, epochs = tv_epochs(len(cols), hidden)
    if epochs_override:
        epochs = epochs_override

    tr, va, te = arrays["train"], arrays["val"], arrays["test"]
    scaler = build_wrapper("mlp", task="regression", seed=seed).ModelClass.named_steps["scale"]
    scaler.fit(pd.DataFrame(tr["x"], columns=cols))
    scale = lambda x: np.asarray(scaler.transform(pd.DataFrame(x, columns=cols)), dtype="float64")

    est = TorchMLPRegressor(hidden_layer_sizes=hidden, activation="relu", random_state=seed,
                            device=device)
    est._n_embed = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    net = est._build(len(cols), torch, nn, generator)
    assert sum(t.numel() for t in net.parameters()) == p
    weights = [m.weight for m in net if isinstance(m, nn.Linear)]
    optimiser = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    bmc = PerPeriodLoss(BMCLoss())

    X = torch.as_tensor(scale(tr["x"]), device=dev)
    Y = torch.as_tensor(tr["y"], device=dev).reshape(-1, 1)
    R = torch.as_tensor(train_reference(tr["meta"], tr["date"]), device=dev).reshape(-1, 1)
    P = torch.as_tensor(tr["date"], device=dev)
    Xv = torch.as_tensor(scale(va["x"]), device=dev)
    Xt = torch.as_tensor(scale(te["x"]), device=dev)

    date_rows = [torch.as_tensor(ix, device=dev) for ix in groups(tr["date"])]
    per_step = max(1, round(batch_rows / np.mean([len(ix) for ix in date_rows])))
    shuffler = np.random.RandomState(seed)
    val_groups = groups(va["date"])

    def predict(x):
        with torch.no_grad():
            return net(x).reshape(-1).cpu().numpy()

    def val_scores():
        pred = predict(Xv)
        rows = [score_date(rank_normalise(pred[ix])[:, None], va["y"][ix], va["meta"][ix])
                for ix in val_groups]
        corr = np.nanmean([r[0][0] for r in rows])
        mmc = np.nanmean([r[2][0] for r in rows])
        return corr, mmc, float(pred.std())

    curve = []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        order = shuffler.permutation(len(date_rows))
        totals = np.zeros(3)
        seen = 0
        for s in range(0, len(order), per_step):
            rows = torch.cat([date_rows[i] for i in order[s:s + per_step]])
            optimiser.zero_grad(set_to_none=True)
            out = net(X[rows])
            if loss == "mse":
                data = 0.5 * ((Y[rows] - out) ** 2).mean()
            else:
                data = bmc(out, Y[rows], reference=R[rows], period=P[rows])
            l2 = 0.5 * alpha * sum((w * w).sum() for w in weights) / len(rows)
            total = data + l2
            if not torch.isfinite(total):
                raise FloatingPointError(f"{name} fold {fold} {loss} seed {seed} diverged at epoch {epoch}")
            total.backward()
            optimiser.step()
            totals += np.array([float(total), float(data), float(l2)]) * len(rows)
            seen += len(rows)
        if epoch % log_every == 0 or epoch == epochs:
            vc, vm, vstd = val_scores()
            wnorm = float(torch.sqrt(sum((w * w).sum() for w in weights)))
            curve.append((name, fold, loss, seed, epoch, 6 * p * seen * epoch, *(totals / seen),
                          vc, vm, vstd, wnorm))
    seconds = perf_counter() - started

    pred = predict(Xt)
    assert np.isfinite(pred).all()
    arm = f"torch_{loss}"
    scores = []
    for ix in groups(te["date"]):
        corr, corr_m, mmc = score_date(rank_normalise(pred[ix])[:, None], te["y"][ix], te["meta"][ix])
        scores.append((name, fold, pd.Timestamp(te["dates"][ix[0]]).date().isoformat(), arm, seed,
                       corr[0], corr_m[0], mmc[0], .1 * corr_m[0] + mmc[0],
                       float(np.square(te["y"][ix] - pred[ix]).mean()), int(np.isfinite(te["meta"][ix]).sum()),
                       len(ix), int(np.unique(pred[ix]).size)))
    fit = dict(model=name, fold=fold, loss=loss, seed=seed, parameters=p, epochs=epochs,
               train_rows=len(tr["y"]), dates_per_step=per_step, seconds=seconds,
               train_meta_coverage=float(np.isfinite(tr["meta"]).mean()),
               train_start=tr["span"][0], train_end=tr["span"][1],
               test_start=te["span"][0], test_end=te["span"][1],
               final_val_corr=curve[-1][9], final_val_mmc=curve[-1][10],
               final_pred_std=curve[-1][11], final_weight_norm=curve[-1][12])
    return fit, curve, scores


def run_job(job):
    name, fold, cols, arrays, loss, seed, device = job
    return fit_one(name, fold, cols, arrays, loss, seed, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    check_scoring()
    work = tasks()
    if args.smoke:
        for name, fold, cols, arrays in work:
            if fold == 0:
                for loss in LOSSES:
                    t = perf_counter()
                    fit, curve, scores = fit_one(name, fold, cols, arrays, loss, 42, args.device,
                                                 epochs_override=3)
                    print(name, loss, f"{(perf_counter() - t) / 3:.2f}s/epoch",
                          "val_corr %.4f val_mmc %.4f" % (fit["final_val_corr"], fit["final_val_mmc"]),
                          "test payout %.4f" % np.mean([s[8] for s in scores]), flush=True)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(n, f, c, a, loss, seed, args.device) for n, f, c, a in work
            for loss in LOSSES for seed in args.seeds]
    # Longest first, so the big nets do not start last.
    jobs.sort(key=lambda j: -tv_epochs(len(j[2]), model_params(j[0])["mlp__hidden_layer_sizes"])[0])
    fits, curves, scores = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for done, fut in enumerate(as_completed(futures), 1):
            fit, curve, daily = fut.result()
            fits.append(fit), curves.extend(curve), scores.extend(daily)
            print(f"[{done}/{len(jobs)}]", {k: fit[k] for k in ("model", "fold", "loss", "seed", "epochs",
                                                                  "seconds", "final_val_mmc", "final_pred_std")},
                  flush=True)
    pd.DataFrame(fits).sort_values(["model", "loss", "fold", "seed"]).to_csv(OUT / "fits.csv", index=False)
    pd.DataFrame(curves, columns=["model", "fold", "loss", "seed", "epoch", "estimated_flops", "train_loss",
                                  "train_data_loss", "train_l2", "val_corr", "val_mmc", "val_pred_std",
                                  "weight_norm"]).to_csv(OUT / "curves.csv", index=False)
    pd.DataFrame(scores, columns=["model", "fold", "date", "arm", "seed", "corr", "corr_matched", "mmc",
                                  "payout", "mse", "meta_rows", "rows", "distinct_preds"]).sort_values(
        ["model", "arm", "seed", "date"]).to_csv(OUT / "daily_scores.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        study="E186", train_sha256=digest(TRAIN), meta_sha256=digest(META),
        models={n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_params(n).items()}
                for n in MODELS},
        losses=LOSSES, seeds=args.seeds, device=args.device, train_periods=PERIODS, gaps=GAP,
        validation_periods=VALID, folds=FOLDS, panel_end=str(END.date()), epochs="TV reference max(floor(0.2 P^0.8),1)+10"), indent=2) + "\n")


if __name__ == "__main__":
    main()
