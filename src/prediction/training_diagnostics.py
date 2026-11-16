"""
Per-epoch diagnostics for a torch network's training -- the panel standard.

    from src.prediction.training_diagnostics import Objective, Snapshotter

    snap = Snapshotter(out_dir, Objective(alpha=100.0, denominator=1024),
                       train=train_rows, held=held_rows, probe=probe_batch)
    snap.snapshot(0, net)                                  # before the first step
    for epoch in range(1, n + 1):
        ...one epoch of training...
        snap.snapshot(epoch, net, lr=current_lr, running_loss=epoch_loss)

What a snapshot measures, all at FIXED weights on data that never changes:

    1  the optimised quantity -- data loss and L2 penalty, separately -- on the
       whole training window and on the held-out block
    2  the metric -- per-day numerai CORR on both, MMC on the held-out block
    3  the gradient of the loss on one fixed training batch, split into its data
       and penalty parts, in total and per layer
    4  the held-out predictions -- within-day spread and resolution, how far
       they moved since the last snapshot, and how they rank each target bin
    +  per-layer weight norms and, for a stack of Linear layers alone, the norm of
       the single linear map it composes to

A snapshot never steps the optimiser, never writes a parameter's .grad (it uses
torch.autograd.grad) and never draws a random number, so a run that takes them
trains bit for bit like one that does not -- src/tests/test_training_diagnostics.py
holds that. The network's train/eval mode is restored after every snapshot.

Everything lands in <out_dir>/log.json, rewritten after each snapshot so the
panel can be drawn while a run is still training (plot_training_diagnostics), and
the held-out predictions at a few milestone epochs in <out_dir>/preds.npz.
"""
import json
import math
import os

import numpy as np
import pandas as pd

# Epochs whose held-out predictions are kept whole; every other epoch keeps only
# the aggregates in log.json. A 39,000-row block at every epoch of a 200-epoch run
# would be 60 MB for nothing the panel draws.
MILESTONES = (0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200)


class Objective(object):
    """
    What the optimiser minimises, kept in its two pieces.

        loss = data + penalty
        data    = 0.5 * mean squared error  (or the day-balanced mean, see below)
                  or, kind="corr", minus the mean per-day Pearson correlation
                  or, kind="mmc", minus the mean per-day contribution over the meta model
        penalty = 0.5 * alpha * sum of squared WEIGHTS / denominator

    That is sklearn's MLPRegressor objective, which every torch MLP in this repo
    reproduces. The denominator is the rows in a training batch: 1024 for the
    sklearn convention, or a pinned value when batches vary in size (E104). With
    day_balanced=True each DAY counts equally in the data loss, which is what the
    day-batch recipes optimise; with equal stocks per day it is the plain mean.

    :param alpha: the L2 strength
    :type alpha: float
    :param denominator: what the penalty is divided by
    :type denominator: float
    :param day_balanced: weight days, not rows, equally in the data loss
    :type day_balanced: bool
    :param kind: "mse", or "corr" for -(mean over days of the within-day Pearson
                 correlation of prediction and target) -- the differentiable stand-in
                 for CORR. It ignores level and scale, so it needs each row's day.
                 "mmc" for -(mean over days of day_contribution) -- the stand-in for
                 MMC; it also needs each row's rank-gaussianised meta model.
    :type kind: str
    """

    def __init__(self, alpha, denominator, day_balanced=False, kind="mse"):
        if kind not in ("mse", "corr", "mmc"):
            raise ValueError("kind must be 'mse', 'corr' or 'mmc', not %r" % kind)
        self.alpha = float(alpha)
        self.denominator = float(denominator)
        self.day_balanced = bool(day_balanced)
        self.kind = kind

    def data(self, out, y, weights=None, days=None, meta=None):
        """
        0.5 * (weighted) mean squared error, `weights` summing to one when given; or,
        for kind="corr", minus the mean per-day Pearson correlation, `days` being each
        row's day as a 0..D-1 integer tensor (each day counts once, weights unused);
        or, for kind="mmc", minus the mean per-day contribution, `meta` being each
        row's meta model already rank-gaussianised within its day (meta_gaussian).
        """
        if self.kind == "mmc":
            if days is None or meta is None:
                raise ValueError("the mmc objective needs each row's day and meta model")
            return -day_contribution(out, y, days, meta).mean()
        if self.kind == "corr":
            if days is None:
                raise ValueError("the corr objective needs each row's day")
            return -day_pearson(out, y, days).mean()
        if weights is None:
            return 0.5 * ((y - out) ** 2).mean()
        return 0.5 * (weights.reshape(-1, 1) * (y - out) ** 2).sum()

    def penalty(self, layer_weights):
        return (0.5 * self.alpha) * sum((w * w).sum() for w in layer_weights) / self.denominator


def day_pearson(out, y, days):
    """
    Pearson correlation of `out` with `y` within each day: a (D,) tensor.

    `days` holds each row's day as 0..D-1. Differentiable throughout; a day whose
    predictions are all equal has correlation 0 and a finite (zero) gradient, which
    is why the variance product gets a tiny floor rather than a clamp.
    """
    import torch
    out, y = out.reshape(-1), y.reshape(-1)
    days = days.reshape(-1)
    n_days = int(days.max()) + 1
    zeros = lambda: torch.zeros(n_days, dtype=out.dtype, device=out.device)
    count = zeros().index_add_(0, days, torch.ones_like(out))
    oc = out - (zeros().index_add_(0, days, out) / count)[days]
    yc = y - (zeros().index_add_(0, days, y) / count)[days]
    cov = zeros().index_add_(0, days, oc * yc)
    var = zeros().index_add_(0, days, oc * oc) * zeros().index_add_(0, days, yc * yc)
    return cov * torch.rsqrt(var + 1e-30)


def day_contribution(out, y, days, meta_g):
    """
    Numerai's correlation_contribution within each day, made differentiable: a (D,) tensor.

    numerai_tools.scoring.correlation_contribution, per day, is
        p = gaussian(rank(pred)), m = gaussian(rank(meta)), n = p - (p.m / m.m) m,
        score = (4y - mean(4y)) . n / len(y)
    The ranking of `pred` has no gradient, so here p is the within-day z-score of the
    prediction (mean 0, std 1 -- the scale gaussian(rank) produces); `meta_g` is the
    meta model ranked and gaussianised exactly as Numerai does (meta_gaussian), and the
    target is bucketed and centred exactly as Numerai does. The only departure from the
    scorer is z-score in place of rank-gaussianise on the prediction.
    """
    import torch
    out, y, m = out.reshape(-1), y.reshape(-1), meta_g.reshape(-1)
    days = days.reshape(-1)
    n_days = int(days.max()) + 1
    zeros = lambda: torch.zeros(n_days, dtype=out.dtype, device=out.device)
    count = zeros().index_add_(0, days, torch.ones_like(out))
    oc = out - (zeros().index_add_(0, days, out) / count)[days]
    z = oc * torch.rsqrt((zeros().index_add_(0, days, oc * oc) / count)[days] + 1e-30)
    proj = zeros().index_add_(0, days, z * m) / zeros().index_add_(0, days, m * m)
    neutral = z - proj[days] * m
    y4 = 4.0 * y
    yc = y4 - (zeros().index_add_(0, days, y4) / count)[days]
    return zeros().index_add_(0, days, yc * neutral) / count


def meta_gaussian(meta, dates):
    """
    The meta model rank-gaussianised within each day, exactly as correlation_contribution
    does it (tie-kept rank, then the normal ppf). Numpy in, numpy out; NaN refuses.
    """
    from numerai_tools.scoring import gaussian, tie_kept_rank
    meta = np.asarray(meta, dtype="float64")
    if np.isnan(meta).any():
        raise ValueError("the meta model is missing on %d rows; the mmc objective needs it "
                         "on every row (diagnose --meta-only)" % int(np.isnan(meta).sum()))
    frame = pd.DataFrame({"m": meta, "d": np.asarray(dates)})
    out = np.empty(len(meta))
    for _, idx in frame.groupby("d").indices.items():
        day = frame.iloc[idx][["m"]].reset_index(drop=True)
        out[idx] = gaussian(tie_kept_rank(day))["m"].to_numpy()
    return out


def day_codes(dates):
    """Each row's day as 0..D-1 (numpy), the form day_pearson takes."""
    return pd.factorize(np.asarray(dates))[0]


def day_weights(dates):
    """Per-row weights giving each day 1/n_days, shared equally by its stocks."""
    codes, uniques = pd.factorize(np.asarray(dates))
    counts = np.bincount(codes)
    return 1.0 / (len(uniques) * counts[codes])


def layer_groups(net):
    """
    Parameter groups for the gradient and weight panels: one per Linear layer.

    Named "L1 44>64" and so on. Anything that is not a Linear layer (an
    embedding, a recurrent cell) gets a group per module that owns parameters, so
    a new architecture is covered without being taught here; give it meaningful
    groups (gates, blocks) when it needs them.

    :return: [(name, [parameters], weight tensor or None)] in forward order
    """
    from torch import nn
    groups, n_linear = [], 0
    for name, module in net.named_modules():
        own = list(module.parameters(recurse=False))
        if not own:
            continue
        if isinstance(module, nn.Linear):
            n_linear += 1
            groups.append(("L%d %d>%d" % (n_linear, module.in_features, module.out_features),
                           own, module.weight))
        else:
            groups.append((name or type(module).__name__, own, None))
    return groups


def net_map_norm(net):
    """
    The norm of the single linear map a pure Linear stack composes to, else None.

    With identity activations the whole network is x -> W x + b, and |W| is what
    decides whether predictions still differ within a day once they are rounded
    to float64 (E104: 1.7e-16 and the predictions held 44 distinct values a day).
    """
    from torch import nn
    import torch
    layers = [m for m in net.modules() if m is not net and not isinstance(m, nn.Sequential)]
    if not layers or not all(isinstance(m, nn.Linear) for m in layers):
        return None
    with torch.no_grad():
        total = None
        for layer in layers:
            total = layer.weight if total is None else layer.weight @ total
        return float(total.norm())


def _device(net):
    """Where the network's parameters live; every input a snapshot builds goes there."""
    return next(net.parameters()).device


class Snapshotter(object):
    """
    Takes the panel's measurements at epoch 0 and after every epoch.

    :param out_dir: where log.json and preds.npz go
    :type out_dir: str
    :param objective: the loss, in pieces
    :type objective: Objective
    :param train: the training window as the network sees it -- dict with X (scaled,
                  float64), y (float64, what the loss sees), date, and optionally
                  y_score (the target as the pipeline scores it), meta (meta-model
                  values) and symbol
    :type train: dict
    :param held: the held-out block in the same form; its predictions are scored and
                 tracked. Nothing may be SELECTED on it, or it is no longer held out.
    :type held: dict
    :param probe: the fixed training batch the gradient is measured on -- dict with
                  rows (indices into train) and, for a day-balanced objective, the
                  per-row weights of that batch
    :type probe: dict
    :param target_col: the target's column name in the scored frames
    :type target_col: str
    :param metric_every: score CORR/MMC every N snapshots (they are the slow part,
                         ~2 s); losses, gradients and predictions are every snapshot
    :type metric_every: int
    :param meta: anything to record verbatim in log.json (the recipe, the data)
    :type meta: dict or None
    :param pred_sink: optional callable(epoch, {"train": array, "held": array}) given
                      every snapshot's predictions, e.g. to keep them all on disk
    :type pred_sink: callable or None
    """

    def __init__(self, out_dir, objective, train, held, probe, target_col="target",
                 metric_every=1, meta=None, pred_sink=None):
        self.out_dir = out_dir
        self.objective = objective
        self.train, self.held, self.probe = train, held, probe
        self.target_col = target_col
        self.metric_every = max(1, int(metric_every))
        # Called as pred_sink(epoch, {"train": ..., "held": ...}) with every snapshot's
        # predictions -- how a run keeps them all (diagnose --save-preds).
        self.pred_sink = pred_sink
        os.makedirs(out_dir, exist_ok=True)
        self._frames = {name: self._frame(part) for name, part in (("train", train), ("held", held))}
        self._prev_rank = None
        self._preds = {}
        self._n = 0
        # The loss of predicting the training mean everywhere: 0.5 * var(y). A data
        # loss at this level has learned the intercept and nothing a ranking can use.
        # For the corr objective a constant prediction has no ranking at all: 0.
        y = np.asarray(train["y"], dtype="float64")
        kind = getattr(objective, "kind", "mse")
        self.corr = kind in ("corr", "mmc")             # a per-day objective: needs each row's day
        self._days = ({name: day_codes(part["date"]) for name, part in (("train", train), ("held", held))}
                      if self.corr else None)
        self._meta_g = ({name: meta_gaussian(part["meta"], part["date"])
                         for name, part in (("train", train), ("held", held))}
                        if kind == "mmc" else None)
        self.log = {"meta": dict(meta or {}), "mean_only_loss": {
            "train": 0.0 if self.corr else float(0.5 * np.mean((y - y.mean()) ** 2)),
            "held": 0.0 if self.corr else float(0.5 * np.mean((np.asarray(held["y"]) - y.mean()) ** 2))},
            "snapshots": []}

    def _frame(self, part):
        # Scored against y_score when given -- the target in the dtype the pipeline
        # scores it in (float32 from the parquet) -- so CORR here equals CORR in
        # results/per_period to the last digit, not to 1e-9.
        target = part["y_score"] if part.get("y_score") is not None else part["y"]
        frame = pd.DataFrame({"date": np.asarray(part["date"]),
                              self.target_col: np.asarray(target)})
        if part.get("meta") is not None:
            frame["meta_model"] = np.asarray(part["meta"], dtype="float64")
        return frame

    # ------------------------------------------------------------------- #

    def snapshot(self, epoch, net, lr=None, running_loss=None, extra=None):
        """
        Measure, append to the log and rewrite log.json.

        :param epoch: 0 before training, then the number of epochs completed
        :param net: the network
        :param lr: the learning rate the NEXT epoch will start with, if known
        :param running_loss: the training loop's own epoch loss (train mode, weights
                             moving under it) -- drawn thin, never compared
        :param extra: anything else to keep for this epoch
        :return: the record
        :rtype: dict
        """
        import torch
        was_training = net.training
        net.eval()
        try:
            record = {"epoch": int(epoch), "lr": None if lr is None else float(lr),
                      "running_loss": None if running_loss is None else float(running_loss)}
            groups = layer_groups(net)
            weights = [w for _, _, w in groups if w is not None]
            preds, device = {}, _device(net)
            with torch.no_grad():
                for name, part in (("train", self.train), ("held", self.held)):
                    X = torch.as_tensor(np.asarray(part["X"], dtype="float64"), device=device)
                    y = torch.as_tensor(np.asarray(part["y"], dtype="float64"),
                                        device=device).reshape(-1, 1)
                    out = net(X)
                    w = (torch.as_tensor(day_weights(part["date"]), device=device)
                         if self.objective.day_balanced and not self.corr else None)
                    d = (torch.as_tensor(self._days[name], device=device) if self.corr else None)
                    mg = (torch.as_tensor(self._meta_g[name], device=device)
                          if self._meta_g is not None else None)
                    data = float(self.objective.data(out, y, w, days=d, meta=mg))
                    preds[name] = out.cpu().numpy().ravel().copy()
                    record[name] = {"data_loss": data}
                penalty = float(self.objective.penalty(weights)) if weights else 0.0
            for name in ("train", "held"):
                record[name]["penalty"] = penalty
                record[name]["loss"] = record[name]["data_loss"] + penalty
            record["gradients"] = self._gradients(net, groups)
            record["weights"] = {name: float(w.detach().norm()) for name, _, w in groups
                                 if w is not None}
            record["net_map_norm"] = net_map_norm(net)
            record["predictions"] = self._dynamics(preds["held"])
            if self.pred_sink is not None:
                self.pred_sink(int(epoch), preds)
            if self._n % self.metric_every == 0:
                record["metric"] = self._metric(preds)
            if extra:
                record["extra"] = dict(extra)
            if epoch in MILESTONES:
                self._preds["epoch_%03d" % epoch] = preds["held"].astype("float64")
                np.savez_compressed(os.path.join(self.out_dir, "preds.npz"), **self._preds)
        finally:
            net.train(was_training)
        self._n += 1
        self.log["snapshots"].append(record)
        self._write()
        return record

    def finish(self, net=None, extra=None):
        """Keep the last epoch's held-out predictions too, and anything else to say."""
        record = self.log["snapshots"][-1] if self.log["snapshots"] else None
        if net is not None and record is not None:
            import torch
            preds = {}
            with torch.no_grad():
                was = net.training; net.eval()
                for name, part in (("train", self.train), ("held", self.held)):
                    out = net(torch.as_tensor(np.asarray(part["X"], dtype="float64"),
                                              device=_device(net)))
                    preds[name] = out.cpu().numpy().ravel().astype("float64")
                net.train(was)
            key = "epoch_%03d" % record["epoch"]
            if key not in self._preds:
                self._preds[key] = preds["held"]
                np.savez_compressed(os.path.join(self.out_dir, "preds.npz"), **self._preds)
            if "metric" not in record:          # the last epoch is always scored
                record["metric"] = self._metric(preds)
        if extra:
            self.log.setdefault("summary", {}).update(extra)
        self._write()

    # ------------------------------------------------------------------- #

    def _gradients(self, net, groups):
        """
        The loss gradient on the probe batch, data and penalty parts separately.

        torch.autograd.grad, not .backward(): nothing is written into a parameter's
        .grad, so the optimiser's next step is untouched.
        """
        import torch
        rows = np.asarray(self.probe["rows"])
        device = _device(net)
        X = torch.as_tensor(np.asarray(self.train["X"], dtype="float64")[rows], device=device)
        y = torch.as_tensor(np.asarray(self.train["y"], dtype="float64")[rows],
                            device=device).reshape(-1, 1)
        w = self.probe.get("weights")
        w = None if w is None else torch.as_tensor(np.asarray(w, dtype="float64"), device=device)
        params = [p for _, own, _ in groups for p in own]
        weights = [lw for _, _, lw in groups if lw is not None]
        d = (torch.as_tensor(day_codes(np.asarray(self.train["date"])[rows]), device=device)
             if self.corr else None)
        mg = (torch.as_tensor(self._meta_g["train"][rows], device=device)
              if self._meta_g is not None else None)
        data = self.objective.data(net(X), y, w, days=d, meta=mg)
        g_data = torch.autograd.grad(data, params, allow_unused=True)
        if weights:
            g_pen = torch.autograd.grad(self.objective.penalty(weights), params, allow_unused=True)
        else:
            g_pen = [None] * len(params)
        zero = lambda g, p: torch.zeros_like(p) if g is None else g
        g_data = [zero(g, p) for g, p in zip(g_data, params)]
        g_pen = [zero(g, p) for g, p in zip(g_pen, params)]
        norm = lambda gs: float(math.sqrt(sum(float((g * g).sum()) for g in gs)))
        result = {"data": norm(g_data), "penalty": norm(g_pen),
                  "total": norm([a + b for a, b in zip(g_data, g_pen)]),
                  "probe_rows": int(len(rows)), "groups": {}}
        at = 0
        for name, own, _ in groups:
            d, p = g_data[at:at + len(own)], g_pen[at:at + len(own)]
            result["groups"][name] = {"data": norm(d), "penalty": norm(p),
                                      "total": norm([a + b for a, b in zip(d, p)])}
            at += len(own)
        return result

    def _dynamics(self, pred):
        """Spread, resolution, movement and the per-target-bin ranking of held-out predictions."""
        frame = self._frames["held"].assign(prediction=pred)
        by_day = frame.groupby("date")["prediction"]
        rank = by_day.rank(pct=True)
        frame = frame.assign(rank=rank)
        mean = float(np.mean(pred))
        stats = {
            "spread": float(by_day.std(ddof=0).median()),
            "distinct": float(by_day.nunique().median()),
            "stocks": float(by_day.size().median()),
            "mean": mean,
            # The gap between adjacent float64 values at the predictions' level. A
            # within-day spread near this is a ranking made of rounding error.
            "float_spacing": float(np.spacing(abs(mean)) if mean else np.finfo("float64").tiny),
            "bin_profile": {str(k): float(v) for k, v in
                            frame.groupby(self.target_col)["rank"].mean().items()},
            "movement": None,
        }
        if self._prev_rank is not None:
            # 1 - the within-day correlation of this snapshot's ranks with the last
            # one's: 0 = no stock changed place, 1 = a fresh shuffle.
            pair = pd.DataFrame({"date": frame["date"], "a": rank.to_numpy(),
                                 "b": self._prev_rank})
            corr = pair.groupby("date")[["a", "b"]].corr().xs("a", level=1)["b"]
            stats["movement"] = float(1.0 - corr.mean()) if corr.notna().any() else None
        self._prev_rank = rank.to_numpy()
        return stats

    def _metric(self, preds):
        """Per-day numerai CORR on both sets; MMC on held-out (and on train where covered)."""
        from src.prediction.metrics import per_period_corr, per_period_bmc
        out = {}
        for name in ("train", "held"):
            frame = self._frames[name].assign(prediction=preds[name])
            # A day whose predictions are all equal earned 0, it is not missing.
            corr = per_period_corr(frame, "prediction", self.target_col, "date").fillna(0.0)
            out["%s_corr" % name] = float(corr.mean())
            out["%s_corr_positive_days" % name] = float((corr > 0).mean())
            if "meta_model" in frame and frame["meta_model"].notna().mean() > 0.5:
                mmc = per_period_bmc(frame, "prediction", "meta_model", self.target_col,
                                     "date").fillna(0.0)
                out["%s_mmc" % name] = float(mmc.mean())
        if "held_mmc" in out:
            out["held_payout"] = 0.1 * out["held_corr"] + out["held_mmc"]
        return out

    def _write(self):
        path = os.path.join(self.out_dir, "log.json")
        with open(path + ".tmp", "w") as handle:
            json.dump(self.log, handle, indent=1, default=float)
        os.replace(path + ".tmp", path)
