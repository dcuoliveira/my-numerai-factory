"""
Draw the training-diagnostics panel from a run's log.json.

    python -m src.analysis.plot_training_diagnostics                 # every run
    python -m src.analysis.plot_training_diagnostics e105_const_s42_r0

Reads results/diag/crypto/<tag>/log.json (written by
src.prediction.training_diagnostics.Snapshotter, e.g. via src.prediction.diagnose)
and writes diagnostics.png beside it. Works on a run that is still training.

    row 1   1a loss, the optimised quantity   1b fit beyond the mean   2 the metric
    row 2   3a gradient: data vs penalty      3b gradient per layer    5 lr and weights
    row 3   4a ranking of each target bin     4b within-day spread     4c resolution, movement

Every panel has one y-axis. Time panels share epochs on x and mark the epoch at
which sklearn's stop rule fires (the run ends there unless it was --no-stop).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join("results", "diag", "crypto")
# dataviz reference palette: categorical slots in order, the blue sequential ramp, chrome
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _series(snaps, *path):
    out = []
    for s in snaps:
        v = s
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
            if v is None:
                break
        out.append(np.nan if v is None else float(v))
    return np.asarray(out, dtype="float64")


def _frame(ax, title, subtitle, log=False):
    ax.set_facecolor("none")
    ax.grid(axis="y", color=GRID, linewidth=0.7); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(which="both", length=0, labelsize=8.5)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=17)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=8, color=INK2)


def _line(ax, x, y, color, label, width=1.8, alpha=1.0):
    ok = np.isfinite(y)
    if ax.get_yscale() == "log":
        ok &= y > 0                    # e.g. the penalty part of an alpha=0 run: nothing to draw
    if not ok.any():
        return
    ax.plot(x[ok], y[ok], color=color, linewidth=width, alpha=alpha, label=label,
            solid_capstyle="round", solid_joinstyle="round")


def _legend(ax, loc="best"):
    # A legend, not end labels: these lines converge (train onto held-out, every
    # metric onto 0 when a model collapses), and stacked end labels then collide.
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=loc, fontsize=7.5, frameon=False, labelcolor=INK2, handlelength=1.6,
                  borderaxespad=0.2)


def _zoom(ax, x, ys, start=5, pad=0.15):
    """
    Fit the y-axis to epochs >= start and say what was left off it.

    Epoch 0 is a random network whose level is far from the target's, so its R^2
    against the mean is in the thousands of percent and would flatten the curve
    that matters into a line.
    """
    keep = x >= min(start, x.max())
    vals = np.concatenate([y[keep][np.isfinite(y[keep])] for y in ys])
    if not len(vals):
        return
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or max(abs(hi), 1e-12)
    ax.set_ylim(lo - pad * span, hi + pad * span)
    early = np.concatenate([y[~keep] for y in ys])
    early = early[np.isfinite(early)]
    if len(early) and (early.min() < lo - pad * span or early.max() > hi + pad * span):
        ax.text(0.01, 0.02, "epochs < %d off scale (down to %.0f)" % (start, early.min()),
                transform=ax.transAxes, fontsize=7.5, color=MUTED)


def _stop(ax, epoch):
    if epoch:
        ax.axvline(epoch, color=AXIS, linewidth=0.9, zorder=1)


def plot_run(tag, root=ROOT):
    """Draw <root>/<tag>/diagnostics.png and return its path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = os.path.join(root, tag)
    log = json.load(open(os.path.join(run_dir, "log.json")))
    snaps, meta = log["snapshots"], log.get("meta", {})
    summary = log.get("summary", {})
    stop = summary.get("stop_rule_epoch")
    ep = _series(snaps, "epoch")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": AXIS,
                         "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(16.5, 13.2), dpi=130, facecolor=SURFACE)
    outer = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.27, left=0.05, right=0.955,
                             top=0.885, bottom=0.075)

    corr_loss = meta.get("loss") in ("corr", "mmc")   # a per-day, scale-free objective
    mmc_loss = meta.get("loss") == "mmc"
    # 1a -- the optimised quantity, log scale: it spans orders of magnitude at alpha 100.
    # A -corr loss is negative and bounded, so it gets a linear axis.
    ax = fig.add_subplot(outer[0, 0])
    _frame(ax, "1a · Loss — what the optimiser minimises",
           ("−(mean per-day contribution over the meta model) + L2 penalty, eval mode" if mmc_loss else
            "−(mean per-day Pearson) + L2 penalty, eval mode, fixed weights" if corr_loss else
            "data loss + L2 penalty, eval mode, fixed weights · log"), not corr_loss)
    _line(ax, ep, _series(snaps, "train", "loss"), SLOTS[0], "train")
    _line(ax, ep, _series(snaps, "held", "loss"), SLOTS[1], "held-out")
    if not corr_loss or float(meta.get("alpha") or 0) > 0:
        _line(ax, ep, _series(snaps, "train", "penalty"), SLOTS[2], "penalty part")
    _line(ax, ep, _series(snaps, "running_loss"), MUTED, "loop's running loss", width=0.9)
    ax.axhline(log["mean_only_loss"]["train"], color=INK2, linewidth=0.8,
               label="constant prediction" if corr_loss else "predict the mean")
    _legend(ax, "upper right"); _stop(ax, stop)

    if corr_loss:
        # 1b -- the surrogate the loss maximises against the rank-based CORR that pays.
        # Level, intercept and pooling are gone from this loss; what is left between
        # it and the metric is Pearson-on-values versus Numerai's ranked, tail-weighted CORR.
        ax = fig.add_subplot(outer[0, 1])
        what, paid = ("z-scored contribution", "mmc") if mmc_loss else ("Pearson", "corr")
        _frame(ax, "1b · Optimised %s vs paid %s" % (what.split()[-1], paid.upper()),
               "per-day %s (the loss, negated) and Numerai %s, same rows" % (what, paid.upper()))
        for name, color, label in (("train", SLOTS[0], "train"), ("held", SLOTS[1], "held-out")):
            _line(ax, ep, -_series(snaps, name, "data_loss"), color, "%s %s" % (label, what.split()[-1]))
            _line(ax, ep, _series(snaps, "metric", "%s_%s" % (name, paid)), color,
                  "%s %s" % (label, paid.upper()), width=1.0, alpha=0.6)
        ax.axhline(0, color=AXIS, linewidth=0.9)
        _legend(ax); _stop(ax, stop)
    else:
        # 1b -- how much of the data loss the network explains beyond the training mean
        ax = fig.add_subplot(outer[0, 1]); _frame(ax, "1b · Fit beyond predicting the mean",
                                                  "1 − data loss ÷ mean-only loss, % (R² against the training mean)")
        r2 = []
        for name, color, label in (("train", SLOTS[0], "train"), ("held", SLOTS[1], "held-out")):
            r2.append(100.0 * (1.0 - _series(snaps, name, "data_loss") / log["mean_only_loss"][name]))
            _line(ax, ep, r2[-1], color, label)
        ax.axhline(0, color=AXIS, linewidth=0.9)
        _zoom(ax, ep, r2); _legend(ax); _stop(ax, stop)

    # 2 -- the metric we are paid on
    ax = fig.add_subplot(outer[0, 2]); _frame(ax, "2 · Metric — what we are paid on",
                                              "mean per-day CORR; MMC against the real meta model")
    _line(ax, ep, _series(snaps, "metric", "train_corr"), SLOTS[0], "train CORR")
    _line(ax, ep, _series(snaps, "metric", "held_corr"), SLOTS[1], "held-out CORR")
    _line(ax, ep, _series(snaps, "metric", "held_mmc"), SLOTS[2], "held-out MMC")
    ax.axhline(0, color=AXIS, linewidth=0.9)
    _legend(ax); _stop(ax, stop)

    # 3a -- the gradient on one fixed training batch, data part vs penalty part
    ax = fig.add_subplot(outer[1, 0]); _frame(ax, "3a · Gradient on a fixed training batch",
                                              "L2 norm, total and its two parts · log", True)
    _line(ax, ep, _series(snaps, "gradients", "total"), INK, "total")
    _line(ax, ep, _series(snaps, "gradients", "data"), SLOTS[0], "data part")
    _line(ax, ep, _series(snaps, "gradients", "penalty"), SLOTS[1], "penalty part")
    clip = meta.get("max_grad_norm")
    if clip:
        ax.axhline(clip, color=INK2, linewidth=0.8)
    _legend(ax); _stop(ax, stop)

    # 3b -- per layer
    ax = fig.add_subplot(outer[1, 1]); _frame(ax, "3b · Gradient by layer",
                                              "L2 norm of each layer's total gradient · log", True)
    names = list((snaps[-1].get("gradients") or {}).get("groups", {}))
    for i, name in enumerate(names):
        _line(ax, ep, _series(snaps, "gradients", "groups", name, "total"),
              SLOTS[i % len(SLOTS)], name)
    _legend(ax, "lower left"); _stop(ax, stop)

    # 5 -- what the optimiser did to the weights, with the learning rate above it
    cell = outer[1, 2].subgridspec(2, 1, height_ratios=[1, 2.4], hspace=0.12)
    ax_lr = fig.add_subplot(cell[0]); _frame(ax_lr, "5 · Learning rate and weights",
                                             "learning rate · log", True)
    _line(ax_lr, ep, _series(snaps, "lr"), INK2, "lr")
    ax_lr.tick_params(labelbottom=False); _stop(ax_lr, stop)
    ax = fig.add_subplot(cell[1], sharex=ax_lr)
    ax.set_facecolor("none"); ax.set_yscale("log"); ax.grid(axis="y", color=GRID, linewidth=0.7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(which="both", length=0, labelsize=8.5)
    ax.text(0, 1.02, "Frobenius norm per layer; the net linear map when there is one · log",
            transform=ax.transAxes, fontsize=8, color=INK2)
    for i, name in enumerate(list(snaps[-1].get("weights", {}))):
        _line(ax, ep, _series(snaps, "weights", name), SLOTS[i % len(SLOTS)], name)
    _line(ax, ep, _series(snaps, "net_map_norm"), INK, "net map |W|")
    _legend(ax, "lower left"); _stop(ax, stop)

    # 4a -- the scatter's replacement for a five-level target: how each bin is ranked
    ax = fig.add_subplot(outer[2, 0]); _frame(ax, "4a · How held-out predictions rank each target bin",
                                              "mean within-day percentile of the prediction, by target")
    # Initialisation, early, middle, the epoch the stop rule picks (the model as it
    # is actually used) and the last one trained.
    wanted = {0, 2, 10, 30, stop if stop is not None else -1, int(ep[-1])}
    marks = [i for i, s in enumerate(snaps) if s["epoch"] in wanted]
    for j, i in enumerate(marks):
        prof = snaps[i]["predictions"]["bin_profile"]
        xs = np.array([float(k) for k in prof]); ys = np.array([prof[k] for k in prof])
        color = RAMP[min(len(RAMP) - 1, int(round(j * (len(RAMP) - 1) / max(1, len(marks) - 1))))]
        ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label="epoch %d%s" % (snaps[i]["epoch"], " (stop)" if snaps[i]["epoch"] == stop else ""))
    ax.axhline(0.5, color=AXIS, linewidth=0.9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", ncol=3, fontsize=7.5, frameon=False, labelcolor=INK2,
                  handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.35 * (hi - lo))   # room for the legend
    ax.set_xlabel("target", fontsize=8.5, color=INK2)

    # 4b -- within-day spread against float64 spacing at the predictions' level
    ax = fig.add_subplot(outer[2, 1]); _frame(ax, "4b · Spread of held-out predictions within a day",
                                              "median within-day std, and the float64 spacing at that level · log", True)
    _line(ax, ep, _series(snaps, "predictions", "spread"), SLOTS[0], "spread")
    _line(ax, ep, _series(snaps, "predictions", "float_spacing"), MUTED, "float64 spacing", width=1.0)
    _legend(ax); _stop(ax, stop)

    # 4c -- resolution and movement, both 0..1
    ax = fig.add_subplot(outer[2, 2]); _frame(ax, "4c · Resolution and movement",
                                              "share of a day's stocks with distinct predictions; rank change since last epoch")
    stocks = _series(snaps, "predictions", "stocks")
    _line(ax, ep, _series(snaps, "predictions", "distinct") / stocks, SLOTS[0], "distinct values")
    _line(ax, ep, _series(snaps, "predictions", "movement"), SLOTS[1], "movement")
    ax.set_ylim(-0.03, 1.05)
    _legend(ax); _stop(ax, stop)

    for a in fig.axes:                          # 4a already says "target"; lr shares x below it
        if a.get_xlabel() == "" and a is not ax_lr:
            a.set_xlabel("epoch", fontsize=8.5, color=INK2)

    recipe = "%s %s, %s loss, alpha %g, lr %g %s, %s batches%s, L2/%s, %s init, seed %s" % (
        meta.get("activation", "?"), meta.get("hidden", "?"),
        {"corr": "−Pearson", "mmc": "−MMC"}.get(meta.get("loss"), "MSE"), meta.get("alpha", float("nan")),
        meta.get("lr", float("nan")), meta.get("schedule", ""), meta.get("batch", "?"),
        " of %s days" % meta.get("days_per_batch") if meta.get("days_per_batch") else "",
        meta.get("l2", "?"), meta.get("init", "?"), meta.get("seed", "?"))
    fig.suptitle("Training diagnostics · %s" % tag, x=0.05, y=0.975, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.05, 0.948, recipe, fontsize=10, color=INK2)
    fig.text(0.05, 0.928, "%s · refit %s: train %s→%s (%s days), held-out %s→%s (%s days) · "
             "stop rule fires at epoch %s%s" % (
                 meta.get("data_version", "?"), meta.get("refit", "?"), meta.get("train_start", "?"),
                 meta.get("train_end", "?"), meta.get("train_days", "?"), meta.get("held_start", "?"),
                 meta.get("held_end", "?"), meta.get("held_days", "?"), stop,
                 " (trained on to %s)" % summary.get("epochs_trained") if meta.get("no_stop") else ""),
             fontsize=9, color=INK2)
    fig.text(0.05, 0.018, "Fixed data throughout: the whole training window, the whole held-out block, "
             "one fixed training batch for the gradient. Grey vertical line = where the stop rule "
             "fires. Read 1a with 2 first, then the rest — see .claude/skills/training-diagnostics.",
             fontsize=8.5, color=INK2)
    out = os.path.join(run_dir, "diagnostics.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    tags = (argv if argv is not None else sys.argv[1:]) or sorted(
        d for d in os.listdir(ROOT) if os.path.exists(os.path.join(ROOT, d, "log.json")))
    for tag in tags:
        print("wrote %s" % plot_run(tag))


if __name__ == "__main__":
    main()
