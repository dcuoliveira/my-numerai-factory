"""
Differentiable training losses for the tabular torch MLP (E42 / E43).

The plain torch MLP (`torch_mlp.py`) trains on a fixed half-MSE. This module is
what lets `torch_mlp_loss` train on the objective the tournament actually pays on
-- correlation and MMC/BMC contribution -- and on risk-adjusted ratios.

Two design facts drive the shapes here:

  * `nn/losses.py` already has TESTED differentiable `CorrLoss` and `BMCLoss`
    (Pearson surrogates with the BMC residualisation and its degeneracy guards),
    but they POOL over the whole batch. `numerai_corr` scores PER PERIOD then
    averages, so `PerPeriodLoss` wraps those modules and applies them per period
    code, averaging -- faithful to "per-period then average", and reusing the
    guarded math rather than re-deriving it.

  * sharpe and calmar are ratios over a DISTRIBUTION of per-period scores; they
    are meaningless inside a 1024-row minibatch that spans ~2 dates. So they are
    FULL-WINDOW losses (`full_batch=True`): the estimator forwards every training
    row once per epoch, this builds the whole per-period correlation series, and
    the ratio is taken over it. The caller then takes one optimiser step/epoch.
    calmar's max-drawdown is not smoothly differentiable, so a soft surrogate
    (log-cumsum-exp running max, log-sum-exp shortfall) stands in for it.

All losses are Pearson (linear) surrogates, NOT the rank-scored metric -- the
same caveat `nn/losses.py` documents. Period codes are expected to be an integer
`pd.factorize(date)` in chronological order (calmar's cumulative relies on it).

The loss callable's signature is uniform: `fn(signal, target, reference, period)`
-> scalar tensor with a gradient path (lower is better).
"""

# One row per loss: does it need a single full-window step, the period column,
# and/or the reference series. The estimator reads this to plumb the right
# passthrough columns and choose the training loop.
_LOSS_META = {
    "mse":    {"full_batch": False, "needs_period": False, "needs_reference": False},
    "corr":   {"full_batch": False, "needs_period": True,  "needs_reference": False},
    "bmc":    {"full_batch": False, "needs_period": True,  "needs_reference": True},
    "sharpe": {"full_batch": True,  "needs_period": True,  "needs_reference": False},
    "calmar": {"full_batch": True,  "needs_period": True,  "needs_reference": False},
}

LOSSES = tuple(_LOSS_META)


def loss_meta(name):
    """Plumbing requirements for a loss name (see _LOSS_META)."""
    if name not in _LOSS_META:
        raise ValueError(
            "unknown loss %r; expected one of %s" % (name, sorted(_LOSS_META)))
    return dict(_LOSS_META[name])


class PerPeriodLoss:
    """
    Apply a pooled nn/losses module per period code and average.

    `inner` is a `CorrLoss` or `BMCLoss` instance -- their degeneracy guards
    (constant/too-few-rows -> grad-carrying zero) are reused unchanged, so a
    thin period contributes 0 rather than a NaN. With no period column it falls
    back to the pooled inner loss (the LSTM's behaviour), which is a cheap
    ablation.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, signal, target, reference=None, period=None):
        import torch

        if period is None:
            return self.inner(signal, target, reference=reference)
        codes = period.reshape(-1).long()
        vals = []
        for code in torch.unique(codes):
            mask = codes == code
            ref = reference[mask] if reference is not None else None
            vals.append(self.inner(signal[mask], target[mask], reference=ref))
        if not vals:
            return signal.sum() * 0.0
        return torch.stack(vals).mean()


def _period_corr_series(signal, target, period):
    """
    The per-period Pearson correlation series, in ascending period-code order.

    Degenerate periods (constant or <2 rows on either side) are skipped, exactly
    as `metrics.per_period_corr` drops them. Returns None if fewer than two
    scorable periods remain (a ratio needs a distribution).
    """
    import torch
    from src.prediction.nn.losses import _pearson, EPS

    signal = signal.reshape(-1)
    target = target.reshape(-1)
    codes = period.reshape(-1).long()
    corrs = []
    for code in torch.sort(torch.unique(codes)).values:
        mask = codes == code
        s, t = signal[mask], target[mask]
        if s.numel() < 2 or s.std() <= EPS or t.std() <= EPS:
            continue
        corrs.append(_pearson(s, t))
    if len(corrs) < 2:
        return None
    return torch.stack(corrs)


def _sharpe_loss(signal, target, reference=None, period=None, eps=1e-6):
    """-(mean / std) of the per-period corr series. Full-window."""
    series = _period_corr_series(signal, target, period)
    if series is None:
        return signal.sum() * 0.0
    return -(series.mean() / (series.std(unbiased=False) + eps))


def _soft_max_drawdown(cumulative, beta_max=50.0, beta_dd=50.0):
    """
    Smooth surrogate for the max peak-to-trough fall of a cumulative series.

    running_max_t via a soft-max (log-cumsum-exp), shortfall dd_t = runmax - cum,
    then a soft-max over dd via log-sum-exp. As beta -> inf this -> the true
    max-drawdown `metrics.summarize` computes; finite beta keeps a gradient.
    """
    import torch

    runmax = torch.logcumsumexp(beta_max * cumulative, dim=0) / beta_max
    shortfall = runmax - cumulative
    return torch.logsumexp(beta_dd * shortfall, dim=0) / beta_dd


def _calmar_loss(signal, target, reference=None, period=None,
                 eps=1e-6, beta_max=50.0, beta_dd=50.0):
    """-(mean per-period corr / soft max-drawdown of its cumulative). Full-window."""
    import torch

    series = _period_corr_series(signal, target, period)
    if series is None:
        return signal.sum() * 0.0
    drawdown = _soft_max_drawdown(torch.cumsum(series, dim=0), beta_max, beta_dd)
    return -(series.mean() / (drawdown + eps))


def build_torch_mlp_loss(name, **kwargs):
    """
    Build a loss callable and its plumbing metadata.

    :param name: one of LOSSES
    :param kwargs: loss options (bmc guards for bmc; eps/beta for sharpe/calmar)
    :return: (loss_fn or None for mse, meta dict). mse returns None because the
             estimator keeps its byte-identical inline half-MSE for the control.
    """
    meta = loss_meta(name)
    if name == "mse":
        return None, meta

    from src.prediction.nn.losses import CorrLoss, BMCLoss

    if name == "corr":
        return PerPeriodLoss(CorrLoss()), meta
    if name == "bmc":
        return PerPeriodLoss(BMCLoss()), meta
    if name == "sharpe":
        eps = float(kwargs.get("eps", 1e-6))
        return (lambda s, t, reference=None, period=None:
                _sharpe_loss(s, t, period=period, eps=eps)), meta
    if name == "calmar":
        eps = float(kwargs.get("eps", 1e-6))
        bmax = float(kwargs.get("beta_max", 50.0))
        bdd = float(kwargs.get("beta_dd", 50.0))
        return (lambda s, t, reference=None, period=None:
                _calmar_loss(s, t, period=period, eps=eps,
                             beta_max=bmax, beta_dd=bdd)), meta
    raise ValueError("unhandled loss %r" % (name,))   # unreachable; loss_meta guards
