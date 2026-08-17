"""
Differentiable training objectives for the LSTM pipeline.

These replace iwtsf's `SeqtoSeqSharpeLoss` but keep its shape and its convention:
compute a statistic *ex post* on the batch's own predictions, and return the
NEGATIVE of it so the trainer always minimises. Nothing here is a per-row loss --
there is no meaningful "error" on a single Numerai row, only a correlation over a
population of them.

    CorrLoss      -numerai-style correlation between prediction and target
    BMCLoss       -contribution of the prediction over a reference series
    MixtureLoss   a weighted sum of the two

------------------------------------------------------------------------------
Two things these are NOT
------------------------------------------------------------------------------

1. **Not the scored metric.** metrics.py scores with `numerai_tools.scoring`,
   which ranks, gaussianises and variance-normalises before correlating. Ranking
   is piecewise-constant, so it has no useful gradient and cannot be trained
   through. What follows are differentiable *surrogates*. The run's own reported
   CORR/BMC -- computed with the real functions -- are the check on whether the
   surrogate tracked the thing we actually care about.

2. **Not per-period.** The statistic is pooled over every row in the batch,
   ignoring date boundaries, which is the literal port of iwtsf's Sharpe loss.
   numerai_corr is computed per period and averaged, so a pooled Pearson mixes
   cross-sectional skill with time-series variation. This is a deliberate choice;
   grouping by period is a contained change to `_pooled_statistic` if the
   reported numbers ever suggest it matters.

------------------------------------------------------------------------------
Scale, and why the mixture weights are not a convex blend
------------------------------------------------------------------------------

`prediction/functions.py:build_scorer` blends two *scoring* metrics convexly:
`w * m1 + (1 - w) * m2` with `w` in [0, 1]. MixtureLoss deliberately does NOT
follow that convention -- its weights are independent multipliers that need not
sum to anything. The default is `bmc_weight=1.0, corr_weight=0.1`.

That matters because the components are not on a common scale. build_scorer's own
docstring records rough magnitudes from real runs: on Crypto, corr ~ 0.14 and
bmc ~ 0.05. So at 1.0/0.1 the BMC term contributes roughly 0.05 against corr's
0.014 -- BMC dominates the gradient about 3.6:1. That is the intent of those
defaults, but it is worth knowing rather than discovering.
"""

import torch
import torch.nn as nn

# Guards a division, nothing more. Deliberately smaller than iwtsf's 1e-6: it
# sits under a product of two norms rather than a standard deviation, so it needs
# to be negligible against values that can legitimately be small.
EPS = 1e-8

#: Fewer usable rows than this and a correlation is not defined.
MIN_ROWS = 2

#: How small an orthogonalised residual has to be, RELATIVE to the prediction it
#: came from, before BMC is treated as exactly zero.
#:
#: This has to be relative, not absolute. When the prediction is an affine
#: function of the reference the residual is zero in exact arithmetic, but float32
#: cancellation leaves ~1e-7 of the original scale behind -- and Pearson divides
#: by the residual's own norm, which amplifies that numerical dust straight back
#: up to an arbitrary correlation near +/-1. An absolute threshold small enough not
#: to swallow real signal is far too small to catch this, so the comparison is
#: made against the prediction's standard deviation instead.
RESIDUAL_REL_TOL = 1e-6


def _squeeze_trailing(tensor):
    """
    Drop a trailing singleton dimension, so (B, L, 1) and (B, L) both work.

    iwtsf's loss did the mirror image of this -- it *unsqueezed* 2-D inputs to
    3-D. Either convention is fine as long as it is applied to every argument;
    collapsing to 2-D is chosen here because the mask is naturally 2-D.

    :param tensor: any tensor, or None
    :type tensor: torch.Tensor or None
    :return: the tensor with a trailing dim of size 1 removed
    :rtype: torch.Tensor or None
    """
    if tensor is None:
        return None
    if tensor.dim() >= 2 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor


def _pearson(a, b):
    """
    Pearson correlation of two 1-D tensors, differentiable in both.

    :param a: first series
    :type a: torch.Tensor
    :param b: second series
    :type b: torch.Tensor
    :return: scalar correlation in [-1, 1]
    :rtype: torch.Tensor
    """
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + EPS)


def _zero_like(signal):
    """
    A zero that still carries a gradient path back to the network.

    Returning a plain `torch.tensor(0.0)` from a degenerate batch would detach the
    graph and make `.backward()` fail; returning NaN would silently poison every
    parameter on the next step and is very hard to trace back. Multiplying the
    signal by zero gives a loss of 0 with a valid (all-zero) gradient, so the
    optimiser simply takes no step for that batch.

    :param signal: the prediction tensor, used only for its graph and device
    :type signal: torch.Tensor
    :return: scalar zero with grad
    :rtype: torch.Tensor
    """
    return signal.sum() * 0.0


def _flatten(signal, target, reference=None, mask=None):
    """
    Collapse a batch to 1-D and drop rows that cannot be scored.

    Rows are dropped, not filled. Filling a missing target with the neutral bin
    (as `fill_missing_features` legitimately does for *features*) would invent
    target values, and inventing target values in a correlation drags it toward
    zero -- the model would be trained to predict a number nobody observed.
    `target_binned_return_60` has ~2.4% NaN, so this is a live concern, not a
    hypothetical.

    :param signal: predictions, (B, L) or (B, L, 1)
    :type signal: torch.Tensor
    :param target: targets, same shape as signal
    :type target: torch.Tensor
    :param reference: reference predictions for BMC, or None
    :type reference: torch.Tensor or None
    :param mask: True where the row has a usable target, or None for all rows
    :type mask: torch.Tensor or None
    :return: (signal, target, reference) flattened to 1-D and masked; reference is
             None if it was not supplied
    :rtype: (torch.Tensor, torch.Tensor, torch.Tensor or None)
    """
    signal = _squeeze_trailing(signal).reshape(-1)
    target = _squeeze_trailing(target).reshape(-1)
    reference = _squeeze_trailing(reference)
    if reference is not None:
        reference = reference.reshape(-1)

    # A NaN target that slipped through without a mask would propagate through
    # the mean and poison the whole batch, so treat non-finite targets as masked
    # regardless of what the caller passed.
    usable = torch.isfinite(target)
    if mask is not None:
        usable = usable & _squeeze_trailing(mask).reshape(-1).bool()
    if reference is not None:
        usable = usable & torch.isfinite(reference)

    if not bool(usable.all()):
        signal = signal[usable]
        target = target[usable]
        if reference is not None:
            reference = reference[usable]

    return signal, target, reference


def _degenerate(signal, target):
    """
    Whether a correlation over these rows would be undefined.

    Two ways it can be: too few rows, or a constant series on either side (a
    zero-variance prediction has no correlation to anything). Both happen in
    practice -- a constant prediction is exactly what an over-regularised model
    produces, and metrics.py has the same guard on the scoring side.

    :param signal: masked 1-D predictions
    :type signal: torch.Tensor
    :param target: masked 1-D targets
    :type target: torch.Tensor
    :return: True if the batch cannot be scored
    :rtype: bool
    """
    if signal.numel() < MIN_ROWS:
        return True
    return bool(signal.std() <= EPS or target.std() <= EPS)


class CorrLoss(nn.Module):
    """
    Negative Pearson correlation between prediction and target, pooled over a batch.

    The differentiable stand-in for `numerai_corr`. It skips the rank ->
    gaussianise -> pow-1.5 chain that Numerai applies, so it rewards *linear*
    agreement where the tournament rewards monotone agreement. In exchange it has
    a clean gradient everywhere.

    Note the loss is invariant to any positive affine rescaling of the prediction
    (`a * signal + b` for `a > 0`), which is why the network's output layer is
    free to sit anywhere on the real line and why no auxiliary calibration term is
    needed to make training work.
    """

    def forward(self, signal, target, reference=None, mask=None):
        """
        :param signal: predictions, (B, L) or (B, L, 1)
        :type signal: torch.Tensor
        :param target: targets, same shape
        :type target: torch.Tensor
        :param reference: ignored; accepted so every loss shares one signature
        :type reference: torch.Tensor or None
        :param mask: True where the row has a usable target
        :type mask: torch.Tensor or None
        :return: scalar loss in [-1, 1], lower is better
        :rtype: torch.Tensor
        """
        flat_signal, flat_target, _ = _flatten(signal, target, None, mask)
        if _degenerate(flat_signal, flat_target):
            return _zero_like(signal)
        return -_pearson(flat_signal, flat_target)


class BMCLoss(nn.Module):
    """
    Negative benchmark-model contribution, pooled over a batch.

    Numerai's `correlation_contribution` asks: once the reference series has been
    projected out of your prediction, does what remains still correlate with the
    target? Predicting the reference exactly scores zero no matter how good the
    reference is -- you contributed nothing.

    The differentiable form is that same projection, on raw values instead of
    gaussianised ranks:

        p     = signal - mean(signal)
        r     = reference - mean(reference)
        resid = p - r * (r . p) / (r . r)      # p orthogonalised against r
        bmc   = pearson(resid, target)

    Degenerate reference: if the reference is constant within the batch then
    `r . r` is ~0, the projection removes nothing, and this decays to CorrLoss.
    That is the right behaviour -- a constant reference contains no information to
    contribute over -- but it means a silently-broken reference column shows up as
    "BMC that behaves suspiciously like CORR" rather than as an error. The metrics
    file records reference coverage so that is checkable.
    """

    def forward(self, signal, target, reference=None, mask=None):
        """
        :param signal: predictions, (B, L) or (B, L, 1)
        :type signal: torch.Tensor
        :param target: targets, same shape
        :type target: torch.Tensor
        :param reference: reference predictions, same shape; required
        :type reference: torch.Tensor
        :param mask: True where the row has a usable target
        :type mask: torch.Tensor or None
        :return: scalar loss in [-1, 1], lower is better
        :rtype: torch.Tensor
        :raises ValueError: if no reference is supplied
        """
        if reference is None:
            raise ValueError(
                "BMCLoss needs a reference series to contribute over; pass "
                "reference=, or use CorrLoss instead"
            )

        flat_signal, flat_target, flat_reference = _flatten(
            signal, target, reference, mask
        )
        if _degenerate(flat_signal, flat_target):
            return _zero_like(signal)

        centred_signal = flat_signal - flat_signal.mean()
        centred_reference = flat_reference - flat_reference.mean()

        projection = (
            (centred_reference * centred_signal).sum()
            / ((centred_reference * centred_reference).sum() + EPS)
        )
        residual = centred_signal - centred_reference * projection

        # Orthogonalising can leave nothing behind -- exactly what happens when the
        # prediction is any affine function of the reference. That is a legitimate
        # BMC of zero, and it must be caught here: Pearson normalises by the
        # residual's norm, so left alone, float32 dust becomes a confident-looking
        # correlation. See RESIDUAL_REL_TOL.
        if bool(residual.std() <= RESIDUAL_REL_TOL * centred_signal.std() + EPS):
            return _zero_like(signal)

        return -_pearson(residual, flat_target)


class MixtureLoss(nn.Module):
    """
    Weighted sum of BMC and CORR: `-(bmc_weight * bmc + corr_weight * corr)`.

    The defaults (BMC at 1.0, CORR at 0.1) say "optimise contribution, but do not
    let the model wander somewhere with no predictive content at all" -- the CORR
    term is a regulariser on the BMC objective, not a co-equal goal.

    These are independent multipliers, NOT the convex `w / (1 - w)` blend that
    build_scorer uses for scoring metrics, and they are applied to components that
    live on different scales. See the module docstring.
    """

    def __init__(self, bmc_weight=1.0, corr_weight=0.1):
        """
        :param bmc_weight: multiplier on the BMC term
        :type bmc_weight: float
        :param corr_weight: multiplier on the CORR term
        :type corr_weight: float
        :raises ValueError: if both weights are zero, which would make the loss
                            constant and training a no-op
        """
        super().__init__()

        if bmc_weight == 0.0 and corr_weight == 0.0:
            raise ValueError(
                "bmc_weight and corr_weight are both zero, so the loss is "
                "constant and no training would happen"
            )

        self.bmc_weight = float(bmc_weight)
        self.corr_weight = float(corr_weight)
        self.bmc = BMCLoss()
        self.corr = CorrLoss()

    def forward(self, signal, target, reference=None, mask=None):
        """
        :param signal: predictions, (B, L) or (B, L, 1)
        :type signal: torch.Tensor
        :param target: targets, same shape
        :type target: torch.Tensor
        :param reference: reference predictions, same shape; required unless
                          bmc_weight is zero
        :type reference: torch.Tensor or None
        :param mask: True where the row has a usable target
        :type mask: torch.Tensor or None
        :return: scalar loss, lower is better
        :rtype: torch.Tensor
        """
        total = _zero_like(signal)

        # Each component is already negated, so the weights add directly.
        if self.bmc_weight != 0.0:
            total = total + self.bmc_weight * self.bmc(
                signal, target, reference, mask
            )
        if self.corr_weight != 0.0:
            total = total + self.corr_weight * self.corr(
                signal, target, None, mask
            )

        return total


#: Same plain-dict dispatch as prediction/models.py::MODELS.
LOSSES = {
    "corr": CorrLoss,
    "bmc": BMCLoss,
    "mixture": MixtureLoss,
}


def build_loss(name, bmc_weight=1.0, corr_weight=0.1):
    """
    Instantiate a loss by name.

    :param name: key into LOSSES
    :type name: str
    :param bmc_weight: multiplier on the BMC term, used by "mixture" only
    :type bmc_weight: float
    :param corr_weight: multiplier on the CORR term, used by "mixture" only
    :type corr_weight: float
    :return: the configured loss module
    :rtype: nn.Module
    :raises ValueError: if the name is unknown
    """
    if name not in LOSSES:
        raise ValueError(
            "unknown loss '{}', expected one of {}".format(name, sorted(LOSSES))
        )

    if name == "mixture":
        return MixtureLoss(bmc_weight=bmc_weight, corr_weight=corr_weight)
    return LOSSES[name]()


def needs_reference(name, bmc_weight=1.0):
    """
    Whether a loss cannot be computed without a reference series.

    Used by the CLI to fail before loading data rather than at the first batch.
    A mixture with `bmc_weight=0` is really just CorrLoss and needs nothing, which
    is why the weight is part of the question.

    :param name: key into LOSSES
    :type name: str
    :param bmc_weight: the mixture's BMC multiplier, ignored for other losses
    :type bmc_weight: float
    :return: True if a reference is required
    :rtype: bool
    """
    if name == "bmc":
        return True
    if name == "mixture":
        return bmc_weight != 0.0
    return False
