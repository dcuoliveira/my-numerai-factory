"""
The LSTM forecasting network.

A close port of `LSTMPortfolioNetwork` from github.com/dcuoliveira/iwtsf, with one
substantive change: the final `Tanh` is replaced by a multiclass head over the
target's own bins.

    features (B, L, F) + symbol_ids (B, L)
      -> nn.Embedding(num_symbols + 1, embed_dim)     symbol identity
      -> concat                                       (B, L, F + embed_dim)
      -> dropout                                      on the LSTM's INPUT
      -> nn.LSTM(hidden_dim, num_layers, batch_first) 1 layer, unidirectional
      -> nn.LayerNorm(hidden_dim)                     optional
      -> dropout                                      on the LSTM's OUTPUT
      -> nn.Linear(hidden_dim, n_classes)             replaces Linear(.., 1) + Tanh
      -> softmax, then expectation against the bin values

This is a **sequence-to-sequence** model: it emits a prediction for every timestep
in the window, not just the last one. That is what makes non-overlapping windows
efficient -- every row in the panel is predicted exactly once.

------------------------------------------------------------------------------
Why a multiclass head, and how it becomes a single number
------------------------------------------------------------------------------

The crypto target is not continuous. It is already binned to five values,
{0, 0.25, 0.5, 0.75, 1}, with half its mass in the middle bin. `Tanh` on [-1, 1]
was the right output for iwtsf, where the network emitted a portfolio *weight*;
here the natural object is a distribution over the bins the target actually takes.

CORR and BMC need one continuous number per row, so the distribution is reduced by
its expectation:

    signal = sum_k softmax(logits)_k * bin_value_k

which lands inside [0, 1] -- the target's own domain, which Tanh's [-1, 1] was
not. Three properties make this the right reduction:

  * Differentiable everywhere, unlike argmax or a hard bin choice.
  * Monotone in where the distribution's mass sits, so ordering behaves sensibly.
  * It keeps the *confidence* information a hard label would throw away: a row
    split 50/50 between bins 0 and 1 scores 0.5, distinct from a row confidently
    in bin 0.5, and the tournament ranks on exactly that kind of gradation.

`bin_values` is passed in, derived from the training block's target, rather than
hard-coded -- so nothing breaks if a target's binning ever changes.

------------------------------------------------------------------------------
Dropout, kept exactly as iwtsf had it
------------------------------------------------------------------------------

`nn.LSTM` is constructed with `dropout=0.0` and dropout is applied by hand to its
input and output instead. Two reasons this is deliberate rather than an oversight:

  * `nn.LSTM(dropout=...)` only applies between *stacked layers*, so at
    num_layers=1 -- the default here -- it does nothing at all.
  * `dropout_type='variational'` samples one mask per sample and reuses it across
    every timestep, which is the right thing for a recurrent net: a fresh mask per
    timestep injects noise that the recurrence has to fight, whereas a shared mask
    perturbs a *feature* consistently along the sequence. This is the Gal &
    Ghahramani idea, applied to the input and output paths only -- the
    recurrent-to-recurrent weights are untouched, so it is not the full method.

iwtsf's actual runs used `dropout_type='regular'`, so that stays the default here.
"""

import torch
import torch.nn as nn


class LSTMForecastNetwork(nn.Module):
    """
    Symbol-embedded LSTM with a multiclass head over the target's bins.

    :param input_dim: number of features per timestep
    :type input_dim: int
    :param bin_values: the target's distinct values, ascending; one class each
    :type bin_values: [float]
    :param hidden_dim: LSTM hidden size
    :type hidden_dim: int
    :param num_layers: stacked LSTM layers
    :type num_layers: int
    :param num_symbols: how many distinct symbols the embedding must cover; one
                        extra row is added for unseen/padding symbols
    :type num_symbols: int or None
    :param embed_dim: symbol embedding width, or 0 to disable embeddings entirely
    :type embed_dim: int
    :param dropout: dropout probability, applied to the LSTM input and output
    :type dropout: float
    :param dropout_type: "regular" (fresh mask per timestep) or "variational"
                         (one mask per sample, shared across timesteps)
    :type dropout_type: str
    :param use_layer_norm: whether to normalise the LSTM output
    :type use_layer_norm: bool
    """

    def __init__(self, input_dim, bin_values, hidden_dim=60, num_layers=1,
                 num_symbols=None, embed_dim=50, dropout=0.3,
                 dropout_type="regular", use_layer_norm=True):
        super().__init__()

        if dropout_type not in ("regular", "variational"):
            raise ValueError(
                "dropout_type must be 'regular' or 'variational', got {!r}".format(
                    dropout_type
                )
            )
        if not len(bin_values):
            raise ValueError("bin_values is empty; the head would have no classes")
        if embed_dim and not num_symbols:
            raise ValueError(
                "embed_dim={} needs num_symbols; pass embed_dim=0 to train without "
                "symbol embeddings".format(embed_dim)
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_classes = len(bin_values)
        self.dropout_rate = float(dropout)
        self.dropout_type = dropout_type
        self.use_layer_norm = bool(use_layer_norm)
        self.use_symbol_embedding = bool(embed_dim)

        # Kept as a buffer, not a plain attribute: it has to follow the module
        # across .to(device) and be restored with the weights, since a checkpoint
        # loaded against different bin values would silently rescale every
        # prediction.
        self.register_buffer(
            "bin_values", torch.tensor(list(bin_values), dtype=torch.float32)
        )

        if self.use_symbol_embedding:
            # One extra row, indexed num_symbols, absorbs symbols that were not in
            # the training block. padding_idx zero-initialises it and holds its
            # gradient at zero, so an unseen symbol contributes no identity signal
            # rather than a random one.
            self.symbol_embedding = nn.Embedding(
                int(num_symbols) + 1, int(embed_dim), padding_idx=int(num_symbols)
            )
            lstm_input_dim = self.input_dim + int(embed_dim)
        else:
            self.symbol_embedding = None
            lstm_input_dim = self.input_dim

        self.regular_dropout = nn.Dropout(self.dropout_rate)

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=int(num_layers),
            bidirectional=False,
            batch_first=True,
            dropout=0.0,          # see the module docstring; not an oversight
        )

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.output_proj = nn.Linear(self.hidden_dim, self.n_classes)

    # ----------------------------------------------------------------------- #
    # Dropout
    # ----------------------------------------------------------------------- #

    def _variational_mask(self, x):
        """
        One dropout mask per sample and feature, broadcast across all timesteps.

        Shape (B, 1, D) so it broadcasts along the sequence axis -- that shared
        axis is the whole point. Inverted scaling keeps the expectation unchanged
        so nothing needs rescaling at eval time.

        :param x: the tensor about to be masked, (B, L, D)
        :type x: torch.Tensor
        :return: multiplicative mask broadcastable to x
        :rtype: torch.Tensor
        """
        if not self.training or self.dropout_rate == 0:
            return torch.ones(
                x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device
            )

        keep = 1.0 - self.dropout_rate
        mask = torch.bernoulli(
            torch.full(
                (x.shape[0], 1, x.shape[2]), keep, dtype=x.dtype, device=x.device
            )
        )
        return mask / keep

    def _apply_dropout(self, x):
        """
        Apply the configured dropout flavour.

        :param x: tensor to perturb, (B, L, D)
        :type x: torch.Tensor
        :return: the perturbed tensor
        :rtype: torch.Tensor
        """
        if self.dropout_rate <= 0:
            return x
        if self.dropout_type == "variational":
            return x * self._variational_mask(x)
        return self.regular_dropout(x)

    # ----------------------------------------------------------------------- #
    # Forward
    # ----------------------------------------------------------------------- #

    def forward_logits(self, features, symbol_ids=None):
        """
        Class logits for every timestep.

        Exposed separately from forward() so an auxiliary cross-entropy term can be
        added without recomputing anything. No such term is used by default: CORR is
        invariant to positive affine rescaling of the signal, so the head only needs
        to get the *ordering* of expectations right, and a cross-entropy term would
        additionally demand calibrated per-bin probabilities -- pulling against the
        objective that is actually scored.

        :param features: (B, L, input_dim)
        :type features: torch.Tensor
        :param symbol_ids: (B, L) integer symbol ids, required iff embeddings are on
        :type symbol_ids: torch.Tensor or None
        :return: (B, L, n_classes) logits
        :rtype: torch.Tensor
        """
        if self.use_symbol_embedding:
            if symbol_ids is None:
                raise ValueError(
                    "this network was built with symbol embeddings, so forward() "
                    "needs symbol_ids"
                )
            embedded = self.symbol_embedding(symbol_ids)
            x = torch.cat([features, embedded], dim=-1)
        else:
            x = features

        x = self._apply_dropout(x)

        # The first return of nn.LSTM is the per-timestep output, not the final
        # hidden state -- that is what makes this seq-to-seq. (iwtsf named this
        # variable `h_n`, which is the name torch uses for the final state; the
        # code was right and the name was misleading.)
        outputs, _ = self.lstm(x)

        if self.use_layer_norm:
            outputs = self.layer_norm(outputs)

        outputs = self._apply_dropout(outputs)

        return self.output_proj(outputs)

    def forward(self, features, symbol_ids=None):
        """
        The continuous signal the losses consume, one value per timestep.

        :param features: (B, L, input_dim)
        :type features: torch.Tensor
        :param symbol_ids: (B, L) integer symbol ids, required iff embeddings are on
        :type symbol_ids: torch.Tensor or None
        :return: (B, L) signal, inside the closed range spanned by bin_values
        :rtype: torch.Tensor
        """
        logits = self.forward_logits(features, symbol_ids)
        probabilities = torch.softmax(logits, dim=-1)

        # Expectation over the bins. Because the weights are a probability
        # simplex, the result is a convex combination of bin_values and therefore
        # cannot leave [min(bin_values), max(bin_values)].
        return (probabilities * self.bin_values).sum(dim=-1)


def infer_bin_values(target_values):
    """
    Read the head's classes off the training target.

    Derived from data rather than hard-coded to [0, .25, .5, .75, 1] so the same
    code survives a change in the target's binning -- and so a target that is *not*
    binned fails loudly here instead of silently training a 500-class head.

    :param target_values: the training block's target column
    :type target_values: np.ndarray or pd.Series
    :return: distinct target values, ascending
    :rtype: [float]
    :raises ValueError: if the target has no values, or too many distinct ones to
                        be a binned target
    """
    import numpy as np

    values = np.asarray(target_values, dtype="float64")
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("the training target has no finite values")

    distinct = np.unique(values)

    # Numerai's binned targets have 5 levels; Classic's integer bins have 5 too.
    # Anything much past that is a continuous target, which this head cannot
    # represent -- better to say so than to build a head with hundreds of classes.
    if len(distinct) > 32:
        raise ValueError(
            "the target has {} distinct values, which does not look like a binned "
            "target; the multiclass head needs one class per bin".format(
                len(distinct)
            )
        )

    return [float(v) for v in distinct]
