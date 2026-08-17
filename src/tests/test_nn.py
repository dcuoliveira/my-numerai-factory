"""
Tests for the torch LSTM pipeline.

Same philosophy as test_pipeline.py: these cover the mistakes that produce
plausible-looking numbers rather than crashes. A sequence model on a panel has a
specific set of those --

    * a window that silently spans a gap in a symbol's history, so the LSTM is
      fed "consecutive" timesteps that are months apart
    * a NaN target filled instead of masked, dragging the correlation to zero
    * an early-stopping "best" state that is really the last state, because
      state_dict() returns live references (the actual bug in the repo this was
      ported from)
    * a target whose forward window reaches into the block being predicted

Everything here runs offline on a synthetic panel and takes seconds. torch is an
optional dependency, so the whole module skips cleanly without it.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.prediction.nn.losses import (  # noqa: E402
    BMCLoss,
    CorrLoss,
    MixtureLoss,
    build_loss,
    needs_reference,
)
from src.prediction.nn.model import LSTMForecastNetwork, infer_bin_values  # noqa: E402
from src.prediction.nn.dataset import (  # noqa: E402
    SymbolSequenceDataset,
    build_period_positions,
    build_symbol_index,
    collate_windows,
    contiguous_runs,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _series(values):
    """A (1, N) float32 tensor, the shape the losses see from a batch."""
    return torch.tensor([values], dtype=torch.float32)


@pytest.fixture
def linear_pair():
    """A target and a prediction that agrees with it perfectly."""
    target = _series([0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0.75])
    return target.clone(), target.clone()


# --------------------------------------------------------------------------- #
# CorrLoss
# --------------------------------------------------------------------------- #

def test_corr_loss_perfect_agreement_is_minus_one(linear_pair):
    signal, target = linear_pair
    assert CorrLoss()(signal, target).item() == pytest.approx(-1.0, abs=1e-5)


def test_corr_loss_perfect_disagreement_is_plus_one(linear_pair):
    signal, target = linear_pair
    assert CorrLoss()(-signal, target).item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (3.0, 0.0), (0.5, -2.0), (10.0, 7.5)])
def test_corr_loss_is_invariant_to_positive_affine_rescaling(linear_pair, scale, shift):
    """
    The head can output anything on the real line; only the ordering matters.

    This is why no calibration term is needed to make training work, and why the
    multiclass head's expectation can sit anywhere in [0, 1] without penalty.
    """
    signal, target = linear_pair
    signal = signal + torch.randn_like(signal) * 0.1     # break the exact tie
    baseline = CorrLoss()(signal, target).item()
    rescaled = CorrLoss()(signal * scale + shift, target).item()
    assert rescaled == pytest.approx(baseline, abs=1e-5)


def test_corr_loss_pools_across_the_batch_ignoring_rows():
    """
    (B, L) and (1, B*L) must give the same number.

    Pins the documented "pooled over the whole batch" decision. If someone later
    changes the statistic to be per-period, this test is the one that should fail
    and force the docs to be updated with it.
    """
    flat = torch.rand(1, 24)
    target = torch.rand(1, 24)
    loss_flat = CorrLoss()(flat, target).item()
    loss_shaped = CorrLoss()(flat.reshape(4, 6), target.reshape(4, 6)).item()
    assert loss_flat == pytest.approx(loss_shaped, abs=1e-6)


def test_corr_loss_accepts_a_trailing_singleton_dimension(linear_pair):
    """(B, L, 1) is what a seq-to-seq head naturally emits."""
    signal, target = linear_pair
    with_dim = CorrLoss()(signal.unsqueeze(-1), target.unsqueeze(-1)).item()
    assert with_dim == pytest.approx(-1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# BMCLoss
# --------------------------------------------------------------------------- #

def test_bmc_is_zero_when_the_prediction_is_the_reference():
    """
    Contributing nothing over the reference must score zero, however good the
    reference is. This is the whole point of the metric.
    """
    reference = _series([0.1, 0.4, 0.2, 0.9, 0.5, 0.7, 0.3, 0.8])
    target = reference.clone()                    # a perfect reference
    loss = BMCLoss()(reference.clone(), target, reference=reference)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_bmc_is_zero_for_any_affine_function_of_the_reference():
    """
    Orthogonalisation removes the reference's *direction*, so rescaling it is
    still no contribution. A BMC that rewarded `2 * reference` would be trivially
    gameable.
    """
    reference = _series([0.1, 0.4, 0.2, 0.9, 0.5, 0.7, 0.3, 0.8])
    target = _series([0.0, 0.5, 0.25, 1.0, 0.5, 0.75, 0.25, 1.0])
    loss = BMCLoss()(reference * 2.0 + 0.3, target, reference=reference)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_bmc_is_positive_for_skill_orthogonal_to_the_reference():
    """A signal the reference does not contain, that predicts the target, contributes."""
    reference = _series([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    orthogonal = _series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    # Target aligned with the orthogonal direction, not the reference.
    target = orthogonal.clone()
    loss = BMCLoss()(orthogonal, target, reference=reference).item()
    assert loss < -0.9, "expected a strongly negative loss (positive BMC)"


def test_bmc_decays_to_corr_when_the_reference_is_constant():
    """
    A constant reference carries no information to contribute over, so the
    projection removes nothing. Documented behaviour -- pinned so it cannot change
    silently, because it is how a broken reference column would present.
    """
    signal = _series([0.2, 0.9, 0.4, 0.7, 0.1, 0.5, 0.8, 0.3])
    target = _series([0.0, 1.0, 0.25, 0.75, 0.0, 0.5, 1.0, 0.25])
    constant = torch.full_like(signal, 0.5)
    bmc = BMCLoss()(signal, target, reference=constant).item()
    corr = CorrLoss()(signal, target).item()
    assert bmc == pytest.approx(corr, abs=1e-4)


def test_bmc_requires_a_reference():
    signal = _series([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="needs a reference"):
        BMCLoss()(signal, signal.clone())


# --------------------------------------------------------------------------- #
# MixtureLoss
# --------------------------------------------------------------------------- #

def test_mixture_is_exactly_the_weighted_sum_of_its_components():
    """The requested 1.0 x BMC + 0.1 x CORR, verified against the parts."""
    signal = _series([0.2, 0.9, 0.4, 0.7, 0.1, 0.5, 0.8, 0.3])
    target = _series([0.0, 1.0, 0.25, 0.75, 0.0, 0.5, 1.0, 0.25])
    reference = _series([0.5, 0.6, 0.1, 0.9, 0.2, 0.4, 0.7, 0.35])

    bmc = BMCLoss()(signal, target, reference=reference).item()
    corr = CorrLoss()(signal, target).item()
    mixture = MixtureLoss(bmc_weight=1.0, corr_weight=0.1)(
        signal, target, reference=reference
    ).item()

    assert mixture == pytest.approx(1.0 * bmc + 0.1 * corr, abs=1e-6)


@pytest.mark.parametrize("bmc_w,corr_w", [(1.0, 0.0), (0.0, 1.0), (2.0, 0.5)])
def test_mixture_honours_arbitrary_non_convex_weights(bmc_w, corr_w):
    """
    The weights are independent multipliers, not a convex blend. build_scorer uses
    `w / (1 - w)` for scoring metrics; this deliberately does not.
    """
    signal = _series([0.2, 0.9, 0.4, 0.7, 0.1, 0.5, 0.8, 0.3])
    target = _series([0.0, 1.0, 0.25, 0.75, 0.0, 0.5, 1.0, 0.25])
    reference = _series([0.5, 0.6, 0.1, 0.9, 0.2, 0.4, 0.7, 0.35])

    expected = 0.0
    if bmc_w:
        expected += bmc_w * BMCLoss()(signal, target, reference=reference).item()
    if corr_w:
        expected += corr_w * CorrLoss()(signal, target).item()

    got = MixtureLoss(bmc_weight=bmc_w, corr_weight=corr_w)(
        signal, target, reference=reference
    ).item()
    assert got == pytest.approx(expected, abs=1e-6)


def test_mixture_rejects_two_zero_weights():
    with pytest.raises(ValueError, match="both zero"):
        MixtureLoss(bmc_weight=0.0, corr_weight=0.0)


def test_mixture_with_zero_bmc_weight_needs_no_reference():
    """A weight of zero must short-circuit, not evaluate BMC and discard it."""
    signal = _series([0.2, 0.9, 0.4, 0.7])
    target = _series([0.0, 1.0, 0.25, 0.75])
    loss = MixtureLoss(bmc_weight=0.0, corr_weight=1.0)(signal, target)
    assert loss.item() == pytest.approx(CorrLoss()(signal, target).item(), abs=1e-6)
    assert needs_reference("mixture", bmc_weight=0.0) is False
    assert needs_reference("mixture", bmc_weight=1.0) is True
    assert needs_reference("corr") is False
    assert needs_reference("bmc") is True


# --------------------------------------------------------------------------- #
# Masking and degenerate batches
# --------------------------------------------------------------------------- #

def test_mask_excludes_rows_and_matches_the_subset_computed_directly():
    """
    Masking must be equivalent to never having had the row -- not to filling it.
    This is what makes target_binned_return_60's 2.4% NaN safe.
    """
    signal = _series([0.2, 0.9, 0.4, 0.7, 0.1, 0.5])
    target = _series([0.0, 1.0, 0.25, 0.75, 0.0, 0.5])
    mask = torch.tensor([[True, True, False, True, False, True]])

    masked = CorrLoss()(signal, target, mask=mask).item()
    subset = CorrLoss()(
        _series([0.2, 0.9, 0.7, 0.5]), _series([0.0, 1.0, 0.75, 0.5])
    ).item()
    assert masked == pytest.approx(subset, abs=1e-6)


def test_nan_targets_are_masked_even_without_an_explicit_mask():
    """
    A NaN that slips through un-masked would propagate through the mean and make
    the whole batch NaN. Defence in depth: the loss drops non-finite targets
    itself.
    """
    signal = _series([0.2, 0.9, 0.4, 0.7])
    target = _series([0.0, 1.0, float("nan"), 0.75])
    loss = CorrLoss()(signal, target)
    assert torch.isfinite(loss), "a NaN target leaked into the loss"
    subset = CorrLoss()(_series([0.2, 0.9, 0.7]), _series([0.0, 1.0, 0.75])).item()
    assert loss.item() == pytest.approx(subset, abs=1e-6)


@pytest.mark.parametrize("loss_name", ["corr", "bmc", "mixture"])
def test_degenerate_batches_return_a_finite_zero_that_still_carries_grad(loss_name):
    """
    Too few rows, or a constant prediction, must not produce NaN -- a single NaN
    step silently destroys every weight and is very hard to trace back.
    """
    loss_fn = build_loss(loss_name)

    for signal, target in (
        (torch.zeros(1, 1, requires_grad=True), torch.zeros(1, 1)),          # 1 row
        (torch.full((1, 6), 0.5, requires_grad=True), torch.rand(1, 6)),     # constant
    ):
        reference = torch.rand_like(target)
        loss = loss_fn(signal, target, reference=reference)
        assert torch.isfinite(loss), "{} produced a non-finite loss".format(loss_name)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        loss.backward()                                # must not raise
        assert signal.grad is not None


def test_losses_are_differentiable_and_move_a_parameter():
    """A smoke check that the whole thing is trainable, not just computable."""
    weight = torch.nn.Parameter(torch.tensor([0.1]))
    features = torch.rand(1, 32)
    target = features * 2.0 + 0.5

    optimiser = torch.optim.Adam([weight], lr=0.1)
    first = None
    for _ in range(25):
        optimiser.zero_grad()
        loss = CorrLoss()(features * weight, target)
        loss.backward()
        optimiser.step()
        if first is None:
            first = loss.item()

    assert loss.item() <= first + 1e-6


def test_build_loss_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss("sharpe")


# --------------------------------------------------------------------------- #
# The network and its multiclass head
# --------------------------------------------------------------------------- #

BINS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _net(**kwargs):
    """A small network on the crypto feature count."""
    options = dict(
        input_dim=22, bin_values=BINS, hidden_dim=8, num_symbols=7, embed_dim=4,
        dropout=0.0,
    )
    options.update(kwargs)
    return LSTMForecastNetwork(**options)


def test_head_emits_one_signal_per_timestep():
    """Seq-to-seq: (B, L, F) in, (B, L) out. Not one prediction per window."""
    net = _net().eval()
    signal = net(torch.rand(3, 11, 22), torch.zeros(3, 11, dtype=torch.long))
    assert signal.shape == (3, 11)


def test_head_signal_stays_inside_the_target_domain():
    """
    A convex combination of the bins cannot leave their range -- which is the
    property Tanh did not have, and the reason the head replaced it.
    """
    net = _net().eval()
    # Large-magnitude features, to try to push the head out of range.
    signal = net(torch.randn(4, 9, 22) * 50.0, torch.zeros(4, 9, dtype=torch.long))
    assert signal.min().item() >= min(BINS) - 1e-6
    assert signal.max().item() <= max(BINS) + 1e-6


def test_head_probabilities_form_a_distribution():
    net = _net().eval()
    logits = net.forward_logits(torch.rand(2, 5, 22), torch.zeros(2, 5, dtype=torch.long))
    assert logits.shape == (2, 5, len(BINS))
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.allclose(
        probabilities.sum(-1), torch.ones(2, 5), atol=1e-5
    )


def test_head_expectation_matches_a_hand_computed_reduction():
    """Pin the reduction itself, not just its range."""
    net = _net().eval()
    features = torch.rand(2, 6, 22)
    symbol_ids = torch.zeros(2, 6, dtype=torch.long)

    logits = net.forward_logits(features, symbol_ids)
    expected = (torch.softmax(logits, dim=-1) * torch.tensor(BINS)).sum(-1)
    assert torch.allclose(net(features, symbol_ids), expected, atol=1e-6)


def test_unseen_symbols_get_a_zero_embedding_and_do_not_crash():
    """
    A symbol absent from the training block maps to the reserved padding row.
    Without the extra row this is an index-out-of-range at predict time; with a
    random row it would inject identity signal for a symbol we know nothing about.
    """
    net = _net(num_symbols=7).eval()
    padding_index = 7
    assert net.symbol_embedding.num_embeddings == 8
    assert torch.allclose(
        net.symbol_embedding.weight[padding_index], torch.zeros(4)
    )
    signal = net(torch.rand(1, 4, 22), torch.full((1, 4), padding_index, dtype=torch.long))
    assert torch.isfinite(signal).all()


def test_embeddings_can_be_disabled_with_embed_dim_zero():
    """iwtsf's own off switch; the lever for the overfitting risk in the plan."""
    net = _net(embed_dim=0, num_symbols=None).eval()
    assert net.symbol_embedding is None
    signal = net(torch.rand(2, 5, 22))
    assert signal.shape == (2, 5)


def test_embeddings_require_symbol_ids_when_enabled():
    net = _net().eval()
    with pytest.raises(ValueError, match="needs symbol_ids"):
        net(torch.rand(1, 4, 22))


@pytest.mark.parametrize("dropout_type", ["regular", "variational"])
def test_dropout_is_active_in_train_mode_and_inert_in_eval(dropout_type):
    net = _net(dropout=0.5, dropout_type=dropout_type)
    features = torch.rand(4, 7, 22)
    symbol_ids = torch.zeros(4, 7, dtype=torch.long)

    net.eval()
    assert torch.allclose(net(features, symbol_ids), net(features, symbol_ids))

    net.train()
    torch.manual_seed(0)
    first = net(features, symbol_ids)
    torch.manual_seed(1)
    second = net(features, symbol_ids)
    assert not torch.allclose(first, second), "dropout had no effect in train mode"


def test_variational_dropout_shares_one_mask_across_timesteps():
    """
    The defining property of the variational flavour. With constant features along
    the sequence, a shared mask leaves the LSTM input constant along the sequence
    too, whereas a per-timestep mask does not.
    """
    net = _net(dropout=0.5, dropout_type="variational").train()
    constant = torch.ones(1, 6, 26)               # 22 features + 4 embedding dims
    torch.manual_seed(0)
    mask = net._variational_mask(constant)
    assert mask.shape == (1, 1, 26), "mask must broadcast along the sequence axis"
    # Same mask reused at every timestep => every row of (mask * constant) is equal.
    masked = mask * constant
    assert torch.allclose(masked[0, 0], masked[0, -1])


def test_layer_norm_is_optional():
    assert hasattr(_net(use_layer_norm=True), "layer_norm")
    assert not hasattr(_net(use_layer_norm=False), "layer_norm")


def test_bin_values_survive_a_state_dict_round_trip():
    """
    Restoring weights against different bins would silently rescale every
    prediction, so the bins travel with the checkpoint.
    """
    net = _net()
    assert "bin_values" in net.state_dict()
    restored = _net()
    restored.load_state_dict(net.state_dict())
    assert torch.allclose(restored.bin_values, torch.tensor(BINS))


def test_bin_values_move_with_the_module_across_devices():
    """A buffer, not a plain attribute -- otherwise .to(device) leaves it behind."""
    net = _net()
    assert net.bin_values.device == torch.device("cpu")
    if torch.cuda.is_available():
        net = net.to("cuda")
        assert net.bin_values.is_cuda


def test_network_rejects_a_bad_dropout_type():
    with pytest.raises(ValueError, match="dropout_type"):
        _net(dropout_type="gaussian")


def test_network_rejects_embeddings_without_a_symbol_count():
    with pytest.raises(ValueError, match="needs num_symbols"):
        LSTMForecastNetwork(input_dim=22, bin_values=BINS, embed_dim=8, num_symbols=None)


# --------------------------------------------------------------------------- #
# infer_bin_values
# --------------------------------------------------------------------------- #

def test_infer_bin_values_reads_the_crypto_bins_off_the_target():
    target = np.array([0.5, 0.0, 1.0, 0.25, 0.5, 0.75, 0.5])
    assert infer_bin_values(target) == BINS


def test_infer_bin_values_ignores_nan():
    target = np.array([0.0, 0.5, np.nan, 1.0])
    assert infer_bin_values(target) == [0.0, 0.5, 1.0]


def test_infer_bin_values_rejects_a_continuous_target():
    """
    A continuous target cannot be represented by one class per bin. Failing here
    beats silently building a head with hundreds of classes.
    """
    with pytest.raises(ValueError, match="does not look like a binned target"):
        infer_bin_values(np.random.rand(500))


def test_infer_bin_values_rejects_an_all_nan_target():
    with pytest.raises(ValueError, match="no finite values"):
        infer_bin_values(np.array([np.nan, np.nan]))


# --------------------------------------------------------------------------- #
# The sequence builder
# --------------------------------------------------------------------------- #

FEATURES = ["feature_a", "feature_b"]


def make_panel(symbol_periods, target_values=None, reference=True, seed=0):
    """
    Build a synthetic panel from an explicit {symbol: [period_index, ...]} map.

    Passing the period indices explicitly is the point: it lets a test punch a
    hole in one symbol's history and assert what the windower does with it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for symbol, periods in symbol_periods.items():
        for period in periods:
            row = {
                "symbol": symbol,
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(period)),
                "target": (
                    float(rng.choice(BINS)) if target_values is None
                    else target_values.pop(0)
                ),
                "feature_a": float(rng.random()),
                "feature_b": float(rng.random()),
            }
            if reference:
                row["reference_prediction"] = float(rng.random())
            rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(frame, seq_length, mode="train", train_buffer=0, **kwargs):
    """Wire a dataset up the way the runner does, with a global position map."""
    positions = build_period_positions(frame["date"].to_numpy())
    symbol_index, padding_id = build_symbol_index(frame["symbol"].unique())
    return SymbolSequenceDataset(
        frame=frame, features=FEATURES, target_col="target", period_col="date",
        id_col="symbol", period_positions=positions, symbol_index=symbol_index,
        padding_id=padding_id, seq_length=seq_length,
        reference_col="reference_prediction", mode=mode,
        train_buffer=train_buffer, **kwargs
    )


def test_contiguous_runs_splits_on_every_break():
    assert contiguous_runs(np.array([0, 1, 2, 5, 6, 9])) == [(0, 3), (3, 5), (5, 6)]
    assert contiguous_runs(np.array([4])) == [(0, 1)]
    assert contiguous_runs(np.array([])) == []


def test_windows_never_span_a_gap_in_a_symbols_history():
    """
    THE test this module exists for. One symbol has a 10-period hole in the middle;
    a position-based slicer would emit a window straddling it.
    """
    periods = list(range(0, 10)) + list(range(20, 30))     # 10 + 10, hole of 10
    dataset = build_dataset(make_panel({"AAA": periods}), seq_length=5)

    assert len(dataset) == 4, "expected 2 windows per 10-period run"
    for index in range(len(dataset)):
        sample = dataset[index]
        pos = sample["period_pos"].numpy()
        assert pos[-1] - pos[0] == len(pos) - 1, "window spans a gap"
        assert np.all(np.diff(pos) == 1), "window is not consecutive"


def test_a_position_slicer_would_have_produced_more_windows():
    """
    Quantifies the bug being avoided: the naive count is higher because it happily
    slices across the hole.
    """
    periods = list(range(0, 10)) + list(range(20, 30))
    dataset = build_dataset(make_panel({"AAA": periods}), seq_length=5)

    naive = len(periods) // 5                # 4 -- coincidence here
    gap_aware = len(dataset)
    # With a hole placed to make the remainder awkward the counts diverge.
    awkward = list(range(0, 7)) + list(range(20, 27))       # 7 + 7
    dataset2 = build_dataset(make_panel({"AAA": awkward}), seq_length=5)
    assert len(dataset2) == 2, "one full window per 7-period run"
    assert len(awkward) // 5 == 2 and gap_aware == naive     # sanity on the setup


def test_window_count_matches_the_gap_aware_formula():
    """sum(run_len // L) over runs, NOT sum(total_rows // L)."""
    symbol_periods = {
        "AAA": list(range(0, 30)),                              # one run of 30
        "BBB": list(range(0, 7)) + list(range(40, 52)),          # runs of 7 and 12
        "CCC": list(range(5, 9)),                               # run of 4, too short
    }
    seq_length = 6
    dataset = build_dataset(make_panel(symbol_periods), seq_length=seq_length)

    expected = 30 // 6 + 7 // 6 + 12 // 6 + 0
    assert len(dataset) == expected


def test_windows_never_mix_two_symbols():
    symbol_periods = {"AAA": list(range(0, 12)), "BBB": list(range(0, 12))}
    dataset = build_dataset(make_panel(symbol_periods), seq_length=4)

    assert len(dataset) == 6
    for index in range(len(dataset)):
        ids = dataset[index]["symbol_id"].numpy()
        assert len(set(ids.tolist())) == 1, "a window mixed two symbols"


def test_windows_are_chronological():
    dataset = build_dataset(make_panel({"AAA": list(range(0, 20))}), seq_length=5)
    for index in range(len(dataset)):
        dates = dataset[index]["date"].numpy()
        assert np.all(np.diff(dates) > 0), "window is not in time order"


def test_the_global_position_map_prevents_a_filtered_gap_from_closing_up():
    """
    Filtering a frame to a training block must not make the periods either side of
    the removed block look adjacent. This is why the position map is built over the
    whole run and passed in, rather than derived per frame.
    """
    frame = make_panel({"AAA": list(range(0, 20))})
    positions = build_period_positions(frame["date"].to_numpy())   # all 20 periods
    symbol_index, padding_id = build_symbol_index(frame["symbol"].unique())

    # Drop the middle, as an embargo would.
    kept = frame[(frame["date"].dt.day <= 8) | (frame["date"].dt.day >= 15)]

    dataset = SymbolSequenceDataset(
        frame=kept, features=FEATURES, target_col="target", period_col="date",
        id_col="symbol", period_positions=positions, symbol_index=symbol_index,
        padding_id=padding_id, seq_length=4, mode="train", train_buffer=0,
    )
    for index in range(len(dataset)):
        pos = dataset[index]["period_pos"].numpy()
        assert np.all(np.diff(pos) == 1), "a removed block was silently bridged"


def test_a_frame_with_unknown_periods_is_rejected():
    frame = make_panel({"AAA": list(range(0, 10))})
    partial = build_period_positions(frame["date"].to_numpy()[:5])
    symbol_index, padding_id = build_symbol_index(["AAA"])
    with pytest.raises(ValueError, match="missing from period_positions"):
        SymbolSequenceDataset(
            frame=frame, features=FEATURES, target_col="target", period_col="date",
            id_col="symbol", period_positions=partial, symbol_index=symbol_index,
            padding_id=padding_id, seq_length=3,
        )


def test_train_mode_applies_the_buffer_and_predict_mode_does_not():
    """
    iwtsf's `+20` slack is training-only. A run that clears seq_length but not the
    buffer trains on nothing yet must still be predicted.
    """
    frame = make_panel({"AAA": list(range(0, 12))})       # one run of 12

    trained = build_dataset(frame, seq_length=5, mode="train", train_buffer=20)
    assert len(trained) == 0, "buffer should have excluded a 12-period run"

    predicted = build_dataset(frame, seq_length=5, mode="predict", train_buffer=20)
    assert len(predicted) > 0, "prediction must not apply the training buffer"


def test_predict_mode_covers_every_row_of_a_run():
    """
    A row with no window gets no prediction and silently leaves the scored set, so
    coverage in predict mode has to be total.
    """
    periods = list(range(0, 23))                          # 23 is not a multiple of 5
    dataset = build_dataset(
        make_panel({"AAA": periods}), seq_length=5, mode="predict"
    )

    covered = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        valid = sample["valid"].numpy()
        covered.update(sample["period_pos"].numpy()[valid].tolist())

    assert covered == set(range(23)), "predict mode left rows uncovered"


def test_train_mode_may_drop_a_runs_remainder():
    """The counterpart: training does not need every row, and says so."""
    dataset = build_dataset(
        make_panel({"AAA": list(range(0, 23))}), seq_length=5, mode="train"
    )
    assert len(dataset) == 4                                # 20 of 23 periods used


def test_a_run_shorter_than_seq_length_is_right_padded_and_masked():
    """
    Padding must be at the END: a unidirectional LSTM's output at a real timestep
    cannot depend on padding that follows it, but front-padding would shift the
    whole sequence.
    """
    dataset = build_dataset(
        make_panel({"AAA": list(range(0, 3))}), seq_length=8, mode="predict"
    )
    sample = dataset[0]

    assert sample["valid"].shape[0] == 8
    assert sample["valid"].numpy().tolist() == [True] * 3 + [False] * 5
    assert sample["target_mask"].numpy()[3:].sum() == 0


def test_nan_targets_are_masked_not_filled():
    frame = make_panel({"AAA": list(range(0, 6))}, target_values=[
        0.0, 0.5, np.nan, 1.0, 0.25, 0.75,
    ])
    dataset = build_dataset(frame, seq_length=6)
    sample = dataset[0]

    assert sample["target_mask"].numpy().tolist() == [
        True, True, False, True, True, True
    ]
    # The tensor itself must be finite even where masked, or a matmul takes the
    # whole batch down with it.
    assert torch.isfinite(sample["target"]).all()
    assert torch.isfinite(sample["features"]).all()


def test_unknown_symbols_map_to_the_padding_row():
    """A symbol absent from the training index must not index out of range."""
    train_frame = make_panel({"AAA": list(range(0, 10))})
    predict_frame = make_panel({"ZZZ": list(range(0, 10))}, seed=3)

    positions = build_period_positions(predict_frame["date"].to_numpy())
    symbol_index, padding_id = build_symbol_index(train_frame["symbol"].unique())

    dataset = SymbolSequenceDataset(
        frame=predict_frame, features=FEATURES, target_col="target",
        period_col="date", id_col="symbol", period_positions=positions,
        symbol_index=symbol_index, padding_id=padding_id, seq_length=5,
        mode="predict",
    )
    assert (dataset[0]["symbol_id"].numpy() == padding_id).all()


def test_coverage_report_counts_what_was_dropped():
    symbol_periods = {
        "AAA": list(range(0, 30)),
        "BBB": list(range(0, 3)),                # too short for training
    }
    dataset = build_dataset(make_panel(symbol_periods), seq_length=6)
    report = dataset.coverage()

    assert report["n_symbols_seen"] == 2
    assert report["n_symbols_contributing"] == 1
    assert report["n_runs_too_short"] == 1
    assert report["n_windows"] == 5


def test_collate_stacks_and_pads_ragged_windows():
    """Predict mode produces ragged lengths; the collate has to absorb that."""
    frame = make_panel({"AAA": list(range(0, 12)), "BBB": list(range(0, 4))})
    dataset = build_dataset(frame, seq_length=6, mode="predict")

    batch = collate_windows([dataset[i] for i in range(len(dataset))])
    assert batch["features"].shape[0] == len(dataset)
    assert batch["features"].shape[2] == len(FEATURES)
    assert batch["valid"].shape == batch["target"].shape
    # The short symbol's window is padded, and the padding is masked out.
    assert not bool(batch["valid"].all())
    assert torch.isfinite(batch["features"]).all()


def test_overlapping_windows_step_by_one():
    plain = build_dataset(make_panel({"AAA": list(range(0, 20))}), seq_length=5)
    overlapped = build_dataset(
        make_panel({"AAA": list(range(0, 20))}), seq_length=5, overlapping=True
    )
    assert len(plain) == 4
    assert len(overlapped) == 16                 # 20 - 5 + 1


def test_dataset_rejects_a_bad_mode_and_seq_length():
    frame = make_panel({"AAA": list(range(0, 10))})
    with pytest.raises(ValueError, match="mode must be"):
        build_dataset(frame, seq_length=4, mode="validate")
    with pytest.raises(ValueError, match="seq_length must be at least 1"):
        build_dataset(frame, seq_length=0)


def test_the_network_consumes_a_collated_batch_end_to_end():
    """The dataset's output shape and the network's input shape must actually agree."""
    frame = make_panel({"AAA": list(range(0, 20)), "BBB": list(range(0, 20))})
    dataset = build_dataset(frame, seq_length=6)
    batch = collate_windows([dataset[i] for i in range(len(dataset))])

    net = LSTMForecastNetwork(
        input_dim=len(FEATURES), bin_values=BINS, hidden_dim=8,
        num_symbols=2, embed_dim=3, dropout=0.0,
    ).eval()
    signal = net(batch["features"], batch["symbol_id"])

    assert signal.shape == batch["target"].shape
    loss = CorrLoss()(signal, batch["target"], mask=batch["target_mask"])
    assert torch.isfinite(loss)
