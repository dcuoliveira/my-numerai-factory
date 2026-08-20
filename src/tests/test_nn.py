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
from src.prediction.nn.train import (  # noqa: E402
    chronological_split,
    clone_state,
    load_checkpoint,
    make_loader,
    predict_rows,
    run_epoch,
    select_seeds,
    slice_loss,
    train_until_early_stop,
    walk_forward,
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


# --------------------------------------------------------------------------- #
# The trainer, and the three iwtsf bugs it fixes
# --------------------------------------------------------------------------- #

TRAIN_CONFIG = {
    "hidden": 8, "num_layers": 1, "embed_dim": 4, "dropout": 0.0,
    "dropout_type": "regular", "layer_norm": True,
    "seq_length": 6, "overlapping": False, "batch_size": 8, "shuffle": True,
    "train_buffer": 0,
    "lr": 0.01, "epochs": 6, "patience": 3, "max_grad_norm": 1.0,
    "val_frac": 0.3, "embargo": 2, "seed": 42, "max_trials": 3, "top_frac": 0.5,
    "cv_scheme": "expanding", "refit_every": 6, "min_train": 0,
    "reference_col": "reference_prediction",
}


def _loaders(frame, seq_length=6):
    """A train loader and a val loader over the same small panel."""
    dataset = build_dataset(frame, seq_length=seq_length)
    return (
        make_loader(dataset, batch_size=8, shuffle=False),
        make_loader(dataset, batch_size=8, shuffle=False),
        dataset,
    )


def test_clone_state_snapshots_weights_instead_of_aliasing_them():
    """
    REGRESSION TEST FOR THE IWTSF BUG.

    `model.state_dict()` hands back references to the live parameter tensors, so a
    snapshot taken that way keeps changing as training continues -- which made
    iwtsf's walk-forward restore the LAST epoch's weights while believing it had
    restored the best. Assert the clone is immune and, for contrast, that the naive
    version really is not.
    """
    net = _net()
    aliased = net.state_dict()
    cloned = clone_state(net)
    before = net.output_proj.weight.detach().clone()

    optimiser = torch.optim.SGD(net.parameters(), lr=1.0)
    for _ in range(3):
        optimiser.zero_grad()
        net(torch.rand(2, 6, 22), torch.zeros(2, 6, dtype=torch.long)).sum().backward()
        optimiser.step()

    assert not torch.allclose(before, net.output_proj.weight), "weights did not move"
    # The naive snapshot tracked the live weights -- the bug.
    assert torch.allclose(aliased["output_proj.weight"], net.output_proj.weight)
    # The clone held still -- the fix.
    assert torch.allclose(cloned["output_proj.weight"], before)


def test_early_stopping_restores_the_best_epoch_not_the_last():
    """
    End-to-end version of the same bug: drive validation loss to get worse after an
    early best, then assert the returned state is the good one.
    """
    frame = make_panel({"AAA": list(range(0, 40)), "BBB": list(range(0, 40))})
    train_loader, val_loader, _ = _loaders(frame)

    net = LSTMForecastNetwork(
        input_dim=len(FEATURES), bin_values=BINS, hidden_dim=6,
        num_symbols=2, embed_dim=3, dropout=0.0,
    )
    config = dict(TRAIN_CONFIG, epochs=8, patience=8, lr=0.5)   # lr high enough to overshoot
    result = train_until_early_stop(
        net, train_loader, val_loader, CorrLoss(), config,
        torch.device("cpu"), verbose=False,
    )

    assert result["best_epoch"] >= 1
    assert result["best_val_loss"] == pytest.approx(min(result["val_history"]))

    # Loading the returned state must reproduce the best score, not the final one.
    restored = LSTMForecastNetwork(
        input_dim=len(FEATURES), bin_values=BINS, hidden_dim=6,
        num_symbols=2, embed_dim=3, dropout=0.0,
    )
    restored.load_state_dict(result["best_state"])
    reproduced = slice_loss(restored, val_loader, CorrLoss(), torch.device("cpu"))
    assert reproduced == pytest.approx(result["best_val_loss"], abs=1e-5)


def test_slice_loss_is_not_the_mean_of_per_batch_losses():
    """
    Fix 4. A correlation is not linear in its sample, so batching changes the mean of
    per-batch correlations but must not change the slice statistic.
    """
    frame = make_panel({"AAA": list(range(0, 40)), "BBB": list(range(0, 40))})
    dataset = build_dataset(frame, seq_length=6)
    net = LSTMForecastNetwork(
        input_dim=len(FEATURES), bin_values=BINS, hidden_dim=6,
        num_symbols=2, embed_dim=3, dropout=0.0,
    ).eval()

    whole = slice_loss(
        net, make_loader(dataset, batch_size=len(dataset), shuffle=False),
        CorrLoss(), torch.device("cpu"),
    )
    split = slice_loss(
        net, make_loader(dataset, batch_size=2, shuffle=False),
        CorrLoss(), torch.device("cpu"),
    )
    assert whole == pytest.approx(split, abs=1e-5), "slice loss moved with batch_size"


def test_run_epoch_clips_gradients():
    """
    Fix 2. Clipping now lives in the one function both phases call, so it cannot be
    applied in seed selection and forgotten in the walk-forward.
    """
    frame = make_panel({"AAA": list(range(0, 30)), "BBB": list(range(0, 30))})
    train_loader, _, _ = _loaders(frame)

    net = LSTMForecastNetwork(
        input_dim=len(FEATURES), bin_values=BINS, hidden_dim=6,
        num_symbols=2, embed_dim=3, dropout=0.0,
    )
    seen = {}
    original = torch.nn.utils.clip_grad_norm_

    def _spy(parameters, max_norm, *args, **kwargs):
        seen["max_norm"] = max_norm
        return original(parameters, max_norm, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = _spy
    try:
        run_epoch(
            net, train_loader, CorrLoss(),
            torch.optim.Adam(net.parameters(), lr=0.01),
            torch.device("cpu"), max_grad_norm=1.0,
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = original

    assert seen.get("max_norm") == 1.0, "run_epoch did not clip"


def test_chronological_split_leaves_an_embargo_gap():
    periods = list(range(20))
    train, val = chronological_split(periods, val_frac=0.25, embargo=3)
    assert val == list(range(15, 20))
    assert train == list(range(12))                 # 20 - 5 - 3
    assert max(train) + 3 < min(val), "embargo gap is too small"


def test_predict_rows_averages_overlaps_and_leaves_uncovered_rows_nan():
    frame = make_panel({"AAA": list(range(0, 23))})
    positions = build_period_positions(frame["date"].to_numpy())
    symbol_index, padding_id = build_symbol_index(frame["symbol"].unique())
    dataset = SymbolSequenceDataset(
        frame=frame, features=FEATURES, target_col="target", period_col="date",
        id_col="symbol", period_positions=positions, symbol_index=symbol_index,
        padding_id=padding_id, seq_length=5, mode="predict",
    )
    net = LSTMForecastNetwork(
        input_dim=len(FEATURES), bin_values=BINS, hidden_dim=6,
        num_symbols=1, embed_dim=3, dropout=0.0,
    ).eval()

    predictions, counts = predict_rows(
        net, make_loader(dataset, batch_size=4, shuffle=False),
        torch.device("cpu"), len(frame),
    )
    assert np.isfinite(predictions).all(), "predict mode left a row unpredicted"
    assert counts.min() >= 1
    assert counts.max() >= 2, "the overlapping tail window should double-cover rows"
    assert predictions.min() >= min(BINS) - 1e-6
    assert predictions.max() <= max(BINS) + 1e-6


def test_no_training_target_lies_within_the_embargo_of_a_predicted_block():
    """
    THE LEAKAGE TEST.

    Feature history may reach through the embargo -- a feature at date t is known at
    t. A *target* may not: a 20-day forward target from inside the gap overlaps the
    block being predicted. So assert that every period contributing a loss term is
    at least `embargo` periods before the block starts.
    """
    n_periods = 60
    frame = make_panel({
        "AAA": list(range(n_periods)), "BBB": list(range(n_periods)),
    })
    all_periods = np.sort(frame["date"].unique())
    embargo = 5
    block_start_position = 40
    block = all_periods[block_start_position:block_start_position + 10]

    train_end = block_start_position - embargo
    train_periods = all_periods[:train_end]
    train_frame = frame[frame["date"].isin(set(train_periods))]

    positions = build_period_positions(all_periods)
    symbol_index, padding_id = build_symbol_index(train_frame["symbol"].unique())
    dataset = SymbolSequenceDataset(
        frame=train_frame, features=FEATURES, target_col="target",
        period_col="date", id_col="symbol", period_positions=positions,
        symbol_index=symbol_index, padding_id=padding_id, seq_length=6,
        mode="train", train_buffer=0,
    )

    for index in range(len(dataset)):
        sample = dataset[index]
        contributing = sample["period_pos"].numpy()[sample["target_mask"].numpy()]
        assert contributing.max() < block_start_position - embargo + 1, (
            "a training target sits inside the embargo gap"
        )
        assert contributing.max() <= train_end - 1


def test_a_prediction_window_may_read_features_through_the_embargo():
    """
    The other half of the rule: prediction is *allowed* to use the gap as input, and
    it has to be, or the first rows of a block would have no history at all.
    """
    n_periods = 60
    frame = make_panel({"AAA": list(range(n_periods))})
    all_periods = np.sort(frame["date"].unique())
    block_start = 40
    seq_length = 6

    lookback = set(all_periods[block_start - seq_length:block_start + 5])
    predict_frame = frame[frame["date"].isin(lookback)].reset_index(drop=True)

    positions = build_period_positions(all_periods)
    symbol_index, padding_id = build_symbol_index(["AAA"])
    dataset = SymbolSequenceDataset(
        frame=predict_frame, features=FEATURES, target_col="target",
        period_col="date", id_col="symbol", period_positions=positions,
        symbol_index=symbol_index, padding_id=padding_id, seq_length=seq_length,
        mode="predict",
    )
    covered = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        covered.update(sample["period_pos"].numpy()[sample["valid"].numpy()].tolist())

    assert any(position < block_start for position in covered), (
        "prediction windows must be able to read pre-block feature history"
    )


def test_seed_selection_keeps_the_best_fraction_and_is_reproducible(tmp_path):
    frame = make_panel({
        "AAA": list(range(0, 60)), "BBB": list(range(0, 60)), "CCC": list(range(0, 60)),
    })
    frame["split"] = "train"

    config = dict(TRAIN_CONFIG, max_trials=4, top_frac=0.5, epochs=2, patience=2)
    first = select_seeds(
        frame, FEATURES, "target", "date", "symbol", BINS, config,
        CorrLoss(), torch.device("cpu"),
        checkpoint_path=str(tmp_path / "seeds.json"), verbose=False,
    )

    assert first["n_trials"] == 4
    assert len(first["selected"]) == 2
    # The selected seeds really are the best-scoring ones.
    ranked = sorted(first["val_scores"].items(), key=lambda item: item[1])
    assert set(first["selected"]) == {int(seed) for seed, _ in ranked[:2]}


def test_seed_selection_resumes_from_its_checkpoint(tmp_path):
    """At 100 trainings an interrupted run is near-certain, so resume must work."""
    frame = make_panel({
        "AAA": list(range(0, 60)), "BBB": list(range(0, 60)), "CCC": list(range(0, 60)),
    })
    frame["split"] = "train"
    checkpoint = tmp_path / "seeds.json"

    config = dict(TRAIN_CONFIG, max_trials=3, top_frac=0.5, epochs=2, patience=2)
    select_seeds(
        frame, FEATURES, "target", "date", "symbol", BINS, config,
        CorrLoss(), torch.device("cpu"),
        checkpoint_path=str(checkpoint), verbose=False,
    )
    assert checkpoint.exists()
    saved = load_checkpoint(str(checkpoint))
    assert len(saved) == 3
    assert all(isinstance(key, int) for key in saved), "seed keys must survive as ints"

    # A second call finds every seed already done and trains nothing new.
    again = select_seeds(
        frame, FEATURES, "target", "date", "symbol", BINS, config,
        CorrLoss(), torch.device("cpu"),
        checkpoint_path=str(checkpoint), verbose=False,
    )
    assert again["val_scores"] == saved


def test_seed_selection_says_so_when_seq_length_is_too_long():
    """
    The most likely first-run failure on this panel, so it must name the cause rather
    than dying on an empty loader.
    """
    frame = make_panel({"AAA": list(range(0, 30))})
    frame["split"] = "train"
    config = dict(TRAIN_CONFIG, seq_length=500, max_trials=1)

    with pytest.raises(ValueError, match="seq_length"):
        select_seeds(
            frame, FEATURES, "target", "date", "symbol", BINS, config,
            CorrLoss(), torch.device("cpu"), verbose=False,
        )


def test_walk_forward_predicts_only_validation_periods_once_each():
    frame = make_panel({
        "AAA": list(range(0, 80)), "BBB": list(range(0, 80)), "CCC": list(range(0, 80)),
    })
    all_periods = np.sort(frame["date"].unique())
    frame["split"] = np.where(
        frame["date"].isin(set(all_periods[56:])), "validation", "train"
    )

    config = dict(
        TRAIN_CONFIG, epochs=2, patience=2, refit_every=8, embargo=2, seq_length=5,
    )
    predictions, records, skipped = walk_forward(
        frame, FEATURES, "target", "date", "symbol", BINS, seeds=[1, 2],
        config=config, loss_fn=CorrLoss(), device=torch.device("cpu"), verbose=False,
    )

    validation_dates = set(all_periods[56:])
    assert set(predictions["date"]) <= validation_dates, "predicted a training period"
    # Every (symbol, date) appears once: the lookback rows are dropped and the
    # overlapping windows are averaged, not duplicated.
    assert not predictions.duplicated(subset=["symbol", "date"]).any()
    assert predictions["prediction"].notna().all()
    assert len(records) == len(range(0, len(all_periods[56:]), 8))
    assert records[0]["seeds"] == "1,2"
    assert records[0]["n_seeds"] == 2
    assert "seed1_val" in records[0]


def test_walk_forward_trains_a_fresh_model_per_block_by_default(monkeypatch):
    """
    Fix 3. iwtsf built the model once per seed OUTSIDE the block loop, so every block
    warm-started from the previous one and the whole back-test was really a single
    continuous training run.

    Asserted by observation rather than by inspecting internals: count how often a
    network is constructed. One per (block, seed) means fresh each time; fewer would
    mean state is being reused.
    """
    frame = make_panel({
        "AAA": list(range(0, 80)), "BBB": list(range(0, 80)), "CCC": list(range(0, 80)),
    })
    all_periods = np.sort(frame["date"].unique())
    frame["split"] = np.where(
        frame["date"].isin(set(all_periods[56:])), "validation", "train"
    )

    from src.prediction.nn import train as train_module

    built = []
    original = train_module.build_network

    def _counting_build(*args, **kwargs):
        built.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "build_network", _counting_build)

    config = dict(
        TRAIN_CONFIG, epochs=2, patience=2, refit_every=8, embargo=2, seq_length=5,
        warm_start=False,
    )
    _, records, _ = walk_forward(
        frame, FEATURES, "target", "date", "symbol", BINS, seeds=[1, 2],
        config=config, loss_fn=CorrLoss(), device=torch.device("cpu"), verbose=False,
    )

    assert len(built) == len(records) * 2, (
        "expected one fresh network per block per seed, got {} for {} blocks x 2 "
        "seeds".format(len(built), len(records))
    )


def test_warm_start_carries_weights_between_blocks_when_asked():
    """The opt-in path still works -- it is just not the default any more."""
    frame = make_panel({
        "AAA": list(range(0, 80)), "BBB": list(range(0, 80)), "CCC": list(range(0, 80)),
    })
    all_periods = np.sort(frame["date"].unique())
    frame["split"] = np.where(
        frame["date"].isin(set(all_periods[56:])), "validation", "train"
    )
    config = dict(
        TRAIN_CONFIG, epochs=1, patience=1, refit_every=8, embargo=2, seq_length=5,
        warm_start=True,
    )
    predictions, records, _ = walk_forward(
        frame, FEATURES, "target", "date", "symbol", BINS, seeds=[1],
        config=config, loss_fn=CorrLoss(), device=torch.device("cpu"), verbose=False,
    )
    assert len(records) > 1, "need at least two blocks for warm start to mean anything"
    assert predictions["prediction"].notna().all()


def test_a_run_without_a_validation_slice_reports_nan_not_infinity(capsys):
    """
    Found by the first smoke run. When the early-stopping slice is too short to hold
    a single window, early stopping is off -- and the initial +inf sentinel used to
    leak out as the reported "best" score, get compared against real losses when
    ranking seeds, and land in the results json as the non-standard literal
    `Infinity`. It must be NaN, and the caller must be told.
    """
    frame = make_panel({"AAA": list(range(0, 30)), "BBB": list(range(0, 30))})
    train_loader, _, _ = _loaders(frame)

    net = LSTMForecastNetwork(
        input_dim=len(FEATURES), bin_values=BINS, hidden_dim=6,
        num_symbols=2, embed_dim=3, dropout=0.0,
    )
    result = train_until_early_stop(
        net, train_loader, None, CorrLoss(), dict(TRAIN_CONFIG, epochs=2),
        torch.device("cpu"), verbose=False,
    )

    assert result["early_stopped"] is False
    assert np.isnan(result["best_val_loss"]), "the +inf sentinel leaked out"
    assert result["epochs_run"] == 2, "without early stopping it must run every epoch"
    assert result["best_state"] is not None


def test_walk_forward_warns_when_the_stop_slice_yields_no_windows(capsys):
    """The same condition end to end: it changes what the run does, so it is loud."""
    frame = make_panel({
        "AAA": list(range(0, 70)), "BBB": list(range(0, 70)),
    })
    all_periods = np.sort(frame["date"].unique())
    frame["split"] = np.where(
        frame["date"].isin(set(all_periods[50:])), "validation", "train"
    )
    # For block 0 the training block is 48 periods, split into a 32-period fit slice
    # and a 14-period early-stopping slice. A buffer of 15 makes a window need 20
    # periods: too many for the stop slice, few enough for the fit slice -- which is
    # exactly the condition being tested. Push it higher and the fit slice starves
    # too, the block is skipped, and the run fails for an unrelated reason.
    config = dict(
        TRAIN_CONFIG, epochs=2, patience=2, refit_every=10, embargo=2,
        seq_length=5, train_buffer=15,
    )
    _, records, _ = walk_forward(
        frame, FEATURES, "target", "date", "symbol", BINS, seeds=[3],
        config=config, loss_fn=CorrLoss(), device=torch.device("cpu"), verbose=False,
    )

    captured = capsys.readouterr().out
    assert "Early stopping is DISABLED" in captured
    assert records[0]["early_stopping"] is False
    assert records[0]["stop_windows"] == 0
    assert np.isnan(records[0]["best_val_loss"])


def test_block_records_are_flat_and_csv_safe():
    """
    write_outputs feeds these straight to hyperparams_table and then to_csv, so a
    nested dict or a list would land in a cell as a Python repr.
    """
    frame = make_panel({
        "AAA": list(range(0, 70)), "BBB": list(range(0, 70)),
    })
    all_periods = np.sort(frame["date"].unique())
    frame["split"] = np.where(
        frame["date"].isin(set(all_periods[50:])), "validation", "train"
    )
    config = dict(
        TRAIN_CONFIG, epochs=2, patience=2, refit_every=10, embargo=2, seq_length=5,
    )
    _, records, _ = walk_forward(
        frame, FEATURES, "target", "date", "symbol", BINS, seeds=[1, 2],
        config=config, loss_fn=CorrLoss(), device=torch.device("cpu"), verbose=False,
    )

    for record in records:
        for key, value in record.items():
            assert not isinstance(value, (dict, list, tuple)), (
                "{} is not a scalar, so it would be repr'd into the CSV".format(key)
            )
    assert records[0]["seeds"] == "1,2"
    assert "seed1_val" in records[0] and "seed2_epochs" in records[0]


# --------------------------------------------------------------------------- #
# prepare_panel's reference handling
# --------------------------------------------------------------------------- #

def _patch_loaders(monkeypatch, frame, features):
    """
    Make prepare_panel offline: a fake panel, and no reference or meta model.

    load_reference / load_meta_model are wrapped in try/except by prepare_panel, so
    making them raise is the supported way to say "this tournament has neither" --
    and it is what forces the meta proxy to be unavailable.
    """
    from src.prediction.nn import pipeline

    monkeypatch.setattr(
        pipeline, "load_tournament_data",
        lambda **kwargs: (frame.copy(), list(features)),
    )
    for name in ("load_reference", "load_meta_model"):
        monkeypatch.setattr(
            pipeline, name,
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not available here")),
        )


def _panel_frame():
    frame = make_panel({"AAA": list(range(0, 40)), "BBB": list(range(0, 40))},
                       reference=False)
    frame = frame.rename(columns={"target": "target_binned_return_20"})
    frame["split"] = "train"
    return frame


def test_an_unbuildable_proxy_does_not_sink_a_corr_run(monkeypatch, capsys):
    """
    Found by a smoke run on target_binned_return_60.

    --bmc-reference defaults to meta_proxy because a BMC *training* loss needs
    gradient on every batch. But CORR never reads a reference, so failing a CORR run
    because the proxy could not be built is refusing to do work that does not depend
    on the thing that broke. It must degrade to the benchmark and say so.
    """
    from src.prediction.nn.pipeline import prepare_panel

    frame = _panel_frame()
    _patch_loaders(monkeypatch, frame, FEATURES)

    panel = prepare_panel(
        "crypto", target="target_binned_return_20", bmc_reference="meta_proxy",
        require_reference=False, verbose=False,
    )

    assert panel["bmc_reference"] == "benchmark", "should have downgraded"
    assert panel["reference_col"] is None
    assert "meta proxy could not be built" in capsys.readouterr().out


def test_an_unbuildable_proxy_does_sink_a_bmc_run(monkeypatch):
    """The other half: a loss that genuinely needs the reference must not proceed."""
    from src.prediction.nn.pipeline import prepare_panel

    frame = _panel_frame()
    _patch_loaders(monkeypatch, frame, FEATURES)

    with pytest.raises(ValueError, match="meta_proxy is unavailable"):
        prepare_panel(
            "crypto", target="target_binned_return_20", bmc_reference="meta_proxy",
            require_reference=True, verbose=False,
        )


def test_prepare_panel_rejects_an_unknown_reference_kind(monkeypatch):
    from src.prediction.nn.pipeline import prepare_panel

    frame = _panel_frame()
    _patch_loaders(monkeypatch, frame, FEATURES)

    with pytest.raises(ValueError, match="must be 'benchmark' or 'meta_proxy'"):
        prepare_panel(
            "crypto", target="target_binned_return_20", bmc_reference="mmc",
            verbose=False,
        )


def test_walk_forward_is_deterministic_for_a_fixed_seed():
    frame = make_panel({
        "AAA": list(range(0, 70)), "BBB": list(range(0, 70)),
    })
    all_periods = np.sort(frame["date"].unique())
    frame["split"] = np.where(
        frame["date"].isin(set(all_periods[50:])), "validation", "train"
    )
    config = dict(
        TRAIN_CONFIG, epochs=2, patience=2, refit_every=10, embargo=2, seq_length=5,
    )

    runs = []
    for _ in range(2):
        predictions, _, _ = walk_forward(
            frame, FEATURES, "target", "date", "symbol", BINS, seeds=[7],
            config=config, loss_fn=CorrLoss(), device=torch.device("cpu"),
            verbose=False,
        )
        runs.append(predictions.sort_values(["symbol", "date"])["prediction"].to_numpy())

    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# Symbol-embedding export, and the dataset builder on top of it
# --------------------------------------------------------------------------- #

from src.data_mani.build_symbol_embedding import attach_embeddings  # noqa: E402
from src.prediction.nn.embeddings import (  # noqa: E402
    FEATURE_PREFIX,
    embedding_columns,
    extract_embedding,
    load_embeddings,
    save_embeddings,
    select_best_seed,
)


def _export_frame(blocks=(0, 1), seeds=(11, 22), dim=3, symbols=("AAA", "BBB")):
    """A synthetic export shaped exactly like the real one."""
    rows = []
    for block in blocks:
        for seed in seeds:
            for symbol in symbols:
                row = {
                    "block": block, "seed": seed, "symbol": symbol,
                    "train_end": pd.Timestamp("2020-01-01") + pd.Timedelta(days=20 * (block + 1)),
                    "val_loss": -0.05 - 0.01 * (seed == 11) - 0.001 * block,
                }
                for d in range(dim):
                    row["emb_{}".format(d)] = float(block + 1) + d / 10.0
                rows.append(row)
    return pd.DataFrame(rows)


def test_extract_embedding_returns_one_row_per_symbol_without_the_padding_row():
    """
    The padding row is a structural zero the network never trains, so exporting it
    would put a fake 'unknown symbol' vector in the table.
    """
    net = _net(num_symbols=3, embed_dim=4)
    symbol_index = {"AAA": 0, "BBB": 1, "CCC": 2}

    frame = extract_embedding(
        net, symbol_index, block=0, seed=7, train_end="2021-01-01", val_loss=-0.04,
    )

    assert len(frame) == 3, "padding row should not be exported"
    assert list(frame["symbol"]) == ["AAA", "BBB", "CCC"]
    assert embedding_columns(frame) == ["emb_0", "emb_1", "emb_2", "emb_3"]
    assert frame["block"].eq(0).all() and frame["seed"].eq(7).all()

    # The values really are the network's weights, in symbol_index order.
    weights = net.symbol_embedding.weight.detach().numpy()
    np.testing.assert_allclose(
        frame[embedding_columns(frame)].to_numpy(), weights[:3], rtol=1e-6
    )


def test_extract_embedding_refuses_a_network_without_embeddings():
    net = _net(embed_dim=0, num_symbols=None)
    with pytest.raises(ValueError, match="embed_dim=0"):
        extract_embedding(net, {}, block=0, seed=1, train_end="2021-01-01", val_loss=0.0)


def test_embedding_columns_sort_numerically_not_lexically():
    """
    Plain sorted() puts emb_10 before emb_2, which would permute the feature space
    silently -- everything downstream would still 'work'.
    """
    frame = pd.DataFrame(columns=["emb_0", "emb_1", "emb_2", "emb_10", "emb_11"])
    assert embedding_columns(frame) == ["emb_0", "emb_1", "emb_2", "emb_10", "emb_11"]


def test_save_embeddings_appends_across_blocks(tmp_path):
    """Written per block so an interrupted run keeps what it finished."""
    path = str(tmp_path / "emb.parquet")
    first = _export_frame(blocks=(0,), seeds=(11,))
    second = _export_frame(blocks=(1,), seeds=(11,))

    save_embeddings(path, [first])
    save_embeddings(path, [second])

    loaded = load_embeddings(path)
    assert set(loaded["block"]) == {0, 1}
    assert len(loaded) == len(first) + len(second)


def test_load_embeddings_rejects_a_frame_that_is_not_an_export(tmp_path):
    path = str(tmp_path / "junk.parquet")
    pd.DataFrame({"symbol": ["AAA"], "x": [1.0]}).to_parquet(path)
    with pytest.raises(ValueError, match="not a symbol-embedding export"):
        load_embeddings(path)


def test_select_best_seed_keeps_one_seed_per_block():
    """
    Seeds must be CHOSEN, never averaged: each network's embedding space has its own
    arbitrary basis, so averaging them cancels the structure.
    """
    export = _export_frame(blocks=(0, 1), seeds=(11, 22))
    picked = select_best_seed(export)

    assert set(picked["block"]) == {0, 1}
    for block in (0, 1):
        rows = picked[picked["block"] == block]
        assert rows["seed"].nunique() == 1
        # seed 11 was given the lower (better) val_loss in the fixture
        assert rows["seed"].iloc[0] == 11


def test_attach_embeddings_never_uses_a_vector_from_the_future():
    """
    THE LEAKAGE TEST for this builder.

    A vector may only reach a row whose date is at least `lag` periods after the
    network's train_end. An embedding is trained against the target, so a violation
    here hands the model a compressed summary of the answer.
    """
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "symbol": ["AAA"] * 60,
        "date": dates,
        "target_binned_return_20": 0.5,
    })
    # One export, trained through period index 20.
    export = pd.DataFrame([{
        "block": 0, "seed": 1, "symbol": "AAA", "train_end": dates[20],
        "val_loss": -0.05, "emb_0": 7.0, "emb_1": -3.0,
    }])

    lag = 5
    out, covered = attach_embeddings(df, export, "symbol", "date", lag=lag)
    col = "{}_0".format(FEATURE_PREFIX)

    got = out[col].to_numpy() != 0.0
    first_allowed = 20 + lag       # train_end position + lag
    assert not got[:first_allowed].any(), (
        "a vector reached a row before train_end + lag -- future information"
    )
    assert got[first_allowed:].all(), "rows after the lag should all be covered"
    assert covered == 60 - first_allowed


def test_attach_embeddings_uses_the_most_recent_legal_export():
    """With several exports, a row must get the newest one it is allowed to see."""
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    df = pd.DataFrame({"symbol": ["AAA"] * 80, "date": dates})
    export = pd.DataFrame([
        {"block": 0, "seed": 1, "symbol": "AAA", "train_end": dates[10],
         "val_loss": -0.05, "emb_0": 1.0},
        {"block": 1, "seed": 1, "symbol": "AAA", "train_end": dates[40],
         "val_loss": -0.06, "emb_0": 2.0},
    ])

    out, _ = attach_embeddings(df, export, "symbol", "date", lag=5)
    col = "{}_0".format(FEATURE_PREFIX)

    assert out[col].iloc[12] == 0.0          # before 10 + 5
    assert out[col].iloc[20] == 1.0          # block 0 only
    assert out[col].iloc[44] == 1.0          # block 1 not legal yet (40 + 5)
    assert out[col].iloc[50] == 2.0          # now on block 1


def test_attach_embeddings_gives_unknown_symbols_zeros():
    """
    A symbol the network never saw gets the same zero vector the network itself uses
    for an unknown symbol -- the truthful encoding of 'nothing is known'.
    """
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "symbol": ["AAA"] * 40 + ["ZZZ"] * 40,
        "date": list(dates) * 2,
    })
    export = pd.DataFrame([{
        "block": 0, "seed": 1, "symbol": "AAA", "train_end": dates[5],
        "val_loss": -0.05, "emb_0": 4.0,
    }])

    out, _ = attach_embeddings(df, export, "symbol", "date", lag=2)
    col = "{}_0".format(FEATURE_PREFIX)

    assert (out.loc[out["symbol"] == "ZZZ", col] == 0.0).all()
    assert (out.loc[(out["symbol"] == "AAA") & (out["date"] > dates[7]), col] == 4.0).all()


def test_attach_embeddings_honours_the_dims_cap():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame({"symbol": ["AAA"] * 20, "date": dates})
    export = _export_frame(blocks=(0,), seeds=(11,), dim=5, symbols=("AAA",))
    export["train_end"] = dates[0]

    out, _ = attach_embeddings(df, export, "symbol", "date", lag=1, dims=2)
    made = [c for c in out.columns if c.startswith(FEATURE_PREFIX)]
    assert made == ["{}_0".format(FEATURE_PREFIX), "{}_1".format(FEATURE_PREFIX)]


def test_exported_features_use_the_loader_discovery_prefix():
    """
    The crypto loader finds features by the 'feature_' prefix, which is what lets a
    built dataset work with no modelling changes at all.
    """
    assert FEATURE_PREFIX.startswith("feature_")
