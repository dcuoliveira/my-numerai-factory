"""
Tests for the bottleneck-autoencoder feature builder.

Same philosophy as test_nn.py: these cover the mistakes that produce
plausible-looking numbers rather than crashes. A learned feature block has its
own specific set of those --

    * an encoder fitted on rows it should not have seen, which shows up as a
      backtest that cannot be reproduced live and never as an error
    * a code column that is not on [0, 1] with 0.5 neutral, so the pipeline's
      missing-value fill quietly asserts an extreme reading
    * a rank taken across the panel instead of within a date, which smuggles
      cross-sectional level into a within-date signal
    * a feature matrix handed to the encoder in a different column order than it
      was fitted on -- a dense matmul, so every code is wrong with no NaN, no
      exception and a perfectly plausible distribution
    * a code that is constant, or a duplicate of another code, shipped as a
      column anyway

Everything here runs offline on a synthetic panel and takes seconds. torch is an
optional dependency, so the whole module skips cleanly without it.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.data_mani import build_autoencoder as builder  # noqa: E402
from src.prediction.nn.autoencoder import (  # noqa: E402
    FeatureAutoencoder,
    apply_swap_noise,
    encode_values,
    iter_batches,
    steps_per_epoch,
    to_tensor,
    train_autoencoder,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

#: The five bins every Crypto feature takes, so the synthetic panel has the same
#: shape as the real one -- rank-binned within each date.
BINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

TARGET = "target_binned_return_20"


def make_panel(n_dates=80, n_symbols=40, n_features=10, n_factors=4, seed=0):
    """
    A synthetic panel with real latent structure, rank-binned within each date.

    Structure matters here in a way it does not for the LSTM tests: an
    autoencoder over pure noise has nothing to compress, so its held-out R^2
    would be zero and the builder would correctly refuse to write the dataset.
    Features are built from `n_factors` shared factors so a bottleneck of that
    width can actually reconstruct them.

    :return: frame with symbol, date, feature_*, and the target
    :rtype: pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        np.arange(n_dates), unit="D")

    rows = []
    loadings = rng.normal(size=(n_factors, n_features))

    for date in dates:
        factors = rng.normal(size=(n_symbols, n_factors))
        raw = factors @ loadings + 0.3 * rng.normal(size=(n_symbols, n_features))

        # Rank-bin within the date, exactly as the shipped features are.
        binned = np.empty_like(raw)
        for column in range(n_features):
            order = pd.Series(raw[:, column]).rank(method="first").to_numpy()
            bucket = ((order - 1) * len(BINS) // n_symbols).astype("int64")
            binned[:, column] = BINS[np.clip(bucket, 0, len(BINS) - 1)]

        frame = pd.DataFrame(
            binned, columns=["feature_f%02d" % i for i in range(n_features)])
        frame.insert(0, "date", date)
        frame.insert(0, "symbol", ["SYM%03d" % s for s in range(n_symbols)])
        frame[TARGET] = BINS[rng.integers(0, len(BINS), size=n_symbols)]
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def write_source(root, panel, version="crypto/vtest"):
    """
    Put a panel on disk where the builder's cache lookup will find it.

    :return: the version string to pass as --source-version
    :rtype: str
    """
    directory = os.path.join(str(root), "crypto", version.split("/")[-1])
    os.makedirs(directory, exist_ok=True)
    panel.set_index("symbol").to_parquet(
        os.path.join(directory, "train.parquet"))
    # Every builder copies this forward so BMC stays comparable across arms.
    pd.DataFrame({"date": panel["date"].unique(), "prediction": 0.5}).to_parquet(
        os.path.join(directory, "historical_meta_models.parquet"))
    return version


def build(tmp_path, monkeypatch, panel=None, **flags):
    """
    Run the builder end to end against a temporary data root.

    :return: (written frame, encoder metadata)
    :rtype: (pd.DataFrame, dict)
    """
    monkeypatch.setattr("src.data_mani.utils.DATA_DIR", str(tmp_path))
    # No network: the source is already cached, so the api is never touched.
    monkeypatch.setattr(builder, "get_api", lambda tournament: None)

    source = write_source(tmp_path, make_panel() if panel is None else panel)

    # A higher lr and a wider epoch budget than the build defaults, because
    # 2400 synthetic rows at batch 512 is five gradient steps an epoch: at the
    # real defaults the codes are still degenerate when the run ends, and the
    # duplicate-code refusal fires on an undertrained network rather than on
    # anything these tests are about.
    argv = ["--source-version", source, "--code-dim", "3", "--epochs", "150",
            "--patience", "20", "--batch-size", "512", "--lr", "0.01",
            "--seed", "42", "-q"]
    for name, value in flags.items():
        argv.append("--%s" % name.replace("_", "-"))
        if value is not None:
            argv.append(str(value))

    builder.main(argv)

    version = source + builder.default_suffix(
        flags.get("method", "ae"), flags.get("fit_scope", "train"))
    written = pd.read_parquet(
        builder.local_path("crypto", version, "train.parquet")).reset_index()
    with open(builder.encoder_meta_path(version)) as handle:
        meta = json.load(handle)
    return written, meta


# --------------------------------------------------------------------------- #
# What the encoder is allowed to see
# --------------------------------------------------------------------------- #

def test_the_encoder_never_sees_a_period_at_or_after_its_fit_boundary(
        tmp_path, monkeypatch):
    """
    The behavioural version of the leakage claim, not the arithmetic one.

    Scramble every feature value after the boundary and rebuild. If the encoder
    had seen those rows the weights would move; identical weight hashes are the
    only evidence that matters.
    """
    panel = make_panel()
    _, clean = build(tmp_path / "a", monkeypatch, panel=panel)

    boundary = pd.Timestamp(clean["fit_boundary_period"])
    scrambled = panel.copy()
    after = scrambled["date"] > boundary
    features = [c for c in scrambled.columns if c.startswith("feature_")]
    rng = np.random.default_rng(7)
    scrambled.loc[after, features] = BINS[
        rng.integers(0, len(BINS), size=(int(after.sum()), len(features)))]

    _, dirty = build(tmp_path / "b", monkeypatch, panel=scrambled)

    assert clean["weights_sha1"] == dirty["weights_sha1"]
    assert clean["fit_boundary_period"] == dirty["fit_boundary_period"]


def test_the_fit_boundary_matches_the_pipelines_own_split_arithmetic():
    periods = np.arange(100)

    # 30 validation periods, 20 embargoed, so 50 remain to fit on.
    assert builder.fit_boundary(periods, 0.3, 20, "train") == 50
    # The lookahead upper bound takes everything, by construction.
    assert builder.fit_boundary(periods, 0.3, 20, "all") == 100

    with pytest.raises(ValueError, match="no fit periods"):
        builder.fit_boundary(periods, 0.9, 20, "train")


def test_the_lookahead_upper_bound_is_fitted_on_every_period(
        tmp_path, monkeypatch):
    _, meta = build(tmp_path, monkeypatch, fit_scope="all")
    assert meta["fit_scope"] == "all"
    assert meta["fit_periods"] == 80


# --------------------------------------------------------------------------- #
# Coverage and scale
# --------------------------------------------------------------------------- #

def test_every_source_row_gets_a_code_and_no_row_is_dropped(
        tmp_path, monkeypatch):
    """
    A forecaster may score only the rows it covered; a dataset builder may not.

    An uncovered row would be filled with the neutral 0.5 downstream, which
    means "no view" -- and the rows most at risk are the young, sparse tokens
    whose 60-day indicators are already missing.
    """
    panel = make_panel()
    written, meta = build(tmp_path, monkeypatch, panel=panel)

    assert len(written) == len(panel)
    assert not written[meta["code_columns"]].isna().to_numpy().any()


def test_codes_land_strictly_inside_the_unit_interval_and_average_exactly_one_half(
        tmp_path, monkeypatch):
    written, meta = build(tmp_path, monkeypatch)
    codes = written[meta["code_columns"]]

    assert codes.to_numpy().min() > 0.0
    assert codes.to_numpy().max() < 1.0

    # (rank - 0.5) / n is exactly centred, unlike rank(pct=True) whose mean is
    # (n + 1) / 2n and whose top row is exactly 1.0.
    per_date = written.groupby("date")[meta["code_columns"]].mean()
    np.testing.assert_allclose(per_date.to_numpy(), 0.5, atol=1e-6)


def test_the_rank_is_taken_within_a_date_not_across_the_panel():
    """Two dates on disjoint activation ranges must each span (0, 1)."""
    periods = pd.Series(["a", "a", "a", "b", "b", "b"])
    values = np.array([[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]])

    ranked = builder.rank_within_period(values, periods)

    np.testing.assert_allclose(ranked[:3, 0], [1 / 6, 3 / 6, 5 / 6])
    np.testing.assert_allclose(ranked[3:, 0], [1 / 6, 3 / 6, 5 / 6])


def test_tied_activations_get_the_same_code_whatever_the_row_order():
    """
    The bug this pins produced a plausible number and no error.

    The raw features live on a five-point grid, so rows sharing a whole feature
    vector get bit-identical activations -- about 8 of 241 on a typical date.
    Under method="first" those ties were broken by parquet row order, so the
    SAME encoder emitted different codes for the same row depending on which
    dataset it was attached to, which quietly breaks any arm table that assumes
    the added columns mean one thing.
    """
    periods = pd.Series(["d"] * 6)
    values = np.array([[1.0], [2.0], [2.0], [2.0], [3.0], [4.0]])

    ranked = builder.rank_within_period(values, periods)

    # The three tied rows must share a value...
    assert ranked[1, 0] == ranked[2, 0] == ranked[3, 0]
    # ...and reordering the rows must not move any of them.
    order = [4, 0, 3, 1, 5, 2]
    reordered = builder.rank_within_period(values[order], periods)
    np.testing.assert_allclose(reordered[:, 0], ranked[order, 0])

    # And the column is still exactly centred, which is what 0.5 rests on.
    assert ranked[:, 0].mean() == pytest.approx(0.5)


def test_the_written_columns_are_the_raw_features_plus_exactly_k_codes(
        tmp_path, monkeypatch):
    written, meta = build(tmp_path, monkeypatch)

    emitted = [c for c in written.columns if c.startswith("feature_")]
    assert len(emitted) == meta["n_features"] + meta["code_dim"]
    assert meta["code_columns"] == ["feature_ae_00", "feature_ae_01",
                                    "feature_ae_02"]


# --------------------------------------------------------------------------- #
# Reproducibility and the column-order trap
# --------------------------------------------------------------------------- #

def test_the_same_seed_reproduces_the_same_codes(tmp_path, monkeypatch):
    panel = make_panel()
    first, meta = build(tmp_path / "a", monkeypatch, panel=panel)
    second, other = build(tmp_path / "b", monkeypatch, panel=panel)

    assert meta["weights_sha1"] == other["weights_sha1"]
    np.testing.assert_allclose(
        first[meta["code_columns"]].to_numpy(),
        second[meta["code_columns"]].to_numpy())


def test_the_saved_encoder_reproduces_its_codes_after_a_round_trip(
        tmp_path, monkeypatch):
    """The live path's whole guarantee: reload and get the same columns back."""
    written, meta = build(tmp_path, monkeypatch)

    _, encode = builder.read_encoder(meta["out_version"])
    reranked = builder.rank_within_period(encode(written), written["date"])

    np.testing.assert_allclose(
        reranked, written[meta["code_columns"]].to_numpy(), atol=1e-6)


def test_encoding_reindexes_to_the_saved_column_order(tmp_path, monkeypatch):
    """
    The sharpest live/backtest trap in the design.

    build_derived takes feature columns in parquet order; live/features.py
    returns them sorted. For a dense first layer a permuted input is silently
    wrong, so the codes must be IDENTICAL under permutation, not merely finite.
    """
    written, meta = build(tmp_path, monkeypatch)
    _, encode = builder.read_encoder(meta["out_version"])

    rng = np.random.default_rng(3)
    permuted = written[list(rng.permutation(written.columns))]

    np.testing.assert_array_equal(encode(permuted), encode(written))


def test_encoding_refuses_a_frame_missing_a_saved_column(tmp_path, monkeypatch):
    written, meta = build(tmp_path, monkeypatch)
    _, encode = builder.read_encoder(meta["out_version"])

    dropped = meta["feature_columns"][2]
    with pytest.raises(ValueError, match=dropped):
        encode(written.drop(columns=[dropped]))


# --------------------------------------------------------------------------- #
# The network and its loop
# --------------------------------------------------------------------------- #

def test_a_code_dim_at_or_above_the_feature_count_is_refused():
    with pytest.raises(ValueError, match="not a bottleneck"):
        FeatureAutoencoder(input_dim=10, code_dim=10)
    with pytest.raises(ValueError, match="not a bottleneck"):
        FeatureAutoencoder(input_dim=10, code_dim=11)


def test_the_decoder_output_stays_inside_the_unit_interval():
    """The reconstruction target is a rank bin, so an unbounded head is wrong."""
    net = FeatureAutoencoder(input_dim=10, code_dim=3, dropout=0.0)
    reconstruction, code = net(torch.rand(64, 10))

    assert reconstruction.min() >= 0.0 and reconstruction.max() <= 1.0
    assert code.shape == (64, 3)


def test_swap_noise_draws_donors_from_the_same_column_only():
    """
    Every corrupted cell must hold a value its own feature could take.

    Columns here are given disjoint value ranges, so a cross-column donor would
    be unmistakable.
    """
    values = torch.stack([torch.full((50,), float(c)) for c in range(4)], dim=1)
    generator = torch.Generator()
    generator.manual_seed(0)

    noisy = apply_swap_noise(values, 0.9, generator)

    for column in range(values.shape[1]):
        assert torch.all(noisy[:, column] == float(column))


def test_swap_noise_is_a_no_op_when_disabled():
    values = torch.rand(20, 5)
    generator = torch.Generator()
    generator.manual_seed(0)

    assert torch.equal(apply_swap_noise(values, 0.0, generator), values)


def test_early_stopping_restores_the_best_epoch_not_the_last():
    """
    The same pin as the LSTM path's, for the same reason.

    state_dict() returns live references, so a snapshot that is not cloned is
    the weights *now* rather than the weights at capture -- early stopping that
    selects an epoch and then throws it away.
    """
    rng = np.random.default_rng(0)
    fit = rng.random((256, 8)).astype("float32")
    stop = rng.random((64, 8)).astype("float32")

    config = {"code_dim": 3, "hidden": (8,), "dropout": 0.0, "lr": 0.01,
              "epochs": 12, "patience": 3, "batch_size": 32, "swap_noise": 0.0,
              "max_grad_norm": 1.0, "seed": 42}
    result = train_autoencoder(fit, stop, config, torch.device("cpu"),
                               verbose=False)

    assert result["best_epoch"] <= result["epochs_run"]
    assert result["best_val_mse"] == pytest.approx(min(result["val_history"]))

    # The returned model IS the best state, not whatever the last epoch left.
    for key, value in result["model"].state_dict().items():
        assert torch.allclose(value.cpu(), result["best_state"][key])


def test_a_run_that_ends_at_the_epoch_cap_is_flagged_as_truncated():
    """
    A truncated network's R^2 understates the architecture, and comparing it
    with a control that did converge quietly answers the wrong question. The
    real build hit exactly this: still improving at epoch 199 of 200, and it
    lost to its own PCA control.
    """
    rng = np.random.default_rng(0)
    fit = rng.random((256, 8)).astype("float32")
    stop = rng.random((64, 8)).astype("float32")
    base = {"code_dim": 3, "hidden": (8,), "dropout": 0.0, "lr": 0.01,
            "batch_size": 32, "swap_noise": 0.0, "max_grad_norm": 1.0,
            "seed": 42}

    # Patience it cannot exhaust, so only the ceiling can end it.
    capped = train_autoencoder(fit, stop, dict(base, epochs=3, patience=999),
                               torch.device("cpu"), verbose=False)
    assert capped["hit_epoch_cap"]
    assert capped["epochs_run"] == 3

    # Patience of zero stops it on the first non-improving epoch instead.
    stopped = train_autoencoder(fit, stop, dict(base, epochs=500, patience=1),
                                torch.device("cpu"), verbose=False)
    assert not stopped["hit_epoch_cap"]
    assert stopped["epochs_run"] < 500


def test_training_without_a_held_out_slice_reports_nan_not_infinity():
    """+inf is the loop's sentinel; reporting it as a score would be a lie."""
    rng = np.random.default_rng(1)
    config = {"code_dim": 2, "hidden": (8,), "dropout": 0.0, "lr": 0.01,
              "epochs": 3, "patience": 2, "batch_size": 32, "swap_noise": 0.0,
              "max_grad_norm": 1.0, "seed": 42}

    result = train_autoencoder(rng.random((64, 6)).astype("float32"), None,
                               config, torch.device("cpu"), verbose=False)

    assert np.isnan(result["best_val_mse"])
    assert not result["early_stopped"]


def test_encode_values_preserves_row_order():
    net = FeatureAutoencoder(input_dim=6, code_dim=2, dropout=0.0)
    values = np.random.default_rng(2).random((37, 6)).astype("float32")

    codes = encode_values(net, values, torch.device("cpu"), batch_size=8)

    assert codes.shape == (37, 2)
    np.testing.assert_allclose(
        codes[5:9],
        encode_values(net, values[5:9], torch.device("cpu"), batch_size=8),
        atol=1e-6)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #

def test_every_row_is_seen_exactly_once_per_pass():
    """
    The property TensorDataset gave for free and hand-rolled batching can lose.

    A shuffle that sampled with replacement, or an off-by-one in the final
    ragged batch, would train on a quietly different dataset than the one that
    was passed in -- and would still produce a perfectly plausible loss curve.
    """
    tensor = to_tensor(np.arange(101 * 3).reshape(101, 3), torch.device("cpu"))
    generator = torch.Generator()
    generator.manual_seed(0)

    for shuffle in (False, True):
        seen = torch.cat([
            batch[:, 0] for batch in
            iter_batches(tensor, 16, shuffle, generator)
        ])
        assert len(seen) == 101
        assert torch.equal(torch.sort(seen).values, tensor[:, 0])


def test_steps_per_epoch_matches_the_batches_actually_yielded():
    """The tuner converts a step budget into epochs with this; if it disagrees
    with the loop, every trial silently gets the wrong amount of training."""
    tensor = to_tensor(np.zeros((1000, 4)), torch.device("cpu"))

    for batch_size in (1, 7, 256, 1000, 4096):
        counted = sum(1 for _ in iter_batches(tensor, batch_size, shuffle=False))
        assert counted == steps_per_epoch(1000, batch_size), batch_size


def test_batching_is_deterministic_for_a_fixed_generator():
    tensor = to_tensor(np.arange(50 * 2).reshape(50, 2), torch.device("cpu"))

    def pass_over(seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.cat([b[:, 0] for b in iter_batches(tensor, 8, True, generator)])

    assert torch.equal(pass_over(0), pass_over(0))
    assert not torch.equal(pass_over(0), pass_over(1))


def test_a_batch_larger_than_the_slice_is_clamped_not_dropped():
    tensor = to_tensor(np.zeros((5, 2)), torch.device("cpu"))
    batches = list(iter_batches(tensor, 4096, shuffle=False))

    assert len(batches) == 1
    assert batches[0].shape == (5, 2)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

def test_a_dead_code_is_refused_rather_than_shipped():
    diagnostics = {"code_std_raw": [0.4, 1e-9, 0.3],
                   "max_code_code_corr": 0.2, "max_code_code_pair": [0, 1]}

    with pytest.raises(SystemExit, match="code 1 is dead"):
        builder.check_codes(diagnostics, allow_dead=False)

    # ...unless the escape hatch is asked for explicitly.
    builder.check_codes(diagnostics, allow_dead=True)


def test_two_codes_that_are_one_code_are_refused():
    diagnostics = {"code_std_raw": [0.4, 0.3],
                   "max_code_code_corr": 0.997, "max_code_code_pair": [0, 1]}

    with pytest.raises(SystemExit, match="they are one column"):
        builder.check_codes(diagnostics, allow_dead=False)


# --------------------------------------------------------------------------- #
# Naming, and the control arm
# --------------------------------------------------------------------------- #

def test_code_names_match_none_of_the_ablation_block_regexes():
    """
    Pins section 6 of the design against a future rename.

    These are the five patterns from experiments/ablation_crypto.sh. A code name
    that matched one would be silently swept into another block's ablation --
    and note the last one is why the prefix must carry no bare "x".
    """
    patterns = [r"^feature_sym_(?!tenc$)", r"^feature_sym_tenc$",
                r"_rm[0-9]+$", r"_spread$", r"_x_|_accel_"]

    for name in builder.code_columns("feature_ae", 12):
        for pattern in patterns:
            assert not re.search(pattern, name), (name, pattern)


def test_code_names_are_zero_padded_so_they_sort_in_code_order():
    """live/features.py sorts columns lexically; ae_10 must not precede ae_2."""
    names = builder.code_columns("feature_ae", 12)
    assert names == sorted(names)


def test_the_pca_control_and_the_autoencoder_emit_the_same_column_names_and_shape(
        tmp_path, monkeypatch):
    """Otherwise the control arm is not comparable with the arm it controls."""
    panel = make_panel()
    neural, neural_meta = build(tmp_path / "ae", monkeypatch, panel=panel)
    linear, linear_meta = build(tmp_path / "pca", monkeypatch, panel=panel,
                                method="pca")

    assert neural_meta["code_columns"] == linear_meta["code_columns"]
    assert neural.shape == linear.shape
    assert linear_meta["method"] == "pca"
    # The control must be loadable without torch having built it.
    assert "weights_sha1" not in linear_meta


def test_each_arm_writes_to_its_own_version_so_none_overwrites_another():
    assert builder.default_suffix("ae", "train") == "-ae"
    assert builder.default_suffix("pca", "train") == "-aepca"
    assert builder.default_suffix("ae", "all") == "-aeglobal"
    assert builder.default_suffix("pca", "all") == "-aepcaglobal"
