"""
Tests for the symbol-embedding fitter.

The failure this file mostly guards against is a silent boundary drift. --val-frac
is a run-time flag, so an embedding fitted at one value and an autoencoder fitted
at another sit on DIFFERENT train samples while both artifacts still describe
themselves as "the train block" -- and the only symptom is a slightly better
number.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.data_mani.fit_symbol_embedding import (
    check_boundary,
    rank_split,
    train_block_periods,
)

CRYPTO_TRAIN = "data/crypto/v2.0/train.parquet"


# --------------------------------------------------------------------------- #
# train_block_periods -- must BE make_validation_split's train block
# --------------------------------------------------------------------------- #

def test_train_block_is_read_off_the_split_not_recomputed():
    """
    REGRESSION TEST for a bug --verify-boundary caught.

    The loaded frame has already had the embargo periods removed, so applying the
    embargo again moved the boundary 20 periods early (2024-06-06 against the
    encoder's 2024-06-26) while every message still said "the train block". Deriving
    from the `split` column cannot drift, because it is the same object the splitter
    produced.
    """
    from src.data_mani.utils import make_validation_split

    periods = pd.date_range("2020-01-01", periods=200, freq="D")
    frame = pd.DataFrame({"date": list(periods) * 3, "x": 1.0})
    split = make_validation_split(frame, "date", val_frac=0.3, embargo=20)

    ours = train_block_periods(split, "date")
    theirs = np.sort(split.loc[split["split"] == "train", "date"].unique())

    assert len(ours) == len(theirs)
    assert pd.Timestamp(ours[-1]) == pd.Timestamp(theirs[-1])
    # And the embargo is NOT applied twice: the loaded frame no longer contains it.
    assert len(split["date"].unique()) == 200 - 20


def test_train_block_refuses_an_empty_split():
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5),
                          "split": ["validation"] * 5})
    with pytest.raises(ValueError, match="train split has"):
        train_block_periods(frame, "date")


# --------------------------------------------------------------------------- #
# rank_split -- the embargo is the difference from the autoencoder
# --------------------------------------------------------------------------- #

def test_rank_split_leaves_an_embargo_gap():
    """
    The autoencoder's stop slice starts the period after its fit slice ends, because
    reconstruction involves no target. This loss DOES involve the target, so a gap
    is mandatory -- without it a 20-day forward target from the last trained period
    reaches into the slice used to rank seeds.
    """
    fit = np.arange(100)
    inner, rank = rank_split(fit, rank_frac=0.15, embargo=20)

    assert len(rank) == 15
    assert list(rank) == list(range(85, 100))
    assert len(inner) == 65                       # 100 - 15 - 20
    assert rank.min() - inner.max() > 20 - 1, "the embargo gap is too small"


def test_rank_split_with_zero_frac_trains_on_everything():
    fit = np.arange(100)
    inner, rank = rank_split(fit, rank_frac=0.0, embargo=20)
    assert len(rank) == 0
    assert len(inner) == 100


def test_rank_split_refuses_to_leave_nothing_to_train_on():
    with pytest.raises(ValueError, match="no periods to train on"):
        rank_split(np.arange(25), rank_frac=0.9, embargo=20)


# --------------------------------------------------------------------------- #
# check_boundary
# --------------------------------------------------------------------------- #

def test_check_boundary_accepts_a_match(tmp_path, capsys):
    path = tmp_path / "encoder.json"
    path.write_text(json.dumps({"fit_boundary_period": "2024-06-26",
                                "val_frac": 0.3, "embargo": 20}))
    check_boundary(str(path), "2024-06-26")
    assert "boundary matches" in capsys.readouterr().out


def test_check_boundary_rejects_a_mismatch_and_names_both_dates(tmp_path):
    path = tmp_path / "encoder.json"
    path.write_text(json.dumps({"fit_boundary_period": "2024-06-26",
                                "val_frac": 0.3, "embargo": 20}))
    with pytest.raises(SystemExit) as excinfo:
        check_boundary(str(path), "2025-01-15")

    message = str(excinfo.value)
    assert "2024-06-26" in message and "2025-01-15" in message
    assert "0.3" in message, "the mismatch must name the encoder's val_frac"


# --------------------------------------------------------------------------- #
# Parity with the real autoencoder, when the data and an encoder are present
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(CRYPTO_TRAIN),
                    reason="crypto dataset not downloaded")
def test_fit_block_reproduces_the_autoencoders_recorded_boundary():
    """
    The claim this module rests on: 'the same train sample as the autoencoder'.
    Checked against whatever encoder.json is actually on disk, so it keeps holding
    if either side's defaults ever move.
    """
    encoders = [
        os.path.join(root, "encoder.json")
        for root, _, files in os.walk("data/crypto")
        if "encoder.json" in files
    ]
    if not encoders:
        pytest.skip("no encoder.json built yet")

    dates = pd.read_parquet(CRYPTO_TRAIN, columns=["date"])["date"]
    periods = np.sort(dates.unique())

    checked = 0
    for path in encoders:
        meta = json.load(open(path))
        if meta.get("fit_scope") != "train":
            continue
        frame = pd.DataFrame({"date": dates})
        from src.data_mani.utils import make_validation_split
        split = make_validation_split(frame, "date", meta["val_frac"], meta["embargo"])
        ours = train_block_periods(split, "date")
        assert str(pd.Timestamp(ours[-1]).date()) == meta["fit_boundary_period"], (
            "boundary drifted against {}".format(path)
        )
        assert len(ours) == meta["fit_periods"]
        checked += 1

    assert checked, "no train-scope encoder to compare against"
