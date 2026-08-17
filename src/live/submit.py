"""
Turn a cached model into a submitted prediction file, with a fallback chain.

Nothing here refits. By the time this runs the estimator either exists or it
does not, and the job is to get *something* defensible uploaded before the round
closes. A missed submission scores nothing and, on a staked slot, is strictly
worse than a mediocre one.

------------------------------------------------------------------------------
Predictions are rank-normalised before upload
------------------------------------------------------------------------------

Numerai scores cross-sectionally by rank, so mapping the raw output onto evenly
spaced values in (0, 1) changes no score. It does two useful things: it
guarantees the file passes range validation whatever the estimator emitted, and
it makes the submitted file comparable across slots that produce quite different
raw scales (a fixed-alpha Ridge and a glmnet-path Ridge do not share units).

Ties are broken by "first" rather than averaged, so the output is a strict
permutation -- a block of identical values would otherwise submit as a block of
identical ranks and lose the ordering the model did express elsewhere.

------------------------------------------------------------------------------
The fallback chain
------------------------------------------------------------------------------

Tried in order, first success wins, and whatever succeeded is recorded so the
round email says plainly which slot ran a stand-in:

  1. own model      the cached estimator on its own feature set
  2. sibling        CURUPIRA_02's estimator on the raw 22 features. It is the
                    fallback for everything because it is the only slot with no
                    build step: no encoding, no vocabulary, no rolling history.
                    If the raw features loaded at all, this can run.
  3. previous round the last CSV this slot successfully submitted. A one-round-
                    stale signal on a 20-day forward target is still most of a
                    signal -- consecutive dates overlap 95%.
  4. example preds  Numerai's own live_example_preds. Scores near zero, but it
                    is a valid submission and keeps the slot's streak alive.

Only the last is really a surrender, and it is still better than nothing.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

from src.live import registry
from src.live.features import PERIOD, SYMBOL, raw_features

SUBMISSION_DIR = "results/live"
EXAMPLE_PREDS = "data/crypto/live/live_example_preds.csv"

# Upload retry policy. The API occasionally 502s or times out under round-open
# load; these are transient and worth waiting out. 5 attempts over ~75s total.
MAX_UPLOAD_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 5


def rank_normalise(values):
    """Map any real-valued vector onto evenly spaced points in (0, 1)."""
    ranks = pd.Series(values).rank(method="first").to_numpy()
    return (ranks - 0.5) / len(ranks)


def predict(estimator, panel, columns, live_date, task="regression"):
    """
    Predict today's rows and return a submission frame.

    :return: DataFrame with columns [symbol, prediction], one row per live symbol
    """
    live = panel[panel["is_live"] & (panel[PERIOD] == live_date)]
    if live.empty:
        raise ValueError("no live rows at %s" % live_date.date())

    X = live[columns]
    if task == "classification":
        raw = estimator.predict_proba(X)[:, 1]
    else:
        raw = estimator.predict(X)

    if not np.isfinite(raw).all():
        raise ValueError("estimator produced %d non-finite predictions"
                         % int((~np.isfinite(raw)).sum()))

    return pd.DataFrame({SYMBOL: live[SYMBOL].to_numpy(),
                         "prediction": rank_normalise(raw)})


def submission_path(round_number, name):
    directory = os.path.join(SUBMISSION_DIR, str(round_number))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "%s.csv" % name)


def previous_submission(name, round_number):
    """Most recent successfully written CSV for this slot, from an older round."""
    candidates = []
    for path in glob.glob(os.path.join(SUBMISSION_DIR, "*", "%s.csv" % name)):
        try:
            other = int(os.path.basename(os.path.dirname(path)))
        except ValueError:
            continue
        if other < round_number:
            candidates.append((other, path))
    if not candidates:
        return None
    return max(candidates)[1]


def load_example_predictions(live_symbols):
    """Numerai's own example predictions, restricted to today's universe."""
    if not os.path.exists(EXAMPLE_PREDS):
        raise FileNotFoundError(EXAMPLE_PREDS)
    frame = pd.read_csv(EXAMPLE_PREDS)
    frame = frame[frame[SYMBOL].isin(set(live_symbols))]
    return frame[[SYMBOL, "prediction"]]


def build_predictions(name, context):
    """
    Run the fallback chain for one slot.

    `context` carries everything the chain might need, assembled once per round
    by the orchestrator:
        estimators   {slot: (estimator, columns, panel, metadata)}
        live_date, round_number, live_symbols

    :return: (frame, source, note) where source is one of
             "model" | "sibling" | "previous_round" | "example"
    """
    spec = registry.get(name)
    errors = []

    entry = context["estimators"].get(name)
    if entry is not None:
        estimator, columns, panel, _ = entry
        try:
            frame = predict(estimator, panel, columns, context["live_date"],
                            spec["fit"]["task"])
            return frame, "model", None
        except Exception as error:
            errors.append("own model: %s" % error)

    sibling = context["estimators"].get("CURUPIRA_02")
    if sibling is not None and name != "CURUPIRA_02":
        estimator, columns, panel, _ = sibling
        try:
            frame = predict(estimator, panel, columns, context["live_date"])
            return frame, "sibling", ("ran CURUPIRA_02 (raw 22 features) as a "
                                      "stand-in; %s" % "; ".join(errors))
        except Exception as error:
            errors.append("sibling: %s" % error)

    previous = previous_submission(name, context["round_number"])
    if previous:
        try:
            frame = pd.read_csv(previous)[[SYMBOL, "prediction"]]
            frame = frame[frame[SYMBOL].isin(set(context["live_symbols"]))]
            if len(frame):
                return frame, "previous_round", (
                    "reused %s; %s" % (previous, "; ".join(errors)))
        except Exception as error:
            errors.append("previous round: %s" % error)

    try:
        frame = load_example_predictions(context["live_symbols"])
        return frame, "example", ("fell all the way through to Numerai's example "
                                  "predictions; %s" % "; ".join(errors))
    except Exception as error:
        errors.append("example: %s" % error)

    raise RuntimeError("every fallback failed for %s -- %s" % (name, "; ".join(errors)))


def write_csv(frame, round_number, name):
    """Write the submission file and sanity-check it before it leaves the box."""
    if frame["prediction"].isna().any():
        raise ValueError("%s: submission contains NaN predictions" % name)
    if frame[SYMBOL].duplicated().any():
        raise ValueError("%s: submission contains duplicate symbols" % name)
    if not ((frame["prediction"] > 0) & (frame["prediction"] < 1)).all():
        raise ValueError("%s: predictions outside (0, 1)" % name)

    path = submission_path(round_number, name)
    frame[[SYMBOL, "prediction"]].to_csv(path, index=False)
    return path


def upload(api, path, model_id, name, dry_run=False, verbose=True):
    """
    Upload with exponential backoff.

    Returns (submission_id, attempts). A dry run returns ("DRY_RUN", 0) without
    touching the network, which is what lets the whole pipeline be exercised on
    a closed round.
    """
    if dry_run:
        return "DRY_RUN", 0

    last_error = None
    for attempt in range(1, MAX_UPLOAD_ATTEMPTS + 1):
        try:
            submission_id = api.upload_predictions(file_path=path, model_id=model_id)
            return submission_id, attempt
        except Exception as error:
            last_error = error
            if attempt == MAX_UPLOAD_ATTEMPTS:
                break
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            if verbose:
                print("   %s upload attempt %d/%d failed (%s), retrying in %ds"
                      % (name, attempt, MAX_UPLOAD_ATTEMPTS, error, delay))
            time.sleep(delay)

    raise RuntimeError("%s: upload failed after %d attempts -- %s"
                       % (name, MAX_UPLOAD_ATTEMPTS, last_error))
