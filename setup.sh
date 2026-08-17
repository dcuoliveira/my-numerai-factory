#!/usr/bin/env bash
#
# Bootstrap this repository on a fresh machine.
#
#   bash setup.sh                 # deps + base crypto dataset + verify
#   bash setup.sh --with-derived  # also build the 812-feature chain (~3 min)
#   bash setup.sh --all           # everything, including numerai Classic data
#   bash setup.sh --check         # verify an existing install, change nothing
#
# What this does, in order:
#
#   1. check python, pip and a writable tree
#   2. install the python packages src/ actually imports
#   3. download the crypto dataset (train + historical meta models, ~5 MB)
#   4. optionally build the derived feature datasets the experiments use
#   5. write a .env template for live submission credentials (never overwritten)
#   6. run the test suite and one tiny end-to-end forecast, to prove it works
#
# Nothing here needs Numerai credentials. The tournament DATA is public and
# downloads anonymously; credentials are only needed to SUBMIT, which is
# src/live/ and out of scope for a first run.
#
# Everything is idempotent -- re-running skips what is already present -- and
# the downloads and builds land in data/, which is gitignored.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

WITH_DERIVED=0
WITH_NUMERAI=0
CHECK_ONLY=0
SKIP_TESTS="${SKIP_TESTS:-0}"
PYTHON="${PYTHON:-python3}"
PIP_ARGS="${PIP_ARGS:-}"

for arg in "$@"; do
    case "$arg" in
        --with-derived) WITH_DERIVED=1 ;;
        --with-numerai) WITH_NUMERAI=1 ;;
        --all)          WITH_DERIVED=1; WITH_NUMERAI=1 ;;
        --check)        CHECK_ONLY=1 ;;
        --skip-tests)   SKIP_TESTS=1 ;;
        -h|--help)      sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown option '$arg' (try --help)" >&2; exit 2 ;;
    esac
done

step()  { echo; echo "=============================================================="; \
          echo "$*"; echo "=============================================================="; }
ok()    { printf '  ok    %s\n' "$*"; }
warn()  { printf '  WARN  %s\n' "$*"; }
fail()  { printf '  FAIL  %s\n' "$*" >&2; exit 1; }

started=$(date +%s)

# --------------------------------------------------------------------------- #
step "1/6  environment"
# --------------------------------------------------------------------------- #

command -v "$PYTHON" >/dev/null 2>&1 || fail "no '$PYTHON' on PATH. Set PYTHON=/path/to/python3"

# 3.9 is the floor: src/ uses dict-ordered kwargs, f-strings and
# functools.cache-era idioms, and numerapi itself no longer supports 3.8.
"$PYTHON" - <<'PY' || exit 1
import sys
v = sys.version_info
print("  ok    python %d.%d.%d at %s" % (v.major, v.minor, v.micro, sys.executable))
if (v.major, v.minor) < (3, 9):
    sys.exit("  FAIL  python >= 3.9 required, found %d.%d" % (v.major, v.minor))
PY

[ -w "$REPO" ] || fail "$REPO is not writable"
ok "repo at $REPO"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    warn "no virtualenv or conda env active -- packages go to the system python."
    warn "consider:  python3 -m venv .venv && source .venv/bin/activate"
fi

# --------------------------------------------------------------------------- #
step "2/6  python packages"
# --------------------------------------------------------------------------- #
#
# Derived from what src/ actually imports, not from a wish list. Split into
# required and optional so a failure in an optional wheel does not block a
# working install.
#
#   numerapi        dataset download + submission (CryptoAPI / NumerAPI)
#   numerai-tools   the official numerai_corr, so scores match the leaderboard
#   pandas pyarrow  parquet datasets
#   numpy scipy     numerics, and the scipy distributions in the search grids
#   scikit-learn    Ridge/LogisticRegression, RandomizedSearchCV, the CV machinery
#   lightgbm        the lgb model wrapper
#   joblib          model persistence for the live path
#   matplotlib      the analysis plots
#   pytest          src/tests/{test_weighting,test_live}.py use pytest.raises and
#                   pytest.mark.parametrize, so they will not even IMPORT without
#                   it -- and unittest cannot collect the parametrised ones at all
#                   (see the verify step)
#
REQUIRED=(numerapi numerai-tools pandas pyarrow numpy scipy scikit-learn lightgbm joblib matplotlib pytest)

#   xgboost catboost   two more model wrappers in prediction/models.py; the
#                      pipeline imports them lazily, so they are only needed if
#                      you actually run --model xgb / catboost
#   scikit-optimize    --search-type bayes (BayesSearchCV). Without it the
#                      random search still works; the CLI fails early and says so
#   torch              the LSTM pipeline (src/run_nn_forecast.py). Optional and
#                      lazily imported for the same reason as the boosters: it is
#                      a large wheel, run_forecast.py never touches it, and a
#                      machine that only runs ridge/lgb should not have to carry
#                      it. run_nn_forecast fails early with a clear message.
OPTIONAL=(xgboost catboost scikit-optimize torch)

if [ "$CHECK_ONLY" = "1" ]; then
    ok "--check: skipping install"
else
    # Deliberately NOT --upgrade. On a fresh machine pip resolves the latest
    # anyway; on a machine that already works, upgrading pandas/numpy/sklearn
    # underneath a running experiment is a good way to break it mid-walk-forward.
    # Pass PIP_ARGS=--upgrade if you actually want that.
    echo "  installing ${#REQUIRED[@]} required packages ..."
    # shellcheck disable=SC2086
    "$PYTHON" -m pip install --quiet $PIP_ARGS "${REQUIRED[@]}" \
        || fail "pip install failed for the required set"
    ok "required packages installed"

    echo "  installing ${#OPTIONAL[@]} optional packages ..."
    for pkg in "${OPTIONAL[@]}"; do
        # shellcheck disable=SC2086
        if "$PYTHON" -m pip install --quiet $PIP_ARGS "$pkg" 2>/dev/null; then
            ok "$pkg"
        else
            warn "$pkg failed to install -- continuing (it is optional)"
        fi
    done
fi

# Import them rather than trusting pip's exit code: a wheel can install and
# still fail to load (a missing libgomp is the classic one for lightgbm).
"$PYTHON" - <<'PY' || fail "a required package installed but does not import"
import importlib, sys
required = {"numerapi": "numerapi", "numerai_tools": "numerai-tools",
            "pandas": "pandas", "pyarrow": "pyarrow", "numpy": "numpy",
            "scipy": "scipy", "sklearn": "scikit-learn", "lightgbm": "lightgbm",
            "joblib": "joblib", "matplotlib": "matplotlib", "pytest": "pytest"}
optional = {"xgboost": "xgboost", "catboost": "catboost", "skopt": "scikit-optimize",
            "torch": "torch"}
bad = []
for module, pkg in required.items():
    try:
        m = importlib.import_module(module)
        print("  ok    %-16s %s" % (pkg, getattr(m, "__version__", "")))
    except Exception as exc:
        print("  FAIL  %-16s %s" % (pkg, exc)); bad.append(pkg)
for module, pkg in optional.items():
    try:
        m = importlib.import_module(module)
        print("  ok    %-16s %s (optional)" % (pkg, getattr(m, "__version__", "")))
    except Exception:
        print("  WARN  %-16s missing -- optional" % pkg)
if bad:
    sys.exit(1)
PY

# --------------------------------------------------------------------------- #
step "3/6  crypto dataset"
# --------------------------------------------------------------------------- #
#
# The pipeline downloads lazily on first use, so this step only front-loads it:
# better to fail on a network problem now than 40 minutes into a walk-forward.
#
"$PYTHON" - <<'PY' || fail "crypto dataset download failed"
import os
from src.data_mani.utils import download_if_missing, get_api, get_spec

spec = get_spec("crypto")
version = spec["version"]
api = get_api("crypto")
for filename in ("train.parquet", spec["meta_model_file"]):
    before = os.path.exists(os.path.join("data", "crypto",
                                         version.split("/")[-1], filename))
    path = download_if_missing(api, "crypto", version, filename, verbose=True)
    print("  ok    %-32s %6.1f MB%s" % (
        filename, os.path.getsize(path) / 1024 / 1024,
        "" if before else "  (downloaded)"))
PY

"$PYTHON" - <<'PY' || fail "the downloaded dataset does not look right"
import pyarrow.parquet as pq
path = "data/crypto/v2.0/train.parquet"
names = pq.ParquetFile(path).schema.names
n_feat = sum(c.startswith("feature_") for c in names)
rows = pq.ParquetFile(path).metadata.num_rows
print("  ok    %s rows, %d features" % (format(rows, ","), n_feat))
if n_feat != 22:
    raise SystemExit("  FAIL  expected 22 features, found %d" % n_feat)
for column in ("symbol", "date", "target_binned_return_20"):
    if column not in names:
        raise SystemExit("  FAIL  missing column '%s'" % column)
PY

if [ "$WITH_NUMERAI" = "1" ]; then
    echo "  downloading numerai Classic v5.3 (several GB, this is slow) ..."
    "$PYTHON" - <<'PY' || warn "numerai Classic download failed -- crypto is unaffected"
from src.data_mani.utils import download_if_missing, get_api, get_spec
spec = get_spec("numerai"); api = get_api("numerai")
for filename in ("train.parquet", "validation.parquet", spec["meta_model_file"]):
    path = download_if_missing(api, "numerai", spec["version"], filename, verbose=True)
    print("  ok    %s" % path)
PY
else
    ok "numerai Classic skipped (pass --with-numerai for it; it is several GB)"
fi

# --------------------------------------------------------------------------- #
step "4/6  derived feature datasets"
# --------------------------------------------------------------------------- #
#
# The base 22 features are enough for most arms. The experiments that reference
# a wider set need this chain, each step reading the previous one:
#
#     v2.0                                     22   downloaded
#     +tenc-onehot                            763   +1 target encoding, +740 ticker one-hot
#     +rollsel                                787   +24 rolling means (8 features x 5/20/60)
#     +spread                                 798   +11 horizon spreads (20d - 60d)
#     +interact                               812   +14 products / acceleration gaps
#
# Only the last one is consumed directly (the "allfeat++" arm), but the chain
# has to be built in order.
#
ROLL_FEATURES="feature_momentum_20d,feature_momentum_60d,\
feature_relative_strength_index_20d,feature_relative_strength_index_60d,\
feature_sharpe_ratio_20d,feature_sharpe_ratio_60d,\
feature_volatility_20d,feature_volatility_60d"

build_step() {
    # $1 = output version, $2.. = the command that produces it
    local out="$1"; shift
    if [ -f "data/${out}/train.parquet" ]; then
        ok "${out} already built"
        return 0
    fi
    echo "  building ${out} ..."
    "$@" >/dev/null 2>&1 || { warn "failed to build ${out}"; return 1; }
    ok "${out}"
}

if [ "$WITH_DERIVED" = "1" ] && [ "$CHECK_ONLY" != "1" ]; then
    build_step "crypto/v2.0-tenc-onehot" \
        "$PYTHON" -m src.data_mani.build_target_encoding \
            --smoothing 50 --keep-onehot --out-suffix=-tenc-onehot
    build_step "crypto/v2.0-tenc-onehot-rollsel" \
        "$PYTHON" -m src.data_mani.build_rolling_features \
            --source-version crypto/v2.0-tenc-onehot \
            --features "$ROLL_FEATURES" --windows 5,20,60 --out-suffix=-rollsel
    build_step "crypto/v2.0-tenc-onehot-rollsel-spread" \
        "$PYTHON" -m src.data_mani.build_derived --kind spread \
            --source-version crypto/v2.0-tenc-onehot-rollsel --out-suffix=-spread
    build_step "crypto/v2.0-tenc-onehot-rollsel-spread-interact" \
        "$PYTHON" -m src.data_mani.build_derived --kind interact \
            --source-version crypto/v2.0-tenc-onehot-rollsel-spread --out-suffix=-interact

    # Assert the shape rather than trusting that four builders all did their
    # job: a silent feature-count drift attaches a plausible number to the
    # wrong experiment, which is worse than a crash.
    "$PYTHON" - <<'PY'
import os, pyarrow.parquet as pq
expected = {"crypto/v2.0": 22, "crypto/v2.0-tenc-onehot": 763,
            "crypto/v2.0-tenc-onehot-rollsel": 787,
            "crypto/v2.0-tenc-onehot-rollsel-spread": 798,
            "crypto/v2.0-tenc-onehot-rollsel-spread-interact": 812}
for version, want in expected.items():
    path = "data/%s/train.parquet" % version
    if not os.path.exists(path):
        print("  WARN  %-50s not built" % version); continue
    got = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
    print("  %s %-50s %4d features%s" % (
        "ok   " if got == want else "WARN ", version, got,
        "" if got == want else "  (expected %d)" % want))
PY
else
    ok "derived datasets skipped (pass --with-derived; ~3 min, ~45 MB)"
fi

# --------------------------------------------------------------------------- #
step "5/6  credentials template"
# --------------------------------------------------------------------------- #
#
# Only src/live/ needs these, and only to submit. Everything in this setup, and
# every backtest, runs without them.
#
if [ -f .env ]; then
    ok ".env already exists -- left untouched"
elif [ "$CHECK_ONLY" = "1" ]; then
    warn "no .env (only needed for live submission)"
else
    cat > .env <<'ENV'
# Numerai API credentials -- required ONLY for live submission (src/live/).
# Create a key pair at https://numer.ai/account -> Automation
# This file is gitignored. Never commit real keys.
NUMERAI_PUBLIC_ID=
NUMERAI_SECRET_KEY=
ENV
    ok "wrote .env template (gitignored; fill in only if you will submit)"
fi

git check-ignore -q .env 2>/dev/null \
    && ok ".env is gitignored" \
    || warn ".env is NOT gitignored -- do not commit it"

# --------------------------------------------------------------------------- #
step "6/6  verify"
# --------------------------------------------------------------------------- #

if [ "$SKIP_TESTS" = "1" ]; then
    ok "tests skipped (SKIP_TESTS=1)"
else
    # pytest, not unittest. Two test modules use pytest.mark.parametrize on
    # plain functions, and unittest only ever collects TestCase subclasses --
    # so `python -m unittest discover` runs 54 of the 105 tests and reports OK,
    # silently skipping the other 51. A green run that covered half the suite
    # is exactly the kind of false assurance this repo is built to refuse.
    echo "  running the test suite (~30 s) ..."
    if "$PYTHON" -m pytest src/tests -q >/tmp/setup_tests.$$ 2>&1; then
        ok "$(grep -E "passed|failed" /tmp/setup_tests.$$ | tail -1)"
    else
        tail -25 /tmp/setup_tests.$$
        rm -f /tmp/setup_tests.$$
        fail "test suite failed -- the install is not usable"
    fi
    rm -f /tmp/setup_tests.$$

    # A green test suite does not prove the DATA is usable: the tests are
    # offline by design. One tiny walk-forward exercises download, load,
    # embargo, search, scoring and output writing end to end.
    echo "  running a smoke forecast (~1 min) ..."
    if "$PYTHON" -m src.run_forecast crypto ridge \
            --cv-scheme rolling --train-window 6month --refit-every 6month \
            --n-iter 4 --n-splits 2 --era-step 16 --seed 42 \
            --run-name setup_smoke -q >/tmp/setup_smoke.$$ 2>&1; then
        "$PYTHON" - <<'PY'
import json, pandas as pd
d = json.load(open("results/metrics/crypto/ridge/setup_smoke.json"))
periods = len(pd.read_csv("results/per_period/crypto/ridge/setup_smoke.csv"))
print("  ok    smoke forecast: CORR %.4f on %d features over %d periods"
      % (d["corr"]["mean"], d["config"]["n_features"], periods))
PY
        # Leave no fake result behind to be mistaken for an experiment.
        rm -f results/*/crypto/ridge/setup_smoke.*
        ok "smoke artifacts removed"
    else
        tail -15 /tmp/setup_smoke.$$
        rm -f /tmp/setup_smoke.$$
        fail "smoke forecast failed -- packages are installed but the pipeline is not working"
    fi
    rm -f /tmp/setup_smoke.$$
fi

# --------------------------------------------------------------------------- #
echo
echo "=============================================================="
echo "ready in $(( ($(date +%s) - started) / 60 ))m$(( ($(date +%s) - started) % 60 ))s"
echo "=============================================================="
cat <<'NEXT'

Run one model:
  python3 -m src.run_forecast crypto ridge_glmnet --cv-scheme rolling \
      --train-window 6month --refit-every 6month --n-iter 50 --n-splits 5 \
      --seed 42 --run-name my_first_run

Run a full study (9 arms, ~8 h on 32 cores):
  bash experiments/lgb_6m6m_reg_crypto.sh

See what a runner would do without running it:
  DRY_RUN=1 bash experiments/lgb_6m6m_reg_crypto.sh

Results land in results/{forecast,per_period,hyperparams,metrics}/ -- all
gitignored, all reproducible from the command recorded in each metrics json.

NEXT
[ "$WITH_DERIVED" != "1" ] && echo "NOTE: the allfeat++ arm needs --with-derived. Re-run: bash setup.sh --with-derived"
exit 0
