#!/usr/bin/env python3
"""
E87 -- SPONGE-cluster bake-off on the 753f combo.

Tune every model class FROM SCRATCH on each of the 10 SPONGE signed-spectral
clusters of crypto/v2.0-vol-infl-carver-ix-fillmask (753f), rank the (model x
cluster) cells, and screen ensembles. Clusters come from the user's file
data/crypto/v2.0-vol-infl-carver-ix-fillmask/sponge_feature_groups.json (Cucuringu
et al. AISTATS 2019; signed Pearson affinity, k=10 by silhouette). They partition
all 753 features; each cluster is selected with a --feature-include anchored
alternation of its exact feature names.

10 clusters x 5 models (ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz) = 50
from-scratch runs at the shared protocol (rolling 130/130, embargo auto=20,
5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20). Each cell is
paired (NW lag 20) vs the FULL combo (e82_*_combo, reused) to show how much of the
combo's signal the cluster carries; the ensemble screen over the 50 e87_* cells +
book is run afterwards with report_html.

Clusters run CONCURRENTLY (CLUSTER_JOBS at a time, default 5); models run
sequentially within a cluster. Env knobs: DRY_RUN=1, SCOUT=1 (era-step 6/n_iter 4,
not comparable), NO_MAIL=1, CLUSTER_JOBS=N, SKIP_DONE=1 (default).
INTERPRETER: this file is run with /home/danielco/miniconda3/bin/python3.
"""
import os, sys, json, re, time, subprocess, concurrent.futures as cf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO); os.makedirs("logs", exist_ok=True)
PY = sys.executable
# Env-driven so the same core runs any SPONGE-cluster bake-off; defaults = E87.
DATA = os.environ.get("SPONGE_DATA", "crypto/v2.0-vol-infl-carver-ix-fillmask")
SPEC = os.environ.get("SPONGE_SPEC", "data/%s/sponge_feature_groups.json" % DATA)
TGT = "target_binned_return_20"
EXPERIMENT = os.environ.get("SPONGE_EXP", "E87")
PREFIX = os.environ.get("SPONGE_PREFIX", "e87")
BPREF = os.environ.get("SPONGE_BASE_PREFIX", "e82")
BSUF = os.environ.get("SPONGE_BASE_SUFFIX", "combo")
BLABEL = os.environ.get("SPONGE_BASE_LABEL", "combo")
WIN, REFIT, EMB, NSPL, VF, SEED, NITER, ERA = 130, 130, "auto", 5, 0.3, 42, 50, 1
if os.environ.get("SCOUT") == "1":
    PREFIX, ERA, NSPL, NITER = PREFIX + "scout", 6, 2, 4
    print("SCOUT MODE (not comparable)")
DRY = os.environ.get("DRY_RUN") == "1"
NOMAIL = os.environ.get("NO_MAIL") == "1"
SKIP_DONE = os.environ.get("SKIP_DONE", "1") == "1"
CLUSTER_JOBS = int(os.environ.get("CLUSTER_JOBS", "5"))

MODELS = ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"]
SJOBS = {"ridge": 4, "lgb_ktz": 2, "catboost_ktz": 1, "rf_ktz": 4, "mlp_ktz": 4}
NJOBS = {"ridge": 1, "lgb_ktz": 1, "catboost_ktz": 2, "rf_ktz": 1, "mlp_ktz": 1}
CELLS = {"ridge": 100, "lgb_ktz": 432, "catboost_ktz": 12, "rf_ktz": 60, "mlp_ktz": 120}

# ---- load clusters ----
spec = json.load(open(SPEC))
clusters = [(c.get("id") or c.get("name"), (c.get("features") or c.get("members")))
            for c in spec["clusters"]]

# ---- preflight ----
def preflight():
    import pyarrow.parquet as pq
    from sklearn.model_selection import ParameterGrid
    from src.prediction.models import build_wrapper
    names = pq.ParquetFile("data/%s/train.parquet" % DATA).schema.names
    feats = set(n for n in names if n.startswith("feature_"))
    bad = []
    if TGT not in names:
        bad.append("%s lacks %s" % (DATA, TGT))
    seen = []
    for cid, fs in clusters:
        miss = [f for f in fs if f not in feats]
        if miss:
            bad.append("cluster %s: %d feats not in dataset (%s...)" % (cid, len(miss), miss[:2]))
        seen += fs
    if len(set(seen)) != len(feats):
        bad.append("clusters cover %d unique feats, dataset has %d" % (len(set(seen)), len(feats)))
    for m in MODELS:
        try:
            w = build_wrapper(m, "regression", seed=42, n_jobs=1)
            gc = len(list(ParameterGrid(w.param_grid)))
        except Exception as e:
            bad.append("%s: %s" % (m, e)); continue
        if gc != CELLS[m]:
            bad.append("%s grid %d != %d" % (m, gc, CELLS[m]))
    if bad:
        sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
    print("preflight OK: %s (%df), %d clusters partition it, 5 wrappers construct."
          % (DATA, len(feats), len(clusters)))

preflight()

def regex_for(fs):
    return "^(?:%s)$" % "|".join(re.escape(f) for f in fs)

if DRY:
    for cid, fs in clusters:
        for m in MODELS:
            print("  %-26s %-14s %4d feats" % ("%s_%s_%s" % (PREFIX, m, cid), m, len(fs)))
    sys.exit(0)

t0 = time.time()
if not NOMAIL:
    try:
        from src.live.notify import send_text
        send_text("[crypto] %s started" % EXPERIMENT,
                  "%s SPONGE-cluster bake-off started %s\n  %d clusters x 5 models = %d from-scratch runs on %s, n_iter=%d.\n  clusters concurrent (%d at a time); paired vs %s (%s_*_%s). report logs/%s_report.txt"
                  % (EXPERIMENT, time.strftime("%F %T"), len(clusters), len(clusters) * len(MODELS), DATA, NITER, CLUSTER_JOBS, BLABEL, BPREF, BSUF, PREFIX))
    except Exception as e:
        print("note: start email failed: %s" % e)

def run_arm(m, cid, rx):
    run = "%s_%s_%s" % (PREFIX, m, cid)
    mj = "results/metrics/crypto/%s/%s.json" % (m, run)
    if SKIP_DONE and os.path.exists(mj) and os.path.getsize(mj) > 0:
        return (run, "skip")
    cmd = [PY, "-u", "-m", "src.run_forecast", "crypto", m, "--task", "regression",
           "--metric", "corr", "--bmc-reference", "benchmark", "--cv-scheme", "rolling",
           "--train-window", str(WIN), "--refit-every", str(REFIT), "--embargo", EMB,
           "--n-splits", str(NSPL), "--n-iter", str(NITER), "--val-frac", str(VF),
           "--era-step", str(ERA), "--seed", str(SEED), "--search-jobs", str(SJOBS[m]),
           "--n-jobs", str(NJOBS[m]), "--data-version", DATA, "--feature-include", rx,
           "--target", TGT, "--run-name", run]
    with open("logs/%s.log" % run, "w") as lg:
        st = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT).returncode
    return (run, "done" if st == 0 else "FAILED(%d)" % st)

def run_cluster(cid, fs):
    rx = regex_for(fs)
    out = []
    for m in MODELS:
        run, status = run_arm(m, cid, rx)
        out.append("  %s %s" % (run, status)); print("  %s %s" % (run, status), flush=True)
    return out

print("=== %s: %d SPONGE clusters x 5 models from scratch on %s | n_iter=%d ===" % (EXPERIMENT, len(clusters), DATA, NITER), flush=True)
with cf.ThreadPoolExecutor(max_workers=CLUSTER_JOBS) as ex:
    futs = {ex.submit(run_cluster, cid, fs): cid for cid, fs in clusters}
    for f in cf.as_completed(futs):
        f.result()

# ---- rank + reference ----
def load(m, run):
    p = "results/metrics/crypto/%s/%s.json" % (m, run)
    try:
        d = json.load(open(p))
    except Exception:
        return None
    c = (d.get("corr") or {}).get("mean") or 0.0
    mm = (d.get("mmc") or {}).get("mean") or 0.0
    s = (d.get("corr") or {}).get("sharpe") or 0.0
    return c, mm, 0.1 * c + mm, s

rows = []
sizes = {cid: len(fs) for cid, fs in clusters}
for cid, fs in clusters:
    for m in MODELS:
        r = load(m, "%s_%s_%s" % (PREFIX, m, cid))
        if r:
            rows.append((cid, sizes[cid], m, r[0], r[1], r[2], r[3]))
rows.sort(key=lambda x: -x[5])

report = "logs/%s_report.txt" % PREFIX
lines = []
lines.append("%s -- SPONGE-cluster bake-off: %d clusters x 5 models from scratch on %s" % (EXPERIMENT, len(clusters), DATA))
lines.append("finished in %dm, %d/%d cells on disk. Each paired (NW lag 20) vs %s (%s_<model>_%s)." % ((time.time() - t0) / 60, len(rows), len(clusters) * len(MODELS), BLABEL, BPREF, BSUF))
lines.append("=" * 62)
lines.append("rank  cluster(nfeat)      model          CORR      MMC   payout  c.shrp")
for k, (cid, n, m, c, mm, p, s) in enumerate(rows, 1):
    lines.append("%4d  %-10s(%4d)  %-13s %8.5f %8.5f %8.5f %7.3f" % (k, cid, n, m, c, mm, p, s))
lines.append("\n%s reference (%s_*_%s):" % (BLABEL, BPREF, BSUF))
for m in MODELS:
    r = load(m, "%s_%s_%s" % (BPREF, m, BSUF))
    if r:
        lines.append("  %-13s CORR %.5f MMC %.5f payout %.5f" % (m, r[0], r[1], r[2]))
lines.append("\nEnsemble screen (promising blends) -- run report_html on the %d %s_* cells + book:" % (len(rows), PREFIX))
lines.append("  %s -m src.analysis.report_html --model lgb_ktz --runs $(for c in %s; do for m in %s; do echo $m/%s_${m}_$c; done; done)"
             % (PY, " ".join(cid for cid, _ in clusters), " ".join(MODELS), PREFIX))
text = "\n".join(lines)
open(report, "w").write(text + "\n")
print("\n" + text, flush=True)

if not NOMAIL:
    try:
        from src.live.notify import send_text
        send_text("[crypto] %s finished" % EXPERIMENT, text)
    except Exception as e:
        print("note: finish email failed: %s" % e)
