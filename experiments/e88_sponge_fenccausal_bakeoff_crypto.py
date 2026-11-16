#!/usr/bin/env python3
"""
E88 -- SPONGE-cluster bake-off on the 1506f causal-combo (fenccausal).

Tune every model class FROM SCRATCH on each SPONGE signed-spectral cluster of
crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal (1506f), rank the (model x
cluster) cells, and screen ensembles. Clusters come from the user's file
data/crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal/sponge_feature_groups.json
(Cucuringu et al. AISTATS 2019; signed Pearson affinity, k=25 by silhouette).
They partition all 1506 features EXACTLY (0 missing, 0 overlap); each cluster is
selected with a --feature-include anchored alternation of its exact feature
names, so membership is honoured verbatim (not by a prefix guess).

This is the 1506f twin of E87 (which ran the 753f combo WITHOUT the causal
per-symbol means). The feature universe here adds a leak-safe causal expanding
per-symbol mean (_symmean) for every combo feature -- the fenccausal idea behind
IA_01, the best MLP slot -- so the clusters here mix each primitive with its own
causal symbol-level twin.

25 clusters x 5 models (ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz) = 125
from-scratch runs at the shared protocol (rolling 130/130, embargo auto=20,
5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20) -- identical to
E85/E86/E87 so every cell is comparable to the book. Each cell is paired (NW lag
20) vs the FULL 1506f combo (e85_*_fenc, reused) to show how much of the combo's
signal the cluster carries; the ensemble screen over the 125 e88_* cells + book is
run afterwards with report_html.

The four singleton clusters and the size-2/3 ones are the break-even-inflation
(BEI) macro features; they are kept so the partition stays exact and so their
near-zero rank is recorded (consistent with E80/E81), but a 1-4 feature search is
near-trivial and their cells are expected to sit at the bottom.

Clusters run CONCURRENTLY (CLUSTER_JOBS at a time, default 5); models run
sequentially within a cluster. Env knobs: DRY_RUN=1 (preflight + list, no fits),
SCOUT=1 (era-step 6 / n_iter 4 / 2 folds, wiring ONLY -- not comparable),
NO_MAIL=1, CLUSTER_JOBS=N, SKIP_DONE=1 (default -- an arm whose metrics json
already exists is skipped, so a relaunch resumes).
INTERPRETER: run with /home/danielco/miniconda3/bin/python3 (bare python3 refuses).
"""
import os, sys, json, re, time, subprocess, concurrent.futures as cf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO); os.makedirs("logs", exist_ok=True)
PY = sys.executable
DATA = "crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal"
SPEC = "data/%s/sponge_feature_groups.json" % DATA
TGT = "target_binned_return_20"
EXPERIMENT, PREFIX = "E88", "e88"
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
# the full-1506f reference these cells are read against (reused, not re-run)
REF = {m: "e85_%s_fenc" % m for m in MODELS}

# ---- load clusters ----
spec = json.load(open(SPEC))
clusters = [(c.get("id") or c.get("name"), (c.get("features") or c.get("members")))
            for c in spec["clusters"]]
NC = len(clusters)
NCELL = NC * len(MODELS)
sizes_all = {cid: len(fs) for cid, fs in clusters}

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
        if not fs:
            bad.append("cluster %s has no feature list" % cid); continue
        miss = [f for f in fs if f not in feats]
        if miss:
            bad.append("cluster %s: %d feats not in dataset (%s...)" % (cid, len(miss), miss[:2]))
        seen += fs
    if len(seen) != len(set(seen)):
        bad.append("clusters overlap: %d feats listed, %d unique" % (len(seen), len(set(seen))))
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
    print("preflight OK: %s (%df), %d clusters partition it EXACTLY, 5 wrappers construct."
          % (DATA, len(feats), NC))

preflight()

def regex_for(fs):
    # exact-match alternation; re.escape handles any regex-special char in a name
    return "^(?:%s)$" % "|".join(re.escape(f) for f in fs)

def degenerate_clusters():
    """Clusters whose every feature is CONSTANT WITHIN each date.

    numerai_corr ranks within a date, so a prediction that is constant within a
    date has zero within-date std and the metric is undefined -- the search
    "could not score any candidate" and the arm crashes (ridge/lgb/catboost/rf);
    mlp only "survives" on random-init noise, scoring ~0 (meaningless). These are
    exactly the pure day-level break-even-inflation (BEI) clusters: a BEI series
    is one macro value per date broadcast to every coin, so a cluster made only of
    raw BEI levels/changes carries no cross-section. Detected data-drivenly on a
    ~30-date sample (not a hard-coded list), so it self-adapts if the grouping
    changes; here it resolves to {02,03,06,14,19,21,24} (12 raw BEI feats), which
    matched the scout failures exactly. Excluded from the fitted roster and NAMED
    in the report (per the repo's 'name anything excluded' rule). The _symmean BEI
    clusters vary across symbols and are kept."""
    import pyarrow.parquet as pq
    import numpy as np
    path = "data/%s/train.parquet" % DATA
    uniq = np.sort(pq.read_table(path, columns=["date"]).column("date").to_pandas().unique())
    sample = list(uniq[:: max(1, len(uniq) // 30)][:30])
    allfeat = sorted({f for _, fs in clusters for f in fs})
    df = pq.read_table(path, columns=["date"] + allfeat,
                       filters=[("date", "in", sample)]).to_pandas()
    gb = df.groupby("date")
    varying = {f: float((gb[f].nunique() > 1).mean()) for f in allfeat}  # frac of dates f varies in
    deg = set(cid for cid, fs in clusters if max(varying[f] for f in fs) < 1e-9)
    return deg

DEGEN = degenerate_clusters()
ACTIVE = [(cid, fs) for cid, fs in clusters if cid not in DEGEN]
NA = len(ACTIVE); NACELL = NA * len(MODELS)
_degs = sorted(DEGEN, key=lambda c: int(c.split("_")[-1]))
print("roster: %d/%d clusters active (%d cells) | %d excluded as within-date degenerate (day-level BEI): %s"
      % (NA, NC, NACELL, len(DEGEN), ", ".join("%s(%d)" % (c, sizes_all[c]) for c in _degs) or "none"))

if DRY:
    for cid, fs in ACTIVE:
        for m in MODELS:
            print("  %-30s %-14s %4d feats" % ("%s_%s_%s" % (PREFIX, m, cid), m, len(fs)))
    if DEGEN:
        print("-- EXCLUDED (within-date degenerate, day-level BEI): %s"
              % ", ".join("%s(%d)" % (c, sizes_all[c]) for c in _degs))
    print("== %d cells (%d active clusters x %d models); %d clusters excluded ==" % (NACELL, NA, len(MODELS), len(DEGEN)))
    sys.exit(0)

t0 = time.time()
if not NOMAIL:
    try:
        from src.live.notify import send_text
        send_text("[crypto] %s started" % EXPERIMENT,
                  "%s SPONGE-cluster bake-off (1506f fenccausal) started %s\n  %d active clusters x 5 models = %d from-scratch runs on %s, n_iter=%d (%d day-level-BEI clusters excluded as within-date degenerate).\n  clusters concurrent (%d at a time); paired vs full combo e85_*_fenc. report logs/%s_report.txt"
                  % (EXPERIMENT, time.strftime("%F %T"), NA, NACELL, DATA, NITER, len(DEGEN), CLUSTER_JOBS, PREFIX))
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

print("=== %s: %d active SPONGE clusters x 5 models from scratch on %s | n_iter=%d (%d excluded) ===" % (EXPERIMENT, NA, DATA, NITER, len(DEGEN)), flush=True)
with cf.ThreadPoolExecutor(max_workers=CLUSTER_JOBS) as ex:
    futs = {ex.submit(run_cluster, cid, fs): cid for cid, fs in ACTIVE}
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
for cid, fs in ACTIVE:
    for m in MODELS:
        r = load(m, "%s_%s_%s" % (PREFIX, m, cid))
        if r:
            rows.append((cid, sizes_all[cid], m, r[0], r[1], r[2], r[3]))
rows.sort(key=lambda x: -x[5])

report = "logs/%s_report.txt" % PREFIX
lines = []
lines.append("%s -- SPONGE-cluster bake-off: %d active clusters x 5 models from scratch on %s" % (EXPERIMENT, NA, DATA))
lines.append("finished in %dm, %d/%d cells on disk. Each paired (NW lag 20) vs FULL combo e85_<model>_fenc." % ((time.time() - t0) / 60, len(rows), NACELL))
lines.append("=" * 66)
lines.append("rank  cluster(nfeat)      model          CORR      MMC   payout  c.shrp")
for k, (cid, n, m, c, mm, p, s) in enumerate(rows, 1):
    lines.append("%4d  %-12s(%4d)  %-13s %8.5f %8.5f %8.5f %7.3f" % (k, cid, n, m, c, mm, p, s))
lines.append("\nFULL combo reference (e85_*_fenc, 1506f):")
for m in MODELS:
    r = load(m, REF[m])
    if r:
        lines.append("  %-13s CORR %.5f MMC %.5f payout %.5f" % (m, r[0], r[1], r[2]))
if DEGEN:
    lines.append("\nEXCLUDED %d clusters -- within-date DEGENERATE (day-level break-even-inflation; one macro value/date" % len(DEGEN))
    lines.append("broadcast to every coin => constant within date => numerai_corr undefined; ridge/lgb/catboost/rf cannot")
    lines.append("score any candidate, mlp fits only init noise ~0). Not a bug, a structural property of the BEI block:")
    _cd = dict(clusters)
    for cid in sorted(DEGEN, key=lambda c: int(c.split("_")[-1])):
        lines.append("  %-12s(%4d)  %s" % (cid, sizes_all[cid], ", ".join(_cd[cid][:3]) + ("..." if sizes_all[cid] > 3 else "")))
lines.append("\nEnsemble screen (promising blends) -- run report_html on the %d e88_* cells + book, e.g.:" % NACELL)
lines.append("  %s -m src.analysis.report_html --model lgb_ktz --runs $(for c in %s; do for m in %s; do echo $m/%s_${m}_$c; done; done)"
             % (PY, " ".join(cid for cid, _ in ACTIVE), " ".join(MODELS), PREFIX))
text = "\n".join(lines)
open(report, "w").write(text + "\n")
print("\n" + text, flush=True)

if not NOMAIL:
    try:
        from src.live.notify import send_text
        send_text("[crypto] %s finished" % EXPERIMENT, text)
    except Exception as e:
        print("note: finish email failed: %s" % e)
