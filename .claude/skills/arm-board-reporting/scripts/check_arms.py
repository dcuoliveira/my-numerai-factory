"""
Is each arm actually a measurement?

    python3 -m ... check_arms.py --model lgb --runs fe_lgb_reg_raw fe_lgb_reg_onehot

Answers three questions the metrics json cannot:

  1. Do any two arms produce IDENTICAL predictions? Then one of them is not
     evidence about its feature block. This happens whenever a block is present
     but unusable -- LightGBM drops columns no split can satisfy, so a 762-column
     arm can come back byte-identical to a 22-column one.
  2. Which feature blocks does each arm's dataset actually contain? Derived by
     comparing parquet schemas, because the run NAME does not say: allfeat,
     allfeat++ and allfeat+++ all carry the 740 ticker columns silently.
  3. Do the recorded feature counts match the datasets on disk?
"""
import argparse, glob, itertools, json, os
import pandas as pd
import pyarrow.parquet as pq

def feats(version):
    path = "data/%s/train.parquet" % version
    if not os.path.exists(path):
        return None
    return {c for c in pq.ParquetFile(path).schema.names if c.startswith("feature_")}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tournament", default="crypto")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run names, or globs like 'fe_lgb_reg_*'")
    args = ap.parse_args()

    base = "results/metrics/%s/%s" % (args.tournament, args.model)
    runs = []
    for pattern in args.runs:
        hits = sorted(os.path.basename(p)[:-5]
                      for p in glob.glob(os.path.join(base, pattern + ".json")))
        runs.extend(hits or ([pattern] if os.path.exists(
            os.path.join(base, pattern + ".json")) else []))
    runs = list(dict.fromkeys(runs))
    if not runs:
        raise SystemExit("no runs matched under %s" % base)

    raw = feats("%s/v2.0" % args.tournament)
    blocks = {}
    for name, version in (("onehot", "v2.0-onehot"), ("tenc", "v2.0-tenc"),
                          ("spread", "v2.0-spread"), ("interact", "v2.0-interact"),
                          ("rollsel", "v2.0-rollsel")):
        f = feats("%s/%s" % (args.tournament, version))
        if f and raw:
            blocks[name] = f - raw

    print("%-34s %6s %-46s %s" % ("run", "feat", "data_version", "blocks present"))
    print("-" * 128)
    preds = {}
    for run in runs:
        cfg = json.load(open(os.path.join(base, run + ".json"))).get("config", {})
        version = cfg.get("data_version")
        present = feats(version) if version else None
        have = [n for n, b in blocks.items() if present and b <= present] if present else []
        drift = ""
        if present is not None and len(present) != cfg.get("n_features"):
            drift = "  *** recorded %s, dataset has %d" % (cfg.get("n_features"), len(present))
        print("%-34s %6s %-46s %s%s" % (run, cfg.get("n_features"), version,
                                        ", ".join(have) or "-", drift))
        path = "results/forecast/%s/%s/%s.csv" % (args.tournament, args.model, run)
        if os.path.exists(path):
            preds[run] = pd.read_csv(path)["prediction"]

    print()
    identical = []
    for a, b in itertools.combinations(preds, 2):
        if len(preds[a]) == len(preds[b]):
            d = float((preds[a] - preds[b]).abs().max())
            if d == 0.0:
                identical.append((a, b))
    if identical:
        print("IDENTICAL PREDICTION PAIRS -- one of each pair is not a measurement:")
        for a, b in identical:
            print("   %s  ==  %s   (max|diff| = 0.00e+00)" % (a, b))
    else:
        print("no two arms produce identical predictions: every run is a distinct measurement")

if __name__ == "__main__":
    main()
