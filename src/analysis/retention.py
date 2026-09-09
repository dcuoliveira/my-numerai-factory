"""
Which forecasts can be reclaimed, and why each survivor survives.

results/forecast/ is 99% of results/ -- 15 GB against 178 MB for every other
artifact combined -- because a walk-forward writes a full per-row prediction
panel per arm and nothing ever removed one. per_period/, metrics/ and
hyperparams/ are the scientific record and are never in scope here; a forecast
is the only artifact this module can delete.

The precedent is E30's runner, which already does
`rm -f results/forecast/crypto/lgb/e30_${ds}_nl*.csv` once its ensemble is
built, with the comment "Reclaim the per-row cell FORECASTS now that the
ensemble is built... on a SHARED QUOTA". That is correct and it is also
unavailable to a runner in general: a runner cannot know about registry.MODELS
or about which runs a published report cites. This module can, so the rule it
generalises is "a study's forecasts become reclaimable when its findings are
written" -- writing the conclusion is what unlocks the disk.

WHAT MAKES THIS SAFE, in the order it matters:

1. A forecast is the primary record of the label a run was scored on
   (results_table.target_of reads the header). Deleting one used to collapse
   the arm into report_data's "unknown" block and mix 20-day with 60-day rows.
   scored_target now falls back to the metrics config -- so a run whose config
   is SILENT about its target keeps its forecast, unconditionally. That is
   K3 below and it protects 494 runs.
2. The delete-set is derived, never hand-listed. mtime cannot discriminate here
   (every file is 2026-08 or 2026-09) and orphan-hunting against metrics finds
   about one file, so a rule-of-thumb sweep is not available.
3. Nothing is deleted unless its per_period AND metrics siblings are still on
   disk at the moment of unlink, so the arm survives in the book without its
   row-level panel -- exactly the state E30's 486 cells are already in.
4. Metrics files are opened read-only. experiment_db.scan_arms uses
   os.path.getmtime on them as the arm's "when did this land" timestamp, so
   rewriting or copying one without -p silently destroys a column.

Usage::

    python3 -m src.analysis.retention                  # plan only, changes nothing
    python3 -m src.analysis.retention --why lgb/e33_raw_a32_relu_lre3_al1
    python3 -m src.analysis.retention --study E33
    python3 -m src.analysis.retention --families a4_,x3_
    python3 -m src.analysis.retention --apply --plan-hash <hash from the plan>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

from src.analysis import experiment_db as edb

FORECAST_ROOT = os.path.join("results", "forecast")
METRICS_ROOT = os.path.join("results", "metrics")
PER_PERIOD_ROOT = os.path.join("results", "per_period")
REPORTS_GLOB = os.path.join("results", "reports")
MANIFEST_ROOT = os.path.join("results", "retention")

# Where a run name might be cited in prose or in a command. registry.py names
# runs in comments as often as in fields, and cross_target.py / corr_heatmap.py
# hardcode paths -- none of which a structural walk would find.
CITATION_DIRS = ("src", "experiments", ".claude")
CITATION_FILES = ("CLAUDE.md", "README.md")
TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _index(root, tournament, suffix):
    """Map (model, run) -> path for one artifact tree."""
    out = {}
    base = os.path.join(root, tournament)
    if not os.path.isdir(base):
        return out
    for model in sorted(os.listdir(base)):
        d = os.path.join(base, model)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith(suffix):
                continue
            run = name[: -len(suffix)]
            if run.endswith(".seeds"):
                continue
            out[(model, run)] = os.path.join(d, name)
    return out


def read_metrics(path):
    """The config of one metrics file. Read-only: mtime is load-bearing."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("config") or {}
    except (ValueError, OSError):
        return {}


def citation_tokens():
    """Every identifier-shaped token in the tracked corpus.

    Substring-free on purpose: a run is cited as a bare word (`w6r6_fixed_reg`
    in a registry comment), so tokenising and matching whole words is both
    cheaper and less trigger-happy than a substring scan.
    """
    seen = set()
    paths = [p for p in CITATION_FILES if os.path.exists(p)]
    for d in CITATION_DIRS:
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            paths.extend(os.path.join(root, f) for f in files)
    for path in paths:
        if os.path.splitext(path)[1] not in (".py", ".sh", ".md", ".json", ".txt", ""):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                seen.update(TOKEN.findall(handle.read()))
        except OSError:
            continue
    return seen


def report_tokens():
    """Run names named by an already-published report, which must stay re-renderable."""
    seen = set()
    for root, _dirs, files in os.walk(REPORTS_GLOB):
        for name in files:
            if not name.endswith((".html", ".md")):
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8",
                          errors="replace") as handle:
                    seen.update(TOKEN.findall(handle.read()))
            except OSError:
                continue
    return seen


def registry_roots():
    """(model, run) for every deployed slot's backtest, plus the reconstructions."""
    from src.analysis import slots as slots_mod
    from src.live import registry

    roots = set()
    for spec in registry.MODELS.values():
        run = spec.get("backtest_run")
        if not run or str(run).startswith("("):
            continue
        model = spec.get("scored_under") or spec.get("wrapper")
        if model:
            roots.add((model, run))
    for model, run, _why in slots_mod.RECONSTRUCTED.values():
        roots.add((model, run))
    return roots


def member_closure(seeds, metrics, tournament):
    """Follow config["members"] transitively.

    A blend's members are themselves runs, and blends of blends exist
    (CURUPIRA_08 sits inside CURUPIRA_SACI_03). Members are written either bare
    -- same tree -- or qualified as "model/run", the convention paired_delta
    uses.
    """
    out, queue = set(), list(seeds)
    while queue:
        key = queue.pop()
        if key in out:
            continue
        out.add(key)
        path = metrics.get(key)
        if not path:
            continue
        for member in read_metrics(path).get("members") or []:
            if "/" in str(member):
                model, run = str(member).split("/", 1)
            else:
                model, run = key[0], str(member)
            if (model, run) not in out:
                queue.append((model, run))
    return out


def build(tournament="crypto"):
    """Index every tree once and classify each forecast. Returns (state, keep, why)."""
    forecasts = _index(FORECAST_ROOT, tournament, ".csv")
    metrics = _index(METRICS_ROOT, tournament, ".json")
    per_period = _index(PER_PERIOD_ROOT, tournament, ".csv")

    cited = citation_tokens()
    reported = report_tokens()
    roots = registry_roots()

    why = {}

    def mark(keys, reason):
        for key in keys:
            why.setdefault(key, reason)

    # K1 -- deployed slots and everything they are built from.
    mark(member_closure(roots, metrics, tournament) & set(forecasts),
         "deployed slot backtest or a member of one")
    # K2 -- named in tracked code, a runner, or the conventions docs.
    mark({k for k in forecasts if k[1] in cited},
         "run name is cited in tracked code or docs")
    # K3 -- the target firebreak. The forecast is the ONLY record of the label.
    mark({k for k in forecasts
          if not read_metrics(metrics.get(k, "")).get("target")},
         "metrics config has no target, so the forecast is the only record of it")
    # K4 -- an incomplete run: deleting the forecast would erase it entirely.
    mark({k for k in forecasts if k not in metrics or k not in per_period},
         "run has no metrics or no per_period sibling")
    # K5 -- a published report names it and must stay re-renderable.
    mark({k for k in forecasts if k[1] in reported},
         "named by a published report")

    return {"forecasts": forecasts, "metrics": metrics,
            "per_period": per_period}, set(why), why


def candidates(state, keep, families=(), studies=()):
    """The delete-set: not kept, and positively opted in by study or family."""
    out = {}
    for key in sorted(state["forecasts"]):
        if key in keep:
            continue
        model, run = key
        study, is_scout = edb.study_of(run, model=model)
        spec = edb.STUDIES.get(study) or {}
        if is_scout:
            out[key] = ("scout arm of %s" % study)
        elif study and spec.get("forecasts") == "reclaim":
            out[key] = ("%s is flagged forecasts=reclaim" % study)
        elif study and study in studies:
            out[key] = ("%s named on the command line" % study)
        elif study is None and any(run.startswith(f) for f in families):
            out[key] = "pre-numbering family named on the command line"
    return out


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def plan_hash(delete):
    digest = hashlib.sha256()
    for key in sorted(delete):
        digest.update(("%s/%s\n" % key).encode())
    return digest.hexdigest()[:16]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tournament", default="crypto")
    parser.add_argument("--study", action="append", default=[],
                        help="also reclaim this study, without flagging it in STUDIES")
    parser.add_argument("--families", default="",
                        help="comma-separated prefixes of pre-numbering runs to reclaim")
    parser.add_argument("--why", metavar="MODEL/RUN",
                        help="explain one run's classification and exit")
    parser.add_argument("--apply", action="store_true",
                        help="delete; requires --plan-hash matching the printed plan")
    parser.add_argument("--plan-hash", default=None)
    parser.add_argument("--quarantine", default=None,
                        help="move into this directory instead of unlinking")
    args = parser.parse_args(argv)

    state, keep, why = build(args.tournament)
    families = tuple(f for f in args.families.split(",") if f)

    if args.why:
        model, run = args.why.split("/", 1)
        key = (model, run)
        print("%s/%s" % key)
        print("  forecast on disk : %s" % (key in state["forecasts"]))
        print("  metrics sibling  : %s" % (key in state["metrics"]))
        print("  per_period sibling: %s" % (key in state["per_period"]))
        study, is_scout = edb.study_of(run, model=model)
        print("  study            : %s%s" % (study, " (scout)" if is_scout else ""))
        if key in keep:
            print("  VERDICT: KEEP -- %s" % why[key])
        else:
            found = candidates(state, keep, families, tuple(args.study)).get(key)
            print("  VERDICT: %s" % ("DELETE -- %s" % found if found
                                     else "KEEP -- no rule opts it in for reclaim"))
        return 0

    delete = candidates(state, keep, families, tuple(args.study))
    sizes = {k: os.path.getsize(state["forecasts"][k]) for k in delete}
    total = sum(sizes.values())
    digest = plan_hash(delete)

    by_study = {}
    for key, reason in delete.items():
        study = edb.study_of(key[1], model=key[0])[0] or "(unnumbered)"
        entry = by_study.setdefault(study, [0, 0])
        entry[0] += 1
        entry[1] += sizes[key]

    print("forecasts on disk : %d" % len(state["forecasts"]))
    print("kept              : %d" % len(keep & set(state["forecasts"])))
    print("to reclaim        : %d  (%s)" % (len(delete), human(total)))
    print("plan hash         : %s" % digest)
    if by_study:
        print("\n  study        files      bytes  reason")
        for study in sorted(by_study):
            n, b = by_study[study]
            sample = next(k for k in delete if
                          (edb.study_of(k[1], model=k[0])[0] or "(unnumbered)") == study)
            print("  %-10s %6d  %9s  %s" % (study, n, human(b), delete[sample]))

    reasons = {}
    for key in keep & set(state["forecasts"]):
        reasons[why[key]] = reasons.get(why[key], 0) + 1
    print("\n  kept because:")
    for reason in sorted(reasons, key=lambda r: -reasons[r]):
        print("  %6d  %s" % (reasons[reason], reason))

    if not args.apply:
        print("\nplan only -- nothing changed. To act:")
        print("  python3 -m src.analysis.retention --apply --plan-hash %s" % digest)
        return 0

    if args.plan_hash != digest:
        print("\nREFUSED: --plan-hash %s does not match this plan (%s). The set "
              "changed since it was printed; re-read it." % (args.plan_hash, digest))
        return 1

    if args.quarantine:
        os.makedirs(args.quarantine, exist_ok=True)
    os.makedirs(MANIFEST_ROOT, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    records, moved = [], 0
    for key in sorted(delete):
        model, run = key
        path = state["forecasts"][key]
        # Re-assert at the moment of action, not just at plan time.
        if key not in state["metrics"] or key not in state["per_period"]:
            print("SKIP (siblings vanished): %s/%s" % key)
            continue
        expected = os.path.join(FORECAST_ROOT, args.tournament, model, run + ".csv")
        if os.path.normpath(path) != os.path.normpath(expected):
            print("SKIP (path shape): %s" % path)
            continue
        records.append({
            "model": model, "run": run, "bytes": sizes[key],
            "study": edb.study_of(run, model=model)[0],
            "target": read_metrics(state["metrics"][key]).get("target"),
            "reason": delete[key],
        })
        if args.quarantine:
            dest = os.path.join(args.quarantine, args.tournament, model)
            os.makedirs(dest, exist_ok=True)
            os.rename(path, os.path.join(dest, run + ".csv"))
        else:
            os.unlink(path)
        moved += 1

    manifest = os.path.join(MANIFEST_ROOT, "%s.json" % stamp)
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump({"plan_hash": digest, "quarantine": args.quarantine,
                   "count": moved, "bytes": total, "records": records},
                  handle, indent=2, sort_keys=True)
    print("\n%s %d forecasts (%s). Manifest: %s"
          % ("quarantined" if args.quarantine else "deleted", moved, human(total),
             manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
