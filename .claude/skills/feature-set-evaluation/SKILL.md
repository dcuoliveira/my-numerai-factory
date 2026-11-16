---
name: feature-set-evaluation
description: "Decide whether a candidate feature set / feature block / dataset variant (crypto/v2.0-*) adds marginal value over the base features on Numerai Crypto, and report the verdict as a self-contained HTML document. Covers the experimental design (fixed model configs, factorial of feature groups x model classes, seeds, purge, untouched test block), the controls and placebos that catch leakage (survivorship in missingness indicators, look-ahead coverage, tenure), the statistics (paired Newey-West + block bootstrap, never naive t-tests), equal-budget retuning, blend sweeps, and the verdict vocabulary. Use whenever asked 'does feature set X help', 'evaluate this dataset variant', 'is this block worth adding', 'ablate these feature groups', 'compare v2.0-foo vs v2.0', or to write up such a study -- and before building yet another feature block, read the prior-results table to avoid repeating a known null."
---

# Feature-set evaluation

Distilled from E64-E131 (litmom, momswap, macro, carver, xinteractions, vol,
infl, stat-arb, basis, allmom, fm/fmx representation blocks, tree-leaf blocks,
fillind, feed-fillind). Building the dataset is `crypto-derived-features`; running and
tabulating arms is `crypto-experiment-reporting` / `arm-board-reporting`. This
skill is the judgement layer between them: **how to tell a real marginal gain
from noise, leakage or search budget.**

## Base rate first

Almost every feature block tried here has been null or harmful against raw
`crypto/v2.0` (22 features). Prior to E131: litmom (E64/E65), momentum swap
(E66/E67), xsmom (E68), macro (E69, ridge-inert), engineered/interactions
(E70/E71/E82), carver (E72), mommask (E73/E75-77), inflation (E80/E81),
stat-arb/BAB/coint/basis (E90-E94), allmom (E108-E112), foundation-model reps
(E109/E113/E118), RF/catboost leaf blocks (E129/E130), litmom-fillind (E131,
REJECT), feed-fillind (E132, REJECT), own-history histfe (E133, REJECT), feed-fillind routed (E134, REJECT), histfe routed (E135, REJECT), histfe full procedure (E136, REJECT), fenccausal-binned (E137, NO IMPROVEMENT, REJECT on ridge), Chronos-2 states as 16 PCs zero-shot / fine-tuned (E143, E144, NO IMPROVEMENT), all 3,072 raw Chronos-2 dims (E146, REJECT on all three; worse than the PCs), Chronos-2 forecasts of the BTC+ETH residual return path with peer covariates / grouped context (E148, NO IMPROVEMENT, REJECT on ridge and MLP; residualising helps MMC standalone, peers and C2 encoding add nothing). The few wins were **blends**, not feature sets (E69 b5, E131 lgb
B0+0.3*ADDED_ONLY). Treat a positive feature-set result as surprising and
demand the full control set below before believing it. Check
`knowledge_base/` and `experiment_db` for a prior study of the same idea first.

## Design

1. **Fixed, named model configs.** Use the repo's tuned configs (ridge,
   `lgb_ktz`, `mlp_ktz`, catboost_ktz, rf_ktz from E65-style grids) -- never
   generic defaults. Say which names don't exist (e.g. there is no `ridge_ktz`)
   and what substituted. Note `mlp_ktz` is identity-activation = linear.
2. **At least three model classes** (linear, booster, NN/forest). A block that
   helps only one class is model-specific, report it per model.
3. **Factorial of feature groups**, not just base vs full:
   `B0` base · `B0C` base + controls · group-wise adds (`M1` raw block,
   `M2` + its indicators, ...) · `FULL` · `ADDED_ONLY` (new features alone) ·
   leave-one-group-out ablations. This is what tells you *which* part helps or hurts.
4. **Walk-forward, purged:** train 130 / purge 22 (≥ 20-day target overlap) /
   predict 130. ≥ 5 seeds for stochastic models; report seed spread.
5. **Final test block held out** and never used for any selection (blend
   weights, retuning, arm choice). Say if earlier studies already scored that
   block -- then it is not virgin.
6. **Equal-budget retuning arm:** re-search hyper-params on B0 and on FULL with
   the same budget. Rules out "the fixed config was tuned for B0". E62 showed an
   apparent edge that was purely 12 vs 50 search candidates.
7. **Same rows for every arm.** Common dates and symbols; name any arm or block
   that is missing (e.g. an MLP pipeline drops all-0/1 columns, so an
   indicators-only arm cannot be fit -- report it as missing, don't silently drop).

## Leakage controls (mandatory for any missingness / coverage feature)

- **Point-in-time coverage.** Indicators like "has spot data" built from a
  coverage snapshot taken today encode survivorship (coins that later died
  never got a feed). E131: matched 2026-09-22 → indicators looked useful on
  dev, flipped on test.
- **Placebo arm `P_NM`:** add only the look-ahead flag. If it "helps" on dev
  and collapses on test (E131: t −3.6..−5.4), the block's gain is leakage.
- **Tenure control `P_HIST`:** a causal days-in-panel feature. If a block's gain
  vanishes once tenure is controlled, the block was proxying listing age.
  (E131: tenure alone +0.0085 dev, t +3.5 -- a lead worth its own study.)
- **Warm-up placebo `P_WARM`:** when a block has no look-ahead flag, add one causal feature instead: the binned share of the block observed. A gain it can match is depth of history, not the block (E136: lgb dev payout +0.0099, F_RANK only +0.0025 above it).
- **Audit on the real panel:** rebuild the block on data truncated at several dates and compare with the dataset. A synthetic truncation test does not replace it (E136: 0 of 422,400 cells differ).
- Fill semantics: `fill`/fillmask with 0.5 == the loader's own fill
  (E79) -- don't re-run byte-identical datasets. `data/crypto/v2.0` is
  live-refreshed; reproduce old arms on frozen `v2.0-asof`.

## Statistics

- Score per period: `numerai_corr` on target_binned_return_20 (never mix
  targets), payout `0.1*corr + mmc` built **per period**, MMC via
  numerai_tools `correlation_contribution` against the historical meta model,
  and call it *proxy* MMC -- never official.
- **Paired deltas** (arm − B0, same dates, seed-matched) with Newey-West HAC SE
  at lag 20 and a moving-block bootstrap CI. No naive t-tests (95% overlap
  between consecutive periods).
- Bar: |t| ≥ 2 on dev **and** same sign on test. With many cells, mention
  multiplicity (Bonferroni ~3 for ~25 pairs); 6/102 cells at |t|≥2 is noise (E118).
- Also break down by block/year and by seed -- a gain from one block or one
  seed is not a gain (E104: seed 42 alone gave a false t +4.18).
- **Read a near-identical arm before a large MLP t.** The final block is one fit per seed, so Newey-West covers date noise, not fit noise. E137's mlp FULL had test payout t +3.2, and one extra feature moved that payout by -0.018.
- **Check prediction spread on a tuned MLP.** A large alpha collapses it to a constant (E137: spread 1e-14 in 6 of 12 blocks, one seed). That row is then unreadable in both directions.
- **More dimensions of a null representation make it worse, not better.** E146's 768 raw dims per state lost to their own 16 PCs on every state (ridge dev CORR −0.034..−0.041). If the PCs are null, do not widen.
- **Compare CORR and payout on the same dates.** Dev CORR starts in 2020, payout in May 2024, so "CORR down, payout up" can just be two windows (E136). A reading from that narrower window is a lead, not a verdict.
- Before calling a window spent, check for labelled dates past it, and keep a short one for a registered forward test instead of reading it (E136: 28 dates, about one independent period).
- Do not use training loss, in-sample fit or feature importance as evidence.

## Blends

Rank-blend `B0 + w*ADDED_ONLY` (or FULL) for w ∈ {0, .05, .1, .2, .3, .5};
choose w by dev CV only, then report test. Also report prediction correlation
with B0 -- low correlation + non-negative CORR is the only route by which a
weak block has paid off here. Label a test-only success "promising but
unconfirmed" and recommend an **unstaked** live slot, not a stake.

## Verdict vocabulary

Per model, per group, overall: **ADOPT** (dev |t|≥2 positive, test same sign,
survives retune + placebo) · **PROMISING BUT UNCONFIRMED** · **NO IMPROVEMENT**
(null) · **REJECT** (significantly harmful or leakage-driven). State which
controls each verdict passed.

## The report

**The deliverable is always a PDF**, with a name that states the question and the
verdict, e.g. `results/reports/crypto/<study>/E131_does_litmom_help_REJECT.pdf`. Send the
PDF, never the HTML alone and never a `report.md`. Author the report as a self-contained HTML
file (figures inline, opens offline; `experiments/e137_report.py` is the template), then print it:

```bash
python3 -m src.analysis.html_pdf results/reports/crypto/<study>/<name>.html
```

It writes `<name>.pdf` beside the HTML: A4 landscape, light theme, nav hidden, rows and charts
kept whole across pages. Chrome is on `lovelace` and `tolstoi` (deepfive has snap chromium), not
on `hamilton`; the tool runs it over ssh on `--host lovelace` when the local box has none.
Rasterise a few pages (`pdftoppm -r 45 -png`) and look at them before sending. Keep the HTML
next to the PDF as the source. If the analysis emits Markdown, convert it to HTML first:

```python
import markdown, base64, re, pathlib
md = pathlib.Path("report.md").read_text()
md = re.sub(r'!\[([^\]]*)\]\(([^)]+\.png)\)', lambda m:
     f'![{m[1]}](data:image/png;base64,{base64.b64encode(pathlib.Path(m[2]).read_bytes()).decode()})', md)
body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
```

Sections, in order: verdict up front (one paragraph + per-model table) ·
design & what deviated from spec · point-in-time audit · levels table (CORR,
MMC proxy, c.sharpe, payout, p.sharpe, dd, **calmar**, burn, n) · paired deltas
dev and test with NW t and CI · group ablations · per-model sensitivity ·
equal-budget retune · placebo/tenure controls · blends · missingness over
time · by-block and by-seed stability · gaps (missing arms, named) ·
recommended next steps · appendix per-block tables. Figures: cumulative CORR,
cumulative delta vs B0, rolling CORR, delta distribution, fold×seed grid,
model×group heatmap, blend sensitivity, missingness over time, ablation CIs,
prediction correlation with B0. Keep CSVs (`summary_levels`, `paired_deltas*`,
`blend_summary`, `scores_by_date_seed.parquet`) next to it.

When a study evaluates several feature sets, also build one PDF per set (`experiments/e132_pdf.py` is the
template: its own verdict, figures redrawn for that set, shared controls labelled as shared, every arm in an appendix).

In chat: verdict first, one static table, the leads, the caveats, then the
PDF. Afterwards update `experiments/EXPERIMENTS.md` (findings), the `STUDIES`
entry, and memory.

## Operational traps (from E131 and kin)

- Register the study (STUDIES + EXPERIMENTS.md row) before launch; check the
  next free E-id -- concurrent sessions collide.
- `src.cluster run` execs the command: no `VAR=` prefix. Units can be stopped
  externally and jobs die silently (quota, reboot) -- verify outputs exist per
  arm/refit before analysing, relaunch only the missing arms.
- User lingering is off on these hosts, so a host's `nmr-` units stop cleanly about 10 minutes after the last
  ssh session to it closes (`ControlPersist 10m`). No traceback, journal says `Stopping`. Hold a session open per
  host until the outputs exist (E132 lost 3 retunes twice this way).
- If a dataset drops rows, score every arm on the full universe and add a baseline fitted on the kept rows, so
  row handling and features are measured apart. Score deployment with the live rule, not a neutral 0.5: the base model
  ranks every coin and the candidate reorders only the coins it covers, inside their rank slots (`experiments/e134_route_feature_sets.py`, E134). A coin dropped for having no feed today makes the row set look-ahead.
  With no feed mask (E135, histfe), read covered from each arm's own columns on that date; that is warm-up, known at the time, and routing then moves few coins. A control arm can pass the bar on a near-zero test delta (E135 P_HIST, |t| < 0.5): the bar only asks for a positive sign, so report the size.
- Never edit a running runner script; only append.
- Don't upload, create models or stake as part of an evaluation.
