# Experiment index

One row per numbered study. The ID is the reference handle: every artefact a
study writes carries it, so `e01_*` retrieves the whole thing — runs, logs and
report — in one glob, years later.

| ID | Name | Runner | What it asks | Arms | Status |
|---|---|---|---|---|---|
| **E01** | LightGBM feature blocks on the 60-day target | `e01_blocks60_lgb_reg_crypto.sh` | Does any feature block beat raw when the label is `target_binned_return_60` instead of `_20`? Same 23 arms, same pinned estimator, same cadence — the label is the only thing varying. | 23 | done, 23/23, 38 min |

## What each ID owns

```
results/{metrics,forecast,per_period,hyperparams}/crypto/<model>/e01_*
logs/e01_*.log            one per arm
logs/e01_report.txt       the final table
experiments/e01_*.sh      the runner
e01.txt                   console log of the run
```

## Naming rule

`E<NN>` plus a one-line description, recorded here before the study launches.

An ID beats a descriptive prefix because a description stops being unique the
moment a second study shares it — `b60_lgb_reg_*` names a configuration, and the
next 60-day LightGBM study would match the same glob. It also beats a codename:
`E01` sorts, increments, and tells you the order things were run in.

Run names inside a study stay descriptive: `e01_<arm>_m<leaf>` reads as
"study E01, this arm, this leaf size".

## E01 — findings

Newey-West at **lag 60**, not 20: a 60-day target overlaps ~98% between
consecutive daily periods, so 470 periods are roughly **eight** effective
independent windows. Nulls are the expected outcome.

| arm | leaf | payout | Δ vs raw | t |
|---|---:|---:|---:|---:|
| `fencglobal` ▲ | 313 | 0.162860 | +0.078671 | **+3.65** |
| `allfeatpp` | 50 | 0.093312 | +0.014690 | +1.10 |
| `tenc_onehot` | 50 | 0.093018 | +0.014397 | +1.06 |
| `raw` | 313 | 0.084189 | baseline | — |
| `interact` | 313 | 0.077502 | −0.006687 | **−3.56** |
| `fenccausal` | 313 | 0.064105 | −0.020084 | **−2.24** |

1. **`interact` reverses sign between labels.** +0.005710 (t = +2.99) on the
   20-day target, −0.006687 (t = −3.56) here. Same columns, same pinned model,
   same cadence. Combined with it not reproducing at leaf 50 (+2.99 → +0.36),
   the 20-day result reads as conditional on the label and the leaf size rather
   than a property of the columns. **SACI_01 is deployed on that 20-day result.**
2. **`fencglobal` gets worse, not better.** +2.83 on 20-day, +3.65 here — its
   lookahead strengthens with horizon, exactly as `build_feature_encoding`'s
   docstring predicts: 62% of tokens do not survive three years, and a
   whole-sample profile partly states whether the token survived. Still not
   deployable.
3. **`fenccausal` is consistently harmful** on both labels (−2.15, −2.24).
4. **Every leaf-50 wide arm is positive** (+0.83 to +1.10) and none significant.
5. **The 60-day label is far riskier.** Max drawdowns run 4.7–7.9 against
   0.2–1.2 for the same datasets on the 20-day label.
