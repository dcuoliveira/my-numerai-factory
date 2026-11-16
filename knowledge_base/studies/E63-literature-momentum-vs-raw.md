# E63 — Literature crypto momentum features vs the raw 22

- **State:** not started (0/6 scouted) — built and registered but never run (multi-hour), per memory
- **Runner:** experiments/e63_momentum_features_crypto.sh
- **Models / datasets:** models `lgb`, `ridge_glmnet`; datasets v2.0 (base 22) vs v2.0-litmom (22 base + 17 literature momentum) vs momentum-only (^feature_mom_)
- **Artifacts on disk:** 0.0 MB across 0 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: litmom-momentum-feature-pipeline.md, e64-ktz-litmom-bakeoff.md, e66-momentum-proxy-swap.md, litmom-eda-signal-verdict.md; metrics: none on disk

## Main question
The 22 shipped features are rank-binned and hide the price path. Does a 17-feature literature-momentum block (rebuilt from spot prices, binned to Numerai's {0,.25,.5,.75,1}, merged point-in-time) add ranking info the binned features lack — i.e. does base+mom beat matched base on numerai_corr at paired |t|>=2?

## What was tried
Planned 6 arms, matched protocol (rolling 130/130, embargo 20, 5 folds, 50 draws, seed 42): {lgb, ridge_glmnet} x {base v2.0, mom v2.0-litmom, momonly ^feature_mom_}. The 17 features (Liu&Tsyvinski, Grobys&Sapkota, Tzouvanas, Li, Tzotchev-2018 robust trend, etc.) land on Numerai's exact scale and merge point-in-time (row D uses spot closes <= D-1). momonly is the standalone diagnostic. Per memory (litmom-momentum-feature-pipeline.md) the study was registered but "NOT yet RUN" and not wired live; the e63_ prefix produced 0 runs.

## Trained parameters
config JSON not on disk; no runs were produced (0/6). The dataset itself (crypto/v2.0-litmom) is 513,496 rows x 39 features (22 shipped + 17 feature_mom_).

## What was found
MEMORY-BACKED — E63 produced no runs of its own, but its scientific question is answered by its pre-run EDA and by the sibling studies that ran the same litmom-vs-base test across more models:
- Pre-run EDA (litmom-eda-signal-verdict.md): all momentum features are NEGATIVELY signed on both targets (short-horizon crypto REVERSAL, not momentum); the flashy MAX/volscaled features are ~volatility/Sharpe proxies (redundant); the real incremental signal is the medium-horizon RETURN ladder (mom_ret_28d/56d/14d/21d, partial |IC| 0.014-0.016 at |t| 4.2-4.6 after residualizing on the base 22; 8/10 clear |t|>=2 incrementally). Expected win, if any, is small (~0.015 incremental IC) and comes from the return ladder, not MAX/volscaled.
- Downstream (e64-ktz-litmom-bakeoff.md, e66-momentum-proxy-swap.md): ADDING litmom is null-to-negative across model classes, attributed to the new momentum being redundant with the base's own 8 momentum proxies (feature_momentum/rsi/sharpe/bollinger_{20,60}d).

Direction: null-to-weak. The block is also designed-not-wired for live (src/live/features.py never imports the momentum modules).
