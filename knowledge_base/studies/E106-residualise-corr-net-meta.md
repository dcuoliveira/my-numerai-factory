# E106 — Residualise the −CORR network on the meta model (no training)

- **State:** done (202/202 arms)
- **Runner:** experiments/e106_meta_residualise_crypto.py
- **Models / datasets:** meta-residualisation transform over the from-scratch −CORR network (model dir torch_scratch, identity 64-32-16, alpha 0, epoch 500; source torchscratch_corr_500ep_wf, seeds 42/43); no new dataset, operates on stitched held-out forecasts
- **Artifacts on disk:** 86.4 MB across 21 forecast runs (plus 202 metrics + 202 per_period series) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E106 — residualising on the meta model raises payout, and a 5-day-old meta model keeps most of it"; memory: e106-meta-residualise.md; metrics: results/metrics/crypto/torch_scratch/e106_lag20_s075_s42.json, e106_base_s000_s42.json

## Main question
Does moving the −CORR network's predictions off the meta model per day — in the space MMC is computed in — raise payout, and does a deployable (lagged, non-oracle) meta model keep the gain?

## What was tried
No training. Per day on 547 meta days: `rank(gauss(rank(p)) − s·β·gauss(rank(m)))`, β = p·m/m·m, with s swept 0..1 in 21 steps. The same-day meta model is an oracle (published only after the round closes); four deployable arms use the latest meta model at least 1/5/20/30 days old. Every arm scored against the same-day meta model, paired against s=0 of its seed (Newey-West lag 20). 2 seeds × (1 + 5 variants × 20 strengths) = 202 runs. Gate: s=0 reproduces the source per-period CORR/MMC to 1e-12 (required casting the target back to float32).

## Trained parameters
No model fit. From results/metrics/crypto/torch_scratch/e106_lag20_s075_s42.json config: model meta_resid applied to source_run "torch_scratch/torchscratch_corr_500ep_wf_s42_ep500"; study E106; target target_binned_return_20; task regression; meta_variant "lag20", days_unadjusted 14, strength 0.75, seed 42; transform "rank(gauss(rank(p)) − s * beta * gauss(rank(m))), per day"; n_periods 547. The base arm (e106_base_s000_s42.json) is strength 0.0, meta_variant "same" — CORR.mean 0.14466, MMC.mean 0.07013; at lag20 s=0.75 the arm scores CORR.mean 0.12770, MMC.mean 0.07698.

## What was found
FINDINGS-BACKED. Residualising raises payout; a 5-day-old meta model keeps most of it. Best strength per arm (payout gain over s=0, seed 42 / seed 43): same-day oracle +0.0081/+0.0075 (t +3.6/+3.5) at s≈0.75; ≥1 day +0.0073/+0.0068 (t +3.3/+3.2); ≥5 days +0.0069/+0.0064 (t +2.9/+2.8) at s=0.80, every s clearing t>=2 on both seeds; ≥20 days +0.0053/+0.0050 (t +1.7/+1.8), only s<=0.70 clearing; ≥30 days +0.0021/+0.0019 (t +0.7/+0.7, null). Mechanism: MMC re-ranks/gaussianises/orthogonalises but does not rescale, so removing the meta component scales up the part MMC pays; under the oracle at s=0.75 CORR falls 0.145→0.110 (t −6.7) while MMC rises 0.070→0.082 (t +5.4). It buys level, not smoothness: at 5-day lag s=0.75, dd(pay) 1.51→1.78, calmar 0.056→0.051. Against the book (470-day window) the 5-day s=0.75 arm pays 0.0919, tying SACI_06 (0.0920, t −0.02) but calmar 0.051 vs 0.087. The network's per-day rank correlation with the meta model is 0.52, β median 0.53; the meta model is published within ~1.5–2.5 days. Caveats in the record: one source model, ~27 independent windows, strength picked from 21 steps; `src/live` does not download the meta model, so deployment needs a per-round fetch.
