## Reading

**What changed from the CORR report:** every panel now uses MMC instead of CORR.
- P1: held-out MMC minus the same-window ridge's MMC, every fold × every epoch.
- P2: the best epoch is picked on MMC.
- P3: the MMC gap at the kept epoch (500).

**Which folds remain.** MMC exists only where the meta model does: 547 days, from 2024-05-07.
- r08–r11 are covered in full (120–130 days each).
- r07 is scored on its last 37 days.
- r00–r06 have no meta-model day and are left out.

**Check:** held-out MMC was rescored at every epoch from the saved predictions. Where the training log also recorded it (r08–r11, both seeds), the two agree to 3e-17.

**In one line: the same answer as for the −CORR network.** On CORR this network beats the ridge in 4 of the 5 folds (r10 is level at −0.0003). On MMC it beats it in 2, and they are the same two folds, r09 and r11. MMC removes the part of the network's edge that leans toward the meta model, so only about a quarter of the CORR edge reaches MMC.

| fold | meta days | CORR gap, same days | MMC gap @500 | MMC gap @stop | best epoch on MMC (flat range) |
|---|---|---|---|---|---|
| r07 | 37 | +0.0013 | −0.0012 | +0.0600 | 81 (81–81) |
| r08 | 130 | +0.0032 | −0.0025 | −0.0383 | 81 (15–81) |
| r09 | 130 | +0.0066 | +0.0064 | −0.0292 | 379 (21–497) |
| r10 | 130 | −0.0003 | −0.0053 | −0.0235 | 71 (4–71) |
| r11 | 120 | +0.0112 | +0.0069 | −0.0215 | 27 (27–27) |

- **P1 and P2: under MSE, MMC peaks late.**
  - Best epochs on MMC have a median of 81 [IQR 71, 81]; r09's is at 379.
  - That matches MSE's slow ranking, which settles only around epoch 190.
  - Seed 43's best epochs fall anywhere from 14 to 137.
- **P3: level with the ridge at epoch 500, far below it at the stop rule.**
  - At epoch 500 the MMC gap has a median of −0.0012 [IQR −0.0025, +0.0064]; 3 of 5 folds are at or below zero.
  - At the stop-rule epoch (16–27) the median is −0.0235 [−0.0292, −0.0215], with 4 of 5 folds far below zero. Stitched, stopping there gives MMC 0.046 instead of 0.070, and calmar 0.011 instead of 0.063.
  - So on MMC, as on CORR, MSE needs the full run. (r07's +0.060 at its stop epoch rests on 37 days.)
- **P6: why CORR and MMC disagree by fold.**
  - The two gaps share a sign in 3 of 5 folds and in 22 of 25 calendar months. The daily gaps correlate +0.93.
  - Daily fit: **MMC gap = −0.0026 + 0.76 × CORR gap.**
  - Network and ridge rank the coins almost identically (per-day correlation 0.974). The network's departure from the ridge correlates +0.082 with the meta model, and its correlation with the meta model is 0.533 against the ridge's 0.516.
  - Over the 547 days, a CORR gap of +0.0048 becomes an MMC gap of +0.0011 and a payout gap of +0.0016 (t +0.6).

**What it means.** For both losses, the fold-level edge over the ridge is a CORR edge. On MMC, which is most of payout, both networks are level with the ridge and win the same two folds. That points to the linear map the networks share with the ridge, not the loss. The part of the edge that MMC does not pay for is the meta-leaning part that E106's residualisation removes.

**Caveats:**
- Five folds, one of them 37 days long, make fold counts weak evidence. The conclusion rests on the monthly and daily views.
- Neighbouring months share information through the 20-day target, so 25 months are not 25 independent tests.
- The table is seed 42. Seed 43 appears only in P2 (hollow markers) and in the churn floor.
