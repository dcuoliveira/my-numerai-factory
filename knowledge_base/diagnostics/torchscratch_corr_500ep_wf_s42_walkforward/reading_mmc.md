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

**In one line: the fold-level story does not survive the switch. On CORR the network beats the ridge in all 5 of these folds; on MMC, in only 2 (r09 and r11).** The two gaps are closely linked, and move together day by day. But MMC removes the part of the network's edge that leans toward the meta model, and that costs about 0.0024 a day. Only about a quarter of the CORR edge reaches MMC.

| fold | meta days | CORR gap, same days | MMC gap @500 | MMC gap @stop | best epoch on MMC (flat range) |
|---|---|---|---|---|---|
| r07 | 37 | +0.0052 | −0.0004 | +0.0232 | 2 (2–2) |
| r08 | 130 | +0.0030 | −0.0010 | −0.0061 | 15 (2–498) |
| r09 | 130 | +0.0054 | +0.0056 | +0.0047 | 1 (1–497) |
| r10 | 130 | +0.0053 | −0.0017 | +0.0020 | 11 (3–310) |
| r11 | 120 | +0.0081 | +0.0037 | +0.0014 | 4 (4–485) |

- **P1 and P2: MMC is set within the first few epochs, then barely moves.**
  - Best epochs on MMC have a median of 4 [IQR 2, 11].
  - Three folds stay within 0.005 of their best for almost the whole run, so the exact peak is noise. Seed 43 puts it anywhere from epoch 8 to 361.
  - The CORR edge is also set within about 5 epochs (see the CORR report).
- **P3: level with the ridge on MMC.**
  - At epoch 500 the MMC gap has a median of −0.0004 [IQR −0.0010, +0.0037]; 3 of 5 folds are at or below zero.
  - At the stop-rule epoch (29–116) the median is +0.0020 [+0.0014, +0.0047], with 4 of 5 folds above zero.
  - Stitched over the 547 days, the stop rule gives MMC 0.0707 against 0.0701 at epoch 500, and calmar 0.065 against 0.056. The payout difference from the ridge is not significant at either epoch (t +0.8 and +1.0).
- **P6: why CORR and MMC disagree by fold.**
  - The two gaps share a sign in only 2 of 5 folds, but in 21 of 25 calendar months. The daily CORR and MMC gaps correlate +0.90.
  - Daily fit: **MMC gap = −0.0024 + 0.71 × CORR gap.**
  - The cause: network and ridge rank the coins almost identically (per-day correlation 0.973, in the rank-gaussianised space MMC uses). Where the network departs from the ridge, it leans toward the meta model.
    - That departure correlates +0.093 with the meta model.
    - The network's correlation with the meta model is 0.536, against the ridge's 0.516.
    - MMC removes the meta-model component before it pays, so that part of the CORR edge earns nothing.
  - Over the 547 days, a CORR gap of +0.0054 (t +2.5) becomes an MMC gap of +0.0014 and a payout gap of +0.0020 (t +1.0).
  - A fold needs a CORR edge of about 0.0034 (0.0024 / 0.71) just to break even on MMC. r08 (+0.0030) falls short; r07 and r10 sit at that line and land below it.

**What it means.** "Beats the ridge in 11 of 12 folds" is a CORR result. On MMC, which is most of payout, the network is level with the ridge. It also explains why E106 works: residualising on the meta model removes exactly the meta-leaning part of the prediction. The MSE network shows the same pattern, and beats the ridge on MMC in the same two folds (see its report).

**Caveats:**
- Five folds, one of them 37 days long, make fold counts weak evidence. The conclusion rests on the monthly and daily views.
- Neighbouring months share information through the 20-day target, so 25 months are not 25 independent tests.
- The table is seed 42. Seed 43 appears only in P2 (hollow markers) and in the churn floor.
