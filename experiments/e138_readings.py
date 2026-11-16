"""Per-study reading for e138_report.py, written AFTER each study's numbers were read.
The verdict labels in the report are computed from the bar; this text only explains them."""

READING = {
    "E138": dict(
        verdict="Routed, the way it would run live, the 55-feature set does not beat v2.0 on any model: every FULL cell is under |t| 2 on dev, "
                "and no arm reaches ADOPT. The two blocks fail in different ways. The macro block is inert on the linear models, as a date-constant "
                "column must be, and on LightGBM it costs dev CORR (REJECT). The trend block hurts the linear MLP on both windows (REJECT), and on "
                "LightGBM it is the only lead, below the bar. The audit is clean, so this is not leakage. Do not deploy the dataset.",
        leads=[
            "<b>Macro is inert on the linear models, harmful on LightGBM.</b> Ridge F_MACRO equals v2.0 to the fifth decimal, and ridge F_TF equals ridge FULL, "
            "which is the expected sanity check for a within-date-constant block. On LightGBM, F_MACRO is dev CORR −0.0113 (t −2.30) and test +0.0063 (t +2.56): "
            "the sign flips between windows, the pattern of a regime split fitted to the dev period. The yields and slopes carry it. Removing them (NO_RATES) "
            "brings LightGBM FULL back to v2.0 on dev (−0.0001), and FULL against NO_RATES is dev CORR −0.0084 (t −2.77).",
            "<b>The trend block is the only lead, on LightGBM only.</b> Routed F_TF: dev CORR −0.0004 (t −0.2), test +0.0045 (t +1.6), test payout +0.0044 (t +2.0). "
            "All 5 seeds are positive on test and only 1 of 5 on dev, so this is a test-window reading. It also does not clear the warm-up placebo: F_TF against P_WARM "
            "is test CORR +0.0081 (t +1.2). Unrouted, F_TF looks better (test payout +0.0090, t +2.8), but those rows feed no-feed coins a neutral 0.5, which E134 showed is a look-ahead row set.",
            "<b>The trend block hurts the linear MLP.</b> Routed F_TF is REJECT on the MLP (dev −0.0065, t −2.1; test −0.0080, t −2.9). On ridge it is dev −0.0024 (t −1.0) routed, so linear models get nothing from it.",
            "<b>Routing matters.</b> About 8% of coins are routed back to v2.0, and routing turns the unrouted LightGBM FULL test CORR of +0.0112 (t +2.3) into +0.0043 (t +1.6). "
            "Much of the unrouted gain comes from how the model treats coins without a feed, which is not something it could do live.",
            "<b>Look-ahead is worth something on dev.</b> P_NM, a flag for a coin that never gets a feed, earns LightGBM dev CORR +0.0054 (t +3.1) and payout +0.0084 (t +3.5), and nothing on test. "
            "That is the survivorship channel a coverage indicator would buy, which is why none is shipped. Tenure (P_HIST) again helps dev on all three models and is flat on test, as in E136/E137.",
            "<b>MLP retune.</b> With alpha capped at 1 the tuned net no longer collapses (minimum within-date spread 0.02). tuneFULL against tuneB0 is test CORR +0.0111 (t +1.9) and payout +0.0127 (t +2.9), "
            "with dev CORR +0.0045 (t +1.2). That is one seed and one test block, unrouted, and the fixed-config MLP shows the opposite sign, so it is noted but not a lead.",
            "<b>Blends do nothing:</b> the dev-picked weights are 0 to 0.05, and ADDED_ONLY alone is about −0.06 CORR on every model.",
        ],
        next=["If anything here is followed up, it is a forward test of LightGBM with the trend block only (F_TF, routed) on the 28 unread dates plus new rounds, as an unstaked candidate. Drop the macro block, rates first.",
              "E139 (inflation on top) and E140 (trend × macro interactions) use this dataset as their parent; read them against CM."]),
    "E139": dict(
        verdict="Adding the 12 break-even inflation features to carver-macro makes it worse. Routed against v2.0, FULL is REJECT on LightGBM (dev CORR −0.0115, t −2.6) and on the linear MLP "
                "(dev −0.0078, t −2.2), and on ridge it is exactly carver-macro, because a date-constant column cannot reorder coins in a linear model. "
                "Against its parent CM (E138's FULL), inflation costs LightGBM dev CORR −0.0043 (t −2.7), and payout falls on all 5 seeds on both windows. Do not deploy; drop the block.",
        leads=[
            "<b>Ridge is the sanity check:</b> FULL against CM is zero to the fifth decimal on every window, so whatever the other two models do with these columns, there is no cross-sectional information in them for a linear fit.",
            "<b>LightGBM uses them to split by regime, and it does not generalise.</b> F_INFL alone is dev CORR −0.0144 (t −3.6) and test payout −0.0027 (t −2.0). The same pattern as E138's macro block: a date-level split fitted to the dev period.",
            "<b>The linear MLP's moves are fit noise.</b> FULL against CM is dev payout −0.0118 (t −4.7), but the per-seed changes range from −0.021 to +0.009, with mixed signs on test. "
            "A linear net cannot use a date-constant input any more than ridge can, so these columns only change its optimisation path. This is the E137 lesson: a large t on one MLP fit per seed covers date noise, not fit noise.",
            "<b>Levels against changes:</b> the levels' own margin (FULL against NO_BEILVL) is negative, REJECT on LightGBM and the MLP, so keeping them hurts. FULL against NO_BEICHG reads PROMISING on LightGBM only because test CORR is +0.0016 (t +2.3) on dev −0.0002, a near-zero margin on a harmful parent.",
            "<b>The MLP retune is PROMISING by the letter</b> (tuneFULL against tuneB0: test CORR +0.0139, t +2.4; payout +0.0168, t +2.9; dev CORR −0.0010). In E138 the same retune without inflation gave +0.0111 on test, so it is mostly the retune on this test block, not inflation. One seed, unrouted: not a lead.",
            "Audit clean: 0 of 18,000 inflation cells differ at 5 cut dates, and 0 of 28.8 million cells shared with E138's dataset differ. The ADDED_ONLY, tenure and P_WARM / P_NM readings repeat E138's.",
        ],
        next=["Do not carry the inflation block into later datasets. E80/E81 (inflation on momswap-macro and on raw) were null as well; this is the third null."]),
    "E140": dict(
        verdict="The 270 trend x macro interactions are clearly harmful. Routed against v2.0, FULL is REJECT on all three models (dev CORR: ridge −0.0449, t −6.3; LightGBM −0.0191, t −3.9; MLP −0.0540, t −8.2), "
                "on test as well for ridge and the MLP, and against its parent CMI (E139's FULL) on both windows for every model. The equal-budget retune does not rescue it. The audit is clean, so this is not leakage. Do not deploy, and do not build further sign-flip interactions of this kind.",
        leads=[
            "<b>Why it fails.</b> A per-date rank ignores a positive scale and reverses under a negative one, so each interaction is its trend signal, reversed on dates when the macro signal is negative: "
            "mean |Spearman| with its own trend signal is 0.986. That is 270 near-copies of 15 trend signals whose direction flips with 18 regime signs. A linear model cannot fit a sign that changes from date to date, "
            "so it averages opposite-signed copies into noise, which is why ridge and the MLP lose 0.04 to 0.05 CORR.",
            "<b>LightGBM is hurt least</b> (it can split on the macro columns to undo the flip), but it is still REJECT. Against CMI, its test payout is −0.0100 (t −4.5), and all 5 seeds are negative on both windows.",
            "<b>Neither half works.</b> The return interactions (IX_RET) and the break-even interactions (IX_BEI) are each REJECT on ridge and the MLP against the parent, and neither is above the parent on LightGBM.",
            "<b>The interactions alone</b> (ADDED_ONLY, without the shipped features) lose 0.07 to 0.10 CORR, and every dev-picked blend weight is 0.",
            "Audit clean: 0 of 405,000 interaction cells differ at 5 cut dates. Coverage equals the trend block's (90.9% of key cells), so routing treats both blocks the same.",
            "This repeats, on stronger terms, E71/E79 (momswap-macro xinteractions, null).",
        ],
        next=["Close the macro-interaction line. If regime conditioning is tried again, give a tree the raw macro sign as a separate column, rather than multiplying it into every coin signal before binning."]),
    "E141": dict(
        verdict="Turning macro into a per-coin exposure does not help either. Routed against v2.0, the 32 rolling macro betas are REJECT on all three models: dev CORR ridge −0.0181 (t −3.8), "
                "LightGBM −0.0145 (t −3.1, 5/5 seeds lower, and lower on test too, −0.0245, t −2.5), MLP −0.0189 (t −4.0). No factor group and neither window clears the bar alone, "
                "and every dev-picked blend weight is 0. The audit is clean (0 of 48,000 beta cells differ when rebuilt from truncated data), so this is not leakage. Do not deploy the dataset.",
        leads=[
            "<b>The betas carry little rank information on their own.</b> ADDED_ONLY (the 32 betas without the shipped features) is about −0.08 to −0.12 dev CORR on every model, i.e. close to no signal. "
            "They are not redundant with the shipped features (max |Spearman| with a base feature 0.26) — they are new, just not predictive of 20-day rank.",
            "<b>Price betas (gold, silver, S&P, DXY) are the worst group:</b> F_PRICE is REJECT on dev and test for ridge and LightGBM (ridge test payout −0.0083, t −2.2; LightGBM test CORR −0.0159, t −3.4).",
            "<b>Rate betas move payout, not CORR.</b> F_RATES lowers dev CORR on every model (t −2.3 to −4.0) but raises payout where MMC carries it: LightGBM dev payout +0.0183 (t +3.2, 5/5 seeds) with flat test, "
            "MLP test payout +0.0151 (t +3.5, 5/5 seeds) with dev +0.0103 (t +1.6). Removing the rates from FULL (NO_RATES) costs payout on both windows. A rate-beta block makes the ranking less like the meta model "
            "while making it less correct, so it fails the bar on CORR; it is the only thing here worth a second look, and only as an MMC idea.",
            "<b>Window does not matter:</b> W60 and W120 are both REJECT on dev for every model, and adding the other window (FULL against W60 or W120) changes CORR by under 0.008 with no CORR t beyond 2.5.",
            "<b>Placebos repeat E138:</b> tenure (P_HIST) and the warm-up share (P_WARM) help dev only; the look-ahead no-feed flag (P_NM) earns LightGBM dev CORR +0.0058 (t +3.4) and nothing on test. "
            "FULL loses to P_WARM on dev for every model, so the block is worth less than knowing whether it is observed.",
            "<b>MLP retune:</b> tuneFULL against tuneB0 is dev CORR −0.0313 (t −3.7) with test payout +0.0190 (t +2.5): one seed, unrouted, opposite signs across windows — not a lead.",
            "Coverage: betas on 90.7% of keys; 6.1% are coins with no spot feed and 3.2% are warm-up (fewer than 80% of the window observed), all routed to v2.0.",
        ],
        next=["Close the macro line in its three forms tried here: raw levels/changes (E138/E139), sign-flip interactions (E140), per-coin betas (E141). None reorders coins usefully at the 20-day horizon.",
              "If anything is followed up, it is rate betas as an MMC-only candidate on LightGBM (F_RATES routed), forward-tested on the unread dates — expect lower CORR by construction."]),
    "E143": dict(
        verdict="Chronos-2-Synth&#39;s pre-head representations do not help, and on two of three models they hurt. Routed against v2.0, FULL (22 + 32 PCs) is REJECT on ridge "
                "(dev CORR −0.0060, t −2.2; dev payout −0.0102, t −2.1) and on the MLP (dev CORR −0.0079, t −2.5), and NO IMPROVEMENT on LightGBM, whose test is negative too "
                "(CORR −0.0060, t −1.9; payout −0.0072, t −2.0; 5/5 seeds lower on test). The audit is clean: rebuilding five dates from panels cut at each date reproduces "
                "all 2.1 million state values exactly, so this is not leakage. Do not deploy the dataset.",
        leads=[
            "<b>Strong standalone IC, no marginal value.</b> 13 of 32 PCs have per-date IC |t| &ge; 2 with the label (best ±0.06, t ±7), yet ADDED_ONLY is 0.04-0.08 dev CORR below v2.0 "
            "on every model and FULL is never above it. What the states know about a coin&#39;s past 20-day returns the 22 shipped features already carry, and the remaining variance is noise to the fit.",
            "<b>Covariates beat temporal-only, but only relative to each other.</b> F_COV beats F_AR on dev CORR on every model (ridge +0.0039 t +2.6, LightGBM +0.0023 t +2.0, MLP +0.0056 t +2.5), "
            "and F_AR is the worse block on all three (MLP dev CORR −0.0138, t −5.7; LightGBM test payout −0.0082, t −3.8). F_COV itself is still at or below v2.0 everywhere: "
            "letting the model see the features as covariates makes the representation less harmful, not useful.",
            "<b>Controls:</b> FULL loses to the warm-up placebo P_WARM on ridge and LightGBM, and FULL_H loses to tenure (P_HIST) on every model, both windows (MLP test CORR −0.0178, t −5.1). "
            "The look-ahead flag P_NM earns its usual dev gain (LightGBM dev CORR +0.0107, t +7.7), which is the survivorship channel, not the block.",
            "<b>Blends:</b> the dev-picked blend weight is 0 or 0.05 for ridge and the MLP; LightGBM picks 0.10 on dev (+0.0022 CORR, t +1.9), which loses on test (−0.0085, t −2.8).",
            "<b>Equal-budget retune</b> does not rescue it: tuneFULL against tuneB0 is ridge dev payout −0.0115 (t −2.2); LightGBM +0.0060 on dev (t +1.7) but −0.0141 on test (t −2.9); MLP −0.0103 dev, +0.0049 test, both |t| &lt; 2. One seed, unrouted.",
            "Coverage: states on 90.1% of scored keys; the rest is warm-up (fewer than 32 resolved labels) or the 60 burn-in dates. Routing therefore moves about 10% of coins back to v2.0.",
        ],
        next=["Together with E109/E113 (TabPFN, TimesFM, Chronos-2 on spot) and E118 x06, this closes time-series foundation-model representations as features: zero-shot states of a coin&#39;s own label history add nothing the shipped features lack.",
              "Untested variants, if the idea is reopened: the forecast itself (the head&#39;s quantiles) rather than the state, a fine-tuned Chronos-2 rather than zero-shot, or more than 16 PCs. None is recommended on this evidence."]),
    "E144": dict(
        verdict="Fine-tuning Chronos-2-Synth on the panel does not rescue its representations. By the study&#39;s bar the result is NO IMPROVEMENT overall and REJECT on the MLP "
                "(routed dev CORR −0.0055, t −2.1), but the substance is worse than null on the other two: LightGBM&#39;s test is significantly harmed "
                "(CORR −0.0101, t −4.4; payout −0.0108, t −5.2; 5/5 seeds lower), more than with E143&#39;s zero-shot states, and ridge&#39;s dev payout is −0.0093 (t −2.5). "
                "The bar escapes REJECT on those two only because it reads dev CORR (LightGBM +0.0016, t +0.8; ridge −0.0045, t −1.8). "
                "The audit is clean: with each block&#39;s model held fixed, panels cut at five dates reproduce all 2.1 million state values exactly, so this is not leakage. Do not deploy the dataset.",
        leads=[
            "<b>Against E143&#39;s zero-shot states</b> (same protocol, same routing): LightGBM FULL test CORR −0.0101 (t −4.4) here against −0.0060 (t −1.9) zero-shot, and F_AR −0.0138 (t −7.2) against −0.0081 (t −3.6). "
            "The MLP is less negative on dev (−0.0055 against −0.0079) and ridge about the same (dev payout −0.0093 against −0.0102). Ten LoRA fine-tunes per mode, about 8 GPU-hours, bought nothing on the metric that pays.",
            "<b>LightGBM looks positive on dev and then breaks.</b> The equal-budget retune shows it most clearly: tuneFULL against tuneB0 is +0.0045 dev CORR (t +2.0) and +0.0080 dev payout (t +2.0), "
            "then −0.0262 test CORR (t −4.3) and −0.0200 test payout (t −5.3). By block, LightGBM FULL swings from +0.0207 (block 3) to −0.0169 (block 11): a booster that splits on block-specific axes picks up "
            "what each block&#39;s fine-tune happened to learn, and that does not carry forward.",
            "<b>The features change meaning every block.</b> Each block has its own models and its own PCA basis, so PC 1 in block 5 is not PC 1 in block 6. Pooled over the file, only 1 of 32 PCs reaches "
            "|t| &ge; 2 on per-date IC (cov PC 1, +0.048, t +3.9) against 13 of 32 for E143&#39;s fixed basis. That is the cost of a point-in-time fine-tune: a fixed axis would need one model fitted on future labels.",
            "<b>F_COV is again the less harmful block.</b> F_COV beats F_AR on LightGBM (test payout +0.0152, t +6.0) and on ridge and the MLP on test (payout +0.0079 t +3.2; +0.0083 t +2.4), "
            "but F_COV is still at or below v2.0 everywhere (routed LightGBM test payout −0.0015, t −0.6).",
            "<b>Controls:</b> FULL loses to the warm-up placebo P_WARM on ridge (dev payout −0.0150, t −3.2) and LightGBM on test, and FULL_H loses to tenure (P_HIST) on every model in both windows (LightGBM dev only −0.0008, t −0.5). "
            "P_NM shows its usual look-ahead dev gain (LightGBM dev CORR +0.0154, t +6.6).",
            "<b>One cell is positive and is not a lead:</b> the MLP tuneFULL is +0.0114 test payout (t +3.3), but its dev payout is −0.0057 and dev CORR −0.0163 (t −2.9). It is one seed and unrouted, and E137 showed a tuned MLP&#39;s test payout moving by 0.018 on one feature.",
            "<b>Blends:</b> ridge and the MLP pick w = 0 on dev payout. LightGBM picks 0.20 on dev (CORR +0.0047, t +2.5), which loses on test (CORR −0.0185, t −3.4; payout −0.0174, t −4.0).",
            "Coverage: states on 75.8% of scored keys. The rest is warm-up, plus blocks 0 and 1 (2020-08-28 to 2021-08-26), which have no fine-tuned model: block 1&#39;s cut leaves about six months of labels, "
            "short of the 134-step minimum. Their columns are a constant 0.5, so FULL equals B0 there on ridge and LightGBM. Restricting dev to blocks 2-10 does not change the reading (routed LightGBM FULL dev CORR +0.0019, t +0.8; ridge payout −0.0093, t −2.5).",
        ],
        next=["With E143 this closes Chronos-2 representations as features, zero-shot and fine-tuned. Together with E109/E113 and E118 x06, no time-series foundation-model state has added marginal value over the 22 shipped features.",
              "If the idea is reopened, test the fine-tuned model&#39;s FORECAST (its 20-day quantiles, one stable meaning in every block) as a feature or a blend leg, not its hidden state. That is the only variant left that does not change meaning between blocks."]),
    "E146": dict(
        verdict="Keeping all 3,072 Chronos-2-Synth dimensions instead of 16 PCs makes the representations clearly harmful. REJECT on all three models by the study&#39;s bar "
                "(routed dev CORR: ridge −0.0514, t −7.3; MLP −0.0489, t −7.5; LightGBM −0.0095, t −2.5). On ridge and the MLP the loss is about the size of deleting the shipped volatility group, "
                "and it shows in every one of the 12 blocks and on all 5 MLP seeds. LightGBM is near-null on dev (payout +0.0026, t +0.5) and harmed on test (CORR −0.0190, t −5.3; payout −0.0169, t −5.0; 5/5 seeds lower). "
                "The audit is clean: the raw states pass the E143/E144 truncation audits, and re-binning them reproduces the dataset cell for cell. This is a capacity/noise result, not leakage. Do not deploy the dataset.",
        leads=[
            "<b>Dims lose to their own PCs.</b> Every state is worse with 768 dims than with its 16-PC summary from E143/E144, unrouted. Ridge dev CORR runs from −0.034 to −0.041 (t −7 to −8) per state, "
            "the MLP from −0.030 to −0.040, and LightGBM from −0.000 to −0.007 on dev and −0.008 to −0.012 on test. The PCs were already null-to-harmful, so the extra 752 dims per state carry noise, not signal.",
            "<b>Fine-tuned against zero-shot: no gain.</b> F_FT against F_ZS is within |t| &lt; 2 on dev CORR for every model (zero-shot minus fine-tuned: ridge +0.0003, MLP +0.0041, LightGBM −0.0032). "
            "FULL (both families) is below F_ZS alone on ridge (dev CORR −0.0159, t −7.7) and the MLP (−0.0139, t −4.0), so stacking the second family adds more harm, not information.",
            "<b>Covariates against label-only (cov against ar):</b> no consistent difference. Zero-shot ar minus cov is within |t| &lt; 2 on dev for every model. Fine-tuned cov is less harmful than fine-tuned ar on LightGBM "
            "(dev CORR +0.0068, t +3.0; test +0.0185, t +3.5), the same ordering E143/E144 saw for the PCs.",
            "<b>LightGBM dev is flattered by drawdown, not level.</b> Unrouted FULL has a dev calmar of 0.070 against 0.023 for v2.0 (dd(pay) 1.13 against 3.01) at almost the same payout. "
            "On test, though, payout falls from 0.0389 to 0.0127 and calmar from 0.029 to 0.008. The dev drawdown gain does not carry forward. The best dev blend (w 0.3 on payout) also loses on test (payout −0.0188, t −2.8).",
            "<b>Equal-budget retune</b> does not rescue it. tuneFULL against tuneB0: ridge dev CORR −0.0494 (t −6.2), MLP −0.0666 (t −7.0), LightGBM −0.0041 dev (t −0.9) and −0.0378 test (t −4.5). One seed, unrouted.",
            "<b>Controls:</b> FULL loses to the warm-up placebo P_WARM on every model, and FULL_H loses to tenure (P_HIST) on every model in both windows. P_NM shows its usual look-ahead dev gain "
            "(LightGBM dev CORR +0.0103, t +7.8), which is why routing reads observed coverage, not the file-wide flag.",
            "<b>Scale from the shipped groups.</b> Removing volatility from v2.0 costs dev CORR −0.031 (LightGBM), −0.053 (MLP) and −0.052 (ridge). No other shipped group moves dev CORR by more than 0.008 on any model "
            "(momentum, reversion, volume, price, size). Adding the dims costs ridge and the MLP about as much as losing volatility.",
            "Coverage: zero-shot states on 90.1% of scored keys and fine-tuned on 75.8% (none in blocks 0-1). Routing therefore sends about 10% of coins (zero-shot arms) or 25% (fine-tuned arms) back to v2.0.",
        ],
        next=["With E143 and E144 this closes Chronos-2 hidden states as features, as PCs or as raw dims, zero-shot or fine-tuned. More dimensions made it worse, not better.",
              "The one untested variant is still the fine-tuned model&#39;s FORECAST (its 20-day quantiles) as a feature or blend leg. Nothing here argues for it beyond E144&#39;s note."]),
    "E148": dict(
        verdict="No residual-path forecast improves v2.0. By the bar the result is NO IMPROVEMENT, and REJECT on ridge and the MLP: adding all six signals moves routed dev CORR by "
                "−0.0042 (t −2.2) on ridge, −0.0052 (t −2.2) on the MLP and −0.0004 (t −0.2) on LightGBM. No single signal raises dev CORR on any model. "
                "The Chronos-2 residual signals do give small dev payout gains (C2 RES +0.0039, t +2.7 on ridge; +0.0049, t +3.0 on LightGBM), but the warm-up placebo, "
                "which only says how much of the block is observed, earns as much (+0.0048 ridge, +0.0107 LightGBM), and FULL loses to it on dev CORR on every model. "
                "The audit is clean: rebuilding every path from spot cut at D−1 reproduces all four Chronos-2 arms exactly on three dates. Do not deploy the dataset.",
        leads=[
            "<b>Where the signal comes from (Part A, each signal alone).</b> Residualising the input is the one component that pays: Chronos-2 on the residual path beats it on the raw path by "
            "+0.038 payout on dev (t +4.0) and +0.024 on test (t +3.1), almost all through MMC (+0.035, t +4.1). Adding BTC, ETH and the peer index as covariates changes nothing "
            "(COV − RES payout −0.004, t −1.3), nor does grouped size-bucket context (GRP − RES −0.001, t −0.4; GRP − COV +0.003, t +0.9): the three residual forecasts correlate 0.93 to 0.95 per date. "
            "Encoding with Chronos-2 does not beat a ridge on the same inputs (COV − RIDGE dev payout +0.014, t +0.5; test CORR −0.032, t −1.8).",
            "<b>The same ordering survives inside v2.0 (Part B), at a tenth of the size.</b> RES over RAW is +0.0039 dev payout on ridge (t +2.3; test +0.0020, t +2.1) and +0.0051 on LightGBM (t +2.5). "
            "COV over RES is null or negative (ridge −0.0016, t −2.9). GRP and COV beat RIDGE on payout for LightGBM (+0.0043, +0.0040, t +2.6) and the MLP (GRP +0.0069, t +2.7), "
            "mostly because the ridge signal itself hurts as a feature.",
            "<b>The ridge signal is the best standalone forecast and the worst feature.</b> Alone it has the highest test CORR of the six (0.051, t +2.2) and the best dev calmar (0.019), "
            "and it is nearly uncorrelated with the Chronos-2 forecasts (per-date Spearman 0.02 to 0.04). Added to v2.0 it is REJECT on all three models (dev CORR −0.0023, −0.0019, −0.0065, |t| ≥ 2.4).",
            "<b>Residual momentum is reversal-signed</b> (standalone CORR −0.032, t −3.8) and REJECT as a feature on ridge and LightGBM.",
            "<b>Controls:</b> tenure (P_HIST), warm-up (P_WARM) and the look-ahead flag (P_NM) all reach ADOPT by the letter, as they do in every study; they are controls. FULL loses to P_WARM on dev CORR "
            "on all three models (ridge −0.0094, t −3.8) and FULL_H loses to tenure on ridge and the MLP. The one unrouted ADOPT, C2 RAW on LightGBM (dev CORR +0.0022, t +2.0; test +0.0001), is one cell in 18 at the edge of the bar, "
            "and routed it is +0.0019 (t +1.8).",
            "<b>Equal-budget retune:</b> LightGBM tuneFULL is +0.0102 dev payout (t +3.9) but +0.0038 on test (t +0.8); the MLP loses dev CORR (−0.0164, t −2.3). One seed, unrouted. "
            "<b>Blends:</b> ridge and the MLP pick w = 0; LightGBM picks 0.5 of the six-signal model on dev payout, +0.0055 (t +1.1), test +0.0165 (t +1.3).",
            "Coverage: the Chronos-2 residual signals and the ridge exist on 85.9% of scored keys, the raw-path forecast on 89.2%. The rest has no spot feed, too short a history for a 120-day beta, or is BTC or ETH.",
        ],
        next=["Residualising on BTC and ETH is the one component that carried signal, and it carried MMC, not CORR. If the idea is reopened, test the residual forecast as a blend leg or against "
              "the meta-residualised book (E107), not as a feature over v2.0.",
              "Peer covariates and grouped context add nothing measurable to Chronos-2 here; there is no case for a fine-tuned version of this setup on this evidence."]),
    "E149": dict(
        verdict="The proposed realised-volatility block does not help v2.0 and hurts the linear models. REJECT on ridge (routed dev CORR −0.0138, t −2.9; test −0.0070, t −3.1) and the MLP (−0.0179, t −4.3); "
                "LightGBM is null on dev (−0.0019, t −0.6) and loses on test (payout −0.0100, t −2.7). No sub-block (levels, own-history ranks, any estimator family, short or long windows) raises dev CORR on any model. "
                "The shipped feature_volatility_20d/60d were kept in the base throughout; dropping them (F_SWAP) costs a further −0.0105 to −0.0201 dev CORR, so the block cannot stand in for them. Builder audit clean.",
        leads=["Levels alone are the least harmful part (ridge −0.0049, t −1.7; LightGBM dev payout +0.0054, t +1.8, test −0.0022); the 132 own-history ranks do the damage (ridge −0.0113, t −3.6).",
               "FULL loses to the warm-up placebo P_WARM on every model and FULL_H loses to tenure on ridge and the MLP: whatever the block knows, its coverage pattern knows more.",
               "Equal-budget retune does not rescue it (ridge tuneFULL dev payout −0.0261, t −3.0)."],
        next=["Volatility is already the strongest shipped group (E147); re-estimating it with other estimators and windows adds nothing over the two shipped columns."]),
}
