# Counterparty Credit Risk (CVA): An ML Portfolio Project

I'm building a system that estimates counterparty credit
risk (CVA) by combining a default-probability classifier, a Monte-Carlo-based
exposure simulator, and a deep-learning-accelerated pricer.

This is is an ML project applied to finance, not a finance project with
ML sprinkled in. Every phase below is a standard ML problem type -
classification, regression, time-series estimation, deep learning - with the
finance content as the applied domain, not the organizing structure.

## Problem decomposition

| Finance concept | ML problem type | Techniques |
|---|---|---|
| Probability of Default (PD) | Binary classification + probability calibration | Logistic regression & gradient boosting (XGBoost/LightGBM); Merton structural model used as an engineered feature, not the final answer |
| Expected Exposure, EE(t) | Regression / conditional expectation estimation under uncertainty | Polynomial OLS baseline, random forest, gradient boosted regression, small MLP - benchmarked against simulation-based ground truth |
| Interest rate dynamics | Time-series parameter estimation | AR(1)-style regression (method of moments), residual diagnostics |
| Fast derivative pricing | Physics-informed deep learning / scientific ML | PINN in PyTorch - custom PDE-residual loss, autograd, no labeled training data |
| Scarcity of real default data | Simulation-based data augmentation | Monte Carlo to generate a large, well-understood synthetic dataset with known ground truth |
| Everything above | Rigorous evaluation methodology | Time-based train/test splits, held-out validation, ROC-AUC/PR-AUC/calibration curves, MAE/RMSE against ground truth |

## Results

All 10 phases are implemented, tested (90 pytest tests, all offline except
the live-data ingestion functions), and runnable end-to-end via
`config.yaml`. A risk-focused extension pack (bootstrapped CVA confidence
interval, wrong-way risk, external spread benchmarking, MRM documentation
-- see [Risk-focused extensions](#risk-focused-extensions)) builds on top
of the base pipeline. Headline numbers from the current run (valuation date
2026-08-15):

| Phase | Result |
|---|---|
| 1. Data engineering | 10-company panel from FRED + yfinance + SEC EDGAR; found and fixed a stale-XBRL-tag bug (AT&T) during the initial build |
| 2. Merton PD baseline | AMC correctly flagged as far riskiest (6.8% 1y PD); investment-grade names collapse to implausible 1e-40-to-1e-76 PDs - a known Merton-with-physical-inputs limitation, not a bug |
| 2b. Supervised PD classifier | XGBoost ROC-AUC 0.95 / PR-AUC 0.44 vs. logistic regression's 0.86 / 0.26 on the Taiwan bankruptcy dataset; found and fixed a calibration-overconfidence issue (Platt scaling) |
| 3. Hull-White calibration | a=0.373, sigma=1.68% from 2,088 days of real SOFR; theta(t) reproduces the input curve exactly (closed-form-verified) |
| 4. Synthetic exposure dataset | 45,000 (state, noisy-label) rows from 5,000 simulated paths; unbiasedness confirmed against the analytic conditional expectation |
| 5. Exposure model comparison | Naive baseline MAE 267,699 vs. polynomial OLS's 3,482 against a nested-simulation ground truth; found and fixed a silent MLP-undertraining bug (target scaling) along the way |
| 6. LGD | 60% (40% recovery, ISDA CDS Standard Model convention) |
| 7. CVA assembly | $19,073 (AMC, best exposure model) vs. $26,907 (naive) - a **29% swing from exposure-model choice alone**; CVA moves strictly monotonically under PD and recovery-rate shocks |
| 8. PINN pricer | Mean abs price error 0.0032 against the closed form; error concentration at t=0 confirmed quantitatively (correlation 0.708) rather than eyeballed; found and fixed a 10-minute test-suite hang (torch thread contention) |
| 9. Portfolio application | Spearman rank correlation 0.767 (p=0.010) between Merton-implied CVA rank and approximate credit-rating rank across the panel - directionally right, with one honest, explained outlier (Ford) |

See [Methodology](#methodology) below for the full per-phase writeup and
[Limitations](#limitations) for what not to over-trust in these numbers.

## Methodology


### Phase 1 - Data engineering

`src/data_pipeline.py` pulls, as of a configured valuation date:

- The Treasury/SOFR curve from FRED (public `fredgraph.csv` export - no API
  key needed), fit with a cubic spline and round-trip validated.
- Daily equity history and a market-cap/shares snapshot per ticker via
  `yfinance`.
- Balance-sheet debt figures per ticker from the SEC EDGAR company-facts
  API, using the plan's convention (`short-term debt + 0.5 * long-term
  debt`), with CIK lookup via SEC's `company_tickers.json`.

Run it with:

```bash
python -m src.data_pipeline
```

This reads `config.yaml` (valuation date, FRED series, company panel),
caches raw pulls under `data/raw/` (gitignored), and writes the curve and
company panel to `data/processed/`.

**Validation implemented:**
- Curve round-trip check (`validate_curve_roundtrip`) - fits the curve and
  confirms it reproduces the input yields.
- Missing-data audit (`audit_missing_data`) - null counts/percentages per
  column.
- Distribution / sanity-range check (`check_panel_sanity`) - flags
  non-positive equity value, implausible equity volatility, missing debt,
  and **stale debt data**: SEC filers sometimes stop using a given XBRL tag
  for years (e.g. AT&T reported `LongTermDebtNoncurrent` only through 2012,
  moving to `LongTermDebtAndCapitalLeaseObligations` afterward), so debt
  extraction checks every candidate tag and takes the single freshest value
  rather than the first candidate with any data - and anything still older
  than ~1.5 years relative to the valuation date gets flagged rather than
  silently used. In the current 10-ticker panel, Ford is the one flagged
  case: no candidate XBRL tag has debt data more recent than 2020-12-31,
  likely because Ford discloses debt through company-specific extension
  tags outside the standard `us-gaap` set this pipeline scans.

Company panel: `AAPL, MSFT, JNJ, KO, XOM, T, F, CCL, BA, AMC` - chosen to
span investment-grade to stressed credit across sectors (config in
`config.yaml`), so Phase 2's Merton PD and Phase 2b's classifier have
something to rank-order.

### Phase 2 - Merton structural PD baseline

`src/pd_model.py` solves the two-equation Merton system (`scipy.optimize.fsolve`)
for asset value V and asset volatility sigma_V given each company's equity
value/vol (Phase 1) and debt (Phase 1, short-term + 0.5*long-term), then
computes Distance-to-Default and `PD ~= N(-DD)`. Risk-free rate is
interpolated from the Phase 1 FRED curve at the configured horizon
(`merton.horizon_years` in `config.yaml`, default 1 year).

Run it with:

```bash
python -m src.pd_model
```

Reads `data/processed/{fred_curve,company_panel}_<date>.csv` from Phase 1,
writes `data/processed/merton_pd_<date>.csv`.

**Validation implemented:**
- Solver convergence check, judged on the equation residual *relative* to
  each deal's own scale (E, D span ~1e8 to ~1e12 across the panel, so a
  fixed absolute residual tolerance is meaningless) - fsolve is retried from
  several initial guesses for sigma_V if the first doesn't converge. 10/10
  companies converge on the current panel.
- Known-answer recovery test (`tests/test_pd_model.py`): generate synthetic
  (E, sigma_E) by running the Merton equations *forward* from a chosen
  (V, sigma_V), then confirm `solve_merton` recovers the original (V, sigma_V).
- Rank-ordering: PD should track credit quality. On the current panel it
  does at the extremes - AMC (highest-risk name) comes out ~750x riskier
  than the next-highest name - but two honest caveats surfaced:
  - **Ford's PD looks artificially low** because Phase 1 flagged its debt
    figure as stale (no SEC tag more recent than 2020-12-31, so its
    `total_debt_face_value` badly understates real leverage). A clean
    illustration of garbage-in-garbage-out, and exactly why the sanity-check
    flag from Phase 1 matters downstream.
  - **Investment-grade names cluster at implausibly small PDs** (1e-40 to
    1e-76) rather than realistic sub-1% figures. This is a well-known
    property of the plain Merton model with physical-measure inputs, not a
    bug - it's the standard motivation for Phase 2b's supervised classifier,
    which this Merton output feeds as an engineered feature rather than
    serving as the final PD.

### Phase 2b - Supervised PD classifier

`src/pd_model.py` (same module as Phase 2) and `src/features.py` train and
evaluate a logistic regression and an XGBoost classifier on the public
[Taiwanese Bankruptcy Prediction dataset](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)
(UCI mirror of the Kaggle dataset the plan references - Taiwan Economic
Journal, ~6,800 companies, ~95 pre-computed accounting-ratio features, no
API key needed). This is a separate, larger, labeled corpus used to train
the classifier itself - it's not tied to the Phase 1/2 ten-ticker panel,
which has zero labeled defaults of its own.

Run it with:

```bash
python -m src.pd_model   # runs Phase 2 then Phase 2b
```

Caches the raw dataset to `data/raw/taiwan_bankruptcy.csv`, writes
`data/processed/pd_classifier_comparison.csv` and
`outputs/pd_classifier_calibration.png`.

**Validation implemented:**
- Class-imbalance-aware evaluation: ROC-AUC and PR-AUC (not accuracy) on a
  held-out test set, for both the raw class-weighted models and calibrated
  versions of each (see below). Current results: Logistic Regression
  ROC-AUC 0.86 / PR-AUC 0.26; XGBoost ROC-AUC 0.95 / PR-AUC 0.44 - gradient
  boosting clearly wins on this dataset.
- Calibration curve, and an honest overconfidence finding: the raw models'
  `class_weight='balanced'` / `scale_pos_weight` settings (needed to fix
  ranking on a 3.2%-positive dataset) push predicted probabilities well
  above observed frequencies - the raw logistic regression's top bin
  predicts ~85% but only ~20% of those companies actually went bankrupt.
  Wrapping each model in cross-validated Platt scaling
  (`calibrate_classifier`, sigmoid method - isotonic would overfit with
  only ~165 positive training examples) brings predicted probability back
  in line with observed frequency, visibly closing the gap to the diagonal
  in `outputs/pd_classifier_calibration.png`, without hurting ROC-AUC/PR-AUC.
- Small-synthetic-dataset pipeline test (`tests/test_pd_classifier.py`):
  both classifiers fit and clear chance-level ROC-AUC on a synthetic,
  separable dataset, offline and fast.

**Two deviations from the plan, documented rather than glossed over:**
- **No time-based split.** The plan calls for a time-based train/test split
  to avoid lookahead bias. This public dataset's released form carries no
  per-row date or fiscal-year field to split on, so a stratified random
  split is used instead (`split_bankruptcy_dataset`) - it at least protects
  the severely-imbalanced positive class from further distortion by chance,
  but it does not protect against lookahead bias the way a true time split
  would. A self-built, dated SEC EDGAR panel (the plan's Path 2) would fix
  this at substantially more data-collection cost.
- **No head-to-head Merton comparison.** The plan asks to compare the
  classifier honestly against the Phase 2 Merton baseline. That comparison
  needs the same companies scored both ways, and neither available dataset
  supports it: the Taiwan dataset has no market equity value/volatility for
  Merton to consume, and the Phase 1/2 ten-ticker panel has zero labeled
  defaults for the classifier to be evaluated against. Stated here as a
  genuine data-availability limitation rather than a skipped step.

### Phase 3 - Hull-White rate model calibration

`src/ratemodel.py` calibrates the Hull-White short-rate model in two parts:

1. **Mean-reversion speed `a` and volatility `sigma`** from historical daily
   SOFR (FRED) via an AR(1)-style regression: `dr_t = intercept + slope *
   r_{t-1} + residual`, giving `a = -slope/dt` and `sigma` from the residual
   std, annualized.
2. **`theta(t)`**, calibrated analytically so the model's expected short
   rate reproduces the Phase 1 forward curve exactly:
   `theta(t) = df(0,t)/dt + a*f(0,t) + sigma^2/(2a)*(1 - e^{-2at})`, where
   `f(0,t)` is the instantaneous forward rate from the fitted zero curve.

Run it with:

```bash
python -m src.ratemodel
```

Reads `data/processed/fred_curve_<date>.csv` from Phase 1, writes
`data/processed/hull_white_params_<date>.json` and
`outputs/hull_white_paths.png`.

**Validation implemented:**
- Residual diagnostics on the AR(1) fit (`residual_diagnostics`): lag-1
  autocorrelation and a heteroscedasticity proxy (correlation of `|residual|`
  with the rate level). On the current SOFR history (2018-2026, 2088 daily
  observations): heteroscedasticity is fine (correlation 0.02), but lag-1
  autocorrelation is **-0.25**, not the ~0 a well-specified AR(1) would
  show. Reported honestly rather than glossed over - a plain daily AR(1)
  doesn't fully capture SOFR's short-term dynamics (plausible causes:
  settlement/day-of-week effects, or mean reversion happening faster than a
  1-day lag resolves).
- Closed-form correctness check for `calibrate_theta`
  (`tests/test_ratemodel.py`): independently derived
  `E[r(t)] = f(0,t) + sigma^2/(2a^2)*(1-e^{-at})^2` by solving the
  deterministic mean-reversion ODE, then confirmed the calibrated `theta(t)`
  numerically integrates (via `scipy.integrate.solve_ivp`) to that same
  closed form - a genuine correctness check, not a tautology, since the two
  are computed independently.
- Known-parameter recovery test: `calibrate_ar1` recovers synthetic
  (a, sigma) from a simulated path with known parameters. Uses monthly
  (not daily) synthetic steps deliberately - at daily frequency the AR(1)
  coefficient is extremely close to 1 (a near-unit-root process), and
  OLS's well-documented finite-sample bias in that regime gets amplified
  roughly 500x when converted to `a` (dividing a tiny, noisy slope by a
  tiny `dt`). This is the same reason the real daily-SOFR `a` estimate
  above should be read as directionally reasonable rather than precise -
  an inherent limitation of AR(1)-from-daily-data calibration, not a code
  bug.
- Simulation sanity check: Monte Carlo mean path (1000 sims, 5y horizon)
  tracks the closed-form `E[r(t)]` to within 0.04bp mean absolute error,
  and both track the input forward curve almost exactly (see
  `outputs/hull_white_paths.png`) - confirming `theta(t)` does what it's
  supposed to. Individual simulated paths fan out plausibly around that
  mean, including some paths dipping negative - an expected feature of
  Gaussian short-rate models like Hull-White, not a bug (negative rates
  are also a real phenomenon historically, e.g. EUR/JPY/CHF).

### Phase 4 - Monte Carlo synthetic exposure dataset

`src/exposure_models.py` simulates Hull-White paths for a hypothetical
interest rate swap (spec in `config.yaml`) and, at each payment-date
checkpoint along every path, computes the **realized** future swap value
along that specific path -- a noisy, one-sample estimate of the true
conditional expectation, in the spirit of Longstaff-Schwartz least-squares
Monte Carlo. This (state, noisy label) dataset is the training data Phase 5
will build regression models against.

The floating leg is modeled SOFR-OIS style (compounded-in-arrears, only
fixed once each accrual period ends), which makes the discounted floating
leg telescope exactly to `notional*(D(t,T_first) - D(t,T_last))` along any
single path using that path's own realized money-market discounting - no
extra simulation error beyond discretizing the rate-path integral itself.
Checkpoints are restricted to payment/reset dates specifically to avoid the
"already-fixed-but-unpaid cash flow" complication a mid-period checkpoint
would introduce - a documented simplification, not a silent one.

Also added: `hull_white_zero_coupon_bond` (analytic Hull-White bond pricing
formula, in `ratemodel.py`) and `analytic_swap_value` - the closed-form
conditional expectation given a path's state, used here purely as a
Phase-4-level sanity check that the noisy-label generator is unbiased (a
different, earlier check than Phase 5's naive-baseline flooring-bias story).

Run it with:

```bash
python -m src.exposure_models
```

Re-derives the Phase 3 Hull-White calibration, reads
`data/processed/fred_curve_<date>.csv`, writes
`data/processed/exposure_training_data_<date>.csv` and
`outputs/exposure_training_data_example_checkpoint.png`.

**Validation implemented:**
- Exact reproduction test (`tests/test_exposure_models.py`): at t=0 with
  `r_t = f(0,0)`, `hull_white_zero_coupon_bond` reproduces the input curve's
  own zero-coupon prices to 1e-6 relative tolerance - a fundamental
  property of Hull-White calibration, checked at the bond-pricing-formula
  level (Phase 3 checked it at the expected-short-rate level).
- Par-swap-rate test: the fixed rate that zeroes out `analytic_swap_value`
  at inception, computed independently from the curve alone, matches the
  textbook definition of an at-the-money swap - confirms `analytic_swap_value`
  is implemented correctly, not just internally self-consistent.
- Unbiasedness check (`run_phase4`, and a Monte Carlo test in the test
  suite): pooled `mean(realized - analytic)` per checkpoint should be ~0.
  On the live run (5,000 paths, 9 checkpoints), the largest
  `|bias / standard error|` across checkpoints is 2.34 - unremarkable given
  9 simultaneous checks, not evidence of a bug.
- Visual check (`outputs/exposure_training_data_example_checkpoint.png`):
  realized values scatter tightly around the smooth analytic curve at an
  example checkpoint - exactly the shape Phase 5's regression models need
  to recover.

**Swap rate:** `fixed_rate` in `config.yaml` is set to 3.9217%, the
par/at-the-money rate implied by the valuation-date curve (computed by
solving `analytic_swap_value(t=0) = 0` via the Hull-White bond-pricing
formula - the same calculation `tests/test_exposure_models.py` checks
independently). That gives the genuinely two-sided exposure profile CVA
work expects: realized/analytic swap values scatter around zero at each
checkpoint rather than skewing to one side, visible in
`outputs/exposure_training_data_example_checkpoint.png`.

### Phase 5 - Exposure regression model comparison

`src/exposure_models.py` (same module as Phase 4) fits, per checkpoint, a
naive baseline and four regression models on the raw (unfloored) realized
swap values, then benchmarks all five against a fresh nested-simulation
ground truth.

The comparison follows the plan's wording closely and deliberately
asymmetrically:
- **naive baseline** = "average the raw noisy labels directly (floor, then
  average)" -> a single constant per checkpoint: `mean(max(V_i, 0))`.
- **ground truth** (`nested_simulation_ground_truth`) = a fresh, larger
  Monte Carlo simulation launched from a fixed state at the checkpoint,
  "average WITHOUT flooring" -> `E[V | state]`, the raw signed conditional
  mean.
- **poly / random forest / gradient boosting / MLP** are all fit on the
  same *unfloored* labels the ground truth targets, one model per
  checkpoint, feature = short rate.

Since `max(V,0) >= V` pointwise for any `V`, naive's floored constant is
provably `>=` the unfloored ground truth whenever a checkpoint has any
chance of a negative value - a structural, not just empirical, source of
upward bias.

Run it with:

```bash
python -m src.exposure_models   # runs Phase 4 then Phase 5
```

Reads `data/processed/exposure_training_data_<date>.csv` from Phase 4,
writes `data/processed/exposure_model_comparison_<date>.csv` (+ `_summary`)
and `outputs/exposure_model_comparison.png`.

**Validation implemented:**
- Nested-sim vs. closed-form cross-check: mean `|ground_truth - analytic|`
  across all tested (checkpoint, rate) points is ~$1,580 against a $10M
  notional - the expensive nested-simulation methodology agrees almost
  exactly with the exact Hull-White formula, which is a nice validation
  since a real path-dependent product without a closed form would have to
  rely on the nested-sim number *alone*.
- Model comparison table (3 checkpoints x 5 rate percentiles, 15 points),
  MAE against ground truth:

  | Model | MAE | RMSE | mean bias | % error positive |
  |---|---|---|---|---|
  | naive | 267,699 | 325,292 | +120,353 | 60% |
  | poly (degree 3) | 3,482 | 4,671 | -3,126 | 20% |
  | random forest | 110,920 | 151,852 | +46,565 | 60% |
  | gradient boosting | 28,290 | 36,010 | -7,580 | 47% |
  | mlp | 8,068 | 11,515 | +4,680 | 60% |

  Polynomial OLS wins clearly - unsurprising in hindsight, since the
  Hull-White swap value is an exponential-affine function of the short
  rate, which a cubic approximates well over the simulated range.
- **Naive's bias is real but its error isn't dominated by flooring the way
  the initial writeup assumed.** Test rates span the 5th-95th percentile of
  each checkpoint's short-rate distribution, so most of naive's error
  magnitude comes from being state-blind (one constant vs. a steeply
  state-varying true curve, visible in `outputs/exposure_model_comparison.png`);
  the flooring/Jensen effect is real and does push `mean_bias` positive
  with `pct_error_positive` above 50% for naive, but it's a secondary
  contributor to the error size here, not the dominant one. Said plainly
  rather than overstated.
- **Random forest visibly overfits the training noise** (unconstrained
  `max_depth`, deliberately - see the fit function's docstring): its fitted
  curve is wiggly through the training range and flat at both tails, since
  trees can't extrapolate past their leaf boundaries - exactly the
  overfitting risk the plan asks to actually check rather than assume.
  This is checked, not assumed: it's the second-worst model here, clearly
  worse than gradient boosting despite both being tree ensembles.

**A real bug found and fixed during this phase:** the first MLP version
scaled the input feature but not the target - swap values are ~$10^5-10^6
in scale, well outside `MLPRegressor`'s default-tuned range, and it
silently converged to a flat near-zero prediction (MAE ~254,000, nearly as
bad as naive) rather than erroring. That looked like a plausible "MLP
underfits here" finding at first glance, but plotting the fitted curve
made it obvious it wasn't a real capacity issue - it hadn't learned
anything. Wrapping the model in `TransformedTargetRegressor` to scale the
target too fixed it immediately (MAE 8,068, second-best model). Worth
remembering when a scikit-learn neural net result looks suspiciously flat.

### Phase 6 - LGD assumption

`src/cva.py`: `compute_lgd(recovery_rate) = 1 - recovery_rate`. Recovery
rate is set to 40% in `config.yaml` - the ISDA CDS Standard Model's
convention for senior unsecured claims (2009 "Big Bang" protocol),
consistent with Moody's long-run historical average senior-unsecured
recovery (~37-40%). A standard market assumption, cited rather than
invented, appropriate to this phase's intentionally small scope per the
plan.

### Phase 7 - CVA assembly and sensitivity analysis

`src/cva.py` combines a PD term structure, Phase 5's EE(t) curve, Phase 6's
LGD, and the Phase 1 discount curve into a CVA number:

```
CVA = LGD * sum_k DF(0,t_k) * EE(t_k) * [Q_default(t_k) - Q_default(t_{k-1})]
```

**PD term structure:** the plan calls for "the Phase 2b PD term
structure," but Phase 2b's classifier isn't applicable here - no shared
features or company universe with the Taiwan dataset (documented in Phase
2b's section). This uses Phase 2's Merton 1-year PD for the chosen
counterparty instead, extrapolated to a term structure via a
constant-hazard-rate assumption (`Q_survival(t) = (1-PD_1y)^t`) - a small,
explicit simplification, not a full credit-curve calibration, matching the
plan's "intentionally small" framing for this phase.

**Counterparty:** `AMC`, chosen deliberately (`cva.counterparty_ticker` in
`config.yaml`) - it's the only Phase 1/2 panel name with a non-negligible
Merton PD. Every investment-grade name collapses to a PD near 1e-40 or
smaller (documented in Phase 2), which would make a sensitivity analysis
against those names trivially uninteresting.

**EE(t) source:** built from Phase 5's regression models via
`compute_ee_curve` (`exposure_models.py`) - fits the chosen model per
checkpoint on `short_rate -> realized_swap_value`, then floors *the fitted
prediction* at each path's own state and averages. This is a lower-
variance EE(t) than the naive floor-raw-labels-then-average approach,
since flooring a smooth fitted curve (instead of each noisy raw label)
removes exactly the continuation-noise Phase 5 quantified - the concrete
mechanism by which Phase 5's model comparison feeds into the CVA number.

Run it with:

```bash
python -m src.cva
```

Reads Phase 1's curve, Phase 2's Merton PDs, Phase 4's training data, and
Phase 5's model-comparison summary (to pick the best model by MAE - `poly`
on the current run). Writes `data/processed/cva_sensitivity_<date>.csv`,
`data/processed/cva_portfolio_<date>.csv`, and `outputs/cva_sensitivity.png`.

**Results on the current run:**
- CVA (AMC, naive EE(t)): $26,907. CVA (AMC, poly EE(t)): $19,073 - a
  **-29.1% difference from the exposure-model choice alone**, exactly the
  "using a better-calibrated exposure model changes the estimated CVA by
  $X" business-relevant framing the plan asks for. Naive's higher number
  is consistent with Phase 5's finding that naive over-estimates exposure.
- Sensitivity analysis: CVA moves from $10,188 to $33,384 as the 1y PD is
  shocked 0.5x-2x, and from $25,430 to $12,715 as recovery rises 20%-60%
  (LGD falls 80%-40%) - both **strictly monotonic**, asserted in code, not
  just visually inspected (`outputs/cva_sensitivity.png`).
- Bonus portfolio table (same swap, each panel company's own Merton PD):
  CVA is ~$0 for every name except AMC ($19,073) and a barely-nonzero
  $29 for CCL - the direct, expected continuation of Phase 2's
  Merton-PD-collapse finding for investment-grade names, not a new issue.

### Phase 8 - PINN fast pricer (scientific ML)

`src/pinn.py` trains a small PyTorch MLP to solve the Hull-White
zero-coupon bond-pricing PDE directly - no labeled price data anywhere in
training, just the PDE residual (via second-order autograd) and a
hard-constrained terminal condition:

```
dP/dt + (theta(t) - a*r) dP/dr + 0.5*sigma^2 d2P/dr2 - r*P = 0,   P(T,r) = 1
```

The terminal condition is baked into the network's output parametrization,
`P(t,r) = 1 + (T-t)*NN(t,r)`, rather than added as a soft penalty term - it
holds exactly, for any weights, not just approximately. This is scientific
ML in the literal sense the plan means: the loss is a physics constraint,
not a supervised label.

Run it with:

```bash
python -m src.pinn
```

Re-derives the Phase 3 Hull-White calibration, reads the Phase 1 curve,
trains for 3,000 epochs (~16s on CPU), and writes `outputs/pinn_validation.png`.

**Validation implemented:**
- Max/mean absolute price error against `ratemodel.hull_white_zero_coupon_bond`:
  0.0127 / 0.0032 (bond prices here run ~0.79-1.00, so that's roughly
  0.3-1.3% relative error).
- **Error concentrated away from the hard-constrained boundary, checked
  quantitatively, not just eyeballed:** correlation(time-to-maturity, abs
  error) = 0.708, and the error heatmap in `outputs/pinn_validation.png`
  shows a clear band of higher error at t=0 fading to ~0 at t=T - exactly
  the pattern the plan predicts from hard-constraining the terminal
  condition rather than the initial one.
- Known-derivative test (`tests/test_pinn.py`): the autograd-computed
  `dP/dt`, `dP/dr`, `d2P/dr2` match finite-difference derivatives of the
  same (untrained) network - validates the autograd wiring itself,
  independent of whether training converges.
- Exact-terminal-condition test: `P(T,r)=1` holds to float precision for
  an *untrained*, randomly-initialized network - confirms it's a property
  of the architecture, not something training has to learn.
- Training-improves-fit test: loss decreases over training, and the
  trained network is closer to the closed form than an untrained baseline
  on the same validation grid.

**Honest scope, stated plainly per the plan:** in this 1D setting the
closed form wins outright - 0.81ms for 10,000 closed-form price
evaluations vs. 7.32ms for the same on the trained PINN, plus the PINN
needed ~16s of training the closed form never pays. The value here isn't
speed; it's that this exact recipe - PDE residual + hard-constrained
boundary, zero labeled data - is what you'd reach for in genuinely
higher-dimensional short-rate models (multi-factor, stochastic vol) where
no closed form exists and grid-based PDE solvers become infeasible. This
is a working, validated instance of that recipe, not a production speed-up.

**A real performance bug found and fixed:** the full test suite hung for
10+ minutes on what should have been a few seconds of PINN training,
consistently stalling at the same test. It wasn't a bug in the training
logic - torch's default multi-threaded BLAS was contending with
numpy/scikit-learn/xgboost's own thread pools once all four were loaded in
the same process (the PINN's network is tiny - hidden_size=64, batches of
a few hundred points - so threading overhead swamps any parallelism
benefit at that scale, and gets far worse under contention). Confirmed by
timing the same training call with and without `torch.set_num_threads(1)`
in the exact import order the full suite uses: unbounded threads took
minutes, one thread took 3 seconds. Fixed by pinning `torch.set_num_threads(1)`
at module import - full suite now runs in ~5s.

### Phase 9 - Portfolio-level application

`src/cva.py` (`run_phase9`) applies the same swap and the same Phase 5
best-model EE(t) curve across the entire Phase 1/2 panel - "a model
deployed across a segment," per the plan, not just more single-company
examples - then sanity-checks the resulting CVA ranking against
approximate public credit ratings.

**Ratings caveat, stated up front:** `APPROXIMATE_CREDIT_RATINGS` in
`cva.py` is illustrative, from general knowledge, not pulled from a live
agency feed (there's no free public API for this). Used only for a
rank-order sanity check, never as an input to any CVA number.

Run it with:

```bash
python -m src.cva   # runs Phase 6+7, then Phase 9
```

Writes `data/processed/portfolio_application_<date>.csv` and
`outputs/portfolio_cva.png`.

**Results and discussion:**
- Spearman rank correlation between CVA rank and approximate-rating rank:
  **0.767 (p=0.010)** - statistically significant despite only 10 names,
  and directionally exactly right: AMC (the panel's only clearly
  speculative-grade name by rating) is correctly identified as by far the
  riskiest, and the genuinely investment-grade names cluster at the
  bottom.
- **Two real limitations show up in the fine-grained ranking, both already
  documented elsewhere and simply surfacing again here:**
  1. Ford ranks among the *safest* names despite a BB+ (speculative-grade)
     rating - the direct consequence of Phase 1's stale-debt-data finding
     for Ford, not a new problem. Visible in `outputs/portfolio_cva.png`
     as the one red (speculative-grade) bar sitting at investment-grade
     height.
  2. Within the investment-grade cluster, PDs span 1e-76 to 1e-41 - over
     twenty orders of magnitude of "differentiation" between names that
     are, in reality, all roughly equally very safe. Ordering names within
     that cluster by Merton PD would be overinterpreting noise, not signal.
- **Net takeaway:** plain Merton is a reasonable coarse categorical screen
  (correctly flags the one clearly-risky name) but not a reliable
  fine-grained ranking tool - exactly the gap Phase 2b's supervised
  classifier exists to close, if it could be applied to this same company
  universe (it can't - see Phase 2b's documented limitation).

## Risk-focused extensions

Four extensions from `cva_risk_extensions_pack.md`, built on top of the
base 10-phase pipeline above. Each reuses the base pipeline's existing
models rather than reimplementing them (Merton's solved `V`/`sigma_V`,
the Hull-White simulation machinery, `analytic_swap_value`, the FRED
ingestion helper) -- these are extensions, not a second project.

### Bootstrapped CVA confidence interval

`src/cva_uncertainty.py` propagates uncertainty from three sources through
to a CVA *distribution* instead of a single point number: the Monte Carlo
exposure simulation itself (resample whole simulated paths, preserving the
fact that one Hull-White draw determines a path's value at every
checkpoint jointly), the Merton PD estimate's own sampling uncertainty
(bootstrap the trailing daily equity log returns behind `sigma_E`,
re-solve Merton), and LGD (drawn from a triangular distribution over the
published plausible recovery-rate range, not held at a single point
value).

Run it with:

```bash
python -m src.cva_uncertainty
```

**Validation implemented:** an explicit ablation check that each source
actually injects variance -- turning all three off collapses the
distribution to exactly zero standard deviation (asserted in code), and
each source turned on alone is confirmed to have positive standard
deviation. This directly targets the pack's documented pitfall: a
bootstrap that resamples a seed without actually varying PD/LGD produces
an interval that looks precise without being meaningful.

**Results on the current run (AMC, 500 iterations):** mean **$28,941**,
median $27,551 (tracks Phase 7's naive-EE CVA of $26,907 reasonably
closely, as expected -- the resampling here mirrors naive's own
floor-then-average estimator), 90% interval **[$14,046, $47,611]**. The
PD-resampling source dominates the interval's width (std $9,864 alone,
vs. $10,469 combined) -- Merton's well-documented tail sensitivity
(Merton memo, Section 6) shows up here as the largest contributor to CVA
*uncertainty*, not just to the point estimate's known magnitude issue.
Path-resampling alone contributes comparatively little (std $495) at this
notional/path-count.

### Wrong-way risk (Gaussian copula)

`src/wrong_way_risk.py` links the counterparty's simulated Merton
asset-value process to the same Hull-White short-rate path driving
exposure, via a single Cholesky-correlated pair of standard normal shocks
per simulation step. Default is assessed by discrete monitoring at the
swap's own payment-date checkpoints (`V(t_k) < D`) -- the same checkpoint
grid `exposure_models.py` already uses, sidestepping a full continuous
first-passage model. `V0`/`sigma_V` are the counterparty's own Phase 2
solved Merton values, not a separately-fit process.

**Correlation parameter:** rather than an arbitrary or sector-level
number, `rho` is the historical realized correlation between the
counterparty's own daily equity log returns and daily SOFR changes, over
the same trailing window Phase 1 uses for `equity_vol` -- directly
computable from data already in this project.

Run it with:

```bash
python -m src.wrong_way_risk
```

**Validation implemented:** the pack's explicit ask -- confirm the
direction of the effect makes sense given the swap's structure and the
correlation's sign. For this project's receive-fixed/pay-floating swap
(value rises when rates fall), positive `rho` means low-rate/high-exposure
paths coincide with low-asset-value/default-likely paths -- the classic
wrong-way setup -- and negative `rho` is the right-way case. Checked with
an offline test using extreme synthetic `rho = +-0.8` on a hand-built
setup (`tests/test_wrong_way_risk.py`), confirming `CVA(rho=+0.8) >
CVA(rho=-0.8)`, not just asserted from the formula.

**Results on the current run (AMC):** the historical `rho` between AMC's
equity returns and SOFR changes comes out to **+0.014** -- essentially
zero, giving a **+2.3%** uplift (CVA $35,587 to $36,406). This is a
genuinely small effect, and it's directly consistent with something this
project's own Limitations section already said before this extension was
built: "AMC \[has\] no obvious economic link between rates and its own
default risk." The mechanism validates correctly (direction check passes,
uplift sign matches `rho`'s sign); it's simply the case that AMC's own
data doesn't support a large wrong-way effect. Note the discrete-
monitoring cumulative default probability (~43% over the 5y horizon) runs
well above Phase 2's 1-year terminal-only Merton PD (6.8%) by
construction -- a cumulative, discretely-monitored probability over 5
years is not directly comparable to a single-horizon terminal-only number,
and the gap is the expected, documented consequence of that different
default-timing mechanism (see the module docstring), not a
recalibration of Phase 2's PD.

### External benchmarking vs. market spreads

`src/spread_benchmark.py` compares each panel company's PD x LGD implied
spread against FRED's free ICE BofA rating-bucket corporate OAS series
(AAA through CCC-and-lower), as an external sanity check independent of
this project's own models.

Run it with:

```bash
python -m src.spread_benchmark
```

**Validation implemented:** a completeness check
(`test_every_approximate_credit_rating_maps_to_a_known_fred_series`) that
every rating in `APPROXIMATE_CREDIT_RATINGS` maps to a bucket actually
present in `FRED_OAS_SERIES` -- the same class of guard `test_cva.py`
already has for the ratings-rank table, since a typo'd bucket here would
silently drop a company from the comparison rather than erroring.

**Results on the current run:** market OAS by bucket comes out
monotonically increasing with credit risk as expected (AAA 41bps, AA
58bps, A 66bps, BBB 98bps, BB 160bps, B 288bps, CCC-and-lower 1,024bps),
confirming the series IDs are correctly mapped. Every investment-grade
name's model-implied spread collapses to near-zero (the same Merton-PD-
collapse finding from Phase 2, now externally quantified rather than just
qualitatively caveated): AAPL implies ~6e-70 bps against a market 58bps,
a gap of -58bps. AMC, the panel's one high-PD name, still understates the
market by -618bps (model 406bps vs. market 1,024bps) -- consistent with
Merton-with-physical-measure-inputs systematically underpricing credit
risk relative to the market's risk-neutral pricing (which also embeds a
risk premium Merton's physical-measure PD doesn't), not a new finding but
now measured in the same units the market actually trades in.

### Model Risk Management (MRM) documentation

`docs/model_validation/` -- one SR-11-7-structured validation memo per
model (Merton PD baseline, supervised PD classifier, Hull-White
calibration, exposure regression models, PINN pricer), plus
[`model_inventory.md`](docs/model_validation/model_inventory.md)
summarizing all five with their production-path status and materiality
tier. Written to engage with each model's *specific* documented
weaknesses (Ford's stale-debt PD, the daily-AR(1) lag-1-autocorrelation
finding, the MLP target-scaling bug, etc.) rather than restating generic
SR 11-7 language -- and, notably, two of the five models (the supervised
PD classifier, the PINN pricer) are found to carry **zero materiality** to
this project's actual reported CVA numbers, stated as an explicit,
load-bearing finding rather than glossed over.

## Repository structure

```
data/
  raw/                    # cached raw pulls (gitignored)
  processed/              # cleaned, feature-engineered datasets (gitignored)
src/
  data_pipeline.py        # Phase 1: FRED + SEC EDGAR + yfinance ingestion
  features.py             # feature engineering (ratios, Merton DD, etc.)
  pd_model.py             # Phase 2/2b: Merton baseline + supervised PD classifier
  ratemodel.py            # Phase 3: Hull-White calibration + simulation
  exposure_models.py      # Phase 4/5: EE(t) regression model comparison
  cva.py                  # Phase 6/7/9: LGD + CVA assembly + sensitivity + portfolio application
  pinn.py                 # Phase 8: PINN fast pricer
  cva_uncertainty.py      # Extension: bootstrapped CVA confidence interval
  wrong_way_risk.py       # Extension: wrong-way risk via Gaussian copula
  spread_benchmark.py     # Extension: external benchmarking vs. FRED OAS spreads
docs/
  model_validation/       # Extension: MRM validation memos, one per model + inventory
notebooks/                # exploratory analysis, results walkthroughs
tests/                    # pytest unit tests on core functions
outputs/                  # saved charts, tables, model artifacts (gitignored)
app.py                    # Streamlit demo (Phase 10): interactive CVA explorer
config.yaml               # config-driven entry point (valuation date, panel, swap spec)
requirements.txt
```

## Setup and quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest              # 90 tests, ~7s, all offline
```

Run the full pipeline end to end (each step reads the previous steps'
saved `data/processed/` outputs, so order matters the first time; after
that, any single phase can be re-run on its own):

```bash
python -m src.data_pipeline    # Phase 1: ingest FRED/yfinance/SEC data
python -m src.pd_model         # Phase 2 + 2b: Merton baseline + supervised classifier
python -m src.ratemodel        # Phase 3: Hull-White calibration
python -m src.exposure_models  # Phase 4 + 5: synthetic exposure data + model comparison
python -m src.cva              # Phase 6 + 7 + 9: LGD, CVA assembly, portfolio application
python -m src.pinn             # Phase 8: PINN pricer (independent of the CVA chain)
```

Then, optionally, the risk-focused extensions (each reads Phase 1-3's
saved outputs; `cva_uncertainty` and `wrong_way_risk` also need Phase 4's
exposure training data and the raw equity history Phase 1 caches):

```bash
python -m src.cva_uncertainty  # Extension: bootstrapped CVA confidence interval
python -m src.wrong_way_risk   # Extension: wrong-way risk via Gaussian copula
python -m src.spread_benchmark # Extension: external benchmarking vs. FRED OAS spreads
```

All of it is config-driven (`config.yaml`): valuation date, company panel,
Merton horizon, Hull-White short-rate series, swap spec, LGD, CVA
counterparty and shock sizes, and PINN training settings can all be
changed there without touching source.

Then explore interactively (needs Phases 1-5 to have run at least once,
per above):

```bash
streamlit run app.py
```

Pick a counterparty, toggle naive vs. best-model exposure, and drag the
PD-shock / recovery-rate sliders to see CVA update live - the app calls
straight into `src.cva` / `src.exposure_models`, so every number matches
what the batch phase scripts and test suite produce; it's a view over the
pipeline, not a second implementation of it.

## Roadmap

- [x] Phase 1 - Data engineering (FRED / yfinance / SEC EDGAR pipeline)
- [x] Phase 2 - Merton structural PD baseline
- [x] Phase 2b - Supervised PD classifier (logistic regression vs. gradient boosting)
- [x] Phase 3 - Hull-White calibration (AR(1) regression)
- [x] Phase 4 - Monte Carlo synthetic dataset generation
- [x] Phase 5 - EE(t) regression model comparison vs. nested-sim ground truth
- [x] Phase 6 - LGD assumption
- [x] Phase 7 - CVA assembly + sensitivity analysis
- [x] Phase 8 - PINN fast pricer
- [x] Phase 9 - Portfolio-level application across the company panel
- [x] Phase 10 - Tests, config-driven pipeline, DS-facing README
- [x] Phase 10 (optional stretch) - Streamlit demo app (`app.py`)
- [x] Extension - Bootstrapped CVA confidence interval (`src/cva_uncertainty.py`)
- [x] Extension - Wrong-way risk via Gaussian copula (`src/wrong_way_risk.py`)
- [x] Extension - External benchmarking vs. FRED market spreads (`src/spread_benchmark.py`)
- [x] Extension - MRM validation memos (`docs/model_validation/`)
- [ ] Extension - Real Fed stress-test scenarios (deferred, see [Risk-focused extensions](#risk-focused-extensions))
- [ ] Extension - NLP-augmented PD signal (deferred, see [Risk-focused extensions](#risk-focused-extensions))

## Limitations

Consolidated from the per-phase writeups above - the individual sections
have the full detail and code references; this is the reviewer-facing
summary of what to not over-trust in these numbers.

**PD modeling:**
- Plain Merton with physical-measure inputs collapses investment-grade PDs
  to implausible magnitudes (1e-40 to 1e-76) - a well-known property of the
  model, not a bug, but it means Merton's PD *magnitudes* shouldn't be
  taken at face value anywhere in this project; only its coarse rank-
  ordering is reasonably trustworthy (Phase 2, Phase 9).
- The Phase 2b classifier and the Phase 2 Merton baseline were never
  compared head-to-head on the same data, because no available dataset
  supports it: the Taiwan bankruptcy dataset has no market-equity data for
  Merton, and the Phase 1/2 panel has zero labeled defaults for the
  classifier. Phase 7/9's CVA numbers use Merton, not the classifier, for
  this reason - stated explicitly rather than silently substituted.
- The Phase 2b classifier's train/test split is stratified-random, not
  time-based, because the public dataset carries no per-row date field.
  This protects against class-imbalance distortion but not lookahead bias
  the way a true time split would.
- Ford's Merton PD (and everywhere it's used downstream, including the
  Phase 9 portfolio ranking) is unreliable because Phase 1 found no SEC
  debt figure newer than 2020 for it - flagged by the pipeline's own
  sanity check, not silently trusted.

**Rate modeling:**
- Hull-White is a single-factor Gaussian short-rate model - it allows
  negative rates (a real historical phenomenon, not obviously wrong, but
  worth naming) and cannot represent decorrelated moves at different
  points on the curve the way a multi-factor model could.
- The AR(1) calibration of mean-reversion speed `a` from daily SOFR data
  is inherently imprecise: at daily frequency the implied autoregressive
  coefficient is extremely close to 1 (near-unit-root), where OLS's
  well-documented finite-sample bias is amplified roughly 500x in the
  `a = -slope/dt` conversion. The daily AR(1) fit also shows a lag-1
  residual autocorrelation of -0.25, not the ~0 a well-specified fit would
  show - the model doesn't fully capture SOFR's short-term dynamics.
- `theta(t)` is calibrated to match the curve's *historical* shape today;
  this project doesn't separately calibrate to market-implied swaption
  volatilities, so the model's forward-looking volatility assumption rests
  entirely on the historical AR(1) `sigma`.

**Exposure modeling:**
- The swap chosen for Phases 4/5/7/9 is a single, simple vanilla
  fixed-for-floating swap with a closed-form price under Hull-White. That
  closed form was used throughout to validate the Monte Carlo/regression
  machinery - a genuinely path-dependent product (no closed form) would
  need to rely on the nested-simulation ground truth alone, with less
  independent verification available.
- Regression models in Phase 5 were evaluated at only 3 checkpoints x 5
  rate values (15 points) against the nested-sim ground truth, for
  runtime's sake - a larger test grid would sharpen (but is unlikely to
  qualitatively change) the MAE comparison.

**CVA assembly:**
- The PD term structure is a constant-hazard-rate extrapolation from a
  single 1-year PD, not a calibrated credit curve - a deliberate,
  documented simplification matching the plan's "intentionally small"
  scope for this phase.
- LGD (60%) is a standard market convention (ISDA CDS Standard Model,
  senior unsecured), not a counterparty- or seniority-specific estimate.
- This is unilateral CVA only - no own-credit adjustment (DVA), no
  collateral/CSA modeling, no wrong-way risk between the swap's rate
  exposure and the counterparty's credit quality (though AMC, chosen as
  the base-case counterparty, has no obvious economic link between rates
  and its own default risk, so this is a lesser concern for this specific
  example than in general).

**Deep learning extension:**
- The Phase 8 PINN is a proof of the technique in the one setting where a
  closed form already exists to validate against. Its value is
  methodological (a validated recipe), not a demonstrated speed or
  accuracy advantage over the closed form.

