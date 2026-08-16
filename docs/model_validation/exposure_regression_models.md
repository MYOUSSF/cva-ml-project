# Model Validation Memo: Exposure Regression Models (EE(t) Estimation)

**Model owner:** Phase 4/5 (`src/exposure_models.py` — `fit_polynomial_regression`,
`fit_random_forest`, `fit_gradient_boosting`, `fit_mlp`, `compute_ee_curve`)
**Validation date:** 2026-08-16
**Reviewer framing:** SR 11-7 pillars, portfolio-project documentation exercise (see caveat in the Merton memo).

## 1. Purpose and scope

Five candidate estimators of Expected (positive) Exposure EE(t) at each
swap checkpoint — naive floor-then-average baseline, degree-3 polynomial
OLS, random forest, gradient boosting (XGBoost), and an MLP — fit on
Hull-White-simulated (state, noisy realized-value) pairs, benchmarked
against a nested-simulation ground truth, with the lowest-MAE model
selected for production use.

**In scope:** choosing the EE(t) input to Phase 7's CVA formula, and by
extension every reported CVA number and this extension pack's bootstrapped
CI and wrong-way-risk calculations. **Out of scope:** the underlying
Hull-White simulation itself (Phase 3's model, separately validated) and
the swap's payoff definition (fixed spec in `config.yaml`, not learned).

## 2. Conceptual soundness

The model comparison is conceptually sound and, notably, harder on itself
than it needed to be: rather than comparing model predictions to each
other, it benchmarks against a genuinely independent nested-simulation
ground truth (`nested_simulation_ground_truth` — fresh forward Monte Carlo
launched from each test state), which is itself cross-checked against the
closed-form analytic value before being trusted as ground truth. This is
the right structure for evaluating a regression-model choice: an
independently-sourced target, not a self-referential comparison.

**A structural asymmetry is the central, correctly-identified finding of
this phase, not a bug:** the naive baseline floors *raw noisy labels* then
averages (`mean(max(V_i, 0))`), while the ground truth and all four
regression models target the *unfloored* conditional mean. Since
`max(V,0) ≥ V` pointwise, naive's constant is provably ≥ the unfloored
truth whenever a checkpoint has any chance of a negative value — a
mathematical, not empirical, source of upward bias. Correctly reasoning
about *why* a baseline is structurally biased, rather than just measuring
that it's worse, is exactly the kind of finding a validation memo should
credit.

## 3. Data lineage and quality

Training data is entirely simulation-generated (Phase 4, 5,000 paths x 9
checkpoints from the already-validated Hull-White model), not observed
market data — there is no external data-quality risk analogous to the
Merton memo's Ford finding. The relevant quality question is instead
**simulation fidelity**, addressed by Phase 4's own unbiasedness check
(pooled `mean(realized - analytic)` per checkpoint, max `|bias/stderr|`
2.34 across 9 simultaneous checks on the live run — unremarkable, not
evidence of a bug).

## 4. Methodology summary (plain language)

At each of the swap's future payment dates, thousands of simulated
interest-rate scenarios produce thousands of "what would this swap be
worth in this scenario" numbers. Averaging those (after zeroing out
negative values, since exposure to a counterparty can't be negative — you
don't owe them if the trade is underwater) gives one candidate estimate of
expected future exposure (the naive baseline). A smarter approach instead
fits a curve through *how swap value depends on the interest rate at that
checkpoint*, then floors and averages the smooth *fitted* curve instead of
each individual noisy scenario — removing noise that would otherwise get
mistaken for real variation in exposure. Four different curve-fitting
methods (a simple cubic, and three more flexible machine-learning
regressors) are compared for which recovers the truth most accurately.

## 5. Performance / benchmarking results (3 checkpoints x 5 rate percentiles, MAE vs. nested-sim ground truth)

| Model | MAE | RMSE | mean bias | % error positive |
|---|---|---|---|---|
| naive | 267,699 | 325,292 | +120,353 | 60% |
| **poly (degree 3)** | **3,482** | **4,671** | -3,126 | 20% |
| random forest | 110,920 | 151,852 | +46,565 | 60% |
| gradient boosting | 28,290 | 36,010 | -7,580 | 47% |
| mlp | 8,068 | 11,515 | +4,680 | 60% |

- **Polynomial OLS wins clearly and for an identifiable structural
  reason**, not by chance: the Hull-White swap value is an
  exponential-affine function of the short rate, which a cubic
  approximates well over the simulated range — a case where the simplest
  candidate wins because it happens to match the problem's true functional
  form, worth stating explicitly rather than treating "more flexible model
  wins" as a prior.
- **Naive's bias is real but not the dominant error source** — test rates
  span the 5th–95th percentile of each checkpoint's simulated distribution,
  so most of naive's error comes from being state-blind (one constant vs.
  a steeply state-varying truth); the flooring/Jensen effect is real
  (positive `mean_bias`, `pct_error_positive` above 50%) but secondary.
  Correctly not overstated relative to what the numbers actually show.
- **Random forest visibly overfits training noise** — unconstrained
  `max_depth` by deliberate choice (to make overfitting checkable rather
  than assumed away), and it shows: second-worst MAE, wiggly through the
  training range, flat at both tails (trees can't extrapolate past leaf
  boundaries).
- **A real bug found and fixed during this phase**, worth recording in a
  validation memo as process evidence, not just a footnote: the first MLP
  scaled its input feature but not its target. Swap values are ~$10^5–10^6
  in scale, well outside `MLPRegressor`'s default-tuned range, and it
  silently converged to a near-zero flat prediction (MAE ~254,000) rather
  than erroring — a plausible-looking "MLP underfits here" finding that
  was actually a scaling bug, caught by plotting the fitted curve rather
  than trusting the MAE number alone. Fixed via
  `TransformedTargetRegressor` (MAE dropped to 8,068).

## 6. Known limitations and weaknesses (specific, not generic)

- **Only 15 (checkpoint, rate) test points** (3 checkpoints x 5
  percentiles) were benchmarked against the expensive nested-sim ground
  truth, for runtime reasons. A larger grid would sharpen but is unlikely
  to qualitatively change the ranking — stated as a real, if probably
  minor, coverage gap.
- **Single product, single closed form.** The entire benchmarking
  methodology leans on this specific swap having a Hull-White closed
  form. A genuinely path-dependent product without one would have to rely
  on the nested-sim ground truth *alone*, with materially less independent
  verification available — this comparison's methodology, not just its
  numbers, is untested outside this favorable case.
- **Per-checkpoint, single-feature models.** Each checkpoint gets its own
  independently-fit model on a single feature (short rate) — no
  cross-checkpoint smoothing or shared structure is exploited, which is
  fine given the closed form is exponential-affine in the rate alone, but
  would not obviously extend to a product whose exposure depends on
  multiple state variables.
- **Model selection is MAE-only**, on the pooled 15-point comparison — no
  separate check of tail behavior (the checkpoints/percentiles that matter
  most for a CVA calculation, which is itself sensitive mostly to the
  right tail of exposure) beyond what's implicit in testing at the 95th
  percentile.

## 7. Ongoing monitoring plan

- **Re-selection trigger:** any change to the swap spec (notional, tenor,
  fixed rate, payment frequency) in `config.yaml` should re-run the full
  Phase 5 comparison rather than assuming poly's win generalizes — poly's
  advantage here is tied to this specific swap's exponential-affine
  functional form, not a property of the estimator in general.
- **Track over time:** the nested-sim-vs-analytic cross-check MAE
  ($1,580 vs. $10M notional on the current run) as a canary for a broken
  ground-truth generator before it silently corrupts the model comparison
  itself.
- **Watch for the MLP-flat-prediction failure mode recurring** if the
  swap spec changes push values to a different scale — the underlying
  cause (default MLP settings assuming O(1) targets) is a property of the
  library default, not something this fix permanently forecloses if the
  target-scaling wrapper were ever removed.

## 8. Model tiering / materiality

**Tier: production-selected, high materiality.** The chosen model (poly)
directly determines the EE(t) curve behind every CVA number in Phases 7
and 9 and both new extensions — Phase 7's own reported finding that
naive-vs-poly EE(t) alone swings CVA by -29% is direct evidence of how
materially this choice moves the final number. Given that materiality,
the 15-point benchmark grid (Section 6) is a real gap worth closing before
this comparison's conclusion is treated as fully settled, even though the
qualitative ranking (poly wins, naive is structurally biased, RF
overfits) is well-supported by what has been checked.
