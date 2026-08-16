# Model Validation Memo: Supervised PD Classifier (Logistic Regression / XGBoost)

**Model owner:** Phase 2b (`src/pd_model.py` — `train_logistic_regression`,
`train_gradient_boosting`, `calibrate_classifier`)
**Validation date:** 2026-08-16
**Reviewer framing:** SR 11-7 pillars, portfolio-project documentation exercise (see caveat in the Merton memo, Section header).

## 1. Purpose and scope

Binary bankruptcy classifier trained on the public Taiwanese Bankruptcy
Prediction dataset (~6,800 companies, ~95 pre-computed accounting-ratio
features, 3.2% positive rate), evaluated with class-imbalance-aware
metrics (ROC-AUC, PR-AUC) and probability calibration.

**In scope:** demonstrating a supervised-learning approach to PD
estimation and its calibration behavior, as the project's actual
supervised-ML component (per the project's framing: "an ML project
applied to finance," not "finance with ML sprinkled in").
**Explicitly out of scope, and this is the memo's most important line:**
**this model is not used anywhere in the project's CVA calculation.** It
scores the Taiwan dataset's companies, not the Phase 1/2 ten-ticker panel
Phase 7/9 build CVA for. There is no path by which this model's output
reaches a CVA number in the current codebase.

**Decision this model informs:** none, operationally, in this project as
built. Its purpose here is methodological demonstration and a documented,
honest illustration of a real deployment constraint (Section 3) — which is
itself a legitimate and common real-world model-risk finding, not a null
result.

## 2. Conceptual soundness

Two complementary approaches, both standard: logistic regression
(interpretable, linear-in-features baseline) and gradient-boosted trees
(XGBoost, flexible nonlinear baseline), both trained on the same
class-weighted data (`class_weight='balanced'` / `scale_pos_weight`) to
address the 3.2% positive rate, then independently recalibrated via
cross-validated Platt (sigmoid) scaling. This pairing is conceptually
sound for a from-scratch supervised PD model: an interpretable baseline
plus a stronger flexible model, evaluated on ranking (AUC-family metrics,
appropriate for imbalanced data — not accuracy) and separately on
calibration (whether predicted probabilities mean what they say), which
are genuinely different properties and were correctly not conflated.

**Alternative considered:** a Merton-baseline row in the same comparison
table (the plan's explicit ask). Not implementable — see Section 3.

## 3. Data lineage and quality

- Source: UCI's public mirror of the Taiwan Economic Journal bankruptcy
  dataset (1999–2009), fetched via `data_pipeline.fetch_bankruptcy_dataset`,
  cached to `data/raw/taiwan_bankruptcy.csv`.
- Features: ~95 pre-computed accounting ratios, engineered upstream by the
  dataset's original publishers, not by this project (`src/features.py`
  drops constant/near-duplicate columns but does not re-derive the ratios
  from raw statements).
- **Critical lineage gap: no per-row date or fiscal-year field.** The
  public release carries no timestamp, which forecloses the plan's called-
  for time-based train/test split — a stratified random split
  (`split_bankruptcy_dataset`) is used instead. This protects the severely
  imbalanced positive class from further distortion by chance, but **does
  not protect against lookahead bias** the way a genuine time split would.
  This is a data-provenance limitation inherited from the public dataset,
  not a modeling choice, and it means any claim of this classifier's
  real-world temporal generalization is unverified.
- **No overlap with the Phase 1/2 company panel.** The Taiwan dataset's
  companies and the ten-ticker panel (AAPL, MSFT, etc.) are two entirely
  separate universes with zero shared identifiers, and the accounting-
  ratio feature set here has no market-equity-value/volatility fields
  Merton needs. This is why Section 1's "not used in CVA" scope statement
  isn't a choice this project made lightly — there is no data-level path
  to connect this model's training corpus to the panel it would need to
  score.

## 4. Methodology summary (plain language)

Both models are shown ~95 financial ratios per company (profitability,
leverage, liquidity measures) and learn to separate the ~3.2% that later
went bankrupt from the rest. Because bankruptcies are rare, both models
are told to weight the rare positive examples more heavily during
training so they don't just learn to always predict "safe" — but that
correction, useful for *ranking* companies by risk, tends to overstate how
likely any individual company actually is to default. A second step
(Platt scaling) recalibrates the raw scores so that "40% predicted risk"
actually corresponds to roughly 40% of similarly-scored companies
defaulting in the held-out test data, without changing which companies
rank as riskier than which others.

## 5. Performance / benchmarking results (held-out test set)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.86 | 0.26 |
| XGBoost | **0.95** | **0.44** |

- Gradient boosting clearly wins on both ranking metrics on this dataset.
- **Calibration finding, found and fixed during this phase:** both raw
  (`class_weight`/`scale_pos_weight`-adjusted) models are meaningfully
  overconfident — the raw logistic regression's top predicted-probability
  bin predicts ~85% but only ~20% of those companies actually went
  bankrupt. Cross-validated sigmoid (Platt) calibration closes this gap
  visibly (`outputs/pd_classifier_calibration.png`) without moving
  ROC-AUC/PR-AUC — confirming the fix addresses calibration specifically,
  not ranking quality (which was never the broken property).
- Sigmoid rather than isotonic calibration was a deliberate choice: with
  only ~165 positive training examples spread across 5 CV folds,
  isotonic's more flexible mapping risks overfitting the calibration curve
  itself.
- **Pipeline correctness test** (`tests/test_pd_classifier.py`): both
  classifiers fit and clear chance-level ROC-AUC on a synthetic, linearly
  separable dataset — validates the training/eval pipeline mechanically,
  independent of the real dataset's specific numbers.

## 6. Known limitations and weaknesses (specific, not generic)

- **Zero real-world applicability to this project's actual CVA
  counterparty universe** (Section 1/3) — the single most important
  finding of this memo. A reviewer signing off on "this classifier is
  good" must not read that as "this classifier informs any number this
  project reports."
- **Lookahead-bias exposure from the non-time-based split** (Section 3) —
  ROC-AUC/PR-AUC above are trustworthy as *held-out* numbers under a
  random split, but the random split does not rule out subtle lookahead
  leakage a true chronological split would catch (e.g. features computed
  using information that would not have been available at prediction
  time, systematically differing between eventual bankrupts and
  survivors in ways correlated with filing date).
- **No Merton comparison ever performed** — the plan's explicit
  requested check, blocked by a genuine data-availability gap (Section 3),
  not skipped for convenience.
- **Feature engineering opacity** — the ~95 ratios are pre-computed
  upstream; this project did not independently verify their construction
  from raw financial statements, so any latent error in the public
  dataset's own feature engineering would propagate silently.
- **Historical/regional generalization is untested** — Taiwan, 1999–2009.
  No evidence this project has generated speaks to how well this
  classifier would generalize to, say, current-day US large-cap credit
  risk (the very universe the CVA calculation actually needs).

## 7. Ongoing monitoring plan

Given Section 1's scope finding, the honest monitoring plan for THIS
project is: **none required, because this model is not in the production
path.** If a future iteration of this project connected this classifier to
a live scoring use (e.g. by building a dated, self-collected panel per the
plan's alternate "Path 2"), the monitoring plan would need, at minimum:
population stability checks on the input ratios, periodic recalibration
(Platt scaling parameters drift as base rates and score distributions
shift), and a genuine backtest against realized defaults in whatever new
universe it's applied to — none of which exists today because there is no
live use to monitor.

## 8. Model tiering / materiality

**Tier: methodological demonstration, zero materiality to this project's
actual reported numbers.** This is not a weakness of the model itself —
ROC-AUC 0.95 / PR-AUC 0.44 (XGBoost) is a genuinely strong result on this
dataset — it is a scope finding about *this project's* use of it. Any
reader interpreting this project's CVA numbers as "informed by machine-
learning PD estimation" would be wrong: the CVA numbers are entirely
Merton-baseline-driven (see the Merton memo). This classifier's real
contribution to the project is demonstrating the supervised-learning
methodology honestly, including its calibration failure mode and fix, on
the one dataset available to demonstrate it on.
