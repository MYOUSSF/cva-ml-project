# Counterparty Credit Risk (CVA): A Data Science / ML Portfolio Project

**Framing:** this is an ML project that happens to be applied to finance, not a finance project with ML sprinkled in. Every phase below is built around a standard ML problem type — classification, regression, time-series estimation, deep learning — with the finance content serving as the applied domain, not the organizing structure. Use the table in Section 1 as your mental model, and lean on that language in your README, resume bullets, and interview answers.

---

## 1. Problem Decomposition — Translate Finance Into ML

| Finance concept | ML problem type | Techniques |
|---|---|---|
| Probability of Default (PD) | **Binary classification + probability calibration** | Logistic regression & gradient boosting (XGBoost/LightGBM) trained on cross-sectional data; structural (Merton) model used as an *engineered feature*, not the final answer |
| Expected Exposure, EE(t) | **Regression / conditional expectation estimation under uncertainty** | Polynomial OLS (classical baseline), random forest regression, gradient boosted regression, small MLP — benchmarked against a simulation-based ground truth |
| Interest rate dynamics | **Time-series parameter estimation** | AR(1)-style regression (method of moments), residual diagnostics |
| Fast derivative pricing | **Physics-informed deep learning / scientific ML** | PINN in PyTorch — custom PDE-residual loss, autograd, no labeled training data |
| Scarcity of real default data | **Simulation-based data augmentation** | Monte Carlo used to generate a large, well-understood synthetic dataset with known ground truth, since real defaults are rare events |
| Everything above | **Rigorous evaluation methodology** | Time-based train/test splits (no lookahead), held-out validation, ROC-AUC/PR-AUC/calibration curves, MAE/RMSE against ground truth, bias-variance analysis |

Keep this table (or a version of it) near the top of your actual README — it's doing a lot of work to signal "ML thinking" to a reviewer skimming your repo.

---

## 2. Tech Stack

- Python 3.x, `numpy`, `pandas`, `scipy`
- `scikit-learn` (logistic regression, random forest, train/test utilities, metrics)
- `xgboost` or `lightgbm` (gradient boosting classifier/regressor)
- `torch` (PINN, small MLP regressor)
- `matplotlib` / `seaborn` for diagnostics
- `yfinance`, `requests` (FRED, SEC EDGAR)
- `pytest` for a handful of unit tests on your core functions (a real DS/MLE differentiator versus a pile of notebooks)
- Optional: `streamlit` for an interactive results app, `optuna` for hyperparameter tuning

---

## 3. Suggested Repository Structure

```
cva-ml-project/
  data/
    raw/                    # cached raw pulls
    processed/               # cleaned, feature-engineered datasets
  src/
    data_pipeline.py          # FRED + SEC EDGAR + yfinance ingestion
    features.py                # feature engineering (ratios, Merton DD, etc.)
    pd_model.py                  # Merton baseline + supervised classifier
    ratemodel.py                  # Hull-White calibration + simulation
    exposure_models.py              # regression model comparison for EE(t)
    cva.py                            # final CVA assembly + sensitivity
    pinn.py                            # PINN fast pricer
  notebooks/                # exploratory analysis, results walkthroughs
  tests/                     # pytest unit tests on core functions
  outputs/                   # saved charts, tables, model artifacts
  app.py                     # optional Streamlit demo
  requirements.txt
  README.md                  # written for a DS audience — see Section 10
```

---

## 4. Phase 0 — Setup & Problem Framing

- Write a one-paragraph problem statement *before* touching data: "I'm building a system that estimates counterparty credit risk (CVA) by combining a default-probability classifier, a Monte-Carlo-based exposure simulator, and a deep-learning-accelerated pricer." This becomes your README opener and your elevator pitch in interviews.
- Decide on a cross-sectional universe of companies up front (you'll need multiple companies from the start now, not just one — see Phase 2).

---

## 5. Phase 1 — Data Engineering

- Pull the Treasury/SOFR curve from FRED for a chosen valuation date (series like `DGS1MO` through `DGS30`, `SOFR` — confirm exact IDs on the FRED site).
- Pull equity price history (`yfinance`) and balance sheet data (SEC EDGAR company facts API, `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, CIK lookup via `https://www.sec.gov/files/company_tickers.json`) for a **panel of companies** (aim for enough breadth to make Phase 2b's classifier meaningful — see below for dataset options).
- Build a clean, reusable ingestion pipeline (`data_pipeline.py`) rather than one-off notebook cells — this is a real signal of engineering maturity to a DS reviewer.
- **Validation:** null/missing-value audit, basic distribution checks (sanity-check ranges), and a fitted-curve round-trip check (fit the discount curve, confirm it reproduces the input yields).

---

## 6. Phase 2 — PD Baseline: Structural (Merton) Model

- Implement the Merton structural model as a **baseline and feature source**, not the final model:
  - Solve simultaneously for asset value V and asset volatility σ_V given observed equity value E, equity volatility σ_E, debt D, and risk-free rate r (`scipy.optimize.fsolve` on the two-equation system).
  - Compute Distance to Default: **DD = [ln(V/D) + (r − 0.5σ_V²)T] / (σ_V√T)**, and **PD ≈ N(−DD)**.
- Treat the resulting DD as an **engineered feature** you'll feed into Phase 2b's classifier, not as your final PD output.
- **Validation:** solver convergence check; PD magnitudes should rank-order sensibly across companies of different credit quality.

---

## 7. Phase 2b — PD Classifier: The Real Supervised Learning Component

This is the phase that makes PD estimation genuinely "ML" rather than "a formula."

- **Get a labeled dataset.** Two realistic paths:
  1. A public cross-sectional bankruptcy-prediction dataset with financial-ratio features and default labels (search Kaggle for "company bankruptcy prediction" — a well-known dataset built from Taiwan Economic Journal data with ~6,800 companies and ~95 accounting-ratio features exists and is commonly used for exactly this task; verify current availability when you start).
  2. Build your own panel: cross-sectional financial ratios from SEC EDGAR for many companies, labeled using a public bankruptcy list (e.g. the UCLA-LoPucki Bankruptcy Research Database for large public companies, or manually cross-referencing Chapter 11 filings). More work, more original, more impressive if you pull it off.
- **Feature engineering:** standard accounting ratios (leverage, liquidity, profitability — Altman Z-score components are a well-known starting set), plus your Phase 2 Merton-derived Distance-to-Default as an additional engineered feature. This hybrid (domain-theory feature + market/accounting features) is a genuinely strong thing to describe in an interview.
- **Train/test split: time-based, not random.** If your data spans multiple years, split by time (train on earlier years, test on later ones) to avoid lookahead bias — financial data has temporal structure, and a random split would leak future information into training. Say this explicitly in your write-up; it's exactly the kind of thing a good DS interviewer probes for.
- **Handle class imbalance.** Defaults are rare — don't report plain accuracy. Use class weighting or resampling, and evaluate with **ROC-AUC and precision-recall AUC**, not accuracy.
- **Models to compare:** logistic regression (interpretable baseline) vs. gradient boosting (XGBoost/LightGBM). Report a proper comparison table.
- **Calibration matters here specifically:** PD needs to be a genuinely well-calibrated *probability*, not just a good ranking — plot a calibration curve (predicted probability vs. observed default frequency in bins) and discuss whether your classifier is over/under-confident. This is an underused diagnostic that will make your project stand out.
- **Validation:** ROC-AUC, PR-AUC, calibration curve, and a comparison against the Merton-only baseline from Phase 2 (does the ML classifier actually beat the structural baseline? Report this honestly either way — a well-reasoned "it doesn't help much, and here's why" is a *better* interview story than blind model-stacking).

---

## 8. Phase 3 — Time-Series Model Calibration (Hull-White)

- Estimate mean-reversion speed *a* and volatility σ from a historical short-rate time series (FRED) via an AR(1)-style regression: regress the rate change Δr_t on the lagged level r_t; the slope gives −a·Δt, the residual standard deviation (annualized) gives σ.
- Calibrate θ(t) analytically to match the current forward curve (Phase 1).
- **Validation:** residual diagnostics on the AR(1) fit (are residuals roughly homoscedastic and uncorrelated — the standard regression-diagnostic checks you'd run on any time-series model); simulated paths should look plausible around the current curve.

---

## 9. Phase 4 — Monte Carlo as Synthetic Data Generation

- Frame this explicitly as a **data generation step**, not just "finance simulation": since real derivative exposure labels effectively don't exist (you can't observe the "true" conditional value of a trade at a future date without waiting for it to happen), Monte Carlo simulation generates a large synthetic dataset with a known, controllable data-generating process — directly analogous to using simulators to augment scarce real-world labels in other ML domains (robotics, autonomous driving, etc.).
- Simulate short-rate paths under your calibrated Hull-White parameters for a hypothetical swap.
- This dataset (path states + realized future cash flows at each checkpoint) is the **training data** for Phase 5.

---

## 10. Phase 5 — Exposure Estimation as a Regression Model Comparison

This is the central modeling phase of the whole project — treat it like a genuine ML benchmarking study.

- **The learning task:** given the state at a checkpoint (current rate, plus any already-fixed-but-unpaid cash flow info), predict the conditional expected future value of the trade. Each simulated path gives one noisy training example (its own realized future cash flow); the *true* target is the smooth conditional expectation, which you don't directly observe — a nice, honest "noisy labels" framing.
- **Build a naive baseline first** — average the raw noisy labels directly (floor, then average) — and show, using Jensen's inequality reasoning plus an empirical check, that this baseline is systematically biased. This bias-variance story is a genuinely strong thing to lead with in a portfolio project; it's a real statistical insight, not just "I fit a model."
- **Compare multiple regression models** against that baseline:
  - Polynomial OLS on a small hand-chosen basis (the classical "Longstaff-Schwartz" approach — cheap, interpretable, a good baseline-plus-one)
  - Random forest regression
  - Gradient boosted regression
  - A small MLP (PyTorch or `sklearn.neural_network.MLPRegressor`)
- **Build a held-out ground truth to evaluate against:** run an expensive nested-simulation estimate at a handful of checkpoints (bin states, launch fresh forward simulations from each bin, average without flooring) — this plays the role of a genuine held-out test set with (nearly) noise-free labels, letting you report real **MAE/RMSE** for each model against something trustworthy, not just against each other.
- **Report a clean model comparison table**: naive baseline vs. each regression model, MAE/RMSE against nested-sim ground truth, plus a qualitative discussion of overfitting risk (does the random forest / MLP overfit the simulation noise if given too much flexibility relative to the smoothness of the true underlying function? This is worth actually checking, not assuming).
- **Validation:** the naive baseline's error should be systematically one-directional (upward bias) versus the more flexible models', which should be closer to zero-centered — confirm this rather than asserting it.

---

## 11. Phase 6 — LGD Assumptions (Brief)

- Use a current published aggregate recovery rate (Moody's/S&P) appropriate to seniority. LGD = 1 − Recovery Rate. This phase is intentionally small — it's domain input, not a modeling exercise.

---

## 12. Phase 7 — CVA Assembly & Sensitivity Analysis

- Combine your Phase 2b PD term structure, Phase 5's best-performing regression model's EE(t) curve, Phase 6's LGD, and Phase 1/3's discount curve into the final CVA number.
- Run a basic sensitivity analysis: how does CVA change if PD is shocked, or if the exposure model choice (naive vs. best regression model) is swapped in? This turns your Phase 5 model comparison into a *business-relevant* result — "using a better-calibrated exposure model changes the estimated CVA by $X" is a much stronger takeaway than a bare accuracy table.
- **Validation:** CVA should move monotonically with PD/LGD shocks (a basic model-sanity check, cheap to run, good to show).

---

## 13. Phase 8 — Deep Learning Extension: Physics-Informed Neural Network

- Train a PINN (PyTorch, `autograd`) to solve the bond-pricing PDE directly using your calibrated Hull-White parameters — no labeled price data in training, just the PDE residual and a hard-constrained terminal condition.
- Validate against your model's closed-form solution: report max/mean absolute error, a training-loss curve, and error concentrated where you'd expect (away from the hard-constrained boundary).
- Frame this explicitly as "scientific ML" in your write-up — it's a genuinely strong, currently-fashionable technique to have hands-on experience with, and it demonstrates comfort with custom loss functions and automatic differentiation beyond standard supervised learning.
- **Be honest about scope:** in one dimension this doesn't beat a closed form on speed — say so, and explain that the real value is in higher-dimensional settings where grid-based PDE solvers become infeasible. This kind of honest limitations discussion is exactly what separates a thoughtful DS candidate from someone padding a resume with buzzwords.

---

## 14. Phase 9 — Portfolio-Level Application

- Apply your full pipeline (PD classifier + exposure model + CVA) across your panel of companies from Phase 1.
- Present this as a **model deployed across a segment**, not just "more examples" — show how PD, EE(t) shape, and CVA vary systematically with credit quality, and discuss what that implies about your PD classifier's discriminative power in a way that's genuinely useful (e.g. rank companies by predicted CVA and sanity-check against public credit ratings if available).

---

## 15. Phase 10 — Packaging, Reproducibility, and the DS-Facing Write-Up

- Add a handful of `pytest` unit tests on your core functions (e.g. the discount curve round-trips, the Merton solver converges on a synthetic test case with a known answer, the regression pipeline runs end-to-end on a small synthetic dataset). Even 5–10 tests meaningfully changes how a reviewer reads your repo.
- Make the pipeline config-driven (a simple YAML/argparse setup so someone can rerun it for a different company or date without editing source) rather than hardcoded notebook cells.
- Optional: a small Streamlit app letting a reviewer pick a company and see PD, EE(t), and CVA update live — a strong, low-effort differentiator for a DS portfolio, since it shows product/communication instinct beyond modeling.
- Write the README for a DS audience: lead with the problem-decomposition table (Section 1), then results, then methodology detail, then an explicit "Limitations" section (Merton-vs-classifier PD tradeoffs, historical-vs-market-implied calibration, single-factor PINN scope, synthetic-data caveats). A clear, honest limitations section reads as more senior than a project that claims no weaknesses.

---

## 16. Validation Checklist (Recap, ML-Flavored)

- [ ] Data pipeline: missing-value audit, distribution sanity checks
- [ ] Curve fit: round-trips to input yields
- [ ] Merton baseline: solver converges; PD ranks companies sensibly
- [ ] PD classifier: time-based split (no lookahead); ROC-AUC / PR-AUC / calibration curve reported; compared honestly against the Merton baseline
- [ ] Rate model: AR(1) residual diagnostics look reasonable
- [ ] Exposure models: naive baseline shows the expected upward bias; more flexible models checked for overfitting, not just accepted because they're more flexible; MAE/RMSE reported against nested-sim ground truth
- [ ] CVA: moves monotonically under PD/LGD shocks
- [ ] PINN: PDE residual loss converges; error against closed-form reported honestly, including where it's worst
- [ ] Repo: has tests, a requirements.txt, and a config-driven entry point — not just notebooks

---

## 17. Suggested Kickoff Prompts

**Phase 1:**
> I'm building a DS/ML portfolio project on counterparty credit risk (plan attached). Phase 1: build a reusable data ingestion pipeline pulling the FRED Treasury/SOFR curve and, for a panel of [N] companies, equity history (yfinance) and balance sheet data (SEC EDGAR). Include a missing-data audit and a curve round-trip validation check.

**Phase 2 + 2b:**
> Phase 2/2b: implement the Merton structural PD model as a baseline and feature source, then train a supervised PD classifier (logistic regression + gradient boosting) on [DATASET], using a time-based train/test split, class-imbalance-aware evaluation (ROC-AUC, PR-AUC), and a calibration curve. Compare the classifier honestly against the Merton baseline.

**Phase 3:**
> Phase 3: estimate Hull-White a and σ from FRED short-rate history via an AR(1)-style regression, run residual diagnostics, and calibrate θ(t) to the current forward curve.

**Phase 4 + 5:**
> Phase 4/5: simulate Hull-White paths as a synthetic dataset for a hypothetical swap, then build and compare exposure-estimation models (naive baseline, polynomial OLS, random forest, gradient boosting, small MLP) against a nested-simulation ground truth, reporting MAE/RMSE and discussing the bias-variance tradeoff.

**Phase 7:**
> Phase 7: assemble CVA from the Phase 2b PD, Phase 5's best exposure model, and an LGD assumption, and run a sensitivity analysis comparing CVA under the naive vs. best exposure model.

**Phase 8:**
> Phase 8: train a PINN on my calibrated Hull-White parameters to solve the bond-pricing PDE with no labeled data, validate against the closed-form solution, and discuss honestly where this approach would and wouldn't provide real value.

**Phase 10:**
> Phase 10: add pytest unit tests for the core pipeline functions, make the pipeline config-driven, and help me write a DS-audience README leading with a problem-decomposition table and an honest limitations section.

---

## 18. Pitfalls to Watch For

- **Lookahead bias:** any time-based split done carelessly (e.g. shuffling before splitting) silently invalidates your classifier evaluation — this is one of the most common real mistakes DS interviewers ask about, so get it right and be ready to explain it.
- **Class imbalance blindness:** reporting accuracy on a rare-event classifier (defaults) is a classic beginner mistake — use AUC-based metrics and say why.
- **Overfitting the simulation noise in Phase 5:** a sufficiently flexible model (deep random forest, unregularized MLP) can fit noise in the synthetic training data rather than the true smooth conditional expectation — check this against your nested-sim ground truth rather than assuming more flexibility is strictly better.
- **Debt-figure ambiguity in Merton:** be explicit and consistent about your debt convention (short-term debt + 0.5×long-term debt is standard).
- **PINN domain coverage:** collocation points must cover the full range your simulation actually visits, or the network extrapolates poorly outside training range.

---

## 19. Stretch Goals

- Hyperparameter tuning (`optuna` or grid search) for the classifier and regression models, with a proper nested cross-validation setup.
- SHAP-value interpretability analysis on the PD classifier — which features actually drive predicted default risk?
- A simple deployed API (FastAPI) wrapping the trained models, beyond just a Streamlit demo.
- Extend Phase 5's model comparison to a genuinely path-dependent product (a simple Bermudan-style feature) where the naive baseline would be *unusable*, not just biased — this sharpens the "why regression is necessary, not just nice" story even further.
