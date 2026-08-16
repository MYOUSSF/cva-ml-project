# Model Inventory Summary

Structured per Extension 1 of `cva_risk_extensions_pack.md` — one row per
model, its purpose, and current validation status. Individual memos have
the full detail (conceptual soundness, data lineage, limitations,
monitoring plan, materiality); this is the reviewer-facing index.

**Note on scope:** this is a documentation exercise for a portfolio
project, framed against SR 11-7's pillars because that's the standard
reference point for what a bank model-risk function's output looks like —
it is not performed by an independent validation function and should not
be read as an equivalent of one.

| Model | Module | Purpose | In production path? | ROC-AUC / MAE / key metric | Materiality tier | Memo |
|---|---|---|---|---|---|---|
| Merton structural PD | `src/pd_model.py` (Phase 2) | Counterparty 1y PD feeding CVA's PD term structure | **Yes** — sole PD input to every CVA number in this project | Rank-order correct at extremes; magnitude collapses to 1e-40 to 1e-76 for IG names (known Merton-with-physical-inputs limitation) | Baseline / feature-source; PD magnitude never independently trustworthy, only coarse rank signal | [merton_pd_baseline.md](merton_pd_baseline.md) |
| Supervised PD classifier | `src/pd_model.py` (Phase 2b) | Bankruptcy classification on Taiwan dataset | **No** — different company universe than the CVA panel, zero data-level connection | XGBoost ROC-AUC 0.95 / PR-AUC 0.44 (vs. LR 0.86 / 0.26) | Methodological demonstration only; zero materiality to reported numbers | [pd_classifier.md](pd_classifier.md) |
| Hull-White rate model | `src/ratemodel.py` (Phase 3) | Short-rate dynamics driving every simulated exposure and discount factor | **Yes** — most structurally load-bearing model in the project | MC-vs-closed-form 0.04bp MAE; lag-1 residual autocorrelation -0.25 (daily AR(1) imprecision, honestly flagged) | Structurally critical, highest materiality | [hull_white_calibration.md](hull_white_calibration.md) |
| Exposure regression models (naive / poly / RF / GBM / MLP) | `src/exposure_models.py` (Phase 4/5) | EE(t) estimation feeding CVA | **Yes** — selected model (poly) directly sets the CVA exposure curve | poly MAE 3,482 vs. naive's 267,699 vs. nested-sim ground truth; naive-vs-poly choice alone swings CVA -29% | Production-selected, high materiality | [exposure_regression_models.md](exposure_regression_models.md) |
| PINN pricer | `src/pinn.py` (Phase 8) | Physics-informed neural bond pricer | **No** — Phase 4/5/7 all use the closed form directly | Mean abs price error 0.0032 vs. closed form; closed form is faster (0.81ms vs. 7.32ms per 10k evals) | Methodological demonstration only; zero materiality to reported numbers | [pinn_pricer.md](pinn_pricer.md) |

## Cross-model observations

- **Two of five models (PD classifier, PINN) carry zero materiality to
  this project's actual reported numbers**, by explicit, honest design —
  both exist to demonstrate a technique (supervised classification,
  physics-informed learning) in a setting where it can be independently
  checked, not because the project pretends they're load-bearing. A
  reviewer should weight validation findings on these two models
  accordingly: rigor in *how* they were validated matters for the
  demonstration's credibility, but neither model's weaknesses can move a
  single number this project reports.
- **The two models actually driving every CVA number (Merton PD, exposure
  regression) both have known, quantified magnitude/selection caveats**
  that any consumer of this project's CVA figures should treat as live
  limitations, not resolved issues: Merton's PD magnitude is not
  independently trustworthy outside coarse rank-ordering, and the
  exposure-model selection was validated on only 15 (checkpoint, rate)
  points against the nested-sim ground truth.
- **Hull-White sits underneath everything**, including this extension
  pack's own bootstrapped-CI and wrong-way-risk additions (both directly
  resample or extend its simulated paths) — its daily-calibration
  imprecision (Section 6 of its memo) is arguably the single highest-
  leverage limitation in the whole project, since nothing downstream can
  average it away.
- **This extension pack's own three new models are validated in their
  respective module docstrings and test suites** (`src/cva_uncertainty.py`,
  `src/wrong_way_risk.py`, `src/spread_benchmark.py`) rather than getting
  separate SR 11-7-style memos here — they are direct, small extensions of
  already-inventoried models (the bootstrap resamples Merton/exposure
  outputs; wrong-way risk reuses Merton's own solved `V`/`sigma_V` inside
  a correlated Hull-White simulation; the spread benchmark is a pure
  read-only comparison against external market data), not new standalone
  models with their own conceptual-soundness story to tell.
