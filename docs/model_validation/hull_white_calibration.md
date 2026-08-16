# Model Validation Memo: Hull-White Interest Rate Model Calibration

**Model owner:** Phase 3 (`src/ratemodel.py` — `calibrate_ar1`,
`calibrate_theta`, `simulate_hull_white_paths`)
**Validation date:** 2026-08-16 | **Calibration date:** 2026-08-15
**Reviewer framing:** SR 11-7 pillars, portfolio-project documentation exercise (see caveat in the Merton memo).

## 1. Purpose and scope

Single-factor Gaussian short-rate model (Hull-White), calibrated in two
parts: mean-reversion speed `a` and volatility `sigma` from historical
daily SOFR via an AR(1)-style regression, and `theta(t)` analytically
fit to exactly reproduce the current FRED forward curve.

**In scope:** generating the short-rate paths every downstream module
depends on — Phase 4/5's exposure training data, Phase 7's discounting,
and both new extensions (bootstrapped CI's path resampling, wrong-way
risk's correlated asset-value simulation). This is the single most
structurally load-bearing model in the project — every other phase either
consumes its simulated paths directly or its calibrated curve for
discounting.
**Out of scope:** multi-factor rate dynamics, market-implied (as opposed
to historical) volatility calibration (see Section 6), and any credit-
spread or default-related dynamics (those live in the PD models).

## 2. Conceptual soundness

Hull-White is standard, well-published, and analytically tractable —
its closed-form zero-coupon bond pricing formula
(`hull_white_zero_coupon_bond`) is exactly what makes Phase 4/5's
regression-model benchmarking possible (an exact ground truth to check
noisy Monte Carlo labels against) and what the PINN pricer (Phase 8)
validates against. This tractability is a deliberate, sound choice for a
project whose actual goal is validating a *pipeline* (Monte Carlo →
regression → CVA), not proving a rate model can't be beaten — a
genuinely path-dependent, no-closed-form product would remove that
independent-verification advantage everywhere downstream.

**Two-part calibration is conceptually clean but couples two different
statistical regimes:** `a`/`sigma` come from a purely historical,
physical-measure regression; `theta(t)` is fit to force agreement with
today's *market* forward curve. This is a defensible and common
practitioner shortcut (calibrate dynamics historically, calibrate level to
market), but it means the model's volatility assumption and its curve-
level assumption come from genuinely different sources — worth naming
explicitly rather than presenting the combined calibration as a single
internally-consistent market-implied fit.

## 3. Data lineage and quality

- Historical short rate: daily SOFR from FRED's public `fredgraph.csv`
  export (no API key needed), 2,088 observations spanning 2018–2026 in
  the current run.
- Forward curve: the Phase 1 FRED Treasury/SOFR curve snapshot, itself
  round-trip validated (`validate_curve_roundtrip`) before this model
  consumes it.
- No data quality issues specific to this model surfaced during
  development — SOFR is a clean, continuously-published series with no
  missing-tag/staleness analog to the Merton memo's Ford finding.

## 4. Methodology summary (plain language)

Interest rates tend to drift back toward some long-run average level
rather than wandering off unboundedly — Hull-White captures that with a
"pull strength" parameter (`a`, how hard rates get pulled back) and a
"jumpiness" parameter (`sigma`, how much daily noise gets added on top).
Both are estimated by looking at how much SOFR actually moved on days when
it started high vs. low, historically. A third piece, `theta(t)`, is a
time-varying adjustment calculated (not statistically fit) so that,
averaged across all the model's randomness, the expected rate at every
future date exactly matches what today's yield curve already implies —
so the model doesn't just have plausible *dynamics*, it's anchored to
today's actual market pricing.

## 5. Performance / benchmarking results

- **Residual diagnostics** (`residual_diagnostics`): heteroscedasticity
  proxy (`|residual|` vs. rate level correlation) is fine at 0.02, but
  **lag-1 residual autocorrelation is -0.25**, not the ~0 a well-specified
  AR(1) would show — reported plainly rather than smoothed over (Section 6).
- **Closed-form correctness test** (`tests/test_ratemodel.py`): the
  analytically-derived `E[r(t)]` closed form is independently verified by
  numerically integrating the calibrated `theta(t)` (via
  `scipy.integrate.solve_ivp`) and confirming agreement — a genuine
  correctness check, not circular, since the two are computed via
  different code paths.
- **Known-parameter recovery test:** `calibrate_ar1` recovers synthetic
  `(a, sigma)` from a simulated path with known ground-truth parameters —
  using *monthly*, not daily, synthetic steps deliberately, because at
  daily frequency the AR(1) coefficient sits extremely close to a unit
  root, where OLS's known finite-sample bias gets amplified ~500x by the
  `a = -slope/dt` conversion. This is itself evidence for Section 6's
  daily-calibration-imprecision finding, not just a testing convenience.
- **Simulation sanity check:** Monte Carlo mean path (1,000 sims, 5y
  horizon) tracks the closed-form `E[r(t)]` to within **0.04bp** mean
  absolute error, and both track the input forward curve almost exactly —
  confirms `theta(t)` does what it's calibrated to do.

## 6. Known limitations and weaknesses (specific, not generic)

- **Daily AR(1) `a`/`sigma` estimates should be read as directionally
  reasonable, not precise.** At daily frequency the process sits close to
  a unit root; the -0.25 lag-1 residual autocorrelation (want: ~0)
  confirms a plain daily AR(1) does not fully capture SOFR's short-term
  dynamics (plausible causes: settlement/day-of-week effects, or mean
  reversion resolving faster than a 1-day lag). **This directly affects
  every downstream number**: Phase 4/5's exposure distribution width, and
  by extension Phase 7's CVA sensitivity, all inherit whatever imprecision
  exists in `sigma`.
- **No market-implied volatility calibration.** `sigma` comes entirely
  from historical realized rate volatility, not from swaption-implied
  volatility. The model's *forward-looking* volatility assumption is
  therefore backward-looking by construction — a materially different
  choice than a trading desk would typically make, and one this project
  does not attempt to correct.
- **Single-factor limitation, inherent to Hull-White**, not a calibration
  flaw: cannot represent decorrelated moves at different points on the
  curve (e.g. short end rallying while the long end sells off) — every
  point on the simulated curve moves through the same single Brownian
  driver.
- **Negative rates are a feature of the model, correctly not treated as a
  bug** (Gaussian short-rate models allow them, and negative rates are a
  real historical phenomenon in EUR/JPY/CHF), but worth flagging since it
  is nonetheless a real behavioral quirk relative to some jump/CIR-family
  alternatives.

## 7. Ongoing monitoring plan

- **Re-calibration cadence:** at minimum with every new valuation date
  (the config-driven pipeline already re-derives `a`/`sigma`/`theta(t)`
  per run — the risk is *not* re-running it, not a stale-model risk here).
- **Track over time:** lag-1 residual autocorrelation (currently -0.25) —
  a persistent, non-shrinking value across regimes would support switching
  the calibration window or considering a richer short-rate dynamic; a
  regime where it moves toward 0 would validate the AR(1) form itself is
  fine and the current reading is a feature of the current data window,
  not the specification.
- **MC-vs-closed-form drift check:** the 0.04bp MAE sanity check
  (`run_phase3`) should be re-run every recalibration — a sudden
  divergence would flag a `theta(t)` calibration bug before it silently
  corrupts every downstream Monte Carlo number.

## 8. Model tiering / materiality

**Tier: structurally critical, highest materiality in the project.**
Unlike the two PD models (which are either explicitly baseline-only or
entirely out of the production path), this model's calibrated parameters
flow directly and unavoidably into every simulated exposure, every
discount factor, and — via this extension pack — the correlated
asset-value process behind the wrong-way-risk CVA number. Its Section 6
limitations (daily-frequency imprecision in particular) should be weighted
accordingly: they are the most likely single source of quantitative error
in this project's headline CVA figures, more so than either PD model's
known weaknesses, because nothing downstream can average away or bound a
systematically mis-calibrated `sigma`.
