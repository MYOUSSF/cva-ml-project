# Model Validation Memo: Merton Structural PD Baseline

**Model owner:** Phase 2 (`src/pd_model.py`, `solve_merton` / `distance_to_default` / `merton_pd`)
**Validation date:** 2026-08-16 | **Valuation date scored:** 2026-08-15
**Reviewer framing:** structured against the SR 11-7 validation pillars (conceptual soundness, ongoing monitoring, outcomes analysis) as a documentation exercise -- this is a portfolio project, not a regulated institution, and this memo is not a substitute for an independent model validation function.

## 1. Purpose and scope

Estimates a 1-year probability of default for a single counterparty from
observable market/balance-sheet inputs (equity value, equity volatility,
face value of debt, risk-free rate), by treating equity as a call option
on the firm's assets (Black-Scholes/Merton).

**In scope:** producing the counterparty PD term structure that feeds
Phase 7's CVA calculation (`compute_cva`), for the ten-name Phase 1/2
company panel. **Out of scope:** default timing/path (the model only
scores default-by-horizon-T, not when within [0,T]); any use of its raw
PD magnitude as a standalone credit judgment (see Section 6); recovery/LGD
estimation (handled separately in Phase 6, a fixed market convention, not
this model's output).

**Decision this model informs:** which counterparty's default risk feeds a
CVA number, and — via `pd_term_structure_from_1y_pd`'s constant-hazard
extrapolation — the shape of the term structure used in that calculation.
Also feeds two extensions built on top of it: the wrong-way-risk copula
(`src/wrong_way_risk.py`, reuses this model's own solved `V`/`sigma_V`)
and the external spread benchmark (`src/spread_benchmark.py`).

## 2. Conceptual soundness

Standard structural credit risk model (Merton 1974): solves the two-
equation system relating observed equity value/volatility to unobserved
asset value/volatility via `scipy.optimize.fsolve`, then computes Distance
to Default and `PD ≈ N(-DD)`. This is textbook methodology, well-published,
and appropriate as a *baseline* -- the plan's own framing, not this
memo's addition (`src/pd_model.py`'s Phase 2 docstring: "feature source,
not the final model").

**Alternatives considered (per the plan):** a purely accounting-ratio
supervised classifier (Phase 2b) was built specifically because Merton's
known weaknesses (Section 6) make it unsuitable as a standalone final
answer. The two were not merged into a single model because they need
different input data the project doesn't have for the same company
universe (Section 3) -- documented as a real limitation, not silently
worked around.

**Key structural assumption not separately tested here:** equity markets
are informationally efficient enough that quoted equity value/volatility
are unbiased signals of the firm's true asset value/volatility. This is
plausible for the panel's large-cap, liquid names (AAPL, MSFT, JNJ, etc.)
but would be a materially weaker assumption for a thinly-traded small-cap
counterparty -- this model has not been validated on any name outside the
current ten-ticker panel and shouldn't be assumed to generalize there
without re-checking that assumption.

## 3. Data lineage and quality

- Equity value/volatility: `yfinance` market cap snapshot + 252-day
  trailing realized volatility of daily log returns, as of the valuation
  date (`data_pipeline.compute_realized_equity_vol`).
- Debt: SEC EDGAR XBRL company-facts API, `short_term_debt + 0.5 *
  long_term_debt` convention, most-recent-tag-value-as-of-valuation-date
  logic that explicitly checks every candidate tag rather than the first
  one with any data (`data_pipeline._latest_fact_asof`).
- Risk-free rate: interpolated from the Phase 1 FRED Treasury/SOFR curve
  at the configured horizon.

**Known data quality issue, materially affecting this model's output for
one panel name:** Ford (`F`)'s debt figure is flagged stale by Phase 1's
own sanity check (`check_panel_sanity`) -- no SEC tag more recent than
2020-12-31 was found, likely because Ford discloses debt via
company-specific extension tags outside the standard `us-gaap` set this
pipeline scans. Ford's Merton PD is consequently understated (its debt
input badly understates real leverage), which visibly propagates all the
way to Phase 9's portfolio ranking (Ford ranks among the safest names
despite a BB+ speculative-grade rating). This is a garbage-in-garbage-out
case, not a defect in the Merton solver itself, but it means **Ford's
score from this model should not be trusted** until the debt figure is
fixed at the source.

## 4. Methodology summary (plain language)

Equity is treated as a call option the shareholders hold on the firm's
total assets, struck at the face value of debt: shareholders only have
value left over if the assets are worth more than what's owed once the
debt matures. Given today's equity value and how much that equity price
bounces around, the model backs out an implied "true" asset value and
asset volatility (via a 2-equation solve, since asset value/volatility
aren't directly observable — only their equity-option payoff is). From
there it measures how many standard deviations of asset-value movement
would separate the firm from insolvency (Distance to Default) and
converts that distance to a probability via the normal CDF.

## 5. Performance / benchmarking results

- **Solver convergence:** 10/10 companies in the current panel converge
  (relative equation residual < 1e-8), with a multi-start retry (six
  initial `sigma_V` guesses) added specifically because highly-levered
  names like AMC needed a different starting point than AAPL for `fsolve`
  to find the economically valid root.
- **Known-answer recovery test** (`tests/test_pd_model.py`): generates
  synthetic `(E, sigma_E)` by running the Merton equations *forward* from
  a chosen `(V, sigma_V)`, then confirms `solve_merton` recovers the
  original values -- validates solver correctness independent of any real
  data quality issue.
- **Rank-ordering at the extremes:** correct. AMC (chosen precisely because
  it's the panel's one clearly speculative-grade name) comes out ~750x
  riskier than the next-highest PD, and the genuinely investment-grade
  names cluster at the bottom.
- **No formal backtest against realized defaults exists or is possible**:
  the panel has zero realized defaults in its history. Performance
  evidence here is limited to solver-correctness and coarse rank-order
  plausibility, not calibration against outcomes -- a real limitation
  for any model risk sign-off, stated plainly rather than implied away.
- **External benchmark (this extension pack, `src/spread_benchmark.py`):**
  comparing PD x LGD to FRED's rating-bucket corporate OAS series shows
  the model-implied spread understates the market spread by 40-620bps
  depending on rating bucket, growing with credit quality -- i.e. the
  model's *magnitude* miscalibration (Section 6) is now externally
  quantified, not just asserted.

## 6. Known limitations and weaknesses (specific, not generic)

- **Investment-grade PDs collapse to implausible magnitudes** (1e-40 to
  1e-76 across AAPL/MSFT/JNJ/KO/XOM) rather than realistic sub-1% figures.
  This is a well-documented property of plain Merton with physical-measure
  (not risk-neutral, not market-implied) inputs for very safe firms — the
  model is extremely sensitive to Distance-to-Default in the tail of the
  normal CDF, and physical-measure inputs push DD far into that tail for
  low-leverage names. **Practical consequence: this model's PD magnitude
  must never be used directly** (e.g. as a spread, as a capital number);
  only its coarse categorical/rank signal is usable, and even that is
  unreliable within the investment-grade cluster (see Section 5 external
  benchmark).
- **Ford's specific data-quality failure** (Section 3) — a live example of
  this model's total dependence on debt-data freshness, with no internal
  cross-check to catch a stale-but-plausible-looking number on its own.
- **No head-to-head comparison against Phase 2b's supervised classifier**
  exists on the same companies, because no available dataset supports it
  (Section 3) — this model's discriminative power has never been measured
  against a real alternative on this specific universe.
- **Single-point-in-time equity vol** (252-day trailing realized vol) as
  the volatility input, not an implied/market-forward-looking measure --
  the model's forward PD estimate rests entirely on a backward-looking
  statistic.

## 7. Ongoing monitoring plan

If this were operationalized rather than a one-time portfolio-project run:
- **Re-run trigger:** any refresh of the FRED curve, equity snapshot, or
  SEC debt figures for a scored counterparty (i.e., re-score whenever the
  inputs change, not on a fixed calendar).
- **Track over time:** solver convergence rate (currently 10/10; a drop
  would signal an input-data regime the multi-start retry doesn't cover),
  and the magnitude of PD for investment-grade names (should stay near-zero
  under this methodology by construction — an abrupt jump would flag a
  data bug before it flags real credit deterioration).
- **Staleness check:** the debt-staleness flag from `check_panel_sanity`
  (>548 days) should gate whether a name's score is used at all, not just
  be logged — Ford is the live example of what happens when it isn't
  enforced as a hard gate.
- **Re-validation trigger:** any extension of the panel beyond the current
  ten large-cap names, given the informational-efficiency assumption in
  Section 2 is untested outside that universe.

## 8. Model tiering / materiality

**Tier: baseline / feature-source, not standalone-decision-grade**, per the
plan's own framing. Appropriate weight in this project: it is the sole PD
input to every downstream CVA number (Phases 7, 9, and both new
extensions), so its coarse-categorical signal ("AMC is much riskier than
the rest of the panel") is load-bearing for those results' *direction*.
Its PD *magnitude* should carry near-zero independent weight in any
decision beyond that coarse signal — this is stated explicitly throughout
the project's own README, not a new finding of this memo, but this memo's
job is to make that materiality judgment the formal, structured
conclusion a validation memo is supposed to reach, rather than a caveat
buried in a docstring.
