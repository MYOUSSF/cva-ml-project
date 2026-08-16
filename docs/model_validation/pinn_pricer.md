# Model Validation Memo: PINN Fast Pricer

**Model owner:** Phase 8 (`src/pinn.py`)
**Validation date:** 2026-08-16
**Reviewer framing:** SR 11-7 pillars, portfolio-project documentation exercise (see caveat in the Merton memo).

## 1. Purpose and scope

A small PyTorch MLP trained to solve the Hull-White zero-coupon
bond-pricing PDE directly via physics-informed learning (PDE-residual
loss + hard-constrained terminal condition, autograd-computed derivatives,
zero labeled price data).

**In scope:** a validated demonstration of the PINN technique in a setting
where ground truth (the closed-form Hull-White bond price) already exists
to check against. **Explicitly out of scope, stated up front per the
project's own honest framing:** this model is **not used anywhere in the
project's CVA pipeline.** Phase 4/5/7 all use the analytic closed form
(`hull_white_zero_coupon_bond`) directly, not this network. This memo's
scope finding mirrors the PD classifier memo's Section 1 finding, for the
same reason: demonstrating a technique honestly does not require pretending
it's load-bearing when it isn't.

## 2. Conceptual soundness

Genuinely correct application of the PINN recipe: the loss is the PDE
residual (via second-order autograd on network outputs with respect to
`t` and `r`), not a supervised label — this is scientific ML in the literal
sense, not supervised learning rebranded. The terminal condition is
**hard-constrained into the output parametrization**,
`P(t,r) = 1 + (T-t)*NN(t,r)`, rather than added as a soft penalty term —
this holds exactly for any network weights, a materially stronger
guarantee than the more common soft-penalty approach, and a deliberate,
well-justified architectural choice.

**Conceptual soundness caveat:** this is a 1D problem (state variables:
time, short rate) with a known closed form. The technique's actual value
proposition — tractability in higher-dimensional settings (multi-factor
rate models, stochastic vol) where no closed form exists and grid-based
PDE solvers become infeasible — is **asserted, not demonstrated**, by
this implementation. That's stated explicitly in the project's own
README, and this memo agrees it's the correct honest framing: what's
validated here is the recipe's correctness in a checkable setting, not its
value proposition in the setting where it would actually matter.

## 3. Data lineage and quality

No external data — the model is trained entirely against the PDE and the
already-validated Phase 3 Hull-White calibration (`theta(t)`, `a`,
`sigma`), read from the Phase 1 curve. No data-quality risk analogous to
the Merton or exposure-model memos; the relevant risk is purely in
training/architecture correctness (Section 5).

## 4. Methodology summary (plain language)

Bond prices, as a function of time and the current interest rate, must
satisfy a well-known differential equation (the same kind of equation that
describes heat diffusion). Rather than showing the network thousands of
example prices and having it learn to imitate them, this trains the
network to satisfy that equation directly everywhere it's evaluated,
using automatic differentiation to compute the equation's required
derivatives of the network's own output. The one thing known for certain —
a bond is worth exactly its face value at maturity — is built directly
into the network's mathematical form rather than left as something
training might or might not learn well.

## 5. Performance / benchmarking results

- **Accuracy vs. closed form:** max absolute price error 0.0127, mean
  absolute error **0.0032** (bond prices run ~0.79–1.00, so roughly
  0.3–1.3% relative error) — genuinely accurate for a from-scratch PDE
  solve with zero labeled data.
- **Error concentration checked quantitatively, not eyeballed:**
  correlation(time-to-maturity, absolute error) = **0.708** — error
  concentrates near `t=0`, fading toward `t=T`, exactly the pattern
  predicted by hard-constraining the *terminal* (not initial) condition.
  This is a real, testable, correctly-predicted consequence of the
  architectural choice in Section 2, and confirming it quantitatively
  (rather than just noting the heatmap looks that way) is good validation
  practice.
- **Known-derivative test** (`tests/test_pinn.py`): autograd-computed
  `dP/dt`, `dP/dr`, `d²P/dr²` match finite-difference derivatives of the
  *same untrained* network — validates the autograd wiring itself,
  independent of whether training converges. This is the right
  decomposition (architecture correctness vs. training-outcome
  correctness tested separately).
- **Exact-terminal-condition test:** `P(T,r)=1` holds to float precision
  for an untrained, randomly-initialized network — confirms it's a
  property of the architecture (Section 2's claim), not something
  training happens to have learned.
- **Training-improves-fit test:** loss decreases over training, and the
  trained network beats an untrained baseline on the same validation grid.
- **Speed, measured honestly against the actual alternative:** 0.81ms for
  10,000 closed-form evaluations vs. 7.32ms for the same on the trained
  network, plus ~16s of training the closed form never pays. **The closed
  form wins outright in this setting** — reported as the headline
  performance finding, not buried, because pretending otherwise would
  misrepresent the model's actual value here (Section 2).

## 6. Known limitations and weaknesses (specific, not generic)

- **No demonstrated advantage over the closed form, in the one setting
  tested.** This is the model's central, self-disclosed limitation, not a
  hedge — see Section 5's speed numbers.
- **Never tested in a setting where it would matter** (multi-factor,
  no-closed-form). The recipe's value proposition rests entirely on an
  untested extrapolation to a harder problem class.
- **A real infrastructure bug found during this phase, worth recording
  for its process lesson:** the full test suite hung 10+ minutes on what
  should have been seconds of PINN training — not a training-logic bug,
  but torch's default multi-threaded BLAS contending with
  numpy/scikit-learn/xgboost's own thread pools once all four were loaded
  in one process (confirmed by timing with/without
  `torch.set_num_threads(1)`: minutes vs. 3 seconds). Fixed by pinning
  thread count at module import. Recorded here because it's exactly the
  kind of environment-coupling failure a model validation process should
  care about — it doesn't affect this model's price-accuracy conclusions,
  but it did silently threaten the test suite's ability to keep validating
  them.
- **Single trained instance, default hyperparameters** (`hidden_size=64`,
  3,000 epochs) — no hyperparameter sensitivity analysis or seed-stability
  check across multiple training runs is reported, so the 0.0032 MAE
  figure should be read as one realization, not a characterized
  distribution.

## 7. Ongoing monitoring plan

Same finding as the PD classifier memo: **none required in production,
because this model is not in the production path.** If a future
extension of this project actually deployed a PINN in a setting without a
closed form (the scenario Section 2 says is the real test), the monitoring
plan would need: training-convergence checks per re-fit (loss curve, not
just final loss), a held-out grid of *some* independently-obtained
reference prices wherever possible (even partial — e.g. limiting cases,
known bounds) since a true closed form won't exist to check against
end-to-end, and seed-stability characterization given Section 6's single-
instance caveat.

## 8. Model tiering / materiality

**Tier: methodological demonstration, zero materiality to this project's
reported CVA numbers** — the same tier and reasoning as the PD classifier
memo. Its value is entirely in being a *correctly validated instance of a
recipe* (hard-constrained PDE residual, zero labeled data) that a reader
could reasonably extend to a setting where it would actually be needed.
Nothing in this project's headline results depends on this model's
accuracy, and no future change to this model's weights or training would
move any number this project reports.
