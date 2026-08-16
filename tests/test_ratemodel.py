"""Unit tests for Phase 3 (Hull-White calibration and simulation).

Two "known answer" checks, offline, no network calls:
  - calibrate_ar1 recovers approximately the true (a, sigma) used to
    simulate a synthetic short-rate path in the first place.
  - calibrate_theta's closed-form theta(t) is checked against an
    independently-derived closed form for E[r(t)] by numerically
    integrating the deterministic mean-reversion ODE and comparing -- this
    validates the calibration formula itself, not just self-consistency
    with simulate_hull_white_paths.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from src.ratemodel import (
    calibrate_ar1,
    calibrate_theta,
    expected_short_rate,
    fit_decimal_yield_curve,
    instantaneous_forward_rate,
    residual_diagnostics,
    simulate_hull_white_paths,
)


def _synthetic_short_rate_series(a=0.4, sigma=0.015, theta_const=0.012, r0=0.03,
                                  n_steps=5000, dt=1 / 12, seed=7) -> pd.Series:
    """A long, densely-sampled synthetic short-rate path from known (a,
    sigma) and a constant theta, as a date-indexed pd.Series -- the same
    shape calibrate_ar1 expects from a real FRED series.
    """
    horizon = n_steps * dt
    times, paths = simulate_hull_white_paths(
        a=a, sigma=sigma, theta=lambda t: theta_const, r0=r0,
        horizon=horizon, n_paths=1, n_steps=n_steps, seed=seed,
    )
    dates = pd.date_range("2000-01-01", periods=n_steps + 1, freq=f"{int(round(dt * 365))}D")
    return pd.Series(paths[0], index=dates)


def test_calibrate_ar1_recovers_known_parameters():
    # Uses monthly (dt=1/12), not daily, synthetic steps deliberately: at
    # a=0.4 the AR(1) coefficient exp(-a*dt) is 0.9992 at daily frequency
    # (dt=1/252) -- a near-unit-root process where OLS's well-documented
    # finite-sample bias in the autoregressive coefficient gets amplified
    # ~500x when converted to "a" (since a = -slope/dt divides a tiny,
    # noisy slope by a tiny dt). Monthly steps keep the same true dynamics
    # but with a large enough a*dt signal that OLS recovery is well-behaved
    # -- this test is about validating calibrate_ar1's regression math, not
    # re-deriving small-sample AR(1) estimation theory. The real Phase 3 run
    # (daily SOFR data) is genuinely noisier for `a`; that's a documented
    # limitation, not something this test needs to reproduce.
    true_a, true_sigma = 0.4, 0.015
    n_steps, dt = 3000, 1 / 12
    series = _synthetic_short_rate_series(a=true_a, sigma=true_sigma, n_steps=n_steps, dt=dt)

    calibration = calibrate_ar1(series, dt=dt)

    assert calibration["a"] == pytest.approx(true_a, rel=0.2)
    assert calibration["sigma"] == pytest.approx(true_sigma, rel=0.1)
    assert calibration["n_obs"] == n_steps


def test_calibrate_ar1_rejects_non_mean_reverting_series():
    # Construct an explosive (non-mean-reverting) process by design:
    # dr_t = +0.01 * r_{t-1} + noise, i.e. positive dependence of the rate
    # change on the lagged level. That gives a positive regression slope,
    # which implies a = -slope/dt < 0.
    rng = np.random.default_rng(0)
    n = 200
    r = np.empty(n)
    r[0] = 0.01
    for i in range(1, n):
        r[i] = r[i - 1] + 0.01 * r[i - 1] + rng.normal(0, 0.0002)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    r_series = pd.Series(r, index=dates)

    with pytest.raises(ValueError):
        calibrate_ar1(r_series, dt=1 / 252)


def test_residual_diagnostics_near_zero_for_white_noise():
    rng = np.random.default_rng(1)
    n = 2000
    resid = rng.normal(0, 0.01, n)
    r_lag = rng.uniform(0.01, 0.05, n)
    calibration = {"residuals": resid, "r_lag": r_lag}

    diagnostics = residual_diagnostics(calibration)

    assert abs(diagnostics["lag1_autocorrelation"]) < 0.1
    assert abs(diagnostics["heteroscedasticity_corr_abs_resid_vs_level"]) < 0.1


def _flat_curve(level_pct: float = 4.0) -> pd.DataFrame:
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "yield_pct": np.full_like(maturities, level_pct),
    })


def _upward_sloping_curve() -> pd.DataFrame:
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    yields = 3.5 + 0.4 * np.log1p(maturities)
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "yield_pct": yields,
    })


@pytest.mark.parametrize("curve_fn", [_flat_curve, _upward_sloping_curve])
def test_calibrate_theta_matches_closed_form_expected_short_rate(curve_fn):
    """Independently verify calibrate_theta's theta(t) by numerically
    integrating the deterministic ODE dm/dt = theta(t) - a*m(t) from
    m(0) = f(0,0), and checking it matches expected_short_rate's closed
    form -- both are derived from the same theta(t), but one is a
    numerical ODE solve and the other an analytic formula, so agreement
    is a genuine correctness check, not a tautology.
    """
    curve = curve_fn()
    a, sigma = 0.4, 0.015
    theta = calibrate_theta(curve, a, sigma)

    f0 = float(instantaneous_forward_rate(fit_decimal_yield_curve(curve), 0.0))

    horizon = 10.0
    sol = solve_ivp(
        lambda t, m: theta(t) - a * m, t_span=(0, horizon), y0=[f0],
        t_eval=np.linspace(0.1, horizon, 20), rtol=1e-10, atol=1e-12,
    )
    ode_mean = sol.y[0]
    closed_form_mean = expected_short_rate(curve, a, sigma, sol.t)

    np.testing.assert_allclose(ode_mean, closed_form_mean, atol=1e-6)


def test_simulate_hull_white_paths_mc_mean_tracks_closed_form():
    curve = _upward_sloping_curve()
    a, sigma = 0.4, 0.015
    theta = calibrate_theta(curve, a, sigma)
    r0 = float(instantaneous_forward_rate(fit_decimal_yield_curve(curve), 0.0))

    times, paths = simulate_hull_white_paths(
        a, sigma, theta, r0, horizon=5.0, n_paths=5000, n_steps=250, seed=42,
    )
    mc_mean = paths.mean(axis=0)
    analytic_mean = expected_short_rate(curve, a, sigma, times)

    # Monte Carlo standard error of the mean at each time step is roughly
    # sigma*sqrt(t)/sqrt(n_paths); use a generous multiple of that as the
    # tolerance so this isn't flaky, while still being a real check.
    se = sigma * np.sqrt(np.maximum(times, 1e-6)) / np.sqrt(paths.shape[0])
    assert np.all(np.abs(mc_mean - analytic_mean) < 6 * se + 1e-4)


def test_simulate_hull_white_paths_shape_and_initial_condition():
    times, paths = simulate_hull_white_paths(
        a=0.3, sigma=0.01, theta=lambda t: 0.03, r0=0.025,
        horizon=2.0, n_paths=100, n_steps=50, seed=0,
    )
    assert paths.shape == (100, 51)
    assert times.shape == (51,)
    assert np.all(paths[:, 0] == 0.025)
