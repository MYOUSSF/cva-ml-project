"""Unit tests for Phase 4 (Monte Carlo synthetic exposure dataset).

Key checks, offline, no network calls:
  - hull_white_zero_coupon_bond exactly reproduces the input curve at t=0,
    r_t=f(0,0) -- a fundamental property of the Hull-White calibration
    (the model must reproduce the initial term structure by construction),
    checked directly at the pricing-formula level this time (Phase 3
    checked it at the expected-short-rate level).
  - The par swap rate (computed from the curve alone, no simulation) gives
    exactly zero swap value at inception -- the textbook definition of an
    at-the-money swap, and a strong correctness check on analytic_swap_value.
  - simulate_swap_exposure_paths's realized (noisy) labels are unbiased
    relative to the analytic conditional expectation, checked with a real
    Monte Carlo run and a statistical tolerance (not exact equality).
"""

import numpy as np
import pandas as pd
import pytest

from src.ratemodel import calibrate_theta, fit_decimal_yield_curve, hull_white_zero_coupon_bond, instantaneous_forward_rate
from src.exposure_models import analytic_swap_value, build_payment_schedule, simulate_swap_exposure_paths


def _upward_sloping_curve() -> pd.DataFrame:
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    yields = 3.5 + 0.4 * np.log1p(maturities)
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "yield_pct": yields,
    })


@pytest.mark.parametrize("T", [0.5, 1.0, 3.0, 7.0, 15.0])
def test_hull_white_bond_price_reproduces_input_curve_at_t0(T):
    curve = _upward_sloping_curve()
    a, sigma = 0.4, 0.015
    yield_spline = fit_decimal_yield_curve(curve)
    f0 = float(instantaneous_forward_rate(yield_spline, 0.0))

    model_price = hull_white_zero_coupon_bond(curve, a, sigma, t=0.0, T=T, r_t=f0)
    curve_price = float(np.exp(-yield_spline(T) * T))

    assert model_price == pytest.approx(curve_price, rel=1e-6)


def test_par_swap_rate_gives_zero_value_at_inception():
    curve = _upward_sloping_curve()
    a, sigma = 0.4, 0.015
    payment_dates, accrual = build_payment_schedule(tenor_years=5.0, payment_frequency="semiannual")
    yield_spline = fit_decimal_yield_curve(curve)
    f0 = float(instantaneous_forward_rate(yield_spline, 0.0))

    P = lambda T: hull_white_zero_coupon_bond(curve, a, sigma, 0.0, T, f0)
    par_rate = (P(payment_dates[0]) - P(payment_dates[-1])) / sum(accrual * P(T) for T in payment_dates)

    value_at_par = analytic_swap_value(
        curve, a, sigma, t=0.0, remaining_payment_dates=payment_dates, accrual=accrual,
        notional=1_000_000.0, fixed_rate=par_rate, r_t=f0,
    )
    assert value_at_par == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("tenor,freq,expected_n,expected_last", [
    (5.0, "semiannual", 10, 5.0),
    (2.0, "quarterly", 8, 2.0),
    (3.0, "annual", 3, 3.0),
])
def test_build_payment_schedule(tenor, freq, expected_n, expected_last):
    dates, accrual = build_payment_schedule(tenor, freq)
    assert len(dates) == expected_n
    assert dates[-1] == pytest.approx(expected_last)
    assert accrual == pytest.approx(1.0 / {"annual": 1, "semiannual": 2, "quarterly": 4}[freq])


def test_simulate_swap_exposure_paths_shape_and_columns():
    curve = _upward_sloping_curve()
    a, sigma = 0.4, 0.015
    theta = calibrate_theta(curve, a, sigma)
    yield_spline = fit_decimal_yield_curve(curve)
    r0 = float(instantaneous_forward_rate(yield_spline, 0.0))

    dataset = simulate_swap_exposure_paths(
        curve, a, sigma, theta, r0, notional=1_000_000.0, fixed_rate=0.04,
        tenor_years=2.0, payment_frequency="semiannual", n_paths=200, seed=1,
    )

    expected_cols = {"path_id", "checkpoint_time", "time_to_maturity", "short_rate",
                      "realized_swap_value", "analytic_conditional_value"}
    assert expected_cols <= set(dataset.columns)
    # semiannual over 2y => 4 payment dates, 3 checkpoints (excludes the last payment)
    assert dataset["checkpoint_time"].nunique() == 3
    assert len(dataset) == 200 * 3


def test_simulate_swap_exposure_paths_realized_labels_are_unbiased():
    curve = _upward_sloping_curve()
    a, sigma = 0.4, 0.015
    theta = calibrate_theta(curve, a, sigma)
    yield_spline = fit_decimal_yield_curve(curve)
    r0 = float(instantaneous_forward_rate(yield_spline, 0.0))

    dataset = simulate_swap_exposure_paths(
        curve, a, sigma, theta, r0, notional=1_000_000.0, fixed_rate=0.04,
        tenor_years=2.0, payment_frequency="semiannual", n_paths=8000, seed=7,
    )

    for t_k, group in dataset.groupby("checkpoint_time"):
        diff = group["realized_swap_value"] - group["analytic_conditional_value"]
        # Mean of the noisy-label minus the analytic conditional expectation
        # should be statistically indistinguishable from zero -- a real,
        # not-hand-tuned tolerance based on the sample's own standard error.
        t_stat = diff.mean() / diff.sem()
        assert abs(t_stat) < 4, f"checkpoint {t_k}: |t-stat|={abs(t_stat):.2f} suggests a real bias, not MC noise"
