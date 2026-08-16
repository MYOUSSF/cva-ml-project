"""Unit tests for Phase 5 (exposure regression model comparison).

Key checks, offline, no network calls:
  - nested_simulation_ground_truth's Monte Carlo mean matches the exact
    analytic value at an arbitrary mid-life checkpoint (not just t=0) --
    validates the "shifted-time" nested-simulation machinery specifically.
  - The central Phase 5 story, as an integration test rather than just
    described in prose: naive_baseline is upward-biased relative to the
    unfloored ground truth, while a regression model fit on the same raw
    labels the ground truth targets is not.
"""

import numpy as np
import pandas as pd
import pytest

from src.exposure_models import (
    analytic_swap_value,
    build_payment_schedule,
    fit_polynomial_regression,
    naive_baseline,
    nested_simulation_ground_truth,
    predict_regression_model,
    simulate_swap_exposure_paths,
)
from src.ratemodel import calibrate_theta, fit_decimal_yield_curve, instantaneous_forward_rate


def _upward_sloping_curve() -> pd.DataFrame:
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    yields = 3.5 + 0.4 * np.log1p(maturities)
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "yield_pct": yields,
    })


def test_naive_baseline_floors_then_averages():
    y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    # floor: [0, 0, 0, 1, 2] -> mean = 0.6
    assert naive_baseline(y) == pytest.approx(0.6)


def test_naive_baseline_ge_raw_mean_always():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 5, 10000)
    assert naive_baseline(y) >= y.mean()  # max(V,0) >= V pointwise, always


def test_polynomial_regression_recovers_noiseless_polynomial():
    x = np.linspace(-1, 1, 50)
    y = 3.0 - 2.0 * x + 0.5 * x ** 2  # exact quadratic, no noise
    model = fit_polynomial_regression(x, y, degree=3)
    x_test = np.array([-0.5, 0.0, 0.7])
    preds = predict_regression_model(model, x_test)
    np.testing.assert_allclose(preds, 3.0 - 2.0 * x_test + 0.5 * x_test ** 2, atol=1e-8)


def test_nested_simulation_ground_truth_matches_analytic_at_mid_life_checkpoint():
    curve = _upward_sloping_curve()
    a, sigma = 0.4, 0.015
    theta = calibrate_theta(curve, a, sigma)
    tenor_years = 5.0
    payment_dates, accrual = build_payment_schedule(tenor_years, "semiannual")

    checkpoint_time = 2.0
    short_rate_value = 0.045
    remaining_dates = payment_dates[payment_dates >= checkpoint_time]

    gt = nested_simulation_ground_truth(
        curve, a, sigma, theta, checkpoint_time, short_rate_value, remaining_dates, accrual,
        notional=1_000_000.0, fixed_rate=0.04, tenor_years=tenor_years,
        n_inner_paths=20000, seed=99,
    )

    # Statistical tolerance based on the Monte Carlo estimate's own standard
    # error, not a hand-picked absolute number.
    t_stat = (gt["ground_truth_mean"] - gt["analytic_value"]) / gt["ground_truth_stderr"]
    assert abs(t_stat) < 4


def test_naive_baseline_is_upward_biased_vs_ground_truth_while_regression_is_not():
    """End-to-end integration test of Phase 5's central claim."""
    curve = _upward_sloping_curve()
    a, sigma = 0.4, 0.015
    theta = calibrate_theta(curve, a, sigma)
    yield_spline = fit_decimal_yield_curve(curve)
    r0 = float(instantaneous_forward_rate(yield_spline, 0.0))
    tenor_years = 3.0
    payment_dates, accrual = build_payment_schedule(tenor_years, "semiannual")
    notional, fixed_rate = 1_000_000.0, 0.04

    dataset = simulate_swap_exposure_paths(
        curve, a, sigma, theta, r0, notional=notional, fixed_rate=fixed_rate,
        tenor_years=tenor_years, payment_frequency="semiannual", n_paths=6000, seed=11,
    )

    checkpoint_time = 1.5
    group = dataset[dataset["checkpoint_time"] == checkpoint_time]
    x, y = group["short_rate"].to_numpy(), group["realized_swap_value"].to_numpy()
    remaining_dates = payment_dates[payment_dates >= checkpoint_time]

    test_rate = float(np.median(x))
    gt = nested_simulation_ground_truth(
        curve, a, sigma, theta, checkpoint_time, test_rate, remaining_dates, accrual,
        notional, fixed_rate, tenor_years, n_inner_paths=20000, seed=123,
    )

    naive_pred = naive_baseline(y)
    poly_model = fit_polynomial_regression(x, y, degree=3)
    poly_pred = float(predict_regression_model(poly_model, test_rate)[0])

    naive_error = naive_pred - gt["ground_truth_mean"]
    poly_error = poly_pred - gt["ground_truth_mean"]

    # Naive's floored constant should sit meaningfully above the unfloored
    # ground truth -- well beyond the ground truth's own Monte Carlo noise.
    assert naive_error > 5 * gt["ground_truth_stderr"]
    # The polynomial regression, fit on the same unfloored labels the
    # ground truth targets, should land much closer to it than naive does.
    assert abs(poly_error) < abs(naive_error)
