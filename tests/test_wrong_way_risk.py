"""Unit tests for the wrong-way-risk (Gaussian copula) extension.

Key checks, offline, no network calls:
  - simulate_correlated_paths: the empirical correlation between the rate
    and asset-value shock increments matches the requested rho.
  - discrete_monitoring_default_checkpoint: hand-constructed breach pattern
    recovers the exact expected first-breach index per path.
  - wrong_way_cva: the plan's explicit validation ask -- for THIS project's
    receive-fixed swap, a strongly positive rho (wrong-way: low rates/high
    exposure paths coincide with low-asset-value/default-likely paths)
    should give a materially higher CVA than a strongly negative rho
    (right-way), holding everything else fixed.
  - compute_historical_rate_equity_correlation: recovers a known planted
    correlation from synthetic aligned series.
"""

import numpy as np
import pandas as pd
import pytest

from src.ratemodel import calibrate_theta
from src.wrong_way_risk import (
    compute_historical_rate_equity_correlation,
    discrete_monitoring_default_checkpoint,
    simulate_correlated_paths,
    wrong_way_cva,
)


def _flat_curve(level_pct: float = 4.0) -> pd.DataFrame:
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "yield_pct": np.full_like(maturities, level_pct),
    })


def test_simulate_correlated_paths_shocks_match_requested_rho():
    curve = _flat_curve()
    theta = calibrate_theta(curve, a=0.3, sigma=0.01)

    for rho in (0.6, -0.6, 0.0):
        times, r_paths, V_paths = simulate_correlated_paths(
            a=0.3, sigma_r=0.01, theta=theta, r0=0.04, V0=1000.0, sigma_V=0.3,
            rho=rho, horizon=1.0, n_paths=4000, n_steps=100, seed=0,
        )
        dr = np.diff(r_paths, axis=1).ravel()
        d_log_v = np.diff(np.log(V_paths), axis=1).ravel()
        empirical_rho = np.corrcoef(dr, d_log_v)[0, 1]
        assert empirical_rho == pytest.approx(rho, abs=0.05)


def test_simulate_correlated_paths_rejects_rho_outside_open_interval():
    curve = _flat_curve()
    theta = calibrate_theta(curve, a=0.3, sigma=0.01)
    with pytest.raises(ValueError):
        simulate_correlated_paths(a=0.3, sigma_r=0.01, theta=theta, r0=0.04, V0=1000.0,
                                   sigma_V=0.3, rho=1.0, horizon=1.0, n_paths=10, n_steps=5)


def test_discrete_monitoring_default_checkpoint_hand_constructed():
    D = 100.0
    # 4 paths x 3 checkpoints: never breaches / breaches at k=0 / breaches at k=1 / breaches at k=2
    V = np.array([
        [150.0, 140.0, 130.0],  # never below D
        [90.0, 200.0, 200.0],   # breaches at checkpoint 0
        [120.0, 80.0, 200.0],   # breaches at checkpoint 1
        [110.0, 105.0, 95.0],   # breaches at checkpoint 2
    ])
    result = discrete_monitoring_default_checkpoint(V, D)
    assert list(result) == [-1, 0, 1, 2]


def test_discrete_monitoring_default_checkpoint_locks_in_first_breach():
    D = 100.0
    # Dips below D at checkpoint 0, recovers above D at checkpoint 1 --
    # should still be recorded as defaulting at checkpoint 0.
    V = np.array([[90.0, 150.0]])
    result = discrete_monitoring_default_checkpoint(V, D)
    assert list(result) == [0]


def test_wrong_way_cva_positive_rho_exceeds_negative_rho_for_receive_fixed_swap():
    curve = _flat_curve(level_pct=4.0)
    theta = calibrate_theta(curve, a=0.3, sigma=0.01)
    kwargs = dict(
        curve=curve, a=0.3, sigma_r=0.01, theta=theta, r0=0.04,
        V0=100.0, sigma_V=0.5, D=90.0, notional=1_000_000.0, fixed_rate=0.045,
        tenor_years=1.5, payment_frequency="semiannual", recovery_rate=0.40,
        n_paths=4000, seed=7,
    )
    wrong_way = wrong_way_cva(rho=0.8, **kwargs)
    right_way = wrong_way_cva(rho=-0.8, **kwargs)

    assert wrong_way["cva"] > right_way["cva"]


def test_wrong_way_cva_zero_paths_survive_gives_zero_cva():
    curve = _flat_curve(level_pct=4.0)
    theta = calibrate_theta(curve, a=0.3, sigma=0.01)
    # D far below V0 with low vol -- essentially no defaults, CVA should be ~0.
    result = wrong_way_cva(
        curve=curve, a=0.3, sigma_r=0.01, theta=theta, r0=0.04,
        V0=1_000_000.0, sigma_V=0.05, D=10.0, rho=0.5,
        notional=1_000_000.0, fixed_rate=0.045, tenor_years=1.5,
        payment_frequency="semiannual", recovery_rate=0.40, n_paths=2000, seed=1,
    )
    assert result["cva"] == pytest.approx(0.0, abs=1.0)
    assert result["cumulative_default_prob"] == pytest.approx(0.0, abs=1e-6)


def test_compute_historical_rate_equity_correlation_recovers_planted_correlation():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    mean = [0.0, 0.0]
    true_rho = 0.5
    cov = [[1.0, true_rho], [true_rho, 1.0]]
    samples = rng.multivariate_normal(mean, cov, size=500)
    equity_returns = pd.Series(samples[:, 0], index=dates)
    rate_changes = pd.Series(samples[:, 1], index=dates)

    recovered = compute_historical_rate_equity_correlation(equity_returns, rate_changes)
    assert recovered == pytest.approx(true_rho, abs=0.1)


def test_compute_historical_rate_equity_correlation_raises_on_no_overlap():
    equity_returns = pd.Series([0.01], index=pd.to_datetime(["2024-01-01"]))
    rate_changes = pd.Series([0.001], index=pd.to_datetime(["2025-01-01"]))
    with pytest.raises(ValueError):
        compute_historical_rate_equity_correlation(equity_returns, rate_changes)
