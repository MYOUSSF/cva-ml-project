"""Unit tests for the bootstrapped-CVA-confidence-interval extension.

Key checks, offline, no network calls:
  - bootstrap_annualized_vol / bootstrap_pd_1y / bootstrap_recovery_rate:
    each resampling primitive is reproducible given a seed and produces
    varying output across seeds (not degenerate).
  - bootstrap_ee_curve: resampling whole paths recovers something close to
    the un-resampled naive EE(t) on average, and stays non-negative.
  - bootstrap_cva_distribution: percentile ordering holds, and -- the
    documented pitfall this extension exists to avoid -- turning all three
    uncertainty sources off collapses the distribution to exactly zero
    variance, while turning any single source on injects nonzero variance.
"""

import numpy as np
import pandas as pd
import pytest

from src.cva_uncertainty import (
    bootstrap_annualized_vol,
    bootstrap_cva_distribution,
    bootstrap_ee_curve,
    bootstrap_pd_1y,
    bootstrap_recovery_rate,
    summarize_distribution,
)
from src.exposure_models import compute_ee_curve


def _flat_curve(level_pct: float = 4.0) -> pd.DataFrame:
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "yield_pct": np.full_like(maturities, level_pct),
    })


def _synthetic_exposure_dataset_with_paths(n_paths: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for t_k in [1.0, 2.0, 3.0]:
        x = rng.normal(0.04, 0.02, n_paths)
        true_mean = 500_000.0 * (0.04 - x)
        y = true_mean + rng.normal(0, 20_000, n_paths)
        rows.append(pd.DataFrame({
            "path_id": np.arange(n_paths), "checkpoint_time": t_k,
            "short_rate": x, "realized_swap_value": y,
        }))
    return pd.concat(rows, ignore_index=True)


def test_bootstrap_annualized_vol_reproducible_and_varies_across_seeds():
    log_returns = np.random.default_rng(1).normal(0, 0.02, 300)
    v1 = bootstrap_annualized_vol(log_returns, np.random.default_rng(42))
    v2 = bootstrap_annualized_vol(log_returns, np.random.default_rng(42))
    v3 = bootstrap_annualized_vol(log_returns, np.random.default_rng(99))
    assert v1 == pytest.approx(v2)
    assert v1 != pytest.approx(v3)
    assert v1 > 0


def test_bootstrap_pd_1y_returns_valid_probability():
    log_returns = np.random.default_rng(1).normal(0, 0.03, 300)
    pd_1y = bootstrap_pd_1y(E=1e9, D=5e8, r=0.04, log_returns=log_returns, rng=np.random.default_rng(3))
    assert 0.0 <= pd_1y <= 1.0


def test_bootstrap_recovery_rate_bounded_and_reproducible():
    r1 = bootstrap_recovery_rate(np.random.default_rng(5), low=0.2, mode=0.4, high=0.6)
    r2 = bootstrap_recovery_rate(np.random.default_rng(5), low=0.2, mode=0.4, high=0.6)
    assert r1 == pytest.approx(r2)
    assert 0.2 <= r1 <= 0.6


def test_bootstrap_ee_curve_nonnegative_and_close_to_naive_on_average():
    dataset = _synthetic_exposure_dataset_with_paths()
    naive_ee = compute_ee_curve(dataset, method="naive")

    rng = np.random.default_rng(11)
    draws = [bootstrap_ee_curve(dataset, rng) for _ in range(200)]
    mean_ee = pd.concat(draws, axis=1).mean(axis=1)

    assert (mean_ee >= 0).all()
    # Bootstrap mean should track the un-resampled naive estimate closely
    # given enough resamples (both are floor-then-average of the same pool).
    assert mean_ee.subtract(naive_ee).abs().max() < 0.05 * naive_ee.abs().max()


def test_bootstrap_cva_distribution_percentiles_ordered():
    dataset = _synthetic_exposure_dataset_with_paths()
    curve = _flat_curve()
    log_returns = np.random.default_rng(1).normal(0, 0.03, 300)

    cvas = bootstrap_cva_distribution(
        dataset, curve, E=1e9, D=5e8, r=0.04, log_returns=log_returns,
        recovery_rate_base=0.40, n_iterations=200, seed=1,
    )
    summary = summarize_distribution(cvas)
    assert summary["p5"] <= summary["median"] <= summary["p95"]
    assert len(cvas) == 200


def test_bootstrap_cva_distribution_degenerate_when_nothing_varies():
    dataset = _synthetic_exposure_dataset_with_paths()
    curve = _flat_curve()
    log_returns = np.random.default_rng(1).normal(0, 0.03, 300)

    cvas = bootstrap_cva_distribution(
        dataset, curve, E=1e9, D=5e8, r=0.04, log_returns=log_returns,
        recovery_rate_base=0.40, n_iterations=50, seed=1,
        vary_paths=False, vary_pd=False, vary_lgd=False,
    )
    assert np.std(cvas) == pytest.approx(0.0)


@pytest.mark.parametrize("vary_paths,vary_pd,vary_lgd", [
    (True, False, False), (False, True, False), (False, False, True),
])
def test_bootstrap_cva_distribution_each_source_injects_variance(vary_paths, vary_pd, vary_lgd):
    dataset = _synthetic_exposure_dataset_with_paths()
    curve = _flat_curve()
    log_returns = np.random.default_rng(1).normal(0, 0.03, 300)

    cvas = bootstrap_cva_distribution(
        dataset, curve, E=1e9, D=5e8, r=0.04, log_returns=log_returns,
        recovery_rate_base=0.40, n_iterations=100, seed=1,
        vary_paths=vary_paths, vary_pd=vary_pd, vary_lgd=vary_lgd,
    )
    assert np.std(cvas) > 0.0
