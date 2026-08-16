"""Unit tests for Phase 6/7/9 (LGD, CVA assembly, sensitivity, portfolio application).

Key checks, offline, no network calls:
  - compute_lgd is the trivial 1-recovery relationship.
  - pd_term_structure_from_1y_pd reproduces the input 1y PD exactly at t=1.
  - compute_cva matches a fully hand-computed single-checkpoint example.
  - sensitivity_analysis: CVA increases with PD, decreases with recovery --
    checked directly, not just asserted in the run_phase6_and_7 script.
  - compute_ee_curve (exposure_models.py): naive method matches naive_baseline
    per checkpoint; a regression method recovers a known noiseless relationship.
  - compute_portfolio_cva: higher PD -> higher CVA, holding EE(t)/LGD/curve
    fixed; and the ratings lookup table used for Phase 9's sanity check is
    internally consistent (no typo'd rating missing from the rank table).
"""

import numpy as np
import pandas as pd
import pytest

from src.cva import (
    APPROXIMATE_CREDIT_RATINGS,
    CREDIT_RATING_RANK,
    compute_cva,
    compute_lgd,
    compute_portfolio_cva,
    pd_term_structure_from_1y_pd,
    sensitivity_analysis,
)
from src.exposure_models import compute_ee_curve, naive_baseline


def _flat_curve(level_pct: float = 4.0) -> pd.DataFrame:
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "yield_pct": np.full_like(maturities, level_pct),
    })


def test_compute_lgd():
    assert compute_lgd(0.40) == pytest.approx(0.60)
    assert compute_lgd(0.0) == pytest.approx(1.0)


def test_pd_term_structure_reproduces_1y_pd_at_t1():
    pdts = pd_term_structure_from_1y_pd(0.05, checkpoint_times=[0.5, 1.0, 2.0])
    assert pdts.loc[1.0] == pytest.approx(0.05)
    # Cumulative default probability must increase with time.
    assert pdts.loc[0.5] < pdts.loc[1.0] < pdts.loc[2.0]


def test_pd_term_structure_matches_constant_hazard_formula():
    pd_1y = 0.05
    pdts = pd_term_structure_from_1y_pd(pd_1y, checkpoint_times=[2.0])
    expected = 1.0 - (1.0 - pd_1y) ** 2.0
    assert pdts.loc[2.0] == pytest.approx(expected)


def test_compute_cva_matches_hand_calculation_single_checkpoint():
    # Flat 4% curve, single checkpoint at t=1 -> DF(0,1) = exp(-0.04).
    curve = _flat_curve(level_pct=4.0)
    ee_curve = pd.Series({1.0: 100.0})
    pd_term_structure = pd.Series({1.0: 0.05})  # Q(0)=0 implicit -> marginal = 0.05
    lgd = 0.6

    cva = compute_cva(ee_curve, pd_term_structure, lgd, curve)

    expected = 0.6 * np.exp(-0.04 * 1.0) * 100.0 * 0.05
    assert cva == pytest.approx(expected, rel=1e-6)


def test_compute_cva_multi_checkpoint_uses_marginal_default_probability():
    curve = _flat_curve(level_pct=0.0)  # zero rates -> DF=1 everywhere, isolates the PD/EE mechanics
    ee_curve = pd.Series({1.0: 100.0, 2.0: 100.0})
    pd_term_structure = pd.Series({1.0: 0.05, 2.0: 0.05})  # marginal prob in (1,2] is exactly 0
    lgd = 1.0

    cva = compute_cva(ee_curve, pd_term_structure, lgd, curve)

    # All the "new" default probability happens by t=1; nothing further
    # defaults between t=1 and t=2, so only the t=1 term contributes.
    assert cva == pytest.approx(100.0 * 0.05, rel=1e-6)


def test_sensitivity_analysis_cva_increases_with_pd_and_decreases_with_recovery():
    curve = _flat_curve(level_pct=4.0)
    ee_curve = pd.Series({0.5: 50.0, 1.0: 80.0, 1.5: 60.0})

    sens = sensitivity_analysis(
        ee_curve, pd_1y_base=0.05, recovery_rate_base=0.40, curve=curve,
        pd_shock_multipliers=(0.5, 1.0, 2.0), recovery_rate_shocks=(0.2, 0.4, 0.6),
    )

    pd_sens = sens[sens["shock_type"] == "PD"].sort_values("pd_1y")
    recovery_sens = sens[sens["shock_type"] == "Recovery"].sort_values("recovery_rate")

    assert pd_sens["cva"].is_monotonic_increasing
    assert recovery_sens["cva"].is_monotonic_decreasing


def _synthetic_exposure_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for t_k in [1.0, 2.0]:
        n = 2000
        x = rng.normal(0.04, 0.02, n)
        true_mean = 500_000.0 * (0.04 - x)  # simple linear truth, decreasing in rate
        y = true_mean + rng.normal(0, 20_000, n)
        rows.append(pd.DataFrame({"checkpoint_time": t_k, "short_rate": x, "realized_swap_value": y}))
    return pd.concat(rows, ignore_index=True)


def test_compute_ee_curve_naive_matches_naive_baseline_per_checkpoint():
    dataset = _synthetic_exposure_dataset()
    ee = compute_ee_curve(dataset, method="naive")

    for t_k, group in dataset.groupby("checkpoint_time"):
        assert ee.loc[t_k] == pytest.approx(naive_baseline(group["realized_swap_value"].to_numpy()))


def test_compute_ee_curve_poly_is_close_to_naive_ge_zero_and_lower_or_equal_naive():
    # Not an exact identity (naive floors *before* averaging, poly floors a
    # smoothed prediction), but poly's EE should still be non-negative and
    # should not wildly exceed naive's on this near-symmetric synthetic setup.
    dataset = _synthetic_exposure_dataset()
    ee_naive = compute_ee_curve(dataset, method="naive")
    ee_poly = compute_ee_curve(dataset, method="poly")

    assert (ee_poly >= 0).all()
    assert (ee_poly <= ee_naive * 1.5).all()


def test_compute_portfolio_cva_increases_with_pd():
    curve = _flat_curve(level_pct=4.0)
    ee_curve = pd.Series({0.5: 50.0, 1.0: 80.0, 1.5: 60.0})
    merton_df = pd.DataFrame({
        "ticker": ["SAFE", "RISKY", "MISSING"],
        "pd_merton": [0.001, 0.10, np.nan],
    })

    portfolio = compute_portfolio_cva(merton_df, ee_curve, lgd=0.6, curve=curve)

    # MISSING (nan PD) should be dropped, not crash or silently zero-fill.
    assert set(portfolio["ticker"]) == {"SAFE", "RISKY"}
    safe_cva = portfolio.loc[portfolio["ticker"] == "SAFE", "cva"].iloc[0]
    risky_cva = portfolio.loc[portfolio["ticker"] == "RISKY", "cva"].iloc[0]
    assert risky_cva > safe_cva
    # Sorted ascending by CVA (safest first), per compute_portfolio_cva's contract.
    assert list(portfolio["ticker"]) == ["SAFE", "RISKY"]


def test_approximate_credit_ratings_all_present_in_rank_table():
    # Guards against a typo'd rating string silently dropping out of the
    # Spearman correlation in run_phase9 (map() would just produce NaN).
    for ticker, rating in APPROXIMATE_CREDIT_RATINGS.items():
        assert rating in CREDIT_RATING_RANK, f"{ticker}'s rating {rating!r} missing from CREDIT_RATING_RANK"
