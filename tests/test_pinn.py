"""Unit tests for Phase 8 (PINN bond pricer).

Key checks, offline, no network calls:
  - The hard-constrained terminal condition P(T,r)=1 holds exactly for an
    UNTRAINED (random-weight) network -- it's a property of the
    architecture, not something training needs to learn.
  - compute_derivatives' autograd output matches finite-difference
    derivatives of the same (untrained) network -- validates the autograd
    wiring itself, independent of whether training converges.
  - A short training run on a small, cheap problem: loss decreases, and
    the trained network is closer to the closed form than an untrained one.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.pinn import BondPricingPINN, compute_derivatives, train_pinn, validate_against_closed_form
from src.ratemodel import calibrate_theta


def _upward_sloping_curve() -> pd.DataFrame:
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    yields = 3.5 + 0.4 * np.log1p(maturities)
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "yield_pct": yields,
    })


def test_terminal_condition_is_exact_for_untrained_network():
    torch.manual_seed(0)
    T = 5.0
    model = BondPricingPINN(T)

    r = torch.tensor(np.linspace(-0.05, 0.15, 20), dtype=torch.float32).reshape(-1, 1)
    t = torch.full_like(r, T)

    with torch.no_grad():
        P = model(t, r)

    np.testing.assert_allclose(P.numpy().flatten(), 1.0, atol=1e-6)


def test_compute_derivatives_matches_finite_differences():
    torch.manual_seed(1)
    model = BondPricingPINN(T=5.0)

    t0, r0 = 2.0, 0.04
    t = torch.tensor([[t0]], dtype=torch.float64)
    r = torch.tensor([[r0]], dtype=torch.float64)
    model = model.double()

    _, dP_dt, dP_dr, d2P_dr2 = compute_derivatives(model, t, r)

    def P_of(t_val, r_val):
        with torch.no_grad():
            return model(
                torch.tensor([[t_val]], dtype=torch.float64),
                torch.tensor([[r_val]], dtype=torch.float64),
            ).item()

    h = 1e-4
    fd_dP_dt = (P_of(t0 + h, r0) - P_of(t0 - h, r0)) / (2 * h)
    fd_dP_dr = (P_of(t0, r0 + h) - P_of(t0, r0 - h)) / (2 * h)
    fd_d2P_dr2 = (P_of(t0, r0 + h) - 2 * P_of(t0, r0) + P_of(t0, r0 - h)) / (h ** 2)

    assert dP_dt.item() == pytest.approx(fd_dP_dt, abs=1e-3)
    assert dP_dr.item() == pytest.approx(fd_dP_dr, abs=1e-3)
    assert d2P_dr2.item() == pytest.approx(fd_d2P_dr2, abs=1e-2)


def test_training_reduces_loss_and_improves_on_untrained_baseline():
    curve = _upward_sloping_curve()
    a, sigma = 0.4, 0.015
    theta = calibrate_theta(curve, a, sigma)
    T = 2.0
    r_range = (-0.02, 0.10)

    torch.manual_seed(7)
    untrained = BondPricingPINN(T)
    untrained_validation = validate_against_closed_form(
        untrained, curve, a, sigma, T, r_range=r_range, n_test_t=10, n_test_r=10,
    )

    model, loss_history = train_pinn(
        T, a, sigma, theta, r_range=r_range,
        n_collocation=500, n_epochs=800, seed=7,
    )

    # Loss should be lower at the end than at the start, and not just by
    # noise -- compare a late-training window average to the initial loss.
    assert np.mean(loss_history[-50:]) < loss_history[0]

    trained_validation = validate_against_closed_form(
        model, curve, a, sigma, T, r_range=r_range, n_test_t=10, n_test_r=10,
    )
    assert trained_validation["mean_abs_error"] < untrained_validation["mean_abs_error"]
