"""Phase 8 -- Physics-informed neural network (PINN) fast pricer.

Trains a small PyTorch MLP to solve the Hull-White zero-coupon bond-pricing
PDE directly -- no labeled price data, just the PDE residual (via autograd)
and a hard-constrained terminal condition:

    dP/dt + (theta(t) - a*r) dP/dr + 0.5*sigma^2 d2P/dr2 - r*P = 0,   P(T,r) = 1

Framed explicitly as scientific ML: the loss is a physics constraint, not a
supervised label, and the terminal condition is baked into the network's
output parametrization (P = 1 + (T-t)*NN(t,r)) rather than penalized as a
soft loss term -- so P(T,r)=1 holds exactly, for any network weights, not
just approximately.

Honest scope note: in this one-dimensional setting the PINN does not beat
`ratemodel.hull_white_zero_coupon_bond` on speed -- that closed form exists
precisely because Hull-White is analytically tractable in 1D. The value of
having built this is that the *same* training recipe (no labeled data, just
a PDE residual + boundary condition) is what you'd reach for in genuinely
higher-dimensional short-rate models (multi-factor, or with stochastic
volatility) where no closed form exists and grid-based PDE solvers become
infeasible -- this is a working, validated instance of that recipe, not a
production speed-up.

Run Phase 8 directly against Phase 1/3's saved outputs:
    python -m src.pinn
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# The network here is tiny (hidden_size=64, batches of a few hundred to a
# few thousand points) -- torch's default multi-threaded BLAS is pure
# overhead at this scale, and when this module is imported alongside
# numpy/sklearn/xgboost (each spinning up their own thread pools in the
# same process, e.g. when running the full test suite) that overhead
# compounds into severe contention. Single-threaded is both faster here
# and avoids fighting the other libraries for cores.
torch.set_num_threads(1)

from src.data_pipeline import fetch_fred_series, load_config
from src.ratemodel import DEFAULT_SHORT_RATE_SERIES, calibrate_ar1, calibrate_theta, hull_white_zero_coupon_bond


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class BondPricingPINN(nn.Module):
    """MLP approximating the Hull-White zero-coupon bond price P(t,r) for a
    FIXED maturity T.

    The terminal condition P(T,r)=1 is hard-constrained into the
    architecture: P(t,r) = 1 + (T-t)*NN(t,r), so it holds exactly at t=T
    regardless of network weights -- the plan's "hard-constrained terminal
    condition," not a soft penalty term competing with the PDE loss.
    """

    def __init__(self, T: float, hidden_size: int = 64):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        correction = self.net(torch.cat([t, r], dim=1))
        return 1.0 + (self.T - t) * correction


# --------------------------------------------------------------------------
# PDE residual and training
# --------------------------------------------------------------------------

def compute_derivatives(model: BondPricingPINN, t: torch.Tensor, r: torch.Tensor):
    """P(t,r) and its first/second-order partials via autograd, split out
    from pde_residual so the derivative computation itself is directly
    testable against finite differences (rather than only checking the
    composite residual, where a sign error in one term could cancel
    against another and hide a bug).
    """
    t = t.clone().requires_grad_(True)
    r = r.clone().requires_grad_(True)
    P = model(t, r)

    dP_dt = torch.autograd.grad(P, t, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    dP_dr = torch.autograd.grad(P, r, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    d2P_dr2 = torch.autograd.grad(dP_dr, r, grad_outputs=torch.ones_like(dP_dr), create_graph=True)[0]

    return P, dP_dt, dP_dr, d2P_dr2


def pde_residual(model: BondPricingPINN, t: torch.Tensor, r: torch.Tensor,
                  a: float, sigma: float, theta_t: torch.Tensor) -> torch.Tensor:
    """Hull-White bond-pricing PDE residual at (t,r). `theta_t` = theta(t)
    evaluated as a plain tensor -- it doesn't depend on network parameters,
    only on t's numeric value, so it's computed outside the graph (in the
    caller) rather than re-implementing calibrate_theta's spline math in
    torch.
    """
    P, dP_dt, dP_dr, d2P_dr2 = compute_derivatives(model, t, r)
    return dP_dt + (theta_t - a * r) * dP_dr + 0.5 * sigma ** 2 * d2P_dr2 - r * P


def train_pinn(
    T: float,
    a: float,
    sigma: float,
    theta,
    r_range: tuple[float, float] = (-0.06, 0.15),
    n_collocation: int = 2000,
    n_epochs: int = 3000,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[BondPricingPINN, list[float]]:
    """Train purely from the PDE residual -- no labeled prices anywhere in
    this function. Collocation points are resampled fresh each epoch
    (standard PINN practice, avoids overfitting to one fixed point cloud).

    `r_range` should cover the range of rates the Hull-White simulation
    actually visits (Phase 4's simulated short_rate spans roughly
    -0.04 to 0.12 on the current calibration) -- per the plan's own
    pitfalls list, a PINN extrapolates poorly outside its collocation
    domain, so the default here is set with a margin around that.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = BondPricingPINN(T)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for _ in range(n_epochs):
        t_np = rng.uniform(0.0, T, size=n_collocation)
        r_np = rng.uniform(r_range[0], r_range[1], size=n_collocation)
        theta_np = theta(t_np)

        t_t = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1)
        r_t = torch.tensor(r_np, dtype=torch.float32).reshape(-1, 1)
        theta_t = torch.tensor(theta_np, dtype=torch.float32).reshape(-1, 1)

        optimizer.zero_grad()
        residual = pde_residual(model, t_t, r_t, a, sigma, theta_t)
        loss = torch.mean(residual ** 2)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return model, loss_history


# --------------------------------------------------------------------------
# Validation against the closed form
# --------------------------------------------------------------------------

def validate_against_closed_form(
    model: BondPricingPINN,
    curve: pd.DataFrame,
    a: float,
    sigma: float,
    T: float,
    r_range: tuple[float, float] = (-0.06, 0.15),
    n_test_t: int = 25,
    n_test_r: int = 25,
) -> dict:
    """Compare the trained PINN against ratemodel.hull_white_zero_coupon_bond
    on a dense (t,r) grid. Excludes t=T itself (error is exactly 0 there by
    the hard constraint -- not an interesting point to report).
    """
    t_grid = np.linspace(0.0, T * 0.998, n_test_t)
    r_grid = np.linspace(r_range[0], r_range[1], n_test_r)
    TT, RR = np.meshgrid(t_grid, r_grid, indexing="ij")

    t_flat = torch.tensor(TT.flatten(), dtype=torch.float32).reshape(-1, 1)
    r_flat = torch.tensor(RR.flatten(), dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        P_pinn = model(t_flat, r_flat).numpy().flatten()

    P_closed = hull_white_zero_coupon_bond(curve, a, sigma, TT.flatten(), T, RR.flatten())
    abs_error = np.abs(P_pinn - P_closed)

    return {
        "t_grid": TT, "r_grid": RR,
        "P_pinn": P_pinn.reshape(TT.shape), "P_closed": P_closed.reshape(TT.shape),
        "abs_error": abs_error.reshape(TT.shape),
        "max_abs_error": float(abs_error.max()),
        "mean_abs_error": float(abs_error.mean()),
    }


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def run_phase8(config_path: str | Path = "config.yaml"):
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    hw_cfg = cfg.get("hull_white", {})
    pinn_cfg = cfg.get("pinn", {})

    print("=== Phase 8: PINN bond pricer (Hull-White PDE) ===\n")

    series_id = hw_cfg.get("short_rate_series", DEFAULT_SHORT_RATE_SERIES)
    series = fetch_fred_series(series_id) / 100.0
    series = series[series.index <= pd.Timestamp(valuation_date)]
    calibration = calibrate_ar1(series)
    a, sigma = calibration["a"], calibration["sigma"]

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    theta = calibrate_theta(curve, a, sigma)

    T = pinn_cfg.get("bond_maturity_years", 5.0)
    r_range = tuple(pinn_cfg.get("r_range", [-0.06, 0.15]))
    n_epochs = pinn_cfg.get("n_epochs", 3000)

    print(f"Pricing a T={T}y zero-coupon bond under Hull-White (a={a:.4f}, sigma={sigma:.4f})")
    print(f"Collocation domain: t in [0,{T}], r in {r_range} "
          f"(covers Phase 4's simulated short-rate range with margin)")

    t0 = time.perf_counter()
    model, loss_history = train_pinn(T, a, sigma, theta, r_range=r_range, n_epochs=n_epochs)
    train_time = time.perf_counter() - t0
    print(f"\nTrained {n_epochs} epochs in {train_time:.1f}s "
          f"(loss {loss_history[0]:.3e} -> {loss_history[-1]:.3e})")

    validation = validate_against_closed_form(model, curve, a, sigma, T, r_range=r_range)
    print("\n--- Validation against closed form ---")
    print(f"Max abs price error:  {validation['max_abs_error']:.6f}")
    print(f"Mean abs price error: {validation['mean_abs_error']:.6f}")

    time_to_maturity = T - validation["t_grid"].flatten()
    corr = float(np.corrcoef(time_to_maturity, validation["abs_error"].flatten())[0, 1])
    print(f"Correlation(time-to-maturity, abs error) = {corr:.3f} "
          f"(positive -> error grows away from the hard-constrained t=T boundary, as expected)")

    n_speed = 10000
    rng = np.random.default_rng(0)
    t_speed = rng.uniform(0, T * 0.99, n_speed)
    r_speed = rng.uniform(*r_range, n_speed)

    t0 = time.perf_counter()
    hull_white_zero_coupon_bond(curve, a, sigma, t_speed, T, r_speed)
    closed_form_time = time.perf_counter() - t0

    t_tensor = torch.tensor(t_speed, dtype=torch.float32).reshape(-1, 1)
    r_tensor = torch.tensor(r_speed, dtype=torch.float32).reshape(-1, 1)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(t_tensor, r_tensor)
    pinn_inference_time = time.perf_counter() - t0

    print(f"\n--- Speed comparison ({n_speed:,} price evaluations) ---")
    print(f"Closed-form eval: {closed_form_time * 1000:.2f}ms")
    print(f"PINN inference:   {pinn_inference_time * 1000:.2f}ms")
    print(f"PINN training:    {train_time:.1f}s (one-time cost the closed form never pays)")
    print(
        "\nHonest takeaway: in 1D, the closed form wins on speed outright -- it's exact, needs no "
        "training, and evaluates faster per call. The PINN's value isn't demonstrated here by speed; "
        "it's that this exact recipe (PDE residual + hard-constrained boundary, no labeled data) is "
        "what scales to genuinely higher-dimensional short-rate models with no closed form, where "
        "grid-based PDE solvers become infeasible. See the README for the full discussion."
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].semilogy(loss_history)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("PDE residual MSE loss")
    axes[0].set_title("Training loss")

    mid_r_idx = len(validation["r_grid"][0]) // 2
    t_slice = validation["t_grid"][:, mid_r_idx]
    axes[1].plot(t_slice, validation["P_closed"][:, mid_r_idx], color="black", linewidth=2, label="Closed form")
    axes[1].plot(t_slice, validation["P_pinn"][:, mid_r_idx], color="tab:red", linestyle="--", label="PINN")
    axes[1].set_xlabel("t (years)")
    axes[1].set_ylabel("P(t,r)")
    axes[1].set_title(f"Price vs. t at r={validation['r_grid'][0, mid_r_idx]:.3f}")
    axes[1].legend()

    im = axes[2].imshow(
        validation["abs_error"], origin="lower", aspect="auto", cmap="viridis",
        extent=[r_range[0], r_range[1], 0, T],
    )
    axes[2].set_xlabel("r")
    axes[2].set_ylabel("t (years)")
    axes[2].set_title("Absolute error |PINN - closed form|")
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "pinn_validation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {fig_path}")
    return model, loss_history, validation


if __name__ == "__main__":
    run_phase8()
