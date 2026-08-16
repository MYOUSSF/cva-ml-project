"""Phase 3 -- Hull-White interest rate model calibration and simulation.

  - Estimate mean-reversion speed `a` and volatility `sigma` from a historical
    short-rate time series (FRED) via an AR(1)-style regression: regress the
    rate change dr_t on the lagged level r_t; the slope gives -a*dt, the
    residual standard deviation (annualized) gives sigma.
  - Calibrate theta(t) analytically to match the current forward curve
    (from Phase 1): theta(t) = df(0,t)/dt + a*f(0,t) + sigma^2/(2a)*(1-e^{-2at}),
    the standard closed form that makes the model reproduce the input curve
    exactly in expectation.
  - Simulate short-rate paths under the calibrated parameters (feeds Phase 4).

Run Phase 3 directly against Phase 1's saved curve:
    python -m src.ratemodel
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.data_pipeline import fetch_fred_series, load_config

DEFAULT_SHORT_RATE_SERIES = "SOFR"
TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------
# AR(1) calibration
# --------------------------------------------------------------------------

def calibrate_ar1(short_rate: pd.Series, dt: float = 1 / TRADING_DAYS_PER_YEAR) -> dict:
    """Estimate Hull-White mean-reversion speed `a` and volatility `sigma`
    from a historical short-rate time series via an AR(1)-style regression:
    regress the rate change dr_t on the lagged level r_t.

        dr_t = intercept + slope * r_{t-1} + residual
        a = -slope / dt
        sigma = std(residual) * sqrt(1/dt)   (annualized)

    `short_rate` must be in decimal (not percent) units, indexed by date,
    sorted or not (this sorts it). Raises ValueError if the fit implies
    non-mean-reverting dynamics (a <= 0).
    """
    r = short_rate.sort_index()
    dr = r.diff().dropna()
    r_lag = r.shift(1).reindex(dr.index)

    slope, intercept = np.polyfit(r_lag.to_numpy(), dr.to_numpy(), deg=1)
    fitted = intercept + slope * r_lag.to_numpy()
    residuals = dr.to_numpy() - fitted

    a = -slope / dt
    if a <= 0:
        raise ValueError(
            f"AR(1) fit implies non-mean-reverting dynamics (a={a:.4f} <= 0) -- "
            "check the input series / date range."
        )
    sigma = float(residuals.std(ddof=2) * np.sqrt(1 / dt))
    long_run_mean = float(-intercept / slope) if slope != 0 else float("nan")

    return {
        "a": float(a),
        "sigma": sigma,
        "long_run_mean": long_run_mean,
        "intercept": float(intercept),
        "slope": float(slope),
        "dt": dt,
        "residuals": residuals,
        "r_lag": r_lag.to_numpy(),
        "dr": dr.to_numpy(),
        "n_obs": len(dr),
    }


def residual_diagnostics(calibration: dict) -> dict:
    """Residual diagnostics on the AR(1) fit: are residuals roughly
    homoscedastic (magnitude uncorrelated with rate level) and uncorrelated
    across time (small lag-1 autocorrelation)? Both should be near zero for
    a well-specified AR(1) fit.
    """
    resid = calibration["residuals"]
    r_lag = calibration["r_lag"]

    return {
        "mean_residual": float(resid.mean()),
        "std_residual": float(resid.std(ddof=1)),
        "lag1_autocorrelation": float(np.corrcoef(resid[:-1], resid[1:])[0, 1]),
        "heteroscedasticity_corr_abs_resid_vs_level": float(np.corrcoef(np.abs(resid), r_lag)[0, 1]),
    }


# --------------------------------------------------------------------------
# theta(t) calibration to the current forward curve
# --------------------------------------------------------------------------

def fit_decimal_yield_curve(curve: pd.DataFrame) -> CubicSpline:
    """Cubic spline of zero yield y(T) (decimal, not percent) vs maturity,
    for computing instantaneous forward rates. Separate from
    data_pipeline.fit_curve, which fits in percent for Phase 1's own
    round-trip check.
    """
    x = curve["maturity_years"].to_numpy()
    y = curve["yield_pct"].to_numpy() / 100.0
    return CubicSpline(x, y)


def instantaneous_forward_rate(yield_spline: CubicSpline, t):
    """f(0,t) = y(t) + t*y'(t), from P(0,t) = exp(-y(t)*t)."""
    return yield_spline(t) + t * yield_spline.derivative(1)(t)


def calibrate_theta(curve: pd.DataFrame, a: float, sigma: float):
    """Analytic Hull-White theta(t) that makes the model's expected short
    rate reproduce the input forward curve exactly:

        theta(t) = df(0,t)/dt + a*f(0,t) + sigma^2/(2a) * (1 - exp(-2at))

    Returns a callable theta(t) (t in years, scalar or array).
    """
    yield_spline = fit_decimal_yield_curve(curve)
    d1 = yield_spline.derivative(1)
    d2 = yield_spline.derivative(2)

    def theta(t):
        y, yp, ypp = yield_spline(t), d1(t), d2(t)
        f = y + t * yp                # f(0,t)
        df_dt = 2 * yp + t * ypp      # d/dt [y(t) + t*y'(t)]
        return df_dt + a * f + (sigma ** 2 / (2 * a)) * (1 - np.exp(-2 * a * t))

    return theta


def expected_short_rate(curve: pd.DataFrame, a: float, sigma: float, t):
    """Closed-form E[r(t)] under the theta(t) from calibrate_theta:

        E[r(t)] = f(0,t) + sigma^2/(2a^2) * (1 - exp(-a*t))^2

    Derived by solving the deterministic mean-reversion ODE
    dE[r]/dt = theta(t) - a*E[r(t)] in closed form. Used both as a
    simulation sanity check (does the Monte Carlo mean path track this?)
    and as a standalone closed-form target to test calibrate_theta's
    correctness against.
    """
    yield_spline = fit_decimal_yield_curve(curve)
    f = instantaneous_forward_rate(yield_spline, t)
    return f + (sigma ** 2 / (2 * a ** 2)) * (1 - np.exp(-a * t)) ** 2


def hull_white_zero_coupon_bond(curve: pd.DataFrame, a: float, sigma: float, t, T, r_t):
    """Analytic Hull-White zero-coupon bond price P(t,T) given short rate
    r_t at time t (Brigo & Mercurio's standard closed form):

        B(t,T) = (1 - exp(-a(T-t))) / a
        P(t,T) = [P(0,T)/P(0,t)] * exp(B(t,T)*f(0,t) - sigma^2/(4a)*(1-exp(-2at))*B(t,T)^2 - B(t,T)*r_t)

    Consumed in Phase 4/5 to price a swap analytically as a function of the
    simulated short rate, giving the "true" smooth conditional expectation
    that the noisy per-path Monte Carlo labels are checked against.
    """
    yield_spline = fit_decimal_yield_curve(curve)
    P0t = np.exp(-yield_spline(t) * t)
    P0T = np.exp(-yield_spline(T) * T)
    f0t = instantaneous_forward_rate(yield_spline, t)

    B = (1 - np.exp(-a * (T - t))) / a
    A = (P0T / P0t) * np.exp(B * f0t - (sigma ** 2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B ** 2)
    return A * np.exp(-B * r_t)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate_hull_white_paths(
    a: float,
    sigma: float,
    theta,
    r0: float,
    horizon: float,
    n_paths: int = 1000,
    n_steps: int = 252,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler-Maruyama simulation of dr = (theta(t) - a*r) dt + sigma dW.

    Returns (times, paths): times has shape (n_steps+1,), paths has shape
    (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0.0, horizon, n_steps + 1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    for i in range(n_steps):
        z = rng.standard_normal(n_paths)
        paths[:, i + 1] = paths[:, i] + (theta(times[i]) - a * paths[:, i]) * dt + sigma * sqrt_dt * z

    return times, paths


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def run_phase3(config_path: str | Path = "config.yaml"):
    """Calibrate Hull-White (a, sigma) from historical short-rate data via
    AR(1), calibrate theta(t) to the Phase 1 forward curve, simulate paths,
    and report the residual-diagnostic and simulation-sanity validation
    called for in the plan's Phase 3.
    """
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    hw_cfg = cfg.get("hull_white", {})
    series_id = hw_cfg.get("short_rate_series", DEFAULT_SHORT_RATE_SERIES)
    n_paths = hw_cfg.get("n_sim_paths", 1000)
    horizon = hw_cfg.get("sim_horizon_years", 5.0)

    print(f"=== Phase 3: Hull-White calibration (short_rate_series={series_id}, "
          f"valuation_date={valuation_date}) ===\n")

    series = fetch_fred_series(series_id) / 100.0
    series = series[series.index <= pd.Timestamp(valuation_date)]

    calibration = calibrate_ar1(series)
    print(f"AR(1) fit on {calibration['n_obs']} daily observations "
          f"({series.index.min().date()} to {series.index.max().date()})")
    print(f"a (mean-reversion speed) = {calibration['a']:.4f}")
    print(f"sigma (annualized vol)   = {calibration['sigma']:.4f}")
    print(f"long-run mean            = {calibration['long_run_mean']:.4%}")

    diagnostics = residual_diagnostics(calibration)
    print("\n--- Residual diagnostics (want: both near 0) ---")
    print(f"lag-1 autocorrelation                    = {diagnostics['lag1_autocorrelation']:.4f}")
    print(f"|residual| vs. level correlation (heteroscedasticity) = "
          f"{diagnostics['heteroscedasticity_corr_abs_resid_vs_level']:.4f}")

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    a, sigma = calibration["a"], calibration["sigma"]
    theta = calibrate_theta(curve, a, sigma)

    r0 = float(series.iloc[-1])
    times, paths = simulate_hull_white_paths(a, sigma, theta, r0, horizon=horizon, n_paths=n_paths, seed=42)

    mc_mean = paths.mean(axis=0)
    analytic_mean = expected_short_rate(curve, a, sigma, times)
    forward = instantaneous_forward_rate(fit_decimal_yield_curve(curve), times)
    mae_mc_vs_analytic = float(np.mean(np.abs(mc_mean - analytic_mean)))

    print(f"\n--- Simulation sanity check ---")
    print(f"{n_paths} paths, {horizon}y horizon, r0={r0:.4%}")
    print(f"Mean abs error, MC mean path vs. closed-form E[r(t)]: {mae_mc_vs_analytic:.6f} "
          f"(should be small -- pure Monte Carlo noise, since theta(t) is calibrated "
          f"to reproduce the curve exactly)")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i in range(min(60, n_paths)):
        ax.plot(times, paths[i], color="lightgray", linewidth=0.5, zorder=1)
    ax.plot(times, mc_mean, label="Monte Carlo mean path", color="tab:blue", linewidth=2)
    ax.plot(times, analytic_mean, label="Closed-form E[r(t)]", color="tab:orange",
            linestyle="--", linewidth=2)
    ax.plot(times, forward, label="Input forward curve f(0,t)", color="tab:green",
            linestyle=":", linewidth=2)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Short rate")
    ax.set_title(f"Hull-White simulated paths vs. calibrated curve (a={a:.3f}, sigma={sigma:.3f})")
    ax.legend()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "hull_white_paths.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    params = {
        "valuation_date": valuation_date,
        "short_rate_series": series_id,
        "a": a,
        "sigma": sigma,
        "r0": r0,
        "long_run_mean": calibration["long_run_mean"],
        "n_obs": calibration["n_obs"],
        **diagnostics,
        "mc_vs_analytic_mae": mae_mc_vs_analytic,
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"hull_white_params_{valuation_date}.json"
    out_path.write_text(json.dumps(params, indent=2))

    print(f"\nSaved: {out_path}, {fig_path}")
    return calibration, diagnostics, theta, (times, paths)


if __name__ == "__main__":
    run_phase3()
