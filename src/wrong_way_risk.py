"""Extension -- Wrong-way risk via a Gaussian copula.

Links the counterparty's simulated Merton asset-value process to the same
Hull-White short-rate path that drives the swap's exposure, via a single
Cholesky-correlated pair of standard normal shocks (Z_r, Z_V) at every
simulation step:

    dr        = (theta(t) - a*r) dt        + sigma_r*sqrt(dt)*Z_r
    d(ln V)   = (r_t - 0.5*sigma_V^2) dt    + sigma_V*sqrt(dt)*Z_V,   corr(Z_r, Z_V) = rho

V0 and sigma_V are the counterparty's own Phase 2 solved Merton values
(same V, sigma_V already saved in merton_pd_<date>.csv) -- this reuses
Phase 2's structural model rather than inventing a separate one. Default is
assessed by discrete monitoring at the swap's own payment-date checkpoints
(V(t_k) < D), the same checkpoint grid `exposure_models.py` already uses
for exposure -- a natural fit given that module's own "checkpoints aligned
to payment dates" design choice, and it sidesteps building a full
continuous first-passage default model.

Correlation parameter: rather than an arbitrary picked number, rho is the
historical realized correlation between the counterparty's own daily
equity log returns and daily SOFR changes, over the same trailing window
Phase 1 uses for equity_vol -- directly computable from data already in
this project (`data/raw/equity_history.csv`, the FRED SOFR series), not a
sector-level guess. Equity return is used as a practical proxy for the
latent asset-value shock (equity itself isn't directly observable under
Merton's own model, but as a call option on V it co-moves strongly with V
except close to default).

Sign convention for THIS project's swap (receive-fixed / pay-floating,
`exposure_models.analytic_swap_value`): swap value rises when rates fall,
so exposure is largest exactly when rates are low. rho > 0 (rates and
asset value moving together) means the paths where rates fall furthest are
also the paths where the counterparty's asset value falls furthest --
exposure-high paths coincide with default-likely paths -- the classic
wrong-way setup for a receive-fixed payer. rho < 0 is the right-way case.
Whichever sign the empirical rho comes out to for a given counterparty,
CVA_correlated should move relative to CVA_independent in that direction --
checked explicitly in run_wrong_way_risk, not just asserted.

Run directly against Phase 1/2/3's saved outputs:
    python -m src.wrong_way_risk
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cva import compute_lgd
from src.data_pipeline import fetch_fred_series, load_config
from src.exposure_models import (
    DEFAULT_SHORT_RATE_SERIES,
    analytic_swap_value,
    build_payment_schedule,
)
from src.pd_model import get_risk_free_rate
from src.ratemodel import calibrate_ar1, calibrate_theta, fit_decimal_yield_curve

# Empirical correlation can be noisy on a single ~1y window of daily data;
# clipped away from +-1 so the Cholesky-implied Z_V doesn't degenerate into
# an exact copy (or exact mirror) of Z_r.
RHO_CLIP = 0.95


# --------------------------------------------------------------------------
# Correlated simulation
# --------------------------------------------------------------------------

def simulate_correlated_paths(
    a: float,
    sigma_r: float,
    theta,
    r0: float,
    V0: float,
    sigma_V: float,
    rho: float,
    horizon: float,
    n_paths: int = 5000,
    n_steps: int = 252,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euler-Maruyama simulation of the Hull-White short rate jointly with a
    Merton-consistent GBM asset-value process, correlated via a single pair
    of standard normal shocks per step. The asset-value process uses the
    path's own simulated short rate as its instantaneous drift, consistent
    with Merton's risk-neutral GBM assumption.

    Returns (times, r_paths, V_paths); each *_paths has shape
    (n_paths, n_steps+1).
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")

    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0.0, horizon, n_steps + 1)

    r_paths = np.empty((n_paths, n_steps + 1))
    log_V_paths = np.empty((n_paths, n_steps + 1))
    r_paths[:, 0] = r0
    log_V_paths[:, 0] = np.log(V0)

    for i in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z_v = rho * z1 + np.sqrt(1.0 - rho ** 2) * z2  # z1 plays the role of z_r

        r_paths[:, i + 1] = r_paths[:, i] + (theta(times[i]) - a * r_paths[:, i]) * dt + sigma_r * sqrt_dt * z1
        log_V_paths[:, i + 1] = (
            log_V_paths[:, i] + (r_paths[:, i] - 0.5 * sigma_V ** 2) * dt + sigma_V * sqrt_dt * z_v
        )

    return times, r_paths, np.exp(log_V_paths)


def discrete_monitoring_default_checkpoint(V_at_checkpoints: np.ndarray, D: float) -> np.ndarray:
    """First checkpoint index at which each path's asset value falls below
    debt threshold D, monitored discretely at the swap's own payment-date
    checkpoints (not continuous first-passage). -1 for paths that never
    breach D within the monitored horizon.

    V_at_checkpoints: shape (n_paths, n_checkpoints), checkpoints in
    ascending time order.
    """
    breached = V_at_checkpoints < D
    n_paths, n_checkpoints = breached.shape
    first_breach = np.full(n_paths, -1, dtype=int)
    for k in range(n_checkpoints):
        newly = breached[:, k] & (first_breach == -1)
        first_breach[newly] = k
    return first_breach


# --------------------------------------------------------------------------
# WWR CVA
# --------------------------------------------------------------------------

def wrong_way_cva(
    curve: pd.DataFrame,
    a: float,
    sigma_r: float,
    theta,
    r0: float,
    V0: float,
    sigma_V: float,
    D: float,
    rho: float,
    notional: float,
    fixed_rate: float,
    tenor_years: float,
    payment_frequency: str,
    recovery_rate: float,
    n_paths: int = 5000,
    steps_per_year: int = 252,
    seed: int | None = 42,
) -> dict:
    """CVA = LGD * sum_k DF(0,t_k) * mean_over_paths[EE(path,t_k) * 1{defaults in (t_{k-1},t_k]}]

    EE(path,t_k) is the analytic (closed-form) swap value given that path's
    own simulated short rate at t_k, floored at 0 -- smooth per-path
    exposure, so the only extra noise relative to Phase 7's formula is the
    correlation effect itself, not an additional realized-value Monte Carlo
    label. Default timing comes from the SAME correlated simulation
    (discrete monitoring, not Phase 7's constant-hazard PD term structure)
    so that comparing rho != 0 against rho = 0 isolates exactly the
    correlation effect without also switching default-probability
    methodology between the two runs.
    """
    payment_dates, accrual = build_payment_schedule(tenor_years, payment_frequency)
    checkpoint_times = payment_dates[:-1]  # no exposure after the final payment (exposure_models' convention)
    n_steps = int(round(tenor_years * steps_per_year))

    times, r_paths, V_paths = simulate_correlated_paths(
        a, sigma_r, theta, r0, V0, sigma_V, rho, horizon=tenor_years,
        n_paths=n_paths, n_steps=n_steps, seed=seed,
    )

    def grid_index(t: float) -> int:
        idx = int(round(t * steps_per_year))
        assert abs(times[idx] - t) < 1e-9, f"checkpoint {t} not aligned to the simulation grid"
        return idx

    checkpoint_idx = [grid_index(t) for t in checkpoint_times]
    V_at_checkpoints = V_paths[:, checkpoint_idx]
    default_checkpoint = discrete_monitoring_default_checkpoint(V_at_checkpoints, D)

    yield_spline = fit_decimal_yield_curve(curve)
    lgd = compute_lgd(recovery_rate)

    total = 0.0
    marginal_default_probs = {}
    for k, t_k in enumerate(checkpoint_times):
        remaining_dates = payment_dates[payment_dates >= t_k]
        r_k = r_paths[:, checkpoint_idx[k]]
        exposure = np.maximum(
            analytic_swap_value(curve, a, sigma_r, t_k, remaining_dates, accrual, notional, fixed_rate, r_k),
            0.0,
        )
        defaults_here = default_checkpoint == k
        marginal_default_probs[t_k] = float(defaults_here.mean())
        discount = float(np.exp(-yield_spline(t_k) * t_k))
        total += discount * float(np.mean(exposure * defaults_here))

    return {
        "cva": lgd * total,
        "marginal_default_probs": pd.Series(marginal_default_probs),
        "cumulative_default_prob": float((default_checkpoint >= 0).mean()),
    }


# --------------------------------------------------------------------------
# Correlation parameter estimation
# --------------------------------------------------------------------------

def compute_historical_rate_equity_correlation(equity_log_returns: pd.Series, rate_changes: pd.Series) -> float:
    """Historical correlation between daily equity log returns and daily
    changes in a short-rate series, aligned by date. Used as rho -- a
    directly-computable, data-grounded correlation parameter rather than an
    arbitrary or sector-level guess.
    """
    aligned = pd.DataFrame({"equity_return": equity_log_returns, "rate_change": rate_changes}).dropna()
    if len(aligned) < 2:
        raise ValueError("Not enough overlapping observations to compute a correlation")
    return float(aligned["equity_return"].corr(aligned["rate_change"]))


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def run_wrong_way_risk(config_path: str | Path = "config.yaml"):
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    recovery_rate = cfg["lgd"]["recovery_rate"]
    cva_cfg = cfg.get("cva", {})
    counterparty_ticker = cva_cfg.get("counterparty_ticker", "AMC")
    hw_cfg = cfg.get("hull_white", {})
    swap_cfg = cfg["swap"]
    merton_cfg = cfg.get("merton", {})
    T = merton_cfg.get("horizon_years", 1.0)

    print(f"=== Extension: wrong-way risk via Gaussian copula (counterparty={counterparty_ticker}) ===\n")

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    merton = pd.read_csv(processed_dir / f"merton_pd_{valuation_date}.csv")
    row = merton.loc[merton["ticker"] == counterparty_ticker].iloc[0]
    V0, sigma_V, D = float(row["V"]), float(row["sigma_V"]), float(row["total_debt_face_value"])
    r0_valuation = get_risk_free_rate(curve, T)
    print(f"Counterparty Merton state (Phase 2): V={V0:,.0f}, sigma_V={sigma_V:.4f}, D={D:,.0f}")

    series_id = hw_cfg.get("short_rate_series", DEFAULT_SHORT_RATE_SERIES)
    rate_series = fetch_fred_series(series_id) / 100.0
    rate_series = rate_series[rate_series.index <= pd.Timestamp(valuation_date)]
    calibration = calibrate_ar1(rate_series)
    a, sigma_r = calibration["a"], calibration["sigma"]
    theta = calibrate_theta(curve, a, sigma_r)
    r0 = float(rate_series.iloc[-1])
    print(f"Hull-White params (re-calibrated as in Phase 3): a={a:.4f}, sigma={sigma_r:.4f}, r0={r0:.4%}")

    equity_history = pd.read_csv(Path("data/raw/equity_history.csv"), parse_dates=["date"])
    hist = equity_history.loc[equity_history["ticker"] == counterparty_ticker].sort_values("date")
    hist = hist[hist["date"] <= pd.Timestamp(valuation_date)].tail(253)
    equity_log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    equity_log_returns.index = hist["date"].iloc[1:].to_numpy()

    rate_changes = rate_series.diff().dropna()
    rho_raw = compute_historical_rate_equity_correlation(equity_log_returns, rate_changes)
    rho = float(np.clip(rho_raw, -RHO_CLIP, RHO_CLIP))
    print(f"\nHistorical correlation, {counterparty_ticker} daily equity log returns vs. daily {series_id} "
          f"changes (same trailing window as Phase 1's equity_vol): rho = {rho_raw:.4f} (clipped to {rho:.4f})")

    kwargs = dict(
        curve=curve, a=a, sigma_r=sigma_r, theta=theta, r0=r0, V0=V0, sigma_V=sigma_V, D=D,
        notional=swap_cfg["notional"], fixed_rate=swap_cfg["fixed_rate"], tenor_years=swap_cfg["tenor_years"],
        payment_frequency=swap_cfg["payment_frequency"], recovery_rate=recovery_rate,
        n_paths=swap_cfg.get("n_exposure_paths", 5000), seed=42,
    )
    independent = wrong_way_cva(rho=0.0, **kwargs)
    correlated = wrong_way_cva(rho=rho, **kwargs)

    uplift = (correlated["cva"] / independent["cva"] - 1.0) if independent["cva"] else float("nan")
    print(f"\nCVA (independent, rho=0):    ${independent['cva']:,.2f}  "
          f"(cumulative default prob {independent['cumulative_default_prob']:.4%})")
    print(f"CVA (correlated, rho={rho:.3f}): ${correlated['cva']:,.2f}  "
          f"(cumulative default prob {correlated['cumulative_default_prob']:.4%})")
    print(f"Wrong-way-risk uplift: {uplift:+.1%}")

    expected_sign = "increase" if rho > 0 else "decrease" if rho < 0 else "be unchanged"
    actual_direction = "increased" if correlated["cva"] > independent["cva"] else "decreased"
    print(
        f"\nDirection check (receive-fixed swap: exposure is largest when rates are low): "
        f"with rho {'> 0' if rho > 0 else '< 0' if rho < 0 else '= 0'}, CVA should {expected_sign} "
        f"relative to independence -- it {actual_direction}."
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"wrong_way_risk_{valuation_date}.csv"
    pd.DataFrame([
        {"scenario": "independent", "rho": 0.0, "cva": independent["cva"],
         "cumulative_default_prob": independent["cumulative_default_prob"]},
        {"scenario": "correlated", "rho": rho, "cva": correlated["cva"],
         "cumulative_default_prob": correlated["cumulative_default_prob"]},
    ]).to_csv(out_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(["Independent (rho=0)", f"Correlated (rho={rho:.2f})"],
                   [independent["cva"], correlated["cva"]], color=["tab:blue", "tab:red"])
    ax.set_ylabel("CVA ($)")
    ax.set_title(f"Wrong-way risk: {counterparty_ticker}, {uplift:+.1%} uplift from correlation")
    for bar, val in zip(bars, [independent["cva"], correlated["cva"]]):
        ax.annotate(f"${val:,.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "wrong_way_risk.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}, {fig_path}")
    return {"independent": independent, "correlated": correlated, "rho": rho, "uplift": uplift}


if __name__ == "__main__":
    run_wrong_way_risk()
