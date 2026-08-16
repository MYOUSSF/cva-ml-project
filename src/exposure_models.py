"""Phase 4/5 -- Monte Carlo synthetic data generation + exposure (EE(t)) regression comparison.

Phase 4 (implemented here): simulate Hull-White short-rate paths for a
hypothetical interest rate swap, then compute, at each checkpoint aligned
with a swap payment/reset date, the REALIZED future swap value along that
specific path -- a noisy one-sample estimate of the true conditional
expectation, in the spirit of Longstaff-Schwartz least-squares Monte Carlo.
This (state, noisy label) dataset is the training data for Phase 5.

Realized future value at a checkpoint uses the path's own realized
money-market discounting: since the floating leg is modeled SOFR-OIS style
(compounded-in-arrears -- rate for period [T_i, T_i+1] is only fixed once
the period ends), the discounted floating leg telescopes exactly to
notional*(D(t,T_first) - D(t,T_last)) along ANY single path, using that
path's own realized discount factors D(t,T) = B(t)/B(T). No cash-flow-by-
cash-flow simulation error accumulates beyond the discretization of the
money-market integral itself.

Design choice: checkpoints are restricted to payment/reset dates. The plan
notes a checkpoint's state may need "any already-fixed-but-unpaid cash flow
info" -- that only arises for a mid-period checkpoint, which the
compounded-in-arrears convention here doesn't have (nothing is fixed until
a period ends). Aligning checkpoints with reset dates sidesteps that
complication entirely; documented here as a real simplification, not a
silent assumption.

Phase 5 (implemented here): fit polynomial OLS / random forest / gradient
boosting / MLP regressions of the RAW (unfloored, signed) realized_swap_value
on short_rate, one set of models per checkpoint, and benchmark them plus a
naive baseline against a nested-simulation ground truth.

The naive-vs-ground-truth comparison is deliberately asymmetric, following
the plan's own wording closely:
  - naive baseline = "average the raw noisy labels directly (floor, then
    average)" -> a single constant per checkpoint, mean(max(V_i, 0)).
  - ground truth = a fresh nested simulation, "average WITHOUT flooring" ->
    E[V | state], the raw signed conditional mean.
Since max(V,0) >= V pointwise for any V, naive's floored constant is
*always* >= the unfloored ground truth whenever the checkpoint has any
chance of a negative value -- a near-algebraic upward bias, not just an
empirical tendency. The regression models are fit on the same *unfloored*
labels the ground truth targets, so (given enough data and a
reasonably-specified model) they have no such structural bias -- that
contrast is the central result of this phase.

Run Phase 4 or 5 directly against Phase 1/3's saved outputs:
    python -m src.exposure_models
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.integrate import cumulative_trapezoid
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import fetch_fred_series, load_config
from src.ratemodel import (
    DEFAULT_SHORT_RATE_SERIES,
    calibrate_ar1,
    calibrate_theta,
    hull_white_zero_coupon_bond,
    simulate_hull_white_paths,
)

PAYMENT_FREQUENCIES_PER_YEAR = {"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12}


# --------------------------------------------------------------------------
# Swap schedule and analytic reference value
# --------------------------------------------------------------------------

def build_payment_schedule(tenor_years: float, payment_frequency: str) -> tuple[np.ndarray, float]:
    """Payment dates and per-period accrual (year fraction) for a plain
    fixed-frequency swap with no stub periods, starting at the first period
    end (e.g. semiannual over 5y -> 0.5, 1.0, ..., 5.0).
    """
    freq_per_year = PAYMENT_FREQUENCIES_PER_YEAR[payment_frequency]
    accrual = 1.0 / freq_per_year
    n_payments = round(tenor_years * freq_per_year)
    payment_dates = np.array([accrual * (i + 1) for i in range(n_payments)])
    return payment_dates, accrual


def analytic_swap_value(curve, a, sigma, t, remaining_payment_dates, accrual, notional, fixed_rate, r_t):
    """Closed-form value at time t given short rate r_t, using the
    Hull-White zero-coupon bond formula -- the smooth "true" conditional
    expectation E[realized future value | r(t)] that the noisy per-path
    labels from simulate_swap_exposure_paths are a one-sample draw of.

    Receive-fixed / pay-floating convention: value = fixed leg - float leg.
    """
    if len(remaining_payment_dates) == 0:
        return np.zeros_like(r_t) if np.ndim(r_t) else 0.0

    T_first, T_last = remaining_payment_dates[0], remaining_payment_dates[-1]
    P_first = hull_white_zero_coupon_bond(curve, a, sigma, t, T_first, r_t)
    P_last = hull_white_zero_coupon_bond(curve, a, sigma, t, T_last, r_t)
    float_leg = notional * (P_first - P_last)

    fixed_leg = sum(
        notional * fixed_rate * accrual * hull_white_zero_coupon_bond(curve, a, sigma, t, T, r_t)
        for T in remaining_payment_dates
    )
    return fixed_leg - float_leg


# --------------------------------------------------------------------------
# Monte Carlo synthetic dataset (Phase 4)
# --------------------------------------------------------------------------

def simulate_swap_exposure_paths(
    curve: pd.DataFrame,
    a: float,
    sigma: float,
    theta,
    r0: float,
    notional: float,
    fixed_rate: float,
    tenor_years: float,
    payment_frequency: str,
    n_paths: int = 5000,
    steps_per_year: int = 252,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate Hull-White paths and compute, at each payment-date
    checkpoint along each path, the REALIZED future swap value (receive
    fixed / pay floating) and the analytic conditional expectation given
    that path's own state -- the training data for Phase 5's regression
    comparison, plus a built-in reference to check it against.

    `steps_per_year` must make every payment date land exactly on a
    simulation grid point (252 works for annual/semiannual/quarterly/
    monthly schedules; asserted at runtime).

    Returns a long DataFrame: path_id, checkpoint_time, time_to_maturity,
    short_rate (state), realized_swap_value (noisy label),
    analytic_conditional_value (smooth reference, Phase-4-level sanity
    check only -- Phase 5 builds its own nested-simulation ground truth).
    """
    payment_dates, accrual = build_payment_schedule(tenor_years, payment_frequency)
    n_steps = int(round(tenor_years * steps_per_year))

    times, paths = simulate_hull_white_paths(
        a, sigma, theta, r0, horizon=tenor_years, n_paths=n_paths, n_steps=n_steps, seed=seed,
    )

    # Money-market numeraire: log B(t) = int_0^t r(s) ds along each path,
    # via cumulative trapezoidal integration on the simulation grid.
    log_B = cumulative_trapezoid(paths, times, axis=1, initial=0.0)  # (n_paths, n_steps+1)

    def grid_index(t: float) -> int:
        idx = int(round(t * steps_per_year))
        assert abs(times[idx] - t) < 1e-9, f"payment date {t} not aligned to the simulation grid"
        return idx

    payment_idx = [grid_index(T) for T in payment_dates]
    checkpoint_positions = range(len(payment_dates) - 1)  # no exposure after the final payment

    records = []
    for pos in checkpoint_positions:
        t_k = payment_dates[pos]
        idx_k = payment_idx[pos]
        remaining_dates = payment_dates[pos:]
        remaining_idx = payment_idx[pos:]

        log_B_k = log_B[:, idx_k]
        D_first = np.exp(log_B_k - log_B[:, remaining_idx[0]])
        D_last = np.exp(log_B_k - log_B[:, remaining_idx[-1]])
        float_leg = notional * (D_first - D_last)

        fixed_leg = np.zeros(n_paths)
        for ridx in remaining_idx:
            D_i = np.exp(log_B_k - log_B[:, ridx])
            fixed_leg += notional * fixed_rate * accrual * D_i

        realized_value = fixed_leg - float_leg
        short_rate_k = paths[:, idx_k]
        analytic_value = analytic_swap_value(
            curve, a, sigma, t_k, remaining_dates, accrual, notional, fixed_rate, short_rate_k,
        )

        records.append(pd.DataFrame({
            "path_id": np.arange(n_paths),
            "checkpoint_time": t_k,
            "time_to_maturity": tenor_years - t_k,
            "short_rate": short_rate_k,
            "realized_swap_value": realized_value,
            "analytic_conditional_value": analytic_value,
        }))

    return pd.concat(records, ignore_index=True)


# --------------------------------------------------------------------------
# Phase 5: naive baseline, regression models, nested-simulation ground truth
# --------------------------------------------------------------------------

def naive_baseline(realized_values: np.ndarray) -> float:
    """Floor each raw noisy label, then average -- a single constant
    prediction per checkpoint, independent of state. See module docstring
    for why this is structurally biased upward relative to an unfloored
    ground truth.
    """
    return float(np.mean(np.maximum(realized_values, 0.0)))


def fit_polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.poly1d:
    """Small hand-chosen polynomial basis -- the classical Longstaff-Schwartz
    approach. Fit on the RAW (unfloored) realized_swap_value.
    """
    return np.poly1d(np.polyfit(x, y, degree))


def fit_random_forest(x: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Deliberately unconstrained (no max_depth) -- flexible enough that
    whether it overfits the noisy training labels is a real, checkable
    question here, not something to assume away with an a priori depth cap.
    """
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(x.reshape(-1, 1), y)
    return model


def fit_gradient_boosting(x: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(x.reshape(-1, 1), y)
    return model


def fit_mlp(x: np.ndarray, y: np.ndarray) -> TransformedTargetRegressor:
    """Scales both the input (StandardScaler in the inner pipeline) AND the
    target (TransformedTargetRegressor) -- swap values here are ~1e5-1e6 in
    scale, and MLPRegressor's default optimizer settings assume roughly
    O(1) targets. Without target scaling this silently converges to a flat
    prediction near zero rather than raising an error, which it did in an
    earlier version of this code (looked like severe underfitting, not a
    real "MLP struggles here" finding -- worth remembering when reading
    scikit-learn neural net results that look suspiciously flat).
    """
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=5000,
                              early_stopping=True, random_state=42)),
    ])
    model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    model.fit(x.reshape(-1, 1), y)
    return model


def predict_regression_model(model, x) -> np.ndarray:
    """Uniform predict() across np.poly1d (polynomial) and sklearn-style
    (.predict, expects 2D X) models.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if isinstance(model, np.poly1d):
        return model(x)
    return model.predict(x.reshape(-1, 1))


MODEL_FIT_FUNCTIONS = {
    "poly": fit_polynomial_regression,
    "random_forest": fit_random_forest,
    "gradient_boosting": fit_gradient_boosting,
    "mlp": fit_mlp,
}


def compute_ee_curve(dataset: pd.DataFrame, method: str = "naive") -> pd.Series:
    """Expected (positive) exposure profile EE(t), one value per checkpoint
    in `dataset`, for use in Phase 7's CVA formula.

    method="naive": naive_baseline per checkpoint (floor each raw noisy
    label, then average).

    method=<name in MODEL_FIT_FUNCTIONS>: fit that regression model per
    checkpoint on (short_rate -> realized_swap_value), then floor its
    PREDICTION at each path's own state and average. This is a lower-
    variance EE(t) estimate than the naive approach -- flooring a smooth
    fitted curve, rather than each noisy raw label, removes the extra
    continuation-noise Phase 5 quantified -- which is the concrete way
    Phase 5's model comparison feeds into Phase 7's CVA number.
    """
    checkpoints = np.sort(dataset["checkpoint_time"].unique())
    values = {}
    for t_k in checkpoints:
        group = dataset[dataset["checkpoint_time"] == t_k]
        x, y = group["short_rate"].to_numpy(), group["realized_swap_value"].to_numpy()
        if method == "naive":
            values[t_k] = naive_baseline(y)
        else:
            model = MODEL_FIT_FUNCTIONS[method](x, y)
            fitted = predict_regression_model(model, x)
            values[t_k] = float(np.mean(np.maximum(fitted, 0.0)))
    return pd.Series(values).sort_index()


def nested_simulation_ground_truth(
    curve: pd.DataFrame,
    a: float,
    sigma: float,
    theta,
    checkpoint_time: float,
    short_rate_value: float,
    remaining_payment_dates: np.ndarray,
    accrual: float,
    notional: float,
    fixed_rate: float,
    tenor_years: float,
    n_inner_paths: int = 20000,
    steps_per_year: int = 252,
    seed: int | None = None,
) -> dict:
    """Expensive nested-simulation estimate of E[V | r(checkpoint_time) =
    short_rate_value]: launch fresh forward Hull-White paths STARTING AT
    short_rate_value at checkpoint_time, simulate to swap maturity, and
    average the realized future value WITHOUT flooring.

    Reuses the same realized-value telescoping-discount-factor trick as
    simulate_swap_exposure_paths, just shifted to start mid-life at a fixed
    (not simulated-from-t=0) state -- this is what "bin states, launch
    fresh forward simulations from each bin" means operationally.

    Also returns the closed-form analytic value at the same point: since
    this product happens to have a Hull-White closed form, the expensive
    nested simulation can be cross-checked against it -- validating the
    nested-sim *methodology* itself, which is what you'd have to rely on
    alone for a product without a closed form.
    """
    horizon = tenor_years - checkpoint_time
    n_steps = int(round(horizon * steps_per_year))

    def theta_shifted(s):
        return theta(checkpoint_time + s)

    inner_times, inner_paths = simulate_hull_white_paths(
        a, sigma, theta_shifted, short_rate_value, horizon=horizon,
        n_paths=n_inner_paths, n_steps=n_steps, seed=seed,
    )
    log_B = cumulative_trapezoid(inner_paths, inner_times, axis=1, initial=0.0)

    local_dates = remaining_payment_dates - checkpoint_time

    def local_grid_index(s: float) -> int:
        idx = int(round(s * steps_per_year))
        assert abs(inner_times[idx] - s) < 1e-9, f"local date {s} not aligned to the inner simulation grid"
        return idx

    idx_first, idx_last = local_grid_index(local_dates[0]), local_grid_index(local_dates[-1])
    D_first = np.exp(-log_B[:, idx_first])
    D_last = np.exp(-log_B[:, idx_last])
    float_leg = notional * (D_first - D_last)

    fixed_leg = np.zeros(n_inner_paths)
    for s in local_dates:
        D_i = np.exp(-log_B[:, local_grid_index(s)])
        fixed_leg += notional * fixed_rate * accrual * D_i

    realized = fixed_leg - float_leg
    analytic_value = float(analytic_swap_value(
        curve, a, sigma, checkpoint_time, remaining_payment_dates, accrual,
        notional, fixed_rate, short_rate_value,
    ))

    return {
        "ground_truth_mean": float(realized.mean()),
        "ground_truth_stderr": float(realized.std(ddof=1) / np.sqrt(n_inner_paths)),
        "analytic_value": analytic_value,
    }


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def run_phase4(config_path: str | Path = "config.yaml"):
    """Calibrate Hull-White (from Phase 3's method, re-run here so this
    phase is runnable standalone), simulate the swap-exposure training
    dataset, and validate that the noisy realized labels are unbiased
    relative to the analytic conditional expectation at each checkpoint --
    a distinct, earlier check than Phase 5's naive-baseline-bias story,
    which specifically concerns flooring at E[max(V,0)].
    """
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    hw_cfg = cfg.get("hull_white", {})
    swap_cfg = cfg["swap"]

    print(f"=== Phase 4: Monte Carlo synthetic exposure dataset (valuation_date={valuation_date}) ===\n")

    series_id = hw_cfg.get("short_rate_series", DEFAULT_SHORT_RATE_SERIES)
    series = fetch_fred_series(series_id) / 100.0
    series = series[series.index <= pd.Timestamp(valuation_date)]
    calibration = calibrate_ar1(series)
    a, sigma = calibration["a"], calibration["sigma"]
    r0 = float(series.iloc[-1])
    print(f"Hull-White params (re-calibrated as in Phase 3): a={a:.4f}, sigma={sigma:.4f}, r0={r0:.4%}")

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    theta = calibrate_theta(curve, a, sigma)

    n_paths = swap_cfg.get("n_exposure_paths", 5000)
    dataset = simulate_swap_exposure_paths(
        curve, a, sigma, theta, r0,
        notional=swap_cfg["notional"], fixed_rate=swap_cfg["fixed_rate"],
        tenor_years=swap_cfg["tenor_years"], payment_frequency=swap_cfg["payment_frequency"],
        n_paths=n_paths, seed=42,
    )
    print(f"\nSwap: notional={swap_cfg['notional']:,.0f}, fixed_rate={swap_cfg['fixed_rate']:.2%}, "
          f"tenor={swap_cfg['tenor_years']}y, freq={swap_cfg['payment_frequency']}")
    print(f"Generated {len(dataset):,} (state, noisy-label) rows: {n_paths} paths x "
          f"{dataset['checkpoint_time'].nunique()} checkpoints")

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"exposure_training_data_{valuation_date}.csv"
    dataset.to_csv(out_path, index=False)

    print("\n--- Unbiasedness check: pooled mean(realized - analytic) per checkpoint (want: ~0) ---")
    summary = dataset.groupby("checkpoint_time").apply(
        lambda g: pd.Series({
            "mean_realized": g["realized_swap_value"].mean(),
            "mean_analytic": g["analytic_conditional_value"].mean(),
            "bias": (g["realized_swap_value"] - g["analytic_conditional_value"]).mean(),
            "bias_stderr": (g["realized_swap_value"] - g["analytic_conditional_value"]).sem(),
            "corr_realized_vs_analytic": g["realized_swap_value"].corr(g["analytic_conditional_value"]),
        }),
        include_groups=False,
    )
    print(summary.to_string())
    max_abs_t_stat = float((summary["bias"] / summary["bias_stderr"]).abs().max())
    print(f"\nMax |bias / stderr| across checkpoints: {max_abs_t_stat:.2f} "
          f"(a large value here would flag a real bug in the realized-value calculation, "
          f"not just Monte Carlo noise)")

    example_checkpoint = float(dataset["checkpoint_time"].median())
    example = dataset[np.isclose(dataset["checkpoint_time"], example_checkpoint)].sort_values("short_rate")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(example["short_rate"], example["realized_swap_value"], s=4, alpha=0.25,
               color="tab:blue", label="Realized value (noisy label, one path)")
    ax.plot(example["short_rate"], example["analytic_conditional_value"], color="tab:orange",
            linewidth=2, label="Analytic E[value | short rate]")
    ax.set_xlabel("Short rate at checkpoint")
    ax.set_ylabel("Swap value")
    ax.set_title(f"Phase 4 synthetic dataset at checkpoint t={example_checkpoint:.2f}y "
                 f"(this is exactly what Phase 5 will regress)")
    ax.legend()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "exposure_training_data_example_checkpoint.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}, {fig_path}")
    return dataset, summary


def run_phase5(config_path: str | Path = "config.yaml"):
    """Fit the naive baseline and four regression models per checkpoint on
    Phase 4's training data, benchmark all of them against a fresh nested-
    simulation ground truth, and report the MAE/RMSE comparison table and
    bias-direction check called for in the plan's Phase 5.
    """
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    hw_cfg = cfg.get("hull_white", {})
    swap_cfg = cfg["swap"]

    print(f"=== Phase 5: EE(t) regression model comparison (valuation_date={valuation_date}) ===\n")

    series_id = hw_cfg.get("short_rate_series", DEFAULT_SHORT_RATE_SERIES)
    series = fetch_fred_series(series_id) / 100.0
    series = series[series.index <= pd.Timestamp(valuation_date)]
    calibration = calibrate_ar1(series)
    a, sigma = calibration["a"], calibration["sigma"]

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    theta = calibrate_theta(curve, a, sigma)

    dataset_path = processed_dir / f"exposure_training_data_{valuation_date}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found -- run Phase 4 first (python -m src.exposure_models).")
    dataset = pd.read_csv(dataset_path)

    notional, fixed_rate = swap_cfg["notional"], swap_cfg["fixed_rate"]
    tenor_years, payment_frequency = swap_cfg["tenor_years"], swap_cfg["payment_frequency"]
    payment_dates, accrual = build_payment_schedule(tenor_years, payment_frequency)

    all_checkpoints = np.sort(dataset["checkpoint_time"].unique())
    test_checkpoints = [all_checkpoints[i] for i in
                         (len(all_checkpoints) // 4, len(all_checkpoints) // 2, 3 * len(all_checkpoints) // 4)]
    print(f"Testing at {len(test_checkpoints)} checkpoints: {[f'{t:.2f}' for t in test_checkpoints]} "
          f"(of {len(all_checkpoints)} available)")

    model_names = ["naive", "poly", "random_forest", "gradient_boosting", "mlp"]
    rows = []
    fitted_models_by_checkpoint = {}

    for t_k in test_checkpoints:
        group = dataset[dataset["checkpoint_time"] == t_k]
        x, y = group["short_rate"].to_numpy(), group["realized_swap_value"].to_numpy()
        remaining_dates = payment_dates[payment_dates >= t_k]

        naive_pred = naive_baseline(y)
        models = {
            "poly": fit_polynomial_regression(x, y),
            "random_forest": fit_random_forest(x, y),
            "gradient_boosting": fit_gradient_boosting(x, y),
            "mlp": fit_mlp(x, y),
        }
        fitted_models_by_checkpoint[t_k] = models

        test_rates = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        for r_test in test_rates:
            gt = nested_simulation_ground_truth(
                curve, a, sigma, theta, t_k, float(r_test), remaining_dates, accrual,
                notional, fixed_rate, tenor_years, n_inner_paths=20000, seed=123,
            )
            row = {
                "checkpoint_time": t_k, "test_rate": r_test,
                "ground_truth": gt["ground_truth_mean"],
                "ground_truth_stderr": gt["ground_truth_stderr"],
                "analytic": gt["analytic_value"],
                "naive": naive_pred,
                **{name: float(predict_regression_model(model, r_test)[0]) for name, model in models.items()},
            }
            rows.append(row)

    results = pd.DataFrame(rows)
    for name in model_names:
        results[f"{name}_error"] = results[name] - results["ground_truth"]

    print("\n--- Nested-sim vs. closed-form cross-check (want: small gap -- validates the nested-sim methodology) ---")
    cross_check_mae = float((results["ground_truth"] - results["analytic"]).abs().mean())
    print(f"Mean |ground_truth - analytic|: {cross_check_mae:,.2f} "
          f"(vs. notional {notional:,.0f} -- should be small)")

    summary = pd.DataFrame({
        name: {
            "MAE": results[f"{name}_error"].abs().mean(),
            "RMSE": np.sqrt((results[f"{name}_error"] ** 2).mean()),
            "mean_bias": results[f"{name}_error"].mean(),
            "pct_error_positive": (results[f"{name}_error"] > 0).mean(),
        }
        for name in model_names
    }).T
    print("\n--- Model comparison vs. nested-simulation ground truth (all 3 checkpoints x 5 rates pooled) ---")
    print(summary.to_string())
    print(
        "\nNote: test rates span the 5th-95th percentile of each checkpoint's simulated short-rate "
        "distribution, so a large share of naive's error here is simply from being state-blind "
        "(one constant prediction vs. a steeply state-varying true curve) -- the flooring/Jensen "
        "effect from the module docstring is a real, additional, and structurally one-directional "
        "contributor on top of that (naive's mean_bias should be positive, and pct_error_positive "
        "meaningfully above 50%), but it is not the dominant source of naive's error magnitude here."
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_path = processed_dir / f"exposure_model_comparison_{valuation_date}.csv"
    summary_path = processed_dir / f"exposure_model_comparison_summary_{valuation_date}.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path)

    # Overfitting-risk diagnostic: fitted curves vs. the (free, exact)
    # analytic curve and the raw noisy training scatter, at the middle
    # test checkpoint -- does RF/MLP wiggle around noise instead of
    # tracking the smooth truth?
    mid_checkpoint = test_checkpoints[1]
    group = dataset[dataset["checkpoint_time"] == mid_checkpoint]
    x, y = group["short_rate"].to_numpy(), group["realized_swap_value"].to_numpy()
    remaining_dates = payment_dates[payment_dates >= mid_checkpoint]
    x_grid = np.linspace(x.min(), x.max(), 300)
    analytic_grid = analytic_swap_value(curve, a, sigma, mid_checkpoint, remaining_dates, accrual,
                                         notional, fixed_rate, x_grid)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=3, alpha=0.15, color="gray", label="Raw noisy training labels", zorder=1)
    ax.plot(x_grid, analytic_grid, color="black", linewidth=2.5, label="Analytic (exact) curve", zorder=5)
    for name, color in zip(["poly", "random_forest", "gradient_boosting", "mlp"],
                            ["tab:blue", "tab:green", "tab:orange", "tab:red"]):
        y_grid = predict_regression_model(fitted_models_by_checkpoint[mid_checkpoint][name], x_grid)
        ax.plot(x_grid, y_grid, color=color, linewidth=1.5, label=name, zorder=4)
    ax.axhline(naive_baseline(y), color="purple", linestyle="--", linewidth=1.5,
               label="Naive baseline (constant)", zorder=4)
    ax.set_xlabel("Short rate")
    ax.set_ylabel("Swap value")
    ax.set_title(f"Phase 5: fitted models vs. analytic truth at checkpoint t={mid_checkpoint:.2f}y")
    ax.legend(fontsize=8)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "exposure_model_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {results_path}, {summary_path}, {fig_path}")
    return results, summary


if __name__ == "__main__":
    run_phase4()
    print()
    run_phase5()
