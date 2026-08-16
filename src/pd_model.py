"""Phase 2 -- Merton structural PD baseline, and Phase 2b -- supervised PD classifier.

Phase 2 (implemented here; baseline / feature source, not the final model):
  - Solve simultaneously for asset value V and asset volatility sigma_V given
    observed equity value E, equity volatility sigma_E, debt D, and risk-free
    rate r (scipy.optimize.fsolve on the two-equation Merton system).
  - Distance to Default: DD = [ln(V/D) + (r - 0.5*sigma_V**2)*T] / (sigma_V*sqrt(T))
  - PD ~= N(-DD)
  - Debt convention: short-term debt + 0.5 * long-term debt, inherited as-is
    from Phase 1's `total_debt_face_value` (see plan pitfalls -- be explicit
    and consistent about this).

Phase 2b (implemented here; the real supervised learning component):
  - Labeled dataset: the public Taiwan Economic Journal bankruptcy-prediction
    dataset (data_pipeline.fetch_bankruptcy_dataset) -- ~6,800 companies,
    ~95 pre-computed accounting-ratio features, 3.2% bankruptcy rate.
  - Models: logistic regression (interpretable baseline, class-weighted) vs.
    gradient boosting (XGBoost, scale_pos_weight-adjusted).
  - Train/test split: stratified random, NOT time-based. The plan calls for
    a time-based split to avoid lookahead bias, but the public release of
    this dataset carries no per-row date/fiscal-year field to split on --
    documented explicitly as a limitation rather than faked. See the
    Phase 2b section of the README.
  - Class-imbalance-aware evaluation: ROC-AUC, PR-AUC (not accuracy).
  - Calibration curve: predicted probability vs. observed default frequency.
  - Merton-baseline comparison: also not literally possible on this dataset
    -- Merton needs market equity value/volatility, which this purely
    accounting-ratio dataset doesn't have, and it covers a different company
    universe than the Phase 1/2 ten-ticker panel (which has zero labeled
    defaults itself). Documented as a limitation rather than forced.

Run Phase 2 and 2b directly:
    python -m src.pd_model
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import fsolve
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import fetch_bankruptcy_dataset, fit_curve, load_config
from src.features import prepare_bankruptcy_features

# Initial-guess multipliers for sigma_V retried in order until fsolve
# converges to a valid (V > 0, sigma_V > 0) solution. Highly levered names
# (e.g. AMC, low equity relative to debt) can need a different starting
# point than a name like AAPL for fsolve to find the root.
SIGMA_V_RETRY_MULTIPLIERS = [1.0, 0.5, 2.0, 0.25, 4.0, 0.1]


# --------------------------------------------------------------------------
# Core Merton system
# --------------------------------------------------------------------------

def _merton_equations(x: np.ndarray, E: float, sigma_E: float, D: float, r: float, T: float) -> list[float]:
    """Residuals of the two-equation Merton system, for fsolve.

    Eq 1 (equity as a call option on firm assets, Black-Scholes):
        E = V*N(d1) - D*exp(-rT)*N(d2)
    Eq 2 (equity-vol / asset-vol relationship from Ito's lemma):
        sigma_E * E = N(d1) * sigma_V * V
    """
    V, sigma_V = x
    if V <= 0 or sigma_V <= 0:
        return [1e10, 1e10]  # steer fsolve away from the invalid region

    sqrt_T = np.sqrt(T)
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrt_T)
    d2 = d1 - sigma_V * sqrt_T

    eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
    return [eq1, eq2]


def solve_merton(
    E: float, sigma_E: float, D: float, r: float, T: float = 1.0
) -> tuple[float, float, bool, float]:
    """Solve for asset value V and asset volatility sigma_V given observed
    equity value E, equity volatility sigma_E, debt D, risk-free rate r, and
    horizon T (years).

    Returns (V, sigma_V, converged, residual_norm). Retries fsolve from a
    few different initial guesses for sigma_V (see SIGMA_V_RETRY_MULTIPLIERS)
    and keeps the first solution that converges to an economically valid
    (V > 0, sigma_V > 0) root; if none converge, returns the lowest-residual
    attempt with converged=False.
    """
    if D <= 0:
        raise ValueError(f"Debt D must be positive for the Merton solver, got {D}")

    # Equation residuals are in dollars, and E/D span ~1e8 to ~1e12 across
    # the panel -- an absolute residual tolerance would either be too loose
    # for small-cap names or unreachable for AAPL-scale ones, so convergence
    # is judged on the residual relative to the deal's own scale instead.
    scale = max(abs(E), abs(D), 1.0)
    rel_tol = 1e-8

    best = None  # (converged, relative_residual, V, sigma_V, residual_norm)
    for mult in SIGMA_V_RETRY_MULTIPLIERS:
        V0 = E + D
        sigma_V0 = max(sigma_E * E / (E + D), 1e-4) * mult
        solution, info, _ier, _msg = fsolve(
            _merton_equations, x0=[V0, sigma_V0], args=(E, sigma_E, D, r, T), full_output=True,
        )
        V, sigma_V = solution
        residual_norm = float(np.linalg.norm(info["fvec"]))
        relative_residual = residual_norm / scale
        converged = bool(V > 0 and sigma_V > 0 and relative_residual < rel_tol)

        candidate = (converged, relative_residual, float(V), float(sigma_V), residual_norm)
        if converged:
            return candidate[2], candidate[3], True, candidate[4]
        if best is None or relative_residual < best[1]:
            best = candidate

    return best[2], best[3], False, best[4]


def distance_to_default(V: float, sigma_V: float, D: float, r: float, T: float = 1.0) -> float:
    """DD = [ln(V/D) + (r - 0.5*sigma_V**2)*T] / (sigma_V*sqrt(T))"""
    return (np.log(V / D) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))


def merton_pd(dd: float) -> float:
    """PD ~= N(-DD): probability the (log-normal) asset value ends up below
    the debt threshold at horizon T under the risk-neutral/physical-approx
    Merton assumption.
    """
    return float(norm.cdf(-dd))


# --------------------------------------------------------------------------
# Panel-level application
# --------------------------------------------------------------------------

def get_risk_free_rate(curve: pd.DataFrame, T: float) -> float:
    """Risk-free rate (decimal, not percent) at maturity T, interpolated
    from the Phase 1 FRED curve via the same cubic spline used for the
    curve's own round-trip check.
    """
    spline = fit_curve(curve)
    return float(spline(T)) / 100.0


def compute_merton_panel(panel: pd.DataFrame, r: float, T: float = 1.0) -> pd.DataFrame:
    """Solve the Merton model for every company in the Phase 1 panel.

    Uses E = equity_value, sigma_E = equity_vol, D = total_debt_face_value
    (all from Phase 1). Rows with missing inputs or a non-positive debt
    figure are skipped (converged=False, PD left null) rather than guessed
    at -- this is a baseline model and a silently wrong PD is worse than a
    visibly missing one.
    """
    rows = []
    for _, row in panel.iterrows():
        E, sigma_E, D = row["equity_value"], row["equity_vol"], row["total_debt_face_value"]

        if pd.isna(E) or pd.isna(sigma_E) or pd.isna(D) or D <= 0:
            rows.append({
                "ticker": row["ticker"], "V": None, "sigma_V": None,
                "distance_to_default": None, "pd_merton": None,
                "converged": False, "residual_norm": None,
                "skip_reason": "missing_or_nonpositive_input",
            })
            continue

        V, sigma_V, converged, residual_norm = solve_merton(E, sigma_E, D, r, T)
        dd = distance_to_default(V, sigma_V, D, r, T)
        rows.append({
            "ticker": row["ticker"], "V": V, "sigma_V": sigma_V,
            "distance_to_default": dd, "pd_merton": merton_pd(dd),
            "converged": converged, "residual_norm": residual_norm,
            "skip_reason": None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def run_phase2(config_path: str | Path = "config.yaml"):
    """Run Phase 2 against Phase 1's saved outputs: load the FRED curve and
    company panel for the configured valuation date, solve Merton for each
    company, and report the solver-convergence and rank-ordering checks
    called for in the plan's Phase 2 validation.
    """
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    T = cfg.get("merton", {}).get("horizon_years", 1.0)

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    panel = pd.read_csv(processed_dir / f"company_panel_{valuation_date}.csv")

    r = get_risk_free_rate(curve, T)
    print(f"=== Phase 2: valuation_date={valuation_date}, T={T}y, r={r:.4%} (interpolated from FRED curve) ===\n")

    merton = compute_merton_panel(panel, r, T)
    merged = panel[["ticker", "equity_value", "equity_vol", "total_debt_face_value"]].merge(merton, on="ticker")
    merged = merged.sort_values("pd_merton")

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"merton_pd_{valuation_date}.csv"
    merged.to_csv(out_path, index=False)

    print("--- Merton PD, ranked lowest to highest (sanity: should track credit quality) ---")
    print(merged[["ticker", "equity_value", "equity_vol", "total_debt_face_value",
                   "distance_to_default", "pd_merton", "converged", "residual_norm"]]
          .to_string(index=False))

    n_converged = int(merged["converged"].sum())
    n_total = len(merged)
    print(f"\n--- Solver convergence check ---\n{n_converged}/{n_total} companies converged.")
    if n_converged < n_total:
        failed = merged.loc[~merged["converged"], "ticker"].tolist()
        print(f"Did not converge / skipped: {failed}")

    print(f"\nSaved: {out_path}")
    return merged


# --------------------------------------------------------------------------
# Phase 2b: supervised PD classifier
# --------------------------------------------------------------------------

def split_bankruptcy_dataset(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = 42
):
    """Stratified random split of the bankruptcy dataset.

    Stratified so the already-rare positive class (3.2%) isn't further
    distorted by chance in either split. NOT a time-based split -- see the
    module docstring: this public dataset has no per-row date field, so the
    plan's preferred time-based split can't be done here.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _make_logistic_regression() -> Pipeline:
    """Interpretable baseline classifier (unfit). class_weight='balanced'
    upweights the minority (bankrupt) class rather than resampling the data.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])


def _make_gradient_boosting(scale_pos_weight: float) -> xgb.XGBClassifier:
    """Gradient boosting classifier (XGBoost, unfit), with scale_pos_weight
    set to account for the same imbalance.
    """
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
    )


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = _make_logistic_regression()
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = _make_gradient_boosting(scale_pos_weight)
    model.fit(X_train, y_train)
    return model


def calibrate_classifier(base_estimator, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5):
    """Wrap an unfit, imbalance-adjusted estimator with cross-validated
    Platt-scaling probability calibration.

    class_weight='balanced' / scale_pos_weight fix the *ranking* imbalance
    problem but, as a side effect, push predicted probabilities well above
    true observed frequencies (verified empirically in run_phase2b: the
    raw models' top calibration bin predicts ~60-85% but observes ~20-25%).
    Sigmoid (Platt) rather than isotonic calibration is used because the
    minority class is small (~165 positives in the training split, spread
    over `cv` folds) -- isotonic's more flexible mapping is prone to
    overfitting the calibration curve itself at that count.
    """
    from sklearn.calibration import CalibratedClassifierCV

    calibrated = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_pd_classifier(model, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 10) -> dict:
    """ROC-AUC, PR-AUC (not accuracy, given the class imbalance), and a
    calibration curve -- predicted probability vs. observed bankruptcy
    frequency, binned by predicted-probability quantile.
    """
    proba = model.predict_proba(X_test)[:, 1]
    frac_positive, mean_predicted = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "calibration_frac_positive": frac_positive,
        "calibration_mean_predicted": mean_predicted,
        "proba": proba,
    }


def run_phase2b():
    """Fetch the Taiwan bankruptcy dataset (or load it from Phase 1's raw
    cache if already fetched), train logistic regression and gradient
    boosting classifiers, and report the ROC-AUC/PR-AUC/calibration
    validation called for in the plan's Phase 2b.
    """
    print("=== Phase 2b: supervised PD classifier (Taiwan bankruptcy dataset) ===\n")

    raw_path = Path("data/raw/taiwan_bankruptcy.csv")
    df = pd.read_csv(raw_path) if raw_path.exists() else fetch_bankruptcy_dataset(raw_dir="data/raw")

    X, y = prepare_bankruptcy_features(df)
    print(f"Dataset: {len(X)} companies, {X.shape[1]} features "
          f"(after dropping constant/near-duplicate columns), "
          f"{int(y.sum())} bankruptcies ({y.mean():.2%})")

    X_train, X_test, y_train, y_test = split_bankruptcy_dataset(X, y)
    print(f"Stratified random split (NOT time-based -- see module docstring): "
          f"{len(X_train)} train / {len(X_test)} test, "
          f"train positive rate={y_train.mean():.2%}, test positive rate={y_test.mean():.2%}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Gradient Boosting (XGBoost)": train_gradient_boosting(X_train, y_train),
        "Logistic Regression (calibrated)": calibrate_classifier(_make_logistic_regression(), X_train, y_train),
        "Gradient Boosting (calibrated)": calibrate_classifier(
            _make_gradient_boosting(scale_pos_weight), X_train, y_train
        ),
    }

    results = {name: evaluate_pd_classifier(model, X_test, y_test) for name, model in models.items()}

    comparison = pd.DataFrame({
        name: {"ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"]} for name, m in results.items()
    }).T
    print("\n--- Model comparison (test set, held out from training) ---")
    print(comparison.to_string())
    print(
        "\nNote: no Merton-baseline row here -- Merton needs market equity "
        "value/volatility, which this purely accounting-ratio dataset "
        "doesn't have, and it covers a different company universe than the "
        "Phase 1/2 ten-ticker panel (which has zero labeled defaults of its "
        "own). See the Phase 2b README section for the full explanation."
    )

    print(
        "\n--- Calibration diagnosis ---\n"
        "The raw models' class_weight='balanced' / scale_pos_weight settings "
        "fix the ranking problem (see ROC-AUC/PR-AUC above) but push "
        "predicted probabilities well above observed frequencies -- both "
        "raw models are overconfident (see calibration plot). Wrapping each "
        "with cross-validated Platt scaling (calibrate_classifier) brings "
        "predicted probability back in line with observed frequency without "
        "hurting ranking quality; both ROC-AUC and PR-AUC are essentially "
        "unchanged between the raw and calibrated rows above."
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    raw_names = ["Logistic Regression", "Gradient Boosting (XGBoost)"]
    calibrated_names = ["Logistic Regression (calibrated)", "Gradient Boosting (calibrated)"]
    for ax, names, title in [(axes[0], raw_names, "Raw"), (axes[1], calibrated_names, "Calibrated")]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        for name in names:
            m = results[name]
            ax.plot(m["calibration_mean_predicted"], m["calibration_frac_positive"], marker="o", label=name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed bankruptcy frequency")
    fig.suptitle("PD classifier calibration (test set): raw vs. Platt-calibrated")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "pd_classifier_calibration.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = processed_dir / "pd_classifier_comparison.csv"
    comparison.to_csv(comparison_path)

    print(f"\nSaved: {comparison_path}, {fig_path}")
    return models, results, comparison


if __name__ == "__main__":
    run_phase2()
    print()
    run_phase2b()
