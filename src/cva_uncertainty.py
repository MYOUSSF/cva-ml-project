"""Extension -- Bootstrapped CVA confidence interval.

Phase 7's CVA is a single point number. This propagates uncertainty from
three sources through to a CVA *distribution* instead:

  1. The Monte Carlo exposure simulation itself -- resample Phase 4's
     simulated paths (with replacement, whole paths at a time so a single
     path's value at every checkpoint stays linked, since one Hull-White
     draw determines all of them jointly) and recompute EE(t) via the same
     naive-style floor-then-average estimator `naive_baseline` uses.
  2. The Merton PD estimate's own sampling uncertainty -- bootstrap the
     trailing daily equity log returns behind sigma_E, re-solve the Merton
     system (`pd_model.solve_merton`) with the resampled sigma_E, holding
     E, D, r fixed.
  3. LGD -- recovery rate isn't a single number in practice; published
     recovery studies for senior unsecured claims report a range around
     the ~40% central estimate (roughly 20% in stressed cycles to 60% in
     benign ones), not a point value. Drawn here from a triangular
     distribution peaked at the config recovery rate.

Each source can be toggled independently (`vary_paths`/`vary_pd`/
`vary_lgd`) -- run_bootstrap_cva_ci uses this to check that each one is
actually injecting variance (the documented pitfall: a bootstrap that
resamples the seed but keeps PD/LGD fixed produces an artificially narrow
interval that looks precise without being meaningful).

Deliberately NOT refitting Phase 5's chosen regression model per bootstrap
iteration -- 500+ refits per checkpoint would be expensive for a CI whose
purpose is capturing the interval's *width*, not sharpening the point
estimate (already validated separately in Phase 5). The naive-style
resampling here is the bootstrap analogue of Phase 7's own naive_baseline
EE(t) estimator, so its mean should track Phase 7's naive-EE CVA number as
a sanity check.

Run directly against Phase 1/2/4's saved outputs:
    python -m src.cva_uncertainty
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cva import compute_cva, compute_lgd, pd_term_structure_from_1y_pd
from src.data_pipeline import load_config
from src.exposure_models import compute_ee_curve
from src.pd_model import distance_to_default, get_risk_free_rate, merton_pd, solve_merton

DEFAULT_N_ITERATIONS = 500
# Plausible published range for senior-unsecured recovery around the
# project's 40% ISDA point estimate (see config.yaml `lgd` section) --
# roughly the spread historical studies report between stressed and benign
# credit cycles, not a single number.
DEFAULT_RECOVERY_RATE_RANGE = (0.20, 0.60)


# --------------------------------------------------------------------------
# Per-source resampling primitives
# --------------------------------------------------------------------------

def bootstrap_annualized_vol(
    log_returns: np.ndarray, rng: np.random.Generator, trading_days: int = 252
) -> float:
    """Resample daily log returns with replacement (same length as input)
    and recompute annualized volatility -- the bootstrap analogue of
    `data_pipeline.compute_realized_equity_vol`.
    """
    n = len(log_returns)
    resampled = log_returns[rng.integers(0, n, size=n)]
    return float(resampled.std(ddof=1) * np.sqrt(trading_days))


def bootstrap_pd_1y(E: float, D: float, r: float, log_returns: np.ndarray, rng: np.random.Generator, T: float = 1.0) -> float:
    """Resample sigma_E, re-solve Merton (E, D, r held fixed), return the
    resulting 1y PD -- the Merton-PD-estimation-uncertainty source.
    """
    sigma_E = bootstrap_annualized_vol(log_returns, rng)
    V, sigma_V, _converged, _residual = solve_merton(E, sigma_E, D, r, T)
    dd = distance_to_default(V, sigma_V, D, r, T)
    return merton_pd(dd)


def bootstrap_ee_curve(dataset: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Resample whole simulated paths (by path_id, with replacement) and
    recompute EE(t) per checkpoint via floor-then-average -- the Monte-
    Carlo-simulation-uncertainty source. Resampling whole paths (rather than
    resampling each checkpoint's rows independently) preserves the fact
    that a single Hull-White draw determines one path's value at every
    checkpoint jointly.
    """
    path_ids = dataset["path_id"].unique()
    sampled_ids = rng.choice(path_ids, size=len(path_ids), replace=True)
    counts = pd.Series(sampled_ids).value_counts()

    checkpoints = np.sort(dataset["checkpoint_time"].unique())
    values = {}
    for t_k in checkpoints:
        group = dataset.loc[dataset["checkpoint_time"] == t_k].set_index("path_id")
        resampled_values = group.loc[counts.index, "realized_swap_value"].to_numpy()
        floored = np.maximum(resampled_values, 0.0)
        values[t_k] = float(np.average(floored, weights=counts.to_numpy()))
    return pd.Series(values).sort_index()


def bootstrap_recovery_rate(rng: np.random.Generator, low: float, mode: float, high: float) -> float:
    """Triangular draw from the plausible published recovery-rate range,
    peaked at the config point estimate.
    """
    return float(rng.triangular(low, mode, high))


# --------------------------------------------------------------------------
# Combined bootstrap
# --------------------------------------------------------------------------

def bootstrap_cva_distribution(
    dataset: pd.DataFrame,
    curve: pd.DataFrame,
    E: float,
    D: float,
    r: float,
    log_returns: np.ndarray,
    recovery_rate_base: float,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    recovery_rate_range: tuple[float, float] | None = None,
    vary_paths: bool = True,
    vary_pd: bool = True,
    vary_lgd: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Run n_iterations of resampled CVA. Any source not toggled on is held
    fixed at its point estimate for every iteration -- used by
    run_bootstrap_cva_ci's ablation check that each source actually injects
    variance (all three off should give an exactly-degenerate, zero-std
    distribution).
    """
    rng = np.random.default_rng(seed)
    recovery_low, recovery_high = recovery_rate_range or DEFAULT_RECOVERY_RATE_RANGE
    checkpoint_times = np.sort(dataset["checkpoint_time"].unique())

    base_ee = compute_ee_curve(dataset, method="naive")
    V0, sigma_V0, _c, _r = solve_merton(E, log_returns.std(ddof=1) * np.sqrt(252), D, r, 1.0)
    base_pd = merton_pd(distance_to_default(V0, sigma_V0, D, r, 1.0))

    cvas = np.empty(n_iterations)
    for i in range(n_iterations):
        ee_curve = bootstrap_ee_curve(dataset, rng) if vary_paths else base_ee
        pd_1y = bootstrap_pd_1y(E, D, r, log_returns, rng) if vary_pd else base_pd
        recovery = (
            bootstrap_recovery_rate(rng, recovery_low, recovery_rate_base, recovery_high)
            if vary_lgd else recovery_rate_base
        )
        lgd = compute_lgd(recovery)
        pd_term_structure = pd_term_structure_from_1y_pd(pd_1y, checkpoint_times)
        cvas[i] = compute_cva(ee_curve, pd_term_structure, lgd, curve)

    return cvas


def summarize_distribution(cvas: np.ndarray, lower_pct: float = 5, upper_pct: float = 95) -> dict:
    return {
        "mean": float(np.mean(cvas)),
        "std": float(np.std(cvas, ddof=1)) if len(cvas) > 1 else 0.0,
        "median": float(np.median(cvas)),
        f"p{lower_pct}": float(np.percentile(cvas, lower_pct)),
        f"p{upper_pct}": float(np.percentile(cvas, upper_pct)),
    }


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def run_bootstrap_cva_ci(config_path: str | Path = "config.yaml", n_iterations: int = DEFAULT_N_ITERATIONS):
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    recovery_rate = cfg["lgd"]["recovery_rate"]
    cva_cfg = cfg.get("cva", {})
    counterparty_ticker = cva_cfg.get("counterparty_ticker", "AMC")
    merton_cfg = cfg.get("merton", {})
    T = merton_cfg.get("horizon_years", 1.0)

    print(f"=== Extension: bootstrapped CVA confidence interval (counterparty={counterparty_ticker}) ===\n")

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    merton = pd.read_csv(processed_dir / f"merton_pd_{valuation_date}.csv")
    row = merton.loc[merton["ticker"] == counterparty_ticker].iloc[0]
    E, D = float(row["equity_value"]), float(row["total_debt_face_value"])
    r = get_risk_free_rate(curve, T)

    equity_history = pd.read_csv(Path("data/raw/equity_history.csv"), parse_dates=["date"])
    hist = equity_history.loc[equity_history["ticker"] == counterparty_ticker].sort_values("date")
    hist = hist[hist["date"] <= pd.Timestamp(valuation_date)].tail(253)
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna().to_numpy()
    print(f"{len(log_returns)} trailing daily log returns for {counterparty_ticker} "
          f"(same lookback window Phase 1 uses for equity_vol)")

    dataset_path = processed_dir / f"exposure_training_data_{valuation_date}.csv"
    dataset = pd.read_csv(dataset_path)

    variants = {
        "all_sources": dict(vary_paths=True, vary_pd=True, vary_lgd=True),
        "paths_only": dict(vary_paths=True, vary_pd=False, vary_lgd=False),
        "pd_only": dict(vary_paths=False, vary_pd=True, vary_lgd=False),
        "lgd_only": dict(vary_paths=False, vary_pd=False, vary_lgd=True),
        "none_degenerate_check": dict(vary_paths=False, vary_pd=False, vary_lgd=False),
    }

    summaries = {}
    full_cvas = None
    for name, kwargs in variants.items():
        cvas = bootstrap_cva_distribution(
            dataset, curve, E, D, r, log_returns, recovery_rate,
            n_iterations=n_iterations, seed=42, **kwargs,
        )
        summaries[name] = summarize_distribution(cvas)
        if name == "all_sources":
            full_cvas = cvas

    summary_df = pd.DataFrame(summaries).T
    print("\n--- Ablation: does each uncertainty source actually inject variance? ---")
    print(summary_df.to_string())
    print(
        "\nnone_degenerate_check's std should be exactly 0 (nothing varying -> point estimate every "
        "iteration); each *_only variant's std should be > 0 -- confirms every source claimed as "
        "'propagated' is actually varying, not just resampling a seed that doesn't move the number."
    )
    assert summary_df.loc["none_degenerate_check", "std"] == 0.0, "degenerate check should have exactly zero variance"
    for name in ("paths_only", "pd_only", "lgd_only"):
        assert summary_df.loc[name, "std"] > 0.0, f"{name} should inject nonzero variance"

    print(f"\n--- Full distribution (all sources, {n_iterations} iterations) ---")
    full_summary = summaries["all_sources"]
    print(f"Mean: ${full_summary['mean']:,.0f}  Median: ${full_summary['median']:,.0f}  "
          f"Std: ${full_summary['std']:,.0f}")
    print(f"90% interval: [${full_summary['p5']:,.0f}, ${full_summary['p95']:,.0f}]")

    processed_dir.mkdir(parents=True, exist_ok=True)
    dist_path = processed_dir / f"cva_bootstrap_distribution_{valuation_date}.csv"
    summary_path = processed_dir / f"cva_bootstrap_summary_{valuation_date}.csv"
    pd.DataFrame({"cva": full_cvas}).to_csv(dist_path, index=False)
    summary_df.to_csv(summary_path)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(full_cvas, bins=40, color="tab:blue", alpha=0.75)
    ax.axvline(full_summary["p5"], color="tab:red", linestyle="--", linewidth=1.5, label="5th percentile")
    ax.axvline(full_summary["p95"], color="tab:red", linestyle="--", linewidth=1.5, label="95th percentile")
    ax.axvline(full_summary["median"], color="black", linestyle="-", linewidth=1.5, label="Median")
    ax.set_xlabel("CVA ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bootstrapped CVA distribution ({counterparty_ticker}, {n_iterations} iterations)\n"
                 "Monte Carlo paths + Merton PD sampling + LGD range, all resampled jointly")
    ax.legend()
    fig.tight_layout()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "cva_bootstrap_histogram.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {dist_path}, {summary_path}, {fig_path}")
    return summary_df, full_cvas


if __name__ == "__main__":
    run_bootstrap_cva_ci()
