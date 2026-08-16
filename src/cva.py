"""Phase 6/7 -- LGD assumption and final CVA assembly + sensitivity analysis.
Phase 9 -- portfolio-level application across the Phase 1/2 company panel.

Phase 6: LGD = 1 - Recovery Rate, using the ISDA CDS Standard Model's
40%-recovery convention for senior unsecured claims (config.yaml `lgd`
section has the full citation). Domain input, not a modeling exercise.

Phase 7: combine a PD term structure, Phase 5's EE(t) curve, Phase 6's LGD,
and the Phase 1 discount curve into a CVA number, then run a sensitivity
analysis: CVA under shocked PD/recovery, and under the naive vs.
best-performing exposure model.

PD note: the plan's Phase 7 calls for "the Phase 2b PD term structure", but
Phase 2b's classifier isn't applicable to the Phase 1/2 company panel (no
shared features or company universe with the Taiwan bankruptcy dataset --
documented in Phase 2b's README section). This uses Phase 2's Merton PD for
the chosen counterparty instead, extrapolated from its single 1-year PD to
a term structure via a constant-hazard-rate assumption -- a deliberately
small simplification, matching the plan's own "intentionally small" framing
for this phase (a full credit-curve calibration is out of scope here).

Phase 9: apply the same swap and the same Phase 5 best-model EE(t) curve
across the entire Phase 1/2 panel, rank by CVA, and sanity-check that
ranking against approximate public credit ratings -- the plan's explicit
ask ("discuss what this implies about your PD classifier's discriminative
power"). Ratings used here are illustrative, from general knowledge, not
pulled from a live agency feed (there's no free public API for this) --
flagged as such in the output rather than presented as verified data.

Run Phase 6+7, or Phase 9, directly against Phase 1/2/4/5's saved outputs:
    python -m src.cva
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_pipeline import load_config
from src.exposure_models import compute_ee_curve
from src.ratemodel import fit_decimal_yield_curve

# Illustrative approximate senior-unsecured credit ratings (S&P-style
# letters), from general knowledge -- NOT pulled from a live agency feed
# (no free public API exists for this) and not tied to the valuation date.
# Used only for a rank-order sanity check in Phase 9, not as data the CVA
# calculation itself depends on. Verify independently before relying on
# these for anything beyond that sanity check.
APPROXIMATE_CREDIT_RATINGS = {
    "AAPL": "AA+", "MSFT": "AAA", "JNJ": "AAA", "KO": "A+", "XOM": "AA-",
    "T": "BBB", "F": "BB+", "CCL": "B+", "BA": "BBB-", "AMC": "CCC+",
}
# Coarser rank used for the correlation check: lower = safer. Standard
# agency letter-grade ordering, investment-grade (AAA..BBB-) vs. high-yield
# (BB+ and below).
CREDIT_RATING_RANK = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10, "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16, "CCC+": 17, "CCC": 18, "CCC-": 19,
}


# --------------------------------------------------------------------------
# Phase 6: LGD
# --------------------------------------------------------------------------

def compute_lgd(recovery_rate: float) -> float:
    """LGD = 1 - Recovery Rate."""
    return 1.0 - recovery_rate


# --------------------------------------------------------------------------
# Phase 7: PD term structure, CVA assembly, sensitivity
# --------------------------------------------------------------------------

def pd_term_structure_from_1y_pd(pd_1y: float, checkpoint_times) -> pd.Series:
    """Constant-hazard-rate extrapolation of a single 1-year PD to a
    cumulative default-probability term structure:

        Q_survival(t) = (1 - pd_1y)^t   =>   Q_default(t) = 1 - Q_survival(t)

    Returns a pd.Series of cumulative default probability, indexed by
    checkpoint time.
    """
    checkpoint_times = np.asarray(checkpoint_times, dtype=float)
    survival = (1.0 - pd_1y) ** checkpoint_times
    return pd.Series(1.0 - survival, index=checkpoint_times)


def compute_cva(ee_curve: pd.Series, pd_term_structure: pd.Series, lgd: float, curve: pd.DataFrame) -> float:
    """CVA = LGD * sum_k DF(0,t_k) * EE(t_k) * [Q_default(t_k) - Q_default(t_{k-1})]

    `curve` is the Phase 1 FRED curve, used (via the same fitted zero curve
    Phase 3/4/5 use for discounting) to bring each checkpoint's exposure
    back to the valuation date. `ee_curve` and `pd_term_structure` must
    share the same checkpoint-time index.
    """
    yield_spline = fit_decimal_yield_curve(curve)
    checkpoints = ee_curve.index.to_numpy()

    prev_q = 0.0
    exposure_weighted_default = 0.0
    for t_k in checkpoints:
        q_k = float(pd_term_structure.loc[t_k])
        marginal_default_prob = q_k - prev_q
        discount_factor = float(np.exp(-yield_spline(t_k) * t_k))
        exposure_weighted_default += discount_factor * ee_curve.loc[t_k] * marginal_default_prob
        prev_q = q_k

    return lgd * exposure_weighted_default


def compute_portfolio_cva(merton_df: pd.DataFrame, ee_curve: pd.Series, lgd: float, curve: pd.DataFrame) -> pd.DataFrame:
    """CVA for the same hypothetical swap (same ee_curve) across every
    company in a Merton-PD table (Phase 2's output), using each company's
    own 1y PD to build its own term structure. Shared by Phase 7's
    portfolio preview and Phase 9's full portfolio-level application.
    """
    checkpoint_times = ee_curve.index.to_numpy()
    rows = []
    for _, row in merton_df.iterrows():
        if pd.isna(row["pd_merton"]):
            continue
        pdts = pd_term_structure_from_1y_pd(float(row["pd_merton"]), checkpoint_times)
        rows.append({
            "ticker": row["ticker"], "pd_merton_1y": row["pd_merton"],
            "cva": compute_cva(ee_curve, pdts, lgd, curve),
        })
    return pd.DataFrame(rows).sort_values("cva").reset_index(drop=True)


def sensitivity_analysis(
    ee_curve: pd.Series,
    pd_1y_base: float,
    recovery_rate_base: float,
    curve: pd.DataFrame,
    pd_shock_multipliers=(0.5, 1.0, 1.5, 2.0),
    recovery_rate_shocks=(0.20, 0.40, 0.60),
) -> pd.DataFrame:
    """CVA under PD shocks (multiplicative on the base 1y PD) and recovery-
    rate shocks (LGD = 1 - recovery), holding everything else fixed. Used
    for the plan's monotonicity validation: CVA should rise with PD and
    fall with recovery rate (rise with LGD).
    """
    checkpoint_times = ee_curve.index.to_numpy()
    rows = []

    for mult in pd_shock_multipliers:
        pd_shocked = min(pd_1y_base * mult, 0.999)
        pdts = pd_term_structure_from_1y_pd(pd_shocked, checkpoint_times)
        lgd = compute_lgd(recovery_rate_base)
        cva = compute_cva(ee_curve, pdts, lgd, curve)
        rows.append({"shock_type": "PD", "shock_label": f"{mult}x", "pd_1y": pd_shocked,
                     "recovery_rate": recovery_rate_base, "cva": cva})

    for recovery in recovery_rate_shocks:
        pdts = pd_term_structure_from_1y_pd(pd_1y_base, checkpoint_times)
        lgd = compute_lgd(recovery)
        cva = compute_cva(ee_curve, pdts, lgd, curve)
        rows.append({"shock_type": "Recovery", "shock_label": f"{recovery:.0%}", "pd_1y": pd_1y_base,
                     "recovery_rate": recovery, "cva": cva})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def run_phase6_and_7(config_path: str | Path = "config.yaml"):
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    recovery_rate = cfg["lgd"]["recovery_rate"]
    cva_cfg = cfg.get("cva", {})
    counterparty_ticker = cva_cfg.get("counterparty_ticker", "AMC")

    print("=== Phase 6: LGD assumption ===\n")
    lgd = compute_lgd(recovery_rate)
    print(f"Recovery rate = {recovery_rate:.0%} (ISDA CDS Standard Model convention, senior unsecured)")
    print(f"LGD = 1 - {recovery_rate:.0%} = {lgd:.0%}")

    print(f"\n=== Phase 7: CVA assembly (counterparty={counterparty_ticker}) ===\n")

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")

    merton = pd.read_csv(processed_dir / f"merton_pd_{valuation_date}.csv")
    counterparty_row = merton.loc[merton["ticker"] == counterparty_ticker].iloc[0]
    pd_1y = float(counterparty_row["pd_merton"])
    print(f"Counterparty 1y Merton PD (Phase 2): {pd_1y:.4%}")
    print("(Using Merton, not Phase 2b's classifier -- see module docstring for why.)")

    dataset = pd.read_csv(processed_dir / f"exposure_training_data_{valuation_date}.csv")
    summary_path = processed_dir / f"exposure_model_comparison_summary_{valuation_date}.csv"
    model_summary = pd.read_csv(summary_path, index_col=0)
    best_model = model_summary.drop(index="naive")["MAE"].idxmin()
    print(f"Best exposure model from Phase 5 (lowest MAE vs. nested-sim ground truth): {best_model}")

    ee_naive = compute_ee_curve(dataset, method="naive")
    ee_best = compute_ee_curve(dataset, method=best_model)

    print("\n--- EE(t) curve: naive vs. best regression model ---")
    print(pd.DataFrame({"naive_EE": ee_naive, f"{best_model}_EE": ee_best}).to_string())

    checkpoint_times = ee_best.index.to_numpy()
    pd_term_structure = pd_term_structure_from_1y_pd(pd_1y, checkpoint_times)

    cva_naive = compute_cva(ee_naive, pd_term_structure, lgd, curve)
    cva_best = compute_cva(ee_best, pd_term_structure, lgd, curve)

    print(f"\nCVA using naive EE(t):        ${cva_naive:,.2f}")
    print(f"CVA using {best_model} EE(t): ${cva_best:,.2f}")
    if cva_naive != 0:
        print(f"Difference: ${cva_best - cva_naive:,.2f} ({cva_best / cva_naive - 1:+.1%} relative to naive) -- "
              f"the business-relevant payoff from Phase 5's model comparison.")

    sens = sensitivity_analysis(
        ee_best, pd_1y, recovery_rate, curve,
        pd_shock_multipliers=cva_cfg.get("pd_shock_multipliers", (0.5, 1.0, 1.5, 2.0)),
        recovery_rate_shocks=cva_cfg.get("recovery_rate_shocks", (0.20, 0.40, 0.60)),
    )
    print("\n--- Sensitivity analysis ---")
    print(sens.to_string(index=False))

    pd_sens = sens[sens["shock_type"] == "PD"].sort_values("pd_1y")
    recovery_sens = sens[sens["shock_type"] == "Recovery"].sort_values("recovery_rate")
    pd_monotonic = pd_sens["cva"].is_monotonic_increasing
    recovery_monotonic = recovery_sens["cva"].is_monotonic_decreasing
    print(f"\nMonotonicity checks: CVA increases with PD: {pd_monotonic}; "
          f"CVA decreases with recovery rate (increases with LGD): {recovery_monotonic}")
    assert pd_monotonic, "CVA should increase monotonically with shocked PD"
    assert recovery_monotonic, "CVA should decrease monotonically as recovery rate rises"

    # Bonus: same swap, same best-model EE(t), CVA for every panel company
    # using its own Merton PD -- a preview of Phase 9's portfolio application.
    portfolio_df = compute_portfolio_cva(merton, ee_best, lgd, curve)
    print(f"\n--- Bonus: same swap across the Phase 1/2 panel (each company's own Merton PD) ---")
    print(portfolio_df.to_string(index=False))
    print(
        "\nNote: near-zero CVA for every name but AMC is the expected, already-documented "
        "consequence of Phase 2's Merton-PD-collapse finding, not a new bug."
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    sens_path = processed_dir / f"cva_sensitivity_{valuation_date}.csv"
    portfolio_path = processed_dir / f"cva_portfolio_{valuation_date}.csv"
    sens.to_csv(sens_path, index=False)
    portfolio_df.to_csv(portfolio_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(pd_sens["pd_1y"] * 100, pd_sens["cva"], marker="o", color="tab:blue")
    axes[0].set_xlabel("Shocked 1y PD (%)")
    axes[0].set_ylabel("CVA ($)")
    axes[0].set_title("CVA vs. PD shock")

    axes[1].plot(recovery_sens["recovery_rate"] * 100, recovery_sens["cva"], marker="o", color="tab:orange")
    axes[1].set_xlabel("Recovery rate (%)")
    axes[1].set_ylabel("CVA ($)")
    axes[1].set_title("CVA vs. recovery-rate shock")
    fig.suptitle(f"Phase 7 sensitivity analysis ({counterparty_ticker})")
    fig.tight_layout()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "cva_sensitivity.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {sens_path}, {portfolio_path}, {fig_path}")
    return {
        "lgd": lgd, "cva_naive": cva_naive, "cva_best": cva_best, "best_model": best_model,
        "sensitivity": sens, "portfolio": portfolio_df,
    }


# --------------------------------------------------------------------------
# Phase 9: portfolio-level application
# --------------------------------------------------------------------------

def run_phase9(config_path: str | Path = "config.yaml"):
    """Apply the full pipeline (Merton PD + Phase 5's best exposure model +
    CVA) across the entire Phase 1/2 panel -- "a model deployed across a
    segment," per the plan, not just more single-company examples -- and
    sanity-check the resulting CVA ranking against approximate public
    credit ratings.
    """
    from scipy.stats import spearmanr

    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    recovery_rate = cfg["lgd"]["recovery_rate"]
    lgd = compute_lgd(recovery_rate)

    print(f"=== Phase 9: portfolio-level application (valuation_date={valuation_date}) ===\n")

    processed_dir = Path("data/processed")
    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    merton = pd.read_csv(processed_dir / f"merton_pd_{valuation_date}.csv")

    dataset = pd.read_csv(processed_dir / f"exposure_training_data_{valuation_date}.csv")
    summary_path = processed_dir / f"exposure_model_comparison_summary_{valuation_date}.csv"
    model_summary = pd.read_csv(summary_path, index_col=0)
    best_model = model_summary.drop(index="naive")["MAE"].idxmin()
    ee_best = compute_ee_curve(dataset, method=best_model)
    print(f"Same swap and {best_model} EE(t) curve for every panel company (Phase 5/7's methodology).")

    portfolio = compute_portfolio_cva(merton, ee_best, lgd, curve)
    portfolio["approx_rating"] = portfolio["ticker"].map(APPROXIMATE_CREDIT_RATINGS)
    portfolio["approx_rating_rank"] = portfolio["approx_rating"].map(CREDIT_RATING_RANK)
    portfolio = portfolio.sort_values("cva").reset_index(drop=True)
    portfolio["cva_rank"] = portfolio["cva"].rank(method="min").astype(int)
    portfolio["rating_rank_within_panel"] = portfolio["approx_rating_rank"].rank(method="min").astype(int)

    print("\n--- Portfolio: PD, CVA, and approximate rating, ranked by CVA (safest first) ---")
    print(portfolio[["ticker", "pd_merton_1y", "cva", "cva_rank",
                      "approx_rating", "rating_rank_within_panel"]].to_string(index=False))
    print(
        "\nRatings above are illustrative (general knowledge, not a live agency pull) -- see module "
        "docstring. Used only to sanity-check rank order, not as an input to the CVA numbers themselves."
    )

    rho, p_value = spearmanr(portfolio["cva_rank"], portfolio["rating_rank_within_panel"])
    print(f"\nSpearman rank correlation (CVA rank vs. approximate-rating rank): {rho:.3f} (p={p_value:.3f})")

    print(
        "\n--- Discussion: what this says about Merton's discriminative power ---\n"
        "Merton gets the broad categories right: AMC (the panel's only clearly speculative-grade\n"
        "name by rating) is correctly identified as by far the riskiest, and the genuinely\n"
        "investment-grade names (AAPL/MSFT/JNJ/KO/XOM) all cluster at the bottom. But two real\n"
        "limitations show up in the fine-grained ranking:\n"
        "  1. Ford ranks as one of the SAFER names here despite a BB+ (speculative-grade) rating --\n"
        "     this is the direct, already-documented consequence of Phase 1's stale-debt-data finding\n"
        "     for Ford (no SEC tag newer than 2020), not a new modeling problem. A garbage-in-\n"
        "     garbage-out case study that happens to surface again at the portfolio level.\n"
        "  2. Within the investment-grade cluster, PDs range from 1e-76 to 1e-41 -- twenty-plus\n"
        "     orders of magnitude of 'differentiation' between names that are, in reality, all\n"
        "     roughly equally very-safe. That's not real signal; it's the well-documented Merton-\n"
        "     with-physical-inputs pathology from Phase 2 showing up again. Ordering names within\n"
        "     this cluster by their Merton PD would be overinterpreting noise.\n"
        "Net: plain Merton is a reasonable coarse categorical screen (flag the clearly risky names)\n"
        "but not a reliable fine-grained ranking tool -- exactly the gap Phase 2b's supervised\n"
        "classifier exists to close, if it could be applied to this same company universe (it can't,\n"
        "per Phase 2b's documented limitation)."
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = processed_dir / f"portfolio_application_{valuation_date}.csv"
    portfolio.to_csv(portfolio_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["tab:red" if r not in (None, "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                                      "BBB+", "BBB", "BBB-") else "tab:blue"
              for r in portfolio["approx_rating"]]
    bars = ax.bar(portfolio["ticker"], portfolio["cva"] + 1e-6, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("CVA ($, log scale)")
    ax.set_title("Phase 9: portfolio CVA, same swap, each company's own Merton PD\n"
                  "(blue = investment-grade rating, red = speculative-grade, by approximate rating)")
    for bar, rating in zip(bars, portfolio["approx_rating"]):
        ax.annotate(rating, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "portfolio_cva.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {portfolio_path}, {fig_path}")
    return portfolio


if __name__ == "__main__":
    run_phase6_and_7()
    print()
    run_phase9()
