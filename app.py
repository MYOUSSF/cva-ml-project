"""Streamlit demo (Phase 10, optional stretch).

Interactive, reviewer-facing view over Phases 1/2/4/5/6/7/9's already-saved
outputs: pick a counterparty from the Phase 1 panel and see its Merton PD,
EE(t) curve (naive vs. Phase 5's best regression model), and live CVA --
with adjustable PD-shock and recovery-rate sliders, plus a portfolio-wide
comparison across the whole panel.

This is a VIEW over the pipeline, not a second implementation of it: every
number here is computed by calling straight into src.cva / src.exposure_models,
the same functions the batch phase scripts use and the test suite checks.

Run with:
    streamlit run app.py

Requires Phases 1, 2, 3, 4, 5 to have been run at least once (so
data/processed/ has the files this app reads) -- Phase 6/7/9's outputs
aren't required since this app recomputes CVA live from the sliders.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.cva import (
    APPROXIMATE_CREDIT_RATINGS,
    compute_cva,
    compute_lgd,
    compute_portfolio_cva,
    pd_term_structure_from_1y_pd,
    sensitivity_analysis,
)
from src.data_pipeline import load_config
from src.exposure_models import compute_ee_curve

st.set_page_config(page_title="CVA Explorer", layout="wide")


@st.cache_data
def load_pipeline_outputs(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    processed_dir = Path("data/processed")

    curve = pd.read_csv(processed_dir / f"fred_curve_{valuation_date}.csv")
    merton = pd.read_csv(processed_dir / f"merton_pd_{valuation_date}.csv")
    dataset = pd.read_csv(processed_dir / f"exposure_training_data_{valuation_date}.csv")
    model_summary = pd.read_csv(
        processed_dir / f"exposure_model_comparison_summary_{valuation_date}.csv", index_col=0
    )
    return cfg, valuation_date, curve, merton, dataset, model_summary


@st.cache_data
def load_ee_curves(dataset: pd.DataFrame, best_model: str):
    return compute_ee_curve(dataset, method="naive"), compute_ee_curve(dataset, method=best_model)


def missing_data_message() -> str:
    return (
        "No processed data found under `data/processed/`. Run the pipeline first:\n\n"
        "```\n"
        "python -m src.data_pipeline\n"
        "python -m src.pd_model\n"
        "python -m src.ratemodel\n"
        "python -m src.exposure_models\n"
        "```"
    )


def main():
    st.title("CVA Explorer")
    st.caption(
        "Interactive view of this project's Phase 2 (Merton PD), Phase 5 (exposure model "
        "comparison), and Phase 6/7/9 (CVA assembly + portfolio application) outputs. "
        "See the README for the full methodology and an honest limitations section -- "
        "the numbers here inherit every caveat documented there."
    )

    processed_dir = Path("data/processed")
    if not processed_dir.exists() or not list(processed_dir.glob("fred_curve_*.csv")):
        st.error(missing_data_message())
        return

    try:
        cfg, valuation_date, curve, merton, dataset, model_summary = load_pipeline_outputs()
    except FileNotFoundError:
        st.error(missing_data_message())
        return

    best_model = model_summary.drop(index="naive")["MAE"].idxmin()
    ee_naive, ee_best = load_ee_curves(dataset, best_model)

    swap_cfg = cfg["swap"]
    recovery_rate_default = float(cfg["lgd"]["recovery_rate"])

    st.sidebar.header("Controls")

    tickers = merton["ticker"].tolist()
    default_ticker = cfg.get("cva", {}).get("counterparty_ticker", tickers[0])
    ticker = st.sidebar.selectbox(
        "Counterparty", tickers,
        index=tickers.index(default_ticker) if default_ticker in tickers else 0,
    )

    exposure_label = f"Best regression model ({best_model})"
    exposure_choice = st.sidebar.radio("Exposure model EE(t)", [exposure_label, "Naive baseline"], index=0)
    ee_curve = ee_best if exposure_choice == exposure_label else ee_naive

    recovery_rate = st.sidebar.slider("Recovery rate", 0.0, 0.90, recovery_rate_default, 0.05)
    pd_shock_mult = st.sidebar.slider("PD shock (multiplier on base 1y Merton PD)", 0.25, 3.0, 1.0, 0.25)

    st.sidebar.caption(
        "Naive floors each raw noisy simulated path, then averages. The best model floors a "
        "smoothed, fitted curve, then averages -- Phase 5 found naive systematically "
        "over-estimates exposure. See README Phase 5."
    )

    lgd = compute_lgd(recovery_rate)
    row = merton.loc[merton["ticker"] == ticker].iloc[0]
    pd_1y_base = float(row["pd_merton"])
    pd_1y_shocked = min(pd_1y_base * pd_shock_mult, 0.999)
    rating = APPROXIMATE_CREDIT_RATINGS.get(ticker, "n/a")

    checkpoint_times = ee_curve.index.to_numpy()
    pd_term_structure = pd_term_structure_from_1y_pd(pd_1y_shocked, checkpoint_times)
    cva = compute_cva(ee_curve, pd_term_structure, lgd, curve)

    def format_pd(p: float) -> str:
        return f"{p:.4%}" if p >= 1e-4 else f"{p:.2e}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1y Merton PD (base)", format_pd(pd_1y_base))
    col2.metric("1y PD (shocked)", format_pd(pd_1y_shocked),
                delta=f"{pd_shock_mult}x" if pd_shock_mult != 1.0 else None)
    col3.metric("Approx. rating (illustrative)", rating)
    col4.metric("CVA", f"${cva:,.0f}")

    if pd_1y_base < 1e-10:
        st.info(
            f"{ticker}'s Merton PD is essentially zero ({pd_1y_base:.2e}) -- a well-documented "
            "property of plain Merton with physical-measure inputs for very safe firms, not a bug. "
            "See README Phase 2. CVA will be ~$0 regardless of the exposure model or recovery slider."
        )

    st.subheader("Expected exposure profile EE(t)")
    ee_df = pd.DataFrame({"naive": ee_naive, best_model: ee_best})
    ee_df.index.name = "checkpoint (years)"
    st.line_chart(ee_df)

    st.subheader("CVA sensitivity (around the current recovery-rate slider, base PD)")
    sens = sensitivity_analysis(ee_curve, pd_1y_base, recovery_rate, curve)
    col_a, col_b = st.columns(2)
    with col_a:
        pd_sens = sens[sens["shock_type"] == "PD"].sort_values("pd_1y").set_index("pd_1y")["cva"]
        st.line_chart(pd_sens)
        st.caption("CVA vs. shocked 1y PD (x-axis: shocked PD)")
    with col_b:
        recovery_sens = sens[sens["shock_type"] == "Recovery"].sort_values("recovery_rate").set_index("recovery_rate")["cva"]
        st.line_chart(recovery_sens)
        st.caption("CVA vs. recovery rate")

    st.subheader("Portfolio: same swap, same exposure model, across the full panel")
    portfolio = compute_portfolio_cva(merton, ee_curve, lgd, curve)
    portfolio["approx_rating"] = portfolio["ticker"].map(APPROXIMATE_CREDIT_RATINGS)
    st.bar_chart(portfolio.set_index("ticker")["cva"])
    st.dataframe(
        portfolio[["ticker", "pd_merton_1y", "cva", "approx_rating"]],
        width="stretch", hide_index=True,
        column_config={
            "pd_merton_1y": st.column_config.NumberColumn("1y Merton PD", format="%.4f"),
            "cva": st.column_config.NumberColumn("CVA", format="$%.2f"),
        },
    )
    st.caption(
        "Ratings are illustrative (general knowledge, not a live agency pull -- see README "
        "Limitations). CVA near $0 for every investment-grade name is the expected consequence "
        "of Merton's documented PD-collapse behavior for very safe firms, not a bug -- and Ford's "
        "position here reflects a known stale-debt-data issue from Phase 1, not a modeling error."
    )

    st.divider()
    st.caption(
        f"Valuation date {valuation_date}. Swap: notional ${swap_cfg['notional']:,.0f}, "
        f"fixed rate {swap_cfg['fixed_rate']:.4%}, {swap_cfg['tenor_years']}y {swap_cfg['payment_frequency']}, "
        f"LGD {lgd:.0%}."
    )


if __name__ == "__main__":
    main()
