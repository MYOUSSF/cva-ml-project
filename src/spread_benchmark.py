"""Extension -- external benchmarking vs. market credit spreads.

A cheap, credible sanity check comparing Phase 2's model-implied PDs to
what the market actually charges: pulls FRED's free ICE BofA rating-bucket
corporate OAS (option-adjusted spread) series and compares each panel
company's implied spread (PD x LGD, a standard rough approximation --
ignoring risk premium and recovery timing) against the market OAS for the
rating bucket its illustrative rating falls into.

FRED series used (ICE BofA US Corporate / High Yield indices by rating,
public, no API key needed -- reuses data_pipeline.fetch_fred_series):
    AAA   BAMLC0A1CAAA
    AA    BAMLC0A2CAA
    A     BAMLC0A3CA
    BBB   BAMLC0A4CBBB
    BB    BAMLH0A1HYBB
    B     BAMLH0A2HYB
    CCC and lower   BAMLH0A3HYC

Run directly against Phase 1/2's saved outputs:
    python -m src.spread_benchmark
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cva import APPROXIMATE_CREDIT_RATINGS, compute_lgd
from src.data_pipeline import fetch_fred_series, load_config

FRED_OAS_SERIES = {
    "AAA": "BAMLC0A1CAAA",
    "AA": "BAMLC0A2CAA",
    "A": "BAMLC0A3CA",
    "BBB": "BAMLC0A4CBBB",
    "BB": "BAMLH0A1HYBB",
    "B": "BAMLH0A2HYB",
    "CCC_OR_LOWER": "BAMLH0A3HYC",
}


def rating_to_oas_bucket(rating: str) -> str:
    """Map a full-notch approximate rating (e.g. 'AA+', 'BBB-') to the
    coarser bucket FRED's OAS index families are published at.
    """
    base = rating.rstrip("+-")
    if base in ("AAA", "AA", "A", "BBB", "BB", "B"):
        return base
    if base == "CCC":
        return "CCC_OR_LOWER"
    raise ValueError(f"No OAS bucket mapping for rating {rating!r}")


def implied_spread_bps(pd_1y: float, lgd: float) -> float:
    """spread ~= PD * LGD, in basis points -- a standard rough approximation
    (ignores risk premium and recovery timing), used here only as an
    external sanity check, not as a CVA input.
    """
    return pd_1y * lgd * 10_000.0


def fetch_oas_asof(series_id: str, valuation_date: str) -> tuple[float, pd.Timestamp]:
    """Most recent OAS observation (in bps) on or before valuation_date,
    same "most recent on or before" convention `fetch_fred_curve` uses.
    Series values are published in percentage points, hence *100 for bps.
    """
    series = fetch_fred_series(series_id)
    as_of = series[series.index <= pd.Timestamp(valuation_date)]
    if as_of.empty:
        raise ValueError(f"No {series_id} observations on or before {valuation_date}")
    return float(as_of.iloc[-1]) * 100.0, as_of.index[-1]


def build_benchmark_table(merton_df: pd.DataFrame, lgd: float, oas_lookup: dict[str, tuple[float, pd.Timestamp]]) -> pd.DataFrame:
    """Comparison table: model-implied spread vs. market OAS, per company.
    Rows with an unmapped/missing rating or a null PD are skipped.
    """
    rows = []
    for _, row in merton_df.iterrows():
        ticker = row["ticker"]
        rating = APPROXIMATE_CREDIT_RATINGS.get(ticker)
        pd_1y = row["pd_merton"]
        if rating is None or pd.isna(pd_1y):
            continue
        bucket = rating_to_oas_bucket(rating)
        if bucket not in oas_lookup:
            continue
        market_oas_bps, obs_date = oas_lookup[bucket]
        model_bps = implied_spread_bps(float(pd_1y), lgd)
        rows.append({
            "ticker": ticker, "approx_rating": rating, "oas_bucket": bucket,
            "pd_merton_1y": pd_1y, "model_implied_spread_bps": model_bps,
            "market_oas_bps": market_oas_bps, "market_oas_obs_date": obs_date,
            "diff_bps": model_bps - market_oas_bps,
        })
    columns = ["ticker", "approx_rating", "oas_bucket", "pd_merton_1y", "model_implied_spread_bps",
               "market_oas_bps", "market_oas_obs_date", "diff_bps"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("model_implied_spread_bps")


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def run_spread_benchmark(config_path: str | Path = "config.yaml"):
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    recovery_rate = cfg["lgd"]["recovery_rate"]
    lgd = compute_lgd(recovery_rate)

    print(f"=== Extension: external benchmarking vs. FRED rating-bucket OAS (valuation_date={valuation_date}) ===\n")

    processed_dir = Path("data/processed")
    merton = pd.read_csv(processed_dir / f"merton_pd_{valuation_date}.csv")

    needed_buckets = {
        rating_to_oas_bucket(APPROXIMATE_CREDIT_RATINGS[t])
        for t in merton["ticker"] if t in APPROXIMATE_CREDIT_RATINGS
    }
    oas_lookup = {}
    for bucket in needed_buckets:
        series_id = FRED_OAS_SERIES[bucket]
        bps, obs_date = fetch_oas_asof(series_id, valuation_date)
        oas_lookup[bucket] = (bps, obs_date)
        print(f"{bucket:15s} ({series_id}): {bps:,.0f} bps as of {obs_date.date()}")

    table = build_benchmark_table(merton, lgd, oas_lookup)
    print("\n--- Model-implied spread (PD x LGD) vs. market OAS by rating bucket ---")
    print(table[["ticker", "approx_rating", "oas_bucket", "pd_merton_1y",
                 "model_implied_spread_bps", "market_oas_bps", "diff_bps"]].to_string(index=False))
    print(
        "\nNote: investment-grade names collapse to near-0 implied spread here, the direct downstream "
        "consequence of Phase 2's documented Merton-PD-collapse finding (see README) -- not a new bug, "
        "and exactly what this external check is useful for catching: it makes the magnitude gap between "
        "model and market visible and quantified, rather than left as a qualitative caveat."
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"spread_benchmark_{valuation_date}.csv"
    table.to_csv(out_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(table))
    width = 0.35
    ax.bar(x - width / 2, table["model_implied_spread_bps"] + 1e-6, width, label="Model-implied (PD x LGD)", color="tab:blue")
    ax.bar(x + width / 2, table["market_oas_bps"], width, label="Market OAS (rating bucket)", color="tab:orange")
    ax.set_yscale("log")
    ax.set_ylabel("Spread (bps, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["ticker"])
    ax.set_title("Model-implied vs. market credit spread, by rating bucket")
    ax.legend()
    fig.tight_layout()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = outputs_dir / "spread_benchmark.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}, {fig_path}")
    return table


if __name__ == "__main__":
    run_spread_benchmark()
