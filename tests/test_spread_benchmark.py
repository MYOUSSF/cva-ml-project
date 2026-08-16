"""Unit tests for the external spread-benchmarking extension.

Key checks, offline, no network calls:
  - implied_spread_bps matches a hand-computed value.
  - rating_to_oas_bucket maps every rating actually used in
    APPROXIMATE_CREDIT_RATINGS to a bucket present in FRED_OAS_SERIES (a
    typo'd bucket here would silently drop a company from the benchmark
    table, the same class of bug test_cva.py guards against for the
    ratings-rank table).
  - build_benchmark_table: correct join/skip behavior for missing rating
    or missing PD, and diff_bps sign/magnitude matches a hand calculation.
"""

import numpy as np
import pandas as pd
import pytest

from src.cva import APPROXIMATE_CREDIT_RATINGS
from src.spread_benchmark import (
    FRED_OAS_SERIES,
    build_benchmark_table,
    implied_spread_bps,
    rating_to_oas_bucket,
)


def test_implied_spread_bps_matches_hand_calculation():
    assert implied_spread_bps(pd_1y=0.02, lgd=0.6) == pytest.approx(0.02 * 0.6 * 10_000)
    assert implied_spread_bps(pd_1y=0.0, lgd=0.6) == pytest.approx(0.0)


@pytest.mark.parametrize("rating,expected_bucket", [
    ("AAA", "AAA"), ("AA+", "AA"), ("AA-", "AA"), ("A+", "A"), ("A-", "A"),
    ("BBB+", "BBB"), ("BBB-", "BBB"), ("BB+", "BB"), ("B+", "B"), ("CCC+", "CCC_OR_LOWER"),
])
def test_rating_to_oas_bucket_mapping(rating, expected_bucket):
    assert rating_to_oas_bucket(rating) == expected_bucket


def test_rating_to_oas_bucket_raises_on_unknown_rating():
    with pytest.raises(ValueError):
        rating_to_oas_bucket("NR")


def test_every_approximate_credit_rating_maps_to_a_known_fred_series():
    # Guards against a typo'd bucket silently dropping a company out of
    # build_benchmark_table (the same failure mode test_cva.py checks for
    # the CREDIT_RATING_RANK table).
    for ticker, rating in APPROXIMATE_CREDIT_RATINGS.items():
        bucket = rating_to_oas_bucket(rating)
        assert bucket in FRED_OAS_SERIES, f"{ticker}'s bucket {bucket!r} missing from FRED_OAS_SERIES"


def test_build_benchmark_table_skips_missing_rating_and_missing_pd():
    merton_df = pd.DataFrame({
        "ticker": ["AAPL", "UNKNOWN_TICKER", "AMC"],
        "pd_merton": [1e-50, 0.05, np.nan],
    })
    oas_lookup = {"AA": (50.0, pd.Timestamp("2026-08-14")), "CCC_OR_LOWER": (900.0, pd.Timestamp("2026-08-14"))}

    table = build_benchmark_table(merton_df, lgd=0.6, oas_lookup=oas_lookup)

    # UNKNOWN_TICKER has no rating -> skipped. AMC has nan PD -> skipped.
    assert list(table["ticker"]) == ["AAPL"]


def test_build_benchmark_table_diff_bps_matches_hand_calculation():
    merton_df = pd.DataFrame({"ticker": ["AMC"], "pd_merton": [0.05]})
    oas_lookup = {"CCC_OR_LOWER": (900.0, pd.Timestamp("2026-08-14"))}

    table = build_benchmark_table(merton_df, lgd=0.6, oas_lookup=oas_lookup)

    expected_model_bps = 0.05 * 0.6 * 10_000
    row = table.iloc[0]
    assert row["model_implied_spread_bps"] == pytest.approx(expected_model_bps)
    assert row["diff_bps"] == pytest.approx(expected_model_bps - 900.0)


def test_build_benchmark_table_skips_bucket_without_oas_lookup():
    merton_df = pd.DataFrame({"ticker": ["AAPL"], "pd_merton": [1e-50]})
    table = build_benchmark_table(merton_df, lgd=0.6, oas_lookup={})
    assert table.empty
