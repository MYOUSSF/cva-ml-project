"""Unit tests for Phase 1 core functions.

These exercise the pure/synthetic-data logic (curve fitting, missing-data
audit, SEC fact extraction) with no network calls, so they run offline and
in CI. The live fetch_* functions (fetch_fred_series, fetch_equity_history,
fetch_sec_company_facts, ...) hit real external APIs and are intentionally
left untested here.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    audit_missing_data,
    check_panel_sanity,
    extract_debt_figures,
    validate_curve_roundtrip,
)


def _synthetic_curve() -> pd.DataFrame:
    maturities = np.array([1 / 365, 1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    # A smooth, monotonically-increasing-ish synthetic curve -- shape doesn't
    # matter for a round-trip check, only that CubicSpline can fit it exactly
    # at its own knot points.
    yields = 3.5 + 0.4 * np.log1p(maturities)
    return pd.DataFrame({
        "series_id": [f"S{i}" for i in range(len(maturities))],
        "maturity_years": maturities,
        "obs_date": pd.Timestamp("2026-08-14"),
        "yield_pct": yields,
    })


def test_validate_curve_roundtrip_reproduces_input_yields():
    curve = _synthetic_curve()
    out = validate_curve_roundtrip(curve)

    assert out["roundtrip_ok"].all()
    np.testing.assert_allclose(out["fitted_yield_pct"], out["yield_pct"], atol=1e-6)


def test_validate_curve_roundtrip_raises_on_duplicate_maturities():
    # CubicSpline interpolates exactly at its own knots, so the only way the
    # round-trip check can fail is a malformed curve going in -- e.g. a
    # config bug that lists the same maturity twice, which breaks the
    # strictly-increasing-x requirement of the spline fit.
    curve = _synthetic_curve()
    dup = curve.copy()
    dup.loc[1, "maturity_years"] = dup.loc[0, "maturity_years"]

    with pytest.raises(ValueError):
        validate_curve_roundtrip(dup)


def test_audit_missing_data_counts_and_percentages():
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": [None, None, None, None],
        "c": [1, 2, 3, 4],
    })
    audit = audit_missing_data(df)

    assert audit.loc["a", "n_missing"] == 1
    assert audit.loc["a", "pct_missing"] == 25.0
    assert audit.loc["b", "n_missing"] == 4
    assert audit.loc["b", "pct_missing"] == 100.0
    assert audit.loc["c", "n_missing"] == 0
    # sorted descending by pct_missing
    assert list(audit.index)[0] == "b"


def _fake_company_facts(long_term_debt=100.0, short_term_debt=20.0, filed="2026-01-30"):
    return {
        "facts": {
            "us-gaap": {
                "LongTermDebtNoncurrent": {
                    "units": {"USD": [
                        {"end": "2025-12-31", "val": long_term_debt, "filed": filed, "form": "10-Q"},
                    ]}
                },
                "LongTermDebtCurrent": {
                    "units": {"USD": [
                        {"end": "2025-12-31", "val": short_term_debt, "filed": filed, "form": "10-Q"},
                    ]}
                },
            }
        }
    }


def test_extract_debt_figures_uses_short_plus_half_long_convention():
    facts = _fake_company_facts(long_term_debt=100.0, short_term_debt=20.0)
    debt = extract_debt_figures(facts, valuation_date="2026-08-15")

    assert debt["long_term_debt"] == 100.0
    assert debt["short_term_debt"] == 20.0
    assert debt["total_debt_face_value"] == pytest.approx(20.0 + 0.5 * 100.0)


def test_extract_debt_figures_excludes_facts_filed_after_valuation_date():
    # Filed in the future relative to the valuation date -- must not leak in.
    facts = _fake_company_facts(filed="2026-09-01")
    debt = extract_debt_figures(facts, valuation_date="2026-08-15")

    assert debt["long_term_debt"] is None
    assert debt["short_term_debt"] is None
    assert debt["total_debt_face_value"] is None


def test_extract_debt_figures_missing_tags_returns_none():
    debt = extract_debt_figures({"facts": {"us-gaap": {}}}, valuation_date="2026-08-15")

    assert debt["long_term_debt"] is None
    assert debt["short_term_debt"] is None
    assert debt["total_debt_face_value"] is None


def test_extract_debt_figures_prefers_freshest_across_candidate_tags():
    # Real-world case (AT&T): LongTermDebtNoncurrent has only stale data,
    # but a different candidate tag has current data. The freshest value
    # across all candidates should win, not just the first tag with any data.
    facts = {
        "facts": {
            "us-gaap": {
                "LongTermDebtNoncurrent": {
                    "units": {"USD": [
                        {"end": "2012-09-30", "val": 999.0, "filed": "2012-11-01", "form": "10-Q"},
                    ]}
                },
                "LongTermDebtAndCapitalLeaseObligations": {
                    "units": {"USD": [
                        {"end": "2026-06-30", "val": 134.0, "filed": "2026-07-22", "form": "10-Q"},
                    ]}
                },
            }
        }
    }
    debt = extract_debt_figures(facts, valuation_date="2026-08-15")

    assert debt["long_term_debt"] == 134.0
    assert debt["long_term_debt_tag"] == "LongTermDebtAndCapitalLeaseObligations"


def _panel_row(**overrides):
    row = {
        "ticker": "XYZ",
        "equity_value": 1_000.0,
        "equity_vol": 0.3,
        "total_debt_face_value": 100.0,
        "long_term_debt_asof": pd.Timestamp("2026-06-30"),
        "short_term_debt_asof": pd.Timestamp("2026-06-30"),
    }
    row.update(overrides)
    return row


def test_check_panel_sanity_flags_stale_debt_data():
    panel = pd.DataFrame([_panel_row(long_term_debt_asof=pd.Timestamp("2012-09-30"))])
    sanity = check_panel_sanity(panel, valuation_date="2026-08-15")

    assert sanity.loc[0, "n_flags"] == 1
    assert any(f.startswith("long_term_debt_stale") for f in sanity.loc[0, "flags"])


def test_check_panel_sanity_flags_missing_debt_and_bad_vol():
    panel = pd.DataFrame([_panel_row(
        total_debt_face_value=None,
        equity_vol=5.0,
        long_term_debt_asof=pd.NaT,
        short_term_debt_asof=pd.NaT,
    )])
    sanity = check_panel_sanity(panel, valuation_date="2026-08-15")

    assert "debt_missing" in sanity.loc[0, "flags"]
    assert "equity_vol_out_of_plausible_range" in sanity.loc[0, "flags"]


def test_check_panel_sanity_no_flags_for_clean_row():
    panel = pd.DataFrame([_panel_row()])
    sanity = check_panel_sanity(panel, valuation_date="2026-08-15")

    assert sanity.loc[0, "n_flags"] == 0
