"""Phase 1 -- Data engineering.

Reusable ingestion pipeline for:
  - The Treasury/SOFR curve from FRED (via the public `fredgraph.csv` export,
    which needs no API key) for a chosen valuation date.
  - Equity price history and a point-in-time market-cap/shares snapshot from
    yfinance, for a panel of companies.
  - Balance-sheet debt figures from the SEC EDGAR company-facts API
    (https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json), with
    CIK lookup via https://www.sec.gov/files/company_tickers.json.

Design goal: functions callable from notebooks, tests, or the config-driven
entry point (`run_phase1`, driven by config.yaml) -- not one-off notebook
cells.

Run directly to execute the full Phase 1 pipeline:
    python -m src.data_pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests
import yfinance as yf

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Taiwanese Bankruptcy Prediction dataset (Taiwan Economic Journal,
# 1999-2009; ~6,800 companies, ~95 accounting-ratio features, binary
# 'Bankrupt?' label) -- the public UCI mirror of the dataset referenced in
# the plan's Phase 2b (commonly distributed via Kaggle). No API key needed.
BANKRUPTCY_DATASET_URL = "https://archive.ics.uci.edu/static/public/572/data.csv"

DEFAULT_FRED_SERIES = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2",
    "DGS5", "DGS10", "DGS20", "DGS30", "SOFR",
]

# Approximate maturity in years for each FRED series, used to build the curve
# and as the x-axis for the round-trip fit check. SOFR is an overnight rate,
# treated here as the ~0 end of the curve.
FRED_MATURITY_YEARS = {
    "SOFR": 1 / 365,
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS5": 5.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

# XBRL tag candidates for the Merton debt convention (short-term debt +
# 0.5 * long-term debt). Different filers use different tags -- and the same
# filer can switch tags over time (e.g. AT&T reported under
# LongTermDebtNoncurrent through 2012, then moved to
# LongTermDebtAndCapitalLeaseObligations) -- so all candidates are checked
# and the single most recent balance-sheet date wins, not just the first
# candidate with any data.
DEBT_TAG_CANDIDATES = {
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
    ],
    "short_term_debt": [
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "UnsecuredDebtCurrent",
    ],
}

# If the freshest debt fact we can find is older than this relative to the
# valuation date, treat it as unreliable for a point-in-time Merton input
# rather than silently using it.
DEBT_STALENESS_DAYS = 548  # ~1.5 years


# --------------------------------------------------------------------------
# FRED: Treasury/SOFR curve
# --------------------------------------------------------------------------

def fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch a single FRED series as a date-indexed pd.Series.

    Uses the public `fredgraph.csv` export, which does not require an API
    key (unlike the FRED REST API).
    """
    df = pd.read_csv(f"{FRED_CSV_URL}?id={series_id}", parse_dates=["observation_date"])
    series = df.set_index("observation_date")[series_id]
    return pd.to_numeric(series, errors="coerce").dropna()


def fetch_fred_curve(valuation_date: str, series_ids: Sequence[str] = DEFAULT_FRED_SERIES) -> pd.DataFrame:
    """Build a snapshot yield curve as of `valuation_date`.

    Treasury/SOFR series aren't published every calendar day (weekends,
    holidays), so for each series this takes the most recent observation on
    or before `valuation_date`.
    """
    valuation_ts = pd.Timestamp(valuation_date)
    rows = []
    for series_id in series_ids:
        series = fetch_fred_series(series_id)
        as_of = series[series.index <= valuation_ts]
        if as_of.empty:
            continue
        rows.append({
            "series_id": series_id,
            "maturity_years": FRED_MATURITY_YEARS[series_id],
            "obs_date": as_of.index[-1],
            "yield_pct": as_of.iloc[-1],
        })
    if not rows:
        raise ValueError(f"No FRED observations found on or before {valuation_date}")
    return pd.DataFrame(rows).sort_values("maturity_years").reset_index(drop=True)


def fit_curve(curve: pd.DataFrame):
    """Fit a cubic spline through the curve's (maturity_years, yield_pct) points."""
    from scipy.interpolate import CubicSpline

    x = curve["maturity_years"].to_numpy()
    y = curve["yield_pct"].to_numpy()
    return CubicSpline(x, y)


def validate_curve_roundtrip(curve: pd.DataFrame, atol: float = 1e-6) -> pd.DataFrame:
    """Fit the curve, re-evaluate at the input maturities, and confirm the fit
    reproduces the input yields (the round-trip check called for in Phase 1).

    Raises ValueError if any point fails to round-trip within `atol`.
    """
    spline = fit_curve(curve)
    out = curve.copy()
    out["fitted_yield_pct"] = spline(out["maturity_years"].to_numpy())
    out["abs_error"] = (out["fitted_yield_pct"] - out["yield_pct"]).abs()
    out["roundtrip_ok"] = out["abs_error"] <= atol

    if not out["roundtrip_ok"].all():
        bad = out.loc[~out["roundtrip_ok"], "series_id"].tolist()
        raise ValueError(f"Curve round-trip check failed for series: {bad}")
    return out


# --------------------------------------------------------------------------
# yfinance: equity history and market snapshot
# --------------------------------------------------------------------------

def fetch_equity_history(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """Long-format daily OHLCV history for a panel of tickers via yfinance."""
    raw = yf.download(
        list(tickers), start=start, end=end,
        auto_adjust=True, group_by="ticker", progress=False,
    )
    frames = []
    for ticker in tickers:
        try:
            df = raw[ticker].copy()
        except KeyError:
            continue
        df = df.dropna(how="all")
        df["ticker"] = ticker
        frames.append(df.reset_index())
    if not frames:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "ticker"])
    return pd.concat(frames, ignore_index=True).rename(columns={"Date": "date"})


def fetch_equity_snapshot(ticker: str) -> dict:
    """Point-in-time equity snapshot (market cap, shares outstanding, last price)."""
    fast_info = yf.Ticker(ticker).fast_info
    return {
        "ticker": ticker,
        "market_cap": fast_info.get("marketCap"),
        "shares_outstanding": fast_info.get("shares"),
        "last_price": fast_info.get("lastPrice"),
    }


def compute_realized_equity_vol(
    price_history: pd.DataFrame, ticker: str, valuation_date: str, lookback_days: int = 252
) -> float | None:
    """Annualized realized equity volatility from trailing daily log returns,
    as of `valuation_date`. Used as the sigma_E input to the Phase 2 Merton solver.
    """
    df = price_history[price_history["ticker"] == ticker].sort_values("date")
    df = df[df["date"] <= pd.Timestamp(valuation_date)].tail(lookback_days + 1)
    if len(df) < 2:
        return None
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if log_returns.empty:
        return None
    return float(log_returns.std(ddof=1) * np.sqrt(252))


# --------------------------------------------------------------------------
# SEC EDGAR: CIK lookup and company facts (debt figures)
# --------------------------------------------------------------------------

def sec_headers(contact_email: str) -> dict:
    """SEC requires a descriptive User-Agent identifying the requester."""
    return {"User-Agent": f"cva-ml-project research {contact_email}"}


def load_sec_ticker_map(contact_email: str) -> pd.DataFrame:
    """Ticker -> CIK lookup table from SEC's company_tickers.json."""
    resp = requests.get(SEC_TICKERS_URL, headers=sec_headers(contact_email), timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json().values())
    df["ticker"] = df["ticker"].str.upper()
    return df


def lookup_cik(ticker: str, ticker_map: pd.DataFrame) -> int:
    row = ticker_map.loc[ticker_map["ticker"] == ticker.upper()]
    if row.empty:
        raise ValueError(f"No CIK found for ticker {ticker!r}")
    return int(row.iloc[0]["cik_str"])


def fetch_sec_company_facts(cik: int, contact_email: str) -> dict:
    resp = requests.get(SEC_FACTS_URL.format(cik=cik), headers=sec_headers(contact_email), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _latest_fact_asof(
    company_facts: dict, tag_candidates: Sequence[str], valuation_date: pd.Timestamp, unit: str = "USD"
):
    """Most recent XBRL fact value across all of `tag_candidates`.

    Only considers facts with `filed` <= valuation_date, so nothing filed
    after the valuation date leaks in (the same lookahead-bias concern the
    plan flags for the PD classifier's time split -- it applies here too).
    Every candidate tag is checked and the single most recent balance-sheet
    `end` date wins -- not just the first candidate with any data, since a
    filer can have a tag with only stale history alongside a different tag
    with current data.
    """
    gaap = company_facts.get("facts", {}).get("us-gaap", {})
    best = None  # (end, filed, val, tag)
    for tag in tag_candidates:
        entries = gaap.get(tag, {}).get("units", {}).get(unit)
        if not entries:
            continue
        df = pd.DataFrame(entries)
        df["filed"] = pd.to_datetime(df["filed"])
        df["end"] = pd.to_datetime(df["end"])
        df = df[df["filed"] <= valuation_date]
        if df.empty:
            continue
        row = df.sort_values(["end", "filed"]).iloc[-1]
        if best is None or row["end"] > best[0]:
            best = (row["end"], row["filed"], float(row["val"]), tag)
    if best is None:
        return None, None, None
    end, _filed, val, tag = best
    return val, end, tag


def extract_debt_figures(company_facts: dict, valuation_date: str) -> dict:
    """Debt figures as of `valuation_date`, using the plan's Merton debt
    convention: total_debt_face_value = short_term_debt + 0.5 * long_term_debt.
    """
    valuation_ts = pd.Timestamp(valuation_date)
    lt_val, lt_end, lt_tag = _latest_fact_asof(company_facts, DEBT_TAG_CANDIDATES["long_term_debt"], valuation_ts)
    st_val, st_end, st_tag = _latest_fact_asof(company_facts, DEBT_TAG_CANDIDATES["short_term_debt"], valuation_ts)

    total_debt = (st_val or 0.0) + 0.5 * (lt_val or 0.0)
    return {
        "long_term_debt": lt_val,
        "long_term_debt_tag": lt_tag,
        "long_term_debt_asof": lt_end,
        "short_term_debt": st_val,
        "short_term_debt_tag": st_tag,
        "short_term_debt_asof": st_end,
        "total_debt_face_value": total_debt if (lt_val is not None or st_val is not None) else None,
    }


# --------------------------------------------------------------------------
# Phase 2b: public bankruptcy-prediction dataset
# --------------------------------------------------------------------------

def fetch_bankruptcy_dataset(raw_dir: str | Path | None = "data/raw") -> pd.DataFrame:
    """Cross-sectional labeled dataset for the Phase 2b PD classifier.

    Unlike the FRED/yfinance/SEC panel above (Phase 1), this dataset is not
    tied to the company panel or valuation date -- it's a separate, larger,
    labeled corpus used purely to train and evaluate the classifier itself.
    """
    df = pd.read_csv(BANKRUPTCY_DATASET_URL)
    df.columns = [c.strip() for c in df.columns]
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_dir / "taiwan_bankruptcy.csv", index=False)
    return df


# --------------------------------------------------------------------------
# Panel assembly and validation
# --------------------------------------------------------------------------

def build_company_panel(
    tickers: Sequence[str],
    valuation_date: str,
    contact_email: str,
    lookback_days: int = 252,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Assemble one row per company: equity value/vol from yfinance, debt
    figures from SEC EDGAR. This is the raw input to Phase 2's Merton solver
    and Phase 2b's feature matrix.
    """
    valuation_ts = pd.Timestamp(valuation_date)
    start = (valuation_ts - pd.Timedelta(days=int(lookback_days * 1.6) + 10)).date().isoformat()
    end = (valuation_ts + pd.Timedelta(days=1)).date().isoformat()

    price_history = fetch_equity_history(tickers, start=start, end=end)
    ticker_map = load_sec_ticker_map(contact_email)

    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        price_history.to_csv(raw_dir / "equity_history.csv", index=False)

    rows = []
    for ticker in tickers:
        snapshot = fetch_equity_snapshot(ticker)
        equity_vol = compute_realized_equity_vol(price_history, ticker, valuation_date, lookback_days)

        debt = {
            "long_term_debt": None, "long_term_debt_tag": None, "long_term_debt_asof": None,
            "short_term_debt": None, "short_term_debt_tag": None, "short_term_debt_asof": None,
            "total_debt_face_value": None,
        }
        cik = None
        try:
            cik = lookup_cik(ticker, ticker_map)
            facts = fetch_sec_company_facts(cik, contact_email)
            if raw_dir is not None:
                (raw_dir / f"sec_facts_{ticker}.json").write_text(json.dumps(facts))
            debt = extract_debt_figures(facts, valuation_date)
        except (ValueError, requests.HTTPError) as exc:
            print(f"[warn] SEC data unavailable for {ticker}: {exc}")

        rows.append({
            "ticker": ticker,
            "cik": cik,
            "equity_value": snapshot["market_cap"],
            "shares_outstanding": snapshot["shares_outstanding"],
            "last_price": snapshot["last_price"],
            "equity_vol": equity_vol,
            **debt,
        })
        time.sleep(0.2)  # polite pacing for SEC EDGAR

    return pd.DataFrame(rows)


def check_panel_sanity(panel: pd.DataFrame, valuation_date: str, stale_days: int = DEBT_STALENESS_DAYS) -> pd.DataFrame:
    """Basic distribution/range sanity checks on the assembled company panel.

    Flags rows with implausible values rather than letting them pass
    silently into Phase 2's Merton solver -- e.g. debt figures that round-
    tripped through `_latest_fact_asof` but are older than `stale_days`
    relative to the valuation date (a real issue found in this panel: some
    filers stop using a given XBRL tag for years, so "the most recent value
    we could find" can still be badly out of date).
    """
    valuation_ts = pd.Timestamp(valuation_date)
    flags = []
    for _, row in panel.iterrows():
        row_flags = []
        if pd.isna(row["equity_value"]) or row["equity_value"] <= 0:
            row_flags.append("equity_value_nonpositive_or_missing")
        if pd.isna(row["equity_vol"]) or not (0 < row["equity_vol"] < 3):
            row_flags.append("equity_vol_out_of_plausible_range")
        if pd.isna(row["total_debt_face_value"]):
            row_flags.append("debt_missing")
        elif row["total_debt_face_value"] < 0:
            row_flags.append("debt_negative")
        for leg in ("long_term_debt", "short_term_debt"):
            asof = row.get(f"{leg}_asof")
            if pd.notna(asof) and (valuation_ts - asof).days > stale_days:
                row_flags.append(f"{leg}_stale_as_of_{pd.Timestamp(asof).date()}")
        flags.append({"ticker": row["ticker"], "flags": row_flags, "n_flags": len(row_flags)})
    return pd.DataFrame(flags)


def audit_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Null/missing-value audit: count and percentage missing per column."""
    n = len(df)
    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().sum() / n * 100).round(2) if n else 0.0,
    })
    return audit.sort_values("pct_missing", ascending=False)


# --------------------------------------------------------------------------
# Config-driven entry point
# --------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_phase1(config_path: str | Path = "config.yaml", contact_email: str = "youssef.mousaaid@gmail.com"):
    """Run the full Phase 1 pipeline: FRED curve + company panel, with the
    missing-data audit and curve round-trip validation from the plan.
    """
    cfg = load_config(config_path)
    valuation_date = cfg["valuation_date"]
    series_ids = cfg["fred"]["series_ids"]
    tickers = cfg["company_panel"]["tickers"]

    print(f"=== Phase 1: valuation_date={valuation_date}, panel={tickers} ===\n")

    curve = fetch_fred_curve(valuation_date, series_ids)
    curve_validated = validate_curve_roundtrip(curve)
    print("--- FRED curve (round-trip validated) ---")
    print(curve_validated[["series_id", "maturity_years", "obs_date", "yield_pct", "fitted_yield_pct", "abs_error"]]
          .to_string(index=False))

    panel = build_company_panel(tickers, valuation_date, contact_email, raw_dir="data/raw")

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    curve_validated.to_csv(processed_dir / f"fred_curve_{valuation_date}.csv", index=False)
    panel.to_csv(processed_dir / f"company_panel_{valuation_date}.csv", index=False)

    print("\n--- Company panel ---")
    print(panel.to_string(index=False))

    print("\n--- Missing-data audit: company panel ---")
    print(audit_missing_data(panel).to_string())

    sanity = check_panel_sanity(panel, valuation_date)
    flagged = sanity[sanity["n_flags"] > 0]
    print("\n--- Distribution / sanity-range check: company panel ---")
    if flagged.empty:
        print("No sanity-check flags raised.")
    else:
        print(flagged.to_string(index=False))

    print(f"\nSaved: data/processed/fred_curve_{valuation_date}.csv, "
          f"data/processed/company_panel_{valuation_date}.csv")

    return curve_validated, panel


if __name__ == "__main__":
    run_phase1()
