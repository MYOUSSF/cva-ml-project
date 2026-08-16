"""Feature engineering.

For Phase 2b: cleans the raw Taiwan bankruptcy dataset (fetched by
`data_pipeline.fetch_bankruptcy_dataset`) into a model-ready (X, y). The
dataset already ships as ~95 pre-computed accounting ratios (leverage,
liquidity, profitability -- the same families as Altman Z-score components),
so no further ratio construction is needed; this only drops columns that
carry no information or duplicate another column, found via EDA (see
BANKRUPTCY_DROP_COLS below).

Note: the plan's original idea for this phase was to also add the Phase 2
Merton-derived Distance-to-Default as an engineered feature alongside the
accounting ratios. That's not possible for *this* dataset -- the Taiwan
bankruptcy data has no market equity value/volatility columns, which Merton
requires, and it covers a different company universe than the Phase 1/2
ten-ticker panel. See the Phase 2b section of the README for the full
explanation.
"""

from __future__ import annotations

import pandas as pd

BANKRUPTCY_LABEL_COL = "Bankrupt?"

# Found via EDA on the raw dataset: 'Net Income Flag' is constant (zero
# information); the rest are near-duplicate (>0.999 correlation) of another
# column already kept, or an exact complement (Net worth/Assets == 1 - Debt
# ratio %). Dropping the redundant half keeps the feature set non-degenerate
# for the logistic regression without losing information for either model.
BANKRUPTCY_DROP_COLS = [
    "Net Income Flag",
    "Realized Sales Gross Margin",       # ~duplicate of Operating Gross Margin
    "Gross Profit to Sales",             # ~duplicate of Operating Gross Margin
    "Net Value Per Share (A)",           # ~duplicate of Net Value Per Share (B)
    "Net Value Per Share (C)",           # ~duplicate of Net Value Per Share (B)
    "Net worth/Assets",                  # == 1 - Debt ratio %
    "Current Liability to Liability",    # duplicate of Current Liabilities/Liability
    "Current Liability to Equity",       # duplicate of Current Liabilities/Equity
]


def prepare_bankruptcy_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the raw bankruptcy dataframe into (X, y), dropping the label
    column and the constant/redundant columns identified in
    BANKRUPTCY_DROP_COLS.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    y = df[BANKRUPTCY_LABEL_COL].astype(int)
    drop_cols = [BANKRUPTCY_LABEL_COL] + [c for c in BANKRUPTCY_DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y
