"""Unit tests for Phase 2b (supervised PD classifier).

Uses a small synthetic dataset (a mildly-imbalanced, linearly separable
classification problem, not the real Taiwan data) so these run offline and
fast -- this is the "regression/classification pipeline runs end-to-end on
a small synthetic dataset" check the plan calls for in Phase 10.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import BANKRUPTCY_LABEL_COL, prepare_bankruptcy_features
from src.pd_model import (
    evaluate_pd_classifier,
    split_bankruptcy_dataset,
    train_gradient_boosting,
    train_logistic_regression,
)


def _synthetic_bankruptcy_df(n=400, n_features=5, positive_rate=0.1, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pos = int(n * positive_rate)
    n_neg = n - n_pos

    # Positive class shifted in feature space so the problem is learnable
    # but not trivial (some overlap), like real financial-ratio data.
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(n_neg, n_features))
    X_pos = rng.normal(loc=1.5, scale=1.0, size=(n_pos, n_features))

    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * n_neg + [1] * n_pos)

    df = pd.DataFrame(X, columns=[f"ratio_{i}" for i in range(n_features)])
    df[BANKRUPTCY_LABEL_COL] = y
    # Shuffle rows so the split isn't handed a pre-sorted label column.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_prepare_bankruptcy_features_drops_label_and_known_redundant_cols():
    df = _synthetic_bankruptcy_df(n=50)
    df["Net Income Flag"] = 1  # constant column that should always be dropped

    X, y = prepare_bankruptcy_features(df)

    assert BANKRUPTCY_LABEL_COL not in X.columns
    assert "Net Income Flag" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == 50


def test_split_bankruptcy_dataset_is_stratified():
    df = _synthetic_bankruptcy_df(n=400, positive_rate=0.1)
    X, y = prepare_bankruptcy_features(df)

    X_train, X_test, y_train, y_test = split_bankruptcy_dataset(X, y, test_size=0.25)

    assert len(X_train) + len(X_test) == len(X)
    # Stratified split should keep the positive rate close to the full-data rate.
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.03)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.05)


@pytest.mark.parametrize("train_fn", [train_logistic_regression, train_gradient_boosting])
def test_classifiers_fit_and_beat_chance_on_separable_synthetic_data(train_fn):
    df = _synthetic_bankruptcy_df(n=600, positive_rate=0.15, seed=1)
    X, y = prepare_bankruptcy_features(df)
    X_train, X_test, y_train, y_test = split_bankruptcy_dataset(X, y)

    model = train_fn(X_train, y_train)
    metrics = evaluate_pd_classifier(model, X_test, y_test)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    # The synthetic classes are shifted apart, so a fitted model should
    # comfortably beat random-chance ROC-AUC (0.5) and the base positive
    # rate for PR-AUC.
    assert metrics["roc_auc"] > 0.8
    assert metrics["pr_auc"] > y_test.mean()
    assert len(metrics["proba"]) == len(y_test)
    assert len(metrics["calibration_frac_positive"]) == len(metrics["calibration_mean_predicted"])
