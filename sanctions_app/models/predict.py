"""Inference pipeline for sanctions detection.

Computes composite risk scores and flags transactions.
Composite score = 0.4 * logit_proba + 0.6 * xgb_proba (XGBoost weighted higher
for its superior capture of non-linear interactions).
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.features import compute_features
from utils.constants import (
    DEFAULT_HIGH_THRESHOLD, DEFAULT_GREY_THRESHOLD,
    SANCTIONED_COUNTRIES, FEATURE_NAMES
)


def get_flag_reasons(row: pd.Series, high_threshold: float = DEFAULT_HIGH_THRESHOLD) -> str:
    """
    Generate human-readable flag reasons for a transaction.

    Args:
        row: Transaction row with feature columns and scores
        high_threshold: Score threshold for high risk

    Returns:
        Semicolon-separated string of flag reasons
    """
    reasons = []

    if row.get('sanctioned_country_flag', 0) == 1:
        reasons.append(f"Sanctioned country ({row.get('receiver_country', '?')})")

    match_score = row.get('sanctions_list_match_score', 0)
    if match_score >= 0.8:
        reasons.append(f"High entity match ({match_score:.0%})")
    elif match_score >= 0.6:
        reasons.append(f"Moderate entity match ({match_score:.0%})")

    if row.get('country_risk_score', 0) >= 0.7:
        reasons.append(f"High-risk jurisdiction (score: {row.get('country_risk_score', 0):.2f})")

    if row.get('currency_mismatch', 0) == 1:
        reasons.append("Currency mismatch")

    benford_dev = row.get('benford_deviation', 0)
    if benford_dev > 15:
        reasons.append(f"Benford violation (chi2={benford_dev:.1f})")

    if row.get('amount_to_revenue_ratio', 0) > 0.1:
        reasons.append("High amount-to-revenue ratio")

    if not reasons and row.get('composite_score', 0) >= high_threshold:
        reasons.append("Elevated composite ML score")

    return '; '.join(reasons) if reasons else 'Low risk — no specific flags'


def predict_batch(
    df: pd.DataFrame,
    logit_model,
    xgb_model,
    scaler,
    imputer,
    sanctions_df: pd.DataFrame,
    governance_scores: dict,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    grey_threshold: float = DEFAULT_GREY_THRESHOLD
) -> pd.DataFrame:
    """
    Run the full inference pipeline on a batch of transactions.

    Args:
        df: Transactions DataFrame
        logit_model: Trained LogisticRegression
        xgb_model: Trained XGBClassifier (or None)
        scaler: Fitted StandardScaler
        imputer: Fitted SimpleImputer
        sanctions_df: Combined sanctions entity list
        governance_scores: Country risk scores dict
        high_threshold: Score above which transaction is High Risk
        grey_threshold: Score above which transaction is Grey Zone

    Returns:
        Input df with added columns: feature columns, logit_proba, xgb_proba,
        composite_score, risk_level, flag_reasons
    """
    result = df.copy()

    # Compute features
    features = compute_features(df, sanctions_df, governance_scores)
    X_imputed = imputer.transform(features)
    X_scaled = scaler.transform(X_imputed)

    # Logit predictions
    logit_proba = logit_model.predict_proba(X_scaled)[:, 1]
    result['logit_proba'] = logit_proba

    # XGBoost predictions
    if xgb_model is not None:
        xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
        result['xgb_proba'] = xgb_proba
        # Composite score: weight XGBoost more heavily
        composite = 0.4 * logit_proba + 0.6 * xgb_proba
    else:
        result['xgb_proba'] = logit_proba
        composite = logit_proba

    result['composite_score'] = composite

    # Risk levels (Altman Z-Score zone analogy)
    result['risk_level'] = pd.cut(
        composite,
        bins=[-np.inf, grey_threshold, high_threshold, np.inf],
        labels=['Low', 'Grey', 'High']
    )

    # Add key feature columns for display
    for col in ['sanctions_list_match_score', 'country_risk_score',
                'sanctioned_country_flag', 'currency_mismatch', 'benford_deviation']:
        if col in features.columns:
            result[col] = features[col].values

    # Flag reasons
    result['flag_reasons'] = result.apply(
        lambda row: get_flag_reasons(row, high_threshold), axis=1
    )

    return result
