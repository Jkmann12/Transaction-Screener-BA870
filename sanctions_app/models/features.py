"""Feature engineering for the sanctions detection ML models.

Computes 9 features per transaction for Logistic Regression and XGBoost models.
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import SANCTIONED_COUNTRIES, COUNTRY_CURRENCIES, FEATURE_NAMES
from utils.benford import get_benford_deviation
from utils.fuzzy_match import match_entity


def compute_features(
    df: pd.DataFrame,
    sanctions_df: pd.DataFrame,
    governance_scores: Dict[str, float]
) -> pd.DataFrame:
    """
    Compute all 9 ML features for each transaction.

    Args:
        df: Transactions DataFrame with required columns
        sanctions_df: Combined sanctions entity list
        governance_scores: Dict mapping country_code -> risk_score (0-1)

    Returns:
        DataFrame with exactly FEATURE_NAMES columns
    """
    features = pd.DataFrame(index=df.index)

    # Feature 1: Sanctions list match score (fuzzy match receiver_name)
    # Match once per unique name, then join back — avoids re-running fuzzy match
    # for every row when the same counterparty appears thousands of times.
    if not sanctions_df.empty:
        unique_names = df['receiver_name'].astype(str).unique()
        name_to_score = {
            name: match_entity(name, sanctions_df, threshold=60)['match_score']
            for name in unique_names
        }
        match_scores = df['receiver_name'].astype(str).map(name_to_score)
    else:
        match_scores = [0.0] * len(df)
    features['sanctions_list_match_score'] = match_scores

    # Feature 2: Country risk score from governance indicators
    features['country_risk_score'] = df['receiver_country'].map(
        lambda c: governance_scores.get(c, 0.5)
    )

    # Feature 3: Sanctioned country flag
    features['sanctioned_country_flag'] = df['receiver_country'].isin(SANCTIONED_COUNTRIES).astype(int)

    # Feature 4: Log-transformed transaction amount
    features['transaction_amount_log'] = np.log1p(df['amount'].clip(lower=0))

    # Feature 5: Currency mismatch flag
    features['currency_mismatch'] = df.apply(
        lambda row: int(
            row.get('currency', '') != COUNTRY_CURRENCIES.get(row.get('receiver_country', ''), row.get('currency', ''))
            and COUNTRY_CURRENCIES.get(row.get('receiver_country', ''), '') != ''
        ),
        axis=1
    )

    # Feature 6: Amount-to-revenue ratio (default to 0.5; yfinance lookup is on-demand)
    features['amount_to_revenue_ratio'] = 0.5

    # Feature 7: Benford deviation (chi-squared statistic per receiver_name)
    benford_map = {}
    for name, group in df.groupby('receiver_name'):
        amounts = group['amount'].tolist()
        benford_map[name] = get_benford_deviation(amounts)
    features['benford_deviation'] = df['receiver_name'].map(benford_map).fillna(0.0)

    # Feature 8: Transaction frequency (normalized count per receiver_name)
    freq_counts = df['receiver_name'].value_counts()
    max_count = freq_counts.max() if len(freq_counts) > 0 else 1
    features['transaction_frequency'] = df['receiver_name'].map(freq_counts) / max_count

    # Feature 9: Interaction term (country risk * log amount)
    features['high_risk_geo_interaction'] = (
        features['country_risk_score'] * features['transaction_amount_log']
    )

    # Ensure correct column order
    return features[FEATURE_NAMES]
