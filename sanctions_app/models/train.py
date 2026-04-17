"""ML training pipeline for sanctions detection.

Trains Logistic Regression and XGBoost models on synthetic data.
Models are saved to models/saved/ for use in the Streamlit app.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.features import compute_features
from utils.constants import FEATURE_NAMES

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Will use only Logistic Regression.")

SAVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')


def train_models(df: pd.DataFrame, sanctions_df: pd.DataFrame, governance_scores: dict):
    """
    Train logit and XGBoost models on the provided data.

    Args:
        df: Training DataFrame with transactions and 'is_sanctioned' label
        sanctions_df: Sanctions entity list for feature computation
        governance_scores: Country risk scores dict

    Returns:
        Tuple of (logit_model, xgb_model, scaler, imputer, X_test, y_test, feature_names)
    """
    print("Computing features...")
    X = compute_features(df, sanctions_df, governance_scores)
    y = df['is_sanctioned'].values

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Logistic Regression...")
    logit_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    logit_model.fit(X_train, y_train)
    logit_auc = roc_auc_score(y_test, logit_model.predict_proba(X_test)[:, 1])
    print(f"Logit AUC: {logit_auc:.4f}")
    print(classification_report(y_test, logit_model.predict(X_test)))

    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            eval_metric='aucpr',
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
        print(f"XGBoost AUC: {xgb_auc:.4f}")
    else:
        xgb_model = None

    return logit_model, xgb_model, scaler, imputer, X_test, y_test, FEATURE_NAMES


def save_models(logit_model, xgb_model, scaler, imputer):
    """Save trained models and preprocessors to disk."""
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(logit_model, os.path.join(SAVED_DIR, 'logit_model.pkl'))
    joblib.dump(scaler, os.path.join(SAVED_DIR, 'scaler.pkl'))
    joblib.dump(imputer, os.path.join(SAVED_DIR, 'imputer.pkl'))
    if xgb_model is not None:
        joblib.dump(xgb_model, os.path.join(SAVED_DIR, 'xgb_model.pkl'))
    print(f"Models saved to {SAVED_DIR}")


def load_saved_models():
    """
    Load previously saved models from disk.

    Returns:
        Tuple of (logit_model, xgb_model, scaler, imputer) or None if not found.
    """
    try:
        logit_path = os.path.join(SAVED_DIR, 'logit_model.pkl')
        scaler_path = os.path.join(SAVED_DIR, 'scaler.pkl')
        imputer_path = os.path.join(SAVED_DIR, 'imputer.pkl')
        xgb_path = os.path.join(SAVED_DIR, 'xgb_model.pkl')

        if not os.path.exists(logit_path):
            return None

        logit_model = joblib.load(logit_path)
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None

        return logit_model, xgb_model, scaler, imputer
    except Exception as e:
        print(f"Could not load saved models: {e}")
        return None


def load_or_train_models():
    """
    Load saved models if available; otherwise generate data and train.

    Returns:
        Tuple of (logit_model, xgb_model, scaler, imputer)
    """
    saved = load_saved_models()
    if saved is not None:
        print("Loaded saved models.")
        return saved

    print("No saved models found. Training from scratch...")

    # Generate synthetic training data
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.generate_synthetic_data import generate_data
    from data.load_sanctions_lists import combine_lists
    from data.world_bank import get_all_governance_scores

    df = generate_data(n=10000)
    sanctions_df = combine_lists()
    countries = list(df['receiver_country'].unique())
    governance_scores = get_all_governance_scores(countries)

    logit_model, xgb_model, scaler, imputer, X_test, y_test, feature_names = train_models(
        df, sanctions_df, governance_scores
    )
    save_models(logit_model, xgb_model, scaler, imputer)

    return logit_model, xgb_model, scaler, imputer


if __name__ == '__main__':
    print("=== Sanctions Detection Model Training ===")
    logit, xgb, scaler, imputer = load_or_train_models()
    print("Training complete!")
