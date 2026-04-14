"""
Page 4: Model Performance
BA775 Financial Analytics | Boston University

Transparency into how the ML models work: feature importance, logit coefficients,
ROC curves, and confusion matrices.

Lecture 3 Connection: Displays logit regression coefficients and odds ratios,
mirroring the logit framework used in Altman's financial distress prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import joblib

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_ROOT)
from utils.constants import FEATURE_NAMES, DEFAULT_HIGH_THRESHOLD
from models.train import load_or_train_models, train_models
from models.features import compute_features
from data.generate_synthetic_data import generate_data
from data.load_sanctions_lists import combine_lists
from data.world_bank import get_all_governance_scores

try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance")
st.markdown(
    "Transparency into the ML sanctions detection models: "
    "feature importance, regression coefficients, ROC curves, and confusion matrices."
)

st.info(
    "**📊 Lecture 3 Connection:** The **Logistic Regression coefficients** below mirror the logit framework "
    "used in Altman's financial distress prediction model. Each coefficient shows the log-odds impact "
    "of that feature on the probability of sanctions involvement — directly analogous to how Altman's "
    "Z-Score weights financial ratios to classify distress probability."
)

# Load models
@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return load_or_train_models()

@st.cache_data(show_spinner="Generating test data...")
def get_test_data():
    df = generate_data(n=2000, random_state=99)
    sanctions_df = combine_lists()
    countries = list(df['receiver_country'].unique())
    governance_scores = get_all_governance_scores(countries)
    return df, sanctions_df, governance_scores

with st.spinner("Loading models..."):
    try:
        logit_model, xgb_model, scaler, imputer = get_models()
        st.success("✅ Models loaded")
    except Exception as e:
        st.error(f"❌ Could not load models: {e}")
        st.stop()

with st.spinner("Preparing test data for evaluation..."):
    try:
        df_test, sanctions_df, governance_scores = get_test_data()
        from sklearn.impute import SimpleImputer
        features = compute_features(df_test, sanctions_df, governance_scores)
        X_imputed = imputer.transform(features)
        X_scaled = scaler.transform(X_imputed)
        y_test = df_test['is_sanctioned'].values

        logit_proba = logit_model.predict_proba(X_scaled)[:, 1]
        if xgb_model is not None:
            xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
        else:
            xgb_proba = logit_proba
    except Exception as e:
        st.error(f"❌ Test data preparation failed: {e}")
        st.stop()

high_threshold = st.session_state.get('high_threshold', DEFAULT_HIGH_THRESHOLD)

st.markdown("---")

# === VISUALIZATION 1: Feature Importance ===
st.markdown("### 🏆 XGBoost Feature Importances")

if xgb_model is not None:
    importances = xgb_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=feat_imp_df['Importance'],
        y=feat_imp_df['Feature'],
        orientation='h',
        marker_color='#1f77b4',
        text=[f"{v:.4f}" for v in feat_imp_df['Importance']],
        textposition='outside'
    ))
    fig_imp.update_layout(
        title='XGBoost Feature Importances (F-Score)',
        xaxis_title='Importance Score',
        height=400,
        margin=dict(l=200)
    )
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.warning("XGBoost model not available.")

st.markdown("---")

# === VISUALIZATION 2: Logit Coefficients Table ===
st.markdown("### 📋 Logistic Regression Coefficients")
st.caption("*(Lecture 3: Logit Framework — analogous to Altman Z-Score ratio weights)*")

coef = logit_model.coef_[0]
coef_df = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'Coefficient': coef,
    'Odds_Ratio': np.exp(coef),
    'Direction': ['↑ Increases Risk' if c > 0 else '↓ Decreases Risk' for c in coef]
}).sort_values('Coefficient', ascending=False)
coef_df['Coefficient'] = coef_df['Coefficient'].round(4)
coef_df['Odds_Ratio'] = coef_df['Odds_Ratio'].round(4)

def color_coef(val):
    try:
        f = float(val)
        if f > 0:
            return 'background-color: #ffcccc'
        elif f < 0:
            return 'background-color: #ccffcc'
    except:
        pass
    return ''

st.dataframe(
    coef_df.style.applymap(color_coef, subset=['Coefficient']),
    use_container_width=True
)
st.caption("Positive coefficient = feature increases sanctions probability (risk). Odds ratio > 1 means increased odds.")

st.markdown("---")

# === VISUALIZATION 3: ROC Curves ===
st.markdown("### 📉 ROC Curves: Logit vs XGBoost")

if SKLEARN_AVAILABLE:
    fig_roc = go.Figure()

    # Logit ROC
    fpr_l, tpr_l, _ = roc_curve(y_test, logit_proba)
    auc_l = auc(fpr_l, tpr_l)
    fig_roc.add_trace(go.Scatter(x=fpr_l, y=tpr_l, mode='lines',
                                  name=f'Logit (AUC = {auc_l:.4f})',
                                  line=dict(color='blue', width=2)))

    # XGBoost ROC
    if xgb_model is not None:
        fpr_x, tpr_x, _ = roc_curve(y_test, xgb_proba)
        auc_x = auc(fpr_x, tpr_x)
        fig_roc.add_trace(go.Scatter(x=fpr_x, y=tpr_x, mode='lines',
                                      name=f'XGBoost (AUC = {auc_x:.4f})',
                                      line=dict(color='red', width=2)))

    # Diagonal
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  name='Random Classifier',
                                  line=dict(color='gray', width=1, dash='dash')))

    fig_roc.update_layout(
        title='ROC Curves: Logistic Regression vs XGBoost',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=450,
        legend=dict(x=0.6, y=0.1)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("---")

# === VISUALIZATION 4: Confusion Matrices ===
st.markdown("### 🔲 Confusion Matrices")

if SKLEARN_AVAILABLE:
    logit_pred = (logit_proba >= high_threshold).astype(int)
    xgb_pred = (xgb_proba >= high_threshold).astype(int) if xgb_model is not None else logit_pred

    cm_l = confusion_matrix(y_test, logit_pred)
    cm_x = confusion_matrix(y_test, xgb_pred)

    fig_cm = make_subplots(rows=1, cols=2,
                            subplot_titles=['Logistic Regression', 'XGBoost'])

    for i, (cm, col) in enumerate([(cm_l, 1), (cm_x, 2)], 1):
        fig_cm.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted: Legit', 'Predicted: Sanctioned'],
                y=['Actual: Legit', 'Actual: Sanctioned'],
                colorscale='Blues',
                text=cm.astype(str),
                texttemplate="%{text}",
                showscale=(col == 2)
            ),
            row=1, col=col
        )

    fig_cm.update_layout(
        title=f'Confusion Matrices at Threshold = {high_threshold:.0%}',
        height=400
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption(f"Threshold: {high_threshold:.0%}. Adjust in sidebar or on Risk Dashboard page.")
