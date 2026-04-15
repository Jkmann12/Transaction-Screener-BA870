"""
TransactIQ
BA775 Financial Analytics | Boston University

Main entry point for the multi-page Streamlit application.
"""
import streamlit as st

st.set_page_config(
    page_title="TransactIQ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ TransactIQ")
st.markdown(
    "**ML-powered screening of financial transactions against global sanctions lists.**  "
    "Upload your own data or explore with synthetic demo transactions."
)
st.markdown("*TransactIQ — Graduate Project | BA775 Financial Analytics | Boston University*")
st.markdown("---")

# Sidebar: global controls
st.sidebar.header("⚙️ Global Settings")
st.sidebar.markdown("*Adjust risk thresholds below. Changes apply across all pages.*")

risk_threshold_high = st.sidebar.slider(
    "High-Risk Threshold",
    min_value=0.0, max_value=1.0, value=0.7, step=0.05,
    help="Composite score above this = High Risk (analogous to Altman Z < 1.80)",
    key="sidebar_high_thresh"
)
risk_threshold_grey = st.sidebar.slider(
    "Grey-Zone Threshold",
    min_value=0.0, max_value=1.0, value=0.4, step=0.05,
    help="Composite score above this = Grey Zone (analogous to 1.80 < Z < 2.99)",
    key="sidebar_grey_thresh"
)

# Persist thresholds in session state
st.session_state['high_threshold'] = risk_threshold_high
st.session_state['grey_threshold'] = risk_threshold_grey

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.caption(
    "Data sources: OFAC SDN, UN Sanctions, OpenSanctions, World Bank, yfinance. "
    "All public/free APIs. App works offline with synthetic demo data."
)

# Main page: navigation guide
st.markdown("### 🚀 Getting Started")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.info("**1️⃣ Transaction Screening**\nUpload CSV/XLSX or use demo data. Run ML screening pipeline.")
with col2:
    st.info("**2️⃣ Risk Dashboard**\nInteractive world heatmap, score histogram, and top flagged transactions.")
with col3:
    st.info("**3️⃣ Benford's Analysis**\nPer-counterparty first-digit distribution check for fraud detection.")
with col4:
    st.info("**4️⃣ Model Performance**\nROC curves, feature importance, logit coefficients.")
with col5:
    st.info("**5️⃣ Counterparty Lookup**\nFetch live financials via yfinance and assess transaction scale.")

st.markdown("---")
st.markdown("### 📋 Expected Data Schema")
st.code(
    "transaction_id | date | sender_name | sender_country | "
    "receiver_name | receiver_country | amount | currency",
    language="text"
)
st.caption("*Countries should be ISO-2 codes (e.g., US, GB, DE). Amounts in any currency.*")
