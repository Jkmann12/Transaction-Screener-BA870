"""
Page 3: Benford's Law Analysis
BA775 Financial Analytics | Boston University

Per-counterparty Benford's Law conformance check on transaction amounts.

Lecture 4 Connection: Benford's Law states that in naturally occurring financial data,
the leading digit d appears with probability log10(1 + 1/d). This app uses Benford
deviation as a fraud-detection feature, mirroring the Beneish M-Score's use of
financial ratios to detect earnings manipulation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_ROOT)
from utils.benford import benford_test, BENFORD_EXPECTED
from data.generate_synthetic_data import generate_data

st.set_page_config(page_title="Benford's Analysis", page_icon="🔢", layout="wide")

st.title("🔢 Benford's Law Analysis")
st.markdown(
    "Per-counterparty first-digit frequency analysis to detect fabricated or "
    "manipulated transaction amounts."
)

with st.expander("📖 What is Benford's Law?", expanded=False):
    st.markdown("""
    **Benford's Law** (also called the First-Digit Law) states that in naturally occurring collections
    of numbers — including financial transactions — the leading digit is **not** uniformly distributed.
    Instead, smaller digits appear more frequently:

    | Digit | Expected Probability |
    |-------|---------------------|
    | 1 | 30.1% |
    | 2 | 17.6% |
    | 3 | 12.5% |
    | 4 | 9.7% |
    | 5 | 7.9% |
    | 6 | 6.7% |
    | 7 | 5.8% |
    | 8 | 5.1% |
    | 9 | 4.6% |

    The formula is: **P(d) = log₁₀(1 + 1/d)**

    **Why does this matter for financial fraud detection?**
    - Humans fabricating numbers tend to distribute digits more uniformly (e.g., using digits 5-9 more than expected)
    - Structured transactions (e.g., just-below-reporting-threshold amounts) cluster around specific digits
    - Significant deviations from Benford's distribution are a red flag, used by forensic accountants and auditors

    **This directly connects to the Beneish M-Score framework (Lecture 4)**, which uses financial ratios
    to detect earnings manipulation — both approaches flag statistical anomalies in financial data.
    """)

st.info(
    "**📊 Lecture 4 Connection:** Benford's Law deviation is one of the 9 features in our ML model "
    "(computed via chi-squared test per counterparty). The synthetic data generator deliberately "
    "violates Benford for suspicious transactions by over-representing digits 5-9."
)

# Load data
if 'screening_results' in st.session_state:
    results = st.session_state['screening_results']
    st.success(f"✅ Using screened results: {len(results):,} transactions")
else:
    st.warning("⚠️ No screening results. Loading demo data for analysis.")
    @st.cache_data
    def load_demo():
        demo_path = os.path.join(APP_ROOT, 'data', 'synthetic_transactions.csv')
        if os.path.exists(demo_path):
            return pd.read_csv(demo_path)
        return generate_data(n=5000)
    results = load_demo()

if 'amount' not in results.columns or 'receiver_name' not in results.columns:
    st.error("❌ Data must have 'amount' and 'receiver_name' columns.")
    st.stop()

st.markdown("---")

# Counterparty selector
counterparties = sorted(results['receiver_name'].dropna().unique().tolist())
if not counterparties:
    st.error("No counterparties found in data.")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    selected_counterparty = st.selectbox(
        "Select Counterparty",
        options=counterparties,
        index=0,
        help="Choose a counterparty to analyze their transaction amount distribution"
    )
with col2:
    cp_txns = results[results['receiver_name'] == selected_counterparty]
    st.metric("Transactions for Selected Counterparty", len(cp_txns))

# Run Benford test
amounts = cp_txns['amount'].dropna().tolist()
benford_result = benford_test(amounts)

if benford_result is None:
    st.warning(f"⚠️ Insufficient data for {selected_counterparty} (need ≥ 10 transactions, found {len(amounts)})")
else:
    # Bar chart: observed vs expected
    digits = list(range(1, 10))
    observed_pct = benford_result['observed'] / benford_result['n'] * 100
    expected_pct = benford_result['expected'] / benford_result['n'] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=digits, y=observed_pct,
        name='Observed', marker_color='#1f77b4',
        text=[f"{v:.1f}%" for v in observed_pct], textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=digits, y=expected_pct,
        name='Expected (Benford)', marker_color='#ff7f0e',
        text=[f"{v:.1f}%" for v in expected_pct], textposition='outside'
    ))
    fig.update_layout(
        title=f"First-Digit Distribution: {selected_counterparty} (n={benford_result['n']})",
        xaxis_title="Leading Digit",
        yaxis_title="Frequency (%)",
        barmode='group',
        height=450,
        xaxis=dict(tickmode='array', tickvals=digits, ticktext=[str(d) for d in digits])
    )
    st.plotly_chart(fig, use_container_width=True)

    # Test results
    chi2 = benford_result['chi2']
    p_value = benford_result['p_value']

    col1, col2, col3 = st.columns(3)
    col1.metric("Chi-Squared Statistic", f"{chi2:.3f}")
    col2.metric("P-Value", f"{p_value:.4f}")
    col3.metric("Sample Size", benford_result['n'])

    if p_value < 0.05:
        st.error(
            f"⚠️ **SUSPICIOUS: Significant deviation from Benford's Law** "
            f"(p = {p_value:.4f} < 0.05). "
            "This counterparty's transaction amounts do not follow the expected natural distribution. "
            "This may indicate fabricated amounts, structured transactions, or manipulation."
        )
    else:
        st.success(
            f"✅ **PASS: Distribution consistent with Benford's Law** "
            f"(p = {p_value:.4f} ≥ 0.05). "
            "Transaction amounts appear to follow natural digit distribution."
        )

st.markdown("---")

# Summary table: all counterparties by Benford deviation
st.markdown("### 📋 All Counterparties — Benford Deviation Summary")
st.caption("Sorted by chi-squared statistic (most suspicious first). Requires ≥ 10 transactions.")

summary_rows = []
for name, group in results.groupby('receiver_name'):
    amounts_grp = group['amount'].dropna().tolist()
    result = benford_test(amounts_grp)
    if result:
        summary_rows.append({
            'Counterparty': name,
            'Transactions': result['n'],
            'Chi² Statistic': round(result['chi2'], 3),
            'P-Value': round(result['p_value'], 4),
            'Status': '⚠️ Suspicious' if result['p_value'] < 0.05 else '✅ Normal'
        })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows).sort_values('Chi² Statistic', ascending=False)
    st.dataframe(summary_df, use_container_width=True, height=400)
else:
    st.info("No counterparties with sufficient transaction history for Benford analysis.")
