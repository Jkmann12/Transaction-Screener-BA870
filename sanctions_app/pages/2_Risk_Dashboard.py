"""
Page 2: Risk Dashboard
BA775 Financial Analytics | Boston University

Visual overview of screening results: world heatmap, risk score histogram,
and top flagged transactions table.

Lecture 3-4 Connection: Risk score histogram uses High/Grey/Low threshold zones
directly modeled on Altman Z-Score discrimination zones:
- Score > high_threshold = High Risk (analogous to Z < 1.80 distress zone)
- Score between thresholds = Grey Zone (analogous to 1.80 < Z < 2.99)
- Score < grey_threshold = Low Risk (analogous to Z > 2.99 safe zone)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_ROOT)
from utils.constants import DEFAULT_HIGH_THRESHOLD, DEFAULT_GREY_THRESHOLD

st.set_page_config(page_title="Risk Dashboard", page_icon="📊", layout="wide")

st.title("📊 Risk Dashboard")
st.markdown("Visual overview of sanctions screening results across all screened transactions.")

# Check for results
if 'screening_results' not in st.session_state:
    st.warning("⚠️ No screening results yet. Please run Transaction Screening first.")
    st.page_link("pages/1_Transaction_Screening.py", label="Go to Transaction Screening →")
    st.stop()

results = st.session_state['screening_results']
high_threshold = st.session_state.get('high_threshold', DEFAULT_HIGH_THRESHOLD)
grey_threshold = st.session_state.get('grey_threshold', DEFAULT_GREY_THRESHOLD)

# Threshold controls
st.markdown("### ⚙️ Risk Thresholds")
col1, col2 = st.columns(2)
with col1:
    high_threshold = st.slider("High-Risk Threshold", 0.0, 1.0, high_threshold, 0.05,
                               help="Altman Z: analogous to Z < 1.80 distress zone")
with col2:
    grey_threshold = st.slider("Grey-Zone Threshold", 0.0, 1.0, grey_threshold, 0.05,
                               help="Altman Z: analogous to 1.80 < Z < 2.99")
st.session_state['high_threshold'] = high_threshold
st.session_state['grey_threshold'] = grey_threshold

# Recompute risk levels with updated thresholds
results = results.copy()
results['risk_level'] = pd.cut(
    results['composite_score'],
    bins=[-np.inf, grey_threshold, high_threshold, np.inf],
    labels=['Low', 'Grey', 'High']
)

st.markdown("---")

# === VISUALIZATION 1: World Choropleth ===
st.markdown("### 🗺️ Geographic Risk Distribution")

# ISO-2 to ISO-3 mapping (Plotly choropleth requires ISO-3)
ISO2_TO_ISO3 = {
    'AF': 'AFG', 'AE': 'ARE', 'AR': 'ARG', 'AM': 'ARM', 'AU': 'AUS',
    'AT': 'AUT', 'AZ': 'AZE', 'BD': 'BGD', 'BE': 'BEL', 'BH': 'BHR',
    'BY': 'BLR', 'BR': 'BRA', 'CA': 'CAN', 'CF': 'CAF', 'CH': 'CHE',
    'CL': 'CHL', 'CM': 'CMR', 'CN': 'CHN', 'CO': 'COL', 'CU': 'CUB',
    'CZ': 'CZE', 'DE': 'DEU', 'DK': 'DNK', 'EC': 'ECU', 'EG': 'EGY',
    'ES': 'ESP', 'ET': 'ETH', 'FR': 'FRA', 'GB': 'GBR', 'GE': 'GEO',
    'GH': 'GHA', 'GR': 'GRC', 'HK': 'HKG', 'HT': 'HTI', 'HU': 'HUN',
    'ID': 'IDN', 'IL': 'ISR', 'IN': 'IND', 'IQ': 'IRQ', 'IR': 'IRN',
    'IT': 'ITA', 'JP': 'JPN', 'KE': 'KEN', 'KG': 'KGZ', 'KP': 'PRK',
    'KR': 'KOR', 'KW': 'KWT', 'KZ': 'KAZ', 'LB': 'LBN', 'LY': 'LBY',
    'MA': 'MAR', 'ML': 'MLI', 'MM': 'MMR', 'MX': 'MEX', 'MY': 'MYS',
    'NG': 'NGA', 'NL': 'NLD', 'NO': 'NOR', 'NZ': 'NZL', 'PE': 'PER',
    'PH': 'PHL', 'PK': 'PAK', 'PL': 'POL', 'PT': 'PRT', 'QA': 'QAT',
    'RU': 'RUS', 'SA': 'SAU', 'SD': 'SDN', 'SE': 'SWE', 'SG': 'SGP',
    'SN': 'SEN', 'SO': 'SOM', 'SS': 'SSD', 'SY': 'SYR', 'TH': 'THA',
    'TJ': 'TJK', 'TM': 'TKM', 'TR': 'TUR', 'TZ': 'TZA', 'UA': 'UKR',
    'US': 'USA', 'UZ': 'UZB', 'VE': 'VEN', 'VN': 'VNM', 'YE': 'YEM',
    'ZA': 'ZAF', 'CF': 'CAF', 'ZW': 'ZWE',
}

country_risk = results.groupby('receiver_country').agg(
    avg_score=('composite_score', 'mean'),
    transaction_count=('composite_score', 'count'),
    high_risk_count=('risk_level', lambda x: (x == 'High').sum())
).reset_index()
country_risk['pct_high_risk'] = country_risk['high_risk_count'] / country_risk['transaction_count'] * 100
country_risk['iso3'] = country_risk['receiver_country'].map(ISO2_TO_ISO3)
country_risk = country_risk.dropna(subset=['iso3'])

fig_map = px.choropleth(
    country_risk,
    locations='iso3',
    locationmode='ISO-3',
    color='avg_score',
    hover_name='receiver_country',
    hover_data={
        'iso3': False,
        'avg_score': ':.3f',
        'transaction_count': True,
        'pct_high_risk': ':.1f'
    },
    color_continuous_scale='RdYlGn_r',
    range_color=[0, 1],
    title='Average Composite Risk Score by Receiver Country',
    labels={'avg_score': 'Avg Risk Score', 'pct_high_risk': '% High Risk'}
)
fig_map.update_layout(
    height=500,
    coloraxis_colorbar_title="Risk Score",
    geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular')
)

selected_country = None
chart_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_chart")
if chart_event and hasattr(chart_event, 'selection') and chart_event.selection:
    points = chart_event.selection.get('points', [])
    if points:
        selected_country = points[0].get('location')
        st.info(f"📍 Filtered to: **{selected_country}**")

st.markdown("---")

# === VISUALIZATION 2: Risk Score Histogram ===
st.markdown("### 📈 Risk Score Distribution")
st.caption("*Modeled on Altman Z-Score discrimination zones — see annotations below*")

fig_hist = go.Figure()

# Color bars by zone
scores = results['composite_score'].values
high_scores = scores[scores > high_threshold]
grey_scores = scores[(scores > grey_threshold) & (scores <= high_threshold)]
low_scores = scores[scores <= grey_threshold]

fig_hist.add_trace(go.Histogram(x=low_scores, name='Low Risk', marker_color='#28a745',
                                 nbinsx=30, opacity=0.8))
fig_hist.add_trace(go.Histogram(x=grey_scores, name='Grey Zone', marker_color='#ffc107',
                                 nbinsx=20, opacity=0.8))
fig_hist.add_trace(go.Histogram(x=high_scores, name='High Risk', marker_color='#dc3545',
                                 nbinsx=15, opacity=0.8))

# Threshold lines
fig_hist.add_vline(x=grey_threshold, line_dash='dash', line_color='orange',
                   annotation_text=f"Grey Zone ({grey_threshold:.0%})<br>Z≈2.99", annotation_position="top")
fig_hist.add_vline(x=high_threshold, line_dash='dash', line_color='red',
                   annotation_text=f"High Risk ({high_threshold:.0%})<br>Z≈1.80", annotation_position="top")

# Background zones
fig_hist.add_vrect(x0=0, x1=grey_threshold, fillcolor='green', opacity=0.05,
                   annotation_text="Safe Zone (Z > 2.99)")
fig_hist.add_vrect(x0=grey_threshold, x1=high_threshold, fillcolor='yellow', opacity=0.07,
                   annotation_text="Grey Zone (1.80 < Z < 2.99)")
fig_hist.add_vrect(x0=high_threshold, x1=1.0, fillcolor='red', opacity=0.07,
                   annotation_text="Distress Zone (Z < 1.80)")

fig_hist.update_layout(
    title='Risk Score Distribution (Altman Z-Score Zone Analogy)',
    xaxis_title='Composite Risk Score',
    yaxis_title='Number of Transactions',
    barmode='overlay',
    height=450,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# === VISUALIZATION 3: Top Flagged Transactions ===
st.markdown("### 🚨 Top Flagged Transactions")

# Filter by selected country if applicable
filtered_results = results.copy()
if selected_country:
    filtered_results = filtered_results[filtered_results['receiver_country'] == selected_country]
    st.caption(f"Showing transactions involving: **{selected_country}**")
else:
    st.caption("Click a country on the map to filter. Showing all transactions sorted by risk score.")

# Top 20 highest risk
top_flagged = filtered_results.nlargest(20, 'composite_score')

display_cols = ['transaction_id', 'date', 'sender_name', 'receiver_name', 'receiver_country',
                'amount', 'currency', 'composite_score', 'risk_level', 'flag_reasons']
available_cols = [c for c in display_cols if c in top_flagged.columns]

def color_risk(val):
    if val == 'High':
        return 'background-color: #ffcccc; color: #cc0000; font-weight: bold'
    elif val == 'Grey':
        return 'background-color: #fff3cc; color: #996600; font-weight: bold'
    return 'background-color: #ccffcc; color: #006600'

display_df = top_flagged[available_cols].copy()
if 'composite_score' in display_df.columns:
    display_df['composite_score'] = display_df['composite_score'].round(4)

st.dataframe(
    display_df.style.applymap(color_risk, subset=['risk_level']) if 'risk_level' in display_df.columns else display_df,
    use_container_width=True,
    height=450
)
