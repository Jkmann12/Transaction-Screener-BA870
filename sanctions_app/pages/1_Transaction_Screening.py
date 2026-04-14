"""
Page 1: Transaction Screening
BA775 Financial Analytics | Boston University

Core screening interface — upload transactions and run the ML sanctions detection pipeline.

Lecture 3 Connection: This pipeline uses Logistic Regression (same logit framework as
Altman's Z-Score distress prediction) combined with XGBoost to generate composite
risk probabilities for each transaction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import io

# Add app root to path for imports
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_ROOT)

from models.train import load_or_train_models
from models.predict import predict_batch
from data.load_sanctions_lists import combine_lists
from data.world_bank import get_all_governance_scores
from data.generate_synthetic_data import generate_data
from utils.constants import DEFAULT_HIGH_THRESHOLD, DEFAULT_GREY_THRESHOLD

st.set_page_config(page_title="Transaction Screening", page_icon="🔍", layout="wide")

st.title("🔍 Transaction Screening")
st.markdown(
    "Screen financial transactions against global sanctions lists using ML-powered composite scoring."
)

st.info(
    "**📊 Lecture 3 Connection:** This screening pipeline uses **Logistic Regression** — "
    "the same logit framework used in Altman's Z-Score financial distress prediction — "
    "combined with XGBoost to generate composite risk probabilities. "
    "The composite score mirrors the Z-Score's use of financial ratios to classify entities "
    "into risk zones (High / Grey / Low)."
)

# Get thresholds from session state or defaults
high_threshold = st.session_state.get('high_threshold', DEFAULT_HIGH_THRESHOLD)
grey_threshold = st.session_state.get('grey_threshold', DEFAULT_GREY_THRESHOLD)

# Load models and data with caching
@st.cache_resource(show_spinner="Loading ML models...")
def get_models():
    return load_or_train_models()

@st.cache_data(ttl=86400, show_spinner="Loading sanctions lists...")
def get_sanctions():
    return combine_lists()

@st.cache_data(show_spinner="Fetching governance scores...")
def get_governance(countries):
    return get_all_governance_scores(list(countries))

@st.cache_data(show_spinner="Generating demo data...")
def get_demo_data():
    demo_path = os.path.join(APP_ROOT, 'data', 'synthetic_transactions.csv')
    if os.path.exists(demo_path):
        return pd.read_csv(demo_path)
    df = generate_data(n=10000)
    df.to_csv(demo_path, index=False)
    return df

# Load models
with st.spinner("Initializing models (this may take a moment on first run)..."):
    try:
        logit_model, xgb_model, scaler, imputer = get_models()
        st.success("✅ Models loaded successfully")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

# Load sanctions lists
with st.spinner("Loading sanctions lists..."):
    try:
        sanctions_df = get_sanctions()
        st.success(f"✅ Sanctions list loaded: {len(sanctions_df):,} entities")
    except Exception as e:
        st.warning(f"⚠️ Using fallback sanctions data: {e}")
        sanctions_df = pd.DataFrame(columns=['entity_name', 'country', 'source', 'entity_type'])

st.markdown("---")

# Data input tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "🎲 Demo Data", "🔧 Format Data"])

df_input = st.session_state.get('loaded_transactions', None)

with tab1:
    st.markdown("**Upload a CSV or XLSX file with transaction data.**")
    st.caption("Required columns: transaction_id, date, sender_name, sender_country, receiver_name, receiver_country, amount, currency")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file with transaction data"
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df_input = pd.read_excel(uploaded_file)
            else:
                df_input = pd.read_csv(uploaded_file)
            if 'transaction_id' not in df_input.columns:
                df_input.insert(0, 'transaction_id', [f"TXN-{i+1}" for i in range(len(df_input))])
                st.info(f"ℹ️ No 'transaction_id' column found — auto-generated TXN-1 through TXN-{len(df_input):,}")
            st.session_state['loaded_transactions'] = df_input
            st.success(f"✅ Loaded {len(df_input):,} transactions from {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

with tab2:
    st.markdown("**Use synthetic demo data (10,000 transactions, ~15% flagged as suspicious).**")
    st.caption("Generated following FATF typology patterns with Benford's Law amount distributions.")

    if st.button("🔄 Load Demo Data", type="primary"):
        with st.spinner("Loading demo data..."):
            df_input = get_demo_data()
            st.session_state['loaded_transactions'] = df_input
            st.success(f"✅ Loaded {len(df_input):,} demo transactions")

with tab3:
    st.markdown("**Upload any transaction dataset and map its columns to the required schema.**")
    st.caption("Use this to reformat datasets like PaySim, IEEE-CIS, or any custom source before screening.")

    REQUIRED_SCHEMA = [
        'transaction_id', 'date', 'sender_name', 'sender_country',
        'receiver_name', 'receiver_country', 'amount', 'currency'
    ]

    fmt_file = st.file_uploader(
        "Upload dataset to reformat",
        type=['csv', 'xlsx'],
        key="formatter_upload",
        help="Upload the raw dataset — we'll help you map its columns"
    )

    if fmt_file:
        try:
            if fmt_file.name.endswith('.xlsx'):
                fmt_df = pd.read_excel(fmt_file)
            else:
                fmt_df = pd.read_csv(fmt_file)

            st.success(f"✅ Loaded {len(fmt_df):,} rows with {len(fmt_df.columns)} columns")

            with st.expander("👁️ Raw Data Preview", expanded=True):
                st.dataframe(fmt_df.head(5), use_container_width=True)

            st.markdown("#### Map Your Columns to Required Schema")
            st.caption("For each required field, select the matching column from your dataset. Select **-- Leave Blank --** to auto-fill with a default value.")

            source_cols = ['-- Leave Blank --'] + list(fmt_df.columns)
            mapping = {}

            col1, col2 = st.columns(2)
            for i, field in enumerate(REQUIRED_SCHEMA):
                target_col = col1 if i % 2 == 0 else col2
                # Try to auto-detect a matching column name
                auto = next((c for c in fmt_df.columns if field.lower().replace('_', '') in c.lower().replace('_', '')), '-- Leave Blank --')
                with target_col:
                    mapping[field] = st.selectbox(
                        f"`{field}`",
                        options=source_cols,
                        index=source_cols.index(auto) if auto in source_cols else 0,
                        key=f"map_{field}"
                    )

            st.markdown("#### Default Fill Values")
            st.caption("Used for any fields left blank above.")
            col1, col2 = st.columns(2)
            with col1:
                default_currency = st.text_input("Default currency", value="USD")
                default_sender_country = st.text_input("Default sender_country", value="US")
            with col2:
                default_receiver_country = st.text_input("Default receiver_country", value="US")
                default_sender_name = st.text_input("Default sender_name", value="Unknown Sender")

            defaults = {
                'transaction_id': None,  # auto-generate
                'date': pd.Timestamp.today().strftime('%Y-%m-%d'),
                'sender_name': default_sender_name,
                'sender_country': default_sender_country,
                'receiver_name': 'Unknown Receiver',
                'receiver_country': default_receiver_country,
                'amount': 0.0,
                'currency': default_currency
            }

            if st.button("🔄 Preview Reformatted Data", key="preview_fmt"):
                reformatted = pd.DataFrame()
                for field in REQUIRED_SCHEMA:
                    if mapping[field] != '-- Leave Blank --':
                        reformatted[field] = fmt_df[mapping[field]].values
                    else:
                        if field == 'transaction_id':
                            reformatted[field] = [f"TXN-{i+1}" for i in range(len(fmt_df))]
                        else:
                            reformatted[field] = defaults[field]

                # Ensure amount is numeric
                try:
                    reformatted['amount'] = pd.to_numeric(reformatted['amount'], errors='coerce').fillna(0.0)
                except Exception:
                    pass

                st.session_state['reformatted_df'] = reformatted
                st.success("✅ Preview ready")

            if 'reformatted_df' in st.session_state:
                reformatted = st.session_state['reformatted_df']

                st.markdown("#### Reformatted Preview")
                st.dataframe(reformatted.head(10), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    # Download button
                    csv_out = io.StringIO()
                    reformatted.to_csv(csv_out, index=False)
                    st.download_button(
                        label="📥 Download Reformatted CSV",
                        data=csv_out.getvalue(),
                        file_name="reformatted_transactions.csv",
                        mime="text/csv"
                    )
                with col2:
                    # Load directly into screener
                    if st.button("➡️ Load into Screener", type="primary", key="load_fmt"):
                        st.session_state['loaded_transactions'] = reformatted
                        df_input = reformatted
                        st.success("✅ Loaded into screener — scroll down to run screening!")

        except Exception as e:
            st.error(f"❌ Could not process file: {e}")

# Show data preview
if df_input is not None:
    with st.expander("👁️ Data Preview", expanded=False):
        st.dataframe(df_input.head(10), use_container_width=True)
        st.caption(f"Schema: {list(df_input.columns)}")

    # Check required columns
    required_cols = ['transaction_id', 'date', 'sender_name', 'sender_country',
                     'receiver_name', 'receiver_country', 'amount', 'currency']
    missing_cols = [c for c in required_cols if c not in df_input.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    st.markdown("---")
    st.markdown(f"**Ready to screen {len(df_input):,} transactions**")
    col1, col2 = st.columns([1, 3])
    with col1:
        run_screening = st.button("🚀 Run Screening", type="primary", use_container_width=True)
    with col2:
        st.caption(f"Thresholds: High Risk > {high_threshold:.0%} | Grey Zone > {grey_threshold:.0%}")

    if run_screening:
        try:
            from models.features import compute_features
            from models.predict import get_flag_reasons
            from utils.constants import FEATURE_NAMES

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Governance scores
            status_text.markdown("**Step 1/6 — Loading governance scores...**")
            all_countries = list(
                set(df_input['sender_country'].unique()) |
                set(df_input['receiver_country'].unique())
            )
            governance_scores = get_governance(tuple(all_countries))
            progress_bar.progress(10)

            # Step 2: Feature engineering (slowest — fuzzy matching)
            status_text.markdown("**Step 2/6 — Computing features (fuzzy matching sanctions lists)...**")
            features = compute_features(df_input, sanctions_df, governance_scores)
            progress_bar.progress(50)

            # Step 3: Impute + scale
            status_text.markdown("**Step 3/6 — Preprocessing features...**")
            X_imputed = imputer.transform(features)
            X_scaled = scaler.transform(X_imputed)
            progress_bar.progress(60)

            # Step 4: Model predictions
            status_text.markdown("**Step 4/6 — Running Logistic Regression & XGBoost...**")
            logit_proba = logit_model.predict_proba(X_scaled)[:, 1]
            xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1] if xgb_model else logit_proba
            composite = 0.4 * logit_proba + 0.6 * xgb_proba
            progress_bar.progress(75)

            # Step 5: Build results DataFrame
            status_text.markdown("**Step 5/6 — Computing risk levels...**")
            results = df_input.copy()
            results['logit_proba'] = logit_proba
            results['xgb_proba'] = xgb_proba
            results['composite_score'] = composite
            results['risk_level'] = pd.cut(
                composite,
                bins=[-np.inf, grey_threshold, high_threshold, np.inf],
                labels=['Low', 'Grey', 'High']
            )
            for col in ['sanctions_list_match_score', 'country_risk_score',
                        'sanctioned_country_flag', 'currency_mismatch', 'benford_deviation']:
                if col in features.columns:
                    results[col] = features[col].values
            progress_bar.progress(90)

            # Step 6: Flag reasons
            status_text.markdown("**Step 6/6 — Generating flag reasons...**")
            results['flag_reasons'] = results.apply(
                lambda row: get_flag_reasons(row, high_threshold), axis=1
            )
            progress_bar.progress(100)

            status_text.markdown(f"**✅ Screening complete — {len(results):,} transactions processed.**")
            st.session_state['screening_results'] = results

        except Exception as e:
            st.error(f"❌ Screening failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# Show results
if 'screening_results' in st.session_state:
    results = st.session_state['screening_results']

    st.markdown("---")
    st.markdown("### 📊 Screening Results")

    # Summary metrics
    total = len(results)
    high_risk = (results['risk_level'] == 'High').sum()
    grey_zone = (results['risk_level'] == 'Grey').sum()
    low_risk = (results['risk_level'] == 'Low').sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Screened", f"{total:,}")
    col2.metric("🔴 High Risk", f"{high_risk:,}", f"{high_risk/total:.1%}")
    col3.metric("🟡 Grey Zone", f"{grey_zone:,}", f"{grey_zone/total:.1%}")
    col4.metric("🟢 Low Risk", f"{low_risk:,}", f"{low_risk/total:.1%}")

    # Display columns
    display_cols = ['transaction_id', 'date', 'sender_name', 'receiver_name',
                    'receiver_country', 'amount', 'currency', 'composite_score',
                    'risk_level', 'flag_reasons']
    available_cols = [c for c in display_cols if c in results.columns]
    display_df = results[available_cols].copy()
    if 'composite_score' in display_df.columns:
        display_df['composite_score'] = display_df['composite_score'].round(4)

    # Color-coded display
    def color_risk(val):
        if val == 'High':
            return 'background-color: #ffcccc; color: #cc0000; font-weight: bold'
        elif val == 'Grey':
            return 'background-color: #fff3cc; color: #996600; font-weight: bold'
        elif val == 'Low':
            return 'background-color: #ccffcc; color: #006600'
        return ''

    st.dataframe(
        display_df.style.applymap(color_risk, subset=['risk_level']),
        use_container_width=True,
        height=500
    )

    # Export button
    high_risk_df = results[results['risk_level'] == 'High'][available_cols]
    if len(high_risk_df) > 0:
        csv_buffer = io.StringIO()
        high_risk_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"📥 Download High-Risk Transactions ({len(high_risk_df):,})",
            data=csv_buffer.getvalue(),
            file_name="high_risk_transactions.csv",
            mime="text/csv"
        )
else:
    if df_input is None:
        st.info("👆 Upload data or load demo data above, then click **Run Screening** to see results.")
    else:
        st.info("👆 Click **Run Screening** to analyze the loaded transactions.")
