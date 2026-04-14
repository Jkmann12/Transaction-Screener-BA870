"""
Page 5: Counterparty Lookup
BA775 Financial Analytics | Boston University

Look up a publicly traded counterparty via yfinance and assess whether its
financial profile is consistent with transaction volumes observed in screened data.

Lecture 2 Connection: Applies Du Pont / ratio analysis to assess counterparty
financial health, mirroring the framework from Lecture 2. Revenue-to-transaction
volume mismatch detection parallels Beneish M-Score ratio anomaly detection.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import warnings
warnings.filterwarnings('ignore')

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_ROOT)

st.set_page_config(page_title="Counterparty Lookup", page_icon="🔎", layout="wide")

st.title("🔎 Counterparty Financial Lookup")
st.markdown(
    "Look up a publicly traded counterparty's financial profile and assess whether "
    "its financial scale is consistent with the transaction volumes in the screened data."
)

st.info(
    "**📊 Lecture 2 Connection:** This page applies **Du Pont / ratio analysis** to assess counterparty "
    "financial health. Revenue, total assets, current ratio, and debt-to-equity are computed from "
    "live yfinance data — mirroring the financial ratio framework from Lecture 2. "
    "Large mismatches between a company's revenue scale and its transaction volume are a red flag, "
    "similar to how the **Beneish M-Score** (Lecture 4) uses ratio anomalies to detect manipulation."
)

# yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance not installed. Install with: pip install yfinance")

st.markdown("---")

# Ticker input
col1, col2 = st.columns([2, 1])
with col1:
    ticker_input = st.text_input(
        "Enter Stock Ticker Symbol",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, JPM, GS",
        help="Enter any publicly traded company's ticker symbol"
    ).upper().strip()
with col2:
    lookup_btn = st.button("🔍 Look Up", type="primary", use_container_width=True)

if lookup_btn and ticker_input:
    if not YFINANCE_AVAILABLE:
        st.error("yfinance is not available. Please install it.")
        st.stop()

    with st.spinner(f"Fetching financial data for {ticker_input}..."):
        try:
            ticker = yf.Ticker(ticker_input)
            info = ticker.info or {}

            # Company overview
            st.markdown(f"### 🏢 {info.get('longName', ticker_input)}")
            st.caption(
                f"{info.get('sector', 'N/A')} | "
                f"{info.get('industry', 'N/A')} | "
                f"Country: {info.get('country', 'N/A')}"
            )

            # Key financial ratios (Du Pont framework)
            st.markdown("#### 📊 Key Financial Ratios (Du Pont Analysis)")
            col1, col2, col3, col4, col5 = st.columns(5)

            market_cap = info.get('marketCap', 0) or 0
            revenue = info.get('totalRevenue', 0) or 0
            total_assets = info.get('totalAssets', 0) or 0
            current_ratio = info.get('currentRatio', None)
            debt_equity = info.get('debtToEquity', None)

            col1.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap > 0 else "N/A")
            col2.metric("Annual Revenue", f"${revenue/1e9:.2f}B" if revenue > 0 else "N/A")
            col3.metric("Total Assets", f"${total_assets/1e9:.2f}B" if total_assets > 0 else "N/A")
            col4.metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "N/A",
                        help="Current Assets / Current Liabilities — Lecture 2: liquidity ratio")
            col5.metric("Debt-to-Equity", f"{debt_equity:.2f}" if debt_equity else "N/A",
                        help="Total Debt / Equity — Lecture 2: leverage ratio")

            # Additional ratios
            st.markdown("#### 📋 Additional Metrics")
            col1, col2, col3, col4 = st.columns(4)
            pe_ratio = info.get('trailingPE', None)
            roe = info.get('returnOnEquity', None)
            roa = info.get('returnOnAssets', None)
            profit_margin = info.get('profitMargins', None)

            col1.metric("P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio else "N/A")
            col2.metric("Return on Equity", f"{roe*100:.1f}%" if roe else "N/A",
                        help="Lecture 2: Du Pont ROE decomposition")
            col3.metric("Return on Assets", f"{roa*100:.1f}%" if roa else "N/A")
            col4.metric("Profit Margin", f"{profit_margin*100:.1f}%" if profit_margin else "N/A")

            # Screening results comparison
            if 'screening_results' in st.session_state:
                st.markdown("---")
                st.markdown("#### 🔍 Transaction Volume vs Revenue Comparison")

                results = st.session_state['screening_results']
                company_name = info.get('longName', ticker_input)

                # Search for matching entities (fuzzy, simplified)
                name_lower = company_name.lower()
                ticker_lower = ticker_input.lower()

                # Check both sender and receiver
                mask = (
                    results['receiver_name'].str.lower().str.contains(ticker_lower, na=False) |
                    results['sender_name'].str.lower().str.contains(ticker_lower, na=False)
                )
                matched_txns = results[mask]

                total_txn_volume = matched_txns['amount'].sum() if len(matched_txns) > 0 else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Matched Transactions", f"{len(matched_txns):,}")
                col2.metric("Total Transaction Volume", f"${total_txn_volume:,.0f}")
                col3.metric("Annual Revenue", f"${revenue:,.0f}" if revenue > 0 else "N/A")

                if revenue > 0 and total_txn_volume > 0:
                    ratio = total_txn_volume / revenue
                    st.metric("Transaction Volume / Revenue Ratio", f"{ratio:.4f}",
                              delta="⚠️ MISMATCH" if ratio > 0.1 else "✅ Normal")
                    if ratio > 0.1:
                        st.error(
                            f"⚠️ **Revenue-Transaction Mismatch Detected!** "
                            f"Transaction volume (${total_txn_volume:,.0f}) represents "
                            f"{ratio:.1%} of this company's annual revenue (${revenue:,.0f}). "
                            "High ratios may indicate the entity is being used as a conduit "
                            "for transactions disproportionate to its financial scale — a key "
                            "red flag in FATF typologies and Beneish M-Score analysis."
                        )
                    else:
                        st.success("✅ Transaction volume is consistent with the company's revenue scale.")
                elif len(matched_txns) == 0:
                    st.info(f"No transactions found matching '{ticker_input}' or '{company_name}' in screened data.")

            # Beta calculation (CAPM stretch goal)
            st.markdown("---")
            st.markdown("#### 📉 Beta & Risk Profile (CAPM — Lecture 5 Stretch)")

            try:
                import yfinance as yf2
                hist = ticker.history(period='1y')
                spy = yf2.Ticker('SPY').history(period='1y')

                if len(hist) > 50 and len(spy) > 50:
                    stock_returns = hist['Close'].pct_change().dropna()
                    market_returns = spy['Close'].pct_change().dropna()

                    # Align on common dates
                    common_dates = stock_returns.index.intersection(market_returns.index)
                    if len(common_dates) > 30:
                        r_stock = stock_returns.loc[common_dates]
                        r_market = market_returns.loc[common_dates]
                        cov = np.cov(r_stock, r_market)[0, 1]
                        var_market = np.var(r_market)
                        beta = cov / var_market if var_market > 0 else info.get('beta', None)

                        col1, col2 = st.columns(2)
                        col1.metric("Calculated Beta (1Y)", f"{beta:.3f}",
                                    help="CAPM beta: covariance(stock, market) / variance(market)")
                        col2.metric("Beta (from yfinance)", f"{info.get('beta', 'N/A')}")

                        if beta > 1.5:
                            st.warning(f"⚠️ High beta ({beta:.2f}) indicates this stock is significantly more volatile than the market.")
                        elif beta < 0.5:
                            st.info(f"ℹ️ Low beta ({beta:.2f}) — defensive stock with below-market volatility.")
                        else:
                            st.success(f"✅ Beta ({beta:.2f}) — market-like volatility.")
            except Exception:
                beta_val = info.get('beta', None)
                if beta_val:
                    st.metric("Beta (from yfinance)", f"{beta_val:.3f}")
                else:
                    st.caption("Beta calculation requires price history data.")

            # Financial statements
            st.markdown("---")
            with st.expander("📄 Income Statement", expanded=False):
                try:
                    financials = ticker.financials
                    if financials is not None and not financials.empty:
                        st.dataframe(financials.round(0), use_container_width=True)
                    else:
                        st.info("Financial statements not available for this ticker.")
                except Exception as e:
                    st.info(f"Could not load income statement: {e}")

            with st.expander("📄 Balance Sheet", expanded=False):
                try:
                    balance = ticker.balance_sheet
                    if balance is not None and not balance.empty:
                        st.dataframe(balance.round(0), use_container_width=True)
                    else:
                        st.info("Balance sheet not available for this ticker.")
                except Exception as e:
                    st.info(f"Could not load balance sheet: {e}")

        except Exception as e:
            st.error(f"❌ Could not fetch data for '{ticker_input}': {e}")
            st.caption("Make sure the ticker symbol is correct and you have an internet connection.")

elif lookup_btn and not ticker_input:
    st.warning("Please enter a ticker symbol.")
else:
    st.info("👆 Enter a stock ticker symbol above and click **Look Up**.")
    st.markdown("**Example tickers:** AAPL (Apple), JPM (JPMorgan), GS (Goldman Sachs), HSBC, BAC (Bank of America)")
