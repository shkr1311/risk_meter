"""
Professional Risk Meter - Unified Edition
Real-time portfolio risk analysis using yfinance
Supports both US and Indian stocks automatically
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    st.error("Install yfinance: pip install yfinance")
    st.stop()

# Page Setup
st.set_page_config(page_title="Professional Risk Meter", layout="wide", initial_sidebar_state="expanded")

# CSS for clean, professional look and alignment
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: bold;
        color: #e2e8f0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    .term-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #667eea;
        margin-top: 1.5rem;
    }
    .term-desc {
        margin-left: 1rem;
        color: #e2e8f0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Multi-page navigation
page = st.sidebar.radio("Navigate", ["Risk Meter Dashboard", "Glossary & Explanations"])

# Data Fetcher
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(symbol, period='1y', start_date=None, end_date=None):
    attempts = [
        symbol,
        f"{symbol}.NS" if not symbol.endswith(('.NS', '.BO')) else None,
        f"{symbol}.BO" if not symbol.endswith(('.NS', '.BO')) else None,
        symbol.replace('.NS', '').replace('.BO', '')
    ]
    attempts = [a for a in attempts if a is not None]
    
    for sym in attempts:
        try:
            ticker = yf.Ticker(sym)
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                df = ticker.history(period=period, auto_adjust=True, raise_errors=False)
            
            if df is not None and not df.empty and len(df) > 0:
                info_data = ticker.info
                currency = info_data.get('currency', 'USD') if isinstance(info_data, dict) else 'USD'
                market = 'India' if currency == 'INR' else 'US'
                
                info = {
                    'name': info_data.get('longName', sym) if isinstance(info_data, dict) else sym,
                    'sector': info_data.get('sector', 'N/A') if isinstance(info_data, dict) else 'N/A',
                    'beta': info_data.get('beta', 1.0) if isinstance(info_data, dict) else 1.0,
                    'market_cap': info_data.get('marketCap', 0) if isinstance(info_data, dict) else 0,
                    'currency': currency
                }
                return df, info, market, sym
        except Exception:
            continue
    return None, None, None, None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market(period='1y', market='US', start_date=None, end_date=None):
    try:
        index = '^NSEI' if market == 'India' else '^GSPC'
        ticker = yf.Ticker(index)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        else:
            df = ticker.history(period=period, auto_adjust=True)
        return df if not df.empty else None
    except:
        return None

# Technical Indicators
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

# Risk Metrics Calculation
def calculate_metrics(df, market_df=None, rfr=0.02):
    try:
        close = df['Close']
        returns = close.pct_change().dropna()
        if len(returns) < 10:
            return None
        
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100
        avg_return = returns.mean() * 252
        annual_return = avg_return * 100
        
        excess = avg_return - rfr
        sharpe = excess / (daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = excess / downside if downside > 0 else 0
        
        cum_ret = (1 + returns).cumprod()
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        var_95 = np.percentile(returns, 5) * 100
        
        beta = 1.0
        alpha = 0.0
        if market_df is not None:
            try:
                mkt_ret = market_df['Close'].pct_change().dropna()
                common = returns.index.intersection(mkt_ret.index)
                if len(common) > 30:
                    stock_r = returns.loc[common]
                    market_r = mkt_ret.loc[common]
                    cov = stock_r.cov(market_r)
                    var = market_r.var()
                    beta = cov / var if var > 0 else 1.0
                    mkt_return = market_r.mean() * 252
                    expected = rfr + beta * (mkt_return - rfr)
                    alpha = (avg_return - expected) * 100
            except:
                pass
        
        vol_score = min(annual_vol * 1.2, 30)
        dd_score = min(abs(max_dd) * 0.7, 25)
        var_score = min(abs(var_95) * 2, 20)
        beta_score = min(abs(beta - 1) * 12, 15)
        sharpe_penalty = max(0, (1 - sharpe) * 10)
        risk_score = min(vol_score + dd_score + var_score + beta_score + sharpe_penalty, 100)
        
        return {
            'volatility': annual_vol,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_dd': max_dd,
            'var_95': var_95,
            'beta': beta,
            'alpha': alpha,
            'risk_score': risk_score,
            'total_return': ((close.iloc[-1] / close.iloc[0]) - 1) * 100,
            'win_rate': (returns > 0).sum() / len(returns) * 100,
            'returns': returns
        }
    except Exception:
        return None

def categorize_risk(score):
    if score <= 20: return "Very Low", "#10b981"
    elif score <= 35: return "Low", "#059669"
    elif score <= 50: return "Moderate", "#f59e0b"
    elif score <= 65: return "Elevated", "#f97316"
    elif score <= 80: return "High", "#ef4444"
    else: return "Very High", "#dc2626"

def get_recommendation(m):
    if not m: return "HOLD", "Insufficient data"
    score, sharpe, ret = m['risk_score'], m['sharpe'], m['annual_return']
    if score < 35 and sharpe > 1.5 and ret > 10: return "STRONG BUY", "Excellent metrics"
    elif score < 50 and sharpe > 1.0 and ret > 5: return "BUY", "Good profile"
    elif score < 60 and sharpe > 0.5: return "HOLD", "Moderate risk"
    elif score < 75: return "REDUCE", "Elevated risk"
    else: return "SELL", "High risk"

def detect_patterns(df):
    patterns = []
    close = df['Close'].values
    if len(close) < 20: return ["Insufficient data"]
    recent = close[-20:]
    slope = np.polyfit(range(len(recent)), recent, 1)[0]
    if slope > close[-1] * 0.01: patterns.append("Strong Uptrend")
    elif slope < -close[-1] * 0.01: patterns.append("Strong Downtrend")
    else: patterns.append("Sideways")
    current = close[-1]
    resistance = np.percentile(close[-30:], 95)
    support = np.percentile(close[-30:], 5)
    if current >= resistance * 0.98: patterns.append("Near Resistance")
    elif current <= support * 1.02: patterns.append("Near Support")
    if len(close) >= 30:
        recent_vol = np.std(close[-10:])
        hist_vol = np.std(close[-30:-10])
        if recent_vol > hist_vol * 1.5: patterns.append("High Volatility")
        elif recent_vol < hist_vol * 0.6: patterns.append("Low Volatility")
    return patterns

# Session State Initialization
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = ['AAPL', 'MSFT', 'GOOGL', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS']
if 'period' not in st.session_state:
    st.session_state.period = '1y'
if 'custom_start' not in st.session_state:
    st.session_state.custom_start = None
if 'custom_end' not in st.session_state:
    st.session_state.custom_end = None

# -----------------------------
# PAGE 1: Risk Meter Dashboard
# -----------------------------
if page == "Risk Meter Dashboard":
    st.markdown('<div class="main-header">Professional Risk Meter</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Institutional-Grade Portfolio Risk Analytics | US & India Markets</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        period_options = {
            '1 Day': '1d', '5 Days': '5d', '1 Month': '1mo', '3 Months': '3mo',
            '6 Months': '6mo', '1 Year': '1y', '2 Years': '2y', '5 Years': '5y',
            'YTD': 'ytd', 'Max': 'max', 'Custom': 'custom'
        }
        selected_period = st.selectbox("Analysis Period", list(period_options.keys()), index=5)
        period = period_options[selected_period]
        
        if period == 'custom':
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
            st.session_state.custom_start = start_date.strftime('%Y-%m-%d')
            st.session_state.custom_end = end_date.strftime('%Y-%m-%d')
        else:
            st.session_state.period = period
            st.session_state.custom_start = None
            st.session_state.custom_end = None
        
        rfr = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.5) / 100
        
        st.divider()
        st.subheader("Portfolio Management")
        new_symbol = st.text_input("Add Stock Symbol", placeholder="e.g., AAPL or RELIANCE").upper()
        
        col_add, col_refresh = st.columns([1, 4])
        with col_add:
            if st.button("➕") and new_symbol.strip():
                if new_symbol not in st.session_state.portfolio:
                    st.session_state.portfolio.append(new_symbol)
                    st.success(f"{new_symbol} added")
                    st.rerun()
        with col_refresh:
            if st.button("🔄 Refresh All Data"):
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("**Current Holdings**")
        for sym in st.session_state.portfolio[:]:
            col_sym, col_remove = st.columns([4, 1])
            col_sym.write(sym)
            if col_remove.button("✖", key=f"rm_{sym}"):
                st.session_state.portfolio.remove(sym)
                st.rerun()

    # Main Dashboard Logic
    if not st.session_state.portfolio:
        st.info("Add at least one stock symbol to begin analysis.")
        st.stop()

    data, info, markets, metrics, used_symbols = {}, {}, {}, {}, {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    failed_symbols = []
    
    start_date = st.session_state.custom_start
    end_date = st.session_state.custom_end
    period = st.session_state.period if not start_date else None

    for i, sym in enumerate(st.session_state.portfolio):
        status_text.text(f"Fetching {sym}... ({i+1}/{len(st.session_state.portfolio)})")
        progress_bar.progress((i + 1) / len(st.session_state.portfolio))
        
        df, inf, mkt, used_sym = fetch_data(sym, period, start_date, end_date)
        if df is not None and len(df) > 10:
            data[sym] = df
            info[sym] = inf
            markets[sym] = mkt
            used_symbols[sym] = used_sym
            market_df = fetch_market(period, mkt, start_date, end_date)
            m = calculate_metrics(df, market_df, rfr)
            if m:
                metrics[sym] = m
        else:
            failed_symbols.append(sym)

    progress_bar.empty()
    status_text.empty()

    if failed_symbols:
        st.warning(f"Failed to fetch: {', '.join(failed_symbols)}. Try adding .NS or .BO for Indian stocks.")

    if not metrics:
        st.error("No valid data. Check symbols and try again.")
        st.stop()

    # Portfolio Overview
    st.markdown('<div class="section-header">Portfolio Overview</div>', unsafe_allow_html=True)
    avg_risk = np.mean([m['risk_score'] for m in metrics.values()])
    avg_ret = np.mean([m['annual_return'] for m in metrics.values()])
    avg_sharpe = np.mean([m['sharpe'] for m in metrics.values()])
    avg_vol = np.mean([m['volatility'] for m in metrics.values()])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cat, color = categorize_risk(avg_risk)
        st.markdown(f"""
        <div class="metric-box" style="border-left: 5px solid {color};">
            <div class="metric-value" style="color: {color};">{int(avg_risk)}</div>
            <div class="metric-label">Risk Score</div>
            <div style="margin-top: 0.5rem;">{cat}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        ret_color = "#10b981" if avg_ret > 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {ret_color};">{avg_ret:+.1f}%</div>
            <div class="metric-label">Annual Return</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        sharpe_color = "#10b981" if avg_sharpe > 1 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {sharpe_color};">{avg_sharpe:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #667eea;">{avg_vol:.1f}%</div>
            <div class="metric-label">Volatility</div>
        </div>
        """, unsafe_allow_html=True)

    # Portfolio Visualizations
    st.markdown('<div class="section-header">Portfolio Visualizations</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        scatter_data = [{'Symbol': sym, 'Risk': m['risk_score'], 'Return': m['annual_return'], 'Volatility': m['volatility']} 
                        for sym, m in metrics.items()]
        df_scatter = pd.DataFrame(scatter_data)
        if not df_scatter.empty:
            fig = px.scatter(df_scatter, x='Risk', y='Return', size='Volatility', color='Symbol',
                             title='Risk vs Return', template='plotly_dark')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        sectors = [info[sym]['sector'] for sym in info if info[sym]['sector'] != 'N/A']
        if sectors and len(set(sectors)) > 1:
            sector_counts = pd.Series(sectors).value_counts()
            fig = px.pie(values=sector_counts.values, names=sector_counts.index, title='Sector Allocation', template='plotly_dark')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Limited or uniform sector data")

    # Individual Analysis Table
    st.markdown('<div class="section-header">Individual Stock Analysis</div>', unsafe_allow_html=True)
    table_data = []
    for sym, m in metrics.items():
        rec, _ = get_recommendation(m)
        cat, _ = categorize_risk(m['risk_score'])
        table_data.append({
            'Symbol': sym,
            'Company': info[sym]['name'][:25],
            'Market': markets[sym],
            'Risk Score': int(m['risk_score']),
            'Category': cat,
            'Annual Return': f"{m['annual_return']:+.1f}%",
            'Volatility': f"{m['volatility']:.1f}%",
            'Sharpe': f"{m['sharpe']:.2f}",
            'Beta': f"{m['beta']:.2f}",
            'Max Drawdown': f"{m['max_dd']:.1f}%",
            'Recommendation': rec
        })
    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Detailed Stock View
    st.markdown('<div class="section-header">Detailed Stock Analysis</div>', unsafe_allow_html=True)
    if metrics:
        selected = st.selectbox("Select stock", options=list(metrics.keys()))
        if selected:
            df = data[selected]
            m = metrics[selected]
            inf = info[selected]
            mkt = markets[selected]
            used = used_symbols.get(selected, selected)
            
            st.subheader(f"{inf['name']} ({mkt})")
            st.caption(f"Data source: {used}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sector", inf['sector'])
            currency = "₹" if inf['currency'] == 'INR' else "$"
            cap = inf['market_cap']
            c2.metric("Market Cap", f"{currency}{cap/1e9:.1f}B" if cap > 0 else "N/A")
            c3.metric("Beta", f"{m['beta']:.2f}")
            c4.metric("Total Return", f"{m['total_return']:+.1f}%")

            tab1, tab2 = st.tabs(["Technical Analysis", "Risk Analysis"])
            
            with tab1:
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                    row_heights=[0.5, 0.2, 0.15, 0.15],
                                    subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'))
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                             low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                ma20 = df['Close'].rolling(20).mean()
                ma50 = df['Close'].rolling(50).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma20, name='MA20', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=ma50, name='MA50', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
                rsi = calculate_rsi(df['Close'])
                fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                macd_line, signal_line = calculate_macd(df['Close'])
                fig.add_trace(go.Scatter(x=df.index, y=macd_line, name='MACD', line=dict(color='blue')), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal', line=dict(color='red')), row=4, col=1)
                fig.update_layout(height=800, template='plotly_dark', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    returns = m['returns']
                    cum_ret = (1 + returns).cumprod()
                    peak = cum_ret.expanding().max()
                    drawdown = (cum_ret - peak) / peak * 100
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', fillcolor='red', line=dict(color='darkred')))
                    fig_dd.update_layout(title='Historical Drawdown', template='plotly_dark', height=400)
                    st.plotly_chart(fig_dd, use_container_width=True)
                with col2:
                    fig_hist = px.histogram(x=returns * 100, nbins=50, title='Daily Returns Distribution', template='plotly_dark')
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Key Risk Metrics**")
                st.metric("Risk Score", int(m['risk_score']))
                st.metric("Annual Volatility", f"{m['volatility']:.1f}%")
                st.metric("Max Drawdown", f"{m['max_dd']:.1f}%")
                st.metric("VaR (95%)", f"{m['var_95']:.2f}%")
                st.metric("Beta", f"{m['beta']:.2f}")
                st.markdown("**Performance Metrics**")
                st.metric("Annual Return", f"{m['annual_return']:+.1f}%")
                st.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
                st.metric("Sortino Ratio", f"{m['sortino']:.2f}")
                st.metric("Win Rate", f"{m['win_rate']:.1f}%")
                rec, reason = get_recommendation(m)
                st.success(f"**Recommendation: {rec}**\n\n{reason}")
            
            with col2:
                st.markdown("**Current Technical Signals**")
                patterns = detect_patterns(df)
                for pattern in patterns:
                    st.info(pattern)
                
                col_a, col_b, col_c, col_d = st.columns(4)
                curr_rsi = rsi.iloc[-1]
                rsi_status = "Neutral" if 30 < curr_rsi < 70 else ("Overbought" if curr_rsi > 70 else "Oversold")
                col_a.metric("RSI (14)", f"{curr_rsi:.1f}", rsi_status)
                macd_val = macd_line.iloc[-1]
                macd_status = "Bullish" if macd_val > signal_line.iloc[-1] else "Bearish"
                col_b.metric("MACD", f"{macd_val:.3f}", macd_status)
                price = df['Close'].iloc[-1]
                col_c.metric("Current Price", f"{currency}{price:.2f}")
                ma20_val = ma20.iloc[-1] if not ma20.empty else price
                vs_ma = ((price / ma20_val) - 1) * 100
                col_d.metric("vs MA20", f"{vs_ma:+.1f}%")

    st.markdown("---")
    st.caption("For educational purposes only. Not financial advice. Data via yfinance. Built with Streamlit.")

# -----------------------------
# PAGE 2: Glossary & Explanations
# -----------------------------
else:
    st.markdown('<div class="main-header">Glossary & Explanations</div>', unsafe_allow_html=True)
    st.markdown("### Key Concepts and Metrics Explained")

    st.markdown("""
    <div class="term-header">Risk Score (0–100)</div>
    <div class="term-desc">Composite score based on volatility, drawdown, downside risk, beta, and Sharpe. Lower = safer.</div>

    <div class="term-header">Annual Volatility</div>
    <div class="term-desc">Annualized standard deviation of returns. Measures price fluctuation.</div>

    <div class="term-header">Sharpe Ratio</div>
    <div class="term-desc">Return per unit of risk. >1 good, >2 excellent.</div>

    <div class="term-header">Sortino Ratio</div>
    <div class="term-desc">Like Sharpe, but only penalizes downside volatility.</div>

    <div class="term-header">Max Drawdown</div>
    <div class="term-desc">Largest historical peak-to-trough decline.</div>

    <div class="term-header">VaR (95%)</div>
    <div class="term-desc">Worst expected daily loss with 95% confidence.</div>

    <div class="term-header">Beta</div>
    <div class="term-desc">Sensitivity to market movements. 1 = market pace.</div>

    <div class="term-header">Alpha</div>
    <div class="term-desc">Excess return beyond market expectation.</div>

    <div class="term-header">RSI</div>
    <div class="term-desc">Momentum indicator. >70 overbought, <30 oversold.</div>

    <div class="term-header">MACD</div>
    <div class="term-desc">Trend-following momentum from moving averages.</div>
    """, unsafe_allow_html=True)

    st.markdown("### Risk Categories")
    st.markdown("""
    - Very Low (0–20): Extremely stable  
    - Low (21–35): Safe with potential  
    - Moderate (36–50): Balanced  
    - Elevated (51–65): Higher volatility  
    - High (66–80): Significant risk  
    - Very High (81–100): Speculative
    """)

    st.caption("Historical metrics only. Past performance ≠ future results.")