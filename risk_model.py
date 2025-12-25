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
st.set_page_config(page_title="Professional Risk Meter", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown("""
<style>
.main-header {font-size: 2.5rem; font-weight: bold; background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
-webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.metric-box {background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 1.5rem; border-radius: 12px;
border: 1px solid rgba(102, 126, 234, 0.2); text-align: center;}
.alert-box {padding: 1rem; border-radius: 8px; margin: 1rem 0;}
.term-header {font-size: 1.3rem; font-weight: bold; color: #667eea; margin-top: 1.5rem;}
.term-desc {margin-left: 1rem; color: #e2e8f0;}
</style>
""", unsafe_allow_html=True)

# Multi-page navigation
page = st.sidebar.radio("Navigate", ["📊 Risk Meter Dashboard", "📖 Glossary & Explanations"])

# Data Fetcher with Auto-Market Detection and Robust Symbol Handling
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(symbol, period='1y', start_date=None, end_date=None):
    """Fetch data from yfinance and auto-detect market with multiple symbol attempts"""
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

# Risk Calculator
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
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
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
            'cvar_95': cvar_95,
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
    if score <= 20:
        return "Very Low", "🟢", "#10b981"
    elif score <= 35:
        return "Low", "🟢", "#059669"
    elif score <= 50:
        return "Moderate", "🟡", "#f59e0b"
    elif score <= 65:
        return "Elevated", "🟠", "#f97316"
    elif score <= 80:
        return "High", "🔴", "#ef4444"
    else:
        return "Very High", "🔴", "#dc2626"

def get_recommendation(m):
    if not m:
        return "HOLD", "🟡", "Insufficient data"
    score, sharpe, ret = m['risk_score'], m['sharpe'], m['annual_return']
    if score < 35 and sharpe > 1.5 and ret > 10:
        return "STRONG BUY", "🟢", "Excellent metrics"
    elif score < 50 and sharpe > 1.0 and ret > 5:
        return "BUY", "🟢", "Good profile"
    elif score < 60 and sharpe > 0.5:
        return "HOLD", "🟡", "Moderate risk"
    elif score < 75:
        return "REDUCE", "🟠", "Elevated risk"
    else:
        return "SELL", "🔴", "High risk"

def detect_patterns(df):
    patterns = []
    close = df['Close'].values
    if len(close) < 20:
        return ["Insufficient data"]
    recent = close[-20:]
    slope = np.polyfit(range(len(recent)), recent, 1)[0]
    if slope > close[-1] * 0.01:
        patterns.append("🚀 Strong Uptrend")
    elif slope < -close[-1] * 0.01:
        patterns.append("📉 Strong Downtrend")
    else:
        patterns.append("↔️ Sideways")
    current = close[-1]
    resistance = np.percentile(close[-30:], 95)
    support = np.percentile(close[-30:], 5)
    if current >= resistance * 0.98:
        patterns.append("🔴 Near Resistance")
    elif current <= support * 1.02:
        patterns.append("🟢 Near Support")
    if len(close) >= 30:
        recent_vol = np.std(close[-10:])
        hist_vol = np.std(close[-30:-10])
        if recent_vol > hist_vol * 1.5:
            patterns.append("⚡ High Volatility")
        elif recent_vol < hist_vol * 0.6:
            patterns.append("😴 Low Volatility")
    return patterns

# Initialize Session State
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
if page == "📊 Risk Meter Dashboard":
    st.markdown('<div class="main-header">🛡️ Professional Risk Meter - Unified</div>', unsafe_allow_html=True)
    st.markdown("Institutional-Grade Portfolio Risk Analytics | Auto-Detects US & India Markets")

    # Sidebar Settings (same as before)
    with st.sidebar:
        st.header("⚙️ Settings")
        period_options = {'1 Day': '1d', '5 Days': '5d', '1 Month': '1mo', '3 Months': '3mo',
                          '6 Months': '6mo', '1 Year': '1y', '2 Years': '2y', '5 Years': '5y',
                          'YTD': 'ytd', 'Max': 'max', 'Custom': 'custom'}
        selected_period = st.selectbox("Period", list(period_options.keys()), index=5)
        period = period_options[selected_period]
        
        if period == 'custom':
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", value=datetime.now())
            st.session_state.custom_start = start_date.strftime('%Y-%m-%d')
            st.session_state.custom_end = end_date.strftime('%Y-%m-%d')
        else:
            st.session_state.period = period
            st.session_state.custom_start = None
            st.session_state.custom_end = None
        
        rfr = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.5) / 100
        
        st.divider()
        st.subheader("💼 Portfolio")
        col1, col2 = st.columns([3, 1])
        new = col1.text_input("Add Symbol (e.g., AAPL or RELIANCE)").upper()
        if col2.button("➕") and new:
            if new not in st.session_state.portfolio:
                st.session_state.portfolio.append(new)
                st.rerun()
        
        if st.button("🔄 Refresh All"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("**Holdings:**")
        for sym in st.session_state.portfolio:
            col1, col2 = st.columns([4, 1])
            col1.text(f"📈 {sym}")
            if col2.button("❌", key=f"rm_{sym}"):
                st.session_state.portfolio.remove(sym)
                st.rerun()

    # Main Dashboard Code (unchanged from previous version)
    active_portfolio = st.session_state.portfolio
    if not active_portfolio:
        st.info("Add stocks to begin")
        st.stop()

    data, info, markets, metrics, used_symbols = {}, {}, {}, {}, {}
    prog = st.progress(0)
    stat = st.empty()
    start_date = st.session_state.custom_start
    end_date = st.session_state.custom_end
    period = st.session_state.period if not start_date else None
    failed_symbols = []

    for i, sym in enumerate(active_portfolio):
        stat.text(f"Analyzing {sym}... ({i+1}/{len(active_portfolio)})")
        prog.progress((i + 1) / len(active_portfolio))
        
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

    prog.empty()
    stat.empty()

    for sym in failed_symbols:
        st.warning(f"⚠️ {sym}: No data available. Try adding .NS or .BO suffix for Indian stocks.")

    if not data:
        st.error("No valid data fetched. Please check symbols and try again.")
        st.stop()

    # Portfolio Overview, Visualizations, Table, Detailed View (same as before)
    st.markdown("### 📊 Portfolio Dashboard")
    avg_risk = np.mean([m['risk_score'] for m in metrics.values()])
    avg_ret = np.mean([m['annual_return'] for m in metrics.values()])
    avg_sharpe = np.mean([m['sharpe'] for m in metrics.values()])
    avg_vol = np.mean([m['volatility'] for m in metrics.values()])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cat, emoji, color = categorize_risk(avg_risk)
        st.markdown(f"""<div class="metric-box" style="border-left: 4px solid {color};">
        <div style="font-size: 2.5rem; font-weight: bold; color: {color};">{int(avg_risk)}</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">Risk Score</div>
        <div style="margin-top: 0.5rem;">{emoji} {cat}</div></div>""", unsafe_allow_html=True)
    with col2:
        col_ret = "#10b981" if avg_ret > 0 else "#ef4444"
        st.markdown(f"""<div class="metric-box">
        <div style="font-size: 2rem; font-weight: bold; color: {col_ret};">{avg_ret:.1f}%</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">Annual Return</div></div>""", unsafe_allow_html=True)
    with col3:
        col_sharpe = "#10b981" if avg_sharpe > 1 else "#f59e0b"
        st.markdown(f"""<div class="metric-box">
        <div style="font-size: 2rem; font-weight: bold; color: {col_sharpe};">{avg_sharpe:.2f}</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">Sharpe Ratio</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-box">
        <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{avg_vol:.1f}%</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">Volatility</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Portfolio Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        scatter_data = [{'Symbol': sym, 'Risk': m['risk_score'], 'Return': m['annual_return'], 'Volatility': m['volatility']} 
                        for sym, m in metrics.items()]
        df_scatter = pd.DataFrame(scatter_data)
        if not df_scatter.empty:
            fig_scatter = px.scatter(df_scatter, x='Risk', y='Return', size='Volatility', color='Symbol',
                                     title='Risk-Return Scatter', template='plotly_dark')
            st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        sector_data = [info[sym]['sector'] for sym in info if info[sym]['sector'] != 'N/A']
        if len(set(sector_data)) > 1:
            fig_pie = px.pie(names=sector_data, title='Sector Distribution', template='plotly_dark')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Limited sector data available")

    st.markdown("---")
    st.markdown("### 📈 Individual Analysis")
    table = []
    for sym, m in metrics.items():
        rec, emoji, _ = get_recommendation(m)
        cat, _, _ = categorize_risk(m['risk_score'])
        table.append({
            'Symbol': sym,
            'Used Symbol': used_symbols.get(sym, sym),
            'Company': info.get(sym, {}).get('name', sym)[:30],
            'Market': markets.get(sym, 'US'),
            'Risk': int(m['risk_score']),
            'Category': cat,
            'Return': f"{m['annual_return']:.1f}%",
            'Volatility': f"{m['volatility']:.1f}%",
            'Sharpe': f"{m['sharpe']:.2f}",
            'Beta': f"{m['beta']:.2f}",
            'Max DD': f"{m['max_dd']:.1f}%",
            'Rec': f"{emoji} {rec}"
        })
    if table:
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔍 Detailed Analysis")
    valid_symbols = list(data.keys())
    if valid_symbols:
        sel = st.selectbox("Select stock", valid_symbols)
        if sel:
            df = data[sel]
            m = metrics[sel]
            inf = info.get(sel, {})
            mkt = markets.get(sel, 'US')
            used_sym = used_symbols.get(sel, sel)
            
            st.markdown(f"#### {inf.get('name', sel)} ({mkt}) - Using: {used_sym}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sector", inf.get('sector', 'N/A'))
            mcap = inf.get('market_cap', 0)
            currency_sym = "₹" if inf['currency'] == 'INR' else "$"
            col2.metric("Market Cap", f"{currency_sym}{mcap/1e9:.1f}B" if mcap > 0 else "N/A")
            col3.metric("Beta", f"{inf.get('beta', 1):.2f}")
            col4.metric("Total Return", f"{m['total_return']:.1f}%")

            tab1, tab2 = st.tabs(["Technical Charts", "Risk Charts"])
            with tab1:
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                    row_heights=[0.4, 0.2, 0.2, 0.2],
                                    subplot_titles=('Price Chart', 'Volume', 'RSI (14)', 'MACD'))
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                             low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                ma20 = df['Close'].rolling(20).mean()
                ma50 = df['Close'].rolling(50).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma20, name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=ma50, name='MA50', line=dict(color='blue', width=1.5)), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='gray'), row=2, col=1)
                rsi = calculate_rsi(df['Close'])
                fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                macd, signal = calculate_macd(df['Close'])
                fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='red')), row=4, col=1)
                fig.update_layout(height=800, template='plotly_dark', showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    returns = m['returns']
                    cum_ret = (1 + returns).cumprod()
                    running_max = cum_ret.expanding().max()
                    drawdown = (cum_ret - running_max) / running_max * 100
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name='Drawdown', fill='tozeroy', line=dict(color='red')))
                    fig_dd.update_layout(title='Drawdown Chart', template='plotly_dark', height=400)
                    st.plotly_chart(fig_dd, use_container_width=True)
                with col2:
                    fig_hist = px.histogram(returns * 100, nbins=50, title='Daily Returns Distribution', template='plotly_dark')
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**📊 Risk Metrics**")
                st.metric("Risk Score", int(m['risk_score']))
                st.metric("Volatility", f"{m['volatility']:.1f}%")
                st.metric("Max Drawdown", f"{m['max_dd']:.1f}%")
                st.metric("VaR (95%)", f"{m['var_95']:.2f}%")
                st.metric("Beta", f"{m['beta']:.2f}")
                st.markdown("---")
                st.markdown("**📈 Performance**")
                st.metric("Annual Return", f"{m['annual_return']:.1f}%")
                st.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
                st.metric("Sortino Ratio", f"{m['sortino']:.2f}")
                st.metric("Win Rate", f"{m['win_rate']:.1f}%")
                st.markdown("---")
                rec, emoji, reason = get_recommendation(m)
                st.info(f"{emoji} **{rec}**\n\n{reason}")

            st.markdown("---")
            st.markdown("**🔍 Detected Patterns**")
            patterns = detect_patterns(df)
            cols = st.columns(min(len(patterns), 4))
            for i, p in enumerate(patterns):
                cols[i % 4].info(p)

            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            curr_rsi = rsi.iloc[-1]
            rsi_stat = "🟢 Neutral" if 30 < curr_rsi < 70 else ("🔴 Overbought" if curr_rsi > 70 else "🟢 Oversold")
            col1.metric("RSI", f"{curr_rsi:.1f}", rsi_stat)
            curr_macd = macd.iloc[-1]
            macd_stat = "🟢 Bullish" if curr_macd > signal.iloc[-1] else "🔴 Bearish"
            col2.metric("MACD", f"{curr_macd:.2f}", macd_stat)
            curr_price = df['Close'].iloc[-1]
            col3.metric("Current Price", f"{currency_sym}{curr_price:.2f}")
            price_vs_ma = ((curr_price / ma20.iloc[-1]) - 1) * 100 if not ma20.empty else 0
            col4.metric("vs MA20", f"{price_vs_ma:+.1f}%")

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice. | Data via yfinance | Built with Streamlit")

# -----------------------------
# PAGE 2: Glossary & Explanations
# -----------------------------
else:
    st.markdown('<div class="main-header">📖 Glossary & Explanations</div>', unsafe_allow_html=True)
    st.markdown("### Understanding Key Risk & Performance Metrics")

    st.markdown("""
    <div class="term-header">Risk Score (0–100)</div>
    <div class="term-desc">A composite proprietary score combining volatility, drawdown, downside risk, beta deviation, and Sharpe penalty. Lower is safer (0 = extremely safe, 100 = extremely risky).</div>
    
    <div class="term-header">Volatility (Annualized)</div>
    <div class="term-desc">Measures how much the stock price fluctuates. Calculated as standard deviation of daily returns × √252. Higher values indicate greater price swings.</div>
    
    <div class="term-header">Annual Return</div>
    <div class="term-desc">Average annualized total return based on historical price changes.</div>
    
    <div class="term-header">Sharpe Ratio</div>
    <div class="term-desc">Risk-adjusted return: (Return − Risk-Free Rate) / Volatility. >1 is good, >2 is excellent.</div>
    
    <div class="term-header">Sortino Ratio</div>
    <div class="term-desc">Similar to Sharpe but only penalizes downside volatility (negative returns).</div>
    
    <div class="term-header">Max Drawdown</div>
    <div class="term-desc">Largest peak-to-trough decline in portfolio value. Shows worst-case historical loss.</div>
    
    <div class="term-header">Value at Risk (VaR 95%)</div>
    <div class="term-desc">Worst expected daily loss with 95% confidence (i.e., 5% chance of losing more than this in a day).</div>
    
    <div class="term-header">Beta</div>
    <div class="term-desc">Measures stock sensitivity to market movements. Beta = 1 moves with market; >1 more volatile; <1 less volatile.</div>
    
    <div class="term-header">Alpha</div>
    <div class="term-desc">Excess return above what is expected given the stock’s beta (outperformance vs. benchmark).</div>
    
    <div class="term-header">Win Rate</div>
    <div class="term-desc">Percentage of trading days with positive returns.</div>
    
    <div class="term-header">RSI (Relative Strength Index)</div>
    <div class="term-desc">Momentum indicator (0–100). >70 = overbought (possible pullback), <30 = oversold (possible bounce).</div>
    
    <div class="term-header">MACD</div>
    <div class="term-desc">Trend-following momentum indicator showing relationship between two moving averages. Bullish when MACD > Signal line.</div>
    
    <div class="term-header">Moving Averages (MA20, MA50)</div>
    <div class="term-desc">Average closing price over 20 or 50 days. Price above MA indicates uptrend.</div>
    """, unsafe_allow_html=True)

    st.markdown("### Risk Categories")
    st.markdown("""
    - **Very Low (0–20)** – Extremely stable assets  
    - **Low (21–35)** – Safe with good return potential  
    - **Moderate (36–50)** – Balanced risk-reward  
    - **Elevated (51–65)** – Higher volatility, caution advised  
    - **High (66–80)** – Significant risk  
    - **Very High (81–100)** – Extremely risky, speculative
    """)

    st.markdown("### Recommendation Logic")
    st.markdown("""
    - **STRONG BUY** – Very low risk + excellent risk-adjusted returns  
    - **BUY** – Low risk + solid returns  
    - **HOLD** – Moderate risk  
    - **REDUCE** – Elevated risk  
    - **SELL** – High risk profile
    """)

    st.caption("All metrics are historical and for educational purposes only. Past performance is not indicative of future results.")