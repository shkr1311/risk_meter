"""
Professional Risk Meter - Production Ready
Real-time portfolio risk analysis using yfinance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    st.error("Install yfinance: pip install yfinance")
    st.stop()

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False

# Page Setup
st.set_page_config(page_title="Professional Risk Meter", page_icon="🛡️", layout="wide")

# CSS
st.markdown("""
<style>
.main-header {font-size: 2.5rem; font-weight: bold; background: linear-gradient(120deg, #667eea 0%, #764ba2 100%); 
-webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.metric-box {background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 1.5rem; border-radius: 12px; 
border: 1px solid rgba(102, 126, 234, 0.2); text-align: center;}
</style>
""", unsafe_allow_html=True)

# Data Fetcher
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(symbol, period='1y'):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)
        if df.empty:
            return None, None
        info = ticker.info
        return df, {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'beta': info.get('beta', 1.0),
            'market_cap': info.get('marketCap', 0)
        }
    except:
        return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market(period='1y'):
    try:
        return yf.Ticker('^GSPC').history(period=period, auto_adjust=True)
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
        
        # Core metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100
        avg_return = returns.mean() * 252
        annual_return = avg_return * 100
        
        # Sharpe Ratio
        excess = avg_return - rfr
        sharpe = excess / (daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        
        # Sortino Ratio
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = excess / downside if downside > 0 else 0
        
        # Maximum Drawdown
        cum_ret = (1 + returns).cumprod()
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Beta
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
        
        # Risk Score (0-100)
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
            'win_rate': (returns > 0).sum() / len(returns) * 100
        }
    except Exception as e:
        st.error(f"Metric error: {e}")
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

# Pattern Detection
def detect_patterns(df):
    patterns = []
    close = df['Close'].values
    
    if len(close) < 20:
        return ["Insufficient data"]
    
    # Trend
    recent = close[-20:]
    slope = np.polyfit(range(len(recent)), recent, 1)[0]
    if slope > 1:
        patterns.append("🚀 Strong Uptrend")
    elif slope < -1:
        patterns.append("📉 Strong Downtrend")
    else:
        patterns.append("↔️ Sideways")
    
    # Support/Resistance
    current = close[-1]
    resistance = np.percentile(close[-30:], 95)
    support = np.percentile(close[-30:], 5)
    
    if current >= resistance * 0.98:
        patterns.append("🔴 Near Resistance")
    elif current <= support * 1.02:
        patterns.append("🟢 Near Support")
    
    # Volatility
    if len(close) >= 30:
        recent_vol = np.std(close[-10:])
        hist_vol = np.std(close[-30:-10])
        if recent_vol > hist_vol * 1.5:
            patterns.append("⚡ High Volatility")
        elif recent_vol < hist_vol * 0.6:
            patterns.append("😴 Low Volatility")
    
    return patterns

# Initialize
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
if 'period' not in st.session_state:
    st.session_state.period = '1y'

# Header
st.markdown('<div class="main-header">🛡️ Professional Risk Meter</div>', unsafe_allow_html=True)
st.markdown("Institutional-Grade Portfolio Risk Analytics")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    periods = {'1M': '1mo', '3M': '3mo', '6M': '6mo', '1Y': '1y', '2Y': '2y', '5Y': '5y'}
    period = st.selectbox("Period", list(periods.keys()), index=3)
    st.session_state.period = periods[period]
    
    rfr = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.5) / 100
    
    st.divider()
    st.subheader("💼 Portfolio")
    
    new = st.text_input("Add Symbol").upper()
    if st.button("➕ Add") and new:
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

# Main
if not st.session_state.portfolio:
    st.info("Add stocks to begin")
    st.stop()

# Fetch data
market = fetch_market(st.session_state.period)
data, info, metrics = {}, {}, {}

prog = st.progress(0)
stat = st.empty()

for i, sym in enumerate(st.session_state.portfolio):
    stat.text(f"Analyzing {sym}... ({i+1}/{len(st.session_state.portfolio)})")
    prog.progress((i + 1) / len(st.session_state.portfolio))
    
    df, inf = fetch_data(sym, st.session_state.period)
    if df is not None and len(df) > 10:
        data[sym] = df
        info[sym] = inf
        m = calculate_metrics(df, market, rfr)
        if m:
            metrics[sym] = m
    else:
        st.warning(f"⚠️ {sym}: No data")

prog.empty()
stat.empty()

if not data:
    st.error("No valid data")
    st.stop()

# Portfolio Overview
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

# Stock Table
st.markdown("---")
st.markdown("### 📈 Individual Analysis")

table = []
for sym, m in metrics.items():
    rec, emoji, _ = get_recommendation(m)
    cat, _, _ = categorize_risk(m['risk_score'])
    table.append({
        'Symbol': sym,
        'Company': info.get(sym, {}).get('name', sym)[:25],
        'Risk': int(m['risk_score']),
        'Category': cat,
        'Return': f"{m['annual_return']:.1f}%",
        'Volatility': f"{m['volatility']:.1f}%",
        'Sharpe': f"{m['sharpe']:.2f}",
        'Beta': f"{m['beta']:.2f}",
        'Max DD': f"{m['max_dd']:.1f}%",
        'Rec': f"{emoji} {rec}"
    })

st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

# Detailed View
st.markdown("---")
st.markdown("### 🔍 Detailed Analysis")

sel = st.selectbox("Select stock", list(data.keys()))

if sel:
    df = data[sel]
    m = metrics[sel]
    inf = info.get(sel, {})
    
    st.markdown(f"#### {inf.get('name', sel)}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sector", inf.get('sector', 'N/A'))
    col2.metric("Market Cap", f"${inf.get('market_cap', 0)/1e9:.1f}B" if inf.get('market_cap') else 'N/A')
    col3.metric("Beta", f"{inf.get('beta', 1):.2f}")
    col4.metric("Total Return", f"{m['total_return']:.1f}%")
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           row_heights=[0.5, 0.25, 0.25],
                           subplot_titles=('Price Chart', 'RSI (14)', 'MACD'))
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        
        # MAs
        ma20 = df['Close'].rolling(20).mean()
        ma50 = df['Close'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma20, name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ma50, name='MA50', line=dict(color='blue', width=1.5)), row=1, col=1)
        
        # RSI
        rsi = calculate_rsi(df['Close'])
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        macd, signal = calculate_macd(df['Close'])
        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='red')), row=3, col=1)
        
        fig.update_layout(height=700, template='plotly_dark', showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
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
    
    # Patterns
    st.markdown("---")
    st.markdown("**🔍 Detected Patterns**")
    patterns = detect_patterns(df)
    cols = st.columns(min(len(patterns), 4))
    for i, p in enumerate(patterns):
        cols[i % 4].info(p)
    
    # Technical Summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    curr_rsi = rsi.iloc[-1]
    rsi_stat = "🟢 Neutral" if 30 < curr_rsi < 70 else ("🔴 Overbought" if curr_rsi > 70 else "🟢 Oversold")
    col1.metric("RSI", f"{curr_rsi:.1f}", rsi_stat)
    
    curr_macd = macd.iloc[-1]
    macd_stat = "🟢 Bullish" if curr_macd > signal.iloc[-1] else "🔴 Bearish"
    col2.metric("MACD", f"{curr_macd:.2f}", macd_stat)
    
    curr_price = df['Close'].iloc[-1]
    col3.metric("Current Price", f"${curr_price:.2f}")
    
    price_vs_ma = ((curr_price / ma20.iloc[-1]) - 1) * 100
    col4.metric("vs MA20", f"{price_vs_ma:+.1f}%")

# Footer
st.markdown("---")
st.caption("⚠️ For educational purposes only. Not financial advice. | Built with yfinance & Streamlit")