Professional Risk Meter

A Streamlit-based portfolio and stock risk analytics dashboard for US and Indian equities. The application fetches historical market data using yfinance, calculates financial risk and performance metrics, generates technical indicators, and presents the results through interactive Plotly visualizations.

Disclaimer: This project is for educational and analytical purposes only. It is not financial advice and should not be used as the sole basis for investment decisions.

📌 Table of Contents

Overview

Problem Statement

Key Features

How the Application Works

Application Architecture

Project Workflow

Technology Stack

Financial Metrics

Technical Indicators

Risk Scoring Model

Recommendation Engine

US and Indian Market Support

Data Handling and Caching

Dashboard Pages

Installation

Running the Application

Example Usage

Project Structure

Important Implementation Details

Limitations

Future Enhancements

Interview Explanation

License

🔎 Overview

Professional Risk Meter is an interactive financial analytics application built with Python and Streamlit.

The application allows a user to enter stocks such as:

AAPL

MSFT

GOOGL

RELIANCE.NS

TCS.NS

INFY.NS

It then:

Fetches historical market data.

Identifies the market as US or India.

Selects an appropriate market benchmark.

Calculates return and risk metrics.

Calculates technical indicators.

Generates a custom 0–100 risk score.

Categorizes the risk.

Generates a rule-based recommendation.

Displays portfolio-level and stock-level analytics.

Provides an educational glossary explaining the metrics.

🎯 Problem Statement

Investors often look at only the current stock price or percentage return when evaluating an investment.

That approach ignores important questions such as:

How volatile is the stock?

How large was its historical drawdown?

How much downside risk does it have?

How does it behave relative to the market?

Is the return attractive relative to the risk?

Is current momentum bullish or bearish?

This project combines these dimensions into a single analytics dashboard.

The objective is not to predict the future, but to make historical risk and performance information easier to understand and compare.

🚀 Key Features

Portfolio Management

Add multiple stock symbols.

Remove stocks from the portfolio.

Supports US and Indian equities.

Automatically attempts common Indian exchange suffixes.

Portfolio persists during Streamlit session reruns.

Analysis Periods

Supported periods include:

1 Day

5 Days

1 Month

3 Months

6 Months

1 Year

2 Years

5 Years

YTD

Max

Custom date range

Risk Analytics

The application calculates:

Annualized Return

Total Return

Annualized Volatility

Sharpe Ratio

Sortino Ratio

Maximum Drawdown

Historical VaR (95%)

Beta

Alpha

Win Rate

Composite Risk Score

Technical Analysis

The dashboard includes:

Candlestick chart

MA20

MA50

Volume

RSI

MACD

Trend detection

Support/resistance approximation

Recent volatility regime

Portfolio Visualization

Risk vs Return scatter plot

Sector allocation

Historical drawdown

Daily return distribution

User Experience

Streamlit interactive UI

Sidebar controls

Progress indicator while fetching data

Cached market data

Detailed stock drill-down

Separate glossary/explanation page

Dark Plotly charts

Responsive wide layout

🧠 How the Application Works

The application follows this high-level pipeline:

User Input
    ↓
Stock Symbols
    ↓
Yahoo Finance Data
    ↓
Data Validation
    ↓
Market Detection
    ↓
Benchmark Selection
    ↓
Daily Returns
    ↓
Risk & Performance Metrics
    ↓
Custom Risk Score
    ↓
Risk Category
    ↓
Recommendation
    ↓
Technical Analysis
    ↓
Interactive Dashboard

🏗️ Application Architecture

The current project is implemented as a single Streamlit Python application.

Conceptually, it contains these layers:

┌───────────────────────────────────────┐
│              Streamlit UI             │
│ Sidebar / Tables / Tabs / Charts      │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│          Portfolio Management         │
│ Session State / Add / Remove Stocks   │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│            Data Acquisition            │
│             yfinance / Yahoo           │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│            Data Analytics              │
│ Returns / Volatility / Risk / Beta    │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│        Technical Analysis Engine      │
│ RSI / MACD / MA / Trend Detection     │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│          Risk Scoring Engine           │
│        Composite 0–100 Score           │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│           Visualization Layer         │
│ Plotly / Metrics / Tables / Charts    │
└───────────────────────────────────────┘

🔄 Project Workflow

1. User selects an analysis period

The user chooses a predefined period or enters custom dates.

The selected period controls the historical data requested from Yahoo Finance.

2. User enters stock symbols

For example:

AAPL
TCS
RELIANCE

Indian symbols can be entered without the exchange suffix because the application attempts:

TCS
TCS.NS
TCS.BO

3. Data is fetched

For each valid symbol, the application retrieves historical OHLCV data.

The application also retrieves company information such as:

Company name

Sector

Beta

Market capitalization

Currency

4. Market benchmark is selected

The project uses:

Market

Benchmark

India

Nifty 50 (^NSEI)

US

S&P 500 (^GSPC)

The benchmark is required primarily for beta and alpha calculations.

5. Daily returns are calculated

Daily percentage returns are calculated from adjusted closing prices.

Conceptually:

Today's Return =
(Today's Close - Yesterday's Close)
/
Yesterday's Close

6. Risk metrics are calculated

The application calculates volatility, Sharpe, Sortino, drawdown, VaR, beta, and alpha.

7. Risk score is generated

The individual risk components are combined into a custom score from 0 to 100.

8. Technical indicators are calculated

The application calculates:

MA20

MA50

RSI

MACD

9. Recommendation is generated

A transparent rule-based engine maps the risk and performance metrics to:

STRONG BUY

BUY

HOLD

REDUCE

SELL

10. Results are displayed

The dashboard presents both portfolio-level and individual-stock analysis.

📊 Financial Metrics

1. Annualized Return

The current implementation calculates an arithmetic annualized return:

avg_return = returns.mean() * 252

There are approximately 252 trading days in a year.

Example

If average daily return is:

0.05%

then:

0.05% × 252 ≈ 12.6%

Important

This is not CAGR.

For long-term investment growth, CAGR would be a more appropriate metric.

2. Annualized Volatility

Volatility measures the dispersion of daily returns.

The project annualizes daily volatility using:

annual_vol = daily_vol * np.sqrt(252) * 100

Why √252?

Variance scales approximately with time, while standard deviation scales with the square root of time.

Example:

Daily volatility = 1%

Annualized volatility
≈ 1% × √252
≈ 15.9%

Higher volatility generally indicates greater price fluctuation.

3. Sharpe Ratio

Sharpe Ratio measures excess return relative to total volatility.

Conceptually:

Sharpe =
(Annualized Return - Risk-Free Rate)
/
Annualized Volatility

Interpretation:

Sharpe

General Interpretation

< 0

Poor risk-adjusted performance

0–0.5

Weak

0.5–1

Moderate

> 1

Good

> 2

Very strong

These are general guidelines, not strict investment rules.

4. Sortino Ratio

Sortino is similar to Sharpe but focuses on downside volatility.

The project calculates downside deviation using negative returns.

This is useful because investors generally care more about harmful volatility than positive volatility.

Example

Two stocks both have 20% return:

Stock A:
High upside volatility + controlled downside

Stock B:
Low upside volatility + large downside movements

Sortino can distinguish their downside-risk profiles better than Sharpe.

5. Maximum Drawdown

Maximum Drawdown measures the largest historical decline from a previous peak.

Example:

₹100 → ₹150 → ₹110

Peak:

₹150

Trough:

₹110

Drawdown:

(110 - 150) / 150
= -26.67%

Therefore:

Maximum Drawdown = -26.67%

6. Historical VaR (95%)

The project calculates historical VaR using the 5th percentile:

var_95 = np.percentile(returns, 5) * 100

If:

VaR 95% = -3%

the historical interpretation is that approximately 5% of observed daily returns were worse than -3%.

VaR is a statistical estimate, not a guarantee or maximum possible loss.

7. Beta

Beta measures sensitivity to a market benchmark.

The project calculates:

Beta =
Covariance(Stock Returns, Market Returns)
/
Variance(Market Returns)

General interpretation:

Beta

Meaning

< 1

Less sensitive than benchmark

≈ 1

Similar sensitivity

> 1

More sensitive

< 0

Tends to move opposite to benchmark

Example

If beta is 1.5, a simplified interpretation is:

Market +1%
Stock ≈ +1.5%

This is an average relationship, not a guaranteed daily movement.

8. Alpha

The project uses a CAPM-style expected return:

Expected Return =
Risk-Free Rate
+
Beta × (Market Return - Risk-Free Rate)

Then:

Alpha =
Actual Return - Expected Return

Example

Risk-free rate = 2%
Market return = 10%
Beta = 1.5

Expected return:

2% + 1.5 × (10% - 2%)
= 14%

If actual return is 18%:

Alpha = 18% - 14%
      = +4%

Positive alpha means the stock outperformed the model's expected return.

9. Total Return

Total return measures the change between the first and last closing price in the selected period.

Example:

Start = ₹100
End   = ₹125

Total Return = +25%

10. Win Rate

Win rate represents the percentage of observed trading days with positive returns.

Example:

100 trading days
60 positive days

Win Rate = 60%

A high win rate does not automatically mean a good investment because the magnitude of losses and gains also matters.

📈 Technical Indicators

MA20

20-day simple moving average:

df['Close'].rolling(20).mean()

It provides a short-term trend reference.

MA50

50-day simple moving average:

df['Close'].rolling(50).mean()

It provides a medium-term trend reference.

RSI

The Relative Strength Index is a momentum indicator.

The implementation uses a 14-period calculation.

General interpretation:

RSI > 70 → potentially overbought
RSI < 30 → potentially oversold
30–70    → neutral range

RSI should not be interpreted as an automatic buy/sell signal.

MACD

The project calculates:

EMA12
EMA26

MACD Line = EMA12 - EMA26

Signal Line = 9-period EMA of MACD

When MACD is above its signal line, the dashboard labels momentum as bullish; otherwise bearish.

⚠️ Risk Scoring Model

The application creates a custom 0–100 risk score.

The score combines:

Volatility Score
+
Drawdown Score
+
VaR Score
+
Beta Deviation Score
+
Sharpe Penalty

Current component limits

Component

Maximum Contribution

Volatility

30

Maximum Drawdown

25

VaR

20

Beta deviation

15

Sharpe penalty

No fixed independent cap before final 100 cap

Final score

100

The final score is capped at 100.

Risk Categories

Score

Category

0–20

Very Low

21–35

Low

36–50

Moderate

51–65

Elevated

66–80

High

81–100

Very High

Important

This is a custom heuristic scoring model.

It is not an official industry-standard risk score.

The weights were chosen to make multiple risk dimensions understandable through a single number.

🎯 Recommendation Engine

The application uses deterministic rules.

STRONG BUY

Requires:

Risk Score < 35
Sharpe > 1.5
Annual Return > 10%

BUY

Requires:

Risk Score < 50
Sharpe > 1.0
Annual Return > 5%

HOLD

Requires:

Risk Score < 60
Sharpe > 0.5

REDUCE

Used when:

Risk Score < 75

but the stock does not satisfy the stronger conditions.

SELL

Used for:

Risk Score >= 75

Why rule-based?

A rule-based system is:

Transparent

Easy to explain

Easy to debug

Deterministic

Suitable for an educational dashboard

It is not a machine-learning prediction model.

🇺🇸 🇮🇳 US and Indian Market Support

The application attempts to automatically identify Indian equities.

For example:

RELIANCE

may be tried as:

RELIANCE
RELIANCE.NS
RELIANCE.BO

For Indian stocks:

Benchmark = Nifty 50

For US stocks:

Benchmark = S&P 500

Currency is also used to display:

₹ for INR
$ for USD

⚡ Data Handling and Caching

The project uses:

@st.cache_data(ttl=1800)

This means fetched data can be cached for approximately 30 minutes.

Benefits

Reduces repeated API requests.

Improves application speed.

Reduces unnecessary network traffic.

Makes Streamlit reruns more efficient.

The dashboard also provides a manual refresh button that clears the Streamlit cache.

🖥️ Dashboard Pages

Page 1 — Risk Meter Dashboard

Contains:

Portfolio Overview

Risk Score

Annual Return

Sharpe Ratio

Volatility

Portfolio Visualizations

Risk vs Return

Sector Allocation

Individual Stock Analysis

A table containing:

Symbol

Company

Market

Risk Score

Risk Category

Annual Return

Volatility

Sharpe

Beta

Maximum Drawdown

Recommendation

Detailed Stock Analysis

Includes:

Company information

Market capitalization

Sector

Beta

Total return

Technical analysis

Risk analysis

Technical signals

Current price

RSI

MACD

Price vs MA20

📚 Page 2 — Glossary & Explanations

The second page explains major financial and technical concepts:

Risk Score

Annual Volatility

Sharpe Ratio

Sortino Ratio

Maximum Drawdown

VaR

Beta

Alpha

RSI

MACD

Risk Categories

This makes the dashboard more accessible to users who are not finance specialists.

🛠️ Installation

1. Clone the repository

git clone <your-repository-url>
cd professional-risk-meter

2. Create a virtual environment

Windows

python -m venv venv
venv\Scripts\activate

macOS/Linux

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

If requirements.txt does not exist yet, install:

pip install streamlit pandas numpy plotly yfinance

▶️ Running the Application

Run:

streamlit run app.py

Then open the local Streamlit URL shown in the terminal, usually:

http://localhost:8501

📦 Recommended requirements.txt

streamlit
pandas
numpy
plotly
yfinance

For reproducible production deployments, pin tested package versions.

🧪 Example Usage

Example 1 — US Portfolio

Add:

AAPL
MSFT
GOOGL

The application:

AAPL → S&P 500 benchmark
MSFT → S&P 500 benchmark
GOOGL → S&P 500 benchmark

Example 2 — Indian Portfolio

Add:

RELIANCE.NS
TCS.NS
INFY.NS

The application:

RELIANCE.NS → Nifty 50 benchmark
TCS.NS      → Nifty 50 benchmark
INFY.NS     → Nifty 50 benchmark

Example 3 — Mixed Portfolio

Add:

AAPL
MSFT
RELIANCE.NS
TCS.NS

The dashboard analyzes both markets using their respective benchmarks.

📁 Project Structure

For the current single-file version:

professional-risk-meter/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

app.py

Contains:

Streamlit UI

Data fetching

Financial calculations

Technical indicators

Risk scoring

Recommendation engine

Visualization

Glossary

requirements.txt

Contains Python dependencies required to run the application.

README.md

Project documentation, setup instructions, architecture, calculations, and limitations.

.gitignore

Prevents unnecessary files from being committed to GitHub.

Recommended entries:

venv/
.venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
.env

🧱 Recommended Advanced Project Structure

For a production-quality version, the application should be modularized:

professional-risk-meter/
│
├── app.py
│
├── pages/
│   └── glossary.py
│
├── services/
│   ├── market_data.py
│   ├── risk_metrics.py
│   ├── technical_analysis.py
│   └── recommendations.py
│
├── components/
│   ├── metric_cards.py
│   ├── charts.py
│   └── tables.py
│
├── utils/
│   ├── validators.py
│   └── formatters.py
│
├── tests/
│   ├── test_metrics.py
│   ├── test_risk_score.py
│   └── test_indicators.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

This separation would make the project easier to maintain, test, and scale.

⚠️ Important Implementation Details

1. Portfolio aggregation is currently equal-weighted

The portfolio overview uses the arithmetic mean of stock-level metrics.

For example:

Stock A Risk = 30
Stock B Risk = 60

Portfolio Risk = (30 + 60) / 2
               = 45

This does not account for how much money is invested in each stock.

2. Portfolio volatility is not covariance-based

True portfolio volatility depends on:

Position weights

Individual asset volatility

Covariance/correlation between assets

The current implementation displays average stock volatility rather than true portfolio volatility.

A proper portfolio model would use:

Portfolio Volatility =
sqrt(wᵀ Σ w)

where:

w = portfolio weight vector

Σ = covariance matrix

3. Sector allocation is stock-count based

If a portfolio contains:

3 Technology stocks
1 Energy stock

the chart represents:

Technology = 75%
Energy = 25%

This is based on the number of holdings, not invested capital.

A future version should support capital-weighted sector allocation.

4. Historical metrics are backward-looking

All major risk metrics depend on historical data.

Historical performance does not guarantee future performance.

5. Yahoo Finance data availability

Data availability, metadata quality, symbol resolution, and request behavior may vary.

A production system should use a dedicated market-data provider appropriate for the application's requirements.

🚧 Limitations

Current limitations include:

No true position-weighted portfolio model.

No covariance-based portfolio volatility.

No correlation matrix.

Annual return is arithmetic annualized return rather than CAGR.

VaR is historical percentile VaR.

No CVaR/Expected Shortfall.

No Monte Carlo simulation.

Recommendation engine is heuristic.

Technical pattern detection is simplified.

No fundamental analysis.

No news or sentiment analysis.

ticker.info may be slow or incomplete.

Broad exception handling can hide specific errors.

No authentication or persistent user portfolio database.

No formal backtesting of the recommendation engine.

No transaction costs, taxes, slippage, or liquidity modeling.

🔮 Future Enhancements

Phase 1 — Portfolio Analytics

Position quantity and investment amount

Capital-weighted portfolio return

True portfolio volatility

Covariance matrix

Correlation heatmap

Weighted sector allocation

Portfolio beta

Phase 2 — Advanced Risk

CAGR

CVaR / Expected Shortfall

Rolling volatility

Rolling Sharpe

Downside deviation

Stress testing

Scenario analysis

Phase 3 — Quant Analytics

Monte Carlo simulation

Efficient Frontier

Maximum Sharpe portfolio

Minimum volatility portfolio

Risk parity

Portfolio optimization

Phase 4 — Intelligence

Fundamental analysis

P/E and P/B ratios

ROE

Revenue growth

Earnings analysis

News sentiment

Market regime detection

Machine-learning risk prediction

Phase 5 — Production

User authentication

Persistent portfolios

Database

Logging

Unit tests

Automated testing

Error monitoring

API-based market data

Report generation

Cloud deployment

🧪 Testing Strategy

A production version should include unit tests for:

Return calculations

Verify daily and annualized returns.

Volatility

Test annualization and edge cases.

Sharpe/Sortino

Test:

Positive returns

Negative returns

Zero volatility

No downside observations

Drawdown

Test known price sequences.

VaR

Compare percentile results against expected values.

Beta

Use synthetic stock and benchmark returns with known relationships.

Risk Score

Test boundary values:

20
35
50
65
80
100

Recommendation Engine

Test each rule:

STRONG BUY
BUY
HOLD
REDUCE
SELL

🔐 Security and Configuration

The current application does not require API keys for its basic yfinance workflow.

If future versions introduce paid APIs or authentication:

Store secrets outside source code.

Use Streamlit secrets or environment variables.

Never commit API keys to GitHub.

Add secrets to .gitignore.

Rotate exposed credentials immediately.

📈 Example Risk Interpretation

Suppose a stock produces:

Risk Score       = 42
Annual Return    = 14%
Volatility       = 18%
Sharpe           = 1.15
Beta             = 1.20
Max Drawdown     = -16%
VaR 95%          = -2.4%

The application would classify it approximately as:

Risk Category = Moderate

Because:

36–50 → Moderate

If the other recommendation conditions are satisfied, it may generate:

BUY

This should be interpreted as the output of the project's rule engine, not a professional investment recommendation.

🎤 Interview Explanation

30-Second Version

I built a Streamlit-based Professional Risk Meter that analyzes US and Indian stocks using historical market data from yfinance. It calculates return, volatility, Sharpe, Sortino, maximum drawdown, VaR, beta, and alpha, then combines multiple risk factors into a custom 0–100 risk score. I also added RSI, MACD, moving averages, interactive Plotly charts, sector analysis, and a rule-based recommendation engine.

60-Second Version

Professional Risk Meter is a financial analytics dashboard built with Python, Streamlit, Pandas, NumPy, Plotly, and yfinance. Users can add US or Indian stocks and select predefined or custom analysis periods. The application fetches adjusted OHLCV data and automatically maps stocks to S&P 500 or Nifty 50 benchmarks. It calculates daily returns and derives annualized return, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, historical 95% VaR, beta, alpha, and win rate. These metrics feed into a custom composite risk score from 0 to 100. The dashboard also provides technical analysis using RSI, MACD, MA20, MA50, volume, trend detection, and drawdown visualizations. The current version uses equal-weighted portfolio aggregation and heuristic recommendations; the next major improvement would be a true weighted portfolio engine using covariance, correlation, and portfolio optimization.

💡 Key Learning Outcomes

This project demonstrates practical experience with:

Python

Functions

Exception handling

Data structures

Numerical computation

Date handling

Pandas

DataFrames

Time series

Percentage returns

Rolling calculations

Statistical operations

NumPy

Percentiles

Standard deviation

Covariance

Linear regression

Mathematical transformations

Streamlit

Interactive widgets

Session state

Caching

Tabs

Sidebar

Progress indicators

Reruns

Plotly

Scatter plots

Pie charts

Candlesticks

Histograms

Subplots

Interactive financial charts

Finance

Returns

Volatility

Sharpe

Sortino

Drawdown

VaR

Beta

Alpha

RSI

MACD

⭐ Project Highlights

The project demonstrates an end-to-end analytics workflow:

Data Collection
       ↓
Data Validation
       ↓
Data Transformation
       ↓
Statistical Analysis
       ↓
Financial Risk Modeling
       ↓
Technical Analysis
       ↓
Visualization
       ↓
Decision Support

This makes the project particularly relevant for roles involving:

Data Analytics

Business Intelligence

Financial Analytics

Python Development

Dashboard Development

Quantitative Analytics

Risk Analytics

📜 License

This project can be released under the MIT License.

Add a LICENSE file containing the standard MIT License text if you intend to distribute the project publicly.

⚠️ Disclaimer

Professional Risk Meter is an educational software project.

The calculations are based on historical market data and simplified analytical assumptions. Risk scores and recommendations are custom outputs of the application and are not professional financial advice, investment recommendations, or guarantees of future performance.

Always perform independent research and consider your own financial circumstances before making investment decisions.
