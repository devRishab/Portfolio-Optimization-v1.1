import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.optimize as sco
from datetime import date
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #131720 100%);
        border-right: 1px solid #2d3748;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px !important;
    }
    [data-testid="stMetricLabel"] { color: #8892a4 !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 26px !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: 13px !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892a4;
        font-weight: 600;
        font-size: 14px;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb !important;
        color: white !important;
    }

    /* Headers */
    h1 { color: #ffffff !important; font-weight: 800 !important; }
    h2 { color: #e2e8f0 !important; font-weight: 700 !important; }
    h3 { color: #cbd5e0 !important; font-weight: 600 !important; }

    /* Info/Success/Warning boxes */
    .stAlert { border-radius: 10px !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Section card */
    .section-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    /* Badge */
    .badge-green {
        background: #064e3b;
        color: #34d399;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-red {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Divider */
    hr { border-color: #2d3748 !important; }

    /* Sidebar text */
    .sidebar-header {
        color: #2563eb;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* ── Safety net for native Streamlit text on the forced dark
       background. The .streamlit/config.toml dark theme should make
       most of this redundant, but keep it in case the app runs without
       that config file (e.g. deployed without the .streamlit folder). */
    .stCaption, [data-testid="stCaptionContainer"] { color: #8892a4 !important; }
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p { color: #cbd5e0 !important; }
    .stTextInput input, .stTextArea textarea,
    .stDateInput input, .stNumberInput input {
        color: #e2e8f0 !important;
        background-color: #1a1f2e !important;
    }
    .stSlider label, .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] { color: #cbd5e0 !important; }
    .stAlert, .stAlert p, .stAlert div { color: #e2e8f0 !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] p,
    [data-testid="stExpander"] li { color: #cbd5e0 !important; }
    .stMarkdown p, .stMarkdown li { color: #cbd5e0; }
    [data-testid="stDataFrame"] * { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(tickers, market_ticker, start, end):
    # De-duplicate in case the user also typed the market ticker into the
    # stock list — a duplicate column would otherwise turn market_data into
    # a DataFrame instead of a Series further down.
    stock_tickers = [t for t in tickers if t != market_ticker]
    all_tickers = list(dict.fromkeys(stock_tickers + [market_ticker]))

    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True)["Close"]
    raw.dropna(how="all", inplace=True)

    # yfinance still creates a column for every requested ticker even when
    # the download fails for it (filled entirely with NaN), so checking
    # column *presence* never actually detects a failed/delisted ticker.
    # Check for columns that actually have data instead.
    available = [
        t for t in stock_tickers
        if t in raw.columns and raw[t].notna().any()
    ]
    stock_data = raw[available].dropna()
    market_data = raw[market_ticker].dropna()
    return stock_data, market_data, available


def compute_stats(stock_data, market_data):
    returns = stock_data.pct_change().dropna()
    mkt_ret = market_data.pct_change().dropna()
    returns, mkt_ret = returns.align(mkt_ret, join="inner", axis=0)

    mean_ret  = returns.mean() * 252
    cov_mat   = returns.cov() * 252
    corr_mat  = returns.corr()
    mkt_annual = mkt_ret.mean() * 252
    mkt_var    = np.var(mkt_ret, ddof=1)  # match np.cov's default ddof=1
    return returns, mkt_ret, mean_ret, cov_mat, corr_mat, mkt_annual, mkt_var


def portfolio_performance(weights, mean_returns, cov_matrix, rf):
    ret   = np.dot(weights, mean_returns)
    vol   = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe


def neg_sharpe(weights, mean_returns, cov_matrix, rf):
    return -portfolio_performance(weights, mean_returns, cov_matrix, rf)[2]


def optimize_portfolio(mean_returns, cov_matrix, rf):
    n = len(mean_returns)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = n * [1.0 / n]
    result = sco.minimize(neg_sharpe, init,
                          args=(mean_returns, cov_matrix, rf),
                          method="SLSQP", bounds=bounds,
                          constraints=constraints)
    return result.x


def compute_beta(port_daily, mkt_ret):
    cov = np.cov(port_daily, mkt_ret)[0, 1]
    var = np.var(mkt_ret, ddof=1)  # match np.cov's default ddof=1
    return cov / var


def individual_betas(returns, mkt_ret, mkt_var):
    betas = {}
    for col in returns.columns:
        cov = np.cov(returns[col], mkt_ret)[0, 1]
        betas[col] = cov / mkt_var
    return betas


def min_variance(mean_returns, cov_matrix, target_return):
    n = len(mean_returns)
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "eq", "fun": lambda x, t=target_return: np.dot(x, mean_returns) - t}
    ]
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = n * [1.0 / n]
    result = sco.minimize(
        lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
        init, method="SLSQP", bounds=bounds, constraints=constraints
    )
    if result.success:
        return result.fun
    return None


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px 0;'>
        <div style='font-size:36px;'>📊</div>
        <div style='font-size:20px; font-weight:800; color:#ffffff;'>Portfolio Optimizer</div>
        <div style='font-size:12px; color:#8892a4; margin-top:4px;'>Powered by SLSQP · CAPM · MPT</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-header">📅 Date Range</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=date(2020, 1, 1), max_value=date.today())
    with col2:
        end_date   = st.date_input("End",   value=date.today(),    max_value=date.today())

    st.markdown("---")
    st.markdown('<div class="sidebar-header">📈 Stock Tickers</div>', unsafe_allow_html=True)
    st.caption("Enter NSE tickers (e.g. TCS.NS) or US tickers (e.g. AAPL). One per line.")

    default_tickers = """TCS.NS
INFY.NS
HDFCBANK.NS
ICICIBANK.NS
RELIANCE.NS
ITC.NS
SUNPHARMA.NS
BHARTIARTL.NS
M&M.NS
GOLDBEES.NS"""
    ticker_input = st.text_area("Tickers", value=default_tickers, height=220, label_visibility="collapsed")
    tickers = [t.strip().upper() for t in ticker_input.strip().split("\n") if t.strip()]

    st.markdown("---")
    st.markdown('<div class="sidebar-header">🏦 Market Settings</div>', unsafe_allow_html=True)
    market_ticker = st.text_input("Market Index", value="^NSEI", help="Nifty 50 = ^NSEI, S&P 500 = ^GSPC")
    rf_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=6.5, step=0.1) / 100

    st.markdown("---")
    run_btn = st.button("🚀  Run Optimization", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px 0;'>
    <h1 style='font-size:32px; margin:0;'>📈 Portfolio Optimization Dashboard</h1>
    <p style='color:#8892a4; margin:4px 0 0 0; font-size:15px;'>
        Modern Portfolio Theory · CAPM · Efficient Frontier · Sharpe Maximization
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style='background:#1a1f2e; border:1px solid #2d3748; border-radius:16px;
                padding:48px; text-align:center; margin-top:40px;'>
        <div style='font-size:64px; margin-bottom:16px;'>🏦</div>
        <div style='font-size:22px; font-weight:700; color:#e2e8f0; margin-bottom:8px;'>
            Configure your portfolio on the left
        </div>
        <div style='color:#8892a4; font-size:15px; max-width:500px; margin:0 auto;'>
            Enter your stock tickers, choose a date range, set the risk-free rate,
            and click <b style='color:#2563eb;'>Run Optimization</b> to generate
            the full analysis.
        </div>
        <div style='margin-top:32px; display:flex; justify-content:center; gap:24px; flex-wrap:wrap;'>
            <span style='background:#0f2c5a; color:#60a5fa; padding:8px 18px;
                         border-radius:999px; font-size:13px; font-weight:600;'>
                📊 Efficient Frontier
            </span>
            <span style='background:#0d3d2e; color:#34d399; padding:8px 18px;
                         border-radius:999px; font-size:13px; font-weight:600;'>
                📉 Capital Allocation Line
            </span>
            <span style='background:#3b1f57; color:#c084fc; padding:8px 18px;
                         border-radius:999px; font-size:13px; font-weight:600;'>
                🎯 Security Market Line
            </span>
            <span style='background:#4a1f1f; color:#fca5a5; padding:8px 18px;
                         border-radius:999px; font-size:13px; font-weight:600;'>
                🔥 Correlation Heatmap
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Fetch & Compute ──────────────────────────
with st.spinner("⏳ Fetching market data and running optimization..."):
    try:
        stock_data, market_data, available_tickers = fetch_data(
            tickers, market_ticker, str(start_date), str(end_date))

        if len(available_tickers) < 2:
            st.error("❌ Need at least 2 valid tickers. Please check your input.")
            st.stop()

        (returns, mkt_ret, mean_ret, cov_mat,
         corr_mat, mkt_annual, mkt_var) = compute_stats(stock_data, market_data)

        opt_weights = optimize_portfolio(mean_ret, cov_mat, rf_rate)
        opt_ret, opt_vol, opt_sharpe = portfolio_performance(opt_weights, mean_ret, cov_mat, rf_rate)

        port_daily = returns.dot(opt_weights)
        port_beta  = compute_beta(port_daily, mkt_ret)
        capm_ret   = rf_rate + port_beta * (mkt_annual - rf_rate)

        ind_betas  = individual_betas(returns, mkt_ret, mkt_var)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# ── Skipped Tickers Warning ──────────────────
skipped = [t for t in tickers if t not in available_tickers]
if skipped:
    st.warning(f"⚠️ Could not fetch data for: {', '.join(skipped)} — they have been excluded.")

# ─────────────────────────────────────────────
#  KEY METRICS ROW
# ─────────────────────────────────────────────
st.markdown("### 🎯 Optimal Portfolio — Key Metrics")
m1, m2, m3, m4, m5, m6 = st.columns(6)

active_stocks = sum(1 for w in opt_weights if w > 0.001)
alpha = opt_ret - capm_ret

m1.metric("Expected Annual Return", f"{opt_ret:.2%}",  delta=f"+{opt_ret - rf_rate:.2%} vs Rf")
m2.metric("Annual Volatility",      f"{opt_vol:.2%}",  delta="Portfolio Risk")
m3.metric("Sharpe Ratio",           f"{opt_sharpe:.4f}", delta="Max Sharpe")
m4.metric("Portfolio Beta",         f"{port_beta:.4f}", delta="Systematic Risk")
m5.metric("CAPM Expected Return",   f"{capm_ret:.2%}",  delta=f"Alpha: {alpha:+.2%}")
m6.metric("Active Stocks",          f"{active_stocks}/{len(available_tickers)}", delta="Non-zero weights")

st.markdown("---")

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Optimal Portfolio",
    "📊 CAL & Efficient Frontier",
    "🎯 Security Market Line",
    "🔥 Correlation Matrix",
    "📉 Stock Valuation",
    "📋 CAL Simulation Table"
])

# ════════════════════════════════════════════
#  TAB 1 — OPTIMAL PORTFOLIO
# ════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown("#### 📦 Asset Allocation Weights")

        weight_df = pd.DataFrame({
            "Ticker": available_tickers,
            "Weight": opt_weights
        }).sort_values("Weight", ascending=False)
        weight_df["Weight %"] = weight_df["Weight"].map(lambda x: f"{x:.2%}")
        weight_df["Expected Return"] = weight_df["Ticker"].map(lambda t: f"{mean_ret[t]:.2%}")
        weight_df["Beta"] = weight_df["Ticker"].map(lambda t: f"{ind_betas[t]:.4f}")
        weight_df["Status"] = weight_df["Weight"].map(
            lambda w: "✅ Active" if w > 0.001 else "⭕ Zero")

        # Color bar chart
        active_df = weight_df[weight_df["Weight"] > 0.001]
        # px.colors.qualitative.Plotly only has 10 colors — cycle it so
        # this doesn't break (mismatched marker-color length) once more
        # than 10 assets get a non-zero optimal weight.
        palette = px.colors.qualitative.Plotly
        colors = [palette[i % len(palette)] for i in range(len(active_df))]

        fig_bar = go.Figure(go.Bar(
            x=active_df["Ticker"],
            y=active_df["Weight"] * 100,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{w:.1f}%" for w in active_df["Weight"] * 100],
            textposition="outside",
            textfont=dict(color="white", size=13)
        ))
        fig_bar.update_layout(
            paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
            font_color="white",
            xaxis=dict(tickfont=dict(size=12), gridcolor="#2d3748"),
            yaxis=dict(title="Weight (%)", gridcolor="#2d3748", ticksuffix="%"),
            margin=dict(t=20, b=20, l=10, r=10),
            height=300, showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Table
        display_df = weight_df[["Ticker", "Weight %", "Expected Return", "Beta", "Status"]].reset_index(drop=True)
        st.dataframe(
            display_df.style.apply(
                lambda row: ["background-color: #0d3d2e" if row["Status"] == "✅ Active"
                             else "background-color: #1a1f2e"] * len(row),
                axis=1
            ),
            use_container_width=True, hide_index=True
        )

    with col_right:
        st.markdown("#### 🥧 Portfolio Composition")

        fig_pie = go.Figure(go.Pie(
            labels=active_df["Ticker"],
            values=active_df["Weight"] * 100,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>"
        ))
        fig_pie.update_layout(
            paper_bgcolor="#1a1f2e",
            font_color="white",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            annotations=[dict(
                text=f"<b>{opt_sharpe:.2f}</b><br>Sharpe",
                x=0.5, y=0.5, font_size=16,
                showarrow=False, font_color="white"
            )]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Portfolio summary card
        st.markdown(f"""
        <div class="section-card">
            <div style='color:#8892a4; font-size:12px; font-weight:700;
                        letter-spacing:1px; margin-bottom:14px;'>PORTFOLIO SUMMARY</div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
                <div>
                    <div style='color:#8892a4; font-size:12px;'>Annual Return</div>
                    <div style='color:#34d399; font-size:20px; font-weight:700;'>{opt_ret:.2%}</div>
                </div>
                <div>
                    <div style='color:#8892a4; font-size:12px;'>Volatility</div>
                    <div style='color:#fbbf24; font-size:20px; font-weight:700;'>{opt_vol:.2%}</div>
                </div>
                <div>
                    <div style='color:#8892a4; font-size:12px;'>Sharpe Ratio</div>
                    <div style='color:#60a5fa; font-size:20px; font-weight:700;'>{opt_sharpe:.4f}</div>
                </div>
                <div>
                    <div style='color:#8892a4; font-size:12px;'>Beta</div>
                    <div style='color:#c084fc; font-size:20px; font-weight:700;'>{port_beta:.4f}</div>
                </div>
                <div>
                    <div style='color:#8892a4; font-size:12px;'>CAPM Return</div>
                    <div style='color:#f87171; font-size:20px; font-weight:700;'>{capm_ret:.2%}</div>
                </div>
                <div>
                    <div style='color:#8892a4; font-size:12px;'>Jensen's Alpha</div>
                    <div style='color:{"#34d399" if alpha >= 0 else "#f87171"}; font-size:20px; font-weight:700;'>{alpha:+.2%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════
#  TAB 2 — CAL & EFFICIENT FRONTIER
# ════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 Capital Allocation Line & Efficient Frontier")

    with st.spinner("Generating efficient frontier (5,000 simulations)..."):
        np.random.seed(42)
        n = len(available_tickers)
        sim_returns, sim_vols, sim_sharpes = [], [], []

        for _ in range(5000):
            w = np.random.random(n)
            w /= np.sum(w)
            r, v, s = portfolio_performance(w, mean_ret, cov_mat, rf_rate)
            sim_returns.append(r)
            sim_vols.append(v)
            sim_sharpes.append(s)

        sim_returns = np.array(sim_returns)
        sim_vols    = np.array(sim_vols)
        sim_sharpes = np.array(sim_sharpes)

        # Minimum-variance frontier
        target_rets = np.linspace(mean_ret.min(), mean_ret.max(), 60)
        ef_vols, ef_rets = [], []
        for tr in target_rets:
            v = min_variance(mean_ret, cov_mat, tr)
            if v is not None:
                ef_vols.append(v)
                ef_rets.append(tr)

        # Keep only the EFFICIENT (upper) branch: everything from the
        # global minimum-variance point upward. Below that point, every
        # portfolio is dominated by one with the same risk and higher
        # return further up the curve, so it isn't part of the true
        # efficient frontier even though it's on the min-variance curve.
        if ef_vols:
            min_idx = int(np.argmin(ef_vols))
            ef_vols, ef_rets = ef_vols[min_idx:], ef_rets[min_idx:]

        # CAL line
        cal_x = np.linspace(0, max(sim_vols) * 1.05, 100)
        cal_y = rf_rate + opt_sharpe * cal_x

    fig_cal = go.Figure()

    # Scatter — Monte Carlo portfolios
    fig_cal.add_trace(go.Scatter(
        x=sim_vols * 100, y=sim_returns * 100,
        mode="markers",
        marker=dict(
            color=sim_sharpes,
            colorscale="Viridis",
            size=4,
            opacity=0.6,
            colorbar=dict(
                title=dict(text="Sharpe Ratio", font=dict(color="white")),
                tickfont=dict(color="white"),
                x=1.02
            ),
            showscale=True
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Random Portfolio</extra>"
    ))

    # Efficient frontier curve
    if len(ef_vols) > 3:
        fig_cal.add_trace(go.Scatter(
            x=[v * 100 for v in ef_vols],
            y=[r * 100 for r in ef_rets],
            mode="lines",
            line=dict(color="#f59e0b", width=3, dash="solid"),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Efficient Frontier</extra>"
        ))

    # CAL line
    fig_cal.add_trace(go.Scatter(
        x=cal_x * 100, y=cal_y * 100,
        mode="lines",
        line=dict(color="#3b82f6", width=2.5, dash="dash"),
        name=f"CAL (Slope={opt_sharpe:.2f})",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>CAL</extra>"
    ))

    # Risk-free point
    fig_cal.add_trace(go.Scatter(
        x=[0], y=[rf_rate * 100],
        mode="markers+text",
        marker=dict(color="#60a5fa", size=12, symbol="circle"),
        text=["Rf"], textposition="top right",
        textfont=dict(color="#60a5fa", size=12),
        name=f"Risk-Free ({rf_rate:.1%})",
        hovertemplate=f"Risk-Free Rate: {rf_rate:.2%}<extra></extra>"
    ))

    # Optimal portfolio (red star)
    fig_cal.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_ret * 100],
        mode="markers+text",
        marker=dict(color="#ef4444", size=20, symbol="star"),
        text=["Optimal"], textposition="top right",
        textfont=dict(color="#ef4444", size=13, family="Arial Black"),
        name=f"Optimal Portfolio (Sharpe={opt_sharpe:.2f})",
        hovertemplate=f"Return: {opt_ret:.2%}<br>Vol: {opt_vol:.2%}<br>Sharpe: {opt_sharpe:.4f}<extra>Optimal</extra>"
    ))

    fig_cal.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#131720",
        font_color="white",
        xaxis=dict(title="Annual Volatility (%)", gridcolor="#2d3748",
                   ticksuffix="%", zeroline=False),
        yaxis=dict(title="Expected Annual Return (%)", gridcolor="#2d3748",
                   ticksuffix="%", zeroline=False),
        legend=dict(bgcolor="#1a1f2e", bordercolor="#2d3748",
                    borderwidth=1, font=dict(size=12)),
        margin=dict(t=20, b=20, l=10, r=80),
        height=560,
        hovermode="closest"
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # Explanation
    with st.expander("📖 How to read this chart"):
        st.markdown("""
        - **Coloured dots** = 5,000 randomly generated portfolios. Brighter (yellow) = higher Sharpe ratio.
        - **Yellow curve** = Efficient Frontier — the optimal boundary of risky-only portfolios.
        - **Blue dashed line** = Capital Allocation Line (CAL) — combinations of risk-free asset + optimal portfolio.
        - **Red ★** = Optimal (Tangency) Portfolio — maximum Sharpe ratio point where CAL is tangent to the frontier.
        - **Blue dot** = Risk-Free Rate — the starting point of the CAL (zero risk).
        """)

# ════════════════════════════════════════════
#  TAB 3 — SECURITY MARKET LINE
# ════════════════════════════════════════════
with tab3:
    st.markdown("#### 🎯 Security Market Line (SML) & Individual Stock Valuation")

    beta_range  = np.linspace(0, max(list(ind_betas.values())) * 1.2, 100)
    sml_returns = rf_rate + beta_range * (mkt_annual - rf_rate)

    fig_sml = go.Figure()

    # SML line
    fig_sml.add_trace(go.Scatter(
        x=beta_range, y=sml_returns * 100,
        mode="lines",
        line=dict(color="#3b82f6", width=2.5),
        name="Security Market Line",
        hovertemplate="Beta: %{x:.2f}<br>CAPM Return: %{y:.2f}%<extra>SML</extra>"
    ))

    # Individual stocks
    for ticker in available_tickers:
        b = ind_betas[ticker]
        actual = mean_ret[ticker]
        capm_r = rf_rate + b * (mkt_annual - rf_rate)
        alpha_i = actual - capm_r
        is_under = actual > capm_r

        fig_sml.add_trace(go.Scatter(
            x=[b], y=[actual * 100],
            mode="markers+text",
            marker=dict(
                color="#34d399" if is_under else "#f87171",
                size=14,
                symbol="circle",
                line=dict(color="white", width=1.5)
            ),
            text=[ticker.replace(".NS", "").replace(".BO", "")],
            textposition="top center",
            textfont=dict(size=10, color="white"),
            name=ticker,
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                f"Beta: {b:.4f}<br>"
                f"Actual Return: {actual:.2%}<br>"
                f"CAPM Return: {capm_r:.2%}<br>"
                f"Alpha: {alpha_i:+.2%}<br>"
                f"{'✅ Undervalued' if is_under else '❌ Overvalued'}"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    # Optimal portfolio on SML
    fig_sml.add_trace(go.Scatter(
        x=[port_beta], y=[capm_ret * 100],
        mode="markers+text",
        marker=dict(color="#ef4444", size=20, symbol="star"),
        text=["Optimal Portfolio"],
        textposition="top right",
        textfont=dict(color="#ef4444", size=12, family="Arial Black"),
        name="Optimal Portfolio",
        hovertemplate=f"Beta: {port_beta:.4f}<br>CAPM Return: {capm_ret:.2%}<extra>Optimal Portfolio</extra>"
    ))

    # Market
    fig_sml.add_trace(go.Scatter(
        x=[1.0], y=[mkt_annual * 100],
        mode="markers+text",
        marker=dict(color="#a78bfa", size=14, symbol="square"),
        text=["Market"], textposition="top right",
        textfont=dict(color="#a78bfa", size=12),
        name="Market Portfolio",
        hovertemplate=f"Market Return: {mkt_annual:.2%}<extra>Market</extra>"
    ))

    # Risk-free
    fig_sml.add_trace(go.Scatter(
        x=[0], y=[rf_rate * 100],
        mode="markers+text",
        marker=dict(color="#60a5fa", size=12, symbol="circle"),
        text=["Rf"], textposition="top right",
        textfont=dict(color="#60a5fa", size=12),
        name=f"Risk-Free ({rf_rate:.1%})",
        hovertemplate=f"Risk-Free Rate: {rf_rate:.2%}<extra></extra>"
    ))

    fig_sml.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#131720",
        font_color="white",
        xaxis=dict(title="Beta (Systematic Risk)", gridcolor="#2d3748", zeroline=False),
        yaxis=dict(title="Expected Return (%)", gridcolor="#2d3748",
                   ticksuffix="%", zeroline=False),
        legend=dict(bgcolor="#1a1f2e", bordercolor="#2d3748",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(t=20, b=20, l=10, r=10),
        height=560,
        hovermode="closest"
    )
    st.plotly_chart(fig_sml, use_container_width=True)

    with st.expander("📖 How to read this chart"):
        st.markdown("""
        - **Blue line** = Security Market Line — shows CAPM required return for each level of beta.
        - **Green dots** = Undervalued stocks (actual return > CAPM required return → positive alpha).
        - **Red dots** = Overvalued stocks (actual return < CAPM required return → negative alpha).
        - **Purple square** = Market portfolio (beta = 1).
        - **Red ★** = Optimal portfolio position on SML.
        """)

# ════════════════════════════════════════════
#  TAB 4 — CORRELATION MATRIX
# ════════════════════════════════════════════
with tab4:
    st.markdown("#### 🔥 Correlation Matrix Heatmap")

    # Short labels
    short_labels = [t.replace(".NS", "").replace(".BO", "") for t in available_tickers]
    corr_display = corr_mat.copy()
    corr_display.columns = short_labels
    corr_display.index   = short_labels

    fig_heat = go.Figure(go.Heatmap(
        z=corr_display.values,
        x=short_labels,
        y=short_labels,
        colorscale=[
            [0.0, "#1e3a5f"],
            [0.3, "#1e3a5f"],
            [0.5, "#1a1f2e"],
            [0.7, "#7f1d1d"],
            [1.0, "#dc2626"]
        ],
        zmin=-1, zmax=1,
        text=np.round(corr_display.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hoverongaps=False,
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color="white")),
            tickfont=dict(color="white"),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0", "-0.5", "0.0", "+0.5", "+1.0"]
        )
    ))

    fig_heat.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="white",
        xaxis=dict(tickangle=-45, tickfont=dict(size=12), side="bottom"),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(t=20, b=80, l=80, r=20),
        height=560
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Stats below heatmap
    col_a, col_b, col_c = st.columns(3)
    flat_corr = corr_display.values[np.triu_indices_from(corr_display.values, k=1)]

    col_a.metric("Average Correlation", f"{flat_corr.mean():.4f}")
    col_b.metric("Max Correlation",     f"{flat_corr.max():.4f}")
    col_c.metric("Min Correlation",     f"{flat_corr.min():.4f}")

    with st.expander("📖 How to read this chart"):
        st.markdown("""
        - **Red cells** = high positive correlation → assets move together → less diversification benefit.
        - **Blue cells** = low or negative correlation → assets move independently → strong diversification benefit.
        - **Diagonal** = always 1.0 (asset perfectly correlated with itself).
        - Lower average correlation = better diversification = lower portfolio risk for the same return.
        """)

# ════════════════════════════════════════════
#  TAB 5 — STOCK VALUATION (UNDERVALUED / OVERVALUED)
# ════════════════════════════════════════════
with tab5:
    st.markdown("#### 📉 Individual Stock Valuation using CAPM & SML Framework")

    rows = []
    for ticker in available_tickers:
        b         = ind_betas[ticker]
        actual    = mean_ret[ticker]
        capm_r    = rf_rate + b * (mkt_annual - rf_rate)
        alpha_i   = actual - capm_r
        verdict   = "✅ Undervalued" if actual > capm_r else "❌ Overvalued"
        opt_w     = opt_weights[available_tickers.index(ticker)]

        rows.append({
            "Ticker":          ticker,
            "Beta":            round(b, 4),
            "Actual Return":   f"{actual:.2%}",
            "CAPM Required":   f"{capm_r:.2%}",
            "Alpha":           f"{alpha_i:+.2%}",
            "Verdict":         verdict,
            "Portfolio Wt":    f"{opt_w:.2%}",
            "_alpha_val":      alpha_i,
            "_actual":         actual,
            "_capm":           capm_r
        })

    val_df = pd.DataFrame(rows).sort_values("_alpha_val", ascending=False)

    # ── Split cards ──
    under = val_df[val_df["_alpha_val"] > 0]
    over  = val_df[val_df["_alpha_val"] <= 0]

    col_u, col_o = st.columns(2, gap="large")

    with col_u:
        st.markdown(f"""
        <div style='background:#064e3b; border:1px solid #065f46;
                    border-radius:12px; padding:12px 18px; margin-bottom:12px;'>
            <span style='color:#34d399; font-weight:700; font-size:16px;'>
                ✅ Undervalued Stocks ({len(under)})
            </span>
            <div style='color:#6ee7b7; font-size:12px; margin-top:4px;'>
                Actual return > CAPM required return → Positive Alpha
            </div>
        </div>
        """, unsafe_allow_html=True)

        for _, row in under.iterrows():
            st.markdown(f"""
            <div style='background:#1a1f2e; border:1px solid #065f46;
                        border-radius:10px; padding:14px 18px; margin-bottom:8px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='color:#34d399; font-weight:700; font-size:15px;'>
                            {row['Ticker'].replace('.NS','').replace('.BO','')}
                        </span>
                        <span style='color:#8892a4; font-size:12px; margin-left:8px;'>
                            β = {row['Beta']}
                        </span>
                    </div>
                    <div style='background:#064e3b; color:#34d399; padding:3px 10px;
                                border-radius:999px; font-size:12px; font-weight:700;'>
                        Alpha {row['Alpha']}
                    </div>
                </div>
                <div style='display:grid; grid-template-columns:1fr 1fr 1fr;
                            gap:8px; margin-top:10px;'>
                    <div>
                        <div style='color:#8892a4; font-size:11px;'>Actual Return</div>
                        <div style='color:#34d399; font-weight:700;'>{row['Actual Return']}</div>
                    </div>
                    <div>
                        <div style='color:#8892a4; font-size:11px;'>CAPM Required</div>
                        <div style='color:#fbbf24; font-weight:700;'>{row['CAPM Required']}</div>
                    </div>
                    <div>
                        <div style='color:#8892a4; font-size:11px;'>Portfolio Wt</div>
                        <div style='color:#60a5fa; font-weight:700;'>{row['Portfolio Wt']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_o:
        st.markdown(f"""
        <div style='background:#7f1d1d; border:1px solid #991b1b;
                    border-radius:12px; padding:12px 18px; margin-bottom:12px;'>
            <span style='color:#fca5a5; font-weight:700; font-size:16px;'>
                ❌ Overvalued Stocks ({len(over)})
            </span>
            <div style='color:#fca5a5; font-size:12px; margin-top:4px; opacity:0.8;'>
                Actual return &lt; CAPM required return → Negative Alpha
            </div>
        </div>
        """, unsafe_allow_html=True)

        for _, row in over.iterrows():
            st.markdown(f"""
            <div style='background:#1a1f2e; border:1px solid #991b1b;
                        border-radius:10px; padding:14px 18px; margin-bottom:8px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='color:#f87171; font-weight:700; font-size:15px;'>
                            {row['Ticker'].replace('.NS','').replace('.BO','')}
                        </span>
                        <span style='color:#8892a4; font-size:12px; margin-left:8px;'>
                            β = {row['Beta']}
                        </span>
                    </div>
                    <div style='background:#7f1d1d; color:#fca5a5; padding:3px 10px;
                                border-radius:999px; font-size:12px; font-weight:700;'>
                        Alpha {row['Alpha']}
                    </div>
                </div>
                <div style='display:grid; grid-template-columns:1fr 1fr 1fr;
                            gap:8px; margin-top:10px;'>
                    <div>
                        <div style='color:#8892a4; font-size:11px;'>Actual Return</div>
                        <div style='color:#f87171; font-weight:700;'>{row['Actual Return']}</div>
                    </div>
                    <div>
                        <div style='color:#8892a4; font-size:11px;'>CAPM Required</div>
                        <div style='color:#fbbf24; font-weight:700;'>{row['CAPM Required']}</div>
                    </div>
                    <div>
                        <div style='color:#8892a4; font-size:11px;'>Portfolio Wt</div>
                        <div style='color:#60a5fa; font-weight:700;'>{row['Portfolio Wt']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Alpha bar chart
    st.markdown("#### 📊 Alpha Comparison Chart")
    val_sorted = val_df.sort_values("_alpha_val")

    fig_alpha = go.Figure(go.Bar(
        x=[t.replace(".NS","").replace(".BO","") for t in val_sorted["Ticker"]],
        y=val_sorted["_alpha_val"] * 100,
        marker=dict(
            color=["#34d399" if a > 0 else "#f87171" for a in val_sorted["_alpha_val"]],
            line=dict(width=0)
        ),
        text=[f"{a:+.2f}%" for a in val_sorted["_alpha_val"] * 100],
        textposition="outside",
        textfont=dict(color="white", size=11),
        hovertemplate="<b>%{x}</b><br>Alpha: %{y:.2f}%<extra></extra>"
    ))
    fig_alpha.add_hline(y=0, line_color="#8892a4", line_width=1.5)
    fig_alpha.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#131720",
        font_color="white",
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(title="Jensen's Alpha (%)", gridcolor="#2d3748", ticksuffix="%"),
        margin=dict(t=20, b=20, l=10, r=10),
        height=360, showlegend=False
    )
    st.plotly_chart(fig_alpha, use_container_width=True)

# ════════════════════════════════════════════
#  TAB 6 — CAL SIMULATION TABLE
# ════════════════════════════════════════════
with tab6:
    st.markdown("#### 📋 Capital Allocation Line — 101 Portfolio Combinations")
    st.caption("Shifting allocation from 100% Risk-Free → 100% Optimal Risky Portfolio (1% steps)")

    cal_rows = []
    for i in range(0, 101):
        w_rf    = i / 100.0
        w_risky = 1.0 - w_rf
        c_ret   = w_rf * rf_rate + w_risky * opt_ret
        c_vol   = w_risky * opt_vol
        c_beta  = w_risky * port_beta
        c_capm  = rf_rate + c_beta * (mkt_annual - rf_rate)
        c_sharpe = (c_ret - rf_rate) / c_vol if c_vol > 0 else 0
        cal_rows.append({
            "Risk-Free Wt":    f"{w_rf:.0%}",
            "Risky Wt":        f"{w_risky:.0%}",
            "Exp. Return":     f"{c_ret:.2%}",
            "Volatility":      f"{c_vol:.2%}",
            "Portfolio Beta":  f"{c_beta:.4f}",
            "CAPM Req. Return":f"{c_capm:.2%}",
            "Sharpe Ratio":    f"{c_sharpe:.4f}",
        })

    cal_table = pd.DataFrame(cal_rows)

    # Line chart of CAL table
    w_risky_vals = np.linspace(0, 1, 101)
    c_rets_arr   = rf_rate + w_risky_vals * (opt_ret - rf_rate)
    c_vols_arr   = w_risky_vals * opt_vol

    fig_cal_tab = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Expected Return vs Risky Weight", "Volatility vs Risky Weight"]
    )
    fig_cal_tab.add_trace(
        go.Scatter(x=w_risky_vals * 100, y=c_rets_arr * 100,
                   mode="lines", line=dict(color="#34d399", width=2.5),
                   name="Expected Return",
                   hovertemplate="Risky Wt: %{x:.0f}%<br>Return: %{y:.2f}%<extra></extra>"),
        row=1, col=1
    )
    fig_cal_tab.add_trace(
        go.Scatter(x=w_risky_vals * 100, y=c_vols_arr * 100,
                   mode="lines", line=dict(color="#f59e0b", width=2.5),
                   name="Volatility",
                   hovertemplate="Risky Wt: %{x:.0f}%<br>Volatility: %{y:.2f}%<extra></extra>"),
        row=1, col=2
    )

    fig_cal_tab.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#131720",
        font_color="white",
        showlegend=False,
        height=300,
        margin=dict(t=40, b=20, l=10, r=10)
    )
    for axis in ["xaxis", "xaxis2"]:
        fig_cal_tab.update_layout(**{axis: dict(
            title="Risky Asset Weight (%)", gridcolor="#2d3748",
            ticksuffix="%", zeroline=False
        )})
    for axis in ["yaxis", "yaxis2"]:
        fig_cal_tab.update_layout(**{axis: dict(
            gridcolor="#2d3748", ticksuffix="%", zeroline=False
        )})

    st.plotly_chart(fig_cal_tab, use_container_width=True)

    # Table
    st.dataframe(
        cal_table.style.set_properties(**{
            "background-color": "#1a1f2e",
            "color": "white",
            "border-color": "#2d3748"
        }),
        use_container_width=True,
        hide_index=True,
        height=400
    )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#8892a4; font-size:12px; padding:12px 0;'>
    📊 Portfolio Optimizer &nbsp;|&nbsp; Data from Yahoo Finance &nbsp;|&nbsp;
    Period: {start_date} to {end_date} &nbsp;|&nbsp;
    Risk-Free Rate: {rf_rate:.1%} &nbsp;|&nbsp;
    {len(available_tickers)} assets optimized via SLSQP
    <br><br>
    <span style='color:#475569; font-size:11px;'>
        ⚠️ This dashboard is for educational purposes only. Past performance does not guarantee future results.
    </span>
</div>
""", unsafe_allow_html=True)
