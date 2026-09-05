import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu                                 { visibility: hidden; }
header[data-testid="stHeader"]            { background: transparent !important;
                                            height: 0px !important;
                                            min-height: 0 !important; }
[data-testid="stToolbar"]                 { display: none !important; }
.stDeployButton                           { display: none !important; }
footer                                    { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: flex !important;
                                            visibility: visible !important; }
.stApp                        { background: #f1f5f9; }
.block-container              { padding-top: 1.2rem !important;
                                padding-bottom: 1rem !important; }
[data-testid="stSidebar"]            { background: #1e293b !important; }
[data-testid="stSidebar"] label      { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stCaption { color: #94a3b8 !important; }
[data-testid="stSidebar"] p          { color: #cbd5e1 !important; }
[data-testid="stSidebar"] small      { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3         { color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stTextArea textarea {
    background: #0f172a !important; color: #e2e8f0 !important;
    border: 1px solid #475569 !important; border-radius: 8px !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: #0f172a !important; color: #e2e8f0 !important;
    border: 1px solid #475569 !important; border-radius: 8px !important; }
[data-testid="stSidebar"] .stDateInput input {
    background: #0f172a !important; color: #e2e8f0 !important;
    border: 1px solid #475569 !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] div { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox div { color: #e2e8f0 !important; }

[data-testid="stMetric"]      { background: #ffffff; border: 1px solid #e2e8f0;
                                 border-radius: 12px; padding: 14px !important;
                                 box-shadow: 0 1px 3px rgba(0,0,0,.07); }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 12px !important;
                                 font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #0f172a !important; font-size: 22px !important;
                                 font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

.stTabs [data-baseweb="tab-list"] { background: #ffffff; border-radius: 10px;
                                     padding: 4px; gap: 3px;
                                     border: 1px solid #e2e8f0;
                                     box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.stTabs [data-baseweb="tab"]      { border-radius: 7px; color: #64748b !important;
                                     font-weight: 600; font-size: 13px;
                                     padding: 7px 14px; border: none !important; }
.stTabs [aria-selected="true"]    { background: #2563eb !important;
                                     color: #ffffff !important; }
h1,h2,h3,h4 { color: #0f172a !important; }
p           { color: #334155; }
[data-testid="stDataFrame"] { border-radius: 10px; border: 1px solid #e2e8f0;
                               box-shadow: 0 1px 3px rgba(0,0,0,.05); }
[data-testid="stExpander"]  { background: #ffffff; border: 1px solid #e2e8f0;
                               border-radius: 10px; }
hr { border-color: #e2e8f0 !important; margin: 0.8rem 0 !important; }
::-webkit-scrollbar        { width:5px; height:5px; }
::-webkit-scrollbar-track  { background:#f1f5f9; }
::-webkit-scrollbar-thumb  { background:#cbd5e1; border-radius:3px; }
.stButton>button { border-radius: 9px !important; font-weight: 700 !important;
                   font-size: 14px !important; height: 44px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  THEME TOKENS
# ─────────────────────────────────────────────
C = {
    "bg"    : "#ffffff", "page"  : "#f1f5f9",
    "border": "#e2e8f0", "head"  : "#0f172a",
    "body"  : "#334155", "muted" : "#64748b",
    "blue"  : "#2563eb", "green" : "#16a34a",
    "red"   : "#dc2626", "amber" : "#d97706",
    "purple": "#7c3aed",
}
PALETTE = ["#2563eb","#16a34a","#d97706","#dc2626",
           "#7c3aed","#0891b2","#ea580c","#65a30d"]

# ─────────────────────────────────────────────
#  MARKET INDEX MAP  — friendly name → Yahoo ticker
# ─────────────────────────────────────────────
MARKET_MAP = {
    "Nifty 50"          : "^NSEI",
    "Nifty 500"         : "^CRSLDX",
    "Nifty Bank"        : "^NSEBANK",
    "Nifty IT"          : "^CNXIT",
    "Nifty Midcap 100"  : "^NSEMDCP100",
    "BSE Sensex"        : "^BSESN",
    "S&P 500 (US)"      : "^GSPC",
    "Nasdaq 100 (US)"   : "^NDX",
    "Dow Jones (US)"    : "^DJI",
}

# ─────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────
def apply_white_theme(fig, height=500, margin=None, hovermode="closest"):
    m = margin or dict(t=30, b=30, l=10, r=10)
    fig.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        height=height, margin=m, hovermode=hovermode,
        font=dict(color=C["body"], family="Inter, sans-serif"),
        legend=dict(bgcolor=C["bg"], bordercolor=C["border"],
                    borderwidth=1, font=dict(size=11, color=C["body"]))
    )

def style_axes(fig, xkw=None, ykw=None):
    xd = dict(gridcolor=C["border"], zeroline=False,
               tickfont=dict(size=11, color=C["muted"]),
               linecolor=C["border"], showline=True,
               title_font=dict(color=C["body"], size=12))
    yd = dict(gridcolor=C["border"], zeroline=False,
               tickfont=dict(size=11, color=C["muted"]),
               linecolor=C["border"], showline=True,
               title_font=dict(color=C["body"], size=12))
    if xkw: xd.update(xkw)
    if ykw: yd.update(ykw)
    fig.update_xaxes(**xd)
    fig.update_yaxes(**yd)

# ─────────────────────────────────────────────
#  TICKER AUTO-RESOLVER
#  User types  TCS / INFY / AAPL  (no suffix needed)
#  We try:  TCS.NS  →  TCS.BO  →  TCS  (US / crypto)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def resolve_ticker(raw: str) -> str | None:
    """Return the first valid Yahoo Finance ticker for a raw symbol."""
    raw = raw.strip().upper()
    # If user already added a suffix, honour it directly
    if "." in raw:
        try:
            d = yf.download(raw, period="5d", progress=False, auto_adjust=True)
            if not d.empty:
                return raw
        except Exception:
            pass
        return None

    # Otherwise try suffixes in order
    for suffix in [".NS", ".BO", ""]:
        candidate = raw + suffix
        try:
            d = yf.download(candidate, period="5d", progress=False, auto_adjust=True)
            if not d.empty:
                return candidate
        except Exception:
            continue
    return None


def short(t: str) -> str:
    """Strip exchange suffix for display."""
    return t.replace(".NS","").replace(".BO","").replace(".BSE","")

# ─────────────────────────────────────────────
#  FINANCE FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(resolved_tickers, mkt, start, end):
    all_t = list(resolved_tickers) + [mkt]
    raw   = yf.download(all_t, start=start, end=end,
                        auto_adjust=True, progress=False)["Close"]
    raw.dropna(how="all", inplace=True)
    avail = [t for t in resolved_tickers if t in raw.columns]
    return raw[avail].dropna(), raw[mkt].dropna(), avail


def compute_stats(sd, md):
    ret, mkt = sd.pct_change().dropna(), md.pct_change().dropna()
    ret, mkt = ret.align(mkt, join="inner", axis=0)
    return (ret, mkt,
            ret.mean()*252, ret.cov()*252, ret.corr(),
            float(mkt.mean()*252), float(np.var(mkt)))


def port_perf(w, mu, sig, rf):
    r  = float(np.dot(w, mu))
    v  = float(np.sqrt(np.dot(w.T, np.dot(sig, w))))
    sh = (r - rf) / v if v > 0 else 0.0
    return r, v, sh


def optimize_portfolio(mu, sig, rf):
    n = len(mu)
    res = sco.minimize(
        lambda w: -port_perf(w, mu, sig, rf)[2],
        n*[1/n], method="SLSQP",
        bounds=[(0,1)]*n,
        constraints={"type":"eq","fun":lambda x: np.sum(x)-1}
    )
    return res.x


def find_gmvp(mu, sig):
    n = len(mu)
    res = sco.minimize(
        lambda w: float(np.sqrt(np.dot(w.T, np.dot(sig, w)))),
        n*[1/n], method="SLSQP",
        bounds=[(0,1)]*n,
        constraints={"type":"eq","fun":lambda x: np.sum(x)-1}
    )
    if res.success:
        return float(np.dot(res.x, mu)), float(res.fun)
    return None, None


def min_var_vol(mu, sig, target):
    n = len(mu)
    res = sco.minimize(
        lambda w: float(np.sqrt(np.dot(w.T, np.dot(sig, w)))),
        n*[1/n], method="SLSQP",
        bounds=[(0,1)]*n,
        constraints=[
            {"type":"eq","fun":lambda x: np.sum(x)-1},
            {"type":"eq","fun":lambda x, t=target: np.dot(x,mu)-t}
        ]
    )
    return float(res.fun) if res.success else None


def get_betas(returns, mkt_ret, mkt_var):
    return {c: float(np.cov(returns[c], mkt_ret)[0,1]/mkt_var)
            for c in returns.columns}


def get_port_beta(port_daily, mkt_ret):
    return float(np.cov(port_daily, mkt_ret)[0,1]/np.var(mkt_ret))


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 18px 0;'>
        <div style='font-size:38px; line-height:1;'>📊</div>
        <div style='font-size:18px; font-weight:800; color:#f8fafc; margin-top:8px;'>
            Portfolio Optimizer</div>
        <div style='font-size:11px; color:#94a3b8; margin-top:4px;'>
            SLSQP · CAPM · MPT</div>
    </div>
    <hr style='border-color:#334155; margin:0 0 14px 0;'>
    """, unsafe_allow_html=True)

    # ── Date Range ──
    st.markdown("<p style='color:#93c5fd; font-size:10px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;'>📅 DATE RANGE</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("S", value=date(2020,1,1),
                                   max_value=date.today(), label_visibility="collapsed")
        st.caption("Start date")
    with c2:
        end_date = st.date_input("E", value=date.today(),
                                 max_value=date.today(), label_visibility="collapsed")
        st.caption("End date")

    st.markdown("<hr style='border-color:#334155; margin:10px 0;'>", unsafe_allow_html=True)

    # ── Stock Tickers ──
    st.markdown("<p style='color:#93c5fd; font-size:10px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px;'>📈 STOCK TICKERS</p>", unsafe_allow_html=True)
    st.caption("Just type the symbol — TCS, INFY, AAPL. One per line. No .NS needed.")

    ticker_input = st.text_area("Tickers", label_visibility="collapsed", height=210,
        value="TCS\nINFY\nHDFCBANK\nICICIBANK\nRELIANCE\n"
              "ITC\nSUNPHARMA\nBHARTIARTL\nM&M\nGOLDBEES")
    raw_tickers = [t.strip().upper() for t in ticker_input.strip().split("\n") if t.strip()]

    st.markdown("<hr style='border-color:#334155; margin:10px 0;'>", unsafe_allow_html=True)

    # ── Market Index ── friendly dropdown
    st.markdown("<p style='color:#93c5fd; font-size:10px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px;'>🏦 MARKET INDEX</p>", unsafe_allow_html=True)

    market_choice = st.selectbox(
        "Market Index",
        options=list(MARKET_MAP.keys()),
        index=0,          # default: Nifty 50
        label_visibility="collapsed"
    )
    market_ticker = MARKET_MAP[market_choice]
    st.caption(f"Yahoo Finance ticker: `{market_ticker}`")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Risk-Free Rate ──
    st.markdown("<p style='color:#93c5fd; font-size:10px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px;'>💰 RISK-FREE RATE</p>", unsafe_allow_html=True)
    rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 6.5, 0.1) / 100

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Run Optimization", use_container_width=True, type="primary")

    st.markdown("""
    <div style='margin-top:12px; background:#0f172a; border:1px solid #334155;
                border-radius:10px; padding:10px 12px; font-size:11px; color:#94a3b8;'>
        <b style='color:#60a5fa;'>💡 Tips:</b><br>
        • Just enter symbols: <b>TCS</b>, <b>AAPL</b><br>
        • NSE stocks auto-detected (.NS)<br>
        • US stocks also supported<br>
        • 3–20 stocks work best
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
            padding:16px 22px; margin-bottom:18px;
            box-shadow:0 1px 4px rgba(0,0,0,.06);'>
    <div style='display:flex; align-items:center; gap:12px;'>
        <div style='font-size:28px;'>📈</div>
        <div>
            <div style='font-size:22px; font-weight:800; color:#0f172a; line-height:1.2;'>
                Portfolio Optimization Dashboard</div>
            <div style='font-size:12px; color:#64748b; margin-top:3px;'>
                Modern Portfolio Theory &nbsp;·&nbsp; CAPM &nbsp;·&nbsp;
                Efficient Frontier &nbsp;·&nbsp; SLSQP Optimizer</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LANDING
# ─────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
                padding:48px 36px; text-align:center;
                box-shadow:0 1px 4px rgba(0,0,0,.06);'>
        <div style='font-size:56px; margin-bottom:12px;'>🏦</div>
        <div style='font-size:20px; font-weight:800; color:#0f172a; margin-bottom:8px;'>
            Configure your portfolio on the left panel</div>
        <div style='color:#64748b; font-size:13px; max-width:480px; margin:0 auto 24px;'>
            Type ticker symbols (no suffix needed) · Select market index ·
            Set date range · Click <b style='color:#2563eb;'>Run Optimization</b>
        </div>
        <div style='display:flex; justify-content:center; gap:10px; flex-wrap:wrap;'>
            <span style='background:#eff6ff; color:#1d4ed8; padding:6px 14px;
                         border-radius:999px; font-size:12px; font-weight:700;'>
                📊 Efficient Frontier</span>
            <span style='background:#f0fdf4; color:#15803d; padding:6px 14px;
                         border-radius:999px; font-size:12px; font-weight:700;'>
                📉 Capital Allocation Line</span>
            <span style='background:#faf5ff; color:#7e22ce; padding:6px 14px;
                         border-radius:999px; font-size:12px; font-weight:700;'>
                🎯 Security Market Line</span>
            <span style='background:#fff7ed; color:#c2410c; padding:6px 14px;
                         border-radius:999px; font-size:12px; font-weight:700;'>
                🔥 Correlation Matrix</span>
            <span style='background:#fefce8; color:#a16207; padding:6px 14px;
                         border-radius:999px; font-size:12px; font-weight:700;'>
                📋 CAPM Valuation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  RESOLVE TICKERS  (auto-detect exchange suffix)
# ─────────────────────────────────────────────
with st.spinner("🔍 Resolving ticker symbols..."):
    resolve_progress = st.empty()
    resolved_map   = {}   # raw → resolved ticker (e.g. TCS → TCS.NS)
    failed_tickers = []

    for raw in raw_tickers:
        resolved = resolve_ticker(raw)
        if resolved:
            resolved_map[raw] = resolved
        else:
            failed_tickers.append(raw)

    tickers = list(resolved_map.values())   # resolved tickers for download

if failed_tickers:
    st.warning(f"⚠️ Could not find: **{', '.join(failed_tickers)}** — skipped. "
               f"Check spelling or try adding suffix manually (e.g. TCS.NS).")

if len(tickers) < 2:
    st.error("❌ Need at least 2 valid tickers. Please check your inputs.")
    st.stop()

# Show which tickers were resolved
with st.expander(f"✅ {len(tickers)} tickers resolved — click to see mapping", expanded=False):
    res_df = pd.DataFrame([
        {"You entered": raw, "Resolved to": resolved_map[raw],
         "Exchange": "NSE" if resolved_map[raw].endswith(".NS")
                    else "BSE" if resolved_map[raw].endswith(".BO")
                    else "US / Other"}
        for raw in resolved_map
    ])
    st.dataframe(res_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  FETCH & COMPUTE
# ─────────────────────────────────────────────
with st.spinner(f"⏳ Fetching data from {start_date} to {end_date} and running SLSQP..."):
    try:
        stock_data, market_data, available = fetch_data(
            tuple(tickers), market_ticker, str(start_date), str(end_date))

        if len(available) < 2:
            st.error("❌ Not enough data returned. Try a different date range.")
            st.stop()

        (returns, mkt_ret, mean_ret,
         cov_mat, corr_mat, mkt_annual, mkt_var) = compute_stats(stock_data, market_data)

        opt_w                    = optimize_portfolio(mean_ret, cov_mat, rf_rate)
        opt_ret, opt_vol, opt_sh = port_perf(opt_w, mean_ret, cov_mat, rf_rate)
        gmvp_ret, gmvp_vol       = find_gmvp(mean_ret, cov_mat)

        port_daily = returns.dot(opt_w)
        port_beta  = get_port_beta(port_daily, mkt_ret)
        capm_ret   = rf_rate + port_beta*(mkt_annual - rf_rate)
        alpha      = opt_ret - capm_ret
        betas      = get_betas(returns, mkt_ret, mkt_var)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()


# ─────────────────────────────────────────────
#  KEY METRICS
# ─────────────────────────────────────────────
st.markdown("<p style='font-size:15px; font-weight:700; color:#0f172a; margin-bottom:10px;'>🎯 Optimal Portfolio — Key Metrics</p>", unsafe_allow_html=True)

m1,m2,m3,m4,m5,m6 = st.columns(6)
active_n = sum(1 for w in opt_w if w > 0.001)
m1.metric("📈 Annual Return",  f"{opt_ret:.2%}",   delta=f"+{opt_ret-rf_rate:.2%} vs Rf")
m2.metric("📉 Volatility",     f"{opt_vol:.2%}",   delta="Total Risk")
m3.metric("⚡ Sharpe Ratio",   f"{opt_sh:.4f}",    delta="Max Sharpe")
m4.metric("🔵 Beta",           f"{port_beta:.4f}", delta="Systematic Risk")
m5.metric("🏦 CAPM Return",    f"{capm_ret:.2%}",  delta=f"Alpha: {alpha:+.2%}")
m6.metric("✅ Active Stocks",  f"{active_n}/{len(available)}", delta="Non-zero weights")

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🏆  Optimal Portfolio",
    "📊  CAL & Efficient Frontier",
    "🎯  Security Market Line",
    "🔥  Correlation Matrix",
    "📉  Stock Valuation",
    "📋  CAL Simulation"
])


# ════════════════════════════════════════════
#  TAB 1 — OPTIMAL PORTFOLIO
# ════════════════════════════════════════════
with tab1:
    L, R = st.columns([1.1, 1], gap="large")

    with L:
        st.markdown("<p style='font-size:14px; font-weight:700; color:#0f172a;'>📦 Asset Allocation Weights</p>", unsafe_allow_html=True)

        wdf = pd.DataFrame({
            "Ticker": available,
            "Label":  [short(t) for t in available],
            "Weight": opt_w,
            "Return": [float(mean_ret[t]) for t in available],
            "Beta":   [betas[t] for t in available],
        }).sort_values("Weight", ascending=False).reset_index(drop=True)

        adf    = wdf[wdf["Weight"] > 0.001].copy()
        colors = PALETTE[:len(adf)]

        fig_bar = go.Figure(go.Bar(
            x=adf["Label"], y=adf["Weight"]*100,
            marker=dict(color=colors, line=dict(color="#fff", width=2)),
            text=[f"{w*100:.1f}%" for w in adf["Weight"]],
            textposition="outside",
            textfont=dict(color=C["head"], size=12),
            hovertemplate="<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>"
        ))
        apply_white_theme(fig_bar, height=280, margin=dict(t=10,b=10,l=10,r=10))
        style_axes(fig_bar,
                   xkw=dict(tickfont=dict(size=12, color=C["body"])),
                   ykw=dict(title="Weight (%)", ticksuffix="%"))
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        tbl = pd.DataFrame({
            "Ticker":       [short(t) for t in wdf["Ticker"]],
            "Weight %":     [f"{w:.2%}" for w in wdf["Weight"]],
            "Exp. Return":  [f"{r:.2%}" for r in wdf["Return"]],
            "Beta":         [f"{b:.4f}" for b in wdf["Beta"]],
            "In Portfolio": ["✅ Yes" if w > 0.001 else "⭕ No" for w in wdf["Weight"]]
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=275)

    with R:
        st.markdown("<p style='font-size:14px; font-weight:700; color:#0f172a;'>🥧 Portfolio Composition</p>", unsafe_allow_html=True)

        fig_pie = go.Figure(go.Pie(
            labels=adf["Label"],
            values=(adf["Weight"]*100).round(2),
            hole=0.56,
            marker=dict(colors=colors, line=dict(color="#fff", width=3)),
            textinfo="label+percent",
            textfont=dict(size=11, color=C["head"]),
            insidetextorientation="horizontal",
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>",
            direction="clockwise", sort=True
        ))
        apply_white_theme(fig_pie, height=280, margin=dict(t=10,b=10,l=10,r=10))
        fig_pie.update_layout(
            showlegend=False,
            annotations=[dict(
                text=f"<b>{opt_sh:.2f}</b><br><span style='font-size:11px'>Sharpe</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color=C["head"])
            )]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(f"""
        <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
                    padding:16px 18px; box-shadow:0 1px 4px rgba(0,0,0,.06);'>
            <p style='color:#64748b; font-size:10px; font-weight:800;
                      letter-spacing:1.5px; margin:0 0 12px 0;'>PORTFOLIO SUMMARY</p>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
                <div><div style='color:#64748b; font-size:11px;'>Annual Return</div>
                     <div style='color:#16a34a; font-size:20px; font-weight:800;'>{opt_ret:.2%}</div></div>
                <div><div style='color:#64748b; font-size:11px;'>Volatility</div>
                     <div style='color:#d97706; font-size:20px; font-weight:800;'>{opt_vol:.2%}</div></div>
                <div><div style='color:#64748b; font-size:11px;'>Sharpe Ratio</div>
                     <div style='color:#2563eb; font-size:20px; font-weight:800;'>{opt_sh:.4f}</div></div>
                <div><div style='color:#64748b; font-size:11px;'>Beta</div>
                     <div style='color:#7c3aed; font-size:20px; font-weight:800;'>{port_beta:.4f}</div></div>
                <div><div style='color:#64748b; font-size:11px;'>CAPM Return</div>
                     <div style='color:#dc2626; font-size:20px; font-weight:800;'>{capm_ret:.2%}</div></div>
                <div><div style='color:#64748b; font-size:11px;'>Jensen's Alpha</div>
                     <div style='color:{"#16a34a" if alpha>=0 else "#dc2626"}; font-size:20px; font-weight:800;'>{alpha:+.2%}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
#  TAB 2 — CAL & EFFICIENT FRONTIER
# ════════════════════════════════════════════
with tab2:
    st.markdown("<p style='font-size:14px; font-weight:700; color:#0f172a; margin-bottom:2px;'>📊 Capital Allocation Line & Efficient Frontier</p>", unsafe_allow_html=True)
    st.caption("5,000 random portfolios · Efficient frontier (upper portion from GMVP) · CAL from Rf through tangency point")

    with st.spinner("Generating efficient frontier..."):
        np.random.seed(42)
        n = len(available)
        sim_r, sim_v, sim_s = [], [], []
        for _ in range(5000):
            w = np.random.random(n); w /= w.sum()
            r, v, s = port_perf(w, mean_ret, cov_mat, rf_rate)
            sim_r.append(r); sim_v.append(v); sim_s.append(s)
        sim_r = np.array(sim_r)
        sim_v = np.array(sim_v)
        sim_s = np.array(sim_s)

        ef_start = gmvp_ret if gmvp_ret is not None else float(mean_ret.min())
        ef_end   = float(mean_ret.max())
        ef_v, ef_r = [], []
        for tr in np.linspace(ef_start, ef_end, 60):
            v = min_var_vol(mean_ret, cov_mat, tr)
            if v is not None:
                ef_v.append(v); ef_r.append(tr)

    cal_max = opt_vol * 1.55
    cal_x   = np.linspace(0, cal_max, 100)
    cal_y   = rf_rate + opt_sh * cal_x

    fig_cal = go.Figure()

    fig_cal.add_trace(go.Scatter(
        x=sim_v*100, y=sim_r*100, mode="markers",
        marker=dict(
            color=sim_s,
            colorscale=[[0,"#bfdbfe"],[0.5,"#3b82f6"],[1,"#1e3a8a"]],
            size=3.5, opacity=0.5,
            colorbar=dict(
                title=dict(text="Sharpe", font=dict(color=C["body"], size=11)),
                tickfont=dict(color=C["body"], size=10),
                x=1.01, thickness=12, len=0.65,
                bgcolor=C["bg"], bordercolor=C["border"]
            ), showscale=True
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Random Portfolio</extra>"
    ))

    if len(ef_v) > 3:
        fig_cal.add_trace(go.Scatter(
            x=[v*100 for v in ef_v], y=[r*100 for r in ef_r],
            mode="lines", line=dict(color=C["amber"], width=3.5),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Efficient Frontier</extra>"
        ))

    if gmvp_ret and gmvp_vol:
        fig_cal.add_trace(go.Scatter(
            x=[gmvp_vol*100], y=[gmvp_ret*100], mode="markers+text",
            marker=dict(color=C["amber"], size=11, symbol="diamond",
                        line=dict(color="white", width=2)),
            text=["GMVP"], textposition="bottom right",
            textfont=dict(color=C["amber"], size=11),
            name=f"GMVP ({gmvp_ret:.1%})",
            hovertemplate=(f"<b>GMVP</b><br>Return: {gmvp_ret:.2%}<br>"
                           f"Vol: {gmvp_vol:.2%}<extra></extra>")
        ))

    fig_cal.add_trace(go.Scatter(
        x=cal_x*100, y=cal_y*100, mode="lines",
        line=dict(color=C["blue"], width=2.5, dash="dash"),
        name=f"CAL (Sharpe={opt_sh:.2f})",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>CAL</extra>"
    ))

    fig_cal.add_trace(go.Scatter(
        x=[0], y=[rf_rate*100], mode="markers+text",
        marker=dict(color=C["blue"], size=10, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"Rf={rf_rate:.1%}"], textposition="top right",
        textfont=dict(color=C["blue"], size=11),
        name=f"Risk-Free ({rf_rate:.1%})",
        hovertemplate=f"Risk-Free: {rf_rate:.2%}<extra></extra>"
    ))

    fig_cal.add_trace(go.Scatter(
        x=[opt_vol*100], y=[opt_ret*100], mode="markers+text",
        marker=dict(color=C["red"], size=18, symbol="star",
                    line=dict(color="white", width=1.5)),
        text=["Optimal"], textposition="top right",
        textfont=dict(color=C["red"], size=12),
        name=f"Optimal (Sharpe={opt_sh:.2f})",
        hovertemplate=(f"<b>Optimal Portfolio</b><br>"
                       f"Return: {opt_ret:.2%}<br>Vol: {opt_vol:.2%}<br>"
                       f"Sharpe: {opt_sh:.4f}<extra></extra>")
    ))

    x_max = max(float(np.percentile(sim_v,99)), opt_vol)*100*1.08
    y_min = rf_rate*100*0.7
    y_max = max(float(np.percentile(sim_r,99)), opt_ret)*100*1.08

    apply_white_theme(fig_cal, height=550, margin=dict(t=20,b=20,l=10,r=70))
    style_axes(fig_cal,
               xkw=dict(title="Annual Volatility (%)", ticksuffix="%", range=[0, x_max]),
               ykw=dict(title="Expected Annual Return (%)", ticksuffix="%", range=[y_min, y_max]))
    fig_cal.update_layout(hovermode="closest")
    st.plotly_chart(fig_cal, use_container_width=True)

    with st.expander("📖 How to read this chart"):
        a, b = st.columns(2)
        with a:
            st.markdown("""
            - 🔵 **Blue dots** = 5,000 random portfolios. Darker = higher Sharpe.
            - 🟡 **Yellow curve** = Efficient Frontier — upper portion only (GMVP upward).
            - 🔷 **Yellow diamond** = GMVP — lowest-risk portfolio.
            """)
        with b:
            st.markdown("""
            - 🔵 **Blue dashed line** = CAL — Rf through optimal portfolio. Same Sharpe everywhere.
            - ⭐ **Red star** = Optimal (Tangency) Portfolio — maximum Sharpe.
            - 🔵 **Blue dot** = Risk-Free Rate at zero volatility.
            """)

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Optimal Return",     f"{opt_ret:.2%}")
    s2.metric("Optimal Volatility", f"{opt_vol:.2%}")
    s3.metric("Max Sharpe Ratio",   f"{opt_sh:.4f}")
    if gmvp_ret and gmvp_vol:
        s4.metric("GMVP", f"{gmvp_ret:.2%} ret | {gmvp_vol:.2%} vol")


# ════════════════════════════════════════════
#  TAB 3 — SECURITY MARKET LINE
# ════════════════════════════════════════════
with tab3:
    st.markdown("<p style='font-size:14px; font-weight:700; color:#0f172a; margin-bottom:2px;'>🎯 Security Market Line — Individual Stock Positioning</p>", unsafe_allow_html=True)

    beta_range  = np.linspace(0, max(betas.values())*1.25, 120)
    sml_returns = rf_rate + beta_range*(mkt_annual - rf_rate)

    fig_sml = go.Figure()
    fig_sml.add_trace(go.Scatter(
        x=beta_range, y=sml_returns*100, mode="lines",
        line=dict(color=C["blue"], width=2.5),
        name="Security Market Line",
        hovertemplate="Beta: %{x:.2f}<br>CAPM Return: %{y:.2f}%<extra>SML</extra>"
    ))

    for ticker in available:
        b      = betas[ticker]
        actual = float(mean_ret[ticker])
        capm_r = rf_rate + b*(mkt_annual - rf_rate)
        under  = actual > capm_r
        lbl    = short(ticker)
        fig_sml.add_trace(go.Scatter(
            x=[b], y=[actual*100], mode="markers+text",
            marker=dict(color=C["green"] if under else C["red"],
                        size=12, symbol="circle",
                        line=dict(color="white", width=1.5)),
            text=[lbl], textposition="top center",
            textfont=dict(size=10, color=C["head"]),
            name=lbl, showlegend=False,
            hovertemplate=(f"<b>{ticker}</b><br>Beta: {b:.4f}<br>"
                           f"Actual: {actual:.2%}<br>CAPM: {capm_r:.2%}<br>"
                           f"Alpha: {actual-capm_r:+.2%}<br>"
                           f"{'✅ Undervalued' if under else '❌ Overvalued'}<extra></extra>")
        ))

    fig_sml.add_trace(go.Scatter(
        x=[port_beta], y=[capm_ret*100], mode="markers+text",
        marker=dict(color=C["red"], size=17, symbol="star",
                    line=dict(color="white", width=1.5)),
        text=["Optimal"], textposition="top right",
        textfont=dict(color=C["red"], size=12),
        name="Optimal Portfolio",
        hovertemplate=f"Beta: {port_beta:.4f}<br>CAPM: {capm_ret:.2%}<extra>Optimal</extra>"
    ))
    fig_sml.add_trace(go.Scatter(
        x=[1.0], y=[mkt_annual*100], mode="markers+text",
        marker=dict(color=C["purple"], size=13, symbol="square",
                    line=dict(color="white", width=1.5)),
        text=["Market"], textposition="top right",
        textfont=dict(color=C["purple"], size=12),
        name=f"Market ({market_choice})",
        hovertemplate=f"Market Return: {mkt_annual:.2%}<extra>Market</extra>"
    ))
    fig_sml.add_trace(go.Scatter(
        x=[0], y=[rf_rate*100], mode="markers+text",
        marker=dict(color=C["blue"], size=10, symbol="circle",
                    line=dict(color="white", width=2)),
        text=["Rf"], textposition="top right",
        textfont=dict(color=C["blue"], size=12),
        name=f"Risk-Free ({rf_rate:.1%})",
        hovertemplate=f"Rf: {rf_rate:.2%}<extra></extra>"
    ))

    apply_white_theme(fig_sml, height=530)
    style_axes(fig_sml,
               xkw=dict(title="Beta (Systematic Risk)"),
               ykw=dict(title="Expected Return (%)", ticksuffix="%"))
    fig_sml.update_layout(hovermode="closest")
    st.plotly_chart(fig_sml, use_container_width=True)

    with st.expander("📖 How to read this chart"):
        a, b = st.columns(2)
        with a:
            st.markdown("""
            - **Blue line** = SML — CAPM required return for each beta level
            - 🟢 **Green dots** = Undervalued (actual > CAPM → positive alpha)
            - 🔴 **Red dots** = Overvalued (actual < CAPM → negative alpha)
            """)
        with b:
            st.markdown("""
            - 🟣 **Purple square** = Market portfolio (beta = 1 by definition)
            - ⭐ **Red star** = Optimal portfolio on SML
            - Hover over any dot for full alpha details
            """)


# ════════════════════════════════════════════
#  TAB 4 — CORRELATION MATRIX
# ════════════════════════════════════════════
with tab4:
    st.markdown("<p style='font-size:14px; font-weight:700; color:#0f172a; margin-bottom:2px;'>🔥 Correlation Matrix — Diversification Analysis</p>", unsafe_allow_html=True)

    slbls = [short(t) for t in available]
    cvals = corr_mat.values

    fig_heat = go.Figure(go.Heatmap(
        z=cvals, x=slbls, y=slbls,
        colorscale=[
            [0.00,"#1e40af"],[0.35,"#93c5fd"],
            [0.50,"#f8fafc"],[0.65,"#fca5a5"],[1.00,"#dc2626"]
        ],
        zmin=-1, zmax=1,
        text=np.round(cvals,2), texttemplate="%{text}",
        textfont=dict(size=11, color=C["head"]),
        hoverongaps=False,
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color=C["body"], size=12)),
            tickfont=dict(color=C["body"], size=11),
            tickvals=[-1,-0.5,0,0.5,1],
            ticktext=["-1.0","-0.5","0.0","+0.5","+1.0"],
            thickness=14, len=0.8, bgcolor=C["bg"], bordercolor=C["border"]
        )
    ))
    fig_heat.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        height=530, margin=dict(t=20,b=80,l=80,r=20),
        font=dict(color=C["body"], family="Inter, sans-serif"),
        hovermode="closest"
    )
    fig_heat.update_xaxes(tickangle=-40, tickfont=dict(size=12,color=C["body"]),
                          side="bottom", showgrid=False, linecolor=C["border"])
    fig_heat.update_yaxes(tickfont=dict(size=12,color=C["body"]),
                          autorange="reversed", showgrid=False, linecolor=C["border"])
    st.plotly_chart(fig_heat, use_container_width=True)

    flat = cvals[np.triu_indices_from(cvals, k=1)]
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Average Correlation", f"{flat.mean():.4f}", delta="Lower = more diversification")
    s2.metric("Max Correlation",     f"{flat.max():.4f}", delta="Most similar pair")
    s3.metric("Min Correlation",     f"{flat.min():.4f}", delta="Most different pair")
    s4.metric("Negative Pairs",      f"{(flat<0).sum()} / {len(flat)}", delta="Pairs with hedge benefit")

    with st.expander("📖 How to read this chart"):
        st.markdown("""
        - 🔴 **Red** = high positive correlation → stocks move together → less diversification
        - 🔵 **Blue** = low/negative correlation → stocks move independently → more diversification
        - **Diagonal** is always 1.0 (stock with itself)
        """)


# ════════════════════════════════════════════
#  TAB 5 — STOCK VALUATION
# ════════════════════════════════════════════
with tab5:
    st.markdown("<p style='font-size:14px; font-weight:700; color:#0f172a; margin-bottom:2px;'>📉 Individual Stock Valuation — CAPM vs Actual Return</p>", unsafe_allow_html=True)

    rows = []
    for t in available:
        b      = betas[t]
        actual = float(mean_ret[t])
        capm_r = rf_rate + b*(mkt_annual - rf_rate)
        ai     = actual - capm_r
        rows.append({"Ticker":t, "Label":short(t), "Beta":b,
                     "Actual":actual, "CAPM":capm_r, "Alpha":ai,
                     "Under": actual > capm_r,
                     "Weight": opt_w[available.index(t)]})

    vdf   = pd.DataFrame(rows).sort_values("Alpha", ascending=False)
    un_df = vdf[vdf["Under"]]
    ov_df = vdf[~vdf["Under"]]

    def val_card(row, is_under):
        color    = C["green"] if is_under else C["red"]
        brd      = "#bbf7d0" if is_under else "#fecaca"
        bg_badge = "#dcfce7" if is_under else "#fee2e2"
        return f"""
        <div style='background:#ffffff; border:1px solid {brd};
                    border-radius:10px; padding:13px 16px; margin-bottom:8px;
                    box-shadow:0 1px 3px rgba(0,0,0,.05);'>
            <div style='display:flex; justify-content:space-between;
                        align-items:center; margin-bottom:8px;'>
                <span style='color:{color}; font-weight:800; font-size:15px;'>
                    {row["Label"]}</span>
                <span style='background:{bg_badge}; color:{color}; padding:3px 10px;
                             border-radius:999px; font-size:12px; font-weight:700;'>
                    α = {row["Alpha"]:+.2%}</span>
            </div>
            <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:6px;'>
                <div><div style='color:#64748b;font-size:10px;'>Beta</div>
                     <div style='color:#0f172a;font-weight:700;font-size:13px;'>{row["Beta"]:.4f}</div></div>
                <div><div style='color:#64748b;font-size:10px;'>Actual</div>
                     <div style='color:{color};font-weight:700;font-size:13px;'>{row["Actual"]:.2%}</div></div>
                <div><div style='color:#64748b;font-size:10px;'>CAPM Req.</div>
                     <div style='color:{C["amber"]};font-weight:700;font-size:13px;'>{row["CAPM"]:.2%}</div></div>
                <div><div style='color:#64748b;font-size:10px;'>Port. Wt</div>
                     <div style='color:{C["blue"]};font-weight:700;font-size:13px;'>{row["Weight"]:.2%}</div></div>
            </div>
        </div>"""

    hc1, hc2 = st.columns(2, gap="large")

    with hc1:
        st.markdown(f"""
        <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px;
                    padding:12px 16px; margin-bottom:12px;'>
            <span style='color:{C["green"]}; font-weight:800; font-size:15px;'>
                ✅ Undervalued — {len(un_df)} stock{"s" if len(un_df)!=1 else ""}</span>
            <div style='color:#15803d; font-size:12px; margin-top:3px;'>
                Actual return &gt; CAPM → Positive Alpha → Above SML</div>
        </div>""", unsafe_allow_html=True)
        for _, row in un_df.iterrows():
            st.markdown(val_card(row, True), unsafe_allow_html=True)

    with hc2:
        st.markdown(f"""
        <div style='background:#fef2f2; border:1px solid #fecaca; border-radius:12px;
                    padding:12px 16px; margin-bottom:12px;'>
            <span style='color:{C["red"]}; font-weight:800; font-size:15px;'>
                ❌ Overvalued — {len(ov_df)} stock{"s" if len(ov_df)!=1 else ""}</span>
            <div style='color:#b91c1c; font-size:12px; margin-top:3px;'>
                Actual return &lt; CAPM → Negative Alpha → Below SML</div>
        </div>""", unsafe_allow_html=True)
        for _, row in ov_df.iterrows():
            st.markdown(val_card(row, False), unsafe_allow_html=True)

    st.markdown("<p style='font-size:14px; font-weight:700; color:#0f172a; margin:16px 0 6px;'>📊 Jensen's Alpha — All Stocks</p>", unsafe_allow_html=True)
    vs = vdf.sort_values("Alpha")
    fig_a = go.Figure(go.Bar(
        x=vs["Label"], y=vs["Alpha"]*100,
        marker=dict(color=[C["green"] if a>0 else C["red"] for a in vs["Alpha"]],
                    opacity=0.85, line=dict(color="rgba(255,255,255,0.6)", width=1)),
        text=[f"{a:+.2f}%" for a in vs["Alpha"]*100],
        textposition="outside", textfont=dict(color=C["head"], size=11),
        hovertemplate="<b>%{x}</b><br>Alpha: %{y:.2f}%<extra></extra>"
    ))
    fig_a.add_hline(y=0, line_color=C["muted"], line_width=1.5, line_dash="dot")
    apply_white_theme(fig_a, height=310, margin=dict(t=20,b=20,l=10,r=10))
    style_axes(fig_a,
               xkw=dict(tickfont=dict(size=12, color=C["body"])),
               ykw=dict(title="Jensen's Alpha (%)", ticksuffix="%"))
    fig_a.update_layout(showlegend=False, hovermode="closest")
    st.plotly_chart(fig_a, use_container_width=True)


# ════════════════════════════════════════════
#  TAB 6 — CAL SIMULATION
# ════════════════════════════════════════════
with tab6:
    st.markdown("<p style='font-size:14px; font-weight:700; color:#0f172a; margin-bottom:2px;'>📋 Capital Allocation Line — 101 Portfolio Combinations</p>", unsafe_allow_html=True)
    st.caption("Shifting 1% at a time: 100% Risk-Free → 100% Optimal Risky Portfolio")

    w_arr = np.linspace(0, 1, 101)
    r_arr = rf_rate + w_arr*(opt_ret - rf_rate)
    v_arr = w_arr * opt_vol

    fig_lines = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Expected Return vs Risky Weight",
                        "Volatility vs Risky Weight"]
    )
    fig_lines.add_trace(go.Scatter(
        x=w_arr*100, y=r_arr*100, mode="lines",
        line=dict(color=C["green"], width=2.5), name="Return",
        hovertemplate="Risky Wt: %{x:.0f}%<br>Return: %{y:.2f}%<extra></extra>"
    ), 1, 1)
    fig_lines.add_trace(go.Scatter(
        x=w_arr*100, y=v_arr*100, mode="lines",
        line=dict(color=C["amber"], width=2.5), name="Volatility",
        hovertemplate="Risky Wt: %{x:.0f}%<br>Vol: %{y:.2f}%<extra></extra>"
    ), 1, 2)

    fig_lines.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        height=270, margin=dict(t=40,b=20,l=10,r=10),
        showlegend=False, font=dict(color=C["body"], family="Inter, sans-serif")
    )
    fig_lines.update_xaxes(gridcolor=C["border"], zeroline=False, ticksuffix="%",
                           tickfont=dict(size=10, color=C["muted"]),
                           title_text="Risky Weight (%)",
                           title_font=dict(color=C["body"], size=11))
    fig_lines.update_yaxes(gridcolor=C["border"], zeroline=False,
                           tickfont=dict(size=10, color=C["muted"]))
    for ann in fig_lines.layout.annotations:
        ann.font.color = C["head"]; ann.font.size = 12

    st.plotly_chart(fig_lines, use_container_width=True)

    cal_rows = []
    for i in range(101):
        wrf=i/100; wr=1-wrf
        cr = wrf*rf_rate + wr*opt_ret
        cv = wr*opt_vol
        cb = wr*port_beta
        cc = rf_rate + cb*(mkt_annual - rf_rate)
        cs = (cr-rf_rate)/cv if cv > 0 else 0.0
        cal_rows.append({
            "Risk-Free Wt":     f"{wrf:.0%}",
            "Risky Wt":         f"{wr:.0%}",
            "Exp. Return":      f"{cr:.2%}",
            "Volatility":       f"{cv:.2%}",
            "Beta":             f"{cb:.4f}",
            "CAPM Req. Return": f"{cc:.2%}",
            "Sharpe Ratio":     f"{cs:.4f}",
        })
    st.dataframe(pd.DataFrame(cal_rows),
                 use_container_width=True, hide_index=True, height=420)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:{C["muted"]}; font-size:12px; padding:8px 0;'>
    📊 Portfolio Optimizer &nbsp;·&nbsp; Data: Yahoo Finance &nbsp;·&nbsp;
    {start_date} → {end_date} &nbsp;·&nbsp; Rf: {rf_rate:.1%} &nbsp;·&nbsp;
    Market: {market_choice} &nbsp;·&nbsp; {len(available)} assets via SLSQP
    <br><span style='color:#94a3b8; font-size:11px; margin-top:4px; display:block;'>
        ⚠️ Educational purposes only. Past performance does not guarantee future results.
    </span>
</div>
""", unsafe_allow_html=True)
