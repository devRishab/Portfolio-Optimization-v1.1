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
#  GLOBAL COLORS  (change here to retheme)
# ─────────────────────────────────────────────
BG_PAGE   = "#f0f4f8"   # page background
BG_CARD   = "#ffffff"   # card / chart background
BG_SIDE   = "#1e293b"   # sidebar background (keep dark for contrast)
BORDER    = "#e2e8f0"   # card borders
TXT_HEAD  = "#0f172a"   # heading text
TXT_BODY  = "#334155"   # body text
TXT_MUTED = "#64748b"   # muted / label text
ACCENT    = "#2563eb"   # primary blue
GREEN     = "#16a34a"
RED       = "#dc2626"
YELLOW    = "#d97706"

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
    /* ── Hide Streamlit chrome ── */
    #MainMenu                        {{ visibility: hidden; }}
    header[data-testid="stHeader"]   {{ display: none !important; }}
    [data-testid="stToolbar"]        {{ display: none !important; }}
    .stDeployButton                  {{ display: none !important; }}
    footer                           {{ display: none !important; }}

    /* ── Page background ── */
    .stApp                           {{ background-color: {BG_PAGE}; }}
    .block-container                 {{ padding-top:1.4rem !important;
                                       padding-bottom:1rem !important; }}

    /* ── Sidebar (keep dark) ── */
    [data-testid="stSidebar"]        {{ background:{BG_SIDE} !important;
                                       border-right:1px solid #334155; }}
    [data-testid="stSidebar"] *      {{ color:#e2e8f0 !important; }}
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stTextInput input  {{
        background:#0f172a !important;
        color:#e2e8f0 !important;
        border:1px solid #475569 !important;
        border-radius:8px !important;
    }}
    [data-testid="stSidebar"] .stDateInput input  {{
        background:#0f172a !important;
        color:#e2e8f0 !important;
        border:1px solid #475569 !important;
    }}

    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    [data-testid="stMetricLabel"] {{
        color: {TXT_MUTED} !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {TXT_HEAD} !important;
        font-size: 22px !important;
        font-weight: 800 !important;
    }}
    [data-testid="stMetricDelta"] {{ font-size: 12px !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {BG_CARD};
        border-radius: 12px;
        padding: 4px 6px;
        gap: 4px;
        border: 1px solid {BORDER};
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {TXT_MUTED} !important;
        font-weight: 600;
        font-size: 13px;
        padding: 7px 16px;
        border: none !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {ACCENT} !important;
        color: #ffffff !important;
    }}
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
        background: #eff6ff !important;
        color: {ACCENT} !important;
    }}

    /* ── Headings on main canvas ── */
    h1,h2,h3,h4 {{ color: {TXT_HEAD} !important; }}
    p, li        {{ color: {TXT_BODY}; }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid {BORDER};
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}

    /* ── Expander ── */
    [data-testid="stExpander"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}

    /* ── Divider ── */
    hr {{ border-color: {BORDER} !important; margin: 0.8rem 0 !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar          {{ width:6px; height:6px; }}
    ::-webkit-scrollbar-track    {{ background:{BG_PAGE}; }}
    ::-webkit-scrollbar-thumb    {{ background:{BORDER}; border-radius:3px; }}

    /* ── Card helper ── */
    .wcard {{
        background:{BG_CARD};
        border:1px solid {BORDER};
        border-radius:14px;
        padding:18px 20px;
        margin-bottom:14px;
        box-shadow:0 1px 6px rgba(0,0,0,0.06);
    }}

    /* ── Sidebar label ── */
    .slbl {{
        color:#93c5fd;
        font-size:10px;
        font-weight:800;
        letter-spacing:1.8px;
        text-transform:uppercase;
        margin-bottom:6px;
    }}

    /* ── Button ── */
    .stButton>button {{
        border-radius:10px !important;
        font-weight:700 !important;
        font-size:15px !important;
        height:46px;
    }}

    /* ── Caption / small text on main ── */
    .stCaption {{ color:{TXT_MUTED} !important; }}
    small       {{ color:{TXT_MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────
def base_layout(h=520, margin=None):
    m = margin or dict(t=30, b=30, l=10, r=10)
    return dict(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TXT_BODY, family="Inter, sans-serif"),
        height=h, margin=m, hovermode="closest",
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER,
                    borderwidth=1, font=dict(size=12, color=TXT_BODY))
    )


def ax_style(title="", sfx=""):
    return dict(
        title=dict(text=title, font=dict(color=TXT_BODY, size=12)),
        gridcolor=BORDER, zeroline=False,
        tickfont=dict(size=11, color=TXT_MUTED),
        ticksuffix=sfx,
        linecolor=BORDER, showline=True,
        color=TXT_BODY
    )


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def short(t):
    return (t.replace(".NS","").replace(".BO","")
             .replace(".BSE","").replace(".NYSE",""))


@st.cache_data(show_spinner=False)
def fetch_data(tickers, mkt, start, end):
    raw = yf.download(list(tickers)+[mkt], start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw.dropna(how="all", inplace=True)
    avail = [t for t in tickers if t in raw.columns]
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
    sh = (r-rf)/v if v > 0 else 0
    return r, v, sh


def optimize_portfolio(mu, sig, rf):
    n = len(mu)
    res = sco.minimize(
        lambda w: -port_perf(w, mu, sig, rf)[2],
        n*[1/n], method="SLSQP",
        bounds=tuple((0,1) for _ in range(n)),
        constraints={"type":"eq","fun":lambda x:np.sum(x)-1}
    )
    return res.x


def find_gmvp(mu, sig):
    n = len(mu)
    res = sco.minimize(
        lambda w: np.sqrt(np.dot(w.T, np.dot(sig, w))),
        n*[1/n], method="SLSQP",
        bounds=tuple((0,1) for _ in range(n)),
        constraints={"type":"eq","fun":lambda x:np.sum(x)-1}
    )
    if res.success:
        return float(np.dot(res.x, mu)), float(res.fun)
    return None, None


def min_var_vol(mu, sig, target):
    n = len(mu)
    res = sco.minimize(
        lambda w: np.sqrt(np.dot(w.T, np.dot(sig, w))),
        n*[1/n], method="SLSQP",
        bounds=tuple((0,1) for _ in range(n)),
        constraints=[
            {"type":"eq","fun":lambda x:np.sum(x)-1},
            {"type":"eq","fun":lambda x,t=target:np.dot(x,mu)-t}
        ]
    )
    return float(res.fun) if res.success else None


def ind_betas(returns, mkt_ret, mkt_var):
    return {c: np.cov(returns[c], mkt_ret)[0,1]/mkt_var
            for c in returns.columns}


def compute_beta(pd_series, mkt_ret):
    return np.cov(pd_series, mkt_ret)[0,1] / np.var(mkt_ret)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 20px 0;'>
        <div style='font-size:42px;line-height:1;'>📊</div>
        <div style='font-size:19px;font-weight:800;color:#ffffff;margin-top:8px;'>
            Portfolio Optimizer</div>
        <div style='font-size:11px;color:#94a3b8;margin-top:3px;'>
            SLSQP · CAPM · MPT · Efficient Frontier</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="slbl">📅 Date Range</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        start_date = st.date_input("S", value=date(2020,1,1),
                                   max_value=date.today(), label_visibility="collapsed")
        st.caption("Start")
    with c2:
        end_date = st.date_input("E", value=date.today(),
                                 max_value=date.today(), label_visibility="collapsed")
        st.caption("End")

    st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
    st.markdown('<div class="slbl">📈 Stock Tickers</div>', unsafe_allow_html=True)
    st.caption("NSE → TCS.NS  |  US → AAPL  |  One per line")

    ticker_input = st.text_area("T", label_visibility="collapsed", height=210,
        value="TCS.NS\nINFY.NS\nHDFCBANK.NS\nICICIBANK.NS\nRELIANCE.NS\n"
              "ITC.NS\nSUNPHARMA.NS\nBHARTIARTL.NS\nM&M.NS\nGOLDBEES.NS")
    tickers = [t.strip().upper() for t in ticker_input.strip().split("\n") if t.strip()]

    st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
    st.markdown('<div class="slbl">🏦 Market Settings</div>', unsafe_allow_html=True)
    market_ticker = st.text_input("M", value="^NSEI", label_visibility="collapsed",
                                  help="Nifty 50 = ^NSEI | S&P 500 = ^GSPC")
    st.caption("Market index ticker")
    rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 6.5, 0.1) / 100

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Run Optimization", use_container_width=True, type="primary")

    st.markdown("""
    <div style='margin-top:14px;background:#0f172a;border:1px solid #334155;
                border-radius:10px;padding:10px 12px;font-size:11px;color:#94a3b8;'>
        <b style='color:#60a5fa;'>💡 Tip:</b> Use 3–20 stocks for best results.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:14px;
            padding:16px 22px;margin-bottom:20px;
            box-shadow:0 1px 6px rgba(0,0,0,0.06);'>
    <div style='display:flex;align-items:center;gap:12px;'>
        <div style='font-size:30px;'>📈</div>
        <div>
            <div style='font-size:24px;font-weight:800;color:{TXT_HEAD};line-height:1.15;'>
                Portfolio Optimization Dashboard</div>
            <div style='font-size:13px;color:{TXT_MUTED};margin-top:3px;'>
                Modern Portfolio Theory &nbsp;·&nbsp; CAPM &nbsp;·&nbsp;
                Efficient Frontier &nbsp;·&nbsp; SLSQP Optimizer</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LANDING
# ─────────────────────────────────────────────
if not run_btn:
    st.markdown(f"""
    <div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:16px;
                padding:52px 40px;text-align:center;
                box-shadow:0 1px 6px rgba(0,0,0,0.06);'>
        <div style='font-size:60px;margin-bottom:14px;'>🏦</div>
        <div style='font-size:22px;font-weight:800;color:{TXT_HEAD};margin-bottom:8px;'>
            Configure your portfolio on the left panel</div>
        <div style='color:{TXT_MUTED};font-size:14px;max-width:480px;margin:0 auto 28px auto;'>
            Enter stock tickers, set a date range and risk-free rate,
            then click <b style='color:{ACCENT};'>Run Optimization</b>.</div>
        <div style='display:flex;justify-content:center;gap:12px;flex-wrap:wrap;'>
            <span style='background:#eff6ff;color:#1d4ed8;padding:7px 16px;
                         border-radius:999px;font-size:12px;font-weight:700;'>
                📊 Efficient Frontier</span>
            <span style='background:#f0fdf4;color:#15803d;padding:7px 16px;
                         border-radius:999px;font-size:12px;font-weight:700;'>
                📉 Capital Allocation Line</span>
            <span style='background:#faf5ff;color:#7e22ce;padding:7px 16px;
                         border-radius:999px;font-size:12px;font-weight:700;'>
                🎯 Security Market Line</span>
            <span style='background:#fff7ed;color:#c2410c;padding:7px 16px;
                         border-radius:999px;font-size:12px;font-weight:700;'>
                🔥 Correlation Heatmap</span>
            <span style='background:#fefce8;color:#a16207;padding:7px 16px;
                         border-radius:999px;font-size:12px;font-weight:700;'>
                📋 CAPM Valuation</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  FETCH & COMPUTE
# ─────────────────────────────────────────────
with st.spinner("⏳ Fetching market data and running SLSQP optimization..."):
    try:
        stock_data, market_data, available = fetch_data(
            tuple(tickers), market_ticker, str(start_date), str(end_date))
        if len(available) < 2:
            st.error("❌ Need at least 2 valid tickers.")
            st.stop()

        (returns, mkt_ret, mean_ret,
         cov_mat, corr_mat, mkt_annual, mkt_var) = compute_stats(stock_data, market_data)

        opt_w                    = optimize_portfolio(mean_ret, cov_mat, rf_rate)
        opt_ret, opt_vol, opt_sh = port_perf(opt_w, mean_ret, cov_mat, rf_rate)
        gmvp_ret, gmvp_vol       = find_gmvp(mean_ret, cov_mat)

        port_daily = returns.dot(opt_w)
        port_beta  = compute_beta(port_daily, mkt_ret)
        capm_ret   = rf_rate + port_beta*(mkt_annual - rf_rate)
        alpha      = opt_ret - capm_ret
        betas      = ind_betas(returns, mkt_ret, mkt_var)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

skipped = [t for t in tickers if t not in available]
if skipped:
    st.warning(f"⚠️ Could not fetch: **{', '.join(skipped)}** — excluded.")

# ─────────────────────────────────────────────
#  KEY METRICS
# ─────────────────────────────────────────────
st.markdown(f"<div style='font-size:16px;font-weight:700;color:{TXT_HEAD};margin-bottom:10px;'>🎯 Optimal Portfolio — Key Metrics</div>", unsafe_allow_html=True)

m1,m2,m3,m4,m5,m6 = st.columns(6)
active_n = sum(1 for w in opt_w if w > 0.001)
m1.metric("📈 Annual Return",   f"{opt_ret:.2%}",   delta=f"+{opt_ret-rf_rate:.2%} vs Rf")
m2.metric("📉 Volatility",      f"{opt_vol:.2%}",   delta="Total Risk")
m3.metric("⚡ Sharpe Ratio",    f"{opt_sh:.4f}",    delta="Max Sharpe")
m4.metric("🔵 Portfolio Beta",  f"{port_beta:.4f}", delta="Systematic Risk")
m5.metric("🏦 CAPM Return",     f"{capm_ret:.2%}",  delta=f"Alpha: {alpha:+.2%}")
m6.metric("✅ Active Stocks",   f"{active_n}/{len(available)}", delta="Non-zero weights")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
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

# palette for pie / bar
PALETTE = ["#2563eb","#16a34a","#d97706","#dc2626",
           "#7c3aed","#0891b2","#ea580c","#65a30d",
           "#db2777","#0369a1"]

# ════════════════════════════════════════════
#  TAB 1 — OPTIMAL PORTFOLIO
# ════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1.1,1], gap="large")

    with col_l:
        st.markdown(f"<div style='font-size:15px;font-weight:700;color:{TXT_HEAD};margin-bottom:8px;'>📦 Asset Allocation Weights</div>", unsafe_allow_html=True)

        wdf = pd.DataFrame({
            "Ticker": available,
            "Label":  [short(t) for t in available],
            "Weight": opt_w,
            "Return": [float(mean_ret[t]) for t in available],
            "Beta":   [betas[t] for t in available],
        }).sort_values("Weight", ascending=False).reset_index(drop=True)

        active_df = wdf[wdf["Weight"]>0.001].copy()
        max_w     = active_df["Weight"].max()
        colors    = PALETTE[:len(active_df)]

        fig_bar = go.Figure(go.Bar(
            x=active_df["Label"],
            y=active_df["Weight"]*100,
            marker=dict(color=colors,
                        line=dict(color="rgba(255,255,255,0.6)", width=2)),
            text=[f"{w*100:.1f}%" for w in active_df["Weight"]],
            textposition="outside",
            textfont=dict(color=TXT_HEAD, size=13, family="Inter"),
            hovertemplate="<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>"
        ))
        fig_bar.update_layout(
            **base_layout(h=290, margin=dict(t=10,b=10,l=10,r=10)),
            xaxis=dict(tickfont=dict(size=12,color=TXT_BODY),
                       gridcolor=BORDER, linecolor=BORDER),
            yaxis=dict(title="Weight (%)", gridcolor=BORDER,
                       ticksuffix="%", zeroline=False,
                       tickfont=dict(color=TXT_BODY)),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        tbl = pd.DataFrame({
            "Ticker":       [short(t) for t in wdf["Ticker"]],
            "Weight %":     [f"{w:.2%}" for w in wdf["Weight"]],
            "Exp. Return":  [f"{r:.2%}" for r in wdf["Return"]],
            "Beta":         [f"{b:.4f}" for b in wdf["Beta"]],
            "In Portfolio": ["✅ Yes" if w>0.001 else "⭕ No" for w in wdf["Weight"]]
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=280)

    with col_r:
        st.markdown(f"<div style='font-size:15px;font-weight:700;color:{TXT_HEAD};margin-bottom:8px;'>🥧 Portfolio Composition</div>", unsafe_allow_html=True)

        fig_pie = go.Figure(go.Pie(
            labels=active_df["Label"],
            values=(active_df["Weight"]*100).round(2),
            hole=0.58,
            marker=dict(colors=colors,
                        line=dict(color="#ffffff", width=3)),
            textinfo="label+percent",
            textfont=dict(size=12, color=TXT_HEAD),
            insidetextorientation="horizontal",
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>",
            direction="clockwise", sort=True
        ))
        fig_pie.update_layout(
            **base_layout(h=290, margin=dict(t=10,b=10,l=10,r=10)),
            showlegend=False,
            annotations=[dict(
                text=f"<b style='color:{TXT_HEAD}'>{opt_sh:.2f}</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=22, color=TXT_HEAD)
            )]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Summary card
        st.markdown(f"""
        <div class="wcard">
            <div style='color:{TXT_MUTED};font-size:10px;font-weight:800;
                        letter-spacing:1.5px;margin-bottom:14px;'>PORTFOLIO SUMMARY</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>
                <div><div style='color:{TXT_MUTED};font-size:11px;'>Annual Return</div>
                     <div style='color:{GREEN};font-size:22px;font-weight:800;'>{opt_ret:.2%}</div></div>
                <div><div style='color:{TXT_MUTED};font-size:11px;'>Volatility</div>
                     <div style='color:{YELLOW};font-size:22px;font-weight:800;'>{opt_vol:.2%}</div></div>
                <div><div style='color:{TXT_MUTED};font-size:11px;'>Sharpe Ratio</div>
                     <div style='color:{ACCENT};font-size:22px;font-weight:800;'>{opt_sh:.4f}</div></div>
                <div><div style='color:{TXT_MUTED};font-size:11px;'>Beta</div>
                     <div style='color:#7c3aed;font-size:22px;font-weight:800;'>{port_beta:.4f}</div></div>
                <div><div style='color:{TXT_MUTED};font-size:11px;'>CAPM Return</div>
                     <div style='color:{RED};font-size:22px;font-weight:800;'>{capm_ret:.2%}</div></div>
                <div><div style='color:{TXT_MUTED};font-size:11px;'>Jensen's Alpha</div>
                     <div style='color:{GREEN if alpha>=0 else RED};font-size:22px;font-weight:800;'>{alpha:+.2%}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
#  TAB 2 — CAL & EFFICIENT FRONTIER
# ════════════════════════════════════════════
with tab2:
    st.markdown(f"<div style='font-size:15px;font-weight:700;color:{TXT_HEAD};margin-bottom:4px;'>📊 Capital Allocation Line & Efficient Frontier</div>", unsafe_allow_html=True)
    st.caption("5,000 simulated portfolios · Mathematical efficient frontier (upper portion only from GMVP) · CAL through tangency point")

    with st.spinner("Generating frontier..."):
        np.random.seed(42)
        n = len(available)
        sim_r,sim_v,sim_s = [],[],[]
        for _ in range(5000):
            w = np.random.random(n); w/=w.sum()
            r,v,s = port_perf(w, mean_ret, cov_mat, rf_rate)
            sim_r.append(r); sim_v.append(v); sim_s.append(s)
        sim_r=np.array(sim_r); sim_v=np.array(sim_v); sim_s=np.array(sim_s)

        # Efficient frontier — UPPER portion only (from GMVP upward)
        ef_start = gmvp_ret if gmvp_ret is not None else float(mean_ret.min())
        ef_end   = float(mean_ret.max())
        ef_v,ef_r = [],[]
        for tr in np.linspace(ef_start, ef_end, 60):
            v = min_var_vol(mean_ret, cov_mat, tr)
            if v is not None:
                ef_v.append(v); ef_r.append(tr)

        # CAL — capped at 1.6× optimal vol
        cal_max = opt_vol * 1.6
        cal_x   = np.linspace(0, cal_max, 120)
        cal_y   = rf_rate + opt_sh * cal_x

    fig_cal = go.Figure()

    # Monte Carlo scatter
    fig_cal.add_trace(go.Scatter(
        x=sim_v*100, y=sim_r*100, mode="markers",
        marker=dict(
            color=sim_s, colorscale="Blues",
            reversescale=False,
            size=3.5, opacity=0.55,
            colorbar=dict(
                title=dict(text="Sharpe Ratio",
                           font=dict(color=TXT_BODY, size=11)),
                tickfont=dict(color=TXT_BODY, size=10),
                x=1.01, thickness=12, len=0.7,
                bgcolor=BG_CARD,
                bordercolor=BORDER
            ), showscale=True
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Random Portfolio</extra>"
    ))

    # Efficient frontier (upper only)
    if len(ef_v) > 3:
        fig_cal.add_trace(go.Scatter(
            x=[v*100 for v in ef_v],
            y=[r*100 for r in ef_r],
            mode="lines",
            line=dict(color="#d97706", width=3.5),
            name="Efficient Frontier (upper only)",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Efficient Frontier</extra>"
        ))

    # GMVP marker
    if gmvp_ret and gmvp_vol:
        fig_cal.add_trace(go.Scatter(
            x=[gmvp_vol*100], y=[gmvp_ret*100],
            mode="markers+text",
            marker=dict(color="#d97706", size=11, symbol="diamond",
                        line=dict(color="white", width=2)),
            text=["GMVP"], textposition="bottom right",
            textfont=dict(color="#d97706", size=11, family="Inter"),
            name=f"GMVP  ({gmvp_ret:.1%} return)",
            hovertemplate=(f"<b>Global Min-Variance Portfolio</b><br>"
                           f"Return: {gmvp_ret:.2%}<br>Vol: {gmvp_vol:.2%}<extra></extra>")
        ))

    # CAL
    fig_cal.add_trace(go.Scatter(
        x=cal_x*100, y=cal_y*100, mode="lines",
        line=dict(color=ACCENT, width=2.5, dash="dash"),
        name=f"CAL  (slope = Sharpe = {opt_sh:.2f})",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>CAL</extra>"
    ))

    # Risk-free
    fig_cal.add_trace(go.Scatter(
        x=[0], y=[rf_rate*100], mode="markers+text",
        marker=dict(color=ACCENT, size=11, symbol="circle",
                    line=dict(color="white", width=2)),
        text=["Rf"], textposition="top right",
        textfont=dict(color=ACCENT, size=12),
        name=f"Risk-Free  {rf_rate:.1%}",
        hovertemplate=f"Risk-Free Rate: {rf_rate:.2%}<extra></extra>"
    ))

    # Optimal portfolio
    fig_cal.add_trace(go.Scatter(
        x=[opt_vol*100], y=[opt_ret*100], mode="markers+text",
        marker=dict(color=RED, size=18, symbol="star",
                    line=dict(color="white", width=1.5)),
        text=["Optimal"], textposition="top right",
        textfont=dict(color=RED, size=12, family="Inter"),
        name=f"Optimal Portfolio  (Sharpe={opt_sh:.2f})",
        hovertemplate=(f"<b>Optimal (Tangency) Portfolio</b><br>"
                       f"Return: {opt_ret:.2%}<br>Vol: {opt_vol:.2%}<br>"
                       f"Sharpe: {opt_sh:.4f}<extra></extra>")
    ))

    x_max = max(float(np.max(sim_v)), opt_vol, cal_max)*100*1.05
    y_min = max(rf_rate*100*0.85, 0)
    y_max = float(np.max(sim_r))*100*1.08

    fig_cal.update_layout(
        **base_layout(h=560, margin=dict(t=20,b=20,l=10,r=60)),
        xaxis=dict(**ax_style("Annual Volatility (%)","%"), range=[0,x_max]),
        yaxis=dict(**ax_style("Expected Annual Return (%)","%"), range=[y_min,y_max]),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    with st.expander("📖 How to read this chart"):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""
            - 🟡 **Yellow curve** = Efficient Frontier — **upper portion only** (from GMVP upward).
            - 🔷 **Yellow diamond** = GMVP — starting point of the efficient frontier.
            - **Blue dots** = 5,000 random portfolios. Darker blue = higher Sharpe ratio.
            """)
        with c2:
            st.markdown("""
            - 🔵 **Blue dashed line** = CAL — combinations of Rf + optimal portfolio.
              Slope = Sharpe ratio (same for all points on the line).
            - ⭐ **Red star** = Optimal (Tangency) Portfolio — maximum Sharpe ratio.
            - 🔵 **Blue dot** = Risk-Free Rate — zero-risk starting point.
            """)

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Optimal Return",     f"{opt_ret:.2%}")
    s2.metric("Optimal Volatility", f"{opt_vol:.2%}")
    s3.metric("Max Sharpe Ratio",   f"{opt_sh:.4f}")
    if gmvp_ret and gmvp_vol:
        s4.metric("GMVP", f"Ret {gmvp_ret:.2%}  |  Vol {gmvp_vol:.2%}")

# ════════════════════════════════════════════
#  TAB 3 — SECURITY MARKET LINE
# ════════════════════════════════════════════
with tab3:
    st.markdown(f"<div style='font-size:15px;font-weight:700;color:{TXT_HEAD};margin-bottom:4px;'>🎯 Security Market Line — Individual Stock Positioning</div>", unsafe_allow_html=True)

    beta_range  = np.linspace(0, max(betas.values())*1.25, 120)
    sml_returns = rf_rate + beta_range*(mkt_annual - rf_rate)

    fig_sml = go.Figure()
    fig_sml.add_trace(go.Scatter(
        x=beta_range, y=sml_returns*100, mode="lines",
        line=dict(color=ACCENT, width=2.5),
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
            marker=dict(color=GREEN if under else RED,
                        size=13, symbol="circle",
                        line=dict(color="white", width=1.5)),
            text=[lbl], textposition="top center",
            textfont=dict(size=10, color=TXT_HEAD),
            name=lbl, showlegend=False,
            hovertemplate=(
                f"<b>{ticker}</b><br>Beta: {b:.4f}<br>"
                f"Actual: {actual:.2%}<br>CAPM: {capm_r:.2%}<br>"
                f"Alpha: {actual-capm_r:+.2%}<br>"
                f"{'✅ Undervalued' if under else '❌ Overvalued'}<extra></extra>"
            )
        ))

    fig_sml.add_trace(go.Scatter(
        x=[port_beta], y=[capm_ret*100], mode="markers+text",
        marker=dict(color=RED, size=18, symbol="star",
                    line=dict(color="white", width=1)),
        text=["Optimal"], textposition="top right",
        textfont=dict(color=RED, size=12),
        name="Optimal Portfolio",
        hovertemplate=f"Beta: {port_beta:.4f}<br>CAPM: {capm_ret:.2%}<extra>Optimal</extra>"
    ))
    fig_sml.add_trace(go.Scatter(
        x=[1.0], y=[mkt_annual*100], mode="markers+text",
        marker=dict(color="#7c3aed", size=14, symbol="square",
                    line=dict(color="white", width=1)),
        text=["Market"], textposition="top right",
        textfont=dict(color="#7c3aed", size=12),
        name="Market Portfolio",
        hovertemplate=f"Market Return: {mkt_annual:.2%}<extra>Market</extra>"
    ))
    fig_sml.add_trace(go.Scatter(
        x=[0], y=[rf_rate*100], mode="markers+text",
        marker=dict(color=ACCENT, size=11, symbol="circle",
                    line=dict(color="white", width=1)),
        text=["Rf"], textposition="top right",
        textfont=dict(color=ACCENT, size=12),
        name=f"Risk-Free  {rf_rate:.1%}",
        hovertemplate=f"Rf: {rf_rate:.2%}<extra></extra>"
    ))
    fig_sml.update_layout(
        **base_layout(h=540),
        xaxis=ax_style("Beta (Systematic Risk)"),
        yaxis=ax_style("Expected Return (%)","%"),
    )
    st.plotly_chart(fig_sml, use_container_width=True)

    with st.expander("📖 How to read this chart"):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""
            - **Blue line** = SML — CAPM required return for each beta level
            - 🟢 **Green dots** = Undervalued (actual > CAPM → positive alpha → above SML)
            - 🔴 **Red dots** = Overvalued (actual < CAPM → negative alpha → below SML)
            """)
        with c2:
            st.markdown("""
            - 🟣 **Purple square** = Market portfolio (beta = 1 by definition)
            - ⭐ **Red star** = Optimal portfolio CAPM position
            - Hover over any dot for full details
            """)

# ════════════════════════════════════════════
#  TAB 4 — CORRELATION MATRIX
# ════════════════════════════════════════════
with tab4:
    st.markdown(f"<div style='font-size:15px;font-weight:700;color:{TXT_HEAD};margin-bottom:4px;'>🔥 Correlation Matrix — Diversification Analysis</div>", unsafe_allow_html=True)

    slbls = [short(t) for t in available]
    cvals = corr_mat.values

    fig_heat = go.Figure(go.Heatmap(
        z=cvals, x=slbls, y=slbls,
        colorscale=[
            [0.00, "#1e40af"],
            [0.40, "#bfdbfe"],
            [0.50, "#f8fafc"],
            [0.65, "#fecaca"],
            [1.00, "#dc2626"]
        ],
        zmin=-1, zmax=1,
        text=np.round(cvals, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color=TXT_HEAD),
        hoverongaps=False,
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color=TXT_BODY, size=12)),
            tickfont=dict(color=TXT_BODY, size=11),
            tickvals=[-1,-0.5,0,0.5,1],
            ticktext=["-1.0","-0.5","0.0","+0.5","+1.0"],
            thickness=14, len=0.85,
            bgcolor=BG_CARD, bordercolor=BORDER
        )
    ))
    fig_heat.update_layout(
        **base_layout(h=530, margin=dict(t=20,b=70,l=70,r=20)),
        xaxis=dict(tickangle=-40, tickfont=dict(size=12,color=TXT_BODY),
                   side="bottom", gridcolor="transparent"),
        yaxis=dict(tickfont=dict(size=12,color=TXT_BODY),
                   autorange="reversed", gridcolor="transparent"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    flat = cvals[np.triu_indices_from(cvals, k=1)]
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Average Correlation", f"{flat.mean():.4f}", delta="Lower = better diversification")
    s2.metric("Max Correlation",     f"{flat.max():.4f}", delta="Most similar pair")
    s3.metric("Min Correlation",     f"{flat.min():.4f}", delta="Most different pair")
    s4.metric("Negative Pairs",      f"{(flat<0).sum()} / {len(flat)}", delta="Pairs with hedging benefit")

    with st.expander("📖 How to read this chart"):
        st.markdown("""
        - 🔴 **Red cells** → high positive correlation → stocks move together → **less diversification**
        - 🔵 **Blue cells** → low/negative correlation → stocks move independently → **more diversification**
        - **Diagonal** is always 1.0 — a stock is perfectly correlated with itself
        - Lower average correlation across all pairs = better portfolio diversification
        """)

# ════════════════════════════════════════════
#  TAB 5 — STOCK VALUATION
# ════════════════════════════════════════════
with tab5:
    st.markdown(f"<div style='font-size:15px;font-weight:700;color:{TXT_HEAD};margin-bottom:4px;'>📉 Individual Stock Valuation — CAPM vs Actual Return</div>", unsafe_allow_html=True)

    rows = []
    for t in available:
        b      = betas[t]
        actual = float(mean_ret[t])
        capm_r = rf_rate + b*(mkt_annual - rf_rate)
        ai     = actual - capm_r
        rows.append({"Ticker":t,"Label":short(t),"Beta":b,
                     "Actual":actual,"CAPM":capm_r,"Alpha":ai,
                     "Under":actual>capm_r,
                     "Weight":opt_w[available.index(t)]})

    vdf   = pd.DataFrame(rows).sort_values("Alpha", ascending=False)
    un_df = vdf[vdf["Under"]]
    ov_df = vdf[~vdf["Under"]]

    hc1,hc2 = st.columns(2, gap="large")

    with hc1:
        st.markdown(f"""
        <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
                    padding:12px 18px;margin-bottom:14px;'>
            <span style='color:{GREEN};font-weight:800;font-size:15px;'>
                ✅ Undervalued — {len(un_df)} stock{"s" if len(un_df)!=1 else ""}</span>
            <div style='color:#15803d;font-size:12px;margin-top:3px;'>
                Actual return &gt; CAPM required → Positive Alpha → Above SML</div>
        </div>""", unsafe_allow_html=True)

        for _, r in un_df.iterrows():
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #bbf7d0;
                        border-radius:10px;padding:13px 16px;margin-bottom:8px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.05);'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:8px;'>
                    <span style='color:{GREEN};font-weight:800;font-size:15px;'>
                        {r["Label"]}</span>
                    <span style='background:#dcfce7;color:{GREEN};padding:3px 10px;
                                 border-radius:999px;font-size:12px;font-weight:700;'>
                        α = {r["Alpha"]:+.2%}</span>
                </div>
                <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:6px;'>
                    <div><div style='color:{TXT_MUTED};font-size:10px;'>Beta</div>
                         <div style='color:{TXT_HEAD};font-weight:700;font-size:13px;'>{r["Beta"]:.4f}</div></div>
                    <div><div style='color:{TXT_MUTED};font-size:10px;'>Actual</div>
                         <div style='color:{GREEN};font-weight:700;font-size:13px;'>{r["Actual"]:.2%}</div></div>
                    <div><div style='color:{TXT_MUTED};font-size:10px;'>CAPM Req.</div>
                         <div style='color:{YELLOW};font-weight:700;font-size:13px;'>{r["CAPM"]:.2%}</div></div>
                    <div><div style='color:{TXT_MUTED};font-size:10px;'>Port. Wt</div>
                         <div style='color:{ACCENT};font-weight:700;font-size:13px;'>{r["Weight"]:.2%}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with hc2:
        st.markdown(f"""
        <div style='background:#fef2f2;border:1px solid #fecaca;border-radius:12px;
                    padding:12px 18px;margin-bottom:14px;'>
            <span style='color:{RED};font-weight:800;font-size:15px;'>
                ❌ Overvalued — {len(ov_df)} stock{"s" if len(ov_df)!=1 else ""}</span>
            <div style='color:#b91c1c;font-size:12px;margin-top:3px;'>
                Actual return &lt; CAPM required → Negative Alpha → Below SML</div>
        </div>""", unsafe_allow_html=True)

        for _, r in ov_df.iterrows():
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #fecaca;
                        border-radius:10px;padding:13px 16px;margin-bottom:8px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.05);'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:8px;'>
                    <span style='color:{RED};font-weight:800;font-size:15px;'>
                        {r["Label"]}</span>
                    <span style='background:#fee2e2;color:{RED};padding:3px 10px;
                                 border-radius:999px;font-size:12px;font-weight:700;'>
                        α = {r["Alpha"]:+.2%}</span>
                </div>
                <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:6px;'>
                    <div><div style='color:{TXT_MUTED};font-size:10px;'>Beta</div>
                         <div style='color:{TXT_HEAD};font-weight:700;font-size:13px;'>{r["Beta"]:.4f}</div></div>
                    <div><div style='color:{TXT_MUTED};font-size:10px;'>Actual</div>
                         <div style='color:{RED};font-weight:700;font-size:13px;'>{r["Actual"]:.2%}</div></div>
                    <div><div style='color:{TXT_MUTED};font-size:10px;'>CAPM Req.</div>
                         <div style='color:{YELLOW};font-weight:700;font-size:13px;'>{r["CAPM"]:.2%}</div></div>
                    <div><div style='color:{TXT_MUTED};font-size:10px;'>Port. Wt</div>
                         <div style='color:{ACCENT};font-weight:700;font-size:13px;'>{r["Weight"]:.2%}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Alpha bar chart
    st.markdown(f"<div style='font-size:15px;font-weight:700;color:{TXT_HEAD};margin:16px 0 8px 0;'>📊 Jensen's Alpha — All Stocks</div>", unsafe_allow_html=True)
    vs = vdf.sort_values("Alpha")
    fig_a = go.Figure(go.Bar(
        x=vs["Label"], y=vs["Alpha"]*100,
        marker=dict(
            color=[GREEN if a>0 else RED for a in vs["Alpha"]],
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.6)", width=1)
        ),
        text=[f"{a:+.2f}%" for a in vs["Alpha"]*100],
        textposition="outside",
        textfont=dict(color=TXT_HEAD, size=11),
        hovertemplate="<b>%{x}</b><br>Alpha: %{y:.2f}%<extra></extra>"
    ))
    fig_a.add_hline(y=0, line_color=TXT_MUTED, line_width=1.5, line_dash="dot")
    fig_a.update_layout(
        **base_layout(h=320, margin=dict(t=20,b=20,l=10,r=10)),
        xaxis=dict(tickfont=dict(size=12,color=TXT_BODY), gridcolor=BORDER),
        yaxis=ax_style("Jensen's Alpha (%)","%" ),
        showlegend=False
    )
    st.plotly_chart(fig_a, use_container_width=True)

# ════════════════════════════════════════════
#  TAB 6 — CAL SIMULATION
# ════════════════════════════════════════════
with tab6:
    st.markdown(f"<div style='font-size:15px;font-weight:700;color:{TXT_HEAD};margin-bottom:4px;'>📋 Capital Allocation Line — 101 Portfolio Combinations</div>", unsafe_allow_html=True)
    st.caption("Shifting 1% at a time from 100% Risk-Free → 100% Optimal Risky Portfolio")

    # ── Key insight box ───────────────────────
    st.markdown(f"""
    <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                padding:14px 18px;margin-bottom:16px;'>
        <div style='font-size:13px;font-weight:700;color:{ACCENT};margin-bottom:6px;'>
            💡 Why does Sharpe Ratio stay the same (1.3441) for all 101 combinations?
        </div>
        <div style='font-size:13px;color:{TXT_BODY};line-height:1.6;'>
            This is a fundamental mathematical property of the CAL. For any portfolio on the CAL
            with weight <b>w</b> in the risky portfolio:
            <br><br>
            <b>Sharpe = (E(Rp) − Rf) / σp
            = [Rf + w·(E(RT)−Rf) − Rf] / (w·σT)
            = w·(E(RT)−Rf) / (w·σT)
            = (E(RT)−Rf) / σT</b>
            <br><br>
            The weight <b>w cancels out</b> — so the Sharpe ratio equals the tangency portfolio's
            Sharpe ratio for every single point on the CAL.
            This is why investors on the CAL get the same reward-per-unit-risk regardless of
            how conservative or aggressive they are — they just scale up or down the same portfolio.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Line charts ───────────────────────────
    w_arr = np.linspace(0, 1, 101)
    r_arr = rf_rate + w_arr*(opt_ret - rf_rate)
    v_arr = w_arr * opt_vol
    s_arr = np.where(v_arr > 0, (r_arr - rf_rate)/v_arr, 0)

    fig_lines = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Expected Return vs Risky Weight",
                        "Volatility vs Risky Weight",
                        "Sharpe Ratio vs Risky Weight"]
    )
    common = dict(mode="lines")
    fig_lines.add_trace(go.Scatter(
        x=w_arr*100, y=r_arr*100,
        line=dict(color=GREEN, width=2.5), name="Return",
        hovertemplate="Risky Wt: %{x:.0f}%<br>Return: %{y:.2f}%<extra></extra>",
        **common), 1, 1)
    fig_lines.add_trace(go.Scatter(
        x=w_arr*100, y=v_arr*100,
        line=dict(color=YELLOW, width=2.5), name="Volatility",
        hovertemplate="Risky Wt: %{x:.0f}%<br>Vol: %{y:.2f}%<extra></extra>",
        **common), 1, 2)
    fig_lines.add_trace(go.Scatter(
        x=w_arr*100, y=s_arr,
        line=dict(color=ACCENT, width=2.5), name="Sharpe",
        hovertemplate="Risky Wt: %{x:.0f}%<br>Sharpe: %{y:.4f}<extra></extra>",
        **common), 1, 3)

    fig_lines.update_layout(
        **base_layout(h=280, margin=dict(t=40,b=20,l=10,r=10)),
        showlegend=False
    )
    for axis in ["xaxis","xaxis2","xaxis3"]:
        fig_lines.update_layout(**{axis: dict(
            title="Risky Weight (%)", gridcolor=BORDER,
            ticksuffix="%", zeroline=False,
            tickfont=dict(color=TXT_BODY)
        )})
    for axis in ["yaxis","yaxis2","yaxis3"]:
        fig_lines.update_layout(**{axis: dict(
            gridcolor=BORDER, zeroline=False,
            tickfont=dict(color=TXT_BODY)
        )})
    for ann in fig_lines.layout.annotations:
        ann.font.color = TXT_HEAD
        ann.font.size  = 13

    st.plotly_chart(fig_lines, use_container_width=True)

    st.markdown(f"""
    <div style='background:#fefce8;border:1px solid #fef08a;border-radius:10px;
                padding:10px 16px;margin-bottom:12px;font-size:13px;color:{TXT_BODY};'>
        📌 <b>Note:</b> The Sharpe Ratio chart shows a flat horizontal line at <b>{opt_sh:.4f}</b>
        — this confirms that every portfolio on the CAL has the same Sharpe ratio as the
        tangency portfolio. The weight <b>w</b> cancels out mathematically.
    </div>""", unsafe_allow_html=True)

    # ── Table ─────────────────────────────────
    cal_rows = []
    for i in range(101):
        wrf=i/100; wr=1-wrf
        cr = wrf*rf_rate + wr*opt_ret
        cv = wr*opt_vol
        cb = wr*port_beta
        cc = rf_rate + cb*(mkt_annual - rf_rate)
        cs = (cr-rf_rate)/cv if cv > 0 else 0
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
<div style='text-align:center;color:{TXT_MUTED};font-size:12px;padding:10px 0 4px 0;'>
    📊 Portfolio Optimizer &nbsp;·&nbsp; Data: Yahoo Finance &nbsp;·&nbsp;
    Period: {start_date} → {end_date} &nbsp;·&nbsp;
    Rf: {rf_rate:.1%} &nbsp;·&nbsp;
    {len(available)} assets via SLSQP (scipy)
    <br><br>
    <span style='color:#94a3b8;font-size:11px;'>
        ⚠️ For educational purposes only. Past performance does not guarantee future results.
    </span>
</div>""", unsafe_allow_html=True)
