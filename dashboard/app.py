"""
RegimeTrader — Streamlit dashboard.
Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Make sure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

from brokers.alpaca_broker import AlpacaBroker
from brokers.position_tracker import PositionTracker
from config import settings

st.set_page_config(
    page_title="RegimeTrader Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("RegimeTrader")
st.sidebar.caption("Automated Regime-Based Trading Bot")
refresh_secs = st.sidebar.slider("Auto-refresh (s)", 30, 300, settings.DASHBOARD_REFRESH_SECS)
selected_ticker = st.sidebar.selectbox("Primary ticker", settings.TICKERS)

# ── Connect to Alpaca ─────────────────────────────────────────────────────────
@st.cache_resource
def get_broker():
    b = AlpacaBroker()
    b.connect()
    return b

broker = get_broker()


def load_account():
    return broker.get_account() or {}


def load_positions():
    tracker = PositionTracker(broker)
    return tracker.get_positions()


# ── Header row ────────────────────────────────────────────────────────────────
account = load_account()
equity   = account.get("portfolio_value", 0.0)
bp       = account.get("buying_power", 0.0)

# Detect regime (lightweight — load recent price history via yfinance)
@st.cache_data(ttl=300)
def get_regime_info(ticker: str):
    try:
        from core.hmm_engine import RegimeDetectionEngine
        from core.market_data import MarketDataProvider

        dp  = MarketDataProvider()
        df  = dp.get_history(ticker, days=settings.HMM_TRAINING_DAYS)
        eng = RegimeDetectionEngine()
        eng.train(df)
        recent = dp.get_history(ticker, days=settings.MARKET_DATA_LOOKBACK_DAYS)
        regime, conf = eng.predict(recent)
        summary = eng.get_regime_summary()
        return regime, conf, summary, df
    except Exception as exc:
        return "unknown", 0.0, {}, pd.DataFrame()


regime, confidence, summary, hist_df = get_regime_info(selected_ticker)

# ── Row 1: KPI cards ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

REGIME_COLORS = {
    "crash":    "#d62728",
    "bear":     "#ff7f0e",
    "neutral":  "#7f7f7f",
    "bull":     "#2ca02c",
    "euphoria": "#9467bd",
}
regime_color = REGIME_COLORS.get(regime, "#1f77b4")

c1.metric("Detected Regime", regime.upper())
c2.metric("Confidence", f"{confidence:.1%}")
c3.metric("Portfolio Value", f"${equity:,.0f}")
c4.metric("Buying Power",    f"${bp:,.0f}")
c5.metric("# Regimes",       summary.get("n_regimes", "N/A"))

st.divider()

# ── Row 2: Price + regime overlay ─────────────────────────────────────────────
st.subheader(f"{selected_ticker} — Price & Regime Overlay")

if not hist_df.empty:
    @st.cache_data(ttl=300)
    def build_regime_series(ticker):
        from core.hmm_engine import RegimeDetectionEngine
        from core.market_data import MarketDataProvider
        dp  = MarketDataProvider()
        df  = dp.get_history(ticker, days=settings.HMM_TRAINING_DAYS)
        eng = RegimeDetectionEngine()
        eng.train(df)
        regimes = eng.classify_states(df)
        return df, regimes

    price_df, regime_series = build_regime_series(selected_ticker)
    merged = price_df[["close"]].join(regime_series, how="inner")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged.index, y=merged["close"],
        mode="lines", name="Close", line=dict(color="#1f77b4", width=1.5),
    ))

    for reg_label, color in REGIME_COLORS.items():
        mask = merged["regime"] == reg_label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=merged.index[mask], y=merged["close"][mask],
                mode="markers",
                marker=dict(color=color, size=4),
                name=reg_label.capitalize(),
            ))

    fig.update_layout(height=350, margin=dict(t=20, b=20), legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3: Volume, confidence, drawdown ───────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Volume")
    if not hist_df.empty:
        fig_v = px.bar(hist_df.tail(60), y="volume", title="")
        fig_v.update_layout(height=220, margin=dict(t=10, b=10))
        st.plotly_chart(fig_v, use_container_width=True)

with col_b:
    st.subheader("Regime Confidence Distribution")
    st.caption("Based on current trained model posterior")

    # Show regime label distribution
    if summary.get("regime_labels"):
        alloc_data = {
            label: settings.REGIME_ALLOCATIONS.get(label, {}).get("equity_pct", 0.5)
            for label in summary["regime_labels"]
        }
        fig_alloc = px.bar(
            x=list(alloc_data.keys()),
            y=list(alloc_data.values()),
            labels={"x": "Regime", "y": "Target Equity %"},
            color=list(alloc_data.keys()),
            color_discrete_map=REGIME_COLORS,
        )
        fig_alloc.update_layout(height=220, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig_alloc, use_container_width=True)

st.divider()

# ── Row 4: Signal feed + risk controls ───────────────────────────────────────
col_sig, col_risk = st.columns([3, 2])

with col_sig:
    st.subheader("Signal Feed — Open Positions")
    positions = load_positions()
    if positions:
        rows = []
        for t, p in positions.items():
            rows.append({
                "Ticker":       t,
                "Qty":          p.qty,
                "Entry":        f"${p.avg_entry:.2f}",
                "Current":      f"${p.current_price:.2f}",
                "Notional":     f"${p.notional:,.0f}",
                "Unrealized P&L": f"${p.unrealized_pnl:,.2f}",
                "P&L %":        f"{p.unrealized_pnl_pct:.2%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No open positions. Market may be closed or bot is in cash.")

with col_risk:
    st.subheader("Risk Controls")
    lock_exists = os.path.exists(settings.LOCK_FILE_PATH)

    st.markdown("**Circuit Breakers**")
    st.markdown(f"- Daily loss halve: `{settings.CB_DAILY_LOSS_HALVE:.0%}`")
    st.markdown(f"- Daily loss close-all: `{settings.CB_DAILY_LOSS_CLOSE:.0%}`")
    st.markdown(f"- Weekly loss resize: `{settings.CB_WEEKLY_LOSS_RESIZE:.0%}`")
    st.markdown(f"- Peak drawdown stop: `{settings.CB_PEAK_DRAWDOWN_STOP:.0%}`")

    st.markdown("**Status**")
    if lock_exists:
        st.error("LOCK FILE ACTIVE — bot is halted.")
    else:
        st.success("All circuit breakers: OK")

    st.markdown(f"**Max leverage:** `{settings.MAX_LEVERAGE}x`")
    st.markdown(f"**Max pos. risk:** `{settings.MAX_POSITION_RISK_PCT:.0%}` per trade")

st.divider()

# ── Row 5: Regime allocation table ───────────────────────────────────────────
st.subheader("Regime Allocation Strategy")
alloc_rows = []
for label, cfg in settings.REGIME_ALLOCATIONS.items():
    alloc_rows.append({
        "Regime":    label.capitalize(),
        "Equity %":  f"{cfg['equity_pct']:.0%}",
        "Leverage":  f"{cfg['leverage']}x",
        "Strategy":  cfg["strategy"],
        "Max Exposure": f"{cfg['equity_pct'] * cfg['leverage']:.0%}",
    })
st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True)

# ── Row 6: Trade History ──────────────────────────────────────────────────────
st.subheader("Trade History")


@st.cache_data(ttl=60)
def load_trade_history():
    if os.path.exists(settings.TRADE_LOG_PATH):
        return pd.read_csv(settings.TRADE_LOG_PATH)
    return pd.DataFrame()


trade_df = load_trade_history()
if not trade_df.empty:
    # Summary metrics
    th1, th2, th3, th4 = st.columns(4)
    th1.metric("Total Trades", len(trade_df))
    th2.metric("Realized P&L", f"${trade_df['realized_pnl'].sum():,.2f}")
    win_rate = (trade_df["realized_pnl"] > 0).mean() if len(trade_df) > 0 else 0
    th3.metric("Win Rate", f"{win_rate:.1%}")
    th4.metric("Last Regime", trade_df["regime"].iloc[-1].upper() if len(trade_df) else "—")

    # Regime breakdown bar chart
    regime_pnl = trade_df.groupby("regime")["realized_pnl"].sum().reset_index()
    fig_pnl = px.bar(
        regime_pnl, x="regime", y="realized_pnl",
        color="regime", color_discrete_map=REGIME_COLORS,
        labels={"realized_pnl": "P&L ($)", "regime": "Regime"},
        title="Realized P&L by Regime",
    )
    fig_pnl.update_layout(height=250, margin=dict(t=30, b=10), showlegend=False)
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Raw table (last 50 trades, newest first)
    display_cols = ["timestamp", "ticker", "side", "qty", "entry_price",
                    "realized_pnl", "regime", "confidence", "allocation_pct"]
    show_cols = [c for c in display_cols if c in trade_df.columns]
    st.dataframe(trade_df[show_cols].tail(50).iloc[::-1], use_container_width=True)
else:
    st.info("No trades recorded yet — log will populate once the bot places its first order.")

st.divider()

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
time.sleep(refresh_secs)
st.rerun()
