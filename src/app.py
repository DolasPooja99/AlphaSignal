import sys
import os
import re
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from urllib.parse import urlparse

from data_fetcher import fetch_daily_prices, get_sector_etf
from anomaly_detector import detect_anomalies
from forecaster import forecast_prices
from news_rag import fetch_and_store_news, retrieve_relevant_news
from claude_analyst import analyze, build_context_message, stream_chat_response
from technical_indicators import add_rsi, add_macd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaSignal",
    page_icon="📈",
    layout="wide",
)

# ── Session state initialisation ──────────────────────────────────────────────
# Everything persists within the browser session — cleared on page refresh.
if "results"   not in st.session_state: st.session_state.results   = {}
if "watchlist" not in st.session_state: st.session_state.watchlist = []
if "chat"      not in st.session_state: st.session_state.chat      = {}


# ── Sidebar — Watchlist ────────────────────────────────────────────────────────
# The watchlist is a list of saved tickers. Clicking one populates the main input.
# It doesn't auto-fetch data — that keeps API usage under control.
with st.sidebar:
    st.header("📋 Watchlist")
    st.caption("Save tickers you follow. Click to load.")

    # Add a new ticker
    with st.form("add_watchlist", clear_on_submit=True):
        new_ticker = st.text_input("Add ticker", placeholder="e.g. TSLA").upper().strip()
        if st.form_submit_button("Add") and new_ticker:
            if new_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker)

    # Display saved tickers
    if st.session_state.watchlist:
        for wt in st.session_state.watchlist:
            col_t, col_x = st.columns([4, 1])
            with col_t:
                # Clicking the ticker button sets it as the active ticker
                if st.button(wt, key=f"wl_{wt}", use_container_width=True):
                    st.session_state.active_ticker = wt
            with col_x:
                if st.button("✕", key=f"rm_{wt}"):
                    st.session_state.watchlist.remove(wt)
                    st.rerun()
    else:
        st.caption("No tickers saved yet.")

    st.divider()
    st.caption("AlphaSignal — AI stock analysis\nData: Alpha Vantage · NewsAPI")


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("📈 AlphaSignal")

# Ticker input — pre-fill from watchlist click if available
default_ticker = st.session_state.get("active_ticker", "AAPL")
col_input, col_btn, col_spacer = st.columns([2, 1, 5])

with col_input:
    ticker = st.text_input("Ticker symbol", value=default_ticker).upper().strip()
with col_btn:
    st.write("")  # vertical align
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)


# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(ticker: str) -> dict:
    with st.status(f"Analyzing {ticker}...", expanded=True) as status:

        status.update(label="📡 Fetching 90 days of price data...")
        df = fetch_daily_prices(ticker, days=90)

        # Fetch sector ETF data for comparison (if ticker is mapped)
        sector_etf = get_sector_etf(ticker)
        sector_df = None
        if sector_etf:
            status.update(label=f"📡 Fetching sector ETF ({sector_etf}) for comparison...")
            try:
                sector_df = fetch_daily_prices(sector_etf, days=90)
            except Exception:
                sector_df = None  # sector fetch failed — not critical, continue

        status.update(label="📐 Computing RSI and MACD...")
        df = add_rsi(df)
        df = add_macd(df)

        status.update(label="🔍 Detecting anomalies across 90-day history...")
        anomalies = detect_anomalies(df)
        anomalies["ticker"] = ticker

        status.update(label="🔮 Running 7-day forecast (Prophet)...")
        forecast = forecast_prices(df, days_ahead=7)

        status.update(label="📰 Fetching and indexing news...")
        fetch_and_store_news(ticker)
        query = f"{ticker} stock price earnings forecast outlook"
        news = retrieve_relevant_news(ticker, query, top_k=5)

        status.update(label="🤖 Asking Claude to synthesize everything...")
        report = analyze(ticker, df, anomalies, forecast, news)

        # Build and store the context string for the chat feature
        context = build_context_message(ticker, df, anomalies, forecast, news)

        status.update(label=f"✅ Done — {ticker} analysis complete", state="complete")

    return {
        "df": df,
        "sector_df": sector_df,
        "sector_etf": sector_etf,
        "anomalies": anomalies,
        "forecast": forecast,
        "news": news,
        "report": report,
        "context": context,
    }


if analyze_clicked and ticker:
    st.session_state.results.pop(ticker, None)
    st.session_state.chat.pop(ticker, None)    # clear old chat when re-analyzing
    st.session_state.results[ticker] = run_pipeline(ticker)


# ── Render results ─────────────────────────────────────────────────────────────
if ticker in st.session_state.results:
    r            = st.session_state.results[ticker]
    df           = r["df"]
    sector_df    = r["sector_df"]
    sector_etf   = r["sector_etf"]
    anomalies    = r["anomalies"]
    forecast     = r["forecast"]
    news         = r["news"]
    report       = r["report"]
    context      = r["context"]

    # ── Chart — 3 subplots: Price, RSI, MACD ──────────────────────────────────
    # Why subplots?
    # RSI and MACD live on a different scale than price (RSI is 0-100, MACD
    # is near zero). Stacking them on separate rows with a shared x-axis lets
    # you see exactly how the indicator aligns with the price move above it.
    st.subheader(f"{ticker} — Price Chart")

    subplot_titles = ["Price" + (f" vs {sector_etf} (normalized to 100)" if sector_etf else ""),
                      "RSI (14)", "MACD (12, 26, 9)"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,                    # zoom/pan all three in sync
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    # ── Row 1: Price history ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        name=ticker,
        line=dict(color="#4C9BE8", width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Sector ETF overlay — normalized so both series start at 100
    # This answers: "is the stock outperforming or underperforming its sector?"
    if sector_df is not None:
        # Find the common start date between the two series
        common_start = max(df["date"].iloc[0], sector_df["date"].iloc[0])
        stock_base   = df[df["date"] >= common_start]["close"].iloc[0]
        sector_base  = sector_df[sector_df["date"] >= common_start]["close"].iloc[0]

        stock_norm   = df[df["date"] >= common_start]["close"] / stock_base * 100
        sector_norm  = sector_df[sector_df["date"] >= common_start]["close"] / sector_base * 100
        stock_dates  = df[df["date"] >= common_start]["date"]
        sector_dates = sector_df[sector_df["date"] >= common_start]["date"]

        fig.add_trace(go.Scatter(
            x=sector_dates, y=sector_norm,
            name=sector_etf,
            line=dict(color="#aaaaaa", width=1.5, dash="dot"),
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{sector_etf}: %{{y:.1f}}<extra></extra>",
        ), row=1, col=1)

        # Replace the stock trace with normalized version when sector is shown
        fig.data[0].y = stock_norm
        fig.data[0].hovertemplate = f"<b>%{{x|%Y-%m-%d}}</b><br>{ticker}: %{{y:.1f}}<extra></extra>"
        fig.data[0].x = stock_dates

    # Forecast confidence band (shaded area)
    predictions  = forecast["predictions"]
    pred_dates   = [p["date"] for p in predictions]
    pred_upper   = [p["upper"] for p in predictions]
    pred_lower   = [p["lower"] for p in predictions]
    pred_mid     = [p["predicted"] for p in predictions]

    fig.add_trace(go.Scatter(
        x=pred_dates + pred_dates[::-1],
        y=pred_upper + pred_lower[::-1],
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Forecast band",
        hoverinfo="skip",
        showlegend=True,
    ), row=1, col=1)

    # Forecast line
    last_date  = str(df["date"].iloc[-1].date())
    last_close = df["close"].iloc[-1]
    if sector_df is not None:
        last_close_norm = float(fig.data[0].y[-1])
        first_pred_norm = pred_mid[0] / stock_base * 100
        pred_mid_norm   = [p / stock_base * 100 for p in pred_mid]
        fig.add_trace(go.Scatter(
            x=[last_date] + pred_dates, y=[last_close_norm] + pred_mid_norm,
            name="Forecast", line=dict(color="orange", width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Forecast: %{y:.1f}<extra></extra>",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=[last_date] + pred_dates, y=[last_close] + pred_mid,
            name="Forecast", line=dict(color="orange", width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Forecast: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # Anomaly markers — color-coded by type
    history    = anomalies.get("history", [])
    price_down = [h for h in history if h["type"] == "price_spike" and h.get("direction") == "down"]
    price_up   = [h for h in history if h["type"] == "price_spike" and h.get("direction") == "up"]
    vol_spikes = [h for h in history if h["type"] == "volume_spike"]

    def _close_to_y(close_val):
        """Convert raw close to normalized value if sector overlay is active."""
        if sector_df is not None:
            return close_val / stock_base * 100
        return close_val

    for group, color, label, symbol in [
        (price_down, "red",        "Price spike ↓", "triangle-down"),
        (price_up,   "green",      "Price spike ↑", "triangle-up"),
        (vol_spikes, "darkorange", "Volume spike",  "diamond"),
    ]:
        if not group:
            continue
        fig.add_trace(go.Scatter(
            x=[h["date"]  for h in group],
            y=[_close_to_y(h["close"]) for h in group],
            mode="markers", name=label,
            marker=dict(color=color, size=10, symbol=symbol,
                        line=dict(color="black", width=1)),
            hovertemplate=(
                "<b>%{x}</b><br>Close: $%{customdata[0]:.2f}<br>"
                + ("Change: %{customdata[1]:+.2f}% (z=%{customdata[2]})<extra></extra>"
                   if group[0]["type"] == "price_spike"
                   else "Volume: %{customdata[1]:.2f}x avg<extra></extra>")
            ),
            customdata=(
                [[h["close"], h["value"], h["zscore"]] for h in group]
                if group[0]["type"] == "price_spike"
                else [[h["close"], h["value"]] for h in group]
            ),
        ), row=1, col=1)

    # ── Row 2: RSI ─────────────────────────────────────────────────────────────
    rsi_valid = df.dropna(subset=["rsi"])
    fig.add_trace(go.Scatter(
        x=rsi_valid["date"], y=rsi_valid["rsi"],
        name="RSI", line=dict(color="#9B59B6", width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>RSI: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    # Overbought / oversold reference lines
    for level, color, label in [(70, "red", "Overbought"), (30, "green", "Oversold")]:
        fig.add_hline(
            y=level, line_dash="dot", line_color=color,
            annotation_text=label, annotation_position="right",
            row=2, col=1,
        )

    # ── Row 3: MACD ────────────────────────────────────────────────────────────
    macd_valid = df.dropna(subset=["macd"])

    # Histogram bars — green when positive (momentum building up), red when negative
    hist_colors = ["rgba(0,180,0,0.6)" if v >= 0 else "rgba(220,0,0,0.6)"
                   for v in macd_valid["macd_hist"]]
    fig.add_trace(go.Bar(
        x=macd_valid["date"], y=macd_valid["macd_hist"],
        name="MACD Histogram", marker_color=hist_colors,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Histogram: %{y:.3f}<extra></extra>",
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=macd_valid["date"], y=macd_valid["macd"],
        name="MACD", line=dict(color="#2196F3", width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>MACD: %{y:.3f}<extra></extra>",
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=macd_valid["date"], y=macd_valid["macd_signal"],
        name="Signal", line=dict(color="#FF9800", width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Signal: %{y:.3f}<extra></extra>",
    ), row=3, col=1)

    # ── Chart layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        height=650,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="relative",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(range=[0, 100], row=2, col=1)   # RSI always 0–100

    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current Price", f"${forecast['current_price']}")
    m2.metric(
        "7-Day Forecast", f"${forecast['predicted_price_7d']}",
        delta=f"{forecast['trend_pct']}%",
        delta_color="normal" if forecast["trend"] == "bullish" else
                    "inverse" if forecast["trend"] == "bearish" else "off",
    )
    m3.metric("Trend",      forecast["trend"].capitalize())
    m4.metric("Confidence", forecast["confidence_level"].capitalize())

    # Current RSI reading with context
    current_rsi = df["rsi"].dropna().iloc[-1]
    rsi_label   = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "⚪ Neutral"
    m5.metric("RSI (14)", f"{current_rsi:.1f}", delta=rsi_label, delta_color="off")

    st.divider()

    # ── Report + News ──────────────────────────────────────────────────────────
    col_report, col_news = st.columns([3, 1])

    with col_report:
        st.subheader("🤖 AI Research Report")
        safe_report = re.sub(r'\$(\d)', r'\\$\1', report)
        st.markdown(safe_report)

    with col_news:
        st.subheader("📰 News & Sentiment")

        # Aggregate sentiment across all articles — gives a quick overall read
        if news:
            labels    = [a["sentiment"]["label"] for a in news]
            bull_ct   = labels.count("bullish")
            bear_ct   = labels.count("bearish")
            neut_ct   = labels.count("neutral")
            st.caption(f"Overall: 🟢 {bull_ct} bullish · 🔴 {bear_ct} bearish · ⚪ {neut_ct} neutral")

        REDIRECT_DOMAINS = {"yahoo.com", "finance.yahoo.com", "msn.com"}
        PAYWALL_DOMAINS  = {"fool.com", "barrons.com", "wsj.com", "ft.com", "bloomberg.com"}

        for article in news:
            sentiment = article.get("sentiment", {})
            emoji     = sentiment.get("emoji", "⚪")
            label_tag = f"{emoji} {sentiment.get('label', '').capitalize()}"

            url    = article.get("url", "")
            domain = urlparse(url).netloc.replace("www.", "") if url else ""

            with st.expander(f"{emoji} {article['title'][:55]}..."):
                st.caption(f"{article['published_at']} · {label_tag} · score: {sentiment.get('compound', 0)}")

                if article["description"]:
                    st.write(article["description"])

                if url:
                    if domain in REDIRECT_DOMAINS:
                        st.caption(f"⚠️ {domain} uses a consent redirect — search the headline to find the article.")
                    else:
                        st.markdown(
                            f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
                            f'Read on {domain} ↗</a>',
                            unsafe_allow_html=True,
                        )
                        if domain in PAYWALL_DOMAINS:
                            st.caption("⚠️ May require a subscription.")

                st.caption(f"Relevance: {article['relevance_score']}")

    # ── Anomaly detail ─────────────────────────────────────────────────────────
    if anomalies["anomalies_found"] > 0:
        st.divider()
        st.subheader("🚨 Today's Anomalies")
        for a in anomalies["anomalies"]:
            sev = "🔴" if a["severity"] == "high" else "🟡"
            if a["type"] == "price_spike":
                st.write(f"{sev} **{a['type']}** ({a['severity']}): "
                         f"{a['direction']} move of {a['value']}%, z-score = {a['zscore']}")
            elif a["type"] == "volume_spike":
                st.write(f"{sev} **{a['type']}** ({a['severity']}): {a['value']}x normal volume")
            else:
                st.write(f"{sev} **{a['type']}** ({a['severity']}): "
                         f"{a['days']} consecutive days, total {a['total_change']}%")

    st.divider()

    # ── Chat interface ─────────────────────────────────────────────────────────
    # This is the most educational part to understand:
    #
    # Streamlit's st.chat_message() renders a message bubble (user or assistant).
    # st.chat_input() is a persistent text box at the bottom of the page.
    # st.write_stream() accepts a generator and streams text into the page in real time.
    #
    # The conversation history is stored in st.session_state.chat[ticker] as a
    # list of {"role": "user"/"assistant", "content": "..."} dicts — the same
    # format Claude's API expects. We append each new exchange to this list so
    # the full conversation is always visible and always passed to Claude.
    st.subheader("💬 Ask Claude a follow-up question")
    st.caption("Ask anything about this stock — Claude has access to all the data above.")

    # Initialise chat history for this ticker if it doesn't exist
    if ticker not in st.session_state.chat:
        st.session_state.chat[ticker] = []

    # Render existing conversation
    for msg in st.session_state.chat[ticker]:
        with st.chat_message(msg["role"]):
            safe_content = re.sub(r'\$(\d)', r'\\$\1', msg["content"])
            st.markdown(safe_content)

    # Chat input — st.chat_input() pins itself to the bottom of the page
    if user_question := st.chat_input(f"Ask about {ticker}..."):

        # Show the user's message immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Stream Claude's response into the page
        with st.chat_message("assistant"):
            response = st.write_stream(
                stream_chat_response(
                    context=context,
                    report=report,
                    history=st.session_state.chat[ticker],
                    user_message=user_question,
                )
            )

        # Save both turns to history so the next message has full context
        st.session_state.chat[ticker].append({"role": "user",      "content": user_question})
        st.session_state.chat[ticker].append({"role": "assistant",  "content": response})

else:
    st.info("Enter a ticker above and click **Analyze** to run the full pipeline.")
