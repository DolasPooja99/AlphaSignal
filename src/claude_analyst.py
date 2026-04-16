import os
import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# System prompt is stable — cache it so repeat analyses are cheap
SYSTEM_PROMPT = """You are a quantitative stock analyst. You receive four signals for a given ticker:

1. Recent price data (90 days of daily closes and volume)
2. Anomaly detection results (unusual price/volume movements)
3. A 7-day Prophet forecast with confidence bands
4. Relevant recent news articles retrieved by semantic similarity

Your job is to synthesize these signals into a clear, plain-English analysis report.

Structure your report exactly as follows:

## Summary
One paragraph (3–4 sentences). State the current situation, the forecast direction, and the key risk factor. Write for a sophisticated investor who wants signal, not noise.

## Price Action & Anomalies
What has the stock been doing? Note any detected anomalies — price spikes, volume surges, consecutive winning/losing streaks. Be specific about dates and magnitudes when relevant.

## 7-Day Forecast
Describe the Prophet forecast: direction, magnitude, and confidence level. Explain what the confidence band width means in plain English (e.g., "the model is uncertain" vs "the model sees a clear trend"). Call out if the forecast seems contradicted by other signals.

## News Context
Summarize the most relevant recent news. Does the news explain any anomalies? Does it support or contradict the forecast? Cite article titles where useful.

## Key Risks & Uncertainty
What could make this analysis wrong? Identify 2–3 specific risks. Be honest about model limitations (Prophet can't predict earnings surprises, news sentiment is a lagging signal, etc.).

## Verdict
One sentence. Bullish / Bearish / Neutral, and the single most important reason why.

Rules:
- Never give a buy/sell recommendation. You analyze, not advise.
- Quantify when possible: use actual dollar figures and percentages from the data.
- Acknowledge uncertainty explicitly — don't manufacture false confidence.
- If signals conflict, say so clearly rather than picking a side.
"""


def build_context_message(
    ticker: str,
    price_data: pd.DataFrame,
    anomalies: dict,
    forecast: dict,
    news: list,
) -> str:
    """
    Builds the structured data context string shared by both analyze() and chat().

    Extracted as a standalone function so the chat feature can pass the same
    context to Claude on every follow-up question without re-running the pipeline.
    """
    price_summary    = _format_price_summary(price_data)
    anomaly_summary  = _format_anomaly_summary(anomalies, ticker)
    forecast_summary = _format_forecast_summary(forecast)
    news_summary     = _format_news_summary(news)

    return f"""Ticker: {ticker}

### SIGNAL 1: Recent Price Data (last 10 trading days shown)
{price_summary}

### SIGNAL 2: Anomaly Detection
{anomaly_summary}

### SIGNAL 3: 7-Day Price Forecast
{forecast_summary}

### SIGNAL 4: Relevant News (retrieved by semantic similarity)
{news_summary}"""


def analyze(
    ticker: str,
    price_data: pd.DataFrame,
    anomalies: dict,
    forecast: dict,
    news: list,
) -> str:
    """
    Synthesizes price data, anomalies, forecast, and news into a plain-English
    analyst report using Claude.

    Uses prompt caching on the system prompt (stable) and streams the response
    to avoid timeout issues on long outputs.

    Returns the full report as a string.
    """
    context = build_context_message(ticker, price_data, anomalies, forecast, news)
    user_message = f"""Please analyze the following stock data and generate your full report.

---
{context}
---
Generate your full analysis report now."""

    # --- Call Claude with prompt caching on the system prompt ---
    # cache_control on the system prompt caches it for 5 minutes (default TTL)
    # Subsequent analyses within that window pay ~0.1x the system prompt cost
    report_parts = []

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            report_parts.append(text)
            print(text, end="", flush=True)

    final = stream.get_final_message()
    cache_read = final.usage.cache_read_input_tokens
    cache_write = final.usage.cache_creation_input_tokens
    if cache_read > 0:
        print(f"\n\n[Cache hit: {cache_read:,} tokens read from cache]")
    elif cache_write > 0:
        print(f"\n\n[Cache miss: {cache_write:,} tokens written to cache]")

    return "".join(report_parts)


CHAT_SYSTEM_PROMPT = """You are a financial analyst assistant. The user has just read a stock analysis report and wants to ask follow-up questions about it.

You have access to the full underlying data: price history, anomaly detection results, the 7-day forecast, and recent news. Use this data to answer questions precisely — cite specific numbers, dates, and percentages when relevant.

Rules:
- Keep answers concise: 2–4 sentences for simple questions, longer only when complexity demands it.
- Never give buy/sell recommendations. You explain and analyze, not advise.
- If the question is outside the data you have (e.g. "what will happen next month?"), say so honestly.
- Use plain English. No jargon without explanation."""


def stream_chat_response(context: str, report: str, history: list, user_message: str):
    """
    Generator that streams Claude's response to a follow-up chat question.

    How the context is structured:
      - system prompt: short chat-focused instructions (cached)
      - first user message: the full stock data context (cached — same every turn)
      - first assistant message: the original analysis report Claude wrote
      - subsequent turns: the user's follow-up questions and Claude's answers

    Why this structure?
    Claude remembers what it said in the report because that's the first
    assistant turn. It can answer "what did you mean by..." or "expand on..."
    without us re-explaining anything. The data context is also always available
    so Claude can answer factual questions like "what was the exact z-score on Feb 12?"

    Prompt caching:
    The system prompt + context block are stable across all turns in the same
    session. With cache_control, turns 2+ pay ~10% of the context token cost.
    Over a 10-message conversation, this is roughly a 6x cost reduction.
    """
    # Build the full message history Claude sees:
    # [data context as user] → [report as assistant] → [follow-up Q&A...]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the stock data for this analysis:\n\n{context}",
                    "cache_control": {"type": "ephemeral"},  # cached — same every turn
                }
            ],
        },
        {
            "role": "assistant",
            "content": report,   # Claude's original analysis — it can refer back to this
        },
    ]

    # Append the conversation so far, then the new question
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": CHAT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    ) as stream:
        yield from stream.text_stream


# --- Formatting helpers ---

def _format_price_summary(df: pd.DataFrame) -> str:
    recent = df.tail(10).copy()
    lines = ["Date         | Close    | Volume"]
    lines.append("-------------|----------|------------")
    for _, row in recent.iterrows():
        lines.append(
            f"{str(row['date'].date())} | ${row['close']:.2f}   | {int(row['volume']):,}"
        )
    return "\n".join(lines)


def _format_anomaly_summary(anomalies: dict, ticker: str) -> str:
    stats = anomalies["stats"]
    history = anomalies.get("history", [])

    lines = [
        f"Ticker: {ticker}",
        f"Latest date: {anomalies['latest_date']}",
        f"Latest close: ${anomalies['latest_close']}",
        f"Today's change: {anomalies['latest_change_pct']}%",
        f"Volume vs 20d avg: {anomalies['latest_volume_spike']}x",
        f"",
        f"Base stats (90-day window):",
        f"  Mean daily change: {stats['mean_daily_change']}%",
        f"  Std daily change:  {stats['std_daily_change']}%",
        f"  Avg volume (20d):  {stats['avg_volume_20d']:,}",
        f"  Anomalous days:    {stats['total_anomaly_days']} out of ~70 trading days",
        f"  (frequency: 1 anomaly every ~{round(70 / max(stats['total_anomaly_days'], 1), 1)} days)",
    ]

    # Today's anomalies
    if anomalies["anomalies"]:
        lines.append("\nToday's anomalies:")
        for a in anomalies["anomalies"]:
            if a["type"] == "price_spike":
                lines.append(
                    f"  - price_spike ({a['severity']}): {a['direction']} "
                    f"{a['value']}% move, z-score={a['zscore']}"
                )
            elif a["type"] == "volume_spike":
                lines.append(
                    f"  - volume_spike ({a['severity']}): {a['value']}x normal volume"
                )
            elif a["type"] in ("consecutive_gains", "consecutive_losses"):
                lines.append(
                    f"  - {a['type']} ({a['severity']}): {a['days']} days, "
                    f"total move {a['total_change']}%"
                )
    else:
        lines.append("\nNo anomalies detected today.")

    # Full 90-day anomaly history — gives Claude frequency + pattern context
    if history:
        lines.append(f"\nFull anomaly history ({len(history)} events):")
        for h in history:
            if h["type"] == "price_spike":
                lines.append(
                    f"  {h['date']} | price_spike {h['direction']:4s} | "
                    f"{h['value']:+.2f}% | z={h['zscore']} | close=${h['close']} | {h['severity']}"
                )
            elif h["type"] == "volume_spike":
                lines.append(
                    f"  {h['date']} | volume_spike       | "
                    f"{h['value']:.2f}x avg vol | close=${h['close']} | {h['severity']}"
                )
    else:
        lines.append("\nNo anomalies detected in the 90-day window.")

    return "\n".join(lines)


def _format_forecast_summary(forecast: dict) -> str:
    lines = [
        f"Current price: ${forecast['current_price']}",
        f"Predicted price in 7 days: ${forecast['predicted_price_7d']}",
        f"Trend: {forecast['trend']} ({forecast['trend_pct']}%)",
        f"Confidence: {forecast['confidence_level']} (band width: {forecast['confidence_width_pct']}%)",
        "",
        "Day-by-day predictions:",
    ]
    for p in forecast["predictions"]:
        lines.append(
            f"  {p['date']}: ${p['predicted']} (range: ${p['lower']} – ${p['upper']})"
        )
    return "\n".join(lines)


def _format_news_summary(news: list) -> str:
    if not news:
        return "No recent news found."

    lines = []
    for i, article in enumerate(news, 1):
        lines.append(f"{i}. [{article['published_at']}] {article['title']}")
        if article["description"]:
            desc = article["description"]
            if len(desc) > 200:
                desc = desc[:200] + "..."
            lines.append(f"   {desc}")
        lines.append(f"   Relevance score: {article['relevance_score']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_fetcher import fetch_daily_prices
    from anomaly_detector import detect_anomalies
    from forecaster import forecast_prices
    from news_rag import fetch_and_store_news, retrieve_relevant_news

    ticker = "AAPL"
    print(f"Running full analysis for {ticker}...\n")

    print("Step 1: Fetching price data...")
    df = fetch_daily_prices(ticker, days=90)

    print("Step 2: Detecting anomalies...")
    anomalies = detect_anomalies(df)
    anomalies["ticker"] = ticker

    print("Step 3: Forecasting...")
    forecast = forecast_prices(df, days_ahead=7)

    print("Step 4: Fetching/retrieving news...")
    fetch_and_store_news(ticker)
    query = f"{ticker} stock price earnings forecast outlook"
    news = retrieve_relevant_news(ticker, query, top_k=5)

    print("\nStep 5: Claude analysis...\n")
    print("=" * 60)
    report = analyze(ticker, df, anomalies, forecast, news)
    print("=" * 60)
