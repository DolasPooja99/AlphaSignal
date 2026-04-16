import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Scans price + volume data for unusual movements.
    
    Inputs:  DataFrame with columns: date, close, volume
    Outputs: dict with anomalies found and severity scores
    """
    df = df.copy()

    # --- Price change % each day ---
    df["price_change_pct"] = df["close"].pct_change() * 100

    # --- Z-score of price change ---
    mean_change = df["price_change_pct"].mean()
    std_change = df["price_change_pct"].std()
    df["price_zscore"] = (df["price_change_pct"] - mean_change) / std_change

    # --- Volume spike: today vs 20-day average ---
    df["volume_ma20"] = df["volume"].rolling(window=20).mean()
    df["volume_spike"] = df["volume"] / df["volume_ma20"]

    # --- Scan every row for anomalies (full 90-day history) ---
    # We skip the first 20 rows because volume_ma20 needs 20 days of data
    # to produce a valid average. Before that, volume_spike is NaN.
    history = []
    for _, row in df.iterrows():
        if pd.isna(row["volume_spike"]):
            continue  # not enough history yet for a valid volume baseline

        # Price spike: z-score beyond ±2 (roughly the top/bottom 5% of moves)
        if abs(row["price_zscore"]) > 2:
            direction = "up" if row["price_change_pct"] > 0 else "down"
            history.append({
                "date": str(row["date"].date()),
                "close": round(row["close"], 2),
                "type": "price_spike",
                "direction": direction,
                "value": round(row["price_change_pct"], 2),
                "zscore": round(row["price_zscore"], 2),
                "severity": "high" if abs(row["price_zscore"]) > 3 else "medium",
            })

        # Volume spike: more than 2x the 20-day average volume
        # A volume spike on the same day as a price spike is a separate signal —
        # it means the price move had conviction behind it, not just noise
        elif row["volume_spike"] > 2:
            history.append({
                "date": str(row["date"].date()),
                "close": round(row["close"], 2),
                "type": "volume_spike",
                "value": round(row["volume_spike"], 2),
                "severity": "high" if row["volume_spike"] > 3 else "medium",
            })

    # --- Today's anomalies (last row only) — kept for backward compatibility ---
    latest = df.iloc[-1]
    todays_anomalies = [h for h in history if h["date"] == str(latest["date"].date())]

    # Multi-day streak check — still only checked on the most recent 5 days
    # (a rolling streak check across all 90 days would create too many flags)
    recent_changes = df.tail(5)["price_change_pct"].values
    if all(x > 0 for x in recent_changes):
        todays_anomalies.append({
            "type": "consecutive_gains",
            "days": 5,
            "total_change": round(sum(recent_changes), 2),
            "severity": "medium",
        })
    elif all(x < 0 for x in recent_changes):
        todays_anomalies.append({
            "type": "consecutive_losses",
            "days": 5,
            "total_change": round(sum(recent_changes), 2),
            "severity": "medium",
        })

    return {
        "ticker": None,
        "latest_date": str(latest["date"].date()),
        "latest_close": round(latest["close"], 2),
        "latest_change_pct": round(latest["price_change_pct"], 2),
        "latest_volume_spike": round(latest["volume_spike"], 2),
        "anomalies_found": len(todays_anomalies),
        "anomalies": todays_anomalies,       # today only — used by app.py anomaly detail section
        "history": history,                  # all 90 days — used by chart + Claude
        "stats": {
            "mean_daily_change": round(mean_change, 2),
            "std_daily_change": round(std_change, 2),
            "avg_volume_20d": int(df["volume_ma20"].iloc[-1]),
            "total_anomaly_days": len(history),   # frequency signal for Claude
        }
    }


if __name__ == "__main__":
    from data_fetcher import fetch_daily_prices
    import time

    df = fetch_daily_prices("AAPL", days=90)
    time.sleep(12)
    
    result = detect_anomalies(df)
    result["ticker"] = "AAPL"

    print(f"\nTicker: AAPL")
    print(f"Latest close: ${result['latest_close']}")
    print(f"Today's change: {result['latest_change_pct']}%")
    print(f"Volume spike: {result['latest_volume_spike']}x average")
    print(f"\nAnomalies found: {result['anomalies_found']}")
    for a in result['anomalies']:
        print(f"  → {a['type']} ({a['severity']}): {a}")
    print(f"\nStats: {result['stats']}")