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

    # --- Flag anomalies ---
    latest = df.iloc[-1]
    recent = df.tail(10)

    anomalies = []

    # Price anomaly — z-score above 2 or below -2
    if abs(latest["price_zscore"]) > 2:
        direction = "up" if latest["price_change_pct"] > 0 else "down"
        anomalies.append({
            "type": "price_spike",
            "direction": direction,
            "value": round(latest["price_change_pct"], 2),
            "zscore": round(latest["price_zscore"], 2),
            "severity": "high" if abs(latest["price_zscore"]) > 3 else "medium"
        })

    # Volume anomaly — volume 2x above 20-day average
    if latest["volume_spike"] > 2:
        anomalies.append({
            "type": "volume_spike",
            "value": round(latest["volume_spike"], 2),
            "severity": "high" if latest["volume_spike"] > 3 else "medium"
        })

    # Multi-day trend — 5 consecutive up or down days
    recent_changes = df.tail(5)["price_change_pct"].values
    if all(x > 0 for x in recent_changes):
        anomalies.append({
            "type": "consecutive_gains",
            "days": 5,
            "total_change": round(sum(recent_changes), 2),
            "severity": "medium"
        })
    elif all(x < 0 for x in recent_changes):
        anomalies.append({
            "type": "consecutive_losses",
            "days": 5,
            "total_change": round(sum(recent_changes), 2),
            "severity": "medium"
        })

    return {
        "ticker": None,
        "latest_date": str(latest["date"].date()),
        "latest_close": round(latest["close"], 2),
        "latest_change_pct": round(latest["price_change_pct"], 2),
        "latest_volume_spike": round(latest["volume_spike"], 2),
        "anomalies_found": len(anomalies),
        "anomalies": anomalies,
        "stats": {
            "mean_daily_change": round(mean_change, 2),
            "std_daily_change": round(std_change, 2),
            "avg_volume_20d": int(df["volume_ma20"].iloc[-1])
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