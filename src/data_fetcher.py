import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def fetch_daily_prices(ticker: str, days: int = 90) -> pd.DataFrame:
    """
    Fetches daily price + volume data for a stock ticker.
    Returns a DataFrame with columns: date, open, high, low, close, volume
    """
    print(f"Fetching {days} days of data for {ticker}...")

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": ALPHAVANTAGE_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError(f"Could not fetch data for {ticker}. Response: {data}")

    time_series = data["Time Series (Daily)"]

    rows = []
    for date, values in time_series.items():
        rows.append({
            "date": pd.to_datetime(date),
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "volume": int(values["5. volume"])
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.tail(days)

    print(f"Got {len(df)} days of data. Latest close: ${df['close'].iloc[-1]:.2f}")
    return df


def fetch_current_price(ticker: str) -> dict:
    """
    Fetches the latest price quote for a ticker.
    Returns price, change, change percent, volume.
    """
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker,
        "apikey": ALPHAVANTAGE_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "Global Quote" not in data:
        raise ValueError(f"Could not fetch quote for {ticker}")

    quote = data["Global Quote"]
    return {
        "ticker": ticker,
        "price": float(quote["05. price"]),
        "change": float(quote["09. change"]),
        "change_pct": float(quote["10. change percent"].replace("%", "")),
        "volume": int(quote["06. volume"]),
        "latest_trading_day": quote["07. latest trading day"]
    }


if __name__ == "__main__":
    import time
    
    df = fetch_daily_prices("AAPL", days=90)
    print(df.tail(5))
    
    print("\nWaiting 15 seconds for API rate limit...")
    time.sleep(15)
    
    quote = fetch_current_price("AAPL")
    print(f"\nCurrent: ${quote['price']} ({quote['change_pct']}%)")