import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

def forecast_prices(df: pd.DataFrame, days_ahead: int = 7) -> dict:
    """
    Uses Facebook Prophet to forecast stock prices.
    
    Why Prophet?
    - Handles missing days (weekends, holidays) automatically
    - Captures weekly and yearly seasonality in stock data
    - Returns confidence intervals — upper and lower bounds
    - No manual parameter tuning needed for a baseline model
    
    Inputs:  DataFrame with date, close columns
    Outputs: dict with forecast, confidence bands, and trend direction
    """

    # Prophet expects columns named 'ds' (date) and 'y' (value)
    prophet_df = df[["date", "close"]].rename(
        columns={"date": "ds", "close": "y"}
    )

    # Fit the model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,  # 90 days is not enough to learn a yearly cycle
        changepoint_prior_scale=0.05  # lower = less sensitive to recent changes
    )
    model.fit(prophet_df)

    # Create future dates to predict
    # freq='B' = business days only — skips weekends so we never forecast
    # on non-trading days (which cause negative price artifacts)
    future = model.make_future_dataframe(periods=days_ahead, freq='B')
    forecast = model.predict(future)

    # Get only the future predictions
    future_forecast = forecast.tail(days_ahead)

    # Calculate trend direction
    first_pred = future_forecast["yhat"].iloc[0]
    last_pred = future_forecast["yhat"].iloc[-1]
    trend_pct = ((last_pred - first_pred) / first_pred) * 100

    if trend_pct > 2:
        trend = "bullish"
    elif trend_pct < -2:
        trend = "bearish"
    else:
        trend = "neutral"

    # Calculate confidence width — wider band = more uncertainty
    avg_confidence_width = (
        future_forecast["yhat_upper"] - future_forecast["yhat_lower"]
    ).mean()
    current_price = df["close"].iloc[-1]
    confidence_pct = (avg_confidence_width / current_price) * 100

    if confidence_pct < 3:
        confidence_level = "high"
    elif confidence_pct < 6:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # Build predictions list
    predictions = []
    for _, row in future_forecast.iterrows():
        predictions.append({
            "date": str(row["ds"].date()),
            "predicted": round(row["yhat"], 2),
            "lower": round(row["yhat_lower"], 2),
            "upper": round(row["yhat_upper"], 2)
        })

    return {
        "days_ahead": days_ahead,
        "trend": trend,
        "trend_pct": round(trend_pct, 2),
        "confidence_level": confidence_level,
        "confidence_width_pct": round(confidence_pct, 2),
        "current_price": round(current_price, 2),
        "predicted_price_7d": round(last_pred, 2),
        "predictions": predictions
    }


if __name__ == "__main__":
    from data_fetcher import fetch_daily_prices

    df = fetch_daily_prices("AAPL", days=90)

    result = forecast_prices(df, days_ahead=7)

    print(f"\nCurrent price: ${result['current_price']}")
    print(f"Predicted price in 7 days: ${result['predicted_price_7d']}")
    print(f"Trend: {result['trend']} ({result['trend_pct']}%)")
    print(f"Confidence: {result['confidence_level']} (band width: {result['confidence_width_pct']}%)")
    print(f"\nDay by day predictions:")
    for p in result['predictions']:
        print(f"  {p['date']}: ${p['predicted']} (${p['lower']} — ${p['upper']})")