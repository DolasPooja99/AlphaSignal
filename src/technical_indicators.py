import pandas as pd


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adds a 'rsi' column to the DataFrame.

    RSI (Relative Strength Index) measures momentum on a 0–100 scale:
      > 70  →  overbought  (price moved up too fast, pullback possible)
      < 30  →  oversold    (price moved down too fast, bounce possible)
      ~50   →  neutral momentum

    How it's calculated:
      1. Find daily price changes
      2. Separate into gains (positive days) and losses (negative days)
      3. Average the gains and losses over the last `period` days (14 is standard)
      4. RS = avg_gain / avg_loss
      5. RSI = 100 - (100 / (1 + RS))

    We use exponential moving average (EWM) instead of simple average —
    this gives more weight to recent days, which makes the indicator more
    responsive to current conditions.
    """
    df = df.copy()
    delta = df["close"].diff()

    gain = delta.clip(lower=0)   # keep only positive moves, set negatives to 0
    loss = -delta.clip(upper=0)  # keep only negative moves (flip sign so positive)

    # ewm(com=period-1) is the standard Wilder smoothing used for RSI
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Adds three MACD columns to the DataFrame:
      'macd'        — the MACD line itself (fast EMA minus slow EMA)
      'macd_signal' — 9-day EMA of the MACD line (smoothed version)
      'macd_hist'   — histogram: MACD minus signal (shows momentum direction)

    How to read it:
      - MACD crosses above signal line  →  bullish crossover (momentum turning up)
      - MACD crosses below signal line  →  bearish crossover (momentum turning down)
      - Histogram bars growing positive →  upward momentum accelerating
      - Histogram bars growing negative →  downward momentum accelerating

    EMA (exponential moving average) gives more weight to recent prices
    than a simple average — so the 12-day EMA reacts faster than the 26-day EMA.
    The difference between them tells you whether short-term momentum is
    outpacing long-term momentum (bullish) or lagging behind it (bearish).
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df
