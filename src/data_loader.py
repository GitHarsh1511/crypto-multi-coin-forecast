import yfinance as yf
import pandas as pd

# 🔒 Project-wide fixed end date
PROJECT_END_DATE = pd.Timestamp("2025-12-31")

def load_crypto_data(symbol, start_date):
    """
    Fetch historical crypto data and cap it till 31-12-2025.
    """

    ticker = f"{symbol}-USD"
    df = yf.download(ticker, start=start_date, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # 🔹 IMPORTANT: Cap data till 31-12-2025
    df = df[df["Date"] <= PROJECT_END_DATE]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.reset_index(drop=True)

    return df
