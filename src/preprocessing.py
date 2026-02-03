import pandas as pd
import numpy as np

def preprocess_crypto_data(df):
    """
    Clean and enrich crypto price data for analysis and modeling.
    """

    # Sort by date (important for time series)
    df = df.sort_values("Date").reset_index(drop=True)

    # Daily returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Volatility (rolling 7-day standard deviation)
    df["Volatility_7D"] = df["Daily_Return"].rolling(window=7).std()

    # Moving averages
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_30"] = df["Close"].rolling(window=30).mean()

    # Drop initial NaN rows
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    from data_loader import load_crypto_data

    df = load_crypto_data("BTC", "2016-01-01")
    df = preprocess_crypto_data(df)

    print(df.head())
    print("\nColumns:", df.columns)
