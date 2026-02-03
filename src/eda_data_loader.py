import pandas as pd

def load_eda_data(csv_path: str):
    """
    Loads and cleans data STRICTLY for EDA & candlestick charts.
    This function guarantees mplfinance-safe output.
    """

    # Read raw CSV
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip().capitalize() for c in df.columns]

    # Required columns for EDA
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[required_cols]

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

    # Convert OHLCV columns to numeric safely
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with ANY invalid values
    df = df.dropna()

    # Force float type (required by mplfinance)
    df = df.astype(float)

    # Sort by date
    df = df.sort_index()

    return df
