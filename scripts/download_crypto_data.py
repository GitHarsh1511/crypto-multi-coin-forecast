import yfinance as yf
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "ADA": "ADA-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
    "DOT": "DOT-USD",
    "AVAX": "AVAX-USD",
    "MATIC": "MATIC-USD",
    "LTC": "LTC-USD",
    "BCH": "BCH-USD",
    "TRX": "TRX-USD",
    "LINK": "LINK-USD",
    "UNI": "UNI-USD",
}

START_DATE = "2016-01-01"
OUTPUT_DIR = Path("data/eda")

# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# DOWNLOAD & CLEAN DATA
# --------------------------------------------------
for coin, symbol in SYMBOLS.items():
    print(f"\n⬇️ Downloading {coin} ({symbol}) ...")

    df = yf.download(
        symbol,
        start=START_DATE,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        print(f"⚠️ No data for {coin}, skipping.")
        continue

    # Fix MultiIndex columns (yfinance issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to get Date column
    df.reset_index(inplace=True)

    # Keep required columns ONLY
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Force numeric (critical for mplfinance)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows
    df.dropna(inplace=True)

    # Save CSV
    output_path = OUTPUT_DIR / f"{coin}.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Saved: {output_path}")

print("\n🎉 ALL CRYPTO DATA DOWNLOADED SUCCESSFULLY")
