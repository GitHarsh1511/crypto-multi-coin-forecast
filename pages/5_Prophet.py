import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import load_crypto_data
from src.preprocessing import preprocess_crypto_data
from src.prophet_model import prophet_forecast, prophet_forecast_30_days

st.title("📈 Prophet Model")

coins = [
    "BTC", "ETH", "BNB", "ADA", "SOL",
    "XRP", "DOGE", "DOT", "AVAX", "MATIC",
    "LTC", "BCH", "TRX", "LINK", "UNI"
]

coin = st.sidebar.selectbox("Select Coin", coins)

df = preprocess_crypto_data(load_crypto_data(coin, "2016-01-01"))

# -----------------------------
# 1️⃣ HISTORICAL FIT
# -----------------------------
st.subheader("📊 Historical Trend (Actual vs Prophet)")

prophet_df = prophet_forecast(df)

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df["Date"], df["Close"], label="Actual", color="black", alpha=0.6)
ax1.plot(prophet_df["Date"], prophet_df["Prophet_Forecast"], label="Prophet Trend", color="blue")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)

# -----------------------------
# 2️⃣ FUTURE FORECAST
# -----------------------------
st.subheader("🔮 30-Day Forecast (After 31-12-2025)")

forecast_df = prophet_forecast_30_days(df)

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(
    forecast_df["Date"],
    forecast_df["Forecast"],
    label="Prophet 30-Day Forecast",
    color="blue"
)
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

st.info(
    "Prophet focuses on **long-term trend forecasting**. "
    "The second chart shows an **out-of-sample 30-day forecast**."
)
