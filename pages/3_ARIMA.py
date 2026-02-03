import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import load_crypto_data
from src.preprocessing import preprocess_crypto_data
from src.arima_model import arima_forecast, arima_forecast_30_days

# ----------------------------------
# PAGE TITLE
# ----------------------------------
st.title("📈 ARIMA Model")

# ----------------------------------
# COIN SELECTOR
# ----------------------------------
coins = [
    "BTC", "ETH", "BNB", "ADA", "SOL",
    "XRP", "DOGE", "DOT", "AVAX", "MATIC",
    "LTC", "BCH", "TRX", "LINK", "UNI"
]

coin = st.sidebar.selectbox("Select Coin", coins)

# ----------------------------------
# LOAD & PREPROCESS DATA
# ----------------------------------
df = preprocess_crypto_data(load_crypto_data(coin, "2016-01-01"))

# ----------------------------------
# 1️⃣ HISTORICAL FIT
# ----------------------------------
st.subheader("📊 Historical Fit (Actual vs ARIMA)")

arima_df = arima_forecast(df)

fig1, ax1 = plt.subplots(figsize=(12, 4))

ax1.plot(df["Date"], df["Close"], label="Actual Price", color="black", alpha=0.6)
ax1.plot(arima_df["Date"], arima_df["ARIMA_Fitted"], label="ARIMA Fit", color="red")

ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)

# ----------------------------------
# 2️⃣ FUTURE FORECAST (30 DAYS)
# ----------------------------------
st.subheader("🔮 30-Day Forecast (After 31-12-2025)")

forecast_df = arima_forecast_30_days(df)

fig2, ax2 = plt.subplots(figsize=(12, 4))

ax2.plot(
    forecast_df["Date"],
    forecast_df["Forecast"],
    label="ARIMA 30-Day Forecast",
    color="green"
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# ----------------------------------
# INSIGHT
# ----------------------------------
st.info(
    "The first chart shows how ARIMA fits historical data. "
    "The second chart shows the **out-of-sample 30-day forecast** beyond the last known date."
)
