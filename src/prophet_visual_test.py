import matplotlib.pyplot as plt

from data_loader import load_crypto_data
from preprocessing import preprocess_crypto_data
from prophet_model import prophet_forecast

# ----------------------------------
# LOAD & PREPROCESS DATA
# ----------------------------------
df = load_crypto_data("BTC", "2016-01-01")
df = preprocess_crypto_data(df)

# ----------------------------------
# PROPHET FORECAST
# ----------------------------------
prophet_df = prophet_forecast(df)

# ----------------------------------
# PLOT: ACTUAL vs PROPHET
# ----------------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df["Close"],
    label="Actual Price",
    color="black",
    alpha=0.6
)

plt.plot(
    prophet_df["Date"],
    prophet_df["Prophet_Forecast"],
    label="Prophet Trend",
    color="blue",
    linewidth=2
)

plt.title("Bitcoin Price vs Prophet Trend (Log-Transformed)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
