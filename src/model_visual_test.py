import matplotlib.pyplot as plt

from data_loader import load_crypto_data
from preprocessing import preprocess_crypto_data
from arima_model import arima_forecast
from sarima_model import sarima_forecast

# ----------------------------------
# LOAD & PREPROCESS DATA
# ----------------------------------
df = load_crypto_data("BTC", "2016-01-01")
df = preprocess_crypto_data(df)

# ----------------------------------
# RUN MODELS
# ----------------------------------
arima_df = arima_forecast(df)
sarima_df = sarima_forecast(df)

# ----------------------------------
# PLOT ACTUAL VS MODELS
# ----------------------------------
plt.figure(figsize=(14, 6))

plt.plot(df["Date"], df["Close"], label="Actual Price", color="black", alpha=0.6)

plt.plot(
    arima_df["Date"],
    arima_df["ARIMA_Fitted"],
    label="ARIMA",
    linestyle="--"
)

plt.plot(
    sarima_df["Date"],
    sarima_df["SARIMA_Fitted"],
    label="SARIMA",
    linestyle=":"
)

plt.title("Model Fit Comparison (BTC)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
