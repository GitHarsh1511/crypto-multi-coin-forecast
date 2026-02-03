import matplotlib.pyplot as plt

from data_loader import load_crypto_data
from preprocessing import preprocess_crypto_data
from lstm_model import lstm_forecast

# ----------------------------------
# LOAD & PREPROCESS DATA
# ----------------------------------
df = load_crypto_data("BTC", "2016-01-01")
df = preprocess_crypto_data(df)

# ----------------------------------
# LSTM FORECAST
# ----------------------------------
lstm_df = lstm_forecast(df)

# ----------------------------------
# PLOT: ACTUAL vs LSTM
# ----------------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    lstm_df["Date"],
    lstm_df["Actual_Price"],
    label="Actual Price",
    color="black",
    alpha=0.6
)

plt.plot(
    lstm_df["Date"],
    lstm_df["LSTM_Prediction"],
    label="LSTM Prediction",
    color="red"
)

plt.title("Bitcoin Price: Actual vs LSTM Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
