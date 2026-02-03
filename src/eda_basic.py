import matplotlib.pyplot as plt
from data_loader import load_crypto_data
from preprocessing import preprocess_crypto_data

# Load and preprocess
df = load_crypto_data("BTC", "2016-01-01")
df = preprocess_crypto_data(df)

# -----------------------------
# Price & Moving Averages
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Close"], label="Close Price")
plt.plot(df["Date"], df["MA_7"], label="MA 7")
plt.plot(df["Date"], df["MA_30"], label="MA 30")
plt.title("Bitcoin Price & Moving Averages")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Volatility
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(df["Date"], df["Volatility_7D"], color="red")
plt.title("Bitcoin 7-Day Volatility")
plt.grid(True)
plt.show()
