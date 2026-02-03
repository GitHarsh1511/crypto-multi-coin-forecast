from src.data_loader import load_crypto_data
from src.preprocessing import preprocess_crypto_data
from src.lstm_model import lstm_forecast_30_days

df = preprocess_crypto_data(load_crypto_data("BTC", "2016-01-01"))

forecast_df = lstm_forecast_30_days(df)

print(forecast_df.head())
print(forecast_df.tail())
