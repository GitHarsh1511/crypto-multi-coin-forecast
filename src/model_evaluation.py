import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_crypto_data
from src.preprocessing import preprocess_crypto_data
from src.arima_model import arima_forecast
from src.sarima_model import sarima_forecast
from src.prophet_model import prophet_forecast
from src.lstm_model import lstm_forecast


def evaluate_models():
    # ----------------------------------
    # LOAD & PREPROCESS DATA
    # ----------------------------------
    df = load_crypto_data("BTC", "2016-01-01")
    df = preprocess_crypto_data(df)

    actual_prices = df["Close"].values

    # ----------------------------------
    # ARIMA
    # ----------------------------------
    arima_df = arima_forecast(df)
    arima_pred = arima_df["ARIMA_Fitted"].values

    min_len = min(len(actual_prices), len(arima_pred))
    actual = actual_prices[-min_len:]
    arima_pred = arima_pred[-min_len:]

    # ----------------------------------
    # SARIMA
    # ----------------------------------
    sarima_df = sarima_forecast(df)
    sarima_pred = sarima_df["SARIMA_Fitted"].values[-min_len:]

    # ----------------------------------
    # PROPHET
    # ----------------------------------
    prophet_df = prophet_forecast(df)
    prophet_pred = prophet_df["Prophet_Forecast"].values[-min_len:]

    # ----------------------------------
    # LSTM
    # ----------------------------------
    lstm_df = lstm_forecast(df)
    lstm_pred = lstm_df["LSTM_Prediction"].values[-min_len:]
    actual_lstm = lstm_df["Actual_Price"].values[-min_len:]

    # ----------------------------------
    # METRICS
    # ----------------------------------
    results = [
        {
            "Model": "ARIMA",
            "MAE": mean_absolute_error(actual, arima_pred),
            "RMSE": np.sqrt(mean_squared_error(actual, arima_pred))
        },
        {
            "Model": "SARIMA",
            "MAE": mean_absolute_error(actual, sarima_pred),
            "RMSE": np.sqrt(mean_squared_error(actual, sarima_pred))
        },
        {
            "Model": "Prophet",
            "MAE": mean_absolute_error(actual, prophet_pred),
            "RMSE": np.sqrt(mean_squared_error(actual, prophet_pred))
        },
        {
            "Model": "LSTM",
            "MAE": mean_absolute_error(actual_lstm, lstm_pred),
            "RMSE": np.sqrt(mean_squared_error(actual_lstm, lstm_pred))
        }
    ]

    return pd.DataFrame(results)


# ---------------- TEST ----------------
if __name__ == "__main__":
    eval_df = evaluate_models()
    print(eval_df)
