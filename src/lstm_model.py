import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # pyright: ignore[reportMissingImports]

def lstm_forecast(df, lookback=60):
    """
    Train LSTM model and return fitted values for comparison.
    """

    # -----------------------------
    # Prepare data
    # -----------------------------
    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []

    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # -----------------------------
    # Train-test split
    # -----------------------------
    train_size = int(len(X) * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # -----------------------------
    # Build LSTM model
    # -----------------------------
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=0
    )

    # -----------------------------
    # Predict (test set)
    # -----------------------------
    predictions = model.predict(X_test, verbose=0)

    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Align dates
    prediction_dates = df["Date"].iloc[train_size + lookback:]

    result_df = pd.DataFrame({
        "Date": prediction_dates.values,
        "LSTM_Prediction": predictions.flatten(),
        "Actual_Price": y_test_actual.flatten()
    })

    return result_df


# ---------------- TEST ----------------
if __name__ == "__main__":
    from data_loader import load_crypto_data
    from preprocessing import preprocess_crypto_data

    df = load_crypto_data("BTC", "2016-01-01")
    df = preprocess_crypto_data(df)

    lstm_df = lstm_forecast(df)
    print(lstm_df.head())



import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense, Input # pyright: ignore[reportMissingImports]


def lstm_forecast_30_days(df, lookback=60, steps=30):
    """
    Recursive LSTM forecasting for future prices AFTER last available date.
    """

    # -----------------------------
    # Prepare data
    # -----------------------------
    data = df[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])

    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # -----------------------------
    # Build & train LSTM
    # -----------------------------
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, scaled_data[lookback:], epochs=5, batch_size=32, verbose=0)

    # -----------------------------
    # Recursive forecasting
    # -----------------------------
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)

    future_predictions = []

    for _ in range(steps):
        next_scaled = model.predict(last_sequence, verbose=0)[0, 0]
        future_predictions.append(next_scaled)

        last_sequence = np.append(
            last_sequence[:, 1:, :],
            [[[next_scaled]]],
            axis=1
        )

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    # -----------------------------
    # Future dates
    # -----------------------------
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="D"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future_predictions.flatten()
    })

    return forecast_df
