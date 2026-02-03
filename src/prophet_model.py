import numpy as np
import pandas as pd
from prophet import Prophet

def prophet_forecast(df):
    """
    Prophet forecast using log-transformed prices
    (prevents negative predictions).
    """

    # -----------------------------
    # Prepare data
    # -----------------------------
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.to_datetime(df["Date"])

    # 🔥 LOG TRANSFORM (KEY FIX)
    prophet_df["y"] = np.log(df["Close"].values.astype(float))

    prophet_df = prophet_df.dropna().reset_index(drop=True)

    # -----------------------------
    # Build & fit Prophet
    # -----------------------------
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=False
    )

    model.fit(prophet_df)

    # In-sample prediction
    forecast = model.predict(prophet_df)

    # -----------------------------
    # Convert back to price scale
    # -----------------------------
    forecast_df = pd.DataFrame()
    forecast_df["Date"] = forecast["ds"]
    forecast_df["Prophet_Forecast"] = np.exp(forecast["yhat"])

    return forecast_df


# ---------------- TEST ----------------
if __name__ == "__main__":
    from data_loader import load_crypto_data
    from preprocessing import preprocess_crypto_data

    df = load_crypto_data("BTC", "2016-01-01")
    df = preprocess_crypto_data(df)

    print("Last date in dataset:", df["Date"].max())

    prophet_df = prophet_forecast(df)
    print(prophet_df.head())


import pandas as pd
import numpy as np
from prophet import Prophet

def prophet_forecast_30_days(df, steps=30):
    """
    Forecast future prices AFTER the last available date using Prophet.
    Uses log-transform to avoid negative prices.
    """

    # Prepare data for Prophet
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.to_datetime(df["Date"])
    prophet_df["y"] = np.log(df["Close"].values)

    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=False
    )

    model.fit(prophet_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=steps, freq="D")

    forecast = model.predict(future)

    # Extract ONLY future dates
    future_forecast = forecast.tail(steps)

    forecast_df = pd.DataFrame({
        "Date": future_forecast["ds"],
        "Forecast": np.exp(future_forecast["yhat"])
    })

    return forecast_df
