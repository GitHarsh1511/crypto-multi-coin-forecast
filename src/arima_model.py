import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(df):
    """
    Train ARIMA model on data capped till 31-12-2025.
    """

    ts = df.set_index("Date")["Close"]

    # Simple ARIMA baseline
    model = ARIMA(ts, order=(5, 1, 0))
    model_fit = model.fit()

    # In-sample prediction (fitted values)
    fitted_values = model_fit.fittedvalues

    forecast_df = pd.DataFrame({
        "Date": fitted_values.index,
        "ARIMA_Fitted": fitted_values.values
    })

    return forecast_df


# ---------------- TEST ----------------
if __name__ == "__main__":
    from data_loader import load_crypto_data
    from preprocessing import preprocess_crypto_data

    df = load_crypto_data("BTC", "2016-01-01")
    df = preprocess_crypto_data(df)

    print("Last date in dataset:", df["Date"].max())

    arima_df = arima_forecast(df)
    print(arima_df.head())

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast_30_days(df, steps=30):
    """
    Forecast future prices AFTER the last available date.
    """

    # Use closing prices
    series = df["Close"].values

    # Fit ARIMA (same order you already use)
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast future steps
    forecast = model_fit.get_forecast(steps=steps)
    forecast_values = forecast.predicted_mean

    # Generate future dates
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="D"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast_values
    })

    return forecast_df
