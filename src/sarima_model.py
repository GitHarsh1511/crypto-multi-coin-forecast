import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

def sarima_forecast(df):
    """
    Stable SARIMA model for daily crypto prices
    (data capped till 31-12-2025).
    """

    # Set Date as index with explicit daily frequency
    ts = df.set_index("Date")["Close"]
    ts = ts.asfreq("D")

    # Fill missing days (crypto trades daily, but safety)
    ts = ts.fillna(method="ffill")

    # SIMPLER & STABLE SARIMA MODEL
    model = SARIMAX(
        ts,
        order=(1, 1, 1),          # non-seasonal
        seasonal_order=(0, 0, 0, 0),  # ❌ remove seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    fitted_values = model_fit.fittedvalues

    forecast_df = pd.DataFrame({
        "Date": fitted_values.index,
        "SARIMA_Fitted": fitted_values.values
    })

    return forecast_df


# ---------------- TEST ----------------
if __name__ == "__main__":
    from data_loader import load_crypto_data
    from preprocessing import preprocess_crypto_data

    df = load_crypto_data("BTC", "2016-01-01")
    df = preprocess_crypto_data(df)

    print("Last date in dataset:", df["Date"].max())

    sarima_df = sarima_forecast(df)
    print(sarima_df.head())
    
    
    
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast_30_days(df, steps=30):
    """
    Forecast future prices AFTER the last available date using SARIMA.
    """

    series = df["Close"].values

    # Use same SARIMA order as your fitted model
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.get_forecast(steps=steps)
    forecast_values = forecast.predicted_mean

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
