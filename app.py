import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


from src.data_loader import load_crypto_data
from src.preprocessing import preprocess_crypto_data

# --------------------------------------------------
# GLOBAL COIN LIST (USED ACROSS ALL PAGES)
# --------------------------------------------------
coins = [
    "BTC", "ETH", "BNB", "ADA", "SOL",
    "XRP", "DOGE", "DOT", "AVAX", "MATIC",
    "LTC", "BCH", "TRX", "LINK", "UNI"
]

from src.prophet_model import (
    prophet_forecast,
    prophet_forecast_30_days
)

from src.arima_model import (
    arima_forecast,
    arima_forecast_30_days
)

from src.sarima_model import (
    sarima_forecast,
    sarima_forecast_30_days
)

from src.lstm_model import (
    lstm_forecast,
    lstm_forecast_30_days
)
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Time Series Analysis with Cryptocurrency",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# GLOBAL CSS (DARK + GOLD THEME, SIDEBAR FIXED)
# --------------------------------------------------
st.markdown("""
<style>

/* Hide Streamlit default multipage navigation */
[data-testid="stSidebarNav"] {
    display: none;
}

/* App background */
[data-testid="stAppViewContainer"] {
    background-color: rgb(14,17,23);
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #2b2f33;
}

/* Headings */
h1, h2, h3 {
    color: #ffcc00 !important;
}

/* Text */
p, li {
    color: #e5e7eb;
    font-size: 16px;
}

/* Sidebar buttons */
.stButton > button {
    width: 100%;
    background-color: #2f3439;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 0px;
    border: 2px solid #3d4248;
    font-weight: 700;
    transition: background-color 0.2s ease-in-out;
}

/* Button text */
.stButton > button span {
    color: white;
    font-size: 16px;
}

/* Hover */
.stButton > button:hover {
    background-color: #ffcc00;
}


.stButton > button:focus span {
    color: #1e2124 !important;
}

/* Remove focus outline */
button:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Remove footer */
footer {
    visibility: hidden;
}

[data-testid="metric-container"] {
    background-color: #1f2933;
    border: 1px solid #3d4248;
    border-radius: 12px;
    padding: 16px;
}


</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR NAVIGATION (FIXED)
# --------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

with st.sidebar:
        # 🔶 Project Title
    st.markdown(
        "<h2 style='color:#ffcc00; text-align:center;'>Time Series Analysis with Cryptocurrency</h2>",
        unsafe_allow_html=True
    )

    
    menu_items = [
        "Home",
        "Executive KPIs",
        "Data Overview",   
        "Exploratory Data Analysis (EDA)",
        "Forecasting Models",
        "Model Evaluation"
    ]

    for item in menu_items:
        if st.button(item, key=f"nav_{item}"):
            st.session_state.active_page = item

# --------------------------------------------------
# MAIN CONTENT ROUTING
# --------------------------------------------------
page = st.session_state.active_page

# ---------------- HOME PAGE ----------------
if page == "Home":

    st.markdown("# Time Series Analysis with Cryptocurrency")

    col1, col2 = st.columns([0.65, 0.35])

    with col1:
        st.markdown("### Project Description")

        st.write("""
            This project focuses on **cryptocurrency price analysis and forecasting**
            using historical market data collected from real-world sources.

            The dataset was carefully **preprocessed, cleaned, and structured**
            to ensure correct time alignment and accurate price mapping.
            Detailed **Exploratory Data Analysis (EDA)** was performed to study
            long-term trends, price volatility, and overall market behavior.

            Multiple **time series forecasting models** were implemented and compared:
            - **ARIMA** for baseline statistical forecasting  
            - **SARIMA** to capture seasonal patterns  
            - **Prophet** for long-term trend estimation  
            - **LSTM (Long Short-Term Memory)** for deep learning-based prediction  

            Each model was evaluated on **historical data up to 31-12-2025**,
            followed by **30-day future forecasting beyond this date**.

            An **interactive Streamlit dashboard** was developed to visualize
            historical prices, model predictions, and future forecasts
            for multiple cryptocurrencies.
        """)

        st.markdown("### Forecasting Models Implemented")
        st.markdown("""
                        - ARIMA  
                        - SARIMA  
                        - Prophet  
                        - LSTM  
                    """)

    with col2:
        image_path = Path("assets/crypto_chart.png")
        if image_path.exists():
            st.image(str(image_path), width="stretch")
            


# ---------------- EXECUTIVE KPIs ----------------
elif page == "Executive KPIs":

    import pandas as pd
    import numpy as np

    st.markdown("# 📊 Executive Market Overview")

    # ---- Coin selector ----
    coins = [
        "BTC","ETH","BNB","ADA","SOL","XRP","DOGE","DOT",
        "AVAX","MATIC","LTC","BCH","TRX","LINK","UNI"
    ]

    coin = st.radio(
        "Select Cryptocurrency",
        coins,
        horizontal=True
    )

    # ---- Load data ----
    df = preprocess_crypto_data(load_crypto_data(coin, "2016-01-01"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # ---- KPI calculations ----
    current_price = float(latest["Close"])
    prev_price = float(prev["Close"])


    pct_change = (
    (current_price - prev_price) / prev_price * 100
    )

    direction = "🔺" if pct_change >= 0 else "🔻"

    high_30d = float(df.tail(30)["High"].max())
    low_30d = float(df.tail(30)["Low"].min())
    avg_volume = float(df.tail(30)["Volume"].mean())
    volatility = float(df["Close"].pct_change().std() * 100)

    # ---------------- RISK SCORE ----------------
    if volatility < 2:
        risk_label = "🟢 Low Risk"
    elif volatility < 4:
        risk_label = "🟡 Medium Risk"
    else:
        risk_label = "🔴 High Risk"
    
    # ---- KPI CARDS ----
    
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Current Price",
        f"${current_price:,.2f}",
        f"{pct_change:.2f}% {direction}"
    )

    col2.metric(
        "30D High",
        f"${high_30d:,.2f}"
    )

    col3.metric(
        "30D Low",
        f"${low_30d:,.2f}"
    )

    col4.metric(
        "Volatility",
        f"{volatility:.2f}%"
    )

    col5.metric(
        "Avg Volume (30D)",
        f"{avg_volume/1e9:.2f} B"
    )

    col6 = st.columns(1)[0]
    col6.metric(
    "Risk Score",
    risk_label
    )

    # ---- Trend mini chart ----
    st.markdown("### 📈 Price Trend (Last 90 Days)")

    trend_df = df.tail(90)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(trend_df["Date"], trend_df["Close"], color="#00c8ff", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)

    st.pyplot(fig)

    # ---- Insights ----
    st.info(
        f"""
        **Executive Insights for {coin}:**

        • Current market direction is **{'bullish' if pct_change >= 0 else 'bearish'}**  
        • Volatility of **{volatility:.2f}%** indicates {'high' if volatility > 4 else 'moderate'} price movement  
        • Price is trading {'closer to 30-day high' if current_price > (high_30d + low_30d)/2 else 'closer to 30-day low'}  
        • This page provides a **high-level snapshot** before deep analysis
        """
    )


# ---------------- DATA OVERVIEW ----------------
elif page == "Data Overview":

    st.markdown("# 📊 Data Overview")
    
    coins = ["BTC","ETH","BNB","ADA","SOL","XRP","DOGE","DOT","AVAX","MATIC","LTC","BCH","TRX","LINK","UNI"]

    render_coin = st.radio(
        "Select Cryptocurrency",
        coins,
        horizontal=True
    )

    df = preprocess_crypto_data(
        load_crypto_data(render_coin, "2016-01-01")
    )

    st.markdown(f"## 📈 {render_coin} Historical Prices")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["Close"], linewidth=1.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    st.info(
        "**Data Overview Insights:**\n\n"
        "• This chart shows the **historical closing prices** of the selected cryptocurrency.\n"
        "• It helps identify **long-term trends, market cycles, and major price movements**.\n"
        "• Use this overview as a foundation before exploring **EDA and forecasting models**.\n"
        "• Sudden spikes or drops often correspond to **market news, adoption events, or macroeconomic factors**."
    )
    st.pyplot(fig)


# ---------------- EDA ----------------
elif page == "Exploratory Data Analysis (EDA)":

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    st.markdown("# 📊 Exploratory Data Analysis (EDA)")

    # ---------------- LOAD BTC DATA (EDA ONLY) ----------------
    coins = [
    "BTC","ETH","BNB","ADA","SOL","XRP","DOGE","DOT",
    "AVAX","MATIC","LTC","BCH","TRX","LINK","UNI"
    ]

    st.markdown("### 🪙 Select Cryptocurrency (EDA)")

    coin = st.radio(
        label="",
        options=coins,
        horizontal=True,   # shows coins in rows
        index=0
    )


    DATA_PATH = f"data/eda/{coin}.csv"
    df = pd.read_csv(DATA_PATH)


    # ---------------- CLEAN & VALIDATE ----------------
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df = df.sort_index()

    # ---------------- TABS ----------------
    tabs = st.tabs([
        "Closing Price Distribution",
        "Closing Price Trend",
        "Volume Trend",
        "Candlestick",
        "Correlation Heatmap",
        "Moving Averages",
        "Volatility Analysis",
        "Monthly Boxplot"
    ])

    # ---------------- 1️⃣ Distribution ----------------
    with tabs[0]:
        st.subheader("Closing Price Distribution")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df["Close"], bins=40, color="gold", edgecolor="black")
        ax.set_title("Closing Price Distribution")
        ax.set_xlabel("Closing Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        st.info("""
        This histogram shows the **distribution of closing prices** over time.

        It helps us understand:
        - Most frequent price ranges  
        - Whether prices are **skewed** or normally distributed  
        - Presence of **extreme price values (outliers)**

        Such distribution insight is useful before applying forecasting models.
        """)


    # ---------------- 2️⃣ Trend ----------------
    with tabs[1]:
        st.subheader("Closing Price Trend")
        
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            df.index,
            df["Close"],
            color="gold",
            linewidth=2,
            label="Closing Price"   
        )
        ax.set_title("Closing Price Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()                 
        ax.grid(True)

        st.pyplot(fig)
        
        st.info("""
        This line chart represents the **long-term price movement** of the cryptocurrency.

        From this trend, we can observe:
        - Overall **market growth or decline**
        - Major **bull and bear cycles**
        - Long-term momentum patterns

        Trend analysis is essential for time-series forecasting.
        """)


    from matplotlib.ticker import FuncFormatter

    # ---------------- 3️⃣ Volume ----------------
    with tabs[2]:
        st.subheader("Trading Volume Trend")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Volume"], color="orange")
        ax.set_title("Volume Trend Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume (Billions)")
        ax.grid(True)

        # ✅ Format Y-axis to billions
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x/1e9:.1f}B")
        )

        st.pyplot(fig)

        st.info("""
        This plot shows how **trading volume changes over time**.

        Key insights:
        - High volume often indicates **strong market interest**
        - Sudden spikes usually align with **major price movements**
        - Low volume may suggest **market consolidation**

        Volume helps validate price trends.
        """)


    # ---------------- 4️⃣ Candlestick ----------------
    with tabs[3]:
        st.subheader("Candlestick Chart")

        candle_df = df[["Open", "High", "Low", "Close", "Volume"]]

        fig, axes = mpf.plot(
            candle_df,
            type="candle",
            style="yahoo",
            volume=True,
            mav=(20, 50),
            figsize=(12, 6),
            returnfig=True
            
        )
        
        # ✅ FIX PRICE AXIS (TOP CHART)
        price_ax = axes[0]
        price_ax.set_ylim(bottom=0)

        st.pyplot(fig)
        
        st.info("""
        The candlestick chart provides a **detailed view of daily price action**.

        Each candle shows:
        - Opening price  
        - Closing price  
        - High and Low prices  

        It is widely used in technical analysis to identify:
        - Market sentiment  
        - Price reversals  
        - Support and resistance levels
        """)



    # ---------------- 5️⃣ Correlation ----------------
    with tabs[4]:
        st.subheader("Correlation Heatmap")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        
        st.info("""
        This heatmap displays the **correlation between Open, High, Low, Close, and Volume**.

        Interpretation:
        - Values close to **1** indicate strong positive correlation  
        - Values close to **-1** indicate negative correlation  

        High correlation among OHLC prices confirms **data consistency**.
        """)


    # ---------------- 6️⃣ Moving Averages ----------------
    with tabs[5]:
        st.subheader("Moving Averages")
        
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Close"], label="Close", alpha=0.6)
        ax.plot(df.index, df["SMA20"], label="20-Day SMA")
        ax.plot(df.index, df["SMA50"], label="50-Day SMA")
        ax.legend()
        ax.set_title("Moving Averages")
        st.pyplot(fig)
        st.info("""
        This chart shows **price volatility along with moving averages**.

        Insights:
        - Moving averages smooth price fluctuations  
        - Crossovers can indicate **trend changes**
        - Volatility reflects **market risk and uncertainty**

        These indicators are widely used in trading strategies.
        """)

    # ---------------- 7️⃣ Volatility Analysis ----------------
    with tabs[6]:
        st.subheader("Volatility Analysis")

        # Daily returns
        df["Daily_Return"] = df["Close"].pct_change()

        # Rolling volatility (30-day)
        df["Volatility_30"] = df["Daily_Return"].rolling(window=30).std()

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            df.index,
            df["Volatility_30"],
            color="purple",
            label="30-Day Rolling Volatility"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.set_title("30-Day Rolling Volatility of Returns")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)

        st.info(
            "Volatility represents market risk. "
            "Higher volatility indicates larger price fluctuations."
        )

    # ---------------- 8️⃣ Monthly Boxplot ----------------
    with tabs[7]:
        st.subheader("Monthly Returns Distribution")

        # Daily returns
        df["Return"] = df["Close"].pct_change()

        # Month names
        df["Month"] = df.index.month_name()

        month_order = [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ]

        fig, ax = plt.subplots(figsize=(12, 5))

        sns.boxplot(
            x="Month",
            y="Return",
            data=df,
            order=month_order,
            ax=ax
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Daily Returns")
        ax.set_title("Monthly Distribution of Daily Returns")
        ax.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)

        st.info(
            "This boxplot shows how daily returns vary across months, "
            "highlighting seasonal effects and extreme market movements."
        )



# ---------------- FORECASTING MODELS ----------------
elif page == "Forecasting Models":

    st.markdown("# Forecasting Models")

    # ---------------- COIN SELECTOR (RADIO BUTTONS) ----------------
    coins = [
        "BTC","ETH","BNB","ADA","SOL","XRP","DOGE","DOT",
        "AVAX","MATIC","LTC","BCH","TRX","LINK","UNI"
    ]

    st.markdown("### 🪙 Select Cryptocurrency")

    coin = st.radio(
        label="",
        options=coins,
        horizontal=True
    )

    # ---------------- LOAD DATA ----------------
    df = preprocess_crypto_data(
        load_crypto_data(coin, "2016-01-01")
    )


    # Helper function for consistent axis formatting
    def format_axes(fig, ax, title):
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.tick_params(axis="x", rotation=45)
        fig.autofmt_xdate()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
    

    # ==================================================
    # 🔮 PROPHET MODEL
    # ==================================================
    st.subheader("🔮 Prophet Model")

    col1, col2 = st.columns(2)

    # Historical Fit
    with col1:
        hist = prophet_forecast(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Close"], label="Actual", alpha=0.6)
        ax.plot(hist["Date"], hist["Prophet_Forecast"], label="Prophet")
        format_axes(fig, ax, "Historical Fit")
        st.pyplot(fig)

    # Future Forecast
    with col2:
        fut = prophet_forecast_30_days(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fut["Date"], fut["Forecast"], label="30-Day Forecast")
        format_axes(fig, ax, "Future Forecast")
        st.pyplot(fig)

    st.info(
        "**Prophet Model Insights:**\n\n"
        "• Prophet is designed for **long-term trend forecasting** in time-series data.\n"
        "• It automatically handles **trend changes, seasonality, and missing values**.\n"
        "• Best suited for understanding **overall market direction** rather than short-term fluctuations.\n"
        "• The left chart shows how Prophet fits historical prices, while the right chart shows a **30-day future forecast**."
        )    
    
    st.divider()

    # ==================================================
    # 📈 ARIMA MODEL
    # ==================================================
    st.subheader("📈 ARIMA Model")

    col1, col2 = st.columns(2)

    with col1:
        hist = arima_forecast(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Close"], label="Actual", alpha=0.6)
        ax.plot(hist["Date"], hist["ARIMA_Fitted"], label="ARIMA")
        format_axes(fig, ax, "Historical Fit")
        st.pyplot(fig)

    with col2:
        fut = arima_forecast_30_days(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fut["Date"], fut["Forecast"], label="30-Day Forecast")
        format_axes(fig, ax, "Future Forecast")
        st.pyplot(fig)

    st.info(
            "**ARIMA Model Insights:**\n\n"
            "• ARIMA is a **statistical forecasting model** based on past values and errors.\n"
            "• It works best for **short-term predictions** when the data is relatively stable.\n"
            "• Does not explicitly model seasonality.\n"
            "• The historical fit shows how well ARIMA captures past price movements, followed by a **30-day forecast**."
    )

    st.divider()

    # ==================================================
    # 📊 SARIMA MODEL
    # ==================================================
    st.subheader("📊 SARIMA Model")

    col1, col2 = st.columns(2)

    with col1:
        hist = sarima_forecast(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Close"], label="Actual", alpha=0.6)
        ax.plot(hist["Date"], hist["SARIMA_Fitted"], label="SARIMA")
        format_axes(fig, ax, "Historical Fit")
        st.pyplot(fig)

    with col2:
        fut = sarima_forecast_30_days(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fut["Date"], fut["Forecast"], label="30-Day Forecast")
        format_axes(fig, ax, "Future Forecast")
        st.pyplot(fig)

    st.info(
            "**SARIMA Model Insights:**\n\n"
            "• SARIMA extends ARIMA by incorporating **seasonal patterns**.\n"
            "• It is effective when cryptocurrency prices show **repeating cycles or periodic behavior**.\n"
            "• Provides improved accuracy over ARIMA when seasonality exists.\n"
            "• The forecast section visualizes expected prices for the **next 30 days**."
    )

    st.divider()

    # ==================================================
    # 🤖 LSTM MODEL
    # ==================================================
    st.subheader("🤖 LSTM Model")

    col1, col2 = st.columns(2)

    with col1:
        hist = lstm_forecast(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Close"], label="Actual", alpha=0.6)
        ax.plot(hist["Date"], hist["LSTM_Prediction"], label="LSTM")
        format_axes(fig, ax, "Historical Fit")
        st.pyplot(fig)

    with col2:
        fut = lstm_forecast_30_days(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fut["Date"], fut["Forecast"], label="30-Day Forecast")
        format_axes(fig, ax, "Future Forecast")
        st.pyplot(fig)

    st.info(
            "**Overall Model Comparison:**\n\n"
            "• **MAE (Mean Absolute Error):** Measures average prediction error.\n"
            "• **RMSE (Root Mean Squared Error):** Penalizes large prediction errors more heavily.\n\n"
            "**Interpretation:**\n"
            "• Lower MAE and RMSE indicate better model performance.\n"
            "• Statistical models (ARIMA, SARIMA) perform well for structured data.\n"
            "• Prophet excels at trend estimation.\n"
            "• LSTM captures complex market dynamics but requires more data."
    )


    
# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":

    from src.model_evaluation import evaluate_models
    import numpy as np

    st.markdown("# 📊 Model Comparison & Evaluation")

    st.markdown("""
    This section compares **ARIMA, SARIMA, Prophet, and LSTM** models
    using standard statistical error metrics.
    Lower values indicate **better performance**.
    """)

    st.divider()

    # ---------------- LOAD RESULTS ----------------
    results_df = evaluate_models()

    # ---------------- METRICS TABLE ----------------
    st.subheader("📋 Error Metrics Summary")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ================= MAE CHART =================
    st.subheader("📉 MAE Comparison (Lower is Better)")

    fig_mae, ax_mae = plt.subplots(figsize=(9, 4))

    bars_mae = ax_mae.bar(
        results_df["Model"],
        results_df["MAE"],
        color=["#9CA3AF", "#22C55E", "#3B82F6", "#EF4444"]
    )

    ax_mae.set_xlabel("Model")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("Mean Absolute Error by Model")
    ax_mae.grid(axis="y", linestyle="--", alpha=0.6)

    # Value labels
    for bar in bars_mae:
        height = bar.get_height()
        ax_mae.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    st.pyplot(fig_mae)

    st.divider()

    # ================= RMSE CHART =================
    st.subheader("📉 RMSE Comparison (Lower is Better)")

    fig_rmse, ax_rmse = plt.subplots(figsize=(9, 4))

    bars_rmse = ax_rmse.bar(
        results_df["Model"],
        results_df["RMSE"],
        color=["#9CA3AF", "#22C55E", "#3B82F6", "#EF4444"]
    )

    ax_rmse.set_xlabel("Model")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("Root Mean Squared Error by Model")
    ax_rmse.grid(axis="y", linestyle="--", alpha=0.6)

    # Value labels
    for bar in bars_rmse:
        height = bar.get_height()
        ax_rmse.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    st.pyplot(fig_rmse)

    st.divider()

    # ---------------- INSIGHTS ----------------
    st.subheader("🧠 Interpretation")

    st.info("""
    **Model Insights:**
    
    • **ARIMA** – Baseline statistical forecasting  
    • **SARIMA** – Improves ARIMA by modeling seasonality  
    • **Prophet** – Strong for long-term trend estimation  
    • **LSTM** – Best for capturing non-linear and short-term patterns  
    """)

