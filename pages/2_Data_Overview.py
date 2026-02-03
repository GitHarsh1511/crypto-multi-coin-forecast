# import streamlit as st
# import matplotlib.pyplot as plt

# from src.data_loader import load_crypto_data
# from src.preprocessing import preprocess_crypto_data

# # --------------------------------------------------
# # HIDE DEFAULT STREAMLIT SIDEBAR NAV (IMPORTANT)
# # --------------------------------------------------
# st.markdown("""
# <style>
# [data-testid="stSidebarNav"] {
#     display: none;
# }
# </style>
# """, unsafe_allow_html=True)

# # --------------------------------------------------
# # PAGE TITLE
# # --------------------------------------------------
# st.title("📊 Cryptocurrency Historical Prices")

# # --------------------------------------------------
# # COIN SELECTOR (MAIN PAGE — NOT SIDEBAR)
# # --------------------------------------------------
# coins = [
#     "BTC","ETH","BNB","ADA","SOL","XRP","DOGE",
#     "DOT","AVAX","MATIC","LTC","BCH","TRX","LINK","UNI"
# ]

# coin = st.selectbox("Select Cryptocurrency", coins)

# # --------------------------------------------------
# # LOAD & PREPROCESS DATA
# # --------------------------------------------------
# df = preprocess_crypto_data(
#     load_crypto_data(coin, start_date="2016-01-01")
# )

# # --------------------------------------------------
# # PLOT
# # --------------------------------------------------
# st.subheader(f"{coin} Historical Prices")

# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(df["Date"], df["Close"], color="#4cc9f0")
# ax.set_xlabel("Date")
# ax.set_ylabel("Price")
# ax.grid(True)

# st.pyplot(fig)
