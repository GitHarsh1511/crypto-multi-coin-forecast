import streamlit as st
import matplotlib.pyplot as plt

from src.model_evaluation import evaluate_models

# ----------------------------------
# PAGE TITLE
# ----------------------------------
st.title("📊 Model Comparison & Evaluation")

st.markdown("""
This page compares **ARIMA, SARIMA, Prophet, and LSTM** models using standard error metrics.
""")

# ----------------------------------
# LOAD RESULTS
# ----------------------------------
results_df = evaluate_models()

# ----------------------------------
# DISPLAY TABLE
# ----------------------------------
st.subheader("📋 Error Metrics (Lower is Better)")
st.dataframe(results_df, use_container_width=True)

# ----------------------------------
# BAR CHART (RMSE)
# ----------------------------------
st.subheader("📉 RMSE Comparison")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    results_df["Model"],
    results_df["RMSE"],
    color=["gray", "green", "blue", "red"]
)

ax.set_ylabel("RMSE")
ax.set_xlabel("Model")
ax.grid(axis="y", alpha=0.4)

st.pyplot(fig)

# ----------------------------------
# INSIGHTS
# ----------------------------------
st.info("""
**Interpretation:**
- **ARIMA**: Baseline statistical model  
- **SARIMA**: Improves ARIMA using seasonality  
- **Prophet**: Best for long-term trend analysis  
- **LSTM**: Best for short-term accuracy and non-linear patterns  
""")
