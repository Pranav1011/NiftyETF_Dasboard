import streamlit as st
import pandas as pd
import pickle
import altair as alt
from bokeh.plotting import figure
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Nifty ETF Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
        .header-bar {
            background-color: #0d6efd;
            padding: 15px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            border-radius: 10px;
        }
        .description {
            font-size: 14px;
            color: #A0A0A0;
            margin-bottom: 10px;
        }
    </style>
    <div class="header-bar">
        📈 Nifty ETF Price Prediction & Market Analysis
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv("./data/final/final_dataset.csv", parse_dates=["Date"])

df = load_data()

forecast_df = pd.read_csv("./data/final/final_dataset_with_forecast.csv", parse_dates=["Date"])
forecast_df = forecast_df.rename(columns={"ds": "Date", "yhat": "Predicted_Volatility", "yhat_lower": "Lower_Bound", "yhat_upper": "Upper_Bound"})

# 📊 **Forecasted Volatility vs Actual Volatility**
st.subheader("📊 Forecasted Volatility vs Actual Volatility")
st.markdown("<p class='description'>This graph compares the actual market volatility (blue) with the forecasted volatility (red, dashed) using the Prophet model. The gray shaded region represents the uncertainty interval.</p>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(forecast_df["Date"], forecast_df["Volatility_20"], label="Actual Volatility", color="blue", linewidth=1.5)
ax.plot(forecast_df["Date"], forecast_df["Predicted_Volatility"], label="Predicted Volatility", linestyle="dashed", color="red", linewidth=1.5)
ax.fill_between(forecast_df["Date"], forecast_df["Lower_Bound"], forecast_df["Upper_Bound"], color="gray", alpha=0.3, label="Uncertainty Interval")
ax.set_title("Market Volatility Forecast (Prophet)", fontsize=14)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Volatility", fontsize=12)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

# 📊 **Moving Averages (50 & 200 Days)**
st.subheader("📊 Moving Averages (50 & 200 Days)")
st.markdown("<p class='description'>This chart tracks the Nifty ETF closing price along with its 50-day and 200-day moving averages. These indicators help in identifying market trends, potential reversals, and support/resistance levels.</p>", unsafe_allow_html=True)

fig2 = figure(x_axis_type="datetime", title="Moving Averages (50 & 200 Days)", width=700, height=350)
fig2.line(df["Date"], df["Close"], color="blue", legend_label="Close Price", alpha=0.6)
fig2.line(df["Date"], df["50_MA"], color="orange", legend_label="50-Day MA", line_width=2)
fig2.line(df["Date"], df["200_MA"], color="red", legend_label="200-Day MA", line_width=2)
fig2.legend.location = "top_left"
st.bokeh_chart(fig2, use_container_width=True)

# 📊 **RSI & Volatility**
st.subheader("📊 RSI & Volatility")
st.markdown("<p class='description'>The RSI (purple) measures momentum and signals overbought (>70) or oversold (<30) conditions. The 20-day rolling volatility (green) tracks price fluctuations to assess market risk levels.</p>", unsafe_allow_html=True)

chart3 = alt.Chart(df).mark_line().encode(
    x=alt.X("year(Date):T", title="Year"),  
    y=alt.Y("RSI_14", title="RSI (14)", axis=alt.Axis(titleColor="purple")),
    color=alt.value("purple")
) + alt.Chart(df).mark_line().encode(
    x=alt.X("year(Date):T", title="Year"),
    y=alt.Y("Volatility_20", title="Volatility (20-Day)", axis=alt.Axis(titleColor="green")),
    color=alt.value("green")
).properties(title="RSI & Volatility Over Time", width=700, height=350)

st.altair_chart(chart3, use_container_width=True)

# 📊 **Bollinger Bands**
st.subheader("📊 Bollinger Bands")
st.markdown("<p class='description'>Bollinger Bands (upper in red, lower in green) highlight periods of high and low volatility. Prices near the upper band indicate overbought conditions, while those near the lower band suggest oversold conditions.</p>", unsafe_allow_html=True)

fig4 = figure(x_axis_type="datetime", title="Bollinger Bands", width=700, height=350)
fig4.line(df["Date"], df["Close"], color="blue", legend_label="Close Price", alpha=0.5)
fig4.line(df["Date"], df["Bollinger_Upper"], color="red", legend_label="Upper Band", line_dash="dashed")
fig4.line(df["Date"], df["Bollinger_Lower"], color="green", legend_label="Lower Band", line_dash="dashed")
fig4.legend.location = "top_left"
st.bokeh_chart(fig4, use_container_width=True)

# 📊 **MACD & Signal Line**
st.subheader("📊 MACD & Signal Line")
st.markdown("<p class='description'>The MACD (blue) helps detect changes in momentum, while the Signal Line (red, dashed) assists in identifying trend reversals. Crossovers between these lines signal buy or sell opportunities.</p>", unsafe_allow_html=True)

chart5 = alt.Chart(df).mark_line().encode(
    x=alt.X("year(Date):T", title="Year"),  
    y="MACD",
    color=alt.value("blue")
) + alt.Chart(df).mark_line(strokeDash=[5,5]).encode(
    x=alt.X("year(Date):T", title="Year"),  
    y="MACD_Signal",
    color=alt.value("red")
).properties(title="MACD & Signal Line Over Time", width=700, height=350)

st.altair_chart(chart5, use_container_width=True)

# 📊 **Trading Volume Over Time**
st.subheader("📊 Trading Volume Over Time")
st.markdown("<p class='description'>This bar chart illustrates trading volume trends, helping to confirm price movements. Sudden spikes may indicate major market events.</p>", unsafe_allow_html=True)

fig6 = figure(x_axis_type="datetime", title="Trading Volume Over Time", width=700, height=350)
fig6.vbar(x=df["Date"], top=df["Volume"], width=0.5, color="purple", alpha=0.7)
st.bokeh_chart(fig6, use_container_width=True)

st.markdown(
    """
    <div style="text-align:center; font-size:14px; margin-top:20px;">
        🔥 Developed by Sai Pranav | Powered by Streamlit 🚀
    </div>
    """,
    unsafe_allow_html=True
)