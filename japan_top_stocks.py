import streamlit as st
import yfinance as yf
import datetime as d
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.title("📈 Japan Stock Forecasting App (ARIMA Model)")

# -----------------------------
# Date range
# -----------------------------
s = d.datetime(2025, 1, 1)
e = d.datetime(2025, 11, 5)

# -----------------------------
# Function to download stock data
# -----------------------------
def load_stock(ticker, name):
    df = yf.download(ticker, start=s, end=e)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Stock"] = name
    return df

with st.spinner("Downloading stock data..."):
    sony = load_stock("6758.T", "Sony")
    hitachi = load_stock("6501.T", "Hitachi")
    softbank = load_stock("9984.T", "SoftBank_Group_Corp")
    tel = load_stock("8035.T", "Tokyo_Electron_Ltd")
    toyota = load_stock("7203.T", "Toyota_Motor_Corp")
    advan = load_stock("6857.T", "Advantest_Group")
    smfgi = load_stock("8316.T", "Sumitomo_Mitsui_Financial_Group_Inc")
    frcl = load_stock("9983.T", "Fast_Retailing_Co_Ltd")
    mfgi = load_stock("8306.T", "Mitsubishi_UFJ_Financial_Group_Inc")
    ncl = load_stock("7974.T", "Nintendo_Co_Ltd")

# Combine all
df = pd.concat(
    [sony, hitachi, softbank, tel, toyota, advan, smfgi, frcl, mfgi, ncl],
    axis=0
)
df = df.set_index("Date")

# -----------------------------
# Select stock
# -----------------------------
stock_list = df["Stock"].unique()
choice = st.selectbox("Select a Stock", stock_list)

st.subheader(f"📌 Selected Stock: **{choice}**")

st1 = df[df.Stock == choice][["Close"]]
st1["Returns"] = st1["Close"].pct_change()
st1.dropna(inplace=True)

# -----------------------------
# ADF Test Function
# -----------------------------
def check_stationarity(series, title="Series"):
    result = adfuller(series.dropna())
    st.write(f"### ADF Test: {title}")
    st.write(f"ADF Statistic: {result[0]}")
    st.write(f"P-value: {result[1]}")
    if result[1] < 0.05:
        st.success("✔ The series is stationary.")
    else:
        st.warning("✖ The series is NOT stationary.")

# Stationarity checks
check_stationarity(st1["Close"], "Close Price")
st1["Close_Diff"] = st1["Close"].diff()
check_stationarity(st1["Close_Diff"], "Differenced Close Price")

# -----------------------------
# ARIMA Modeling
# -----------------------------
st.write("## 📉 ARIMA Model Forecast")

model = ARIMA(st1["Close"], order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=10)
dates = pd.date_range(start=st1.index[-1], periods=11, freq="B")[1:]

# -----------------------------
# Plot Forecast
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(st1["Close"], label="Actual Prices")
ax.plot(dates, forecast, label="Predicted Prices", linestyle="dashed")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()

st.pyplot(fig)

st.write("### 🔮 Forecasted Values")
st.dataframe(pd.DataFrame({"Date": dates, "Forecast": forecast.values}))
