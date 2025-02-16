import pandas as pd
import numpy as np
import logging

datasets = {
    "nifty_etf": "data/processed/cleaned_nifty_etf.csv",
    "nifty_index": "data/processed/cleaned_nifty_index.csv",
    "sensex": "data/processed/cleaned_sensex.csv",
    "sp500": "data/processed/cleaned_sp500.csv",
    "nasdaq": "data/processed/cleaned_nasdaq.csv",
    "fx_rates": "data/processed/cleaned_fx_rates.csv"
}

dfs = {name: pd.read_csv(path) for name, path in datasets.items()}

for name, df in dfs.items():
    df["Date"] = pd.to_datetime(df["Date"])

final_df = dfs["nifty_etf"]  
for name, df in dfs.items():
    if name not in ["nifty_etf", "fx_rates"]:  
        final_df = final_df.merge(df, on="Date", how="inner", suffixes=("", f"_{name}"))

if "fx_rates" in dfs:
    fx_df = dfs["fx_rates"].rename(columns={"Close": "USD_INR"})
    final_df = final_df.merge(fx_df[["Date", "USD_INR"]], on="Date", how="left")
    final_df["USD_INR"] = final_df["USD_INR"].ffill()
logging.info(f"✅ Merged dataset with FX rates - Shape: {final_df.shape}")

if "Close_sp500" in final_df.columns:
    final_df["Close_SP500_INR"] = final_df["Close_sp500"] * final_df["USD_INR"]
    final_df["High_SP500_INR"] = final_df["High_sp500"] * final_df["USD_INR"]
    final_df["Low_SP500_INR"] = final_df["Low_sp500"] * final_df["USD_INR"]
    final_df["Open_SP500_INR"] = final_df["Open_sp500"] * final_df["USD_INR"]

if "Close_nasdaq" in final_df.columns:
    final_df["Close_NASDAQ_INR"] = final_df["Close_nasdaq"] * final_df["USD_INR"]
    final_df["High_NASDAQ_INR"] = final_df["High_nasdaq"] * final_df["USD_INR"]
    final_df["Low_NASDAQ_INR"] = final_df["Low_nasdaq"] * final_df["USD_INR"]
    final_df["Open_NASDAQ_INR"] = final_df["Open_nasdaq"] * final_df["USD_INR"]

# 🚀 **Rolling Correlations**
if "Close_SP500_INR" in final_df.columns:
    final_df["SP500_Correlation"] = final_df["Close"].rolling(30).corr(final_df["Close_SP500_INR"])

if "Close_sensex" in final_df.columns:
    final_df["Sensex_Correlation"] = final_df["Close"].rolling(30).corr(final_df["Close_sensex"])

# 🚀 **Feature Engineering: Moving Averages**
final_df["50_MA"] = final_df["Close"].rolling(window=50, min_periods=1).mean()
final_df["200_MA"] = final_df["Close"].rolling(window=200, min_periods=1).mean()

# 🚀 **Feature Engineering: Daily Returns**
final_df["1_day_return"] = final_df["Close"].pct_change()
final_df["5_day_return"] = final_df["Close"].pct_change(periods=5)
final_df["10_day_return"] = final_df["Close"].pct_change(periods=10)

# 🚀 **Feature Engineering: Bollinger Bands**
rolling_mean = final_df["Close"].rolling(window=20, min_periods=1).mean()
rolling_std = final_df["Close"].rolling(window=20, min_periods=1).std()
final_df["Bollinger_Upper"] = rolling_mean + (rolling_std * 2)
final_df["Bollinger_Lower"] = rolling_mean - (rolling_std * 2)

# 🚀 **Feature Engineering: Volatility & RSI**
final_df["Volatility_20"] = final_df["Close"].rolling(window=20, min_periods=1).std()

# RSI Calculation
delta = final_df["Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
rs = avg_gain / avg_loss
final_df["RSI_14"] = 100 - (100 / (1 + rs))

# 🚀 **Feature Engineering: Exponential Moving Averages**
final_df["12_EMA"] = final_df["Close"].ewm(span=12, adjust=False).mean()
final_df["26_EMA"] = final_df["Close"].ewm(span=26, adjust=False).mean()
final_df["MACD"] = final_df["12_EMA"] - final_df["26_EMA"]
final_df["MACD_Signal"] = final_df["MACD"].ewm(span=9, adjust=False).mean()

# 🚀 **Feature Engineering: On-Balance Volume (OBV)**
final_df["OBV"] = (np.sign(final_df["Close"].diff()) * final_df["Volume"]).fillna(0).cumsum()

# 🚀 **Handling Missing Values (No Row Drop)**
technical_indicators = [
    "50_MA", "200_MA", "Bollinger_Upper", "Bollinger_Lower",
    "1_day_return", "5_day_return", "10_day_return", "Volatility_20", "RSI_14"
]

# Apply Backward Fill (bfill)
final_df[technical_indicators] = final_df[technical_indicators].bfill()

# Apply Forward Fill (ffill) as a secondary method
final_df[technical_indicators] = final_df[technical_indicators].ffill()

final_df.to_csv("data/final/final_dataset.csv", index=False)
logging.info("✅ Feature Engineering Completed Successfully & Missing Values Handled")