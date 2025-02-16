import pandas as pd
import numpy as np
import pickle
import logging
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train_ml.log"), logging.StreamHandler()],
)

df = pd.read_csv("data/final/final_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

prophet_df = df[["Date", "Volatility_20"]].dropna()
prophet_df = prophet_df.rename(columns={"Date": "ds", "Volatility_20": "y"})

logging.info(f"✅ Prophet Dataset Ready - Shape: {prophet_df.shape}")

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
forecast = forecast.rename(columns={"ds": "Date", "yhat": "Predicted_Volatility"})

df = df.merge(forecast, on="Date", how="left")

actual = df["Volatility_20"].dropna()
predicted = df["Predicted_Volatility"].dropna()

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

logging.info(f"\n📊 Prophet Model Evaluation:")
logging.info(f"   - MAE: {mae:.4f}")
logging.info(f"   - RMSE: {rmse:.4f}")
logging.info(f"   - R² Score: {r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Volatility_20"], label="Actual Volatility", color="blue", alpha=0.6)
plt.plot(df["Date"], df["Predicted_Volatility"], label="Predicted Volatility", color="red", linestyle="dashed")
plt.fill_between(df["Date"], df["yhat_lower"], df["yhat_upper"], color="gray", alpha=0.3, label="Uncertainty Interval")
plt.title("Market Volatility Forecast (Prophet)")
plt.legend()
plt.tight_layout()
plt.savefig("data/images/volatility_forecast.png")
plt.show()

with open("./models/prophet_volatility_model.pkl", "wb") as f:
    pickle.dump(model, f)

df.to_csv("data/final/final_dataset_with_forecast.csv", index=False)
logging.info("✅ Forecast Data Saved Successfully!")