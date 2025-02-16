# 📊 Nifty ETF Price Prediction & Market Analysis


## 🚀 Project Overview  
This project provides an interactive **Streamlit dashboard** for analyzing **Nifty ETF market trends, price predictions, and volatility forecasts** using **Machine Learning & Time Series Forecasting**. It includes key financial indicators like **Moving Averages, Bollinger Bands, RSI, MACD, and Trading Volume**, with **Prophet-based Volatility Forecasting**.

## 📌 Features  
✅ **Market Volatility Forecast**: Prophet model predicts future volatility with uncertainty intervals.  
✅ **Moving Averages (50 & 200 Days)**: Helps identify market trends.  
✅ **RSI & Volatility**: Tracks momentum and risk assessment.  
✅ **Bollinger Bands**: Highlights overbought and oversold market conditions.  
✅ **MACD & Signal Line**: Helps in momentum-based trading strategies.  
✅ **Trading Volume**: Identifies periods of high activity.  

## 🛠️ Tech Stack  
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Altair, Bokeh, Plotly, FB Prophet)  
- **Machine Learning** (Linear Regression for price prediction, Prophet for volatility forecasting)  
- **Streamlit** (Interactive visualization)  
- **Bokeh & Plotly** (Advanced graph rendering)  

## ⚡ Installation  

1️⃣ **Clone the Repository**  

git clone https://github.com/your-username/nifty-etf-dashboard.git
cd nifty-etf-dashboard

2️⃣ **Create & Activate Virtual Environment**

python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Windows

3️⃣ **Install Dependencies**

pip install -r requirements.txt

4️⃣ **Run the Feature Engineering & Model Training Scripts**

python scripts/feature_engineering.py
python scripts/train_ml.py

5️⃣ **Launch the Streamlit Dashboard**

streamlit run dashboards/dashboard.py

📊 **Dashboard Overview**

1️⃣ Market Volatility Forecast

Predicts future volatility using Facebook Prophet, with an uncertainty interval (gray).
🔹 Blue Line → Actual Volatility
🔹 Red Dashed Line → Predicted Volatility
🔹 Gray Region → Uncertainty Interval

2️⃣ Moving Averages (50 & 200 Days)

Tracks short-term & long-term trends for better trend analysis.

3️⃣ RSI & Volatility

🔹 RSI (Relative Strength Index) indicates overbought (>70) or oversold (<30) market conditions.
🔹 Volatility (Green) measures market risk.

4️⃣ Bollinger Bands

Helps identify high-volatility and price reversal zones.
🔹 Upper Band (Red) → Overbought
🔹 Lower Band (Green) → Oversold

5️⃣ MACD & Signal Line

Momentum-based buy/sell indicator using trend shifts.

6️⃣ Trading Volume

Tracks trading activity & price confirmation using volume spikes.

📌 Future Enhancements

🔹 Add Deep Learning LSTM Model for Forecasting
🔹 Incorporate Sentiment Analysis using News Data
🔹 Develop a Real-Time API for Live Market Data

🤝 Contributing

Contributions are welcome! Please fork this repo, create a branch, and submit a PR.

📜 License

This project is licensed under MIT License.