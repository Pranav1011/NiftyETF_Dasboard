import yfinance as yf
import pandas as pd

tickers = {
    "nasdaq": "^IXIC",
    "nifty_etf": "NIFTYBEES.NS",
    "nifty_index": "^NSEI",
    "sensex": "^BSESN",
    "sp500": "^GSPC",
    "usd_inr": "USDINR=X"
}

output_paths = {key: f"data/raw/{key}.csv" for key in tickers.keys()}

standard_columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

start_date = "2009-01-01"

for name, ticker in tickers.items():
    try:
        print(f"Fetching data for {name} ({ticker}) starting from {start_date}...")
        data = yf.Ticker(ticker)
        df = data.history(start=start_date) 

        df.reset_index(inplace=True)

        df.rename(columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }, inplace=True)

        df = df[standard_columns]

        df.to_csv(output_paths[name], index=False)
        print(f"Saved data for {name} to {output_paths[name]}")

    except Exception as e:
        print(f"Error fetching data for {name}: {e}")