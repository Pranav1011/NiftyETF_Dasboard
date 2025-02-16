import pandas as pd
from functools import reduce
import logging
import pytz

class FinancialDataCleaner:
    
    def __init__(self, config):
        self.config = config
        self.cleaned_dfs = {}
        self.cutoff_date = pd.Timestamp('2025-02-07', tz='UTC')
        self._validate_config()
        
    def _validate_config(self):
        required_keys = ['datasets', 'base_currency', 'time_zones']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _load_exchange_rates(self):
        try:
            fx_path = self.config['datasets']['fx_rates']
            fx_df = pd.read_csv(fx_path)
            
            if 'Date' not in fx_df.columns:
                raise KeyError("FX rates file missing 'Date' column")
                
            self._process_dataset('fx_rates', fx_path)
            
            fx_df = self.cleaned_dfs['fx_rates']
            fx_df.ffill(inplace=True)
            
            logging.info(f"✅ Loaded FX rates data | Shape: {fx_df.shape}")
            return fx_df[['Close']].rename(columns={'Close': 'fx_rate'})
            
        except Exception as e:
            logging.error(f"❌ Failed to load FX rates: {str(e)}")
            raise
            
    def _process_dataset(self, name, path):
        try:
            df = pd.read_csv(path)
            
            if 'Date' not in df.columns:
                raise KeyError(f"Dataset {name} missing 'Date' column")
                
            logging.info(f"📂 Processing {name} | Initial shape: {df.shape}")

            if name == 'fx_rates':
                if 'Volume' in df.columns:
                    df.drop(columns=['Volume'], inplace=True)

                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df['Date'] = df['Date'].dt.tz_convert('UTC').dt.normalize()
                df = df[df['Date'] <= self.cutoff_date]
                
                df = df.groupby('Date').last().sort_index()
                df.index = df.index.tz_localize(None).tz_localize('UTC')
                
                self.cleaned_dfs[name] = df
                logging.info(f"✅ Cleaned {name} | Final shape: {df.shape}")
                return 

            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            
            if name in self.config['time_zones']:
                market_tz = self.config['time_zones'][name]
                df['Date'] = df['Date'].dt.tz_convert(market_tz).dt.tz_convert('UTC')
                
            df['Date'] = df['Date'].dt.normalize()
            df = df[df['Date'] <= self.cutoff_date]
            
            df = df.groupby('Date').last().sort_index()
            df = df[~df.index.duplicated()]
            df.index = df.index.tz_localize(None).tz_localize('UTC')
            
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(0).astype(int)
                
            if name in self.config['convert_to_base']:
                fx_rates = self._load_exchange_rates()
                df = df.merge(fx_rates, left_index=True, right_index=True, how='left')
                df['fx_rate'] = df['fx_rate'].ffill()
                
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col] * df['fx_rate']
                
                df.drop(columns=['fx_rate'], inplace=True)
                logging.info(f"💱 Converted {name} to {self.config['base_currency']}")
            
            self.cleaned_dfs[name] = df
            logging.info(f"✅ Cleaned {name} | Final shape: {df.shape}")
            
        except Exception as e:
            logging.error(f"❌ Failed processing {name}: {str(e)}")
            raise

    def _align_datasets(self):
        """Find common dates across all processed datasets"""
        try:
            date_sets = [set(df.index) for df in self.cleaned_dfs.values()]
            self.common_dates = sorted(list(reduce(lambda x,y: x & y, date_sets)))
            
            if not self.common_dates:
                raise ValueError("No common dates found across datasets")
                
            logging.info(f"📆 Found {len(self.common_dates)} common trading days")
            
        except Exception as e:
            logging.error("Alignment failed: " + str(e))
            raise

    def clean_all(self):
        """Main cleaning pipeline"""
        try:
            for name, path in self.config['datasets'].items():
                self._process_dataset(name, path)
            
            self._align_datasets()
            
            for name, df in self.cleaned_dfs.items():
                aligned_df = df.reindex(self.common_dates).ffill()
                save_path = f"data/processed/cleaned_{name}.csv"
                aligned_df.to_csv(save_path)
                logging.info(f"💾 Saved {name} to {save_path}")
            
            return True
            
        except Exception as e:
            logging.error("Cleaning pipeline failed: " + str(e))
            return False

if __name__ == "__main__":
    config = {
        'base_currency': 'INR',
        'convert_to_base': ['nasdaq', 'sp500'],
        'time_zones': {
            'nasdaq': 'America/New_York',
            'sp500': 'America/New_York',
            'nifty_etf': 'Asia/Kolkata',
            'nifty_index': 'Asia/Kolkata',
            'sensex': 'Asia/Kolkata'
        },
        'datasets': {
            'nasdaq': 'data/raw/nasdaq.csv',
            'nifty_etf': 'data/raw/nifty_etf.csv',
            'nifty_index': 'data/raw/nifty_index.csv',
            'sensex': 'data/raw/sensex.csv',
            'sp500': 'data/raw/sp500.csv',
            'fx_rates': 'data/raw/usd_inr.csv'
        }
    }

    cleaner = FinancialDataCleaner(config)
    if cleaner.clean_all():
        logging.info("✨ Data cleaning completed successfully")
    else:
        logging.error("🔥 Data cleaning failed")
