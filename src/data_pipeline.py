import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import sys

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def download_ko_data(years=2):
    """
    Download 2 years of Coca-Cola (KO) historical stock price data.
    
    Args:
        years: Number of years of historical data to download (default: 2)
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    logger.info(f"Downloading {years} years of KO (Coca-Cola) data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    ko_data = yf.download("KO", start=start_date, end=end_date, progress=False)
    logger.info(f"Downloaded {len(ko_data)} records from {ko_data.index[0].date()} to {ko_data.index[-1].date()}")
    
    return ko_data

def handle_missing_values(df):
    """
    Handle missing values in the price data using forward fill.
    
    Args:
        df: Input DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    missing_before = df.isnull().sum().sum()
    df = df.fillna(method='ffill').fillna(method='bfill')
    missing_after = df.isnull().sum().sum()
    
    logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
    return df

def compute_moving_average(df, window=20):
    """
    Compute simple moving average for smoothing.
    
    Args:
        df: Input DataFrame with 'Close' column
        window: Moving average window size (default: 20 days)
        
    Returns:
        pd.Series: Moving average series
    """
    ma = df['Close'].rolling(window=window, min_periods=1).mean()
    logger.info(f"Computed {window}-day moving average")
    return ma

def process_ko_data(output_dir='data', ma_window=20):
    """
    Complete data pipeline: download, clean, and compute moving average.
    
    Args:
        output_dir: Directory to save processed data
        ma_window: Window size for moving average
        
    Returns:
        tuple: (processed_df, moving_avg_series)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"=== Coca-Cola Stock Data Processing Pipeline ===")
    logger.info(f"Random seed set to: {RANDOM_SEED}")
    
    # Download data
    ko_data = download_ko_data(years=2)
    
    # Reset index to have date as column for processing
    ko_data = ko_data.reset_index()
    
    # Handle missing values
    ko_data = handle_missing_values(ko_data)
    
    # Compute moving average on the Close column
    ko_data['MA20'] = compute_moving_average(ko_data, window=ma_window)
    
    ko_data_clean = ko_data
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'ko_processed.csv')
    ko_data_clean.to_csv(csv_path, index=False)
    logger.info(f"Saved processed data to {csv_path}")
    
    logger.info(f"Data shape: {ko_data_clean.shape}")
    logger.info(f"Date range: {ko_data_clean['Date'].min()} to {ko_data_clean['Date'].max()}")
    
    return ko_data_clean, ko_data_clean['MA20']

if __name__ == '__main__':
    processed_data, ma_series = process_ko_data(output_dir='data', ma_window=20)
    print(f"\n=== Data Pipeline Summary ===")
    print(f"Processed records: {len(processed_data)}")
    print(f"Date range: {processed_data['Date'].min()} to {processed_data['Date'].max()}")
    print(f"Closing price range: ${processed_data['Close'].min():.2f} - ${processed_data['Close'].max():.2f}")
    print(f"Moving average (last 5 days):\n{ma_series.tail()}")