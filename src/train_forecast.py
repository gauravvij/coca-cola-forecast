import os
import sys
import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime, timedelta
import pickle

import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_processed_data(data_path='data/ko_processed.csv'):
    """Load processed KO data from CSV."""
    logger.info(f"Loading processed data from {data_path}")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Keep only numeric columns for price analysis
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA20']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def compute_metrics(y_true, y_pred):
    """
    Compute standard time series metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary with MAE, RMSE, MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def check_stationarity(series):
    """Check and optionally difference for stationarity."""
    from statsmodels.tsa.stattools import adfuller
    
    # Ensure series is numeric
    series = pd.to_numeric(series, errors='coerce').dropna()
    
    adf_result = adfuller(series, autolag='AIC')
    p_value = adf_result[1]
    
    logger.info(f"ADF test p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        logger.info("Series appears stationary")
        return False
    else:
        logger.info("Series not stationary, will apply differencing")
        return True

def train_sarima_model(series, train_size=0.8):
    """
    Train a simple trend-based forecasting model.
    
    Args:
        series: Price series (pandas Series)
        train_size: Fraction of data to use for training
        
    Returns:
        tuple: (model_results, train_data, test_data, test_predictions, metrics)
    """
    logger.info(f"=== Training Trend-Based Forecasting Model ===")
    logger.info(f"Series length: {len(series)}")
    
    # Ensure series is numeric
    series = pd.to_numeric(series, errors='coerce').dropna()
    
    # Split data
    train_idx = int(len(series) * train_size)
    train_data = series[:train_idx]
    test_data = series[train_idx:]
    
    logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Calculate trend from training data (linear regression)
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data.values
    
    from sklearn.linear_model import LinearRegression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Generate test predictions
    X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
    test_predictions = pd.Series(lr_model.predict(X_test), index=test_data.index)
    
    logger.info(f"Model fitted successfully")
    logger.info(f"Trend slope: {lr_model.coef_[0]:.4f}")
    
    # Compute metrics
    metrics = compute_metrics(test_data.values, test_predictions.values)
    logger.info(f"Test Metrics - MAE: ${metrics['MAE']:.2f}, RMSE: ${metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
    
    # Create a simple model object to match expected return signature
    class SimpleModel:
        def __init__(self, lr_model, train_data):
            self.lr_model = lr_model
            self.train_data = train_data
            self.coef = lr_model.coef_[0]
            self.intercept = lr_model.intercept_
    
    results = SimpleModel(lr_model, train_data)
    
    return results, train_data, test_data, test_predictions, metrics

def forecast_future_prices(model_results, series, steps=90):
    """
    Forecast future prices (3 months = ~60-90 trading days).
    
    Args:
        model_results: Fitted model results with lr_model
        series: Full price series
        steps: Number of trading days to forecast
        
    Returns:
        tuple: (forecast_values, forecast_ci)
    """
    logger.info(f"Forecasting {steps} trading days into future...")
    
    # Use the linear regression model to extrapolate
    last_idx = len(series)
    X_future = np.arange(last_idx, last_idx + steps).reshape(-1, 1)
    
    forecast_values = pd.Series(
        model_results.lr_model.predict(X_future),
        index=pd.RangeIndex(steps)
    )
    
    # Create confidence intervals (±5% of last price)
    last_price = series.iloc[-1]
    margin = last_price * 0.05
    
    forecast_ci = pd.DataFrame({
        0: forecast_values - margin,
        1: forecast_values + margin
    })
    
    logger.info(f"Forecast range: ${forecast_values.min():.2f} - ${forecast_values.max():.2f}")
    
    return forecast_values, forecast_ci

def train_and_forecast(data_path='data/ko_processed.csv', forecast_days=90):
    """
    Complete training and forecasting pipeline.
    
    Args:
        data_path: Path to processed data
        forecast_days: Number of days to forecast (default: 90)
        
    Returns:
        dict: Results dictionary with all artifacts
    """
    # Initialize W&B
    wandb.init(
        project="coca-cola-forecasting",
        config={
            'ticker': 'KO',
            'data_source': 'yfinance',
            'forecast_horizon_days': forecast_days,
            'ma_window': 20,
            'arima_order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 252),
            'train_test_split': 0.8,
            'random_seed': RANDOM_SEED
        }
    )
    
    logger.info(f"W&B initialized with run ID: {wandb.run.id}")
    
    # Load data
    df = load_processed_data(data_path)
    close_prices = df['Close']
    
    # Check stationarity
    needs_diff = check_stationarity(close_prices)
    
    # Train model
    results, train_data, test_data, test_pred, metrics = train_sarima_model(
        close_prices, 
        train_size=0.8
    )
    
    # Forecast future
    forecast_values, forecast_ci = forecast_future_prices(results, close_prices, steps=forecast_days)
    
    # Log metrics to W&B
    wandb.log({
        'test_mae': metrics['MAE'],
        'test_rmse': metrics['RMSE'],
        'test_mape': metrics['MAPE'],
        'train_size': len(train_data),
        'test_size': len(test_data),
        'forecast_horizon': forecast_days
    })
    
    # Save model artifact using joblib
    import joblib
    model_path = os.path.join('models', 'ko_trend_model.joblib')
    os.makedirs('models', exist_ok=True)
    joblib.dump(results.lr_model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Log model artifact to W&B
    wandb.save(model_path, policy='now')
    
    # Log model parameters to W&B
    wandb.log({'model_type': 'Linear Trend', 'model_coef': results.coef})
    
    # Create results dictionary
    results_dict = {
        'model': results,
        'train_data': train_data,
        'test_data': test_data,
        'test_predictions': test_pred,
        'forecast_values': forecast_values,
        'forecast_ci': forecast_ci,
        'metrics': metrics,
        'df': df,
        'close_prices': close_prices,
        'model_path': model_path
    }
    
    logger.info(f"=== Training Complete ===")
    logger.info(f"Forecast values (first 5 days): {forecast_values.head().values}")
    logger.info(f"Forecast values (last 5 days): {forecast_values.tail().values}")
    
    return results_dict

if __name__ == '__main__':
    results = train_and_forecast(data_path='data/ko_processed.csv', forecast_days=90)
    print(f"\n=== Forecasting Summary ===")
    print(f"Test MAE: ${results['metrics']['MAE']:.2f}")
    print(f"Test RMSE: ${results['metrics']['RMSE']:.2f}")
    print(f"Test MAPE: {results['metrics']['MAPE']:.2f}%")
    print(f"Forecast (last 5 values): {results['forecast_values'].tail().values}")