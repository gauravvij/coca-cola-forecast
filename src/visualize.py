import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import pickle
import warnings

import wandb
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')

def plot_original_with_ma(df, ma_column='MA20'):
    """Plot original close prices with moving average overlay."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Date'], df['Close'], label='Close Price', color='steelblue', linewidth=1.5, alpha=0.8)
    ax.plot(df['Date'], df[ma_column], label=f'{ma_column} Moving Average', 
            color='coral', linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax.set_title('Coca-Cola (KO) Stock Price with 20-Day Moving Average', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_train_test_split(train_data, test_data, train_idx):
    """Plot training and test split with vertical separator."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    train_dates = range(len(train_data))
    test_dates = range(len(train_data), len(train_data) + len(test_data))
    
    ax.plot(train_dates, train_data.values, label='Training Set', color='green', linewidth=2, alpha=0.8)
    ax.plot(test_dates, test_data.values, label='Test Set', color='red', linewidth=2, alpha=0.8)
    
    ax.axvline(x=len(train_data), color='black', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')
    
    ax.set_xlabel('Time Index (Trading Days)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax.set_title('Coca-Cola Stock Price - Training/Test Split', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_forecast_with_ci(close_prices, test_data, test_pred, forecast_values, forecast_ci):
    """Plot historical data, test predictions, and future forecast with confidence intervals."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(range(len(close_prices)), close_prices.values, label='Historical Price', 
            color='steelblue', linewidth=1.5, alpha=0.7)
    
    test_start = len(close_prices) - len(test_data)
    test_range = range(test_start, len(close_prices))
    ax.plot(test_range, test_pred.values, label='Test Predictions', 
            color='orange', linewidth=2, alpha=0.8, linestyle='--')
    
    forecast_start = len(close_prices)
    forecast_range = range(forecast_start, forecast_start + len(forecast_values))
    ax.plot(forecast_range, forecast_values.values, label='3-Month Forecast', 
            color='green', linewidth=2.5, alpha=0.9)
    
    ci_lower = forecast_ci.iloc[:, 0].values
    ci_upper = forecast_ci.iloc[:, 1].values
    ax.fill_between(forecast_range, ci_lower, ci_upper, 
                     color='green', alpha=0.2, label='95% Confidence Interval')
    
    ax.axvline(x=forecast_start, color='red', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time Index (Trading Days)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax.set_title('Coca-Cola Stock Price: Test Predictions & 3-Month Forecast', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_residuals(train_data, test_data, test_pred):
    """Plot model residuals (test set prediction errors)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    residuals = test_data.values - test_pred.values
    ax.plot(residuals, color='steelblue', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.fill_between(range(len(residuals)), residuals, alpha=0.3, color='steelblue')
    
    ax.set_xlabel('Test Sample Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residual ($)', fontsize=11, fontweight='bold')
    ax.set_title('Model Residuals (Test Set Errors)', fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_acf_pacf(train_data, test_data, test_pred):
    """Plot ACF and PACF of residuals for diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals = test_data.values - test_pred.values
    lags = min(20, len(residuals)//2)
    
    try:
        plot_acf(residuals, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Lag', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('ACF', fontsize=10, fontweight='bold')
        
        plot_pacf(residuals, lags=lags, ax=axes[1], alpha=0.05)
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Lag', fontsize=10, fontweight='bold')
        axes[1].set_ylabel('PACF', fontsize=10, fontweight='bold')
    except Exception as e:
        logger.warning(f"ACF/PACF computation failed: {e}. Using histogram instead.")
        axes[0].hist(residuals, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Residual Value ($)', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
        
        axes[1].scatter(range(len(residuals)), residuals, alpha=0.6, color='steelblue')
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Sample Index', fontsize=10, fontweight='bold')
        axes[1].set_ylabel('Residual Value ($)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def generate_all_visualizations(results_dict, output_dir='visualizations'):
    """Generate all visualizations and upload to W&B."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"=== Generating Visualizations ===")
    
    df = results_dict['df']
    close_prices = results_dict['close_prices']
    train_data = results_dict['train_data']
    test_data = results_dict['test_data']
    test_pred = results_dict['test_predictions']
    forecast_values = results_dict['forecast_values']
    forecast_ci = results_dict['forecast_ci']
    
    logger.info("Creating original price + MA plot...")
    fig1 = plot_original_with_ma(df, ma_column='MA20')
    fig1_path = os.path.join(output_dir, 'ko_price_with_ma.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"Saved to {fig1_path}")
    wandb.log({'Original Price with MA': wandb.Image(fig1_path)})
    
    logger.info("Creating train/test split plot...")
    fig2 = plot_train_test_split(train_data, test_data, len(train_data))
    fig2_path = os.path.join(output_dir, 'ko_train_test_split.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved to {fig2_path}")
    wandb.log({'Train/Test Split': wandb.Image(fig2_path)})
    
    logger.info("Creating forecast with confidence intervals plot...")
    fig3 = plot_forecast_with_ci(close_prices, test_data, test_pred, forecast_values, forecast_ci)
    fig3_path = os.path.join(output_dir, 'ko_forecast_with_ci.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    logger.info(f"Saved to {fig3_path}")
    wandb.log({'Forecast with Confidence Intervals': wandb.Image(fig3_path)})
    
    logger.info("Creating residuals plot...")
    fig4 = plot_residuals(train_data, test_data, test_pred)
    fig4_path = os.path.join(output_dir, 'ko_residuals.png')
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    logger.info(f"Saved to {fig4_path}")
    wandb.log({'Model Residuals': wandb.Image(fig4_path)})
    
    logger.info("Creating ACF/PACF diagnostics plot...")
    fig5 = plot_acf_pacf(train_data, test_data, test_pred)
    fig5_path = os.path.join(output_dir, 'ko_acf_pacf.png')
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close(fig5)
    logger.info(f"Saved to {fig5_path}")
    wandb.log({'ACF/PACF Diagnostics': wandb.Image(fig5_path)})
    
    logger.info(f"=== All visualizations completed and logged to W&B ===")
    
    return {
        'original_ma': fig1_path,
        'train_test': fig2_path,
        'forecast': fig3_path,
        'residuals': fig4_path,
        'acf_pacf': fig5_path
    }