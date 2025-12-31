import sys
import os
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from data_pipeline import process_ko_data
from train_forecast import train_and_forecast
from visualize import generate_all_visualizations

def main():
    """Main orchestration script for the Coca-Cola forecasting pipeline."""
    
    logger.info("="*60)
    logger.info("COCA-COLA STOCK FORECASTING PIPELINE")
    logger.info("="*60)
    
    # Step 1: Data Pipeline
    logger.info("\n[STEP 1] Data Pipeline - Download & Process")
    processed_data, ma_series = process_ko_data(output_dir='../data', ma_window=20)
    
    # Step 2: Training & Forecasting
    logger.info("\n[STEP 2] Training & Forecasting - SARIMA Model")
    results = train_and_forecast(data_path='../data/ko_processed.csv', forecast_days=90)
    
    # Step 3: Visualization
    logger.info("\n[STEP 3] Visualization - Generate Plots & Report")
    vis_paths = generate_all_visualizations(results, output_dir='../visualizations')
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Processed data: ../data/ko_processed.csv")
    logger.info(f"Model artifacts: ../models/ko_sarima_model.pkl")
    logger.info(f"Visualizations: ../visualizations/")
    logger.info(f"\nW&B Run: https://wandb.ai")
    
    return results

if __name__ == '__main__':
    main()