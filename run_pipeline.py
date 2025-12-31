import sys
sys.path.insert(0, '/home/ubuntu/neo_projects/wandb_t1/src')

from data_pipeline import process_ko_data
from train_forecast import train_and_forecast
from visualize import generate_all_visualizations
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("COCA-COLA STOCK FORECASTING PIPELINE")
logger.info("="*60)

logger.info("\n[STEP 1] Data Pipeline - Download & Process")
processed_data, ma_series = process_ko_data(output_dir='data', ma_window=20)

logger.info("\n[STEP 2] Training & Forecasting - SARIMA Model")
results = train_and_forecast(data_path='data/ko_processed.csv', forecast_days=90)

logger.info("\n[STEP 3] Visualization - Generate Plots & Report")
vis_paths = generate_all_visualizations(results, output_dir='visualizations')

logger.info("\n" + "="*60)
logger.info("PIPELINE COMPLETE")
logger.info("="*60)