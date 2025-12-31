import sys
sys.path.insert(0, '/home/ubuntu/neo_projects/wandb_t1/src')

from data_pipeline import process_ko_data
from train_forecast import train_and_forecast
from visualize import generate_all_visualizations
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("COCA-COLA STOCK FORECASTING PIPELINE")
logger.info("="*70)

logger.info("\n[STEP 1/3] Data Pipeline - Download & Process KO Stock Data")
processed_data, ma_series = process_ko_data(output_dir='/home/ubuntu/neo_projects/wandb_t1/data', ma_window=20)

logger.info("\n[STEP 2/3] Training & Forecasting - Model Training")
results = train_and_forecast(data_path='/home/ubuntu/neo_projects/wandb_t1/data/ko_processed.csv', forecast_days=90)

logger.info("\n[STEP 3/3] Visualization - Generate Plots & Upload to W&B")
vis_paths = generate_all_visualizations(results, output_dir='/home/ubuntu/neo_projects/wandb_t1/visualizations')

logger.info("\n" + "="*70)
logger.info("PIPELINE COMPLETE - All deliverables generated")
logger.info("="*70)
logger.info(f"✓ Processed data: /home/ubuntu/neo_projects/wandb_t1/data/ko_processed.csv")
logger.info(f"✓ Model artifact: {results.get('model_path', 'N/A')}")
logger.info(f"✓ Visualizations: /home/ubuntu/neo_projects/wandb_t1/visualizations/")
logger.info(f"✓ W&B run logged with metrics, parameters, and artifacts")

import wandb
wandb.finish()