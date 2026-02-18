import os
import argparse
import logging
import sys

# Add the current directory to path so imports work correctly
sys.path.append(os.getcwd())

from data.generate_mock import MockDataGenerator
from src.train import TrainingService
from src.inference import InferenceService

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TutorGPT_Main")

def main():
    parser = argparse.ArgumentParser(description="TutorGPT Dacon Pipeline")
    parser.add_argument('--mode', type=str, default='all', choices=['data', 'train', 'infer', 'all'],
                        help='Pipeline mode: data (generate mock), train, infer, or all (default)')
    
    args = parser.parse_args()
    
    base_dir = "/home/junsikback/.openclaw/workspace/TutorGPT_Dacon"
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Data Generation
    if args.mode in ['data', 'all']:
        logger.info(">>> STEP 1: Generating Mock Data")
        generator = MockDataGenerator(output_dir=data_dir)
        generator.save_csv()
        
    # 2. Training
    if args.mode in ['train', 'all']:
        logger.info(">>> STEP 2: Training Models")
        train_path = os.path.join(data_dir, "train.csv")
        if not os.path.exists(train_path):
            logger.error("Train data not found. Run with --mode data first.")
            return
            
        trainer = TrainingService(data_path=train_path)
        trainer.train_pipeline()
        
    # 3. Inference
    if args.mode in ['infer', 'all']:
        logger.info(">>> STEP 3: Inference")
        test_path = os.path.join(data_dir, "test.csv")
        if not os.path.exists(test_path):
            logger.error("Test data not found.")
            return
            
        inferencer = InferenceService(test_data_path=test_path)
        inferencer.generate_submission()
        
    logger.info(">>> Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    main()
