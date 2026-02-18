import pandas as pd
import numpy as np
import joblib
import os
import glob
from typing import List
import logging

from .preprocessing import PreprocessingService
from .model import EnsembleModel

logger = logging.getLogger(__name__)

class InferenceService:
    """
    Handles the generation of predictions for the test set.
    
    Philosophy: "Care" extends to deployment. We ensemble the predictions 
    from all our cross-validated models to ensure the final submission 
    is robust and stable.
    """
    
    def __init__(self, model_dir: str = "models", test_data_path: str = "data/test.csv"):
        self.model_dir = model_dir
        self.test_data_path = test_data_path
        self.output_path = "submission.csv"
        
    def load_artifacts(self) -> List[tuple]:
        """Loads all saved model/preprocessor pairs."""
        models = []
        
        # Find all model files
        model_files = glob.glob(os.path.join(self.model_dir, "ensemble_fold*.pkl"))
        
        for model_path in model_files:
            # Infer corresponding preprocessor path
            # Assuming naming convention: ensemble_foldX_SCORE.pkl -> preprocessor_foldX_SCORE.pkl
            basename = os.path.basename(model_path)
            prep_basename = basename.replace("ensemble", "preprocessor")
            prep_path = os.path.join(self.model_dir, prep_basename)
            
            if os.path.exists(prep_path):
                logger.info(f"Loading pair: {basename} + {prep_basename}")
                model = joblib.load(model_path)
                preprocessor = joblib.load(prep_path)
                models.append((model, preprocessor))
            else:
                logger.warning(f"Preprocessor not found for {basename}, skipping.")
                
        return models
    
    def generate_submission(self):
        """
        Generates the submission file by averaging predictions from all folds.
        """
        logger.info("Starting Inference...")
        
        # Load Data
        test_df = pd.read_csv(self.test_data_path)
        ids = test_df['ID']
        X_test = test_df.drop(columns=['ID'], errors='ignore')
        
        # Load Models
        artifacts = self.load_artifacts()
        if not artifacts:
            logger.error("No valid model artifacts found. Run training first.")
            return
            
        # Accumulate probabilities
        avg_probs = np.zeros((len(test_df), 2))
        
        for model, preprocessor in artifacts:
            # Transform test data using the SPECIFIC preprocessor for this model
            X_test_proc = preprocessor.transform(X_test)
            probs = model.predict_proba(X_test_proc)
            avg_probs += probs
            
        # Average
        avg_probs /= len(artifacts)
        
        # Thresholding (Class 1 probability)
        final_preds = (avg_probs[:, 1] >= 0.5).astype(int)
        
        # Create Submission
        submission = pd.DataFrame({
            'ID': ids,
            'target': final_preds
        })
        
        submission.to_csv(self.output_path, index=False)
        logger.info(f"Submission saved to {self.output_path} with {len(submission)} rows.")
