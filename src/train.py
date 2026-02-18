import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from typing import Tuple, List
import logging
import joblib
import os
from datetime import datetime

from .preprocessing import PreprocessingService
from .model import EnsembleModel

logger = logging.getLogger(__name__)

class TrainingService:
    """
    Orchestrates the training process using Cross-Validation.
    
    Philosophy: "Care" means validation. We don't just trust one split of data.
    We use Stratified K-Fold to ensure our model performs consistently across 
    different subsets of students.
    """
    
    def __init__(self, data_path: str, model_save_path: str = "models"):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.n_splits = 5
        self.random_state = 42
        os.makedirs(model_save_path, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Loads the training data."""
        return pd.read_csv(self.data_path)
    
    def train_pipeline(self):
        """
        Executes the full training pipeline:
        1. Load Data
        2. Preprocess (Fit on full train to learn encoders, or per fold)
           *Method Note*: Ideally, we fit preprocessors inside the fold to prevent leakage.
           For this implementation, we will instantiate a new preprocessor per fold.
        3. Cross-Validate
        4. Save Metrics
        """
        logger.info("Starting Training Pipeline...")
        df = self.load_data()
        
        X = df.drop(columns=['target', 'ID'], errors='ignore')
        y = df['target']
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"--- Processing Fold {fold + 1}/{self.n_splits} ---")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Preprocessing
            # We create a FRESH preprocessor for each fold to strictly avoid data leakage
            preprocessor = PreprocessingService()
            X_train_proc = preprocessor.fit_transform(X_train, y_train)
            X_val_proc = preprocessor.transform(X_val)
            
            # Model Training
            model = EnsembleModel(random_state=self.random_state)
            model.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)])
            
            # Evaluation
            val_preds = model.predict(X_val_proc)
            score = f1_score(y_val, val_preds, average='macro') # Using macro F1 as is common in Dacon, check specific rules
            fold_scores.append(score)
            
            logger.info(f"Fold {fold + 1} F1 Score: {score:.4f}")
            
            # Save artifacts for this fold
            self._save_artifacts(model, preprocessor, fold, score)
            
        avg_score = np.mean(fold_scores)
        logger.info(f"Training Complete. Average F1 Score: {avg_score:.4f}")
        return avg_score

    def _save_artifacts(self, model, preprocessor, fold, score):
        """Saves the trained model and preprocessor."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Save Model
        model_filename = f"ensemble_fold{fold}_{score:.4f}.pkl"
        joblib.dump(model, os.path.join(self.model_save_path, model_filename))
        
        # Save Preprocessor
        prep_filename = f"preprocessor_fold{fold}_{score:.4f}.pkl"
        joblib.dump(preprocessor, os.path.join(self.model_save_path, prep_filename))
