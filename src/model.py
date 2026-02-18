from typing import Dict, Any, List
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    A robust ensemble model combining the strengths of XGBoost, LightGBM, and CatBoost.
    
    Philosophy: Diversity in learning. Just as students learn differently, these models
    capture different patterns in the data. We combine them (Soft Voting) for a more 
    reliable prediction of student completion.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {
            'xgb': 0.33,
            'lgbm': 0.33,
            'cat': 0.34
        }
        self._initialize_models()

    def _initialize_models(self):
        """Initializes the base models with hyperparameters tuned for stability."""
        
        # XGBoost
        self.models['xgb'] = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss',
            early_stopping_rounds=50
        )
        
        # LightGBM
        self.models['lgbm'] = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        # CatBoost
        self.models['cat'] = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=self.random_state,
            verbose=False,
            allow_writing_files=False
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: List[tuple] = None):
        """
        Trains all underlying models.
        
        Args:
            X: Training features
            y: Target labels
            eval_set: List of (X_val, y_val) tuples for early stopping
        """
        logger.info("Training ensemble models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name.upper()}...")
            
            if name == 'xgb':
                model.fit(X, y, eval_set=eval_set, verbose=False)
            elif name == 'lgbm':
                # LightGBM interface for early stopping in sklearn API
                callbacks = None
                if eval_set:
                    from lightgbm import early_stopping, log_evaluation
                    callbacks = [early_stopping(stopping_rounds=50), log_evaluation(0)]
                model.fit(X, y, eval_set=eval_set, eval_metric='binary_logloss', callbacks=callbacks)
            elif name == 'cat':
                model.fit(X, y, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
                
        logger.info("All models trained successfully.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the weighted average of probabilities from all models.
        """
        weighted_probs = np.zeros((len(X), 2))
        
        for name, model in self.models.items():
            probs = model.predict_proba(X)
            weighted_probs += probs * self.weights[name]
            
        return weighted_probs

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Returns class predictions based on weighted probabilities.
        """
        probs = self.predict_proba(X)[:, 1] # Probability of class 1
        return (probs >= threshold).astype(int)
