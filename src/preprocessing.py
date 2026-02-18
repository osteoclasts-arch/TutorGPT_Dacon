import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict, Any
import logging

# Configure logging to match "Care" philosophy - transparent and informative
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingService(BaseEstimator, TransformerMixin):
    """
    A comprehensive service for data preprocessing in the TutorGPT pipeline.
    
    Philosophy: "Care" for the data. We treat missing values not as errors but as 
    potential signals or gaps in student records that need thoughtful imputation.
    """
    
    def __init__(self):
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.fill_values: Dict[str, Any] = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Analyzes the data to determine how to best process it.
        """
        logger.info("Fitting PreprocessingService...")
        
        # Identify feature types (excluding ID and target if present)
        # We assume X might contain 'ID' but we won't process it as a feature
        cols_to_exclude = ['ID', 'target']
        
        self.numeric_features = [
            col for col in X.columns 
            if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(X[col])
        ]
        
        self.categorical_features = [
            col for col in X.columns 
            if col not in cols_to_exclude and not pd.api.types.is_numeric_dtype(X[col])
        ]
        
        logger.info(f"Identified {len(self.numeric_features)} numeric and {len(self.categorical_features)} categorical features.")
        
        # Learn imputation values
        for col in self.numeric_features:
            self.fill_values[col] = X[col].median()
            
        for col in self.categorical_features:
            self.fill_values[col] = X[col].mode()[0] if not X[col].mode().empty else 'Missing'
            
            # Prepare LabelEncoders for categorical features
            # Note: For tree-based models, Label Encoding is often sufficient and effective
            le = LabelEncoder()
            # Fit on available data, handling NaNs by filling them temporarily for fitting
            le.fit(X[col].fillna(str(self.fill_values[col])).astype(str))
            self.label_encoders[col] = le
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned transformations to the data.
        """
        logger.info("Transforming data...")
        X_processed = X.copy()
        
        # 1. Imputation
        for col, fill_val in self.fill_values.items():
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].fillna(fill_val)
        
        # 2. Feature Engineering
        # Create 'interaction_score' if relevant columns exist
        if 'learning_hours_avg' in X_processed.columns and 'quiz_score_avg' in X_processed.columns:
            X_processed['efficiency_score'] = X_processed['quiz_score_avg'] / (X_processed['learning_hours_avg'] + 1)
            
        # 3. Encoding
        for col in self.categorical_features:
            if col in X_processed.columns:
                le = self.label_encoders[col]
                # Handle unseen labels by mapping them to a default or handling gracefully
                # Here we strictly map, assuming test data resembles train data structure
                # A robust pipeline might use TargetEncoding or specific handling for unseen labels
                
                # Convert to string to match fit type, then map
                X_processed[col] = X_processed[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else -1
                )
                
        # Drop ID if it exists in the feature set to be returned
        if 'ID' in X_processed.columns:
            X_processed = X_processed.drop(columns=['ID'])
            
        return X_processed

    def get_feature_names(self) -> List[str]:
        """Returns the list of processed feature names."""
        # This is an approximation as we might have added engineered features
        # ideally we track the exact columns in transform
        return self.numeric_features + self.categorical_features + ['efficiency_score']
