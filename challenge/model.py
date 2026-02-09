import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List
import xgboost as xgb


class DelayModel:

    def __init__(
        self
    ):
        self._model = None  # Model should be saved in this attribute.
        # Top 10 most important features from the notebook analysis
        self._top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Generate one-hot encoding for categorical features
        features = pd.concat([
            pd.get_dummies(data_copy['OPERA'], prefix='OPERA'),
            pd.get_dummies(data_copy['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data_copy['MES'], prefix='MES')
        ], axis=1)
        
        # Ensure all top 10 features exist (fill with 0 if missing)
        for feature in self._top_10_features:
            if feature not in features.columns:
                features[feature] = 0
        
        # Keep only the top 10 features in the correct order
        features = features[self._top_10_features]
        
        # If target column is specified, create and return target
        if target_column:
            # Calculate delay based on min_diff if not already present
            if target_column not in data_copy.columns:
                # Calculate min_diff if needed
                if 'min_diff' not in data_copy.columns:
                    data_copy['min_diff'] = self._calculate_min_diff(data_copy)
                # Create delay column: 1 if min_diff > 15, else 0
                data_copy[target_column] = np.where(data_copy['min_diff'] > 15, 1, 0)
            
            target = data_copy[[target_column]]
            return features, target
        
        return features
    
    def _calculate_min_diff(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the difference in minutes between scheduled and actual time.
        
        Args:
            data (pd.DataFrame): raw data with Fecha-I and Fecha-O columns.
            
        Returns:
            pd.Series: difference in minutes.
        """
        fecha_o = pd.to_datetime(data['Fecha-O'], format='%Y-%m-%d %H:%M:%S')
        fecha_i = pd.to_datetime(data['Fecha-I'], format='%Y-%m-%d %H:%M:%S')
        min_diff = (fecha_o - fecha_i).dt.total_seconds() / 60
        return min_diff

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Calculate class imbalance ratio for scale_pos_weight
        n_y0 = len(target[target.iloc[:, 0] == 0])
        n_y1 = len(target[target.iloc[:, 0] == 1])
        scale = n_y0 / n_y1
        
        # Initialize and train XGBoost model with class balancing
        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale  # Balance classes
        )
        
        # Fit the model
        self._model.fit(features, target.iloc[:, 0])

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        # If model hasn't been trained, return all zeros (no delay prediction)
        if self._model is None:
            return [0] * len(features)
        
        # Get predictions and convert to list of integers
        predictions = self._model.predict(features)
        return predictions.tolist()