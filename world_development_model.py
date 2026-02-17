"""
world_development_model.py
==========================

World Development Clustering Model Module

This file contains the WorldDevelopmentClusteringModel class
that can be imported and used in other Python scripts or notebooks.

Usage:
------
    from world_development_model import WorldDevelopmentClusteringModel
    
    # Load model
    model = WorldDevelopmentClusteringModel.load('world_development_kmeans.pkl')
    
    # Predict
    predictions = model.predict(df)
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score


class WorldDevelopmentClusteringModel:
    """
    Production-ready KMeans clustering model for World Development data.
    
    This class handles all preprocessing, training, prediction, and 
    model persistence for the world development clustering project.
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize the clustering model.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters to form
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None 
        self.scaler = None                                                                
        self.imputer_stats = None  # Store median values instead of imputer object to avoid pickle issues
        self.feature_names = None
        self.log_transform_cols = [
            'GDP', 'CO2 Emissions', 'Energy Usage',
            'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound'
        ]
        self.currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
        self.metadata = {
            'model_type': 'KMeans',
            'n_clusters': n_clusters,
            'version': '1.0',
            'silhouette_score': None
        }

    def _clean_currency(self, series):
        """
        Clean currency format columns (remove $ and commas).
        
        Parameters:
        -----------
        series : pd.Series
            Column with currency format
            
        Returns:
        --------
        pd.Series : Cleaned numeric column
        """
        if series.dtype == 'object':
            return series.str.replace('$', '', regex=False)\
                        .str.replace(',', '', regex=False)\
                        .astype(float)
        return series
    
    def _preprocess(self, df, fit=False):
        """
        Preprocess data: clean, impute, transform, and scale.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool, default=False
            Whether to fit preprocessing components or just transform
            
        Returns:
        --------
        np.ndarray : Preprocessed and scaled features
        """
        df_clean = df.copy()
        
        # Clean currency columns
        for col in self.currency_cols:
            if col in df_clean.columns:
                df_clean[col] = self._clean_currency(df_clean[col])
        
        # Clean Business Tax Rate
        if 'Business Tax Rate' in df_clean.columns and df_clean['Business Tax Rate'].dtype == 'object':
            df_clean['Business Tax Rate'] = pd.to_numeric(
                df_clean['Business Tax Rate'].str.replace('%', ''), 
                errors='coerce'
            )
        
        # Drop non-feature columns
        df_model = df_clean.drop(columns=['Country', 'Number of Records'], errors='ignore')
        
        # Store or match feature names
        if fit:
            self.feature_names = df_model.columns.tolist()
        else:
            df_model = df_model[self.feature_names]
        
        # Impute missing values using median (avoid pickle issues with SimpleImputer)
        if fit:
            # Store median stats for later use instead of pickling imputer object
            self.imputer_stats = df_model.median().to_dict()
            df_model = df_model.fillna(self.imputer_stats)
        else:
            # Use stored stats to fill missing values
            if self.imputer_stats is not None:
                df_model = df_model.fillna(self.imputer_stats)
            else:
                # Fallback: use median of incoming data
                df_model = df_model.fillna(df_model.median())
        
        # Log transformation
        for col in self.log_transform_cols:
            if col in df_model.columns:
                df_model[col] = np.log1p(df_model[col])
        
        # Scaling
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(df_model)
        else:
            X_scaled = self.scaler.transform(df_model)
        
        return X_scaled

    def fit(self, df):
        """
        Train the clustering model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
            
        Returns:
        --------
        self : Returns the instance itself
        """
        X_scaled = self._preprocess(df, fit=True)
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        self.model.fit(X_scaled)
        self.metadata['silhouette_score'] = silhouette_score(X_scaled, self.model.labels_)
        
        print(f"  Model trained successfully!")
        print(f"  Silhouette Score: {self.metadata['silhouette_score']:.4f}")
        return self

    def predict(self, df):
        """
        Predict cluster assignments for new data.
        Handles missing columns by filling with 0.
        
        Parameters:
        -----------
        df : pd.DataFrame
            New data to predict
            
        Returns:
        --------
        np.ndarray : Cluster assignments
        """
        if self.model is None:
            raise ValueError("Model not trained! Call fit() first.")
        
        try:
            df_copy = df.copy()
            # Add missing columns with 0 (safe filler for normalized data)
            if self.feature_names:
                for col in self.feature_names:
                    if col not in df_copy.columns:
                        df_copy[col] = 0
                # Select only training features in correct order
                df_copy = df_copy[self.feature_names]
            
            X_scaled = self._preprocess(df_copy, fit=False)
            return self.model.predict(X_scaled)
        except KeyError as e:
            raise ValueError(f"Missing required feature(s): {e}")
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")

    def save(self, filepath='world_development_kmeans.pkl'):
        """
        Save model to file.
        
        Parameters:
        -----------
        filepath : str, default='world_development_kmeans.pkl'
            Path where to save the model
            
        Returns:
        --------
        str : Filepath where model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained!")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'imputer_stats': self.imputer_stats,  # Store stats dict instead of imputer object
                'feature_names': self.feature_names,
                'log_transform_cols': self.log_transform_cols,
                'currency_cols': self.currency_cols,
                'metadata': self.metadata
            }, f)
        
        print(f" Model saved to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath='world_development_kmeans.pkl'):
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str, default='world_development_kmeans.pkl'
            Path to the saved model
            
        Returns:
        --------
        WorldDevelopmentClusteringModel : Loaded model instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(n_clusters=data['metadata']['n_clusters'], random_state=42)
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.imputer_stats = data.get('imputer_stats', {})  # Load stats dict (compatible with old pickles)
        instance.feature_names = data['feature_names']
        instance.log_transform_cols = data['log_transform_cols']
        instance.currency_cols = data['currency_cols']
        instance.metadata = data['metadata']
        
        print(f"  Model loaded from: {filepath}")
        print(f"  Silhouette Score: {instance.metadata['silhouette_score']:.4f}")
        
        return instance


# Module-level message
if __name__ == "__main__":
    print("=" * 70)
    print("World Development Clustering Model Module")
    print("=" * 70)
    print()
    print("This module provides the WorldDevelopmentClusteringModel class.")
    print()
    print("Usage:")
    print("  from world_development_model import WorldDevelopmentClusteringModel")
    print()
    print("  # Load model")
    print("  model = WorldDevelopmentClusteringModel.load('model.pkl')")
    print()
    print("  # Predict")
    print("  predictions = model.predict(df)")
    print()
else:
    print(" WorldDevelopmentClusteringModel imported successfully!")
