import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.eda.eda import merge_multiple_dataframes



class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        # Fit method is not needed for this transformer
        return self
    
    def transform(self, X):
        return X
    
    def _extract_seniority(self, title: str) -> str:
        """
        Extracts the seniority level from a job title.
        
        Args:
            title (str): The job title.
        
        Returns:
            str: The seniority level.
        """
        if "senior" in title.lower():
            return "senior"
        elif "junior" in title.lower():
            return "junior"
        else:
            return "mid-level"

    def _extract_role_family(self, title: str) -> str:
        """
        Extracts the role family from a job title.
        
        Args:
            title (str): The job title.
        
        Returns:
            str: The role family.
        """
        if "data" in title.lower():
            return "data"
        elif "software" in title.lower():
            return "software"
        else:
            return "other"
    
    