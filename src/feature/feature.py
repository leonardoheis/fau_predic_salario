import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline



class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer that creates:
    - Seniority level from job titles
    - Role family/category from job titles
    - Experience-education interaction feature

    The transformer performs the following operations:
    1. Extracts seniority level (e.g., Senior, Junior, Director) from job titles
    2. Groups jobs into broader role families (e.g., Data, Engineering, Sales)
    3. Creates an interaction feature combining years of experience and education level
    4. Drops the original 'Job Title' column as it's replaced with engineered features
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Fit method is not needed for this transformer
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed data with new features and original 'Job Title' column removed.
        """
        X = X.copy()
        
        # Extract seniority level from job titles
        X['Seniority'] = X['Job Title'].apply(self._extract_seniority)
        
        # Extract role family/category from job titles
        X['Role_family'] = X['Job Title'].apply(self._extract_role_family)
        
        # Create experience-education interaction feature
        X['Exp_education'] = X['Years of Experience'] * X['Education Level'].map({
            "Bachelor's": 1, "Master's": 2, "PhD": 3
        })
        
        return X.drop(columns=['Job Title'])
    
    def _extract_seniority(self, title: str) -> str:
        """Helper method to extract seniority level from job title
        
        Parameters
        ----------
        title : str
            Job title string to analyze.
            
        Returns
        -------
        str
            Seniority level extracted from the title.
        """
        title = title.lower()
        if 'senior' in title: return 'Senior'
        elif 'junior' in title: return 'Junior'
        elif 'lead' in title or 'manager' in title: return 'Lead'
        elif 'director' in title or 'head' in title: return 'Director'
        elif 'intern' in title: return 'Intern'
        else: return 'Mid'

    def _extract_role_family(self, title: str) -> str:
        """Helper method to extract role family from job title
        
        Parameters
        ----------
        title : str
            Job title string to analyze.
            
        Returns
        -------
        str
            Role family extracted from the title.
        """
        title = title.lower()
        if 'data' in title: return 'Data'
        elif 'engineer' in title or 'developer' in title: return 'Engineering'
        elif 'analyst' in title: return 'Analytics'
        elif 'sales' in title: return 'Sales'
        elif 'marketing' in title: return 'Marketing'
        elif 'product' in title: return 'Product'
        elif 'hr' in title or 'recruit' in title: return 'HR'
        elif 'finance' in title or 'account' in title: return 'Finance'
        elif 'designer' in title: return 'Design'
        else: return 'Other'

def create_preprocessor() -> ColumnTransformer:
    """Creates the ColumnTransformer with all preprocessing steps
    
    The preprocessor performs the following transformations:
    - Ordinal encoding for education level (converts to ordered numeric values)
    - One-hot encoding for categorical features ('Gender', 'Seniority', 'Role_family')
    - Standard scaling for numerical features ('Age', 'Years of Experience', 'Exp_education')
    
    The scaling transformation (StandardScaler) centers the data around mean 0 with standard deviation 1.
    This is important because:
    - Many machine learning algorithms perform optimally when features follow a normal distribution
    - Features with different scales can dominate the model's objective function
    - Scaling helps algorithms that use distance calculations or gradient descent converge faster
    
    Returns
    -------
    ColumnTransformer
        Configured ColumnTransformer with all preprocessing steps.
    """
    # return ColumnTransformer([
    #     ('ordinal', OrdinalEncoder(categories=[["Bachelor's", "Master's", "PhD"]]), ['Education Level']),
    #     ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Seniority', 'Role_family']),
    #     ('scaler', StandardScaler(), ['Age', 'Years of Experience', 'Exp_education'])
    # ])
    ct = ColumnTransformer(
        transformers=[
            ("ord_edu",
             OrdinalEncoder(categories=[["Bachelor's","Master's","PhD"]]),
             ["Education Level"]),
            ("ohe",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["Gender","Seniority","Role_family"]),
            ("scale",
             StandardScaler(),
             ["Age","Years of Experience","Exp_education"]),
        ],
        #remainder="passthrough",           # <— carry through id, Description, Salary, etc.
        verbose_feature_names_out=False,
    )
    # crucial: ask it to return a DataFrame
    return ct.set_output(transform="pandas")

def build_pipeline() -> Pipeline:
    """Builds the complete preprocessing pipeline
    
    The pipeline consists of:
    1. Feature engineering step (FeatureEngineer)
    2. Preprocessing step (ColumnTransformer with encoding and scaling)
    
    Returns
    -------
    Pipeline
        Configured Pipeline with all transformation steps.
    """
    return Pipeline([
        ('features', FeatureEngineer()),
        ('preprocessing', create_preprocessor())
    ])
    
    