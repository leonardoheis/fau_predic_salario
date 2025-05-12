from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.feature.feature import FeatureEngineer

def create_preprocessing_pipeline():
    """
    Creates a preprocessing pipeline for numerical and categorical features.
    
    Returns:
        ColumnTransformer: The preprocessing pipeline.
    """
    # Define the numerical and categorical features
    numerical_features = ['salary', 'years_of_experience']
    categorical_features = ['job_title', 'location']
    
    # Create the numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Create the categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine the transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def build_pipeline():
    """
    Builds a complete pipeline including preprocessing and feature engineering.
    
    Returns:
        Pipeline: The complete pipeline.
    """
    preprocessor = create_preprocessing_pipeline()
    
    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_engineer', FeatureEngineer())
    ])
    
    return pipeline