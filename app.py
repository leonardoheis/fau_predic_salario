from typing import Annotated, List, Literal
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field, conint, confloat, constr, model_validator, ValidationError
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from typing import List
import logging
import traceback

from example import ENDPOINT_EXAMPLES  # Assuming this is a module with OpenAPI examples

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Salary Prediction API",
    description="API for predicting salaries based on demographic and professional information",
    version="1.0.0"
)

# Load models with version checking
try:
    pipeline       = joblib.load("./model/preprocessing_pipeline.pkl")
    model          = joblib.load("./model/salary_model.pkl")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise RuntimeError(f"Model loading failed. Please check scikit-learn versions match between training and serving. Error: {str(e)}")

class InputData(BaseModel):
    """Input data model for salary prediction.
    
    Attributes:
        age (float): Age of the individual in years
        gender (str): Gender of the individual (Male/Female/Other)
        education_level (str): Highest education level (e.g., "Bachelor's", "Master's", "PhD")
        job_title (str): Current job title/position
        years_of_experience (float): Total years of professional experience
        description (str): Professional description or summary
        
    Validations:
        Each field has specific validation rules:
        - `age`: Must be between 18 and 70
        - `gender`: Must be one of Male or Female
        - `education_level`: Must be one of Bachelor's, Master's, or PhD"
        - `job_title`: Must be a non-empty string with max length 256
        - `years_of_experience`: Must be between 0 and 50
        - `description`: Must be a string between 20 and 1000 characters
        
    """
    age: conint(ge=18, le=70) = Field(
        ..., description="Age in years (must be between 18 and 70)"
    )
    gender: Literal["Male","Female"] = Field(
        ..., description="Gender, one of Male or Female"
    )
    education_level: Literal["Bachelor's", "Master's", "PhD"] = Field(
        ..., description="Highest education level"
    )
    job_title: constr(min_length=1, max_length=256) = Field(
        ..., description="Job title (1–256 chars)"
    )
    years_of_experience: confloat(ge=0, le=50) = Field(
        ..., description="Years of experience (0–50)"
    )
    description: constr(min_length=20, max_length=1000) = Field(
        ...,
        description=(
            "Free-text professional description (20–1000 chars). "
            "E.g.: “I am a 36-year-old female Sales Associate …”"
        )
    )
    
    # This is a functon that validates using a business rule, 
    # if the user is old enough to have the years of experience they claim.
    @model_validator(mode="after")
    def check_experience_vs_age(self):
        MIN_WORK_AGE = 18
        max_exp = max(0, self.age - MIN_WORK_AGE)
        if self.years_of_experience > max_exp:
            raise ValueError(
                f"A {self.age}-year-old cannot plausibly have "
                f"{self.years_of_experience} years of experience "
                f"(max {max_exp})."
            )
        return self
    
class PredictionResponse(BaseModel):
    """Response model for salary prediction.
    
    Attributes:
        predicted_salary (float): Predicted annual salary in USD
        confidence_interval (List[float]): Estimated confidence interval for the prediction [lower, upper]
    """
    predicted_salary: float
    confidence_interval: List[float]


@app.post("/predict")
async def predict_salary(
        data: Annotated[
        InputData,
        Body(
            openapi_examples=ENDPOINT_EXAMPLES
        )
    ]
) -> PredictionResponse:
    """Predict salary based on input features.
    
    Args:
        data (InputData): Input data containing demographic and professional information
        
    Returns:
        PredictionResponse: Object containing predicted salary and confidence interval
        
    Raises:
        HTTPException: 400 Bad Request if prediction fails due to invalid input
        or processing error
    """
    try:
        input_df = pd.DataFrame([{
            'Age': data.age,
            'Gender': data.gender,
            'Education Level': data.education_level,
            'Job Title': data.job_title,
            'Years of Experience': data.years_of_experience,
            'Description': data.description
        }])
        
        processed_data = pipeline.transform(input_df)
        clean_description = input_df['Description'].fillna('').str.lower()
        embeddings = sentence_model.encode(clean_description.tolist())
        final_features = np.hstack([processed_data, embeddings])
        
        predictions_by_estimator = np.array([estimator.predict(final_features)[0] for estimator in model.estimators_])
        length = len(predictions_by_estimator)
        bootstrap_samples = int(1e5)
        bootstrap_indexes = np.random.randint(0, length, size=(bootstrap_samples, length))
        predictions = np.mean(predictions_by_estimator[bootstrap_indexes], axis=1) # could be median
        
        point_prediction = np.mean(predictions)
        ci = 95
        alpha = (100 - ci) / 2
        predictions_lower_ci = np.percentile(predictions, alpha)
        predictions_upper_ci = np.percentile(predictions, 100 - alpha)
        confidence_interval = [predictions_lower_ci, predictions_upper_ci]
        
        return {
            "predicted_salary": round(float(point_prediction), 2),
            "confidence_interval": [round(x, 2) for x in confidence_interval]
        }        
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Prediction failed: {str(tb)}")
        raise HTTPException(status_code=400, detail=str(tb))