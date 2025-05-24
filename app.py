from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from typing import List
import logging
import traceback

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
    """
    #id: int = 1
    age: float
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float
    description: str
    
class PredictionResponse(BaseModel):
    """Response model for salary prediction.
    
    Attributes:
        predicted_salary (float): Predicted annual salary in USD
        confidence_interval (List[float]): Estimated confidence interval for the prediction [lower, upper]
    """
    predicted_salary: float
    confidence_interval: List[float]


@app.post("/predict", response_model=PredictionResponse)
async def predict_salary(data: InputData):
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
            #'id': 1,
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
        
        prediction = model.predict(final_features)[0]
        confidence_interval = [prediction * 0.9, prediction * 1.1]
        
        return {
            "predicted_salary": round(float(prediction), 2),
            "confidence_interval": [round(x, 2) for x in confidence_interval]
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Prediction failed: {str(tb)}")
        raise HTTPException(status_code=400, detail=str(tb))