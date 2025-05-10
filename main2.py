from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize app
app = FastAPI()

# Load model and transformers
model = joblib.load("student_score.pkl")
scaler = joblib.load("scaled.pkl")
encoder = joblib.load("encoded.pkl")

# Define input schema
class ModelInput(BaseModel):
    Hours_Studied: float
    Attendance: float
    Previous_Scores: float
    Tutoring_Sessions: float
    Physical_Activity: float
    Internet_Access: str  # e.g., "Yes" or "No"

@app.get("/")
def read_root():
    return {"message": "API for predicting student exam scores using Linear Regression"}

@app.post("/predict/")
def predict(data: ModelInput):
    # Extract numeric values
    numeric_values = [[
        data.Hours_Studied,
        data.Attendance,
        data.Previous_Scores,
        data.Tutoring_Sessions,
        data.Physical_Activity
    ]]
    
    # Scale numeric values
    scaled_values = scaler.transform(numeric_values)

    # Encode categorical value
    encoded_values = encoder.transform([[data.Internet_Access]])
    encoded_array = encoded_values.toarray() if hasattr(encoded_values, "toarray") else encoded_values

    # Combine scaled + encoded
    input_data = list(scaled_values[0]) + list(encoded_array[0])

    # Predict
    prediction = model.predict([input_data])
    
    return {"predicted_exam_score": float(prediction[0])}
