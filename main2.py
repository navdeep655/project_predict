from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

class ModelInput(BaseModel):
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Tutoring_Sessions: float
    Physical_Activity: float
    Parental_Involvement_Low: int
    Parental_Involvement_Medium: int
    Teacher_Quality_Low: int
    Teacher_Quality_Medium: int
    Gender_Male: int

# Load the pre-trained model
model = joblib.load("student_score.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Score Prediction API. Use the /predict/ endpoint to get predictions."}

@app.post("/predict/")
def predict(data: ModelInput):
    # Convert the input data to a DataFrame
    data_dict = data.dict()
    user_data = [[
        data_dict["Hours_Studied"], data_dict["Attendance"], data_dict["Sleep_Hours"], 
        data_dict["Previous_Scores"], data_dict["Tutoring_Sessions"], data_dict["Physical_Activity"], 
        data_dict["Parental_Involvement_Low"], data_dict["Parental_Involvement_Medium"], 
        data_dict["Teacher_Quality_Low"], data_dict["Teacher_Quality_Medium"], data_dict["Gender_Male"]
    ]]
    
    # Make predictions using the loaded model
    prediction = model.predict(user_data)
    
    # Return the prediction as a JSON response
    return {"prediction": float(prediction[0])}
