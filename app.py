from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes based on medical parameters",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = joblib.load("random_forest_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=17)
    Glucose: int = Field(..., gt=0, le=199)
    BloodPressure: int = Field(..., ge=0, le=122)
    SkinThickness: int = Field(..., ge=0, le=99)
    Insulin: int = Field(..., ge=0, le=846)
    BMI: float = Field(..., gt=0, le=67.1)
    DiabetesPedigreeFunction: float = Field(..., ge=0.078, le=2.42)
    Age: int = Field(..., ge=21, le=81)

@app.post("/predict")
async def predict_diabetes(input_data: DiabetesInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        features = np.array([[
            input_data.Pregnancies,
            input_data.Glucose,
            input_data.BloodPressure,
            input_data.SkinThickness,
            input_data.Insulin,
            input_data.BMI,
            input_data.DiabetesPedigreeFunction,
            input_data.Age
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return {
            "prediction": int(prediction),
            "status": "Diabetic" if prediction == 1 else "Not Diabetic"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 