import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Diabetes Prediction API")


class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str


# Load the exported model from the fixed local folder
MODEL_DIR = "model"
model = mlflow.sklearn.load_model(MODEL_DIR)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_source": MODEL_DIR}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    input_data = pd.DataFrame([patient.model_dump()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
        status="diabetic" if prediction == 1 else "healthy",
    )