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


# Load the best model from MLflow (highest test_accuracy)
mlflow.set_experiment("diabetes-prediction")
experiment = mlflow.get_experiment_by_name("diabetes-prediction")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.test_accuracy DESC"])
best_run_id = runs.iloc[0]["run_id"]
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_run_id": best_run_id}


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