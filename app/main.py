from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

app = FastAPI(
    title="Telco Customer Churn API",
    description="Predict customer churn using the trained Telco churn model.",
    version="1.0.0",
)


class CustomerDetails(BaseModel):
    # Example request:
    # {
    #   "gender": "Female",
    #   "SeniorCitizen": 0,
    #   "Partner": "Yes",
    #   "Dependents": "No",
    #   "tenure": 12,
    #   "PhoneService": "Yes",
    #   "MultipleLines": "No",
    #   "InternetService": "Fiber optic",
    #   "OnlineSecurity": "No",
    #   "OnlineBackup": "Yes",
    #   "DeviceProtection": "No",
    #   "TechSupport": "No",
    #   "StreamingTV": "Yes",
    #   "StreamingMovies": "Yes",
    #   "Contract": "Month-to-month",
    #   "PaperlessBilling": "Yes",
    #   "PaymentMethod": "Electronic check",
    #   "MonthlyCharges": 85.5,
    #   "TotalCharges": 1026.0
    # }
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


def get_risk_label(churn_probability: float) -> str:
    if churn_probability >= 0.6:
        return "High"
    if churn_probability >= 0.4:
        return "Medium"
    return "Low"


def customer_to_dict(customer: CustomerDetails) -> dict:
    if hasattr(customer, "model_dump"):
        return customer.model_dump()

    return customer.dict()


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_missing",
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(customer: CustomerDetails):
    if model is None:
        raise HTTPException(status_code=503, detail="Trained model is not available.")

    try:
        input_data = pd.DataFrame([customer_to_dict(customer)])
        prediction = int(model.predict(input_data)[0])
        churn_probability = float(model.predict_proba(input_data)[0][1])

        return {
            "churn_prediction": prediction,
            "churn_probability": round(churn_probability, 4),
            "risk_label": get_risk_label(churn_probability),
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error
