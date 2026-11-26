import pickle
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready REST API serving churn classification.",
    version="1.0.0",
)

# Load the existing pre-saved model and encoders
with open("customer_churn_model.pkl", "rb") as f:
  data = pickle.load(f)
  model = data["model"]
  feature_columns = data["features_names"]

with open("encoders.pkl", "rb") as f:
  encoders = pickle.load(f)

# The 19 training feature columns
feature_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class CustomerPayload(BaseModel):
  gender: str = Field(..., example="Female")
  SeniorCitizen: int = Field(..., example=0)
  Partner: str = Field(..., example="Yes")
  Dependents: str = Field(..., example="No")
  tenure: int = Field(..., example=12)
  PhoneService: str = Field(..., example="Yes")
  MultipleLines: str = Field(..., example="No")
  InternetService: str = Field(..., example="DSL")
  OnlineSecurity: str = Field(..., example="No")
  OnlineBackup: str = Field(..., example="Yes")
  DeviceProtection: str = Field(..., example="No")
  TechSupport: str = Field(..., example="No")
  StreamingTV: str = Field(..., example="No")
  StreamingMovies: str = Field(..., example="No")
  Contract: str = Field(..., example="Month-to-month")
  PaperlessBilling: str = Field(..., example="Yes")
  PaymentMethod: str = Field(..., example="Electronic check")
  MonthlyCharges: float = Field(..., example=55.20)
  TotalCharges: float = Field(..., example=662.40)


@app.get("/")
def root():
  return {
      "service": "Customer Churn Prediction API",
      "status": "healthy",
      "documentation": "/docs",
  }


@app.post("/predict")
def predict_churn(customer: CustomerPayload):
  try:
    df = pd.DataFrame([customer.dict()])

    if isinstance(encoders, dict):
      for col, enc in encoders.items():
        if col in df.columns:
          df[col] = enc.transform(df[col])

    df = df[feature_columns]

    churn_pred = int(model.predict(df)[0])
    churn_prob = (
        float(model.predict_proba(df)[0][1])
        if hasattr(model, "predict_proba")
        else float(churn_pred)
    )

    return {
        "churn_prediction": churn_pred,
        "churn_risk": "High" if churn_pred == 1 else "Low",
        "churn_probability": round(churn_prob, 4),
    }

  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))