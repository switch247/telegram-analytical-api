import pandas as pd
import mlflow.sklearn
import joblib
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import TransactionInput, PredictionOutput
import sys
from pathlib import Path

# Add project root to path to allow imports from src if needed
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

app = FastAPI(title="Credit Risk Fraud Detection API")

# Set tracking URI - assuming mlruns is in the root of the app
# In Docker, we'll set WORKDIR to /app and copy mlruns there
mlflow.set_tracking_uri("file:./mlruns")

model_name = "Credit_Risk_Fraud_Detection_best_model"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"

model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Attempting to load model from {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully from MLflow.")
    except Exception as e:
        print(f"Failed to load from MLflow: {e}")
        fallback_path = project_root / "outputs" / "models" / "model.pkl"
        print(f"Attempting fallback load from {fallback_path}...")
        try:
            model = joblib.load(fallback_path)
            print("Model loaded successfully from fallback path.")
        except Exception as e2:
            print(f"Error loading model from fallback: {e2}")
            # In production, we might want to fail startup, but for dev log it
            pass


@app.get("/")
def read_root():
    return {"message": "Credit Risk Fraud Detection API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert input to DataFrame
    # Pydantic v2 uses model_dump()
    data = transaction.model_dump()
    df = pd.DataFrame([data])

    # Predict
    try:
        prediction = model.predict(df)[0]
        # Check if model supports predict_proba
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df)[0][1]
        else:
            probability = 0.0  # Or handle appropriately

        return {
            "is_fraud": bool(prediction),
            "probability": float(probability),
            "model_version": model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
