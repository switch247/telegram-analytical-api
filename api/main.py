import pandas as pd
import mlflow.sklearn
import joblib
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import text
from .database import engine
from .schemas import TopProduct, ChannelActivity, MessageSearchResult, VisualContentStats
from typing import List
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


@app.get("/api/reports/top-products", response_model=List[TopProduct])
def get_top_products(limit: int = 10):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT word AS product, COUNT(*) AS count
            FROM (
                SELECT unnest(string_to_array(lower(message_text), ' ')) AS word
                FROM medical_warehouse.marts.core.fct_messages
            ) sub
            WHERE word ~ '^[a-zA-Z]+'
            GROUP BY word
            ORDER BY count DESC
            LIMIT :limit
        """), {"limit": limit})
        return [TopProduct(product=row[0], count=row[1]) for row in result]


@app.get("/api/channels/{channel_name}/activity", response_model=ChannelActivity)
def get_channel_activity(channel_name: str):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT channel_name, total_posts, avg_views, first_post_date, last_post_date
            FROM medical_warehouse.marts.core.dim_channels
            WHERE channel_name = :channel_name
        """), {"channel_name": channel_name}).fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Channel not found")
        return ChannelActivity(
            channel_name=result[0],
            total_posts=result[1],
            avg_views=result[2],
            first_post_date=str(result[3]),
            last_post_date=str(result[4])
        )


@app.get("/api/search/messages", response_model=List[MessageSearchResult])
def search_messages(query: str, limit: int = 20):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT message_id, channel_name, message_text, post_date, view_count, forward_count, has_image
            FROM medical_warehouse.marts.core.fct_messages
            WHERE message_text ILIKE :query
            LIMIT :limit
        """), {"query": f'%{query}%', "limit": limit})
        return [MessageSearchResult(
            message_id=row[0],
            channel_name=row[1],
            message_text=row[2],
            post_date=str(row[3]),
            view_count=row[4],
            forward_count=row[5],
            has_image=row[6]
        ) for row in result]


@app.get("/api/reports/visual-content", response_model=List[VisualContentStats])
def get_visual_content_stats():
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT channel_name,
                SUM(CASE WHEN image_category = 'promotional' THEN 1 ELSE 0 END) AS promotional_count,
                SUM(CASE WHEN image_category = 'product_display' THEN 1 ELSE 0 END) AS product_display_count,
                SUM(CASE WHEN image_category = 'lifestyle' THEN 1 ELSE 0 END) AS lifestyle_count,
                SUM(CASE WHEN image_category = 'other' THEN 1 ELSE 0 END) AS other_count
            FROM processed.yolo_detections
            GROUP BY channel_name
        """))
        return [VisualContentStats(
            channel_name=row[0],
            promotional_count=row[1],
            product_display_count=row[2],
            lifestyle_count=row[3],
            other_count=row[4]
        ) for row in result]
