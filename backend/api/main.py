from fastapi import FastAPI
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

from api.predict import predict_fraud
from monitoring.drift import compute_drift
from monitoring.logger import log_prediction
from retraining.retrain_trigger import force_retrain, promote_shadow, load_registry

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = BASE_DIR / "data" / "logs.parquet"

app = FastAPI(title="Fraud Detection API", description="API for detecting credit card fraud")

class TransactionFeatures(BaseModel):
    features: list[float]  # List of 30 features: V1 to V28, Amount, Time

@app.post("/predict")
def predict(transaction: TransactionFeatures):
    if len(transaction.features) != 30:
        return {"error": "Exactly 30 features required: V1-V28, Amount, Time"}
    result = predict_fraud(transaction.features)
    log_prediction(
        transaction.features,
        result["prediction"],
        result["probability"],
        result.get("shadow_prediction"),
        result.get("shadow_confidence"),
    )
    return result

@app.get("/metrics")
def metrics():
    drift_scores = compute_drift()
    return {"drift_scores": drift_scores}

@app.post("/retrain")
def retrain():
    force_retrain()
    return {"message": "Retraining triggered"}

@app.post("/promote")
def promote():
    return promote_shadow()

@app.get("/registry")
def registry():
    return load_registry()

@app.get("/predictions")
def get_predictions(limit: int = 100):
    try:
        df = pd.read_parquet(LOG_FILE)
        return df.tail(limit)[
            ["timestamp", "prediction", "confidence", "shadow_prediction", "shadow_confidence"]
        ].to_dict(orient="records")
    except Exception:
        return []

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}
