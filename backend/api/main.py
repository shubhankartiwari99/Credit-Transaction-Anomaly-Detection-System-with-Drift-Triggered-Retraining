from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = BASE_DIR / "data" / "logs.parquet"


def seed_logs_if_empty():
    import numpy as np
    import pandas as pd

    if LOG_FILE.exists():
        try:
            if not pd.read_parquet(LOG_FILE).empty:
                return
        except Exception:
            pass

    print("==> Seeding logs with sample predictions...")
    LOG_FILE.parent.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    base_time = datetime.utcnow() - timedelta(hours=2)
    rows = []
    for index in range(50):
        amount = float(rng.exponential(80.0))
        event_time = float(index * 120.0)
        features = rng.normal(0.0, 1.0, 30).tolist()
        features[28] = amount
        features[29] = event_time

        prediction = int(rng.choice([0, 1], p=[0.96, 0.04]))
        if prediction == 1:
            confidence = float(rng.uniform(0.75, 0.98))
        else:
            confidence = float(rng.uniform(0.01, 0.18))

        shadow_prediction = prediction if rng.random() < 0.85 else 1 - prediction
        shadow_confidence = float(np.clip(confidence + rng.normal(0.0, 0.08), 0.01, 0.99))

        rows.append(
            {
                "timestamp": base_time + timedelta(minutes=index * 2),
                "features": features,
                "prediction": prediction,
                "confidence": confidence,
                "shadow_prediction": shadow_prediction,
                "shadow_confidence": shadow_confidence,
            }
        )

    pd.DataFrame(rows).to_parquet(LOG_FILE, index=False)
    print("==> Seeded 50 sample log entries")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.predict import get_model

    print("==> Loading model...")
    get_model()
    seed_logs_if_empty()
    print("==> Model loaded, API ready")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Fraud Detection API",
    description="API for detecting credit card fraud",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionFeatures(BaseModel):
    features: list[float]  # List of 30 features: V1 to V28, Amount, Time

@app.post("/predict")
def predict(transaction: TransactionFeatures):
    from api.predict import predict_fraud
    from monitoring.logger import log_prediction

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
    from monitoring.drift import compute_drift

    drift_scores = compute_drift()
    return {"drift_scores": drift_scores}

@app.get("/drift")
def get_drift():
    from monitoring.drift import compute_drift
    from datetime import datetime

    report = compute_drift()
    drift_score = float(max(report.get("amount_psi", 0), report.get("confidence_kl", 0)))
    threshold = 0.1
    status = "HIGH" if drift_score > threshold else "LOW"

    return {
        "drift_score": drift_score,
        "status": status,
        "threshold": threshold,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/drift/history")
def get_drift_history():
    from monitoring.drift import load_drift_history
    return load_drift_history()

@app.post("/retrain")
def retrain():
    try:
        from retraining.retrain_trigger import force_retrain

        result = force_retrain()
        return result
    except Exception as e:
        # Return 200 with error info so CORS headers are included
        return {"status": "error", "message": str(e)}

@app.get("/retrain/status")
def retrain_status():
    from retraining.retrain_trigger import load_retrain_status
    return load_retrain_status()

@app.post("/promote")
def promote():
    from retraining.retrain_trigger import promote_shadow

    return promote_shadow()

@app.get("/registry")
def registry():
    from retraining.retrain_trigger import load_registry

    return load_registry()

@app.get("/predictions")
def get_predictions(limit: int = 100):
    import pandas as pd

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

@app.get("/health")
def health():
    from monitoring.drift import compute_drift
    from retraining.retrain_trigger import load_registry

    report = compute_drift()
    drift_score = float(max(report.get("amount_psi", 0), report.get("confidence_kl", 0)))
    drift_status = "high" if drift_score > 0.1 else "low"

    registry = load_registry()
    active_version = "v1"
    for v in registry.get("versions", []):
        if v.get("status") == "production":
            active_version = f"v{v.get('version')}"

    return {
        "status": "ok",
        "model": active_version,
        "drift": drift_status
    }
