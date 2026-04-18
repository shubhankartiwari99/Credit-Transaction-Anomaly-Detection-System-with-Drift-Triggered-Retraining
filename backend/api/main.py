from contextlib import asynccontextmanager
import subprocess
import sys

from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = BASE_DIR / "data" / "logs.parquet"
RAW_DATA_FILE = BASE_DIR / "data" / "creditcard.csv"
PROCESSED_DATA_FILE = BASE_DIR / "data" / "processed_data.pkl"
MODEL_FILE = BASE_DIR / "models" / "model.pkl"


def _run_backend_script(script_path: Path):
    subprocess.run([sys.executable, str(script_path)], cwd=BASE_DIR, check=True)


def ensure_runtime_artifacts():
    if MODEL_FILE.exists():
        return

    BASE_DIR.joinpath("models").mkdir(exist_ok=True)

    if not PROCESSED_DATA_FILE.exists():
        if not RAW_DATA_FILE.exists():
            raise RuntimeError(
                f"Missing training data at {RAW_DATA_FILE}. "
                "Add creditcard.csv before starting the API without a prebuilt model."
            )
        print("==> processed_data.pkl not found, preprocessing now...")
        _run_backend_script(BASE_DIR / "training" / "preprocess.py")
        print("==> Preprocessing complete")

    print("==> model.pkl not found, training now...")
    _run_backend_script(BASE_DIR / "training" / "train.py")
    print("==> Training complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_artifacts()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Fraud Detection API",
    description="API for detecting credit card fraud",
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

@app.post("/retrain")
def retrain():
    from retraining.retrain_trigger import force_retrain

    force_retrain()
    return {"message": "Retraining triggered"}

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
