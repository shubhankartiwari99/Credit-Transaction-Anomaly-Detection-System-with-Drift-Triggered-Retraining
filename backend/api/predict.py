import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SHADOW_MODEL_PATH = BASE_DIR / "models" / "model_shadow.pkl"
DATA_PATH = BASE_DIR / "data" / "processed_data.pkl"

model = None
model_mtime = None
scaler = None
data_mtime = None
shadow_model = None
shadow_model_mtime = None


def get_model():
    global model
    global model_mtime

    current_mtime = MODEL_PATH.stat().st_mtime
    if model is None or model_mtime != current_mtime:
        model = joblib.load(MODEL_PATH)
        model_mtime = current_mtime
    return model


def get_scaler():
    global scaler
    global data_mtime

    current_mtime = DATA_PATH.stat().st_mtime
    if scaler is None or data_mtime != current_mtime:
        _, _, _, _, _, _, scaler = joblib.load(DATA_PATH)
        data_mtime = current_mtime
    return scaler


def get_shadow_model():
    global shadow_model
    global shadow_model_mtime

    if not SHADOW_MODEL_PATH.exists():
        shadow_model = None
        shadow_model_mtime = None
        return None

    current_mtime = SHADOW_MODEL_PATH.stat().st_mtime
    if shadow_model is None or shadow_model_mtime != current_mtime:
        shadow_model = joblib.load(SHADOW_MODEL_PATH)
        shadow_model_mtime = current_mtime
    return shadow_model


def predict_fraud(features):
    """
    Predict fraud for a single transaction.
    features: list of 30 floats [V1, V2, ..., V28, Amount, Time]
    """
    features = np.array(features).reshape(1, -1)
    # Scale Amount and Time (indices 28 and 29)
    features[0, 28:30] = get_scaler().transform(features[0, 28:30].reshape(1, -1))
    production_model = get_model()
    prediction = production_model.predict(features)[0]
    probability = production_model.predict_proba(features)[0, 1]
    result = {
        "prediction": int(prediction),
        "probability": float(probability),
    }

    # Shadow model prediction
    candidate_shadow_model = get_shadow_model()
    if candidate_shadow_model is not None:
        shadow_pred = candidate_shadow_model.predict(features)[0]
        shadow_proba = candidate_shadow_model.predict_proba(features)[0, 1]
        result["shadow_prediction"] = int(shadow_pred)
        result["shadow_confidence"] = float(shadow_proba)

    return result
