import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SHADOW_MODEL_PATH = BASE_DIR / "models" / "model_shadow.pkl"
DATA_PATH = BASE_DIR / "data" / "processed_data.pkl"

# Load model and scaler
model = joblib.load(MODEL_PATH)
_, _, _, _, _, _, scaler = joblib.load(DATA_PATH)

# Load shadow model if exists
shadow_model = None
if SHADOW_MODEL_PATH.exists():
    shadow_model = joblib.load(SHADOW_MODEL_PATH)

def predict_fraud(features):
    """
    Predict fraud for a single transaction.
    features: list of 30 floats [V1, V2, ..., V28, Amount, Time]
    """
    features = np.array(features).reshape(1, -1)
    # Scale Amount and Time (indices 28 and 29)
    features[0, 28:30] = scaler.transform(features[0, 28:30].reshape(1, -1))
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]
    result = {
        "prediction": int(prediction),
        "probability": float(probability),
    }

    # Shadow model prediction
    if shadow_model is not None:
        shadow_pred = shadow_model.predict(features)[0]
        shadow_proba = shadow_model.predict_proba(features)[0, 1]
        result["shadow_prediction"] = int(shadow_pred)
        result["shadow_confidence"] = float(shadow_proba)

    return result
