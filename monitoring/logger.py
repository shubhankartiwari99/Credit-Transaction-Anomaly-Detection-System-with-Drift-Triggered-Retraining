import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_FILE = os.path.join(BASE_DIR, 'data', 'logs.parquet')

def log_prediction(features, prediction, confidence, shadow_prediction=None, shadow_confidence=None):
    """
    Log a prediction to Parquet file.
    features: list of 30 floats
    prediction: int (0 or 1)
    confidence: float (probability of fraud)
    shadow_prediction: int or None
    shadow_confidence: float or None
    """
    timestamp = datetime.now()
    data = {
        'timestamp': [timestamp],
        'features': [features],
        'prediction': [prediction],
        'confidence': [confidence],
        'shadow_prediction': [shadow_prediction],
        'shadow_confidence': [shadow_confidence]
    }
    df = pd.DataFrame(data)

    # Append to existing parquet or create new
    if os.path.exists(LOG_FILE):
        existing_df = pd.read_parquet(LOG_FILE)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_parquet(LOG_FILE, index=False)
    print(f"Logged prediction to {LOG_FILE}")

def get_logs():
    """Retrieve all logs as DataFrame"""
    if os.path.exists(LOG_FILE):
        return pd.read_parquet(LOG_FILE)
    return pd.DataFrame()