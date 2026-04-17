import numpy as np
from scipy.stats import entropy
import joblib
from .logger import get_logs
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index"""
    breakpoints = np.histogram(expected, bins=bins)[1]
    expected_hist = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_hist = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_hist = np.where(expected_hist == 0, 1e-6, expected_hist)
    actual_hist = np.where(actual_hist == 0, 1e-6, actual_hist)

    psi = np.sum((actual_hist - expected_hist) * np.log(actual_hist / expected_hist))
    return psi

def calculate_kl(expected, actual, bins=10):
    """Calculate KL Divergence"""
    breakpoints = np.histogram(expected, bins=bins)[1]
    expected_hist = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_hist = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_hist = np.where(expected_hist == 0, 1e-6, expected_hist)
    actual_hist = np.where(actual_hist == 0, 1e-6, actual_hist)

    kl = entropy(actual_hist, expected_hist)
    return kl

def compute_drift():
    """Compute drift scores for Amount and confidence distributions"""
    # Load baseline data
    data_path = os.path.join(BASE_DIR, 'data', 'processed_data.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    X_train, _, X_val, y_val, _, _, _ = joblib.load(data_path)
    model = joblib.load(model_path)

    baseline_amount = X_train[:, 28]  # Amount column
    baseline_confidence = model.predict_proba(X_val)[:, 1]  # Confidence on val set

    # Load logs
    logs_df = get_logs()
    if logs_df.empty:
        return {
            'amount_psi': 0.0,
            'amount_kl': 0.0,
            'confidence_psi': 0.0,
            'confidence_kl': 0.0
        }

    # Extract current data from logs
    current_amount = np.array([features[28] for features in logs_df['features']])
    current_confidence = logs_df['confidence'].values

    # Compute drift metrics
    amount_psi = calculate_psi(baseline_amount, current_amount)
    amount_kl = calculate_kl(baseline_amount, current_amount)
    confidence_psi = calculate_psi(baseline_confidence, current_confidence)
    confidence_kl = calculate_kl(baseline_confidence, current_confidence)

    return {
        'amount_psi': float(amount_psi),
        'amount_kl': float(amount_kl),
        'confidence_psi': float(confidence_psi),
        'confidence_kl': float(confidence_kl)
    }