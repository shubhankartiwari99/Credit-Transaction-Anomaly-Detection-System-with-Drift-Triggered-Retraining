import numpy as np
import joblib
import pandas as pd
from scipy.stats import entropy
from .logger import get_logs
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = BASE_DIR / "data" / "baseline_distribution.parquet"
DATA_PATH = BASE_DIR / "data" / "processed_data.pkl"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


def empty_drift_report():
    return {
        "amount_psi": 0.0,
        "amount_kl": 0.0,
        "confidence_psi": 0.0,
        "confidence_kl": 0.0,
    }


def get_amount_series(values):
    if hasattr(values, "columns"):
        if "Amount" in values.columns:
            return values["Amount"].to_numpy()
        return values.iloc[:, 28].to_numpy()
    return np.asarray(values)[:, 28]


def get_amount_time_matrix(values):
    if hasattr(values, "columns"):
        if {"Amount", "Time"}.issubset(values.columns):
            return values[["Amount", "Time"]].to_numpy()
        return values.iloc[:, [28, 29]].to_numpy()
    return np.asarray(values)[:, 28:30]


def load_baseline_distribution():
    if BASELINE_PATH.exists():
        try:
            baseline_df = pd.read_parquet(BASELINE_PATH)
            if not baseline_df.empty and {"amount", "confidence"}.issubset(baseline_df.columns):
                return baseline_df["amount"].to_numpy(), baseline_df["confidence"].to_numpy()
        except Exception:
            pass

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        return None, None

    try:
        _, _, X_val, _, _, _, scaler = joblib.load(DATA_PATH)
        model = joblib.load(MODEL_PATH)
    except Exception:
        return None, None

    raw_amount = scaler.inverse_transform(get_amount_time_matrix(X_val))[:, 0]
    baseline_confidence = model.predict_proba(X_val)[:, 1]
    return raw_amount, baseline_confidence

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
    baseline_amount, baseline_confidence = load_baseline_distribution()
    if baseline_amount is None or baseline_confidence is None:
        return empty_drift_report()

    # Load logs
    logs_df = get_logs()
    if logs_df.empty or "features" not in logs_df or "confidence" not in logs_df:
        return empty_drift_report()

    # Extract current data from logs
    current_amount = np.array(
        [
            features[28]
            for features in logs_df["features"]
            if isinstance(features, (list, tuple, np.ndarray)) and len(features) > 28
        ]
    )
    current_confidence = logs_df["confidence"].dropna().values

    if current_amount.size == 0 or current_confidence.size == 0:
        return empty_drift_report()

    # Compute drift metrics
    amount_psi = calculate_psi(baseline_amount, current_amount)
    amount_kl = calculate_kl(baseline_amount, current_amount)
    confidence_psi = calculate_psi(baseline_confidence, current_confidence)
    confidence_kl = calculate_kl(baseline_confidence, current_confidence)

    return {
        "amount_psi": float(amount_psi),
        "amount_kl": float(amount_kl),
        "confidence_psi": float(confidence_psi),
        "confidence_kl": float(confidence_kl),
    }
