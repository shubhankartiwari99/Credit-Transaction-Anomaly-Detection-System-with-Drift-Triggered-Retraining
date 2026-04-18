import joblib
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from monitoring.drift import compute_drift

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_REGISTRY_PATH = BASE_DIR / "models" / "model_registry.json"
SHADOW_MODEL_PATH = BASE_DIR / "models" / "model_shadow.pkl"
PRODUCTION_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed_data.pkl"

def load_registry():
    if MODEL_REGISTRY_PATH.exists():
        with MODEL_REGISTRY_PATH.open("r") as f:
            return json.load(f)
    return {"versions": []}

def save_registry(registry):
    with MODEL_REGISTRY_PATH.open("w") as f:
        json.dump(registry, f, indent=2, default=str)

def initialize_registry():
    registry = load_registry()
    if not registry["versions"]:
        # Initialize with current model
        registry["versions"].append({
            "version": 1,
            "trained_at": datetime.now(),
            "auc_pr": 0.831,  # From evaluation
            "trigger_reason": "baseline",
            "status": "production"
        })
        save_registry(registry)

def should_retrain(drift_report):
    # Thresholds: PSI > 0.2 or KL > 0.1
    return drift_report['amount_psi'] > 0.2 or drift_report['confidence_kl'] > 0.1

def retrain_model():
    # Guard: processed training data may not be present on the server
    if not DATA_PATH.exists():
        print("WARNING: processed_data.pkl not found, skipping retrain")
        return {"skipped": True, "reason": "training data not available on server"}

    # Retrain on existing processed data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = joblib.load(DATA_PATH)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    joblib.dump(model, SHADOW_MODEL_PATH)
    print(f"Retrained model saved as {SHADOW_MODEL_PATH}")

    # Update registry
    registry = load_registry()
    new_version = len(registry["versions"]) + 1
    registry["versions"].append({
        "version": new_version,
        "trained_at": datetime.now(),
        "auc_pr": None,  # Could compute here
        "trigger_reason": "drift",
        "status": "shadow"
    })
    save_registry(registry)
    return {"skipped": False, "version": new_version}

def shadow_deploy():
    # Shadow model is already saved, just confirm
    if SHADOW_MODEL_PATH.exists():
        print("Shadow model deployed")
    else:
        print("No shadow model found")

def log_retrain_event(reason):
    print(f"Retrained at {datetime.now()}: {reason}")

def trigger_retrain_if_needed():
    report = compute_drift()
    if should_retrain(report):
        retrain_model()
        shadow_deploy()
        log_retrain_event(f"Drift detected: PSI={report['amount_psi']:.3f}, KL={report['confidence_kl']:.3f}")
    else:
        print("No retrain needed")

def force_retrain():
    result = retrain_model()
    if result and result.get("skipped"):
        return result
    shadow_deploy()
    log_retrain_event("Manual retrain")
    return result or {"skipped": False}

def promote_shadow():
    if not SHADOW_MODEL_PATH.exists():
        return {"error": "No shadow model to promote"}

    # Swap files
    backup_path = PRODUCTION_MODEL_PATH.with_suffix(".pkl.bak")
    PRODUCTION_MODEL_PATH.rename(backup_path)
    SHADOW_MODEL_PATH.rename(PRODUCTION_MODEL_PATH)
    backup_path.rename(SHADOW_MODEL_PATH)

    # Update registry
    registry = load_registry()
    for v in registry["versions"]:
        if v["status"] == "shadow":
            v["status"] = "production"
        elif v["status"] == "production":
            v["status"] = "archived"
    save_registry(registry)

    print("Shadow model promoted to production")
    return {"message": "Shadow promoted"}

# Initialize on import
initialize_registry()
