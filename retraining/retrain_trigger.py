import joblib
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from api.monitoring.drift import compute_drift

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_REGISTRY_PATH = os.path.join(BASE_DIR, 'models', 'model_registry.json')
SHADOW_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_shadow.pkl')
PRODUCTION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.pkl')

def load_registry():
    if os.path.exists(MODEL_REGISTRY_PATH):
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return {"versions": []}

def save_registry(registry):
    with open(MODEL_REGISTRY_PATH, 'w') as f:
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

def shadow_deploy():
    # Shadow model is already saved, just confirm
    if os.path.exists(SHADOW_MODEL_PATH):
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
    retrain_model()
    shadow_deploy()
    log_retrain_event("Manual retrain")

def promote_shadow():
    if not os.path.exists(SHADOW_MODEL_PATH):
        return {"error": "No shadow model to promote"}

    # Swap files
    os.rename(PRODUCTION_MODEL_PATH, PRODUCTION_MODEL_PATH + '.bak')
    os.rename(SHADOW_MODEL_PATH, PRODUCTION_MODEL_PATH)
    os.rename(PRODUCTION_MODEL_PATH + '.bak', SHADOW_MODEL_PATH)

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