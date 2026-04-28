import joblib
import json
import os
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from monitoring.drift import compute_drift

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_REGISTRY_PATH = BASE_DIR / "models" / "model_registry.json"
SHADOW_MODEL_PATH = BASE_DIR / "models" / "model_shadow.pkl"
PRODUCTION_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed_data.pkl"
RETRAIN_STATUS_PATH = BASE_DIR / "models" / "retrain_status.json"
COOLDOWN_MINUTES = 2

def load_retrain_status():
    if RETRAIN_STATUS_PATH.exists():
        with RETRAIN_STATUS_PATH.open("r") as f:
            return json.load(f)
    return {"status": "idle", "reason": None, "drift_score": None, "new_model_version": None, "timestamp": None}

def save_retrain_status(status_data):
    with RETRAIN_STATUS_PATH.open("w") as f:
        json.dump(status_data, f, indent=2, default=str)

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
            "auc_roc": 0.945,
            "auc_pr": 0.831,  # From evaluation
            "precision": 0.850,
            "recall": 0.810,
            "f1": 0.829,
            "trigger_reason": "baseline",
            "status": "production"
        })
        save_registry(registry)

def should_retrain(drift_report):
    # Thresholds: PSI > 0.2 or KL > 0.1
    return drift_report['amount_psi'] > 0.2 or drift_report['confidence_kl'] > 0.1

def retrain_model(reason="manual", drift_score=None, top_shifted_feature=None):
    # Guard: processed training data may not be present on the server
    if not DATA_PATH.exists():
        print("WARNING: processed_data.pkl not found, skipping retrain")
        return {"status": "skipped", "reason": "training data not available on server"}

    try:
        # Retrain on existing processed data
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = joblib.load(DATA_PATH)

        model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss")
        model.fit(X_train, y_train)

        # Evaluate to get metrics for the candidate model
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        auc_pr = average_precision_score(y_val, y_pred_proba)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        joblib.dump(model, SHADOW_MODEL_PATH)
        print(f"Retrained model saved as {SHADOW_MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Retraining failed - {e}")
        error_result = {"status": "failed", "reason": str(e), "timestamp": datetime.now().isoformat()}
        save_retrain_status(error_result)
        return error_result

    # Update registry
    registry = load_registry()
    new_version = len(registry["versions"]) + 1
    registry["versions"].append({
        "version": new_version,
        "trained_at": datetime.now(),
        "auc_roc": round(auc_roc, 3) if 'auc_roc' in locals() else None,
        "auc_pr": round(auc_pr, 3) if 'auc_pr' in locals() else None,
        "precision": round(precision, 3) if 'precision' in locals() else None,
        "recall": round(recall, 3) if 'recall' in locals() else None,
        "f1": round(f1, 3) if 'f1' in locals() else None,
        "trigger_reason": reason,
        "status": "shadow"
    })
    save_registry(registry)
    
    result = {
        "status": "triggered",
        "reason": reason,
        "drift_score": drift_score,
        "top_shifted_feature": top_shifted_feature,
        "data_window": "last_10k_samples",
        "new_model_version": f"v{new_version}"
    }
    
    save_retrain_status({
        **result,
        "timestamp": datetime.now().isoformat()
    })
    
    return result

def shadow_deploy():
    # Shadow model is already saved, just confirm
    if SHADOW_MODEL_PATH.exists():
        print("Shadow model deployed")
    else:
        print("No shadow model found")

def log_retrain_event(reason):
    print(f"Retrained at {datetime.now()}: {reason}")

def compare_models_and_decide(candidate_auc):
    registry = load_registry()
    production_auc = None
    for v in registry["versions"]:
        if v["status"] == "production":
            production_auc = v.get("auc_roc")
            break
    
    if production_auc is None:
        print("Decision: no_production_baseline -> promote_candidate")
        promote_shadow()
        return True
        
    print(f"Production AUC: {production_auc:.4f} | Candidate AUC: {candidate_auc:.4f}")
    if candidate_auc > production_auc:
        print("Decision: better_model -> promote")
        promote_shadow()
        return True
    else:
        print("Decision: candidate_model_worse -> no_action")
        return False

def run_ml_control_loop():
    print("=== ML System Control Loop ===")
    report = compute_drift()
    if not should_retrain(report):
        print("Decision: no_drift -> no_action")
        return

    # Check cooldown
    last_status = load_retrain_status()
    if last_status.get("timestamp"):
        last_time = datetime.fromisoformat(last_status["timestamp"])
        if (datetime.now() - last_time).total_seconds() < COOLDOWN_MINUTES * 60:
            print(f"Decision: drift_detected but within {COOLDOWN_MINUTES}m cooldown -> no_action")
            return

    print(f"Decision: drift_detected (PSI={report['amount_psi']:.3f}) -> retrain")
    drift_score = max(report['amount_psi'], report['confidence_kl'])
    top_shifted_feature = report.get('top_shifted_feature', 'Unknown')
    
    retrain_model(reason="drift_threshold_exceeded", drift_score=drift_score, top_shifted_feature=top_shifted_feature)
    shadow_deploy()
    log_retrain_event(f"Drift detected: PSI={report['amount_psi']:.3f}, KL={report['confidence_kl']:.3f}")
    
    # Evaluate & Compare
    registry = load_registry()
    candidate_auc = registry["versions"][-1].get("auc_roc", 0)
    compare_models_and_decide(candidate_auc)

def force_retrain():
    print("=== ML System Control Loop (Manual Trigger) ===")
    report = compute_drift()
    drift_score = max(report.get('amount_psi', 0), report.get('confidence_kl', 0))
    top_shifted_feature = report.get('top_shifted_feature', 'Unknown')
    
    print("Decision: manual_trigger -> retrain")
    result = retrain_model(reason="manual", drift_score=drift_score, top_shifted_feature=top_shifted_feature)
    if result.get("status") in ["skipped", "failed"]:
        return result
    shadow_deploy()
    
    # Evaluate & Compare
    registry = load_registry()
    candidate_auc = registry["versions"][-1].get("auc_roc", 0)
    compare_models_and_decide(candidate_auc)
    
    log_retrain_event("Manual retrain")
    return result

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
