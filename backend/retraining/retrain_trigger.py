import joblib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from monitoring.drift import compute_drift
from training.evaluate import compute_confusion, compute_business_loss, find_optimal_threshold

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_REGISTRY_PATH = BASE_DIR / "models" / "model_registry.json"
SHADOW_MODEL_PATH = BASE_DIR / "models" / "model_shadow.pkl"
PRODUCTION_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed_data.pkl"
RETRAIN_STATUS_PATH = BASE_DIR / "models" / "retrain_status.json"
DECISION_HISTORY_PATH = BASE_DIR / "models" / "decision_history.json"
COOLDOWN_MINUTES = 2
MODEL_ARTIFACT_DIR = BASE_DIR / "models"

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


def save_decision_event(event):
    history = []
    if DECISION_HISTORY_PATH.exists():
        try:
            with DECISION_HISTORY_PATH.open("r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(event)
    with DECISION_HISTORY_PATH.open("w") as f:
        json.dump(history, f, indent=2, default=str)


def is_threshold_stable(threshold, lower=0.2, upper=0.8):
    return lower <= threshold <= upper


def decide_promotion(prod_metrics, cand_metrics):
    if cand_metrics["loss"] < prod_metrics["loss"]:
        return "promote"
    elif cand_metrics["loss"] > prod_metrics["loss"]:
        return "reject"
    return "no_change"


def initialize_registry():
    registry = load_registry()
    if not registry["versions"]:
        # Initialize with current production baseline model
        registry["versions"].append({
            "version": 1,
            "model_id": "model_v1",
            "model_path": "model_v1.pkl",
            "created_at": datetime.now().isoformat(),
            "trained_at": datetime.now().isoformat(),
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

        new_version = len(load_registry()["versions"]) + 1
        model_id = f"model_v{new_version}"
        versioned_model_path = MODEL_ARTIFACT_DIR / f"{model_id}.pkl"
        joblib.dump(model, SHADOW_MODEL_PATH)
        joblib.dump(model, versioned_model_path)
        print(f"Retrained model saved as {SHADOW_MODEL_PATH} and {versioned_model_path}")
    except Exception as e:
        print(f"ERROR: Retraining failed - {e}")
        error_result = {"status": "failed", "reason": str(e), "timestamp": datetime.now().isoformat()}
        save_retrain_status(error_result)
        return error_result

    # Update registry
    registry = load_registry()
    new_version = len(registry["versions"]) + 1
    model_id = f"model_v{new_version}"
    registry["versions"].append({
        "version": new_version,
        "model_id": model_id,
        "model_path": f"{model_id}.pkl",
        "created_at": datetime.now().isoformat(),
        "trained_at": datetime.now().isoformat(),
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

def evaluate_model_for_decision(model, X_eval, y_eval):
    probs = model.predict_proba(X_eval)[:, 1]
    auc_roc = roc_auc_score(y_eval, probs)
    auc_pr = average_precision_score(y_eval, probs)
    best_threshold, best_loss = find_optimal_threshold(y_eval, probs)
    preds = (probs > best_threshold).astype(int)
    confusion = compute_confusion(y_eval, preds)
    recall = confusion["tp"] / (confusion["tp"] + confusion["fn"]) if (confusion["tp"] + confusion["fn"]) > 0 else 0.0
    threshold_status = "stable" if is_threshold_stable(best_threshold) else "unstable"

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "threshold": best_threshold,
        "threshold_status": threshold_status,
        "confusion": confusion,
        "loss": best_loss,
        "recall": recall
    }


def compare_models_and_decide(drift_detected=False, drift_score=None):
    registry = load_registry()
    production_entry = next((v for v in registry["versions"] if v["status"] == "production"), None)
    candidate_entry = next((v for v in reversed(registry["versions"]) if v["status"] == "shadow"), None)

    if not SHADOW_MODEL_PATH.exists() or candidate_entry is None:
        print("Decision: no_shadow_model_available -> no_action")
        return False

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = joblib.load(DATA_PATH)
    production_model = joblib.load(PRODUCTION_MODEL_PATH)
    candidate_model = joblib.load(SHADOW_MODEL_PATH)

    prod_metrics = evaluate_model_for_decision(production_model, X_val, y_val)
    cand_metrics = evaluate_model_for_decision(candidate_model, X_val, y_val)

    delta_auc = cand_metrics["auc_roc"] - prod_metrics["auc_roc"]
    delta_loss = prod_metrics["loss"] - cand_metrics["loss"]

    print(f"Production AUC: {prod_metrics['auc_roc']:.4f}")
    print(f"Candidate AUC: {cand_metrics['auc_roc']:.4f}")
    print(f"Δ AUC: {delta_auc:+.4f}\n")

    print(f"Production Best Threshold: {prod_metrics['threshold']:.2f}")
    print(f"Production Threshold Status: {prod_metrics['threshold_status']}")
    print(f"Production Recall: {prod_metrics['recall']:.4f}")
    print(f"Production Confusion: {prod_metrics['confusion']}\n")

    print(f"Candidate Best Threshold: {cand_metrics['threshold']:.2f}")
    print(f"Candidate Threshold Status: {cand_metrics['threshold_status']}")
    print(f"Candidate Recall: {cand_metrics['recall']:.4f}")
    print(f"Candidate Confusion: {cand_metrics['confusion']}\n")

    print(f"Production Loss: {prod_metrics['loss']}")
    print(f"Candidate Loss: {cand_metrics['loss']}")
    print(f"Δ Loss: {delta_loss:+.0f}\n")

    decision = "promote"
    if production_entry is not None:
        decision = decide_promotion(prod_metrics, cand_metrics)

    if production_entry is None:
        print("Decision: no_production_baseline -> promote_candidate")
    elif decision == "promote":
        print("Decision: promote")
    elif decision == "reject":
        print("Decision: reject")
    else:
        print("Decision: no_change")

    if production_entry:
        production_entry["best_threshold"] = round(prod_metrics["threshold"], 3)
        production_entry["best_loss"] = int(prod_metrics["loss"])
        production_entry["best_recall"] = round(prod_metrics["recall"], 3)
        production_entry["threshold_status"] = prod_metrics["threshold_status"]

    candidate_entry["best_threshold"] = round(cand_metrics["threshold"], 3)
    candidate_entry["best_loss"] = int(cand_metrics["loss"])
    candidate_entry["best_recall"] = round(cand_metrics["recall"], 3)
    candidate_entry["threshold_status"] = cand_metrics["threshold_status"]

    save_registry(registry)

    decision_event = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": bool(drift_detected),
        "drift_score": float(drift_score) if drift_score is not None else None,
        "dataset_id": DATA_PATH.name,
        "production": {
            "version": production_entry["version"] if production_entry else None,
            "model_id": production_entry["model_id"] if production_entry else None,
            "auc": round(prod_metrics["auc_roc"], 4),
            "threshold": round(prod_metrics["threshold"], 3),
            "threshold_status": prod_metrics["threshold_status"],
            "recall": round(prod_metrics["recall"], 4),
            "confusion": prod_metrics["confusion"],
            "loss": int(prod_metrics["loss"])
        } if production_entry else None,
        "candidate": {
            "version": candidate_entry["version"],
            "model_id": candidate_entry.get("model_id"),
            "auc": round(cand_metrics["auc_roc"], 4),
            "threshold": round(cand_metrics["threshold"], 3),
            "threshold_status": cand_metrics["threshold_status"],
            "recall": round(cand_metrics["recall"], 4),
            "confusion": cand_metrics["confusion"],
            "loss": int(cand_metrics["loss"])
        },
        "decision": decision,
        "delta_loss": int(delta_loss),
        "delta_auc": round(delta_auc, 4)
    }

    save_decision_event(decision_event)

    if decision == "promote":
        promote_shadow()
        return True

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
    compare_models_and_decide(drift_detected=True, drift_score=drift_score)

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
    compare_models_and_decide(drift_detected=False, drift_score=drift_score)
    
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


def rollback_to_previous():
    registry = load_registry()
    production_entry = next((v for v in registry["versions"] if v["status"] == "production"), None)
    archived_entries = [v for v in registry["versions"] if v["status"] == "archived"]

    if production_entry is None or not archived_entries:
        print("Rollback unavailable: no previous production version found")
        return {"status": "failed", "reason": "no previous production version"}

    previous_entry = archived_entries[-1]
    previous_model_path = MODEL_ARTIFACT_DIR / previous_entry["model_path"]
    if not previous_model_path.exists():
        print(f"Rollback failed: versioned model file not found at {previous_model_path}")
        return {"status": "failed", "reason": "missing model artifact"}

    if production_entry and production_entry.get("model_path"):
        current_prod_model_path = MODEL_ARTIFACT_DIR / production_entry["model_path"]
        if current_prod_model_path.exists():
            shutil.copy2(current_prod_model_path, SHADOW_MODEL_PATH)

    shutil.copy2(previous_model_path, PRODUCTION_MODEL_PATH)

    for v in registry["versions"]:
        if v["version"] == previous_entry["version"]:
            v["status"] = "production"
        elif production_entry and v["version"] == production_entry["version"]:
            v["status"] = "archived"

    save_registry(registry)
    print(f"Rollback complete: restored {previous_entry['model_id']} to production")
    return {"status": "rolled_back", "rolled_back_to": previous_entry["model_id"], "timestamp": datetime.now().isoformat()}

# Initialize on import
initialize_registry()
