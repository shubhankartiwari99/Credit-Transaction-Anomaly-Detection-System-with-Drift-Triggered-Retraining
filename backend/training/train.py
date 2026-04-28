import joblib
import json
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parents[1]


def get_amount_time_matrix(values):
    if hasattr(values, "columns"):
        if {"Amount", "Time"}.issubset(values.columns):
            return values[["Amount", "Time"]].to_numpy()
        return values.iloc[:, [28, 29]].to_numpy()
    return values[:, 28:30]

def train():
    # Load processed data
    data_path = BASE_DIR / "data" / "processed_data.pkl"
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = joblib.load(data_path)

    # Train model
    model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    print("Validation Classification Report:")
    print(classification_report(y_val, y_pred))
    auc = roc_auc_score(y_val, y_pred_proba)
    auc_pr = average_precision_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"Validation AUC-ROC: {auc:.4f}")
    print(f"Validation AUC-PR: {auc_pr:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    
    # Update registry with initial metrics if possible
    registry_path = BASE_DIR / "models" / "model_registry.json"
    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
            if registry.get("versions"):
                registry["versions"][0]["auc_roc"] = round(auc, 3)
                registry["versions"][0]["auc_pr"] = round(auc_pr, 3)
                registry["versions"][0]["precision"] = round(precision, 3)
                registry["versions"][0]["recall"] = round(recall, 3)
                registry["versions"][0]["f1"] = round(f1, 3)
                with open(registry_path, "w") as f:
                    json.dump(registry, f, indent=2, default=str)
        except Exception as e:
            print(f"Could not update registry: {e}")

    # Save model
    model_path = BASE_DIR / "models" / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save baseline distribution for drift monitoring
    raw_amount = scaler.inverse_transform(get_amount_time_matrix(X_val))[:, 0]
    baseline = pd.DataFrame({
        "amount": raw_amount,
        "confidence": y_pred_proba,
    })
    baseline_path = BASE_DIR / "data" / "baseline_distribution.parquet"
    baseline.to_parquet(baseline_path, index=False)
    print(f"Baseline distribution saved to {baseline_path}")

if __name__ == '__main__':
    train()
