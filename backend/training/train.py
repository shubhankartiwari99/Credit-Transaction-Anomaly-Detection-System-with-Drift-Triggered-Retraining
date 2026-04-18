import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

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
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    print("Validation Classification Report:")
    print(classification_report(y_val, y_pred))
    auc = roc_auc_score(y_val, y_pred_proba)
    print(f"Validation AUC: {auc:.4f}")

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
