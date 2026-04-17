import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train():
    # Load processed data
    data_path = '../data/processed_data.pkl'
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
    model_path = '../models/model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train()