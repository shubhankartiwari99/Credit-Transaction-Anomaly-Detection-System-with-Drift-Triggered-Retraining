import joblib
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score

def evaluate():
    # Load processed data and model
    data_path = '../data/processed_data.pkl'
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = joblib.load(data_path)

    model_path = '../models/model.pkl'
    model = joblib.load(model_path)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Test Classification Report:")
    print(classification_report(y_test, y_pred))

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC-ROC: {auc_roc:.4f}")

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    print(f"Test AUC-PR: {auc_pr:.4f}")

    if auc_pr > 0.70:
        print("✅ AUC-PR > 0.70: Solid performance!")
    else:
        print("⚠️ AUC-PR < 0.70: May need model tuning.")

if __name__ == '__main__':
    evaluate()