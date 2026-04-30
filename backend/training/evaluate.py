import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[1]

def evaluate():
    # Load processed data and model
    data_path = BASE_DIR / "data" / "processed_data.pkl"
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = joblib.load(data_path)

    model_path = BASE_DIR / "models" / "model.pkl"
    model = joblib.load(model_path)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Test Classification Report:")
    print(classification_report(y_test, y_pred))

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    prec_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test AUC-ROC: {auc_roc:.4f}")
    print(f"Test AUC-PR: {auc_pr:.4f}")
    print(f"Test Precision: {prec_score:.4f}")
    print(f"Test Recall: {rec_score:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    if auc_pr > 0.70:
        print("✅ AUC-PR > 0.70: Solid performance!")
    else:
        print("⚠️ AUC-PR < 0.70: May need model tuning.")


def compute_confusion(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def compute_business_loss(confusion, fn_cost=10, fp_cost=1):
    return (
        confusion["fn"] * fn_cost +
        confusion["fp"] * fp_cost
    )


def find_optimal_threshold(y_true, probs, fn_cost=10, fp_cost=1):
    thresholds = np.linspace(0.1, 0.9, 50)

    best_threshold = 0.5
    best_loss = float("inf")

    for t in thresholds:
        preds = (probs > t).astype(int)
        conf = compute_confusion(y_true, preds)
        loss = compute_business_loss(conf, fn_cost, fp_cost)

        if loss < best_loss:
            best_loss = loss
            best_threshold = float(t)

    return best_threshold, best_loss


if __name__ == '__main__':
    evaluate()
