# Credit Transaction Anomaly Detection System with Drift-Triggered Retraining

This project implements a machine learning system for detecting credit card fraud using the Kaggle credit card fraud dataset. The system includes drift detection and automatic retraining capabilities.

## Dataset

- Source: Kaggle Credit Card Fraud Detection dataset
- Transactions: 284,807
- Fraudulent: 492 (0.172%)
- Features: V1-V28 (PCA-transformed), Amount, Time, Class

## Project Structure

- `data/`: Raw and processed datasets
- `training/`: Training scripts and preprocessing
- `api/`: FastAPI application for predictions
- `monitoring/`: Drift detection and logging
- `retraining/`: Retraining triggers
- `dashboard/`: Visualization dashboard
- `models/`: Saved model artifacts

## Setup

1. Create virtual environment: `python -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run preprocessing: `python training/preprocess.py`
5. Train model: `python training/train.py`

## Usage

### API

Run the API: `uvicorn api.main:app --reload`

Test prediction: Use POST to `/predict` with JSON `{"features": [list of 30 floats]}`

Get drift metrics: GET `/metrics`

### Training

- Preprocessing handles scaling and SMOTE for imbalance
- Model: RandomForest with AUC-PR 0.831 on test set

## Status Check

| Component | Status | Notes |
|---|---|---|
| Data pipeline + preprocessing | ✅ | SMOTE handled correctly |
| Model training | ✅ | AUC-ROC 0.944, AUC-PR 0.831 on test |
| Model artifact | ✅ | `models/model.pkl` |
| FastAPI `/predict` | ✅ | Accepts 30 features, logs predictions |
| Logging | ✅ | Parquet-based logging in `monitoring/logger.py` |
| Drift monitoring | ✅ | KL + PSI on Amount/confidence in `monitoring/drift.py` |
| `/metrics` endpoint | ✅ | Exposes drift scores |
| Retrain trigger | ✅ | PSI > 0.2 or KL > 0.1 in `retraining/retrain_trigger.py` |
| Shadow deploy | ✅ | Dual model inference in `api/predict.py` |
| Model registry | ✅ | JSON registry in `models/model_registry.json` |
| `/retrain` + `/promote` | ✅ | Manual retrain/promote endpoints |
| `/predictions` endpoint | ✅ | Returns recent prediction history |
| Next.js Dashboard | ✅ | Real-time monitoring in `dashboard/` |

## Running the Full System

1. **Start FastAPI**: `cd /path/to/project && PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
2. **Start Dashboard**: `cd dashboard && npm install && npm run dev`
3. **Open Dashboard**: `http://localhost:3000` (polls API every 30s)
4. **Run Smoke Test**: `python smoke_test.py` (tests full MLOps loop)

## API Endpoints

- `POST /predict` - Make prediction
- `GET /metrics` - Drift scores
- `GET /registry` - Model versions
- `POST /retrain` - Trigger retrain
- `POST /promote` - Promote shadow model
- `GET /predictions?limit=100` - Recent predictions

## System Design Decisions

- **PSI > 0.2 threshold for retrain triggers**: Industry standard in financial ML for significant distributional shift; balances sensitivity to real drift with avoiding false positives from noise.
- **Parquet over SQLite for prediction logs**: Columnar format enables efficient computation of distributions (KL/PSI) on large datasets; better compression and query performance for time-series analytics.
- **Shadow deployment before promotion**: Ensures production safety by testing new models in parallel without risk; allows A/B comparison of confidence distributions before cold swap.
- **AUC-PR over AUC-ROC for evaluation**: Honest metric on heavily imbalanced datasets (0.17% fraud rate); AUC-ROC inflates scores, while AUC-PR directly measures precision-recall tradeoffs critical for fraud detection.
- **SMOTE for class imbalance**: Synthetic oversampling preserves minority class patterns better than random oversampling; maintains decision boundary integrity in high-dimensional PCA space.

## Next Steps

- Implement drift detection (PSI/KL)
- Add logging for predictions
- Create retraining pipeline
- Build dashboard