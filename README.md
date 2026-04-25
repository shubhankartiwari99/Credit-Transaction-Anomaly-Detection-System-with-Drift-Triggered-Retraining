# Credit Transaction Anomaly Detection System with Drift-Triggered Retraining

## 🧠 What This System Does
This is a production-style ML system that:
- Detects anomalous credit card transactions
- Monitors real-time data drift using distributional metrics (KL divergence)
- Automatically triggers retraining when drift exceeds a threshold
- Maintains a model registry with version control and promotion workflows

---

## ⚙️ Core Insight
Traditional ML systems degrade silently under data drift.
This system treats drift as a first-class signal, enabling:
- proactive retraining
- continuous reliability
- observable model behavior

---

## 🔄 System Flow

```mermaid
graph LR
    A[Incoming Transactions] --> B[Prediction API]
    B --> C[Drift Monitoring]
    C --> D{Drift > Threshold?}
    D -- No --> B
    D -- Yes --> E[Trigger Retraining]
    E --> F[New Model Version]
    F --> G[Model Registry]
    G --> H[Promote to Production]
```

---

## 📊 Key Results
- Drift detection using KL divergence with real-time tracking
- Retraining triggered automatically when drift exceeds threshold
- Cooldown mechanism prevents unstable retraining loops
- Observability: drift timeline + prediction distribution visualization
- Shadow model deployment enables safe A/B comparison before promotion

---

## 🧠 Design Decisions
- **KL Divergence** used for drift detection due to sensitivity to distribution changes
- **Cooldown mechanism** added to avoid retraining instability under noisy drift signals
- **Feature shift explanation** improves interpretability of drift (surfaces top shifted feature)
- **Model registry** enables controlled promotion and rollback via shadow deployments
- **Failure handling** in retraining pipeline ensures system resilience and observable error state

---

## Demo

![Demo](./demo-fraud-ml.gif)

---

## Structure

```text
.
├── backend/
│   ├── api/            # FastAPI endpoints (predict, drift, retrain, health)
│   ├── data/           # Processed data, logs, drift history
│   ├── models/         # Trained models, registry, retrain status
│   ├── monitoring/     # Drift computation (KL, PSI), prediction logging
│   ├── retraining/     # Retrain trigger, cooldown, shadow deploy, promotion
│   ├── training/       # Preprocessing and model training scripts
│   ├── Procfile
│   ├── requirements.txt
│   └── runtime.txt
├── frontend/
│   ├── app/            # Next.js dashboard (drift, registry, charts)
│   ├── public/
│   ├── .env.example
│   ├── next.config.js
│   └── package.json
└── README.md
```

## Data

The raw dataset is not committed. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `backend/data/`.

## Local Setup

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python training/preprocess.py
python training/train.py
PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

The frontend reads the API base URL from `NEXT_PUBLIC_API_URL`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Run fraud prediction on a transaction |
| `GET` | `/drift` | Current drift score, status, threshold |
| `GET` | `/drift/history` | Drift score timeline |
| `GET` | `/metrics` | Raw drift metrics (KL, PSI) |
| `POST` | `/retrain` | Trigger model retraining |
| `GET` | `/retrain/status` | Last retrain result and metadata |
| `POST` | `/promote` | Promote shadow model to production |
| `GET` | `/registry` | Model version registry |
| `GET` | `/predictions` | Recent prediction logs |
| `GET` | `/health` | System health check |

## Deployment

- **Backend (Render)**: set the project root directory to `backend/`.
- **Frontend (Vercel)**: set the project root directory to `frontend/`.

## Live Deployment

- **Backend**: https://credit-transaction-anomaly-detection.onrender.com
- **Frontend**: https://credit-transaction-anomaly-detectio.vercel.app
