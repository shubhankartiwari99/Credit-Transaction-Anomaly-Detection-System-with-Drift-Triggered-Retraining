import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DRIFT_HISTORY_PATH = BASE_DIR / "data" / "drift_history.json"
MODEL_REGISTRY_PATH = BASE_DIR / "models" / "model_registry.json"
DECISION_HISTORY_PATH = BASE_DIR / "models" / "decision_history.json"

def show_timeline():
    print("\n" + "="*50)
    print(" 🧠 ML SYSTEM OBSERVABILITY & GOVERNANCE CLI")
    print("="*50 + "\n")
    
    # Determine Last Decision
    last_decision = "Unknown"
    last_event_summary = None
    if DECISION_HISTORY_PATH.exists():
        try:
            with open(DECISION_HISTORY_PATH, "r") as f:
                history = json.load(f)
                if history:
                    last_event = history[-1]
                    decision = last_event.get("decision", "unknown")
                    drift_score = last_event.get("drift_score")
                    delta_loss = last_event.get("delta_loss")
                    candidate_threshold = last_event.get("candidate", {}).get("threshold")
                    production_threshold = last_event.get("production", {}).get("threshold")
                    production_version = last_event.get("production", {}).get("version")
                    candidate_version = last_event.get("candidate", {}).get("version")
                    production_model_id = last_event.get("production", {}).get("model_id")
                    candidate_model_id = last_event.get("candidate", {}).get("model_id")
                    last_decision = decision
                    last_event_summary = {
                        "drift_score": drift_score,
                        "delta_loss": delta_loss,
                        "production_threshold": production_threshold,
                        "candidate_threshold": candidate_threshold,
                        "production_version": production_version,
                        "candidate_version": candidate_version,
                        "production_model_id": production_model_id,
                        "candidate_model_id": candidate_model_id
                    }
        except Exception:
            pass

    if last_event_summary is None and MODEL_REGISTRY_PATH.exists():
        with open(MODEL_REGISTRY_PATH, "r") as f:
            try:
                registry = json.load(f)
                versions = registry.get("versions", [])
                if versions:
                    last_version = versions[-1]
                    if len(versions) == 1:
                        last_decision = "baseline_established"
                    elif last_version.get("status") == "production":
                        last_decision = "promote (candidate > production)"
                    else:
                        last_decision = "no_action (candidate <= production)"
            except Exception:
                pass

    print(f"Last Decision: {last_decision}\n")
    if last_event_summary:
        print("--- Last Decision Summary ---")
        print(f"Drift Score: {last_event_summary['drift_score']}")
        print(f"Production Version: v{last_event_summary['production_version']} ({last_event_summary['production_model_id']})")
        print(f"Candidate Version: v{last_event_summary['candidate_version']} ({last_event_summary['candidate_model_id']})")
        print(f"Production Threshold: {last_event_summary['production_threshold']}")
        print(f"Candidate Threshold: {last_event_summary['candidate_threshold']}")
        print(f"Δ Loss: {last_event_summary['delta_loss']}\n")
    
    # 1. Drift History
    print("--- Drift & Performance Timeline (Last 10 Runs) ---")
    if DRIFT_HISTORY_PATH.exists():
        with open(DRIFT_HISTORY_PATH, "r") as f:
            try:
                history = json.load(f)
                if not history:
                    print("No drift runs logged yet.")
                else:
                    for i, entry in enumerate(history[-10:]):
                        ts = entry.get('timestamp')
                        try:
                            # Format timestamp nicely
                            ts = datetime.fromisoformat(ts).strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                        score = entry.get('drift_score', 0)
                        
                        # Visual trend indicator
                        bar = "█" * int(score * 20)
                        alert = "⚠️ DRIFT" if score > 0.1 else "✅ STABLE"
                        print(f"[{ts}] {alert:<10} | Score: {score:.3f} | Trend: {bar}")
            except json.JSONDecodeError:
                print("Error reading drift history.")
    else:
        print("drift_history.json not found.")

    print("\n" + "-"*50 + "\n")
    
    # 2. Model Governance Registry
    print("--- Model Version History (Governance) ---")
    if MODEL_REGISTRY_PATH.exists():
        with open(MODEL_REGISTRY_PATH, "r") as f:
            try:
                registry = json.load(f)
                versions = registry.get("versions", [])
                if not versions:
                    print("No model versions in registry.")
                else:
                    for v in versions:
                        ver = f"v{v.get('version')}"
                        status = v.get('status', 'unknown').upper()
                        auc = v.get('auc_roc')
                        auc_str = f"{auc:.3f}" if auc else "N/A"
                        reason = v.get('trigger_reason', 'manual')
                        
                        # Highlight production
                        if status == "PRODUCTION":
                            ver_str = f"🚀 {ver} [{status}]"
                        elif status == "SHADOW":
                            ver_str = f"👻 {ver} [{status}]"
                        else:
                            ver_str = f"📦 {ver} [{status}]"
                            
                        print(f"{ver_str:<20} | AUC: {auc_str:<5} | Trigger: {reason}")
            except json.JSONDecodeError:
                print("Error reading model registry.")
    else:
        print("model_registry.json not found.")
        
    print("\n" + "="*50 + "\n")

if __name__ == '__main__':
    show_timeline()
