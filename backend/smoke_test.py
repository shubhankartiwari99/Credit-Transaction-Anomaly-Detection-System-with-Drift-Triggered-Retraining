#!/usr/bin/env python3
"""
Smoke test script for the Fraud Detection ML System
Run this after starting the FastAPI server with:
cd backend && PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import requests
import time
import json

API_BASE = 'http://localhost:8000'

# Sample transaction features (from dataset)
SAMPLE_FEATURES = [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]

def test_endpoint(method, url, data=None):
    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling {url}: {e}")
        return None

def smoke_test():
    print("🚀 Starting Fraud Detection System Smoke Test")
    print("=" * 50)

    # 1. Check initial registry
    print("1. Checking initial model registry...")
    registry = test_endpoint('GET', f'{API_BASE}/registry')
    if registry:
        print(f"   ✅ Registry: {len(registry.get('versions', []))} versions")
        for v in registry.get('versions', []):
            print(f"      v{v['version']}: {v['status']} ({v['trigger_reason']})")

    # 2. Check initial metrics
    print("\n2. Checking initial drift metrics...")
    metrics = test_endpoint('GET', f'{API_BASE}/metrics')
    if metrics:
        print("   ✅ Metrics received")
        for key, value in metrics.get('drift_scores', {}).items():
            print(f"      {key}: {value:.4f}")

    # 3. Make 50 predictions
    print("\n3. Making 50 predictions...")
    predictions_made = 0
    for i in range(50):
        result = test_endpoint('POST', f'{API_BASE}/predict', {'features': SAMPLE_FEATURES})
        if result:
            predictions_made += 1
            if (i + 1) % 10 == 0:
                print(f"   ✅ {i+1} predictions completed")
        else:
            break
    print(f"   Total predictions logged: {predictions_made}")

    # 4. Check updated metrics after predictions
    print("\n4. Checking drift metrics after predictions...")
    metrics = test_endpoint('GET', f'{API_BASE}/metrics')
    if metrics:
        print("   ✅ Updated metrics:")
        for key, value in metrics.get('drift_scores', {}).items():
            print(f"      {key}: {value:.4f}")

    # 5. Check recent predictions
    print("\n5. Checking recent predictions...")
    predictions = test_endpoint('GET', f'{API_BASE}/predictions?limit=5')
    if predictions:
        print(f"   ✅ Retrieved {len(predictions)} recent predictions")
        for p in predictions[-3:]:  # Show last 3
            print(f"      Pred: {p['prediction']}, Conf: {p['confidence']:.4f}")

    # 6. Trigger retrain
    print("\n6. Triggering manual retrain...")
    retrain_result = test_endpoint('POST', f'{API_BASE}/retrain')
    if retrain_result:
        print("   ✅ Retrain triggered")

    # 7. Check registry after retrain
    print("\n7. Checking registry after retrain...")
    time.sleep(2)  # Wait for retrain to complete
    registry = test_endpoint('GET', f'{API_BASE}/registry')
    if registry:
        print(f"   ✅ Registry: {len(registry.get('versions', []))} versions")
        for v in registry.get('versions', []):
            print(f"      v{v['version']}: {v['status']} ({v['trigger_reason']})")

    # 8. Promote shadow model
    print("\n8. Promoting shadow model...")
    promote_result = test_endpoint('POST', f'{API_BASE}/promote')
    if promote_result:
        print("   ✅ Promotion result:", promote_result)

    # 9. Final registry check
    print("\n9. Final registry check...")
    registry = test_endpoint('GET', f'{API_BASE}/registry')
    if registry:
        print(f"   ✅ Final registry: {len(registry.get('versions', []))} versions")
        for v in registry.get('versions', []):
            print(f"      v{v['version']}: {v['status']} ({v['trigger_reason']})")

    print("\n🎉 Smoke test completed!")
    print("Check the Next.js dashboard at http://localhost:3000 for visual confirmation")

if __name__ == '__main__':
    smoke_test()
