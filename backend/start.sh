#!/bin/bash
echo "=== Testing imports ==="
python -c "from api.main import app; print('Import OK')" 2>&1
EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Import failed ==="
    exit 1
fi
echo "=== Starting server ==="
uvicorn api.main:app --host 0.0.0.0 --port $PORT
