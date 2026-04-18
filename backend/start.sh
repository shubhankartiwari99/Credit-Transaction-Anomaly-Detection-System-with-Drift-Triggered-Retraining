#!/bin/bash
echo "=== Testing imports ==="
python -c "from api.main import app; print('Import OK')"
if [ $? -ne 0 ]; then
    echo "=== Import failed, showing full error above ==="
    exit 1
fi
echo "=== Starting server ==="
uvicorn api.main:app --host 0.0.0.0 --port $PORT
