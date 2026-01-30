#!/bin/bash
# start_backend.sh

# Navigate to directory
cd "$(dirname "$0")/NeuroPhish/backend" || exit

# Activate venv
source venv/bin/activate

# Start Uvicorn
echo "🚀 Starting NeuroPhish Backend..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
