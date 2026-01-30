#!/bin/bash
# start_frontend.sh

# Navigate to directory
cd "$(dirname "$0")/NeuroPhish/frontend" || exit

# Start Vite
echo "🚀 Starting NeuroPhish Frontend..."
npm run dev
