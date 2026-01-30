#!/bin/bash
# stop_servers.sh

echo "🛑 Stopping Servers..."

# Find and kill Uvicorn (Backend)
pids_backend=$(pgrep -f "uvicorn main:app")
if [ -n "$pids_backend" ]; then
  echo "Found Backend (PIDs: $pids_backend). Killing..."
  kill $pids_backend
else
  echo "Backend not running."
fi

# Find and kill Vite/Node (Frontend)
# We look for "vite" processes
pids_frontend=$(pgrep -f "vite")
if [ -n "$pids_frontend" ]; then
  echo "Found Frontend (PIDs: $pids_frontend). Killing..."
  kill $pids_frontend
else
  echo "Frontend not running."
fi

echo "✅ All servers stopped."
