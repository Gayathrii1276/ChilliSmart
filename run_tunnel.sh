#!/usr/bin/env bash
# Simple tunnel starter: prefer localhost.run (SSH) and fallback to localtunnel

set -euo pipefail
PORT=${1:-8000}

echo "Starting tunnel for local port $PORT..."

if command -v ssh >/dev/null 2>&1; then
  echo "Attempting localhost.run SSH reverse tunnel..."
  # anonymous reverse tunnel: maps remote port 80 to local port
  ssh -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -R 80:localhost:${PORT} nokey@localhost.run
  exit 0
fi

if command -v npx >/dev/null 2>&1; then
  echo "ssh not available; falling back to npx localtunnel (requires node)"
  npx localtunnel --port $PORT
  exit 0
fi

echo "No tunnel method available. Install OpenSSH client or Node (for localtunnel)."
exit 1
