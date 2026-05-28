#!/usr/bin/env bash
set -eu
PORT=${1:-8000}
LOG=/tmp/mi_tunnel.log

echo "Starting tunnel for local port $PORT..." | tee "$LOG"

# Prefer ssh -> localhost.run (no interstitial)
if command -v ssh >/dev/null 2>&1; then
  echo "Attempting localhost.run SSH reverse tunnel..." | tee -a "$LOG"
  nohup ssh -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -R 80:localhost:${PORT} nokey@localhost.run >/tmp/mi_tunnel.out 2>&1 &
  sleep 2
  echo "Output (tail):" | tee -a "$LOG"
  tail -n 200 /tmp/mi_tunnel.out | tee -a "$LOG"
  echo "If successful you'll see a public https://*.lhr.life URL above. Open it on your phone." | tee -a "$LOG"
  exit 0
fi

# Fallback to localtunnel via npx
if command -v npx >/dev/null 2>&1; then
  echo "localhost.run not available; falling back to localtunnel (npx)." | tee -a "$LOG"
  nohup npx --yes localtunnel --port ${PORT} >/tmp/mi_tunnel.out 2>&1 &
  sleep 2
  tail -n 200 /tmp/mi_tunnel.out | tee -a "$LOG"
  echo "If you see a https://*.loca.lt URL above, open it on your phone." | tee -a "$LOG"
  exit 0
fi

echo "No tunnel tool available. Install SSH or npm + localtunnel." | tee -a "$LOG"
exit 2
