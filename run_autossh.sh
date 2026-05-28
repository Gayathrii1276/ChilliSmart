#!/usr/bin/env bash
set -euo pipefail
PORT=${1:-8000}
LOG=/tmp/mi_autossh.out

if ! command -v autossh >/dev/null 2>&1; then
  echo "autossh not installed. Install with: brew install autossh"
  exit 2
fi

echo "Starting autossh reverse tunnel to localhost.run for port $PORT (logging to $LOG)"
autossh -M 0 -N -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -R 80:localhost:${PORT} nokey@localhost.run >"$LOG" 2>&1 &
echo "autossh started (pid $!)"
