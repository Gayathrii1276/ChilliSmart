#!/usr/bin/env bash
set -eu
PORT=${1:-8000}
LOG=/tmp/mi_autossh.log
OUT=/tmp/mi_autossh.out

echo "Starting persistent autossh tunnel for local port $PORT..." | tee "$LOG"

if ! command -v autossh >/dev/null 2>&1; then
  cat <<'EOF' | tee -a "$LOG"
autossh is not installed.
Install via Homebrew:
  brew install autossh
Or install via your package manager.
EOF
  exit 2
fi

# Use autossh to keep the reverse tunnel alive
AUTOSSH_CMD=(autossh -M 0 -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -R 80:localhost:${PORT} nokey@localhost.run)

echo "Running: ${AUTOSSH_CMD[*]}" | tee -a "$LOG"
"${AUTOSSH_CMD[@]}" >/tmp/mi_autossh.out 2>&1 || true
sleep 1
echo "Tunnel startup output (tail):" | tee -a "$LOG"
tail -n 200 /tmp/mi_autossh.out | tee -a "$LOG"

echo "If the tunnel succeeded you'll see a https://*.lhr.life URL above. Open it on your phone." | tee -a "$LOG"
