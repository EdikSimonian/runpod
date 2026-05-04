#!/usr/bin/env bash
# SSH local-forward to expose the VM's SGLang :30000 on laptop's localhost:30000.
# Foreground process — Ctrl-C to close. Run in a second terminal.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/../lib/env.sh"
[ -f "$SCRIPT_DIR/../lib/env.local.sh" ] && . "$SCRIPT_DIR/../lib/env.local.sh"

require_var VM_SSH_HOST

LOCAL_PORT="${LOCAL_PORT:-30000}"
REMOTE_PORT="${REMOTE_PORT:-30000}"

log "Opening SSH tunnel: localhost:$LOCAL_PORT → ${VM_SSH_HOST}:${REMOTE_PORT}"
log "Ctrl-C to close. SGLang must be running on the VM (workshop/hotaisle/launch-*.sh)."

exec ssh -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    -p "${VM_SSH_PORT:-22}" \
    "${VM_SSH_USER:-hotaisle}@${VM_SSH_HOST}"
