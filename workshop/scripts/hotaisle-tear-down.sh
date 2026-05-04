#!/usr/bin/env bash
# Hot Aisle tear-down: stop or delete the active VM.
#
# Default: hotaisle vm stop (preserves VM definition + disk for cheap restart).
# Set DELETE=1 to fully destroy with `hotaisle vm delete`.
#
# Run from your laptop.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/../lib/env.sh"
[ -f "$SCRIPT_DIR/../lib/env.local.sh" ] && . "$SCRIPT_DIR/../lib/env.local.sh"

need hotaisle
need jq
require_var HOTAISLE_VM_ID

TEAM="${HOTAISLE_TEAM:-edik-simonians-team}"

# Stop LiteLLM if running on laptop (best-effort)
PID_FILE="/tmp/litellm-${LITELLM_PORT:-4000}.pid"
if [ -f "$PID_FILE" ]; then
    LITELLM_PID=$(cat "$PID_FILE")
    log "Stopping LiteLLM (pid $LITELLM_PID)..."
    kill -INT "$LITELLM_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$LITELLM_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
fi

# Best-effort drain SGLang via the SSH-tunneled endpoint
if [ -n "${VM_SSH_HOST:-}" ]; then
    log "Best-effort draining SGLang on VM (skip if no in-flight requests)..."
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -p "${VM_SSH_PORT:-22}" "${VM_SSH_USER:-hotaisle}@$VM_SSH_HOST" \
        'docker stop --time 30 sglang 2>/dev/null || true' || true
fi

if [ "${DELETE:-0}" = "1" ]; then
    log "DELETING VM $HOTAISLE_VM_ID..."
    hotaisle vm delete --team "$TEAM" --vm "$HOTAISLE_VM_ID"
    rm -f "$SCRIPT_DIR/../lib/env.local.sh"
    log "VM deleted; env.local.sh removed"
else
    log "Stopping VM $HOTAISLE_VM_ID (use DELETE=1 to destroy instead)..."
    hotaisle vm stop --team "$TEAM" --vm "$HOTAISLE_VM_ID" || log "WARN: stop returned non-zero"
    log "VM stopped; env.local.sh preserved for next session"
fi

cat <<EOF
==========================================
TEARDOWN COMPLETE
  VM: $HOTAISLE_VM_ID  ($([ "${DELETE:-0}" = "1" ] && echo "DELETED" || echo "stopped"))
==========================================
EOF
