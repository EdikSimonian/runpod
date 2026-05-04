#!/usr/bin/env bash
# Hot Aisle bring-up: provision a 1× MI300X VM, wait for SSH, print connection info.
#
# Run from your laptop, NOT from inside any VM.
#
# Defaults:
#   --gpu-count 1   (override with GPU_COUNT=2 for the Mistral 3.5 path when stock appears)
#   model is whatever you launch INSIDE the VM via workshop/hotaisle/launch-{qwen,mistral}.sh
#
# Pre-requisites:
#   - hotaisle CLI installed and `hotaisle config get token` returns a token
#   - default-team set: `hotaisle config set default-team <slug>`
#   - SSH public key registered: `hotaisle user ssh-keys list` shows your key
#   - team has credits: `hotaisle team balance --handle <slug>`

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/../lib/env.sh"

need hotaisle
need jq
need ssh

GPU_COUNT="${GPU_COUNT:-1}"
TEAM="${HOTAISLE_TEAM:-edik-simonians-team}"

log "Probing stock for ${GPU_COUNT}× MI300X..."
AVAIL=$(hotaisle vm available --team "$TEAM" 2>/dev/null)
MATCHED=$(echo "$AVAIL" | jq --argjson n "$GPU_COUNT" '
    [ .[] | select(
        ((.Specs.gpus // []) | length > 0)
        and (.Specs.gpus[0].model == "MI300X")
        and (.Specs.gpus[0].count == $n)
        and (.Quantity > 0)
    ) ] | length')

if [ "${MATCHED:-0}" -eq 0 ]; then
    log "FAIL: no available SKU matching MI300X count=${GPU_COUNT}"
    log "Current availability:"
    echo "$AVAIL" | jq .
    exit 2
fi

log "OK: ${GPU_COUNT}× MI300X available. Provisioning..."
PROVISION_OUT=$(hotaisle vm provision \
    --team "$TEAM" \
    --gpu-model MI300X \
    --gpu-count "$GPU_COUNT" 2>&1)

echo "$PROVISION_OUT"
VM_ID=$(echo "$PROVISION_OUT" | jq -r '.id // .vm_id // empty' 2>/dev/null || true)

# Fall back to listing newest VM if jq couldn't extract id from provision output
if [ -z "$VM_ID" ] || [ "$VM_ID" = "null" ]; then
    log "Provision output did not contain id directly; pulling newest VM from list..."
    sleep 2
    VM_ID=$(hotaisle vm list --team "$TEAM" 2>/dev/null \
        | jq -r 'sort_by(.created_at // .createdAt // 0) | reverse | .[0].id // empty')
fi

[ -n "$VM_ID" ] || fail "Could not resolve newly provisioned VM id"
log "VM id: $VM_ID"

# Poll for VM ready (state should report ssh_host/ssh_port or running status)
log "Waiting for VM to be RUNNING (up to 5 min)..."
DEADLINE=$(( $(date +%s) + 5*60 ))
VM_JSON="{}"
SSH_HOST=""
SSH_PORT="22"
SSH_USER="hotaisle"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    VM_JSON=$(hotaisle vm get --team "$TEAM" --vm "$VM_ID" 2>/dev/null || echo "{}")
    STATE=$(echo "$VM_JSON" | jq -r '.state // .status // empty')
    log "  state: ${STATE:-unknown}"
    if [ "$STATE" = "running" ] || [ "$STATE" = "RUNNING" ] || [ "$STATE" = "active" ]; then
        SSH_HOST=$(echo "$VM_JSON" | jq -r '
            .ssh_host // .public_ip // .public_ipv4 // .ip // .ipv4 // empty
        ')
        SSH_PORT=$(echo "$VM_JSON" | jq -r '.ssh_port // 22')
        SSH_USER=$(echo "$VM_JSON" | jq -r '.ssh_user // "hotaisle"')
        [ -n "$SSH_HOST" ] && break
    fi
    sleep 10
done

[ -n "$SSH_HOST" ] || { echo "$VM_JSON" | jq . >&2; fail "VM did not reach RUNNING with SSH info within 5 min"; }

# Persist for tear-down + tunnel + watchdog
ENV_LOCAL="$SCRIPT_DIR/../lib/env.local.sh"
umask 077
{
    echo "# Auto-generated $(date) by hotaisle-bring-up.sh"
    echo "export HOTAISLE_VM_ID='$VM_ID'"
    echo "export VM_SSH_HOST='$SSH_HOST'"
    echo "export VM_SSH_PORT='$SSH_PORT'"
    echo "export VM_SSH_USER='$SSH_USER'"
    echo "export GPU_COUNT='$GPU_COUNT'"
} > "$ENV_LOCAL"
log "Wrote $ENV_LOCAL"

# Wait for sshd to actually accept connections (the API state can flip running before sshd is up)
log "Waiting for sshd on $SSH_HOST:$SSH_PORT..."
for _ in $(seq 1 60); do
    if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
         -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" 'echo ssh-up' 2>/dev/null | grep -q ssh-up; then
        log "  sshd accepting connections"
        break
    fi
    sleep 5
done

cat <<EOF

==========================================
HOT AISLE VM READY

VM ID:    $VM_ID
GPUs:     ${GPU_COUNT}× MI300X
SSH:      ssh -p $SSH_PORT $SSH_USER@$SSH_HOST

Next: copy the workshop scripts to the VM and start prep:
  scp -P $SSH_PORT -r workshop/hotaisle $SSH_USER@$SSH_HOST:~/
  ssh -p $SSH_PORT $SSH_USER@$SSH_HOST 'bash ~/hotaisle/vm-prep.sh'

Then in another shell, open the SGLang tunnel:
  bash workshop/scripts/hotaisle-tunnel.sh

Tear down: bash workshop/scripts/hotaisle-tear-down.sh
==========================================
EOF
