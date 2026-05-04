#!/usr/bin/env bash
# Shared env + helpers for workshop scripts. Source this from other scripts.
# Fill in the empty variables below after RunPod console provisioning.

# RunPod API key — auto-pulled from runpodctl config if present
: "${RUNPOD_API_KEY:=$(grep apikey "$HOME/.runpod/config.toml" 2>/dev/null | sed "s/.*= *'\([^']*\).*/\1/")}"

# Pod IDs — fill in after provisioning each pod via console
: "${MI300X_POD_ID_1GPU:=}"      # A0/A1: 1× MI300X EU-RO-1
: "${MI300X_POD_ID_2GPU:=}"      # A2 + workshop: 2× MI300X EU-RO-1
: "${H100_POD_ID_BACKUP:=}"      # backup: 4× H100 80GB in CUDA DC, kept stopped

# Network volume IDs (500 GB each)
: "${MI300X_VOLUME_ID:=}"        # EU-RO-1
: "${H100_VOLUME_ID:=}"          # CUDA DC

# SGLang auth — generate once with `openssl rand -hex 32` and reuse across phases
: "${SGLANG_API_KEY:=}"
: "${SGLANG_ADMIN_API_KEY:=}"

# Model
MODEL_PATH="mistralai/Mistral-Medium-3.5-128B"
# Pin to the commit AFTER the long-context config.json fix.
# Verify on the HF model page commit history before locking in.
MODEL_REVISION="${MODEL_REVISION:-main}"

# RunPod TCP exposure — populated after pod start
: "${POD_TCP_HOST:=}"
: "${POD_TCP_PORT:=}"

# LiteLLM
LITELLM_PORT="${LITELLM_PORT:-4000}"

# === Helpers ===
log()    { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" >&2; }
fail()   { log "FAIL: $*"; exit 1; }
need()   { command -v "$1" >/dev/null 2>&1 || fail "missing dependency: $1"; }
require_var() {
    local v="$1"
    [ -n "${!v:-}" ] || fail "$v is unset; edit workshop/lib/env.sh"
}

# Preflight — call from laptop-side scripts (bring-up, tear-down, fallback, watchdog)
preflight_laptop() {
    local missing=()
    # LiteLLM runs in Docker, so we don't require the litellm CLI on the host.
    # runpodctl is legacy from the RunPod path — not used by Hot Aisle bring-up.
    for cmd in docker jq curl python3; do
        command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        fail "missing laptop dependencies: ${missing[*]} (install with: brew install jq docker)"
    fi
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        || fail "python3 must be >= 3.10"
}

# Resolve pod TCP host:port for a given internal port. Tries multiple
# RunPod CLI/API output shapes since runpodctl JSON varies by version.
# Args: pod_id, internal_port (e.g. 30000)
# Sets globals: POD_TCP_HOST, POD_TCP_PORT
pod_tcp_resolve() {
    local pod_id="$1"
    local internal_port="$2"
    local pod_json

    # Try runpodctl JSON output first (modern shape)
    pod_json=$(runpodctl pod get "$pod_id" -o json 2>/dev/null || echo "")

    # Fall back to REST API if runpodctl JSON is empty/malformed
    if ! echo "$pod_json" | jq empty 2>/dev/null; then
        require_var RUNPOD_API_KEY
        pod_json=$(curl -sS -m 15 \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://rest.runpod.io/v1/pods/$pod_id" 2>/dev/null || echo "")
    fi
    if ! echo "$pod_json" | jq empty 2>/dev/null; then
        log "pod_tcp_resolve: could not get pod JSON for $pod_id"
        return 1
    fi

    # Try GraphQL-style: .runtime.ports[] | select(.privatePort == N and .type == "tcp")
    local host port
    read -r host port < <(
        echo "$pod_json" | jq -r --argjson p "$internal_port" '
            .runtime.ports // []
            | map(select((.privatePort == $p) and ((.type // "tcp") == "tcp")))
            | if length > 0 then "\(.[0].ip) \(.[0].publicPort)" else empty end
        '
    )

    # Try REST-style: .publicIp + .portMappings["30000"]
    if [ -z "${host:-}" ] || [ -z "${port:-}" ]; then
        host=$(echo "$pod_json" | jq -r '.publicIp // empty')
        port=$(echo "$pod_json" | jq -r --arg p "$internal_port" '.portMappings[$p] // empty')
    fi

    if [ -z "${host:-}" ] || [ -z "${port:-}" ] || [ "$host" = "null" ] || [ "$port" = "null" ]; then
        log "pod_tcp_resolve: no TCP mapping found for port $internal_port"
        log "pod JSON for debugging:"
        echo "$pod_json" | jq . >&2
        return 1
    fi

    POD_TCP_HOST="$host"
    POD_TCP_PORT="$port"
    return 0
}

# Wait for SGLang to drain in-flight requests using /metrics polling.
# Args: base_url, max_wait_s (default 300)
# Returns 0 if drained, 1 if timeout.
sglang_drain() {
    local base_url="$1"
    local max_wait="${2:-300}"
    local clean_in_a_row=0
    local deadline=$(( $(date +%s) + max_wait ))

    log "Draining SGLang at $base_url (max ${max_wait}s)..."
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local metrics
        metrics=$(curl -sS -m 5 -H "Authorization: Bearer ${SGLANG_API_KEY:-}" \
            "$base_url/metrics" 2>/dev/null || echo "")
        if [ -z "$metrics" ]; then
            log "  /metrics not reachable; assuming drained"
            return 0
        fi
        local running queued
        running=$(echo "$metrics" | grep -E '^sglang:num_running_reqs[ {]' | awk '{print $NF}' | head -1)
        queued=$(echo "$metrics"  | grep -E '^sglang:num_queue_reqs[ {]'   | awk '{print $NF}' | head -1)
        running="${running:-0}"
        queued="${queued:-0}"
        # Strip decimal (Prometheus values can be floats)
        running="${running%.*}"
        queued="${queued%.*}"
        if [ "$running" = "0" ] && [ "$queued" = "0" ]; then
            clean_in_a_row=$((clean_in_a_row + 1))
            log "  drained probe ${clean_in_a_row}/3 (running=0 queued=0)"
            [ "$clean_in_a_row" -ge 3 ] && return 0
        else
            clean_in_a_row=0
            log "  in-flight: running=$running queued=$queued"
        fi
        sleep 5
    done
    log "  timeout waiting for drain"
    return 1
}

# Start LiteLLM in a Docker container.
#
# litellm_config.yaml uses os.environ/SGLANG_API_KEY and os.environ/MODEL_API_BASE,
# which we pass through as env vars to the container. MODEL_API_BASE must point
# at the SGLang endpoint reachable from inside the container — when bring-up.sh
# opens an SSH tunnel on the laptop's :30000, the container reaches it via
# host.docker.internal:30000 (Docker Desktop on macOS).
#
# Container name is "workshop-litellm". Idempotent: prior containers are torn
# down before relaunch.
litellm_start() {
    require_var SGLANG_API_KEY
    require_var MODEL_API_BASE
    require_var LITELLM_MASTER_KEY

    need docker

    local cfg="$1"
    local cname="workshop-litellm"
    local pgname="workshop-litellm-db"
    local netname="workshop-litellm-net"
    local image="${LITELLM_IMAGE:-ghcr.io/berriai/litellm:main-stable}"
    local pg_image="postgres:16-alpine"
    local pg_pass="${LITELLM_PG_PASSWORD:-litellm-local}"

    # Tear down prior LiteLLM (Postgres can persist between runs)
    docker rm -f "$cname" >/dev/null 2>&1 || true

    # Dedicated bridge network so the proxy can resolve postgres by name
    docker network inspect "$netname" >/dev/null 2>&1 \
        || docker network create "$netname" >/dev/null

    if ! docker ps --filter "name=$pgname" --filter "status=running" --format '{{.Names}}' | grep -q "^${pgname}$"; then
        docker rm -f "$pgname" >/dev/null 2>&1 || true
        log "Starting Postgres sidecar ($pg_image)"
        docker run -d \
            --name "$pgname" \
            --network "$netname" \
            --restart unless-stopped \
            -e POSTGRES_USER=litellm \
            -e POSTGRES_PASSWORD="$pg_pass" \
            -e POSTGRES_DB=litellm \
            -v workshop-litellm-pgdata:/var/lib/postgresql/data \
            "$pg_image" >/dev/null
        # Wait for PG ready
        for _ in $(seq 1 30); do
            docker exec "$pgname" pg_isready -U litellm >/dev/null 2>&1 && break
            sleep 1
        done
    fi

    local db_url="postgresql://litellm:${pg_pass}@${pgname}:5432/litellm"

    log "Starting LiteLLM container ($image) on :$LITELLM_PORT"
    log "  MODEL_API_BASE=$MODEL_API_BASE"
    docker run -d \
        --name "$cname" \
        --network "$netname" \
        --add-host=host.docker.internal:host-gateway \
        --restart unless-stopped \
        -p "${LITELLM_PORT}:4000" \
        -v "$cfg:/app/config.yaml:ro" \
        -e SGLANG_API_KEY="$SGLANG_API_KEY" \
        -e MODEL_API_BASE="$MODEL_API_BASE" \
        -e LITELLM_MASTER_KEY="$LITELLM_MASTER_KEY" \
        -e DATABASE_URL="$db_url" \
        -e STORE_MODEL_IN_DB="True" \
        "$image" \
        --config /app/config.yaml --port 4000 \
        >/dev/null

    # Wait for /health/liveliness with explicit failure
    for _ in $(seq 1 90); do
        if curl -sf -m 2 "http://127.0.0.1:${LITELLM_PORT}/health/liveliness" >/dev/null 2>&1 \
            || curl -sf -m 2 "http://127.0.0.1:${LITELLM_PORT}/health" >/dev/null 2>&1; then
            log "  LiteLLM is up"
            return 0
        fi
        sleep 1
    done
    log "FAIL: LiteLLM did not become healthy. Last 60 log lines:"
    docker logs --tail 60 "$cname" 2>&1 >&2 || true
    return 1
}

# Persist resolved POD_TCP_HOST/PORT/ACTIVE_POD_ID for other shells/scripts
write_env_local() {
    local pod_id="$1"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    local out="$script_dir/../lib/env.local.sh"
    umask 077
    {
        echo "# Auto-generated $(date)"
        echo "export POD_TCP_HOST='$POD_TCP_HOST'"
        echo "export POD_TCP_PORT='$POD_TCP_PORT'"
        echo "export ACTIVE_POD_ID='$pod_id'"
    } > "$out"
    log "Wrote $out"
}
