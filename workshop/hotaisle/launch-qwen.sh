#!/usr/bin/env bash
# Launch SGLang serving Qwen3.6-27B-FP8 on 1× MI300X (the workshop fallback path).
# Run INSIDE the Hot Aisle VM.
#
# Codex-validated config:
#  - Qwen/Qwen3.6-27B-FP8 (NOT cyankiwi compressed-tensors AWQ — known broken on SGLang)
#  - --attention-backend triton + --linear-attn-backend triton (GDN models on AMD need triton, NOT AITER)
#  - BF16 KV cache (default; no --kv-cache-dtype) — fits trivially with Qwen's hybrid attention
#  - --reasoning-parser qwen3 + --tool-call-parser qwen3_coder per Qwen3.6 model card

set -euo pipefail

IMAGE="${SGLANG_IMAGE:-qwen-sglang:rocm720-aiter112-sgmain}"
HF_CACHE_HOST="${HF_CACHE_HOST:-/var/lib/hf-cache}"
CONTAINER="${CONTAINER:-sglang}"

# Required from env
: "${HF_TOKEN:?set HF_TOKEN env var (Hugging Face read token, accept Qwen license if gated)}"
: "${SGLANG_API_KEY:?set SGLANG_API_KEY (random secret, used by LiteLLM bearer)}"
: "${SGLANG_ADMIN_API_KEY:?set SGLANG_ADMIN_API_KEY (random secret)}"

# Stop any previous container
docker stop "$CONTAINER" 2>/dev/null || true
docker rm   "$CONTAINER" 2>/dev/null || true

echo "Launching SGLang with Qwen3.6-27B-FP8 (1× MI300X, BF16 KV, triton backends)..."
echo "First boot: ~15-25 min (model download ~30 GB + engine init). Tail logs with:"
echo "  docker logs -f $CONTAINER"
echo ""

docker run -d \
    --name "$CONTAINER" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --shm-size 32g --ipc host \
    --restart unless-stopped \
    -v "$HF_CACHE_HOST:/root/.cache/huggingface" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e SGLANG_USE_AITER=1 \
    -e SGLANG_ENABLE_SPEC_V2=1 \
    -p 30000:30000 \
    "$IMAGE" \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3.6-27B \
        --dtype bfloat16 \
        --host 0.0.0.0 --port 30000 \
        --tp 1 \
        --attention-backend triton \
        --linear-attn-backend triton \
        --context-length 131072 \
        --max-running-requests 8 \
        --max-total-tokens 1100000 \
        --mem-fraction-static 0.92 \
        --cuda-graph-max-bs 8 \
        --cuda-graph-bs 1 2 4 8 \
        --triton-attention-num-kv-splits 16 \
        --chunked-prefill-size 32768 \
        --max-prefill-tokens 32768 \
        --num-continuous-decode-steps 2 \
        --schedule-policy lpm \
        --enable-tokenizer-batch-encode \
        --speculative-algorithm EAGLE \
        --speculative-num-steps 5 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 6 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --api-key "$SGLANG_API_KEY" \
        --admin-api-key "$SGLANG_ADMIN_API_KEY" \
        --enable-metrics \
        --enable-cache-report \
        --disable-piecewise-cuda-graph

echo "Container started: $CONTAINER"
echo "Logs:  docker logs -f $CONTAINER"
echo "Health: curl -sf -H 'Authorization: Bearer \$SGLANG_API_KEY' http://127.0.0.1:30000/health"

# Wait for /health and assert the KV pool is large enough for 16×64k.
# Mistral 3.5 silently profiled max_total_num_tokens 11x below the requested
# value — we don't want users to discover that under load.
MIN_KV_TOKENS="${MIN_KV_TOKENS:-1000000}"
URL="http://127.0.0.1:30000"
echo ""
echo "=== Waiting for /health (up to 25 min for first model load) ==="
DEADLINE=$(( $(date +%s) + 25*60 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if curl -sf -m 5 -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/health" >/dev/null 2>&1; then
        echo "  /health 200"
        break
    fi
    sleep 10
done
curl -sf -m 5 -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/health" >/dev/null \
    || { echo "FAIL: /health timeout. Last 80 lines:"; docker logs --tail 80 "$CONTAINER" 2>&1; exit 1; }

echo ""
echo "=== KV pool size gate (must be >= $MIN_KV_TOKENS tokens) ==="
INFO=$(curl -sS -m 10 -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/get_server_info" 2>/dev/null || echo "{}")
ACTUAL=$(echo "$INFO" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    # SGLang nests under "internal_states"[0] in 0.5.x; fall back to top level
    states = d.get("internal_states") or [{}]
    s = states[0] if states else {}
    v = s.get("max_total_num_tokens") or d.get("max_total_num_tokens") or 0
    print(int(v))
except Exception:
    print(0)
')
echo "  max_total_num_tokens = $ACTUAL"
if [ "${ACTUAL:-0}" -lt "$MIN_KV_TOKENS" ]; then
    echo "  FAIL: KV pool $ACTUAL < required $MIN_KV_TOKENS — engine cannot service 16×64k."
    echo "  This is the same failure mode that killed Mistral 3.5. Aborting."
    echo "  Last 60 lines from container:"
    docker logs --tail 60 "$CONTAINER" 2>&1
    exit 1
fi
echo "  OK: KV pool sized for 16×64k workshop spec."
