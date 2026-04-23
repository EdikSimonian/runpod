#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL_NAME:-qwen3.6:27b}"
PORT="${PORT:-80}"

# ── Network volume: prefer persistent storage when RunPod mounts one ─────────
if [ -d "/runpod-volume" ]; then
    mkdir -p /runpod-volume/.ollama/models
    export OLLAMA_MODELS="/runpod-volume/.ollama/models"
    echo "[start] Using network volume: $OLLAMA_MODELS"
fi

# ── Start Ollama server ───────────────────────────────────────────────────────
echo "[start] Launching Ollama..."
ollama serve &

# ── Wait for Ollama HTTP API ──────────────────────────────────────────────────
MAX_WAIT=300; ELAPSED=0
until curl -sf --max-time 5 \
    ${OLLAMA_API_KEY:+-H "Authorization: Bearer ${OLLAMA_API_KEY}"} \
    http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
    (( ELAPSED >= MAX_WAIT )) && { echo "[start] ERROR: Ollama timeout" >&2; exit 1; }
    echo "[start] Waiting for Ollama... (${ELAPSED}s)"
    sleep 3; (( ELAPSED += 3 ))
done
echo "[start] Ollama ready (${ELAPSED}s)."

# ── Pull model if not cached ──────────────────────────────────────────────────
if ollama list 2>/dev/null | grep -qF "${MODEL}"; then
    echo "[start] Model '${MODEL}' cached — skipping pull."
else
    echo "[start] Pulling '${MODEL}'..."
    ollama pull "${MODEL}"
    echo "[start] Pull complete."
fi

# ── Pre-warm: load model into VRAM before the first real request ──────────────
echo "[start] Pre-warming model into VRAM..."
curl -sf http://127.0.0.1:11434/api/generate \
    -H "Content-Type: application/json" \
    ${OLLAMA_API_KEY:+-H "Authorization: Bearer ${OLLAMA_API_KEY}"} \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"\",\"keep_alive\":-1}" > /dev/null \
    && echo "[start] Model loaded." \
    || echo "[start] WARN: pre-warm failed (non-fatal)."

# ── Start FastAPI on port 80 — RunPod Load Balancer default ─────────────────
echo "[start] Starting FastAPI on port ${PORT}..."
exec uvicorn server:app --host 0.0.0.0 --port "${PORT}" --workers 1
