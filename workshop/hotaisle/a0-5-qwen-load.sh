#!/usr/bin/env bash
# A0.5 — sustained-load + tool-call correctness for Qwen3.6-27B on 1× MI300X.
# Run INSIDE the Hot Aisle VM, AFTER launch-qwen.sh + a0-validate.sh both pass.
#
# Codex flagged this as required before committing Qwen as the workshop fallback:
#  - Qwen3.5/3.6 hybrid GDN has documented 20-concurrent crash issues (#20069)
#  - 0.5.10rc0 had a ~20% quality regression on Qwen3.5-27B-FP8 (#21696)
#  - Smoke alone is not enough; need real sustained load + multilingual + tool battery
#
# Tests:
#  1. sglang.bench_serving — 16 concurrent, 60k input, 4k output, chat completions
#  2. claude_code_replay.py — 16 sessions × 5 turns, bursty, shared system prompt
#  3. quality battery — runs the full case set including multilingual
#  4. control-plane probe — concurrent /v1/models + /metrics during load (catches "API hangs while worker reports running")
#
# Pass criteria printed at end. Exits non-zero on hard failure.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SGLANG_API_KEY:?set SGLANG_API_KEY}"
URL="${URL:-http://127.0.0.1:30000}"
OUT_DIR="/tmp/a0-5-qwen"
mkdir -p "$OUT_DIR"

# Quality-battery + replay scripts must be available; copied from workshop/quality-battery/
QB_DIR="${QB_DIR:-$HOME/quality-battery}"
[ -d "$QB_DIR" ] || { echo "FAIL: $QB_DIR missing. Copy workshop/quality-battery/ to the VM first."; exit 1; }

# Spawn control-plane probe in background — catches engine wedge during load.
# /v1/models alone only catches HTTP-level death; /health_generate forces a real
# generation through the engine path, which is the actual wedge mode (#20069).
PROBE_LOG="$OUT_DIR/probe.log"
(
    while true; do
        T=$(date +%s)
        MODELS=FAIL
        METRICS=FAIL
        GEN=FAIL
        curl -sf -m 3 -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/v1/models" >/dev/null 2>&1 && MODELS=OK
        curl -sf -m 3 -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/metrics" >/dev/null 2>&1 && METRICS=OK
        curl -sf -m 8 -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/health_generate" >/dev/null 2>&1 && GEN=OK
        echo "$T models=$MODELS metrics=$METRICS gen=$GEN" >> "$PROBE_LOG"
        sleep 10
    done
) &
PROBE_PID=$!
trap "kill $PROBE_PID 2>/dev/null || true" EXIT

echo ""
echo "=========================================="
echo "A0.5 Test 1: bench_serving (16 concurrent, 60k input, 4k output)"
echo "=========================================="
docker exec \
    -e PYTHONPATH=/sgl-workspace/sglang/python \
    -e OPENAI_API_KEY="$SGLANG_API_KEY" \
    sglang python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --base-url http://127.0.0.1:30000 \
    --header "Authorization: Bearer $SGLANG_API_KEY" \
    --apply-chat-template \
    --dataset-name random \
    --num-prompts 32 \
    --random-input-len 60000 \
    --random-output-len 4000 \
    --max-concurrency 16 \
    --request-rate inf \
    --output-file /tmp/bench-random.json \
    2>&1 | tee "$OUT_DIR/bench-random.txt" \
    || { echo "FAIL: bench_serving failed"; exit 1; }

echo ""
echo "=========================================="
echo "A0.5 Test 2: claude_code_replay (16 sessions × 5 turns, bursty)"
echo "=========================================="
python3 "$QB_DIR/claude_code_replay.py" \
    --base-url "$URL" \
    --api-key "$SGLANG_API_KEY" \
    --num-sessions 16 \
    --turns-per-session 5 \
    --seed 42 \
    --output "$OUT_DIR/replay.json" \
    2>&1 | tee "$OUT_DIR/replay.txt" \
    || { echo "FAIL: claude_code_replay had errors"; exit 1; }

echo ""
echo "=========================================="
echo "A0.5 Test 3: quality battery (tool / JSON / code / multilingual)"
echo "=========================================="
ALLOW_SMALL_BATTERY=1 python3 "$QB_DIR/run.py" \
    --base-url "$URL" \
    --api-key "$SGLANG_API_KEY" \
    --cases "$QB_DIR/cases.json" \
    --output "$OUT_DIR/battery-results.json" \
    --label "qwen-fp8" \
    --model "Qwen/Qwen3.6-27B" \
    || { echo "FAIL: quality battery had errors"; exit 1; }

# Stop the probe and analyze
kill $PROBE_PID 2>/dev/null || true
wait 2>/dev/null || true

echo ""
echo "=========================================="
echo "A0.5 Control-plane probe summary"
echo "=========================================="
TOTAL=$(wc -l < "$PROBE_LOG")
MODELS_FAIL=$(awk '/models=FAIL/{n++} END{print n+0}' "$PROBE_LOG")
METRICS_FAIL=$(awk '/metrics=FAIL/{n++} END{print n+0}' "$PROBE_LOG")
GEN_FAIL=$(awk '/gen=FAIL/{n++} END{print n+0}' "$PROBE_LOG")
echo "  total probes: $TOTAL"
echo "  /v1/models     fails: $MODELS_FAIL"
echo "  /metrics       fails: $METRICS_FAIL"
echo "  /health_generate fails: $GEN_FAIL  (engine wedge canary)"
# /health_generate failures are the real wedge signal — control-plane endpoints
# can stay green while the engine is hung. Threshold tighter on gen.
if [ "$GEN_FAIL" -gt 2 ]; then
    echo "  FAIL: /health_generate failed >2 times — engine wedged under load"
    echo "  Last 20 probe lines:"; tail -20 "$PROBE_LOG"
    exit 1
fi
if [ "$MODELS_FAIL" -gt 3 ] || [ "$METRICS_FAIL" -gt 3 ]; then
    echo "  FAIL: control-plane endpoints failed >3 times — likely API hang"
    echo "  Last 20 probe lines:"; tail -20 "$PROBE_LOG"
    exit 1
fi

echo ""
echo "=========================================="
echo "A0.5 PASS for Qwen3.6-27B on 1× MI300X"
echo "  Outputs in $OUT_DIR"
echo "  - bench-random.txt   (worst-case throughput)"
echo "  - replay.json        (Claude-Code-shaped traffic, look for TTFT P95/P99)"
echo "  - battery-results.json  (tool/JSON/code/multilingual correctness)"
echo "  - probe.log          (control-plane responsiveness during load)"
echo ""
echo "Qwen path is workshop-ready as the fallback."
echo "=========================================="
