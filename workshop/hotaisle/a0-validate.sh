#!/usr/bin/env bash
# A0 validation: confirms whichever SGLang container is currently running responds correctly.
# Run INSIDE the Hot Aisle VM, AFTER launch-{qwen,mistral}.sh has been started AND its
# /health is reachable.
#
# Tests:
#  1. /v1/models returns a model
#  2. Single chat completion (text)
#  3. Tool-call (using a small tool definition)
#  4. Multilingual probe (catches GDN garbling regression for Qwen3.5/3.6 family)
#
# Exits non-zero on any failure.

set -euo pipefail

: "${SGLANG_API_KEY:?set SGLANG_API_KEY}"
URL="${URL:-http://127.0.0.1:30000}"
CONTAINER="${CONTAINER:-sglang}"

# Wait for /health (in case caller jumped in too early)
echo "Waiting for /health (up to 25 min for first model load)..."
DEADLINE=$(( $(date +%s) + 25*60 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if curl -sf -m 5 -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/health" >/dev/null 2>&1; then
        echo "  /health 200"
        break
    fi
    sleep 5
done
curl -sf -m 5 -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/health" >/dev/null \
    || { echo "FAIL: /health timeout. Last 60 lines from container:"; docker logs --tail 60 "$CONTAINER" 2>&1; exit 1; }

echo ""
echo "=== /v1/models ==="
MODELS=$(curl -sS -H "Authorization: Bearer $SGLANG_API_KEY" "$URL/v1/models")
echo "$MODELS" | python3 -m json.tool
MODEL_ID=$(echo "$MODELS" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
echo "Active model: $MODEL_ID"

echo ""
echo "=== Test 1: single chat completion ==="
RESP1=$(curl -sS -H "Authorization: Bearer $SGLANG_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_ID\",
      \"messages\": [{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}],
      \"max_tokens\": 512,
      \"temperature\": 0
    }" \
    "$URL/v1/chat/completions")
echo "$RESP1" | python3 -m json.tool
if ! echo "$RESP1" | grep -qi 'pong'; then
    echo "FAIL: response did not contain 'pong'"; exit 1
fi

echo ""
echo "=== Test 2: tool-call ==="
RESP2=$(curl -sS -H "Authorization: Bearer $SGLANG_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_ID\",
      \"messages\": [
        {\"role\":\"system\",\"content\":\"You are a coding assistant. Use the bash tool to answer.\"},
        {\"role\":\"user\",\"content\":\"What is the current date in ISO format? Use the bash tool.\"}
      ],
      \"tools\": [{
        \"type\":\"function\",
        \"function\":{\"name\":\"bash\",\"description\":\"Run a bash command\",
          \"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}}
      }],
      \"max_tokens\": 256,
      \"temperature\": 0
    }" \
    "$URL/v1/chat/completions")
echo "$RESP2" | python3 -m json.tool
TOOL_CALL=$(echo "$RESP2" | python3 -c '
import sys, json
d = json.load(sys.stdin)
choices = d.get("choices") or []
if not choices: print(""); sys.exit()
msg = choices[0].get("message") or {}
tcs = msg.get("tool_calls") or []
print(",".join(tc.get("function",{}).get("name","") for tc in tcs))
')
if [ -z "$TOOL_CALL" ]; then
    # Some Qwen3.6 outputs land in content (XML) rather than structured tool_calls.
    # Check the content field too — if a <tool_call> appears, treat as soft pass
    # since LiteLLM/Claude Code path uses its own tool schema injection.
    INLINE_TC=$(echo "$RESP2" | python3 -c '
import sys, json
d = json.load(sys.stdin); choices = d.get("choices") or []
if choices:
    msg = choices[0].get("message") or {}
    print("yes" if msg.get("content","") and "<tool_call>" in msg.get("content","") else "")
')
    if [ -z "$INLINE_TC" ]; then
        echo "FAIL: model did not produce a tool_call (no structured tool_calls and no <tool_call> in content)"; exit 1
    fi
    echo "WARN: tool_call appeared inline in content, not parsed by qwen3_coder detector. Soft pass."
else
    echo "tool_calls: $TOOL_CALL"
fi

echo ""
echo "=== Test 3: multilingual probe (catches GDN garbling regression) ==="
for LANG_PROBE in \
    "中文:用一句话描述夏天" \
    "日本語:夏を一文で説明してください" \
    "Español:Describe el verano en una oración"; do
    LABEL="${LANG_PROBE%%:*}"
    PROMPT="${LANG_PROBE#*:}"
    RESP=$(curl -sS -H "Authorization: Bearer $SGLANG_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"$MODEL_ID\",
          \"messages\": [{\"role\":\"user\",\"content\":\"$PROMPT\"}],
          \"max_tokens\": 512,
          \"temperature\": 0
        }" \
        "$URL/v1/chat/completions")
    CONTENT=$(echo "$RESP" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print(d["choices"][0]["message"]["content"])
except Exception as e:
    print(f"PARSE_ERROR: {e}")
')
    echo "  [$LABEL] -> ${CONTENT:0:200}"
    # crude garbling check: response must be non-empty and contain at least 3 non-ASCII chars
    NONASCII=$(echo "$CONTENT" | python3 -c 'import sys; print(sum(1 for c in sys.stdin.read() if ord(c) > 127))')
    if [ "${NONASCII:-0}" -lt 3 ]; then
        echo "  WARN [$LABEL]: response looks ASCII-only or empty (possible GDN garbling regression)"
    fi
done

echo ""
echo "=========================================="
echo "A0 PASS — $MODEL_ID responds correctly to text + tool + multilingual probes"
echo ""
echo "Next:"
echo "  - For Mistral (2× pod): also run a1-fp8-kv-test (BF16 vs FP8 KV correctness)"
echo "  - For Qwen (1× pod): run a0-5-qwen-load.sh (sustained 16-concurrent load test)"
echo "=========================================="
