"""
RunPod Load Balancer worker — FastAPI gateway over Ollama.

Endpoints:
  POST /v1/messages           Anthropic Messages API  ← Claude Code
  POST /v1/chat/completions   OpenAI Chat Completions API
  ANY  /api/*                 Raw Ollama API passthrough
  GET  /health                Liveness + model status
"""
import json
import os
import time
import uuid
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

OLLAMA_BASE = "http://127.0.0.1:11434"
MODEL = os.environ.get("MODEL_NAME", "qwen3:27b")
API_KEY = os.environ.get("OLLAMA_API_KEY", "")

app = FastAPI(title="Ollama Gateway")


def _ollama_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


async def _auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    key = request.headers.get("x-api-key", "")
    if auth != f"Bearer {API_KEY}" and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── /v1/messages — Anthropic Messages API (Claude Code) ──────────────────────

async def _anthropic_sse(payload: dict, msg_id: str) -> AsyncGenerator[str, None]:
    """Convert Ollama streaming response to Anthropic SSE event format."""
    model = payload["model"]

    yield (
        f"event: message_start\n"
        f"data: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','model':model,'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    )
    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"

    input_tokens = output_tokens = 0

    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST", f"{OLLAMA_BASE}/api/chat",
            headers=_ollama_headers(), json=payload,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield (
                        f"event: content_block_delta\n"
                        f"data: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':content}})}\n\n"
                    )
                if chunk.get("done"):
                    input_tokens = chunk.get("prompt_eval_count", 0)
                    output_tokens = chunk.get("eval_count", 0)

    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
    yield (
        f"event: message_delta\n"
        f"data: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':output_tokens}})}\n\n"
    )
    yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


@app.post("/v1/messages")
async def messages(request: Request):
    await _auth(request)
    body = await request.json()

    system = body.get("system", "")
    if isinstance(system, list):
        system = " ".join(s.get("text", "") for s in system if s.get("type") == "text")

    msgs = list(body.get("messages", []))
    if system:
        msgs = [{"role": "system", "content": system}] + msgs

    payload = {
        "model":      body.get("model", MODEL),
        "messages":   msgs,
        "stream":     body.get("stream", False),
        "keep_alive": -1,
        "options": {
            "temperature": body.get("temperature", 0.7),
            "num_predict": body.get("max_tokens", -1),
        },
    }
    # Qwen3 extended thinking — enabled via {"thinking": {"type": "enabled", "budget_tokens": N}}
    if isinstance(body.get("thinking"), dict) and body["thinking"].get("type") == "enabled":
        payload["think"] = True

    msg_id = f"msg_{uuid.uuid4().hex[:20]}"

    if body.get("stream", False):
        return StreamingResponse(
            _anthropic_sse(payload, msg_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/chat", headers=_ollama_headers(), json=payload,
        )
        resp.raise_for_status()
    result = resp.json()

    msg = result.get("message", {})
    content_blocks = []
    if msg.get("thinking"):
        content_blocks.append({"type": "thinking", "thinking": msg["thinking"]})
    content_blocks.append({"type": "text", "text": msg.get("content", "")})

    return {
        "id":             msg_id,
        "type":           "message",
        "role":           "assistant",
        "model":          result.get("model", MODEL),
        "content":        content_blocks,
        "stop_reason":    "end_turn",
        "stop_sequence":  None,
        "usage": {
            "input_tokens":  result.get("prompt_eval_count", 0),
            "output_tokens": result.get("eval_count", 0),
        },
    }


# ── /v1/chat/completions — OpenAI API ────────────────────────────────────────

async def _openai_sse(payload: dict) -> AsyncGenerator[str, None]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model = payload["model"]
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST", f"{OLLAMA_BASE}/api/chat",
            headers=_ollama_headers(), json=payload,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                done = chunk.get("done", False)
                oc = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content} if content else {},
                        "finish_reason": "stop" if done else None,
                    }],
                }
                yield f"data: {json.dumps(oc)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await _auth(request)
    body = await request.json()

    payload = {
        "model":      body.get("model", MODEL),
        "messages":   body.get("messages", []),
        "stream":     body.get("stream", False),
        "keep_alive": -1,
        "options": {
            "temperature": body.get("temperature", 0.7),
            "top_p":       body.get("top_p", 0.9),
            "num_predict": body.get("max_tokens", -1),
        },
    }

    if body.get("stream", False):
        return StreamingResponse(
            _openai_sse(payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/chat", headers=_ollama_headers(), json=payload,
        )
        resp.raise_for_status()
    result = resp.json()

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   result.get("model", MODEL),
        "choices": [{"index": 0, "message": result.get("message", {}), "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens":     result.get("prompt_eval_count", 0),
            "completion_tokens": result.get("eval_count", 0),
            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
        },
    }


# ── /api/* — raw Ollama passthrough ──────────────────────────────────────────

@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE"])
async def ollama_passthrough(path: str, request: Request):
    await _auth(request)
    body = await request.body()
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.request(
            method=request.method,
            url=f"{OLLAMA_BASE}/api/{path}",
            headers=_ollama_headers(),
            content=body,
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags", headers=_ollama_headers())
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok" if ollama_ok else "degraded", "ollama": ollama_ok, "model": MODEL}
