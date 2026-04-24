"""RunPod serverless handler — thin HTTP proxy in front of vLLM's OpenAI server.

Rationale: we used to import vLLM's Python classes directly (worker-vllm pattern),
but vLLM's internal API churns between minor releases (0.16 -> 0.17 -> 0.19 broke
OpenAIServingChat signatures). This handler instead spawns `vllm serve` as a
subprocess and forwards OpenAI-shaped requests over HTTP. vLLM's HTTP contract is
stable across versions; we stay upstream-compatible by not touching internals.

Input shapes accepted (compatible with worker-vllm v2.14.0 tests_json):

  # OpenAI chat route — preferred
  {"input": {"openai_route": "/v1/chat/completions",
             "openai_input": {"messages": [...], ...}}}

  # OpenAI completion / models
  {"input": {"openai_route": "/v1/completions", "openai_input": {...}}}
  {"input": {"openai_route": "/v1/models"}}

  # Bare prompt — treated as single-user-message chat completion
  {"input": {"prompt": "...", "max_tokens": 200, "temperature": 0.7, ...}}
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, AsyncGenerator

import httpx
import runpod
from runpod import RunPodLogger

log = RunPodLogger()

VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}"
# `vllm serve` downloads the model if needed, then sets up the engine. For a
# 27 B model on a fresh host this is ~5–10 min, so give it plenty of room.
VLLM_READY_TIMEOUT = int(os.getenv("VLLM_READY_TIMEOUT", "1200"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))

# vllm serve flag mapping: ENV_VAR -> --cli-flag. Only forwarded when set.
_STRING_FLAGS: dict[str, str] = {
    "MAX_MODEL_LEN": "--max-model-len",
    "MAX_NUM_SEQS": "--max-num-seqs",
    "MAX_NUM_BATCHED_TOKENS": "--max-num-batched-tokens",
    "GPU_MEMORY_UTILIZATION": "--gpu-memory-utilization",
    "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
    "PIPELINE_PARALLEL_SIZE": "--pipeline-parallel-size",
    "DTYPE": "--dtype",
    "KV_CACHE_DTYPE": "--kv-cache-dtype",
    "QUANTIZATION": "--quantization",
    "TOKENIZER_NAME": "--tokenizer",
    "MODEL_REVISION": "--revision",
    "TOOL_CALL_PARSER": "--tool-call-parser",
    "REASONING_PARSER": "--reasoning-parser",
    "CHAT_TEMPLATE": "--chat-template",
    "RESPONSE_ROLE": "--response-role",
    "LIMIT_MM_PER_PROMPT": "--limit-mm-per-prompt",
    "SEED": "--seed",
    "CUSTOM_CHAT_TEMPLATE": "--chat-template",
    "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "--served-model-name",
}

_BOOL_FLAGS: dict[str, str] = {
    "TRUST_REMOTE_CODE": "--trust-remote-code",
    "ENABLE_AUTO_TOOL_CHOICE": "--enable-auto-tool-choice",
    "ENABLE_PREFIX_CACHING": "--enable-prefix-caching",
    "ENABLE_CHUNKED_PREFILL": "--enable-chunked-prefill",
    "ENFORCE_EAGER": "--enforce-eager",
    "DISABLE_LOG_STATS": "--disable-log-stats",
    "DISABLE_LOG_REQUESTS": "--disable-log-requests",
}


def _build_vllm_cmd() -> list[str]:
    model = os.getenv("MODEL_NAME")
    if not model:
        raise RuntimeError("MODEL_NAME env var is required")
    cmd: list[str] = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", model]
    cmd += ["--host", VLLM_HOST, "--port", str(VLLM_PORT)]
    for env, flag in _STRING_FLAGS.items():
        v = os.getenv(env)
        if v not in (None, ""):
            cmd += [flag, v]
    for env, flag in _BOOL_FLAGS.items():
        if os.getenv(env, "false").lower() == "true":
            cmd.append(flag)
    # Pass-through: anything with VLLM_RUNPOD_ prefix maps to --<lowercased-kebab>
    for key, val in os.environ.items():
        if key.startswith("VLLM_RUNPOD_") and val:
            cmd += [f"--{key[len('VLLM_RUNPOD_'):].lower().replace('_', '-')}", val]
    return cmd


_vllm_proc: subprocess.Popen[bytes] | None = None


def _spawn_vllm() -> subprocess.Popen[bytes]:
    cmd = _build_vllm_cmd()
    log.info(f"Spawning vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setsid)
    return proc


def _wait_for_vllm() -> None:
    deadline = time.time() + VLLM_READY_TIMEOUT
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Bail early if vllm subprocess has exited
        if _vllm_proc is not None and _vllm_proc.poll() is not None:
            raise RuntimeError(f"vllm subprocess exited with code {_vllm_proc.returncode} before becoming ready")
        try:
            r = httpx.get(f"{VLLM_BASE}/health", timeout=5.0)
            if r.status_code == 200:
                log.info(f"vLLM ready after {attempts} probes ({int(time.time() - (deadline - VLLM_READY_TIMEOUT))}s)")
                return
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM did not become ready within {VLLM_READY_TIMEOUT}s")


def _shutdown(*_: Any) -> None:
    global _vllm_proc
    if _vllm_proc and _vllm_proc.poll() is None:
        log.info("SIGTERM received — terminating vllm subprocess")
        try:
            os.killpg(os.getpgid(_vllm_proc.pid), signal.SIGTERM)
            _vllm_proc.wait(timeout=15)
        except Exception as exc:  # noqa: BLE001
            log.warn(f"vllm shutdown error: {exc}")
            try:
                os.killpg(os.getpgid(_vllm_proc.pid), signal.SIGKILL)
            except Exception:
                pass
    sys.exit(0)


def _normalize_input(job_input: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
    """Resolve the RunPod job input to (http_method, path, json_body)."""
    route = job_input.get("openai_route")
    if route:
        body = job_input.get("openai_input")
        if route == "/v1/models":
            return ("GET", route, None)
        if body is None:
            raise ValueError(f"openai_route={route} requires openai_input")
        return ("POST", route, body)

    # Legacy bare-prompt shape → wrap as chat completion.
    prompt = job_input.get("prompt")
    if prompt is not None:
        body = {
            "model": os.getenv("MODEL_NAME"),
            "messages": [{"role": "user", "content": prompt}],
        }
        for k in ("max_tokens", "temperature", "top_p", "top_k", "stop",
                 "frequency_penalty", "presence_penalty", "seed", "stream",
                 "tools", "tool_choice"):
            if k in job_input:
                body[k] = job_input[k]
        return ("POST", "/v1/chat/completions", body)

    raise ValueError("Job input must contain either 'openai_route'+'openai_input' or 'prompt'")


async def handler(job: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
    try:
        method, path, body = _normalize_input(job.get("input") or {})
    except ValueError as exc:
        yield {"error": str(exc)}
        return

    url = f"{VLLM_BASE}{path}"
    stream = bool(body and body.get("stream"))

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            if method == "GET":
                r = await client.get(url)
                yield r.json()
                return

            if not stream:
                r = await client.post(url, json=body)
                if r.status_code >= 400:
                    yield {"error": f"vllm returned {r.status_code}", "detail": r.text}
                    return
                yield r.json()
                return

            # Streaming: consume SSE and yield each data chunk to the client.
            async with client.stream("POST", url, json=body) as r:
                if r.status_code >= 400:
                    err_text = await r.aread()
                    yield {"error": f"vllm returned {r.status_code}", "detail": err_text.decode(errors="replace")}
                    return
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        log.warn(f"Non-JSON SSE frame skipped: {data!r}")
        except httpx.RequestError as exc:
            yield {"error": "vllm http error", "detail": repr(exc)}


def _max_concurrency_modifier(_: int) -> int:
    return int(os.getenv("MAX_CONCURRENCY", "32"))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        _vllm_proc = _spawn_vllm()
        _wait_for_vllm()
    except Exception as exc:  # noqa: BLE001
        log.error(f"vLLM bring-up failed: {exc}")
        _shutdown()
        raise

    log.info("vLLM ready — starting RunPod serverless loop")
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": _max_concurrency_modifier,
            "return_aggregate_stream": True,
        }
    )
