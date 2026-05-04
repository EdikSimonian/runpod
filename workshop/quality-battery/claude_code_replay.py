#!/usr/bin/env python3
"""Synthetic Claude-Code-shaped traffic generator for A2 load testing.

Simulates N concurrent agentic sessions with:
- Shared system prompt + tool definitions (RadixAttention should cache these)
- Bursty per-session traffic (5 sequential turns then idle)
- Variable output lengths
- Growing conversation prefix per session

Usage:
    python3 claude_code_replay.py \\
        --base-url http://127.0.0.1:30000 \\
        --api-key $SGLANG_API_KEY \\
        --num-sessions 16 \\
        --turns-per-session 5 \\
        --output replay.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import statistics
import sys
import time

import httpx

SHARED_SYSTEM_PROMPT = (
    "You are Claude Code, an AI coding assistant. Help the user with software "
    "engineering tasks. Use the available tools to read files, edit code, and "
    "run commands. Be concise. Always read files before editing them. Verify "
    "your work by running tests when possible. Stay focused on the task at hand."
) * 8  # ~1k tokens of shared system prompt

SHARED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing old text with new",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "grep across files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
]

USER_PROMPTS = [
    "Find all TODO comments in src/ and summarize them.",
    "Refactor the function `calculate_tax` in billing.py to use the new rate constants.",
    "Why is the test `test_user_login` failing? Read the test and the implementation.",
    "Add type hints to all public functions in api/handlers.py.",
    "Create a new module `utils/retry.py` with a generic exponential-backoff decorator.",
    "Audit the codebase for any uses of deprecated `requests.get(verify=False)`.",
]

TOOL_OBSERVATIONS = [
    "(file contents: 800 lines of Python)",
    "(grep matches: 23 lines across 7 files)",
    "(command stdout: tests passed, 14/14)",
    "(file contents: empty)",
    "(error: file not found)",
]


async def run_session(
    session_id: int,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    num_turns: int,
    model: str,
) -> dict:
    messages: list[dict] = [{"role": "system", "content": SHARED_SYSTEM_PROMPT}]
    turn_metrics: list[dict] = []

    for turn in range(num_turns):
        # Add a user prompt
        prompt = random.choice(USER_PROMPTS) + f" (session {session_id}, turn {turn})"
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": model,
            "messages": messages,
            "tools": SHARED_TOOLS,
            "max_tokens": random.choice([256, 1024, 4096]),
            "temperature": 0.7,
            "stream": True,
        }

        t0 = time.time()
        ttft = None
        chunks = 0
        last_content = ""
        try:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=600,
            ) as r:
                if r.status_code != 200:
                    err = await r.aread()
                    turn_metrics.append(
                        {
                            "turn": turn,
                            "error": f"status {r.status_code}: {err[:200]!r}",
                        }
                    )
                    return {
                        "session": session_id,
                        "turns": turn_metrics,
                        "error": f"http {r.status_code}",
                    }
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks += 1
                    try:
                        d = json.loads(data)
                        delta = (d.get("choices", [{}])[0].get("delta", {}) or {}).get(
                            "content"
                        )
                        if delta:
                            last_content += delta
                    except Exception:
                        pass
        except Exception as e:
            turn_metrics.append({"turn": turn, "error": repr(e)})
            return {"session": session_id, "turns": turn_metrics, "error": repr(e)}

        elapsed = time.time() - t0
        turn_metrics.append(
            {
                "turn": turn,
                "ttft_s": round(ttft, 2) if ttft else None,
                "elapsed_s": round(elapsed, 2),
                "chunks": chunks,
                "output_chars": len(last_content),
            }
        )
        # Append assistant's reply + a fake tool result to grow the prefix
        messages.append({"role": "assistant", "content": last_content[:4000]})
        messages.append(
            {
                "role": "user",
                "content": "Result of last action: " + random.choice(TOOL_OBSERVATIONS),
            }
        )

    return {"session": session_id, "turns": turn_metrics, "error": None}


async def main_async(args) -> int:
    print(
        f"Spawning {args.num_sessions} sessions × {args.turns_per_session} turns",
        file=sys.stderr,
    )
    limits = httpx.Limits(
        max_connections=args.num_sessions * 2,
        max_keepalive_connections=args.num_sessions * 2,
    )
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        tasks = [
            run_session(
                i,
                client,
                args.base_url,
                args.api_key,
                args.turns_per_session,
                args.model,
            )
            for i in range(args.num_sessions)
        ]
        # Stagger session starts slightly to mimic bursty behavior
        results = []

        async def run_with_delay(t, delay):
            await asyncio.sleep(delay)
            return await t

        results = await asyncio.gather(
            *[run_with_delay(t, random.uniform(0, 5)) for t in tasks]
        )

    # Summarize
    all_ttft = [m["ttft_s"] for r in results for m in r["turns"] if m.get("ttft_s")]
    errors = [r for r in results if r.get("error")]
    print("\n--- Summary ---")
    print(f"  sessions completed: {len(results) - len(errors)}/{len(results)}")
    print(f"  errors: {len(errors)}")
    if all_ttft:
        all_ttft.sort()
        print(f"  TTFT mean: {statistics.mean(all_ttft):.2f}s")
        print(f"  TTFT P50: {all_ttft[len(all_ttft) // 2]:.2f}s")
        print(f"  TTFT P95: {all_ttft[int(len(all_ttft) * 0.95)]:.2f}s")
        print(
            f"  TTFT P99: {all_ttft[min(int(len(all_ttft) * 0.99), len(all_ttft) - 1)]:.2f}s"
        )

    args.output.write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "num_sessions": args.num_sessions,
                "turns_per_session": args.turns_per_session,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output}", file=sys.stderr)
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--num-sessions", type=int, default=16)
    ap.add_argument("--turns-per-session", type=int, default=5)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument(
        "--seed", type=int, default=42, help="Seed module random for reproducibility"
    )
    args = ap.parse_args()
    random.seed(args.seed)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
