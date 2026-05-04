#!/usr/bin/env python3
"""Run the quality battery cases against an OpenAI-compatible endpoint.

Outputs a JSON file with the request/response pairs, one per case.
Use compare.py afterward to diff two output files (e.g., BF16 vs FP8 KV).

Filler expansion: <<INJECT_FILLER_NK>> in case messages is replaced with
content sized to N*1000 ACTUAL TOKENS, using the model's tokenizer when
available. Falls back to a 3.5 chars/token heuristic if neither
mistral_common nor transformers is installed.

Refuses to run with fewer than MIN_CASES_PER_CATEGORY cases per category
unless ALLOW_SMALL_BATTERY=1 is set in the env.

Usage:
    python3 run.py \\
        --base-url http://127.0.0.1:30000 \\
        --api-key $SGLANG_API_KEY \\
        --cases cases.json \\
        --output results.json \\
        --label bf16-kv
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from collections import Counter
from typing import Any, Callable

import httpx

FILLER_TOKEN = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "

# Codex-recommended minimums per category for a meaningful FP8 KV correctness test
MIN_PER_CATEGORY = {
    "tool": 20,
    "json": 10,
    "code": 10,
    "needle": 5,
    "multiturn": 5,
    "vision": 5,
}


def make_filler_fn(model_path: str, revision: str) -> Callable[[int], str]:
    """Return a function: target_tokens -> filler_text matching that token count.

    Tries mistral_common first (most accurate for Mistral 3.x), then transformers
    AutoTokenizer, then a char-based heuristic.
    """
    # Attempt mistral_common
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tok = MistralTokenizer.from_hf_hub(model_path, revision=revision)

        def encode(s: str) -> int:
            return len(tok.instruct_tokenizer.tokenizer.encode(s, bos=False, eos=False))

        backend = "mistral_common"
    except Exception:
        try:
            from transformers import AutoTokenizer

            t = AutoTokenizer.from_pretrained(
                model_path, revision=revision, trust_remote_code=True
            )

            def encode(s: str) -> int:
                return len(t.encode(s, add_special_tokens=False))

            backend = "transformers"
        except Exception:
            encode = None
            backend = "char-heuristic-3.5"

    print(f"  filler tokenizer: {backend}", file=sys.stderr)

    def make_filler(target_tokens: int) -> str:
        if encode is None:
            # Heuristic fallback (3.5 chars/token, slightly conservative)
            return (
                FILLER_TOKEN * ((target_tokens * 35 // 10 // len(FILLER_TOKEN)) + 1)
            )[: target_tokens * 35 // 10]
        # Binary-search-ish approach: append until we reach target, then truncate
        s = FILLER_TOKEN
        while encode(s) < target_tokens:
            s = s * 2
        # Trim down by character until token count matches
        while encode(s) > target_tokens and len(s) > len(FILLER_TOKEN):
            s = s[: int(len(s) * target_tokens / encode(s)) + len(FILLER_TOKEN)]
        return s

    return make_filler


def expand_fillers(text: str, make_filler: Callable[[int], str]) -> str:
    def repl(m: re.Match) -> str:
        n_k = int(m.group(1))
        return make_filler(n_k * 1000)

    return re.sub(r"<<INJECT_FILLER_(\d+)K>>", repl, text)


def normalize_messages(
    messages: list[dict], make_filler: Callable[[int], str]
) -> list[dict]:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append({**m, "content": expand_fillers(c, make_filler)})
        else:
            out.append(m)
    return out


def run_case(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    case: dict,
    model: str,
    make_filler: Callable[[int], str],
) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "messages": normalize_messages(case["messages"], make_filler),
        "max_tokens": case.get("max_tokens", 256),
        "temperature": 0,
        "top_p": 1.0,
        "seed": 42,
        "stream": False,
    }
    if "tools" in case:
        body["tools"] = case["tools"]

    t0 = time.time()
    try:
        r = client.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=600,
        )
        r.raise_for_status()
        return {
            "id": case["id"],
            "category": case["category"],
            "elapsed_s": round(time.time() - t0, 2),
            "request": body,
            "response": r.json(),
            "error": None,
        }
    except Exception as e:
        return {
            "id": case["id"],
            "category": case["category"],
            "elapsed_s": round(time.time() - t0, 2),
            "request": body,
            "response": None,
            "error": repr(e),
        }


def check_battery_size(cases: list[dict]) -> None:
    counts = Counter(c["category"] for c in cases)
    short: list[str] = []
    for cat, want in MIN_PER_CATEGORY.items():
        got = counts.get(cat, 0)
        if got < want:
            short.append(f"{cat}={got}/{want}")
    if short:
        if os.environ.get("ALLOW_SMALL_BATTERY") == "1":
            print(
                f"  WARN: battery is undersized but ALLOW_SMALL_BATTERY=1 is set: {', '.join(short)}",
                file=sys.stderr,
            )
        else:
            print(f"  ERROR: battery undersized: {', '.join(short)}", file=sys.stderr)
            print(
                "  Expand cases.json or set ALLOW_SMALL_BATTERY=1 to override.",
                file=sys.stderr,
            )
            sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:30000")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--cases", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--label", required=True, help="e.g. bf16-kv or fp8-kv")
    ap.add_argument("--model", default="mistralai/Mistral-Medium-3.5-128B")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    cases_doc = json.loads(args.cases.read_text())
    cases = cases_doc["cases"]
    check_battery_size(cases)

    print(f"Running {len(cases)} cases under label '{args.label}'", file=sys.stderr)
    make_filler = make_filler_fn(args.model, args.revision)

    results: list[dict] = []
    with httpx.Client(timeout=600) as client:
        for i, case in enumerate(cases, 1):
            print(
                f"  [{i}/{len(cases)}] {case['id']} ({case['category']})",
                file=sys.stderr,
            )
            res = run_case(
                client, args.base_url, args.api_key, case, args.model, make_filler
            )
            results.append(res)
            if res["error"]:
                print(f"    ERROR: {res['error']}", file=sys.stderr)

    args.output.write_text(
        json.dumps(
            {
                "label": args.label,
                "model": args.model,
                "base_url": args.base_url,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
