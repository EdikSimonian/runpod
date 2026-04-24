"""Run the canonical test cases against a live RunPod endpoint.

Source of truth for the test cases: tests/tests_json — lifted verbatim from
runpod-workers/worker-vllm v2.14.0's .runpod/tests_json so our handler stays
compatible with the same input contract.

Usage:
    export RUNPOD_API_KEY=...        # or ~/.runpod/config.toml (runpodctl)
    python tests/run_tests.py <ENDPOINT_ID>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import httpx

TESTS_JSON = pathlib.Path(__file__).parent / "tests_json"


def resolve_api_key() -> str:
    key = os.getenv("RUNPOD_API_KEY")
    if key:
        return key
    cfg = pathlib.Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.strip().startswith("apikey"):
                return line.split("=", 1)[1].strip().strip("'").strip('"')
    raise RuntimeError("RUNPOD_API_KEY not set and ~/.runpod/config.toml missing apikey")


def run_one(endpoint: str, api_key: str, test: dict[str, Any]) -> dict[str, Any]:
    timeout_ms = test.get("timeout", 60000)
    timeout_s = timeout_ms / 1000

    # Use /run (async) + status polling so we don't depend on /runsync's 30s cap.
    base = f"https://api.runpod.ai/v2/{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    r = httpx.post(f"{base}/run", headers=headers, json={"input": test["input"]}, timeout=30)
    r.raise_for_status()
    job_id = r.json()["id"]

    deadline = time.time() + timeout_s + 60  # +60s slack for scheduler
    while time.time() < deadline:
        s = httpx.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
        s.raise_for_status()
        data = s.json()
        status = data.get("status")
        if status in {"COMPLETED", "FAILED"}:
            return data
        time.sleep(2)

    return {"status": "TIMEOUT", "id": job_id}


def validate(test: dict[str, Any], result: dict[str, Any]) -> tuple[bool, str]:
    if result.get("status") != "COMPLETED":
        return False, f"status={result.get('status')} error={result.get('error')!r}"

    output = result.get("output")
    if output is None:
        return False, "no output in result"

    # Output is a list (return_aggregate_stream) of OpenAI chat completion chunks
    if not isinstance(output, list) or not output:
        return False, f"unexpected output shape: {type(output).__name__}"

    # For chat completion results, expect choices[0].message.content to contain text
    first = output[0]
    if isinstance(first, dict) and "choices" in first:
        choices = first.get("choices") or []
        if not choices:
            return False, "no choices in response"
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if not content.strip():
            return False, f"empty content: {msg!r}"
        return True, f"got {len(content)} chars: {content[:80]!r}"
    return False, f"no 'choices' in {first!r}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("endpoint", help="RunPod endpoint id (e.g. vllm-siz44ua4w8iett)")
    p.add_argument("--tests", default=str(TESTS_JSON),
                   help=f"Path to tests JSON (default: {TESTS_JSON})")
    args = p.parse_args()

    tests_doc = json.loads(pathlib.Path(args.tests).read_text())
    tests = tests_doc.get("tests") or []
    if not tests:
        print("no tests in file", file=sys.stderr)
        return 2

    api_key = resolve_api_key()

    failures = 0
    for t in tests:
        name = t.get("name", "<unnamed>")
        print(f"\n=== {name} ===")
        print(f"  input: {json.dumps(t['input'])[:120]}")
        t0 = time.time()
        result = run_one(args.endpoint, api_key, t)
        dt = time.time() - t0
        ok, msg = validate(t, result)
        status = "PASS" if ok else "FAIL"
        print(f"  {status} ({dt:.1f}s) — {msg}")
        if not ok:
            failures += 1
            # Print full result body for debugging
            print("  full result:")
            print("    " + json.dumps(result, indent=2).replace("\n", "\n    ")[:2000])

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
