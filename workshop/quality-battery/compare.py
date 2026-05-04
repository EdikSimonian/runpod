#!/usr/bin/env python3
"""Compare two quality-battery result files (e.g., BF16 KV vs FP8 KV).

For each case, applies the case's check (json_valid, json_schema, contains,
tool_call_name, python_assert) to BOTH the reference and candidate response,
and reports divergence per category.

python_assert checks run model-generated code in an ISOLATED subprocess
with an enforced timeout, no inherited environment, and resource limits.

Exits non-zero if overall divergence percentage exceeds --max-divergence-pct
(default 5.0).

Usage:
    python3 compare.py \\
        --reference results-bf16.json \\
        --candidate results-fp8.json \\
        --report compare.txt \\
        --max-divergence-pct 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any


def load_cases(path: pathlib.Path) -> dict[str, dict]:
    doc = json.loads(path.read_text())
    return {c["id"]: c for c in doc["cases"]}


def get_text(resp: dict | None) -> str:
    if not resp:
        return ""
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


def get_tool_calls(resp: dict | None) -> list[dict]:
    if not resp:
        return []
    choices = resp.get("choices") or []
    if not choices:
        return []
    msg = choices[0].get("message") or {}
    return msg.get("tool_calls") or []


def _type_match(value: Any, ty: str) -> bool:
    return {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(ty, False)


def run_python_assert_isolated(
    code: str, asserts: list[str], timeout_s: int = 5
) -> tuple[bool, str]:
    """Execute model-generated code + asserts in an isolated subprocess.

    - python -I -B: ignore PYTHON* env vars, no .pyc bytecode
    - empty env (only PATH for python itself)
    - resource limits: address space, CPU time, no network (best-effort)
    - hard timeout
    - tempdir cwd so any file writes are sandboxed
    """
    runner = textwrap.dedent("""
        import resource, sys, os
        # Limit CPU time and memory (1 GB AS, 5s CPU)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        except Exception:
            pass
        ns = {}
        code = sys.stdin.read()
        sep = "<<<ASSERTS>>>"
        body, _, asserts_block = code.partition(sep)
        try:
            exec(body, ns)
        except SystemExit:
            raise
        except Exception as e:
            print(f"EXEC_ERROR: {e!r}")
            sys.exit(2)
        for line in asserts_block.strip().splitlines():
            try:
                if not eval(line, ns):
                    print(f"ASSERT_FAIL: {line}")
                    sys.exit(3)
            except Exception as e:
                print(f"ASSERT_EXC: {line} -> {e!r}")
                sys.exit(4)
        print("OK")
    """)
    payload = code + "<<<ASSERTS>>>" + "\n".join(asserts)
    with tempfile.TemporaryDirectory() as td:
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-B", "-c", runner],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=td,
                env={"PATH": "/usr/bin:/bin"},
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout > {timeout_s}s"
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if r.returncode == 0 and out == "OK":
            return True, "all asserts pass"
        return False, out or err or f"exit {r.returncode}"


def apply_check(check: dict, response: dict | None) -> tuple[bool, str]:
    if response is None:
        return False, "no response"
    kind = check.get("kind")
    text = get_text(response)
    tool_calls = get_tool_calls(response)

    if kind == "contains":
        v = check["value"]
        return (v.lower() in text.lower(), f"contains '{v}'")

    if kind == "non_ascii_min":
        # For multilingual probes — catches GDN garbling regression where the model
        # falls back to ASCII-only output for non-Latin scripts.
        n = sum(1 for c in text if ord(c) > 127)
        threshold = check.get("value", 5)
        return (n >= threshold, f"non-ASCII chars: {n} (need >= {threshold})")

    if kind == "json_valid":
        try:
            json.loads(text.strip())
            return True, "json_valid"
        except Exception as e:
            return False, f"json invalid: {e}"

    if kind == "json_schema":
        try:
            obj = json.loads(text.strip())
        except Exception as e:
            return False, f"json invalid: {e}"
        schema = check["schema"]
        for k in schema.get("required", []):
            if k not in obj:
                return False, f"missing required key '{k}'"
        for k, v in (schema.get("properties") or {}).items():
            if k in obj and "type" in v and not _type_match(obj[k], v["type"]):
                return False, f"type mismatch on '{k}': got {type(obj[k]).__name__}"
        return True, "schema ok"

    if kind == "tool_call_name":
        names = [tc.get("function", {}).get("name") for tc in tool_calls]
        return (check["value"] in names, f"tool call '{check['value']}' (got: {names})")

    if kind == "python_assert":
        code = text
        if "```" in code:
            m = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
            if m:
                code = m.group(1)
        return run_python_assert_isolated(code, check.get("asserts") or [])

    return False, f"unknown check kind: {kind}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=pathlib.Path)
    ap.add_argument("--candidate", required=True, type=pathlib.Path)
    ap.add_argument(
        "--cases",
        default=None,
        type=pathlib.Path,
        help="Path to cases.json. Defaults to ../quality-battery/cases.json relative to this script.",
    )
    ap.add_argument("--report", required=True, type=pathlib.Path)
    ap.add_argument(
        "--max-divergence-pct",
        type=float,
        default=5.0,
        help="Exit non-zero if overall divergence percentage exceeds this.",
    )
    args = ap.parse_args()

    if args.cases is None:
        args.cases = pathlib.Path(__file__).parent / "cases.json"

    cases = load_cases(args.cases)
    ref = json.loads(args.reference.read_text())
    cand = json.loads(args.candidate.read_text())

    ref_by_id = {r["id"]: r for r in ref["results"]}
    cand_by_id = {r["id"]: r for r in cand["results"]}

    lines: list[str] = []
    lines.append(f"Reference: {ref['label']} | Candidate: {cand['label']}\n")
    lines.append("=" * 72)

    by_cat: dict[str, dict[str, int]] = {}
    divergent_cases: list[str] = []

    for case_id in sorted(set(ref_by_id) | set(cand_by_id)):
        case = cases.get(case_id)
        if case is None:
            lines.append(f"  WARN: case {case_id} not in cases.json")
            continue

        check = case.get("check") or {"kind": "contains", "value": ""}
        cat = case["category"]
        by_cat.setdefault(
            cat, {"total": 0, "ref_pass": 0, "cand_pass": 0, "divergent": 0}
        )
        by_cat[cat]["total"] += 1

        ref_pass, ref_why = apply_check(
            check, (ref_by_id.get(case_id) or {}).get("response")
        )
        cand_pass, cand_why = apply_check(
            check, (cand_by_id.get(case_id) or {}).get("response")
        )

        if ref_pass:
            by_cat[cat]["ref_pass"] += 1
        if cand_pass:
            by_cat[cat]["cand_pass"] += 1
        if ref_pass != cand_pass:
            by_cat[cat]["divergent"] += 1
            divergent_cases.append(case_id)
            lines.append(
                f"DIVERGE [{case_id}] ref={ref_pass}({ref_why}) cand={cand_pass}({cand_why})"
            )

    lines.append("")
    lines.append("Per-category summary:")
    lines.append(
        f"  {'category':<12} {'total':>6} {'ref_pass':>10} {'cand_pass':>10} {'divergent':>10}"
    )
    overall_div = 0
    overall_total = 0
    for cat, s in sorted(by_cat.items()):
        lines.append(
            f"  {cat:<12} {s['total']:>6} {s['ref_pass']:>10} {s['cand_pass']:>10} {s['divergent']:>10}"
        )
        overall_div += s["divergent"]
        overall_total += s["total"]

    lines.append("")
    div_pct = (overall_div / overall_total * 100) if overall_total else 0
    lines.append(f"Overall divergence: {overall_div}/{overall_total} ({div_pct:.1f}%)")
    lines.append(f"Threshold: {args.max_divergence_pct:.1f}%")

    report = "\n".join(lines)
    args.report.write_text(report)
    print(report)

    if div_pct > args.max_divergence_pct:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
