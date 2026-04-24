# tests/

Canonical test cases lifted verbatim from
[runpod-workers/worker-vllm v2.14.0](https://github.com/runpod-workers/worker-vllm/blob/v2.14.0/.runpod/tests_json)
so our proxy handler is verified against the same input contract the upstream
worker promises to honor.

- `tests_json` — the two canonical cases (`basic_inference_test`,
  `openai_messages_test`) plus the recommended Hub config
- `run_tests.py` — submits each case to a RunPod endpoint via `/run` + polls
  `/status/<job-id>` and validates the response shape

## Run against a live endpoint

```bash
export RUNPOD_API_KEY=...  # or have ~/.runpod/config.toml (runpodctl default)
python tests/run_tests.py <ENDPOINT_ID>
```

Example:
```bash
python tests/run_tests.py vllm-siz44ua4w8iett
```

Exits 0 on all-pass, 1 on any-fail.

## What each test verifies

- **basic_inference_test** — the bare-prompt input shape
  (`{"input": {"prompt": "..."}}`). Our handler must wrap this into an OpenAI
  `/v1/chat/completions` request with a single user message.
- **openai_messages_test** — the full OpenAI chat completions input shape,
  forwarded verbatim to vLLM's HTTP server.

Both assert that the response contains a non-empty
`choices[0].message.content`.
