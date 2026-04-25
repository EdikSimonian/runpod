# RunPod — Qwen3.6-27B vLLM deployments

## Pick up here next session

**Endpoint and volume were torn down at pause time.** What survived is zero-cost
to keep and reusable:

| Resource | ID | Status |
|---|---|---|
| Template | `7rr31v1f4o` | still on RunPod, ready to attach to a new endpoint |
| Registry auth | `cmoc95oxr0086jv06c3t9n2es` (`ghcr-ediksimonian`) | attached to template |
| RunPod secret | `HF_TOKEN` (console-only) | template references it |
| CUDA image | `ghcr.io/ediksimonian/runpod-cuda:main` (sha-bc34eda, public) | ready to pull |
| ROCm image | `ghcr.io/ediksimonian/runpod-rocm:main` (public) | ready to pull |

**Resume steps** (~5 min of CLI + a cold-boot wait):

1. **Commit the local split-Dockerfile change first** so the next build picks
   up the layer-splitting optimization:
   ```bash
   git diff docker-cuda/Dockerfile    # review
   git add docker-cuda/Dockerfile CLAUDE.md
   git commit -m "Split CUDA Dockerfile into torch/flashinfer/vllm layers"
   git push origin main
   # Wait ~15 min for GHA to rebuild and push a new :main image
   ```

2. **Recreate volume + endpoint** (template already exists and is wired up):
   ```bash
   SUFFIX=$(date +%H%M%S)
   VOL=$(runpodctl network-volume create \
     --name "qwen3-models-eu-se-$SUFFIX" --size 30 \
     --data-center-id EU-SE-1 | jq -r .id)
   EP=$(runpodctl serverless create \
     --name "qwen3.6-27b-vllm-a6000-$SUFFIX" \
     --template-id 7rr31v1f4o \
     --network-volume-id "$VOL" \
     --data-center-ids EU-SE-1 \
     --gpu-id "NVIDIA RTX A6000" \
     --gpu-count 1 --workers-min 1 --workers-max 1 | jq -r .id)
   echo "VOL=$VOL EP=$EP"
   ```

3. **Watch the cold boot** (expect ~8–12 min if split-Dockerfile change landed,
   otherwise ~12–15 min):
   ```bash
   KEY=$(grep apikey ~/.runpod/config.toml | sed "s/.*= *'\([^']*\).*/\1/")
   watch -n 20 "curl -sS https://api.runpod.ai/v2/$EP/health -H 'Authorization: Bearer \$KEY' | jq .workers"
   ```

4. **Verify with canonical tests** (both should PASS now that
   `LIMIT_MM_PER_PROMPT` JSON format is in the template):
   ```bash
   python tests/run_tests.py "$EP"
   ```

5. **Fire a real `/openai/v1` request** (what LiteLLM + Claude Code will do):
   ```bash
   curl -sS https://api.runpod.ai/v2/$EP/openai/v1/chat/completions \
     -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"model":"cyankiwi/Qwen3.6-27B-AWQ-INT4","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
   ```

6. **After green**, drop Active-tier billing:
   ```bash
   curl -sS -X PATCH "https://rest.runpod.io/v1/endpoints/$EP" \
     -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"workersMin":0}'
   ```

7. **Wire up LiteLLM + Claude Code** — recipe at the bottom of this file.

**Uncommitted working-tree changes at session end** (check `git status`):
- `docker-cuda/Dockerfile` — three-way layer split (torch / flashinfer / vllm)
  so parallel-pull halves first-boot time. Pins torch 2.10.0, flashinfer 0.6.6
  per vLLM 0.19.1 `requirements/cuda.txt`.
- `CLAUDE.md` — this document itself.
- Also `D server.py D start.sh` — leftover Ollama-era files that were removed
  from the working tree long ago but not staged. Decide separately.

**Known issues resolved but not yet end-to-end verified:**
- `log_error_stack` vLLM-0.19 API drift → solved by replacing worker-vllm's
  Python wrapper with the subprocess HTTP proxy handler.
- `compressed-tensors` vs `awq` quant mismatch → `QUANTIZATION=compressed-tensors`
  in template env.
- `LIMIT_MM_PER_PROMPT` format change from `image=5,video=2` →
  `{"image":5,"video":2}` → template env now uses JSON.

Expected spend to green-light on resume: ~$0.50–1.00 of Active-tier billing
during the final cold boot + test run.

**If you want to wipe everything instead of resuming:**
```bash
runpodctl template delete 7rr31v1f4o
runpodctl registry delete cmoc95oxr0086jv06c3t9n2es
# HF_TOKEN secret: delete in the console at
# https://www.runpod.io/console/user/secrets
# GHCR images cost nothing to leave; delete at
# https://github.com/EdikSimonian?tab=packages if you want
```

---


## Target model: Qwen3.6-27B (locked)

**The target model for every deployment is `Qwen/Qwen3.6-27B`** (or its quants).
Do not silently swap to Qwen3-32B or another family — the user has explicitly
committed to Qwen3.6-27B. If a constraint (GPU, vLLM version, quant availability)
makes it infeasible, surface the tradeoff; don't substitute.

Qwen3.6-27B is a **vision-language model** (architecture
`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`). Takes text + image +
video input, outputs **text only** (no image/video generation). Has no text-only
sibling. Implications:

- Cannot use the stock `runpod/worker-v1-vllm` image — that worker is text-only.
- Requires **vLLM ≥ 0.17.0** (Qwen3.5 architecture support landed in 0.17.0).
- Model repo ships no Python modeling files, so `trust_remote_code` doesn't rescue
  older vLLM versions.

## Serving stack: HTTP proxy in front of `vllm serve`

Queue-based serverless endpoint. The container runs a 240-line RunPod handler
(`src/handler.py`) that:

1. Spawns `python -m vllm.entrypoints.openai.api_server` as a subprocess at
   container start
2. Waits up to 20 min for `/health` to return 200
3. Forwards RunPod job inputs to localhost:8000 via HTTP (handles streaming +
   non-streaming + `/v1/models`)

**Why we don't use worker-vllm's Python-layer wrapper.** worker-vllm v2.14.0
imports `vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat` and
friends directly. vLLM 0.17+ restructured that layer (new required
`openai_serving_render` kwarg, removed `log_error_stack`, split Chat/Render), so
every public fork that bumped vLLM without touching engine.py (YeDaxia, bgeneto,
linllm, and upstream itself) crashes on engine init. vLLM's **HTTP** API is
stable across versions; its Python surface is not. The proxy handler treats vLLM
as a subprocess with a stable REST contract and survives future upgrades with
zero code changes.

Endpoint exposes `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/*`, consumed
by Claude Code through an Anthropic↔OpenAI translation proxy (LiteLLM or
claude-code-router — not yet set up). vLLM's continuous batching handles
concurrency inside a single worker.

Auth is the account API key at `~/.runpod/config.toml` (Bearer token). RunPod
validates at the edge — you can't set a custom LB bearer.

The handler supports three input shapes (compatible with worker-vllm v2.14.0
tests_json):
- `{"input": {"openai_route": "/v1/chat/completions", "openai_input": {...}}}`
- `{"input": {"openai_route": "/v1/models"}}` (GET)
- `{"input": {"prompt": "...", "max_tokens": ..., ...}}` (legacy shorthand; wrapped
  into a single-user-message chat completion)

## Quantization ladder by GPU

| GPU        | VRAM   | Quant used        | Model variant                       |
|------------|--------|-------------------|-------------------------------------|
| RTX A6000  | 48 GB  | **Q4** (compressed-tensors / AWQ Int4) | `cyankiwi/Qwen3.6-27B-AWQ-INT4` |
| RTX 5090   | 32 GB  | **Q4** (same) — tight | `cyankiwi/Qwen3.6-27B-AWQ-INT4` |
| H100 80GB  | 80 GB  | **FP8**           | `Qwen/Qwen3.6-27B-FP8`              |
| MI300X     | 192 GB | **FP16** (BF16)   | `Qwen/Qwen3.6-27B` (native)         |

**Quant gotcha for `cyankiwi/Qwen3.6-27B-AWQ-INT4`:** despite "AWQ" in the repo
name, the model's `config.json` declares
`quantization_config.format: pack-quantized` which is the **compressed-tensors**
library's packed format, NOT classic AWQ. vLLM validates the model-config quant
against the `--quantization` argument at load time, so passing
`QUANTIZATION=awq` fails with:
```
Quantization method specified in the model config (compressed-tensors) does not
match the quantization method specified in the `quantization` argument (awq)
```
Set **`QUANTIZATION=compressed-tensors`**. Always read
`huggingface.co/<repo>/raw/main/config.json → quantization_config.format` before
picking the vLLM quant arg; the HF repo name can lie.

## Repo layout

```
docker-cuda/            CUDA image context (5090, H100)
  Dockerfile            nvidia/cuda:12.8.0 + vllm[flashinfer]==0.19.1 + cu128 torch
  src/handler.py        The 240-line subprocess + HTTP proxy handler
  builder/requirements.txt  runpod, httpx, hf-transfer (that's it)

docker-rocm/            ROCm image context (MI300X)
  Dockerfile            rocm/vllm-dev:preview_releases_v0.20.0_20260422
  src/handler.py        Same handler as docker-cuda
  builder/requirements.txt  Same trim

.github/workflows/
  docker-cuda.yml       Builds on docker-cuda/**, pushes ghcr.io/.../runpod-cuda
  docker-rocm.yml       Builds on docker-rocm/**, pushes ghcr.io/.../runpod-rocm

tests/
  tests_json            Lifted verbatim from worker-vllm v2.14.0/.runpod/tests_json
  run_tests.py          Fires both cases at a live endpoint, validates response
  README.md             How to run
```

## Building the images (CI on GitHub Actions)

Two images, two contexts, two workflows. Both push to GHCR as public packages.

| Platform       | Context        | Workflow                | Image |
|----------------|----------------|-------------------------|-------|
| CUDA (5090 / A6000 / H100) | `docker-cuda/` | `.github/workflows/docker-cuda.yml` | `ghcr.io/ediksimonian/runpod-cuda:main` |
| ROCm (MI300X)  | `docker-rocm/` | `.github/workflows/docker-rocm.yml` | `ghcr.io/ediksimonian/runpod-rocm:main` |

Each workflow triggers only on changes to its own context or workflow file, on
its own version tags (`cuda-v*` / `rocm-v*`), and via `workflow_dispatch` for
manual runs. Caches are scoped per-platform (`cache-from: type=gha,scope=cuda` /
`scope=rocm`) so they don't collide.

Resulting tags per workflow:
- `:main` — latest main build
- `:sha-<short-sha>` — pinned to a commit
- `:<semver>` — when a `cuda-vX.Y.Z` or `rocm-vX.Y.Z` tag is pushed

First build per workflow pushes the package as **public** (verified). RunPod
pulls with anonymous auth on public packages; we also have a registered GHCR
PAT as belt-and-suspenders (see "Reusable state" below).

### CUDA image (docker-cuda/)
- Base: `nvidia/cuda:12.8.0-base-ubuntu22.04` (5090 hosts reject CUDA ≥ 12.9 —
  Blackwell drivers top out at 12.8).
- Pip: `vllm[flashinfer]==0.19.1` + cu128 torch wheels.
- Runs on 5090 (Blackwell), A6000 (Ampere), H100 (Hopper) — same image, the GPU
  choice lives on the endpoint.
- Compressed size: ~8 GB.

### ROCm image (docker-rocm/)
- Base: `rocm/vllm-dev:preview_releases_v0.20.0_20260422` (AMD-published, vLLM
  0.20, ROCm 7.12). If this tag ages off, pick the closest newer one from
  <https://hub.docker.com/r/rocm/vllm-dev/tags>. Don't use `rocm/vllm:latest` —
  it's stuck at vLLM 0.16 (pre-Qwen3.5).
- Installs our 3 requirements on top with `--no-deps` so pip doesn't try to
  re-resolve torch/transformers.
- Compressed size: ~13 GB.

Weights are **not** baked in — they download to the network volume at first
run, controlled by `MODEL_NAME` on the template.

### Speed up the HF model download: `HF_HUB_ENABLE_HF_TRANSFER=1`
Both Dockerfiles set it by default. `hf-transfer` is a Rust parallel-chunk
downloader that delivers 100–250 MB/s vs 30–60 MB/s from the stock client, so
the 20 GB AWQ weights download in ~2–3 min instead of 5–10.

### Baking weights into the image (optional, not currently used)
Add build args to a workflow:
```
--build-arg MODEL_NAME=cyankiwi/Qwen3.6-27B-AWQ-INT4
--build-arg BASE_PATH=/models
--secret id=HF_TOKEN,env=HF_TOKEN
```
Swap `BASE_PATH` to a local path (not `/runpod-volume`) so the download doesn't
land in the serverless volume convention.

## Tests

`tests/tests_json` — lifted verbatim from worker-vllm v2.14.0 so our handler is
verified against the exact same input contract the upstream worker promises.
Two cases:
- `basic_inference_test` — bare prompt shorthand
- `openai_messages_test` — full OpenAI chat-completions body with system + user
  messages

Run against a live endpoint:
```bash
python tests/run_tests.py <ENDPOINT_ID>
```
Exits 0 on all-pass. API key comes from `RUNPOD_API_KEY` env or
`~/.runpod/config.toml`.

## Current state

**Endpoint and volume torn down.** Template survived and is pre-configured:

```
template:        7rr31v1f4o  (qwen3.6-27b-awq-vllm-a6000-221511)
  image          ghcr.io/ediksimonian/runpod-cuda:main (sha-bc34eda)
  env            MODEL_NAME=cyankiwi/Qwen3.6-27B-AWQ-INT4
                 QUANTIZATION=compressed-tensors
                 MAX_MODEL_LEN=16384
                 MAX_NUM_SEQS=16
                 GPU_MEMORY_UTILIZATION=0.92
                 LIMIT_MM_PER_PROMPT={"image":5,"video":2}  (JSON, vLLM 0.19 req)
                 ENABLE_AUTO_TOOL_CHOICE=true
                 TOOL_CALL_PARSER=hermes
                 TRUST_REMOTE_CODE=true
                 BASE_PATH=/runpod-volume
                 HF_HOME=/runpod-volume/huggingface-cache/hub
                 HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
                 HF_HUB_ENABLE_HF_TRANSFER=1
  container      30 GB
  volume mount   /runpod-volume
  registry auth  cmoc95oxr0086jv06c3t9n2es (ghcr-ediksimonian) — attached
```

Target GPU on resume: **NVIDIA RTX A6000** in **EU-SE-1** (only DC with both
volume support AND current A6000 stock).

Pricing: Flex $1.22/hr active | Active $1.04/hr always-on. With
`workers-min=0` post-verification, idle cost is $0.

Template env (full):
```
MODEL_NAME                 = cyankiwi/Qwen3.6-27B-AWQ-INT4
QUANTIZATION               = compressed-tensors  # NOT 'awq' (see quant gotcha)
MAX_MODEL_LEN              = 16384
GPU_MEMORY_UTILIZATION     = 0.92
MAX_NUM_SEQS               = 16
LIMIT_MM_PER_PROMPT        = {"image":5,"video":2}  # vision + video input enabled
ENABLE_AUTO_TOOL_CHOICE    = true
TOOL_CALL_PARSER           = hermes
TRUST_REMOTE_CODE          = true
BASE_PATH                  = /runpod-volume
HF_HOME                    = /runpod-volume/huggingface-cache/hub
HF_TOKEN                   = {{ RUNPOD_SECRET_HF_TOKEN }}
HF_HUB_ENABLE_HF_TRANSFER  = 1
```

Why A6000 in EU-SE-1:
- A6000 is the cheapest serverless GPU with real headroom for 16 users (48 GB =
  20 GB weights + 24 GB KV cache = 16 users × 12k context easily).
- EU-SE-1 is the only DC with both volume support AND current A6000 stock.
- GPUs with wider availability (H100) cost 3× more per hour at Flex tier.

## Reusable state (persists across re-deploys)

- **GHCR registry auth**: id `cmoc95oxr0086jv06c3t9n2es`, name
  `ghcr-ediksimonian`. Created with:
  ```bash
  runpodctl registry create --name ghcr-ediksimonian \
    --username ediksimonian --password $(gh auth token)
  ```
  Attach to a new template via REST (no CLI flag yet):
  ```bash
  curl -sS -X PATCH "https://rest.runpod.io/v1/templates/<TPL>" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d '{"containerRegistryAuthId":"cmoc95oxr0086jv06c3t9n2es"}'
  ```
- **`HF_TOKEN` RunPod secret** — created in the console at
  <https://www.runpod.io/console/user/secrets>. Templates reference it as
  `{{ RUNPOD_SECRET_HF_TOKEN }}`. Required because anon HF pulls are rate-limited
  (you'll see `Warning: You are sending unauthenticated requests to the HF Hub`
  in worker logs if the secret isn't resolved).
- **Published GHCR images** (public): `ghcr.io/ediksimonian/runpod-cuda:main`,
  `ghcr.io/ediksimonian/runpod-rocm:main`, each tagged also with `:sha-<short>`.
  They persist forever on GHCR; no re-cost to re-pull.

## Deploy playbook — from zero (A6000 in EU-SE-1)

```bash
set -e
SUFFIX=$(date +%H%M%S)
KEY=$(grep apikey ~/.runpod/config.toml | sed "s/.*= *'\([^']*\).*/\1/")
GHCR_AUTH=cmoc95oxr0086jv06c3t9n2es

# 1. Volume (30 GB fits one AWQ Int4 ~20 GB + overhead)
VOL=$(runpodctl network-volume create \
  --name "qwen3-models-eu-se-$SUFFIX" --size 30 --data-center-id EU-SE-1 \
  | jq -r .id)

# 2. Template
ENV_JSON=$(python3 -c "import json; print(json.dumps({
  'MODEL_NAME': 'cyankiwi/Qwen3.6-27B-AWQ-INT4',
  'QUANTIZATION': 'compressed-tensors',
  'MAX_MODEL_LEN': '16384',
  'GPU_MEMORY_UTILIZATION': '0.92',
  'MAX_NUM_SEQS': '16',
  'LIMIT_MM_PER_PROMPT': '{"image":5,"video":2}',
  'ENABLE_AUTO_TOOL_CHOICE': 'true',
  'TOOL_CALL_PARSER': 'hermes',
  'TRUST_REMOTE_CODE': 'true',
  'BASE_PATH': '/runpod-volume',
  'HF_HOME': '/runpod-volume/huggingface-cache/hub',
  'HF_TOKEN': '{{ RUNPOD_SECRET_HF_TOKEN }}',
  'HF_HUB_ENABLE_HF_TRANSFER': '1'}))")
TPL=$(runpodctl template create \
  --name "qwen3.6-27b-awq-vllm-a6000-$SUFFIX" \
  --image "ghcr.io/ediksimonian/runpod-cuda:main" \
  --serverless --container-disk-in-gb 30 \
  --volume-mount-path /runpod-volume \
  --env "$ENV_JSON" | jq -r .id)

# Attach GHCR registry auth
curl -sS -X PATCH "https://rest.runpod.io/v1/templates/$TPL" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"containerRegistryAuthId\":\"$GHCR_AUTH\"}" > /dev/null

# 3. Endpoint (workers-min=1 holds the host through first image pull — avoids
#    the throttle-and-redeploy loop on low-stock DCs)
EP=$(runpodctl serverless create \
  --name "qwen3.6-27b-vllm-a6000-$SUFFIX" \
  --template-id "$TPL" \
  --network-volume-id "$VOL" \
  --data-center-ids EU-SE-1 \
  --gpu-id "NVIDIA RTX A6000" \
  --gpu-count 1 --workers-min 1 --workers-max 1 | jq -r .id)

echo "VOL=$VOL TPL=$TPL EP=$EP"

# 4. Verify (takes ~10–15 min for first cold boot)
python tests/run_tests.py "$EP"

# 5. After tests pass, drop workers-min to 0 via REST (saves idle cost)
curl -sS -X PATCH "https://rest.runpod.io/v1/endpoints/$EP" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"workersMin":0}' > /dev/null
```

## Upgrade path — H100 80GB (FP8)

Drop to FP8 weights, raise MAX_MODEL_LEN and MAX_NUM_SEQS. Same docker-cuda
image. DCs with H100 + volume support: `US-KS-2, US-GA-2, US-CA-2, US-MO-1,
US-NE-1`. Check stock with `runpodctl datacenter list` first.

```bash
# 1. New volume in target DC (50 GB for FP8 ~31 GB + HF cache + headroom)
runpodctl network-volume create --name qwen3-models-us-ks --size 50 \
  --data-center-id US-KS-2

# 2. Retune template env — change MODEL_NAME + drop QUANTIZATION + bump ctx/seqs
ENV_JSON=$(python3 -c "import json; print(json.dumps({
  'MODEL_NAME': 'Qwen/Qwen3.6-27B-FP8',
  'MAX_MODEL_LEN': '32768',
  'GPU_MEMORY_UTILIZATION': '0.90',
  'MAX_NUM_SEQS': '32',
  'LIMIT_MM_PER_PROMPT': '{"image":5,"video":2}',
  'ENABLE_AUTO_TOOL_CHOICE': 'true',
  'TOOL_CALL_PARSER': 'hermes',
  'TRUST_REMOTE_CODE': 'true',
  'BASE_PATH': '/runpod-volume',
  'HF_HOME': '/runpod-volume/huggingface-cache/hub',
  'HF_TOKEN': '{{ RUNPOD_SECRET_HF_TOKEN }}',
  'HF_HUB_ENABLE_HF_TRANSFER': '1'}))")
runpodctl template update <TEMPLATE_ID> --env "$ENV_JSON"

# 3. Delete old endpoint, create new one on H100 + new volume
runpodctl serverless delete <OLD_ENDPOINT_ID>
runpodctl serverless create \
  --name qwen3.6-27b-vllm-h100 \
  --template-id <TEMPLATE_ID> \
  --network-volume-id <NEW_VOLUME_ID> \
  --data-center-ids US-KS-2 \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 --workers-min 0 --workers-max 3
```

## Upgrade path — AMD MI300X (BF16)

Why: 192 GB VRAM + 5.3 TB/s memory bandwidth. Full BF16 with huge KV cache for
32+ users, no quantization loss. **Availability: EU-RO-1 only, Low stock,
secure cloud only** — the "community MI300X" $0.50/hr price RunPod displays is
phantom (the `communityCloud: false` flag on the GPU type confirms no community
providers offer it).

```bash
IMAGE="ghcr.io/ediksimonian/runpod-rocm:main"

# 1. EU volume (100 GB fits BF16 ~54 GB + HF cache + other quants later)
runpodctl network-volume create --name qwen3-models-eu-ro --size 100 \
  --data-center-id EU-RO-1

# 2. Template
ENV_JSON=$(python3 -c "import json; print(json.dumps({
  'MODEL_NAME': 'Qwen/Qwen3.6-27B',
  'MAX_MODEL_LEN': '32768',
  'GPU_MEMORY_UTILIZATION': '0.90',
  'MAX_NUM_SEQS': '32',
  'DTYPE': 'bfloat16',
  'LIMIT_MM_PER_PROMPT': '{"image":5,"video":2}',
  'ENABLE_AUTO_TOOL_CHOICE': 'true',
  'TOOL_CALL_PARSER': 'hermes',
  'TRUST_REMOTE_CODE': 'true',
  'BASE_PATH': '/runpod-volume',
  'HF_HOME': '/runpod-volume/huggingface-cache/hub',
  'HF_TOKEN': '{{ RUNPOD_SECRET_HF_TOKEN }}',
  'HF_HUB_ENABLE_HF_TRANSFER': '1'}))")
runpodctl template create \
  --name qwen3.6-27b-bf16-vllm-rocm \
  --image "$IMAGE" \
  --serverless --container-disk-in-gb 40 \
  --volume-mount-path /runpod-volume \
  --env "$ENV_JSON"

# 3. Endpoint
runpodctl serverless create \
  --name qwen3.6-27b-vllm-mi300x \
  --template-id <NEW_TEMPLATE_ID> \
  --network-volume-id <NEW_VOLUME_ID> \
  --data-center-ids EU-RO-1 \
  --gpu-id "AMD Instinct MI300X OAM" \
  --gpu-count 1 --workers-min 1 --workers-max 2
```

## Lessons banked from this session

- **Low-stock GPU DCs throttle mid-init.** On 5090 in US-IL-1 (Low stock) our
  worker spent ~3 min `initializing` then RunPod reclaimed the host
  (`throttled=1`) before the image pull finished. Retrying didn't help — same
  pattern repeated. **Mitigation**: start new endpoints with `workers-min=1`
  during the first cold boot to hold the host; drop to 0 after tests pass.
- **Image pull is per-host.** FlashBoot biases toward cached hosts but can't
  force it. First pull on a new host = full 8 GB transfer. On sparse-GPU DCs
  you keep eating cold pulls until enough hosts cache the image.
- **EUR-IS-2 has great 5090 stock but no volume support.** Error text:
  `"Data center EUR-IS-2 not found or does not support network volumes"`. DCs
  that do support volumes:
  `AP-JP-1, CA-MTL-3, CA-MTL-4, EU-CZ-1, EU-NL-1, EU-RO-1, EU-SE-1, EUR-IS-1,
  EUR-IS-3, EUR-NO-1, US-CA-2, US-GA-2, US-IL-1, US-KS-2, US-MO-1, US-MO-2,
  US-NC-1, US-NC-2, US-NE-1, US-TX-3, US-WA-1`.
- **`cyankiwi/Qwen3.6-27B-AWQ-INT4` uses compressed-tensors format, not AWQ.**
  The repo name is misleading. Read `config.json` → `quantization_config.format`
  to pick the right vLLM `--quantization` arg.
- **worker-vllm v2.14.0 is stale against vLLM 0.17+.** Upstream hasn't updated
  engine.py in 6+ weeks. All the "updated" forks (YeDaxia, bgeneto, linllm)
  bumped the pip pin but kept the broken imports; they'd crash the same way.
  This is why we rewrote as a subprocess+HTTP proxy handler — decouples us from
  vLLM's Python-API churn.
- **vLLM 0.17+ API drift specifics** (for future archaeology):
  - Removed `log_error_stack` kwarg from OpenAIServingChat / OpenAIServingCompletion
  - Added required `openai_serving_render` kwarg (new class in
    `vllm/entrypoints/serve/render/serving.py`) — no default
  - Some kwargs moved from Chat into Render (chat_template,
    trust_request_chat_template, enable_auto_tools, etc.)
- **`--limit-mm-per-prompt` CLI format changed in vLLM 0.19.** Old format
  `image=5,video=2` now errors with `cannot be converted to <function loads>`.
  New format is **JSON**: `{"image":5,"video":2}`. The env var string passed
  into the template must be the JSON representation, e.g.
  `LIMIT_MM_PER_PROMPT: '{"image":5,"video":2}'`.
- **Image CUDA toolchain MUST match the vLLM wheel's compiled CUDA toolchain.**
  vLLM 0.19.1's published wheel embeds Marlin PTX compiled with CUDA 12.9.1
  (per their `docker/Dockerfile`'s `ARG CUDA_VERSION=12.9.1`). If our image
  uses any older CUDA base (e.g. 12.8), the wheel's PTX is rejected at engine
  init with:
  ```
  torch.AcceleratorError: CUDA error: the provided PTX was compiled with an
  unsupported toolchain
  ```
  triggered from `vllm/model_executor/kernels/linear/mixed_precision/marlin.py
  ::transform_w_s` during `process_weights_after_loading`. **This is NOT a
  host-driver problem and can't be fixed with `allowedCudaVersions`** — the
  image itself ships PTX the runtime can't accept. Our `docker-cuda/Dockerfile`
  pins to `nvidia/cuda:12.9.1-base` + cu129 torch wheels so it matches.
  Trade: the resulting image won't run on RTX 5090 hosts (drivers cap at 12.8).
- **`allowedCudaVersions` is the right tool for host-driver filtering**, but
  only after you've matched the image's CUDA toolchain to the wheel. Set
  `["12.9"]` on every endpoint via REST PATCH after creation
  (`runpodctl serverless create` doesn't expose this flag):
  ```bash
  curl -sS -X PATCH "https://rest.runpod.io/v1/endpoints/$EP" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d '{"allowedCudaVersions":["12.9"]}'
  ```
- **Blackwell silicon (sm_120) requires newer toolchain than vLLM 0.19.1
  ships.** RTX PRO 6000 (Blackwell Server) and RTX 5090 fail with the same
  PTX error even on hosts with new drivers, because vLLM's Marlin PTX simply
  doesn't include sm_120. Stick to Ampere (A6000, A100), Hopper (H100, H200),
  or Ada (L40, L40S, RTX 6000 Ada) until vLLM ≥ 0.20 lands with sm_120 PTX.
- **"Phantom" community pricing.** RunPod's GraphQL `gpuTypes` returns a
  `communityPrice` even for GPUs that have `communityCloud: false` (MI300X is
  the case we hit — listed at $0.50/hr but no community provider actually
  offers it; only secure cloud at $1.99). Always check `communityCloud` /
  `secureCloud` booleans before trusting a price.
- **Template name collisions are sticky.** RunPod soft-deletes templates and
  the names remain reserved. Always append `-$(date +%H%M%S)` to template +
  endpoint names.

## RunPod CLI + API gotchas

- `runpodctl template list` **returns only RunPod-public templates**, not the
  user's own. Use the REST API for user templates:
  ```bash
  curl -sS "https://rest.runpod.io/v1/templates" -H "Authorization: Bearer $KEY"
  ```
- `runpodctl serverless create` has no `--scaler-type` / `--scaler-value` /
  `--idle-timeout` / `--execution-timeout` flags; create first, then
  `runpodctl serverless update <id>` for those.
- `runpodctl serverless update` has no `--gpu-id` — changing GPU means
  delete + recreate.
- `runpodctl template create/update` has no `--container-registry-auth-id`
  flag. Attach via REST PATCH
  `{"containerRegistryAuthId":"..."}` after creation.
- `runpodctl template create/update` has no `--volume-mount-path` on update,
  and the create flag is effectively ignored for serverless (volumes always
  mount at `/runpod-volume`). Pin `BASE_PATH=/runpod-volume` in env.
- RunPod **secrets** are manageable via GraphQL `secretCreate` / `secretDelete`
  mutations (`https://api.runpod.io/graphql?api_key=$KEY`), or via the console
  at <https://www.runpod.io/console/user/secrets>. REST and runpodctl don't
  expose them. Reference in env as `{{ RUNPOD_SECRET_<NAME> }}`.
- No CLI/REST field selects **Load Balancer** endpoint type — only the console
  UI. That's why we're on queue-based serverless (which is fine — vLLM's
  batching is the real concurrency win).
- Network volumes are **region-locked**; changing DC requires a new volume.
- **5090 hosts reject CUDA ≥ 12.9 images.** Blackwell drivers top out at 12.8.
  Any image with `nvidia/cuda:12.9*` base (e.g. stock
  `runpod/worker-v1-vllm:v2.13+`) fails with `nvidia-container-cli: requirement
  error: unsatisfied condition: cuda>=12.9`. Our docker-cuda image pins the
  12.8 base for this reason.
- Worker logs are only accessible via the RunPod console (**Serverless → endpoint
  → Workers → click worker → Logs**). No REST or runpodctl command returns
  them. They're purged when the worker terminates.

## Claude Code integration (not yet done)

Claude Code speaks the **Anthropic Messages API**, not OpenAI. Point it at a
translation proxy:

```bash
pip install 'litellm[proxy]'

# litellm_config.yaml
model_list:
  - model_name: qwen3.6-27b
    litellm_params:
      model: openai/cyankiwi/Qwen3.6-27B-AWQ-INT4
      api_base: https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
      api_key: os.environ/RUNPOD_API_KEY

litellm --config litellm_config.yaml --port 4000
```

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=anything  # LiteLLM auth
claude
```
