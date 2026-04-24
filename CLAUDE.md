# RunPod — Qwen3.6-27B vLLM deployments

## Target model: Qwen3.6-27B (locked)

**The target model for every deployment is `Qwen/Qwen3.6-27B`** (or its quants).
Do not silently swap to Qwen3-32B or another family — the user has explicitly committed
to Qwen3.6-27B. If a constraint (GPU, vLLM version, quant availability) makes it
infeasible, surface the tradeoff; don't substitute.

Qwen3.6-27B is a **vision-language model** (architecture `Qwen3_5ForConditionalGeneration`,
`model_type: qwen3_5`). It has no text-only sibling. Implications:
- Cannot use the stock `runpod/worker-v1-vllm` image — that worker is text-only.
- Requires **vLLM ≥ 0.17.0** (Qwen3.5 architecture support landed in 0.17.0).
- Model repo ships no Python modeling files, so `trust_remote_code` doesn't rescue
  older vLLM versions.

## Serving stack

Queue-based serverless endpoint running a **custom image** (in `docker/`) that is
worker-vllm's handler forked from v2.14.0 with the base layers bumped:

- `nvidia/cuda:12.8.0-base-ubuntu22.04` (was 12.9.1 — 5090 host drivers reject ≥12.9)
- `vllm[flashinfer]==0.19.1` with PyTorch cu128 wheels (was 0.16.0 cu129)
- Everything else in `src/` and `builder/requirements.txt` is a verbatim copy of
  worker-vllm v2.14.0

Endpoint exposes `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/*`, consumed by
Claude Code through an Anthropic↔OpenAI translation proxy (LiteLLM or
claude-code-router — not yet set up). vLLM's continuous batching handles concurrency
inside a single worker.

Auth is the account API key at `~/.runpod/config.toml` (Bearer token). RunPod validates
at the edge — you can't set a custom LB bearer.

`handler.py` in the repo root is the old Ollama queue handler and is obsolete under the
vLLM setup.

## Quantization ladder by GPU

| GPU        | VRAM   | Quant used        | Model variant                       |
|------------|--------|-------------------|-------------------------------------|
| RTX 5090   | 32 GB  | **Q4** (AWQ Int4) | `cyankiwi/Qwen3.6-27B-AWQ-INT4`     |
| H100 80GB  | 80 GB  | **FP8**           | `Qwen/Qwen3.6-27B-FP8`              |
| MI300X     | 192 GB | **FP16** (BF16)   | `Qwen/Qwen3.6-27B` (native)         |

## Building the images

Two images, two contexts, two workflows — one per GPU platform. Both push to GHCR.

| Platform | Context        | Workflow                              | Resulting image                              |
|----------|----------------|---------------------------------------|----------------------------------------------|
| CUDA (5090, H100) | `docker-cuda/` | `.github/workflows/docker-cuda.yml` | `ghcr.io/ediksimonian/runpod-cuda:main`      |
| ROCm (MI300X)     | `docker-rocm/` | `.github/workflows/docker-rocm.yml` | `ghcr.io/ediksimonian/runpod-rocm:main`      |

Each workflow triggers only on changes to its own context or workflow file, and on
its own version tags (`cuda-v*` / `rocm-v*`). Also has `workflow_dispatch` for manual
runs.

Extra image tags produced by each workflow:
- `:main` — latest main build
- `:sha-<short-sha>` — pinned to a commit
- `:<semver>` — when a `cuda-vX.Y.Z` or `rocm-vX.Y.Z` tag is pushed

### CUDA image (docker-cuda/)

- Base: `nvidia/cuda:12.8.0-base-ubuntu22.04` (5090 hosts reject CUDA ≥ 12.9)
- Pip: `vllm[flashinfer]==0.19.1` + cu128 torch wheels
- Works on both RTX 5090 (Blackwell) and H100 (Hopper) — same image, different GPU.
- Compressed size: ~10 GB.

### ROCm image (docker-rocm/)

- Base: `rocm/vllm-dev:preview_releases_v0.20.0_20260422` (AMD-published; vLLM 0.20,
  ROCm 7.12). This is a *preview* tag; if it rolls off Docker Hub or you need newer
  vLLM, pick the closest tag from <https://hub.docker.com/r/rocm/vllm-dev/tags>.
  Reason we can't use `rocm/vllm:latest`: that stable tag is stuck at vLLM 0.16
  which pre-dates Qwen3.5 support.
- Just layers worker-vllm's `src/` handler and the runpod SDK on top.
- Compressed size: ~11 GB.

Weights are NOT baked into either image — they download to the network volume at
first run, controlled by `MODEL_NAME` on the template.

### First-time package visibility

After each workflow's first successful run, the resulting GHCR package is private.
For each package at
`https://github.com/EdikSimonian/runpod/pkgs/container/runpod-cuda` and
`https://github.com/EdikSimonian/runpod/pkgs/container/runpod-rocm`, either:

- Flip it public (Package → Settings → Change visibility → Public) — RunPod pulls
  with no auth; or
- Keep it private and register a GitHub PAT (scope `read:packages`):
  ```bash
  runpodctl registry create --name ghcr \
    --username ediksimonian --password <GITHUB_PAT_WITH_read:packages>
  ```
  Then attach that registry auth to each template in the console (no CLI flag yet).

### Baking weights into the image (optional)

If cold-start model download is unacceptable, add build args to a workflow:
```
--build-arg MODEL_NAME=cyankiwi/Qwen3.6-27B-AWQ-INT4 \
--build-arg BASE_PATH=/models \
--secret id=HF_TOKEN,env=HF_TOKEN
```
(Swap `BASE_PATH` to a local path so the download doesn't land in the serverless
`/runpod-volume` convention.)

## Current state

Fully torn down — no endpoint, no template, no network volume on RunPod.
Secret `HF_TOKEN` is created in the RunPod console and ready to inject via
`{{ RUNPOD_SECRET_HF_TOKEN }}`.

## Deploy playbook — RTX 5090 in US-IL-1 (Q4, Qwen3.6-27B)

```bash
IMAGE="ghcr.io/ediksimonian/runpod-cuda:main"   # built + pushed from ./docker

# 1. Network volume (30 GB fits one AWQ Int4 quant ~20 GB)
runpodctl network-volume create \
  --name qwen3-models-us-il --size 30 --data-center-id US-IL-1
# -> note volume id

# 2. Template
ENV_JSON=$(python3 -c "import json; print(json.dumps({
  'MODEL_NAME': 'cyankiwi/Qwen3.6-27B-AWQ-INT4',
  'QUANTIZATION': 'awq',
  'MAX_MODEL_LEN': '8192',
  'GPU_MEMORY_UTILIZATION': '0.92',
  'MAX_NUM_SEQS': '16',
  'ENABLE_AUTO_TOOL_CHOICE': 'true',
  'TOOL_CALL_PARSER': 'hermes',
  'TRUST_REMOTE_CODE': 'true',
  'BASE_PATH': '/runpod-volume',
  'HF_HOME': '/runpod-volume/huggingface-cache/hub',
  'HF_TOKEN': '{{ RUNPOD_SECRET_HF_TOKEN }}'}))")
runpodctl template create \
  --name qwen3.6-27b-awq-vllm \
  --image "$IMAGE" \
  --serverless --container-disk-in-gb 30 \
  --volume-mount-path /runpod-volume \
  --env "$ENV_JSON"
# -> note template id

# 3. Endpoint
runpodctl serverless create \
  --name qwen3.6-27b-vllm-5090 \
  --template-id <TEMPLATE_ID> \
  --network-volume-id <VOLUME_ID> \
  --data-center-ids US-IL-1 \
  --gpu-id "NVIDIA GeForce RTX 5090" \
  --gpu-count 1 --workers-min 0 --workers-max 1
```

Rationale for these settings:

- **Custom image (not `runpod/worker-v1-vllm`)** — stock worker ships vLLM 0.15/0.16
  which lack Qwen3.5 arch; our image uses vLLM 0.19.1 on CUDA 12.8. Must be rebuilt
  and pushed from `docker/` before creating the template.
- **`MAX_MODEL_LEN=8192`** — after 20.5 GB of AWQ weights on a 32 GB card, ~9 GB is
  left for KV. 16 users × 8 k context ≈ 34 k tokens total, fits. Raising context or
  concurrency will cause vLLM preemption. Bump either only when moving to H100.
- **`MAX_NUM_SEQS=16`** — matches the 16-user target.
- **`workers-min=0 max=1`** — single worker, scale to zero when idle.
- **`HF_HOME=/runpod-volume/huggingface-cache/hub`** — matches the path baked into
  the image ENV, persists the 20 GB download across cold starts.
- **`HF_TOKEN`** resolves via RunPod secret (console-only at
  <https://www.runpod.io/console/user/secrets>). Already created.

Once deployed (replace `<ENDPOINT_ID>` with the id from `runpodctl serverless create`):
```bash
KEY=$(grep apikey ~/.runpod/config.toml | sed "s/.*= *'\([^']*\).*/\1/")

# Smoke test
curl -sS https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"cyankiwi/Qwen3.6-27B-AWQ-INT4","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

# Watch worker state
curl -sS https://api.runpod.ai/v2/<ENDPOINT_ID>/health -H "Authorization: Bearer $KEY"
```

## Upgrade path — H100 80GB in US-KS-2 (FP8)

Why: more headroom, higher context, FP8 quality. `runpodctl serverless update` has no
`--gpu-id` flag and H100 isn't in US-IL-1, so the endpoint and volume must be recreated
in a new DC.

On H100 hosts you could theoretically drop our custom image and go back to
`runpod/worker-v1-vllm:v2.14.0` (cuda 12.9.1 is fine on Hopper), BUT that image still
has vLLM 0.16.0 which lacks Qwen3.5. **Keep using the same custom image** — vLLM 0.19.1
also runs fine on H100.

```bash
IMAGE="ghcr.io/ediksimonian/runpod-cuda:main"

# 1. New volume (50 GB fits FP8 weights ~31 GB + HF cache + headroom)
runpodctl network-volume create --name qwen3-models-us-ks --size 50 --data-center-id US-KS-2

# 2. Retune template env for H100 / FP8
ENV_JSON=$(python3 -c "import json; print(json.dumps({
  'MODEL_NAME': 'Qwen/Qwen3.6-27B-FP8',
  'MAX_MODEL_LEN': '32768',
  'GPU_MEMORY_UTILIZATION': '0.90',
  'MAX_NUM_SEQS': '32',
  'ENABLE_AUTO_TOOL_CHOICE': 'true',
  'TOOL_CALL_PARSER': 'hermes',
  'TRUST_REMOTE_CODE': 'true',
  'BASE_PATH': '/runpod-volume',
  'HF_HOME': '/runpod-volume/huggingface-cache/hub',
  'HF_TOKEN': '{{ RUNPOD_SECRET_HF_TOKEN }}'}))")
runpodctl template update <TEMPLATE_ID> --env "$ENV_JSON"
# drop QUANTIZATION — vLLM auto-detects FP8

# 3. Recreate endpoint (delete first to avoid double-billing)
runpodctl serverless delete <OLD_ENDPOINT_ID>
runpodctl serverless create \
  --name qwen3.6-27b-vllm-h100 \
  --template-id <TEMPLATE_ID> \
  --network-volume-id <NEW_VOLUME_ID> \
  --data-center-ids US-KS-2 \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 --workers-min 0 --workers-max 3
```

DCs with H100 in USA: `US-KS-2`, `US-GA-2`, `US-CA-2`, `US-TX-3`. Check stock with
`runpodctl datacenter list` before creating the volume.

## Upgrade path — AMD MI300X 192GB in EU-RO-1 (FP16 / BF16)

Why: ~1.6× the memory bandwidth of H100 (5.3 vs 3.35 TB/s → faster decode), 192 GB
VRAM lets you run Qwen3.6-27B in full BF16 with massive KV cache for 32+ users, no
quantization loss. Usually cheaper per hour than H100. Tradeoff: EU-only (no US
MI300X), less mature quant kernels (doesn't matter here since we're running full
precision), prefill of long prompts still faster on H100.

Availability: `EU-RO-1` only, Low stock. `runpodctl datacenter list` to confirm.

Use the ROCm image from `docker-rocm/` — same worker-vllm handler, AMD base.
```bash
IMAGE="ghcr.io/ediksimonian/runpod-rocm:main"

# 1. EU volume. BF16 weights for Qwen3.6-27B ~= 54 GB → size at 100 GB
runpodctl network-volume create --name qwen3-models-eu-ro --size 100 --data-center-id EU-RO-1

# 2. Template
ENV_JSON=$(python3 -c "import json; print(json.dumps({
  'MODEL_NAME': 'Qwen/Qwen3.6-27B',
  'MAX_MODEL_LEN': '32768',
  'GPU_MEMORY_UTILIZATION': '0.90',
  'MAX_NUM_SEQS': '32',
  'DTYPE': 'bfloat16',
  'ENABLE_AUTO_TOOL_CHOICE': 'true',
  'TOOL_CALL_PARSER': 'hermes',
  'TRUST_REMOTE_CODE': 'true',
  'BASE_PATH': '/runpod-volume',
  'HF_HOME': '/runpod-volume/huggingface-cache/hub',
  'HF_TOKEN': '{{ RUNPOD_SECRET_HF_TOKEN }}'}))")
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
  --gpu-count 1 --workers-min 0 --workers-max 2
```

Verify the exact GPU ID string with `runpodctl gpu list` before running.

## RunPod CLI gotchas

- `runpodctl serverless create` has no `--scaler-type`/`--scaler-value`/`--idle-timeout`
  flags; create first, then `runpodctl serverless update <id>` for those.
- `runpodctl serverless update` has no `--gpu-id` — changing GPU means delete + recreate.
- `runpodctl template create/update` has no `--volume-mount-path` on update and the
  create flag is effectively ignored for serverless (volumes always mount at
  `/runpod-volume`). Pin `BASE_PATH=/runpod-volume` in env to be defensive.
- RunPod **secrets** are manageable via GraphQL `secretCreate`/`secretDelete` mutations
  (`https://api.runpod.io/graphql`), or via the console at
  <https://www.runpod.io/console/user/secrets>. REST and runpodctl do not expose them.
  Reference them in env as `{{ RUNPOD_SECRET_<NAME> }}`.
- No CLI/REST field selects **Load Balancer** endpoint type — only the console UI does.
- Network volumes are **region-locked**; changing DC requires a new volume + re-download.
- **5090 hosts reject CUDA ≥ 12.9 images.** Blackwell drivers currently top out at CUDA
  12.8. Any image using a `nvidia/cuda:12.9*` base (including stock
  `runpod/worker-v1-vllm:v2.13+`) will hit `nvidia-container-cli: requirement error:
  unsatisfied condition: cuda>=12.9`. Our custom image uses the 12.8 base for this
  reason.
