# Workshop — Qwen3.6-27B on 1× MI300X via Hot Aisle

Self-hosted Claude Code backend serving Qwen3.6-27B on a single AMD MI300X
rented from [Hot Aisle](https://hotaisle.com). Validated 2026-05-04.

```
[laptop: claude]
   ↓ Anthropic Messages API (stream)
[laptop: LiteLLM container :4000]   ← Anthropic↔OpenAI translation, master-key auth
   ↓ OpenAI Chat Completions
[laptop: SSH tunnel :30000] ──────→ [Hot Aisle MI300X VM]
                                       └─ docker container "sglang"
                                          └─ SGLang main + aiter v0.1.12 + flydsl 0.1.2
                                             └─ Qwen3.6-27B BF16 + EAGLE/MTP draft
```

## Headline numbers (single GPU, 1× MI300X)

| Config | Single-stream tok/s | × baseline |
|---|---|---|
| Vanilla BF16 (no graphs, no spec) | 35 | 1.0× |
| + cuda graphs `bs [1,2,4,8]` | 50 | 1.4× |
| + EAGLE `num_steps=3, draft=4` | 121–148 | 3.5–4.2× |
| **+ EAGLE `5/6` + LPM + 32k prefill + tokenizer-batch + decode-2** | **~190** | **5.4×** |

At ~190 tok/s single-stream, ~280–310 tok/s aggregate at 2 concurrent. With
8 concurrent users sharing the box, expect ~35–45 tok/s per stream
(usable but not blazing). 16 concurrent saturates the engine compute-side
at ~17–22 tok/s per stream.

Full bench data: [`sglang-rocm-image/BENCHMARKS.md`](./sglang-rocm-image/BENCHMARKS.md).

## Resume from cold (≈20 min wall, ≈$2 GPU spend to live)

Prereqs on laptop: `hotaisle` CLI authed (`hotaisle config get token`),
`docker` (Docker Desktop on macOS works), `gh` (only if pushing to GHCR),
`jq`. SSH key registered with Hot Aisle.

```bash
# 1) Provision 1× MI300X (probes stock first; bails if none)
bash workshop/scripts/hotaisle-bring-up.sh
# Writes workshop/lib/env.local.sh with VM_SSH_HOST etc.

# 2) On the new VM: install docker + pull base SGLang image (~10 min)
. workshop/lib/env.local.sh
ssh -p 22 hotaisle@$VM_SSH_HOST 'bash ~/hotaisle/vm-prep.sh'
#   ↑ if scripts not on VM yet:
#     scp -P 22 -r workshop/{hotaisle,quality-battery} hotaisle@$VM_SSH_HOST:~/

# 3) Either pull the prebuilt SGLang image from GHCR ...
ssh -p 22 hotaisle@$VM_SSH_HOST \
    'docker pull ghcr.io/ediksimonian/qwen-sglang-rocm:main \
     && docker tag ghcr.io/ediksimonian/qwen-sglang-rocm:main qwen-sglang:rocm720-aiter112-sgmain'

# ... OR rebuild from source (~15 min, captures any latest SGLang main fixes)
scp -P 22 workshop/sglang-rocm-image/Dockerfile hotaisle@$VM_SSH_HOST:~/sglang-main-build/
ssh -p 22 hotaisle@$VM_SSH_HOST \
    'cd ~/sglang-main-build && docker build -t qwen-sglang:rocm720-aiter112-sgmain .'

# 4) Stage HF + SGLang secrets, launch Qwen on the VM
ssh -p 22 hotaisle@$VM_SSH_HOST "cat > /tmp/launch-env.sh <<EOF
export HF_TOKEN='$HF_TOKEN'
export SGLANG_API_KEY='$SGLANG_API_KEY'
export SGLANG_ADMIN_API_KEY='$SGLANG_ADMIN_API_KEY'
EOF
chmod 600 /tmp/launch-env.sh"
ssh -p 22 hotaisle@$VM_SSH_HOST \
    'set -a; . /tmp/launch-env.sh; set +a; bash ~/hotaisle/launch-qwen.sh'
# Boot is ~3 min (cuda graph capture is the long pole). Watch for the
# "server is fired up and ready to roll!" line.

# 5) Bring up the laptop side: SSH tunnel + LiteLLM (Docker) on :4000
bash workshop/scripts/bring-up.sh
# Prints the master key + ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN exports

# 6) Run claude
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=$LITELLM_MASTER_KEY
claude
```

## Layout

```
workshop/
├── README.md                       ← (this file)
├── sglang-rocm-image/
│   ├── Dockerfile                  ← derivative SGLang image (aiter+flydsl+main)
│   ├── README.md                   ← image build instructions, mandatory flags
│   └── BENCHMARKS.md               ← full FP8/BF16/Stage1-3 measurements
├── litellm/
│   └── litellm_config.yaml         ← Anthropic↔OpenAI proxy + master-key auth
├── hotaisle/                       ← run-on-VM scripts
│   ├── vm-prep.sh                  ← installs docker + pulls base image
│   ├── launch-qwen.sh              ← canonical SGLang launch (Stage 3 flags)
│   ├── a0-validate.sh              ← smoke (chat + tool + multilingual)
│   └── a0-5-qwen-load.sh           ← 16-concurrent sustained load
├── scripts/                        ← run-on-laptop orchestration
│   ├── hotaisle-bring-up.sh        ← provision a 1× MI300X VM
│   ├── hotaisle-tunnel.sh          ← foreground SSH local-forward
│   ├── hotaisle-tear-down.sh       ← delete the VM (ends billing)
│   ├── bring-up.sh                 ← tunnel + LiteLLM docker on :4000
│   └── watchdog.sh                 ← health/cost watch
├── lib/
│   └── env.sh                      ← shared bash helpers (litellm_start, etc.)
└── quality-battery/                ← load-test cases + Claude-Code traffic replay
```

## Key bugs we hit + how we worked around them

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `ImportError: cannot import name 'dynamic_per_tensor_quant'` from aiter at SGLang init | Base image bundles aiter v0.1.11.post1 missing the symbol | Build aiter v0.1.12.post1 from source in derivative image |
| 2 | `AttributeError: module 'aiter' has no attribute 'fmoe'` after fixing #1 | Bundled flydsl 0.0.1.dev disables CK/HIP runtime, so aiter.fmoe never loads | Pin `flydsl==0.1.2` in derivative image |
| 3 | `Qwen/Qwen3.6-27B-FP8` loads with 128 missing-`weight_scale_inv` warnings, output is character soup | SGLang v0.5.10.post1 fails to register fused `gate_up_proj` weight scales — [issue #23687](https://github.com/sgl-project/sglang/issues/23687), fixed in [PR #23062](https://github.com/sgl-project/sglang/pull/23062) (April 30, not in any released ROCm image as of 2026-05-04) | Replace SGLang v0.5.10.post1 with `main` in derivative image |
| 4 | `cyankiwi/Qwen3.6-27B-AWQ-INT4` fails: `NotImplementedError: No compressed-tensors compatible scheme was found` | SGLang's compressed-tensors scheme registry doesn't cover this layout | Use BF16 native instead; INT4 path remains untested for Qwen3.6 |
| 5 | LiteLLM /v1/messages 500s for Claude Code requests | Falls into LiteLLM's `experimental_pass_through/responses_adapters` which calls `/v1/responses` (OpenAI-only) | `litellm_settings.use_chat_completions_url_for_anthropic_messages: true` |
| 6 | LiteLLM 400s on Claude Code's `context_management` param | Anthropic-only field LiteLLM doesn't pass through to OpenAI | `litellm_settings.drop_params: true` |
| 7 | Claude Code feels frozen 5–30s, then answer arrives in burst | LiteLLM's anthropic streaming bridge emits `content_block_start: text` then `thinking_delta` deltas inside (mismatched types). [PR #25212](https://github.com/BerriAI/litellm/pull/25212) fixes it but is not merged | `extra_body.chat_template_kwargs.enable_thinking: false` per-model — Qwen skips reasoning preamble entirely |
| 8 | LiteLLM `/v1/models` 401 vs 200 inconsistency | Master key gates everything when set; readiness checks need it too | `bring-up.sh` mints a master key + threads it through container env |
| 9 | LiteLLM UI `/v2/login` returns 500 (`Not connected to DB!`) | UI auth path requires Postgres for user persistence | Postgres sidecar in `lib/env.sh:litellm_start()` |
| 10 | Claude Code 400 with `requested 81845 tokens > 65536 context` | Claude Code 2.x defaults `max_tokens=32k` for `claude-sonnet-4-6` alias | Bump SGLang `--context-length 131072` OR set `CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192` in client shell |
| 11 | FP8 weights load cleanly with main-branch SGLang BUT decode at half BF16 speed | Triton attention is BF16-bound (75% of layers are GDN linear-attn, AITER GDN broken upstream); aiter FP8 GEMM has no tuned configs for Qwen3.6 shapes | Workshop default stays BF16 |

## Tunable speed levers we *didn't* try

- **AITER GEMM pre-tune** (`gradlib.gemm_tuner`) — reverses the FP8 slowdown, +5-20% on GEMM-heavy paths. Untested; needs 20-60 min tuning pass.
- **Adaptive speculative decoding** (`--speculative-adaptive`) — +0-15%. Unverified on our SGLang main commit.
- **2× MI300X** (TP=2) — recovers most of the speed gap with the prior 2-GPU setup. ~$3.98/hr.

## Image: published on GHCR

```
ghcr.io/ediksimonian/qwen-sglang-rocm:main           ← latest (SGLang main + aiter+flydsl fixes)
ghcr.io/ediksimonian/qwen-sglang-rocm:sg-05aed5e1d   ← pinned to SGLang commit 05aed5e1d
```

Currently **private**. Either flip to public via:
```bash
gh api -X PATCH "/user/packages/container/qwen-sglang-rocm" -f visibility=public
```
or attach a Hot Aisle/RunPod registry-auth secret with a GHCR pull token.

## Cost tracking

Hot Aisle CLI exposes only point-in-time balance + rate (no history endpoint):
```bash
hotaisle team balance --handle edik-simonians-team \
  | jq '{balance_usd: (.available_balance/100), per_hour_usd: (.hourly_rate/100), runout: .estimated_runout_time}'
```
Web UI at hotaisle.app for invoiced history.

This validation session burned ~$7.37 across all VM time + image builds + benchmark passes.
