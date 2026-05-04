# qwen-sglang ROCm derivative image

SGLang for serving Qwen3.5 / Qwen3.6 family models on AMD MI300X.

Built on top of `lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x` to fix three
distinct bugs that block Qwen3.6-FP8 loading on the upstream image:

| Layer | Bundled | We override to | Why |
|---|---|---|---|
| flydsl | `0.0.1.dev95158637` | `0.1.2` | Bundled version disables aiter's CK/HIP runtime ("Unsupported flydsl version: expected 0.1.2"). Cascades to `AttributeError: module 'aiter' has no attribute 'fmoe'` at SGLang init. |
| aiter | `v0.1.11.post1` | `v0.1.12.post1` | SGLang's `quark/schemes/quark_w4a4_mxfp4.py` calls `aiter.dtypes.fp8`. v0.1.11 doesn't expose `dtypes` or `dynamic_per_tensor_quant`. Import fails before the engine starts. |
| sglang | `v0.5.10.post1` (Apr 9) | `main` | Brings PR [#23062](https://github.com/sgl-project/sglang/pull/23062) (`Qwen3_5GatedDeltaNet._make_packed_weight_loader` per-tensor scale broadcast). Without it, `Qwen/Qwen3.6-27B-FP8` loads with 128 missing-`weight_scale_inv` warnings and emits character soup. Tracked at SGLang issue [#23687](https://github.com/sgl-project/sglang/issues/23687). |

## Build

Build on a host that has the rocm720 base image cached (e.g. the Hot Aisle
VM after `workshop/hotaisle/vm-prep.sh`). The build is ~15 min — most of it
is the aiter source compile (`MAX_JOBS=8`) and the recursive
`composable_kernel` submodule clone.

```bash
docker build -t qwen-sglang:rocm720-aiter112-sgmain .
```

To pin to a specific SGLang commit instead of `main`:

```bash
docker build --build-arg SGLANG_REF=<sha-or-tag> \
    -t qwen-sglang:rocm720-aiter112-sg<sha> .
```

After build, the file `/sgl-workspace/SGLANG_VERSION.txt` inside the image
records which SGLang commit was installed.

## Run (Qwen3.6-27B BF16 + EAGLE — workshop default)

The handler-side launch script lives at `workshop/hotaisle/launch-qwen.sh`.
The config below is the validated workshop default — BF16 weights with
Stage-1 (cuda graphs) + Stage-2 (EAGLE/MTP draft) + Stage-3 (deeper draft,
chunked prefill, LPM scheduling) optimizations stacked. Single-stream
decode runs ~190 tok/s vs the 35 tok/s vanilla baseline.

```bash
docker run -d --name sglang \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --shm-size 32g --ipc host \
    -e SGLANG_USE_AITER=1 \
    -e SGLANG_ENABLE_SPEC_V2=1 \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e HF_TOKEN="$HF_TOKEN" \
    -p 30000:30000 \
    qwen-sglang:rocm720-aiter112-sgmain \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3.6-27B \
        --dtype bfloat16 \
        --tp 1 \
        --attention-backend triton \
        --linear-attn-backend triton \
        --context-length 131072 \
        --max-running-requests 8 \
        --max-total-tokens 1100000 \
        --mem-fraction-static 0.92 \
        --cuda-graph-max-bs 8 \
        --cuda-graph-bs 1 2 4 8 \
        --triton-attention-num-kv-splits 16 \
        --chunked-prefill-size 32768 \
        --max-prefill-tokens 32768 \
        --num-continuous-decode-steps 2 \
        --schedule-policy lpm \
        --enable-tokenizer-batch-encode \
        --speculative-algorithm EAGLE \
        --speculative-num-steps 5 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 6 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --api-key "$SGLANG_API_KEY" \
        --admin-api-key "$SGLANG_ADMIN_API_KEY" \
        --enable-metrics --enable-cache-report \
        --disable-piecewise-cuda-graph
```

For an FP8 test (Qwen3.6-27B-FP8 — loads correctly with this image but
runs ~half BF16's decode speed; see BENCHMARKS.md for why), swap
`--model-path` and add `--fp8-gemm-backend aiter`. Drop `--dtype` (the
checkpoint determines dtype). Don't try this for a workshop today — it's
just there as proof the loader fix works.

## Mandatory flags for Qwen3.5 / Qwen3.6 GDN models on AMD

These are **not** optional on this stack — set incorrectly the engine boots
but produces garbled output:

- `--attention-backend triton` and `--linear-attn-backend triton` —
  AITER's GDN attention is broken for Qwen3.5/3.6. Triton kernels are
  slower but correct.
- `--disable-piecewise-cuda-graph` — piecewise capture has known crashes
  on hybrid-attention layouts. Keep regular `--cuda-graph` enabled though.
- `SGLANG_USE_AITER=1` — keep aiter on for non-attention paths
  (RoPE, GEMM, MoE). The per-backend flags above override the
  attention-only path.

## Published image

`ghcr.io/ediksimonian/qwen-sglang-rocm:main` — currently pinned to SGLang
main commit `05aed5e1d` (2026-04-30, includes PR #23062).

Also tagged: `ghcr.io/ediksimonian/qwen-sglang-rocm:sg-05aed5e1d` for an
exact-commit pull.

To bump: rebuild on a host with the rocm720 base image cached, retag with
the new short SHA, push.

## Performance

See [BENCHMARKS.md](./BENCHMARKS.md) for measured numbers across all
optimization stages, FP8-vs-BF16 results, EAGLE acceptance metrics, and
findings on what didn't work (INT4 AWQ, latest-rocm tag).

**TL;DR:** Workshop default is BF16 + EAGLE + Stage-3 tuning →
**~190 tok/s single-stream** (5.4× over the 35 tok/s vanilla baseline),
~280-310 tok/s aggregate at 2 concurrent streams. FP8 loads correctly
with this image but runs at ~half BF16's decode speed because attention
stays Triton/BF16-bound (AITER GDN broken upstream) and aiter's FP8 GEMM
lacks tuned configs for Qwen3.6 shapes — kept here only for the loader
fix proof.

## Pushing manually (no CI)

```bash
echo "$(gh auth token)" | docker login ghcr.io -u EdikSimonian --password-stdin
docker tag qwen-sglang:rocm720-aiter112-sgmain \
    ghcr.io/ediksimonian/qwen-sglang-rocm:main
docker push ghcr.io/ediksimonian/qwen-sglang-rocm:main
```

No GitHub Actions workflow is set up for this image — the build is too
specific (ROCm + AMD GPU + aiter source compile) and the upstream
`lmsysorg/sglang` ROCm cadence is slow enough that we'd rebuild
ad-hoc anyway.
