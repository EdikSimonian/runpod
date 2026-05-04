# Benchmarks — Qwen3.6-27B on 1× MI300X

All measurements taken 2026-05-04 on a Hot Aisle MI300X VM (ROCm 7.2.0,
driver 6.16.13) using `qwen-sglang:rocm720-aiter112-sgmain` (SGLang main at
commit `05aed5e1d`, aiter v0.1.12.post1, flydsl 0.1.2). Identical launch
flags except for `--model-path` and `--dtype`.

Common flags:

```
--tp 1 \
--attention-backend triton --linear-attn-backend triton \
--context-length 131072 --max-running-requests 8 \
--max-total-tokens 1100000 --mem-fraction-static 0.92 \
--cuda-graph-max-bs 8 --cuda-graph-bs 1 2 4 8 \
--triton-attention-num-kv-splits 16 \
--disable-piecewise-cuda-graph
```

## Headline numbers

| | BF16 (`Qwen/Qwen3.6-27B`) | FP8 (`Qwen/Qwen3.6-27B-FP8`) |
|---|---|---|
| **Single-stream decode** | **50 tok/s** | **25.6 tok/s** |
| 256-token wall time (warm) | ~5.6 s | ~11.8 s (3-run consistent) |
| Weight memory | 51 GB | 28.5 GB |
| `max_total_num_tokens` (KV pool) | 1,077,735 | 1,100,000 |
| Mamba SSM cache slots | 388 | 487 |
| Available GPU mem post-init | 14.65 GB | 25.01 GB |
| Boot time (incl. cuda graph capture) | ~3 min | ~5 min |
| `weight_scale_inv` warnings | n/a | **0** (PR #23062 fix worked) |
| Output quality (pong probe) | clean | clean |

**Conclusion:** on this stack, FP8 is roughly half the decode speed of BF16
for the same model. We stay on BF16 until upstream tuning lands.

## Why FP8 is slower here (root cause)

1. **Triton attention is BF16-bound.** Qwen3.6 is hybrid: 16 full-attention
   layers + 48 Gated DeltaNet (linear-attention) layers. AITER's GDN
   attention is broken upstream for Qwen3.5/3.6, so we're stuck on
   `--attention-backend triton`. Triton attention reads/writes BF16
   regardless of weight precision. The expected FP8 win on attention
   simply doesn't apply.
2. **AITER FP8 GEMM has no tuned config for Qwen3.6 shapes.**
   At boot the engine emits `not found tuned config in
   /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config!` for
   our matmul shapes. The FP8 GEMM path is running unoptimized fallback
   kernels.
3. **FP8 → BF16 dequant overhead per layer.** For the 48 GDN layers
   (75% of all layers), weights are dequantized back to BF16 at compute
   time. Pure overhead vs reading BF16 directly.
4. **Per-token activation quantization kernels.** FP8 inference compiles
   `module_rmsnorm_quant` and `module_quant` aiter kernels. These add
   per-token dispatch cost that BF16 doesn't pay.

## Cuda graph impact (tested before FP8 attempt)

Same BF16 model, with vs without cuda graphs:

| | `--disable-cuda-graph` | cuda graphs `--cuda-graph-bs 1 2 4 8` |
|---|---|---|
| Single-stream decode | 35 tok/s | 50 tok/s |

+43%, matching the upper end of Codex's 10-35% prediction. Capture cost is
~115 s at boot (one-time) and 0.24 GB graph memory.

## 16-concurrent sustained load (BF16, before cuda-graph re-enable)

Validated 2026-05-04 with `bench_serving --num-prompts 32 --random-input-len 60000 --random-output-len 4000 --max-concurrency 16` plus `claude_code_replay` (16 sessions × 5 turns) plus quality battery:

| Metric | Value |
|---|---|
| Successful requests | 32 / 32 |
| Replay sessions | 16 / 16, 0 errors |
| TTFT median / P95 / P99 | 3.15 s / 1.83 s / 2.66 s |
| ITL median | 61 ms |
| Per-stream tok/s under 16-conc | ~16 |
| Aggregate decode tok/s | ~250 |
| Control-plane probe failures | 0 / 116 |

Engine never wedged under sustained 16-concurrent load.

## What actually solved the FP8 loader bug

`Qwen/Qwen3.6-27B-FP8` on stock `lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x`
fails with **128 missing-`weight_scale_inv` warnings** (every MLP gate/up
projection) and emits character-soup output. Tracked at
[SGLang #23687](https://github.com/sgl-project/sglang/issues/23687).

The fix is [SGLang PR #23062](https://github.com/sgl-project/sglang/pull/23062)
(merged 2026-04-30, **not in any released ROCm image** — last published
ROCm tag is from 2026-04-09). This image rebuilds SGLang from main on top
of the v0.5.10.post1 base to pick up that PR.

After the rebuild: 0 warnings, model loads cleanly, output is coherent.
Speed is the unrelated issue documented above.

## What didn't work, briefly

- `cyankiwi/Qwen3.6-27B-AWQ-INT4` — SGLang's `compressed-tensors`
  scheme registry doesn't cover this checkpoint's `pack-quantized` layout
  on this ROCm build:
  `NotImplementedError: No compressed-tensors compatible scheme was found.`
  (`compressed_tensors.py:629`).
- `lmsysorg/sglang:latest-rocm` — does not exist as a tag. The ROCm
  release cadence is much slower than CUDA (last ROCm tag: 2026-04-09;
  CUDA gets nightly builds).
- `--enable-torch-compile` — Codex flagged as out-of-maintenance per
  SGLang docs; not tested.

## Open levers (not yet tried)

| Lever | Expected gain | Risk | Why not yet |
|---|---|---|---|
| EAGLE speculative decoding (Stage 2) | ×1.3-2 on top of BF16 baseline | Startup failure, extra memory, unclear acceptance rate on Qwen3.6 | Blocked on validating Stage 1 in real Claude Code use first |
| AMD quark-published FP8 quants | unknown — would need shape-tuned aiter configs | Low correctness risk if quark provides scales SGLang accepts; high speed risk | No quark Qwen3.6-27B FP8 published as of search date |
| Backport GEMM tuning configs from AMD | could reverse the FP8 slowdown | Build complexity | Out of scope for workshop |
