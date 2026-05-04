# Benchmarks — Qwen3.6-27B on 1× MI300X

All measurements taken 2026-05-04 on a Hot Aisle MI300X VM (ROCm 7.2.0,
driver 6.16.13) using `qwen-sglang:rocm720-aiter112-sgmain` (SGLang main at
commit `05aed5e1d`, aiter v0.1.12.post1, flydsl 0.1.2). Identical launch
flags except where called out.

Common flags across all rows:

```
--tp 1 \
--attention-backend triton --linear-attn-backend triton \
--context-length 131072 --max-running-requests 8 \
--max-total-tokens 1100000 --mem-fraction-static 0.92 \
--triton-attention-num-kv-splits 16 \
--disable-piecewise-cuda-graph
```

## Stage stack (BF16, single-stream)

| Stage | Added flags / changes | Single-stream tok/s | × baseline |
|---|---|---|---|
| 0 — vanilla | (no cuda graph, no spec) | **35** | 1.0× |
| 1 — cuda graphs | `--cuda-graph-bs 1 2 4 8 --cuda-graph-max-bs 8` | **50** | 1.4× |
| 2 — + EAGLE | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` + `SGLANG_ENABLE_SPEC_V2=1` | **121–148** | 3.5–4.2× |
| **3 — workshop default** | Stage 2 + `--speculative-num-steps 5 --speculative-num-draft-tokens 6 --chunked-prefill-size 32768 --max-prefill-tokens 32768 --num-continuous-decode-steps 2 --schedule-policy lpm --enable-tokenizer-batch-encode` | **~190** | **5.4×** |

Each stage stacks on the previous — Stage 3 includes everything from
Stage 2 which includes Stage 1.

### Stage 2 / Stage 3 EAGLE acceptance

| | Stage 2 (3-step, 4-draft) | Stage 3 (5-step, 6-draft) |
|---|---|---|
| `accept_len` (mean) | ~3.0 | **~4.5** |
| `accept_len` (range) | 2.55 – 3.65 | 3.66 – 5.12 |
| `accept_rate` (mean) | ~0.7 | ~0.7 |
| Single-stream peak | 148 tok/s | **~200 tok/s** |
| 2-conc aggregate | 240–256 tok/s | **260–310 tok/s** |
| Wall time (300-tok prompt, warm) | ~3.5 s | **~1.7 s** |
| TTFB (warm) | ~0.30 s | **~0.13 s** |

The deeper draft is the key Stage-3 win — `accept_len` rose 50%, meaning
each verify pass commits 4.5 tokens on average instead of 3.

## BF16 vs FP8 (with Stage 1 only — pre-EAGLE)

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
for the same model. The workshop config stays on BF16. FP8 is kept in the
image only as proof of the loader fix.

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

## Concurrent capacity estimate

Based on the Stage 3 measurements:

| Concurrent users | Per-stream tok/s (estimated) | Aggregate |
|---|---|---|
| 1 | ~190 (measured) | ~190 |
| 2 | ~140 (measured) | 280–310 (measured) |
| **8** | **~35–45** | **~280–360** |
| 16 | ~17–22 | ~280–350 (compute-saturated) |

At ≥8 concurrent the engine is compute-bound and aggregate throughput
plateaus. EAGLE's draft compute competes with main-model forward passes,
so the per-stream gain shrinks. LPM scheduling helps offset this when
sessions share prefixes (Claude Code's repeated system prompts).

## Open levers (not yet tried)

| Lever | Expected gain | Risk | Why not yet |
|---|---|---|---|
| AITER GEMM pre-tune (`gradlib.gemm_tuner`) | +5-20% on GEMM-heavy paths | Long tuning pass (20-60 min); needs tuned-config CSV path wired into image | Workshop priority shifted to validation |
| Wider `--cuda-graph-bs` (e.g. include 12, 16) | TTFT smoothing at concurrent batch sizes | Larger graph memory, slower boot capture | Current narrow set is the win — wider is just margin |
| AMD quark-published FP8 quants | unknown — would need shape-tuned aiter configs | Low correctness risk if quark provides scales SGLang accepts; high speed risk | No quark Qwen3.6-27B FP8 published as of search date |
| Adaptive speculative (`--speculative-adaptive`) | +0-15% when accept_len varies by task | Unverified main-build availability | Not tested |
| 2× MI300X (`--tp 2`) | ~2× per-stream | Cost: $3.98/hr vs $1.99/hr; not single-GPU anymore | User explicitly staying on 1× |
