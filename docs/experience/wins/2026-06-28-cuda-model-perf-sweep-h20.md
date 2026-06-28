# CUDA model performance sweep — 8×H20, fresh `deepep` build (HEAD 6e30dd36)

## Context

Re-test of CUDA serving performance across every model staged on the H20 pod,
on a clean rebuild of the latest tree (`6e30dd36`, `--features cuda,nccl,deepep`,
native DeepGEMM). DSv4-Flash-FP8 run at **TP4/EP4** per request. Goal: a fresh
per-model CUDA perf picture + the capability boundary of the rewrite stack.

**Harness — stdlib two-point, NOT guidellm (deviation, documented).** The pod's
PyPI routes were both down at run time (direct = read-timeout on large wheels;
SOCKS proxy 127.0.0.1:1080 = connection refused), and cross-platform wheel
resolution failed on `pyarrow>=21`. guidellm could not be installed. Substitute:
a stdlib-only closed-loop generator (`llmbench.py`) — **arle serve has streaming
deferred** (`stream=true` → 400 "deferred in R5 tranche 2"), so it uses
NON-streaming requests + the two-point method to decompose prefill vs decode:
- `max_tokens=1` latency  ≈ **TTFT** (prefill + 1 token)
- `max_tokens=128` latency ≈ prefill + 128-token decode
- ⇒ **ITL** (ms/tok) = (lat₁₂₈ − lat₁) / 127; decode tok/s = 128 / lat₁₂₈
All models run the **same** harness, so cross-model comparison is apples-to-apples
(comparability to old guidellm wins baselines is intentionally not claimed).
Input length is reported as the server-measured `prompt_tokens` (the synthetic
`wN` filler tokenizes to ~3–4 BPE tokens each, so actual ≫ the nominal target).

## Capability matrix — what the CUDA backend actually serves (HEAD 6e30dd36)

| Model | Weights | CUDA path | Serves? | Concurrency (c>1) |
|---|---|---|---|---|
| **Qwen3-4B** | BF16 dense | `R6-clean` (TileLang HD128/kv8, BF16-only) | ✅ | ❌ **crashes** — `submit()` hard-asserts `rows==1`; the scheduler batching ≥2 rows kills the engine thread ("R6 clean CUDA forward is single-row only") |
| **Qwen3.6-27B-FP8** | FP8 | `qwen35` (FP8 + DeepGEMM) | ✅ | ⚠️ **serializes** — accepts c>1 without crashing but throughput stays flat; requests queue |
| **DeepSeek-V4-Flash-FP8** | FP8 | `dsv4` (MLA + DSA + EP, TP4/EP4) | ✅ | ✅ **batches** — aggregate throughput scales with concurrency |
| **Qwen3-30B-A3B** | BF16 MoE | — | ❌ | n/a — `Qwen3MoeForCausalLM is not supported on the CUDA backend; use --backend metal` |
| **Qwen3.5-122B-A10B** | BF16 MoE | — | ❌ | n/a — `num_kv_heads (2) not divisible by world_size (4)`; 234 GB won't fit on TP≤2 (>97 GB/GPU) ⇒ no viable TP config |

The three serveable models exercise three *different* CUDA executors with three
*different* batching behaviors — the single most important structural finding.

## Results (measured, 1×H20 unless noted; c=1)

| Model (GPUs) | in_tok | decode tok/s | ITL ms/tok | TTFT ms |
|---|---|---|---|---|
| Qwen3-4B (1) | 423 | **99.8** | 9.7 | **53** |
| Qwen3-4B (1) | 4031 | 50.3 | 19.1 | 117 |
| Qwen3.6-27B-FP8 (1) | 425 | **21.0** | ~47 | ~3230 *(JIT-cold)* / 1821 warm@4033in |
| Qwen3.6-27B-FP8 (1) | 4033 | 23.7 | 28.2 | 1821 |
| DSv4-Flash-FP8 TP4/EP4 (4) | 273 | **34.1** † | 28.9 | **79** |
| DSv4-Flash-FP8 TP4/EP4 (4) | 2089 | 33.0 | — | ~5972 ‡ |

† DSv4 c=1 decode is **run-to-run variable** (34.1 tok/s fresh-matrix vs 25.1
tok/s warm-scale run, same shape) — its known serial-bound / MoE-nondeterminism
characteristic; treat as **~25–34 tok/s**.
‡ DSv4 long-context TTFT is noisy (n=3 at ~4 s/req) and heavy — the DSA prefill
path dominates; the absolute value is not reliable, only the "prefill is
expensive at length" trend is.

### Concurrency scaling (decode, in≈128 / out=128)

| c | DSv4 TP4 tok/s | DSv4 lat p50 / p99 ms | Qwen3.6 tok/s | Qwen3.6 lat p50 ms |
|---|---|---|---|---|
| 1 | 25.1 | 5152 / 5753 | 21.0 | 6086 |
| 2 | 31.9 | 8001 / 8441 | 21.0 | 12167 |
| 4 | **43.2** | 10340 / **20879** | 21.0 | 30407 |
| 8 | — | — | 21.1 | 48621 / 66813 |

## Analysis

- **Qwen3-4B (BF16 dense) is the fastest single-stream path** — 99.8 tok/s decode
  at short context, 53 ms TTFT, ~56K tok/s prefill (TTFT 53→117 ms as input
  423→4031). Decode halves to 50 tok/s at 4K context (KV-attention growth: ITL
  9.7→19.1 ms/tok). But it is **c=1-only** — concurrency crashes the engine, so it
  is a single-user latency path, not a serving-throughput path.
- **DSv4-Flash-FP8 TP4/EP4 is the only real batched server.** c=1 decode ~25–34
  tok/s (4 GPUs, FP8 MoE, MLA+DSA) — lower per-stream than the 4B dense model, as
  expected for a much larger MoE with sparse-attention overhead. Its value is
  **aggregate scaling**: 25→32→43 tok/s as c=1→2→4. The cost is tail latency
  (p99 20.9 s at c=4) — throughput-over-latency. TP4 sits below the historical
  TP8 c=1 (~53 tok/s) reference, consistent with halved aggregate bandwidth.
- **Qwen3.6-27B-FP8 serves but does not batch.** Flat 21 tok/s from c=1 to c=8
  while latency grows linearly (6→49 s) ⇒ concurrent requests serialize behind a
  single in-flight decode. Single-stream decode ~21–24 tok/s; the FP8 path also
  pays a DeepGEMM-JIT cold-start tax on the first requests (TTFT 3.2 s cold vs
  1.8 s warm) — **warm up before measuring**.
- **MoE coverage is the CUDA gap.** Both vanilla MoE checkpoints (Qwen3-30B-A3B,
  Qwen3.5-122B-A10B) fail to serve — one by arch (`Qwen3MoeForCausalLM` → Metal
  only), one by sharding geometry (2 KV heads, no fitting TP). The CUDA MoE story
  on this build is **DSv4-only**.

## Key findings

1. **Three executors, three batching behaviors**: R6-clean dense *crashes* on
   c>1, qwen35-FP8 *serializes*, dsv4 *batches*. Batched serving is wired for
   DSv4 only; Qwen-on-CUDA is single-stream today.
2. **Qwen3-4B is the single-user latency champ** (100 tok/s, 53 ms TTFT); **DSv4
   TP4/EP4 is the only throughput-scaling server** (43 tok/s @c=4).
3. **CUDA serves 3 of 5 staged models**; both vanilla-MoE checkpoints are
   unsupported (Metal / no-TP-fit).

## Caveats / SOLID gaps (not yet closed)

- **Harness ≠ guidellm** (pod network). Numbers are internally consistent
  (same harness, two-point method) but not cross-comparable to guidellm wins
  baselines. Re-run under guidellm when the pod proxy is back to confirm.
- **DSv4 c=1 variance unresolved** (25 vs 34 tok/s) — small samples (~4 req/15 s)
  + DSv4's serial-bound/MoE-nondeterminism. Longer-duration repeats needed to
  pin the mean ± σ.
- **Qwen3.6 TTFT is JIT-confounded** in the cold matrix; only warm values trusted.
- **No correctness gate beyond a 1-shot coherent completion** per model (Paris /
  France answered correctly). Not a needle-ladder parity run.

## Rule

The rewrite CUDA backend is **single-stream for Qwen** (R6-clean dense crashes on
batch; qwen35-FP8 serializes) and **batched only for DSv4**. Before quoting a
"CUDA serving throughput", name the executor and the concurrency behavior — a c=1
number for Qwen is a latency number, not a throughput-under-load number. FP8
paths need a warmup pass (DeepGEMM JIT) before any TTFT is measured.
