# DSv4 roofline-driven optimization queue

## Context

Goal: make DSv4-Flash default-worthy on 8xH20 by optimizing the current
correct route systematically instead of extrapolating from mismatched
SGLang numbers.

Current default route, commit `90f940bd`:

```text
FlashMLA attention + local routed experts + EP all-reduce + DeepGEMM-auto
```

The native DeepEP transport is not the default because the current TP/EP
serving path replicates token rows on every rank. DeepEP dispatch/combine needs
token-owned EP rows and over-transports this layout.

## Evidence

Remote artifacts:

- no-trace first request:
  `/sgl-workspace/bench-artifacts/dsv4-roofline-90f940bd-notrace-p2048-o8/default-p2048`
- no-trace same-server warm pair:
  `/sgl-workspace/bench-artifacts/dsv4-roofline-90f940bd-warm2-p2048-o8/default-p2048-warm2`
- trace same-server warm pair:
  `/sgl-workspace/bench-artifacts/dsv4-roofline-90f940bd-trace-warm2-p2048-o8/default-p2048-trace-warm2`

Same-server p2048/o8 no-trace:

| Request | TTFT | Total | TPOT after first token | Notes |
|---|---:|---:|---:|---|
| cold request 1 | 5183 ms | 5973 ms | 112.8 ms | first request pays warmup/lazy overhead |
| warm request 2 | 4380 ms | 5165 ms | 112.2 ms | steady TPOT unchanged |

Interpretation:

- The earlier cold-request TTFT is polluted by first-request warmup.
- The decode TPOT is not materially warmup-polluted.
- `113 ms/token` is a same-runtime p2048/o8 steady decode signal, but it must
  not be compared directly to a different SGLang workload as a final claim.

Trace overhead:

| Shape | no-trace TPOT | trace TPOT | Overhead |
|---|---:|---:|---:|
| p2048/o8 | 112.2 ms | 140.7 ms | about 1.25x |

Use trace for phase attribution, not absolute product numbers.

## Warm Decode Phase Breakdown

p2048/o8 warm request 2, trace-on, token_count=1. Values below are top-level
per-token phase sums after normalizing the mixed 8-rank log by rank and decode
step. The rough no-trace estimate divides by the observed 1.25 trace overhead.

| Stage | Trace ms/token | Estimated no-trace ms/token | Verdict |
|---|---:|---:|---|
| attention total | 92.1 | about 73 | P0 bottleneck |
| `attn_csa_select_kernel` | 49.7 | about 40 | single largest unreasonable stage |
| attention hybrid kernel/math | 23.0 | about 18 | still high |
| FFN total | 39.0 | about 31 | over target budget |
| routed local experts | 13.5 | about 11 | secondary |
| FFN all-reduce | 12.8 | about 10 | secondary, overlap candidate |
| shared expert | 5.6 | about 4.5 | not the primary decode blocker |

Decode target framing:

- To clear SGLang-on-H20-adjacent `60-64 ms/token`, the warm p2048 TPOT needs
  about 43-46% reduction.
- To clear the repo SLO `<=30 ms/token`, it needs about 73% reduction.
- CSA select alone is already above the full SLO budget, so it is the first
  license-or-kill axis.

## Warm Prefill Phase Breakdown

p2048/o8 warm request 2, trace-on, token_count=2047. The no-trace TTFT is
4380 ms; the trace top-level phase total is higher due to synchronization.

| Stage | Trace ms | Estimated no-trace ms | Verdict |
|---|---:|---:|---|
| attention total | 3256 | about 2620 | P0 prefill bottleneck |
| FFN total | 1974 | about 1590 | P0 prefill bottleneck |
| shared expert | 1552 | about 1250 | block-scaled batch GEMV gap |
| `attn_csa_select_kernel` | 722 | about 580 | algorithm/kernel gap |
| attention projection | 679 | about 545 | block-scaled batch GEMV gap |
| attention output projection | 496 | about 400 | block-scaled batch GEMV gap |
| FFN all-reduce | 208 | about 167 | not the main prefill limiter |

The shared expert and attention projections are still using the generic
DSv4 block-scaled batch GEMV path. Their effective throughput is far below a
reasonable Hopper tensor-core roofline, so the optimization target is not
"more DeepEP"; it is moving block-scaled matrix paths onto tensor-core GEMM /
DeepGEMM-class kernels.

## Optimization Queue

### P0.1 Decode CSA select

Problem: `attn_csa_select_kernel` costs about 40 ms/token no-trace estimate at
p2048 decode, before attention math. That single stage makes `<=30 ms/token`
impossible.

Entry points:

- `infer/src/model/deepseek/weights.rs` around `csa_selected_blocks_gpu`
- `crates/cuda-kernels/csrc/misc/dsv4_attention.cu`
- `crates/cuda-kernels/csrc/misc/arle_flashmla_csa_prep.cu`

Hypotheses to license-or-kill:

1. Decode recomputes selected compressed blocks from scratch each token and
   should maintain an incremental cache for the selected block list where the
   semantics allow it.
2. The current kernel does too much per-layer/top-k work for small decode
   batches and needs fusion with FlashMLA index build or a persistent/top-k
   structure.
3. CSA layers and HCA layers need separate targets; do not average them into
   one "attention" bucket.

Gate: same-server warm p2048/o8 and p2048/o32, output/token sanity, and trace
phase attribution. The first accepted win must reduce warm TPOT and specifically
reduce `attn_csa_select_kernel` or prove that attribution wrong.

### P0.2 Block-scaled batch GEMV to tensor-core GEMM

Problem: p2048 prefill spends about 1.25 s no-trace estimate in shared expert
and about 0.95 s in attention in/out projections. These paths are matrix-like
but currently run through DSv4 block-scaled batch GEMV.

Entry points:

- `infer/src/ops/linear.rs`
- `infer/src/model/deepseek/mlp.rs` shared expert path
- DeepGEMM/FP8 cache code in `infer/src/model/deepseek/mlp.rs`

Hypotheses to license-or-kill:

1. Shared expert should use a DeepGEMM/tensor-core path for prefill batches.
2. Attention projection block-scaled weights need an equivalent tensor-core
   path or an explicit per-shape reason why they cannot use one.
3. DeepGEMM routed-expert improvements are not enough if shared/projection
   paths remain on low-utilization GEMV.

Gate: p2048/o8 warm TTFT, p4096/o1 TTFT, and phase reduction in `ffn_shared`,
`attn_proj`, or `attn_output_proj`.

### P1 All-reduce overlap

Problem: decode still pays about 10 ms/token estimated in FFN all-reduce plus a
smaller attention all-reduce component. The path is mostly serial.

Entry points:

- `infer/src/model/deepseek/weights.rs` `post_*_all_reduce_hidden_states`
- `infer/src/model/layer_communicator.rs`

Gate: do not optimize collectives before P0.1 unless trace shows CSA select is
already fixed or the collective work can be overlapped without changing
correctness. A win must show wall-clock TPOT improvement, not only a narrower
NVTX window.

### P1 Warmup hygiene

Problem: cold TTFT adds about 800 ms at p2048/o8 in the same-server pair.

Gate: keep this separate from steady-state decode. A warmup fix is useful for
first-request UX, but it does not license a throughput claim.

## Protocol

- Use same-server warm pairs for TTFT/TPOT claims.
- Use trace-on only for attribution; cross-check with no-trace wall-clock.
- One axis per commit and per wins/errors entry.
- Preserve correctness before perf: usage/token counts, non-empty output, and
  where possible greedy output parity against the current default.
- Do not compare against SGLang unless workload, model, quantization,
  concurrency, and measurement method match.

