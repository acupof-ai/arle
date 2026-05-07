# M6 CUDA vLLM Gap Follow-ups

## Context

The first M6 CUDA P0 snapshot on `48d31ace09f8` completed all four ARLE-vs-vLLM
workloads on the local RTX 4070 Ti SUPER, but ARLE won only 1 of 8 score cells.
This plan tracks the losing cells without blocking the benchmark snapshot.

Primary snapshot:
[2026-05-07-m6-world-rank-snapshot-cuda.md](../experience/wins/2026-05-07-m6-world-rank-snapshot-cuda.md)

## Targets

| workload | gap | target |
|---|---:|---|
| prefill-heavy out tok/s | ARLE -5.4% | close >= 6% without regressing TTFT |
| decode-heavy out tok/s | ARLE -4.9% | close >= 6% and stabilize TTFT p99 |
| longctx-32k out tok/s | ARLE -34.9% | remove 16 GiB pool bottleneck or mark hardware-limited with remote H100/Ada result |
| high-conc out tok/s | ARLE -62.9% | raise effective active concurrency beyond 14 or prove memory-bound limit |

## Work Items

1. Prefill-heavy:
   - Compare ARLE TileLang paged prefill launch shape against vLLM Triton
     attention at 4096-in / 16-out.
   - Confirm whether chunking at 2048 is optimal for single request prefill on
     sm_89.

2. Decode-heavy:
   - Measure per-token decode phase with CUDA graph on and deterministic BF16
     GEMM enabled.
   - Separate model math time from sampling / host response streaming overhead.

3. Longctx-32k:
   - Re-run on a larger GPU before drawing architecture conclusions.
   - On the local card, test whether KV tier promotion/demotion can keep c=4
     progressing without the `active=2 waiting=2` residency ceiling.

4. High-conc:
   - Compare ARLE `max_slots=14` against vLLM `max_num_seqs=64` and an
     equal-slot vLLM run.
   - Profile scheduler admission and decode batch width under c=64; the first
     goal is to increase output tok/s while preserving ARLE's TTFT p50 lead.

## Acceptance

- Publish a follow-up wins or errors entry with at least one closed gap.
- Do not count hardware-limited longctx as a runtime regression until a larger
  CUDA runner repeats the workload.
- Keep the M6 raw command shapes unchanged unless the change is explicitly
  documented as a new benchmark variant.
