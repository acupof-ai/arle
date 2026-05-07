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

## **Priority 0: re-run with T1 host-pinned KV overflow enabled**

The M6 baseline run did NOT explicitly enable the T1 host-pinned tier.
`SchedulerConfig::t1_host_pinned_capacity_bytes` defaulted to `None`
(the constructor's conservative auto-size, see `scheduler/types.rs:177`).
The local box has **31.3 GB system RAM** vs the **16 GB GPU**; the
infrastructure to spill GPU KV to host-pinned RAM **already exists**
(`infer/src/kv_tier/host_pool.rs` + `KvTierAdapter`, M2 of unification)
and the `infer` binary already exposes the CLI flags
(`infer/src/main.rs:203-219`):

- `--t1-host-pinned-capacity-mb <N>` (target ≈ 16384 — leaves ~10 GB for OS / build / driver)
- `--t1-host-pinned-high-water <FRAC>` (default 0.85)
- `--t1-host-pinned-low-water <FRAC>` (default 0.70)

Re-run the M6 canonical sweep with explicit T1 sizing **before** going
deep on combo plan implementation. Hypothesis: longctx-32k -34.9% and
high-conc -62.9% gaps shrink materially because kv_util-100% scenarios
spill to host instead of stalling at `active=2 waiting=2` (longctx) or
`active=14 waiting=∞` (high-conc).

```bash
RUST_LOG=info NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run --release -p infer --no-default-features --features cuda -- \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --port 8000 \
  --max-seq-len 5120 \
  --t1-host-pinned-capacity-mb 16384

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh cuda-m6-with-t1 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B
```

Expected delta:
- longctx-32k: gap closes (host-pinned absorbs the prefix overflow that currently stalls at 2 active).
- high-conc: gap shrinks (host-pinned increases the effective active concurrency ceiling without changing `max_slots`).
- prefill-heavy / decode-heavy: unchanged (no overflow happening in baseline).

If these expected deltas land, the headline "ARLE 1/8 → ARLE ≥3/8 on
local 16 GB" without writing a single line of code. That's the
*configuration-only* win to publish before chasing kernel-level optimizations.

**Acceptance**: re-run produces a `cuda-m6-with-t1` wins entry with
direct delta-vs-baseline rows. Closes longctx-32k or high-conc cell
even on local hardware → counts as a gap closed for this plan's
acceptance criterion.

## Strategic alignment with combo plan (manager note 2026-05-07)

Three of the four gaps overlap directly with already-spec'd combo plan
sub-plans. Do NOT attack them in isolation — they are downstream of work
already in flight:

| M6 gap | Combo plan that addresses it | Relationship |
|---|---|---|
| decode-heavy out tok/s -4.9% | [`M_b.2`](longctx-spec-tilelang-combo.md) sparse-self-spec fusion | spec-decode is the canonical decode-throughput multiplier; M_b.2 targets +10-15% on top of vanilla decode without retraining. Likely closes this gap on its own. |
| longctx-32k out tok/s -34.9% | [`M_d`](M_d-tier-kv-spec-decode-coordination.md) Tier-KV × spec coordination | Long-context throughput is gated by KV residency, exactly what M_d's eager-prefetch + scratch-page commit barrier address. Plus the 16 GB pool ceiling means H100/L4 retest is the right path before architecture changes. |
| high-conc out tok/s -62.9% | [`M3.5`](M3.5-collapse-scheduler-loops.md) shared CPU policy + [`M_b`](M_b-tilelang-fused-draft-verify-kernel.md) batched verify | High-conc throughput is gated by scheduler decisions per tick + per-row verify cost. M3.5 unifies decisions; M_b/M_c add multi-token-per-step credit on the verify path. |
| prefill-heavy out tok/s -5.4% | (no combo plan; isolated tile-shape tuning) | The only gap that's a pure single-shape prefill tuning question. Worth its own focused micro-optimization. |

So the priority order for gap-closing is **M_b.2 first, then M3.5, then M_d**;
prefill-heavy is the only follow-up that needs to be attacked in this
gap-followup plan as a standalone item. Don't burn a sprint reproducing
spec-decode work that the combo plan already specs.

## Acceptance

- Publish a follow-up wins or errors entry with at least one closed gap.
- Do not count hardware-limited longctx as a runtime regression until a larger
  CUDA runner repeats the workload.
- Keep the M6 raw command shapes unchanged unless the change is explicitly
  documented as a new benchmark variant.
- For the three gaps that map to combo plan sub-plans, gap closure happens
  when the combo plan sub-plan lands — this plan tracks the bench delta
  per sub-plan, not duplicate implementations.
