# A6 T1 Retention Gate Miss

## Context

A6 targeted W4 canonical tool-resume retention after A3 proved same-session attach works in a one-session control. The task was to keep long session KV alive by demoting T0 GPU blocks into T1 host-pinned DRAM, then promote T1 back to T0 on session-tagged resume admission.

Code commits:

- `7b9ffb2f` - RadixCache host-retained block metadata, retagging, and leaf-first selection.
- `f83a8e05` - Scheduler T1 demote/promote path, long-session eligibility, explicit T1 capacity, and paged prefill reclaim.
- `f9e47d41` - Radix/T1 retention tests.

## Goal

W4 canonical avoided-prefill ratio >= 90%, matching the A3 one-session control signal (~96.2% avoided prefill / ~151 ms TTFT).

## Hypothesis

Canonical W4 missed because warmup KV was evicted before resume. If long session blocks survive in T1 and resume admission can promote them back to T0, resume requests should attach the 8k warmup prefix and prefill only the tool-output delta.

## Params

- Model: `models/default` (`Qwen3-4B` Instruct weights).
- GPU: NVIDIA L4 24GB.
- Server shape: `./target/release/infer --model-path models/default --port 8000 --num-slots 8 --max-seq-len 12288 --kv-cache-dtype fp8 --mem-fraction-static 0.85`.
- A6 T1 flags: `--t1-host-pinned-min-prompt-tokens 4096`, `--t1-host-pinned-high-water 0.98`, `--t1-host-pinned-low-water 0.95`, capacity tested at 32GiB and 44GiB.
- Trace: `scripts/bench_agent_trace.py --workload agent-w4-tool-resume --num-concurrent 8 --max-tokens 256 --probe-stats`.

## Env

Standard CUDA box env:

```bash
CUDA_HOME=/usr/local/cuda
CARGO_HOME=/tmp/cargo-home-local
PEGAINFER_CUDA_SM=89
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig
INFER_TILELANG_PYTHON=/usr/bin/python3
```

## Results

The gate did not pass. Resume still reported `matched_prefix_tokens=32` / `resume_prefill_tokens ~= 8.2k-8.5k` in live `/v1/stats` snapshots, and resume admissions continued logging `radix_gpu_attach=32`.

Iteration notes:

- `a6-w4-server.log`: host tier entries existed, but resume logged long `radix_hit` values as not reusable because bytes were not on GPU and there was no free slot; paged prefill then failed under pool pressure.
- `a6-w4-server-retag-32g.log`: fixed the demoted-block ID collision by retagging T1 blocks to scheduler-owned logical IDs; exposed paged prefill batch OOM (`needs 128 free pages, only ... available`).
- `a6-w4-server-retag-batch-32g.log`: fixed prefill batch reclaim; canonical resume phase still stayed at `radix_gpu_attach=32` with no staged-prefix admission.
- `a6-w4-server-retag-hostevict-44g.log`: 44GiB T1 was killed by Linux OOM (`infer` RSS about 49GiB); run ended 85/213 turns, no valid final stats.
- `a6-w4-server-retag-hostevict-32g.log` and `a6-w4-server-drain-hostevict-32g.log`: host tier hit full pressure and fell back to dropping GPU blocks; live stats still stayed at `matched_prefix_tokens=32`.

Verification that passed before the final bench attempts:

```bash
cargo check -p infer --no-default-features --features no-cuda
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo test -p infer --release --no-default-features --features no-cuda prefix_cache::tests:: --lib
cargo build -p infer --release --features cuda
```

The `cuda,no-cuda` and CUDA build warnings were the pre-existing `infer/src/speculative/cuda.rs` unused imports.

## Problems

Three implementation bugs were found and fixed:

- Demoted host blocks reused physical GPU page IDs. After those pages were released and reallocated, radix metadata could collide with new T0 blocks. Fix: `retag_block(old, new)` and high-range logical `BlockId`s for below-T0 blocks.
- Paged prefill batches could allocate without first reclaiming prefix-cache pages. Fix: `reclaim_for_paged_appends` before sync/async prefill and mixed decode+prefill launches.
- T1 full pressure could force immediate GPU drops. Fix attempt: leaf-first host eviction for demote headroom, including a Drain path that ignores lookup soft pins while preserving ref-count and leaf invariants.

The gate miss remained after those fixes. The immediate binding issue is not simply T1 capacity: resume lookup still enters through token-walk on the rendered prompt, and W4 chat/tool serialization diverges near token 32. KV in T0/T1/T2 cannot be promoted if admission never selects the session's warmup prefix as the semantic resume base.

Capacity is still a ceiling for this exact A6-only design: T0 has 8583 KV pages; 32GiB T1 stores about 28k more blocks at 1,216,512 bytes/block, so T0+T1 is about 37k blocks versus W4's roughly 66k warmup blocks. 44GiB would still be below 90% retention and OOMs this box. That ceiling is secondary to the observed `matched_prefix_tokens=32` lookup failure.

## Learnings

W4 needs a semantic session-resume lookup key or token-stable prompt serialization before more KV capacity work can move the metric. The plausible fork is:

- A: session-id keyed lookup/readmission that can attach the prior warmup transcript even when the resume prompt diverges after tool-call injection.
- B: chat library serialization that makes warmup and resume token prefixes stable through the tool-call boundary.
- C: ship the current A6 partial as retention substrate, document that W4 canonical is blocked on semantic lookup.

Do not continue capacity-only iterations for W4 canonical; they will not turn `matched_prefix_tokens=32` into an 8k staged prefix.

## Rule

When W4 resume matched-prefix stays at 32, stop capacity experiments and inspect the semantic lookup boundary first. T1/T2 retention only helps after admission can identify the session prefix to promote.
