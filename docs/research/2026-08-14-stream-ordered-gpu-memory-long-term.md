# Stream-ordered GPU memory: long-term plan — CUDA, 2026-08-14

Follow-up to the cuda-oxide borrow assessment
(`2026-08-13-cuda-oxide-borrow-assessment.md`). Covers what shipped, what the
code taught us that the assessment got wrong, and the systematic path for the
remaining host syncs.

## What shipped

Two OPD hot-path syncs replaced with stream-ordered equivalents:

1. **bf16 teacher-logits bridge** (`196eb2bb1`): the D2D copy now runs on the
   source stream (after the lm_head GEMM), gated by a completion event that
   the student stream waits on. The source's `cuMemFreeAsync` is ordered after
   the copy on the same stream. A/B on H20: 2.005 → 0.897 ms/call (2.24x).
2. **KV pool release** (`35a773d52`, revised in-tree): the two full-context
   syncs are replaced by a single event sync (record after drop, wait for the
   frees, then trim). The event sync is cheaper than a context sync because
   it only waits for the frees, not all queued work.

The pre-bridge teacher sync (`49b469456`) and the bf16 matmul tolerance
(`319ca1f9f`) also landed.

## What the assessment got wrong

The assessment recommended a ~150-line deferred-reclamation limbo (the
cuda-oxide `reclaim.rs` pattern). The code showed a simpler answer:

- cudarc's `CudaSlice::drop` already frees on the slice's own stream
  (`cuMemFreeAsync(ptr, slice.stream)`). The source buffer's free is
  stream-ordered. Ordering the *copy* on that same stream makes the free
  safe without a limbo.
- The limbo solves a different problem (cancelled futures) that ARLE does
  not have. Both sync sites were better addressed by stream ordering.

The limbo remains unbuilt and is not needed.

## The pattern

For a cross-stream producer→consumer handoff where the producer frees the
source after the handoff:

1. Run the copy on the producer's stream (ordered after the producer's last
   write).
2. Record a completion event on the producer's stream.
3. Make the consumer's stream wait on the event.
4. The producer's free (on its own stream) is ordered after the copy.

No host sync, no limbo, no background thread. The event wait is a device-side
dependency; the host continues enqueuing work.

For a free→trim handoff (KV pool): record an event after the drop, wait for
the event, then trim. The event sync is cheaper than a context sync.

## Remaining sync sites

Categorized by whether the sync is on the hot path:

### Hot path (serving / OPD step)

| File | Syncs | Why it's there | Can replace? |
|------|--------|----------------|--------------|
| `qwen35_decode.rs:366,624` | 2 | Decode step barriers | Likely — event waits if the consumer is on a known stream |
| `qwen35_state.rs:481,611,677` | 3 | Slot state transitions | Needs audit — some may be load-bearing for graph capture |
| `qwen35_spec.rs:185,413` | 2 | Spec decode verify | Needs audit |
| `qwen35_forward.rs` | 4 | Forward barriers | Needs audit |
| `deepep.rs:376,449,599,659` | 4 | NCCL/comm ordering | Likely load-bearing (NCCL stream semantics) |

### One-time / cold path

| File | Syncs | Why |
|------|--------|-----|
| `qwen35_load.rs` (7) | Weight loading | One-time cost, low priority |
| `loader.rs` (6) | Model loading | One-time cost |
| `qwen35_lora.rs:526` | LoRA merge | Per-step but not hot |

### Profiling (leave as-is)

`profile.rs`, `linear_profile.rs`, `moe.rs` — these sync to measure elapsed
time. They are diagnostic, not ordering.

## The OPD offload UAF (open)

The root-cause doc
(`errors/2026-08-14-opd-engine-offload-starves-autograd-forward.md`) describes
a separate class of problem: `--engine-offload student` re-profiles the KV
pool on reload, ratcheting it up until the co-resident autograd allocator is
starved. The fix is not a sync change — it is to cap the pool re-profile at
the original grant (or carry the token count through offload/reload).

This is the highest-leverage remaining item in the OPD memory path. The
stream-ordering work above is orthogonal: it makes the handoff cheaper, but
the offload UAF is about pool sizing, not ordering.

## Systematic approach

1. **Audit each hot-path sync**: what does it order? If the producer and
   consumer are on known streams, replace with an event wait. If the sync
   orders NCCL or graph capture, leave it.
2. **Cap the KV pool re-profile**: the offload/reload cycle must not
   re-profile from instantaneous free VRAM. Profile once at startup, carry
   the token count.
3. **Fail loud on allocation pressure**: an autograd allocation failure under
   pressure must fail the step, not let a kernel read stale memory.
4. **Launch contract** (low priority): the ~78 hand-written FFI kernels
   declare their geometry in comments. A per-kernel descriptor with
   `requires` checks at registration would catch shape mismatches before
   launch. This is a developer-experience improvement, not a perf item.

## What we did not build

- **The limbo**: unneeded, as explained above.
- **The launch contract**: documented as a future item; the current
  `launch_1d`/`launch_rows` helpers already centralize the geometry.
- **KernelFamily**: YAGNI for ARLE's current variant count.
