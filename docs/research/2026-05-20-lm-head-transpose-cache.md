# 2026-05-20 — `lm_head` transpose-copy is the dominant CPU OPD bottleneck

> **Status:** research / proposal. Companion to
> [`docs/experience/wins/2026-05-20-bench-opd-rollout-last-logits-pending.md`](../experience/wins/2026-05-20-bench-opd-rollout-last-logits-pending.md)
> — the rollout-last-logits investigation flagged this as the dominant
> confounder. This doc consolidates the four candidate `lm_head` perf
> axes with FLOP/bandwidth math + ROI ranking. **Code path:** codex (per
> 2026-05-20 work split — Claude does research/docs, codex does complex
> implementation + verification).

## Problem

At Qwen3-0.6B (`vocab=151_936`, `hidden=1024`) the CPU OPD step is matmul-
dominated, and within the matmul slice `lm_head` accounts for **54 % of
per-step matmul time** (1.82 s of 3.37 s, from
[`2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md`](../experience/wins/2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md)).
That wins entry's §Problems section already noted: *"`lm_head` backward is
bandwidth-bound, not kernel-bound."* But it framed the cost as the matmul
itself streaming the 608 MB weight through main memory.

Investigation during the 2026-05-20 `forward_last_logits` tranche surfaced
a sharper diagnosis: **`linear_forward` (`crates/train/src/qwen35.rs:980`)
physically transposes the lm_head weight on every call**. The relevant
chain:

```rust
// crates/train/src/qwen35.rs:1018
let weight_t = transpose(weight, 0, 1, store, tape)?;
let projected = matmul(flat_x, weight_t, store, tape)?;
```

`transpose` (`crates/autograd/src/ops/layout.rs:110-130`) dispatches on
`device_handle` presence — on the CPU host path it falls into
`transpose_host_eager` (`crates/autograd/src/ops/layout.rs:181-202`), which
calls `transpose_data` (`crates/autograd/src/ops/layout.rs:500-532`). That
function allocates `vec![0.0; data.len()]` and copies every element via a
strided index walk. **This is a full physical 623 MB allocation + copy per
`lm_head` forward call** at Qwen3-0.6B.

### Cost decomposition (Qwen3-0.6B, prompt=3 + rollout=2 → seq=5)

| Op (per OPD step) | Calls | Per-call cost | Cumulative |
|---|---:|---:|---:|
| `lm_head` transpose copy (`vec![0;…]` + scalar gather) | 3 | 623 MB / ~6 GB·s⁻¹ ≈ **100 ms** | **300 ms** |
| `lm_head` matmul (5 × 1024 × 151 936) | 3 | ~90 ms @ 8.55 GF/s | 270 ms |
| `lm_head` backward matmul | 1 | ~245 ms @ 2.95 GF/s | 245 ms |
| **`lm_head` subtotal** | — | — | **~815 ms** (54 % of 1.50 s step\*) |
| All other matmuls | many | — | ~700 ms |

\* The 1.80 s step time in the mixed-dispatch entry includes the full step;
the 1.50 s estimate here scales to the cost share attributable to `lm_head`.

The 6 GB/s estimate for the scalar transpose is conservative — DDR4-3200
single-channel peaks at ~25 GB/s, but `transpose_data`'s scatter pattern
hits a different cache line on every store, so realistic throughput is
3-10 GB/s. **A `ncu`-style trace cannot run on CPU**; the SOLID move is
to instrument the transpose call directly with `Instant::now()` and a
counter. (Codex action item.)

### Why this masks the `forward_last_logits` win

The rollout student calls `lm_head` twice during 2-token rollout (at seq=3
and seq=4). Last-row-only saves `(seq_len - 1)/seq_len` of the matmul work
on the rollout student. But it does *not* eliminate the transpose copy.
Per-rollout-call cost:

| Variant | Transpose (ms) | Matmul (ms) | Total (ms) |
|---|---:|---:|---:|
| `forward` (full) | 100 | 60 + 80 = 140 | 240 |
| `forward_last_logits` | 100 | 20 + 20 = 40 | 140 |
| **Saving** | — | **100** | **100** (~5 % step) |

Compared with the rollout-`lm_head`-matmul saving in isolation (~70 % of
the rollout `lm_head` work), the wall-clock win is squeezed down to ~5 % of
step because the transpose is a fixed cost that survives. **Killing the
transpose copy directly is structurally larger ROI than slicing rows.**

## Candidate axes for the `lm_head` perf surface

Ranked by Claude-side ROI estimate; codex owns benching + final ranking.

### Axis A — Transpose-aware forward matmul (recommended)

**Approach.** Introduce a public `matmul_bt(a, b)` op that computes
`C = A @ B^T` directly, dispatching to the already-implemented
`matmul_a_bt_into` (`crates/autograd/src/backend.rs:1697-1734`). Use it in
`linear_forward`:

```rust
// Before
let weight_t = transpose(weight, 0, 1, store, tape)?;
let projected = matmul(flat_x, weight_t, store, tape)?;

// After
let projected = matmul_bt(flat_x, weight, store, tape)?;
```

**Backward.** Needs a new `BackwardOp::MatmulBT` variant. For `C = A @ B^T`:
- `grad_A = grad_C @ B` (plain matmul — existing kernel)
- `grad_B = grad_C^T @ A` = `(A^T @ grad_C)^T` (transpose-aware via
  `matmul_at_b_into`, then transpose — *or* directly compute
  `grad_B[k, n] = sum_m grad_C[m, k] * A[m, n]` which is a new kernel
  variant)

The cleanest formulation uses both existing transpose-aware kernels:
- `grad_A` via `matmul` (rank-2 plain)
- `grad_B` via `matmul_at_b_into(grad_C, A)` then a single transpose at
  the end (or a new "row-major output transposed" variant)

**ROI estimate.** Eliminates 100 ms × 3 forward calls + 100 ms × 1 backward
call = **~400 ms per step**, ~22 % of the 1.80 s step. Roughly the same
order as the original mixed-dispatch tranche. Compounds with
`forward_last_logits` (both wins additive — last-logits removes matmul
work, transpose-bt removes copy work).

**Risk.** Backward kernel symmetry must be carefully derived. Determinism
test is the safety net; grad-check is the numerical safety net. New tape
op needs full backward-tree coverage in `tape::backward`.

**Complexity.** Medium-high. New public op, new BackwardOp variant, three
backend dispatches (CPU + Metal + CUDA), and the Qwen3.5 linear_forward
switch. Codex's lane.

### Axis B — Cache the transposed lm_head weight (lower ROI but cheaper to implement)

**Approach.** Allocate `lm_head_t` once at `Qwen35Model::new` (or lazily
on first forward). `linear_forward` checks for a cached `_t` companion and
skips the per-call transpose.

**Invalidation.** Weight mutates after every optimizer step → cache must
be regenerated. Two options:
1. **Invalidate on optimizer step:** `AdamW::step` clears the cache; next
   `linear_forward` regenerates. Cost: 1 transpose per step (vs 4 currently
   for teacher+student-full+rollout-student+backward). Wins ~75 % of the
   transpose budget. ~300 ms / step saved.
2. **Maintain alongside step:** optimizer also updates `lm_head_t`. Cost
   doubles weight memory + adds 1 transpose per step. Wins close to 100 %
   of the transpose budget. ~400 ms / step saved.

**ROI estimate.** Similar to Axis A (~22 % step), but:
- Adds 623 MB persistent memory at Qwen3-0.6B
- Doesn't compound cleanly with other transpose elimination work (it's a
  caching hack on top of the existing path, not a structural rewrite)
- Inference-side compat: frozen weights mean cache never invalidates, so
  this is *strictly better than nothing* for the teacher forward

**Complexity.** Low-medium. Add an `Option<TensorId>` field per linear
layer; an invalidation hook on optimizer step (or hand the cache to the
optimizer for in-place update). No autograd-side changes.

**Recommendation.** Implement Axis A *if* the backward kernel work is in
scope. Axis B is the fallback if Axis A's backward symmetry turns out
to be harder than expected.

### Axis C — bf16 lm_head weight (orthogonal, harder)

**Approach.** Store `lm_head` weight as bf16 (half size = 312 MB instead
of 623 MB) and dequantize on-the-fly during matmul.

**ROI estimate.** Halves transpose-copy bandwidth and matmul-streaming
bandwidth. Synergistic with both Axis A and Axis B. But:
- Numerical precision impact on KL distill loss must be measured (likely
  acceptable — bf16 has same exponent range as f32)
- Need bf16-aware matmul (CPU side currently f32-only; matrixmultiply
  supports f32 only)
- Determinism test would need an updated baseline

**Complexity.** High. New dtype path in autograd, new matmul kernel variants,
new safetensors load path. Deferred until Axis A or Axis B lands.

### Axis D — Rayon-parallel K-shard of lm_head matmul (orthogonal)

**Approach.** Parallelise the lm_head matmul across `N` (vocab) using rayon.
The mixed-dispatch entry already showed `matrixmultiply::threading` regresses
22 % at M=4 OPD shapes (per-tile thread overhead). But explicit `N`-axis
sharding (each thread computes `vocab/T` rows of the output) avoids the
per-tile coordination overhead — each thread does an independent dense
`matrixmultiply::sgemm` over its slab.

**ROI estimate.** On 8C/16T Zen 2, 4-8× theoretical ceiling on the lm_head
matmul alone. With bandwidth-bound regime, realistic 2-3× ≈ 50-70 % matmul
saving = ~120 ms / step. Compounds with all three above axes.

**Complexity.** Medium. Need a custom dispatch in `cpu_matmul_forward` for
`N ≥ vocab_threshold`, splitting the output buffer + spawning rayon scope.
Backward grad_B requires per-thread accumulators or a reduction step.

**Recommendation.** Pursue after Axis A lands. The matmul will then be the
binding constraint again, and parallelism is the natural next lever.

## Recommended sequencing

1. **A/B verify `forward_last_logits` at Qwen3-0.6B-vocab shape** (codex,
   in progress). Pass → ship as `verified`. Kill → revert.
2. **Axis A — transpose-aware `matmul_bt` + `linear_forward` rewrite**
   (codex). Verify with determinism + grad-check + bench. Expected ~22 %
   step saving regardless of #1 outcome.
3. **Axis D — N-axis rayon shard of CPU sgemm** (codex). Best to land after
   #2 so the matmul itself is the binding constraint.
4. **Axis C — bf16 lm_head** (codex, lower priority). Defer until #1-3 are
   merged or until a precision-tolerance experiment justifies it.

Axis B is the cheap fallback if Axis A's backward derivation is harder
than expected; it can be inverted in priority with Axis A if codex finds
backward symmetry impractical for a single sprint.

## Acceptance gates (each axis)

- `cargo test -p autograd --release` green (especially numerical kernel tests)
- `cargo test -p train --test test_opd_determinism --release` bit-identical
- `cargo test -p train --test test_opd_grad_check --release` finite-diff agreement
- `cargo clippy -p autograd -p train --all-targets --release -- -D warnings` clean
- Production-shape wall-clock A/B with sigma_pct ≤ 2 % showing the projected
  step-level win (or kill if not measurable)

## Open questions for codex

1. **Backward kernel for `matmul_bt`.** Is the cleanest derivation to call
   `matmul_at_b_into` then transpose, or to add a dedicated kernel that
   writes `grad_B` in `[K, N]` order directly? The latter avoids a final
   transpose but doubles the kernel surface.
2. **Cache-once + invalidate-on-step.** If Axis A turns out to be deferred,
   Axis B needs a cache invalidation hook. Where's the right boundary — at
   `Optimizer::step`, at `TensorStore::get_mut`, or via a generation
   counter on the weight tensor?
3. **Memory budget under cooperative sessions.** The OOM during the
   `forward_last_logits` A/B suggests the dev box's combined Claude+codex
   RSS approaches the limit even on small benches. Should the benches gate
   on a `free -h` check before starting, or split into separate processes
   reading shared snapshots?

## Addendum — rollout intra-iteration `TensorStore` pruning

The `forward_last_logits` A/B harness OOM'd partly because rollout
ephemerals accumulate across iterations (`crates/train/src/opd.rs:109-113`
comment: *"rollout ephemerals (logits per iteration) stay in the store
until the post-step retain_ids prune below"*). Codex's harness fix
(rev'd in `crates/train/examples/rollout_last_logits_ab_bench.rs`) added
`retain_model_tensors` between rollout iters, snapshot-pruning back to
just the model parameters before each new iteration. That works for the
bench because there's no backward through the rollout.

For the *production* `opd_step`, intra-iteration pruning is non-trivial:

- Tape is disabled during rollout, so no backward graph exists for
  rollout activations — they're safe to free.
- The `keep` set must include teacher params, student params, **and**
  cos/sin caches (which are referenced by every forward). The 109-113
  comment notes this explicitly.
- Existing `cleanup_after_backward` (`crates/train/src/trainer.rs:671`)
  only handles the post-step prune. There's no shared helper for
  rollout-intra prune.

**Proposal.** Add a small helper `cleanup_after_rollout_iter(store, tape,
keep_extra)` symmetric to `cleanup_after_backward` but without the
`set_enabled(true)` side effect (rollout iters want tape *disabled*).
The keep-set construction is the caller's responsibility — `opd_step`
constructs it once before the rollout loop from `teacher.all_parameter_ids()
∪ student_params ∪ {cos_cache, sin_cache}` and reuses across iters.

**ROI.** Memory only, not wall-clock. At Qwen3-0.6B with `rollout_len=2`,
the logits ephemerals are ~600 MB per rollout iter (logits tensor =
`[1, seq_len, 151_936] × 4 B`) — adding up to ~1.2 GB peak that survives
until the post-step prune. Pruning intra-iter caps the rollout peak at
~600 MB. Critical for `rollout_len > 4` or for cooperative-session
memory pressure, not a perf win on its own.

**Complexity.** Low. ~30 lines including keep-set construction in
`opd_step`. Codex's lane (touches the production OPD path).

## Cross-links

- Wins (pending verification): [`2026-05-20-bench-opd-rollout-last-logits-pending.md`](../experience/wins/2026-05-20-bench-opd-rollout-last-logits-pending.md)
- Wins (substrate): [`2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md`](../experience/wins/2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md)
- Wins (transpose-aware backward): [`2026-05-19-bench-cpu-matmul-backward-transpose-aware-qwen3-06b.md`](../experience/wins/2026-05-19-bench-cpu-matmul-backward-transpose-aware-qwen3-06b.md)
- Backend kernels: `crates/autograd/src/backend.rs:1697-1810` (`matmul_a_bt_into`, `matmul_at_b_into`)
- Linear forward: `crates/train/src/qwen35.rs:980-1023`
- Transpose copy: `crates/autograd/src/ops/layout.rs:181-202` and `:500-532`
