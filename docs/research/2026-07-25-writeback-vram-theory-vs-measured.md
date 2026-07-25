# Writeback VRAM: 12.4× above the theoretical floor, none of it physics

The masked-writeback backward peaks at **2.72 MiB/token** above a 28.5 GB base,
putting the single-card wall at **seq ≈ 25 400** (H20 96 GB, ThinkingCap-27B-FP8).
The per-layer theoretical floor is **~0.22 MiB/token → seq ≈ 260 000**. Every one
of the intervening 12.4× is an implementation choice, and the largest single item
is not a dtype but a **lifetime bug in a mechanism that already exists**.

Consequence: 256K single-card is reachable in principle; context parallelism is
not the lever, and was never the first question.

## Measured baseline

Source: `/host/ob172/*.log`, 12 synthetic replay arms, 2026-07-24. Ledger line:

```
[opd-vram-ledger] masked-writeback base_used_mib=28521 post_forward_used_mib=61129
                  post_backward_used_mib=93289 post_cleanup_used_mib=62857
                  allocator_retained_delta_mib=34336
```

| seq | post-fwd | post-bwd (peak) | peak − base |
|-|-|-|-|
| 5 120 | 36 297 | 42 729 | 14.2 GB |
| 12 288 | 45 385 | 61 897 | 33.4 GB |
| 24 576 | 61 129 | **93 289** / 97 508 | 64.8 GB |
| 28 672 | 66 345 | **OOM** | — |

Linear, no `seq²` term — both SDPA paths are chunked (`attention.rs:433`, q-chunk
to a ~4 GiB budget; forward tries the fused flash prefill at `attention.rs:65`
first). Attention contributes a **constant** floor, not slope.

Wall model, matching the measurement: `97508 = 28521 + 2.72 × seq → seq_max = 25 400`
(24 576 completed with 4219 MiB free; 28 672 OOM'd).

## Per-token decomposition

| # | Source | MiB/tok | Derivation | Evidence |
|-|-|-|-|-|
| 1 | Checkpoint hidden states | **1.31** | 64 layers × 5120 × 4B = 1.25 | **measured**: offload A/B forward delta 15.3 GB @ 12288 |
| 2 | One layer's recomputed MLP inner | 0.20 | 3 × 17408 × 4B | config arithmetic |
| 3 | Their gradients | 0.20 | idem | config arithmetic |
| 4 | Layer in/out hidden gradients | 0.04 | 2 × 5120 × 4B | config arithmetic |
| 5 | **Unattributed** | **~0.97** | — | see waste ③ |
| | **total** | **2.72** | | measured slope |

Not in the slope (constant): fp8 weights 28.5 GB (`base_used_mib`), attention
transients ~4–12 GiB (q-chunked), linear-attention state ~0.1 GB.

## Theoretical floor

The backward walks layers 64 → 1. At layer *i*, physically required:

| Required | elements/token | bf16 |
|-|-|-|
| layer *i*'s checkpoint input | 5 120 | 10 KB |
| its recomputed MLP inner | 2 × 17 408 | 68 KB |
| their gradients | 2 × 17 408 | 68 KB |
| the dY flowing through | 5 120 | 10 KB |
| LoRA gradients + optimizer state | — | **0** (seq-independent) |
| | | **~0.16 MiB/tok** |

Relaxed to three saved MLP tensors and three gradient tensors: **~0.22 MiB/tok**.

```
budget = 97 508 − 28 521 (base) − 12 GiB (attention floor) ≈ 56 GiB
seq_max = 56 GiB / 0.22 MiB ≈ 260 000
```

## The four wastes

### ① 63 of 64 checkpoints — 1.29 MiB/tok (47% of all waste)

The backward needs one checkpoint at a time; all 64 are resident. This is a
**lifetime** problem, not a dtype one — and `offload_checkpoints` was written for
exactly it (`checkpoint.rs:46-58`), but the A/B shows it does not achieve it:

```
backward-added:  offload off = 16.5 GB     offload on = 31.9 GB
delta 15.4 GB ≈ the full checkpoint set (15.3 GB)
```

Offload moves them to host, then the backward pulls **all of them back at once**.
Net peak benefit: 64 MiB. Net cost: **+45% backward wall time**. The correct
semantics is fetch-per-layer, drop after use.

Fixing this takes 1.31 → **0.02** MiB/tok — more than twice what a bf16 cast on
the same tensors buys (1.31 → 0.66), and it touches no numerics.

### ② f32 everywhere — 2× on every remaining term

The autograd tape is f32-only: **109 `alloc_zeros::<f32>`, zero bf16 activation or
gradient allocations**, with a hard guard at `backend_cuda.rs:554` ("cuda backend
cannot matmul a bf16 handle on this f32-only path"). `CudaBf16Storage` serves
weights and a GEMM bridge that converts straight back to f32.

Training needs bf16 activations and f32 accumulation/optimizer state — and the
optimizer state is LoRA-sized, seq-independent. **No seq-scaling tensor has a
precision reason to be f32.**

The one place a cast is local today is the checkpoint boundary: `checkpoint.rs:46`
states the saved input is untouched between save and backward replay, so no
operator sees the bf16 handle and the guard never fires. Cast down at
`offload_checkpoint_to_host`, cast up at `ensure_device` — two points.

### ③ `add_into`'s third buffer — ~0.5 MiB/tok (hypothesis)

```rust
// backend_cuda.rs:6047
let mut d_out = backend.stream.alloc_zeros::<f32>(size)   // dest + src → a NEW out
```

Gradient accumulation is not in-place, so three full-sequence tensors are live per
accumulation. This is the leading suspect for waste #5 in the decomposition;
arithmetic-consistent, **not isolated by experiment**.

### ④ Two undetermined constants

- **`allocator_retained_delta_mib=34336`** — `post_cleanup` sits 34 GB above base.
  Reusable allocator blocks (benign) or genuine retention (accumulating across
  trajectories)? Untested. Discriminator: whether `post_backward` drifts upward
  across successive trajectories in one process.
- **The inference engine's share of the 28.5 GB base** — ob172 arms are synthetic
  `replay writeback` runs, likely with no engine co-resident. Production
  agent-OPD has the engine on the same card, where it need not be during the
  writeback. `EngineOffloadMode::Student` exists; default `Off`.

## The ladder

| Stage | slope MiB/tok | real wall |
|-|-|-|
| today | 2.72 | 25 K |
| + checkpoint bf16 | 2.07 | 33 K |
| **+ checkpoint fetch-per-layer (①)** | **1.43** | **48 K** |
| + bf16 throughout (②) | 0.72 | 95 K |
| + in-place accumulation (③) | ~0.22 | **~260 K** |

① outranks the bf16 cast: twice the gain, no numerics touched, and it repairs a
mechanism that already ships. The 196 `SKIP trajectory` lines observed in the
sweetspot3 profile run span seq 23 013–34 142 — **stage ① alone clears all 196**;
the current `max_update_seq=23000` clears none of them.

## Ruled out

- **Context parallelism** — 8-way buys 8×; at today's slope 256K still OOMs
  (`708/8 + 28.5 = 117 GB > 95.6`). The coefficient must come down first
  regardless, and once it does, CP is not needed for the lengths this corpus has.
- **Flash backward** — the forward already takes a fused flash prefill when the
  backend accepts the shape; the backward's q-chunker holds its transient at a
  constant ~4 GiB. Flash backward buys **time** (measured fwd:bwd = 125 s : 463 s
  = 3.7×, against a normal ~2×), not slope.
- **Backward-gradient bf16 as a local change** — blocked by the 109-site f32
  monoculture and the `backend_cuda.rs:554` guard. It is a refactor, not a cast.

## What is not verified

All coefficients come from **synthetic replay trajectories with
`total_targets=64`**; production agent-OPD carries thousands (e.g. `total_targets=3232`
at seq 24781). The CE-side share of term #5 will move. Term #1 is
config-determined and does not. So the ranking above is stable, but the post-fix
acceptance numbers need one real-run ledger to calibrate.

`ARLE_SDPA_TRACE=1` (`attention.rs:93`) prints `fused=TAKEN|REJECTED` per call
and has never been captured on a training shape. If the fused prefill is being
rejected, the composed fallback head-chunks only down to one head, leaving a
`seq²` term that would impose a hard wall near 50 K — invisible in the measured
range but decisive above it.

## Method note

The offload A/B is the entry's own cautionary tale: reading it as "peak unchanged
⇒ checkpoints are not at the peak" is wrong. Only decomposing both arms
(forward-added vs backward-added) shows they are resident at the peak in both,
and that the offload merely defers them. **A null result on an aggregate is not a
null result on the mechanism.**
