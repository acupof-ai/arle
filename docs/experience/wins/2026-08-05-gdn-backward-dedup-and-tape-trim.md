# GDN backward: 187 lines of duplicated device algebra out, g/beta off the FlashQLA tape

**Date:** 2026-08-05 · **Status:** pending-remote (#81) — no nvcc here

## Context

Cleanup pass over the FlashQLA port (`4846f8046`). Four candidates were raised;
two shipped, two were killed by evidence, and the evidence is the point.

## Shipped

**`g` / `beta` no longer reach the tape on the FlashQLA route.** They were saved
by the forward, loaded and length-checked by the backward, and never read — that
branch re-derives both from `qkv_conv` via `gdr_fq_prep`. ~31 MB/layer of dead
residency at seq 81920 with 48 value heads, plus two guards per layer.
`LinearAttentionDeviceBackwardArgs::{g, beta}` become `Option`, and the
backward's hard `let Some(g) else return` guard now fires only off the FlashQLA
route.

**187 net lines of duplicated device algebra removed** from
`backend_cuda/kernels/linear_attention.cu` (+174/−361). Four `__device__`
helpers: `la_block_sum` (the block-wide tree reduction, written out 8 lines at a
time in 15 places), `la_rms_gated_backward_row` (templated on a row-accessor pair
so the bf16-global and f32-shared callers share one body, and on the caller's
reduction buffer so the grid topologies are untouched), `la_l2_norm_grad`, and
`la_gate_param_backward`.

The duplication was wider than the review found — the gate tail was also
character-identical in `linear_attention_scan_backward_f32` and
`linear_attention_chunked_scan_backward_f32`, and the RMS-gated block in
`chunked_scan_backward` too. `linear_attention_chunk_transfer_f32` has a similar
but genuinely different RMS block (no `dz` write) and was left alone.

**One float reassociation, and which direction it went matters.** The two
RMS-gated sites did not compute `dz` identically: the new standalone kernel had
`up * x * inv_rms * w * silu_grad(gate)`, the fused kernels
`up * (x*inv_rms*w) * silu_grad(gate)`. Float multiply is not associative, so one
caller had to change. The helper takes **the fused form** — those kernels are the
verified path and stay bit-identical; the FlashQLA kernel, which has never run on
hardware, is the one that conforms. Taking the other branch would have put a
1-ulp change into a path with standing measured references, to preserve one that
has none.

## Killed by evidence

**Saving `q/k/v/g_cumsum/a_inv` to skip the backward's recompute.** The premise
was that under gradient checkpointing those buffers are alive anyway, so keeping
them is free and drops three full-`seq_len` kernel launches per layer. Half the
premise held: `checkpoint_backward` (`tape.rs:983`) snapshots `live_before` at
`:1011`, runs the replay on a fresh inner tape, and frees by store live-id diff
at `:1046` — anything allocated during a replay is reclaimed.

The other half did not. `should_checkpoint` (`qwen35.rs:2886`) is
`gradient_checkpointing && tape.enabled && (forced || modeled×4 > free)`, so the
route is reachable **un**-checkpointed whenever the flag is off or the modeled
tape fits, and all three call sites then run a plain per-layer loop with the tape
enabled. Saving unconditionally would pin the five tensors for the whole forward:
at seq 81920 with H=48, Hg=16, kd=vd=128 that is **~3.5 GB/layer** (q/k/v 1006 MB
each, a_inv 503 MB, g_cumsum 16 MB) — an order of magnitude past the ~400 MB the
port cited as the cost it was avoiding.

The narrow form is a `Tape::in_checkpoint_replay` flag set only on the inner tape
at `tape.rs:1017`, with the five carried as `Option` alongside `raw_output`. That
is a `Tape` API change, not a cleanup, and it is not in this entry.

**Dropping `dht` / `dh0`.** Both are dead by the code's own comment — `dht` is
zero-filled and `dh0` is written and never read. But the AOT kernel reads `dht`
unconditionally (`flashqla_gdr.py:1396`) and writes `dh0` at `:1484`; a null
pointer faults and the signature is fixed by the `fq_bwd_v1` ABI. Both are
already the minimum the ABI needs (`state_len` ≈ 3.1 MB, sequence-independent).

## Verification

`cargo check`/`clippy -D warnings -p autograd --features cuda,no-cuda --lib`,
`cargo test -p autograd` (60 passed), `clang++ -std=c++17 -fsyntax-only` on the
`.cu` with CUDA stubs — all clean. As with the port itself, everything under
`cfg(not(feature = "no-cuda"))` is invisible to the local lane; the pod build is
the first real gate.

Two things the pod must check that nothing here can:

- **Register pressure.** `la_rms_gated_backward_row` is inlined into a kernel
  that is already shared-memory heavy. `-Xptxas -v` / `ncu` must show no
  occupancy change before the dedup is trusted.
- **One extra `__syncthreads()` per reduction.** `la_block_sum` ends with a
  barrier so its buffer is reusable on return; some original sites had none.
  Every converted site was checked to be reached uniformly by the whole block
  (the early returns all precede the loops), so it is safe, but it is 15 extra
  barriers on a hot kernel.

## Rule

When de-duplicating two copies of an expression that are not bit-identical, the
copy with measured references wins and the unverified one conforms. The direction
looks arbitrary while writing the helper and is not: it decides whether the next
parity failure has one suspect or two.
