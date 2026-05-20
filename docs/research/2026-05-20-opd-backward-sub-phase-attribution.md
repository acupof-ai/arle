# 2026-05-20 — OPD backward sub-phase attribution: MatmulBT (56%) + merge_grad (39%) dominate

> **Status:** research / findings. Companion to
> [`../plans/2026-05-20-opd-cpu-perf-codex-handoffs.md`](../plans/2026-05-20-opd-cpu-perf-codex-handoffs.md)
> §"P4 — Backward sub-phase profiling". Data sourced from codex's
> existing `backward_op_summary` print in
> `crates/train/examples/opd_step_cpu_moderate_profile.rs`
> (`bench-output/2026-05-20-opd-backward-op-profile/run.txt`).
> Implementation handed to codex per the 2026-05-20 work split.

## Headline

After `506f02b` (AdamW host-zip-loop) landed, `backward` became the
dominant coarse phase at **29 % of step**. Within backward (4.23 s total
over 5 steps at moderate shape):

| Sub-phase | Seconds (5 steps) | % of `backward` |
|---|---:|---:|
| Op kernels (`op_seconds`) | 2.572 | 60.8 % |
| `merge_grad` host accumulation | 1.653 | 39.1 % |
| Prelude (setup, topo sort) | 0.002 | 0.05 % |
| Unattributed | 0.004 | 0.09 % |

Inside the 2.572 s of op kernels:

| Op | Count | Seconds | % of `backward` total |
|---|---:|---:|---:|
| **MatmulBT** | 1275 | 2.375 | **56.1 %** |
| AddBroadcast | 540 | 0.041 | 0.96 % |
| Transpose | 1080 | 0.038 | 0.90 % |
| Embedding | 15 | 0.029 | 0.69 % |
| Mul | 375 | 0.018 | 0.42 % |
| Slice | 360 | 0.017 | 0.39 % |
| Softmax | 195 | 0.011 | 0.26 % |
| LogSoftmax | 15 | 0.010 | 0.25 % |
| RMSNorm | 735 | 0.008 | 0.19 % |
| Silu | 180 | 0.007 | 0.17 % |
| Matmul (plain) | 360 | 0.007 | 0.16 % |
| Reshape | 4725 | 0.007 | 0.15 % |
| (all other 5 ops combined) | ~915 | <0.005 | <0.10 % |

**Two clear axes dominate the 95+%-tile:**

1. **MatmulBT backward — 56 % of backward, 16 % of step.** Already on
   the transpose-aware kernels (`6e37b91`) plus matmul_bt-symmetric
   backward (`0b593e1`). The remaining cost is purely the
   `matrixmultiply::sgemm` + `matmul_at_b_into` arithmetic. **Likely
   bandwidth-bound** at large shapes — saxpy/matrixmultiply hit ~17 GF/s
   on a 25 GB/s-class memory channel; the OPD step's per-call shapes
   already stream tens of MB.

2. **`merge_grad` host accumulation — 39 % of backward, 11 % of step.**
   Defined in `crates/autograd/src/tape.rs:401-471`. The host fast path
   (no device handle present) does:

   ```rust
   let incoming_data = store.to_host(new_grad_id)?;  // CLONES new_grad data
   let existing = store.tensor_mut(existing_grad_id)?;
   for (dst, src) in existing.data.iter_mut().zip(incoming_data) {
       *dst += src;
   }
   ```

   The `to_host` allocates + memcpys the full incoming grad before the
   scalar add loop. For a `[151_936, 1024]` `lm_head` weight grad
   (622 MB) on CPU, this is ~50 ms clone + ~50 ms add = ~100 ms per
   merge call, paid every backward step.

## Candidate axes for the `backward` perf surface

### Axis E — `merge_grad` clone elimination (lower complexity, ~5-10 % step)

**Approach.** Add a `TensorStore::add_into_host(dst_id, src_id)` helper
that adds `src` into `dst` in-place without the intermediate clone.

Borrow-rule challenge: both `dst` and `src` live in
`self.tensors: Vec<Option<Tensor>>`. Cleanest approach is
`split_at_mut`:

```rust
pub fn add_into_host(&mut self, dst_id: TensorId, src_id: TensorId) -> Result<()> {
    use std::cmp::Ordering;
    self.ensure_host(dst_id)?;
    self.ensure_host(src_id)?;
    match dst_id.cmp(&src_id) {
        Ordering::Less => {
            let (lo, hi) = self.tensors.split_at_mut(src_id);
            let dst = lo[dst_id].as_mut().ok_or(AutogradError::InvalidTensorId(dst_id))?;
            let src = hi[0].as_ref().ok_or(AutogradError::InvalidTensorId(src_id))?;
            // shape check
            for (d, s) in dst.data.iter_mut().zip(&src.data) { *d += s; }
            Ok(())
        }
        Ordering::Greater => {
            let (lo, hi) = self.tensors.split_at_mut(dst_id);
            let src = lo[src_id].as_ref().ok_or(AutogradError::InvalidTensorId(src_id))?;
            let dst = hi[0].as_mut().ok_or(AutogradError::InvalidTensorId(dst_id))?;
            for (d, s) in dst.data.iter_mut().zip(&src.data) { *d += s; }
            Ok(())
        }
        Ordering::Equal => {
            // grad += grad : doubling — caller bug, but technically valid
            let dst = self.tensors[dst_id].as_mut().ok_or(AutogradError::InvalidTensorId(dst_id))?;
            for v in dst.data.iter_mut() { *v *= 2.0; }
            Ok(())
        }
    }
}
```

Then `merge_grad`'s host fast-path becomes:

```rust
} else {
    store.add_into_host(existing_grad_id, new_grad_id)?;
}
```

**ROI estimate.** Saves the clone half of `merge_grad`. ~50 % of
1.65 s = ~0.83 s over 5 steps = ~165 ms per step. On a 0.83-s step
that's **~20 % step saving** (capped by the share of merge_grad with
large operands). The lm_head merge dominates because of its size; other
projections contribute proportionally.

**Acceptance criterion:** End-to-end OPD profile median step time
improves ≥ 1.10 × at σ ≤ 5 %. Determinism + grad-check stay bit-identical.

**Complexity.** Low. ~30 lines including shape check + error mapping.
Codex's lane (touches autograd substrate).

### Axis F — MatmulBT backward kernel investigation (higher complexity, harder win)

**Approach.** 56 % of backward is in the `MatmulBT` op variant. The
kernel is already optimised (transpose-aware, `matrixmultiply` for wide
N, saxpy for M=1). Further wins would require:

- **Bandwidth-bound case analysis.** Instrument the matmul_bt backward
  call to track bytes-streamed-per-second. If we're ≥ 80 % of memory
  bandwidth peak, the kernel is saturated and only algorithmic changes
  (block tiling for L2/L3 reuse, lower precision) help.
- **Rayon N-shard for wide N (lm_head bwd).** The previous decision tree
  flagged this; `matrixmultiply::threading` regresses at M=4 OPD shapes,
  so per-thread independent `sgemm` over N-axis chunks is the right
  shape. The 1275 backward calls include many small projections + a few
  giant lm_head; parallelizing lm_head bwd specifically (largest N) is
  the natural target.
- **bf16 weight storage.** Halves the streaming cost. Complex (new
  dtype path, see prior decision tree).

**ROI estimate.** Bandwidth-bound 2-4 × ceiling on lm_head bwd (largest
N) ≈ ~80 ms per step saved. The other 1270 matmul_bt calls are smaller
and likely compute-bound; less win there. Total step saving estimate:
~100 ms = ~12 % step.

**Acceptance criterion:** End-to-end OPD profile median step time
improves ≥ 1.10 × at σ ≤ 5 %, AND the lm_head bwd call duration
improves ≥ 1.5 × in isolation (single-variable A/B).

**Complexity.** Medium-high. Requires careful single-variable A/B at
each shape, and rayon-shard implementation needs per-thread accumulator
strategy.

### Axis G — `Transpose` backward + `AddBroadcast` backward (tiny)

`AddBroadcast` is 0.96 %, `Transpose` is 0.90 % of backward. Below the
"worth investigating" threshold per the 5 % rule. Defer indefinitely.

## Recommended sequencing

1. **Axis E (`merge_grad` clone elimination)** — quick, scoped, almost
   certain ROI. Land first.
2. **Re-measure backward sub-phase.** Confirm the merge_grad share drops
   from 39 % to ~5 % as expected. New dominant: probably MatmulBT at
   ~85+ % of backward.
3. **Axis F decision.** If post-E backward is still ≥ 20 % of step,
   license the lm_head bwd rayon-shard. If backward dropped below 15 %,
   declare backward complete and move to the next coarse phase.

## Hand-off

Codex owns implementation. The data and the decision-tree are here;
the code change for Axis E is small enough that the snippet above is
copy-paste-ready (modulo error mapping conventions in the surrounding
`tape.rs`).

## Cross-links

- Coarse phase data: `bench-output/2026-05-20-adamw-host-zip-loop-ab/opd_profile_after.txt`
- Backward op breakdown: `bench-output/2026-05-20-opd-backward-op-profile/run.txt`
- Existing matmul_bt backward kernel: `crates/autograd/src/backend.rs::cpu_matmul_bt_backward`
- Existing merge_grad: `crates/autograd/src/tape.rs:401-471`
- Hand-offs index: [`../plans/2026-05-20-opd-cpu-perf-codex-handoffs.md`](../plans/2026-05-20-opd-cpu-perf-codex-handoffs.md)
