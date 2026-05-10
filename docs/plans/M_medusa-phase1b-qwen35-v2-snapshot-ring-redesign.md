# M_medusa Phase 1.B-Qwen3.5 v2 — recurrent snapshot ring redesign

> **Why this brief**: User direction 2026-05-10 = "stop Qwen3
> optimization, only Qwen3.5+". Per `05270b1` codex Step 0 audit
> + `2026-05-10-medusa-qwen35-recurrent-rollback-blocker.md`, Phase
> 1.B-Qwen3 substrate brief is PAUSED. Qwen3.5 hybrid linear-attention
> recurrent state has no partial-truncate operation, so the existing
> speculative `paged_kv_pool.truncate_slot` is insufficient for
> Medusa K+1 verifier rollback. This brief picks audit Option 1
> (snapshot ring) with formula prediction + prototype scope + LOC.
>
> **Status**: ready for codex pickup once Step 0 prototype validates
> snapshot-ring overhead per `kernel-optimization` skill Phase 4
> formula prediction below.

---

## §1 Problem (from blocker doc)

Qwen3.5 decode mutates **two** state classes per token:
- Paged KV (full-attention layers) — supports `truncate_slot` rollback
- Recurrent state (linear-attention conv + GDR layers) — only `reset()`
  to zeros, no partial truncate

Medusa K+1 verifier runs the target `K+1` times to verify K draft
tokens. If only `j ≤ K` are accepted, the existing scheduler:
- Truncates paged KV to accepted length ✓
- Leaves recurrent state advanced through all K+1 verifier tokens ✗

Result: greedy consistency violated; risks anti-pattern #26 same-output-
but-garbage failures.

---

## §2 Picked design — Option 1: snapshot ring

### §2.1 Mechanism

Extend `RecurrentSnapshot` (`infer/src/model/qwen35/recurrent_state.rs`
L91-134) from single-snapshot to N-deep ring buffer:
- N = K+1 (Medusa head count + 1 for bonus token), default 6
- Before each verifier step, push current state to ring tail
- After verification with accepted length j, restore from ring entry j
- Ring rotates per decode step (memcpy reuses CudaSlice allocation)

### §2.2 Formula prediction (per `kernel-optimization` skill Phase 4)

Per-snapshot cost per blocker doc: ~49 MB GPU memcpy / Qwen3.5-4B
(24 layers × ~2 MB each).

```
snapshot_cost_per_step = K_plus_1 × 49_MB × N_concurrent
                       = 6 × 49 MB × 4 = 1.176 GB

snapshot_overhead_ms = snapshot_cost_per_step / hbm_bandwidth
                     = 1.176 GB / 672 GB/s = 1.75 ms

current_qwen35_decode_itl = ~10-15 ms (estimate; needs measurement)
overhead_pct = 1.75 / 12.5 = 14% step cost

E[accepted_tokens / step] (α=0.6, K=5 per `M_medusa-required-path.md`)
                          = 1 + 0.6 + 0.36 + 0.22 + 0.13 = 2.31

throughput_speedup = E[accepted] / (1 + overhead_pct)
                   = 2.31 / 1.14 = 2.03×
```

Target: 2.03× tok/s vs no-spec on Qwen3.5 hybrid model.
License threshold per Phase 1.B brief §1: ≥1.5× → LICENSE
Soft win: ≥1.2× → keep
Kill: <1.0× → revert

### §2.3 Memory cost

Ring buffer = N × 49 MB = **294 MB additional GPU memory** for K=5.
At conc=4: 4 × 294 = ~1.2 GB ring memory total.

Qwen3.5-4B working set fits in ~10 GB on 16 GB sm_89. Adding 1.2 GB
ring = 70% utilization, acceptable.

For larger Qwen3.5 sizes (35B), ring memory scales linearly. May
require K reduction (K=3 → 588 MB ring) or shadow-state Option 2.

---

## §3 Substrate scope (Rust LOC)

| File | Edit | LOC |
|---|---|---:|
| `infer/src/model/qwen35/recurrent_state.rs` | Extend single-snapshot → ring; add `push_ring_slot()`, `restore_from_ring(j)` | +60 |
| `infer/src/model/qwen35/forward.rs` | Add verifier hook `forward_spec_verify_batch` (mirrors qwen3) + ring usage | +80 |
| `infer/src/scheduler/cuda/spec_path.rs` L251-258 | Add model-owned commit hook callback for non-KV state rollback | +40 |
| `infer/src/scheduler/cuda/execution.rs` | Wire commit hook into spec verification path | +20 |
| Tests: `tests/test_qwen35_spec_rollback.rs` | greedy_consistency under reject scenarios | +60 |
| **TOTAL** | | **~260 LOC** |

Plus existing Phase 1.B Medusa scope (~250 LOC for `medusa.rs` + ~80 for
weights + ~50 for speculative.rs integration) = total Qwen3.5 Medusa
substrate ~640 LOC.

---

## §4 Step 0 prototype (license-or-kill BEFORE full impl)

Per blocker doc requirement: "small prototype that measures recurrent
snapshot-ring overhead per decode step".

### §4.1 Prototype scope (~50 LOC)

Add a benchmark function `bench_recurrent_snapshot_ring(K, conc)` to
`infer/src/model/qwen35/recurrent_state.rs` (gated behind
`#[cfg(test)]` or `--features qwen35-snapshot-bench`):

```rust
pub fn bench_snapshot_ring_overhead(
    ctx: &DeviceContext,
    state: &mut RecurrentState,
    k_plus_1: usize,
) -> Result<Duration> {
    let start = std::time::Instant::now();
    let mut ring = Vec::with_capacity(k_plus_1);
    for _ in 0..k_plus_1 {
        let snap = state.clone_to_snapshot(ctx)?;
        ring.push(snap);
    }
    // Restore from arbitrary slot
    state.restore_from(&ring[k_plus_1 / 2], ctx)?;
    Ok(start.elapsed())
}
```

### §4.2 Acceptance gate

| Metric | License | Kill |
|---|---:|---:|
| Snapshot ring overhead at K=5 / Qwen3.5-4B | <2.0 ms | >5 ms (revert to Option 2/3) |
| Snapshot ring overhead at K=5 / Qwen3.5-35B | <8.0 ms | >20 ms |
| Memory cost validated | <1.5 GB at conc=4 | >3 GB at conc=4 |

If license: proceed to full §3 substrate.
If kill: pivot to Option 2 (shadow state) or Option 3 (Qwen3.5 Medusa
defer until measured).

---

## §5 Greedy consistency test plan

Per blocker doc rule: "every mutable state advanced by verifier tokens
has an accepted-length commit or rollback mechanism".

| Test | Scenario | Pass criterion |
|---|---|---|
| `test_qwen35_spec_all_accept` | All K draft tokens accepted | Same output as no-spec greedy, 0% diff |
| `test_qwen35_spec_zero_accept` | Zero draft tokens accepted | Same output as no-spec greedy, 0% diff |
| `test_qwen35_spec_partial_accept` | j∈{1,2,3,4} accepted (param) | Same output as no-spec greedy, 0% diff |
| `test_qwen35_recurrent_rollback_idempotent` | Push K snapshots, restore each, verify state matches | bitwise equal recurrent state at each restore |

---

## §6 Cross-references

- Blocker: `2026-05-10-medusa-qwen35-recurrent-rollback-blocker.md`
- Audit: `2026-05-10-medusa-phase1b-qwen35-step0-audit.md`
- Prior brief: `M_medusa-phase1b-substrate-brief.md` (PAUSED for Qwen3)
- Master path: `M_medusa-required-path.md` Phase 1
- Existing snapshot: `infer/src/model/qwen35/recurrent_state.rs:91-134`
- Existing truncate: `infer/src/model/qwen35/forward.rs:122-134`
- Spec path: `infer/src/scheduler/cuda/spec_path.rs:251-258`
- vLLM Medusa prior-art: `1ccb41f` (still valid for `medusa.rs` core)
- Phase 1.B Qwen3 brief: superseded by THIS for Qwen3.5+

---

## §7 Pickup gate (revised from prior brief §7)

Step 0 prototype:
- [ ] Codex implements §4 prototype (~50 LOC, ~1 hr)
- [ ] Claude runs bench at K=5 / Qwen3.5-4B
- [ ] License decision per §4.2 gate

If LICENSED:
- [ ] User confirms: proceed with Option 1 snapshot ring (vs Option 2 shadow)
- [ ] Codex implements §3 substrate (~260 LOC + Medusa core ~380 LOC = ~640 LOC, ~3-4 days)
- [ ] Claude runs §5 greedy consistency tests
- [ ] License-or-kill at 1.5× tok/s threshold per `M_medusa-required-path.md`

If KILLED:
- [ ] Pivot to Option 2 (shadow state) or Option 3 (Qwen3.5 Medusa defer)
- [ ] New brief documenting picked direction

---

## §8 Why Option 1 over 2/3

| Option | LOC | Memory | Throughput | Risk |
|---|---:|---:|---:|---|
| **1: Snapshot ring** | **~260** | **+294 MB / slot** | **~14% step cost** | **LOW (extend existing)** |
| 2: Shadow state | ~400 | +49 MB × N shadow rows | ~10% step cost (replay) | MED (split forward) |
| 3: Defer | 0 | 0 | 0 | (no Medusa on Qwen3.5) |

Option 1 reuses existing `save_snapshot/restore_snapshot` infra
(L91-134), smallest LOC, predictable memory cost. Option 2 requires
splitting forward into real+shadow paths. Option 3 leaves Medusa
locked to full-attention models only.
