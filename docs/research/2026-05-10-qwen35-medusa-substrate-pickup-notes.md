---
title: 2026-05-10 Qwen3.5 Medusa v2 — §3 substrate pickup notes (concrete file-line refs)
date: 2026-05-10
type: research
status: open (companion to a00113f v2 redesign brief; populates concrete pickup intel)
related_docs: [`a00113f` v2 snapshot-ring brief, `05270b1` step 0 audit + blocker, `1ccb41f` vLLM Medusa prior-art]
---

# Qwen3.5 Medusa v2 — §3 substrate pickup notes

> **Why this**: `a00113f` brief §3 estimates 260 LOC for rollback infra
> + 380 LOC for Medusa core. This entry surveys actual code surfaces
> with file:line refs so the §3 pickup (post Step 0 LICENSE) starts
> with concrete intel, not LOC-estimate guesses.

## §1 Mirror target: qwen3 `forward_spec_verify_batch`

**Source**: `infer/src/model/qwen3/forward.rs:729-795`

Pattern qwen35 must mirror:
```rust
fn forward_spec_verify_batch(
    &self,
    requests: &[SpecVerifyRequest<'_>],
    states: &mut [Self::State],
    pool: &mut PagedKVPool,
) -> Result<Vec<SpecVerifyOutput>> {
    // Step-by-step verifier:
    // for step in 0..max_steps {
    //     forward_decode_batch(tokens, states, slot_indices, pool, ...)
    //     for slot: select_token argmax → outputs[slot].push(token)
    // }
}
```

**Qwen3.5 delta** (per `a00113f` §3):
- BEFORE each step: push current `recurrent_state` to ring slot (~10 LOC)
- AFTER `forward_decode_batch`: same as qwen3 (argmax token select)
- ON commit (called by spec_path scheduler): restore from ring slot j

LOC estimate refined: **+50 LOC** in `qwen35/forward.rs` (was estimated +80
in brief; actual is smaller because most logic mirrors qwen3 verbatim).

## §2 Hidden state Medusa consumes

**Source**: `infer/src/model/qwen35/batch_decode.rs:578-594`

```rust
ops::rms_norm_batch_offset_into(
    &self.ctx,
    hidden,
    &self.norm,
    c.rms_norm_eps,
    &mut bufs.common.normed,    // ← Medusa head input
)?;
let logits_buf = bufs.logits_batch.as_mut().unwrap();
ops::gemm_into(
    &self.ctx,
    &self.embed_tokens,
    &bufs.common.normed,         // ← then projected to logits
    logits_buf,
);
```

`bufs.common.normed` IS the post-RMS-norm pre-LM-head hidden state.
Medusa propose: capture this between `rms_norm_batch_offset_into` and
`gemm_into` calls. ~5 LOC clone or tensor-view share.

## §3 Spec scheduler commit hook

**Source**: `infer/src/scheduler/cuda/spec_path.rs:251-258`

Current code (paged KV only):
```rust
if let Err(err) = pool.truncate_slot(row.slot_idx, keep_target_len) {
    log::error!("spec target KV rollback failed: {err}");
}
```

L264 already has `draft_engine.commit_request_state` pattern for draft
side. The MIRROR for target (with non-KV state like recurrent) is the
missing piece per blocker.

Proposed addition (per `a00113f` §3):
```rust
if let Err(err) = target_engine.commit_request_state(
    row.slot_idx,
    keep_target_len,
    result.num_accepted,    // hint for ring slot restore
) {
    log::error!("spec target non-KV commit failed: {err}");
}
```

Trait method on `InferenceEngine` or model-specific. Default impl =
no-op (qwen3 etc don't need it). Qwen3.5 impl = restore from ring.

LOC estimate refined: **+25 LOC** in `spec_path.rs` (was +40).

## §4 Medusa core (independent of Qwen3.5 rollback)

Per `1ccb41f` vLLM prior-art survey (still valid):
- `infer/src/model/medusa.rs` ~250 LOC (ResidualBlock + Medusa + Config)
- `infer/src/model/medusa/weights.rs` ~80 LOC (separate safetensors load)
- `infer/src/speculative.rs` ~50 LOC delta (replace MockDraftModel)

**These ~380 LOC are model-agnostic**. Same code lands regardless of
target model (Qwen3 / Qwen3.5 / Qwen3.6). The Qwen3.5-specific work
is ONLY the §1-§3 rollback infra (~140 LOC actual, refined down from
260 LOC brief estimate).

## §5 Refined LOC budget

Per actual code surveys above (refines `a00113f` §3 table):

| File | Refined LOC | Brief estimate |
|---|---:|---:|
| `qwen35/recurrent_state.rs` (ring extension) | +60 | +60 |
| `qwen35/forward.rs` (verify hook + ring usage) | +50 | +80 |
| `qwen35/batch_decode.rs` (capture normed) | +5 | (not separated) |
| `scheduler/cuda/spec_path.rs` (commit hook) | +25 | +40 |
| `scheduler/cuda/execution.rs` (wire) | +20 | +20 |
| `infer/src/model/medusa.rs` (NEW) | +250 | +250 |
| `infer/src/model/medusa/weights.rs` (NEW) | +80 | +80 |
| `infer/src/speculative.rs` (delta) | +50 | +50 |
| `tests/test_qwen35_spec_rollback.rs` (NEW) | +60 | +60 |
| **TOTAL** | **~600 LOC** | **~640 LOC** |

Refined estimate ~6% smaller than brief — mostly because qwen3
mirror is closer than expected.

## §6 What this enables for codex post Step 0 LICENSE

Codex pickup directive will be (paste-ready):
```
PICKUP: Qwen3.5 Medusa Phase 1.B v2 §3 full substrate (~600 LOC)
Brief: docs/plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md
Pickup notes: docs/research/2026-05-10-qwen35-medusa-substrate-pickup-notes.md (concrete file:line refs)
Order:
1. recurrent_state.rs ring extension (per §1)
2. spec_path.rs commit hook (per §3)
3. qwen35/forward.rs verify path mirror of qwen3:729-795 + ring usage (per §1)
4. Medusa core: medusa.rs + weights.rs + speculative.rs delta (per `1ccb41f`)
5. Tests: greedy_consistency under j∈{0..K} accept (per brief §5)
6. Bench: tok/s vs no-spec at K=5 / Qwen3.5-4B; gate at 1.5×
```

## §7 Cross-references

- `a00113f` Qwen3.5 v2 redesign brief
- `05270b1` codex Step 0 audit + recurrent rollback blocker
- `1ccb41f` vLLM Medusa prior-art (still valid for §4 core)
- `infer/src/model/qwen3/forward.rs:729-795` (verify pattern)
- `infer/src/model/qwen35/batch_decode.rs:578-594` (hidden state capture point)
- `infer/src/model/qwen35/recurrent_state.rs:91-134` (snapshot infra to extend)
- `infer/src/scheduler/cuda/spec_path.rs:251-264` (commit hook surface)
