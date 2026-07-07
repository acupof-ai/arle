# Commit 1 design — reuse the existing shared-expert scratch in DSv4 batched decode

> Status: Active | 2026-07-07 | part of `2026-07-07-dsv4-decode-launch-bound-plan.md` Step 1

## One-line

The batched decode path (`forward_decode_batch_stream_impl`) allocates a fresh
shared-expert output + FP8 scratch **every layer every step** via
`dsv4_shared_expert_forward`, while the eager/MTP-verify paths already reuse a
**pre-allocated, model-wide** `Dsv4SharedDecodeScratch` held on the kv_adapter.
Switch the batched path to the existing scratch — no new buffer, no new struct.

## Evidence

- nsys `kern141_decode2` (07-03, TP=4, MTP-on): `cuMemAllocAsync`+`Free` = 12.2M
  calls (7.7% wall), `cuMemsetD8Async` = 2.4M (9.1%). Decode is launch-bound
  (launch+sync = 66% wall, zero `cuGraphLaunch`).
- The MTP-on profile runs `forward_decode_batch_stream_impl` (N=2–3 verify rows);
  N=1 never reaches it (`dsv4.rs:2526`).
- Per layer this path does `HiddenStates::uninit(hidden, seq_len)` for `shared`
  (`dsv4.rs:4028`) + the ~6 allocs + 4 tiny H2D inside `dsv4_shared_expert`
  (`moe.rs:4341-4400`, via `dsv4_shared_expert_forward`). = ~7 allocs + memsets
  × 43 layers × 8458 steps.

## What already exists (do NOT rebuild)

- `Dsv4KvAdapter.shared_expert_scratch: Option<Dsv4SharedDecodeScratch>`
  (`kv_layout.rs:159`) — allocated whenever the model has a shared-expert layer
  (`kv_layout.rs:494`, `shared_expert_decode.map(...)`), **independent of decode
  graph**. `max_m = 128` (`moe.rs:2437`) ≥ any decode/MTP `n` (MTP verify ≤
  `MAX_SPEC_VERIFY_ROWS = 64`).
- `Dsv4KvAdapter.shared_expert_out: Option<HiddenStates>` — **always** allocated,
  capacity `MAX_SPEC_VERIFY_ROWS` (`kv_layout.rs:491`).
- Accessor `kv_adapter.shared_expert_decode_mut() -> (Option<&mut HiddenStates>,
  Option<&mut Dsv4SharedDecodeScratch>)` (`kv_layout.rs:766`).
- `dsv4_shared_expert_forward_decode_scratch(ctx, stream, layer, hidden, out,
  swiglu_limit, scratch)` (`moe.rs:3711`) → `dsv4_shared_expert_pooled`, which
  handles `num_tokens > 1` (`moe.rs:4202-4212`) and asserts `n <= max_m`.
- Template: the eager/verify sites already do exactly this
  (`dsv4.rs:5309/5440-5480`): destructure `(shared_out, shared_scratch)`, set
  `shared.seq_len = seq_len`, call the `_decode_scratch` fn.

## The change (minimal, one call site)

`forward_decode_batch_stream_impl`, the MoE-tail shared-expert block
(`dsv4.rs:4027-4037`). Current:

```rust
let mut shared = unsafe { HiddenStates::uninit(&self.ctx, hidden_size, seq_len)? };
crate::moe::dsv4_shared_expert_forward(&self.ctx, &self.ctx.stream,
    layer.moe.as_ref().expect("DSv4 layer.moe"), &normed, &mut shared,
    self.config.swiglu_limit, &mut keepalive)?;
keepalive.keep_hidden(&shared);
crate::ops::add_batch(&self.ctx, &moe_out, &shared, &mut moe_with_shared)?;
```

New (mirror the eager template):

```rust
let (shared_out, shared_scratch) = kv_adapter.shared_expert_decode_mut();
let shared = shared_out.ok_or_else(|| anyhow!("DSv4 batched decode requires shared-expert output buffer"))?;
let scratch = shared_scratch.ok_or_else(|| anyhow!("DSv4 batched decode requires shared-expert scratch"))?;
shared.seq_len = seq_len;   // rows ≤ MAX_SPEC_VERIFY_ROWS ≤ max_m=128
ensure!(shared.hidden_dim == hidden_size, "shared out hidden {} != {}", shared.hidden_dim, hidden_size);
crate::moe::dsv4_shared_expert_forward_decode_scratch(&self.ctx, &self.ctx.stream,
    layer.moe.as_ref().expect("DSv4 layer.moe"), &normed, shared,
    self.config.swiglu_limit, scratch)?;
crate::ops::add_batch(&self.ctx, &moe_out, shared, &mut moe_with_shared)?;
```

## Borrow-checker note (the one real risk)

`kv_adapter` is `&mut` on the function. The MoE-tail block must not hold another
live `&mut kv_adapter` borrow across this. The attention half already returns its
kv_adapter borrows before the MoE half (they run sequentially per layer). The
`shared_expert_decode_mut()` borrow is confined to this block and dropped before
the next layer iteration. If the borrow checker complains, scope the destructure
in a `{ }` block returning nothing that outlives it. **Verify at compile (pod).**

## Correctness

- Byte-identical math: `_decode_scratch` → `dsv4_shared_expert_pooled` is the SAME
  kernel sequence the eager path uses (already needle-gate-licensed under #29).
  Only the buffer provenance changes (pooled vs fresh).
- `shared.seq_len = seq_len` before dispatch: the pooled fn reads `hidden.seq_len`
  = `n` and asserts `n <= max_m`; `shared_expert_out` capacity is
  `MAX_SPEC_VERIFY_ROWS = 64`. MTP verify n ≤ 64 ✓. If a future config allows
  n > 64 batched decode, the ensure fires (fail-closed, not silent corruption).
- No cross-layer/step aliasing: shared/scratch fully overwritten before read each
  layer (serial on `ctx.stream`), exactly the invariant the eager path relies on.

## Scope boundary (NOT in commit 1)

- The `dsv4_moe_forward_decode_fp8` ~8 allocs (counts/offsets/packed/act/
  expert_out/route_out) — that's commit 2 (needs a new pooled scratch keyed by
  `max_rows`, larger change).
- The B1 attn/ffn stream buffers + N-ring prepared buffers — later commits.

## Verify (pod TP=4)

1. `scripts/pod.sh sync && build` — compile clean.
2. Needle gate (`lever_gate.sh`, DSv4 profile, MTP-on) — coherent + DET, no
   regression vs baseline envelope.
3. A/B nsys or `/v1/stats`: `cuMemAllocAsync`/`cuMemsetD8Async` count drop on the
   batched path; wall-clock ms/committed-token Δ (hypothesis: small, launch-bound
   band). Land `wins/` (or `errors/` if wash) with the Δ.
