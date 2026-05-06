# M5 P0 — `ModelForward` trait survey + scope retreat

> Pre-plan output for `backend-unification.md` §M5 (Unified ModelForward
> Trait + Qwen3 Cross-Backend Path). Per the
> [`feedback_p0_survey_before_plan_body`](../../.claude/projects/-home-ckl-projects-arle/memory/feedback_p0_survey_before_plan_body.md)
> rule, do the file survey before drafting milestones — this entry IS the
> survey output, not a milestone plan.

## P0 finding

`infer/src/model.rs::ModelForward` (line 271, 686-line file) is **deeply
CUDA-specific** in its trait signature:

```rust
pub trait ModelForward: Send {
    type State: GenerationState + Send;
    type DecodeContext: DecodeContextOps + Send;
    type PrefillContext: Send;

    fn create_decode_context(
        &self,
        max_batch_size: usize,
        max_seq_len: Option<usize>,
        pool: &PagedKVPool,                      // ← CUDA-only type
    ) -> Result<Self::DecodeContext>;

    fn create_prefill_context(
        &self,
        _max_batch_size: usize,
        _prefill_budget_tokens: usize,
        _pool: &PagedKVPool,                     // ← CUDA-only type
    ) -> Result<Self::PrefillContext>;

    fn forward_with_logits(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(Vec<u32>, DeviceVec)>;          // ← cudarc-only type

    fn forward_sparse_decode_with_logits(
        &self,
        ...,
        _pool: &mut PagedKVPool,                 // ← CUDA-only type
        _decode_ctx: &mut Self::DecodeContext,
        _sparse_view: SparseKvDraftView<'_>,     // ← CUDA-only spec-decode shape
    ) -> Result<u32>;
    ...
}
```

`forward_prefill_batch_with_pool`, `forward_decode_batch` similarly take
`&mut PagedKVPool`.

Metal's KV state is `MetalKVPool` (packed-varlen + left-padding additive
mask) — not paged, not API-compatible with `PagedKVPool`. Metal's tensor
type is `MlxArray` — not API-compatible with `DeviceVec`.

**Implication**: Metal cannot directly implement `ModelForward` as it
stands. Three options for M5:

| Option | Description | Cost |
|---|---|---|
| A. Full generalization | Abstract `PagedKVPool` and `DeviceVec` behind traits in `ModelForward`'s signature. | very high — touches model.rs (686 lines), every Qwen3 forward file (~6000 lines), and forces Metal's MetalKVPool to match a paged-cache shape it doesn't have. |
| B. Two-tier trait | `ModelArch` (backend-neutral, config getters + architecture spec) + `BackendModelForward<B>` (backend-specific, takes B's pool/tensor types). Schedulers go through the per-backend trait; cross-backend code (telemetry, metrics, tokenizer hookup) goes through `ModelArch`. | medium — adds one trait, ~50 lines new code. Doesn't unify the forward path itself. |
| C. Status quo + escape hatch | Keep `ModelForward` CUDA-specific. Document Metal's parallel structure in `backend/metal/AGENTS.md`. Track unification at the M5+ horizon when MLX gets a paged-KV equivalent. | minimal — a doc commit. |

## Scope retreat — recommend Option B (two-tier trait)

Option A blocks on Metal's pool architecture, which is *intentionally*
different (MLX's lazy graph + packed-varlen is the right shape for Metal,
not paged-KV — see `backend/metal/AGENTS.md` §7). Forcing it into a
PagedKVPool shape would lose the MLX-native efficiency.

Option C ships nothing.

Option B is the right cut: extract the *backend-neutral portion* (config
getters: `kv_cache_bytes_per_token`, `num_kv_layers`, `num_kv_heads`,
`head_dim`, `num_q_heads`, plus `model_id` and a few config-shape
methods) into `ModelArch`. **Qwen3** already declares all these
implicitly through its config — the trait extraction is mechanical.
**Metal Qwen3** can then implement `ModelArch` cheaply, even though it
doesn't implement `ModelForward`.

This unblocks:
- Cross-backend telemetry / metrics rendering that needs to know
  `kv_cache_bytes_per_token` etc.
- Cross-backend test harnesses that need to enumerate "this model has
  N layers and head_dim H" without instantiating the full forward path.
- Future M5+ work (if MLX ever grows a paged-KV equivalent) that wants
  to share the *prefill / decode* method bodies — has a base trait
  to extend.

## Recommended sub-plan (writes the M5 milestone body)

`docs/plans/M5-modelarch-trait.md` (NOT M5-modelforward-trait — narrower
scope name) covering:

| Step | Files | Delta | Days |
|---|---|---|---|
| U1. Define `ModelArch` trait in `infer/src/model.rs` (super-trait of `ModelForward`). All 6 config getters move to it. | 1 | +50 / -0 | 0.5 |
| U2. CUDA `Qwen3Model` already has `ModelForward` impl; the 6 getters now satisfy the new super-trait automatically. Run e2e + greedy_consistency to confirm zero regression. | 0 | 0 | 0.5 |
| U3. Metal `Qwen3` (currently a builder pattern in `backend/metal/forward.rs`) gains a thin `impl ModelArch for MetalQwen3Adapter` block. | 1 | +30 | 1 |
| U4. Cross-backend caller update: server_engine telemetry (`metrics/render.rs`) that currently casts to CUDA-specific Qwen3 reads through `ModelArch`. | 1 | refactor | 1 |
| U5. wins entry recording the line delta + verification gate set. | 1 | +1 file | 0.5 |

**Total: ~3.5 days**, not the original Week 7-8 budget (which was
predicated on Option A). M5 retreats from "Unified ModelForward" to
"Unified ModelArch" — narrower scope, but lands cleanly without
fighting MLX.

The "real" forward unification (forward_prefill / forward_decode share
across CUDA + Metal) is **deferred to a future milestone** that gates on
either (a) MLX gaining a paged-KV equivalent or (b) ARLE adopting a
unified KV abstraction across packed-varlen and paged. Both are
multi-month efforts and aren't justified by current bench data — the
two backends already share the model-architecture knowledge that
matters for telemetry / harness; the rest is execution-side.

## Out of scope for M5 (after retreat)

- Forward method-body unification — deferred to undefined future
  milestone gated on MLX paged-KV.
- Spec-decode cross-backend — already explicit in M_c plan
  (`docs/plans/M_c-hybrid-spec-rollback.md`).
- `MetalKVPool` API redesign — orthogonal work.

## References

- `docs/plans/backend-unification.md` §M5 (the original milestone framing
  — needs scope retreat update once this P0 lands)
- `docs/plans/M3-unified-schedule-ir.md` — the precedent for "extract IR
  rather than force structural unification"
- `infer/src/model.rs::ModelForward` — surveyed (line 271)
- `infer/src/backend/metal/forward.rs` — Metal Qwen3 forward graph
  (would gain `ModelArch` impl in U3)
- `infer/src/backend/metal/AGENTS.md` §7 — packed-varlen rationale
